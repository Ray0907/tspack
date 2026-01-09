//! QDC (Quantized Delta with Checkpoint) streaming encoder
//!
//! Delta encoding for time-series data with periodic checkpoints
//! to limit error propagation.

use crate::{Field, Quality, Error, MAX_FIELDS};
use crate::bitpack::{pack_bits, unpack_bits};
use crate::crc::crc8;

/// Default checkpoint interval (absolute value every N samples)
pub const DEFAULT_CHECKPOINT_INTERVAL: usize = 32;

/// Maximum delta bits (signed, so actual range is +/- 2^(bits-1))
pub const MAX_DELTA_BITS: u8 = 16;

/// Delta field configuration
#[derive(Debug, Clone, Copy)]
pub struct DeltaField {
    /// Base field definition
    pub field: Field,
    /// Number of bits for delta encoding (includes sign bit)
    pub delta_bits: u8,
    /// Maximum positive delta value
    max_delta: i32,
}

impl DeltaField {
    /// Create delta field from base field with expected change rate
    ///
    /// # Arguments
    /// * `field` - Base field definition
    /// * `max_change_per_sample` - Maximum expected change between samples
    pub fn new(field: Field, max_change_per_sample: f32) -> Self {
        // Calculate delta bits needed
        // Delta range: [-max_delta, +max_delta]
        let range = field.max - field.min;
        let scale = ((1u32 << field.bits()) - 1) as f32 / range;
        let max_quantized_delta = (max_change_per_sample * scale).ceil() as u32;

        // Need to represent [-max_delta, +max_delta], so we need:
        // 2 * max_delta + 1 values, which requires ceil(log2(2*max_delta+1)) bits
        let delta_range = max_quantized_delta * 2 + 1;
        let delta_bits = (32 - delta_range.leading_zeros()).max(1).min(MAX_DELTA_BITS as u32) as u8;
        let max_delta = (1i32 << (delta_bits - 1)) - 1;

        Self {
            field,
            delta_bits,
            max_delta,
        }
    }

    /// Create delta field with explicit delta bits
    pub fn with_delta_bits(field: Field, delta_bits: u8) -> Self {
        let delta_bits = delta_bits.min(MAX_DELTA_BITS).max(1);
        let max_delta = (1i32 << (delta_bits - 1)) - 1;

        Self {
            field,
            delta_bits,
            max_delta,
        }
    }

    /// Check if delta is within encodable range
    #[inline]
    pub fn can_encode_delta(&self, delta: i32) -> bool {
        delta >= -self.max_delta && delta <= self.max_delta
    }

    /// Encode delta as unsigned value (offset binary)
    #[inline]
    pub fn encode_delta(&self, delta: i32) -> u32 {
        // Offset binary: add max_delta to make it unsigned
        // Range: [0, 2*max_delta]
        (delta + self.max_delta) as u32
    }

    /// Decode unsigned value back to signed delta
    #[inline]
    pub fn decode_delta(&self, encoded: u32) -> i32 {
        encoded as i32 - self.max_delta
    }
}

/// Delta schema for streaming encoding
#[derive(Debug, Clone)]
pub struct DeltaSchema {
    pub id: u8,
    fields: [DeltaField; MAX_FIELDS],
    field_count: usize,
    /// Bits per absolute frame
    absolute_bits: u16,
    /// Bits per delta frame
    delta_bits: u16,
    /// Checkpoint interval
    pub checkpoint_interval: usize,
}

impl DeltaSchema {
    /// Create delta schema from fields with expected change rates
    pub fn new(id: u8, fields: &[(Field, f32)], checkpoint_interval: usize) -> Self {
        assert!(fields.len() <= MAX_FIELDS, "Too many fields");

        let mut delta_fields = [DeltaField::with_delta_bits(Field::new(0.0, 1.0, 1.0), 8); MAX_FIELDS];
        let mut absolute_bits = 0u16;
        let mut delta_bits_total = 0u16;

        for (i, (field, max_change)) in fields.iter().enumerate() {
            let df = DeltaField::new(*field, *max_change);
            delta_fields[i] = df;
            absolute_bits += field.bits() as u16;
            delta_bits_total += df.delta_bits as u16;
        }

        Self {
            id,
            fields: delta_fields,
            field_count: fields.len(),
            absolute_bits,
            delta_bits: delta_bits_total,
            checkpoint_interval,
        }
    }

    /// Get the delta fields
    pub fn fields(&self) -> &[DeltaField] {
        &self.fields[..self.field_count]
    }

    /// Get field count
    pub fn field_count(&self) -> usize {
        self.field_count
    }

    /// Get bytes needed for absolute frame
    pub fn absolute_bytes(&self) -> usize {
        ((self.absolute_bits + 7) / 8) as usize
    }

    /// Get bytes needed for delta frame
    pub fn delta_bytes(&self) -> usize {
        ((self.delta_bits + 7) / 8) as usize
    }
}

/// Streaming encoder state
#[derive(Debug)]
pub struct StreamEncoder {
    /// Previous quantized values for delta calculation
    prev_codes: [u32; MAX_FIELDS],
    /// Sample counter for checkpoint
    sample_count: usize,
    /// Whether we have previous values
    initialized: bool,
}

impl StreamEncoder {
    /// Create new streaming encoder
    pub fn new() -> Self {
        Self {
            prev_codes: [0; MAX_FIELDS],
            sample_count: 0,
            initialized: false,
        }
    }

    /// Reset encoder state
    pub fn reset(&mut self) {
        self.prev_codes = [0; MAX_FIELDS];
        self.sample_count = 0;
        self.initialized = false;
    }

    /// Check if next frame should be absolute (checkpoint)
    pub fn needs_checkpoint(&self, schema: &DeltaSchema) -> bool {
        !self.initialized || self.sample_count % schema.checkpoint_interval == 0
    }

    /// Encode a single sample
    ///
    /// Returns (bytes_written, is_absolute_frame)
    pub fn encode(
        &mut self,
        schema: &DeltaSchema,
        values: &[Option<f32>],
        buf: &mut [u8],
    ) -> Result<(usize, bool), Error> {
        let is_checkpoint = self.needs_checkpoint(schema);

        if is_checkpoint {
            let len = self.encode_absolute(schema, values, buf)?;
            Ok((len, true))
        } else {
            // Try delta encoding, fall back to absolute if delta overflow
            match self.try_encode_delta(schema, values, buf) {
                Ok(len) => Ok((len, false)),
                Err(Error::DeltaOverflow) => {
                    // Delta overflow - encode as absolute checkpoint
                    let len = self.encode_absolute(schema, values, buf)?;
                    Ok((len, true))
                }
                Err(e) => Err(e),
            }
        }
    }

    /// Encode absolute frame (checkpoint)
    fn encode_absolute(
        &mut self,
        schema: &DeltaSchema,
        values: &[Option<f32>],
        buf: &mut [u8],
    ) -> Result<usize, Error> {
        // Frame format: [flags:1][payload][crc:1]
        // flags: bit 7 = absolute frame (1)
        let payload_bytes = schema.absolute_bytes();
        let total_size = 1 + payload_bytes + 1;

        if buf.len() < total_size {
            return Err(Error::BufferTooSmall);
        }

        // Clear buffer
        buf[..total_size].fill(0);

        // Flags: 0x80 = absolute frame
        buf[0] = 0x80;

        // Pack values
        let mut bit_offset = 0usize;
        for (i, value) in values.iter().enumerate().take(schema.field_count()) {
            let df = &schema.fields[i];
            let (code, _quality) = match value {
                Some(v) => df.field.quantize(*v),
                None => (0, Quality::Null),
            };

            pack_bits(&mut buf[1..], bit_offset, df.field.bits() as usize, code);
            bit_offset += df.field.bits() as usize;

            // Store for next delta
            self.prev_codes[i] = code;
        }

        // CRC
        let crc_offset = 1 + payload_bytes;
        buf[crc_offset] = crc8(&buf[..crc_offset]);

        self.initialized = true;
        self.sample_count += 1;

        Ok(total_size)
    }

    /// Try to encode delta frame
    fn try_encode_delta(
        &mut self,
        schema: &DeltaSchema,
        values: &[Option<f32>],
        buf: &mut [u8],
    ) -> Result<usize, Error> {
        // Frame format: [flags:1][payload][crc:1]
        // flags: bit 7 = delta frame (0)
        let payload_bytes = schema.delta_bytes();
        let total_size = 1 + payload_bytes + 1;

        if buf.len() < total_size {
            return Err(Error::BufferTooSmall);
        }

        // Calculate deltas and check if all fit
        let mut deltas = [0i32; MAX_FIELDS];
        let mut new_codes = [0u32; MAX_FIELDS];

        for (i, value) in values.iter().enumerate().take(schema.field_count()) {
            let df = &schema.fields[i];
            let (code, _quality) = match value {
                Some(v) => df.field.quantize(*v),
                None => (0, Quality::Null),
            };

            let delta = code as i32 - self.prev_codes[i] as i32;

            if !df.can_encode_delta(delta) {
                return Err(Error::DeltaOverflow);
            }

            deltas[i] = delta;
            new_codes[i] = code;
        }

        // Clear buffer
        buf[..total_size].fill(0);

        // Flags: 0x00 = delta frame
        buf[0] = 0x00;

        // Pack deltas
        let mut bit_offset = 0usize;
        for (i, delta) in deltas.iter().enumerate().take(schema.field_count()) {
            let df = &schema.fields[i];
            let encoded = df.encode_delta(*delta);
            pack_bits(&mut buf[1..], bit_offset, df.delta_bits as usize, encoded);
            bit_offset += df.delta_bits as usize;
        }

        // CRC
        let crc_offset = 1 + payload_bytes;
        buf[crc_offset] = crc8(&buf[..crc_offset]);

        // Update state
        self.prev_codes = new_codes;
        self.sample_count += 1;

        Ok(total_size)
    }
}

impl Default for StreamEncoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Streaming decoder state
#[derive(Debug)]
pub struct StreamDecoder {
    /// Previous quantized values for delta reconstruction
    prev_codes: [u32; MAX_FIELDS],
    /// Whether we have previous values
    initialized: bool,
}

impl StreamDecoder {
    /// Create new streaming decoder
    pub fn new() -> Self {
        Self {
            prev_codes: [0; MAX_FIELDS],
            initialized: false,
        }
    }

    /// Reset decoder state
    pub fn reset(&mut self) {
        self.prev_codes = [0; MAX_FIELDS];
        self.initialized = false;
    }

    /// Decode a single frame
    pub fn decode(
        &mut self,
        schema: &DeltaSchema,
        buf: &[u8],
    ) -> Result<[f32; MAX_FIELDS], Error> {
        if buf.is_empty() {
            return Err(Error::InvalidFrame);
        }

        let is_absolute = (buf[0] & 0x80) != 0;

        if is_absolute {
            self.decode_absolute(schema, buf)
        } else {
            if !self.initialized {
                return Err(Error::InvalidFrame); // Delta without prior absolute
            }
            self.decode_delta(schema, buf)
        }
    }

    /// Decode absolute frame
    fn decode_absolute(
        &mut self,
        schema: &DeltaSchema,
        buf: &[u8],
    ) -> Result<[f32; MAX_FIELDS], Error> {
        let payload_bytes = schema.absolute_bytes();
        let expected_size = 1 + payload_bytes + 1;

        if buf.len() < expected_size {
            return Err(Error::InvalidFrame);
        }

        // Verify CRC
        let crc_offset = expected_size - 1;
        if buf[crc_offset] != crc8(&buf[..crc_offset]) {
            return Err(Error::CrcMismatch);
        }

        // Unpack values
        let mut values = [0.0f32; MAX_FIELDS];
        let mut bit_offset = 0usize;

        for i in 0..schema.field_count() {
            let df = &schema.fields[i];
            let code = unpack_bits(&buf[1..], bit_offset, df.field.bits() as usize);
            bit_offset += df.field.bits() as usize;

            values[i] = df.field.dequantize(code);
            self.prev_codes[i] = code;
        }

        self.initialized = true;
        Ok(values)
    }

    /// Decode delta frame
    fn decode_delta(
        &mut self,
        schema: &DeltaSchema,
        buf: &[u8],
    ) -> Result<[f32; MAX_FIELDS], Error> {
        let payload_bytes = schema.delta_bytes();
        let expected_size = 1 + payload_bytes + 1;

        if buf.len() < expected_size {
            return Err(Error::InvalidFrame);
        }

        // Verify CRC
        let crc_offset = expected_size - 1;
        if buf[crc_offset] != crc8(&buf[..crc_offset]) {
            return Err(Error::CrcMismatch);
        }

        // Unpack deltas and reconstruct values
        let mut values = [0.0f32; MAX_FIELDS];
        let mut bit_offset = 0usize;

        for i in 0..schema.field_count() {
            let df = &schema.fields[i];
            let encoded = unpack_bits(&buf[1..], bit_offset, df.delta_bits as usize);
            bit_offset += df.delta_bits as usize;

            let delta = df.decode_delta(encoded);
            let code = (self.prev_codes[i] as i32 + delta) as u32;

            values[i] = df.field.dequantize(code);
            self.prev_codes[i] = code;
        }

        Ok(values)
    }
}

impl Default for StreamDecoder {
    fn default() -> Self {
        Self::new()
    }
}

// Add DeltaOverflow error variant
impl Error {
    /// Check if error is delta overflow
    pub fn is_delta_overflow(&self) -> bool {
        matches!(self, Error::DeltaOverflow)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_schema() -> DeltaSchema {
        DeltaSchema::new(
            1,
            &[
                // Temperature: -40 to 80, precision 0.1, max change 2.0 per sample
                (Field::new(-40.0, 80.0, 0.1), 2.0),
                // Humidity: 0 to 100, precision 1, max change 5.0 per sample
                (Field::new(0.0, 100.0, 1.0), 5.0),
                // Pressure: 300 to 1200, precision 0.1, max change 1.0 per sample
                (Field::new(300.0, 1200.0, 0.1), 1.0),
            ],
            32,
        )
    }

    #[test]
    fn test_delta_field_creation() {
        let field = Field::new(-40.0, 80.0, 0.1);
        let df = DeltaField::new(field, 2.0);

        // With 11 bits for absolute (1201 values), and max change of 2.0
        // delta should be around 17 quantized units (2.0 / 0.1 * scale)
        assert!(df.delta_bits >= 5 && df.delta_bits <= 8);
    }

    #[test]
    fn test_delta_encode_decode() {
        let field = Field::new(-40.0, 80.0, 0.1);
        let df = DeltaField::with_delta_bits(field, 8);

        // Test various deltas
        for delta in [-100, -50, -1, 0, 1, 50, 100] {
            if df.can_encode_delta(delta) {
                let encoded = df.encode_delta(delta);
                let decoded = df.decode_delta(encoded);
                assert_eq!(decoded, delta, "Delta {} roundtrip failed", delta);
            }
        }
    }

    #[test]
    fn test_stream_encoder_absolute() {
        let schema = test_schema();
        let mut encoder = StreamEncoder::new();
        let mut buf = [0u8; 32];

        let values = [Some(23.5), Some(65.0), Some(1013.2)];
        let (len, is_absolute) = encoder.encode(&schema, &values, &mut buf).unwrap();

        assert!(is_absolute, "First frame should be absolute");
        assert!(len > 0);
    }

    #[test]
    fn test_stream_encoder_delta() {
        let schema = test_schema();
        let mut encoder = StreamEncoder::new();
        let mut buf = [0u8; 32];

        // First frame - absolute
        let values1 = [Some(23.5), Some(65.0), Some(1013.2)];
        let (_, is_abs1) = encoder.encode(&schema, &values1, &mut buf).unwrap();
        assert!(is_abs1);

        // Second frame - should be delta (small change)
        let values2 = [Some(23.6), Some(65.0), Some(1013.3)];
        let (len2, is_abs2) = encoder.encode(&schema, &values2, &mut buf).unwrap();
        assert!(!is_abs2, "Second frame should be delta");

        // Delta frame should be smaller than absolute
        let abs_size = 1 + schema.absolute_bytes() + 1;
        assert!(len2 < abs_size, "Delta frame should be smaller");
    }

    #[test]
    fn test_stream_roundtrip() {
        let schema = test_schema();
        let mut encoder = StreamEncoder::new();
        let mut decoder = StreamDecoder::new();
        let mut buf = [0u8; 32];

        let test_values = [
            [Some(23.5), Some(65.0), Some(1013.2)],
            [Some(23.6), Some(65.0), Some(1013.3)],
            [Some(23.7), Some(64.0), Some(1013.2)],
            [Some(23.8), Some(64.0), Some(1013.1)],
        ];

        for values in &test_values {
            let (len, _) = encoder.encode(&schema, values, &mut buf).unwrap();
            let decoded = decoder.decode(&schema, &buf[..len]).unwrap();

            // Check values within precision
            for i in 0..3 {
                if let Some(expected) = values[i] {
                    let precision = schema.fields[i].field.precision;
                    assert!(
                        (decoded[i] - expected).abs() < precision,
                        "Field {} mismatch: {} vs {}",
                        i, decoded[i], expected
                    );
                }
            }
        }
    }

    #[test]
    fn test_checkpoint_interval() {
        let schema = DeltaSchema::new(
            1,
            &[(Field::new(0.0, 100.0, 1.0), 5.0)],
            4, // checkpoint every 4 samples
        );
        let mut encoder = StreamEncoder::new();
        let mut buf = [0u8; 32];

        let values = [Some(50.0)];

        // Sample 0: absolute (first)
        let (_, is_abs) = encoder.encode(&schema, &values, &mut buf).unwrap();
        assert!(is_abs, "Sample 0 should be absolute");

        // Samples 1-3: delta
        for i in 1..4 {
            let (_, is_abs) = encoder.encode(&schema, &values, &mut buf).unwrap();
            assert!(!is_abs, "Sample {} should be delta", i);
        }

        // Sample 4: absolute (checkpoint)
        let (_, is_abs) = encoder.encode(&schema, &values, &mut buf).unwrap();
        assert!(is_abs, "Sample 4 should be absolute (checkpoint)");
    }

    #[test]
    fn test_delta_overflow_fallback() {
        let schema = DeltaSchema::new(
            1,
            &[(Field::new(0.0, 100.0, 1.0), 1.0)], // max change 1.0
            32,
        );
        let mut encoder = StreamEncoder::new();
        let mut buf = [0u8; 32];

        // First value
        let (_, is_abs1) = encoder.encode(&schema, &[Some(50.0)], &mut buf).unwrap();
        assert!(is_abs1);

        // Big jump - should fallback to absolute
        let (_, is_abs2) = encoder.encode(&schema, &[Some(80.0)], &mut buf).unwrap();
        assert!(is_abs2, "Large delta should fallback to absolute");
    }
}
