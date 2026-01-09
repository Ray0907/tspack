//! tspack: Schema-driven IoT time-series data encoding
//!
//! A compact binary encoding format for IoT sensor data that uses schema
//! information to achieve near-optimal compression ratios.

#![cfg_attr(not(feature = "std"), no_std)]

mod bitpack;
mod crc;
pub mod delta;

use bitpack::{pack_bits, unpack_bits};
use crc::crc8;

pub use delta::{DeltaSchema, DeltaField, StreamEncoder, StreamDecoder};

/// Maximum number of fields per schema
pub const MAX_FIELDS: usize = 16;

/// Magic number for tspack frames (0xA = 1010)
pub const MAGIC: u8 = 0xA0;

/// Protocol version
pub const VERSION: u8 = 0;

/// Error types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    BufferTooSmall,
    InvalidFrame,
    CrcMismatch,
    SchemaIdMismatch,
    TooManyFields,
    DeltaOverflow,
}

/// Quality flags for each field value
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Quality {
    Valid = 0,
    Null = 1,
    Overflow = 2,
    Underflow = 3,
}

impl Quality {
    fn from_u8(v: u8) -> Self {
        match v & 0x03 {
            0 => Quality::Valid,
            1 => Quality::Null,
            2 => Quality::Overflow,
            3 => Quality::Underflow,
            _ => unreachable!(),
        }
    }
}

/// Field flags
#[derive(Debug, Clone, Copy, Default)]
pub struct FieldFlags {
    pub nullable: bool,
}

/// Field definition in a schema
#[derive(Debug, Clone, Copy)]
pub struct Field {
    pub min: f32,
    pub max: f32,
    pub precision: f32,
    pub flags: FieldFlags,
    // Computed fields
    bits: u8,
    scale: f32,
}

impl Field {
    /// Create a new field definition
    pub fn new(min: f32, max: f32, precision: f32) -> Self {
        let range = max - min;
        let num_values = (range / precision).ceil() as u32 + 1;
        let bits = (32 - num_values.leading_zeros()).max(1) as u8;
        let scale = ((1u32 << bits) - 1) as f32 / range;

        Self {
            min,
            max,
            precision,
            flags: FieldFlags::default(),
            bits,
            scale,
        }
    }

    /// Mark this field as nullable
    pub fn nullable(mut self) -> Self {
        self.flags.nullable = true;
        self
    }

    /// Get the number of bits needed for this field
    pub fn bits(&self) -> u8 {
        self.bits
    }

    /// Quantize a value to an integer code
    pub fn quantize(&self, value: f32) -> (u32, Quality) {
        if value.is_nan() {
            return (0, Quality::Null);
        }
        if value > self.max {
            let code = (1u32 << self.bits) - 1;
            return (code, Quality::Overflow);
        }
        if value < self.min {
            return (0, Quality::Underflow);
        }

        let normalized = (value - self.min) * self.scale;
        let code = (normalized + 0.5) as u32;
        let max_code = (1u32 << self.bits) - 1;
        (code.min(max_code), Quality::Valid)
    }

    /// Dequantize an integer code back to a value
    pub fn dequantize(&self, code: u32) -> f32 {
        let max_code = (1u32 << self.bits) - 1;
        let normalized = code as f32 / max_code as f32;
        self.min + normalized * (self.max - self.min)
    }
}

/// Schema definition
#[derive(Debug, Clone)]
pub struct Schema {
    pub id: u8,
    fields: [Field; MAX_FIELDS],
    field_count: usize,
    total_bits: u16,
}

impl Schema {
    /// Create a new schema
    pub fn new(id: u8, fields: &[Field]) -> Self {
        assert!(fields.len() <= MAX_FIELDS, "Too many fields");

        let mut schema_fields = [Field::new(0.0, 1.0, 1.0); MAX_FIELDS];
        let mut total_bits = 0u16;

        for (i, f) in fields.iter().enumerate() {
            schema_fields[i] = *f;
            total_bits += f.bits as u16;
        }

        Self {
            id,
            fields: schema_fields,
            field_count: fields.len(),
            total_bits,
        }
    }

    /// Get the fields
    pub fn fields(&self) -> &[Field] {
        &self.fields[..self.field_count]
    }

    /// Get the number of fields
    pub fn field_count(&self) -> usize {
        self.field_count
    }

    /// Get total payload bits (excluding header and CRC)
    pub fn payload_bits(&self) -> u16 {
        self.total_bits
    }

    /// Get total payload bytes
    pub fn payload_bytes(&self) -> usize {
        ((self.total_bits + 7) / 8) as usize
    }
}

/// Encode options
#[derive(Debug, Clone, Copy, Default)]
pub struct EncodeOptions {
    pub include_quality: bool,
}

/// Encode result
#[derive(Debug)]
pub struct EncodeResult {
    pub len: usize,
    pub qualities: [Quality; MAX_FIELDS],
}

/// Encode values using the schema
///
/// Returns the number of bytes written to the buffer
pub fn encode(
    schema: &Schema,
    values: &[Option<f32>],
    buf: &mut [u8],
    options: EncodeOptions,
) -> Result<EncodeResult, Error> {
    let has_quality = options.include_quality || values.iter().any(|v| v.is_none());

    // Calculate required size
    let quality_bytes = if has_quality {
        (schema.field_count() * 2 + 7) / 8
    } else {
        0
    };
    let header_size = 2; // magic+version, control
    let payload_size = schema.payload_bytes();
    let crc_size = 1;
    let total_size = header_size + quality_bytes + payload_size + crc_size;

    if buf.len() < total_size {
        return Err(Error::BufferTooSmall);
    }

    // Header byte 0: Magic (4 bits) + Version (4 bits)
    buf[0] = MAGIC | VERSION;

    // Header byte 1: Type (2) + Q (1) + T (1) + Schema ID (4)
    // Type = 00 (single point), Q = has_quality, T = 0 (no timestamp)
    let q_flag = if has_quality { 0x20 } else { 0x00 };
    buf[1] = q_flag | (schema.id & 0x0F);

    let mut offset = 2;
    let mut qualities = [Quality::Valid; MAX_FIELDS];

    // Quality bitmap (if present)
    if has_quality {
        let mut quality_bits = 0u32;
        for (i, value) in values.iter().enumerate().take(schema.field_count()) {
            let quality = match value {
                Some(v) => {
                    let (_, q) = schema.fields[i].quantize(*v);
                    q
                }
                None => Quality::Null,
            };
            qualities[i] = quality;
            quality_bits |= (quality as u32) << (i * 2);
        }

        // Write quality bitmap
        for i in 0..quality_bytes {
            buf[offset + i] = ((quality_bits >> (i * 8)) & 0xFF) as u8;
        }
        offset += quality_bytes;
    }

    // Payload - bit-packed values
    let mut bit_offset = 0usize;
    for (i, value) in values.iter().enumerate().take(schema.field_count()) {
        let field = &schema.fields[i];
        let code = match value {
            Some(v) => {
                let (c, q) = field.quantize(*v);
                if !has_quality {
                    qualities[i] = q;
                }
                c
            }
            None => 0, // Null values encode as 0
        };

        pack_bits(&mut buf[offset..], bit_offset, field.bits as usize, code);
        bit_offset += field.bits as usize;
    }
    offset += payload_size;

    // CRC-8
    buf[offset] = crc8(&buf[..offset]);
    offset += 1;

    Ok(EncodeResult {
        len: offset,
        qualities,
    })
}

/// Decode result
#[derive(Debug)]
pub struct DecodeResult {
    pub values: [f32; MAX_FIELDS],
    pub qualities: [Quality; MAX_FIELDS],
    pub field_count: usize,
}

/// Decode a tspack frame
pub fn decode(schema: &Schema, buf: &[u8]) -> Result<DecodeResult, Error> {
    if buf.len() < 4 {
        return Err(Error::InvalidFrame);
    }

    // Verify magic and version
    if buf[0] & 0xF0 != MAGIC {
        return Err(Error::InvalidFrame);
    }

    // Parse control byte
    let has_quality = (buf[1] & 0x20) != 0;
    let schema_id = buf[1] & 0x0F;

    if schema_id != schema.id {
        return Err(Error::SchemaIdMismatch);
    }

    // Calculate sizes
    let quality_bytes = if has_quality {
        (schema.field_count() * 2 + 7) / 8
    } else {
        0
    };
    let header_size = 2;
    let payload_size = schema.payload_bytes();
    let expected_size = header_size + quality_bytes + payload_size + 1;

    if buf.len() < expected_size {
        return Err(Error::InvalidFrame);
    }

    // Verify CRC
    let crc_offset = expected_size - 1;
    let expected_crc = crc8(&buf[..crc_offset]);
    if buf[crc_offset] != expected_crc {
        return Err(Error::CrcMismatch);
    }

    let mut offset = 2;
    let mut qualities = [Quality::Valid; MAX_FIELDS];

    // Parse quality bitmap
    if has_quality {
        let mut quality_bits = 0u32;
        for i in 0..quality_bytes {
            quality_bits |= (buf[offset + i] as u32) << (i * 8);
        }
        for i in 0..schema.field_count() {
            qualities[i] = Quality::from_u8(((quality_bits >> (i * 2)) & 0x03) as u8);
        }
        offset += quality_bytes;
    }

    // Parse payload
    let mut values = [0.0f32; MAX_FIELDS];
    let mut bit_offset = 0usize;

    for i in 0..schema.field_count() {
        let field = &schema.fields[i];
        let code = unpack_bits(&buf[offset..], bit_offset, field.bits as usize);
        bit_offset += field.bits as usize;

        values[i] = if qualities[i] == Quality::Null {
            f32::NAN
        } else {
            field.dequantize(code)
        };
    }

    Ok(DecodeResult {
        values,
        qualities,
        field_count: schema.field_count(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_bits() {
        // Temperature: -40 to 80, precision 0.1 = 1201 values = 11 bits
        let temp = Field::new(-40.0, 80.0, 0.1);
        assert_eq!(temp.bits(), 11);

        // Humidity: 0 to 100, precision 1 = 101 values = 7 bits
        let humidity = Field::new(0.0, 100.0, 1.0);
        assert_eq!(humidity.bits(), 7);

        // Pressure: 300 to 1200, precision 0.1 = 9001 values = 14 bits
        let pressure = Field::new(300.0, 1200.0, 0.1);
        assert_eq!(pressure.bits(), 14);
    }

    #[test]
    fn test_roundtrip() {
        let schema = Schema::new(
            1,
            &[
                Field::new(-40.0, 80.0, 0.1),
                Field::new(0.0, 100.0, 1.0),
                Field::new(300.0, 1200.0, 0.1),
            ],
        );

        let values = [Some(23.5), Some(65.0), Some(1013.2)];
        let mut buf = [0u8; 32];

        let result = encode(&schema, &values, &mut buf, EncodeOptions::default()).unwrap();
        let decoded = decode(&schema, &buf[..result.len]).unwrap();

        // Check values are within precision
        assert!((decoded.values[0] - 23.5).abs() < 0.1);
        assert!((decoded.values[1] - 65.0).abs() < 1.0);
        assert!((decoded.values[2] - 1013.2).abs() < 0.1);
    }

    #[test]
    fn test_null_handling() {
        let schema = Schema::new(
            1,
            &[
                Field::new(-40.0, 80.0, 0.1).nullable(),
                Field::new(0.0, 100.0, 1.0).nullable(),
            ],
        );

        let values = [Some(23.5), None];
        let mut buf = [0u8; 32];

        let result = encode(&schema, &values, &mut buf, EncodeOptions::default()).unwrap();
        assert_eq!(result.qualities[1], Quality::Null);

        let decoded = decode(&schema, &buf[..result.len]).unwrap();
        assert_eq!(decoded.qualities[1], Quality::Null);
        assert!(decoded.values[1].is_nan());
    }

    #[test]
    fn test_frame_size() {
        let schema = Schema::new(
            1,
            &[
                Field::new(-40.0, 80.0, 0.1),   // 11 bits
                Field::new(0.0, 100.0, 1.0),    // 7 bits
                Field::new(300.0, 1200.0, 0.1), // 14 bits
            ],
        );

        // Total: 32 bits = 4 bytes payload
        assert_eq!(schema.payload_bits(), 32);
        assert_eq!(schema.payload_bytes(), 4);

        let values = [Some(23.5), Some(65.0), Some(1013.2)];
        let mut buf = [0u8; 32];

        let result = encode(&schema, &values, &mut buf, EncodeOptions::default()).unwrap();
        // Header (2) + Payload (4) + CRC (1) = 7 bytes
        assert_eq!(result.len, 7);
    }
}
