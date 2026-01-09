//! Bit-packing utilities for tspack

/// Pack a value into a byte buffer at a specific bit offset
///
/// # Arguments
/// * `buf` - The buffer to write to
/// * `bit_offset` - The starting bit position within the buffer
/// * `bits` - The number of bits to pack (1-32)
/// * `value` - The value to pack
#[inline]
pub fn pack_bits(buf: &mut [u8], bit_offset: usize, bits: usize, value: u32) {
    debug_assert!(bits <= 32);
    debug_assert!(bits > 0);

    let byte_offset = bit_offset / 8;
    let bit_shift = bit_offset % 8;

    // Mask the value to the specified number of bits
    let mask = if bits == 32 { u32::MAX } else { (1u32 << bits) - 1 };
    let value = value & mask;

    // We might need to write to up to 5 bytes for a 32-bit value
    // spanning a byte boundary
    let mut remaining_bits = bits;
    let mut current_byte = byte_offset;
    let mut shift = bit_shift;
    let mut val = value;

    while remaining_bits > 0 {
        let bits_in_this_byte = (8 - shift).min(remaining_bits);
        let byte_mask = ((1u32 << bits_in_this_byte) - 1) as u8;

        buf[current_byte] |= ((val as u8) & byte_mask) << shift;

        val >>= bits_in_this_byte;
        remaining_bits -= bits_in_this_byte;
        current_byte += 1;
        shift = 0;
    }
}

/// Unpack a value from a byte buffer at a specific bit offset
///
/// # Arguments
/// * `buf` - The buffer to read from
/// * `bit_offset` - The starting bit position within the buffer
/// * `bits` - The number of bits to unpack (1-32)
///
/// # Returns
/// The unpacked value
#[inline]
pub fn unpack_bits(buf: &[u8], bit_offset: usize, bits: usize) -> u32 {
    debug_assert!(bits <= 32);
    debug_assert!(bits > 0);

    let byte_offset = bit_offset / 8;
    let bit_shift = bit_offset % 8;

    let mut result = 0u32;
    let mut remaining_bits = bits;
    let mut current_byte = byte_offset;
    let mut shift = bit_shift;
    let mut result_shift = 0;

    while remaining_bits > 0 {
        let bits_in_this_byte = (8 - shift).min(remaining_bits);
        let byte_mask = ((1u32 << bits_in_this_byte) - 1) as u8;

        let byte_val = (buf[current_byte] >> shift) & byte_mask;
        result |= (byte_val as u32) << result_shift;

        result_shift += bits_in_this_byte;
        remaining_bits -= bits_in_this_byte;
        current_byte += 1;
        shift = 0;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_aligned() {
        let mut buf = [0u8; 8];

        // Pack 8 bits at offset 0
        pack_bits(&mut buf, 0, 8, 0xAB);
        assert_eq!(buf[0], 0xAB);
        assert_eq!(unpack_bits(&buf, 0, 8), 0xAB);

        // Pack 16 bits at offset 8
        pack_bits(&mut buf, 8, 16, 0x1234);
        assert_eq!(unpack_bits(&buf, 8, 16), 0x1234);
    }

    #[test]
    fn test_pack_unpack_unaligned() {
        let mut buf = [0u8; 8];

        // Pack 11 bits at offset 0
        pack_bits(&mut buf, 0, 11, 0x7FF);
        assert_eq!(unpack_bits(&buf, 0, 11), 0x7FF);

        // Pack 7 bits at offset 11
        pack_bits(&mut buf, 11, 7, 0x55);
        assert_eq!(unpack_bits(&buf, 11, 7), 0x55);

        // Verify the first value is still correct
        assert_eq!(unpack_bits(&buf, 0, 11), 0x7FF);
    }

    #[test]
    fn test_pack_multiple_fields() {
        let mut buf = [0u8; 8];

        // Simulate packing 3 fields: 11, 7, 14 bits
        pack_bits(&mut buf, 0, 11, 635);   // temperature
        pack_bits(&mut buf, 11, 7, 65);    // humidity
        pack_bits(&mut buf, 18, 14, 7132); // pressure

        assert_eq!(unpack_bits(&buf, 0, 11), 635);
        assert_eq!(unpack_bits(&buf, 11, 7), 65);
        assert_eq!(unpack_bits(&buf, 18, 14), 7132);
    }
}
