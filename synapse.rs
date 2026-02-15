//! SYNAPSE v1.0 — Binary Wire Format
//!
//! Frame format: Magic | Version | Flags | SeqID | Payload | CRC-32C
//! Each instruction encodes as: opcode (1 byte) + operand_count (1 byte) + operands (variable)
//!
//! ML Innovations LLC · M. L. McKnight · Pheba, Mississippi · 2026

use crate::opcodes::Opcode;

/// Magic bytes identifying a SYNAPSE frame
pub const SYNAPSE_MAGIC: [u8; 4] = [0x53, 0x43, 0x52, 0x57]; // "SCRW"
pub const SYNAPSE_VERSION: u8 = 0x10; // v1.0

/// Operand value — can be integer, float, or a list
#[derive(Debug, Clone)]
pub enum Operand {
    Int(i64),
    Float(f64),
    List(Vec<i64>),
}

/// A single SCRAWL instruction with opcode and operands
#[derive(Debug, Clone)]
pub struct Instruction {
    pub opcode: Opcode,
    pub operands: Vec<Operand>,
}

impl Instruction {
    pub fn new(opcode: Opcode, operands: Vec<Operand>) -> Self {
        Self { opcode, operands }
    }

    pub fn mnemonic(&self) -> &'static str {
        self.opcode.mnemonic()
    }
}

/// CRC-32C (Castagnoli) for frame integrity
pub fn crc32c(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0x82F63B78;
            } else {
                crc >>= 1;
            }
        }
    }
    !crc
}

/// Encode a varint (variable-length integer)
pub fn encode_varint(value: u64) -> Vec<u8> {
    let mut result = Vec::new();
    let mut v = value;
    loop {
        let mut byte = (v & 0x7F) as u8;
        v >>= 7;
        if v != 0 {
            byte |= 0x80;
        }
        result.push(byte);
        if v == 0 { break; }
    }
    result
}

/// Decode a varint, returning (value, bytes_consumed)
pub fn decode_varint(data: &[u8]) -> (u64, usize) {
    let mut value: u64 = 0;
    let mut shift = 0;
    let mut i = 0;
    loop {
        if i >= data.len() { break; }
        let byte = data[i];
        value |= ((byte & 0x7F) as u64) << shift;
        i += 1;
        if byte & 0x80 == 0 { break; }
        shift += 7;
    }
    (value, i)
}

/// SYNAPSE frame encoder
pub struct SynapseEncoder {
    sequence_id: u32,
}

impl SynapseEncoder {
    pub fn new() -> Self {
        Self { sequence_id: 0 }
    }

    /// Encode a program into a SYNAPSE frame
    pub fn encode_frame(&mut self, instructions: &[Instruction]) -> Vec<u8> {
        let mut payload = Vec::new();

        // Encode instruction count as varint
        payload.extend(encode_varint(instructions.len() as u64));

        for inst in instructions {
            // Opcode byte
            payload.push(inst.opcode.to_byte());

            // Operand count
            payload.push(inst.operands.len() as u8);

            // Operands
            for op in &inst.operands {
                match op {
                    Operand::Int(v) => {
                        payload.push(0x01); // type tag: int
                        payload.extend(encode_varint(*v as u64));
                    }
                    Operand::Float(v) => {
                        payload.push(0x02); // type tag: float
                        payload.extend(&v.to_le_bytes());
                    }
                    Operand::List(items) => {
                        payload.push(0x03); // type tag: list
                        payload.extend(encode_varint(items.len() as u64));
                        for &item in items {
                            payload.extend(encode_varint(item as u64));
                        }
                    }
                }
            }
        }

        // Build frame
        let mut frame = Vec::new();
        frame.extend(&SYNAPSE_MAGIC);
        frame.push(SYNAPSE_VERSION);
        frame.push(0x00); // flags
        frame.extend(&self.sequence_id.to_le_bytes());
        frame.extend(encode_varint(payload.len() as u64));
        frame.extend(&payload);

        // CRC-32C over everything except the CRC itself
        let checksum = crc32c(&frame);
        frame.extend(&checksum.to_le_bytes());

        self.sequence_id += 1;
        frame
    }
}

impl Default for SynapseEncoder {
    fn default() -> Self { Self::new() }
}

/// SYNAPSE frame decoder
pub struct SynapseDecoder;

impl SynapseDecoder {
    pub fn new() -> Self { Self }

    /// Decode a SYNAPSE frame into instructions
    pub fn decode_frame(&self, frame: &[u8]) -> Result<Vec<Instruction>, String> {
        if frame.len() < 14 {
            return Err("Frame too short".to_string());
        }

        // Verify magic
        if frame[0..4] != SYNAPSE_MAGIC {
            return Err("Invalid SYNAPSE magic".to_string());
        }

        // Verify CRC
        let crc_offset = frame.len() - 4;
        let stored_crc = u32::from_le_bytes([
            frame[crc_offset], frame[crc_offset + 1],
            frame[crc_offset + 2], frame[crc_offset + 3],
        ]);
        let computed_crc = crc32c(&frame[..crc_offset]);
        if stored_crc != computed_crc {
            return Err("CRC mismatch".to_string());
        }

        // Parse header
        let mut offset = 4; // skip magic
        let _version = frame[offset]; offset += 1;
        let _flags = frame[offset]; offset += 1;
        let _seq_id = u32::from_le_bytes([
            frame[offset], frame[offset + 1], frame[offset + 2], frame[offset + 3],
        ]);
        offset += 4;

        // Payload length
        let (payload_len, consumed) = decode_varint(&frame[offset..]);
        offset += consumed;
        let payload = &frame[offset..offset + payload_len as usize];

        // Decode instructions
        let mut pos = 0;
        let (inst_count, consumed) = decode_varint(payload);
        pos += consumed;

        let mut instructions = Vec::with_capacity(inst_count as usize);

        for _ in 0..inst_count {
            if pos >= payload.len() { break; }

            let opcode_byte = payload[pos]; pos += 1;
            let opcode = Opcode::from_byte(opcode_byte)
                .ok_or(format!("Unknown opcode: 0x{:02X}", opcode_byte))?;

            let op_count = payload[pos] as usize; pos += 1;

            let mut operands = Vec::with_capacity(op_count);
            for _ in 0..op_count {
                if pos >= payload.len() { break; }
                let type_tag = payload[pos]; pos += 1;
                match type_tag {
                    0x01 => { // int
                        let (v, c) = decode_varint(&payload[pos..]);
                        pos += c;
                        operands.push(Operand::Int(v as i64));
                    }
                    0x02 => { // float
                        let bytes: [u8; 8] = payload[pos..pos + 8].try_into()
                            .map_err(|_| "Truncated float")?;
                        pos += 8;
                        operands.push(Operand::Float(f64::from_le_bytes(bytes)));
                    }
                    0x03 => { // list
                        let (len, c) = decode_varint(&payload[pos..]);
                        pos += c;
                        let mut items = Vec::with_capacity(len as usize);
                        for _ in 0..len {
                            let (v, c) = decode_varint(&payload[pos..]);
                            pos += c;
                            items.push(v as i64);
                        }
                        operands.push(Operand::List(items));
                    }
                    _ => return Err(format!("Unknown operand type: 0x{:02X}", type_tag)),
                }
            }

            instructions.push(Instruction::new(opcode, operands));
        }

        Ok(instructions)
    }
}

impl Default for SynapseDecoder {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opcodes::*;

    #[test]
    fn test_varint_roundtrip() {
        for v in [0u64, 1, 127, 128, 16383, 16384, 0xCAFE, u64::MAX] {
            let encoded = encode_varint(v);
            let (decoded, _) = decode_varint(&encoded);
            assert_eq!(decoded, v);
        }
    }

    #[test]
    fn test_crc32c() {
        let crc = crc32c(b"hello");
        assert_ne!(crc, 0);
        assert_eq!(crc, crc32c(b"hello")); // deterministic
        assert_ne!(crc, crc32c(b"world")); // different data
    }

    #[test]
    fn test_frame_encode_decode() {
        let instructions = vec![
            Instruction::new(
                Opcode::Identity(IdentityOp::Derive),
                vec![Operand::Int(0), Operand::Int(0xCAFE), Operand::Int(16)],
            ),
            Instruction::new(
                Opcode::Identity(IdentityOp::Verify),
                vec![Operand::Int(0), Operand::Int(0), Operand::Int(1)],
            ),
            Instruction::new(
                Opcode::Execution(ExecutionOp::Halt),
                vec![],
            ),
        ];

        let mut encoder = SynapseEncoder::new();
        let frame = encoder.encode_frame(&instructions);

        let decoder = SynapseDecoder::new();
        let decoded = decoder.decode_frame(&frame).unwrap();

        assert_eq!(decoded.len(), instructions.len());
        assert_eq!(decoded[0].opcode, instructions[0].opcode);
        assert_eq!(decoded[2].opcode, Opcode::Execution(ExecutionOp::Halt));
    }
}
