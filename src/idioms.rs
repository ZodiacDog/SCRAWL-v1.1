//! SCRAWL Idioms 1.0 — Darmok-style phrase compression.
//!
//! A short named phrase stands for a whole multi-step command sequence.
//! See `docs/IDIOMS.md` for the full specification.
//!
//! ML Innovations Research Lab · M. L. McKnight · Korg (AI assistant, Grok Bot) · 2026

use crate::identity::sha256;
use crate::opcodes::*;
use crate::synapse::{Instruction, Operand, SynapseEncoder, SYNAPSE_FLAG_LEXICON};

/// Lexicon name and version shipped with this crate.
pub const LEXICON_NAME: &str = "SCRAWL Idioms 1.0";
pub const LEXICON_VERSION: u16 = 1;

/// Built-in idiom ids. These are sticky: never reused if retired.
pub const ID_HEARTBEAT_AT_CAFE: u16 = 1;
pub const ID_NAMED_HEARTBEAT: u16 = 2;
pub const ID_THE_ROUND_AT_TWO: u16 = 3;
pub const ID_SILENT_RETRANSMIT: u16 = 4;

/// Kind of idiom: a VM instruction sequence, or a host-level behavior
/// (Silent Retransmit is a delta-compressor path, not a VM program).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdiomKind {
    VmProgram,
    HostDelta,
}

/// One entry in the standard lexicon.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Idiom {
    pub id: u16,
    pub phrase: &'static str,
    pub kind: IdiomKind,
    pub description: &'static str,
    pub version_introduced: u16,
}

/// The frozen "SCRAWL Idioms 1.0" table (16 documented phrases; four are
/// executable in this crate — the first four ids below).
pub fn standard_lexicon() -> Vec<Idiom> {
    vec![
        Idiom {
            id: ID_HEARTBEAT_AT_CAFE,
            phrase: "Heartbeat at Cafe",
            kind: IdiomKind::VmProgram,
            description: "I_DERIVE, I_VERIFY, X_YIELD, X_HALT (README quick start)",
            version_introduced: 1,
        },
        Idiom {
            id: ID_NAMED_HEARTBEAT,
            phrase: "Named Heartbeat",
            kind: IdiomKind::VmProgram,
            description: "I_DERIVE, I_VERIFY, I_FINGERPRINT, X_YIELD, X_HALT",
            version_introduced: 1,
        },
        Idiom {
            id: ID_THE_ROUND_AT_TWO,
            phrase: "The Round at Two",
            kind: IdiomKind::VmProgram,
            description: "C_PROPOSE, C_QUORUM 0.5, two APPROVE votes, C_COMMIT, X_HALT",
            version_introduced: 1,
        },
        Idiom {
            id: ID_SILENT_RETRANSMIT,
            phrase: "Silent Retransmit",
            kind: IdiomKind::HostDelta,
            description: "Second compress() of an unchanged buffer — 3 bytes when length <= 255",
            version_introduced: 1,
        },
    ]
}

/// Look up a built-in idiom by id.
pub fn lookup(id: u16) -> Option<Idiom> {
    standard_lexicon().into_iter().find(|i| i.id == id)
}

/// Look up a built-in idiom by exact phrase (after whitespace squeeze).
pub fn lookup_phrase(phrase: &str) -> Option<Idiom> {
    let norm = normalize_phrase(phrase);
    standard_lexicon()
        .into_iter()
        .find(|i| normalize_phrase(i.phrase) == norm)
}

pub fn normalize_phrase(phrase: &str) -> String {
    phrase
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// 8-byte lexicon fingerprint: SHA-256 prefix of the canonical table.
/// Agents include this in `IdentityHandshake::derive_shared_key_with_lexicon`.
pub fn lexicon_fingerprint() -> [u8; 8] {
    let mut data = Vec::new();
    data.extend_from_slice(LEXICON_NAME.as_bytes());
    data.extend_from_slice(&LEXICON_VERSION.to_le_bytes());
    for idiom in standard_lexicon() {
        data.extend_from_slice(&idiom.id.to_le_bytes());
        data.extend_from_slice(idiom.phrase.as_bytes());
        data.push(0);
        // Bind the expansion too, so a phrase with a changed body cannot
        // keep the same fingerprint.
        if idiom.kind == IdiomKind::VmProgram {
            let expanded = expand(idiom.id, &default_params(idiom.id)).unwrap_or_default();
            let mut enc = SynapseEncoder::new();
            let frame = enc.encode_frame(&expanded);
            data.extend_from_slice(&frame);
        } else {
            data.extend_from_slice(idiom.description.as_bytes());
        }
    }
    let hash = sha256(&data);
    let mut fp = [0u8; 8];
    fp.copy_from_slice(&hash[..8]);
    fp
}

fn default_params(id: u16) -> Vec<Operand> {
    match id {
        ID_HEARTBEAT_AT_CAFE => vec![Operand::Int(0xCAFE), Operand::Int(16)],
        ID_NAMED_HEARTBEAT => vec![Operand::Int(0xCAFE), Operand::Int(16)],
        ID_THE_ROUND_AT_TWO => vec![
            Operand::Int(1),
            Operand::Int(0),
            Operand::Int(0),
            Operand::Int(1),
            Operand::Float(0.5),
            Operand::Int(1),
        ],
        _ => vec![],
    }
}

/// Expand a VM idiom id plus parameters into a native instruction list.
///
/// Heartbeat at Cafe params: `[seed, depth]` (defaults 0xCAFE, 16).
/// Named Heartbeat params: `[seed, depth]`.
/// The Round at Two params: `[prop_id, data_reg, agent0, agent1, quorum, dest_reg]`.
pub fn expand(id: u16, params: &[Operand]) -> Result<Vec<Instruction>, String> {
    match id {
        ID_HEARTBEAT_AT_CAFE => {
            let seed = param_int(params, 0, 0xCAFE);
            let depth = param_int(params, 1, 16);
            Ok(heartbeat_at_cafe(seed, depth))
        }
        ID_NAMED_HEARTBEAT => {
            let seed = param_int(params, 0, 0xCAFE);
            let depth = param_int(params, 1, 16);
            Ok(named_heartbeat(seed, depth))
        }
        ID_THE_ROUND_AT_TWO => {
            let prop = param_int(params, 0, 1);
            let data_reg = param_int(params, 1, 0);
            let agent0 = param_int(params, 2, 0);
            let agent1 = param_int(params, 3, 1);
            let quorum = param_float(params, 4, 0.5);
            let dest = param_int(params, 5, 1);
            Ok(the_round_at_two(prop, data_reg, agent0, agent1, quorum, dest))
        }
        ID_SILENT_RETRANSMIT => Err(
            "Silent Retransmit is a host-level delta-compressor idiom, not a VM program".into(),
        ),
        _ => Err(format!(
            "unknown idiom id {} in {} v{}",
            id, LEXICON_NAME, LEXICON_VERSION
        )),
    }
}

fn param_int(params: &[Operand], idx: usize, default: i64) -> i64 {
    match params.get(idx) {
        Some(Operand::Int(v)) => *v,
        Some(Operand::Float(v)) => *v as i64,
        Some(Operand::List(v)) => v.first().copied().unwrap_or(default),
        None => default,
    }
}

fn param_float(params: &[Operand], idx: usize, default: f64) -> f64 {
    match params.get(idx) {
        Some(Operand::Float(v)) => *v,
        Some(Operand::Int(v)) => *v as f64,
        _ => default,
    }
}

/// README quick start: derive, verify, yield, halt.
pub fn heartbeat_at_cafe(seed: i64, depth: i64) -> Vec<Instruction> {
    vec![
        Instruction::new(
            Opcode::Identity(IdentityOp::Derive),
            vec![Operand::Int(0), Operand::Int(seed), Operand::Int(depth)],
        ),
        Instruction::new(
            Opcode::Identity(IdentityOp::Verify),
            vec![Operand::Int(0), Operand::Int(0), Operand::Int(1)],
        ),
        Instruction::new(
            Opcode::Execution(ExecutionOp::Yield),
            vec![Operand::Int(1)],
        ),
        Instruction::new(Opcode::Execution(ExecutionOp::Halt), vec![]),
    ]
}

/// Example heartbeat: derive, verify, fingerprint, yield, halt.
pub fn named_heartbeat(seed: i64, depth: i64) -> Vec<Instruction> {
    vec![
        Instruction::new(
            Opcode::Identity(IdentityOp::Derive),
            vec![Operand::Int(0), Operand::Int(seed), Operand::Int(depth)],
        ),
        Instruction::new(
            Opcode::Identity(IdentityOp::Verify),
            vec![Operand::Int(0), Operand::Int(0), Operand::Int(1)],
        ),
        Instruction::new(
            Opcode::Identity(IdentityOp::Fingerprint),
            vec![Operand::Int(2), Operand::Int(0)],
        ),
        Instruction::new(
            Opcode::Execution(ExecutionOp::Yield),
            vec![Operand::Int(1)],
        ),
        Instruction::new(Opcode::Execution(ExecutionOp::Halt), vec![]),
    ]
}

/// Two-agent propose / quorum 0.5 / two approvals / commit / halt.
/// Vote operands are (prop_id, agent_id, vote) as of v1.2.
pub fn the_round_at_two(
    prop: i64,
    data_reg: i64,
    agent0: i64,
    agent1: i64,
    quorum: f64,
    dest: i64,
) -> Vec<Instruction> {
    vec![
        Instruction::new(
            Opcode::Consensus(ConsensusOp::Propose),
            vec![
                Operand::Int(prop),
                Operand::Int(data_reg),
                Operand::List(vec![agent0, agent1]),
            ],
        ),
        Instruction::new(
            Opcode::Consensus(ConsensusOp::Quorum),
            vec![Operand::Int(prop), Operand::Float(quorum)],
        ),
        Instruction::new(
            Opcode::Consensus(ConsensusOp::Vote),
            vec![Operand::Int(prop), Operand::Int(agent0), Operand::Int(0)],
        ),
        Instruction::new(
            Opcode::Consensus(ConsensusOp::Vote),
            vec![Operand::Int(prop), Operand::Int(agent1), Operand::Int(0)],
        ),
        Instruction::new(
            Opcode::Consensus(ConsensusOp::Commit),
            vec![Operand::Int(prop), Operand::Int(dest)],
        ),
        Instruction::new(Opcode::Execution(ExecutionOp::Halt), vec![]),
    ]
}

/// Build a LEX_CALL instruction.
pub fn lex_call(id: u16, params: Vec<Operand>) -> Instruction {
    let mut ops = vec![Operand::Int(id as i64)];
    ops.extend(params);
    Instruction::new(Opcode::Lexicon(LexiconOp::Call), ops)
}

/// Greedy longest-match packer: replace known sequences with LEX_CALL.
pub fn pack_idioms(instructions: &[Instruction]) -> Vec<Instruction> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < instructions.len() {
        if let Some((n, call)) = match_at(instructions, i) {
            out.push(call);
            i += n;
        } else {
            out.push(instructions[i].clone());
            i += 1;
        }
    }
    out
}

fn match_at(ins: &[Instruction], i: usize) -> Option<(usize, Instruction)> {
    // Longest first: Named Heartbeat (5) before Heartbeat at Cafe (4).
    if let Some(params) = match_named_heartbeat(&ins[i..]) {
        return Some((5, lex_call(ID_NAMED_HEARTBEAT, params)));
    }
    if let Some(params) = match_heartbeat_at_cafe(&ins[i..]) {
        return Some((4, lex_call(ID_HEARTBEAT_AT_CAFE, params)));
    }
    if let Some(params) = match_round_at_two(&ins[i..]) {
        return Some((6, lex_call(ID_THE_ROUND_AT_TWO, params)));
    }
    None
}

fn is_op(inst: &Instruction, expected: Opcode) -> bool {
    inst.opcode == expected
}

fn int_at(inst: &Instruction, idx: usize) -> Option<i64> {
    match inst.operands.get(idx) {
        Some(Operand::Int(v)) => Some(*v),
        _ => None,
    }
}

fn match_named_heartbeat(ins: &[Instruction]) -> Option<Vec<Operand>> {
    if ins.len() < 5 {
        return None;
    }
    if !is_op(&ins[0], Opcode::Identity(IdentityOp::Derive)) {
        return None;
    }
    if !is_op(&ins[1], Opcode::Identity(IdentityOp::Verify)) {
        return None;
    }
    if !is_op(&ins[2], Opcode::Identity(IdentityOp::Fingerprint)) {
        return None;
    }
    if !is_op(&ins[3], Opcode::Execution(ExecutionOp::Yield)) {
        return None;
    }
    if !is_op(&ins[4], Opcode::Execution(ExecutionOp::Halt)) {
        return None;
    }
    let seed = int_at(&ins[0], 1)?;
    let depth = int_at(&ins[0], 2)?;
    Some(vec![Operand::Int(seed), Operand::Int(depth)])
}

fn match_heartbeat_at_cafe(ins: &[Instruction]) -> Option<Vec<Operand>> {
    if ins.len() < 4 {
        return None;
    }
    if !is_op(&ins[0], Opcode::Identity(IdentityOp::Derive)) {
        return None;
    }
    if !is_op(&ins[1], Opcode::Identity(IdentityOp::Verify)) {
        return None;
    }
    if !is_op(&ins[2], Opcode::Execution(ExecutionOp::Yield)) {
        return None;
    }
    if !is_op(&ins[3], Opcode::Execution(ExecutionOp::Halt)) {
        return None;
    }
    let seed = int_at(&ins[0], 1)?;
    let depth = int_at(&ins[0], 2)?;
    Some(vec![Operand::Int(seed), Operand::Int(depth)])
}

fn match_round_at_two(ins: &[Instruction]) -> Option<Vec<Operand>> {
    if ins.len() < 6 {
        return None;
    }
    if !is_op(&ins[0], Opcode::Consensus(ConsensusOp::Propose)) {
        return None;
    }
    if !is_op(&ins[1], Opcode::Consensus(ConsensusOp::Quorum)) {
        return None;
    }
    if !is_op(&ins[2], Opcode::Consensus(ConsensusOp::Vote)) {
        return None;
    }
    if !is_op(&ins[3], Opcode::Consensus(ConsensusOp::Vote)) {
        return None;
    }
    if !is_op(&ins[4], Opcode::Consensus(ConsensusOp::Commit)) {
        return None;
    }
    if !is_op(&ins[5], Opcode::Execution(ExecutionOp::Halt)) {
        return None;
    }
    let prop = int_at(&ins[0], 0)?;
    let data_reg = int_at(&ins[0], 1)?;
    let agents = match ins[0].operands.get(2) {
        Some(Operand::List(v)) if v.len() == 2 => v.clone(),
        _ => return None,
    };
    let quorum = match ins[1].operands.get(1) {
        Some(Operand::Float(v)) => *v,
        Some(Operand::Int(v)) => *v as f64,
        _ => return None,
    };
    let dest = int_at(&ins[4], 1)?;
    Some(vec![
        Operand::Int(prop),
        Operand::Int(data_reg),
        Operand::Int(agents[0]),
        Operand::Int(agents[1]),
        Operand::Float(quorum),
        Operand::Int(dest),
    ])
}

/// Encode a (possibly packed) program. Sets the SYNAPSE lexicon flag if any
/// LEX_CALL instruction is present.
pub fn encode_with_idioms(instructions: &[Instruction]) -> Vec<u8> {
    let packed = pack_idioms(instructions);
    let has_lex = packed
        .iter()
        .any(|i| i.opcode == Opcode::Lexicon(LexiconOp::Call));
    let mut flags = 0u8;
    if has_lex {
        flags |= SYNAPSE_FLAG_LEXICON;
    }
    let mut enc = SynapseEncoder::with_flags(flags);
    enc.encode_frame(&packed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::identity::{DeltaCompressor, IdentityBaseline, IdentityHandshake};
    use crate::synapse::SynapseDecoder;
    use crate::vm::ScrawlVM;

    #[test]
    fn lexicon_fingerprint_is_stable() {
        let a = lexicon_fingerprint();
        let b = lexicon_fingerprint();
        assert_eq!(a, b);
        assert_ne!(a, [0u8; 8]);
    }

    #[test]
    fn lookup_and_normalize() {
        assert_eq!(
            lookup(ID_HEARTBEAT_AT_CAFE).unwrap().phrase,
            "Heartbeat at Cafe"
        );
        assert!(lookup_phrase("  heartbeat   at   CAFE ").is_some());
        assert!(lookup_phrase("no such idiom").is_none());
        assert!(lookup(99).is_none());
    }

    #[test]
    fn expand_unknown_idiom_errors() {
        let err = expand(99, &[]).unwrap_err();
        assert!(err.contains("unknown idiom"));
    }

    #[test]
    fn silent_retransmit_is_host_level() {
        assert!(expand(ID_SILENT_RETRANSMIT, &[]).is_err());
        let baseline = IdentityBaseline::new(0xBEEF, 16);
        let mut c = DeltaCompressor::new(baseline);
        let payload = vec![b'x'; 50];
        let first = c.compress(&payload);
        let second = c.compress(&payload);
        assert_eq!(first.len(), 51);
        assert_eq!(second.len(), 3, "identical retransmit must be 3 bytes");
    }

    #[test]
    fn pack_and_expand_heartbeat_roundtrip() {
        let original = heartbeat_at_cafe(0xCAFE, 16);
        let packed = pack_idioms(&original);
        assert_eq!(packed.len(), 1);
        assert_eq!(packed[0].opcode, Opcode::Lexicon(LexiconOp::Call));
        let expanded = expand(ID_HEARTBEAT_AT_CAFE, &packed[0].operands[1..]).unwrap();
        assert_eq!(expanded.len(), original.len());
        for (a, b) in original.iter().zip(expanded.iter()) {
            assert_eq!(a.opcode, b.opcode);
        }
    }

    #[test]
    fn named_heartbeat_preferred_over_cafe_prefix() {
        let original = named_heartbeat(0xCAFE, 16);
        let packed = pack_idioms(&original);
        assert_eq!(packed.len(), 1);
        match packed[0].operands[0] {
            Operand::Int(id) => assert_eq!(id, ID_NAMED_HEARTBEAT as i64),
            _ => panic!("expected idiom id"),
        }
    }

    #[test]
    fn vm_executes_lex_call_heartbeat() {
        let mut vm = ScrawlVM::new();
        let program = vec![lex_call(
            ID_HEARTBEAT_AT_CAFE,
            vec![Operand::Int(0xCAFE), Operand::Int(16)],
        )];
        let result = vm.execute(&program);
        assert!(result.success, "{}", result.error_message);
        assert_eq!(vm.registers.get_reg(1), 1);
        assert_eq!(result.yielded_values, vec![1]);
    }

    #[test]
    fn vm_traps_unknown_lex_call() {
        let mut vm = ScrawlVM::new();
        let result = vm.execute(&[lex_call(99, vec![])]);
        assert!(!result.success);
        assert!(result.error_message.contains("unknown idiom"));
    }

    #[test]
    fn handshake_includes_lexicon_fingerprint() {
        let (b, fp) = IdentityHandshake::initiate(0xBEEF, 16);
        let (_b2, ok) = IdentityHandshake::respond(0xBEEF, 16, &fp);
        assert!(ok);
        let lex = lexicon_fingerprint();
        let k1 = IdentityHandshake::derive_shared_key_with_lexicon(&b, 0, 1, &lex);
        let k2 = IdentityHandshake::derive_shared_key_with_lexicon(&b, 0, 1, &lex);
        assert_eq!(k1, k2);
        let k3 = IdentityHandshake::derive_shared_key_with_lexicon(&b, 0, 1, &[0xFF; 8]);
        assert_ne!(k1, k3, "mismatched lexicon fingerprint must change the key");
    }

    #[test]
    fn frame_with_lex_call_sets_flag_and_is_smaller() {
        let expanded = heartbeat_at_cafe(0xCAFE, 16);
        let mut plain = SynapseEncoder::new();
        let expanded_frame = plain.encode_frame(&expanded);
        let packed_frame = encode_with_idioms(&expanded);
        assert!(
            packed_frame.len() < expanded_frame.len(),
            "packed {}B should be smaller than expanded {}B",
            packed_frame.len(),
            expanded_frame.len()
        );
        assert_eq!(packed_frame[5] & SYNAPSE_FLAG_LEXICON, SYNAPSE_FLAG_LEXICON);
        let decoded = SynapseDecoder::new().decode_frame(&packed_frame).unwrap();
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].opcode, Opcode::Lexicon(LexiconOp::Call));
    }

    #[test]
    fn phrase_collision_rejected_on_normalize() {
        // Two distinct entries must not share a normalized phrase.
        let phrases: Vec<String> = standard_lexicon()
            .into_iter()
            .map(|i| normalize_phrase(i.phrase))
            .collect();
        let mut unique = phrases.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), phrases.len());
    }
}
