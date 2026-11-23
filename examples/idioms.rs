//! SCRAWL Idioms 1.0 — pack, execute, and measure Darmok-style compression.
//!
//! Usage: cargo run --example idioms

use scrawl::idioms::*;
use scrawl::identity::{DeltaCompressor, IdentityBaseline, IdentityHandshake};
use scrawl::synapse::SynapseEncoder;
use scrawl::vm::ScrawlVM;

fn main() {
    println!("=== SCRAWL Idioms 1.0 ===\n");
    println!("  Lexicon: {} v{}", LEXICON_NAME, LEXICON_VERSION);
    println!("  Fingerprint: {:02x?}", lexicon_fingerprint());
    println!();

    for idiom in standard_lexicon() {
        println!("  [{:02}] {} — {}", idiom.id, idiom.phrase, idiom.description);
    }

    // Heartbeat at Cafe: expanded vs LEX_CALL frame sizes
    let expanded = heartbeat_at_cafe(0xCAFE, 16);
    let mut enc = SynapseEncoder::new();
    let expanded_frame = enc.encode_frame(&expanded);
    let packed_frame = encode_with_idioms(&expanded);

    println!("\n  Heartbeat at Cafe:");
    println!("    Expanded SYNAPSE frame: {:>3} B  ({} instructions)",
        expanded_frame.len(), expanded.len());
    println!("    LEX_CALL frame:         {:>3} B  (1 instruction)", packed_frame.len());
    let saved = expanded_frame.len() as i64 - packed_frame.len() as i64;
    let pct = 100.0 * saved as f64 / expanded_frame.len() as f64;
    println!("    Savings:                {:>3} B  ({:.1}%)", saved, pct);

    let mut vm = ScrawlVM::new();
    let result = vm.execute(&[lex_call(ID_HEARTBEAT_AT_CAFE, vec![
        scrawl::synapse::Operand::Int(0xCAFE),
        scrawl::synapse::Operand::Int(16),
    ])]);
    println!("    VM: success={}, verified={}, yielded={:?}",
        result.success, vm.registers.get_reg(1) == 1, result.yielded_values);

    // Named Heartbeat
    let named = named_heartbeat(0xCAFE, 16);
    let mut enc = SynapseEncoder::new();
    let named_exp = enc.encode_frame(&named);
    let named_pack = encode_with_idioms(&named);
    println!("\n  Named Heartbeat:");
    println!("    Expanded: {:>3} B  →  LEX_CALL: {:>3} B  ({:.1}% smaller)",
        named_exp.len(), named_pack.len(),
        100.0 * (named_exp.len() - named_pack.len()) as f64 / named_exp.len() as f64);

    // The Round at Two
    let round = the_round_at_two(1, 0, 0, 1, 0.5, 1);
    let mut enc = SynapseEncoder::new();
    let round_exp = enc.encode_frame(&round);
    let round_pack = encode_with_idioms(&round);
    println!("\n  The Round at Two:");
    println!("    Expanded: {:>3} B  →  LEX_CALL: {:>3} B  ({:.1}% smaller)",
        round_exp.len(), round_pack.len(),
        100.0 * (round_exp.len() - round_pack.len()) as f64 / round_exp.len() as f64);
    let mut vm2 = ScrawlVM::new();
    let r2 = vm2.execute(&pack_idioms(&round));
    println!("    VM: success={}, committed={}", r2.success, vm2.registers.get_reg(1) == 1);

    // Silent Retransmit
    let mut c = DeltaCompressor::new(IdentityBaseline::new(0xBEEF, 16));
    let payload = vec![b'x'; 50];
    let first = c.compress(&payload);
    let second = c.compress(&payload);
    println!("\n  Silent Retransmit (50-byte payload):");
    println!("    First send:  {} B", first.len());
    println!("    Second send: {} B  (identical retransmit)", second.len());

    // Handshake with lexicon fingerprint
    let (baseline, fp) = IdentityHandshake::initiate(0xBEEF, 16);
    let (_, matched) = IdentityHandshake::respond(0xBEEF, 16, &fp);
    let key = IdentityHandshake::derive_shared_key_with_lexicon(
        &baseline, 0, 1, &lexicon_fingerprint(),
    );
    println!("\n  Handshake: identity match={}, shared key {} bytes (lexicon-bound)",
        matched, key.len());

    println!("\n=== Idioms OK ===");
}
