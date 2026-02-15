//! SCRAWL Agent Heartbeat — Rust
//!
//! The simplest SCRAWL program: derive identity, verify, fingerprint, halt.
//! If this runs, your Rust stack works.
//!
//! Usage: cargo run --example heartbeat

use scrawl::identity::{ml_identity, ml_identity_verify};
use scrawl::vm::ScrawlVM;
use scrawl::synapse::{Instruction, Operand};
use scrawl::opcodes::*;

fn main() {
    println!("=== SCRAWL Agent Heartbeat (Rust) ===\n");

    // 1. Verify the ML Identity holds (pure math)
    for &a in &[1i64, 5, 10, 42, 1000] {
        let (b, lhs) = ml_identity(a);
        assert_eq!(lhs, b * b, "Identity broken at a={}", a);
        println!("  ML Identity: {} + {} + {} = {}  ✓", a, a * a, b, b * b);
    }

    // 2. Build and execute a SCRAWL program
    let program = vec![
        Instruction::new(Opcode::Identity(IdentityOp::Derive),
            vec![Operand::Int(0), Operand::Int(0xCAFE), Operand::Int(16)]),
        Instruction::new(Opcode::Identity(IdentityOp::Verify),
            vec![Operand::Int(0), Operand::Int(0), Operand::Int(1)]),
        Instruction::new(Opcode::Identity(IdentityOp::Fingerprint),
            vec![Operand::Int(2), Operand::Int(0)]),
        Instruction::new(Opcode::Execution(ExecutionOp::Yield),
            vec![Operand::Int(1)]),
        Instruction::new(Opcode::Execution(ExecutionOp::Halt), vec![]),
    ];

    let mut vm = ScrawlVM::new();
    let result = vm.execute(&program);

    println!("\n  VM executed {} instructions in {:.3}ms",
        result.instructions_executed, result.execution_time_ms);
    println!("  Identity verified: {}",
        if vm.registers.get_reg(1) == 1 { "YES" } else { "NO" });
    println!("  Fingerprint: 0x{:016X}", vm.registers.get_reg(2));
    println!("  Yielded: {:?}", result.yielded_values);

    // 3. Show baseline info
    if let Some(baseline) = vm.baselines.get(&0) {
        println!("\n  Baseline: seed=0x{:04X}, depth={}", baseline.seed, baseline.depth);
        println!("  Chain (first 5): {:?}", &baseline.chain[..5.min(baseline.chain.len())]);
    }

    println!("\n=== Heartbeat OK ===");
}
