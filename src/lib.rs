//! # SCRAWL v1.2
//!
//! **Structured Compressed Runtime for Agent-to-Agent Workload Language**
//!
//! Binary instruction set and runtime for AI agent communication.
//! 84 base opcodes across 6 domains, LEX_CALL (0x60) Darmok-style idioms,
//! register-based VM, delta compression from the ML Identity equation,
//! SYNAPSE binary wire format.
//!
//! This crate is the Rust implementation. The Python reference core
//! (`src/*.py`, ROSETTA compiler, 552 Python tests) is **not** included
//! in this repository.
//!
//! ## Zero external runtime dependencies
//!
//! SHA-256, CRC-32C, tensor math, and all other algorithms are implemented
//! in pure Rust. Criterion is a *dev-dependency* used only for `cargo bench`.
//!
//! ## Quick Start
//!
//! ```rust
//! use scrawl::vm::ScrawlVM;
//! use scrawl::synapse::{Instruction, Operand};
//! use scrawl::opcodes::*;
//!
//! let mut vm = ScrawlVM::new();
//! let result = vm.execute(&[
//!     Instruction::new(Opcode::Identity(IdentityOp::Derive),
//!         vec![Operand::Int(0), Operand::Int(0xCAFE), Operand::Int(16)]),
//!     Instruction::new(Opcode::Identity(IdentityOp::Verify),
//!         vec![Operand::Int(0), Operand::Int(0), Operand::Int(1)]),
//!     Instruction::new(Opcode::Execution(ExecutionOp::Yield),
//!         vec![Operand::Int(1)]),
//!     Instruction::new(Opcode::Execution(ExecutionOp::Halt), vec![]),
//! ]);
//! assert!(result.success);
//! assert_eq!(vm.registers.get_reg(1), 1); // Identity verified (chain recomputed)
//! ```
//!
//! ML Innovations Research Lab · M. L. McKnight · Pheba, Mississippi · 2026
//! Contributor: Korg (AI assistant, Grok Bot) — v1.2 fixes and idiom layer

pub mod opcodes;
pub mod identity;
pub mod registers;
pub mod synapse;
pub mod idioms;
pub mod vm;
