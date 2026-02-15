//! # SCRAWL v1.1
//!
//! **Structured Compressed Runtime for Agent-to-Agent Workload Language**
//!
//! Binary instruction set and runtime for AI agent communication.
//! 84 opcodes across 6 domains, register-based VM, delta compression
//! from the ML Identity equation, SYNAPSE binary wire format.
//!
//! ## Zero external dependencies
//!
//! This crate has no runtime dependencies. SHA-256, CRC-32C, tensor
//! math, and all other algorithms are implemented in pure Rust.
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
//! assert_eq!(vm.registers.get_reg(1), 1); // Identity verified
//! ```
//!
//! ML Innovations LLC · M. L. McKnight · Pheba, Mississippi · 2026

pub mod opcodes;
pub mod identity;
pub mod registers;
pub mod synapse;
pub mod vm;
