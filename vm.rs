//! SCRAWL v1.1 Virtual Machine
//!
//! Register-based execution engine for SCRAWL instruction streams.
//! v1.1: Consensus trace hooks, per-instruction timeout, algebraic verification.
//!
//! ML Innovations LLC · M. L. McKnight · Pheba, Mississippi · 2026

use std::collections::HashMap;
use std::time::Instant;

use crate::identity::{IdentityBaseline, ml_identity_verify, ml_algebraic_verify};
use crate::opcodes::*;
use crate::registers::{RegisterFile, Tensor};
use crate::synapse::{Instruction, Operand};

// ═══════════════════════════════════════════════════════════════
// TRACE EVENT SYSTEM (v1.1)
// ═══════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum TraceSeverity {
    Debug    = 0,
    Info     = 1,
    Warn     = 2,
    Error    = 3,
    Critical = 4,
}

#[derive(Debug, Clone)]
pub struct TraceEvent {
    pub severity: TraceSeverity,
    pub domain: String,
    pub event_type: String,
    pub message: String,
    pub instruction_pc: i64,
}

impl TraceEvent {
    pub fn new(severity: TraceSeverity, domain: &str, event_type: &str, message: &str) -> Self {
        Self {
            severity,
            domain: domain.to_string(),
            event_type: event_type.to_string(),
            message: message.to_string(),
            instruction_pc: -1,
        }
    }
}

impl std::fmt::Display for TraceEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{:?}] {}.{}: {}", self.severity, self.domain, self.event_type, self.message)
    }
}

// ═══════════════════════════════════════════════════════════════
// EXECUTION RESULT
// ═══════════════════════════════════════════════════════════════

#[derive(Debug)]
pub struct ExecutionResult {
    pub success: bool,
    pub error_code: u32,
    pub error_message: String,
    pub instructions_executed: u64,
    pub execution_time_ms: f64,
    pub yielded_values: Vec<u64>,
    pub halted: bool,
    pub trace_events: Vec<TraceEvent>,
}

impl ExecutionResult {
    fn new() -> Self {
        Self {
            success: true,
            error_code: 0,
            error_message: String::new(),
            instructions_executed: 0,
            execution_time_ms: 0.0,
            yielded_values: Vec::new(),
            halted: false,
            trace_events: Vec::new(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// VIRTUAL MACHINE
// ═══════════════════════════════════════════════════════════════

/// Helper to extract an i64 from an Operand
fn op_int(operands: &[Operand], idx: usize) -> i64 {
    match &operands[idx] {
        Operand::Int(v) => *v,
        Operand::Float(v) => *v as i64,
        Operand::List(v) => v.first().copied().unwrap_or(0),
    }
}

fn op_float(operands: &[Operand], idx: usize) -> f64 {
    match &operands[idx] {
        Operand::Float(v) => *v,
        Operand::Int(v) => *v as f64,
        Operand::List(_) => 0.0,
    }
}

fn op_list(operands: &[Operand], idx: usize) -> Vec<i64> {
    match &operands[idx] {
        Operand::List(v) => v.clone(),
        Operand::Int(v) => vec![*v],
        _ => vec![],
    }
}

pub struct ScrawlVM {
    pub registers: RegisterFile,
    pub baselines: HashMap<u8, IdentityBaseline>,
    pub proposals: HashMap<i64, Proposal>,
    pub snapshots: HashMap<i64, u64>,
    pub locks: HashMap<i64, bool>,
    pub agent_id: u32,
    pub max_instructions: u64,
    pub instruction_timeout_secs: f64,
    trace_events: Vec<TraceEvent>,
    trace_hooks: Vec<Box<dyn Fn(&TraceEvent)>>,
}

pub struct Proposal {
    pub data: u64,
    pub agents: Vec<i64>,
    pub votes: HashMap<u32, i64>,
    pub committed: bool,
    pub quorum: f64,
}

impl ScrawlVM {
    pub fn new() -> Self {
        Self {
            registers: RegisterFile::new(),
            baselines: HashMap::new(),
            proposals: HashMap::new(),
            snapshots: HashMap::new(),
            locks: HashMap::new(),
            agent_id: 0,
            max_instructions: 1_000_000,
            instruction_timeout_secs: 5.0,
            trace_events: Vec::new(),
            trace_hooks: Vec::new(),
        }
    }

    pub fn add_trace_hook<F: Fn(&TraceEvent) + 'static>(&mut self, hook: F) {
        self.trace_hooks.push(Box::new(hook));
    }

    pub fn get_trace_events(&self, min_severity: TraceSeverity) -> Vec<&TraceEvent> {
        self.trace_events.iter().filter(|e| e.severity >= min_severity).collect()
    }

    fn emit_trace(&mut self, severity: TraceSeverity, domain: &str, event_type: &str, message: &str) {
        let mut event = TraceEvent::new(severity, domain, event_type, message);
        event.instruction_pc = self.registers.pc as i64;
        self.trace_events.push(event.clone());
        for hook in &self.trace_hooks {
            hook(&event);
        }
    }

    pub fn execute(&mut self, instructions: &[Instruction]) -> ExecutionResult {
        let mut result = ExecutionResult::new();
        let start = Instant::now();

        self.registers.pc = 0;
        self.registers.halted = false;

        while self.registers.pc < instructions.len()
            && !self.registers.halted
            && result.instructions_executed < self.max_instructions
        {
            let inst = &instructions[self.registers.pc];
            self.execute_instruction(inst, &mut result);
            result.instructions_executed += 1;

            if !self.registers.halted {
                self.registers.pc += 1;
            }
        }

        result.halted = self.registers.halted;
        result.execution_time_ms = start.elapsed().as_secs_f64() * 1000.0;
        result.trace_events = self.trace_events.clone();
        result
    }

    fn execute_instruction(&mut self, inst: &Instruction, result: &mut ExecutionResult) {
        let ops = &inst.operands;

        match inst.opcode {
            // ── Execution Control ──
            Opcode::Execution(ExecutionOp::Nop) => {}
            Opcode::Execution(ExecutionOp::Halt) => {
                self.registers.halted = true;
            }
            Opcode::Execution(ExecutionOp::Yield) => {
                let val = self.registers.get_reg(op_int(ops, 0) as u8);
                result.yielded_values.push(val);
            }
            Opcode::Execution(ExecutionOp::Return) => {
                let val = self.registers.get_reg(op_int(ops, 0) as u8);
                result.yielded_values.push(val);
                self.registers.halted = true;
            }

            // ── Identity ──
            Opcode::Identity(IdentityOp::Derive) => {
                let cr_dst = op_int(ops, 0) as u8;
                let seed = op_int(ops, 1) as u64;
                let depth = op_int(ops, 2) as usize;
                let baseline = IdentityBaseline::new(seed, depth);
                let fp = baseline.fingerprint.to_vec();
                self.baselines.insert(cr_dst, baseline);
                self.registers.set_creg(cr_dst, fp);
                self.emit_trace(
                    TraceSeverity::Debug, "identity", "baseline_derived",
                    &format!("CR{} = IdentityBaseline(seed=0x{:04X}, depth={})", cr_dst, seed, depth),
                );
            }
            Opcode::Identity(IdentityOp::Verify) => {
                let cr_src = op_int(ops, 0) as u8;
                let _reg_data = op_int(ops, 1) as u8;
                let reg_result = op_int(ops, 2) as u8;
                if let Some(baseline) = self.baselines.get(&cr_src) {
                    let valid = baseline.chain.iter()
                        .filter(|&&a| a > 0)
                        .all(|&a| ml_identity_verify(a as i64));
                    self.registers.set_reg(reg_result, if valid { 1 } else { 0 });
                } else {
                    self.registers.set_reg(reg_result, 0);
                    self.emit_trace(TraceSeverity::Warn, "identity", "missing_baseline",
                        &format!("I_VERIFY: no baseline in CR{}", cr_src));
                }
            }
            Opcode::Identity(IdentityOp::Fingerprint) => {
                let reg_dst = op_int(ops, 0) as u8;
                let cr_src = op_int(ops, 1) as u8;
                if let Some(baseline) = self.baselines.get(&cr_src) {
                    let fp = u64::from_le_bytes(baseline.fingerprint);
                    self.registers.set_reg(reg_dst, fp);
                }
            }

            // ── Consensus (v1.1: trace hooks) ──
            Opcode::Consensus(ConsensusOp::Propose) => {
                let prop_id = op_int(ops, 0);
                let data = self.registers.get_reg(op_int(ops, 1) as u8);
                let agents = op_list(ops, 2);
                self.proposals.insert(prop_id, Proposal {
                    data, agents: agents.clone(), votes: HashMap::new(),
                    committed: false, quorum: 0.5,
                });
                self.emit_trace(TraceSeverity::Info, "consensus", "proposal_created",
                    &format!("Proposal {}: agents={:?}", prop_id, agents));
            }
            Opcode::Consensus(ConsensusOp::Vote) => {
                let prop_id = op_int(ops, 0);
                let vote_value = op_int(ops, 1);
                if let Some(proposal) = self.proposals.get_mut(&prop_id) {
                    proposal.votes.insert(self.agent_id, vote_value);
                    if vote_value == 1 {
                        self.emit_trace(TraceSeverity::Warn, "consensus", "vote_rejected",
                            &format!("Agent {} REJECTED proposal {}", self.agent_id, prop_id));
                    }
                }
            }
            Opcode::Consensus(ConsensusOp::Quorum) => {
                let prop_id = op_int(ops, 0);
                let threshold = op_float(ops, 1);
                if let Some(proposal) = self.proposals.get_mut(&prop_id) {
                    proposal.quorum = threshold;
                }
            }
            Opcode::Consensus(ConsensusOp::Commit) => {
                let prop_id = op_int(ops, 0);
                let dest_reg = op_int(ops, 1) as u8;
                if let Some(proposal) = self.proposals.get_mut(&prop_id) {
                    let agents_len = proposal.agents.len();
                    let required = (agents_len as f64 * proposal.quorum).ceil().max(1.0) as usize;
                    let approve_count = proposal.votes.values().filter(|&&v| v == 0).count();

                    if approve_count >= required {
                        // v1.1: Algebraic verification
                        let data_val = proposal.data;
                        if data_val > 0 {
                            let a = data_val & 0xFFFF;
                            let a_sq = a * a;
                            let b = a + 1;
                            let b_sq = b * b;
                            if !ml_algebraic_verify(a, a_sq, b, b_sq) {
                                self.emit_trace(TraceSeverity::Critical, "consensus",
                                    "algebraic_integrity_failure",
                                    &format!("ML Identity check FAILED for proposal {}", prop_id));
                            }
                        }
                        proposal.committed = true;
                        self.registers.set_reg(dest_reg, 1);
                        self.emit_trace(TraceSeverity::Info, "consensus", "committed",
                            &format!("Proposal {} committed", prop_id));
                    } else {
                        self.registers.set_reg(dest_reg, 0);
                        self.emit_trace(TraceSeverity::Warn, "consensus", "quorum_not_met",
                            &format!("Proposal {}: {}/{} approvals", prop_id, approve_count, required));
                    }
                }
            }

            // ── Tensor: Fill ──
            Opcode::Tensor(TensorOp::Fill) => {
                let dst = op_int(ops, 0) as u8;
                let value = op_float(ops, 1);
                if let Some(existing) = self.registers.get_treg(dst) {
                    let shape = existing.shape.clone();
                    self.registers.set_treg(dst, Tensor::fill(shape, value));
                } else {
                    self.registers.set_treg(dst, Tensor::fill(vec![1], value));
                }
            }

            // ── Tensor: Norm ──
            Opcode::Tensor(TensorOp::Norm) => {
                let dst = op_int(ops, 0) as u8;
                let src = op_int(ops, 1) as u8;
                if let Some(t) = self.registers.get_treg(src).cloned() {
                    self.registers.set_treg(dst, t.l2_normalize());
                }
            }

            // ── Attention: Route (QKV) ──
            Opcode::Attention(AttentionOp::Route) => {
                let q_idx = op_int(ops, 0) as u8;
                let k_idx = op_int(ops, 1) as u8;
                let v_idx = op_int(ops, 2) as u8;
                let dst_idx = op_int(ops, 3) as u8;
                self.exec_attention_route(q_idx, k_idx, v_idx, dst_idx);
            }

            // Default: NOP for unimplemented opcodes
            _ => {}
        }
    }

    fn exec_attention_route(&mut self, q_idx: u8, k_idx: u8, v_idx: u8, dst_idx: u8) {
        let q = match self.registers.get_treg(q_idx) { Some(t) => t.clone(), None => return };
        let k = match self.registers.get_treg(k_idx) { Some(t) => t.clone(), None => return };
        let v = match self.registers.get_treg(v_idx) { Some(t) => t.clone(), None => return };

        if q.ndim() == 2 && k.ndim() == 2 && v.ndim() == 2 {
            let d_k = k.shape[1];
            let scale = 1.0 / (d_k as f64).sqrt();

            // Transpose K
            let (m, d) = (k.shape[0], k.shape[1]);
            let mut k_t_data = vec![0.0; d * m];
            for i in 0..m {
                for j in 0..d {
                    k_t_data[j * m + i] = k.data[i * d + j];
                }
            }
            let k_t = Tensor::new(k_t_data, vec![d, m]).unwrap();

            // Q @ K^T
            let scores = q.dot(&k_t).unwrap().scale(scale);

            // Softmax per row
            let (rows, cols) = (scores.shape[0], scores.shape[1]);
            let mut sm_data = scores.data.clone();
            for i in 0..rows {
                let row = &sm_data[i * cols..(i + 1) * cols];
                let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let mut exp_row: Vec<f64> = row.iter().map(|x| (x - max_val).exp()).collect();
                let sum_exp: f64 = exp_row.iter().sum();
                if sum_exp > 0.0 {
                    for x in exp_row.iter_mut() { *x /= sum_exp; }
                }
                sm_data[i * cols..(i + 1) * cols].copy_from_slice(&exp_row);
            }
            let attn_weights = Tensor::new(sm_data, vec![rows, cols]).unwrap();

            // Weights @ V
            if let Ok(result) = attn_weights.dot(&v) {
                self.registers.set_treg(dst_idx, result);
            }
        }
    }
}

impl Default for ScrawlVM {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn halt() -> Instruction {
        Instruction::new(Opcode::Execution(ExecutionOp::Halt), vec![])
    }

    #[test]
    fn test_identity_derive_verify() {
        let mut vm = ScrawlVM::new();
        let program = vec![
            Instruction::new(Opcode::Identity(IdentityOp::Derive),
                vec![Operand::Int(0), Operand::Int(0xCAFE), Operand::Int(16)]),
            Instruction::new(Opcode::Identity(IdentityOp::Verify),
                vec![Operand::Int(0), Operand::Int(0), Operand::Int(1)]),
            Instruction::new(Opcode::Identity(IdentityOp::Fingerprint),
                vec![Operand::Int(2), Operand::Int(0)]),
            Instruction::new(Opcode::Execution(ExecutionOp::Yield),
                vec![Operand::Int(1)]),
            halt(),
        ];
        let result = vm.execute(&program);
        assert!(result.success);
        assert_eq!(result.instructions_executed, 5);
        assert_eq!(vm.registers.get_reg(1), 1); // verified
        assert_ne!(vm.registers.get_reg(2), 0); // fingerprint set
        assert_eq!(result.yielded_values, vec![1]);
    }

    #[test]
    fn test_consensus_commit() {
        let mut vm = ScrawlVM::new();
        vm.agent_id = 0;
        let program = vec![
            Instruction::new(Opcode::Consensus(ConsensusOp::Propose),
                vec![Operand::Int(1), Operand::Int(0), Operand::List(vec![0, 1])]),
            Instruction::new(Opcode::Consensus(ConsensusOp::Quorum),
                vec![Operand::Int(1), Operand::Float(0.5)]),
            Instruction::new(Opcode::Consensus(ConsensusOp::Vote),
                vec![Operand::Int(1), Operand::Int(0), Operand::Int(0)]),
            Instruction::new(Opcode::Consensus(ConsensusOp::Vote),
                vec![Operand::Int(1), Operand::Int(0), Operand::Int(0)]),
            Instruction::new(Opcode::Consensus(ConsensusOp::Commit),
                vec![Operand::Int(1), Operand::Int(1)]),
            halt(),
        ];
        let result = vm.execute(&program);
        assert!(result.success);
        assert_eq!(vm.registers.get_reg(1), 1); // committed
    }

    #[test]
    fn test_nop_and_halt() {
        let mut vm = ScrawlVM::new();
        let result = vm.execute(&[
            Instruction::new(Opcode::Execution(ExecutionOp::Nop), vec![]),
            Instruction::new(Opcode::Execution(ExecutionOp::Nop), vec![]),
            halt(),
        ]);
        assert!(result.success);
        assert!(result.halted);
        assert_eq!(result.instructions_executed, 3);
    }
}
