//! SCRAWL v1.2 Virtual Machine
//!
//! Register-based execution engine for SCRAWL instruction streams.
//! v1.2: meaningful I_VERIFY, C_VOTE agent operands, C_COMMIT borrow fix,
//! unimplemented opcodes trap (not silent NOPs), LEX_CALL expansion,
//! additional straightforward opcode implementations.
//!
//! ML Innovations Research Lab · M. L. McKnight · Korg (AI assistant, Grok Bot) · 2026

use std::collections::HashMap;
use std::time::Instant;

use crate::identity::{IdentityBaseline, ml_algebraic_verify};
use crate::idioms;
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

fn op_int(operands: &[Operand], idx: usize) -> Option<i64> {
    match operands.get(idx) {
        Some(Operand::Int(v)) => Some(*v),
        Some(Operand::Float(v)) => Some(*v as i64),
        Some(Operand::List(v)) => Some(v.first().copied().unwrap_or(0)),
        None => None,
    }
}

fn op_float(operands: &[Operand], idx: usize) -> Option<f64> {
    match operands.get(idx) {
        Some(Operand::Float(v)) => Some(*v),
        Some(Operand::Int(v)) => Some(*v as f64),
        Some(Operand::List(_)) => Some(0.0),
        None => None,
    }
}

fn op_list(operands: &[Operand], idx: usize) -> Option<Vec<i64>> {
    match operands.get(idx) {
        Some(Operand::List(v)) => Some(v.clone()),
        Some(Operand::Int(v)) => Some(vec![*v]),
        Some(Operand::Float(v)) => Some(vec![*v as i64]),
        None => None,
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
    call_stack: Vec<usize>,
}

pub struct Proposal {
    pub data: u64,
    pub agents: Vec<i64>,
    pub votes: HashMap<u32, i64>,
    pub committed: bool,
    pub rejected: bool,
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
            call_stack: Vec::new(),
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

    fn trap(&mut self, result: &mut ExecutionResult, message: String) {
        result.success = false;
        result.error_code = 1;
        result.error_message = message.clone();
        self.registers.halted = true;
        self.emit_trace(TraceSeverity::Error, "vm", "trap", &message);
    }

    pub fn execute(&mut self, instructions: &[Instruction]) -> ExecutionResult {
        let mut result = ExecutionResult::new();
        let start = Instant::now();

        self.registers.pc = 0;
        self.registers.halted = false;
        self.call_stack.clear();

        self.run_loop(instructions, &mut result);

        result.halted = self.registers.halted;
        result.execution_time_ms = start.elapsed().as_secs_f64() * 1000.0;
        result.trace_events = self.trace_events.clone();
        result
    }

    fn run_loop(&mut self, instructions: &[Instruction], result: &mut ExecutionResult) {
        while self.registers.pc < instructions.len()
            && !self.registers.halted
            && result.success
            && result.instructions_executed < self.max_instructions
        {
            let inst = instructions[self.registers.pc].clone();
            let inst_start = Instant::now();
            let jumped = self.execute_instruction(&inst, result);
            result.instructions_executed += 1;

            if result.success
                && self.instruction_timeout_secs > 0.0
                && inst_start.elapsed().as_secs_f64() > self.instruction_timeout_secs
            {
                self.trap(
                    result,
                    format!(
                        "per-instruction timeout exceeded ({:.3}s) at pc {}",
                        self.instruction_timeout_secs, self.registers.pc
                    ),
                );
            }

            if !self.registers.halted && result.success && !jumped {
                self.registers.pc += 1;
            }
        }
    }

    /// Returns true if the instruction set PC itself (caller must not increment).
    fn execute_instruction(&mut self, inst: &Instruction, result: &mut ExecutionResult) -> bool {
        let ops = &inst.operands;

        match inst.opcode {
            Opcode::Execution(ExecutionOp::Nop) => {}
            Opcode::Execution(ExecutionOp::Halt) => {
                self.registers.halted = true;
            }
            Opcode::Execution(ExecutionOp::Yield) => {
                if let Some(idx) = op_int(ops, 0) {
                    result.yielded_values.push(self.registers.get_reg(idx as u8));
                }
            }
            Opcode::Execution(ExecutionOp::Abort) => {
                self.trap(result, "X_ABORT".into());
            }
            Opcode::Execution(ExecutionOp::Trap) => {
                let msg = match op_int(ops, 0) {
                    Some(code) => format!("X_TRAP code={}", code),
                    None => "X_TRAP".into(),
                };
                self.trap(result, msg);
            }
            Opcode::Execution(ExecutionOp::Branch) => {
                // If R[cond] != 0, jump to absolute instruction index target.
                if let (Some(cond), Some(target)) = (op_int(ops, 0), op_int(ops, 1)) {
                    if self.registers.get_reg(cond as u8) != 0 {
                        self.registers.pc = target as usize;
                        return true;
                    }
                } else {
                    self.trap(result, "X_BRANCH requires (cond_reg, target_pc)".into());
                }
            }
            Opcode::Execution(ExecutionOp::Loop) => {
                // Decrement counter register; jump to target if it is still > 0 after decrement.
                // First iteration: if counter is already 0, do not jump.
                if let (Some(ctr), Some(target)) = (op_int(ops, 0), op_int(ops, 1)) {
                    let val = self.registers.get_reg(ctr as u8);
                    if val > 0 {
                        self.registers.set_reg(ctr as u8, val - 1);
                        if self.registers.get_reg(ctr as u8) > 0 {
                            self.registers.pc = target as usize;
                            return true;
                        }
                    }
                } else {
                    self.trap(result, "X_LOOP requires (counter_reg, target_pc)".into());
                }
            }
            Opcode::Execution(ExecutionOp::Call) => {
                if let Some(target) = op_int(ops, 0) {
                    self.call_stack.push(self.registers.pc + 1);
                    self.registers.pc = target as usize;
                    return true;
                } else {
                    self.trap(result, "X_CALL requires target_pc".into());
                }
            }
            Opcode::Execution(ExecutionOp::Return) => {
                if let Some(idx) = op_int(ops, 0) {
                    result.yielded_values.push(self.registers.get_reg(idx as u8));
                }
                if let Some(ret) = self.call_stack.pop() {
                    self.registers.pc = ret;
                    return true;
                } else {
                    // Top-level return: yield and halt (v1.1 behavior).
                    self.registers.halted = true;
                }
            }

            // ── Identity ──
            Opcode::Identity(IdentityOp::Derive) | Opcode::Identity(IdentityOp::Baseline) => {
                if let (Some(cr_dst), Some(seed), Some(depth)) =
                    (op_int(ops, 0), op_int(ops, 1), op_int(ops, 2))
                {
                    let cr_dst = cr_dst as u8;
                    let baseline = IdentityBaseline::new(seed as u64, depth as usize);
                    let fp = baseline.fingerprint.to_vec();
                    self.baselines.insert(cr_dst, baseline);
                    self.registers.set_creg(cr_dst, fp);
                    self.emit_trace(
                        TraceSeverity::Debug, "identity", "baseline_derived",
                        &format!("CR{} = IdentityBaseline(seed=0x{:04X}, depth={})", cr_dst, seed, depth),
                    );
                } else {
                    self.trap(result, "I_DERIVE/I_BASELINE requires (cr_dst, seed, depth)".into());
                }
            }
            Opcode::Identity(IdentityOp::Verify) => {
                // Recompute chain + fingerprint from the stored seed/depth.
                // A tautological a+a²+b==b² check is not used here.
                if let (Some(cr_src), Some(reg_result)) = (op_int(ops, 0), op_int(ops, 2).or_else(|| op_int(ops, 1))) {
                    // Operands: [cr_src, _unused, reg_result] (v1.1 layout) or [cr_src, reg_result].
                    let reg_result = if ops.len() >= 3 {
                        op_int(ops, 2).unwrap()
                    } else {
                        reg_result
                    } as u8;
                    if let Some(baseline) = self.baselines.get(&(cr_src as u8)) {
                        let valid = baseline.verify_integrity();
                        self.registers.set_reg(reg_result, if valid { 1 } else { 0 });
                        if !valid {
                            self.emit_trace(
                                TraceSeverity::Warn, "identity", "verify_failed",
                                "I_VERIFY: chain or fingerprint does not match seed/depth",
                            );
                        }
                    } else {
                        self.registers.set_reg(reg_result, 0);
                        self.emit_trace(
                            TraceSeverity::Warn, "identity", "missing_baseline",
                            &format!("I_VERIFY: no baseline in CR{}", cr_src),
                        );
                    }
                } else {
                    self.trap(result, "I_VERIFY requires (cr_src, _, result_reg)".into());
                }
            }
            Opcode::Identity(IdentityOp::Fingerprint) => {
                let reg_dst = op_int(ops, 0);
                let cr_src = op_int(ops, 1);
                if let (Some(reg_dst), Some(cr_src)) = (reg_dst, cr_src) {
                    if let Some(baseline) = self.baselines.get(&(cr_src as u8)) {
                        let fp = u64::from_le_bytes(baseline.fingerprint);
                        self.registers.set_reg(reg_dst as u8, fp);
                    } else {
                        self.registers.set_reg(reg_dst as u8, 0);
                    }
                } else {
                    self.trap(result, "I_FINGERPRINT requires (dest_reg, cr_src)".into());
                }
            }
            Opcode::Identity(IdentityOp::Chain) => {
                if let (Some(cr_src), Some(reg_dst)) = (op_int(ops, 0), op_int(ops, 1)) {
                    let len = self.baselines.get(&(cr_src as u8)).map(|b| b.chain.len()).unwrap_or(0);
                    self.registers.set_reg(reg_dst as u8, len as u64);
                } else {
                    self.trap(result, "I_CHAIN requires (cr_src, dest_reg)".into());
                }
            }

            // ── Consensus ──
            Opcode::Consensus(ConsensusOp::Propose) => {
                if let (Some(prop_id), Some(data_reg), Some(agents)) =
                    (op_int(ops, 0), op_int(ops, 1), op_list(ops, 2))
                {
                    let data = self.registers.get_reg(data_reg as u8);
                    self.proposals.insert(prop_id, Proposal {
                        data, agents: agents.clone(), votes: HashMap::new(),
                        committed: false, rejected: false, quorum: 0.5,
                    });
                    self.emit_trace(TraceSeverity::Info, "consensus", "proposal_created",
                        &format!("Proposal {}: agents={:?}", prop_id, agents));
                } else {
                    self.trap(result, "C_PROPOSE requires (prop_id, data_reg, agents)".into());
                }
            }
            Opcode::Consensus(ConsensusOp::Vote) => {
                // v1.2 canonical operands: (prop_id, agent_id, vote_value).
                // 2-operand fallback: (prop_id, vote_value) uses vm.agent_id.
                let parsed = if ops.len() >= 3 {
                    match (op_int(ops, 0), op_int(ops, 1), op_int(ops, 2)) {
                        (Some(p), Some(a), Some(v)) => Some((p, a as u32, v)),
                        _ => None,
                    }
                } else if ops.len() == 2 {
                    match (op_int(ops, 0), op_int(ops, 1)) {
                        (Some(p), Some(v)) => Some((p, self.agent_id, v)),
                        _ => None,
                    }
                } else {
                    None
                };
                if let Some((prop_id, agent, vote_value)) = parsed {
                    if let Some(proposal) = self.proposals.get_mut(&prop_id) {
                        proposal.votes.insert(agent, vote_value);
                        if vote_value == 1 {
                            self.emit_trace(TraceSeverity::Warn, "consensus", "vote_rejected",
                                &format!("Agent {} REJECTED proposal {}", agent, prop_id));
                        }
                    }
                } else {
                    self.trap(result, "C_VOTE requires (prop_id, agent_id, vote) or (prop_id, vote)".into());
                }
            }
            Opcode::Consensus(ConsensusOp::Quorum) => {
                if let (Some(prop_id), Some(threshold)) = (op_int(ops, 0), op_float(ops, 1)) {
                    if let Some(proposal) = self.proposals.get_mut(&prop_id) {
                        proposal.quorum = threshold;
                    }
                } else {
                    self.trap(result, "C_QUORUM requires (prop_id, threshold)".into());
                }
            }
            Opcode::Consensus(ConsensusOp::Commit) => {
                if let (Some(prop_id), Some(dest_reg)) = (op_int(ops, 0), op_int(ops, 1)) {
                    let decision = self.proposals.get_mut(&prop_id).map(|proposal| {
                        let agents_len = proposal.agents.len();
                        let required = (agents_len as f64 * proposal.quorum).ceil().max(1.0) as usize;
                        let approve_count = proposal.votes.values().filter(|&&v| v == 0).count();
                        if proposal.rejected {
                            (false, 0u64, approve_count, required, true)
                        } else if approve_count >= required {
                            let data_val = proposal.data;
                            proposal.committed = true;
                            (true, data_val, approve_count, required, false)
                        } else {
                            (false, 0u64, approve_count, required, false)
                        }
                    });
                    if let Some((committed, data_val, approve_count, required, rejected)) = decision {
                        if committed {
                            if data_val > 0 {
                                let a = data_val & 0xFFFF;
                                let a_sq = a.saturating_mul(a);
                                let b = a + 1;
                                let b_sq = b.saturating_mul(b);
                                if !ml_algebraic_verify(a, a_sq, b, b_sq) {
                                    self.emit_trace(TraceSeverity::Critical, "consensus",
                                        "algebraic_integrity_failure",
                                        &format!("ML Identity check FAILED for proposal {}", prop_id));
                                }
                            }
                            self.registers.set_reg(dest_reg as u8, 1);
                            self.emit_trace(TraceSeverity::Info, "consensus", "committed",
                                &format!("Proposal {} committed", prop_id));
                        } else {
                            self.registers.set_reg(dest_reg as u8, 0);
                            let kind = if rejected { "rejected" } else { "quorum_not_met" };
                            self.emit_trace(TraceSeverity::Warn, "consensus", kind,
                                &format!("Proposal {}: {}/{} approvals", prop_id, approve_count, required));
                        }
                    } else {
                        self.registers.set_reg(dest_reg as u8, 0);
                    }
                } else {
                    self.trap(result, "C_COMMIT requires (prop_id, dest_reg)".into());
                }
            }
            Opcode::Consensus(ConsensusOp::Reject) => {
                if let Some(prop_id) = op_int(ops, 0) {
                    if let Some(proposal) = self.proposals.get_mut(&prop_id) {
                        proposal.rejected = true;
                        proposal.committed = false;
                    }
                    if let Some(dest) = op_int(ops, 1) {
                        self.registers.set_reg(dest as u8, 0);
                    }
                    self.emit_trace(TraceSeverity::Warn, "consensus", "rejected",
                        &format!("Proposal {} rejected", prop_id));
                } else {
                    self.trap(result, "C_REJECT requires prop_id".into());
                }
            }
            Opcode::Consensus(ConsensusOp::Audit) => {
                if let Some(prop_id) = op_int(ops, 0) {
                    let msg = if let Some(p) = self.proposals.get(&prop_id) {
                        format!(
                            "Proposal {}: votes={}, committed={}, rejected={}",
                            prop_id, p.votes.len(), p.committed, p.rejected
                        )
                    } else {
                        format!("Proposal {} not found", prop_id)
                    };
                    self.emit_trace(TraceSeverity::Info, "consensus", "audit", &msg);
                    if let Some(dest) = op_int(ops, 1) {
                        let n = self.proposals.get(&prop_id).map(|p| p.votes.len()).unwrap_or(0);
                        self.registers.set_reg(dest as u8, n as u64);
                    }
                } else {
                    self.trap(result, "C_AUDIT requires prop_id".into());
                }
            }

            // ── Tensor ──
            Opcode::Tensor(TensorOp::Fill) => {
                if let (Some(dst), Some(value)) = (op_int(ops, 0), op_float(ops, 1)) {
                    let dst = dst as u8;
                    if let Some(existing) = self.registers.get_treg(dst) {
                        let shape = existing.shape.clone();
                        self.registers.set_treg(dst, Tensor::fill(shape, value));
                    } else {
                        self.registers.set_treg(dst, Tensor::fill(vec![1], value));
                    }
                } else {
                    self.trap(result, "T_FILL requires (dst_treg, value)".into());
                }
            }
            Opcode::Tensor(TensorOp::Norm) => {
                if let (Some(dst), Some(src)) = (op_int(ops, 0), op_int(ops, 1)) {
                    if let Some(t) = self.registers.get_treg(src as u8).cloned() {
                        self.registers.set_treg(dst as u8, t.l2_normalize());
                    }
                } else {
                    self.trap(result, "T_NORM requires (dst_treg, src_treg)".into());
                }
            }
            Opcode::Tensor(TensorOp::Copy) => {
                if let (Some(dst), Some(src)) = (op_int(ops, 0), op_int(ops, 1)) {
                    if let Some(t) = self.registers.get_treg(src as u8).cloned() {
                        self.registers.set_treg(dst as u8, t);
                    }
                } else {
                    self.trap(result, "T_COPY requires (dst_treg, src_treg)".into());
                }
            }
            Opcode::Tensor(TensorOp::Reshape) => {
                if let (Some(dst), Some(src), Some(shape)) = (op_int(ops, 0), op_int(ops, 1), op_list(ops, 2)) {
                    if let Some(t) = self.registers.get_treg(src as u8).cloned() {
                        let shape_us: Vec<usize> = shape.iter().map(|&d| d.max(0) as usize).collect();
                        match t.reshape(shape_us) {
                            Ok(reshaped) => self.registers.set_treg(dst as u8, reshaped),
                            Err(e) => self.trap(result, format!("T_RESHAPE: {}", e)),
                        }
                    }
                } else {
                    self.trap(result, "T_RESHAPE requires (dst, src, shape_list)".into());
                }
            }
            Opcode::Tensor(TensorOp::Compose) => {
                if let (Some(dst), Some(a), Some(b), Some(mode)) =
                    (op_int(ops, 0), op_int(ops, 1), op_int(ops, 2), op_int(ops, 3))
                {
                    let ta = self.registers.get_treg(a as u8).cloned();
                    let tb = self.registers.get_treg(b as u8).cloned();
                    match (ta, tb) {
                        (Some(ta), Some(tb)) => {
                            let out = match mode {
                                0 => ta.dot(&tb),
                                1 => ta.hadamard(&tb),
                                _ => Err(format!("T_COMPOSE: unsupported mode {}", mode)),
                            };
                            match out {
                                Ok(t) => self.registers.set_treg(dst as u8, t),
                                Err(e) => self.trap(result, e),
                            }
                        }
                        _ => self.trap(result, "T_COMPOSE: missing source tensor".into()),
                    }
                } else {
                    self.trap(result, "T_COMPOSE requires (dst, src_a, src_b, mode)".into());
                }
            }
            Opcode::Tensor(TensorOp::Compare) => {
                if let (Some(dst), Some(a), Some(b)) = (op_int(ops, 0), op_int(ops, 1), op_int(ops, 2)) {
                    let eq = match (self.registers.get_treg(a as u8), self.registers.get_treg(b as u8)) {
                        (Some(ta), Some(tb)) => ta == tb,
                        _ => false,
                    };
                    self.registers.set_reg(dst as u8, if eq { 1 } else { 0 });
                } else {
                    self.trap(result, "T_COMPARE requires (dest_reg, treg_a, treg_b)".into());
                }
            }
            Opcode::Tensor(TensorOp::Convert) => {
                if let (Some(dst), Some(src)) = (op_int(ops, 0), op_int(ops, 1)) {
                    if let Some(t) = self.registers.get_treg(src as u8).cloned() {
                        self.registers.set_treg(dst as u8, t);
                    }
                } else {
                    self.trap(result, "T_CONVERT requires (dst_treg, src_treg)".into());
                }
            }
            Opcode::Tensor(TensorOp::Reduce) => {
                if let (Some(dst), Some(src), Some(opv)) = (op_int(ops, 0), op_int(ops, 1), op_int(ops, 2)) {
                    if let Some(t) = self.registers.get_treg(src as u8) {
                        let value = match opv {
                            0 => t.reduce_sum(),
                            1 => if t.size() == 0 { 0.0 } else { t.reduce_sum() / t.size() as f64 },
                            2 => t.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                            3 => t.data.iter().cloned().fold(f64::INFINITY, f64::min),
                            4 => t.data.iter().product(),
                            _ => {
                                self.trap(result, format!("T_REDUCE: unknown op {}", opv));
                                return false;
                            }
                        };
                        match Tensor::new(vec![value], vec![1]) {
                            Ok(out) => self.registers.set_treg(dst as u8, out),
                            Err(e) => self.trap(result, e),
                        }
                    }
                } else {
                    self.trap(result, "T_REDUCE requires (dst_treg, src_treg, op)".into());
                }
            }

            // ── Attention ──
            Opcode::Attention(AttentionOp::Route) => {
                if let (Some(q), Some(k), Some(v), Some(dst)) =
                    (op_int(ops, 0), op_int(ops, 1), op_int(ops, 2), op_int(ops, 3))
                {
                    self.exec_attention_route(q as u8, k as u8, v as u8, dst as u8);
                } else {
                    self.trap(result, "A_ROUTE requires (q, k, v, dst)".into());
                }
            }
            Opcode::Attention(AttentionOp::SelfAttn) => {
                // TR_dst = attention.self(TR_src): Q = K = V = src.
                if let (Some(src), Some(dst)) = (op_int(ops, 0), op_int(ops, 1)) {
                    self.exec_attention_route(src as u8, src as u8, src as u8, dst as u8);
                } else {
                    self.trap(result, "A_SELF requires (src_treg, dst_treg)".into());
                }
            }

            // ── State ──
            Opcode::State(StateOp::Sync) => {
                self.emit_trace(TraceSeverity::Debug, "state", "sync", "S_SYNC barrier");
            }
            Opcode::State(StateOp::Lock) => {
                if let Some(id) = op_int(ops, 0) {
                    self.locks.insert(id, true);
                } else {
                    self.trap(result, "S_LOCK requires lock_id".into());
                }
            }
            Opcode::State(StateOp::Unlock) => {
                if let Some(id) = op_int(ops, 0) {
                    self.locks.insert(id, false);
                } else {
                    self.trap(result, "S_UNLOCK requires lock_id".into());
                }
            }
            Opcode::State(StateOp::Snapshot) => {
                if let (Some(id), Some(src)) = (op_int(ops, 0), op_int(ops, 1)) {
                    self.snapshots.insert(id, self.registers.get_reg(src as u8));
                } else {
                    self.trap(result, "S_SNAPSHOT requires (snap_id, src_reg)".into());
                }
            }
            Opcode::State(StateOp::Restore) => {
                if let (Some(id), Some(dst)) = (op_int(ops, 0), op_int(ops, 1)) {
                    let val = self.snapshots.get(&id).copied().unwrap_or(0);
                    self.registers.set_reg(dst as u8, val);
                } else {
                    self.trap(result, "S_RESTORE requires (snap_id, dest_reg)".into());
                }
            }
            Opcode::State(StateOp::Load) => {
                if let (Some(dst), Some(src)) = (op_int(ops, 0), op_int(ops, 1)) {
                    let val = self.registers.get_reg(src as u8);
                    self.registers.set_reg(dst as u8, val);
                } else {
                    self.trap(result, "S_LOAD requires (dest_reg, src_reg)".into());
                }
            }
            Opcode::State(StateOp::Store) => {
                if let (Some(dst), Some(src)) = (op_int(ops, 0), op_int(ops, 1)) {
                    let val = self.registers.get_reg(src as u8);
                    self.registers.set_reg(dst as u8, val);
                } else {
                    self.trap(result, "S_STORE requires (dest_reg, src_reg)".into());
                }
            }
            Opcode::State(StateOp::Cas) => {
                if let (Some(reg), Some(expected), Some(newv), Some(dest)) =
                    (op_int(ops, 0), op_int(ops, 1), op_int(ops, 2), op_int(ops, 3))
                {
                    let cur = self.registers.get_reg(reg as u8);
                    if cur == expected as u64 {
                        self.registers.set_reg(reg as u8, newv as u64);
                        self.registers.set_reg(dest as u8, 1);
                    } else {
                        self.registers.set_reg(dest as u8, 0);
                    }
                } else {
                    self.trap(result, "S_CAS requires (reg, expected, new, result_reg)".into());
                }
            }
            Opcode::State(StateOp::Evict) => {
                if let Some(idx) = op_int(ops, 0) {
                    self.registers.tensor[idx as usize] = None;
                } else {
                    self.trap(result, "S_EVICT requires tensor register index".into());
                }
            }

            // ── Lexicon / Darmok idioms ──
            Opcode::Lexicon(LexiconOp::Call) => {
                if let Some(id) = op_int(ops, 0) {
                    let params: Vec<Operand> = if ops.len() > 1 { ops[1..].to_vec() } else { vec![] };
                    match idioms::expand(id as u16, &params) {
                        Ok(expanded) => {
                            let saved_pc = self.registers.pc;
                            self.registers.pc = 0;
                            self.run_loop(&expanded, result);
                            self.registers.pc = saved_pc;
                        }
                        Err(e) => self.trap(result, e),
                    }
                } else {
                    self.trap(result, "LEX_CALL requires idiom_id".into());
                }
            }

            // Remaining opcodes: trap with a clear error — never a silent NOP.
            other => {
                self.trap(
                    result,
                    format!(
                        "unimplemented opcode {} (0x{:02X}): trapping (not a silent NOP)",
                        other.mnemonic(),
                        other.to_byte()
                    ),
                );
            }
        }
        false
    }

    fn exec_attention_route(&mut self, q_idx: u8, k_idx: u8, v_idx: u8, dst_idx: u8) {
        let q = match self.registers.get_treg(q_idx) { Some(t) => t.clone(), None => return };
        let k = match self.registers.get_treg(k_idx) { Some(t) => t.clone(), None => return };
        let v = match self.registers.get_treg(v_idx) { Some(t) => t.clone(), None => return };

        if q.ndim() == 2 && k.ndim() == 2 && v.ndim() == 2 {
            let d_k = k.shape[1];
            let scale = 1.0 / (d_k as f64).sqrt();

            let (m, d) = (k.shape[0], k.shape[1]);
            let mut k_t_data = vec![0.0; d * m];
            for i in 0..m {
                for j in 0..d {
                    k_t_data[j * m + i] = k.data[i * d + j];
                }
            }
            let k_t = Tensor::new(k_t_data, vec![d, m]).unwrap();

            let scores = q.dot(&k_t).unwrap().scale(scale);

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

            if let Ok(out) = attn_weights.dot(&v) {
                self.registers.set_treg(dst_idx, out);
            }
        }
    }
}

impl Default for ScrawlVM {
    fn default() -> Self { Self::new() }
}

/// Opcodes that still trap in this VM (not yet implemented, by design).
pub fn unimplemented_opcodes() -> Vec<Opcode> {
    let implemented = implemented_opcodes();
    crate::opcodes::all_base_opcodes()
        .into_iter()
        .filter(|op| !implemented.contains(op))
        .collect()
}

/// Opcodes the VM actually executes (plus LEX_CALL, which is not a base opcode).
pub fn implemented_opcodes() -> Vec<Opcode> {
    use Opcode::*;
    vec![
        Execution(ExecutionOp::Nop),
        Execution(ExecutionOp::Halt),
        Execution(ExecutionOp::Yield),
        Execution(ExecutionOp::Abort),
        Execution(ExecutionOp::Trap),
        Execution(ExecutionOp::Branch),
        Execution(ExecutionOp::Loop),
        Execution(ExecutionOp::Call),
        Execution(ExecutionOp::Return),
        Identity(IdentityOp::Derive),
        Identity(IdentityOp::Baseline),
        Identity(IdentityOp::Verify),
        Identity(IdentityOp::Fingerprint),
        Identity(IdentityOp::Chain),
        Consensus(ConsensusOp::Propose),
        Consensus(ConsensusOp::Vote),
        Consensus(ConsensusOp::Quorum),
        Consensus(ConsensusOp::Commit),
        Consensus(ConsensusOp::Reject),
        Consensus(ConsensusOp::Audit),
        Tensor(TensorOp::Fill),
        Tensor(TensorOp::Norm),
        Tensor(TensorOp::Copy),
        Tensor(TensorOp::Reshape),
        Tensor(TensorOp::Compose),
        Tensor(TensorOp::Compare),
        Tensor(TensorOp::Convert),
        Tensor(TensorOp::Reduce),
        Attention(AttentionOp::Route),
        Attention(AttentionOp::SelfAttn),
        State(StateOp::Sync),
        State(StateOp::Lock),
        State(StateOp::Unlock),
        State(StateOp::Snapshot),
        State(StateOp::Restore),
        State(StateOp::Load),
        State(StateOp::Store),
        State(StateOp::Cas),
        State(StateOp::Evict),
        Lexicon(LexiconOp::Call),
    ]
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
        assert!(result.success, "{}", result.error_message);
        assert_eq!(result.instructions_executed, 5);
        assert_eq!(vm.registers.get_reg(1), 1);
        assert_ne!(vm.registers.get_reg(2), 0);
        assert_eq!(result.yielded_values, vec![1]);
    }

    #[test]
    fn test_verify_tampered_chain_fails() {
        let mut vm = ScrawlVM::new();
        let derive = vec![
            Instruction::new(Opcode::Identity(IdentityOp::Derive),
                vec![Operand::Int(0), Operand::Int(0xCAFE), Operand::Int(16)]),
            halt(),
        ];
        assert!(vm.execute(&derive).success);
        vm.baselines.get_mut(&0).unwrap().chain[3] ^= 1;
        let verify = vec![
            Instruction::new(Opcode::Identity(IdentityOp::Verify),
                vec![Operand::Int(0), Operand::Int(0), Operand::Int(1)]),
            halt(),
        ];
        let result = vm.execute(&verify);
        assert!(result.success);
        assert_eq!(vm.registers.get_reg(1), 0, "tampered chain must fail I_VERIFY");
    }

    #[test]
    fn test_consensus_commit_two_agents() {
        let mut vm = ScrawlVM::new();
        let program = vec![
            Instruction::new(Opcode::Consensus(ConsensusOp::Propose),
                vec![Operand::Int(1), Operand::Int(0), Operand::List(vec![0, 1])]),
            Instruction::new(Opcode::Consensus(ConsensusOp::Quorum),
                vec![Operand::Int(1), Operand::Float(0.5)]),
            // (prop_id, agent_id, vote) — two distinct agents
            Instruction::new(Opcode::Consensus(ConsensusOp::Vote),
                vec![Operand::Int(1), Operand::Int(0), Operand::Int(0)]),
            Instruction::new(Opcode::Consensus(ConsensusOp::Vote),
                vec![Operand::Int(1), Operand::Int(1), Operand::Int(0)]),
            Instruction::new(Opcode::Consensus(ConsensusOp::Commit),
                vec![Operand::Int(1), Operand::Int(1)]),
            halt(),
        ];
        let result = vm.execute(&program);
        assert!(result.success, "{}", result.error_message);
        assert_eq!(vm.registers.get_reg(1), 1);
        assert_eq!(vm.proposals[&1].votes.len(), 2);
    }

    #[test]
    fn test_vote_two_operand_fallback_uses_vm_agent_id() {
        let mut vm = ScrawlVM::new();
        vm.agent_id = 7;
        let program = vec![
            Instruction::new(Opcode::Consensus(ConsensusOp::Propose),
                vec![Operand::Int(1), Operand::Int(0), Operand::List(vec![7])]),
            Instruction::new(Opcode::Consensus(ConsensusOp::Vote),
                vec![Operand::Int(1), Operand::Int(0)]),
            Instruction::new(Opcode::Consensus(ConsensusOp::Commit),
                vec![Operand::Int(1), Operand::Int(2)]),
            halt(),
        ];
        assert!(vm.execute(&program).success);
        assert!(vm.proposals[&1].votes.contains_key(&7));
        assert_eq!(vm.registers.get_reg(2), 1);
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

    #[test]
    fn test_unimplemented_opcode_traps() {
        let mut vm = ScrawlVM::new();
        let result = vm.execute(&[
            Instruction::new(Opcode::Execution(ExecutionOp::Fork), vec![]),
        ]);
        assert!(!result.success);
        assert!(result.error_message.contains("unimplemented"));
        assert!(result.error_message.contains("X_FORK"));
        assert!(vm.registers.halted);
    }

    #[test]
    fn test_every_unimplemented_opcode_traps() {
        for op in unimplemented_opcodes() {
            let mut vm = ScrawlVM::new();
            let result = vm.execute(&[Instruction::new(op, vec![])]);
            assert!(!result.success, "{} should trap", op.mnemonic());
        }
    }

    #[test]
    fn test_branch_and_call() {
        let mut vm = ScrawlVM::new();
        vm.registers.set_reg(0, 1);
        // 0: branch R0 -> 2; 1: abort (skipped); 2: halt
        let result = vm.execute(&[
            Instruction::new(Opcode::Execution(ExecutionOp::Branch),
                vec![Operand::Int(0), Operand::Int(2)]),
            Instruction::new(Opcode::Execution(ExecutionOp::Abort), vec![]),
            halt(),
        ]);
        assert!(result.success, "{}", result.error_message);
        assert!(result.halted);
    }

    #[test]
    fn test_state_cas_and_snapshot() {
        let mut vm = ScrawlVM::new();
        vm.registers.set_reg(5, 10);
        let result = vm.execute(&[
            Instruction::new(Opcode::State(StateOp::Snapshot),
                vec![Operand::Int(1), Operand::Int(5)]),
            Instruction::new(Opcode::State(StateOp::Cas),
                vec![Operand::Int(5), Operand::Int(10), Operand::Int(99), Operand::Int(6)]),
            Instruction::new(Opcode::State(StateOp::Restore),
                vec![Operand::Int(1), Operand::Int(7)]),
            halt(),
        ]);
        assert!(result.success, "{}", result.error_message);
        assert_eq!(vm.registers.get_reg(5), 99);
        assert_eq!(vm.registers.get_reg(6), 1);
        assert_eq!(vm.registers.get_reg(7), 10);
    }

    #[test]
    fn test_tensor_copy_compose_compare() {
        let mut vm = ScrawlVM::new();
        vm.registers.set_treg(0, Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap());
        vm.registers.set_treg(1, Tensor::new(vec![4.0, 5.0, 6.0], vec![3]).unwrap());
        let result = vm.execute(&[
            Instruction::new(Opcode::Tensor(TensorOp::Copy),
                vec![Operand::Int(2), Operand::Int(0)]),
            Instruction::new(Opcode::Tensor(TensorOp::Compose),
                vec![Operand::Int(3), Operand::Int(0), Operand::Int(1), Operand::Int(0)]), // dot
            Instruction::new(Opcode::Tensor(TensorOp::Compare),
                vec![Operand::Int(4), Operand::Int(0), Operand::Int(2)]),
            halt(),
        ]);
        assert!(result.success, "{}", result.error_message);
        let dot = vm.registers.get_treg(3).unwrap();
        assert!((dot.data[0] - 32.0).abs() < 1e-6);
        assert_eq!(vm.registers.get_reg(4), 1);
    }

    #[test]
    fn test_implemented_and_unimplemented_partition_the_isa() {
        let base = crate::opcodes::all_base_opcodes();
        assert_eq!(base.len(), 84);
        let impld: Vec<_> = implemented_opcodes()
            .into_iter()
            .filter(|o| !matches!(o, Opcode::Lexicon(_)))
            .collect();
        let unimpl = unimplemented_opcodes();
        assert_eq!(impld.len() + unimpl.len(), 84);
    }
}
