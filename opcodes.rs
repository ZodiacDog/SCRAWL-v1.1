//! SCRAWL v1.1 Instruction Set Architecture
//!
//! 84 opcodes organized into 6 operational domains.
//! Mirrors the Python `opcodes.py` implementation exactly.
//!
//! ML Innovations LLC · M. L. McKnight · Pheba, Mississippi · 2026

/// Tensor operations (0x00–0x0E): 15 opcodes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum TensorOp {
    Compose    = 0x00,
    Decompose  = 0x01,
    Transform  = 0x02,
    Reshape    = 0x03,
    Slice      = 0x04,
    Reduce     = 0x05,
    Quantize   = 0x06,
    Broadcast  = 0x07,
    Fill       = 0x08,
    Copy       = 0x09,
    Compare    = 0x0A,
    Convert    = 0x0B,
    Norm       = 0x0C,
    Random     = 0x0D,
    Einsum     = 0x0E,
}

/// Attention routing (0x10–0x1D): 14 opcodes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum AttentionOp {
    Route     = 0x10,
    Mask      = 0x11,
    Focus     = 0x12,
    Scatter   = 0x13,
    Gather    = 0x14,
    Cross     = 0x15,
    SelfAttn  = 0x16,
    MultiHead = 0x17,
    Sparse    = 0x18,
    Linear    = 0x19,
    Flash     = 0x1A,
    Window    = 0x1B,
    Pool      = 0x1C,
    TopK      = 0x1D,
}

/// Execution control (0x20–0x2F): 16 opcodes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ExecutionOp {
    Nop    = 0x20,
    Halt   = 0x21,
    Yield  = 0x22,
    Abort  = 0x23,
    Branch = 0x24,
    Loop   = 0x25,
    Call   = 0x26,
    Return = 0x27,
    Fork   = 0x28,
    Join   = 0x29,
    Trap   = 0x2A,
    Resume = 0x2B,
    Spawn  = 0x2C,
    Kill   = 0x2D,
    Sleep  = 0x2E,
    Wake   = 0x2F,
}

/// State management (0x30–0x3E): 15 opcodes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum StateOp {
    Sync      = 0x30,
    Lock      = 0x31,
    Unlock    = 0x32,
    Delta     = 0x33,
    Apply     = 0x34,
    Snapshot  = 0x35,
    Restore   = 0x36,
    Publish   = 0x37,
    Subscribe = 0x38,
    Watch     = 0x39,
    Cas       = 0x3A,
    Load      = 0x3B,
    Store     = 0x3C,
    Evict     = 0x3D,
    Prefetch  = 0x3E,
}

/// Consensus (0x40–0x4B): 12 opcodes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ConsensusOp {
    Propose  = 0x40,
    Vote     = 0x41,
    Commit   = 0x42,
    Reject   = 0x43,
    Quorum   = 0x44,
    Escalate = 0x45,
    Timeout  = 0x46,
    Revoke   = 0x47,
    Delegate = 0x48,
    Audit    = 0x49,
    Veto     = 0x4A,
    Ratify   = 0x4B,
}

/// Identity (0x50–0x5B): 12 opcodes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum IdentityOp {
    Derive      = 0x50,
    Verify      = 0x51,
    Baseline    = 0x52,
    Reconstruct = 0x53,
    Rotate      = 0x54,
    Challenge   = 0x55,
    Respond     = 0x56,
    Bind        = 0x57,
    Unbind      = 0x58,
    Fingerprint = 0x59,
    Chain       = 0x5A,
    Split       = 0x5B,
}

/// Compose modes for T_COMPOSE
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ComposeMode {
    Dot      = 0,
    Hadamard = 1,
    Outer    = 2,
    Concat   = 3,
    Kronecker = 4,
}

/// Reduce operations for T_REDUCE
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ReduceOp {
    Sum  = 0,
    Mean = 1,
    Max  = 2,
    Min  = 3,
    Prod = 4,
}

/// Vote values for C_VOTE
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum VoteValue {
    Approve = 0,
    Reject  = 1,
    Abstain = 2,
}

/// Unified opcode that covers all 84 instructions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Opcode {
    Tensor(TensorOp),
    Attention(AttentionOp),
    Execution(ExecutionOp),
    State(StateOp),
    Consensus(ConsensusOp),
    Identity(IdentityOp),
}

impl Opcode {
    /// Decode a raw byte into an Opcode
    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            0x00..=0x0E => Some(Self::Tensor(unsafe { std::mem::transmute(b) })),
            0x10..=0x1D => Some(Self::Attention(unsafe { std::mem::transmute(b) })),
            0x20..=0x2F => Some(Self::Execution(unsafe { std::mem::transmute(b) })),
            0x30..=0x3E => Some(Self::State(unsafe { std::mem::transmute(b) })),
            0x40..=0x4B => Some(Self::Consensus(unsafe { std::mem::transmute(b) })),
            0x50..=0x5B => Some(Self::Identity(unsafe { std::mem::transmute(b) })),
            _ => None,
        }
    }

    /// Encode opcode to raw byte
    pub fn to_byte(&self) -> u8 {
        match self {
            Self::Tensor(op) => *op as u8,
            Self::Attention(op) => *op as u8,
            Self::Execution(op) => *op as u8,
            Self::State(op) => *op as u8,
            Self::Consensus(op) => *op as u8,
            Self::Identity(op) => *op as u8,
        }
    }

    /// Get human-readable mnemonic
    pub fn mnemonic(&self) -> &'static str {
        match self {
            Self::Tensor(TensorOp::Compose) => "T_COMPOSE",
            Self::Tensor(TensorOp::Decompose) => "T_DECOMPOSE",
            Self::Tensor(TensorOp::Transform) => "T_TRANSFORM",
            Self::Tensor(TensorOp::Reshape) => "T_RESHAPE",
            Self::Tensor(TensorOp::Slice) => "T_SLICE",
            Self::Tensor(TensorOp::Reduce) => "T_REDUCE",
            Self::Tensor(TensorOp::Quantize) => "T_QUANTIZE",
            Self::Tensor(TensorOp::Broadcast) => "T_BROADCAST",
            Self::Tensor(TensorOp::Fill) => "T_FILL",
            Self::Tensor(TensorOp::Copy) => "T_COPY",
            Self::Tensor(TensorOp::Compare) => "T_COMPARE",
            Self::Tensor(TensorOp::Convert) => "T_CONVERT",
            Self::Tensor(TensorOp::Norm) => "T_NORM",
            Self::Tensor(TensorOp::Random) => "T_RANDOM",
            Self::Tensor(TensorOp::Einsum) => "T_EINSUM",
            Self::Attention(AttentionOp::Route) => "A_ROUTE",
            Self::Attention(AttentionOp::Mask) => "A_MASK",
            Self::Attention(AttentionOp::Focus) => "A_FOCUS",
            Self::Attention(AttentionOp::Scatter) => "A_SCATTER",
            Self::Attention(AttentionOp::Gather) => "A_GATHER",
            Self::Attention(AttentionOp::Cross) => "A_CROSS",
            Self::Attention(AttentionOp::SelfAttn) => "A_SELF",
            Self::Attention(AttentionOp::MultiHead) => "A_MULTI_HEAD",
            Self::Attention(AttentionOp::Sparse) => "A_SPARSE",
            Self::Attention(AttentionOp::Linear) => "A_LINEAR",
            Self::Attention(AttentionOp::Flash) => "A_FLASH",
            Self::Attention(AttentionOp::Window) => "A_WINDOW",
            Self::Attention(AttentionOp::Pool) => "A_POOL",
            Self::Attention(AttentionOp::TopK) => "A_TOPK",
            Self::Execution(ExecutionOp::Nop) => "X_NOP",
            Self::Execution(ExecutionOp::Halt) => "X_HALT",
            Self::Execution(ExecutionOp::Yield) => "X_YIELD",
            Self::Execution(ExecutionOp::Abort) => "X_ABORT",
            Self::Execution(ExecutionOp::Branch) => "X_BRANCH",
            Self::Execution(ExecutionOp::Loop) => "X_LOOP",
            Self::Execution(ExecutionOp::Call) => "X_CALL",
            Self::Execution(ExecutionOp::Return) => "X_RETURN",
            Self::Execution(ExecutionOp::Fork) => "X_FORK",
            Self::Execution(ExecutionOp::Join) => "X_JOIN",
            Self::Execution(ExecutionOp::Trap) => "X_TRAP",
            Self::Execution(ExecutionOp::Resume) => "X_RESUME",
            Self::Execution(ExecutionOp::Spawn) => "X_SPAWN",
            Self::Execution(ExecutionOp::Kill) => "X_KILL",
            Self::Execution(ExecutionOp::Sleep) => "X_SLEEP",
            Self::Execution(ExecutionOp::Wake) => "X_WAKE",
            Self::State(StateOp::Sync) => "S_SYNC",
            Self::State(StateOp::Lock) => "S_LOCK",
            Self::State(StateOp::Unlock) => "S_UNLOCK",
            Self::State(StateOp::Delta) => "S_DELTA",
            Self::State(StateOp::Apply) => "S_APPLY",
            Self::State(StateOp::Snapshot) => "S_SNAPSHOT",
            Self::State(StateOp::Restore) => "S_RESTORE",
            Self::State(StateOp::Publish) => "S_PUBLISH",
            Self::State(StateOp::Subscribe) => "S_SUBSCRIBE",
            Self::State(StateOp::Watch) => "S_WATCH",
            Self::State(StateOp::Cas) => "S_CAS",
            Self::State(StateOp::Load) => "S_LOAD",
            Self::State(StateOp::Store) => "S_STORE",
            Self::State(StateOp::Evict) => "S_EVICT",
            Self::State(StateOp::Prefetch) => "S_PREFETCH",
            Self::Consensus(ConsensusOp::Propose) => "C_PROPOSE",
            Self::Consensus(ConsensusOp::Vote) => "C_VOTE",
            Self::Consensus(ConsensusOp::Commit) => "C_COMMIT",
            Self::Consensus(ConsensusOp::Reject) => "C_REJECT",
            Self::Consensus(ConsensusOp::Quorum) => "C_QUORUM",
            Self::Consensus(ConsensusOp::Escalate) => "C_ESCALATE",
            Self::Consensus(ConsensusOp::Timeout) => "C_TIMEOUT",
            Self::Consensus(ConsensusOp::Revoke) => "C_REVOKE",
            Self::Consensus(ConsensusOp::Delegate) => "C_DELEGATE",
            Self::Consensus(ConsensusOp::Audit) => "C_AUDIT",
            Self::Consensus(ConsensusOp::Veto) => "C_VETO",
            Self::Consensus(ConsensusOp::Ratify) => "C_RATIFY",
            Self::Identity(IdentityOp::Derive) => "I_DERIVE",
            Self::Identity(IdentityOp::Verify) => "I_VERIFY",
            Self::Identity(IdentityOp::Baseline) => "I_BASELINE",
            Self::Identity(IdentityOp::Reconstruct) => "I_RECONSTRUCT",
            Self::Identity(IdentityOp::Rotate) => "I_ROTATE",
            Self::Identity(IdentityOp::Challenge) => "I_CHALLENGE",
            Self::Identity(IdentityOp::Respond) => "I_RESPOND",
            Self::Identity(IdentityOp::Bind) => "I_BIND",
            Self::Identity(IdentityOp::Unbind) => "I_UNBIND",
            Self::Identity(IdentityOp::Fingerprint) => "I_FINGERPRINT",
            Self::Identity(IdentityOp::Chain) => "I_CHAIN",
            Self::Identity(IdentityOp::Split) => "I_SPLIT",
        }
    }
}
