# SCRAWL v1.2

**Structured Compressed Runtime for Agent-to-Agent Workload Language**

```
Rust implementation · 84-opcode ISA + LEX_CALL idioms · zero runtime dependencies
```

SCRAWL is a binary instruction set and runtime for AI agent communication. Where existing protocols treat agent messages as text to be parsed, SCRAWL treats them as programs to be executed — 84 opcodes across six domains, transmitted in a compact binary wire format, with delta compression derived from the ML Identity equation. One agent compiles intent into SCRAWL instructions; the receiving agent executes them directly on a register-based VM.

This repository is the **Rust implementation** (plus documentation and ROSETTA *source* examples). The Python reference core (`python/src/`, the ROSETTA compiler, and the 552 Python tests described in earlier write-ups) is **not included here** and is not fabricated in this tree.

**Creator:** M. L. McKnight / ML Innovations Research Lab  
**v1.2 contributor:** Korg (AI assistant, Grok Bot) — fixes and the idiom layer

## Quick Start

```bash
git clone https://github.com/ZodiacDog/SCRAWL-v1.1.git
cd SCRAWL-v1.1
cargo test
cargo test --release
cargo run --example heartbeat
cargo run --example idioms
```

**5-line agent heartbeat (Rust):**

```rust
use scrawl::vm::ScrawlVM;
use scrawl::synapse::{Instruction, Operand};
use scrawl::opcodes::*;

let mut vm = ScrawlVM::new();
vm.execute(&[
    Instruction::new(Opcode::Identity(IdentityOp::Derive),
        vec![Operand::Int(0), Operand::Int(0xCAFE), Operand::Int(16)]),
    Instruction::new(Opcode::Identity(IdentityOp::Verify),
        vec![Operand::Int(0), Operand::Int(0), Operand::Int(1)]),
    Instruction::new(Opcode::Execution(ExecutionOp::Yield),
        vec![Operand::Int(1)]),
    Instruction::new(Opcode::Execution(ExecutionOp::Halt), vec![]),
]);
assert_eq!(vm.registers.get_reg(1), 1); // chain recomputed from seed and matched
```

`I_VERIFY` recomputes `ml_identity_chain(seed, depth)` and the fingerprint and compares them to the stored baseline. A tampered chain fails verification.

## What's in this repo vs. what isn't

| Included | Not included |
|----------|----------------|
| Rust crate (`src/`, `examples/`, `benches/`, `tests/`) | Python package `src/*.py` (opcodes, vm, rosetta, …) |
| SYNAPSE encoder/decoder, identity, VM, idiom registry | ROSETTA compiler (`rosetta.py`) — `.rosetta` files are source only |
| Unit + integration tests (`cargo test`) | The 552 Python tests (`test_scrawl.py`, `test_v11.py`, `flux_scrawl_tests.py`) |
| `benchmarks/bench_scrawl.py` (source) | A runnable Python benchmark — that file imports the unpublished `src` package |
| `docs/IDIOMS.md` and a working LEX_CALL implementation | FLUX 2.0 (`flux_scrawl.py`) |

Python examples that live in the sibling repo [SCRAWL-Structured-…](https://github.com/ZodiacDog/SCRAWL-Structured-Compressed-Runtime-for-Agent-to-Agent-Workload-Language) also `import src.*` and will not run until that core is published.

## Benchmarks (measured)

Run the Rust benches with `cargo bench`. The tables below replace earlier typographic estimates with numbers measured from this encoder and compressor. Methodology: [BENCHMARKS.md](BENCHMARKS.md).

**Wire format size — SCRAWL (this Rust SYNAPSE encoder) vs the JSON payloads in `benchmarks/bench_scrawl.py`:**

| Operation | SCRAWL | JSON | Savings |
|-----------|--------|------|---------|
| Identity derive + verify + halt | 36 B | 192 B | **81.2%** (5.3× smaller) |
| Attention route (Q,K,V→out) + halt | 28 B | 113 B | **75.2%** (4.0× smaller) |
| 3-agent consensus round + halt | 72 B | 471 B | **84.7%** (6.5× smaller) |
| 10-instruction mixed program | 102 B | 664 B | **84.6%** (6.5× smaller) |

**Delta compression** (Rust `DeltaCompressor`, seed `0xBEEF`, depth 16):

| Payload | Raw | First send | Identical retransmit |
|---------|-----|------------|----------------------|
| Small agent state (`pos=(100,200) hp=95 ammo=30`) | 27 B | 28 B (flag + payload) | **3 B** |
| 50-byte buffer | 50 B | 51 B | **3 B (94%)** |
| Large JSON state | 294 B | 295 B | 5 B (RLE run length is capped at 255) |

The first send of a fresh payload is raw size plus a 1-byte flag: XOR against a SHA-256 expectation stream does not shrink the first message in this compressor. The **identical-retransmit 3-byte** path is real (RLE of a zero XOR against the previous message) for payloads of 255 bytes or less.

## Idioms (Darmok-style compression)

A short named phrase can stand for a whole multi-step command sequence. The standard table is **SCRAWL Idioms 1.0**. Full rules — definition record, versioning, lexicon fingerprint in the handshake, `LEX_CALL` (opcode `0x60`), the SYNAPSE flag bit, trap-on-unknown, longest-match packing, and collision handling — are in [docs/IDIOMS.md](docs/IDIOMS.md).

Four idioms are executable in this crate:

| Phrase | Expansion | Measured SYNAPSE frame |
|--------|-----------|------------------------|
| **Heartbeat at Cafe** | `I_DERIVE, I_VERIFY, X_YIELD, X_HALT` | 40 B expanded → **26 B** `LEX_CALL` (**35.0%**) |
| **Named Heartbeat** | `I_DERIVE, I_VERIFY, I_FINGERPRINT, X_YIELD, X_HALT` | 46 B → **26 B** (**43.5%**) |
| **The Round at Two** | propose, quorum 0.5, two APPROVE votes, commit, halt | 63 B → **39 B** (**38.1%**) |
| **Silent Retransmit** | host-level second `compress()` of an unchanged buffer | 50 B payload: 51 B first, **3 B** second |

```bash
cargo run --example idioms
```

Agents include the 8-byte lexicon fingerprint in `IdentityHandshake::derive_shared_key_with_lexicon`. A `LEX_CALL` for an unknown id traps; it is never a silent NOP.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    ROSETTA v1.1                      │
│     Human-readable form (compiler not in this repo)  │
│     84 opcodes + Idioms 1.0 macro / LEX_CALL layer   │
├─────────────────────────────────────────────────────┤
│                   SYNAPSE v1.0                       │
│          Binary wire format + CRC-32C                │
│     Magic | Version | Flags | SeqID | Payload | CRC  │
│     Flag 0x01 = frame may contain LEX_CALL           │
├─────────────────────────────────────────────────────┤
│                  SCRAWL VM v1.2                      │
│     Register-based execution engine                  │
│     256 GP + 64 tensor + 16 context registers        │
│     Unimplemented opcodes trap (not silent NOPs)     │
├──────────┬──────────┬──────────┬────────────────────┤
│ Tensor   │ Attention│ Execution│ State | Consensus   │
│ 15 ops   │ 14 ops   │ 16 ops   │ 15+12 ops          │
├──────────┴──────────┴──────────┴────────────────────┤
│              ML Identity Foundation                  │
│    a + a² + b = b² (integer-only, i128 intermediates)│
└─────────────────────────────────────────────────────┘
```

**Six operational domains — 84 base opcodes**, plus `LEX_CALL` at `0x60` in the reserved extension range `0x60–0xFF`.

- **Tensor** (15): compose, decompose, transform, reshape, slice, reduce, quantize, broadcast, fill, copy, compare, convert, normalize, random, einsum
- **Attention** (14): route, mask, focus, scatter, gather, cross, self, multi-head, sparse, linear, flash, window, pool, topk
- **Execution** (16): nop, halt, yield, abort, branch, loop, call, return, fork, join, trap, resume, spawn, kill, sleep, wake
- **State** (15): sync, lock, unlock, delta, apply, snapshot, restore, publish, subscribe, watch, cas, load, store, evict, prefetch
- **Consensus** (12): propose, vote, commit, reject, quorum, escalate, timeout, revoke, delegate, audit, veto, ratify
- **Identity** (12): derive, verify, baseline, reconstruct, rotate, challenge, respond, bind, unbind, fingerprint, chain, split

See CHANGELOG.md for which of these the VM actually executes, and which still trap with a clear error.

`C_VOTE` operands are `(proposal_id, agent_id, vote_value)`. A two-operand form `(proposal_id, vote_value)` still works and uses `vm.agent_id`. Vote `0` is APPROVE, `1` is REJECT.

## The ML Identity

SCRAWL's mathematical foundation, discovered by M. L. McKnight in 1999 at Raymond High School, Mississippi:

```
a + a² + b = b²    where b = a + 1
```

This identity holds in any commutative ring. SCRAWL uses it for shared baselines, gnomon incremental chain updates, and a zero-cost algebraic check in `C_COMMIT`. **Identity authentication** (`I_VERIFY`) does *not* use that tautology alone: it recomputes the chain from the seed and depth. The complete treatment is in *The ML Identity Theorem Family* paper (17 theorems, 6 engineering applications), which is not shipped in this repository.

## Project Structure

```
scrawl/
├── src/                        # Rust implementation
│   ├── lib.rs
│   ├── opcodes.rs
│   ├── identity.rs
│   ├── registers.rs
│   ├── synapse.rs
│   ├── idioms.rs               # SCRAWL Idioms 1.0
│   └── vm.rs
├── examples/
│   ├── heartbeat.rs
│   └── idioms.rs
├── tests/
├── benches/                    # Criterion (cargo bench)
├── benchmarks/                 # Python bench source (requires unpublished src/)
├── self/programs/              # ROSETTA examples (compiler not in this repo)
├── docs/IDIOMS.md
├── README.md
├── BENCHMARKS.md
├── CHANGELOG.md
├── AUTHORS.md
├── FORKING.md
├── CONTRIBUTING.md
└── LICENSE                     # Apache 2.0
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Run `cargo test` and `cargo test --release` before opening a PR.

## Fork It

SCRAWL is Apache 2.0. See [FORKING.md](FORKING.md). If you build something with SCRAWL, open an issue titled `[Fork] Your Project Name`.

## License

Apache License 2.0 — use it, modify it, ship it. See [LICENSE](LICENSE).

---

**ML Innovations Research Lab** · M. L. McKnight · ml.innovation.research.lab@gmail.com · 662-295-2269 · Pheba, Mississippi · 2026

*Built on the ML Identity — discovered 1999, Raymond High School, Mississippi*
