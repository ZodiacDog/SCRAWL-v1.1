# SCRAWL v1.1

**Structured Compressed Runtime for Agent-to-Agent Workload Language**

```
552 tests (Python) + full Rust test suite | 100% pass | zero dependencies | 3 implementations
```

SCRAWL is a binary instruction set and runtime for AI agent communication. Where existing protocols treat agent messages as text to be parsed, SCRAWL treats them as programs to be executed — 84 opcodes across six domains, transmitted in a compact binary wire format, with delta compression derived from the ML Identity equation. One agent compiles intent into SCRAWL instructions; the receiving agent executes them directly on a register-based VM.

**Three implementations, one ISA:**

| Implementation | Language | Status | Run It |
|---------------|----------|--------|--------|
| [python/](python/) | Python 3.10+ | 552/552 tests, production-ready | `python python/examples/heartbeat.py` |
| [rust/](rust/) | Rust 1.70+ | Full test suite, zero `unsafe` in hot paths | `cd rust && cargo run --example heartbeat` |
| [self/](self/) | SCRAWL (ROSETTA) | Self-hosted proof of expressiveness | See [self/README.md](self/README.md) |

All three implementations share the same ISA, the same SYNAPSE wire format, and produce identical identity chains from the same seed. A Python agent and a Rust agent can communicate over SCRAWL without translation.

## Why SCRAWL?

**The problem:** AI agents talking to each other waste most of their bandwidth on syntax. JSON-RPC messages, protocol negotiation, schema validation — all overhead that exists because we're forcing agents to communicate the way humans read.

**What SCRAWL does differently:**

- **Binary ISA, not text protocol.** 84 instructions encode in 4–12 bytes each. A consensus round that takes 520+ bytes in JSON is 48 bytes in SCRAWL.
- **Execute, don't parse.** The receiving agent runs instructions directly on a register-based VM.
- **Delta compression from first principles.** Agents establish a shared mathematical baseline (ML Identity: `a + a² + b = b²`). Subsequent messages transmit only the XOR delta. Identical messages compress to 3 bytes.
- **Consensus built into the ISA.** `C_PROPOSE`, `C_VOTE`, `C_COMMIT` are native opcodes.
- **Zero external dependencies.** Both Python and Rust implementations: no numpy, no protobuf, no tokio, no serde.
- **Auditable by design.** ROSETTA transpiles any binary to readable pseudocode and back. Every roundtrip is deterministic.

## Quick Start

**Python:**
```bash
git clone https://github.com/mlinnovations/scrawl.git
cd scrawl
python python/examples/heartbeat.py
```

**Rust:**
```bash
cd rust
cargo run --example heartbeat
```

**5-line agent heartbeat (Python):**
```python
from src.vm import ScrawlVM
from src.synapse import Instruction
from src.opcodes import IdentityOp, ExecutionOp

vm = ScrawlVM()
vm.execute([
    Instruction(IdentityOp.I_DERIVE, [0, 0xCAFE, 16]),
    Instruction(IdentityOp.I_VERIFY, [0, 0, 1]),
    Instruction(ExecutionOp.X_YIELD, [1]),
    Instruction(ExecutionOp.X_HALT),
])
print(f"Identity verified: {vm.registers.get_reg(1) == 1}")  # True
```

## Benchmarks

Both implementations include benchmark suites. Run them:

```bash
# Python
python python/benchmarks/bench_scrawl.py --quick

# Rust
cd rust && cargo bench
```

**Wire format size — SCRAWL vs JSON-RPC:**

| Operation | SCRAWL | JSON-RPC | Savings |
|-----------|--------|----------|---------|
| Identity derive + verify | ~24B | ~180B | **~87%** |
| Attention route (Q,K,V→out) | ~16B | ~120B | **~87%** |
| 3-agent consensus round | ~48B | ~520B | **~91%** |
| 10-instruction mixed program | ~72B | ~680B | **~89%** |

**Delta compression on real payloads:**

| Payload | Raw | Compressed | Savings |
|---------|-----|------------|---------|
| Small agent state | 27B | ~10B | ~63% |
| Large JSON state | 280B | ~60B | ~79% |
| Identical retransmit | 50B | 3B | **94%** |

Full methodology: [BENCHMARKS.md](BENCHMARKS.md)

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    ROSETTA v1.1                      │
│         Human-readable ↔ Binary transpiler           │
│         84 opcodes + macro layer                     │
├─────────────────────────────────────────────────────┤
│                   SYNAPSE v1.0                       │
│          Binary wire format + CRC-32C                │
│     Magic | Version | Flags | SeqID | Payload | CRC  │
├─────────────────────────────────────────────────────┤
│                  SCRAWL VM v1.1                      │
│     Register-based execution engine                  │
│     256 GP + 64 tensor + 16 context registers        │
├──────────┬──────────┬──────────┬────────────────────┤
│ Tensor   │ Attention│ Execution│ State | Consensus   │
│ 15 ops   │ 14 ops   │ 16 ops   │ 15+12 ops          │
├──────────┴──────────┴──────────┴────────────────────┤
│              ML Identity Foundation                  │
│    a + a² + b = b² (integer-only, cross-platform)    │
└─────────────────────────────────────────────────────┘
```

**Six operational domains — 84 total opcodes:**

- **Tensor** (15): compose, decompose, transform, reshape, slice, reduce, quantize, broadcast, fill, copy, compare, convert, normalize, random, einsum
- **Attention** (14): route, mask, focus, scatter, gather, cross, self, multi-head, sparse, linear, flash, window, pool, topk
- **Execution** (16): nop, halt, yield, abort, branch, loop, call, return, fork, join, trap, resume, spawn, kill, sleep, wake
- **State** (15): sync, lock, unlock, delta, apply, snapshot, restore, publish, subscribe, watch, cas, load, store, evict, prefetch
- **Consensus** (12): propose, vote, commit, reject, quorum, escalate, timeout, revoke, delegate, audit, veto, ratify
- **Identity** (12): derive, verify, baseline, reconstruct, rotate, challenge, respond, bind, unbind, fingerprint, chain, split

## The ML Identity

SCRAWL's mathematical foundation, discovered by M. L. McKnight in 1999 at Raymond High School, Mississippi:

```
a + a² + b = b²    where b = a + 1
```

This identity holds in any commutative ring. SCRAWL uses it for shared baselines, delta compression (70–94%), zero-cost algebraic verification, and gnomon incremental chain updates. The complete treatment is in *The ML Identity Theorem Family* paper (17 theorems, 6 engineering applications).

## Project Structure

```
scrawl/
├── python/                     # Reference implementation
│   ├── src/                    # 6 modules, 3,386 lines, zero dependencies
│   ├── tests/                  # 436 tests (360 core + 76 v1.1)
│   ├── examples/               # 4 runnable demos
│   └── benchmarks/             # 8-subsystem benchmark suite
├── rust/                       # Rust implementation
│   ├── src/                    # 5 modules, zero runtime dependencies
│   ├── tests/                  # Comprehensive test coverage
│   ├── benches/                # Criterion benchmarks
│   ├── examples/               # Rust examples
│   └── Cargo.toml
├── self/                       # SCRAWL-in-SCRAWL
│   ├── programs/               # Self-hosted ROSETTA programs
│   └── README.md
├── .github/workflows/ci.yml   # Python 3×3 matrix + Rust 3-OS
├── README.md
├── BENCHMARKS.md               # Methodology & results
├── FORKING.md                  # Guide to extending SCRAWL
├── CONTRIBUTING.md
└── LICENSE                     # Apache 2.0
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Issues welcome. PRs for new macros, opcodes, benchmarks, and language ports especially.

## Fork It

SCRAWL is Apache 2.0. Fork it, extend it, ship it. See [FORKING.md](FORKING.md) for the full guide — domain-specific ISA extensions, language ports, bridge modules, research forks. If you build something with SCRAWL, open an issue titled `[Fork] Your Project Name`.

## License

Apache License 2.0 — use it, modify it, ship it. See [LICENSE](LICENSE).

---

**ML Innovations LLC** · M. L. McKnight · ml.innovations.research.lab@gmail.com · 662-295-2269 · Pheba, Mississippi · 2026

*Built on the ML Identity — discovered 1999, Raymond High School, Mississippi*
