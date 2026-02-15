# SCRAWL v1.1 — Benchmarks

Performance measurements across all major subsystems. Every benchmark runs on a bare Python install with zero external dependencies.

## Run It Yourself

```bash
# Standard (10 trials per benchmark, ~30 seconds)
python benchmarks/bench_scrawl.py

# Quick mode (~5 seconds)
python benchmarks/bench_scrawl.py --quick

# Custom trials for tighter confidence intervals
python benchmarks/bench_scrawl.py --trials 50

# Machine-readable output (writes benchmarks/results.json)
python benchmarks/bench_scrawl.py --json
```

## What Gets Measured

| # | Benchmark | What It Tells You |
|---|-----------|-------------------|
| 1 | **VM Execution Throughput** | Instructions/sec across all 6 domains — the core speed limit |
| 2 | **SYNAPSE Encode/Decode** | Frames/sec and bytes/sec for binary wire format |
| 3 | **Delta Compression** | Compression ratios on real agent payloads, vs raw and vs JSON |
| 4 | **ROSETTA Compile/Decompile** | Transpiler throughput and full roundtrip latency |
| 5 | **Identity Chain Derivation** | Chain generation speed at depths 8–256, handshake latency |
| 6 | **Consensus Round Latency** | Propose→vote→commit for 2, 5, and 10 agents |
| 7 | **Tensor Operations** | In-place operation throughput at varying sizes |
| 8 | **Wire Format Size** | SCRAWL binary vs equivalent JSON — byte-for-byte comparison |

## Methodology

- **Timing:** `time.perf_counter()` wall-clock, highest resolution available
- **Statistics:** Median of N trials (default N=10) after warmup runs — median resists outliers better than mean
- **Isolation:** Each benchmark creates fresh VM/encoder/compressor instances — no shared state
- **Determinism:** Identity chains use fixed seeds; results are reproducible across runs and platforms
- **No cheating:** JSON comparisons use equivalent semantics, not strawman payloads

## Wire Format Size: SCRAWL vs JSON

This is the comparison that matters most for bandwidth-constrained environments. Same operations, different encodings:

| Operation | SCRAWL | JSON-RPC | Savings | Ratio |
|-----------|--------|----------|---------|-------|
| Identity derive + verify | ~24B | ~180B | ~87% | ~7.5x smaller |
| Attention route (Q,K,V→out) | ~16B | ~120B | ~87% | ~7.5x smaller |
| 3-agent consensus round | ~48B | ~520B | ~91% | ~10.8x smaller |
| 10-instruction mixed program | ~72B | ~680B | ~89% | ~9.4x smaller |

> *Exact sizes depend on operand values; these are representative. Run the benchmark for your platform's exact numbers.*

## Delta Compression

When two agents share a baseline (established via `IdentityHandshake`), subsequent messages transmit only the XOR delta:

| Payload Type | Raw Size | Compressed | Savings |
|-------------|----------|------------|---------|
| Small agent state | 27B | ~10B | ~63% |
| Medium agent state | 120B | ~35B | ~71% |
| Large JSON state | 280B | ~60B | ~79% |
| Identical retransmit | 50B | 3B | **94%** |

The identical-retransmit case is real: agents that poll status without changes send 3 bytes instead of repeating the full state. In swarm scenarios with 100+ agents, this is the difference between saturating comms and staying silent.

## What These Numbers Mean

**For agent framework developers:** SCRAWL's binary encoding is 7–11x smaller than JSON-RPC for equivalent operations. If your agents exchange thousands of messages per second, switching the wire format from JSON to SCRAWL eliminates most of your serialization overhead.

**For swarm/multi-agent systems:** Consensus is a VM operation, not an application-layer protocol. A 10-agent propose→vote→commit cycle completes in microseconds, not milliseconds. The trace hook overhead is minimal — you get full auditability without meaningful performance cost.

**For bandwidth-constrained environments:** Delta compression on structured agent state saves 70–94%. Combined with the compact binary encoding, SCRAWL can reduce total bandwidth by an order of magnitude compared to text-based protocols. This matters for satellite links, mesh networks, and contested RF environments.

**For correctness-critical systems:** Integer-only identity chains are deterministic across platforms. The same seed produces the same chain on Linux, macOS, and Windows across Python 3.10–3.12. The benchmark verifies this — `all_int=True` for every depth.

## Contributing Benchmarks

Performance profiling is one of the highest-value contributions (see [CONTRIBUTING.md](CONTRIBUTING.md)). Things we'd especially like to see:

- Benchmarks on ARM (Raspberry Pi, Apple Silicon M-series, Graviton)
- Benchmarks on constrained environments (MicroPython, low-memory)
- Head-to-head comparisons with MCP/A2A message overhead
- Throughput under concurrent multi-agent workloads
- Memory profiling (peak RSS during chain derivation, tensor ops)

Submit results as a PR with your `benchmarks/results.json` and platform details.

---

**ML Innovations LLC** · M. L. McKnight · ml.innovations.research.lab@gmail.com · Pheba, Mississippi · 2026
