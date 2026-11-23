# SCRAWL v1.2 — Benchmarks

Numbers in the size tables below were measured from the **Rust** implementation in this repository. The Python suite `benchmarks/bench_scrawl.py` is kept as source, but it cannot be run from this tree: it imports the unpublished Python `src` package.

## Run It Yourself

```bash
# Rust Criterion benches
cargo bench

# Python suite — requires the unpublished Python core (not in this repo)
# python benchmarks/bench_scrawl.py --quick
```

## What Gets Measured

| # | Benchmark | What It Tells You |
|---|-----------|-------------------|
| 1 | **VM Execution Throughput** | Instructions/sec across implemented opcodes |
| 2 | **SYNAPSE Encode/Decode** | Frames/sec and bytes/sec for the binary wire format |
| 3 | **Delta Compression** | First-send and identical-retransmit sizes on real payloads |
| 4 | **Identity Chain Derivation** | Chain generation speed at depths 8–256, handshake latency |
| 5 | **Consensus Round Latency** | Propose→vote→commit for 2, 5, and 10 agents |
| 6 | **Tensor Operations** | In-place operation throughput at varying sizes |
| 7 | **Wire Format Size** | SCRAWL binary vs equivalent JSON — byte-for-byte |
| 8 | **Idiom packing** | `cargo run --example idioms` — LEX_CALL vs expanded frames |

Timing methodology for the Python harness (when the core is available): `time.perf_counter()`, median of N trials after warmup, fresh instances, fixed seeds. Rust benches use Criterion.

## Wire Format Size: SCRAWL vs JSON (measured)

Same instruction lists as `benchmarks/bench_scrawl.py` `bench_wire_size`, encoded with the Rust `SynapseEncoder`. JSON sizes are `len(json.dumps(...).encode("utf-8"))` of the exact payloads in that file.

| Operation | SCRAWL | JSON | Savings | Ratio |
|-----------|--------|------|---------|-------|
| Identity derive + verify + halt | 36 B | 192 B | 81.2% | 5.3× smaller |
| Attention route (Q,K,V→out) + halt | 28 B | 113 B | 75.2% | 4.0× smaller |
| 3-agent consensus round + halt | 72 B | 471 B | 84.7% | 6.5× smaller |
| 10-instruction mixed program | 102 B | 664 B | 84.6% | 6.5× smaller |

Earlier typographic estimates (~24 B / ~16 B / ~48 B / ~72 B vs ~180 / ~120 / ~520 / ~680) understated the SYNAPSE frame (header + CRC is already about 14 bytes) and slightly overstated some JSON sizes. The compact-ISA savings remain large; the figures above are the ones to quote.

## Delta Compression (measured)

Rust `DeltaCompressor`, seed `0xBEEF`, depth 16. The first send XORs against a SHA-256 expectation stream and prepends a 1-byte flag. That XOR is not compressible by the zero-run RLE coder, so the first send is **raw + 1 byte**. A second send of an *identical* buffer XORs against the previous message (all zeros) and RLE-encodes the run: **3 bytes** when the payload is 255 bytes or less (1 flag + `0x00` + run length). Longer identical payloads use multiple RLE runs (294 B → 5 B).

| Payload | Raw | First send | Identical retransmit |
|---------|-----|------------|----------------------|
| Small agent state `pos=(100,200) hp=95 ammo=30` | 27 B | 28 B | 3 B |
| README example string `agent_position: x=100, y=200, health=95` | 39 B | 40 B | 3 B |
| `delta_state_sync` example state (52 B) | 52 B | 53 B | 3 B |
| 50-byte buffer | 50 B | 51 B | **3 B (94%)** |
| Medium bench string | 128 B | 129 B | 3 B |
| Large JSON bench payload | 294 B | 295 B | 5 B |

Earlier first-send claims (27 B → ~10 B, 120 B → ~35 B, 280 B → ~60 B) do not hold for this compressor. They may have been measured against the unpublished FLUX 2.0 pipeline, which is not in this repository. The **50 B → 3 B identical retransmit (94%)** claim is confirmed.

## Idiom packing (measured)

`cargo run --example idioms`:

| Idiom | Expanded frame | LEX_CALL frame | Savings |
|-------|----------------|----------------|---------|
| Heartbeat at Cafe | 40 B | 26 B | 35.0% |
| Named Heartbeat | 46 B | 26 B | 43.5% |
| The Round at Two | 63 B | 39 B | 38.1% |

## Contributing Benchmarks

Performance profiling is welcome (see [CONTRIBUTING.md](CONTRIBUTING.md)): ARM, constrained environments, comparisons with MCP/A2A, concurrent workloads, memory profiling. Please include `cargo bench` output or a `results.json` plus platform details.

---

**ML Innovations Research Lab** · M. L. McKnight · ml.innovation.research.lab@gmail.com · Pheba, Mississippi · 2026
