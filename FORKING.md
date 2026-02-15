# Fork SCRAWL — Build What We Haven't

SCRAWL is Apache 2.0 licensed. That means you can fork it, modify it, embed it in proprietary systems, and ship it — no copyleft, no license contamination, no asking permission. The patent grant in Apache 2.0 covers the ML Identity applications, so you can build on the mathematical foundation without ambiguity.

We didn't open-source this so it could sit in one repo. We open-sourced it so people would build things we haven't thought of yet.

## What We'd Love to See Forked

### Domain-Specific ISA Extensions

SCRAWL reserves opcode range `0x60–0xFF` for domain-specific extensions. The base ISA covers the six domains every agent needs (tensor, attention, execution, state, consensus, identity). Your domain probably needs more.

**Fork ideas:**
- **Robotics:** Motor control opcodes, sensor fusion primitives, path planning instructions
- **Finance:** Order book operations, risk computation, settlement consensus
- **Healthcare:** HL7/FHIR message encoding, clinical decision support primitives
- **Gaming:** Physics update compression, lobby consensus, anti-cheat verification
- **IoT/Edge:** Low-power sleep scheduling, mesh routing opcodes, sensor aggregation

Build the extension, register opcodes in the 0x60+ range, add ROSETTA compile/decompile support for your new opcodes, and you have a domain-specific agent communication layer that inherits all of SCRAWL's infrastructure: VM execution, binary wire format, delta compression, trace hooks, and algebraic verification.

### Language Ports

SCRAWL is pure Python today. The ISA and wire format are language-independent.

**Port ideas:**
- **Rust:** Zero-copy SYNAPSE decoder, no-alloc VM for embedded agents
- **Go:** Native concurrency for multi-agent consensus workloads
- **C/C++:** Bare-metal deployment for drone swarms, satellites, edge hardware
- **JavaScript/WASM:** Browser-based agent visualization and debugging
- **Zig:** Safety-critical deployment without runtime overhead

The SYNAPSE wire format spec (magic bytes, CRC-32C, varint encoding) is fully documented in `src/synapse.py`. Any implementation that speaks SYNAPSE can interop with any other — regardless of language.

### Bridge Modules

SCRAWL doesn't replace your existing agent framework — it replaces the wire format underneath it.

**Bridge ideas:**
- **MCP bridge:** Translate MCP JSON-RPC messages to/from SCRAWL instruction streams
- **A2A bridge:** Google's Agent-to-Agent protocol backed by SCRAWL transport
- **LangChain adapter:** SCRAWL as the inter-agent communication layer in LangChain pipelines
- **AutoGen bridge:** Microsoft AutoGen agents using SCRAWL for fast, compact message exchange
- **CrewAI adapter:** CrewAI task delegation via SCRAWL consensus opcodes
- **ROS2 bridge:** Robot Operating System agents communicating via SCRAWL

Bridge modules should live in a separate package (not in `src/`) so they can have external dependencies without violating SCRAWL's zero-dependency core.

### Research Forks

The ML Identity (`a + a² + b = b²`) has more applications than we've implemented. The theorem paper documents 17 theorems and 6 engineering applications — we've built 4 of them into SCRAWL. The remaining ones are open territory.

**Research directions:**
- Finite field implementations for post-quantum verification
- Modular arithmetic chains for homomorphic operations
- Triangular-number memory allocators for GPU tensor layouts
- Identity-based zero-knowledge proofs
- Gnomon-accelerated hash chains

## How to Fork Effectively

### Keep the Tests Running

SCRAWL has 552 tests for a reason. When you fork:

```bash
# Your fork should still pass these
python tests/test_scrawl.py       # 360 core tests
python flux_scrawl_tests.py       # 116 FLUX tests
python tests/test_v11.py          # 76 v1.1 tests

# Plus your new tests
python tests/test_your_extension.py
```

If your extension breaks a core test, you've broken interoperability. Fix it before shipping.

### Maintain ROSETTA Roundtrip

If you add opcodes, both the decompiler and compiler must handle them:

```
Your opcode → decompile → human-readable pseudocode → compile → same opcode
```

This roundtrip guarantee is what makes SCRAWL auditable. Break it and your fork loses one of the most valuable properties of the system.

### Preserve Zero Dependencies in Core

Your bridge modules, adapters, and extensions can depend on anything. The `src/` directory cannot. This is the contract that lets SCRAWL run on a bare Python install, a Raspberry Pi, a locked-down government system, or anywhere else that can't install packages.

### Respect Integer Determinism

Identity chains use integer arithmetic only. No `float` in the identity path. This isn't a style choice — it's what guarantees cross-platform determinism. See Section 2.3 of the ML Identity paper for the formal justification.

## Upstream Contributions

Built something good in your fork? We want it back upstream. See [CONTRIBUTING.md](CONTRIBUTING.md) for the process. We especially want:

- New ROSETTA macros that other developers would use
- New opcodes in the extension range with full test coverage
- Performance benchmarks on platforms we haven't tested
- Bridge modules for popular agent frameworks
- Bug reports with reproduction steps

## Let Us Know

If you fork SCRAWL and build something with it, tell us. Open an issue titled `[Fork] Your Project Name` with a link and a one-paragraph description. We'll maintain a list of notable forks in the README.

We built the foundation. Now go build something on top of it.

---

**ML Innovations LLC** · M. L. McKnight · ml.innovations.research.lab@gmail.com · 662-295-2269 · Pheba, Mississippi · 2026

*Apache 2.0 — use it, modify it, ship it.*
