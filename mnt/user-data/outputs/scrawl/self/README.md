# SCRAWL-in-SCRAWL

**SCRAWL programs that implement SCRAWL's own primitives.**

This directory contains ROSETTA pseudocode programs that exercise SCRAWL's core subsystems using SCRAWL's own instruction set. This is self-hosting — the ISA is expressive enough to implement the operations it was designed to support.

## Programs

| Program | What It Implements | Instructions |
|---------|-------------------|-------------|
| `identity_chain.rosetta` | ML Identity chain derivation and fingerprint verification | Identity + Execution |
| `consensus_round.rosetta` | Full propose→vote→commit consensus cycle | Consensus + Identity + Execution |
| `fused_attention.rosetta` | Identity-verified attention computation pipeline | Attention + Identity + Execution |

## Running

These are ROSETTA pseudocode programs. Compile and execute them using the Python ROSETTA compiler:

```python
from src.rosetta import compile_program
from src.vm import ScrawlVM

with open("self/programs/identity_chain.rosetta") as f:
    source = f.read()

program = compile_program(source, strict=False)
vm = ScrawlVM()
result = vm.execute(program)
print(f"Success: {result.success}, Yielded: {result.yielded_values}")
```

## Why This Matters

Self-hosting is a standard expressiveness proof for programming languages. If a language can implement itself, it has sufficient expressive power for general computation within its domain. These programs demonstrate that SCRAWL's 84-opcode ISA is not merely a collection of primitives — it's a coherent computational model that can express its own operational semantics.

**For fork builders:** If you're extending SCRAWL with domain-specific opcodes (0x60–0xFF), write a self-hosted program that exercises your extension. If you can't express your domain's core operation as a SCRAWL program, your opcodes might need redesigning.

---

**ML Innovations LLC** · M. L. McKnight · Pheba, Mississippi · 2026
