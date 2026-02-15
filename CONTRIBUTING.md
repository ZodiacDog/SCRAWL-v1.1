# Contributing to SCRAWL

Thanks for considering a contribution. SCRAWL is built by ML Innovations LLC and we welcome outside involvement — this project gets stronger with more eyes on it.

## What We're Looking For

**High-value contributions (we'll review these fast):**

- **New ROSETTA macros.** If you've built a compound operation that other agent developers would use, submit it. Macros must be deterministic — same arguments, same instruction sequence, every time.
- **New opcodes in the 0x60–0xFF extension range.** The ISA reserves this space for domain-specific extensions. If you need an opcode that doesn't exist, propose it with: (1) the operation it performs, (2) its operand types, (3) a test that exercises it, and (4) a ROSETTA decompile/compile roundtrip.
- **Bug reports with reproduction steps.** If a test fails on your platform, tell us your Python version, OS, and the exact output. Cross-platform determinism is a core guarantee — if it breaks, we fix it immediately.
- **Performance benchmarks.** We test correctness exhaustively but performance profiling is always welcome. Especially: VM execution throughput, SYNAPSE encoding speed, delta compression ratios on real workloads.

**Also welcome:**

- Documentation improvements
- Additional examples
- Integration guides (connecting SCRAWL to other agent frameworks)
- Typo fixes

## How to Contribute

1. **Fork the repo** and create a branch from `main`.
2. **Make your change.** If it touches `src/`, add or update tests in `tests/`.
3. **Run all three test suites:**
   ```bash
   python tests/test_scrawl.py       # Must pass 360/360
   python flux_scrawl_tests.py       # Must pass 116/116
   python tests/test_v11.py          # Must pass 76/76
   ```
4. **Open a PR.** Describe what you changed and why. Link any relevant issues.

## Rules

- **Zero external dependencies.** SCRAWL's core (`src/`) must run on a bare Python 3.10+ install. No numpy, no protobuf, no anything. If your change adds an `import` that isn't in the Python standard library, it belongs in a separate bridge module, not in `src/`.
- **Tests are mandatory.** No PR merges without passing all 552 tests plus any new tests for the new feature.
- **ROSETTA roundtrip.** If you add an opcode, both the decompiler and compiler must handle it. If `decompile → compile → decompile` doesn't produce identical output, it doesn't ship.
- **Integer determinism.** Any code that touches identity chains, baselines, or fingerprints must use integer arithmetic only. No `float` in the identity path. Section 2.3 of the ML Identity paper explains why.

## Code Style

- Pure Python, no type annotations required but appreciated
- Docstrings on public functions
- Module-level comments explaining what the file does
- Orange project titles in documentation (#FF6600 with black trim, per UAIP v1.0)

## Questions?

Open an issue. We respond.

---

**ML Innovations LLC** · M. L. McKnight · ml.innovations.research.lab@gmail.com · Pheba, Mississippi
