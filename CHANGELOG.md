# Changelog

## [1.2.0] — 2026-11-22

Local working tree. Not yet published to GitHub.

Created by M. L. McKnight / ML Innovations Research Lab. v1.2 fixes and the idiom layer by Korg (AI assistant, Grok Bot).

### Fixed

- Proper Cargo layout (`src/`, `examples/`, `benches/`, `tests/`) so `cargo build`, `cargo test`, `cargo test --release`, and `cargo run --example heartbeat` work out of the box.
- `C_COMMIT` no longer fails to compile (E0499: `emit_trace` while a proposal was mutably borrowed).
- `I_VERIFY` / `ml_identity_verify` no longer panic in debug on realistic 32-bit chain values (i128 intermediates).
- `I_VERIFY` is now a real check: it recomputes `ml_identity_chain(seed, depth)` and the fingerprint and compares them to the stored baseline. Tampered chains and fingerprints fail. Tests cover this.
- Unimplemented opcodes no longer silently NOP. They trap with a clear error message.
- `C_VOTE` canonical operands are `(proposal_id, agent_id, vote_value)`. A two-operand form `(proposal_id, vote_value)` still uses `vm.agent_id`.
- README and BENCHMARKS quotes use measured SYNAPSE and delta sizes. The true 3-byte identical-retransmit path is kept. The `github.com/mlinnovations/scrawl` URL is replaced with `https://github.com/ZodiacDog/SCRAWL-v1.1`. The missing Python core is stated clearly, not fabricated.
- LICENSE copyright placeholder filled as `Copyright 2026 M. L. McKnight, ML Innovations Research Lab`. Contact email and phone already present in the project are unchanged.

### Added

- **SCRAWL Idioms 1.0** (`src/idioms.rs`, `docs/IDIOMS.md`, `examples/idioms.rs`): Darmok-style phrase compression. Registry, longest-match packer, `LEX_CALL` (0x60), SYNAPSE flag bit 0x01, lexicon fingerprint in `derive_shared_key_with_lexicon`, trap on unknown idiom.
- Built-in executable idioms: Heartbeat at Cafe, Named Heartbeat, The Round at Two; host-level Silent Retransmit (50 B → 3 B on the second send).
- Additional straightforward opcodes (see lists below).
- Per-instruction timeout is now enforced (the field existed in v1.1 but was unused).
- `IdentityBaseline::verify_integrity`, `Opcode::from_byte` is a safe match (no `transmute`).

### Opcode coverage (VM)

**Implemented (39 of 84 base + `LEX_CALL`):**

- Execution: `X_NOP`, `X_HALT`, `X_YIELD`, `X_ABORT`, `X_TRAP`, `X_BRANCH`, `X_LOOP`, `X_CALL`, `X_RETURN`
- Identity: `I_DERIVE`, `I_BASELINE`, `I_VERIFY`, `I_FINGERPRINT`, `I_CHAIN`
- Consensus: `C_PROPOSE`, `C_VOTE`, `C_QUORUM`, `C_COMMIT`, `C_REJECT`, `C_AUDIT`
- Tensor: `T_FILL`, `T_NORM`, `T_COPY`, `T_RESHAPE`, `T_COMPOSE` (dot, hadamard), `T_COMPARE`, `T_CONVERT`, `T_REDUCE`
- Attention: `A_ROUTE`, `A_SELF`
- State: `S_SYNC`, `S_LOCK`, `S_UNLOCK`, `S_SNAPSHOT`, `S_RESTORE`, `S_LOAD`, `S_STORE`, `S_CAS`, `S_EVICT`
- Extension: `LEX_CALL` (0x60)

**Still trapping (45 of 84) — not silent NOPs:**

- Execution: `X_FORK`, `X_JOIN`, `X_RESUME`, `X_SPAWN`, `X_KILL`, `X_SLEEP`, `X_WAKE`
- Identity: `I_RECONSTRUCT`, `I_ROTATE`, `I_CHALLENGE`, `I_RESPOND`, `I_BIND`, `I_UNBIND`, `I_SPLIT`
- Consensus: `C_ESCALATE`, `C_TIMEOUT`, `C_REVOKE`, `C_DELEGATE`, `C_VETO`, `C_RATIFY`
- Tensor: `T_DECOMPOSE`, `T_TRANSFORM`, `T_SLICE`, `T_QUANTIZE`, `T_BROADCAST`, `T_RANDOM`, `T_EINSUM`
- Attention: `A_MASK`, `A_FOCUS`, `A_SCATTER`, `A_GATHER`, `A_CROSS`, `A_MULTI_HEAD`, `A_SPARSE`, `A_LINEAR`, `A_FLASH`, `A_WINDOW`, `A_POOL`, `A_TOPK`
- State: `S_DELTA`, `S_APPLY`, `S_PUBLISH`, `S_SUBSCRIBE`, `S_WATCH`, `S_PREFETCH`

These still trap because there was no precise operand specification in the published tree, or they need runtime features (threads, async sleep, a real pub/sub fabric) that a conservative first pass should not invent.

### Breaking / behavior changes

- Programs that relied on unimplemented opcodes being silent NOPs will now trap.
- `I_VERIFY` will fail if the stored chain does not recompute from the seed (the old check was a tautology and always passed in release).
- `C_VOTE` with three operands is `(prop, agent, vote)`, not `(prop, vote, unused)`.
- `derive_shared_key_with_lexicon` is the handshake entry point that binds Idioms 1.0.

## [1.1.0] — 2026-02-15

Initial public Rust upload of the 84-opcode ISA, SYNAPSE, identity, and a partial VM. Layout and several behaviors were later corrected in 1.2.0.
