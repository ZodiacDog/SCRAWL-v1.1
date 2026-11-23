# SCRAWL Idioms 1.0

Darmok-style phrase speech for SCRAWL: a short named idiom stands for a whole multi-step command sequence. Agents that share this lexicon can write, speak, or transmit the phrase (or a compact `LEX_CALL`) instead of the expanded instruction list.

This document is the specification. The Rust module `src/idioms.rs` implements the registry, packing, expansion, lexicon fingerprint, and four executable built-ins.

## Why

SCRAWL's binary ISA is already much smaller than JSON-RPC. Idioms compress *further* at two layers:

1. **Human / planner text.** A phrase such as `Heartbeat at Cafe` replaces four ROSETTA lines or a block of host `Instruction(...)` calls.
2. **The wire.** `LEX_CALL` (opcode `0x60`) plus parameters replaces the expanded instruction list inside a SYNAPSE frame.

The starter table is taken from command sequences that actually repeat in the published examples, tests, and docs — not from invented semantics.

## Definition record

An idiom is a record:

| Field | Meaning |
|-------|---------|
| `id` | `u16`, sticky. Never reused if the idiom is retired. Common built-ins may also be executed as `LEX_CALL` (`0x60`) with this id as the first operand. |
| `phrase` | Short name, with optional parameter slots (`Heartbeat at {seed}`). |
| `kind` | `VmProgram` (expands to a SCRAWL instruction list) or `HostDelta` (host-level behavior, not a VM program). |
| `expansion` | Canonical instruction list. Must be deterministic: same slots → same opcodes and operands, every time. |
| `expansion_hash` | SHA-256 of the canonical SYNAPSE encoding of that list (used when computing the lexicon fingerprint). |
| `version_introduced` | Lexicon version in which the idiom first appeared. |
| `retired` | Optional flag. Retired ids stay in the table as holes. |

This is a formalization of the ROSETTA macro idea (`name`, `params`, deterministic expander) plus a stable id and a hash so two agents can agree without sending expander source.

## Standard lexicon (16 entries)

Phrases 1–4 are implemented in this crate. Phrases 5–16 are specified here for a later revision; they must not be sent as `LEX_CALL` until they are assigned a frozen expansion and an id in a minor lexicon bump.

**Text savings** compare the phrase to the compact expansion (characters, and whitespace-separated tokens as a rough proxy — not a model tokenizer). **Frame savings** are measured from this crate's `SynapseEncoder` where an implementation exists.

| # | Phrase | Exact expansion | Text savings | Frame / wire (measured) |
|---:|--------|-----------------|--------------|-------------------------|
| 1 | **Heartbeat at Cafe** | `CR0 = identity.derive(seed=0xCAFE, depth=16)` / `R1 = identity.verify(CR0, R0)` / `yield R1` / `halt`  (`I_DERIVE, I_VERIFY, X_YIELD, X_HALT`). Params: `{seed, depth}` default `0xCAFE, 16`. | Phrase 17c / 3 wt. vs ROSETTA 88c / 11 wt (**81% / 73%**); vs compact Python `Instruction` form 167c / 11 wt (**90% / 73%**). | Expanded **40 B** → `LEX_CALL` **26 B** (**35.0%**). |
| 2 | **Named Heartbeat** | `I_DERIVE, I_VERIFY, I_FINGERPRINT, X_YIELD, X_HALT` (the `heartbeat` example). Params: `{seed, depth}`. | Phrase 15c / 2 wt. vs ROSETTA 119c / 14 wt (**87% / 86%**); vs Python 214c / 14 wt (**93% / 86%**). | Expanded **46 B** → **26 B** (**43.5%**). |
| 3 | **The Round at Two** | `C_PROPOSE [prop, data_reg, [agent0, agent1]]`, `C_QUORUM [prop, 0.5]`, two `C_VOTE` APPROVE, `C_COMMIT`, `X_HALT`. Vote operands are `(prop_id, agent_id, vote)`. | Phrase 16c / 4 wt. vs compact Python 260c / 20 wt (**94% / 80%**). | Expanded **63 B** → **39 B** (**38.1%**). |
| 4 | **Silent Retransmit** | Host-level: second `DeltaCompressor.compress()` of an unchanged buffer (flags `0x02\|0x04`, RLE of zeros). Not a VM program; `expand()` returns an error if asked to emit `LEX_CALL`. | Replaces a repeated full-state command with “same as last.” | 50 B payload: first send 51 B, second send **3 B (94%)**. Payloads ≤ 255 B retransmit in 3 bytes. |
| 5 | **Three Approve at Proposal One** | Full `consensus_round.rosetta` body: identity preface, propose, quorum 0.67, three APPROVE votes, commit, yield, halt. | Phrase 29c / 5 wt. vs that ROSETTA body 409c / 30 wt (**93% / 83%**). | Unpacked 3-agent consensus frame is 72 B. Packing not implemented in 1.0. |
| 6 | **Quorum Failed at Two** | Same shape as #3 with a reject or unmet quorum (the rejection program in `consensus_vote.py`). | Same order of savings as #3 once parameterized. | Not implemented as `LEX_CALL` in 1.0. |
| 7 | **Identity then Propose** | `I_DERIVE, I_FINGERPRINT, C_PROPOSE, C_QUORUM, C_VOTE, C_VOTE, C_COMMIT, X_HALT` (`consensus_vote.py` happy path). | ~90% of the Python instruction list. | Not implemented as `LEX_CALL` in 1.0. |
| 8 | **Route the Query** | `A_ROUTE [0, 1, 2, 3]`, `X_HALT`. | Phrase 15c / 3 wt. vs Python 80c / 6 wt (**81% / 50%**). | Unpacked route+halt frame is 28 B. Packing not implemented in 1.0. |
| 9 | **Trust then Attend** | `I_DERIVE(seed=0xF00D, depth=8)`, `I_VERIFY`, `TR3 = attention.self(TR0)`, `yield R1`, `halt` (`fused_attention.rosetta`). | Phrase 17c / 3 wt. vs ROSETTA 113c / 14 wt (**85% / 79%**). | Not implemented as `LEX_CALL` in 1.0. |
| 10 | **Normalize then Dot** | `T_NORM(src1)`, `T_NORM(src2)`, `T_COMPOSE(dst, src1, src2, DOT)` — the `strided_compose` example macro. | Phrase 18c / 3 wt. vs Python 162c / 13 wt (**89% / 77%**). | Not implemented as `LEX_CALL` in 1.0. |
| 11 | **Fill then Compose** | `T_FILL(0, 1.0)`, `T_FILL(1, 2.0)`, `T_COMPOSE(..., DOT)` from the mixed program bench. | Phrase 17c / 3 wt. vs Python 140c / 11 wt (**88% / 73%**). | Not implemented as `LEX_CALL` in 1.0. |
| 12 | **Handshake at Beef** | `IdentityHandshake.initiate(seed, depth)`, `.respond(...)`, `.derive_shared_key_with_lexicon(...)`. Usual example seed `0xBEEF`. | Phrase 17c / 3 wt. vs that Python block 230c / 16 wt (**93% / 81%**). | Host API, not a VM program. |
| 13 | **Delta the State** | After a shared baseline: `sender.compress(state)` / `receiver.decompress(compressed)` / equality check. | Phrase 15c / 3 wt. vs the three-line body 105c / 10 wt (**86% / 70%**). | Host API. |
| 14 | **Encode the Frame** | `SynapseEncoder.encode_frame(...)` then `SynapseDecoder.decode_frame(...)`. | Phrase 16c / 3 wt. vs the four-line Python block 140c / 13 wt (**89% / 77%**). | Host API. |
| 15 | **Rosetta Roundtrip** | `compile_program` → encode → decode → `decompile` → `vm.execute`. Requires the unpublished Python ROSETTA compiler. | Phrase 18c / 2 wt. vs an 8-line host pipeline 282c / 27 wt (**~94% / ~93%**). | Not implemented here (no compiler in this repo). |
| 16 | **Gnomon Walk** | Loop: `a_sq = ml_gnomon_update(a_sq, a)` then `ml_algebraic_verify(...)`. | A short phrase for the “walk the chain and check the identity” ritual. | Host API. |

A natural additional name, already used in earlier docs, is **`fused_attention`**. Keep that macro name rather than renaming it; it is not a numbered Idioms 1.0 built-in until an expansion is frozen.

Do not start with State-domain sequences (`S_LOCK` / `S_DELTA` / …) or `I_CHALLENGE` / `I_RESPOND` until those have attested expansions.

Ids actually allocated in the crate today:

| Id | Phrase |
|---:|--------|
| 1 | Heartbeat at Cafe |
| 2 | Named Heartbeat |
| 3 | The Round at Two |
| 4 | Silent Retransmit (host-level) |

## Versioning

- The lexicon name is the string `SCRAWL Idioms 1.0`. The numeric `LEXICON_VERSION` starts at `1`.
- **Major bump:** any already-shipped idiom changes expansion or parameter meaning. Old ids are never reused. Retired ids remain as holes so a stale sender gets a clean trap, not a different meaning.
- **Minor bump:** new idioms appended with fresh ids. An old receiver that does not know id 17 must trap or request the expansion, not guess.
- The handshake carries the lexicon fingerprint (see below). Equal versions with unequal fingerprints are a hard error. Unequal versions: intersect on the older set, or send the expanded form for unknown ids.

## Lexicon fingerprint and the handshake

`idioms::lexicon_fingerprint()` returns an 8-byte SHA-256 prefix of a canonical encoding of the table (name, version, each id, phrase, and the SYNAPSE encoding of the default expansion, or the description for host-level idioms).

After an identity handshake (`IdentityHandshake::initiate` / `respond`), agents derive a shared key with:

```rust
IdentityHandshake::derive_shared_key_with_lexicon(
    &baseline, agent_a, agent_b, &lexicon_fingerprint(),
)
```

A mismatched lexicon fingerprint changes the key. Agents that do not share Idioms 1.0 cannot silently disagree about what “Three Approve at Proposal One” (or “Heartbeat at Cafe”) means.

`derive_shared_key` without the lexicon argument still exists; it binds an all-zero fingerprint and is **not** lexicon-safe.

## Wire format

- Opcode **`LEX_CALL = 0x60`** (reserved extension range `0x60–0xFF`).
- Operands: `idiom_id: int`, then the idiom's parameter operands.
- SYNAPSE flag bit **`0x01` (`SYNAPSE_FLAG_LEXICON`)** means the payload may contain `LEX_CALL`. `idioms::encode_with_idioms` sets this flag when packing.
- Expansion always remains legal SYNAPSE: idioms are compression, not a second ISA. A sender that is unsure of the receiver's lexicon should send the expanded list.
- **Unknown id, host-level id used as `LEX_CALL`, or expansion error must trap.** It is never a silent NOP. This matches ROSETTA “strict mode” and the v1.2 VM rule for unimplemented opcodes.

The VM expands `LEX_CALL` at execute time (`idioms::expand`) and runs the resulting instructions against the same register file.

## Packing and longest match

`idioms::pack_idioms` walks an instruction list and replaces known sequences with `LEX_CALL`, **longest match first** (Named Heartbeat, 5 instructions, before Heartbeat at Cafe, 4). It does not automatically pack a subsequence if a longer built-in matches.

Do not automatically pack a subsequence when the surrounding program needs intermediate registers for something else; prefer an explicit `lex_call(...)`.

## Ambiguity and collisions

- **Phrase uniqueness.** Normalize before insert: squeeze whitespace, casefold. A new phrase that normalizes equal to an existing one is rejected unless it is an explicit alias of the same expansion hash.
- **Parse ambiguity.** Longest-phrase match in text; parameters only in declared slots. Prefer bound forms (`Heartbeat at seed=0xCAFE`) when a value looks like a parameter.
- **Id collisions.** Two records may not share an `id` with different expansions. Detect at lexicon compose time and at bind time.
- **Expansion collisions.** Two phrases with the same expansion hash are aliases (one primary phrase). The wire form uses the id, so spoken-phrase aliases do not fork the protocol.
- **Cross-lexicon collisions.** Different lexicons may reuse ids; the handshake fingerprint plus the lexicon name/version scopes them. Never decode an id without a matching fingerprint.
- **Vote layout.** Consensus idioms use the v1.2 canonical `C_VOTE` operands `(proposal_id, agent_id, vote_value)`.

## Sharing a custom lexicon

After a successful identity handshake, an agent may publish a lexicon delta: new `(id, phrase, expansion)` records, themselves delta-compressed against the shared baseline, to be bound only after each expansion hash is checked and collision rules pass. Custom lexicons must not silently overload ids 1–4 of Idioms 1.0. Domain forks (`FORKING.md`, range `0x60+`) should ship a *named* lexicon.

`I_BIND` is reserved in the ISA for a future bind operation; it still traps in this VM.

## Implementation status in this crate

| Feature | Status |
|---------|--------|
| Standard table + lookup + phrase normalize | Implemented (`src/idioms.rs`) |
| Expand + pack Heartbeat at Cafe, Named Heartbeat, The Round at Two | Implemented |
| Silent Retransmit (host `DeltaCompressor`) | Implemented and tested (50 B → 3 B) |
| `LEX_CALL` execution in the VM | Implemented; unknown id traps |
| SYNAPSE flag `0x01` | Implemented |
| Lexicon fingerprint in `derive_shared_key_with_lexicon` | Implemented |
| Idioms 5–16 as `LEX_CALL` | Specified only |
| ROSETTA text parser / compiler | Not in this repository |

```bash
cargo test
cargo run --example idioms
```

---

**ML Innovations Research Lab** · M. L. McKnight · Pheba, Mississippi · 2026  
**Contributor:** Korg (AI assistant, Grok Bot)
