"""
SCRAWL v1.1 — Performance Benchmarks

Measures throughput, latency, and compression across all major subsystems:

    1. VM Execution Throughput         — instructions/sec across all 6 domains
    2. SYNAPSE Encode/Decode           — frames/sec and bytes/sec
    3. Delta Compression               — ratio vs raw, ratio vs JSON baseline
    4. ROSETTA Compile/Decompile       — roundtrip throughput
    5. Identity Chain Derivation       — chains/sec at varying depths
    6. Consensus Round Latency         — propose→commit microseconds
    7. Tensor Operations               — ops/sec for core tensor math
    8. Wire Format Size Comparison     — SCRAWL vs JSON for equivalent payloads

All benchmarks use wall-clock time (time.perf_counter) and report
median of N trials to reduce noise. No external dependencies.

Usage:
    python benchmarks/bench_scrawl.py
    python benchmarks/bench_scrawl.py --trials 20 --warmup 3
    python benchmarks/bench_scrawl.py --json           # machine-readable output
    python benchmarks/bench_scrawl.py --quick           # reduced iterations

M. L. McKnight · ML Innovations LLC · Pheba, Mississippi · 2026
"""

import sys, os, time, json, argparse, statistics, struct, hashlib, platform

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vm import ScrawlVM, TraceSeverity
from src.synapse import Instruction, SynapseEncoder, SynapseDecoder
from src.rosetta import decompile, compile_program, expand_macro_full
from src.registers import Tensor
from src.identity import (
    IdentityBaseline, IdentityHandshake, DeltaCompressor,
    ml_identity, ml_gnomon_update, ml_algebraic_verify,
)
from src.opcodes import (
    TensorOp, AttentionOp, ExecutionOp, StateOp,
    ConsensusOp, IdentityOp, ComposeMode,
)


# ─── Utilities ────────────────────────────────────────────────────────

def median_of(fn, trials=10, warmup=2):
    """Run fn() `warmup` times, then `trials` times, return (median_sec, all_times)."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        result = fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return statistics.median(times), times, result


def fmt_rate(count, seconds, unit="op"):
    """Format a rate as human-readable string."""
    if seconds == 0:
        return f"∞ {unit}s/sec"
    rate = count / seconds
    if rate >= 1_000_000:
        return f"{rate/1_000_000:.2f}M {unit}s/sec"
    elif rate >= 1_000:
        return f"{rate/1_000:.2f}K {unit}s/sec"
    else:
        return f"{rate:.2f} {unit}s/sec"


def fmt_time(seconds):
    """Format time as human-readable string."""
    if seconds < 0.000_001:
        return f"{seconds*1_000_000_000:.1f}ns"
    elif seconds < 0.001:
        return f"{seconds*1_000_000:.1f}µs"
    elif seconds < 1.0:
        return f"{seconds*1_000:.2f}ms"
    else:
        return f"{seconds:.3f}s"


def fmt_bytes(b):
    """Format byte count."""
    if b >= 1_048_576:
        return f"{b/1_048_576:.1f}MB"
    elif b >= 1024:
        return f"{b/1024:.1f}KB"
    else:
        return f"{b}B"


def separator(title):
    width = 64
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


# ─── Benchmark 1: VM Execution Throughput ─────────────────────────────

def bench_vm_throughput(iterations, trials):
    """Measure raw VM instruction throughput across domains."""
    separator("1. VM Execution Throughput")

    # Build a program that exercises multiple domains without halting early
    # Each iteration runs this block
    block = [
        Instruction(IdentityOp.I_DERIVE, [0, 0xCAFE, 8]),
        Instruction(IdentityOp.I_VERIFY, [0, 0, 1]),
        Instruction(IdentityOp.I_FINGERPRINT, [2, 0]),
        Instruction(ExecutionOp.X_NOP),
        Instruction(ExecutionOp.X_NOP),
        Instruction(ExecutionOp.X_NOP),
        Instruction(ExecutionOp.X_NOP),
        Instruction(ExecutionOp.X_NOP),
    ]

    # Repeat block N times, then halt
    program = block * iterations + [Instruction(ExecutionOp.X_HALT)]
    total_instructions = len(block) * iterations + 1

    def run():
        vm = ScrawlVM()
        return vm.execute(program)

    med, times, result = median_of(run, trials=trials)
    rate = total_instructions / med

    print(f"  Program size:      {total_instructions} instructions")
    print(f"  Median time:       {fmt_time(med)}")
    print(f"  Throughput:        {fmt_rate(total_instructions, med, 'instr')}")
    print(f"  Per-instruction:   {fmt_time(med / total_instructions)}")
    print(f"  Min/Max:           {fmt_time(min(times))} / {fmt_time(max(times))}")

    return {
        "name": "vm_throughput",
        "instructions": total_instructions,
        "median_sec": med,
        "rate_per_sec": rate,
        "min_sec": min(times),
        "max_sec": max(times),
    }


# ─── Benchmark 2: SYNAPSE Encode/Decode ──────────────────────────────

def bench_synapse(iterations, trials):
    """Measure SYNAPSE frame encoding and decoding throughput."""
    separator("2. SYNAPSE Encode/Decode Throughput")

    # Representative program with mixed instruction types
    program = [
        Instruction(IdentityOp.I_DERIVE, [0, 0xBEEF, 16]),
        Instruction(IdentityOp.I_VERIFY, [0, 0, 1]),
        Instruction(IdentityOp.I_FINGERPRINT, [2, 0]),
        Instruction(ConsensusOp.C_PROPOSE, [1, 0, [0, 1, 2]]),
        Instruction(ConsensusOp.C_VOTE, [1, 0, 0]),
        Instruction(ConsensusOp.C_COMMIT, [1, 1]),
        Instruction(TensorOp.T_FILL, [0, 1.0]),
        Instruction(AttentionOp.A_ROUTE, [0, 1, 2, 3]),
        Instruction(ExecutionOp.X_YIELD, [1]),
        Instruction(ExecutionOp.X_HALT),
    ]

    encoder = SynapseEncoder()
    decoder = SynapseDecoder()

    # Pre-encode once to get frame size
    sample_frame = encoder.encode_frame(program)
    frame_size = len(sample_frame)

    # Encode benchmark
    def encode_batch():
        enc = SynapseEncoder()
        for _ in range(iterations):
            enc.encode_frame(program)

    med_enc, times_enc, _ = median_of(encode_batch, trials=trials)
    total_bytes_enc = frame_size * iterations

    # Decode benchmark
    frames = [encoder.encode_frame(program) for _ in range(iterations)]

    def decode_batch():
        dec = SynapseDecoder()
        for f in frames:
            dec.decode_frame(f)

    med_dec, times_dec, _ = median_of(decode_batch, trials=trials)

    print(f"  Frame size:        {frame_size} bytes ({len(program)} instructions)")
    print(f"  Iterations:        {iterations}")
    print()
    print(f"  ENCODE:")
    print(f"    Median time:     {fmt_time(med_enc)}")
    print(f"    Throughput:      {fmt_rate(iterations, med_enc, 'frame')}")
    print(f"    Bandwidth:       {fmt_rate(total_bytes_enc, med_enc, 'byte')}/sec")
    print()
    print(f"  DECODE:")
    print(f"    Median time:     {fmt_time(med_dec)}")
    print(f"    Throughput:      {fmt_rate(iterations, med_dec, 'frame')}")
    print(f"    Bandwidth:       {fmt_rate(total_bytes_enc, med_dec, 'byte')}/sec")

    return {
        "name": "synapse",
        "frame_bytes": frame_size,
        "iterations": iterations,
        "encode_median_sec": med_enc,
        "encode_rate": iterations / med_enc if med_enc > 0 else 0,
        "decode_median_sec": med_dec,
        "decode_rate": iterations / med_dec if med_dec > 0 else 0,
    }


# ─── Benchmark 3: Delta Compression ──────────────────────────────────

def bench_delta_compression(trials):
    """Measure compression ratios and throughput for delta encoding."""
    separator("3. Delta Compression Ratios & Throughput")

    seed, depth = 0xBEEF, 16
    baseline = IdentityBaseline(seed=seed, depth=depth)

    # Test payloads of varying structure
    payloads = {
        "agent_state_small": b"pos=(100,200) hp=95 ammo=30",
        "agent_state_medium": (
            b"agent_id=0x00FF pos=(1024,2048,512) vel=(10,-5,2) "
            b"hp=87 shield=45 ammo=120 target=0x00A1 mode=PATROL "
            b"timestamp=1707000000 seq=42"
        ),
        "agent_state_large": (
            b"{"
            b'"id":"agent_00FF","position":{"x":1024,"y":2048,"z":512},'
            b'"velocity":{"x":10,"y":-5,"z":2},'
            b'"health":87,"shield":45,"ammo":120,'
            b'"target":"agent_00A1","mode":"PATROL",'
            b'"inventory":["rifle","medkit","grenade","scope"],'
            b'"waypoints":[[100,200],[300,400],[500,600]],'
            b'"timestamp":1707000000,"sequence":42'
            b"}"
        ),
        "repeated_updates": b"agent_state: position=(100, 200), health=95, ammo=30",
        "binary_payload": bytes(range(256)) * 2,
        "zero_payload": b"\x00" * 128,
    }

    # JSON baseline sizes for comparison
    json_equivalents = {
        "agent_state_small": '{"pos":[100,200],"hp":95,"ammo":30}',
        "agent_state_medium": (
            '{"agent_id":255,"pos":[1024,2048,512],"vel":[10,-5,2],'
            '"hp":87,"shield":45,"ammo":120,"target":161,"mode":"PATROL",'
            '"ts":1707000000,"seq":42}'
        ),
        "agent_state_large": (
            '{"id":"agent_00FF","position":{"x":1024,"y":2048,"z":512},'
            '"velocity":{"x":10,"y":-5,"z":2},'
            '"health":87,"shield":45,"ammo":120,'
            '"target":"agent_00A1","mode":"PATROL",'
            '"inventory":["rifle","medkit","grenade","scope"],'
            '"waypoints":[[100,200],[300,400],[500,600]],'
            '"timestamp":1707000000,"sequence":42}'
        ),
    }

    results_table = []

    for name, payload in payloads.items():
        comp = DeltaCompressor(IdentityBaseline(seed=seed, depth=depth))
        decomp = DeltaCompressor(IdentityBaseline(seed=seed, depth=depth))

        # Compress
        def compress_once():
            c = DeltaCompressor(IdentityBaseline(seed=seed, depth=depth))
            return c.compress(payload)

        med_c, _, compressed = median_of(compress_once, trials=trials)
        compressed = DeltaCompressor(IdentityBaseline(seed=seed, depth=depth)).compress(payload)

        raw_size = len(payload)
        comp_size = len(compressed)
        ratio = (1.0 - comp_size / raw_size) * 100 if raw_size > 0 else 0

        json_size = len(json_equivalents.get(name, "").encode()) if name in json_equivalents else None
        vs_json = ""
        if json_size:
            json_savings = (1.0 - comp_size / json_size) * 100
            vs_json = f" | vs JSON {json_size}B: {json_savings:+.1f}%"

        print(f"  {name}:")
        print(f"    Raw: {raw_size}B → Compressed: {comp_size}B ({ratio:+.1f}%){vs_json}")

        entry = {
            "payload": name,
            "raw_bytes": raw_size,
            "compressed_bytes": comp_size,
            "ratio_pct": round(ratio, 1),
        }
        if json_size:
            entry["json_bytes"] = json_size
            entry["vs_json_pct"] = round((1.0 - comp_size / json_size) * 100, 1)
        results_table.append(entry)

    # Throughput: repeated compression of medium payload
    print()
    payload = payloads["agent_state_medium"]
    iterations = 1000

    def compress_batch():
        c = DeltaCompressor(IdentityBaseline(seed=seed, depth=depth))
        for _ in range(iterations):
            c.compress(payload)

    med_thr, _, _ = median_of(compress_batch, trials=trials)
    print(f"  Compress throughput ({iterations}x medium):")
    print(f"    Median:          {fmt_time(med_thr)}")
    print(f"    Rate:            {fmt_rate(iterations, med_thr, 'msg')}")

    # Identical-message compression (best case)
    print()
    comp_identical = DeltaCompressor(IdentityBaseline(seed=seed, depth=depth))
    first = comp_identical.compress(payload)
    second = comp_identical.compress(payload)
    print(f"  Identical message (2nd send):")
    print(f"    First:  {len(payload)}B → {len(first)}B")
    print(f"    Second: {len(payload)}B → {len(second)}B (delta against previous)")

    return {"name": "delta_compression", "payloads": results_table}


# ─── Benchmark 4: ROSETTA Compile/Decompile ──────────────────────────

def bench_rosetta(iterations, trials):
    """Measure ROSETTA transpiler throughput."""
    separator("4. ROSETTA Compile/Decompile Roundtrip")

    source = """
CR0 = identity.derive(seed=0xF00D, depth=8)
R1 = identity.verify(CR0, R0)
R2 = identity.fingerprint(CR0)
TR3 = attention.self(TR0)
halt
"""

    # Compile benchmark
    def compile_batch():
        for _ in range(iterations):
            compile_program(source, strict=True)

    med_comp, _, compiled = median_of(compile_batch, trials=trials)
    compiled = compile_program(source, strict=True)
    print(f"  Source lines:      {len([l for l in source.strip().split(chr(10)) if l.strip()])}")
    print(f"  Compiled to:       {len(compiled)} instructions")
    print()
    print(f"  COMPILE ({iterations}x):")
    print(f"    Median:          {fmt_time(med_comp)}")
    print(f"    Rate:            {fmt_rate(iterations, med_comp, 'program')}")

    # Decompile benchmark
    def decompile_batch():
        for _ in range(iterations):
            decompile(compiled, include_hex=True)

    med_decomp, _, _ = median_of(decompile_batch, trials=trials)
    print()
    print(f"  DECOMPILE ({iterations}x):")
    print(f"    Median:          {fmt_time(med_decomp)}")
    print(f"    Rate:            {fmt_rate(iterations, med_decomp, 'program')}")

    # Macro expansion benchmark
    def expand_batch():
        for _ in range(iterations):
            expand_macro_full("fused_attention(TR0, TR1, TR2, TR3)")

    med_macro, _, _ = median_of(expand_batch, trials=trials)
    print()
    print(f"  MACRO EXPAND ({iterations}x fused_attention):")
    print(f"    Median:          {fmt_time(med_macro)}")
    print(f"    Rate:            {fmt_rate(iterations, med_macro, 'expansion')}")

    # Full roundtrip: source → compile → encode → decode → decompile
    encoder = SynapseEncoder()
    decoder = SynapseDecoder()

    def full_roundtrip():
        prog = compile_program(source, strict=True)
        frame = encoder.encode_frame(prog)
        decoded, _ = decoder.decode_frame(frame)
        readable = decompile(decoded, include_hex=False)
        return readable

    med_rt, _, _ = median_of(full_roundtrip, trials=trials)
    print()
    print(f"  FULL ROUNDTRIP (compile→encode→decode→decompile):")
    print(f"    Median:          {fmt_time(med_rt)}")

    return {
        "name": "rosetta",
        "compile_median_sec": med_comp,
        "compile_rate": iterations / med_comp if med_comp > 0 else 0,
        "decompile_median_sec": med_decomp,
        "decompile_rate": iterations / med_decomp if med_decomp > 0 else 0,
        "roundtrip_median_sec": med_rt,
    }


# ─── Benchmark 5: Identity Chain Derivation ──────────────────────────

def bench_identity_chains(trials):
    """Measure identity chain derivation at varying depths."""
    separator("5. Identity Chain Derivation")

    depths = [8, 16, 32, 64, 128, 256]
    results = []

    for depth in depths:
        def derive():
            return IdentityBaseline(seed=0xDEAD, depth=depth)

        med, _, baseline = median_of(derive, trials=trials)
        chain_len = len(baseline.chain)
        print(f"  depth={depth:4d}: {fmt_time(med):>10s}  "
              f"chain_len={chain_len:4d}  "
              f"all_int={all(isinstance(v, int) for v in baseline.chain)}")

        results.append({
            "depth": depth,
            "median_sec": med,
            "chain_length": chain_len,
        })

    # Handshake benchmark (initiate + respond + verify)
    print()

    def handshake():
        b_a, fp_a = IdentityHandshake.initiate(0xBEEF, 32)
        b_b, match = IdentityHandshake.respond(0xBEEF, 32, fp_a)
        key = IdentityHandshake.derive_shared_key(b_a, agent_a_id=0, agent_b_id=1)
        return match, key

    med_hs, _, (match, key) = median_of(handshake, trials=trials)
    print(f"  Full handshake (depth=32): {fmt_time(med_hs)}  match={match}")

    # Gnomon throughput: how fast can we advance squares
    iterations = 10_000

    def gnomon_batch():
        a_sq = 0
        for a in range(iterations):
            a_sq = ml_gnomon_update(a_sq, a)
        return a_sq

    med_gn, _, _ = median_of(gnomon_batch, trials=trials)
    print(f"  Gnomon updates ({iterations}x): {fmt_time(med_gn)}  "
          f"{fmt_rate(iterations, med_gn, 'update')}")

    # Algebraic verification throughput
    def verify_batch():
        for a in range(iterations):
            b = a + 1
            ml_algebraic_verify(a, a * a, b, b * b)

    med_av, _, _ = median_of(verify_batch, trials=trials)
    print(f"  Algebraic verify ({iterations}x): {fmt_time(med_av)}  "
          f"{fmt_rate(iterations, med_av, 'verify')}")

    return {"name": "identity_chains", "depths": results}


# ─── Benchmark 6: Consensus Round Latency ────────────────────────────

def bench_consensus(trials):
    """Measure consensus round latency: propose → vote → commit."""
    separator("6. Consensus Round Latency")

    # 2-agent consensus
    def consensus_2():
        vm = ScrawlVM()
        vm.agent_id = 0
        program = [
            Instruction(ConsensusOp.C_PROPOSE, [1, 0, [0, 1]]),
            Instruction(ConsensusOp.C_QUORUM, [1, 0.5]),
            Instruction(ConsensusOp.C_VOTE, [1, 0, 0]),
            Instruction(ConsensusOp.C_VOTE, [1, 0, 0]),
            Instruction(ConsensusOp.C_COMMIT, [1, 1]),
            Instruction(ExecutionOp.X_HALT),
        ]
        return vm.execute(program)

    med_2, _, _ = median_of(consensus_2, trials=trials)
    print(f"  2-agent consensus: {fmt_time(med_2)}")

    # 5-agent consensus
    def consensus_5():
        vm = ScrawlVM()
        vm.agent_id = 0
        agents = list(range(5))
        program = [
            Instruction(ConsensusOp.C_PROPOSE, [1, 0, agents]),
            Instruction(ConsensusOp.C_QUORUM, [1, 0.6]),
        ]
        for agent in agents:
            program.append(Instruction(ConsensusOp.C_VOTE, [1, agent, 0]))
        program.append(Instruction(ConsensusOp.C_COMMIT, [1, 1]))
        program.append(Instruction(ExecutionOp.X_HALT))
        return vm.execute(program)

    med_5, _, _ = median_of(consensus_5, trials=trials)
    print(f"  5-agent consensus: {fmt_time(med_5)}")

    # 10-agent consensus
    def consensus_10():
        vm = ScrawlVM()
        vm.agent_id = 0
        agents = list(range(10))
        program = [
            Instruction(ConsensusOp.C_PROPOSE, [1, 0, agents]),
            Instruction(ConsensusOp.C_QUORUM, [1, 0.5]),
        ]
        for agent in agents:
            program.append(Instruction(ConsensusOp.C_VOTE, [1, agent, 0]))
        program.append(Instruction(ConsensusOp.C_COMMIT, [1, 1]))
        program.append(Instruction(ExecutionOp.X_HALT))
        return vm.execute(program)

    med_10, _, _ = median_of(consensus_10, trials=trials)
    print(f"  10-agent consensus: {fmt_time(med_10)}")

    # Consensus with trace hooks (overhead measurement)
    def consensus_traced():
        vm = ScrawlVM()
        vm.agent_id = 0
        events = []
        vm.add_trace_hook(lambda e: events.append(e))
        program = [
            Instruction(ConsensusOp.C_PROPOSE, [1, 0, [0, 1]]),
            Instruction(ConsensusOp.C_QUORUM, [1, 0.5]),
            Instruction(ConsensusOp.C_VOTE, [1, 0, 0]),
            Instruction(ConsensusOp.C_VOTE, [1, 0, 0]),
            Instruction(ConsensusOp.C_COMMIT, [1, 1]),
            Instruction(ExecutionOp.X_HALT),
        ]
        return vm.execute(program)

    med_traced, _, _ = median_of(consensus_traced, trials=trials)
    trace_overhead = ((med_traced / med_2) - 1.0) * 100 if med_2 > 0 else 0
    print(f"  2-agent + trace hooks: {fmt_time(med_traced)} ({trace_overhead:+.1f}% overhead)")

    return {
        "name": "consensus",
        "2_agent_sec": med_2,
        "5_agent_sec": med_5,
        "10_agent_sec": med_10,
        "traced_sec": med_traced,
        "trace_overhead_pct": round(trace_overhead, 1),
    }


# ─── Benchmark 7: Tensor Operations ──────────────────────────────────

def bench_tensor_ops(iterations, trials):
    """Measure tensor operation throughput."""
    separator("7. Tensor Operations")

    # Small tensors (typical agent state)
    sizes = [(4,), (16,), (64,), (2, 3), (4, 4), (8, 8)]

    for shape in sizes:
        n = 1
        for s in shape:
            n *= s
        data_a = [float(i) for i in range(n)]
        data_b = [float(i + 1) for i in range(n)]

        def inplace_chain():
            for _ in range(iterations):
                t = Tensor(list(data_a), shape)
                bias = Tensor(list(data_b), shape)
                t.add_inplace(bias).scale_inplace(0.5)

        med, _, _ = median_of(inplace_chain, trials=trials)
        label = "x".join(str(s) for s in shape)
        print(f"  Tensor({label:>5s}) add+scale in-place ({iterations}x): "
              f"{fmt_time(med)}  {fmt_rate(iterations, med, 'op')}")

    # VM-level attention routing
    print()
    Q = Tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], (2, 3))
    K = Tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0], (2, 3))
    V = Tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], (2, 3))

    def attention_route():
        vm = ScrawlVM()
        vm.registers.set_treg(0, Q)
        vm.registers.set_treg(1, K)
        vm.registers.set_treg(2, V)
        return vm.execute([
            Instruction(AttentionOp.A_ROUTE, [0, 1, 2, 3]),
            Instruction(ExecutionOp.X_HALT),
        ])

    med_attn, _, _ = median_of(attention_route, trials=trials)
    print(f"  A_ROUTE (2x3 QKV):     {fmt_time(med_attn)}")

    return {"name": "tensor_ops", "attention_route_sec": med_attn}


# ─── Benchmark 8: Wire Format Size Comparison ────────────────────────

def bench_wire_size():
    """Compare SCRAWL wire format sizes against JSON equivalents."""
    separator("8. Wire Format Size Comparison (SCRAWL vs JSON)")

    comparisons = [
        {
            "name": "Identity derive + verify",
            "scrawl": [
                Instruction(IdentityOp.I_DERIVE, [0, 0xCAFE, 16]),
                Instruction(IdentityOp.I_VERIFY, [0, 0, 1]),
                Instruction(ExecutionOp.X_HALT),
            ],
            "json": json.dumps({
                "method": "identity.derive",
                "params": {"register": 0, "seed": 0xCAFE, "depth": 16},
                "id": 1
            }) + "\n" + json.dumps({
                "method": "identity.verify",
                "params": {"chain_reg": 0, "source_reg": 0, "dest_reg": 1},
                "id": 2
            }),
        },
        {
            "name": "Attention route (Q,K,V→out)",
            "scrawl": [
                Instruction(AttentionOp.A_ROUTE, [0, 1, 2, 3]),
                Instruction(ExecutionOp.X_HALT),
            ],
            "json": json.dumps({
                "method": "attention.route",
                "params": {
                    "query_reg": 0, "key_reg": 1,
                    "value_reg": 2, "output_reg": 3
                },
                "id": 1
            }),
        },
        {
            "name": "Consensus round (propose+vote+commit)",
            "scrawl": [
                Instruction(ConsensusOp.C_PROPOSE, [1, 0, [0, 1, 2]]),
                Instruction(ConsensusOp.C_QUORUM, [1, 0.5]),
                Instruction(ConsensusOp.C_VOTE, [1, 0, 0]),
                Instruction(ConsensusOp.C_VOTE, [1, 1, 0]),
                Instruction(ConsensusOp.C_VOTE, [1, 2, 0]),
                Instruction(ConsensusOp.C_COMMIT, [1, 1]),
                Instruction(ExecutionOp.X_HALT),
            ],
            "json": json.dumps([
                {"method": "consensus.propose", "params": {"id": 1, "data": 0, "agents": [0, 1, 2]}},
                {"method": "consensus.quorum", "params": {"id": 1, "threshold": 0.5}},
                {"method": "consensus.vote", "params": {"id": 1, "agent": 0, "vote": "approve"}},
                {"method": "consensus.vote", "params": {"id": 1, "agent": 1, "vote": "approve"}},
                {"method": "consensus.vote", "params": {"id": 1, "agent": 2, "vote": "approve"}},
                {"method": "consensus.commit", "params": {"id": 1, "dest_reg": 1}},
            ]),
        },
        {
            "name": "10-instruction mixed program",
            "scrawl": [
                Instruction(IdentityOp.I_DERIVE, [0, 0xF00D, 8]),
                Instruction(IdentityOp.I_VERIFY, [0, 0, 1]),
                Instruction(IdentityOp.I_FINGERPRINT, [2, 0]),
                Instruction(TensorOp.T_FILL, [0, 1.0]),
                Instruction(TensorOp.T_FILL, [1, 2.0]),
                Instruction(TensorOp.T_COMPOSE, [2, 0, 1, ComposeMode.DOT]),
                Instruction(AttentionOp.A_SELF, [0, 3]),
                Instruction(ConsensusOp.C_PROPOSE, [1, 0, [0, 1]]),
                Instruction(ConsensusOp.C_VOTE, [1, 0, 0]),
                Instruction(ExecutionOp.X_HALT),
            ],
            "json": json.dumps([
                {"method": "identity.derive", "params": {"reg": 0, "seed": "0xF00D", "depth": 8}},
                {"method": "identity.verify", "params": {"chain": 0, "src": 0, "dst": 1}},
                {"method": "identity.fingerprint", "params": {"dst": 2, "chain": 0}},
                {"method": "tensor.fill", "params": {"reg": 0, "value": 1.0}},
                {"method": "tensor.fill", "params": {"reg": 1, "value": 2.0}},
                {"method": "tensor.compose", "params": {"dst": 2, "a": 0, "b": 1, "mode": "dot"}},
                {"method": "attention.self", "params": {"src": 0, "dst": 3}},
                {"method": "consensus.propose", "params": {"id": 1, "data": 0, "agents": [0, 1]}},
                {"method": "consensus.vote", "params": {"id": 1, "agent": 0, "vote": "approve"}},
            ]),
        },
    ]

    encoder = SynapseEncoder()
    results = []

    for comp in comparisons:
        frame = encoder.encode_frame(comp["scrawl"])
        scrawl_bytes = len(frame)
        json_bytes = len(comp["json"].encode("utf-8"))
        reduction = (1.0 - scrawl_bytes / json_bytes) * 100
        ratio = json_bytes / scrawl_bytes if scrawl_bytes > 0 else 0

        print(f"  {comp['name']}:")
        print(f"    SCRAWL:  {scrawl_bytes:4d} bytes")
        print(f"    JSON:    {json_bytes:4d} bytes")
        print(f"    Savings: {reduction:.1f}% ({ratio:.1f}x smaller)")
        print()

        results.append({
            "name": comp["name"],
            "scrawl_bytes": scrawl_bytes,
            "json_bytes": json_bytes,
            "reduction_pct": round(reduction, 1),
            "ratio": round(ratio, 1),
        })

    return {"name": "wire_size", "comparisons": results}


# ─── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SCRAWL v1.1 Performance Benchmarks")
    parser.add_argument("--trials", type=int, default=10, help="Measurement trials per benchmark (default: 10)")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup runs before measuring (default: 2)")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    parser.add_argument("--quick", action="store_true", help="Reduced iterations for fast results")
    args = parser.parse_args()

    if args.quick:
        vm_iter = 100
        syn_iter = 100
        ros_iter = 100
        tensor_iter = 100
        args.trials = 5
    else:
        vm_iter = 1000
        syn_iter = 1000
        ros_iter = 500
        tensor_iter = 500

    print("=" * 64)
    print("  SCRAWL v1.1 — Performance Benchmarks")
    print("  ML Innovations LLC · M. L. McKnight · Pheba, Mississippi")
    print("=" * 64)
    print()
    print(f"  Platform:  {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"  Python:    {platform.python_version()}")
    print(f"  Trials:    {args.trials} (median of N)")
    print(f"  Mode:      {'quick' if args.quick else 'standard'}")

    results = []
    t_start = time.perf_counter()

    results.append(bench_vm_throughput(vm_iter, args.trials))
    results.append(bench_synapse(syn_iter, args.trials))
    results.append(bench_delta_compression(args.trials))
    results.append(bench_rosetta(ros_iter, args.trials))
    results.append(bench_identity_chains(args.trials))
    results.append(bench_consensus(args.trials))
    results.append(bench_tensor_ops(tensor_iter, args.trials))
    results.append(bench_wire_size())

    t_total = time.perf_counter() - t_start

    separator("Summary")
    print(f"  Total benchmark time: {fmt_time(t_total)}")
    print(f"  All benchmarks completed successfully.")
    print()
    print("  Key takeaways:")
    print(f"    • VM throughput:         {fmt_rate(results[0]['instructions'], results[0]['median_sec'], 'instr')}")
    print(f"    • SYNAPSE encode:        {fmt_rate(results[1]['iterations'], results[1]['encode_median_sec'], 'frame')}")
    print(f"    • 2-agent consensus:     {fmt_time(results[5]['2_agent_sec'])}")
    print(f"    • Trace hook overhead:   {results[5]['trace_overhead_pct']:+.1f}%")

    wire = results[7]["comparisons"]
    if wire:
        avg_reduction = sum(w["reduction_pct"] for w in wire) / len(wire)
        print(f"    • Avg wire size savings: {avg_reduction:.1f}% vs JSON")

    print()
    print("=" * 64)
    print("  Zero external dependencies. Pure Python 3.10+.")
    print("  Run: python benchmarks/bench_scrawl.py --quick")
    print("=" * 64)

    if args.json:
        output = {
            "version": "1.1",
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "python": platform.python_version(),
            },
            "config": {
                "trials": args.trials,
                "quick": args.quick,
            },
            "total_time_sec": t_total,
            "benchmarks": results,
        }
        json_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "results.json"
        )
        with open(json_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  JSON results written to: {json_path}")


if __name__ == "__main__":
    main()
