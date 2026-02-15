//! SCRAWL v1.1 Benchmarks (Rust)
//!
//! Run: cargo bench
//!
//! ML Innovations LLC · M. L. McKnight · Pheba, Mississippi · 2026

use criterion::{criterion_group, criterion_main, Criterion, black_box};

use scrawl::identity::*;
use scrawl::vm::ScrawlVM;
use scrawl::synapse::*;
use scrawl::opcodes::*;
use scrawl::registers::Tensor;

fn bench_ml_identity(c: &mut Criterion) {
    c.bench_function("ml_identity (single)", |b| {
        b.iter(|| ml_identity(black_box(42)));
    });

    c.bench_function("ml_identity_verify (single)", |b| {
        b.iter(|| ml_identity_verify(black_box(42)));
    });

    c.bench_function("gnomon_update (single)", |b| {
        b.iter(|| ml_gnomon_update(black_box(1764), black_box(42)));
    });

    c.bench_function("algebraic_verify (single)", |b| {
        b.iter(|| ml_algebraic_verify(black_box(42), black_box(1764), black_box(43), black_box(1849)));
    });
}

fn bench_chain_derivation(c: &mut Criterion) {
    let mut group = c.benchmark_group("chain_derivation");
    for depth in [8, 16, 32, 64, 128] {
        group.bench_function(format!("depth_{}", depth), |b| {
            b.iter(|| ml_identity_chain(black_box(0xBEEF), black_box(depth)));
        });
    }
    group.finish();
}

fn bench_baseline(c: &mut Criterion) {
    c.bench_function("baseline_new (depth=16)", |b| {
        b.iter(|| IdentityBaseline::new(black_box(0xBEEF), black_box(16)));
    });

    c.bench_function("handshake_full (depth=32)", |b| {
        b.iter(|| {
            let (baseline_a, fp_a) = IdentityHandshake::initiate(0xBEEF, 32);
            let (_baseline_b, matched) = IdentityHandshake::respond(0xBEEF, 32, &fp_a);
            let _key = IdentityHandshake::derive_shared_key(&baseline_a, 0, 1);
            matched
        });
    });
}

fn bench_delta_compression(c: &mut Criterion) {
    let state = b"agent_state: position=(100, 200), health=95, ammo=30";

    c.bench_function("delta_compress", |b| {
        b.iter(|| {
            let baseline = IdentityBaseline::new(0xBEEF, 16);
            let mut comp = DeltaCompressor::new(baseline);
            comp.compress(black_box(state))
        });
    });

    c.bench_function("delta_roundtrip", |b| {
        b.iter(|| {
            let baseline = IdentityBaseline::new(0xBEEF, 16);
            let mut sender = DeltaCompressor::new(baseline.clone());
            let mut receiver = DeltaCompressor::new(baseline);
            let compressed = sender.compress(state);
            receiver.decompress(&compressed).unwrap()
        });
    });
}

fn bench_vm_execution(c: &mut Criterion) {
    let block = vec![
        Instruction::new(Opcode::Identity(IdentityOp::Derive),
            vec![Operand::Int(0), Operand::Int(0xCAFE), Operand::Int(8)]),
        Instruction::new(Opcode::Identity(IdentityOp::Verify),
            vec![Operand::Int(0), Operand::Int(0), Operand::Int(1)]),
        Instruction::new(Opcode::Identity(IdentityOp::Fingerprint),
            vec![Operand::Int(2), Operand::Int(0)]),
        Instruction::new(Opcode::Execution(ExecutionOp::Nop), vec![]),
        Instruction::new(Opcode::Execution(ExecutionOp::Nop), vec![]),
    ];

    let mut program: Vec<Instruction> = Vec::new();
    for _ in 0..100 {
        program.extend(block.clone());
    }
    program.push(Instruction::new(Opcode::Execution(ExecutionOp::Halt), vec![]));

    c.bench_function("vm_execute_501_instructions", |b| {
        b.iter(|| {
            let mut vm = ScrawlVM::new();
            vm.execute(black_box(&program))
        });
    });
}

fn bench_synapse(c: &mut Criterion) {
    let instructions = vec![
        Instruction::new(Opcode::Identity(IdentityOp::Derive),
            vec![Operand::Int(0), Operand::Int(0xBEEF), Operand::Int(16)]),
        Instruction::new(Opcode::Identity(IdentityOp::Verify),
            vec![Operand::Int(0), Operand::Int(0), Operand::Int(1)]),
        Instruction::new(Opcode::Consensus(ConsensusOp::Propose),
            vec![Operand::Int(1), Operand::Int(0), Operand::List(vec![0, 1, 2])]),
        Instruction::new(Opcode::Execution(ExecutionOp::Halt), vec![]),
    ];

    c.bench_function("synapse_encode_frame", |b| {
        b.iter(|| {
            let mut encoder = SynapseEncoder::new();
            encoder.encode_frame(black_box(&instructions))
        });
    });

    let mut encoder = SynapseEncoder::new();
    let frame = encoder.encode_frame(&instructions);

    c.bench_function("synapse_decode_frame", |b| {
        let decoder = SynapseDecoder::new();
        b.iter(|| decoder.decode_frame(black_box(&frame)));
    });
}

fn bench_tensor_ops(c: &mut Criterion) {
    let a = Tensor::new(vec![1.0; 64], vec![8, 8]).unwrap();
    let b = Tensor::new(vec![2.0; 64], vec![8, 8]).unwrap();

    c.bench_function("tensor_matmul_8x8", |b_| {
        b_.iter(|| a.dot(black_box(&b)));
    });

    c.bench_function("tensor_add_inplace_64", |b_| {
        b_.iter(|| {
            let mut t = Tensor::new(vec![1.0; 64], vec![64]).unwrap();
            let bias = Tensor::new(vec![0.1; 64], vec![64]).unwrap();
            t.add_inplace(&bias).scale_inplace(2.0);
        });
    });
}

criterion_group!(
    benches,
    bench_ml_identity,
    bench_chain_derivation,
    bench_baseline,
    bench_delta_compression,
    bench_vm_execution,
    bench_synapse,
    bench_tensor_ops,
);
criterion_main!(benches);
