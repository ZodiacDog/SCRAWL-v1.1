//! SCRAWL v1.1 Identity Module
//!
//! ML Identity equation: a + a² + b = b² where b = a + 1
//!
//! All chain derivation is INTEGER ARITHMETIC ONLY.
//! Cross-platform determinism guaranteed by Theorem 1 + Section 2.3.
//!
//! v1.1 Changes:
//!   - Integer-only chain derivation (Theorem 1 + §2.3)
//!   - Gnomon incremental squares: (a+1)² = a² + a + (a+1) (Theorem 4+5)
//!   - Zero-cost algebraic verification (§7.2 + §7.5)
//!   - Generalized identity: a² + a(2k-1) + k(k-1) + b = b² (Theorem 2)
//!   - RLE decode hardening (defensive guard)
//!
//! ML Innovations LLC · M. L. McKnight · Pheba, Mississippi · 2026

use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════
// ML IDENTITY CORE (Integer Arithmetic — Theorem 1)
// ═══════════════════════════════════════════════════════════════

/// Compute ML Identity: a + a² + b = b² where b = a + 1.
/// Returns (b, lhs) where lhs == b² for all a.
#[inline]
pub fn ml_identity(a: i64) -> (i64, i64) {
    let b = a + 1;
    let lhs = a + a * a + b;
    (b, lhs)
}

/// Verify the ML Identity holds for a given value of a.
#[inline]
pub fn ml_identity_verify(a: i64) -> bool {
    let b = a + 1;
    a + a * a + b == b * b
}

/// Generalized ML Identity (Theorem 2):
/// a² + a(2k-1) + k(k-1) + b = b² where b = a + k
#[inline]
pub fn ml_identity_generalized(a: i64, k: i64) -> bool {
    let b = a + k;
    let lhs = a * a + a * (2 * k - 1) + k * (k - 1) + b;
    lhs == b * b
}

/// Gnomon incremental square update (Theorem 4 + 5):
/// (a+1)² = a² + a + (a+1)
/// Replaces multiplication with two additions.
#[inline]
pub fn ml_gnomon_update(a_squared: u64, a: u64) -> u64 {
    a_squared + a + (a + 1)
}

/// ML Consecutive Sum Theorem (Theorem 4):
/// b² - a² = a + b = 2a + 1 where b = a + 1
#[inline]
pub fn ml_consecutive_sum(a: u64) -> u64 {
    2 * a + 1
}

/// ML-Triangular Bridge (Theorem 3): T(n) = n(n+1)/2
#[inline]
pub fn ml_triangular(n: u64) -> u64 {
    n * (n + 1) / 2
}

/// Triangular-Square Duality (Theorem 6): T(n-1) + T(n) = n²
/// Returns (T(n-1), T(n)).
#[inline]
pub fn ml_triangular_square_duality(n: u64) -> (u64, u64) {
    let t_prev = if n > 0 { (n - 1) * n / 2 } else { 0 };
    let t_curr = n * (n + 1) / 2;
    (t_prev, t_curr)
}

/// Zero-cost algebraic error detection (§7.5).
/// Verify ML Identity holds given independently computed a, a², b, b².
/// Any single-value corruption detected. Cost: one add + one comparison.
#[inline]
pub fn ml_algebraic_verify(a: u64, a_sq: u64, b: u64, b_sq: u64) -> bool {
    a + a_sq + b == b_sq
}

/// Chain ML Identity derivations to generate a deterministic integer sequence.
///
/// v1.1: Integer arithmetic only. Uses gnomon incremental updates (Theorem 4).
/// Cross-platform deterministic: same seed → same chain on any architecture.
pub fn ml_identity_chain(seed: u64, depth: usize) -> Vec<u64> {
    const MOD: u128 = 1u128 << 64;

    let mut chain = Vec::with_capacity(depth + 1);
    chain.push(seed);

    let mut a = (seed & 0xFFFFFFFF) as u64;
    let mut a_sq = a as u128 * a as u128;

    for _ in 0..depth {
        let b = a as u128 + 1;
        let b_sq = a_sq + a as u128 + b; // gnomon update

        // Zero-cost identity check (§7.5)
        debug_assert_eq!(
            a as u128 + a_sq + b, b_sq,
            "ML Identity violated at a={}", a
        );

        // Deterministic mixing — pure integer arithmetic
        let mixed = ((b_sq ^ (a as u128).wrapping_mul(2654435761))
            .wrapping_add(b.wrapping_mul(1103515245))
            .wrapping_add(12345))
            % MOD;

        let mixed_u64 = (mixed & 0xFFFFFFFF) as u64;
        chain.push(mixed_u64);

        // Advance
        a = ((mixed >> 16) & 0xFFFF) as u64 | (((mixed & 0xFFFF) << 16) as u64);
        a &= 0xFFFFFFFF;
        a_sq = a as u128 * a as u128;
    }

    chain
}

// ═══════════════════════════════════════════════════════════════
// SHA-256 (minimal, no-dependency implementation)
// ═══════════════════════════════════════════════════════════════

/// Minimal SHA-256 implementation — zero external dependencies.
/// Used only for fingerprints and key derivation.
pub fn sha256(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
        0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
        0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
        0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];

    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];

    // Pad message
    let bit_len = (data.len() as u64) * 8;
    let mut padded = data.to_vec();
    padded.push(0x80);
    while (padded.len() % 64) != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    // Process 512-bit blocks
    for chunk in padded.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                chunk[i * 4], chunk[i * 4 + 1],
                chunk[i * 4 + 2], chunk[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7)
                ^ w[i - 15].rotate_right(18)
                ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17)
                ^ w[i - 2].rotate_right(19)
                ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut result = [0u8; 32];
    for (i, &val) in h.iter().enumerate() {
        result[i * 4..i * 4 + 4].copy_from_slice(&val.to_be_bytes());
    }
    result
}

// ═══════════════════════════════════════════════════════════════
// IDENTITY BASELINE
// ═══════════════════════════════════════════════════════════════

/// Shared structural baseline derived from ML Identity.
/// v1.1: Chain derivation is fully integer-based.
#[derive(Debug, Clone)]
pub struct IdentityBaseline {
    pub seed: u64,
    pub depth: usize,
    pub chain: Vec<u64>,
    pub fingerprint: [u8; 8],
    pattern_cache: HashMap<u32, [u8; 32]>,
}

impl IdentityBaseline {
    pub fn new(seed: u64, depth: usize) -> Self {
        let chain = ml_identity_chain(seed, depth);
        let fingerprint = Self::compute_fingerprint(seed, depth, &chain);
        Self {
            seed,
            depth,
            chain,
            fingerprint,
            pattern_cache: HashMap::new(),
        }
    }

    fn compute_fingerprint(seed: u64, depth: usize, chain: &[u64]) -> [u8; 8] {
        let mut data = Vec::new();
        data.extend_from_slice(&seed.to_le_bytes());
        data.extend_from_slice(&(depth as u32).to_le_bytes());
        for &val in chain {
            data.extend_from_slice(&(val & 0xFFFFFFFFFFFFFFFF).to_le_bytes());
        }
        let hash = sha256(&data);
        let mut fp = [0u8; 8];
        fp.copy_from_slice(&hash[..8]);
        fp
    }

    pub fn derive_pattern(&mut self, pattern_id: u32) -> [u8; 32] {
        if let Some(&cached) = self.pattern_cache.get(&pattern_id) {
            return cached;
        }
        let mut data = Vec::new();
        data.extend_from_slice(&self.fingerprint);
        data.extend_from_slice(&pattern_id.to_le_bytes());
        let chain_idx = (pattern_id as usize) % self.chain.len();
        data.extend_from_slice(&(self.chain[chain_idx] & 0xFFFFFFFFFFFFFFFF).to_le_bytes());
        let pattern = sha256(&data);
        self.pattern_cache.insert(pattern_id, pattern);
        pattern
    }

    pub fn compute_delta(&mut self, data: &[u8]) -> Vec<u8> {
        let expectation = self.generate_expectation(data.len());
        data.iter().zip(expectation.iter()).map(|(a, b)| a ^ b).collect()
    }

    pub fn apply_delta(&mut self, delta: &[u8]) -> Vec<u8> {
        let expectation = self.generate_expectation(delta.len());
        delta.iter().zip(expectation.iter()).map(|(a, b)| a ^ b).collect()
    }

    fn generate_expectation(&mut self, length: usize) -> Vec<u8> {
        let mut result = Vec::with_capacity(length);
        let mut block = 0u32;
        while result.len() < length {
            let pattern = self.derive_pattern(block);
            result.extend_from_slice(&pattern);
            block += 1;
        }
        result.truncate(length);
        result
    }

    /// Serialize baseline to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(&self.seed.to_le_bytes());
        data.extend_from_slice(&(self.depth as u32).to_le_bytes());
        data.extend_from_slice(&self.fingerprint);
        data
    }
}

// ═══════════════════════════════════════════════════════════════
// IDENTITY HANDSHAKE
// ═══════════════════════════════════════════════════════════════

pub struct IdentityHandshake;

impl IdentityHandshake {
    pub fn initiate(seed: u64, depth: usize) -> (IdentityBaseline, [u8; 8]) {
        let baseline = IdentityBaseline::new(seed, depth);
        let fp = baseline.fingerprint;
        (baseline, fp)
    }

    pub fn respond(
        seed: u64, depth: usize, received_fingerprint: &[u8; 8],
    ) -> (IdentityBaseline, bool) {
        let baseline = IdentityBaseline::new(seed, depth);
        let matches = &baseline.fingerprint == received_fingerprint;
        (baseline, matches)
    }

    pub fn derive_shared_key(
        baseline: &IdentityBaseline, agent_a_id: u32, agent_b_id: u32,
    ) -> [u8; 32] {
        let mut data = Vec::new();
        data.extend_from_slice(&baseline.fingerprint);
        let low = agent_a_id.min(agent_b_id);
        let high = agent_a_id.max(agent_b_id);
        data.extend_from_slice(&low.to_le_bytes());
        data.extend_from_slice(&high.to_le_bytes());
        for &val in &baseline.chain {
            data.extend_from_slice(&(val & 0xFFFFFFFFFFFFFFFF).to_le_bytes());
        }
        sha256(&data)
    }
}

// ═══════════════════════════════════════════════════════════════
// DELTA COMPRESSOR (with RLE hardening)
// ═══════════════════════════════════════════════════════════════

/// Stateful delta compressor with RLE.
/// v1.1: RLE decode length verification + stream corruption detection.
pub struct DeltaCompressor {
    baseline: IdentityBaseline,
    pub message_count: u64,
    last_data: Option<Vec<u8>>,
}

impl DeltaCompressor {
    pub fn new(baseline: IdentityBaseline) -> Self {
        Self {
            baseline,
            message_count: 0,
            last_data: None,
        }
    }

    pub fn compress(&mut self, data: &[u8]) -> Vec<u8> {
        let mut flags: u8 = 0;
        let mut best = data.to_vec();

        // Try baseline delta
        let baseline_delta = self.baseline.compute_delta(data);
        if baseline_delta.len() <= best.len() {
            best = baseline_delta.clone();
            flags = 0x01;
        }

        // Try previous-message delta
        if let Some(ref prev) = self.last_data {
            if prev.len() == data.len() {
                let prev_delta: Vec<u8> = data.iter()
                    .zip(prev.iter())
                    .map(|(a, b)| a ^ b)
                    .collect();
                let zero_count = prev_delta.iter().filter(|&&b| b == 0).count();
                if zero_count > prev_delta.len() / 2 {
                    let rle = Self::rle_encode(&prev_delta);
                    if rle.len() < best.len() {
                        best = rle;
                        flags = 0x02 | 0x04;
                    }
                }
            }
        }

        // Try RLE on baseline delta
        if flags == 0x01 {
            let rle = Self::rle_encode(&baseline_delta);
            if rle.len() < best.len() {
                best = rle;
                flags = 0x01 | 0x04;
            }
        }

        self.last_data = Some(data.to_vec());
        self.message_count += 1;

        let mut result = vec![flags];
        result.extend_from_slice(&best);
        result
    }

    pub fn decompress(&mut self, compressed: &[u8]) -> Result<Vec<u8>, String> {
        if compressed.is_empty() {
            return Err("Empty compressed data".to_string());
        }

        let flags = compressed[0];
        let mut payload = compressed[1..].to_vec();

        if flags & 0x04 != 0 {
            payload = Self::rle_decode(&payload)?;
        }

        let data = if flags & 0x02 != 0 {
            let prev = self.last_data.as_ref()
                .ok_or("No previous message for delta decompression")?;
            if payload.len() != prev.len() {
                return Err(format!(
                    "RLE decode length mismatch: got {}, expected {} (stream corruption)",
                    payload.len(), prev.len()
                ));
            }
            payload.iter().zip(prev.iter()).map(|(a, b)| a ^ b).collect()
        } else if flags & 0x01 != 0 {
            self.baseline.apply_delta(&payload)
        } else {
            payload
        };

        self.last_data = Some(data.clone());
        self.message_count += 1;
        Ok(data)
    }

    fn rle_encode(data: &[u8]) -> Vec<u8> {
        let mut result = Vec::new();
        let mut i = 0;
        while i < data.len() {
            if data[i] == 0 {
                let mut run = 0u8;
                while i < data.len() && data[i] == 0 && run < 255 {
                    run += 1;
                    i += 1;
                }
                result.push(0x00);
                result.push(run);
            } else {
                let start = i;
                while i < data.len() && data[i] != 0 && (i - start) < 255 {
                    i += 1;
                }
                let count = (i - start) as u8;
                result.push(0xFF);
                result.push(count);
                result.extend_from_slice(&data[start..start + count as usize]);
            }
        }
        result
    }

    fn rle_decode(data: &[u8]) -> Result<Vec<u8>, String> {
        let mut result = Vec::new();
        let mut i = 0;
        while i < data.len() {
            let marker = data[i];
            i += 1;
            if i >= data.len() {
                break;
            }
            let count = data[i] as usize;
            i += 1;
            if marker == 0x00 {
                result.extend(std::iter::repeat(0u8).take(count));
            } else if marker == 0xFF {
                if i + count > data.len() {
                    return Err(format!(
                        "RLE stream corruption: run claims {} bytes, only {} remain",
                        count, data.len() - i
                    ));
                }
                result.extend_from_slice(&data[i..i + count]);
                i += count;
            }
        }
        Ok(result)
    }
}

// ═══════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ml_identity_basic() {
        for a in [0i64, 1, 5, 10, 42, 100, 1000, -1, -10] {
            let (b, lhs) = ml_identity(a);
            assert_eq!(lhs, b * b, "Identity failed at a={}", a);
        }
    }

    #[test]
    fn test_ml_identity_verify() {
        for a in -100..200 {
            assert!(ml_identity_verify(a), "Verify failed at a={}", a);
        }
    }

    #[test]
    fn test_ml_identity_generalized() {
        for k in 1..=20 {
            for a in -50..50 {
                assert!(
                    ml_identity_generalized(a, k),
                    "Generalized identity failed at a={}, k={}", a, k
                );
            }
        }
    }

    #[test]
    fn test_gnomon_update() {
        let mut a_sq = 0u64;
        for a in 0..1000u64 {
            a_sq = ml_gnomon_update(a_sq, a);
            assert_eq!(a_sq, (a + 1) * (a + 1), "Gnomon failed at a={}", a);
        }
    }

    #[test]
    fn test_algebraic_verify() {
        for a in 0..200u64 {
            let b = a + 1;
            assert!(ml_algebraic_verify(a, a * a, b, b * b));
        }
        // Detect corruption
        assert!(!ml_algebraic_verify(5, 25, 6, 37)); // b_sq corrupted
    }

    #[test]
    fn test_chain_integer_only() {
        let chain = ml_identity_chain(0xCAFE, 16);
        assert_eq!(chain.len(), 17);
        // All values are already u64 in Rust — type system guarantees this
    }

    #[test]
    fn test_chain_determinism() {
        let c1 = ml_identity_chain(0xBEEF, 32);
        let c2 = ml_identity_chain(0xBEEF, 32);
        assert_eq!(c1, c2, "Same seed must produce identical chains");
    }

    #[test]
    fn test_chain_uniqueness() {
        let c1 = ml_identity_chain(0xBEEF, 16);
        let c2 = ml_identity_chain(0xDEAD, 16);
        assert_ne!(c1, c2, "Different seeds must produce different chains");
    }

    #[test]
    fn test_baseline_fingerprint() {
        let b1 = IdentityBaseline::new(0xBEEF, 16);
        let b2 = IdentityBaseline::new(0xBEEF, 16);
        assert_eq!(b1.fingerprint, b2.fingerprint);
    }

    #[test]
    fn test_handshake() {
        let (baseline_a, fp_a) = IdentityHandshake::initiate(0xBEEF, 16);
        let (_baseline_b, matched) = IdentityHandshake::respond(0xBEEF, 16, &fp_a);
        assert!(matched);

        let key = IdentityHandshake::derive_shared_key(&baseline_a, 0, 1);
        assert_eq!(key.len(), 32);
    }

    #[test]
    fn test_delta_compression_roundtrip() {
        let baseline = IdentityBaseline::new(0xBEEF, 16);
        let mut sender = DeltaCompressor::new(baseline.clone());
        let mut receiver = DeltaCompressor::new(baseline);

        let state = b"agent_state: position=(100, 200), health=95, ammo=30";
        let compressed = sender.compress(state);
        let recovered = receiver.decompress(&compressed).unwrap();
        assert_eq!(recovered, state);
    }

    #[test]
    fn test_delta_identical_message() {
        let baseline = IdentityBaseline::new(0xBEEF, 16);
        let mut comp = DeltaCompressor::new(baseline);

        let state = b"agent_state: position=(100, 200)";
        let first = comp.compress(state);
        let second = comp.compress(state);
        assert!(second.len() < first.len(), "Identical message should compress better");
    }

    #[test]
    fn test_consecutive_sum() {
        for a in 0..100u64 {
            let b = a + 1;
            assert_eq!(ml_consecutive_sum(a), a + b);
            assert_eq!(ml_consecutive_sum(a), b * b - a * a);
        }
    }

    #[test]
    fn test_triangular_square_duality() {
        for n in 1..100u64 {
            let (t_prev, t_curr) = ml_triangular_square_duality(n);
            assert_eq!(t_prev + t_curr, n * n, "Duality failed at n={}", n);
        }
    }
}
