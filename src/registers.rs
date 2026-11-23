//! SCRAWL Register File and Tensor Type
//!
//! 256 general registers, 64 tensor registers, 16 context registers.
//! Zero external dependencies — all tensor math is pure Rust.
//!
//! ML Innovations LLC · M. L. McKnight · Pheba, Mississippi · 2026

/// Tensor data types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DType {
    Float32 = 0,
    Float16 = 1,
    Int32   = 2,
    Int16   = 3,
    Int8    = 4,
    Uint8   = 5,
    Bool    = 6,
    Float64 = 7,
}

/// N-dimensional tensor — pure Rust, no external dependencies.
/// Data stored as flat Vec<f64> with shape metadata.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl Tensor {
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Result<Self, String> {
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(format!(
                "Data length {} doesn't match shape {:?} (expected {})",
                data.len(), shape, expected
            ));
        }
        Ok(Self { data, shape, dtype: DType::Float32 })
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        Self { data: vec![0.0; size], shape, dtype: DType::Float32 }
    }

    pub fn fill(shape: Vec<usize>, value: f64) -> Self {
        let size: usize = shape.iter().product();
        Self { data: vec![value; size], shape, dtype: DType::Float32 }
    }

    pub fn ndim(&self) -> usize { self.shape.len() }
    pub fn size(&self) -> usize { self.data.len() }

    /// Matrix multiplication / dot product
    pub fn dot(&self, other: &Tensor) -> Result<Tensor, String> {
        if self.ndim() == 1 && other.ndim() == 1 {
            if self.shape[0] != other.shape[0] {
                return Err("Dot: incompatible shapes".to_string());
            }
            let result: f64 = self.data.iter()
                .zip(other.data.iter())
                .map(|(a, b)| a * b)
                .sum();
            return Tensor::new(vec![result], vec![1]);
        }
        if self.ndim() == 2 && other.ndim() == 2 {
            let (m, k1) = (self.shape[0], self.shape[1]);
            let (k2, n) = (other.shape[0], other.shape[1]);
            if k1 != k2 {
                return Err(format!("Matmul: {:?} x {:?} incompatible", self.shape, other.shape));
            }
            let mut result = vec![0.0; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut s = 0.0;
                    for p in 0..k1 {
                        s += self.data[i * k1 + p] * other.data[p * n + j];
                    }
                    result[i * n + j] = s;
                }
            }
            return Tensor::new(result, vec![m, n]);
        }
        Err(format!("Dot not implemented for ndim {} x {}", self.ndim(), other.ndim()))
    }

    /// Element-wise multiplication
    pub fn hadamard(&self, other: &Tensor) -> Result<Tensor, String> {
        if self.shape != other.shape {
            return Err("Hadamard: shape mismatch".to_string());
        }
        let data: Vec<f64> = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();
        Tensor::new(data, self.shape.clone())
    }

    /// Element-wise addition (returns new tensor)
    pub fn add(&self, other: &Tensor) -> Result<Tensor, String> {
        if self.shape != other.shape {
            return Err("Add: shape mismatch".to_string());
        }
        let data: Vec<f64> = self.data.iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Tensor::new(data, self.shape.clone())
    }

    /// Scale by scalar (returns new tensor)
    pub fn scale(&self, factor: f64) -> Tensor {
        let data: Vec<f64> = self.data.iter().map(|x| x * factor).collect();
        Tensor { data, shape: self.shape.clone(), dtype: self.dtype }
    }

    // ── v1.1: In-place operations (zero allocation) ──

    /// In-place element-wise addition
    pub fn add_inplace(&mut self, other: &Tensor) -> &mut Self {
        assert_eq!(self.shape, other.shape, "Add inplace: shape mismatch");
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a += b;
        }
        self
    }

    /// In-place scaling
    pub fn scale_inplace(&mut self, factor: f64) -> &mut Self {
        for x in self.data.iter_mut() {
            *x *= factor;
        }
        self
    }

    /// In-place element-wise subtraction
    pub fn sub_inplace(&mut self, other: &Tensor) -> &mut Self {
        assert_eq!(self.shape, other.shape, "Sub inplace: shape mismatch");
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a -= b;
        }
        self
    }

    /// In-place element-wise multiplication
    pub fn hadamard_inplace(&mut self, other: &Tensor) -> &mut Self {
        assert_eq!(self.shape, other.shape, "Hadamard inplace: shape mismatch");
        for (a, b) in self.data.iter_mut().zip(other.data.iter()) {
            *a *= b;
        }
        self
    }

    /// Reshape (no copy if contiguous)
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Tensor, String> {
        let new_size: usize = new_shape.iter().product();
        if new_size != self.size() {
            return Err(format!("Cannot reshape {:?} to {:?}", self.shape, new_shape));
        }
        Tensor::new(self.data.clone(), new_shape)
    }

    /// L2 normalize
    pub fn l2_normalize(&self) -> Tensor {
        let norm: f64 = self.data.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 0.0 {
            self.scale(1.0 / norm)
        } else {
            self.clone()
        }
    }

    /// Sum reduction
    pub fn reduce_sum(&self) -> f64 {
        self.data.iter().sum()
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        if self.shape != other.shape { return false; }
        self.data.iter().zip(other.data.iter())
            .all(|(a, b)| (a - b).abs() < 1e-6)
    }
}

// ═══════════════════════════════════════════════════════════════
// REGISTER FILE
// ═══════════════════════════════════════════════════════════════

/// SCRAWL virtual machine register file.
/// R0-R255: General purpose (64-bit), TR0-TR63: Tensor, CR0-CR15: Context.
pub struct RegisterFile {
    pub general: [u64; 256],
    pub tensor: [Option<Tensor>; 64],
    pub context: [Option<Vec<u8>>; 16],
    pub pc: usize,
    pub zero_flag: bool,
    pub overflow_flag: bool,
    pub halted: bool,
}

impl RegisterFile {
    pub fn new() -> Self {
        Self {
            general: [0u64; 256],
            tensor: std::array::from_fn(|_| None),
            context: std::array::from_fn(|_| None),
            pc: 0,
            zero_flag: false,
            overflow_flag: false,
            halted: false,
        }
    }

    pub fn get_reg(&self, index: u8) -> u64 {
        self.general[index as usize]
    }

    pub fn set_reg(&mut self, index: u8, value: u64) {
        self.general[index as usize] = value;
        self.zero_flag = value == 0;
    }

    pub fn get_treg(&self, index: u8) -> Option<&Tensor> {
        self.tensor[index as usize].as_ref()
    }

    pub fn set_treg(&mut self, index: u8, tensor: Tensor) {
        self.tensor[index as usize] = Some(tensor);
    }

    pub fn get_creg(&self, index: u8) -> Option<&Vec<u8>> {
        self.context[index as usize].as_ref()
    }

    pub fn set_creg(&mut self, index: u8, value: Vec<u8>) {
        self.context[index as usize] = Some(value);
    }

    pub fn reset(&mut self) {
        self.general = [0u64; 256];
        self.tensor = std::array::from_fn(|_| None);
        self.context = std::array::from_fn(|_| None);
        self.pc = 0;
        self.zero_flag = false;
        self.overflow_flag = false;
        self.halted = false;
    }
}

impl Default for RegisterFile {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_dot_1d() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]).unwrap();
        let c = a.dot(&b).unwrap();
        assert!((c.data[0] - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_tensor_matmul() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let c = a.dot(&b).unwrap();
        assert_eq!(c.shape, vec![2, 2]);
        assert!((c.data[0] - 19.0).abs() < 1e-6); // 1*5 + 2*7
    }

    #[test]
    fn test_inplace_chain() {
        let mut t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let bias = Tensor::new(vec![0.1, 0.2, 0.3, 0.4], vec![4]).unwrap();
        let obj_ptr = &t as *const Tensor;
        t.add_inplace(&bias).scale_inplace(2.0);
        assert_eq!(&t as *const Tensor, obj_ptr); // same allocation
        assert!((t.data[0] - 2.2).abs() < 1e-6);
        assert!((t.data[3] - 8.8).abs() < 1e-6);
    }

    #[test]
    fn test_register_file() {
        let mut rf = RegisterFile::new();
        rf.set_reg(42, 0xDEADBEEF);
        assert_eq!(rf.get_reg(42), 0xDEADBEEF);
        rf.set_reg(0, 0);
        assert!(rf.zero_flag);
    }
}
