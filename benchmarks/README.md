# Python benchmark suite

`bench_scrawl.py` is the original 8-subsystem Python harness. It is kept here as source documentation.

**It cannot be run from this repository.** It begins with:

```python
from src.vm import ScrawlVM, TraceSeverity
from src.synapse import Instruction, SynapseEncoder, SynapseDecoder
...
```

The Python package `src/` is not included in this tree (see the top-level README). Measured size numbers in `BENCHMARKS.md` were taken from the Rust encoder and compressor instead.

```bash
# When the Python core is available:
python benchmarks/bench_scrawl.py --quick
```

Rust benches: `cargo bench`.
