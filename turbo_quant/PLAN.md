# TurboQuant KV Cache Quantization

## 1. Paper Reference & Summary

**TurboQuant**: Data-oblivious online vector quantization for KV cache compression.
- Paper: https://arxiv.org/html/2504.19874v1
- Target model: Meta-Llama-3.1-8B-Instruct on Tenstorrent Wormhole (N150)

### Core Idea

TurboQuant compresses KV cache vectors during autoregressive LLM inference with
**no calibration data** and **near-optimal distortion**:

1. **Random rotation** (QR decomposition of Gaussian matrix) maps any vector to a
   known Beta distribution over coordinates
2. **Lloyd-Max quantization** per coordinate using precomputed codebooks
3. **Inner-product variant** adds 1-bit QJL on the residual for unbiased attention scores
4. Distortion guarantee: D_mse ≤ (√3π/2) · 4^(-b)

### Algorithm 1 — TurboQuant_mse

```
SETUP:
  Π ∈ ℝ^(d×d) via QR(randn(d,d))
  Precompute optimal centroids c₁,...,c_{2^b} via Lloyd-Max on Beta(d)

QUANTIZE(x):
  y = Π · x                          # rotate
  idx_j = argmin_k |y_j - c_k|       # nearest centroid per coordinate
  return idx, ||y||₂                  # b-bit indices + L2 norm

DEQUANTIZE(idx, norm):
  ỹ_j = c_{idx_j} · norm             # retrieve + rescale
  x̃ = Πᵀ · ỹ                         # rotate back
```

### Algorithm 2 — TurboQuant_prod (Inner-Product-Optimized)

```
QUANTIZE(x):
  idx, norm = mse_quantize(x)         # MSE at (b-1) bits
  r = x - mse_dequantize(idx, norm)   # residual
  qjl = sign(S · r)                   # 1-bit QJL
  return (idx, qjl, ||r||₂)

DEQUANTIZE:
  x̃ = mse_dequantize(idx) + √(π/2)/d · γ · Sᵀ · qjl
```

### Codebook Details

Coordinate distribution after rotation: f(x) = Γ(d/2)/(√π·Γ((d-1)/2)) · (1-x²)^((d-3)/2)

Precomputed MSE distortion per bit-width (d=128):

| Bits | Centroids | MSE (theoretical) | MSE (measured CPU) |
|------|-----------|--------------------|--------------------|
| 1 | 2 | 0.36 | 0.362 |
| 2 | 4 | 0.117 | 0.117 |
| 3 | 8 | 0.03 | 0.034 |
| 4 | 16 | 0.009 | 0.009 |

---

## 2. PyTorch CPU Implementation

### File Structure

```
turbo_quant/
├── __init__.py
├── rotation.py              # Random orthogonal matrix (QR decomposition)
├── codebook.py              # Lloyd-Max codebook on Beta distribution
├── quantizer.py             # TurboQuantMSE, TurboQuantProd, OutlierAwareTurboQuant
├── bitpack.py               # CPU bit-packing/unpacking for 1/2/3/4-bit indices
├── kv_cache.py              # HuggingFace-compatible KV cache wrapper
├── llama_integration.py     # HuggingFace Llama generation integration
├── ttnn_integration.py      # TTNN on-device implementation (Section 3)
├── eval_e2e.py              # End-to-end decode benchmark (teacher-forced + trace)
├── eval_e2e_prefill.py      # Real prefill + TurboQuant decode
├── eval_quality_comparison.py  # Side-by-side quality comparison
└── benchmarks/
    ├── test_correctness.py  # 34 CPU unit tests
    ├── test_ttnn.py         # 9 TTNN hardware tests
    ├── eval_perplexity.py   # Synthetic + real model perplexity
    ├── eval_latency.py      # Throughput and memory profiling
    ├── eval_needle.py       # Needle-in-a-haystack retrieval
    └── results_llama31_8b.json
```

### Components

| Module | What it does |
|--------|-------------|
| `rotation.py` | Generates Π ∈ ℝ^(d×d) via QR decomposition. Seed-based, orthogonality error < 1e-15. |
| `codebook.py` | Lloyd-Max algorithm for Beta distribution. Precomputed centroids for b=1,2,3,4. |
| `quantizer.py` | Three variants: MSE (Algorithm 1), Prod (Algorithm 2), OutlierAware (mixed bit-widths). |
| `bitpack.py` | Packs b-bit indices into uint8. 1-bit: 8×, 2-bit: 4×, 3-bit: 2.67×, 4-bit: 2×. |
| `kv_cache.py` | Drop-in replacement for HF `DynamicCache`. Supports all variants + beam search. |
| `llama_integration.py` | Monkey-patches `prepare_inputs_for_generation` for HF Llama models. |

### CPU Test Suite (34 tests, ~0.8s)

| Test Class | Count | Coverage |
|-----------|-------|---------|
| `TestRotationMatrix` | 3 | Orthogonality, determinism, seed variation |
| `TestCodebook` | 4 | Sorted centroids, symmetry, round-trip, index range |
| `TestTurboQuantMSE` | 4 | Shapes, monotonic MSE, zero input, single token |
| `TestTurboQuantProd` | 4 | Shapes, IP unbiasedness, IP error, min-bits validation |
| `TestOutlierAwareTurboQuant` | 7 | Effective bits, MSE ordering, calibration, configs |
| `TestBitPack` | 6 | Round-trip all bit-widths, compression, shapes, boundaries |
| `TestTurboQuantCache` | 6 | Prefill/decode, all variants, memory, bitpack |

Run: `PYTHONPATH=. python -m turbo_quant.benchmarks.test_correctness`

---

## 3. TTNN On-Device Implementation

### Fused Compute Kernels (C++)

Two custom TTNN device operations replace cascaded TTNN primitives with single-pass
SFPU kernels that operate entirely in DST registers (no DRAM intermediates):

**Fused Bucketize** (`turbo_quant_bucketize.cpp`):
Replaces 13 ops (7×ge + 6×add) with one kernel. Per tile: load input → loop over
boundaries using `copy_tile → unary_ge_tile → add_binary_tile`. ~31 SFPU ops/tile.

**Fused Gather Centroids** (`turbo_quant_gather_centroids.cpp`):
Replaces 21 ops (7×ge + 7×full_like + 7×where) with one kernel. Per tile:
conditional overwrite via `copy_tile → unary_ge_tile → fill_tile → sub/mul/add_binary_tile`.
~52 SFPU ops/tile.

Both use the multi-core program factory (`split_work_to_cores`) and reuse the
standard `reader/writer_unary_interleaved_start_id.cpp` dataflow kernels.

**Python API:**
```python
ttnn.experimental.turbo_quant_bucketize(input_tensor, boundaries)
ttnn.experimental.turbo_quant_gather_centroids(input_tensor, centroids)
```

Auto-detected at import time (`_FUSED_OPS_AVAILABLE`), graceful fallback to cascaded ops.

### C++ File Structure

```
ttnn/cpp/ttnn/operations/experimental/turbo_quant/
├── CMakeLists.txt
├── turbo_quant.hpp / .cpp            # Public API
├── turbo_quant_nanobind.hpp / .cpp   # Python bindings
└── device/
    ├── turbo_quant_device_operation.hpp / .cpp
    ├── turbo_quant_program_factory.cpp
    └── kernels/compute/
        ├── turbo_quant_bucketize.cpp
        └── turbo_quant_gather_centroids.cpp
```

### Key Optimisations

| Optimisation | What | Impact |
|-------------|------|--------|
| Fused kernels (B1+B2) | Single SFPU pass vs 13+21 TTNN ops | 71ms → 47ms |
| Rotated-space SDPA (B3) | Pre-rotate Q, post-rotate output, skip inverse rotation on full cache | +3ms at seq=512 |
| Cache centroid values | Gather at quantize time (1 token) not dequantize time (full cache) | 54ms → 49ms at seq=512 |
| Absorb Π into W_v/W_o | Bake rotation into projection weights (V has no RoPE) | Saves 64 matmuls/step |
| Pre-rescale centroids×norms | Store final values, dequantize = identity | O(1) dequantize |
| Paged BF16 cache | Use paged SDPA on pre-rescaled BF16 values | Flat latency, 128K context |
| BFP4 index cache | Integers 0-7 exact in BFP4 (~0.5 byte/elem) | 2× smaller than baseline |
| rsqrt norm | Replace square+sum+sqrt+div with rsqrt | Fewer ops in quantize |
| Free layer_past | Deallocate BFP8 layer_past when TQ active | Doubles available DRAM |

### TTNN Hardware Test Suite (9 tests)

| Test | What it covers |
|------|---------------|
| `test_setup_tensors` | Rotation matrix upload, orthogonality on device |
| `test_quantize_shapes` | Index range [0, 2^b-1], output shapes |
| `test_dequantize_shapes` | Round-trip shape preservation |
| `test_roundtrip_quality` | MSE within BF16 bounds per bit-width |
| `test_cpu_reference_match` | Device vs CPU cosine > 0.70, index match > 90% |
| `test_monotonic_mse` | More bits → lower MSE (monotonicity) |
| `test_sdpa_decode_loop` | 4-step decode with real `scaled_dot_product_attention_decode` |
| `test_cache_update_and_dequantize` | Full cache scatter + dequantize pipeline |
| `test_latency` | Quantize + dequantize timing |

Run: `PYTHONPATH=. python turbo_quant/benchmarks/test_ttnn.py`

---

## 4. TurboQuant Variants

### 4.1 — Baseline BFP8 (no TurboQuant)

Standard Llama-3.1-8B inference path. BFP8 KV cache with paged SDPA.
No quantization overhead. Used as the performance/memory reference.

- **Flag:** `--no-turbo-quant`
- **KV memory:** ~1 byte/element (BFP8)
- **Latency:** 37ms/tok (flat across all seq lengths)

### 4.2 — TurboQuant Performance (paged BF16 pre-rescaled)

Optimised for **latency parity with baseline**. Stores pre-rescaled centroid×norm
values as BF16 in the model's paged KV cache. Dequantize is a no-op (values are
already in the format SDPA expects). Rotation absorbed into W_v/W_o weights.

- **Flag:** default mode (no flag needed)
- **KV memory:** 2 bytes/element (BF16) — 2× baseline
- **Latency:** 37.2ms/tok (flat, matches baseline)
- **Max context:** 128K tested, no OOM on N150
- **Tradeoff:** 2× more KV memory than baseline

### 4.3 — TurboQuant Memory-Efficient (BFP4 indices, contiguous)

Optimised for **minimal KV cache memory**. Stores BFP4 quantization indices
(~0.5 bytes/elem) + BF16 norms in a contiguous (non-paged) cache. BFP4 exactly
represents integers 0-7 (verified on hardware). Dequantize requires typecast +
gather + mul per step, proportional to max_seq_len.

- **Flag:** `memory_efficient=True` in `TTNNTurboQuantCache`
- **KV memory:** ~0.52 bytes/element (BFP4 indices + BF16 norms) — **2× smaller than baseline**
- **Latency:** 57–61ms/tok (higher, without trace/paging)
- **Max context:** 16K (OOM at 32K due to contiguous dequantize temp allocation)
- **Tradeoff:** slower decode due to contiguous SDPA + full-cache dequantize

---

## 5. TTNN Experiments

All measurements on **Wormhole N150, Llama-3.1-8B-Instruct, batch=1, 3-bit**.

### Performance (ms/tok)

Measured 2026-04-10, Wormhole N150, greedy decoding, 10 tokens generated.

| max_seq | Baseline BFP8 | TQ Performance | TQ Memory-Efficient (BFP4) |
|---------|--------------|----------------|----------------------------|
| 128 | 36.9 | 37.2 | 59.4 |
| 256 | 36.9 | 37.3 | 61.2 |
| 512 | 37.0 | 37.2 | 59.3 |
| 1024 | 37.0 | 37.2 | 56.6 |
| 2048 | 36.9 | 37.2 | 59.0 |
| 4096 | 36.9 | 37.2 | 60.1 |
| 8192 | 37.0 | 37.3 | 57.4 |
| 16384 | 37.0 | 37.2 | 57.4 |
| 32768 | 36.9 | 37.2 | OOM |
| 65536 | 37.0 | 37.2 | OOM |
| 131072 | 37.0 | 37.2 | OOM |

### KV Cache Memory (per batch, 32 layers × 2 K/V × 8 heads × 128 dim)

| max_seq | Baseline BFP8 (~1 B/elem) | TQ Performance (2 B/elem) | TQ Memory-Efficient (~0.5 B/elem) |
|---------|--------------------------|--------------------------|-----------------------------------|
| 128 | 9 MB | 17 MB | **4 MB** |
| 256 | 18 MB | 34 MB | **9 MB** |
| 512 | 36 MB | 67 MB | **18 MB** |
| 1024 | 71 MB | 134 MB | **36 MB** |
| 2048 | 143 MB | 268 MB | **72 MB** |
| 4096 | 285 MB | 537 MB | **145 MB** |
| 8192 | 570 MB | 1.1 GB | **289 MB** |
| 16384 | 1.1 GB | 2.1 GB | **579 MB** |
| 32768 | 2.3 GB | 4.3 GB | **1.2 GB** |
| 65536 | 4.6 GB | 8.6 GB | **2.3 GB** |
| 131072 | 9.1 GB | 17.2 GB | **4.6 GB** |

### Quality (real prefill + decode, greedy sampling)

| Prompt | Bits | Output | Correct? |
|--------|------|--------|----------|
| "What is the capital of France?" | 2 | "The capital of France is Paris." | Yes |
| "What is the capital of France?" | 3 | "The capital of France is Paris." | Yes |
| "What is the capital of France?" | 4 | "The capital of France is Paris." | Yes |
| Quantum computing (3 sentences) | 2 | Correct: superposition, entanglement, parallel processing | Yes |
| Quantum computing (3 sentences) | 3 | Correct: same topics, slightly different wording | Yes |
| Number sequence (continue 90..99) | 3 | Correctly outputs "100" | Yes |
| Megaliths (188-tok, avoid keywords, `******` separator) | 3 | Follows all constraints | Yes |
| Grafton VT (3 paragraphs, start with "send") | 3 | Follows all formatting | Yes |

### Optimisation History

```
183ms/tok   initial TurboQuant (A0)
 71ms/tok   TTNN trace (A2t)
 47ms/tok   fused kernels (B1+B2)
 46ms/tok   cache centroids
 45.6ms     absorb Π into W_v/W_o
 44.1ms     pre-rescale centroids×norms
 43.5ms     rsqrt norm + remove UINT32 typecast
 37.2ms     paged BF16 with paged SDPA (= baseline)
```

---

## 6. Constraints & Next Steps

### Why BFP4/BFP8 paged caches don't work for pre-rescaled values

**Tested 2026-04-10:** Storing pre-rescaled centroid×norm values as BFP8 in the
paged `layer_past` (instead of BF16) produces **garbage output** despite synthetic
tests showing only 0.2% added error (cosine 0.99997). The BFP8 shared exponent
per 32-element block causes selective precision loss on the specific value patterns
that attention relies on. BFP4 is even worse for continuous values.

This means:
- **Paged SDPA requires BF16** for pre-rescaled TQ values (2 bytes/element)
- **BFP4 indices can't be stored in paged `layer_past`** — SDPA interprets them
  as K/V values, not indices
- There's no hook between paged cache read and SDPA computation to dequantize

### Current variant tradeoffs (no further optimisation possible without kernel changes)

| Variant | Latency | KV Memory | Max Context | Bottleneck |
|---------|---------|-----------|-------------|------------|
| TQ Performance | **37ms** (= baseline) | 2× baseline | 128K | BF16 required by SDPA |
| TQ Memory-Efficient | 57-61ms | **0.5× baseline** | 16K | Contiguous dequantize temp OOM |
| Baseline BFP8 | **37ms** | 1× | 128K | — |

### Fused Paged SDPA Reader (would unlock best of both worlds)

Modify the paged SDPA kernel to read BFP4 indices + BF16 norms from paged cache
and dequantize on-the-fly (gather centroids + mul norms) inside the kernel's
chunk-reading loop. This would give:
- **37ms latency** (paged SDPA, same as baseline)
- **~0.5 bytes/element** (BFP4 indices, 2× smaller than baseline)
- **128K+ context** (paged allocation)

**Engineering scope:** Modify `reader_interleaved.cpp` (~500 lines) and
`sdpa_program_factory.cpp` to:
1. Read BFP4 tiles from paged KV cache instead of BF16/BFP8
2. Add a compute stage that typecasts BFP4 → float32 in DST
3. Apply centroid gather (reuse B2 kernel logic) + norm multiply
4. Feed dequantized tiles to existing SDPA attention computation

This is the single remaining optimisation that would make TurboQuant
strictly better than baseline (same speed, half the memory).

### Other Next Steps

**Quality benchmarks:**
- LongBench passkey-retrieval at 3-bit vs FP16 baseline
- WikiText-2 perplexity measurement
- Multi-turn conversation quality at 16K+ context

**Multi-batch / Multi-device:**
- Batch > 1: TQ's compressed cache enables more concurrent sequences
- Galaxy (8+ devices): already has `TT_NUM_DEVICES` support, needs testing

**True 3-bit packing:**
Pack indices to 3 bits/element (0.375 bytes) instead of BFP4's ~0.5 bytes.
Requires custom pack/unpack kernels + ROW_MAJOR scatter. CPU implementation
in `bitpack.py`. Only worthwhile after fused SDPA reader is implemented.
