# TurboQuant KV Cache Quantization

## 1. Paper Reference & Summary

**TurboQuant**: Data-oblivious online vector quantization for KV cache compression.
- Paper: https://arxiv.org/html/2504.19874v1
- Target model: Meta-Llama-3.1-8B-Instruct on Tenstorrent Wormhole (N150)
- Model weights: `HF_HOME=/localdev/proj_sw/user_dev/hf_data`

### Result: same speed, half the KV cache, scales to 2,213 tok/s on T3K

3-bit TurboQuant with BFP4 paged cache vs baseline BFP8:

| | Baseline BFP8 | TurboQuant 3-bit BFP4 |
|--|--|--|
| **Single device (N150), batch=1** | 37.0 ms/tok | **37.2 ms/tok** (+0.2ms, +0.5%) |
| **T3K 8-device, batch=1** | 14.0 ms/tok | **14.2 ms/tok** (71 tok/s, 2.6× speedup) |
| **T3K 8-device, batch=32** | (KV-limited) | **2,213 tok/s** (14.5ms, 31× scaling) |
| **KV cache memory** | 1× (~1 byte/elem) | **0.5×** (~0.5 byte/elem) |
| **Quality** | — | Correct output at all seq lengths 128–131072 |
| **Cosine vs CPU ref** | — | > 0.999 (synthetic SDPA test at all seqlens) |
| **MSE vs float32 CPU** | — | 0.034 (matches paper bound) |
| **Max context** | 128K | **128K** |

Verified 2026-04-14: BFP4 paged cache + standard SDPA decode, flat 37.1–37.2 ms/tok
from seq=128 to seq=131072. Pre-rescaled centroid×norm values stored as BFP4 in paged
`layer_past`, fed directly to `scaled_dot_product_attention_decode` which natively
accepts BFP4 inputs. No custom SDPA kernel needed for this path.

### T3K Multi-Device Result (2026-04-17)

Running on T3K (8× Wormhole) with `TT_NUM_DEVICES=8` and `FABRIC_1D` config:

| | Single device (N150) | T3K (8 devices) | Speedup |
|--|---------------------|-----------------|---------|
| **Baseline BFP8** | 37.0 ms/tok | **14.0 ms/tok** | 2.6× |
| **TQ BFP4 paged** | 37.2 ms/tok | **14.2 ms/tok** | 2.6× |
| **TQ overhead** | +0.2ms (+0.5%) | +0.2ms (+1.4%) | constant |

**70.6 tok/s on T3K, flat across 128 → 131072 seqlens.** Change was 4 lines:
call `ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)` before opening the
mesh when `num_devices > 1`. TQ's `from_torch` calls replicate constants across
devices automatically. KV heads shard across devices: Llama-3.1-8B has 8 KV heads
→ 1 head/device on T3K.

### T3K Batch Throughput (2026-04-17)

Batch scaling on T3K with `--batch-size N`:

| Batch | Latency (ms/tok) | Throughput (tok/s) | Scaling vs batch=1 |
|-------|------------------|--------------------|--------------------|
| 1 | 14.2 | 70.6 | 1.00× |
| 4 | 14.1 | 283.4 | 4.01× |
| 8 | 14.1 | 565.9 | 8.01× |
| 16 | 14.2 | 1,128.2 | 15.98× |
| **32** | **14.5** | **2,213.2** | **31.35×** |

**Perfect linear scaling up to batch=32. 2,213 tok/s peak throughput.**
Latency barely grows (14.2 → 14.5ms). TQ's compressed BFP4 cache (0.5 bytes/elem,
2× smaller than baseline BFP8) enables these large batch sizes at long seqlens
without running out of DRAM — this is the key benefit of KV compression for serving.

### T3K Multi-Device + Multi-Batch Quality (2026-04-17)

End-to-end quality verified on T3K with rotation-absorbed model and BF16 migration:

**10 diverse prompts (capitals, currencies, recipes, math, jokes, biology):**
All outputs factually correct, matches baseline. Spot-checks:
- Mix yellow + blue → green ✓
- 2+2 → 4 ✓
- Capital of USA → Washington D.C. ✓
- Capital of France → Paris ✓
- Currency of Brazil → BRL ✓

**Batch consistency check (batch=1 vs batch=4, same prompt):**
Bit-exact identical token IDs across all 20 generated tokens. No batching artifacts.

**Migration path:** Multi-device KV migration uses `ConcatMesh2dToTensor(dims=(0,1))`
to read sharded KV heads → full tensor, `ShardTensor2dMesh(dims=(None, 1))` to
write quantized values back to correct devices.

### E2E Overhead: TQ BFP4 vs Baseline BFP8 (2026-04-14)

Back-to-back comparison, same machine, same prompt, traced, 10 generated tokens.

| max_seq | Baseline BFP8 | TQ BFP4 Paged | Overhead |
|---------|--------------|---------------|----------|
| 128 | 36.9 ms/tok | 37.1 ms/tok | +0.2ms (+0.5%) |
| 1,024 | 36.9 ms/tok | 37.2 ms/tok | +0.3ms (+0.8%) |
| 4,096 | 37.0 ms/tok | 37.2 ms/tok | +0.2ms (+0.5%) |
| 16,384 | 37.0 ms/tok | 37.2 ms/tok | +0.2ms (+0.5%) |
| 65,536 | 37.0 ms/tok | 37.2 ms/tok | +0.2ms (+0.5%) |
| 131,072 | 37.0 ms/tok | 37.1 ms/tok | +0.1ms (+0.3%) |

Overhead is **constant O(1)** — only touches the 1 new token per step:
permute → centroid lookup (fused bucketize) → norm → pre-rescale (centroid×norm)
→ permute back → paged_update_cache scatter (BF16→BFP4 hardware conversion).
Rotation cost is zero (absorbed into W_v/W_o at model load time).

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

### Fused TQ SDPA Decode Kernel

Custom SDPA decode kernel that reads BFP4 quantization indices + BF16 norms from the
TQ cache and dequantizes on-the-fly during SDPA computation. Eliminates the full-cache
BF16 dequantize temporary that would otherwise be needed at long sequences.

**Pipeline (3 RISC-V cores per Tensix):**

```
Reader (RISC-V 0)           Compute (RISC-V 1-3)              Writer (RISC-V 4)
─────────────────           ──────────────────────             ─────────────────
Read Q (BF16) ──→ c_0       Pass 1: init_sfpu + typecast       Generate scale/
Read K idx (BFP4) → c_10      BFP4→BF16 into c_14              identity tiles
Read K norms (BF16)→ c_11   Pass 2: mm_init + centroid gather   Write output
Read V idx (BFP4) → c_12      + norm bcast_cols + K transpose    from c_16
Read V norms (BF16)→ c_13     → c_1 (K) and c_2 (V)
                             Then: sdpa_standard(c_0,c_1,c_2)
                               → c_16
```

**Key technical details:**
- `init_sfpu(src_cb, dst_cb)` required for BFP4→BF16 typecast (not just `copy_tile`)
- `mm_init()` must be called between SFPU (typecast) and FPU (bcast multiply) modes
- Centroid gather uses in-place `pack_tile<true>` to overwrite index tiles
- K tiles transposed during output via `pack_tile<true>(0, cb, col*Sk+row)`
- Reader reads K NOT transposed (row-major for norm alignment), compute transposes

**Python API:**
```python
ttnn.experimental.turbo_quant_sdpa_decode(
    q, k_indices, k_norms, v_indices, v_norms,
    page_table, cur_pos, centroids, scale
)
```

**Test results (cosine vs CPU reference):**
- 1 head: 0.9998, 8 heads: 0.9996, 32Q/8KV GQA: 0.9996

### C++ File Structure

```
ttnn/cpp/ttnn/operations/experimental/turbo_quant/
├── CMakeLists.txt
├── turbo_quant.hpp / .cpp            # Public API
├── turbo_quant_nanobind.hpp / .cpp   # Python bindings
├── device/
│   ├── turbo_quant_device_operation.hpp / .cpp
│   ├── turbo_quant_program_factory.cpp
│   └── kernels/compute/
│       ├── turbo_quant_bucketize.cpp
│       └── turbo_quant_gather_centroids.cpp
└── sdpa/
    ├── device/
    │   ├── sdpa_tq_device_operation.hpp / .cpp
    │   └── sdpa_tq_program_factory.cpp
    └── kernels/
        ├── compute/sdpa_tq_decode.cpp
        └── dataflow/
            ├── reader_tq_decode.cpp
            └── writer_tq_decode.cpp
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
| Fused TQ SDPA kernel | On-the-fly dequant inside SDPA (typecast+gather+norm) | No BF16 temp tensor |
| Pre-rescaled BFP4 SDPA | Typecast-only path: store centroid×norm as BFP4 | **= baseline latency, 4× less memory** |
| Multi-core fused SDPA | Distribute batch×heads across compute grid | ~8× speedup |
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

## 4. TurboQuant — Main Version

### Baseline BFP8 (no TurboQuant)

Standard Llama-3.1-8B inference path. BFP8 KV cache with paged SDPA.
No quantization overhead. Used as the performance/memory reference.

- **Flag:** `--no-turbo-quant`
- **KV memory:** ~1 byte/element (BFP8)
- **Latency:** 37ms/tok (flat across all seq lengths)

### TurboQuant BFP4 Paged Cache + Standard SDPA — THE MAIN VERSION

**This is the production TurboQuant variant.** Pre-rescales centroid×norm at quantize
time (O(1) per token), stores as BFP4 (~0.5 bytes/elem) in the model's paged KV
cache. Standard `scaled_dot_product_attention_decode` reads BFP4 natively — the
matmul unpacker handles BFP4→internal format conversion automatically. No custom
SDPA kernel needed. Rotation absorbed into W_v/W_o weights.

- **Flag:** `--bfp4-cache` in eval_e2e.py
- **KV memory:** ~0.5 bytes/element (BFP4) — **2× smaller than baseline**
- **Latency:** 37.1–37.2 ms/tok (flat, matches baseline)
- **Max context:** 128K verified (2026-04-14)
- **Cosine:** > 0.999 vs CPU reference at all seqlens
- **Status:** production-ready, verified at all seqlens 128–131072

### TurboQuant Pre-Rescaled BFP4 (fused SDPA) — DEVELOPMENT TRACK

Custom fused SDPA kernel that typecasts BFP4→BF16 on-the-fly with TQ-specific
dequant steps inside the kernel. Preserves fine-grained control over the dequant
pipeline. Currently limited to ~2K seq (L1 overflow at 4K+). Needs chunked online
softmax (Flash Attention style) to work at longer sequences — see Section 6.

- **Flag:** `pre_rescaled=True` in `turbo_quant_sdpa_decode`
- **KV memory:** ~0.5 bytes/element (BFP4) — **2× smaller than baseline**
- **Latency:** matches standard SDPA (0.03ms at seq=128, 0.17ms at seq=2048)
- **Cosine:** > 0.999 vs BFP4 roundtrip reference
- **Status:** multi-core, limited to ~2K seq (chunked dequant in progress)

### Deprecated Variants

The following variants are **deprecated** and should not be used for future
development or testing. They are retained here for historical reference only.

**TQ Performance (paged BF16 pre-rescaled)** — DEPRECATED
Stored pre-rescaled centroid×norm as BF16 in the model's paged KV cache.
Achieved latency parity at all seqlens up to 128K, but uses **2× baseline memory**
(BF16 = 2 bytes/elem vs BFP8 = 1 byte/elem). Superseded by the fused SDPA BFP4
variant which achieves 0.5× baseline memory instead.

**TQ Full Dequant (BFP4 indices + norms, fused SDPA)** — DEPRECATED
Stored BFP4 quantization indices + BF16 norms separately. Fused SDPA kernel did
centroid gather + norm multiply on-the-fly. Higher quality than pre-rescaled but
~15-35× slower due to per-tile centroid gather (~50 SFPU ops/tile). The quality
advantage does not justify the latency cost.

---

## 5. TTNN Experiments

All measurements on **Wormhole N150, Llama-3.1-8B-Instruct, batch=1, 3-bit**.

### Fused TQ SDPA Kernel Benchmark (multi-core) — MAIN VERSION

Synthetic data, 8Q/8KV heads, hd=128, 3-bit, Wormhole N150.
Verified 2026-04-14: passes 128–2048, fails 4096+ (L1 overflow at 34MB > 1.5MB).

| Seq Len | Pre-rescaled (ms) | Std SDPA (ms) | Cosine | BFP4 KV (MB) |
|---------|-------------------|---------------|--------|--------------|
| 128 | **0.03** | 0.03 | 0.9996 | 0.1 |
| 256 | **0.04** | 0.04 | 0.9996 | 0.2 |
| 512 | **0.05** | 0.06 | 0.9996 | 0.5 |
| 1024 | **0.09** | 0.11 | 0.9997 | 1.0 |
| 2048 | **0.17** | 0.20 | 0.9997 | 2.0 |
| 4096 | L1 limit | 0.41 | — | 4.0 |
| 8192 | L1 limit | 0.80 | — | 8.0 |
| 16384 | L1 limit | 1.57 | — | 16.0 |
| 32768 | L1 limit | 3.10 | — | 32.0 |
| 65536 | L1 limit | 6.21 | — | 64.0 |
| 131072 | L1 limit | 12.37 | — | 128.0 |

**Key findings:**
- **Pre-rescaled mode matches standard SDPA latency** — typecast-only path has negligible overhead
- Multi-core gives ~8× speedup over single-core (8 heads on 8 cores)
- L1 limit at 4K+: fused kernel pre-fills full BF16 cache in L1 CBs. Chunked dequant (Section 6) is the fix

### BFP4 K/V + Standard SDPA Decode — Synthetic Validation (2026-04-14)

The standard `scaled_dot_product_attention_decode` natively accepts BFP4 K/V inputs.
Pre-rescaled centroid×norm values stored as BFP4, fed directly to standard SDPA.
No custom kernel — standard SDPA's chunked online softmax handles all seqlens.

Synthetic data, 8Q/8KV heads, hd=128, 3-bit, Wormhole N150.

| Seq Len | Cosine vs CPU ref | Status | BFP4 KV (MB) |
|---------|-------------------|--------|--------------|
| 128 | 0.9996 | PASS | 0.1 |
| 256 | 0.9997 | PASS | 0.2 |
| 512 | 0.9997 | PASS | 0.5 |
| 1,024 | 0.9997 | PASS | 1.0 |
| 2,048 | 0.9997 | PASS | 2.0 |
| 4,096 | 0.9997 | PASS | 4.0 |
| 8,192 | 0.9997 | PASS | 8.0 |
| 16,384 | 0.9995 | PASS | 16.0 |
| 32,768 | 0.9996 | PASS | 32.0 |
| 65,536 | 0.9996 | PASS | 64.0 |
| 131,072 | 0.9995 | PASS | 128.0 |

**All 11 seqlens pass with cosine > 0.999.** No crashes, no L1 limit — standard SDPA
handles BFP4 at full 128K context. BFP4 shared exponent precision is sufficient for
pre-rescaled attention values.

### BFP4 Paged Cache — End-to-End Sweep (2026-04-14) — MAIN VERSION

Full Llama-3.1-8B-Instruct, teacher-forced decode, 3-bit TQ, traced, 10 generated
tokens. BFP4 paged `layer_past` + standard `scaled_dot_product_attention_decode`.
Correct output ("The capital of France is Paris.") at every seqlen.

| max_seq | TQ BFP4 Paged (ms/tok) | Warm avg (ms/tok) |
|---------|------------------------|-------------------|
| 128 | 37.2 | 37.2 |
| 256 | 37.2 | 37.2 |
| 512 | 37.2 | 37.2 |
| 1,024 | 37.1 | 37.1 |
| 2,048 | 37.2 | 37.2 |
| 4,096 | 37.2 | 37.2 |
| 8,192 | 37.1 | 37.1 |
| 16,384 | 37.1 | 37.1 |
| 32,768 | 37.1 | 37.1 |
| 65,536 | 37.2 | 37.2 |
| 131,072 | 37.1 | 37.1 |

**Flat 37.1–37.2 ms/tok across all seqlens.** Same speed as baseline BFP8, half the
KV cache memory (BFP4 = 0.5 bytes/elem). Works at full 128K context with no custom
SDPA kernel — standard SDPA's chunked online softmax handles BFP4 natively.

### KV Cache Memory (per batch, 32 layers × 2 K/V × 8 heads × 128 dim)

| max_seq | Baseline BFP8 (~1 B/elem) | TQ Pre-Rescaled BFP4 (~0.5 B/elem) | Savings |
|---------|--------------------------|-------------------------------------|---------|
| 2,048 | 143 MB | **72 MB** | 2× |
| 4,096 | 285 MB | **145 MB** | 2× |
| 8,192 | 570 MB | **289 MB** | 2× |
| 16,384 | 1.1 GB | **579 MB** | 2× |
| 32,768 | 2.3 GB | **1.2 GB** | 2× |
| 65,536 | 4.6 GB | **2.3 GB** | 2× |
| 131,072 | 9.1 GB | **4.6 GB** | 2× |

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

### WikiText-2 Perplexity + KV Cache Distortion (2026-04-14)

Llama-3.1-8B-Instruct, BF16 on CPU, 4550 tokens, sliding-window PPL.
KV distortion measured on 5 captured windows × 32 layers of real KV tensors.

**Baseline perplexity: 9.91**

| Variant | Key MSE | Key Cosine | Value MSE | Value Cosine |
|---------|---------|------------|-----------|--------------|
| MSE 2-bit | 0.4725 | 0.9400 | 0.0123 | 0.9401 |
| **MSE 3-bit** | **0.1384** | **0.9828** | **0.0036** | **0.9828** |
| MSE 4-bit | 0.0380 | 0.9953 | 0.0010 | 0.9953 |
| Outlier 2.25-bit | 0.3893 | 0.9508 | 0.0101 | 0.9510 |

3-bit (production config): cosine 0.983 on real model KV tensors, matches paper bound.
Monotonic quality: 2-bit → 3-bit → 4-bit. Outlier 2.25-bit outperforms plain 2-bit.

### Needle-in-a-Haystack Retrieval (2026-04-14)

Synthetic test: planted needle key with known high affinity, measured retrieval after
TQ quantize/dequantize. 5 needle positions × 6 haystack lengths × 6 variants = 180 tests.

| Variant | Retrieval Accuracy | Haystack lengths |
|---------|-------------------|------------------|
| FP32 baseline | 30/30 (100%) | 64, 256, 1K, 4K, 16K, 64K |
| MSE 2-bit | 30/30 (100%) | " |
| MSE 3-bit | 30/30 (100%) | " |
| MSE 4-bit | 30/30 (100%) | " |
| Prod 3-bit | 30/30 (100%) | " |
| Outlier 2.25-bit | 30/30 (100%) | " |

**100% retrieval across all variants**, including 2-bit, at all context lengths up to 64K.

### Quality Comparison: 31 Diverse Prompts (2026-04-14)

Side-by-side baseline BFP8 vs TQ 3-bit on 31 prompts (capitals, currencies, recipes,
jokes, math, biology, sports, travel). Prefill + 100 tokens decode per prompt.

- **All 31 prompts produce factually correct, coherent output** — no hallucinations,
  no repetitive text, no quality degradation from TQ quantization
- **Average word overlap: 81%** (range 57–96%)
- No exact matches expected: rotation absorption changes model wording slightly
- Spot-checks: capitals (Washington D.C., Ottawa, London, Berlin, Paris, Tokyo,
  Lisbon, Beijing), currencies (all correct), math (2+2=4), color mixing (green)

### Optimisation History

```
183ms/tok   initial TurboQuant (A0)
 71ms/tok   TTNN trace (A2t)
 47ms/tok   fused kernels (B1+B2)
 46ms/tok   cache centroids
 45.6ms     absorb Π into W_v/W_o
 44.1ms     pre-rescale centroids×norms
 43.5ms     rsqrt norm + remove UINT32 typecast
 37.2ms     paged BF16 with paged SDPA (= baseline, 2 bytes/elem)
 37.2ms     BFP4 index cache (= baseline, FLAT 128→131072, 0.5 bytes/elem)
  0.17ms    fused BFP4 SDPA pre-rescaled @ seq=2048 (= baseline, 0.5 bytes/elem)
 37.1ms     BFP4 paged cache + standard SDPA (= baseline, FLAT 128→131072, 0.5 bytes/elem)
```

---

## 6. Constraints & Next Steps

### Status Summary

| | Latency | KV Memory | Max Context | Status |
|--|---------|-----------|-------------|--------|
| **TQ BFP4 Paged + Std SDPA** | **= baseline** | **0.5× baseline** | **128K** | **Production-ready (2026-04-14)** |
| Baseline BFP8 | = baseline | 1× | 128K | Reference |
| ~~TQ BFP4 Fused SDPA~~ | — | — | — | **DROPPED (2026-04-17)** — BFP4 paged path supersedes |

### Prefill → BFP4 decode: paged prefill path (follow-up)

`eval_e2e_prefill.py --bfp4-cache` works with non-paged model: prefill (BFP8) →
migrate (BFP4) → decode (BFP4, non-paged SDPA). Output is correct ("Paris.") with
no repetitive output issue. Decode latency is ~43ms/tok (non-paged SDPA) vs 37ms
(paged SDPA in eval_e2e.py).

To get 37ms decode after prefill, the model needs paged attention for decode. Paged
prefill currently fails with block_size mismatch (`Input tensor height (128) must be
<= cache tensor height (32)`). This is a model-level issue with how `prepare_inputs_prefill`
handles paged KV caches, not a TurboQuant issue. Follow-up: investigate paged prefill
setup or two-phase model init (non-paged prefill → paged decode).

### ~~Fused SDPA kernel — Chunked dequant~~ DROPPED (2026-04-17)

The custom fused SDPA kernel (`ttnn/cpp/ttnn/operations/experimental/turbo_quant/sdpa/`)
is deprecated. The BFP4 paged + standard SDPA path already delivers the target
(= baseline latency, 0.5× memory, 128K context) with no custom kernel needed.

C++ sources are retained in-tree for reference but are not on the production path.

### ~~Prefill → TQ decode migration (quality issue)~~ RESOLVED

Previously, migrating prefill BFP8 KV into TQ format produced repetitive output.
**Fixed as of 2026-04-14:** `eval_e2e_prefill.py --bfp4-cache` produces correct,
coherent output ("The capital of France is Paris.") with no repetition. The fix
was storing centroid×norm (not just centroids) and using BFP4/BF16 cache.

### ~~Quality benchmarks~~ DONE (2026-04-14)

All completed:
- **WikiText-2 perplexity:** baseline 9.91, 3-bit KV cosine 0.983, MSE 0.138
- **Needle-in-a-haystack:** 100% retrieval across all variants (2/3/4-bit) up to 64K
- **31-prompt quality comparison:** all factually correct, 81% avg word overlap vs baseline

### Remaining Next Steps

**Paged prefill → paged decode (for 37ms decode after prefill):**
Currently prefill uses non-paged model (43ms decode). Paged prefill fails with
block_size mismatch — model-level issue, not TQ-specific.

**Max batch at 128K context:**
Sweep to find batch limit at long seqlen. TQ's 2× KV compression should enable
significantly larger batches than baseline BFP8 at extended context.

**Galaxy (TG, 32 devices):**
Untested. Should work with same `FABRIC_1D` config. Would give ~4× T3K throughput.

**Formal accuracy benchmarks:**
MMLU, HellaSwag, or similar — for a formal "accuracy retained" number.

**~~Multi-batch~~ DONE (2026-04-17):**
- T3K batch sweep: perfect linear scaling 1→32, 2,213 tok/s peak at batch=32

**~~Multi-device~~ DONE (2026-04-17):**
- T3K (8× Wormhole) verified: 14.2 ms/tok, 2.6× speedup vs single device

**~~Fused SDPA kernel~~ DROPPED (2026-04-17):**
- Redundant with BFP4 paged path. C++ sources retained for reference only.

**True 3-bit packing:**
Pack indices to 3 bits/element (0.375 bytes) instead of BFP4's ~0.5 bytes.
Requires custom pack/unpack kernels + ROW_MAJOR scatter.
