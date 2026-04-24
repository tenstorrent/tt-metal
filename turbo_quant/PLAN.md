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
| **Qualitative quality** | — | Correct output at all seq lengths 128–131072 (31 prompts) |
| **Top-1 token accuracy** (T3K) | 96.7% | **72.5%** |
| **Top-5 token accuracy** (T3K) | 99.8% | **88.2%** |
| **Cosine vs CPU ref** | — | > 0.999 (synthetic SDPA test at all seqlens) |
| **MSE vs float32 CPU** | — | 0.034 (matches paper bound) |
| **Max context** | 128K | **128K** |

Verified 2026-04-14: BFP4 paged cache + standard SDPA decode, flat 37.1–37.2 ms/tok
from seq=128 to seq=131072. Pre-rescaled centroid×norm values stored as BFP4 in paged
`layer_past`, fed directly to `scaled_dot_product_attention_decode` which natively
accepts BFP4 inputs. No custom SDPA kernel needed for this path.

**Accuracy caveat (2026-04-20):** Rigorous token-accuracy benchmarking revealed
that BFP4 cache storage itself causes ~25% top-1 accuracy loss vs BFP8 baseline.
TurboQuant doesn't mitigate this — BFP4's shared-exponent compression dominates
the error. Qualitative tests (31 prompts, all factually correct) passed because
top-5 stays at 88-90%, enough for coherent output, but top-1 matching drops
substantially. See "Token Accuracy Benchmark" section below for full analysis.

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

### Token Accuracy Benchmark — BFP4 Storage is the Culprit (2026-04-20)

Rigorous token accuracy on the 1024-token reference corpus
(`models/tt_transformers/tests/reference_outputs/Llama-3.1-8B-Instruct.refpt`)
reveals BFP4 cache itself is lossy — **TurboQuant doesn't mitigate it**.

| Mode | Top-1 | Top-5 | KV memory | Latency |
|------|-------|-------|-----------|---------|
| **Baseline BFP8** | **96.7%** | 99.8% | 1× | 14.1 ms |
| TQ 3-bit + BFP8 cache | 89.1% | 100% | **1× (same!)** | 18.9 ms |
| Baseline BFP4 (no TQ) | 71.3% | 90.1% | 0.5× | 14.1 ms |
| TQ 3-bit + BFP4 cache | 72.5% | 88.2% | 0.5× | 18.9 ms |
| TQ 4-bit + BFP4 cache | 74.3% | 89.1% | 0.5× | 20.1 ms |

**TQ + BFP4 (72.5%) ≈ BFP4 alone (71.3%).** The BFP4 shared-exponent compression
(32-element blocks share 1 exponent) dominates the error — moving from 3-bit to
4-bit TQ barely helps (72.5% → 74.3%). TurboQuant's rotation + centroid approach
is being wasted when the storage format is BFP4.

**TQ + BFP8 is strictly worse than baseline BFP8:**
- Same memory (1 byte/elem, no savings)
- +4.8 ms latency overhead (+34%)
- -7.6% top-1 accuracy (96.7% → 89.1%)

No production use case where TQ + BFP8 wins.

**Qualitative vs Quantitative:** Our 31-prompt test showed factually correct
outputs for all TQ configs, matching the 82-88% top-5 numbers. Top-1 matches
baseline less often, but top-5 remains reasonable, which is enough for coherent
generation. But for rigorous accuracy (teacher-forced matching against a reference),
there's a significant gap.

**Implications:**
- Current "TQ BFP4 paged" path = half memory but ~25% top-1 accuracy loss
- TQ's design assumes high-precision storage (e.g. BF16); BFP4 breaks that assumption
- To recover accuracy, either use higher-precision cache (BFP8 or BF16) or accept
  the quality/memory tradeoff

### The Real Issue: Pre-Rescaled ≠ Paper's TQ Design

The low accuracy isn't inherent to TurboQuant — **our implementation took a shortcut
that breaks TQ's design guarantees**.

**Paper's design (Pure TQ / "Full Dequant"):**
- Store: 3-bit **integer index** (0..7) + BF16 **norm** (one float per vector)
- Memory: ~0.4 bytes/elem (3 bits/elem + 4 bytes per 128-elem vector)
- At read time: `centroid[idx] × norm` reconstructed with full precision
- **Accuracy preserved** — indices 0..7 are exact in any format; norm is BF16

**Our shortcut ("Pre-Rescaled BFP4"):**
- Compute `centroid[idx] × norm` at quantize time, store the product as BFP4
- Memory: ~0.5 bytes/elem (BFP4 storage)
- Storing continuous floats in BFP4 is lossy (shared-exponent per 32-elem block)
- **Accuracy lost** — BFP4's shared-exponent loses precision on pre-rescaled values

The paper's claims (MSE=0.034, cosine>0.999) apply to the **indices-plus-norm**
representation, NOT to pre-rescaled floats stored in BFP4. We conflated the two.

### Next Step: Revive Full Dequant as Main Path

Full Dequant (BFP4 indices + BF16 norms, fused SDPA with centroid reconstruction)
is the **correct TQ implementation**. We had this working previously but deprecated
it due to latency (15-35× slower than std SDPA). With the accuracy data we now have,
the tradeoff is worth revisiting:

- Full Dequant: ~0.4 bytes/elem, expected ~95%+ top-1 accuracy, slower SDPA
- Pre-Rescaled: ~0.5 bytes/elem, 72% top-1 accuracy, fast SDPA

For production, **correctness matters more than raw latency**. See Section 6 for
implementation plan.

### T3K Max Batch at Long Context (2026-04-20)

**The key serving benefit of KV compression:** TQ fits 2× the batch of baseline
at long sequence lengths where KV cache dominates DRAM usage.

| seqlen | TQ BFP4 max batch | Baseline BFP8 max batch | TQ throughput @ max | TQ advantage |
|--------|-------------------|-------------------------|--------------------:|--------------|
| 8K | 32+ (KV < 1GB) | 32+ | 2,222 tok/s | — |
| 32K | 32+ | 32+ | 2,217 tok/s | — |
| **64K** | **32** | 16 | **2,242 tok/s** | **2×** |
| **128K** | **16** | 8 | **1,136 tok/s** | **2×** |

Beyond max batch, baseline OOMs while TQ still runs. Baseline 64K×32 and 128K×16
both fail with DRAM OOM; TQ succeeds at the same config.

**Required fix:** Override `KV_CACHE` precision to BFP4 at **model init time**
(not just after init). Previously the model allocated BFP8 paged cache first,
limiting `max_num_blocks` to BFP8's memory footprint. See `make_optimizations`
in eval_e2e.py.

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

| | Latency | KV Memory | Top-1 Acc | Max Context | Status |
|--|---------|-----------|-----------|-------------|--------|
| Baseline BFP8 | 1× | 1× | **96.7%** | 128K | Reference |
| TQ BFP4 Paged (pre-rescaled) | 1× | 0.5× | 72% | 128K | Works but lossy (2026-04-20) |
| **TQ Full Dequant (indices + norms)** | **TBD** | **~0.4×** | **TBD (expect ~95%+)** | TBD | **NEW MAIN TARGET** (see below) |

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

### NEW MAIN TARGET: Full Dequant (BFP4 indices + BF16 norms) — REVIVED (2026-04-20)

After the accuracy benchmark revealed the pre-rescaled path loses ~25% top-1
accuracy due to BFP4 shared-exponent compression, we're reviving the Full Dequant
path as the correct implementation of TurboQuant. This is the paper's original
design: store quantization indices + per-vector norms separately, reconstruct at
read time. Previously deprecated for latency; now justified by accuracy data.

**What exists already:**
- Custom fused SDPA kernel at `ttnn/cpp/ttnn/operations/experimental/turbo_quant/sdpa/`
  - Device op + program factory + reader + compute + writer (~925 lines C++)
  - Reads BFP4 indices + BF16 norms, does centroid gather + norm multiply in compute
  - Python binding: `ttnn.experimental.turbo_quant_sdpa_decode`
- TTNNTurboQuantCache with `memory_efficient=True` stores BFP4 indices + BF16 norms
- Synthetic tests pass with cosine > 0.999

**Known limitations:**
- Works up to 2K seq; pre-fills full BF16 cache in L1 CBs → OOMs at 4K+
- Uses interleaved tensors (not paged) — separate `k_indices_dev` / `k_norms_dev`
- 15-35× slower than std SDPA (at seqs that fit) due to centroid gather overhead
- Untested on T3K / multi-device

**Implementation Plan:**

**1. Chunked online softmax in fused kernel** (critical, unblocks >2K context) — **DONE (2026-04-24)**

- Modified `sdpa_tq_decode.cpp` to stream K/V chunks instead of pre-filling
- CBs shrunk from `k_num_chunks` to 2 (double-buffer) → no OOM at 4K+
- New per-chunk dequant helpers (`dequant_k_chunk`, `dequant_v_chunk`) that
  match tile layout (transposed K, natural V) for matmul_blocks compatibility
- Interleaved: dequant one K/V chunk → matmul Q×K^T → softmax → V matmul → accumulate

**Multi-chunk bug fix (commit `3da187312b0`, 2026-04-24):**
Root cause was a PACK→UNPACK visibility gap on the ping-pong softmax-state CBs
(cb_max_A/B, cb_sum_A/B, cb_out_im_A/B). PACK writes these via pack_tile+push_back
in iteration N, but when UNPACK reads them as prev_max/sum/out in iteration N+1,
the L1 data hasn't fully settled despite cb_push_back's TTI_STALLWAIT.

Discovered via bisection: adding `DPRINT_UNPACK(DPRINT << TSLICE(...))` at
end-of-iteration fixed it up to 8 chunks. Translated to non-DPRINT form as
`sync_unpack_cb_read()` — a noinline function doing 64 volatile uint32 L1 reads
on UNPACK thread, wrapped in `UNPACK(...)` macro. Reading only 8 words fails;
64 is the empirical minimum.

**Validation (all PASS, cos ≥ 0.9983):**
| seqlen | k_chunks | cos |
|--------|---------:|----:|
| 128    |     1    | 0.9996 |
| 2048   |    16    | 0.9997 |
| 8192   |    64    | 0.9996 |
| 32768  |   256    | 0.9995 |
| 131072 |  1024    | 0.9983 |

**2. Paged cache support for indices + norms**
- Current `TTNNTurboQuantCache` uses `[1, max_seq, n_heads, head_dim]` interleaved
- Refactor to paged: `[max_num_blocks, n_heads, block_size, head_dim]` for indices
- Norms are tiny (per-vector, not per-element) — could stay non-paged or be paged
- Update reader kernel to take page_table and do paged reads
- Effort: 1-2 days

**3. Multi-device / T3K support**
- Replicate centroids across devices, shard indices/norms by heads
- Fix `fused_sdpa_decode` to use mesh_composer/mesh_mapper where needed
- Effort: 1 day

**4. Integration into eval_e2e.py / eval_e2e_prefill.py**
- Add `--tq-full-dequant` flag (mutually exclusive with `--bfp4-cache`)
- Allocate TTNNTurboQuantCache with `memory_efficient=True`
- Route attention decode through `fused_sdpa_decode` (not std SDPA)
- Effort: 1 day

**5. Accuracy + latency validation**
- Run token accuracy benchmark → expect >95% top-1 (vs pre-rescaled's 72%)
- Measure decode latency → target <30ms on T3K (2× baseline acceptable given accuracy)
- Run seqlen + batch sweeps to confirm long context + multi-batch works
- Effort: 1-2 days

**6. Optional: optimize centroid gather** (only if latency is unacceptable)
- Current: ~50 SFPU ops/tile via conditional cascades
- Alternative: LUT-style gather via unpacker + matmul
- Alternative: do centroid×norm expansion post-softmax instead of on K/V read
- Effort: 3-5 days if needed

**Total effort estimate: ~1-2 weeks.**

**Alternative approach (simpler but higher risk):**
Modify the standard SDPA decode reader to support TQ dequant on-read. Reuse std
SDPA's proven chunked online softmax. Only the reader needs TQ logic. Risk:
touching shared infrastructure used by other models.

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

**~~End-to-end demo with real prefill + paged decode~~ DONE (2026-04-20):**

`eval_e2e_prefill.py --bfp4-cache` now runs real prefill + paged decode on device,
no teacher forcing. Correct output across prompts.

Fix was passing `page_table=` to `prepare_inputs_prefill` and `kv_cache=` to
`ttnn_prefill_forward` (previously omitted). Migration function now handles paged
layer_past layout `[max_num_blocks, n_kv_heads, block_size, head_dim]`.

| Config | Prefill+decode (ms/tok) | eval_e2e.py teacher-forced | Gap |
|--------|-------------------------|---------------------------|-----|
| Single device | 42.8 | 37.1 | +5.7ms |
| T3K 8-device | 18.7 | 14.2 | +4.5ms |

**Residual ~5ms gap vs teacher-forced path** — possibly program cache / memory
fragmentation after prefill. Non-blocking but worth investigating. Not TQ-specific
(would likely affect any paged-prefill → paged-decode flow).

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
