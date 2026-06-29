# Verification Report: scaled_dot_product_attention

## Code Review

### Fixes Applied

1. **Missing running-max step in online softmax (CRITICAL correctness bug)**
   - **File**: `kernels/scaled_dot_product_attention_compute.cpp`
   - **Problem**: The online softmax recurrence requires `m_new = max(m_old, m_blk)` (the running max), but the kernel used `m_blk` directly as `m_new`. This caused `alpha = exp(m_old - m_blk)` instead of `exp(m_old - m_new)`. When a KV-block had all -inf scores (fully masked by causal mask), `m_blk = -inf` and `alpha = exp(m_old - (-inf)) = inf`, corrupting the running output O and sum l. This produced PCC ~0.37 on masked multi-block cases.
   - **Fix**: Added Phase 4b between row-max (Phase 4) and alpha computation (Phase 5): an `eltwise_chain` that loads `cb_max_old` and `cb_max_new` into DST (D0, D1), computes `BinaryMax<D0, D1, D0>` = `max(m_old, m_blk)`, and packs the running max back to `cb_max_new`. Subsequent phases (5, 8, 13) now use the correct running max. Also added `#include "ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp"` for `BinaryMax`.
   - **Impact**: Fixed 4 acceptance test failures (custom_mask + multi-KV-block shapes) and 4 golden suite numerical-bug failures.

2. **Partial last Q-block / KV-block out-of-bounds read (CRITICAL correctness bug)**
   - **File**: `scaled_dot_product_attention_program_descriptor.py`
   - **Problem**: `B_q_t = min(MAX_B_Q_T, S_q_tiles)` did not ensure `B_q_t` divides `S_q_tiles`. When `S_q_tiles % B_q_t != 0` (e.g. S_q=192, S_q_tiles=6, B_q_t=4 → last Q-block has 2 tiles but the reader pushed 4), the reader read out-of-bounds tiles past the Q tensor. The compute kernel also processed garbage tiles, producing inf/nan output.
   - **Fix**: Added a divisor-reduction loop: `while S_q_tiles % B_q_t != 0 and B_q_t > 1: B_q_t -= 1` (and same for `B_kv_t`). This ensures every Q-block and KV-block is full — no partial blocks, no out-of-bounds reads.
   - **Impact**: Fixed 4 golden suite numerical-bug failures on shape (2,3,192,96) and (1,1,192,96).

3. **Added `eltwise_binary_sfpu.hpp` include**
   - **File**: `kernels/scaled_dot_product_attention_compute.cpp`
   - **Problem**: `BinaryMax` is defined in `eltwise_binary_sfpu.hpp`, not in `eltwise_math.hpp` or `eltwise_chain.hpp`.
   - **Fix**: Added the include.

### Registry Conformance Notes

- The `INPUT_TAGGERS` dict has three taggers, all with the correct `(inputs, axes)` signature.
- `SUPPORTED` covers all gated axes: dtype, fp32_dest_acc_en, layout, alignment, attention_kind, kv_heads_mode, mask_mode, scale_mode.
- `EXCLUSIONS` is empty (the causal+cross exclusion is commented out since causal is not in SUPPORTED yet).
- `validate()` checks per-axis SUPPORTED first, then cell-level EXCLUSIONS. Both raise from `ttnn.operations._op_contract`. Correct order.
- The public entry point calls `validate()` as its first line. Confirmed.
- The op file does NOT declare `INVALID`. Confirmed.

### Design Conformance

- **Algorithm**: Flash Attention v2 online softmax recurrence — implemented as designed (QK^T → scale → mask → row-max → running-max → alpha → rescale O,l → sub → exp → row-sum → update l → PV matmul → accumulate O → update m). The running-max step (Phase 4b) was missing from the original implementation but is part of the correct algorithm — now fixed.
- **Data pipeline topology**: DRAM → reader → L1 CBs → compute → L1 output CB → writer → DRAM. Matches design.
- **Parallelization**: Embarrassingly parallel across (B, H) pairs via `split_work_to_cores`. Matches design.
- **Inter-core communication**: None — each core processes independently. Matches design.

### Helper Usage

- All compute phases use helpers (`matmul_block`, `mul`, `add`, `sub`, `copy`, `unary`, `reduce`, `eltwise_chain`). No raw LLK calls in compute.
- Raw API is only used for reader-side constant fills (scale factor, reduce scalers) — correctly justified in the design.
- Reader uses `calculate_and_prepare_reduce_scaler` from `reduce_helpers_dataflow.hpp`. Correct.
- Reader uses `TensorAccessor` (not deprecated `InterleavedAddrGen`). Correct.
- All kernels use `void kernel_main() { }` pattern and `api/dataflow/dataflow_api.h` include path. Correct.

### CB Sync Verification

- push count = wait count for every CB, verified per phase.
- The scores re-use pattern (Phase 4 `WaitUpfrontNoPop` → Phase 8 `Streaming`) is correctly implemented.
- The Q retention pattern (`WaitAndRetainOnLastBlock` on in0) keeps Q in cb_q across KV-blocks. The compute drains cb_q between Q-blocks.
- The `cb_alpha` drain (`cb_pop_front(cb_alpha, B_q_t)`) correctly drains the HeldBulk alpha tiles after both rescale phases consume them.

## Registry Conformance

- Confirmed: INPUT_TAGGERS, SUPPORTED, EXCLUSIONS, validate() all present and correctly wired in the op file. Confirmed op file does NOT declare INVALID (it's a test-suite concept).
- No auto-fixes applied to SUPPORTED based on XPASS evidence (xpass_drift = 0).
- INVALID audit (in `eval/golden_tests/scaled_dot_product_attention/feature_spec.py`): `INVALID = []` — correct for SDPA (TILE-only, no ROW_MAJOR in TARGET, so the canonical bf8b+ROW_MAJOR rule is vacuous). No cross-tensor-axis entries. No missing canonical cells.

## Precision Baseline

| Shape | PCC | Max Abs Err | Mean Abs Err | Relative RMS Err |
|-------|-----|-------------|--------------|------------------|
| (1,1,32,32) | 0.999998 | 0.005304 | 0.000520 | 0.002584 |
| (1,1,128,64) | 0.999998 | 0.002397 | 0.000226 | 0.002475 |
| (1,1,256,64) | 0.999997 | 0.003700 | 0.000173 | 0.002663 |
| (1,8,128,64) | 0.999998 | 0.004569 | 0.000245 | 0.002527 |

**Assessment**: Excellent precision — PCC ≥ 0.999997 across all tested shapes. The online softmax with fp32 DEST accumulation produces results matching torch SDPA to within bf16 rounding. Max abs error ≤ 0.005, relative RMS ≤ 0.003.

**Recommended tolerances**: PCC >= 0.995, rtol=0.05, atol=0.01

## Verifier CLI Summary

- supported_pass: 200
- xfail_expected: 2440
- invalid_skipped: 0
- supported_fail: 8 (all OOM on D=512/D=1024 head dims)
- xpass_drift: 0
- xfail_wrong_mode: 0
- xfail_other: 0
- no_axes_found: 119 (regression/loose tests without axes)

The 8 supported_fail cells are all OOM on large head dims (D=512, D_t=16; D=1024, D_t=32). The CBs for cb_o, cb_o_accum, cb_scores, cb_exp_scores scale with D_t, exceeding the 1.5 MB L1 budget. These are `/memory-budget-metal` refinement candidates.

## Recommendations

- The OOM on D=512/D=1024 is the primary Phase 0 gap. The cb_o and cb_o_accum CBs are each `B_q_t * D_t` pages × fp32_tile_size. For D=1024: 4 * 32 = 128 tiles × 4 KB = 512 KB each. Combined with cb_scores (16 tiles × 4KB) and cb_exp_scores (same), total exceeds 1.5 MB. The `/memory-budget-metal` skill's K-blocking/chunking pattern applies.
- The `work_b[16]` / `work_h[16]` arrays in the reader and writer are hard-coded to 16 entries. With B*H_q > 16, a single core assigned >16 work units would overflow. Currently safe (max B*H_q in INPUTS is 8*4=32, split across 32+ cores → 1 per core), but should be noted for future multi-core distribution changes.
- The reader reads tiles one at a time with `noc_async_read_barrier()` per tile. This is correct but slow — a future optimization could batch reads and use a single barrier per block.
