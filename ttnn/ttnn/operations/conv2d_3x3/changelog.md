# Changelog: conv2d_3x3

## Phase 0 — Core Implementation
- **Date**: 2026-11-17
- **What was done**: Initial implementation via the incremental pipeline (planner -> implementer -> verifier). Regime R0 (interleaved-slab), built as designed:
  - M is split row-major over the live grid (11x10 on the measured p150b), one M-block per core on every acceptance shape.
  - K is walked per kh block (`bk = tensor_kkh_tiles`; the deep-K case solves down to 24 of 192 tiles with bf16 weights).
  - Per-core weight stream, fp32 packer-L1 accumulation, row-broadcast bias.
  - CB set, sizes and data-movement budget match `l1_ledger.md`.
- **SUPPORTED at Phase 0**:
  - dtype=[bfloat16], weight_dtype=[bfloat16, bfloat8_b], layout=[ROW_MAJOR], memory_layout=[INTERLEAVED], fp32_dest_acc_en=[True].
  - bias_mode=[with_bias, no_bias], activation=[none], channel_alignment=[tile_aligned], hw_alignment=[hw_aligned].
  - stride_mode=[unit, strided], dilation_mode=[unit, dilated], padding_mode=[valid, same].
- **Accuracy achieved** (HiFi4, fp32 DEST, with bias; 4 shapes via `test_conv2d_3x3_precision_baseline.py`):
  - bf16 weights: PCC >= 0.9999985, max_abs_err <= 2.04 (K = 18432), rel_rms_err ~0.0017, median bf16 ULP 0.
  - bf8_b weights: PCC >= 0.99996, rel_rms_err ~0.0078.
  - The got/true ratio median is 1.0000 on every shape.
- **Golden suite at Phase 0**: 64 / 64 supported cells passing (1399 xfail_expected, 480 invalid_skipped, 0 supported_fail / xpass_drift / xfail_wrong_mode) per `golden_results/verifier_report.json`.
- **Issues encountered**:
  - Verifier drift fix: SUPPORTED under-claimed `weight_dtype=bfloat8_b`, `stride_mode=strided` and `dilation_mode=dilated`. The kernel already handled all three, and a gate-lifted probe passed 80/80 interleaved RM bf16 cells. They are promoted, taking supported_pass from 12 to 64.
  - Verifier contract fix: `validate()` now rejects a bias whose dtype differs from the weight's, because `cb_bias` pages are sized from the weight tile.
  - `c_in_unaligned` also passed the probe but stays unsupported. The reader copies DRAM page-alignment lanes without zeroing them (Refinement 1).
- **Tests added**: test_conv2d_3x3.py (+ `test_bias_dtype_contract_violation`), test_conv2d_3x3_precision_baseline.py, test_conv2d_3x3_perf.py

### Implementer notes (departures from the design text, advisory)
- The slab fill's page runs use `num_contiguous_pages` only when `contiguous_page_stride() == 1`. On an interleaved accessor a "run" is pages `p, p + num_banks, ...` (same bank, not consecutive pixels), so a contiguous slab destination can take only one page per read there.
- Self-aimed gather and zero reads use `noc_async_read(get_noc_addr(l1), ...)`, with the local_copy helpers' trid calls for fencing. Outstanding reads per trid are throttled below `NOC_MAX_TRANSACTION_ID_COUNT`.

### Measured (Blackhole p150b, HiFi4, bf16, with bias; DEVICE KERNEL DURATION)

| Shape | cores | kernel | weights stubbed | binding stage |
|-------|-------|--------|-----------------|---------------|
| 32x64x128 -> 128 | 64 | 97.7 us | 24.8 us | weight stream: 18.4 MB at ~190 GB/s (DRAM peak) |
| 16x16x64 -> 128 | 8 | 16.2 us | 10.8 us | mixed |
| 8x16x512 -> 256 | 4 | 126 us | 68.8 us | weights + compute (4 cores) |
| 8x16x2048 -> 256 | 4 | 468 us | 258 us | compute (4608 tile-MACs/core at HiFi4) + weights |
| 64x128x128 -> 128 | 86 | 134 us | 80 us | slab fill + weights |
| 128x256x64 -> 64 | 103 | 225 us | 212 us | slab fill: 835 x 128 B DRAM reads/core (halo 2.6x) |

Slab-fill classification: sourcing the same reads from local L1 takes 128x256x64 from 212 to 88 us
(weights stubbed). The fill is therefore DRAM-transaction-bound in aggregate, not RISC-issue-bound.

Lamps measured (kept at design defaults):
- P9 bank stagger (`SLAB_READ_STAGGER_PX_PER_CORE`, parked at 0): flat on every shape.
- P2 `im2col_rm_depth_rows = block_m_tiles`: flat (225 -> 218 us at best).
- P3 `core_m_tiles x2`: mixed.
  - Faster: 32x64x128 97 -> 66 us, 64x128x128 134 -> 119 us.
  - Slower: 128x256x64 225 -> 242 us, and the 4-core shapes.
  - Weight-bound shapes want R1 (weight multicast, Refinement 3), not fewer cores.
