# ttnn.bringup.moe_ag_route_plan, moe_ag_local_reduce, moe_ag_sum_rows_tiled, moe_ag_add_rows, moe_ag_untilize_active, moe_ag_untilize_x

## What it does
New op set (no TTNN source on this tree; from `mstaletovic/mimo-v2-dp` @ `b45a923d73c`): the device programs of the
all-gather MoE block (MiMo-V2's MoeAgBlock), i.e. MoE without dispatch / combine. Every chip gathers the tokens and
their top-k, runs its own local experts on the tokens routed to them, and reduces the weighted expert outputs; the
cross-chip sum is a reduce-scatter (or the fused send-back below). Bindings: `moe_ag_nanobind.cpp`.
- `moe_ag_route_plan`: gathered top-k ids [.., T, K] uint16 RM + this chip's local-slot map [.., 1, NG] uint32 ->
  `[counts [1, NG], regions [1, NG], token_index [1, num_rows], y_slot [1, T K]]` uint32 RM DRAM: tokens per global
  expert, each local expert's first flat row (32-row aligned), flat row -> gathered token, per (token, k) the flat row
  of its expert output (or 0xFFFFFFFF). One program on an 8 x 8 core rectangle. Feeds `flat_routed_expert` in indexed
  mode.
- `moe_ag_local_reduce`: per token, sum over this chip's local experts of w[g, k] * y[y_slot[g, k]] (fp32 DEST), whole
  worker grid. Phase 0: one pass over T tokens (row-major, or `tiled` bf16 tiles for a reduce-scatter, or `split`
  into own / other halves). Phases 1 / 2 (two mesh rows, T == 2 S): fused send-back; phase 1 writes the other row's
  tokens' partial (to be all-gathered over axis 0), phase 2 adds this row's tokens' partial and the peer's gathered
  phase-1 partial.
- `moe_ag_sum_rows_tiled`: out[r] = sum of num_blocks row blocks of a bf16 RM source -> bf16 TILE (sum + tilize in
  one pass).
- `moe_ag_add_rows`: out[i] = a[a_offset + i] + b[b_offset + i], bf16 RM (b_offset optionally per device from
  chip_info word 1).
- `moe_ag_untilize_active`: bfp8 TILE y -> bf16 RM, only the tile rows that hold tokens (counts / regions read on
  device).
- `moe_ag_untilize_x`: bf16 TILE x [S, H] -> RM [1, 1, S H / 1024, 1024] (2 KB pages, so an indexed expert's token
  reads spread over H / 1024 DRAM banks).

## Consider it instead of
- `ttnn.bringup.dispatch` + per-expert compute + `ttnn.bringup.combine` (the dispatch / combine MoE): the all-gather
  design replaces the token all-to-all with an all-gather of x / top-k and a reduce-scatter of the weighted sums.
  On GLM-5.3 (LoudBox 2x4) the ag block measured 4.05 ms vs unified 9.57 ms per layer
  (`models/demos/glm53_flash_d_p_lb/bringup/BREADCRUMBS.md`, "Whole model on the LoudBox").
- Host-side or `ttnn.topk` / `ttnn.sort` / scatter compositions to build expert row maps (route plan).
- `ttnn.matmul` / `ttnn.mul` + `ttnn.sum` over k for the weighted combine of expert outputs (local reduce).
- `ttnn.add` + `ttnn.tilize` / `ttnn.untilize` pairs (sum_rows_tiled, add_rows, untilize_*).
Only useful together with `flat_routed_expert` (indexed mode) and the fabric collectives; the ops assume its flat
row layout.

## Options
Realistic (use by default): every option below.

`moe_ag_route_plan(topk_indices, local_slot_map, experts_per_chip, num_rows, *, outputs=None)`
- `experts_per_chip`: required, <= 64.
- `num_rows`: required, multiple of 32, >= roundup(P, 32) + 32 (min(P, experts_per_chip) - 1), P = T min(K,
  experts_per_chip). K <= 32.
- `outputs=None`: preallocated (counts, regions, token_index, y_slot).

`moe_ag_local_reduce(y, y_slot, weights, chip_info, chunk_size_per_chip, *, phase=0, split=False, tiled=False, peer=None, outputs=None)`
- `chip_info`: [.., 1, 16] uint32 per device: word 0 mesh row, 1 other row's block start, 2 this row's.
- `chunk_size_per_chip` (S): required; T a multiple of S.
- `phase=0`: 0 one pass; 1 / 2 the fused send-back (needs T == 2 S).
- `split=False`: phase 0 only, outputs [own, other].
- `tiled=False`: bf16 TILE output; phase 0 (T % 32 == 0) or phase 2 (S % 32 == 0); exclusive with split.
- `peer=None`: required by (only) phase 2.
- `outputs=None`: preallocated outputs.

`moe_ag_sum_rows_tiled(src, num_rows, num_blocks, block_stride, *, output=None)`: num_rows multiple of 32,
num_blocks >= 2.

`moe_ag_add_rows(a, b, chip_info, num_rows, *, a_offset=0, b_offset=0, info_offset=False, output=None)`:
`info_offset=True` takes b_offset per device from chip_info word 1.

`moe_ag_untilize_active(y, counts, regions, local_slot_map, experts_per_chip, *, tiles_per_block=32, output=None)`

`moe_ag_untilize_x(x, *, output=None)`: H a multiple of 1024.

No fork options: CHANGELOG.md lists no changes (mechanical import only).

Precision only: per-step checks or a measured end-to-end gap (rule 7): none.

## Measured
Device ns per call, one BH p150 (QuietBox chip), median of 3, 2026-09-29; perf baselines in
`tests/unit/test_moe_ag_ops.py` (`_PERF_EXPECTED_NS`, margin 5%). Shapes (H, K, NG, experts/chip, T, S): mimo
(4096, 8, 256, 64, 1280, 640), galaxy (4096, 8, 256, 8, 5120, 640), k2 (7168, 8, 384, 48, 2048, 1024), small
(2048, 6, 160, 20, 256, 128):

| case | route_plan | local_reduce tiled | sum_rows_tiled | untilize_x | local_reduce phase 2 |
|---|---|---|---|---|---|
| mimo | 29,758 | 160,110 | 61,755 | 46,583 | 75,089 |
| galaxy | 47,881 | 278,859 | 61,219 | 46,323 | - |
| k2 | 34,776 | 288,256 | 129,889 | 97,264 | 141,866 |
| small | 15,256 | 35,477 | 18,865 | 13,154 | 15,734 |

In the model (GLM-5.3, LoudBox 2x4 p150b, chunk 5120 at 51200): local reduce 0.49 ms of the ag block's 4.05 ms
(`models/demos/glm53_flash_d_p_lb/bringup/BREADCRUMBS.md`, "Whole model on the LoudBox"). Layer 4 (chunk 2048, bfp4):
ag split call 2.34 ms vs unified 5.56 ms; vs golden PCC 0.9883 (unified 0.9884) (same file, "MiMo all-gather MoE ops
on GLM"). Accuracy gate of the unit test: local reduce PCC > 0.9999 and rel < 1e-2 vs a host reference
(`tests/unit/test_moe_ag_ops.py`); route plan / untilizes bit-exact.

## Limits
- bf16 / uint16 / uint32 ROW_MAJOR DRAM inputs as listed in Options; `untilize_active` takes bfp8 TILE y.
- H % 1024 == 0 (local_reduce, sum_rows_tiled, untilize_x); experts_per_chip <= 64; K <= 32.
- Fused send-back (phase 1 / 2, split) only for exactly two mesh rows (T == 2 S); more rows use phase 0 tiled +
  `fabric_reduce_scatter`.
- glm53_flash_d_p's experts_ag builds only at <= 64 experts per chip and never runs on a 2x2
  (`models/demos/deepseek_v41_flash_d_p/bringup/known_issues.md`, DS41-63).
- `moe_ag_sum_rows_tiled`, `moe_ag_add_rows`, `moe_ag_untilize_active`, `moe_ag_untilize_x` have no model caller on
  this tree (unit tests only).
- Unit tests are single-device; the multi-chip path is only exercised through glm53_flash_d_p.

## How to call
```python
counts, regions, token_index, y_slot = ttnn.bringup.moe_ag_route_plan(gidx, lmap, experts_per_chip, num_rows)
(part,) = ttnn.bringup.moe_ag_local_reduce(y, y_slot, gw, chip_info, S, phase=0, tiled=True)
```
(`models/demos/glm53_flash_d_p/tt/experts_ag.py`). Tests: `tests/unit/test_moe_ag_ops.py` (correctness vs host
references, program cache, determinism, validation, perf). `tests/source.yaml` is empty (no upstream tests).

## Used by
- glm53_flash_d_p (`tt/experts_ag.py`, GLM_EXPERTS_MODE=ag; the whole model on the LoudBox via glm53_flash_d_p_lb):
  `moe_ag_route_plan`, `moe_ag_local_reduce` (phase 0 tiled; phases 1 / 2 with peer, tiled phase 2).

## History
CHANGELOG.md
