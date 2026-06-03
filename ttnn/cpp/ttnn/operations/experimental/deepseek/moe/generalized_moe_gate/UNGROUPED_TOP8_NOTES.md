# Ungrouped global top-8 for generalized_moe_gate — implementation notes

Goal: change the gate so that instead of DeepSeek's grouped routing (top-2 sum → top-4
groups → top-8 of 128), it returns the **true global top-8 of all 256 experts** (ranked by
bias-corrected score, returning the normalized non-bias score). The Python golden is already
updated to this semantics in `models/demos/deepseek_v3/tt/generalized_moe_gate/op.py`, and
`models/demos/deepseek_v3/tests/test_generalized_moe_gate.py` validates against it.

This is purely a **device-kernel** change. Everything below is WH B0 specific.

## Confirmed SFPU geometry (LLK expert + assembly.yaml + on-device dump)

- A WH B0 LREG is **4 rows × 8 columns = 32 lanes**. SFPSWAP modes are defined over rows 0..3
  (`tt_metal/tt-llk/tt_llk_wormhole_b0/instructions/assembly.yaml:2680-2700`).
- The **8 SFPU columns are 8 independent SIMD groups**. One SFPLOAD loads the same 4 DEST rows
  for *all 8 groups at once* into one LREG (group g → SFPU column g).
- A per-group **sorted top-8 is held as `LREG0`(rows 0-3) + `LREG1`(rows 4-7)** — i.e. the
  `(offset, offset+4)` pair — and at rest occupies **one face column** (rows 0-7). The on-device
  dump after `sum_top2` confirmed: group g's top-8 (descending) sits at face **col 2g, rows 0-7**
  (value in `bias` region, idx in `indices` LO16, score in `scores` HI16).
- `bitonic_top8_ph3_st4_to_1` merges **exactly two** sorted-8 runs at a time: run A in
  `LREG0/LREG1`, run B in `LREG2/LREG3`, with one run pre-reversed (`reverse_sort_order`) so the
  pair is bitonic. Indices ride along in `LREG4-7` with index-tracking (`SFPCONFIG 0x4`) on.
- DEST regions (single-face op): `scores=0, indices=64, bias=128, interm=192` (units = DEST rows,
  64 rows = 1 tile). `dst_tile_offset=64`.

## Why the v0 attempt was wrong (post-mortem)

v0 (`_gmg_merge4_top8` / `_generalized_moe_gate_top8_ungrouped`, behind `GMG_UNGROUPED_TOP8`)
tried to "load 4 columns → merge → repeat." Two fatal mistakes:

1. The 8 groups are **not** in 8 separately-loadable columns — they are in the **8 SFPU columns of
   one `LREG0/LREG1` pair**. Getting the global top-8 means reducing **across SFPU columns**, which
   needs `SFPTRANSP`/`SFPSHFT` (rotate columns into rows), **not** the run-merge primitive (which
   merges runs living in *different LREGs, same columns*).
2. `bitonic_top8_ph3_st4_to_1` merges only 2 runs, not 4.

Symptom this produced: duplicated indices (e.g. `239,239,239`) and results biased to a few groups —
classic cross-column-not-reduced + broadcast artifacts.

## The pipeline today (grouped, working — `GMG_UNGROUPED_TOP8` off)

`copy/add → sum_top2 → step0(transpose) → sort_top4_groups → step1(transpose) → top8 → step2`
- `step0`/`step1` exist for the **group sort** (bring the 8 group-sums onto a sortable axis, then
  back). `top8`'s own comment "combine 4 groups of 8 → 2 groups of 8 → 1" shows it already does a
  2-level reduction for the 4 selected groups.

## Recommended implementation — Path A′ (preferred)

Reuse the generic, dist-parameterized bitonic top-k that already merges **across the width**:
`tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_topk.h`
- `_bitonic_topk_phases_steps(...)` — sort
- `_bitonic_topk_merge(m_iter, k)` — merge sorted runs at column distance `dist = k<<m_iter`
- `_bitonic_topk_rebuild(idir, m_iter, k, logk, skip_second)` — rebuild after merge
- `bitonic_topk_load16(dist0,dist1)` / `store16` / `load8(offset,dist)` / `store8` — the
  dist-aware loads (these are the parameterized versions of the deepseek `_single_face` helpers).

Plan:
1. After `sum_top2`, the 64 candidates (8 groups × top-8) are in the deepseek DEST regions.
2. Either run the generic `merge`/`rebuild` over those 64 to extract the global top-8 (k=8), or
   reformat into the generic layout (`dst_indices_offset=128`, full-width) and run the generic
   sort+merge+rebuild, then reformat back.
3. Reuse the deepseek `top8` **normalization tail only** (recip + scale + broadcast-multiply,
   `ckernel_sfpu_deepseek_moe_gate_topk_single_face.h:442-479`).
4. Orchestration becomes `copy/add → sum_top2 → <generic merge to top-8> → step2`
   (drop step0/sort_top4/step1).

Open item the expert could NOT fully pin from code: the exact element permutation `step1` performs.
Not needed for Path A′ if the generic merge consumes the post-`sum_top2` layout via its own
`dist`-aware loads — verify on HW with the dump probe.

## Alternative — Path B′

Hand-write the cross-column reduction in the deepseek single-face style: `SFPTRANSP` the 8 group
columns into rows, bitonic-merge down to top-8, then normalize. Stays in the familiar layout but
requires authoring the transpose/swap sequence (the part that needs HW iteration).

## How to iterate (the loop that works)

- Toggle in `device/kernels/generalized_moe_gate_kernel.cpp`:
  `GMG_UNGROUPED_TOP8` (real path) / `GMG_DUMP_AFTER_SUM_TOP2` / `GMG_DUMP_AFTER_STEP1` (probes).
- The dump probe packs `bias`(dst tile 2)→output and `indices`(tile 1)→output_indices; read it via
  `test_generalized_moe_gate.py::test_dump_sum_top2_layout` (rigged inputs: idx=0..255, bias=j+g).
- Validate the real path with `test_generalized_moe_gate.py::test_generalized_moe_gate`.
- Only WH (`tt_llk_wormhole_b0`) variants were touched; the BH tree needs the same edits for a BH build.

## Key references

- Generic topk: `tt_metal/tt-llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_topk.h`
  (load8 :32, load16 :67, store8 :50, ph3_st4 :141, phases_steps :397, merge :571, rebuild :623)
- Deepseek single-face: `.../generalized_moe_gate/device/kernel_includes/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_generalized_moe_gate_topk_single_face.h`
  (store8_even_cols :128, ph3_st4 :242, reverse_sort_order :320, sum_top2 :327, top8 :394, normalize tail :442)
- Transpose dest: `.../llk_lib/llk_math_generalized_moe_gate_transpose_dest_single_face.h:74` (step1)
- Pipeline order: `.../kernel_includes/tt_metal/include/compute_kernel_api/generalized_moe_gate.h:75`
- assembly.yaml ISA notes: `tt_metal/tt-llk/tt_llk_wormhole_b0/instructions/assembly.yaml`
  (SFPSWAP rows :2680, SFPLOAD :2116, SFPTRANSP :2511, MOVD2B/MOVB2D/TRNSPSRCB :316)
