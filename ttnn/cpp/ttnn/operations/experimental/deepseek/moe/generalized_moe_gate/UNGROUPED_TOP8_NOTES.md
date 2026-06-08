# Ungrouped global top-8 for `generalized_moe_gate` — dev log

**Goal.** Make `ttnn.experimental.deepseek.moe.generalized_moe_gate` return the **true global
top-8 over all 256 experts** (8 groups × 32), ranked by the bias-corrected score and returning the
normalized non-bias score — *not* DeepSeek's grouped routing (top-2 sum → top-4 groups → top-8 of
128). It must be a **single fused on-device op** (~µs), not the slow generic `ttnn.topk`.

**Status: DONE & validated** (2026-06-04, WH B0) against a flattened `torch.topk` golden, tie-robust
test, all `seed × {sigmoid} × {batch 1,2}` cases pass. Python golden + test:
`models/demos/deepseek_v3/{tt/generalized_moe_gate/op.py, tests/test_generalized_moe_gate.py}`.

This was a **device-kernel-only** change (WH B0). The BH tree still needs the same edits for a BH build.

---

## 1. Final working architecture

Insight: the proven 4-group merge brick (`merge4_top8`) already produces a correct sorted-8 from 4
groups. Run it **twice** and combine:

```
topA = top8(groups 0-3)      ← lay groups 0-3 out, merge
topB = top8(groups 4-7)      ← lay groups 4-7 out, merge
global top-8 = full-sort(topA ∪ topB)   ← 16 candidates, then normalize
```

Mathematically exact: the global top-8 ⊆ topA ∪ topB (each group contributes ≤ 8 to a top-8).

### Pipeline (under `GMG_UNGROUPED_TOP8`, in `compute_kernel_api/generalized_moe_gate.h`)

```
sum_top2 → step0(transpose: group g → DEST row g) → [ungrouped block] → step2(transpose) → pack
```

The ungrouped block. The SFPU merge can only reliably address **DEST rows 0-7** (see §3), so only
one half can be "in the merge slot" at a time. The FPU (`copy4rows`, a plain MOVD2B→MOVB2D 4-row
copy) stashes the idle half in **rows 8-15**, which the SFPU can't touch but the FPU can:

| step | op | effect |
|------|----|--------|
| 1 | `copy4rows<4,8>` | save groups 4-7 source: rows 4-7 → rows 8-11 |
| 2 | `step1_hi<d2b_dst=0>` + `merge4_top8<read=0, store={0,2}>` | topA = top8(groups 0-3) → cols {0,2} |
| 3 | `copy4rows<0,12>` | park topA: rows 0-3 → rows 12-15 |
| 4 | `copy4rows<8,4>` | restore groups 4-7: rows 8-11 → rows 4-7 |
| 5 | `step1_hi<d2b_dst=4>` + `merge4_top8<read=0, store={4,6}>` | topB = top8(groups 4-7) → cols {4,6} |
| 6 | `copy4rows<12,0>` | restore topA: rows 12-15 → rows 0-3 |
| 7 | `finalize_ungrouped` | full bitonic sort of the 16 candidates → global top-8 + normalize |

Each `copy4rows` uses a **disjoint SrcB scratch window** (16/20/24/28) — see §4, Bug 0.
topA lands at row-cols {0,2} (rows 0-3), topB at {4,6} (rows 4-7): row-disjoint so step 6 can place
both without overlap.

### `finalize_ungrouped` (the part that took the longest to get right)

topA{0,2} + topB{4,6} = 16 candidate values. **Do NOT** try a 2-run bitonic *merge* (`ph3_st4_to_1`
or `merge4_runs_raw`) — those expect a specific run lane-orientation that `merge4_top8`'s output does
not have, and give partial-merge or duplicate results (see §4, Bug 3). Instead, load all 16 as an
**unsorted 16-vector** (topA→LREG0/1, topB→LREG2/3; idx LO16 | score HI16 in LREG4-7, with
`SFPCONFIG(0x4,0xF,1)` index tracking) and run the **full** `bitonic_top8_ph0_to_ph3<idir=false>`.
A full sort is **orientation-independent** — it sidesteps every lane-layout subtlety. Then
`store8_even_cols_split` + the standard normalize tail.

---

## 2. DEST / SFPU layout facts used

- DEST regions (single-face op): `scores=0, indices=64, bias=128, interm=192` (units = DEST rows;
  64 rows = 1 tile; `dst_tile_offset=64`). A stored run keeps idx in `indices` (LO16) and score in
  `scores` (HI16), bias in `bias` — `merge4_top8`'s store convention (`store_lo`/`store_hi`).
- Post-`step0`: group `g` lives at DEST **row g** (rows 0-3 = groups 0-3, rows 4-7 = groups 4-7).
  `step1_hi<d2b_dst=4>` reads rows 4-7 → produces top8(groups 4-7); single-half test confirmed exact.
- A `merge4_top8` output sorted-8 occupies **two columns** {store_lo, store_hi} (LREG0→lo, LREG1→hi),
  4 values each.

---

## 3. The hard HW constraint that shaped everything

**SFPU `SFPLOAD`/`SFPSTORE` can only reliably address rows 0-7 of tiles 0-3** (offsets 0,2,4,6).
Offset ≥ 8 (and ≥ 256) wrap / read stale. So the SFPU merge is confined to rows 0-7. The FPU,
however, *can* address rows 8-15 (`MOVB2D` `b2d_base` relocation, `MOVD2B`, `TRNSPSRCB` all proven).
That asymmetry is the whole reason for the FPU `copy4rows` park dance: rows 8-15 are the only refuge
for the idle half, and only the FPU can put data there / read it back.

`TRNSPSRCB` = in-place 16×16 transpose of SrcB rows 16-31. `step1_hi` loads the 4 source rows into
SrcB 16-19 and 28-31, transposes, then `MOVB2D` reads the even transposed rows → the sortable run.

---

## 4. Bugs encountered & how they were solved

The methodology that worked throughout: **isolate with bisection diagnostics, hard-reset between
runs (`tt-smi -glx_reset`) for determinism, trust only `output[:,0,:8]` / real-test indices** (scratch
DEST is reset-dependent). The `GMG_DIAG_TOPA` / `GMG_DIAG_TOPB` isolation macros and the
`GMG_DUMP_AFTER_*` dump points (read via `test_dump_sum_top2_layout`) drove every fix.

### Bug 0 — back-to-back `copy4rows` SrcB carryover (suspected, pre-empted)
Consecutive `copy4rows` calls share SrcB rows 16-19; a later `MOVB2D` could read the previous copy's
SrcB. **Fix:** give each `copy4rows` a disjoint SrcB window (16/20/24/28). (Turned out not to be the
actual failure — the result was identical with/without — but it's correct and cheap, so kept.)

### Bug 1 — false "topB is corrupted" (the diagnostic was lying): missing `SETRWC`
`GMG_DIAG_TOPB` reported topB mixed with group-1 (topA) experts. Bisection (skip topA work entirely →
topB clean; add back each topA step) pinned it to **`restore-topA` (`copy4rows<12,0>`) running right
before the SFPU readout**. Root cause: **`_gmg_copy_topk_run` / `_gmg_normalize_run` lacked the
`TTI_SETRWC(...,SET_D)`** that `merge4_top8` / `finalize` have at their start. An FPU MOP (`copy4rows`,
whose last `MOVB2D` uses `ADDR_MOD_2` = +64 to the Dst base) leaves the **Dst RWC counter advanced by
+64/tile**; the next SFPU `SFPLOAD` then reads `offset + leftover_RWC` → wrong rows. **Rule: any SFPU
op that reads DEST right after an FPU MOP must reset Dst RWC first.** topB (and topA) were always
correct in DEST — only the SETRWC-less *readout* was biased. Fix: add `SETRWC` to both helpers.

### Bug 2 — confirmed via dump that `log1.txt` (no reset) is stale
A no-reset run produced all-zeros/stale output; only the post-`tt-smi -glx_reset` run is trustworthy.
Reinforced the hard-reset discipline.

### Bug 3 — the 2-run merge orientation (the long one)
With topA and topB both individually correct (proven by `GMG_DIAG_TOPA`/`TOPB` on all data incl.
batch 2), `finalize` still mis-selected by ~1 expert at the boundary. Tried, in order:
- single `ph3_st4_to_1`, topB hi→LREG2 / lo→LREG3: **topB high end dropped**.
- swap topB lo/hi: **"top-4 of each run"** (no cross-comparison) — closer but still wrong.
- `_gmg_merge4_runs_raw` after re-laying topA→{0,4}, topB→{2,6}: **duplicates** (`[71,71,114,114,…]`).

Root cause: `merge4_top8`'s output run is in a lane orientation that the 2-run *merge* primitives
(`ph3_st4_to_1`, `merge4_runs_raw`) don't expect (they want step1-transpose-format runs). **Fix that
worked: stop trying to merge two sorted runs — load all 16 candidates and FULL-SORT them**
(`bitonic_top8_ph0_to_ph3`), which makes no orientation assumptions. See §1, finalize.

A useful debugging fact: the failing cases were **not tie-breaking** — verified by recomputing the
true bias ranks (a dropped expert was rank-5, 0.03 above the cutoff = 3-4 bf16 steps, not a tie).
Don't hand-wave "it's ties" without checking the rank gap vs the bf16 step at that magnitude.

---

## 5. Diagnostics left in place (for the upcoming generalization)

All OFF by default. Toggle in `device/kernels/generalized_moe_gate_kernel.cpp`:
- `GMG_DUMP_AFTER_SUM_TOP2` / `GMG_DUMP_AFTER_STEP0` / `GMG_DUMP_AFTER_STEP1` — stop after that stage,
  pack `bias`(→output) + `indices`(→output_indices) as a 16×16 face; read via
  `test_generalized_moe_gate.py::test_dump_sum_top2_layout` (rigged: idx=arange(256), bias[g,j]=j+g).
- `GMG_DIAG_TOPA` / `GMG_DIAG_TOPB` — in the ungrouped path, output topA (or topB) ALONE; pair with a
  top8(groups 0-3) (or groups 4-7) golden in `op.py` to isolate a half from the finalize.
  Helpers `_gmg_copy_topk_run` / `_gmg_normalize_run` carry the Bug-1 `SETRWC` fix.

---

## 6. Generalization notes (512 experts / softmax / top-n = 6,10)

- **512 experts** (16 groups × 32): breaks the layout assumptions (group g at row g; rows 0-7
  SFPU-addressable; one face). 16 groups exceed a single face — likely needs multi-tile handling and
  more than two `merge4` runs (e.g. 4× quarter merges + a wider finalize).
- **top-n = 6/10**: the bitonic network (`merge4_top8`, `ph0_to_ph3`, the finalize sort) is built for
  k=8. Changing k touches the bitonic stage counts.
- **softmax** (vs current sigmoid): the normalize tail is a plain sum-normalization; softmax needs
  exp + max-stabilization.

---

## Key references

- Ungrouped orchestration: `device/kernel_includes/tt_metal/include/compute_kernel_api/generalized_moe_gate.h`
- SFPU core: `device/kernel_includes/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_generalized_moe_gate_topk_single_face.h`
  (`store8_even_cols_split`, `ph3_st4_to_1`, `ph0_to_ph3`, `reverse_sort_order`, `sum_top2`,
  `merge4_top8`, `merge4_runs_raw`, `top8`, `copy_topk_run`, `normalize_run`, `finalize_ungrouped`)
- Transpose / copy4rows (FPU): `device/kernel_includes/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_generalized_moe_gate_transpose_dest_single_face.h`
  (`step0`, `step1`, `step1_hi`, `copy4rows`, `step2`)
- Dump pack: `device/unified_kernels/generalized_moe_gate.hpp`
- Golden + tests: `models/demos/deepseek_v3/{tt/generalized_moe_gate/op.py, tests/test_generalized_moe_gate.py}`
