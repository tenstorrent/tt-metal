# Generalized MoE Gate — 512/384-expert combine (A2): status, progress, blockers

Target: **Wormhole B0**. Generalize the fused single-op gate from 256 → 256/384/512 experts in **one op**,
computing the true **global top-8** (k=8). 256 stays the fast single op (~2.48 µs). Kimi=384, Qwen=512.
Softmax will later be fused in, so a single op is required (can't split into two ops).

## Architecture (per-block run + combine)
- **Input layout (slice)**: each 256-block → face0 of its own 32×32 tile; logits/bias sharded `num_blocks`
  tiles/core. `num_blocks = ceil(N/256)`.
- **Per block**: run the proven 256 ungrouped pipeline up to a **re-mergeable top-8 RUN**
  (`merge16_to_run`): the run = `(bias, idx, score)` stored at DEST cols `{0,2}` of the
  scores/indices/bias regions (bias=rank key, idx=expert id, score=value).
- **Combine**: place the 2 block-runs at cols `{0,2}` and `{4,6}` and run the proven `finalize`
  (merge16 of the 16 candidates → global top-8 + normalize) + step2.

## DEST layout facts (WH B0, verified)
- Regions: scores @ off 0, indices @ 64, bias @ 128, interm @ 192 (`dst_tile_offset=64`).
  `copy_tile`/`pack_tile` tile index k ↔ SFPU offset k*64.
- **SFPU offsets are mod-32** (offset 32 aliases col 0). Park must be in 16-31, not 32+.
- A run lives at **2 columns** (`{store_lo, store_hi}`), 4 values each. The merge reads cols `{0,2,4,6}`.
- Occupancy probe (arange sentinel in indices region, one full pipeline pass): the pipeline **writes cols
  0-15** of each region; **cols 16-31 are FREE in the *indices* region** (idx isn't transposed). NOT
  verified free for scores/bias (the transpose likely uses 16-31 there).
- `fp32_dest_acc_en = false` (16-bit dest). `dst_full_sync_en = true` (set for the combine).

## What WORKS (verified on device)
1. **`produce_run`** (template `produce_run` on `generalized_moe_gate` → ends at `merge16_to_run`,
   skips normalize/step2): 256 via `produce_run + relocate<0,2,0,4> + normalize_step2` PASSES.
2. **Global indices via per-block `input_indices` tiles** (block b's tile = `arange + b*256`): block1's
   output idx come out correctly global (256-511). Reader sets up `num_blocks` indices tiles; block b
   copies tile b. (The in-kernel offset add was abandoned — see blocker (e).)
3. **block1's pipeline runs correctly as the 2nd block IFF the previous block ended with a full `step2`
   (the OP, not just `step2_init`)**: confirmed — block1 then emits its exact top-8.
4. The proven `finalize`/`merge16` brick, `relocate` (=`copy_topk_run`), `normalize`.

## The BLOCKER: a tight constraint web (why the one-op combine doesn't close)
The two runs must be **co-resident in the SFPU "math" DEST layout** at `{0,2}` and `{4,6}` for the merge.
Getting block0's run to survive block1's processing and land at `{4,6}` hits ALL of:

- **(a)** block1's `produce_run` pipeline writes all of scores/idx/bias **cols 0-15** (the 16×16 face),
  the transpose writes **cols 16-31**, and it uses the **interm** region. → No free 2-col slot in
  scores/bias to park a run during block1's pipeline. (indices 16-31 is free, but a run needs bias+score
  too.) → **in-DEST park is dead.**
- **(b)** The run is in the SFPU **"math" (transposed) layout**; `pack_tile`/`copy_tile` use **standard
  row-major**, so the run does **NOT round-trip through L1** (dump: score bf16 bits land in idx slots).
  → **naive L1 stash is dead.**
- **(c)** The **next FPU op after a `produce_run`** (block1's `transpose_wh`, or the `copy_tile`/
  `transpose_wh` that restores block0) **does not start unless the previous step ended with a full
  `step2`** (resets SrcB/RWC; `step2_init` alone is NOT enough).
- **(d)** But `step2` **(1) scrambles the run still in DEST** and **(2) writes cols 0-31** (incl any park).
  → can't both `step2`-reset AND keep a run parked in DEST.

Every ordering hits one of (a)-(d). Concretely, the L1-stash-with-step2-convert attempt:
- block0: `produce_run → step2 (math→standard, survives L1) → pack standard run to L1`.
- block1: `produce_run → [restore block0 from L1 via transpose_wh standard→math, place at {4,6}] → finalize`.
- The restore (transpose_wh/copy_tile) is AFTER block1's produce_run → broken by (c). Placing it BEFORE
  produce_run → produce_run clobbers {4,6} (a). A step2 reset between → scrambles block1's run (d).
- Symptom: dev_idx = `[506,315,506,315,8062,342,8062,4734]` — block1's experts (256-511) partly present,
  block0 entirely absent, garbage (block1 pipeline leftover at {4,6}). IDENTICAL across copy_tile vs
  transpose_wh vs step2-before-pack → confirms block0 never lands at {4,6} (the restore-after-produce_run
  is consistently broken by (c)).
- **(e)** In-kernel `idx += b*256`: both `sfpi l_reg[LReg4]+o` (SSA doesn't write back to the physical
  LREG the raw TTI_SFPSTORE reads → no-op) and `TTI_SFPIADD(...ARG_IMM)` (no observable change, likely
  SFPCONFIG index-tracking mode) FAILED. → sidestepped with per-block indices (works, see 2).

## MOST PROMISING UNTRIED idea: a 3rd "merge-only" acquire
Stash BOTH block-runs to L1 in **standard** layout (via step2 before pack), then a **separate merge
acquire that has NO `produce_run`** — so (c) doesn't bite, and the only FPU op is `transpose_wh` which
**writes DEST cols 0-15 only** (SrcB rows 16-31 are scratch), leaving **cols 16-31 free to park**:
1. block0: `produce_run → step2 → pack standard to L1`.
2. block1: `produce_run → step2 → pack standard to L1`.
3. merge acquire (no produce_run):
   - `transpose_wh`-unpack block0 (standard→math) → run at math `{0,2}`.
   - `relocate {0,2}→{16,18}` (park; SFPU copy_topk_run, row-selective).
   - `transpose_wh`-unpack block1 (standard→math) → run at math `{0,2}` (writes cols 0-15 only, so the
     `{16,18}` park survives — needs verifying transpose_wh really doesn't touch 16-31 here).
   - `relocate {16,18}→{4,6}` (restore block0). Now `{0,2}`=block1, `{4,6}`=block0.
   - `finalize` (+ step2) → global top-8.
Key bet: in a merge-only acquire (no produce_run-tail), `transpose_wh`-unpack works AND only touches
cols 0-15, so a `{16,18}` park survives. Each transpose_wh-unpack is 3 fields (scores/idx/bias),
done one-at-a-time into the regions (or via interm). Needs: a 2nd run CB set (or 2 pages) for block1's
stash; confirming transpose_wh-unpack→math is the clean inverse of the step2-before-pack; checking that
the per-field transpose_wh + relocate sequence doesn't re-trigger (c) among themselves.

## Linchpin test result (U1+U2) — FAILED
256 path: `produce_run → step2(math→standard) → pack standard to L1 → transpose_wh-unpack(standard→math)
→ relocate<0,2,0,4> → normalize` → **all-0** (dev_idx all 0). So the L1 round-trip + layout-convert does
NOT recover the run. (The earlier math-pack + copy_tile L1 round-trip ALSO gave all-0.) **=> the "stash to
L1" direction is dead** for the run as currently packed. Caveat: all-0 is uninformative — it could be the
L1 round-trip itself, OR transpose_wh-unpack not working on the run CB (needs more setup than just
init_short + srcb_dummy_valid), OR the relocate/normalize AFTER the 3 transpose_wh ops getting broken by an
FPU-tail (the same "op-after-an-FPU-op needs a reset" class of issue that pervades this whole effort). Not
yet localized.

## UPDATE — L1 round-trip WORKS; blocker narrowed to the standard→math DEST convert
New finding (256, GMG_TEST_STASH): `produce_run → step2(math→standard) → pack standard to L1 → copy_tile
unpack` → **block0's run comes back** (golden top-8 ids all present in the dumped idx region, in a
STANDARD layout: the run as a ROW (row 0, cols 0-7), plus the transposed face in cols 0-3). So:
- **The L1 stash is viable** — packing the run in STANDARD layout (step2 BEFORE pack) survives the
  round-trip (the math layout does not; that was the earlier all-0).
- **Remaining blocker = converting the recovered STANDARD run back to MATH `{0,2}`** for the merge.
  Both DEST transposes fail in this standalone (post-copy_tile, fresh-acquire) context:
  - `step2` (DEST→DEST, MOVD2B/TRNSPSRCB/MOVB2D): **HANGS** even with transpose_common_init +
    step2_init + srcb_dummy_valid. step2 is NOT standalone-callable — its TRNSPSRCB depends on the
    SrcB state that step0/step1 set up earlier in the pipeline; called alone it stalls waiting for SrcB.
  - `transpose_wh` (CB→DEST, the input transpose, which IS step2's logical inverse standard→math):
    gives **all-0** for BOTH the bf16 score region AND the uint16 idx region → transpose_wh-unpack is
    totally broken in this fresh-acquire/post-copy_tile context (a setup issue — not just uint16). The
    transpose_wh setup it gets (transpose_wh_init_short + srcb_dummy_valid) is insufficient on its own.
- NEXT to try: (1) a standalone standard→math DEST transpose that actually works (find the right op +
  SrcB/addrmod setup — maybe step0 or step1 rather than step2; ask someone who knows the transpose LLK);
  (2) handle the idx field's transpose separately (it's uint16; bf16 transpose_wh may drop it);
  (3) a small SFPU rearrange (row-0 8-vector → cols {0,2}) instead of a full-face transpose.

## The crisp open question (for a transpose/LLK owner)
We have a top-8 RUN packed to a CB in STANDARD layout (it was step2'd math→standard before pack;
copy_tile loads it back fine, confirming the data is there). We need to load it back into DEST in the
SFPU **"math" layout** — i.e. the pre-step2 layout that the proven `merge16_core` reads at cols {0,2}
(so two runs at {0,2} and {4,6} can be merged). Tried, in a fresh tile_regs_acquire right after a
copy_tile:
- `copy_tile` (unpack + A2D datacopy, no transpose): loads the run but in STANDARD layout (wrong for the merge).
- `step2` (DEST→DEST transpose; transpose_dest_single_face_step2, MOVD2B/TRNSPSRCB/MOVB2D): **HANGS**
  even after transpose_dest_common_init + step2_init + srcb_dummy_valid. Seems not standalone-callable
  (its TRNSPSRCB depends on SrcB state that step0/step1 set up earlier in the pipeline).
- `transpose_wh` (CB→DEST input transpose, the logical standard→math inverse): **all-0**, with BOTH
  transpose_wh_init_short AND the full transpose_wh_init(icb, ocb). Produces nothing in DEST.
**Question: what is the correct standalone way to load a CB tile into DEST *transposed* (standard→math),
in a fresh acquire after a copy_tile, for this single-face gate?** (Or: which of step0/step1/step2 is the
standalone standard→math DEST transpose + its exact SrcB/addrmod/init setup.) Once that one primitive
works, the combine closes: stash one block's run to L1 (standard, step2 before pack — VERIFIED), in the
merge place it at {4,6} via this transpose-load, the other block's run is already at {0,2}, then the
proven finalize → global top-8.

## Overall status: one-op combine BLOCKED
Both stash directions are dead (in-DEST park: no free DEST cols during block1's pipeline; L1 stash: run
doesn't survive the round-trip). The recurring obstacle across ALL attempts is a web of WH B0
micro-architectural constraints that have to be threaded simultaneously: (i) op-sequencing — many FPU/SFPU
ops only start correctly if the previous op ended a certain way (e.g. a full step2 reset); (ii) layout —
the run is in a transposed "math" layout that standard pack/unpack scrambles; (iii) DEST coexistence — two
runs can't both sit in the SFPU-addressable regions while block1's pipeline runs. The proven 256 pipeline
threads (i) internally; every ADDED op (stash/2nd-produce_run/unpack/relocate) re-trips (i)/(ii)/(iii).

## Possible directions (not yet tried / need expertise)
- Localize the linchpin all-0 (dump the regions right after transpose_wh-unpack) to learn whether
  transpose_wh-unpack on a run CB even works.
- FPU-park in DEST tiles 4-7 (dst_full_sync_en=true gives 16 tiles; the gate uses 0-3): move block0's run
  to tile 4+ via FPU (copy4rows-style) during block1's pipeline, move back for the merge. Open question:
  can the SFPU address tiles 4-7 (offset ≥256), or only the FPU? If only FPU, need an FPU move back.
- Deep WH B0 LLK/SFPU expertise on the exact transpose layouts, SFPU offset addressing, and op-sequencing
  resets — these have been reverse-engineered expensively via iteration.

## Test/debug macros (in `generalized_moe_gate_kernel.cpp`)
`GMG_UNGROUPED_TOP8` (default), `GMG_DIAG_BLOCK` (A1: per-block output), `GMG_TEST_PRODUCE_RUN`,
`GMG_TEST_STASH`, `GMG_TEST_PARK`, `GMG_TEST_PARK2`, `GMG_DUMP_OCCUPANCY`, `GMG_COMBINE_DIAG`. Tests:
`test_generalized_moe_gate_512_global`, `test_dump_stash_run` (occupancy). User builds + pastes to
`/home/yuqiaoli/log.txt`.
