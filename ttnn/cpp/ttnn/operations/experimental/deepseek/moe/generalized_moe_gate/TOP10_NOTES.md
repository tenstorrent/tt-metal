# Top-10 (k > 8) support for `generalized_moe_gate` — plan, analysis & decision (DEFERRED)

**Status: deferred.** Chosen approach is **Plan 2 (two-pass: top-8 + top-2)**; not yet implemented.
Rationale (per team guidance): use the fast fused op where it's available/easy (top-4/6/8 — already
done), and fall back to the slower general `ttnn.topk + softmax` path for the hard/rare cases. **Only
Qwen3.5 needs top-10** among the target models; everything else is k ≤ 8. Broad compatibility matters
more than top-10 being fast. Revisit Plan 2 when top-10 is actually scheduled; revisit Plan 1 (below)
only if leadership explicitly asks to optimize top-10. (Top-n k ≤ 8: see `TOPN_NOTES.md`.)

## The core bottleneck (why top-10 ≠ "mask fewer")

For k ≤ 8 we just zero ranks ≥ k before the normalize (`TOPN_NOTES.md`) — cheap, because the pipeline
already produces a sorted global **top-8**. For k > 8 there is nothing to un-mask past rank 7: **every
reduction stage truncates to top-8** (each group of 32 → top-8; `merge4_top8`: 4 groups/128 → top-8;
`merge16_core`: two halves → top-8). A member of the global top-10 can be the rank-9/10 of a single
group or half, which was discarded at an *earlier* stage. So a single-pass "stash the runner-ups"
(Plan 3) does NOT work — the candidates were already thrown away upstream; you'd have to retain ≥16 at
*every* stage, which is Plan 1.

## Plan 2 — two-pass top-8 + top-2 (CHOSEN, deferred)

Correct and reuses the validated top-8 pipeline with **zero bitonic rewrite**:
1. **Pass 1:** run the full 512 pipeline → correct global top-8 (already works).
2. **Mask:** set those 8 experts' keys to **−∞ by INDEX** in the input (see masking note below), so
   pass 2 excludes them.
3. **Pass 2:** re-run the pipeline on the masked input → top-8 of the remaining 504; its **top-2** =
   global ranks 9,10 (each pass is a correct top-8, so nothing is lost).
4. **Combine + normalize-10:** pass-1 top-8 → output cols 0-7, pass-2 top-2 → cols 8-9 (already sorted:
   all of pass-1 > all of pass-2). Normalize denominator = sum of the 10.

**Implementation pieces (effort: moderate kernel restructure, low risk):**
- Drive the pipeline **twice** in one op — keep the input CBs re-readable (don't pop / keep a copy).
- **Inter-pass masking** (the one new primitive): exclude pass-1's 8 winners.
  - **By index (recommended, tie-robust):** extract each of the 8 ids from the sorted top-8, broadcast,
    `v_if(input_id == sel_k){ bias = −inf }` — 8 passes. Needs an "extract lane-k id + broadcast"
    helper. Correct under bf16 ties.
  - By value threshold (`key ≥ rank-7's key`): simpler but **wrong on bf16 ties** at the rank-8/9
    boundary (e.g. keys `[…,7,7,7,7,7,7,7,6,5]` → masks all 7s, pass-2 grabs 6,5). Avoid.
- Combine 8+2 and change the normalize denominator from 8 to 10.
- Plumbing: `topk == 10` (k > 8) branches into the two-pass mode at the kernel top level; the existing
  finalize k ≤ 8 mask is unchanged (each pass is internally a top-8).

**Perf estimate:** ≈ 2 × the single-pass reduction + masking ≈ **~20 µs** (512 top-8 measured at
**9.6 µs**; top-4 also 9.6 µs — k barely affects the reduction cost). The second pass can be trimmed
(only top-2 needed) but the front (sum_top2/step0/combine) can't, so ~17-20 µs is the realistic floor.

## Plan 1 — widen the whole reduction to top-16, then mask to k (DEFERRED option)

Single pass, general for any k ≤ 16, and **likely faster (~12-15 µs est.)** than Plan 2, but a **big,
risky rewrite** of the custom SFPU bitonic (the same lane-layout debugging as the top-6 work, but
wider). Kept as the option to pursue **only if leadership asks to optimize top-10**.

**Resource-constraint findings for 256-top16 (verified — no hard memory wall):**
- **DEST: not a constraint.** `SFP_DESTREG_COUNT = 0x400/2` → 1024 addressable units; we use only
  offsets 0-255 (4 tiles: scores/idx/bias/interm) → ~4× headroom (12 spare tiles to park into).
- **L1: not a constraint** for 256 (single op, no stash; the 512 combine's stash CBs just double page
  size — trivial).
- **SFPU 32-lane register = the ceiling, but top-16 fits:** a top-16 *merge* = sorting 32 candidates =
  exactly fills 32 lanes (zero headroom past 16). Current top-8 uses ~16 candidates, and the layout is
  actually *sparse* (sorted-8 land at lanes 0,8,16,24 — 4 of 32 lanes per row), so there IS lane room;
  the bitonic must be rewritten to pack densely.
- **The real "炸" point = the `copy4rows` park choreography.** SFPU merges run in rows 0-7; FPU
  `copy4rows` parks **4-row** top-8 runs in rows 8-15 (SrcB scratch windows 16/20/24/28). Top-16 runs
  are ~8 rows → topA(8) + topB(8) fills rows 0-15 with no room for the group stashes → overflow. Not a
  hard wall (park into the spare DEST tiles), but the choreography must be re-mapped.

## Plan 1′ — rebuild on the HW `topk` primitive (gpt-oss style) — REJECTED on perf

gpt-oss's `topk_router_gpt` supports k=1..32 by always computing the full sorted **top-32** (via the HW
`topk_local_sort`/`topk_merge`/`topk_rebuild`, intrinsically 32-wide) and applying the user k as a
**softmax mask** (cols 0..k-1 = 0, cols k..31 = −inf) at the end — i.e. "compute the superset once,
mask to k". Clean and general, BUT the HW topk merge is **heavy**: the gpt-oss op is ~67 µs (incl. its
2880×128 matmul, which we don't have, but the merge cost scales with expert count). For 512 = 16 tiles
(15 merges keeping top-32) it would blow far past our 9.6 µs custom single-face top-8. Not worth copying.

## 512 follow-on (once 256-top16 or the two-pass works)

The hard part of 512 (moving block runs via L1 stash → tilize → transpose_wh → place) is **already
proven** and is width-agnostic — a run just carries 16 instead of 8 (stash CB page size doubles). So
512-top-k ≈ "256-top-k per block" + the existing combine + a wider final merge. 256 is the crux.
