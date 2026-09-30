# Operation Requirements: mhc_pre

## Definition
- **Formula**: per token row x (length n·C), with n = 4:
  - `r = rsqrt(mean(x²) + norm_eps)`
  - `mix = (x @ W) · r`
  - `pre = σ(a_pre·mix[0:n] + b[0:n]) + eps`
  - `post = 2σ(a_post·mix[n:2n] + b[n:2n])`
  - `comb = Sinkhorn(a_res·mix[2n:] + b[2n:])`, n×n in DeepSeek-V4 order: row softmax + eps, one column
    normalisation, then (iters−1) × (row, column)
  - `y = Σ_i pre_i · x[i·C:(i+1)·C]`
- **PyTorch Reference**: `eval/golden_tests/mhc_pre/helpers.py::pytorch_mhc_pre` (with `sinkhorn_knopp`).
- **Import Path**: `from ttnn.bringup.mhc_pre import mhc_pre, default_compute_kernel_config`
- **Function Signature**:
  `mhc_pre(input_tensor, proj_weight, proj_bias, *, scale: tuple[float, float, float], sinkhorn_iters: int = 20, eps: float = 1e-6, norm_eps: float = 1e-6, compute_kernel_config: ttnn.ComputeConfigDescriptor = None) -> (y, post, comb)`
  - y is `(..., T, C)` in X's dtype.
  - post is `(..., T, n)` and comb is `(..., T, n·n)`, both float32.
  - All outputs are TILE, DRAM interleaved.

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [float32]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED weight_dtype**: [float32, bfloat16]. bfloat16 was added by the verifier on golden evidence,
  with no kernel change.
- **SUPPORTED shape-derived axes**: alignment ∈ {tile_aligned, h_non_aligned}
- **SUPPORTED op-specific axes**: fp32_dest_acc_en ∈ {True}; any `sinkhorn_iters ≥ 1`
- **Cores**: full grid.
  - Regime R1 `group_ksplit_resident`: token tile-rows go to groups, and nC is split by stream-column slice
    across a group's ranks.
  - The partials are gathered to the root with a push and a monotonic semaphore, folded in fp32 on the SFPU,
    and multicast back (`mcast_pipe`).
  - The Sinkhorn is owned round-robin. X crosses DRAM once.
- **Compute config**: caller's `math_fidelity` / `math_approx_mode`; fp32 DEST forced (anything else is refused).
- **Golden baseline**: 106 supported_pass, 98 xfail_expected (all `dtype = bfloat16`), 1 supported_fail.
  The failure is `test_large_sinkhorn_logits[T64_nC4096]`, a precision failure tracked in Refinement 2.
  There is 0 drift.

Blocking-model classification of what is left (TARGET − SUPPORTED = {dtype: bfloat16}, plus one tracked
failure and one deferred regime):

| Candidate | Kind | Where |
|-----------|------|-------|
| `dtype = bfloat16`, together with a precision fix for fp32 W | knob-turn (CB formats already derived), plus a projection precision change | Refinement 1 |
| fp32-stream exact projection / Σx² | precision (compute-path change gated on dtype) | Refinement 2 |
| W column broadcast (R2 regime, reuse-shared operand) | **scheme-change** (new mcast topology) | Refinement 3 (perf) |
| group-combine latency / overlap for small C | scheme-change or knob-turn (lamp L2 / L1) | Refinement 4 (perf) |
| block-size × buffer-depth co-tune for bf16 streams | knob-turn | Refinement 5 (perf) |

### [x] Refinement 1 — bf16 residual streams (lands the perf-focus contract)

**Goal**: add `ttnn.bfloat16` to `SUPPORTED["dtype"]`, so that all 98 `xfail_expected` cells pass. That is 50
`test_op` cells and 48 `test_op_loose` cells, and it also turns on `test_comb_depth_chain[bf16]`.

The kernel already runs bf16 X. CB formats and page sizes come from `_cb_table`, and the helpers reconfigure
per operand. The blocker is precision, not plumbing. With bf16 X × **fp32 W**, post/comb sit at rel-RMS
5.0–5.5e-4 against the `("coeff", bfloat16)` gate of 5e-4. That failed on 11 cells (17×512, 1000×7168,
256×24576 test_op; 7 loose shapes) and on the bf16 depth chain. bf16 X × bf16 W already passes everywhere.

Root cause, measured on device (see `verification_report.md`, Precision Baseline):
- **(a)** At HiFi4 the FPU consumes a fp32 (tf32-valued) W with only ~9 explicit mantissa bits. This error
  source is worth ~2e-4 of rel-RMS.
- **(b)** The FPU's in-tile matmul accumulation is not fp32-exact. This is a ~2.5e-4 floor that remains with
  an exactly-representable W.

Fix (a) with an exact **bf16 hi/lo split of the resident W**:
- `W_hi = bf16-truncate(W)` and `W_lo = W − W_hi`. Both parts are exactly bf16 for any tf32 W (verified on
  CPU over the golden W's).
- Compute them once per kernel on the SFPU from the resident W tiles (UnpackToDestFp32 copy → mask → subtract
  → pack both as bf16).
- Accumulate `X@W_hi + X@W_lo` in the **same DEST window**. Do not pack and reload a partial: an FPU reload of
  an fp32 partial truncates it.
- Apply the split only when `weight_dtype == float32`. A bf16 W is already exact.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- This refinement is anchored to perf refinement 3. The perf-focus contract is bf16 X, **fp32 W**, TILE,
  DRAM interleaved, `fp32_dest_acc_en = True`, HiFi4. This refinement is the whole unlock, so it must ship
  that exact config green. Do not substitute bf16 W as a proxy.
- **Build it performantly.**
  - `cb_weight_hi` + `cb_weight_lo` as bf16 together cost the same L1 as today's fp32 `cb_weight`
    (`2 · kmax · 2048` B). Put both in `_cb_table`, the single source of CB sizes.
  - The split runs once, before block 0. It must not become a per-block pass.
  - The projection's FPU work doubles. N is 1 tile, so this is small next to the DRAM-bound reader. Measure
    640×7168 bf16 X / fp32 W before and after: the baseline is 270.9 µs, and it must not regress beyond noise.
- Residual source (b) is a hardware floor. It passes the 5e-4 gate with ~2× margin (bf16-W cells today:
  2.5–2.9e-4). If some cell still fails after the split, report its rel-RMS. Do not EXCLUDE it.
- If `matmul_block` cannot accumulate two in1 CBs into one DEST window, a thin custom block op over raw
  `matmul_tiles` is the sanctioned realization (design scope boundary). The X block retention contract
  (in0 kept for the y-mix) still holds.
- Refinement 2 reuses this split machinery for fp32 X, so keep the split a reusable block op.

**Done when**:
- `dtype ∈ {float32, bfloat16}` is in SUPPORTED.
- `verify_supported` shows 0 `xfail_expected`, 0 drift, and every bf16 cell passing, including all 48 bf16
  loose cases and `test_comb_depth_chain[bf16]`.
- No regression on the fp32 cells.
- 640×7168 bf16 device-ns is not worse than 270.9 µs beyond noise.

**Outcome**: landed.
- `dtype = bfloat16` is in SUPPORTED. The golden suite is 205/206; the only failure is the pre-existing
  Refinement 2 cell. Every bf16 cell passes, including `test_comb_depth_chain[bf16]`.
- bf16-X × fp32-W post/comb rel-RMS went from 5.0–5.5e-4 to 2.4–3.1e-4, which is the bf16-W floor.
- 640×7168 bf16 / fp32 W: 267.3 µs baseline on this box → 269.6 µs (+0.9 %, noise).
- The naive split measured 295 µs:
  - The SFPU split sat after the whole W read. Fixed by pushing W in `W_CHUNK_TILES = 8` chunks, so the
    split runs under the W DRAM read.
  - The doubled matmul cost ~6 µs. Fixed by running the W_lo products at `W_LO_FIDELITY = LoFi`.
- This op is still DRAM-bound on the W re-read; that is Refinement 3.

### [x] Refinement 2 — fp32-stream projection precision (large Sinkhorn logits)

**Goal**: move `eval/golden_tests/mhc_pre/test_regression.py::test_large_sinkhorn_logits[T64_nC4096]`
(fp32 X, fp32 W, a_res = 30) from `supported_fail` to passing.

Today it fails only the worst-row clause of the doubly-stochastic gate: device max\|rowsum−1\| = 0.0674 vs
the reference's 0.0672 + 5e-5 slack. The columns are exact, PCC is 0.9999982 and rel-RMS is 1.9e-3.

The Sinkhorn itself is fp32-exact (probe: 1.9e-7 vs fp64 on isolated logits). The gap is tf32-class noise in
the logits, ×30. The FPU reads fp32 X with ~9 explicit bits (truncating) in **both** the projection and the
Σx² path. The two truncation biases partly cancel in `mix·r`, so fixing only one makes it worse (CPU
emulation: projection-only exact → worst row 0.06718; Σx²-only exact → 0.06726).

Lever: exact fp32 → bf16 splits of X in the fp32-stream path.
- Split `x = x_hi + x_mid (+ x_lo)` into bf16 pieces on the SFPU per resident block.
- Feed the projection (`Σ_pieces x_p @ {W_hi, W_lo}`, dropping products below fp32 resolution).
- Feed Σx² (`x² = Σ` cross terms, or an SFPU square of the UnpackToDestFp32 copy).
- Both must be done together.

**Verifier notes**:
- Gate the whole path at compile time on `dtype == float32`. The bf16 perf-focus path must compile to exactly
  what Refinement 1 shipped.
- Depends on Refinement 1's W hi/lo split, which is required here too, so it goes second.
- **L1**: at C=7168 fp32 the footprint is already 1.26 MB of 1.47 MB. Do not make the pieces resident as a
  second copy of the block. Recompute them per K chunk from the resident fp32 X instead: a fixed chunk window
  sized as a knob in `_cb_table`, streamed over K (axis accounting: streams K, does not span it). The x_block
  depth-2 prefetch must survive.
- **Perf bar**: fp32 streams are not the perf focus, but the extra SFPU work must stay under the DRAM shadow
  where possible. Report fp32 640×7168 device-ns: the baseline is 383 µs.
- Optionally extend the split to the y-mix (design lamp L5). That removes the −6.8e-4 y bias for fp32 streams
  at little marginal cost once the pieces exist. It is not required by any gate.
- Escape hatch, if the split is out of reach in one pass: partial-tick `[~]` and report the measured worst-row
  error. **Do not EXCLUDE or shrink SUPPORTED.** Also note that the golden owner has been asked
  (`verification_report.md`, Recommendations) whether `ROW_SUM_SLACK` is intended to hold at this noise level.

**Done when**:
- `test_large_sinkhorn_logits[T64_nC4096]` passes, and so does the whole `test_regression.py` including the
  depth chains.
- `verify_supported` shows 0 `supported_fail` and 0 drift.
- No regression on the bf16 cells (bit-identical bf16 outputs are expected, because the path is gated).

**Outcome**: landed.
- `test_large_sinkhorn_logits[T64_nC4096]` passes: worst row 0.0672204, against reference 0.0672202 and limit
  0.0672702. The post-logit rms error went from 2.9e-4 to 3.45e-5.
- The bf16 pieces alone were insufficient. The FPU's in-tile dot product rounds to ~11 bits below its largest
  product. So the fp32-X path uses an exact-grid split:
  - x0 and W0 lie on power-of-two grids, so x0·W0 sums exactly in-tile;
  - bf16 remainder pieces make up the rest;
  - Σx² is computed exactly on the SFPU.
- The path is compile-time gated. bf16 640×7168 stays at 270.2 µs.
- fp32 640×7168 went from 378 µs to 567 µs. That path is now compute-bound: split SFPU ~60 µs, one-time W split
  ~47 µs, and 5 HiFi products.
- Next I would try either:
  - sharing the W split across groups (it rides on Refinement 3's W broadcast: one split per W slice instead of
    one per group), or
  - a DEST full-sync fp32-X variant that splits 2 x tiles per window.

  I did neither here, because fp32 is not the perf focus and both are out of this heading's scope.

### [x] Refinement 3 — Speed up the perf-focus profile T=640, C=7168, bf16 streams (W column broadcast)

**Type**: perf

**Goal**: optimize the flagged loose case exactly as specified:
- Case: `feature_spec.LOOSE_CASES` "mhc_pre T640 C7168 DeepSeek-V4 (5K/SP8)"
- Config: `(1, 1, 640, 28672)`, dtype bfloat16, **weight_dtype float32**, TILE, DRAM interleaved,
  `fp32_dest_acc_en = True`, HiFi4
- `extras.attention`: PERF FOCUS. Its goal is `extras.target_ns` ≈ 121 µs (80 % of 512 GB/s).
- Measured today (bf16 enabled): 270.9 µs = 2.24× the roofline.

The dominant lever is the design's deferred regime **R2: W column broadcast.**
- W does not vary along the token-group split, so every one of the G_t = 10 groups re-reads its rank's W
  slice. For fp32 W that is 36.7 MB, about 45 % of this shape's ~83 MB of DRAM traffic.
- Read each rank's W slice once, e.g. rank r of group 0, and multicast it down to rank r of every other
  group. With `group_w = grid_x` that is the physical column.
- This changes only who fills `cb_weight` (or `cb_weight_hi/lo` after Refinement 1). Compute, the other CBs
  and the group combine are unchanged.
- Relevant catalog entries in `ttnn/ttnn/operations/examples/master.md`: `shared_input_reuse` and
  `mcast_topology` (T3; `mcast_pipe` `Mcast1D(PerColumn)`). Also `noc_placement`: pick the mcast NoC against
  the X-read NoC. Tensor sizes are unchanged, so SUPPORTED does not change.

**Verifier notes**:
- Validate by building the variant and measuring device-ns. Do not infer the gain from a remove-W ablation.
- Keep the bias read per core. It is 0.6 % of the bytes, or it can ride the same mcast.
- If the one-shot W mcast serializes the start of block 0, overlap it with the first X block's read. The
  reader already owns both.
- This is one T3 lever, a whole phase by itself. Do not bundle knob tunes here; those are Refinement 5.

**Done when**:
- Measured device-ns improves on the flagged 640×7168 bf16 / fp32-W case, moving it toward ~121 µs.
- Its precision gates still hold, and the golden suite is green.
- There is no regression across the config-spanning guard set: one representative per distinct kernel path ×
  dtype × weight_dtype × group shape (`group_h = 1` vs `> 1`, `group_cores = 1`), for example 640×1792 fp32,
  1×28672 decode, 32×128, and 1280×4096 bf16/bf16.

**Outcome**: R2 built (reader-only change: `Mcast1D(PerColumn)`, Counter signal, write-once landing so no handshake;
path-gated to `group_h == 1` with ≥ 2 full active group rows, else R1; knob `W_BCAST`). Receivers issue their X block 0
read before the W receive. BH p150 device-kernel ns, bf16 X / fp32 W: 640×7168 **272.0 → 191.6 µs** (−30 %; 1.58× the
121 µs target), 640×1792 105.8 → 88.4, 1280×4096 263.9 → 197.3, 4096×1792 372.5 → 351.3; fp32 X 567 → 504, 176 → 152,
515 → 422, 660 → 651. X0-before-W measured on vs off: 191.6 vs 193.9 / 197.3 vs 214.5 (kept). Bottleneck now: the
reader (NCRISC) ends at ~122 µs (≈ the DRAM target, X read at ~300 GB/s) while compute/writer run to ~190 µs — the
wall is the post-arrival tail of the last block (projection, combine round trip, coefficients, y-mix, y store,
Sinkhorn) with only 2 blocks per core. Next: overlap/shorten that tail (Refinement 5 knobs: block_token_tiles,
L3 project(b+1) before coefficients(b), split y store), not done here because the verifier scoped this heading to
the one T3 lever.



### [x] Refinement 3b — Speed up the perf-focus profile T=640, C=7168, bf16 streams (W column broadcast) (debug: fix gate violations)

**Goal**: fix the hard violation from Refinement 3 so the completion gate's three bullets hold.

**Verifier notes** (mechanical, from the harness completion gate):

```
Bullet 2 FAIL: acceptance/refinement tests failing:
  - ttnn/ttnn/bringup/mhc_pre/tests/unit/test_mhc_pre_precision_baseline.py::test_mhc_pre_bf16_stream_fp32_weight_precision[X1x1x256x24576] - AssertionError: comb: rel_rms 0.011754175593281494
Bullet 3 FAIL: REGRESSION — prior-passing golden cells no longer pass (responsible cells 205/206). A prior-passing cell that failed, hung, or never ran (suite hung before reaching it) is a regression.
```

**Done when**: the gate passes — zero hangs in SUPPORTED, acceptance + refinement tests pass, golden majority with no regression.

**Outcome**: root cause was a source-L1 race in the R2 W sender. It pushed each W chunk to its own compute *before*
`sender.send()`. The fp32-W hi/lo split rewrites `cb_weight` in place (aliased `cb_weight_split`), so the sender's
compute could overwrite chunk j while the multicast was still reading it. Receivers then got half-split W, giving
intermittent comb rel-RMS of about 1e-2. Reproduced 1 in 40 seeds (256×24576 bf16/fp32-W) before the fix and 0 in
40 after. Fix: push after `send()` returns (the pipe's source guard). Full golden suite 206/206; unit dir 35/35 (×4
runs). Perf, BH device-ns: bf16 640×7168 191.9 µs (unchanged); fp32-X 640×7168 543.5 µs (was 504 with the race;
567 before R3), because the sender's in-place split now waits for each chunk's mcast.

### [x] Refinement 4 — Speed up the perf-focus profile T=640, C=1792, bf16 streams

**Type**: perf

**Goal**: optimize the flagged loose case exactly as specified:
- Case: `feature_spec.LOOSE_CASES` "mhc_pre T640 C1792 DeepSeek-V4 TP4 (5K/SP8)"
- Config: `(1, 1, 640, 7168)`, bf16 X, fp32 W, TILE, DRAM interleaved, fp32 DEST, HiFi4
- `extras.attention`: PERF FOCUS. Its goal is `target_ns` ≈ 30 µs.
- Measured today: 104.7 µs = **3.45×**. This is the worst ratio of the three focus shapes. Even with W re-read
  the bytes are ~21 MB (~50 µs), so it is **latency / synchronization-bound**, not DRAM-bound.

Per-rank work is tiny (`core_k_tiles = 24`, 1 row per block, 2 blocks per core). The per-block gather →
fold → mcast round trip, and the serial phases around it, dominate.

Relevant levers (design perf lamps and `master.md`):
- **L2** (grid sync): a `tensix_all_reduce`-style tree reduce + mcast (T3, measured 1.45–1.60× over a flat
  root on a busy grid), or a narrower group (`group_w < grid_x`: fewer ranks, more groups). The latter is a
  knob-turn on the core-assignment surface.
- **L1** (overlap): `x_block_depth` / `block_token_tiles`, co-tuned (`double_buffer`, `compute_block_size`).
  Today one block's read barely shadows one round trip.
- **L3** (compute skew): issue `project(b+1)` before `coefficients(b)`.

The implementer picks the lever or levers by measurement. Refinement 3's W broadcast also removes 8 MB here,
so measure after it lands.

**Verifier notes**: tree combine and narrower groups are alternatives, not additions. Measure both before
committing to the tree (the T3 cost). Every group-geometry knob must stay derived in `make_plan`, and the
`group_cores = 1` and `group_h > 1` branches must stay covered (acceptance tests).

**Done when**:
- Measured device-ns improves on the flagged 640×1792 bf16 case, moving it toward ~30 µs.
- The golden suite is green.
- There is no regression across the config-spanning guard set (as in Refinement 3, plus 640×7168 bf16).

**Outcome**: 640×1792 bf16 (fp32 W, BH device-ns) went from 104.7 µs (verifier) / 86.5 µs (after R3) to **44.3–45.6 µs**
(1.95× vs R3). Guard set vs R3, bf16 / fp32 X: 640×7168 191.9 → 154 / 543.5 → 409; 1280×4096 199.2 → 176 / 447.6 → 384;
4096×1792 352.8 → 246 / 633.5 → 558; 640×1792 fp32 157.3 → 130. The winning levers: the W fill moved to the writer
(column all-gather, per-share split); narrow groups (L2: group_w = 5, one block per group; this took the place of the
tree combine); owner C discount (`OWNER_C_DISCOUNT = 7`, −2 µs); the owned block fused with the coefficient tile reload
(−1.5 µs); bounded X look-ahead + per-K-chunk streamed proj/Σx² (−1.6 µs vs no chunking). `READER_NOC_FLIP_ROWS`
measured null and is parked at 0. What binds now: the X read is NoC/DRAM-saturated. All 9.2 MB lands by ~24–26 µs
(~370 GB/s aggregate), and the top core rows are starved on NoC0 (their first chunk lands only ~3 µs before the end,
whichever rows are flipped). A ~19 µs serial tail follows on the critical group: proj + Σx² on late data (~5 µs),
gather + fold + mcast (~3.5 µs), the owner's coefficients + Sinkhorn (~8 µs), then y-mix + stores. Next I would try:
a 4-way SFPU fold or a split combine (~1 µs), a register-fused Sinkhorn row/col pass (~1 µs), and Σx² as a matmul
X·Xᵀ diagonal (~1.5 µs). I did not do them because each is ~1 µs, inside the ±1 µs run-to-run noise at this size, and
the lower-fidelity Σx² variant spends precision budget. The ~30 µs target would need the X read itself to get faster.

### [x] Refinement 5 — Speed up the perf-focus profile T=1280, C=4096, bf16 streams (block × depth co-tune)

**Type**: perf

**Goal**: optimize the flagged loose case exactly as specified:
- Case: `feature_spec.LOOSE_CASES` "mhc_pre T1280 C4096 GLM-5.3-Flash (5K/SP4)"
- Config: `(1, 1, 1280, 16384)`, bf16 X, fp32 W, TILE, DRAM interleaved, fp32 DEST, HiFi4
- `extras.attention`: PERF FOCUS. Its goal is `target_ns` ≈ 133 µs.
- Measured today: 264.1 µs = 1.98×.

`BLOCK_TOKEN_TILES_CAP = 1` and `X_BLOCK_DEPTH_DEFAULT = 2` were measured on **fp32** shapes only. bf16
halves the X term of the footprint (`cb_x_resident`), which frees about 200–350 KB per core at C=4096–7168.

Co-tune the block surface the planner exposed for bf16:
- `block_token_tiles` (coarser amortizes the per-block round trip, matmul init and SFPU inits)
- `x_block_depth` (3 = deeper prefetch across the combine shadow)
- `y_chunk_tiles` / `y_depth` (writes in flight per barrier)
- reader bytes in flight

These are several cheap T1/T2 knobs from `master.md` (`double_buffer`, `compute_block_size`,
`noc_placement`). They may be batched in one phase. The chunk floor is whole tiles, and coarser amortizes up
to the point where overlap is lost (lamp L1). Selection must remain a function of (T, C, dtypes, grid), per
the prompt MUST rule: no per-shape constants. A dtype-dependent cap derived from the L1 fit is fine.

**Verifier notes**:
- If a bt > 1 default is adopted, re-check `sumsq_block`'s per-row `cb_sq_acc` handshake (verification
  report, Expression) so the Σx² phase does not become a per-row serialization point.
- `test_mhc_pre_blocking.py` must keep covering the non-default branches.

**Done when**:
- Measured device-ns improves on the flagged 1280×4096 bf16 case, moving it toward ~133 µs.
- The golden suite is green.
- There is no regression across the config-spanning guard set, including the other two focus shapes and the
  fp32 640×7168 case.

**Outcome**: 1280×4096 bf16 X / fp32 W went 175.8 µs (R4 build; 264.1 µs at verifier time) → **148.2 µs**, 1.11× the
133 µs target. BH device-ns, medians. The block × depth knobs measured null or negative and stay at their
defaults: bt 2 → 163.8, bt 4 → 186.7, depth 3 → 148.8, depth 1 → 174.3, and y chunk / y depth / stream chunks /
in-flight all within ±2 µs. A coarser block at C=4096 forces fewer groups or depth 1 and loses the overlap, so
the bf16 L1 headroom stays unspent. The win came from NoC placement: the reader NoC flip is derived for the top
round(0.4·grid_y) rows, and the W column share is read by the reader on its own NoC ahead of X. It is path-gated
off for bf16 X / bf16 W and for bf16 W without the column broadcast, where it lost. Other shapes: 2048×5120 bf16
344 → 315, decode 1×7168 59 → 46, 640×7168 bf16 153 → 146, fp32 X / bf16 W −3…−11 %. 640×7168 fp32 is +0.7 %
(inside the noise band). A latent missing W wait on the fp32-X / bf16-W path was fixed (golden 206/206). What
binds now: the W all-gather ends at ~50–58 µs, bound by its slowest (NoC0-starved) row, and ~70–85 µs of per-core
compute follows (2 × [projection + Σx² 15.5, coefficients 5–17, y-mix 7 µs]). Landing W first only moves the wall
onto the middle rows' late X plus their tail. What I would try next: a row-weighted W share split (fewer tiles on
the NoC-starved rows) or an all-gather that does not wait on the slowest row, then a cheaper projection + Σx² per
K tile. I did not do them because they are a new share topology and a compute rewrite, beyond this knob-turn
heading.
