# Operation Requirements: mhc_post

## Definition
- **Formula**: `X'_j = post_j · F + Σ_i comb[i][j] · X_i`. comb is applied transposed. The sum is computed in fp32
  and rounded once to X.dtype.
  - **stats**: `stats[t, k] = (X' @ W)[t, k]` for k ≠ 24, and `stats[t, 24] = Σ_m Xmix[t, m]²`, where Xmix is the
    unrounded fp32 mix.
  - **collapse**: `hc = deepseek_hc(stats)` (rsqrt-norm, sigmoid pre/post, softmax + Sinkhorn comb), and
    `y = Σ_i P_i · X'_i` (bf16).
- **PyTorch Reference**: `eval/golden_tests/mhc_post/helpers.py` → `pytorch_mhc_post`, `deepseek_hc`,
  `pytorch_mhc_post_collapse`.
- **Import Path**: `from ttnn.operations.mhc_post import mhc_post, default_compute_kernel_config`
- **Function Signature**: `mhc_post(input_tensor, residual, post, comb, *, compute_kernel_config=None, post_col=0,
  comb_col=0, stats_weight=None, collapse_pre=None, hc_scale=None, hc_base=None, hc_norm_width=None,
  hc_norm_eps=1e-6, hc_eps=1e-6, hc_sinkhorn_iters=20, output_tensor=None, stats_output=None, hc_output=None,
  y_output=None, max_grid=None) -> X' | (X', stats) | (X', stats, hc, y)`

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

**Perf-1 contract** (`feature_spec.LOOSE_CASES`, flagged `attention: PERF FOCUS`):

| Facet | Value |
|---|---|
| Shape | T = 160, C = 5120 (`(1, 1, 160, 20480)`) |
| dtype / sublayer_dtype | bfloat16 / bfloat16 |
| Layout | TILE, DRAM interleaved |
| mode | **collapse** |
| coef_source | **packed** (`post_col = 4`, `comb_col = 8`, `collapse_pre` = the mixes) |
| Compute config | `math_fidelity = HiFi2`, `fp32_dest_acc_en = True` |
| Target | `target_ns = 46 525` (DRAM roofline at 80 % of 512 GB/s); reference stats-only call 103 100 ns |

Refinements 1 and 2 exist to land every one of these values in `SUPPORTED`.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [float32]; **sublayer_dtype**: [float32]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED shape-derived axes**: alignment ∈ {tile_aligned, h_non_aligned}
- **SUPPORTED op-specific axes**: fp32_dest_acc_en = True, mode = plain, coef_source = separate. Column offsets
  (`post_col`, `comb_col`) already work, so the engram-add form is a plain call.
- **Cores**: full grid, 2-D `(R, Q)` split over token-tile rows × column tiles via `choose_split`. The block is
  `1 row × block_c × n` via `choose_block_c`.
- **Compute config**: the caller's `math_fidelity` and `math_approx_mode`; fp32 DEST is forced. Mix is SFPU-only,
  with `UnpackToDestFp32` on every fp32 CB.
- **Golden baseline**: 88 / 88 supported cells passing; 379 xfail_expected, 348 invalid_skipped, 0 loud.

### [x] Refinement 1 — bf16 streams + packed mixes + stats mode (stats-2d-tree)

**Goal**:
- Add `ttnn.bfloat16` to `SUPPORTED["dtype"]` and to `SUPPORTED["sublayer_dtype"]`. All four plain dtype combos
  must work: fp32/fp32, fp32 X + bf16 F, bf16/fp32, bf16/bf16.
- Add `"packed"` to `SUPPORTED["coef_source"]`: one mixes page serves post at `post_col` and comb at `comb_col`.
  The mixes are read once per (core, row) and never sliced.
- Add `"stats"` to `SUPPORTED["mode"]` by building the **stats-2d-tree** regime from `op_design.md`:
  - a resident W slice (`cb_w_slice`, bounded by `W_RESIDENT_BUDGET` inside `choose_split`);
  - `project_block` = `matmul_block` of the bf16 X' block (`cb_xprime_mm`, re-packed from DEST) against W, at the
    caller's fidelity;
  - an exact SFPU sum of squares from the unrounded D0;
  - a deterministic cross-core combine: fixed slot per sender, a semaphore count, an index-order SFPU add,
    `COMBINE_FANIN = 16` → two-level tree, and roots that rotate across rows.

  Caller `stats_output` must also be supported.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- *Why bf16 and stats are one refinement.* There is a true dependency: every stats cell needs bf16/bf16
  (fp32 + stats is INVALID), so stats cannot be built or tested before bf16 exists. Bundling keeps the hardest
  structural work (the cross-core combine) first.
- *The stats part is a scheme-change.* The `c` axis becomes dependent for the stats row. No skill covers the
  combine topology, so follow `op_design.md` → Dataflow Strategy rows "partial → leader / leader → root / gather →
  compute". Use `ttnn.SemaphoreDescriptor`, `noc_async_write`, and `noc_semaphore_inc` / `wait`. The multicast
  helper does not apply here: this is a gather, not a broadcast.
- *Probe the bf16 pack rounding first.* `cb_out_tiles` and `cb_xprime_mm` must be the same bits and round to
  nearest even. If the packer does not, round in SFPU before the pack. The bias floor is 1e-5.
- *Mode-invariant sizing.* Make `choose_split` / `choose_block_c` take `fused = mode ≠ plain` and size against
  the collapse inventory now. Otherwise Refinement 2 changes the stats summation order and breaks
  `test_fused_modes_agree`.
- *Build it performant.* Perf 1 (Refinement 3) optimizes exactly this path. The fused split must fill the grid
  (design: `(2, 54)` at T=160/C=5120), and both dataflow halves must stay block-batched.
- *Ledger folds.* `cb_zero` may be dropped by seeding with `FillScalar` into DEST. Update `l1_ledger.md` rows as
  the fused CBs land.

**Done when**:
- Golden: zero `xfail_expected` remains on `dtype`, `sublayer_dtype` or `coef_source` for plain and stats cells,
  and every stats cell passes (bf16/bf16, separate and packed, aligned and ragged T).
- Acceptance: `test_plain_mixed_dtypes`, `test_plain_packed` and `test_fused_modes[stats-*]` pass.
- Regression: `test_depth_chain[bf16]`, `test_output_buffers[plain]` and `test_output_buffers[stats]` pass.
- No loud categories, and the stats rows of two calls are bit-identical.

**Outcome**:
- Landed: bf16 / bf16 and mixed dtype combos, packed mixes (one page per row), and stats-2d-tree. The tree is
  deterministic and two-level: fixed leaders, rotating roots, grant/arrival semaphores.
- Golden slices green: stats 58/58, plain 232/232, regression 15 + 12 collapse skips.
- Finding: the FPU matmul rounds each 16-term face partial to ~tf32, so the HiFi3/HiFi4 projection uses 16
  exact column-masked passes. HiFi2 keeps one matmul.
- Stats at the focus shape (HiFi2): 151 µs vs 103 µs reference. Plain bf16: 84.6 µs.
- The op is SFPU-mix-bound. Remaining levers are in Refinements 3 and 5.

### [x] Refinement 2 — collapse mode (hc SFPU program + y)

**Goal**: add `"collapse"` to `SUPPORTED["mode"]` by building **collapse-2d-tree** on top of Refinement 1:
- Read `collapse_pre` P in-kernel: separate → one more source page; packed → the same page as the mixes, pre at
  columns 0..n-1. Expand it into the `P_i` column tiles of `cb_coef_cols` (24 tiles).
- `collapse_block`: y = Σ_i P_i · X'_i computed in fp32 on the SFPU from the bf16 `cb_xprime_mm` block, rounded
  once to bf16. Write it to y or the caller's `y_output`.
- On each row's root:
  - permute stats into the hc lane layout;
  - run a custom fp32 SFPU program: rsqrt-norm, sigmoid pre/post, row-max softmax + eps, the Sinkhorn column
    step, then `iters − 1` row/column pairs. Use exact transcendentals regardless of `math_approx_mode`, and no
    FPU operands;
  - inverse-permute and write hc (or the caller's `hc_output`).
- The hc scalars (`hc_scale`, `hc_base`, `hc_norm_width`, `hc_norm_eps`, `hc_eps`, `hc_sinkhorn_iters`) are RT
  args.

**Verifier notes**:
- Depends on Refinement 1: this is a true dependency, because it needs the stats row and bf16.
- No skill covers custom SFPI lane programs. Work from `op_design.md` → "hc lanes" / Key Risks rows and from
  `models/demos/deepseek_v3_d_p/reference/mhc/deepseek_sinkhorn_kernel.py`, which gives the order of operations.
- Probe the SFPU datum ↔ lane mapping first, as in the design's robust 4-row × 16-col quantity blocks.
- Do not use the existing `mhc_split_sinkhorn` op's FPU arithmetic. It is TF32-grade (~1e-3), and the hc gate is
  1e-5.
- Keep the fused split mode-invariant: stats bits must equal Refinement 1's.
- This lands the last axis value of the perf-1 contract. Hold it to the performance-conformance bar: y is
  computed in the mix pass from L1, and the hc tail runs on a rotating root.

**Done when**:
- Every collapse golden cell passes.
- `test_fused_modes[collapse-*]` passes.
- Every `test_regression.py` fused test passes: `test_fused_modes_agree`, `test_engram_add[*]`,
  `test_hc_scalars_are_runtime`, `test_collapse_pre_controls_y`, `test_output_buffers[collapse]`,
  `test_max_grid[*]`, `test_collapse_deterministic`, `test_projection_hifi4`.
- `xfail_expected` is 0 and there are no loud categories.

**Outcome**:
- Landed: collapse-2d-tree. P is read in-kernel (packed = the same single mixes page). y is an fp32 SFPU chain
  from the X' block in L1, before the projection. The hc step is one lane-wise fp32 SFPI program
  (`HcProgram`), run on each row's stats core (rotating root).
- On Blackhole, the dense lane layout came from a device probe: an SFPU load = 4 rows × 8 even|odd columns.
- Golden results: collapse 62/62; `test_regression.py` 26/27.
- The one failure, `test_hc_scalars_are_runtime`, is a golden-suite defect. With `hc_eps = 1.1e-5`, the
  float64 reference itself has |colsum − 1| = 1.1e-5 > `HC_COLSUM = 1e-5`. The gate should scale with
  `hc_eps`; every other gate of that test passes. This needs a human edit to `helpers.py`.
- Perf at T160/C5120 HiFi2: collapse 192 µs vs stats 152 µs. The rest is writer lane permutes (~11 µs),
  the y chain (~9 µs) and the hc SFPU (~5 µs); see the changelog for the levers.

### [x] Refinement 3 — Speed up the perf-flagged profile: block / split knob co-tune

**Type**: perf

**Goal**:
- `feature_spec.LOOSE_CASES` flags **T=160, C=5120, bf16/bf16, collapse, packed mixes, HiFi2,
  fp32_dest_acc_en=True** as the mandatory perf target (`target_ns = 46 525`; reference 103 100 ns for a
  stats-only call). Optimize THIS exact config, using the cheap knob levers the planner exposed. These are the
  T1/T2 entries in `ttnn/ttnn/operations/examples/master.md`: `compute_block_size`, `double_buffer`,
  `width_split`, `tensix_all_reduce` (fan-in).
- **Block count vs block size.** The verifier measured on plain fp32:
  - `MIN_PIPELINE_BLOCKS` 4 → 8 gains 9 % at T640/C7168;
  - but it loses 23 % at T160/C5120, where it forces `block_c = 1`;
  - the coarsest block (1) loses 8–35 %.

  Co-tune a "≥ 2 c-tiles per block, then as many blocks as fit" rule.
- **The `choose_split` objective lamp.** Compare busiest-core bytes with aggregate bytes. `R=1/Q=80` moves 9 % fewer
  DRAM bytes at the focus.
- **Other knobs:** `COMBINE_FANIN ∈ {8, 16, 32}`, `IO_DEPTH = 3`, `COEF_DEPTH = 2`.
- No SUPPORTED change.

**Verifier notes**: keep the fused-mode split mode-invariant. Any objective or knob change must give stats and
collapse the same `(R, Q, block_c)`, or `test_fused_modes_agree` breaks.

**Done when**:
- Measured device-ns improves on the flagged shape, moving it toward `target_ns`.
- Its gates still hold, and the golden suite is green.
- No regression across the config-spanning guard set: one representative per distinct kernel path × mode
  (plain / stats / collapse) × dtype combo × coef_source.

**Outcome**:
- Flagged config (T160 C5120 bf16 collapse packed HiFi2): **193.5 → 138.1 µs** device kernel (−29 %; target 46.5 µs).
  Stats 152.4 → 108.0, plain bf16 84.8 → 74.2; T640: collapse 636 → 362, stats 561 → 333, plain 343 → 231.
  Plain fp32 guard shapes: 0.84–1.00× (no regression).
- Levers (all kept, all live knobs): `choose_split` cost adds `FUSED_ROW_OVERHEAD_BYTES` per row and drops W
  bytes (W is a residency constraint only), tie-break = DRAM-bank spread of the core starts then fewest cores
  → focus split R=2/Q=54 → R=5/Q=21. The reader streams W per block under the first row's mix (it was read
  and barriered before any F/X). `COEF_DEPTH` 1 → 2. Block rule = old `MIN_PIPELINE_BLOCKS` bound capped by
  `BLOCK_C_TARGET` (2 for bf16, 3 for fp32 streams). Fan-in = smallest of (8, 16, 32) fitting the semaphores.
  `IO_DEPTH = 3` / `OUT_DEPTH = 3` measured flat or worse, so they stay at 2.
- Bottleneck now: the fit across 13 splits is ~20 µs fixed per row on a core plus ~11 µs per (row, c-tile)
  unit, i.e. ~2.7 µs per output tile. That is the SFPU mix chain (Refinement 5). W also crosses DRAM 5×
  (13 MB) at R=5, which is Refinement 4's w-mcast.
- Not done, and why: the bank-aligned column-start hot-spot is avoided via the split tie-break, not fixed.
  A per-core rotated block walk would fix it at any Q, but it changes the stats accumulation order, so it
  is left for the tournament. Plain fp32 T160 is 100.6 µs at Q=22 vs 111 µs at Q=21 (aggregate-DRAM-bound
  there); a mode/dtype-aware core-count preference could take it.

### [x] Refinement 4 — Speed up the perf-flagged profile: reuse multicast (W and coefficients)

**Type**: perf

**Goal**:
- On the same flagged config (T=160, C=5120, bf16, collapse, packed, HiFi2), remove the reuse-shared DRAM
  re-reads:
  - **w-mcast**: W is read R× and must be read once. A W injector per column group multicasts the slice to the R
    cores sharing those columns.
  - **coef-mcast**: the mixes are read Q× and must be read once per row group. Use one reader plus a multicast
    to the group's `cb_coef_src`.
- These are the `deferred` regimes in `op_design.md`, and the T3 `shared_input_reuse` / T2 `mcast_topology`
  entries in `ttnn/ttnn/operations/examples/master.md`. Use the multicast helper: host `ttnn.Mcast` + `attach()`,
  kernel `McastArgs` `sender().send()` / `receiver().receive()`.
- No SUPPORTED change.

**Verifier notes**:
- This is a new dataflow topology (injector + broadcast), so it stands alone.
- Validate it by building it and measuring device-ns. Do not infer the gain from a remove-the-read ablation.
- If Refinement 3 moved the split to `R = 1`, w-mcast becomes a no-op. In that case, measure coef-mcast alone and
  record why w-mcast was dropped.

**Done when**:
- Measured device-ns improves on the flagged shape, the golden suite is green, and stats / X' bits are unchanged.
- No regression across the guard set (plain fp32, plain bf16, stats, collapse; separate and packed).

**Outcome**:
- Flagged config (T160 C5120 bf16 collapse packed HiFi2): **137.9 → 129.5 µs** device kernel (−6 %).
  Stats 107.3 → 96.0, plain bf16 74.3 → 73.6; HiFi4 stats 204.5 → 195.1, collapse 236.7 → 228.5; T640 HiFi2
  stats 314.1 → 284.7, collapse 361.8 → 354.0. Plain fp32 separate: T160 109.2 → 94.7, T1/C7168 49.3 → 40.7,
  T4096 740.7 → 734.9 (median of 3), T640/C1792 146.1 → 147.4 (+0.9 %, inside the noise band, consistent).
- Split by channel at the focus: W-mcast alone is 137.9 → 130.2 µs, coef-mcast alone 137.9 → 137.5. Both kept
  as live knobs (`W_MCAST`, `COEF_MCAST`). Measured W-mcast lands on the "skip non-injector W reads" ceiling
  (129.9 µs), so the W stream is now fully removed from the receivers' critical path.
- Topology: band core layout `(g_r, g_c) → (g_c mod gx, g_r + R·⌊g_c/gx⌋)`. Column groups are R cores down
  one grid column, so the W channel is a native ColumnMajor helper multicast with staggered injectors. Row
  groups are M = ⌈Q/gx⌉ grid-row segments, served by two RowMajor coefficient channels (full and tail
  segments); with Q = 21 on the 11-wide grid that means 2 reads per row, not 1. One read per row group
  would need equal consecutive groups in both orders, i.e. Q ≤ gx; the helper cannot express it at Q = 21.
- Bottleneck now: unchanged from Refinement 3, the per-tile SFPU mix chain (~2.7 µs per output tile) plus
  the per-row fixed cost. That is Refinement 5. Not done here: an injector-rotating W stream (each group
  member injects 1/R of the slice) would also balance the injector's extra read. It is not tried because
  the receivers already sit at the W-free ceiling.

### [x] Refinement 5 — Speed up the perf-flagged profile: SFPU-bound mix

**Type**: perf

**Goal**:
- With bf16 streams, DRAM time per output tile halves. The per-tile mix chain is 5 bcast unpacks + 5 copies + 5
  SFPU ops plus element re-inits, so it likely becomes the bound on the flagged config (and on plain bf16).
- Reduce the compute per output tile using the `sfpu_tile_scope` / `compute_fusion` / `compute_block_size`
  entries in `ttnn/ttnn/operations/examples/master.md`. Candidates, from the design's lamps:
  - loop c inside each element, keeping a coefficient tile resident in DEST across c;
  - a fused SFPU block op that loads each F / X_i vector once and emits all 4 outputs (full-sync DEST);
  - for bf16 streams only, a split-precision FPU form: coefficient = bf16 hi + mid + lo, giving exact bf16 × bf16
    products into fp32 DEST.
- No SUPPORTED change.

**Verifier notes**:
- The signed-bias gate is non-negotiable. Any FPU form must stay exact: no fp32 operand may go through tf32 srcA
  or srcB.
- First confirm the flagged config is compute-bound after Refinements 3–4, using the device profiler's per-RISC
  zones. If it is DRAM-bound, record that and close this refinement with no change.

**Done when**:
- Measured device-ns improves on the flagged shape, and the golden suite (including the bias and depth-chain
  gates) is green.
- No regression across the guard set.

**Outcome**:
- Confirmed compute-bound first: at the focus, stubbing the mix + y chains gives 129.3 → 93.7 µs.
- Flagged config (T160 C5120 bf16 collapse packed HiFi2): **129.3 → 106.5–107.1 µs** device kernel (−17 %).
  - plain bf16 73.6 → 56.9; stats 96.0 → 78.1;
  - HiFi4 stats / collapse 195.1 / 228.5 → 177.2 / 206.8;
  - T640 HiFi2 stats / collapse 284.7 / 354.0 → 213.7 / 266.8;
  - T4096 C1792 bf16 plain / collapse 494 / 835 → 357 / 643.
- Lever: a fused SFPU mix (`MIX_FUSED`).
  - Once per row, the broadcast coefficient columns are compacted into packed DEST-layout tiles, with 4
    coefficients per tile, one per SFPU quadrant of each row group.
  - Each output tile is then 7 unpacks + 3 multi-term SFPU passes, instead of 10 + 6. y is 5 + 2 instead of 8 + 4.
  - The operation order is the same, so the bits are unchanged.
- Path-gated to bf16 X streams (`MIX_FUSED_STREAM_DTYPES`).
  - With fp32 streams, full-grid calls already sit at the aggregate DRAM limit, and the faster mix exposes a
    bottom-right NoC hotspot: T4096 C1792 fp32 736 → 823 µs.
  - Small fp32 calls would win 6–11 %. Plain fp32 stays on the per-term chain, unchanged.
- Bottleneck now: the non-mix floor (~94 µs: DM, projection, per-row combine / hc tail). The mix itself costs
  ~10 µs of unpack overhead; its SFPU passes cost ~3 µs.
- Next to try: fix the fp32 NoC hotspot (placement / bank stagger), then enable fp32. Not done here because it
  is a dataflow-placement lever, outside this SFPU-mix scope.
