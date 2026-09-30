# E1: pipeline-aware group-width selection for bf16 X (mhc_pre, perf tournament round 2)

This is a host-only experiment. The real op and its kernels run unchanged, and only `make_plan`'s bf16-X group-width choice
changes. Numbers come from BH p150 (11x10 grid) with bf16 X and fp32_dest_acc_en=True. They are in-process
DEVICE KERNEL DURATION values, reported as the median of 3.

## Rule (graduation.patch -> mhc_pre_program_descriptor.py)
The candidates are every `group_w` in `min(grid_x, Ct)..1` whose `fit(group_w, 1)` fits L1. The same `fit`, the same
`group_h = 1`, and the same depth set (2, then 1) are used as before. The pick is
`min(fits, key=(depth < 2 and blocks > 1, _block_schedule_cost))`, and a tie goes to the widest.

The cost is measured in X-tile reads, where one tile read ≈ 0.55 us for one core under full-grid contention:

    H    = 32 + 0.75 * group_cores (+ 4 with a bf16 W: its projection is one unstreamed window)
    cost = blocks*kmax + H + kmax/n  (- 12 if every group has <= 1 token tile-row: owner discount absorbs the Sinkhorn)
         + sum over steps b < blocks-1 of
             pipe_at(b): max(0, H - 0.4*kmax)   # S(b+1)'s round trip hides under tail(b)
             serial    : max(0, H - 0.6*kmax)   # S(b)'s round trip hides under the prefetch of X(b+1) minus tail(b)

Why each term is there:
- **Stream term, blocks*kmax.** A rank's X blocks cross the NoC back to back. Idle cores and narrow groups show up here as a
  larger `blocks*kmax`. Group widths of 2 or 3 at small T leave 30-60 % of the cores idle, and the old rule picked them.
- **Last block, H + kmax/n.** The last block's round trip is always exposed: partial, then gather, then the rank-ordered fold,
  then mcast, then coef, plus the owner's Sinkhorn. Its y-mix and y write are exposed too, and y is 1/n of X.
  The zones for 640x1792 back this up. The root fold is 2 us at G=5 and 5 us/block at G=11, which is the per-rank term.
  The Sinkhorn takes about 7 us.
- **Middle steps.** The pipeline (`pipe_at`, which mirrors the kernels) only hides a round trip if the block carries
  enough K per rank. With a small `kmax` (C <= 2560 at full width) every extra block costs its round trip. That is why
  C <= 2560 stays at w=5, and why C >= 5120 goes to the full row with 2x the blocks.
- **The depth-1 exclusion.** At depth 1, X(b+1) cannot be read before block b is freed, so nothing overlaps. Refinement 4's
  choice of w=5 at C >= 6144 fell to depth 1, and so did its w=2/3 picks at C = 2560-5120.

The constants come from a decision fit over the all-width sweep (`analysis/`). The picks are unchanged across
RT_TILES_BASE 30-34 × TAIL_FRAC 0.35-0.45 (fp32 W).

## Results (Refinement 4 rule -> this rule, median of 3; full tables in data/final_tables.md)
- fp32 W: 640x7168 145.4 -> 128.2 (-11.8 %). 640x1792 and 1280x4096 keep the same plan (43.9/43.6, 138.7/138.6).
  32 of the 42 narrow-group cells are 2.5-55 % faster; for example, 1024x1792 goes 152.1 -> 68.1 and 2048x4096 goes 315.8 -> 241.6.
  1280x5120 is flat (176.2 -> 175.4; over 9 interleaved samples +1.6 %, within noise). 9 cells keep the same plan.
  T = 256 (Mt < grid_y) never reaches this code: 6 cells, same plan.
- bf16 W: large wins (1024x4096 -36 %, 2048x1792 -36 %, 2048x6144 -16 %), and the other cells are within ±1.5 %.
  **There is one exception.** 4096x4096 bf16 W goes 455.9 -> 471.0 (+3.3 %, interleaved 3x3). Refinement 4 picked w=2 at depth 1
  there, which the prefetch exclusion rules out, and the best depth-2 plan (w=5) is 3 % slower.
- Golden: `eval/golden_tests/mhc_pre/` with `-p gw_grad_plugin` passes 206/206. The unit tests test_mhc_pre and _blocking pass 24/24.
  The bench GW_CHECK run with golden tolerances is ok on 10 changed-plan cells × both W dtypes, including T=333.
- X_BLOCK_DEPTH_DEFAULT = 3 was measured as part of the sweep (variants w11d3, w5d3, w3d3). The result is mixed:
  +10 % at 1280x6144 and 2560x7168, and -5 to -7 % at 1024x5120, 4096x5120 and 1280x4096. Depth is left as it is.

## Files
- `gw_plan.py`: make_plan with a pluggable selector (default, forced wN[dD], cand).
- `bench/test_gw.py` + `bench/run.sh`: perf and correctness bench. `test_grad_matches_bench` asserts that grad/ picks the bench plan on 108 cells × 2 W dtypes.
- `bench/gw_grad_plugin.py`, `bench/grad_loader.py`: run the real op with grad/ (the patched descriptor and the real kernels).
- `grad/mhc_pre_program_descriptor.py`: the real descriptor with graduation.patch applied.
- `data/`: raw sweeps, final A/B runs, and zone reports. `analysis/`: the fit scripts, which read /tmp copies of data/.

Reproduce:

    GW_SHAPES=640x7168,1280x4096 GW_VARIANTS=default,cand GW_REPEAT=3 bench/run.sh perf
    PYTHONPATH=<this dir>/bench scripts/run_safe_pytest.sh --run-all eval/golden_tests/mhc_pre/ -p gw_grad_plugin
