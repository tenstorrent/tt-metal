# Changelog: g6_reshard

## Phase 0 — Core Implementation
- **Date**: 2026-11-27
- **What was done**: Initial implementation via incremental pipeline (planner → implementer → verifier). PULL regime: each out-block holder pulls the pieces of in0 overlapping its block straight into its out shard. Two issuers per core, piece walk on device, maximal runs, one read barrier per issuer, no CB, no compute kernel, one `ttnn.generic_op` dispatch. The NoC per issuer comes from a route-model (`balanced`) policy.
- **SUPPORTED at Phase 0**: dtype=[bfloat8_b], layout=[TILE_LAYOUT], no shape-derived axes, every legal BLOCK_SHARDED L1 placement pair. `TARGET − SUPPORTED = ∅`.
- **Accuracy achieved**: bit-exact. PCC=1.000000, max_abs_err=0, mean_abs_err=0, relative RMS=0, max ULP=0, 0 bit mismatches, got/true ratio 1.0 everywhere. Measured on 4 placement pairs (P, I3, I5, O1) of the production shape via `test_g6_reshard_precision_baseline.py`.
- **Golden suite at Phase 0**: 9 / 9 registry cells passing (12 / 12 tests incl. 3 data-distribution regression tests). Verifier CLI: 0 supported_fail, 0 xpass_drift, 0 xfail_wrong_mode (`eval_results/g6_reshard_verifier/verifier_report.json`).
- **Perf at Phase 0 (P, Blackhole)**: 1637 ns Tracy `DEVICE KERNEL DURATION` (warm median); 1980–2041 ns golden `device_kernel_ns`; stock 1990 ns.
- **Issues encountered**:
  - Verifier fixes. `_build` now takes the issuer count from the plan, so the knob is read once per build.
  - Adopted the working-tree reader change that moves arg loads inside the `RESHARD-ISSUE` zone (BB-21).
  - Removed a broken untracked scratch test (`probes/test_floor_scratch.py`) whose `/tmp` kernel source no longer exists.
  - No drift fixes were needed.
- **Tests added**: `test_g6_reshard.py`, `test_g6_reshard_precision_baseline.py`, `test_g6_reshard_device_perf.py`, `test_g6_reshard_noc_policy_perf.py`

## Refinement 1 — Cut P's busiest NoC link
- Date: 2026-11-27
- What was done:
  - Checked the route model against the device first. Single-call A/B, then 5-call medians, device kernel ns. The model's busiest link ranks the per-issuer policies the same way the device does on P: fixed 104 → 2456, wrap_free / balanced 64 → ~1660, split/issuer 48 → ~1510. It does not rank the intra-issuer `cut` shape. `cut` lowers the modeled link to 40 but measures 1698 on P (no win) and 2007 on I5 (+20% vs 1671).
  - Ablation of the cut: an issuer that reads its rows as two segments on one NoC costs nothing (P 1501). A cross-NoC cut at the same modeled busiest link (48, wrap weight 2) costs +330 cycles of issue time per issuer (P 1671). So the cost is switching NoC inside one issuer, which the model (response bytes per link) cannot see.
  - Shipped the per-issuer p-norm search as the default: `SPLIT_SHAPE` changed from `cut` to `issuer`. P's busiest link drops from 64 to 48. The greedy `balanced` policy had stopped at 64 in a local optimum, so per-issuer choice was not floored there. The search stays at 48 from 12 random starts.
  - `cut` stays a live, non-default knob. Added `SPLIT_WRAP_WEIGHT` (default 1.0, plan unchanged, in the cache key). It weights wrapped routes in the split search. At 2.0 it stops the I5 cut regression, but it also drops the wrapped single-issuer moves that win on P.
  - No kernel change, no CB change, and the L1 ledger and data-movement budget are unchanged.
  - Reused: the existing split search, segment encoding and kernel. Added: the default flip, the wrap-weight knob, and perf-test repeat/filter env vars with three new settings. Also added host-only device-log and A/B summarizers under `probes/`.
- Perf (Blackhole, 1350 MHz):
  - Fresh kernel cache, 5-call medians, Tracy `DEVICE KERNEL DURATION`, Phase-0 `balanced` → new default:

    | case | before (ns) | after (ns) | change |
    |---|---|---|---|
    | P | 1641 | 1532 | −6.6% (two earlier sessions: −9% to −10%) |
    | O1 | 2845 | 2688 | −5.5% |
    | I0 | 930 | 933 | unchanged |
    | I3 | 2209 | 2202 | unchanged |
    | I5 | 1693 | 1686 | unchanged |

    I0, I3 and I5 plans are identical to Phase 0's.
  - Golden `device_kernel_ns`, same-session control → new default: P test_op 1996 → 1782 (−10.7%), P loose 1942 → 1821 (−6.2%), O1 3087 → 2833 (−8.2%).
  - Stock is 1990 ns.
- Accuracy achieved: bit-exact (PCC=1.0, relative RMS=0) on P, I0, I3, I5, O1. Every acceptance and A/B call asserts bit equality.
- Golden test progress: the slice that was run passed 4/4: P test_op, P loose, O1 test_op, and the I0 placements_one_process loose case, which covers the I3/P/O1/P/I5 cache keys. The full suite is the harness's run.
- Issues encountered: the tree's default before this refinement was `split`/`cut`, which regressed I5 (+20%) and did not beat `balanced` on P. Launch skew (~600–750 cycles, dispatch side) moves single-call P numbers by ±4%, so every decision used 5-call medians. The device log reports physical NoC columns (2 harvested Tensix columns), while the model uses translated ones. Column order is the same in both, so modeled link maxima are unchanged.
- Tests added: no new test file. `test_g6_reshard_noc_policy_perf.py` gained `G6_PERF_REPEATS` / `G6_PERF_PLACEMENTS` / `G6_PERF_SETTINGS` and three settings: `split_issuer2_halved` (same-NoC segment ablation), `split_issuer2_w2` and `split_cut2_w2`.
