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

## Refinement 2 — Trim P's per-issuer fixed cost and co-tune issue order
- Date: 2026-11-27
- What was done:
  - Attributed P's issuer timeline first, using a zoned probe copy of the kernel (`probes/make_zoned_kernel.py`) and per-core Tracy timelines.
    - Each `noc_async_read` costs ~60–100 cycles to issue, even on I0's idle NoC. It is ~25 instructions, six of them NIU register stores, plus a command-buffer poll.
    - Each piece adds ~95 cycles. Setup before the first read is ~250 cycles.
    - BRISC starts 86–152 cycles after NCRISC on every core.
    - In dynamic NoC mode each RISC's read barrier waits on the NIU's shared response counter. So both issuers of a core end together, at BRISC's last read + ~300 cycles.
    - Before this refinement, P's critical path was BRISC's issue loop.
  - **Shipped `STATEFUL_READS` (default on).**
    - Per piece, `noc_async_read_one_packet_set_state` once, then `noc_async_read_one_packet_with_state` per row: 3 register stores instead of 6.
    - Compiled in only for placement pairs that have per-row runs whose shard row fits one packet (the ≤ 15-tile cap). I0/I3/I5 (merged 16-tile runs) build the plain kernel, byte-identical to before.
    - A 4-row piece now issues in ~180 cycles instead of ~330.
  - **Shipped `EARLY_FIRST_RUN="single_run"`.**
    - The walk's first run (page id, out offset, bytes, NoC: 4 O(1) per-core args) is issued right after the accessor is built, and the walk then skips it.
    - Enabled only when every issuer reads exactly one run. That run is then all of the issuer's work, and the walk is dropped.
    - `"on"` (every issuer) is a live knob. It measured +5–6% on P: an extra address resolution on an issue-bound RISC.
  - **Built, measured and parked as live non-default knobs**, all correct (bit-exact on all five placements):
    - `ISSUER_SPLIT="runs"` (mid-row cut with a start lag): P +2–10%; the cut adds a piece to both issuers.
    - `ISSUER_SPLIT="balanced"` (modeled row cut, and which RISC takes which half, `ISSUER_COST`): P +3% over stateful, I0 +22%, I3 +10%.
    - `PIECE_ORDER="stagger"` (bank_stagger): P +4–7%.
    - `PIECE_ORDER="local_last"`: P +6–10%.
    - Ablation: two same-NoC segments per issuer cost nothing (P 1409 = 1409). So the order itself is what loses, not segment overhead; the natural top-left start is fastest.
  - Kernel: explicit block-column origin arg, a NoC bitmask for barriers (any segment count), the stateful branch and the early-run branch. Host: issuers are lists of rectangles (segments); three new knobs plus the cost model are in the cache key.
  - Reused: the segment encoding, the NoC search and the kernel walk. Added: the knobs above, `has_row_runs` / `first_run_args` / `early_first_run`, `issuer_rects` / `order_rects`.
  - No CB, L1 or data-movement-budget change: same runs, same bytes.
- Perf (Blackhole, 1350 MHz, Tracy `DEVICE KERNEL DURATION`, medians of 3):
  - Fresh kernel cache, same session. `plain` is the Refinement 1 issue path; `default` is the new defaults:

    | case | plain (ns) | default (ns) | change |
    |---|---|---|---|
    | P | 1533 | 1408 | −8.2% |
    | I0 | 913 | 904 | −1% |
    | I3 | 2241 | 2199 | −1.9% |
    | I5 | 1653 | 1651 | 0% |
    | O1 | 2593 | 2558 | −1.4% |

  - Stateful alone on P in four other sessions: −7.1%, −6.9%, −9.0%, −9.4%.
  - `single_run` on the I cases in one session: I0 −5%, I3 −2.7%, I5 −2.8%. In the fresh-cache session it was within noise.
  - Golden `device_kernel_ns`, same-session control → new (two runs): P test_op 1829 → 1718 / 1768; P loose 1831 → 1737 / 1779; O1 3184 → 2994 / 2828; I0 998 → 913 / 930.
  - Refinement 1 recorded P at 1532 (Tracy) and 1782 / 1821 (golden).
- Bottleneck now: 53 of 64 P cores finish issuing >350 cycles before their last data lands, and end times cluster at ~1750 cycles regardless of launch time (71–529 cycles). P is link-bound on NoC1's westward row links (Refinement 1's busiest link: 48 tiles ≈ 816 cycles, fed below saturation), plus ~450 ns of dispatch launch skew. The late-launched west cores are still issue-bound.
- Accuracy achieved: bit-exact (PCC=1.0, relative RMS=0) on P, I0, I3, I5, O1. Every acceptance, A/B and golden call asserts exact equality.
- Golden test progress: the slice that was run passed 7/7: test_op P/I0/I3/I5/O1, P loose, and I0 placements_one_process (cache keys for I3/P/O1/P/I5). The full suite is the harness's run.
- Issues encountered:
  - The first `runs` split applied BRISC's lag to an empty issuer and handed I0's whole block to one RISC (+22%); fixed.
  - I3 measured +2% with the stateful branch compiled in but never taken, so the switch is path-gated on host.
  - The zoned probe's markers cost ~18 cycles each, and the compiler moves address math across them. Per-piece numbers from it are approximate.
- Tests added:
  - `test_g6_reshard_noc_policy_perf.py` gained 18 settings: `default`, `plain`, `stateful`, `runs_lag1-3`, `stagger`, `local_last`, `balanced`, `st_*` combinations, `early`, `st_early*`, `st_single`, `st_halved`, `zoned`, `zoned_stateful`. All are bit-exact (150/150).
  - `probes/make_zoned_kernel.py` + `probes/kernels/reader_zoned.cpp` (timing probe).
