# conv2d_3x3 ablation hooks, A/B runs and merge gate (Perf 13)

`ablation_hooks.patch` applies to `4f2f03f6d20` (the round-13 base). It is the round-12 patch re-cut by three-way merge,
plus stubs the round-12 patch lacked: the stride-2 DEST tap build (`tilize`, `conv2d_3x3_s2_taps.hpp`), the kw-fold
host-planned fill on both RISCs (`fill`) and the BLOCK one-read weight rounds (`weights`). It does not apply to the
round's final tree.

```bash
git apply ttnn/ttnn/operations/conv2d_3x3/perf_experiments/ablation_r13/ablation_hooks.patch
ttnn/ttnn/operations/conv2d_3x3/perf_experiments/ablation_r13/run_peel.sh <label> gather,fill,weights
git apply -R ttnn/ttnn/operations/conv2d_3x3/perf_experiments/ablation_r13/ablation_hooks.patch
```

Files:
- `peel_results.txt`: zoned DEVICE KERNEL DURATION (us) per focus case in `test_conv2d_3x3_perf_focus` order (base).
- `zones_full.txt`, `zones_dm_scatter.txt`, `zones_x_matmul.txt`, `zones_final.txt`: `../zone_report.py` output (base,
  base with all DM stubbed, base with everything stubbed, final).
- `timeline.py <csv> <run index> [x,y]`: one core's zone timeline (default: the core that ends last).
- `focus.txt`: zoned focus runs. `ab_focus_round0.txt`: N-slice round 0 on / off on the host-fill tree (`ab_focus.sh`,
  zones stripped on the working tree).
- `ab_results.txt` (+ `ab_unzoned_ids.txt`, `ab_report.py`): whole perf file, base `4f2f03f6d20` / final
  `9969908e7f9`, 3 alternating pairs, zones stripped.
- `merge_gates/test_focus_exact.py`: `op_copy/` (final tree) vs `op_base/` (`4f2f03f6d20`) on all 22 focus cases; every
  case must be bit-identical. Run with `--import-mode=prepend` and an absolute path.
- Scripts: `focus.sh`, `run_peel.sh`, `ab_unzoned.sh`, `ab_unzoned_ids.sh`, `ab_focus.sh`, `zone_hash_check.py`
  (outputs under `/tmp/abl13/`).
