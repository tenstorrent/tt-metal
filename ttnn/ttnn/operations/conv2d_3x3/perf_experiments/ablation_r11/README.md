# conv2d_3x3 ablation hooks, A/B runs and merge gate (Perf 11)

`ablation_hooks.patch` applies to `039e18f6c81` (the round-11 base). It is the round-10 patch re-cut by three-way
merge, plus two stubs the round-10 paths lacked: the tap shift (`tilize`, `conv2d_3x3_tap_shift.hpp`) and the fused
side rounds' matmul (`matmul`, `conv2d_3x3_kw_fold.hpp`). One line in the server writer's hunk is a comment that only
breaks a profiler zone-hash collision. It does not apply to the round's final tree.

```bash
git apply ttnn/ttnn/operations/conv2d_3x3/perf_experiments/ablation_r11/ablation_hooks.patch
ttnn/ttnn/operations/conv2d_3x3/perf_experiments/ablation_r11/run_peel.sh <label> gather,fill,weights
git apply -R ttnn/ttnn/operations/conv2d_3x3/perf_experiments/ablation_r11/ablation_hooks.patch
```

Files:
- `peel_results.txt`: zoned DEVICE KERNEL DURATION (us) per focus case in `test_conv2d_3x3_perf_focus` order; `hooks_full`, `x_matmul` and the cumulative `dm_*` rows.
- `zones_base.txt`, `zones_dm_all.txt`, `zones_final.txt`: `../zone_report.py` output (base, base all-stubbed, final tree).
- `zone_hash_check.py`: hashes every `MaybeDeviceZoneScope` location the way the profiler does (16-bit) and reports collisions; run it after any edit that shifts line numbers.
- `base_runs.txt`: the first zoned / unzoned base focus runs.
- `ab_results.txt` (+ `ab_unzoned_ids.txt`, `ab_report.py`): whole perf file, base `039e18f6c81` / final `cc1b40769bd`, 3 alternating pairs.
- `focus_only_ab.txt`: focus file alone, 3 alternating pairs (`fbase*` / `ffinal*`).
- `bisect_runs.txt`: per-commit runs: `p1_` / `p2_` / `p3_` focus-only passes, `w2_` / `w3_` whole-file passes, `alone_` (stem.conv2 alone) and `afterblk_` (BLOCK perf tests then stem.conv2).
- `focus.txt`: zoned focus runs (`base_zoned`, `final_zoned`).
- `merge_gates/test_focus_exact.py`: `op_copy/` (final tree) vs `op_base/` (`039e18f6c81`) on all 22 focus cases; every case must be bit-identical. Run with `--import-mode=prepend` and an absolute path.
- Scripts: `focus.sh`, `run_peel.sh`, `ab_unzoned.sh`, `ab_unzoned_ids.sh` (outputs under `/tmp/abl11/`). Under `--profile`, pass node ids, not `-k`: the tracy wrapper loses the `-k` quoting and collects nothing.
