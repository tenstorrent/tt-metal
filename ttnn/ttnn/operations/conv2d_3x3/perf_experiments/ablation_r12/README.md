# conv2d_3x3 ablation hooks, A/B runs and merge gate (Perf 12)

`ablation_hooks.patch` applies to `739a271172e` (the round-12 base). It is the round-11 patch re-cut by three-way
merge (descriptor define lists, the tap-shift stub, the writer weight-send hunks), plus two stubs the round-11 patch
lacked: the kw-fold reader fill (rewritten in Perf 11, so `fill` left it running) and the spare-column forward under
`weights`. It does not apply to the round's final tree.

```bash
git apply ttnn/ttnn/operations/conv2d_3x3/perf_experiments/ablation_r12/ablation_hooks.patch
ttnn/ttnn/operations/conv2d_3x3/perf_experiments/ablation_r12/run_peel.sh <label> gather,fill,weights
git apply -R ttnn/ttnn/operations/conv2d_3x3/perf_experiments/ablation_r12/ablation_hooks.patch
```

Files:
- `peel_results.txt`: zoned DEVICE KERNEL DURATION (us) per focus case in `test_conv2d_3x3_perf_focus` order. `hooks_full`,
  the cumulative `dm_*` rows and `x_matmul` are from the first pass; the `dm_*2`, `x_fill` and `x_weights` rows are
  after the kw-fold fill stub was added (use these for kw-fold cases).
- `zones_base.txt`, `zones_dm_all.txt`, `zones_final.txt`: `../zone_report.py` output (base, base all-stubbed, final).
- `focus.txt`: zoned focus runs (`base_zoned`, `final_zoned`).
- `ab_results.txt` (+ `ab_unzoned_ids.txt`, `ab_report.py`): whole perf file, base `739a271172e` / final, 3 alternating
  pairs, zones stripped.
- `merge_gates/test_focus_exact.py`: `op_copy/` (final tree) vs `op_base/` (`739a271172e`) on all 22 focus cases; every
  case must be bit-identical. Run with `--import-mode=prepend` and an absolute path.
- Scripts: `focus.sh`, `run_peel.sh`, `ab_unzoned.sh`, `ab_unzoned_ids.sh`, `zone_hash_check.py` (outputs under
  `/tmp/abl12/`). Under `--profile`, pass node ids, not `-k`.
