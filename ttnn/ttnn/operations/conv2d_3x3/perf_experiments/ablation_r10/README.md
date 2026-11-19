# conv2d_3x3 ablation hooks, A/B runs and merge gate (Perf 10)

`ablation_hooks.patch` applies to `6981db2a54c` (the round-10 base). It is the round-9 patch re-cut for the Perf 9 tree:
- the `padded_tilize` stub follows its kh_lo / kh_hi signature;
- the N-slice weight hunk follows the polled ring credit (`receive_polling`);
- the `khx_drain` hunk is dropped, since the drain already serves the fill.

```bash
git apply ttnn/ttnn/operations/conv2d_3x3/perf_experiments/ablation_r10/ablation_hooks.patch
ttnn/ttnn/operations/conv2d_3x3/perf_experiments/ablation_r10/run_peel.sh <label> gather,fill,weights
git apply -R ttnn/ttnn/operations/conv2d_3x3/perf_experiments/ablation_r10/ablation_hooks.patch
```

It does not apply to the round's final tree: the tap shift and slab-direct paths moved the hooked prelude code.

Files:

- `peel_results.txt`: DEVICE KERNEL DURATION (us, zoned) per focus case, in `test_conv2d_3x3_perf_focus` order (see `../ablation_r8/README.md`).
  - `full` / `hooks_full` is the base. `x_matmul` stubs the matmul only.
  - The `dm_*` rows peel data movement first, cumulatively: gather, fill, weights, scatter, untilize, tilize, matmul.
- `zones_full.txt`, `zones_x_matmul.txt`, `zones_dm_all.txt`: `../zone_report.py` output. Per-core timelines come from `../ablation_r8/{timeline.py,endgrid.py,servers.py}`.
- `focus.sh <label>`: the 22 focus cases on the working tree, zoned, PCC asserted. `focus.txt` holds the combined-tree step after each graduation (`g1_ring` .. `g7_wearly`).
- `ab_unzoned.sh <rev> <label>`: the whole perf file with zones stripped at `<rev>`.
  - `ab_results.txt` holds three alternating base (`6981db2a54c`) / final (`da787eee785`) runs; `ab_results_pre_carve.txt` the same against `aeba2aed511`, before the early-fill carve-out.
  - `ab_report.py ab_results.txt ab_unzoned_ids.txt` prints the guard and focus tables.
- `ab_unzoned_ids.sh <rev> <label> <node ids...>`: chosen ids only. `ab_guard.txt` holds the H16_W16_Ci64 tile cells at every round-10 commit, two passes.
- `merge_gates/test_focus_exact.py`: `op_copy/` (final tree) vs `op_base/` (`6981db2a54c`) on all 22 focus cases. Run it with `--import-mode=prepend` and an absolute path. stem.conv1 is allowed to differ (fused bias).
