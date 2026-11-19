# conv2d_3x3 ablation hooks and merge gates (Perf 9)

`ablation_hooks.patch` applies to `a51305feccb` (the round-9 base after its first writer fix). It is the round-8 patch
re-cut for the Perf 8 tree:

- `matmul` empties the n-group kw-fold `subblock_body` c loop and the balanced walk's `group_mm`;
- `tilize` also covers `padded_tilize` and counts `prelude_rows`;
- `untilize` also covers the padded slab, including its pushed tail;
- `weights` makes server receivers treat the server-multicast bias as landed and serves the split fill in the halo
  exchange loops.

It does not apply to the round's final tree: the padded tilize changed signature.

```bash
git apply ttnn/ttnn/operations/conv2d_3x3/perf_experiments/ablation_r9/ablation_hooks.patch
ttnn/ttnn/operations/conv2d_3x3/perf_experiments/ablation_r9/run_peel.sh <label> gather,fill,weights
git apply -R ttnn/ttnn/operations/conv2d_3x3/perf_experiments/ablation_r9/ablation_hooks.patch
```

Files:

- `peel_results.txt`: DEVICE KERNEL DURATION (us, zoned) per focus case, in `test_conv2d_3x3_perf_focus` order (see
  `../ablation_r8/README.md`).
  - `full` is the base. `x_matmul` stubs the matmul only.
  - `dm_*` peel data movement first, cumulatively: gather, fill, weights, scatter, untilize, tilize, matmul
    (`dm_all` = everything stubbed).
  - `final_zoned` is the round's final tree.
- `zones_full.txt`, `zones_x_matmul.txt`, `zones_dm_all.txt`, `zones_final.txt`: `../zone_report.py` output.
  The per-core timelines came from `../ablation_r8/timeline.py`.
- `ab_unzoned.sh <rev> <label>`: the whole perf file with zones stripped at `<rev>`.
  - `ab_results.txt` holds three alternating base (`587aef8ec99`) / final (`f7f6b1a5b35`) runs, in `ab_unzoned_ids.txt`
    order.
  - `ab_report.py ab_results.txt ab_unzoned_ids.txt` prints the guard and focus tables.
  - `ab_unzoned_ids.sh <rev> <label> <node ids...>` runs chosen ids only. `ab_guard.txt` holds the H16_W16_Ci64 tile
    cells at every round-9 commit, two passes.
- `merge_gates/test_focus_exact.py`: the final tree vs `587aef8ec99` on all 22 focus cases, bit for bit. Populate
  `op_copy/` and `op_base/` with the op files first, then run with `--import-mode=prepend` and an absolute path.
