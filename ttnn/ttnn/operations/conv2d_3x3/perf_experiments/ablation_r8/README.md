# conv2d_3x3 ablation hooks and merge gates (Perf 8)

`ablation_hooks.patch` applies to `3ca4eb28f46` (the round-8 base, end of Perf 7). It is the round-7 patch with its
kw-fold hunk re-cut for the column-round walk (the `matmul` token empties kw-fold's `matmul_block` c loop). It does not
apply to the round's final tree (the n-group passes, padded slab and balanced split moved the hooked code).

```bash
git apply ttnn/ttnn/operations/conv2d_3x3/perf_experiments/ablation_r8/ablation_hooks.patch
ttnn/ttnn/operations/conv2d_3x3/perf_experiments/ablation_r8/run_peel.sh <label> gather,fill,weights
git apply -R ttnn/ttnn/operations/conv2d_3x3/perf_experiments/ablation_r8/ablation_hooks.patch
```

- `peel_results.txt`: DEVICE KERNEL DURATION (us, zoned) per focus case in `test_conv2d_3x3_perf_focus` order (stem1,
  stem2, stem3, res2, res3.0, res3.x, res4.0, res4.x, res5.0, res5.1, res5.2, aspp1, aspp2, aspp3, sem dec res3 fc0,
  inst dec res3 fc0, sem dec res3 fc1, sem dec res2 fc0, inst dec res2 fc0, sem res2 fc1, inst res2 fc1,
  center_head.1). `dm_*` rows peel data movement first, `x_matmul*` compute first, `weights_only` stubs everything but
  the weight stream; `g1_server` .. `g6_balanced` are the full (unablated) focus set after each graduation.
- `zones_full.txt`, `zones_x_matmul.txt`, `zones_dm_..._matmul.txt`, `zones_final.txt`: `../zone_report.py` output.
- `timeline.py <profile_log_device.csv> <case index>` (one core's zones), `servers.py <csv> <case index>` (the weight
  servers' zones), `endgrid.py`, `zone_compact.py` (one line per case).
- `ab_unzoned.sh <rev> <label>`: the whole perf file with zones stripped at `<rev>`; `ab_results.txt` holds three
  alternating base (`3ca4eb28f46`) / final (`faf15c942a9`) pairs in `ab_unzoned_ids.txt` order; `ab_report.py` turns
  them into the guard and focus tables. `ab_unzoned_ids.sh <rev> <label> <node ids...>` does the same for chosen ids
  (`ab_guard_bisect.txt`: the two H16_W16_Ci64 interleaved cells at every round-8 commit, two passes).
- `merge_gates/`: the bit-exact checks run on the merged tree. Populate `op_copy/` (the tree under test) and `op_base/`
  (a reference rev) with the op files first. `test_npl_exact.py` (kw-fold focus cases), `test_block_exact.py` (BLOCK
  focus cases), `test_prec.py` (HEIGHT decoder layers, PCC / MAE / identical fraction), `test_tdcp_bitexact_merged.py`
  (padded slab on / off on the tree in `op_copy/`, path detected from the kernel defines).
