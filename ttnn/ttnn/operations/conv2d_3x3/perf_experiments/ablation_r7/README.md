# conv2d_3x3 ablation hooks (Perf 7)

`ablation_hooks.patch` applies to `e40c435b625` (the round-7 base, end of Perf 6). It is the round-6 patch rebased onto
the Perf 6 graduations, plus hooks the round-6 patch could not have:

- `weights` also stubs the weight servers: both server kernels return at once and receivers skip the flag wait;
- `gather` also stubs the K-blocked tap-direct FIFO (`TapDirectFifo::unit`): every tap entry points at the zero block;
- `tilize` also stubs `fifo_tilize`, and the `direct_tilize` stub skips the N-slice center tap (`skip_t`). Without that
  skip the stub over-pushes and res4.x hangs; the first `..._tilize` peel row was lost to that hang, and the first
  `..._matmul` row ran with the bad stub (re-run as `x_gather_..._matmul`);
- `matmul` also empties kw-fold's raw `matmul_block` c loop (`SKIP_COMPUTE` does not reach it).

kw-fold's pack stream, final pass and reader seed have no hook: with everything stubbed, the kw-fold cases keep them.

```bash
git apply ttnn/ttnn/operations/conv2d_3x3/perf_experiments/ablation_r7/ablation_hooks.patch
ttnn/ttnn/operations/conv2d_3x3/perf_experiments/ablation_r7/run_peel.sh <label> gather,fill,weights
git apply -R ttnn/ttnn/operations/conv2d_3x3/perf_experiments/ablation_r7/ablation_hooks.patch
```

- `peel_results.txt`: DEVICE KERNEL DURATION (us, zoned) per focus case in `test_conv2d_3x3_perf_focus` order:
  stem1, stem2, stem3, res2, res3.0, res3.x, res4.0, res4.x, res5.0, res5.1, res5.2, aspp1, aspp2, aspp3,
  sem dec res3 fc0, inst dec res3 fc0, sem dec res3 fc1, sem dec res2 fc0, inst dec res2 fc0, sem res2 fc1,
  inst res2 fc1, center_head.1. `dm_*` rows peel data movement first, `x_matmul*` rows compute first; `g1`..`g3d` are
  the full (unablated) focus set after each graduation step.
- `zones_*.txt`: `../zone_report.py` output for the full, matmul-stubbed and everything-stubbed runs.
- `timeline.py <profile_log_device.csv> <case index>`: one core's zone start / end (us from the run's first kernel
  start); `endgrid.py`: every core's kernel end as a grid.
- `ab_unzoned.sh <rev> <label>`: the whole perf file with zones stripped at `<rev>`; `ab_results.txt` holds the
  round's base / final pairs in `ab_unzoned_ids.txt` order.
