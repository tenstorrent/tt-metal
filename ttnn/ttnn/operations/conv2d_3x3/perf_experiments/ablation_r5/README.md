# conv2d_3x3 ablation hooks (Perf 5)

`ablation_hooks.patch` applies to commit `2ac42333eb2` (the round-5 base). It is the round-4 patch re-derived for the
round-4 graduations. Apply it to a scratch tree only; it is not part of the op.

```bash
git apply ttnn/ttnn/operations/conv2d_3x3/perf_experiments/ablation_r5/ablation_hooks.patch
CONV2D_3X3_PERF_ABLATION=1 CONV2D_3X3_ABLATE=gather,fill,weights \
  scripts/run_safe_pytest.sh --profile --run-all \
  "tests/ttnn/unit_tests/operations/conv2d_3x3/test_conv2d_3x3_perf.py::test_conv2d_3x3_perf_focus"
git apply -R ttnn/ttnn/operations/conv2d_3x3/perf_experiments/ablation_r5/ablation_hooks.patch
```

Tokens: `gather, fill, weights, scatter, untilize, tilize, matmul`, with the same meaning as in `../ablation_r2/README.md`.
Changes from round 4:
- `untilize` also stubs the ring-slab untilize (it keeps the ring's tail pops) and the wide untilize.
- `tilize` also stubs the tap-direct `direct_tilize`.
- `gather` also stubs the tap-direct gather: every tap entry of `cb_src_table` points at the unit's own slot.
- `fill` also stubs the ring-slab fill reads on both RISCs.

Every token was checked solo: each one drops its own stage's zone to ~0 on its own run. The first version of the
`untilize` hook emitted `if constexpr (false) {} else` ahead of the real untilize, so the real untilize still ran
after the stub and the program hung (the round-4 trap again, with the arms swapped). The committed patch has
`if constexpr (true)`.

`zone_timeline.py <profile_log_device.csv> <run id> <x,y>` prints one core's zone start/end times (us from the
core's first marker); `../zone_report.py` gives per-zone p50 / max over cores. The measured round-5 peel is in
`changelog.md` "## Perf 5".
