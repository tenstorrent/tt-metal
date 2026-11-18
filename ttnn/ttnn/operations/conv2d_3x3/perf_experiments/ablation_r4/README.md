# conv2d_3x3 ablation hooks (Perf 4)

`ablation_hooks.patch` applies to commit `20cb18c0947` (the round-4 base). It is the round-3 patch re-derived for the
strided-window tilize and the center-tap slots. Apply it to a scratch tree only; it is not part of the op.

```bash
git apply ttnn/ttnn/operations/conv2d_3x3/perf_experiments/ablation_r4/ablation_hooks.patch
CONV2D_3X3_PERF_ABLATION=1 CONV2D_3X3_ABLATE=gather,fill,weights \
  scripts/run_safe_pytest.sh --profile --run-all \
  "tests/ttnn/unit_tests/operations/conv2d_3x3/test_conv2d_3x3_perf.py::test_conv2d_3x3_perf_focus"
git apply -R ttnn/ttnn/operations/conv2d_3x3/perf_experiments/ablation_r4/ablation_hooks.patch
```

Tokens: `gather, fill, weights, scatter, untilize, tilize, matmul`, with the same meaning as in `../ablation_r2/README.md`.
Changes from round 3:
- `gather` also stubs the strided-window gather (the unit's address word then points at the unit itself) and the
  center-tap copies (`CENTER_READ`).
- `tilize` also stubs `window_tilize` and the center-slot tilize.

The first version of this patch emitted `if constexpr (false)` in both arms of the `gather` hook, so the tap-major,
row and strided-window gathers still ran under that token. The committed patch has the fix (`if constexpr (true)` in
the ablated arm). The measured round-4 peel, with the fix, is in `changelog.md` "## Perf 4".
