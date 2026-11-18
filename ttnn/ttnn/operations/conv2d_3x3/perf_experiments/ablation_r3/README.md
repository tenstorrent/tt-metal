# conv2d_3x3 ablation hooks (Perf 3)

`ablation_hooks.patch` applies to commit `0995ef092d3` (the round-3 base). It is the round-2 patch re-derived for the
narrow untilize, the split TILE fill and the polled weight receive. It also stubs the writer's half of the split TILE
fill. Apply it to a scratch tree only; it is not part of the op.

```bash
git apply ttnn/ttnn/operations/conv2d_3x3/perf_experiments/ablation_r3/ablation_hooks.patch
CONV2D_3X3_PERF_ABLATION=1 CONV2D_3X3_ABLATE=gather,fill,weights \
  scripts/run_safe_pytest.sh --profile --run-all \
  "tests/ttnn/unit_tests/operations/conv2d_3x3/test_conv2d_3x3_perf.py::test_conv2d_3x3_perf_focus"
git apply -R ttnn/ttnn/operations/conv2d_3x3/perf_experiments/ablation_r3/ablation_hooks.patch
```

Tokens: `gather, fill, weights, scatter, untilize, tilize, matmul`, with the same meaning as in `../ablation_r2/README.md`.
`fill` now also stubs the writer's split-fill reads. `untilize` stubs both the helper path and the narrow untilize.
The measured round-3 peel is in `changelog.md` "## Perf 3".
