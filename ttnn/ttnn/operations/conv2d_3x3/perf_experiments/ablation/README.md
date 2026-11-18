# conv2d_3x3 ablation hooks (Perf 1)

`ablation_hooks.patch` adds payload stubs gated by kernel defines, driven by an env var read in the program
descriptor. Apply it to a scratch tree only; it is not part of the op.

```bash
git apply ttnn/ttnn/operations/conv2d_3x3/perf_experiments/ablation/ablation_hooks.patch
CONV2D_3X3_PERF_ABLATION=1 CONV2D_3X3_ABLATE=gather,matmul \
  scripts/run_safe_pytest.sh --profile --run-all \
  "tests/ttnn/unit_tests/operations/conv2d_3x3/test_conv2d_3x3_perf.py::test_conv2d_3x3_perf_focus"
git apply -R ttnn/ttnn/operations/conv2d_3x3/perf_experiments/ablation/ablation_hooks.patch
```

Tokens (comma-separated, cumulative peel = list several):
- `gather`  — reader row / tap-major gather: no reads, no address math; reserve / trid / push kept.
- `matmul`  — `SKIP_COMPUTE` on the compute kernel (matmul_block's LLK call; packs and syncs kept).
- `weights` — writer weight K-blocks: no DRAM read, no multicast send or receive; reserve / push kept.
- `scatter` — R4 partial scatter writes (writer and reader halves); credits, arrivals, CB counts kept.
- `fill`    — TILE slab-fill reads; reserve / push kept.

`CONV2D_3X3_PERF_ABLATION=1` skips the PCC check in the perf harness (the output is wrong by design).
Zone breakdowns: `perf_experiments/zone_report.py generated/profiler/.logs/profile_log_device.csv`.
