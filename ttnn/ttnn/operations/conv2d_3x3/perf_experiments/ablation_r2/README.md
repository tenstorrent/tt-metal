# conv2d_3x3 ablation hooks (Perf 2)

`ablation_hooks.patch` applies to commit `c33f4181192` (the round-2 base). Later commits move the gather, untilize and
writer code, so re-derive it before reusing it on a newer tree. It adds payload stubs gated by kernel defines, which an
env var read in the program descriptor turns on. Apply it to a scratch tree only; it is not part of the op.

```bash
git apply ttnn/ttnn/operations/conv2d_3x3/perf_experiments/ablation_r2/ablation_hooks.patch
CONV2D_3X3_PERF_ABLATION=1 CONV2D_3X3_ABLATE=gather,fill,weights \
  scripts/run_safe_pytest.sh --profile --run-all \
  "tests/ttnn/unit_tests/operations/conv2d_3x3/test_conv2d_3x3_perf.py::test_conv2d_3x3_perf_focus"
git apply -R ttnn/ttnn/operations/conv2d_3x3/perf_experiments/ablation_r2/ablation_hooks.patch
```

Tokens (comma-separated; list several for a cumulative peel):
- `gather`: the reader's row / tap-major gather and the writer's split-gather half. The reads and address math are
  removed; reserve, trid and push are kept.
- `fill`: the TILE slab-fill reads. Reserve and push are kept.
- `weights`: the writer's weight DRAM reads and its multicast send / receive, on both the K-blocked and N-slice paths.
  Reserve and push are kept.
- `scatter`: the R4 partial / slice sends, both halves. Credits, arrivals and CB counts are kept.
- `untilize`: compute's slab untilize, replaced by the same per-tile-row CB wait / reserve / push / pop.
- `tilize`: compute's im2col tilize, the same way.
- `matmul`: `SKIP_COMPUTE` on the compute kernel. This removes `matmul_block`'s LLK call only; the packs stay.

`CONV2D_3X3_PERF_ABLATION=1` skips the PCC check in the perf harness, because the output is wrong by design. Zone
breakdowns: `perf_experiments/zone_report.py`. The measured round-2 peel is in `changelog.md` "## Perf 2".
