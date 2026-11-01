# Perf 1 coordinator tools

- `ops.sh [csv]` - DEVICE KERNEL DURATION + per-RISC columns from the newest ops_perf CSV.
- `zones.py` - per-(RISC, zone) summed duration, critical cores vs rest, marker-count check.
- `timeline.py` - zone start/end timeline for the slowest and a typical core.
- `grid.py [zone]` - per-core grid of the first `reader_block` duration, block count and BRISC end.
- `ablation_hooks.patch` - the temporary payload-ablation hooks used for the Perf 1 peel
  (`LN_ABLATE=compute,read,write` env var -> `LN_ABLATE_*` defines). Not applied to the op; context
  predates clang-format, apply by hand.

Bench: `scripts/run_safe_pytest.sh [--run-all] --profile tests/ttnn/unit_tests/operations/layer_norm_tile/test_perf_bench.py`
(one call per LOOSE_CASES perf row; ablated runs fail correctness by design, so use `--run-all`).
