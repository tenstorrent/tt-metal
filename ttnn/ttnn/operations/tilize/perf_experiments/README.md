# tilize perf experiments

Perf tournaments' isolated experiments. One dir per idea; each carries a README with its numbers
and a generator script for its kernel-dir variants (`kernels_*`, not committed — regenerate them;
`scatter_offload/kernels_so/` is hand-authored and committed). The device harnesses live in
`tests/ttnn/unit_tests/operations/tilize/test_tilize_perf1_<idea>.py` (pytest files under
`ttnn/ttnn/...` fail to import: double op registration) and are opt-in:
`TILIZE_PERF_EXPERIMENTS=1 scripts/run_safe_pytest.sh --profile <harness>`.

Per-stage zones (`MaybeDeviceZoneScope`) record only with `TT_METAL_KERNEL_PERF_ZONES=1`.

| round | dir | idea | verdict |
|---|---|---|---|
| Perf 1 | `breakdown/` | measured breakdown: zones + cumulative ablation | — |
| Perf 1 | `write_throttle/` | cap un-ACKed writes / in-flight reads | NULL / REGRESSION |
| Perf 1 | `posted_writes/` | posted tile writes, set_state issue | NULL (posted = incorrect) |
| Perf 1 | `noc_region/` | static per-region NoC swap, tail row rebalance | REGRESSION |
| Perf 1 | `bank_paired_writes/` | coalesce same-bank output tiles into one write | REGRESSION (bound) |
| Perf 1 | `scatter_offload/` | BRISC takes part of the loopback scatter | NULL / REGRESSION |
