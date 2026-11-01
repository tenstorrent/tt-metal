# Refinement 5: sync-only floor

- `run.sh <nodeid>...` - one `--profile` run of test node ids under `tests/ttnn/unit_tests/operations/layer_norm_tile/`, prints DEVICE KERNEL DURATION per op. `LN_KNOBS` works only through `test_perf_bench.py`.
- `inj.py [zone]` - zone timeline of every core that recorded `zone` (default: the injector's `reader_affine_send`).
- `starts.py` - per-core NCRISC kernel start offsets (launch skew).
- Sync-only peel: the `LN_ABLATE` hooks of `_coordinator/ablation_hooks.patch` (read, write, compute), re-applied by hand to the current kernels and reverted after. `LN_ABLATE_ALL` (every `kernel_main` returns at entry) gave the empty-program floor.

Numbers: `changelog.md`, Refinement 5.
