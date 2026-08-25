# Milestone A MLP2D Host Readiness Work Log

## Checkpoint 1: Scope and static audit

- Scope is limited to `models/common/modules/mlp/mlp_2d.py`, its MLP2D tests, and this log.
- TT hardware execution is explicitly excluded.
- Milestone A requires WH Galaxy `(8, 4)` validation, Llama `8192 -> 28672 -> 8192` and Qwen `5120 -> 25600 -> 5120` geometry, independent decode/prefill tuning, sequence-keyed prefill program factories, injected prefetch/CCL resources, preserved gated MLP dataflow, and no `from_model_args` dependency.
- Existing host coverage validates representative shapes, mesh/architecture/device count, weight shapes, independent mode dtype/kernel settings, prefill factories, collaborator injection, collective buffers, decode cleanup, and mode dispatch.
- Audit gaps to verify with tests: activation policy validation and selector behavior across decode/prefill tensor geometries.
- No hardware commands have been run.

## Checkpoint 2: Host baseline

- Command: `pytest -q models/common/tests/modules/mlp/test_mlp_2d.py`
- Result: `18 passed in 2.78s` (process wall time 7.45s).
- Static audit confirmed that a non-callable `collective_resource_selector` is treated as resolved and deferred to the hot path.
- Activation is configurable and used in both mode paths, but construction does not validate the value and prefill propagation is not covered by host tests.
- Prefill cutoff and custom program-config factories are also consumed in the hot path without construction-time validation.
- No hardware commands have been run.

## Checkpoint 3: Bounded implementation and coverage fixes

- Added construction-time validation for the activation enum, collective selector callable, prefill program-config factory callables, and positive prefill cutoff.
- Added selector tests for reduce-scatter, all-gather, and all-reduce across decode and prefill contexts, including runtime tensor forwarding.
- Added decode and prefill hot-path tests for SiLU and GELU activation propagation.
- Added prefill assertions for sequence-keyed program configs, mode-specific collective dispatch, and transient ownership.
- Command: `pytest -q models/common/tests/modules/mlp/test_mlp_2d.py`
- Result: `32 passed in 4.83s` (process wall time 9.55s).
- No hardware commands have been run.

## Checkpoint 4: Final host-only verification

- `python_env/bin/black --check models/common/tests/modules/mlp/test_mlp_2d.py`: passed; file unchanged.
- `python -m py_compile` for the MLP2D implementation, host test, and hardware test: passed.
- `git diff --check` for the scoped MLP2D files and this log: passed.
- `pytest --collect-only -q models/common/tests/modules/mlp/test_mlp_2d_wh_galaxy.py`: `4 tests collected in 0.03s`; no hardware fixture executed.
- The four collected hardware cases cover Llama and Qwen decode batch 32 repeat plus prefill lengths 128 and 2048 repeat.
- A whole-file Black check reports pre-existing formatting drift in `mlp_2d.py` and `test_mlp_2d_wh_galaxy.py`; broad formatting was intentionally not applied because those files contain shared in-progress work outside this bounded audit.
- Final executed host test result remains `32 passed in 4.83s`.
- No TT hardware commands, resets, or device tests were run.
