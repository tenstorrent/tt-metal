# fabric_reduce_scatter_ttnn (ttnn.bringup.fabric_reduce_scatter, Python op)

- Source: `ttnn/ttnn/operations/examples/fabric_reduce_scatter` on `mstaletovic/mimo-v2-dp` @ `b45a923d73c`. New op: no
  TTNN source on this tree (no upstream tests to carry).
- Python: `ttnn.bringup.fabric_reduce_scatter` (PYTHON_OPS in `ttnn/ttnn/bringup/__init__.py`), planned with
  fabric_all_gather_ttnn's `plan()`.
- Used by: glm53_flash_d_p (tt/common.py:scatter_rows, tt/kda_attention.py, tt/experts_ag.py)
- Tests: `tests/unit/` (the op's own suite); model cases `tests/test_fabric_reduce_scatter_ttnn.py`.

## Changes

<!-- One entry per change, newest last:
### <short title>
- What: the change, and the switch or argument that turns it on (default = source behaviour).
- Why: the symptom it fixes or the feature it adds.
- Needed by: <model> <task>
- Files: <paths inside this folder>
-->

### Tests: fork test suite with glm53_flash_d_p cases (O.1, 2026-10-13)
- Model-case suite `tests/cases.py`, `tests/reference.py`, `tests/test_fabric_reduce_scatter_ttnn.py`: GLM's two calls (full-mesh snake ring, and axis 1), vs a float64 sum, plus determinism. No op change.
