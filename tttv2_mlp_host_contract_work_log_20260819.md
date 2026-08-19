# MLP2D Host Contract Work Log

## Checkpoint 1: Coverage review

- Reviewed `MLP2D._double_matmul_reduce_scatter_axis1` and the existing decode host test.
- Found that the decode test mocks the fused helper, leaving its fused-output ordering and decode resource lookup key uncovered.
- Added a hardware-free helper contract test that supplies an input whose `shape` property raises, verifies selection with the static `(1, 1, max_batch_size, hidden_dim // 8)` geometry, checks semaphore/resource forwarding, and asserts the fused `(projection, w3_reduced, w1_reduced)` result is returned as `(w1_reduced, w3_reduced)` while only the projection is deallocated.

## Checkpoint 2: Focused verification

- Ran `pytest -q models/common/tests/modules/mlp/test_mlp_2d.py -k 'decode_fused_matmul_uses_static_resource_key_and_preserves_output_order'`.
- Result: `1 passed, 32 deselected in 0.28s`.
- The test is host-only: all TTNN operations that could dispatch work are mocked, and no hardware reset or device fixture was used.

## Checkpoint 3: Scope and diff validation

- `git diff --check` passed for the two allowed files.
- Changed files are `models/common/tests/modules/mlp/test_mlp_2d.py` and this work log.
- The test file contained pre-existing modifications; this goal added only `test_decode_fused_matmul_uses_static_resource_key_and_preserves_output_order`.
