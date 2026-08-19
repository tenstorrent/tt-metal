# Milestone A Attention2D WH Galaxy Audit

## Scope

- Dedicated goal: audit only the Milestone A `Attention2D` implementation and WH Galaxy hardware tests.
- Host inspection and host-only pytest collection/execution only.
- No TT hardware command, device fixture execution, or `tt-smi` reset was performed.
- No production or test source file was edited.

## Verification

- `pytest --collect-only -q models/common/tests/modules/attention/test_attention_2d_wh_galaxy.py`
  collected 2 hardware cases.
- `pytest --collect-only -q models/common/tests/modules/attention/test_attention_2d.py`
  collected 60 host cases.
- `pytest -q models/common/tests/modules/attention/test_attention_2d.py` passed: `60 passed in 9.11s`.
- `python -m py_compile` passed for the implementation, host test, and WH Galaxy test.

## Exact Runnable Hardware Cases

The hardware file contains one test function parameterized over two model geometries:

1. `wormhole_b0-llama-70b-mesh_device0-device_params0`
   - Geometry: dim 8192, 64 Q heads, 8 KV heads, head dim 128.
   - No Q/K RMSNorm.
2. `wormhole_b0-qwen-7b-mesh_device0-device_params0`
   - Geometry: dim 5120, 64 Q heads, 8 KV heads, head dim 128.
   - Head-local Q/K RMSNorm enabled. Despite the test ID, this is the representative dim-5120 geometry used by the Milestone A Qwen lane.

Each case is a combined qualification, not independently selectable decode/prefill tests. It performs:

- batch-32 contiguous-cache decode at positions 127 and 128;
- single-row causal prefill at sequence lengths 128 and 2048;
- two prefill invocations for each sequence length;
- output PCC >= 0.99 and K/V cache PCC >= 0.99 after every invocation;
- production `GalaxyResources` plus `Prefetcher2D` construction and mode activation;
- cleanup of outputs, inputs, module weights, norm weights, KV caches, and resource owner.

The hardware cases deliberately use identity rotary behavior, regular single-row prefill, regular attention, linear all-reduce, and contiguous KV caches. They do not qualify real RoPE composition, concat-32 prefill, ring-selector recipes, prefix/chunked attention, or paged KV caches.

## Findings

### High: all-reduce persistent output specs have the wrong logical shape

`_resource_plan` in `test_attention_2d_wh_galaxy.py:259-267` multiplies the final tensor dimension by the collective axis size. An all-reduce preserves input shape; unlike all-gather, it does not widen the result. The current plans therefore allocate:

- QKV axis-1 output at 4x the input width; and
- WO axis-0 output at 8x the input width.

This disagrees with the qualified MLP helper, where the all-reduce width scale is explicitly 1. It is likely to fail at the first QKV head split, output activation validation, or persistent-output binding before producing numerical evidence. The 60 host tests cannot catch this because they mock the low-level collective and do not validate hardware resource specs against stage tensor shapes.

### High: persistent CCL result ownership is not represented by the Attention2D boundary

The hardware adapter passes a resource-owned persistent output buffer to `all_reduce_async` (`test_attention_2d_wh_galaxy.py:158-174`) and returns its result as an ordinary tensor. `Attention2D._transition` then records that result as module-owned (`attention_2d.py:669-679`), and the reduced QKV result is released after head creation (`attention_2d.py:779` and `attention_2d.py:934`). Final reduced outputs are also explicitly deallocated by the hardware test (`test_attention_2d_wh_galaxy.py:567` and `test_attention_2d_wh_galaxy.py:595`) before the resource owner later releases its persistent allocations.

The MLP host ownership tests explicitly treat persistent collective results as borrowed and avoid releasing them as module transients. Attention2D has no equivalent ownership signal or host reduction. Depending on TTNN alias/deallocation semantics, this can invalidate a buffer before repeat invocation or lead to duplicate release during resource cleanup. This contract should be corrected and covered host-side before interpreting a repeat hardware result.

### Medium: the host lifecycle test gives a false sense of CCL ownership coverage

`test_decode_direct_ttnn_recipe_is_straight_line_and_owns_stages` uses fresh mock tensors for all low-level results (`test_attention_2d.py:124-144`) and only asserts that every mocked transient is released once. It never injects a borrowed/persistent CCL result and therefore enforces the opposite ownership model from the hardware adapter. Add a host reduction analogous to the MLP persistent-output tests when the production fix is made.

### Coverage caveat: current hardware cases are narrower than host recipe coverage

The host suite covers every row/collective/attention selector combination at length 128 plus a 2048 selector. The real-WH file instantiates only `SINGLE_ROW + REGULAR + REGULAR`. This is sufficient as the first numerical gate, but it must not be reported as hardware qualification of concat-32, ring, chunked, or paged-cache branches.

## Known Status and Blockers

- There is no recorded Attention2D output or KV-cache PCC result on WH `(8, 4)`.
- The current Milestone A status correctly marks Attention2D hardware as unqualified.
- The persistent output shape error is the most likely immediate blocker to the first hardware call.
- Persistent-result ownership is a repeat-invocation and cleanup blocker even if the first invocation happens to run.
- No hardware-side failure signature exists yet because this audit intentionally did not execute the device fixture.

## Recommended First Serialized Hardware Command

After correcting and host-testing the two high-severity issues, run the smaller blast-radius single Llama node first with a hard timeout:

```bash
pytest -svv --timeout=600 'models/common/tests/modules/attention/test_attention_2d_wh_galaxy.py::test_attention_2d_wh_galaxy_decode_and_prefill_repeat[wormhole_b0-llama-70b-mesh_device0-device_params0]'
```

This node is the best first serialized gate because it exercises production prefetch/CCL ownership, repeated decode, both required prefill lengths, output PCC, and KV-cache PCC without adding Qwen's head-local Q/K norm path. If it faults or times out, inspect the process/device state and use `tt-smi -r` only after the pytest process has terminated.

## Assessment

Attention2D is host-contract ready at `60 passed`, and the intended WH test matrix is concrete and collectible. The hardware adapter is not ready to run unchanged: its all-reduce persistent buffers are geometrically incorrect, and its borrowed-buffer lifecycle is not expressed or tested. Fix those host-visible defects before spending the serialized Galaxy slot.
