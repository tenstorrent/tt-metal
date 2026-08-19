# MLP2D Axis-0 All-Reduce Reference Audit

## Goal

Diagnose the exact mismatch between the current `MLP2D` axis-0 `all_reduce_async` launch and the proven Wormhole Galaxy 6U production path using source-only analysis. Do not run hardware and do not edit shared production files.

## Checkpoint 1: Scope and Proven Path Identified

- Read the Milestone A plan and confirmed that the common `MLP2D` must preserve the proven WH Galaxy behavior through injected, model-owned resources.
- Identified the proven decode path as:
  - `TtLlamaMLP.forward()` in `models/demos/llama3_70b_galaxy/tt/llama_mlp.py`.
  - `TT_CCL.line_all_reduce()` and `TT_CCL.get_persistent_buffers()` in `models/demos/llama3_70b_galaxy/tt/llama_ccl.py`.
  - Llama and Qwen residual/output layouts in their Galaxy model-config files.
- Identified the current path as:
  - `MLP2D._all_reduce_tg()` in `models/common/modules/mlp/mlp_2d.py`.
  - `GalaxyResources` / `TTNNGalaxyCCLResourceFactory` in `models/common/models/galaxy/resources.py`.
  - Focused hardware resource plans in `models/common/tests/modules/mlp/test_mlp_2d_wh_galaxy.py`.

## Checkpoint 2: Resource Geometry Compared

The current decode resource geometry is equivalent to the proven reference for the axis-0 MLP all-reduce:

- Both use worker subdevice 1 over 50 cores: `(1,0)-(3,9)` and `(5,0)-(6,9)`.
- Both use a width-sharded L1 persistent buffer with per-core shard `[32,1024]`.
- Both materialize that buffer from a global `(8,4,32,51200)` tensor with `ShardTensor2dMesh(dims=(0,1), mesh_shape=(8,4))`, yielding the same per-device storage.
- Both use a width-sharded residual output with shard `[32,128]`: 16 cores for Llama (`dim=8192`) and 10 cores for Qwen (`dim=5120`).
- Both use axis 0, ring topology, four links, BF8 input/output semantics, one global semaphore per active slot, and worker subdevice 1.

The exact decode launch mismatch under investigation was `use_optimal_ccl_for_llama`:

- The proven MLP requests `use_optimal_ccl_for_llama=True` for its decode axis-0 all-reduce.
- The legacy CCL forwards that flag to `ttnn.experimental.all_reduce_async`.
- The earlier failing launch omitted the argument, so nanobind supplied its default `False`.
- In the C++ program factory this is behavioral, not cosmetic: `True` selects `llama_specific::get_custom_worker_core_placement(num_links)`; `False` dynamically chooses workers from cores remaining after reserving output cores.

## Checkpoint 3: Shared-Workspace Rebaseline

The shared production file changed during this source audit; this audit did not make that edit. The latest `models/common/modules/mlp/mlp_2d.py` now passes `use_optimal_ccl_for_llama=True` and the host test asserts it.

Latest-state decode call comparison:

| Property | Proven 6U decode | Current `MLP2D` decode | Result |
| --- | --- | --- | --- |
| Input | BF8, width-sharded W2 output | BF8, width-sharded W2 output | Match |
| Persistent buffer | Axis-0 buffer, L1 width-sharded `[32,1024]` on 50 worker cores | Same | Match |
| Cluster axis | `0` | `0` | Match |
| Mesh | Explicit `(8,4)` mesh | Explicit `(8,4)` mesh | Match |
| Semaphore | One zero-initialized global semaphore on worker cores, two-slot cycle | Same semantics, keyed per collective | Match |
| Links | `4` on WH 6U | `4` | Match |
| Output memory | L1 width-sharded `[32,128]`, 16 Llama or 10 Qwen cores | Same | Match |
| Dtype | `None`, therefore input BF8 | Explicit BF8 | Equivalent |
| Topology | Ring | Ring | Match |
| Subdevice | Worker subdevice 1 | Worker subdevice 1 | Match |
| `use_noc1_only` | Explicit `False` | Omitted, default `False` | Equivalent |
| Optimized placement | Explicit `True` | Explicit `True` in latest state | Match |
| FP32 accumulation | `False` for MLP | Omitted, default `False` | Equivalent |

The flag changes the actual program: the optimized path fixes the four link workers at `(5,3)`, `(6,3)`, `(2,8)`, and `(3,8)`. The default path chooses four available cores dynamically after subtracting residual output cores. Therefore the omission selects a different kernel placement even when every tensor and semaphore resource is otherwise identical.

Remaining ownership differences are not launch mismatches:

- Legacy semaphores are shared by axis and cycled with `gather_idx`; current semaphores are owned and cycled by exact collective key. Both present one fresh scalar semaphore to this first axis-0 MLP all-reduce.
- Legacy owns one persistent buffer per axis; current owns one per exact operation/geometry key. The selected MLP buffer has equivalent storage geometry and distribution.
- The focused current helper's decode stall group contains only worker subdevice 1. This matches the proven tests at operation time, which switch from prefetcher+worker to worker-only after starting prefetch.

## Checkpoint 4: Separate Prefill Divergence

Decode is aligned in the latest state, but prefill is not equivalent to the proven path:

- Current `MLP2D.prefill_forward()` calls `_all_reduce_tg(..., mode="prefill")`, which invokes `ttnn.experimental.all_reduce_async` and currently also requests the optimized Llama worker placement.
- Proven `TtLlamaMLP.forward_prefill()` calls `TT_CCL.line_all_reduce(..., buffer_key="FF2")`.
- In prefill mode, legacy `line_all_reduce()` implements that operation as reduce-scatter followed by all-gather, not `all_reduce_async`.

This does not explain the decode primitive stall, but it is a real reference mismatch that should be resolved before claiming prefill qualification.

## Ranked Fix Recommendation

1. **Retain and qualify `use_optimal_ccl_for_llama=True` for decode.** This is the exact fix for the discovered decode launch mismatch. It restores the proven custom worker placement. Keep the host assertion and rerun the serialized decode hardware case outside this audit.
2. **Make the optimized-worker choice a resolved collective launch option.** Hardcoding the model-named TTNN flag inside reusable `MLP2D` works for both target models but weakens the injected-resource contract. Put the boolean on the all-reduce plan/resources (or a topology-neutral launch-options value) and have `MLP2D` forward it.
3. **Restore the proven prefill algorithm.** Route prefill axis-0 reduction through injected reduce-scatter plus all-gather resources, or provide an independently proven prefill `all_reduce_async` plan. Do not infer prefill validity from the decode fix.
4. **Do not change buffer shape, mapper, residual core grid, topology, link count, semaphore count, or worker subdevice as the next decode experiment.** Source evidence shows those already match the proven 6U implementation; changing them would move away from the reference.

## Evidence Index

- Proven MLP decode request: `models/demos/llama3_70b_galaxy/tt/llama_mlp.py:282-288`.
- Proven all-reduce forwarding: `models/demos/llama3_70b_galaxy/tt/llama_ccl.py:773-819`.
- Proven persistent axis-0 buffer: `models/demos/llama3_70b_galaxy/tt/llama_ccl.py:373-422`.
- Proven semaphore allocation/cycling: `models/demos/llama3_70b_galaxy/tt/llama_ccl.py:96-137` and `:864-865`.
- Proven worker core set: `models/demos/llama3_70b_galaxy/tt/model_config.py:496-501`.
- Proven operation-time worker-only stall group: `models/demos/llama3_70b_galaxy/tests/unit_tests/test_llama_mlp.py:95-105`.
- Current launch: `models/common/modules/mlp/mlp_2d.py:220-279`.
- Current decode call site: `models/common/modules/mlp/mlp_2d.py:394-415`.
- Current prefill call site: `models/common/modules/mlp/mlp_2d.py:501-521`.
- Current persistent plan: `models/common/tests/modules/mlp/test_mlp_2d_wh_galaxy.py:58-120`.
- Current residual/persistent memory configs: `models/common/tests/modules/mlp/test_mlp_2d_wh_galaxy.py:231-260`.
- Current mapped allocation: `models/common/models/galaxy/resources.py:123-135`.
- Current worker subdevice: `models/common/tests/modules/_wh_galaxy_hardware.py:207-243`.
- TTNN default for omitted optimized flag: `ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/all_reduce_async_nanobind.cpp:123-136`.
- TTNN worker placement branch: `ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/all_reduce_async_program_factory.cpp:216-224`.
- Fixed custom worker coordinates: `ttnn/cpp/ttnn/operations/experimental/ccl/llama_common.cpp:9-29`.
- Proven prefill decomposition: `models/demos/llama3_70b_galaxy/tt/llama_ccl.py:823-865`.

No hardware was run. No shared production file was edited.
