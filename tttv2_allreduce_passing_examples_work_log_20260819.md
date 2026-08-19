# WH Galaxy `all_reduce_async` Passing-Example Investigation

## Goal

Find known passing WH Galaxy `all_reduce_async` tests or model uses in this repository and compare their complete device setup and call sequence with `models/common/tests/modules/mlp/test_mlp_2d_wh_galaxy.py`, focusing on axis-0 eight-device lines and reusable semaphore behavior. No hardware runs and no shared production-file edits.

## Checkpoint 1: Repository Inventory

- Searched the repository for every Python `all_reduce_async` call.
- Identified the most relevant evidence families:
  - `tests/nightly/tg/ccl/test_all_reduce_async.py`: TG/Galaxy CCL correctness coverage with explicit cluster-axis calls.
  - `models/demos/llama3_70b_galaxy/tt/llama_ccl.py`: production Galaxy wrapper with persistent buffers and semaphore selection.
  - `models/demos/deepseek_v3_d_p/tt/tt_lm_head.py` and MoE gate code: model-level persistent semaphore reuse.
  - `models/common/tests/modules/attention/test_attention_2d_wh_galaxy.py`: nearby Milestone A axis-0 call, useful for local setup comparison but not yet accepted as a known passing reference.
- Confirmed the target MLP test currently monkeypatches synchronization immediately after the primitive, so a stall there isolates the primitive or its submitted work rather than the following reshape.
- Next: reconstruct complete setup and exact call signatures for the known Galaxy references, then diff invariants against the target test.

## Checkpoint 2: Complete Setup and Reuse Comparison

### Explicit passing post-commit test

- `tests/nightly/tg/ccl/test_all_reduce_async.py:326-377` defines the repository's direct WH Galaxy `(8,4)` axis-0 case: eight devices per line, four simultaneous lines, 16 iterations, BF16 tile data, DRAM, one link, `Topology.Linear`, and `FABRIC_1D`.
- Its helper (`:119-225`) synchronizes before setup, creates one full-grid worker subdevice, loads its manager, sets the worker stall group, allocates fixed vectors of 3 reduce-scatter, 2 all-gather, and 2 barrier global semaphores, then reuses those same vectors for every iteration and synchronizes the worker subdevice after each call.
- This proves reusable semaphore handles do not require host-side resetting between completed invocations. It does not directly qualify the target's minimal persistent-buffer overload because it uses the composite overload.

### Established WH Galaxy decode model path

- `models/demos/llama3_70b_galaxy/tt/model_config.py:496-501` uses exactly the target's 50-core worker set: `(1,0)-(3,9)` plus `(5,0)-(6,9)`.
- `model_config.py:630-641` selects four links and `Topology.Ring` on a 32-PCIe-device WH 6U Galaxy.
- `llama_ccl.py:96-141` creates two semaphore handles per cluster axis on the worker core set and initializes a per-axis index. `llama_ccl.py:801-819,864-865` passes one current handle to the minimal persistent-buffer overload and alternates the index modulo two after every call.
- `llama_ccl.py:387-422` creates the axis-0 persistent L1 width-sharded buffer over all 50 worker cores with a global `(*cluster_shape, M, N_per_shard * num_cores)` tensor and `ShardTensor2dMesh(dims=(0,1), mesh_shape=(8,4))`.
- `llama_mlp.py:282-288` invokes that wrapper for the decode W2 output with cluster axis 0, four Galaxy links, decode residual output memory config, and **explicitly sets `use_optimal_ccl_for_llama=True`**.
- On the initial read, the target matched the model's mesh, topology, link count, worker subdevice, persistent-buffer distribution, two-slot semaphore cycling, output shard height/width, and axis. The material call-sequence mismatch was the absent optimal-worker flag.

## Checkpoint 3: C++ Causal Evidence and Smallest Diagnostic

- The persistent-buffer Python signature defaults `use_optimal_ccl_for_llama=False` (`all_reduce_async_nanobind.cpp:108-136`), so omission in `MLP2D._all_reduce_tg` is behaviorally significant.
- `all_reduce_async_program_factory.cpp:202-224` obtains the selected worker subdevice, excludes output cores, and chooses sender workers as follows:
  - flag true: `llama_specific::get_custom_worker_core_placement(num_links)`;
  - flag false: generic `ar_choose_worker_cores(...)`.
- `llama_common.cpp:9-30` fixes the four-link Llama worker placement to `(5,3)`, `(6,3)`, `(2,8)`, `(3,8)`, all within the same 50-core worker envelope used by the target.
- Git commit `69586dbc6dc` (`Improving CCL Core Locations for Llama (#24479)`) introduced this flag and added it to the Galaxy decode MLP axis-0 all-reduce. Current blame shows it has remained on that exact model call since 2025-07-07.
- The target's synchronization monkeypatch synchronizes immediately after the primitive returns, so the observed stall occurs in submitted all-reduce work and cannot be caused by the subsequent reshape.

### Smallest high-confidence diagnostic/fix

Pass `use_optimal_ccl_for_llama=True` in the target MLP decode all-reduce call. The minimally scoped production form is `use_optimal_ccl_for_llama=mode == "decode"` in `MLP2D._all_reduce_tg`, mirroring the module's decode-only optimization choice for all-gather. Run only the existing single Llama decode hardware case first. A completed primitive synchronization would directly confirm the worker-placement diagnosis before broad qualification.

This is higher confidence than changing topology, link count, semaphore count, or subdevice layout because the established WH Galaxy MLP already passes with the target's current values for all of those parameters and differs at this exact call by the worker-placement flag.

## Checkpoint 4: Concurrent Fix State and Verification

- During this repository-only investigation, another work lane added `use_optimal_ccl_for_llama=True` to `MLP2D._all_reduce_tg` and added a host assertion. This investigation did not edit either shared file.
- The main work log records `32 passed in 4.91s` for the focused MLP host suite after that change.
- The first hardware retry after the change did not reach model execution: a warm reset exposed a `(16,2)` mesh descriptor that could not satisfy the `(8,4)` fixture. Therefore there is currently no positive or negative hardware result for the flag.
- No hardware command was run by this investigation. The only file created or edited here is this work log.

### Exact next step

After the serialized full Galaxy reset restores `(8,4)`, rerun the single Llama decode case with the now-present flag and retain the immediate post-primitive synchronization. If it completes, the worker-placement mismatch was causal. If it still blocks, the smallest next diagnostic is a standalone two-invocation minimal all-reduce test using the target's existing W2-output layout, axis-0 persistent buffer, two alternating semaphore handles, Ring topology, four links, worker subdevice 1, and optimal flag; this separates all-reduce setup from preceding reduce-scatter/all-gather queue state without changing any geometry.
