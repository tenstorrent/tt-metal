# MLP2D Prefetch/Readback Lifecycle Audit

## Goal

Audit the clean WH Galaxy MLP2D decode timeout without TT hardware or shared-file edits. Determine whether host composition/readback while `dram_prefetcher` is active can hang, and identify the narrow production/test API boundary needed to avoid it.

## Scope and method

- Read-only source audit of `Prefetcher2D`, `GalaxyResources`, the common Galaxy hardware helper, the MLP2D Galaxy test, TTNN host conversion, fast-dispatch subdevice selection, and the DRAM prefetch kernels.
- Compared the common path with the proven Llama and Qwen Galaxy MLP tests.
- Used the main Milestone A work log's clean-timeout and worker-scoped retry as hardware evidence; this audit ran no hardware commands.

## Findings

### 1. Decode activation deliberately leaves a sender program in flight

`Prefetcher2D.activate("decode")` starts `ttnn.dram_prefetcher` and stores its returned tensor as `_prefetch_result` (`models/common/modules/prefetcher/prefetcher_2d.py:367-399`, `491-510`). Before launch it stalls both decode subdevices; immediately after launch it restores `context.stall_group`. The canonical Galaxy decode plan makes that steady-state group worker-only (`models/common/tests/modules/_wh_galaxy_hardware.py:207-243`).

This matches the legacy Llama/Qwen sequence: launch the prefetcher while sender and worker are stalled, then set the stall group to the worker before running MLP (`models/demos/llama3_70b_galaxy/tests/unit_tests/test_llama_mlp.py:95-130`, `test_qwen_mlp.py:131-169`). The sender kernel is therefore allowed to overlap worker execution and must not be included in an incidental whole-device wait.

### 2. The prefetch kernel is finite but may still be running at host readback

The DRAM reader and L1 writer execute `num_layers * num_tensors` loops. The writer then performs the remote-CB barrier and signals the reader; only after that signal can the reader exit (`ttnn/cpp/ttnn/operations/prefetcher/prefetcher/device/kernels/writer_l1.cpp:40-99`, `reader_dram.cpp:40-119`). With `prefetch_num_layers=1`, this is a finite session rather than an unbounded service, but host readback can race its completion.

The Python owner keeps `_prefetch_result` live until repeat activation, mode transition, or cleanup. Its default stop currently deallocates that sentinel (`prefetcher_2d.py:498-545`). Readback is intentionally not a prefetch stop boundary.

### 3. `compose`/`to_torch` while the sender is active is unsafe without an explicit worker wait

`compose_2d_sharded_tensor` calls `ttnn.to_torch` without a subdevice argument (`models/common/tests/modules/_wh_galaxy_hardware.py:246-253`). `ttnn.to_torch` calls `ttnn.from_device` for a device tensor (`ttnn/ttnn/operations/core.py:379-423`), and fast-dispatch reads with an omitted subdevice list resolve to the mesh's current stall group (`tt_metal/impl/buffers/dispatch.cpp:1847-1855`). This makes correctness depend on mutable ambient mesh state and on implementation details below the high-level compose API.

In the intended decode state, that ambient group is worker-only, so the low-level read path should not wait for the sender. Nevertheless, `to_torch` exposes no subdevice-scoped contract, and the clean hardware result is decisive: direct composition timed out, while one explicit synchronization of only `worker_sub_device_id` before composition passed both Llama invocations in 22.93 seconds with the prefetch sender still active (`tttv2_2d_modules_work_log.md:707-725`).

**Conclusion:** yes, compose/readback while `dram_prefetcher` is active can hang when the readback path performs or inherits a wait that includes the sender. Relying on the current stall group alone is too implicit for a qualification harness. Stopping the prefetcher before every readback is unnecessary and would alter the production overlap lifecycle.

### 4. The narrow API fix belongs at the Galaxy resource/readback boundary

`GalaxyResources` already implements the exact safe operation in private `_synchronize(mode)`: it calls the injectable synchronization binding with the selected mode plan's `stall_group` (`models/common/models/galaxy/resources.py:400-401`). Decode's group contains only the worker. `activate` mode transitions and `cleanup` already use this helper (`resources.py:307-321`, `336-355`).

Recommended production API:

```python
def synchronize(self, mode: GalaxyMode | None = None) -> None:
    selected = mode or self._active_mode
    if selected is None:
        raise RuntimeError("Galaxy resources have no active mode")
    self._synchronize(selected)
```

This is a public resource-owner boundary, not an MLP hot-path synchronization. It should neither call `Prefetcher2D.activate`, stop/deallocate `_prefetch_result`, reset the stall group, nor clear the loaded manager.

Recommended test API and call site:

- Add `GalaxyHardwareResources.synchronize(mode)` that delegates to `owner.synchronize(mode)`. A compatibility fallback may call `ttnn.synchronize_device` with only `ccl.context(mode).worker_sub_device_id`.
- In `_invoke`, call `resources.synchronize(mode)` after `module(...)` and before `compose_2d_sharded_tensor(...)`.
- Do not add synchronization inside `compose_2d_sharded_tensor`; that helper lacks mode/resource ownership and would reintroduce an ambient whole-device decision.

### 5. Host tests needed for the API contract

- `GalaxyResources.synchronize("decode")` emits exactly one binding call with the decode worker stall group.
- Synchronization leaves `active_mode` unchanged and does not call prefetch activation/stop/cleanup.
- Calling without a mode uses the active mode; calling with neither an argument nor an active mode fails clearly.
- The hardware adapter delegates to public owner synchronization and its fallback names only the worker subdevice.
- The MLP Galaxy test orders events as module completion, worker synchronization, composition, then tensor deallocation.

## Final assessment

No `Prefetcher2D` lifecycle change is warranted for this timeout. The persistent/overlapped decode session is behaving as designed. The missing contract is a public, worker-scoped wait on the Galaxy resource owner before host readback. Promoting the existing private synchronization operation and routing the qualification helper through it is the smallest production/test API fix.
