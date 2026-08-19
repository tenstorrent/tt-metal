# Attention2D Axis-1 CCL C++ Investigation

## Goal

Investigate the Milestone A Attention2D hang for a WH Galaxy `(8,4)` mesh and local QKV tensor
`(1,1,32,1280)`. Inspect TTNN reduce-scatter, all-gather, and all-reduce validation and worker
scheduling, with emphasis on `cluster_axis=1`, `subdevice_id`, the tensor dimension argument, mesh
mapping, and a minimal reproducer. This checkpoint is source-only: no TT hardware command ran and no
implementation file was edited.

## External Knowledge Access

The requested `tenstorrent/tt-buddy` repository could not be read from this environment:

- The public GitHub page returned 404.
- Direct Git access requested credentials that are not configured.
- `gh` is not installed.
- No local `tt-buddy` checkout or matching installed skill was found under `/home/gwang`.

The investigation therefore applied the same evidence-first workflow to the current local TTNN C++
sources, exact Galaxy tests, model reference path, Git history, and prior Milestone A logs.

## Checkpoint 1: Exact Shape and Dimension Contract

The shape itself is legal for axis-1 reduction.

- `cluster_axis=1` means four devices per independent row of the `(8,4)` mesh. It is not tensor dim
  `1`. `get_topological_dimension()` resolves the ring/line size from global mesh shape, yielding `4`
  (`ttnn/cpp/ttnn/operations/ccl/ccl_common.cpp:167-183`).
- Reduce-scatter's tensor `dim` is independently normalized from the logical tensor rank before the
  primitive call (`reduce_scatter.cpp:63,93`). For `(1,1,32,1280)`, the correct width dimension is
  `dim=3` (equivalently `-1` at the public API).
- The generic all-reduce auto-selector converts tiled dimensions to tile counts. Width is `1280/32 =
  40` tiles, and `40 % 4 == 0`, so it selects logical dim `3`
  (`all_reduce_async.cpp:33-62,337-339`).
- Native reduce-scatter validation maps rank-4 dim `3` to canonical 4D dim `3` and requires width tile
  count divisible by ring size. `40 % 4 == 0`, producing a legal local result width of `320`
  (`reduce_scatter_validate_utils.cpp:33-48`).

Conclusion: confusing mesh axis `1` with tensor dim `1` would be a bug, but the current Attention
adapter passes `cluster_axis=1, dim=3`. There is no shape-divisibility explanation for the hang.

## Checkpoint 2: Mesh Mapping and Neighbor Selection

The CCL host code uses two different domains that must agree:

- Ring size and axis index come from the global mesh shape. For axis 1, physical coordinate `(r,c)`
  gets device index `c`, and neighbors vary only `c` (`ccl_common.cpp:167-205,219-266`).
- Programs are created only for coordinates in the input tensor's device storage
  (`reduce_scatter_program_factory.cpp:53-64`).

There is no validation that the tensor coordinate set contains every participant of every global
axis-1 line. A partially mapped tensor can therefore advertise ring size 4 while scheduling fewer
than four programs, which would deadlock. The current Attention construction appears to cover all 32
coordinates: activation mapping is `(replicate rows, shard tensor dim 3 over columns)`, and the WQKV
weight mapping shards output over rows and K over columns. A reproducer must nevertheless assert and
print the tensor topology and all 32 device coordinates before launch; shape alone does not prove
participant coverage.

All-gather propagates tensor topology and marks a placement replicated only when its tensor shard dim
equals the gather dim (`all_gather_device_operation.cpp:101-118`). That rule is tensor-dimension based,
not mesh-axis based. It is valid for ordinary shard mappings but does not encode an explicit
"partial-sum across mesh axis 1" state. A standalone RS result is therefore the safest AG input for a
reproducer, rather than a synthetic tensor whose topology merely has the same shape.

## Checkpoint 3: Subdevice Scheduling

Standard reduce-scatter does honor the requested subdevice for its own resources:

- It resolves `sd_id` from `subdevice_id` (or the first active subdevice), obtains exactly that
  subdevice's Tensix cores, and allocates its internal global semaphores on those cores
  (`reduce_scatter_program_factory.cpp:34-51`).
- The same `sd_id` supplies the worker core range and core-grid origin to the minimal RS builder
  (`reduce_scatter_program_factory.cpp:105-139`).
- Default RS worker count is bounded by the selected subdevice's core count
  (`reduce_scatter_program_utils.cpp:32-98`).

The runtime does not trust the API argument when dispatching. It derives used subdevices from every
kernel group's actual cores. A program touching kernels in two active subdevices is rejected with
`Programs must be executed on a single sub-device`
(`tt_metal/impl/program/program.cpp:2166-2211`,
`tt_metal/distributed/fd_mesh_command_queue.cpp:387-389`). Thus passing `subdevice_id=1` cannot make a
program valid if any builder-selected core lies in subdevice 0.

This explains the earlier fail-fast result under the sender/worker partition, but not the latest
isolated run: the current CCL-only decode plan creates one full-grid subdevice with ID 0. The isolated
hang must have another cause.

## Checkpoint 4: Standard and Persistent API Differences

The compared APIs are not interchangeable:

1. `ttnn.all_reduce` is a composite wrapper. It calls the cluster-axis `all_reduce_async` overload,
   which either uses standard RS plus the new standard AG or a composite AG plus local reduction
   (`all_reduce.cpp:41-56`, `all_reduce_async.cpp:302-466`).
2. Standard `ttnn.all_gather` was rewritten in July 2026. Its legacy `num_links`, `topology`, worker
   tuning, and L1-small arguments are deprecated and ignored. It derives scheduling from the active
   fabric and accepts only `subdevice_id`/`sub_core_grids` as placement controls
   (`all_gather.cpp:97-135`, `all_gather_nanobind.cpp:45-56`).
3. The persistent `ttnn.experimental.all_reduce_async(input, buffer, ...)` is a separate kernel. It
   requires width-sharded input, output, and scratch; chooses sender cores after reserving output
   cores; and binds the scratch CB on both output and sender cores
   (`all_reduce_async_device_operation.cpp:45-72`,
   `all_reduce_async_program_factory.cpp:202-252,351-381`).
4. The convenience `all_reduce_async(input, cluster_axis, math_op, subdevice_id, memory_config, ...)`
   overload currently ignores its `memory_config` parameter and forwards `nullopt`
   (`all_reduce_async.cpp:469-491`). This is a concrete contract defect. It can silently select input
   placement instead of the requested output placement, although it does not alone prove the hang.

The exact 6U shape reference is
`tests/ttnn/unit_tests/operations/ccl/test_qkv_all_reduce_minimal.py`. It uses
`all_reduce_create_qkv_heads`, not generic RS/AG. Its 6U case uses `FABRIC_1D_RING`, a dedicated single
subdevice, 24 QKV input cores, 10 reduced-output cores, persistent scratch, and fused head creation.
The generic `(1,1,32,1280)` all-reduce cases in `test_new_all_reduce.py` use `FABRIC_1D` and do not
carry an explicit 6U-only qualification marker. They are useful geometry references, but they are not
proof that the same generic kernel is qualified on WH 6U.

## Checkpoint 5: Highest-Confidence Harness Defect

The isolated standard RS/AG adapter force-deallocates the RS result immediately after enqueueing AG:

```python
reduced = ttnn.reduce_scatter(...)
output = ttnn.all_gather(reduced, ...)
deallocate_tensor(reduced)  # calls reduced.deallocate(True)
return output
```

This occurs at
`models/common/tests/modules/attention/test_attention_2d_wh_galaxy.py:347-370`; the shared helper calls
`Tensor.deallocate(True)` (`_wh_galaxy_hardware.py:354-359`). In C++, `force=True` permits underlying
device storage to be freed even when shared owners remain (`ttnn/core/tensor/tensor.cpp:124-149`). The
AG command is asynchronous and still reads that buffer. The exact CCL tests keep RS inputs and
intermediates alive through synchronization, and the earlier RMSNorm2D investigation independently
found that forced queued-intermediate deallocation caused post-compute instability.

This is the most likely cause of the latest isolated standard RS/AG hang. It does not explain every
earlier persistent-kernel experiment, but it invalidates the conclusion that the standard primitive
itself hangs for this shape.

The persistent adapter has the same class of risk: it force-deallocates converted collective input
and temporary placement tensors while the asynchronous CCL remains queued. Those experiments should
also be repeated with all CCL inputs/intermediates retained until a worker-scoped synchronization.

## Ranked Likely Causes

1. **Forced lifetime violation in the Attention test adapter.** Confirmed source defect and direct
   match to the isolated run's RS-to-AG dependency.
2. **Using a generic path where only the fused 6U QKV path is directly qualified.** The exact WH 6U
   reference uses `all_reduce_create_qkv_heads`; generic tests differ in fabric/path qualification.
3. **Actual core placement crossing subdevices in the production sender/worker setup.** Confirmed
   explanation for the earlier fail-fast error; it must be checked from generated kernel cores, not
   inferred from the API's `subdevice_id`.
4. **Fabric/API mismatch.** The Attention fixture passes `fabric_config=True`, while the exact 6U QKV
   reference explicitly uses `FABRIC_1D_RING`; standard AG also ignores requested topology/links and
   derives them from active fabric.
5. **Incomplete tensor coordinate coverage.** Validation gap capable of deadlock, but current mapping
   appears full-mesh and therefore makes this lower probability.
6. **Shape or canonical dim error.** Ruled out for `(1,1,32,1280)`, axis 1, dim 3.

## Minimal Reproducer Proposal

Create a new hardware test file only when hardware work is authorized. Keep it independent of
Attention2D, weights, matmul, Prefetcher2D, KV cache, SDPA, and trace capture.

### Baseline

Copy the single-iteration 6U case from `test_qkv_all_reduce_minimal.py` exactly:

- mesh `(8,4)`;
- explicit `FABRIC_1D_RING`;
- one dedicated worker subdevice;
- BF8 input, BF16 output;
- shape `(1,1,32,1280)` per device;
- 24 input cores, 10 output cores;
- `cluster_axis=1`, three links;
- fused `all_reduce_create_qkv_heads`;
- worker-scoped sync and readback.

This establishes that physical axis-1 routing and the exact geometry work on the local Galaxy.

### Isolation Ladder

Run each case in a fresh process and retain every tensor until after worker synchronization:

1. Standard `ttnn.reduce_scatter(input, dim=3, cluster_axis=1)` only; read back `(1,1,32,320)`.
2. Standard `ttnn.all_gather(rs_output, dim=3, cluster_axis=1)` only, using the live result from step
   1; read back `(1,1,32,1280)`.
3. Repeat step 2 with the old force-deallocation behavior only after the safe case passes. A safe-case
   pass and forced-case hang would prove the lifetime defect.
4. Standard `ttnn.all_reduce(input, cluster_axis=1)` with no requested output re-shard.
5. Persistent experimental all-reduce using the exact 24-input/10-output core geometry and persistent
   scratch from the fused reference, keeping input/scratch/output alive through sync.
6. Change only fabric mode: `FABRIC_1D_RING` versus `FABRIC_1D`.
7. Change only subdevice layout: one dedicated worker subdevice versus production sender+worker
   partition. Record generated kernel core ranges or enable the program/subdevice debug log before
   launch.

Before every launch, assert:

- logical and padded shape;
- dtype/layout/memory config and shard grids;
- tensor topology placements;
- exactly 32 device-storage coordinates, with four participants in every axis-1 row;
- active fabric config and resolved axis topology;
- selected subdevice ID and its complete worker core set.

The first failing transition in this ladder is a useful TTNN reproducer. Starting from the current
Attention test is not minimal because it combines forced lifetimes, placement conversions, model
mapping, and collective scheduling.

## Recommended Next Code Change

When implementation edits are authorized, make one narrow harness change first: retain RS output and
all converted CCL inputs in adapter-owned staging until after `resources.synchronize(mode)`, then
release them. Also replace boolean fabric configuration with an explicit `FabricConfig` enum. Do not
change the collective algorithm in the same experiment.

If standard RS/AG still hangs with safe lifetimes, use the exact fused
`all_reduce_create_qkv_heads` path as the Milestone A decode implementation candidate; it is the only
repository reference that directly qualifies this geometry on 6U and naturally removes the
RS-output-to-AG lifetime boundary.

## Status

Source investigation complete. No hardware ran. No implementation files were edited. The key new
finding is that the latest isolated standard RS/AG result is not valid evidence of a TTNN primitive
hang because its RS output was force-freed before the queued AG consumed it.
