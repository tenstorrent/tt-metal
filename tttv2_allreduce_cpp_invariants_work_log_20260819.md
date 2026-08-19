# TTTV2 `all_reduce_async` C++ Invariants Audit Work Log

## Goal

Audit TTNN/C++ `all_reduce_async` invariants for WH Galaxy axis 0 without running hardware or editing shared production files. Focus on persistent-buffer geometry and mesh mapping, semaphore cardinality/lifecycle, topology, subdevice/core ranges, and input/output sharding. Return concrete validation gaps or deadlock conditions with file:line evidence.

## Checkpoint 1: Call Path And Geometry

Status: complete.

- The MLP path selects the minimal persistent-buffer overload and passes one persistent buffer, one global semaphore, axis 0, explicit mesh, output memory config, topology, link count, and worker subdevice at `models/common/modules/mlp/mlp_2d.py:254-273`.
- The Python binding for this overload defaults to Linear but accepts the explicit topology and scalar `GlobalSemaphore` at `ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/all_reduce_async_nanobind.cpp:108-136`.
- Axis 0 on an `(8,4)` mesh produces `ring_size=8` directly from mesh rows at `ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/all_reduce_async_device_operation.cpp:270-287`.
- Current decode Llama geometry is the same principal geometry as the upstream focused test: input logical shape `(1,1,32,2048)`, 24 input cores, 16 output cores, and 4 links. Current configuration is at `models/common/tests/modules/mlp/test_mlp_2d_wh_galaxy.py:169-227,231-260`; the upstream case is at `tests/ttnn/unit_tests/operations/ccl/test_new_all_reduce.py:305-317`.
- Current input shard is `[32,96]` across 24 cores (2304 padded width), output shard is `[32,128]` across 16 cores, and persistent-buffer shard is `[32,1024]` across all 50 decode worker cores. The buffer therefore exactly satisfies the enforced per-shard volume lower bound: `32*1024 == 8*(32*128)`.
- Current persistent global shape `(8,4,32,50*1024)` with `ShardTensor2dMesh(dims=(0,1), mesh_shape=(8,4))` yields a per-device logical allocation of `(32,50*1024)`, matching 50 local shards of `[32,1024]`; configuration is at `models/common/tests/modules/mlp/test_mlp_2d_wh_galaxy.py:103-112,231-260`.

## Checkpoint 2: Host Validation Coverage

Status: complete.

Validated on cache miss by `AllReduceAsyncDeviceOperation`:

- device allocation and non-null buffers;
- aligned input pages;
- even ring size;
- positive links and links no greater than device grid rows;
- WIDTH_SHARDED input, persistent buffer, and output;
- persistent-buffer core grid contains output core grid;
- persistent-buffer shard volume is at least output shard volume times ring size.

Evidence: `ttnn/cpp/ttnn/operations/experimental/ccl/all_reduce_async/device/all_reduce_async_device_operation.cpp:18-73`.

Important validation gaps:

1. No validation that `cluster_axis` is 0 or 1 before indexing mesh dimensions. `cluster_axis` is used to choose rows versus columns with every nonzero value treated as axis 1 at `all_reduce_async_device_operation.cpp:270-274`, while it is later used as a topology placement index at `all_reduce_async_device_operation.cpp:91-102`.
2. No validation that input tensor, persistent buffer, explicit `mesh_device`, and global semaphore belong to the same mesh or cover the same mesh coordinates. The operation accepts all four independently at `all_reduce_async.cpp:494-525`; validation only checks that each tensor is device allocated at `all_reduce_async_device_operation.cpp:34-38`.
3. No validation of persistent-buffer dtype, tile/page layout, page size, or alignment against the reduction CB. The program binds the persistent allocation as a globally allocated CB but derives its data format and tile size from input/output at `all_reduce_async_program_factory.cpp:241-252,358-368`.
4. No validation that output cores are contained by the selected worker subdevice. The factory silently intersects them with subdevice cores at `all_reduce_async_program_factory.cpp:202-214`, but later partitions and addresses the original output core set at `all_reduce_async_program_factory.cpp:265-296,494-500`. A partial intersection can leave kernels absent on output cores while senders still target them.
5. No validation that the selected subdevice has at least `num_links` non-output worker cores. Selection only warns and returns fewer cores at `all_reduce_async_program_factory.cpp:47-82`; program construction later indexes one sender core per requested link at `all_reduce_async_program_factory.cpp:470-472`.
6. No validation that the global semaphore was allocated on every output core. Reduction kernels wait on the absolute global-semaphore address on every output core at `all_reduce_async_program_factory.cpp:595-604` and `device/kernels/dataflow/reduction_receiver.cpp:21-34`.
7. No cache-hit validator is declared (`all_reduce_async_device_operation.hpp:12-29`). Dynamic overrides update only input address, semaphore address, output CB address, and persistent-buffer CB address at `all_reduce_async_program_factory.cpp:618-653`; all geometry, core placement, topology, and link-derived runtime structure remain compile-time/cached assumptions.

## Checkpoint 3: Topology And Observed Deadlock

Status: complete.

### Primary deadlock condition: requested Ring on neighbor-exchange fabric

The focused MLP hardware test requests `fabric_config=True` at `models/common/tests/modules/mlp/test_mlp_2d_wh_galaxy.py:330-333` and explicitly configures all-reduce as `Topology.Ring` at `models/common/tests/modules/mlp/test_mlp_2d_wh_galaxy.py:58-73`.

Unlike model-specific fixtures, the applicable top-level `device_params` fixture returns the parameter unchanged at `conftest.py:307-311`; `mesh_device` forwards it to `set_fabric` at `conftest.py:616-624`. Python enum conversion was checked without opening hardware: `ttnn.FabricConfig(True)` is `FABRIC_1D_NEIGHBOR_EXCHANGE`. This is fixed by the enum values in `tt_metal/api/tt-metalium/experimental/fabric/fabric_types.hpp:17-26`, where numeric value 1 is neighbor exchange, value 2 is linear fabric, and value 3 is ring fabric. Model-specific fixtures explicitly normalize `True` to `FABRIC_1D_RING` for 6U Galaxy, demonstrating the intended policy at `models/tt_transformers/conftest.py:23-29`; the common test has no equivalent normalization.

`all_reduce_async` does not protect against this mismatch:

- Explicit Ring is accepted whenever the tensor geometry can wrap; `get_usable_topology` does not compare requested topology to active fabric config at `ttnn/cpp/ttnn/operations/ccl/ccl_common.cpp:149-164`.
- A separate helper correctly derives NeighborExchange/Linear/Ring from fabric config, showing that these are distinct active topologies at `tt_metal/fabric/fabric_context.cpp:162-173`, but that helper is not used to validate the explicit request.
- Axis-0 Ring then creates wrapping forward/backward neighbors at `all_reduce_async_program_factory.cpp:114-121`, assigns all seven peer targets around the ring at `ttnn/cpp/ttnn/operations/ccl/ccl_common.cpp:1850-1868`, and opens those fabric connections in `device/kernels/dataflow/worker_writer.cpp:82-109`.

Deadlock mechanism: on neighbor-exchange fabric, the requested wrapping Ring includes an edge connection that is not provisioned as a ring/dateline route. A writer can block in `fabric_connection.open_finish()` (`worker_writer.cpp:107-109`), so no complete set of remote atomic increments reaches the output-core semaphore. Every reduction receiver waits for exactly eight contributions at `all_reduce_async_program_factory.cpp:548-560,595-603` and `device/kernels/dataflow/reduction_receiver.cpp:21-34`; one missing sender leaves the synchronize permanently stalled.

Concrete correction for the current test: use `ttnn.FabricConfig.FABRIC_1D_RING` with `Topology.Ring`, or use a non-wrapping topology compatible with the active fabric. Do not use boolean `True` in this common-test fixture.

### Topology validation gap

Even after correcting this test, the operation should fail fast when explicit topology is incompatible with `GetFabricConfig()`. Current geometry-only resolution means the same deadlock class remains available to any caller that combines Ring/Torus operation topology with Linear or NeighborExchange fabric.

## Checkpoint 4: Semaphore Cardinality And Lifecycle

Status: complete.

- One scalar `GlobalSemaphore` per in-flight minimal all-reduce is the expected cardinality. The operation carries one semaphore at `all_reduce_async_device_operation_types.hpp:22-37`; the program uses its absolute address for all participating output cores at `all_reduce_async_program_factory.cpp:548-560,595-603`.
- Each sender distributes to `ring_size-1` remote peers and contributes locally, so each output-core bank waits for `ring_size` (8 on axis 0). Target counts are computed at `ttnn/cpp/ttnn/operations/ccl/ccl_common.cpp:1850-1868`; remote and local increments are issued at `device/kernels/dataflow/worker_writer.cpp:117-169`; wait value is set at `all_reduce_async_program_factory.cpp:548-560`.
- The reduction receiver resets each participating output-core semaphore bank to zero only after the wait succeeds at `device/kernels/dataflow/reduction_receiver.cpp:29-34`. Therefore an incomplete collective never self-recovers; stale partial counts remain until host reset or resource teardown.
- `GlobalSemaphore` is a height-sharded one-word-per-core allocation over the caller-provided core set at `tt_metal/impl/buffers/global_semaphore.cpp:82-103`. Current allocation on all 50 decode worker cores (`models/common/tests/modules/_wh_galaxy_hardware.py:229-242` and `models/common/models/galaxy/resources.py:180-190,218-224`) correctly covers the 10/16 output cores.
- The current plan has two semaphore slots and one semaphore per slot (`models/common/models/galaxy/resources.py:45-55`). Cycling is deterministic at `models/common/models/galaxy/ccl.py:239-250`. This is sufficient for serial same-queue calls because the receiver resets before completion, but there is no in-flight ownership check: a third asynchronously submitted use can alias slot 0 if callers use independent queues or otherwise bypass ordering.
- Host teardown performs a blocking reset before dropping semaphore ownership at `models/common/models/galaxy/resources.py:231-247`; the low-level reset writes every allocated core and blocks at `tt_metal/impl/buffers/global_semaphore.cpp:59-79`.

Additional gap: the operation never compares `GlobalSemaphore::device()` or its core set (available through `tt_metal/api/tt-metalium/global_semaphore.hpp:44-51`) with the explicit mesh, selected subdevice, or output grid. A semaphore allocated on the wrong mesh or a core set missing any output core can pass validation and deadlock at the fixed wait.

## Checkpoint 5: Persistent Buffer, Mesh Mapping, And Sharding

Status: complete.

### Current geometry is valid

- Input is WIDTH_SHARDED BF8 tiled data on the 24 W2 receiver/ring cores with local shard `[32,96]`; construction is at `models/common/tests/modules/mlp/test_mlp_2d_wh_galaxy.py:169-227`.
- Output is WIDTH_SHARDED on 16 cores for Llama (`[32,128]`) or 10 cores for Qwen (`[32,128]`), all within the decode worker subdevice; construction is at `models/common/tests/modules/mlp/test_mlp_2d_wh_galaxy.py:231-260`.
- Persistent scratch is WIDTH_SHARDED BF8 tiled data on all 50 worker cores with local shard `[32,1024]`. It contains the output grid and provides exactly eight output shards of storage per core, satisfying validation at `all_reduce_async_device_operation.cpp:59-72`.
- The global `(8,4,32,50*1024)` tensor and `ShardTensor2dMesh(dims=(0,1))` give each physical mesh coordinate one complete local scratch tensor. This matches the upstream convention of using `ShardTensor2dMesh` for both input and persistent intermediate at `tests/ttnn/unit_tests/operations/ccl/test_new_all_reduce.py:144-165`.
- Input and buffer mappings both cover the full `(8,4)` mesh. The output topology is derived from input topology and marks axis 0 replicated at `all_reduce_async_device_operation.cpp:91-102`; scratch topology is intentionally not propagated.

### Additional validation gaps and deadlock/corruption conditions

1. **Persistent scratch must reserve sender cores, but validation only requires output cores.** The reduction CB is created on `all_cores = output_cores + sender_worker_cores` and globally bound to the persistent buffer at `all_reduce_async_program_factory.cpp:216-231,358-368`. The sender writer dereferences that CB on each sender core at `device/kernels/dataflow/worker_writer.cpp:42-46`. A buffer grid containing only output cores passes `all_reduce_async_device_operation.cpp:59-61`, yet its address is used on unreserved sender-core L1. This can alias another allocation, corrupt data, or stall. The current 50-core buffer avoids this condition.
2. **Input/output shard tile divisibility is assumed.** Per-core page counts use integer division by `TILE_HW` at `all_reduce_async_program_factory.cpp:175-180,198-200`; no validator requires shard volume to be a whole number of tiles. Truncation can make sender and receiver tile counts disagree.
3. **Fabric packet capacity is assumed.** `num_pages_per_packet = packet_size / page_size` at `all_reduce_async_program_factory.cpp:241-245`. If a page exceeds packet payload, this becomes zero; the writer loop advances by `num_tiles_to_read_this_core`, which then remains zero at `device/kernels/dataflow/worker_writer.cpp:117-161`, producing an infinite loop.
4. **Buffer data format is assumed to equal input reduction format.** Reduction CB format and tile byte size are derived from input dtype, then bound to the caller's buffer allocation at `all_reduce_async_program_factory.cpp:246-252,358-368`. A differently typed buffer passes host validation and can under-allocate or reinterpret the CB.
5. **Output grid/subdevice containment is assumed.** As recorded in Checkpoint 2, partial intersection is accepted at `all_reduce_async_program_factory.cpp:202-214`, but sender partitions still include original output cores. Missing reduction kernels on any targeted output core leave output incomplete and may leave semaphore state undrained.
6. **Input topology must span the complete axis.** Axis ring size is taken from global mesh shape, not participating input coordinates, at `all_reduce_async_device_operation.cpp:270-287`. The workload is created over tensor coordinates at `all_reduce_async_program_factory.cpp:89-101`. A partially mapped input can therefore program fewer participants while each output core still waits for the full axis size.

## Final Synthesis

Status: audit complete; no hardware was run and no shared production file was edited.

Highest-priority action for the current Milestone A stall:

1. Replace `fabric_config=True` in the common WH Galaxy MLP hardware test with `ttnn.FabricConfig.FABRIC_1D_RING` while retaining `Topology.Ring`. This directly removes the statically demonstrated NeighborExchange/Ring mismatch.
2. If a Linear fabric is intentionally desired, configure both active fabric and operation topology consistently; do not rely on `get_usable_topology` to reconcile them.
3. Keep the current axis-0 persistent buffer mapper, 50-core grid, and `[32,1024]` shard. They satisfy all currently enforced geometry invariants and also cover sender cores, which is stronger than C++ validation requires.
4. Keep one global semaphore per slot and reset/serialize after failed runs. The primitive's device reset only happens after all eight contributions arrive, so a deadlocked run leaves partial values.

Recommended C++ hardening, in priority order:

1. Validate explicit operation topology against active fabric topology before program creation.
2. Validate semaphore mesh and core coverage against every output core.
3. Select sender cores before final validation, then require persistent-buffer grid to contain both output and sender cores and require at least `num_links` sender cores.
4. Require input/output shard volumes to be tile divisible, buffer dtype/page layout compatible with input, and packet payload at least one page.
5. Require axis in range, input/buffer/mesh identity and coordinate compatibility, full-axis participant coverage, and output-grid containment in the selected subdevice.

No production edit was made, per task constraints.
