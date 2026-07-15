# CB→DFB Kernel Audit: `all_gather_async` [factory: `AllGatherViaBroadcastFactory`]

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/`

**Scope:** `AllGatherViaBroadcastFactory` (`all_gather_via_broadcast_factory.cpp`) → kernels: `device/kernels/broadcast_rm_reader.cpp`, `device/kernels/broadcast_rm_writer.cpp`. No compute kernel, no donor kernels. Shared headers scanned (via writer's `minimal_ccl_common.hpp`): `ccl/common/kernels/minimal_ccl_common.hpp`, `ccl/shared_with_host/{hetergeneous_data_structs,sharded_tensor_addr_gen}.hpp`, `ccl/kernel_common/sharding_addrgen.hpp` (all zero hits).

**Out of scope for this slice** (referenced only by other factories of this op): `device/kernels/minimal_default_reader.cpp`, `device/kernels/minimal_default_writer.cpp` (→ `AllGatherAsyncDefaultProgramFactory`); `device/kernels/llama_shapes_sharded_reader.cpp`, `device/kernels/llama_shapes_sharded_writer.cpp` (→ `AllGatherAsyncLlamaShardedProgramFactory`). These need their own audit slices if/when those factories are ported.

## Overall verdict: GREEN

**Summary:** Single cross-kernel circular buffer `cb0` is a Class 1 linear FIFO (reader = PRODUCER, writer = CONSUMER) using the `CircularBuffer` object API, which maps 1:1 to `DataflowBuffer`. Zero GATE / silent-wrong / quasar-blocked / LTA-prereq / ptr-surgery hits. Mechanical rename only.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb0` (CTA0) | 1 | `broadcast_rm_reader.cpp` (producer), `broadcast_rm_writer.cpp` (consumer) | Portable | linear FIFO; `CircularBuffer` object API → `DataflowBuffer` 1:1; `get_write_ptr`/`get_read_ptr` are bare L1 addresses | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
