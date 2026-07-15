# CB→DFB Kernel Audit: `all_broadcast` [factory: `AllBroadcastProgramFactory`]

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/ccl/all_broadcast/`

**Scope:** `AllBroadcastProgramFactory` (`all_broadcast_program_factory.cpp`) → **donor kernels** (live under sibling op `ccl/broadcast/`): `broadcast_tile_reader.cpp`, `broadcast_rm_reader.cpp`, `broadcast_tile_writer.cpp`, `broadcast_rm_writer.cpp`. Both `tilized` and row-major branches are reachable, so all four are in scope. Shared headers scanned: `ccl/kernel_common/{sharding_addrgen,worker_routing_utils}.hpp`, `ccl/shared_with_host/{sharded_tensor_addr_gen,hetergeneous_data_structs}.hpp`, `ccl/common/kernels/minimal_ccl_common.hpp` (all zero hits).

## Overall verdict: GREEN

**Summary:** Single cross-kernel circular buffer `c_in0` is a textbook Class 1 linear FIFO (reader = PRODUCER, writer = CONSUMER). Zero GATE / silent-wrong / quasar-blocked / LTA-prereq / ptr-surgery hits across all scanned files. Mechanical `CircularBuffer` → `DataflowBuffer` rename only. Note: the four kernels are donor kernels shared with `ccl/broadcast`; this finding applies to both ops.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `c_in0` (`cb0_id`, CTA0) | 1 | `broadcast_tile_reader.cpp`, `broadcast_rm_reader.cpp` (producer); `broadcast_tile_writer.cpp`, `broadcast_rm_writer.cpp` (consumer) | Portable | linear FIFO → `DataflowBuffer`; `get_write_ptr`/`get_read_ptr` are bare L1 addresses only | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)

## Notes

- **Shared donor kernels:** the four in-scope kernels live under `ccl/broadcast/device/kernels/` and are referenced by `AllBroadcastProgramFactory`. `ccl/broadcast` is not separately on the audit do-list; any port of these kernels covers both ops.
- Writers' fabric-multicast and `noc_semaphore_*` traffic uses raw semaphore-bank addresses via `reinterpret_cast<volatile tt_l1_ptr uint32_t*>` — these are NOC/semaphore addresses, not CB-interface field reads, and are outside the CB audit patterns.
