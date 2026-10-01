# fabric_all_gather

`ttnn.experimental.fabric_all_gather` is a drop-in replacement for `ttnn.experimental.high_bw_all_gather`: the same
Python signature, the same validation and program-cache rules, the same output (including the bytes a partial gather
leaves untouched). Only the implementation differs.

## Contract (identical to high_bw_all_gather)
- Input: DRAM (interleaved or ND-sharded), TILE or ROW_MAJOR, any dtype. Output: preallocated interleaved DRAM, the
  worst-case gathered shape.
- `cluster_axis` 0 / 1 gathers along one mesh axis (the other axis runs independent gathers); `None` gathers across the
  whole 2D mesh, output in row-major chip order.
- `input_batch_index` / `gathered_dim_size`: one cache slot, a prefix of the gathered extent (runtime values, not in the
  program hash). Trace-safe forms: `input_batch_index_tensor` (+ `batch_slot_num_layers`, `batch_slot_layer_idx`) and
  `gathered_prefix_tensor` (+ `gathered_slab_global`), read on device.
- `subdevice_id` + `sub_core_grids` confine the op's cores; `ready_semaphore` + `data_valid_semaphore` (caller-owned,
  zero-initialised, left at zero after every call) make it allocation- and sync-free for sub-device overlap.
- Not supported: a ROW_MAJOR gather along the innermost dim (partial pages).

## Algorithm
Terms and the shard / page model: `device/kernels/fabric_all_gather_chunk_walk.hpp`.

Per chip, per ring, per ring direction and per link, one **link worker** core: a reader (NCRISC, NoC0) reads this
chip's own shard from the input, then the shards it relays from its own output; a sender (BRISC, NoC1) sends each
chunk one fabric hop into the same pages of the neighbour's output. One **copy core** per link writes the chip's own
shard. Link workers sit next to the Ethernet core of their link (`tt_fabric::get_forwarding_eth_core`).

- **Chunks**: runs of up to `payload / page` pages that sit consecutively in one DRAM bank, one packet each. Link
  workers split the banks between them.
- **Relay**: relay entry k is what upstream sent as its entry k - 1. Upstream's last packet of every entry increments
  this worker's arrival counter, so the reader relays entry k once the counter reaches k.
- **Fence**: before sending, each sender waits until its downstream has signalled that it started this call, so a
  reused output is never overwritten early.
- **Rings**: axis line / ring; for `cluster_axis=None` a snake, or on a torus with both sides >= 3 two edge-disjoint
  Hamiltonian cycles (every chip uses all four neighbours). Even rings are balanced: the opposite shard goes half each
  way.

## Tests
`tests/ttnn/unit_tests/operations/ccl/test_fabric_all_gather_op.py`: every GLM prefill call pattern, bit-exact against
a torch reference and against `high_bw_all_gather` (hardware); trace-safe metadata under a captured trace; a sub-device
strip with external semaphores; host dispatch cost. Runs under tt-emule (32-chip Galaxy) for correctness, except trace
and sub-devices (fast dispatch only).
