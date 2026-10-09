# fabric_all_gather

`ttnn.experimental.fabric_all_gather` is a drop-in replacement for `ttnn.experimental.high_bw_all_gather`: the same
Python signature, the same validation and program-cache rules, the same output (including the bytes a partial gather
leaves untouched). Only the program differs: the device operation (`device/fabric_all_gather_device_operation.hpp`)
derives from high_bw_all_gather's, so parameters, validation, program hash and output spec are shared code.

## Contract (identical to high_bw_all_gather)
- Input: DRAM (interleaved or ND-sharded), TILE or ROW_MAJOR, any dtype. Output: preallocated interleaved DRAM, the
  worst-case gathered shape.
- `cluster_axis` 0 / 1 gathers along one mesh axis (the other axis runs independent gathers); `None` gathers across the
  whole 2D mesh, output in row-major chip order.
- `input_batch_index` / `gathered_dim_size`: one cache slot, a prefix of the gathered length (runtime values, not in the
  program hash). Trace-safe forms: `input_batch_index_tensor` (+ `batch_slot_num_layers`, `batch_slot_layer_idx`) and
  `gathered_prefix_tensor` (+ `gathered_slab_global`), read on device.
- `subdevice_id` + `sub_core_grids` confine the op's cores; `ready_semaphore` + `data_valid_semaphore` (caller-owned,
  zero-initialised, left at zero after every call) make it allocation- and sync-free for sub-device overlap.
- Not supported: a ROW_MAJOR gather along the innermost dim (partial pages). A partial length along a TILE gather dim must
  be tile aligned.

## Algorithm
Terms (logical vs physical pages, outer slices, fabric chunks, CB batches, outgoing shards, the counters):
`device/kernels/fabric_all_gather_chunk_walk.hpp`.

Per chip, per ring, per ring direction and per link, one **fabric link worker** core: a reader (NCRISC, NoC0) reads
this chip's own shard from the input, then the shards it forwards from its own output; a sender (BRISC, NoC1) sends
each fabric chunk one fabric hop into the same pages of the neighbour's output. **Local copy cores** write the chip's
own shard: one per link, or two per link for a non-interleaved input (the ND-sharded KV cache), which they convert
block by block into the output for the link workers to read from there. Link workers sit next to the Ethernet core of
their link (`tt_fabric::get_forwarding_eth_core`).

- **Fabric chunks**: up to `payload / page` consecutive pages of one page lane (pages k, k + 8, k + 16, ...), physically contiguous in an interleaved
  tensor, one packet each (a page larger than the payload is split over several packets). Link workers split the
  lanes between them.
- **Forwarding**: outgoing shard k is what upstream sent as its outgoing shard k - 1. Upstream's last packet of every
  outgoing shard (or a bare increment, if the shard has no chunk in that worker's lanes) increments this worker's
  shards-arrived counter, so the reader forwards shard k once the counter reaches k.
- **Fence**: before sending, each sender waits until its downstream has signalled that it started this call (the
  downstream-started counter), so a reused output is never overwritten early.
- **Rings**: axis line / ring; for `cluster_axis=None` a snake, or on a torus with both sides >= 3 two edge-disjoint
  Hamiltonian cycles (every chip uses all four neighbours). Even rings are balanced: the opposite shard goes half each
  way.

## Tests
- `tests/ttnn/unit_tests/operations/ccl/test_fabric_all_gather_op.py`: every GLM prefill call pattern, bit-exact against
  a torch reference and against `high_bw_all_gather` (hardware); trace-safe metadata replayed with changing cache slots and
  lengths; a sub-device strip with external semaphores; pages larger than the fabric payload; host dispatch cost. Runs
  under tt-emule (32-chip Galaxy) for correctness, except trace and sub-devices (fast dispatch only).
- `tests/ttnn/unit_tests/operations/ccl/test_fabric_all_gather_device_time.py`: device time against
  `high_bw_all_gather`, reported as GB/s received per chip and busiest-link utilisation (GLM KV-cache gather, the
  sparse-MLA overlap window, 18 MiB tile gather). The Galaxy set runs in the `high_bw_all_gather` CI job.
