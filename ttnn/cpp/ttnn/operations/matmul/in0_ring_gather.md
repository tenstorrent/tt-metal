# Matmul "in0 ring gather" (1D GATHER_IN0)

This document explains how the ttnn matmul op implements the `gather_in0` path, also
called **in0 ring gather**. It describes the topology, the circular-buffer layout, and
the dataflow/compute kernels that move and consume the activations (`in0`) around a
unidirectional ring of cores.

> Note: the code lives in the core matmul op at
> `ttnn/cpp/ttnn/operations/matmul/`, **not** under
> `ttnn/cpp/ttnn/operations/experimental/matmul/` (which holds `attn_matmul` and
> `group_attn_matmul`). The relevant files are listed at the bottom.

---

## 1. What "ring gather" means here

In a 1D matmul, the inner dimension `K` of the activation tensor `in0` is **sharded
across the worker cores** (width-sharded in tiles). Each worker core `i` owns exactly
one `K`-slice of `in0` (its *local shard*), but to compute its full output tile it must
accumulate over **all** `K` slices:

```
out[M_c, N_c] = sum over k  A[M_c, k] * B[k, N_c]
```

where the `k` slices of `A` live on *different* cores.

`gather_in0` solves this by rotating the `in0` shards through a **logical ring** of
cores. Each core sends its local shard to the next core in the ring and forwards every
shard it receives to the next core, one shard per pipeline step. After `ring_size - 1`
steps every core has seen every other core's shard and can finish the reduction.

The name is a deliberate contrast with the sibling 1D mode `MCAST_IN0`: instead of
multicasting one `in0` block from a single sender to all receivers, `GATHER_IN0` ships
each core's distinct shard around a ring so that all shards meet on every core.

### When it is selected

- Config: `MatmulMultiCoreReuseMultiCast1DProgramConfig` with `gather_in0 == true`
  (see `matmul_program_config_types.hpp`).
- The 1D factory returns the sentinel `ttnn::prim::Matmul1DType::GATHER_IN0`
  (see `matmul_1d_type.hpp`).
- Validation (`matmul_device_operation.cpp`) requires:
  - `in0` (`input_tensor_a`) is **sharded**,
  - no `transpose_a` / `transpose_b`,
  - at least one sub-device id,
  - `hop_cores` (if any) do **not** overlap the `in0` shard grid.

---

## 2. Ring topology: worker cores + optional hop cores

The ring is built from two kinds of cores, distinguished by a runtime `CORE_TYPE`:

```cpp
enum class CORE_TYPE : uint8_t { IDLE_CORE = 0, WORKER_CORE = 1, HOP_CORE = 2 };
```

| Kind | Holds a local `in0` shard? | Role in the ring |
|------|---------------------------|------------------|
| `WORKER_CORE` | Yes (`cb_in0`) | Owns one `K` slice; participates in compute. |
| `HOP_CORE` | No | Pure relay: forwards shards, runs no compute. |
| `IDLE_CORE` | No | Not part of the ring; returns immediately. |

Host-side (`matmul_multicore_reuse_mcast_1d_program_factory.cpp`):

- `all_worker_cores = a.shard_spec().grid` — the `in0` shard grid.
- `ring_size = all_worker_cores.num_cores()` — the number of `K` slices / logical ring
  positions (hop cores do **not** increase `ring_size`).
- `hop_cores` are optional relay cores supplied in the program config.

**Ring direction.** Each worker sends to the *previous* index and the edge between
worker `0` and worker `n-1` is where hop cores are spliced in:

```
worker_0 → (hop_0 → hop_1 → … → hop_{h-1}) → worker_{n-1}
worker_{n-1} → worker_{n-2} → … → worker_1 → worker_0
```

This is set up in the runtime-arg loop: for worker `i`, `next_core` is `worker_{i-1}`
(wrapping `0 → n-1`), except worker `0`, which sends to `hop_cores[0]` when hop cores
are present. Hop core `i` sends to `hop_{i+1}`, and the last hop core
(`end_of_hop`) sends to `worker_{n-1}`.

The purpose of hop cores is to break the physically longest edge (the wrap-around jump
from worker `0` directly to worker `n-1`) into several shorter NoC hops. This improves
per-hop latency/bandwidth on a large ring; the logical gather semantics are unchanged.

---

## 3. Circular buffers

Two CBs implement the gather; a third (`cb_in1`) holds the weights and a semaphore
carries the pipeline credit.

- `cb_in0` (`src0_cb_index`) — the core's **local** shard
  (`per_core_M × in0_shard_width_in_tiles` tiles), globally allocated at `in0`'s
  address. It is pushed to compute directly by the compute kernel, **not** by the reader.
- `cb_in2` (`src2_cb_index`) — holds the **remote** shards that arrive over the ring:

  ```cpp
  uint32_t in2_CB_tiles = (ring_size - 1) * in0_CB_tiles;  // All shards except local
  ```

  The reader reserves `(ring_size - 1) * shard_size_in_tiles` here up front and pushes
  one shard per pipeline step.
- `in0_signal_semaphore_id` — a per-core semaphore used as a credit/token counter for
  the ring pipeline (see §5).

---

## 4. Reader kernel (RISCV_1)

File: `device/kernels/dataflow/reader_bmm_tile_layout_in0_ring_all_gather.cpp`

The reader is instantiated with compile-time args `shard_width_in_tiles`,
`shard_height_in_tiles`, `batch`, `ring_size`, and the signal semaphore id; named args
`cb_in0` and `cb_in2` bind the two buffers. Runtime args are:

```
core_type, ring_idx, next_core_noc_x, next_core_noc_y, noc_id, end_of_hop,
unpadded_in0_shard_widths_in_tiles[ring_size]   // workers only
```

The core loop, simplified:

```cpp
dfb_in2.reserve_back((ring_size - 1) * shard_size_in_tiles);   // room for remote shards
uint32_t local_shard_read_addr = dfb_in0.get_read_ptr();
uint32_t l1_write_addr_in0     = dfb_in2.get_write_ptr();
uint32_t hop_core_offset       = is_hop_core ? 1 : 0;

for (uint32_t shard_cnt = hop_core_offset; shard_cnt < ring_size; shard_cnt++) {
    uint32_t curr_ring_idx = (ring_idx + shard_cnt) % ring_size;

    // 1. Wait until `shard_cnt` shards have reached this core (pipeline credit).
    signal_sem.wait_min(shard_cnt);

    // 2. Forward: step 0 sends the *local* shard, later steps forward the shard
    //    received in the previous step (which sits at the previous write slot).
    uint32_t curr_shard_read_addr =
        shard_cnt == 0 ? local_shard_read_addr
                       : l1_write_addr_in0 + shard_size_bytes * (shard_cnt - 1);
    uint32_t curr_shard_write_addr =
        l1_write_addr_in0 + shard_size_bytes * (shard_cnt - hop_core_offset);

    if (shard_cnt < ring_size - 1 || is_hop_core) {   // workers skip the last send
        noc_obj.async_write(curr_shard_read_addr, next_core, curr_shard_write_addr, shard_size_bytes);
        noc_obj.async_writes_flushed();
        signal_sem.up(noc_obj, next_core, 1);          // credit to successor
    }

    // 3. Make the received shard visible to compute.
    if (shard_cnt > 0) dfb_in2.push_back(shard_size_in_tiles);
}
```

What this implements is a **pipelined ring all-gather**:

1. **Step 0**: every core sends its own shard to its successor (no wait needed,
   `wait_min(0)` passes). The local shard is never pushed into `cb_in2`; compute reads
   it from `cb_in0`.
2. **Steps 1..ring_size-2**: each core forwards the shard it received in the previous
   step and pushes that shard into `cb_in2`.
3. **Step ring_size-1** (workers): the last received shard is pushed into `cb_in2` but
   **not** forwarded — after `ring_size - 1` forwards the ring has fully unwound and no
   one needs it back.

The `signal_sem` is the pipeline token: `up(next_core, +1)` after a write tells the
successor "one more shard has landed at your `cb_in2`", and `wait_min(shard_cnt)`
blocks until that many tokens have been received, guaranteeing the source shard is
valid in L1 before it is read for forwarding. `async_writes_flushed()` before the `up`
is required because the payload write and the atomic increment use different NoC
command buffers; without the flush the atomic could land before the data.

**Hop cores** run the same kernel with `hop_core_offset = 1`, so they start at
`shard_cnt = 1` (they own no local shard) and, because `is_hop_core` keeps the send
guard true, they forward **every** received shard — including the last one, which is
what closes the loop back into `worker_{n-1}`.

For `batch > 1` the extra batches need no re-gather (the shard layout is identical):
the reader simply re-credits `cb_in2` once per remaining batch:

```cpp
for (uint32_t b = 0; b < batch - 1; ++b) {
    dfb_in2.reserve_back((ring_size - 1) * shard_size_in_tiles);
    dfb_in2.push_back((ring_size - 1) * shard_size_in_tiles);
}
```

---

## 5. Compute kernel (consumes the gathered shards)

File: `device/kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp`

Compute mirrors the reader's rotation. It processes `num_blocks == ring_size` blocks;
for block `b` it reads `in0` from `cb_in0` when `b == 0` and from `cb_in2` otherwise,
always rotating by `ring_idx`:

```cpp
const uint32_t curr_ring_idx = (ring_idx + block) % ring_size;
const uint32_t input0_dfb_id = block == 0 ? in0_dfb_id : in2_dfb_id;
...
for (uint32_t inner_dim_idx = 0; inner_dim_idx < unpadded_in0_block_w; ++inner_dim_idx)
    matmul_block(input0_dfb_id, in1_dfb_id, in0_index, in1_index, dst_index, ...);
```

Key details:

- The **local** shard (`cb_in0`) is credited to compute here, not by the reader:
  `input0_dfb.reserve_back(...); input0_dfb.push_back(...)` on block 0, matching the
  comment in the reader ("Reserving/pushing the local shard is done in compute").
- `unpadded_in0_shard_widths_in_tiles[curr_ring_idx]` bounds the inner-dimension loop
  per block. `K` is not necessarily a multiple of `ring_size × in0_block_w`, so the
  last slice(s) may be narrower; the host precomputes each core's actual width and
  passes the array as a runtime-arg tail to both reader and compute.
- The `in1` (weight) block is consumed in the **same** rotated order (see §6), so at
  block `b` the core multiplies `in0` slice `curr_ring_idx` against the matching `in1`
  slice and accumulates into its `per_core_M × per_core_N` output tile.

`IDLE_CORE` and `HOP_CORE` return immediately after reading `core_type`; only
`WORKER_CORE` performs the matmul.

---

## 6. The `in1` (weight) side

File: `device/kernels/dataflow/reader_bmm_tile_layout_in1_ring_all_gather.cpp` (RISCV_0)

`in1` is **not** gathered over the ring. Each core reads its own weight blocks from
DRAM (interleaved or DRAM-sharded) or from a global circular buffer (`ENABLE_GLOBAL_CB`),
but in **ring-rotated order** so the weights line up with the rotating `in0` shards:

```cpp
uint32_t block_idx = (ring_idx + block) % num_blocks;
```

- DRAM paths double-buffer `cb_in1` (`2 * in0_shard_width_in_tiles * per_core_N` tiles)
  and push blocks front-to-back; compute consumes them with `in1_index_subblock_offset = 0`.
- Non-DRAM/L1-resident case keeps the full tensor in `cb_in1` and compute indexes it
  directly with `in1_index_subblock_offset = in1_block_num_tiles * curr_ring_idx`.
- With `STREAMING_IN1` the prefetcher delivers blocks in ring-rotated FIFO order through
  the GCB; the kernel gates `remote_cb_pop_front` on the `in1` CB's consumer ack so a
  slot is never recycled before the unpacker HW has drained it.

---

## 7. Summary of the pipeline

```
        ┌──────────── worker_i ────────────┐
        │ cb_in0 (local K-slice i)         │
        │ cb_in2 (remote slices, ring-1)   │──push_back──▶ compute
        │ signal_sem (credit counter)      │
        └──────────────┬───────────────────┘
                       │ async_write(shard) + sem.up(+1)
                       ▼
                 next core (i-1, or hop / wrap)
```

1. Host builds a unidirectional ring over the `in0` shard grid, optionally splicing hop
   relay cores into the wrap-around edge.
2. Every worker starts by sending its local shard; it then forwards each received shard,
   one per step, and pushes it into `cb_in2`.
3. A per-core semaphore carries write→signal credit so a core never forwards data that
   has not landed yet.
4. Compute walks `ring_size` blocks in `(ring_idx + block) % ring_size` order: block 0
   from `cb_in0`, blocks `1..ring_size-1` from `cb_in2`, accumulating against the
   matching `in1` block (read in the same rotated order).
5. After `ring_size - 1` forwards, every core holds all shards and has produced its
   `per_core_M × per_core_N` output tile.

---

## 8. Files

| File | Role |
|------|------|
| `ttnn/cpp/ttnn/operations/matmul/device/config/matmul_program_config_types.hpp` | `MatmulMultiCoreReuseMultiCast1DProgramConfig` (`gather_in0`, `hop_cores`, `stream_in1`) |
| `ttnn/cpp/ttnn/operations/matmul/device/matmul_1d_type.hpp` | `Matmul1DType::GATHER_IN0` |
| `ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp` | gather_in0 validation / dispatch |
| `ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp` | ring construction, CB sizing, runtime/compile args |
| `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_ring_all_gather.cpp` | **in0 ring gather reader** (this doc's focus) |
| `ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_ring_all_gather.cpp` | weight reader (ring-rotated) |
| `ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation_gathered.cpp` | compute consuming gathered shards |
