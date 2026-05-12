# Data Transfer Analysis: glu_fused

## Algorithm Summary

`glu_fused` is a **pure pointwise** operation along the last dim — every output
element depends on exactly **two** input elements (one from each half of the
W dimension). There is no reduction, no broadcast, and no cross-core
communication.

The split happens at the **tile-id level inside the reader**: for each
output tile `out_idx`, the reader reads one A-tile from the first half and
one B-tile from the second half of the same input tile-row. Both halves
are tile-aligned (`W % 64 == 0` is validated host-side), so no sub-tile
masking is needed.

---

## Tensor Inventory

| Tensor | Role | Shape | Pages (tiles) | Page Size | Total Bytes | Memory |
|--------|------|-------|---------------|-----------|-------------|--------|
| input  | x    | `(N, C, H, W)` | `N · C · (H/32) · (W/32)` | 4096 B (fp32 tile) | `pages × 4096` | DRAM or L1 interleaved |
| output | glu(x) | `(N, C, H, W/2)` | `N · C · (H/32) · (W/64)` | 4096 B (fp32 tile) | `pages × 4096` | inherits input |

- `Wt = W / 32` — tile-cols per row of input.
- `Wt_half = W / 64` — tile-cols per row of output (also tile-cols per half
  of the input row).
- `Ht = H / 32`.
- `total_output_tiles = N · C · Ht · Wt_half`.
- `total_input_tiles  = 2 · total_output_tiles = N · C · Ht · Wt` (each input
  tile is read exactly once).

---

## DRAM Read Profile

| Source | Reads per output tile | Reads per core (g1) | Reads (total, all cores) | Transaction size | Bytes (total) |
|--------|----------------------|---------------------|--------------------------|------------------|---------------|
| input (A half) | 1 | `tiles_per_core_g1` | `total_output_tiles` | 4096 B | `total_output_tiles × 4096` |
| input (B half) | 1 | `tiles_per_core_g1` | `total_output_tiles` | 4096 B | `total_output_tiles × 4096` |

Both reads go through the same `TensorAccessor` (one input buffer, distinct
tile IDs). They are issued **concurrently** on NoC0 with a single
`noc_async_read_barrier` per output tile — the fix this verification round
applied to the reader. Before the fix: 2 barriers per output tile; after:
1 barrier per output tile.

**Total DRAM read transactions**: `2 × total_output_tiles` per program.
**Total bytes read**: `2 × total_output_tiles × 4096` = exactly `total_input_tiles × 4096`.

**Read amplification**: **1.0×** of the input tensor (each input tile is
read exactly once across all cores — `split_work_to_cores` partitions
output tiles, and each output tile owns its unique A and B input tiles).

---

## DRAM Write Profile

| Destination | Writes per output tile | Writes per core (g1) | Writes (total) | Transaction size | Bytes (total) |
|-------------|------------------------|----------------------|----------------|------------------|---------------|
| output      | 1 | `tiles_per_core_g1` | `total_output_tiles` | 4096 B | `total_output_tiles × 4096` |

One `noc_async_write_tile` per output tile on NoC1, followed by a per-tile
`noc_async_write_barrier` and `cb_pop_front`. The writer drains
`cb_output_tiles` (2-page double buffer) one tile at a time.

**Total DRAM write transactions**: `total_output_tiles`.
**Total bytes written**: `total_output_tiles × 4096` = exactly the output
tensor size.

---

## Cross-Core Data Duplication

| Tensor | Duplication | Reason |
|--------|------------|--------|
| input  | None | `split_work_to_cores` assigns unique, contiguous output tile-id ranges to each core. A and B tile IDs are functions of `out_idx` only, so distinct cores read distinct input tiles. |
| output | None | Each core writes its own unique output tile-id range. |

**Multicast used**: No.
**Multicast opportunity**: None — there is no shared input data. Every
input tile is needed by exactly one output tile (which is owned by exactly
one core).

---

## NoC Channel Balance

| Channel | Direction | Total Bytes | Load |
|---------|-----------|-------------|------|
| NoC0 | Read (DRAM → L1) | `2 × total_output_tiles × 4096` | 2/3 of total |
| NoC1 | Write (L1 → DRAM) | `1 × total_output_tiles × 4096` | 1/3 of total |

**Balance ratio**: **2:1** (read-heavy). This is **inherent** to the GLU
algorithm — the input is twice the size of the output, so the read side
necessarily moves twice the data the write side does. There is no
algorithmic way to reduce this without changing what `glu` computes.

Both reader (NCRISC/NoC0) and writer (BRISC/NoC1) use their default NoC
assignments via `ReaderConfigDescriptor()` / `WriterConfigDescriptor()`.

---

## Work Distribution

| Metric | Value (worked example: shape (1,1,32,128) on 8×8 grid, total=2 tiles) |
|--------|----------------------------------------------------------------------|
| Total work units | `total_output_tiles = N · C · Ht · Wt_half` |
| Grid | `device.compute_with_storage_grid_size()` (8×8 = 64 cores on Wormhole) |
| Cores used | `min(total_output_tiles, num_grid_cores)` |
| Per-core (group 1) | `ceil(total / num_cores)` |
| Per-core (group 2) | `floor(total / num_cores)` (one fewer than g1, or absent if even split) |
| Load imbalance | at most 1 tile, i.e. `(g1 - g2) / g1 = 1 / g1` |

For tiny shapes (`total_output_tiles ≤ num_cores`) the parallelism is
limited by the work — only `total_output_tiles` cores get any work.
This is acceptable for Phase 0; refinements may explore intra-tile
splitting for tiny shapes if perf demands it.

---

## Per-Iteration Transaction Pattern

### Reader (one iteration per output tile, after the verifier fix)

```cpp
cb_reserve_back(cb_input_a, 1);
cb_reserve_back(cb_input_b, 1);
noc_async_read_tile(a_tile_idx, src_accessor, l1_write_addr_a);
noc_async_read_tile(b_tile_idx, src_accessor, l1_write_addr_b);
noc_async_read_barrier();   // single barrier covers both reads
cb_push_back(cb_input_a, 1);
cb_push_back(cb_input_b, 1);
```

Two outstanding NoC0 transactions per iteration, one barrier per iteration.

### Writer (one iteration per output tile)

```cpp
cb_wait_front(cb_output_tiles, 1);
noc_async_write_tile(out_idx, dst_accessor, l1_read_addr);
noc_async_write_barrier();
cb_pop_front(cb_output_tiles, 1);
```

One outstanding NoC1 transaction per iteration, one barrier per iteration.

---

## Key Observations

1. **No data duplication anywhere.** Read amplification is 1.0× — the
   theoretical minimum for this algorithm. The reader's tile-id math is
   what guarantees this: each output tile owns its A and B input tiles
   uniquely.

2. **Read-heavy is inherent.** The 2:1 NoC0:NoC1 ratio cannot be improved
   — the input is twice the output by definition. Both channels are
   active (no severe idle channel).

3. **Concurrent in-flight reads.** Both A and B tile reads share NoC0 and
   are issued without a barrier between them. This overlaps the two NoC
   transactions and removes one barrier per output tile compared to a
   naive one-barrier-per-read pattern.

4. **Tiny-shape parallelism.** When `total_output_tiles ≤ num_cores`,
   parallelism is bounded by the work, not the grid. This is benign for
   small shapes (the launch is fast anyway) but is a candidate for
   intra-tile work splitting in a future refinement if a perf use case
   demands it.

5. **TensorAccessor is bank-aware.** Both reader and writer use
   `TensorAccessor` with `TensorAccessorArgs` compiled in — bank/offset
   resolution for interleaved tensors is handled automatically. No
   manual bank addressing in the kernels.
