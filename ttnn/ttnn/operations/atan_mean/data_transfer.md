# Data Transfer Analysis: atan_mean

> Static analysis of the data transfer profile from reading
> `atan_mean.py`, `atan_mean_program_descriptor.py`, and the three kernels.

## 1. Tensors and Roles

| Tensor | Role | Shape | Pages (tiles) | Page Size | Total Bytes |
|--------|------|-------|---------------|-----------|-------------|
| `input_tensor` | DRAM input | `(N, C, H, W)` fp32 TILE | `N·C·Ht·Wt` | 4096 B | `N·C·Ht·Wt × 4096` |
| `output_tensor` | DRAM output | `(N, C, H, 1)` fp32 TILE (padded to `(N,C,H,32)`) | `N·C·Ht` (one W-tile per row-tile) | 4096 B | `N·C·Ht × 4096` |

Where `Ht = H/32`, `Wt = W/32`.

`cb_scaler` is an L1-only persistent CB. It carries a single bf16 tile (2048 B) computed once per core at program startup; it is not read from DRAM.

## 2. Reader Kernel (NCRISC / NoC0)

`atan_mean_reader.cpp`:

```cpp
for r in [start_row_tile, start_row_tile + num_row_tiles):
    base_tile = r * Wt
    for wt in [0, Wt):
        noc_async_read_tile(base_tile + wt, src_accessor, l1)
        noc_async_read_barrier()       // per-tile barrier
        cb_push_back(cb_input_tiles, 1)
```

Plus a single program-startup call to `calculate_and_prepare_reduce_scaler<cb_scaler, AVG, REDUCE_ROW, W>()` — this writes the `1/W` value into `cb_scaler` *locally* (no DRAM traffic).

**Per core**: `num_row_tiles_this_core × Wt` `noc_async_read_tile` calls — each fetches one 4096-byte fp32 tile.

**Globally** (sum over all cores): exactly `total_row_tiles × Wt = N·C·Ht·Wt` reads = one read per input tile.

## 3. Writer Kernel (BRISC / NoC1)

`atan_mean_writer.cpp`:

```cpp
for r in [start_row_tile, start_row_tile + num_row_tiles):
    cb_wait_front(cb_output_tiles, 1)
    noc_async_write_tile(r, dst_accessor, l1)
    noc_async_write_barrier()
    cb_pop_front(cb_output_tiles, 1)
```

**Per core**: `num_row_tiles_this_core` `noc_async_write_tile` calls (4096 B each).

**Globally**: `total_row_tiles = N·C·Ht` writes — one per output tile.

## 4. Transaction Inventory

| Path | Direction | Count (global) | Bytes per Txn | Total Bytes |
|------|-----------|----------------|---------------|-------------|
| DRAM → reader CB | NoC0 read | `N·C·Ht·Wt` | 4096 | `N·C·Ht·Wt × 4096` |
| writer CB → DRAM | NoC1 write | `N·C·Ht` | 4096 | `N·C·Ht × 4096` |

**Scaler tile**: written into `cb_scaler` locally on each core via host-driven CB push (no NoC traffic). No cross-core multicast and no DRAM hit.

## 5. Bandwidth Summary

- **Input bytes (DRAM → device)**: `N·C·H·W × 4` (one read per element, no amplification).
- **Output bytes (device → DRAM)**: `N·C·H × 4` of valid data; physically `N·C·H × 32 × 4` because the trailing tile is padded to 32 columns (only column 0 holds the mean). The physical write volume is therefore `32×` the valid-output size.

**Read amplification**: 1.0× (every input element is fetched exactly once globally).
**Write amplification**: 32× over valid output bytes (intrinsic to the col-0-fill REDUCE_ROW output layout — the cost of pushing one whole tile per row-tile to avoid a host-side gather).

## 6. Cross-Core Duplication

| Tensor | Per-core read pattern | Duplication |
|--------|----------------------|-------------|
| `input_tensor` | Disjoint slice via `split_work_to_cores` — core `k` gets `[start_row_tile_k, start_row_tile_k + num_row_tiles_k)`. No overlap. | **None.** |
| `output_tensor` | Disjoint writes — same partition as the reader. | **None.** |
| Scaler tile | Each core independently produces its own `1/W` tile via `calculate_and_prepare_reduce_scaler`. The value is a constant. | Computed redundantly on all `num_cores` cores, but **not** read from DRAM — so this is L1 duplication, not DRAM duplication. Cost is negligible (one ~32-element write per core at startup). |

There is **no broadcast, multicast, or weights tensor**. The work distribution gives perfect 1.0× DRAM amplification on both input and output.

## 7. NoC Channel Balance

```
NoC0 (reads)  = N·C·Ht·Wt × 4096 bytes
NoC1 (writes) = N·C·Ht × 4096 bytes
Balance ratio = Wt : 1
```

| Shape | `Wt` | Read:Write ratio |
|-------|------|------------------|
| `(1, 1, 32, 32)` | 1 | 1:1 (perfectly balanced) |
| `(1, 1, 2048, 64)` | 2 | 2:1 (read-leaning) |
| `(1, 256, 64, 64)` | 2 | 2:1 |
| `(1, 1, 1024, 128)` | 4 | 4:1 (clearly read-heavy) |

For the operation's expected workloads (rows of 32–128 elements collapsing to 1 mean), the channel is mildly read-leaning. The writer is idle for `Wt - 1` ticks per row-tile while the reader is fetching, so on multi-Wt shapes NoC1 has spare capacity that the algorithm cannot exploit.

## 8. Compute-to-Data Ratio

Per row-tile (`Wt` input tiles in, 1 output tile out):

| Phase | Work | DRAM cost |
|-------|------|-----------|
| Reader | `Wt` 4096-byte reads | `Wt × 4096` B |
| Compute (SFPU atan) | `Wt × 1024` SFPU elementwise atan_tile ops (one per fp32 datum in a tile) | 0 (L1-only) |
| Compute (matmul-mode REDUCE_ROW AVG) | One 32×32 × 32×Wt matmul (because the matmul reduce path is dispatched for AVG+REDUCE_ROW) | 0 |
| Writer | 1 × 4096-byte write | 4096 B |

For Wt=2 (common), the compute kernel does 2048 atan SFPU evaluations + one matmul against a bf16 scaler tile per ~12 KB of NoC traffic. This is **strongly compute-bound** on the atan operation in fp32/HiFi4 — the atan SFPU instruction is multi-cycle and the unpacker reconfig (fp32 input ↔ bf16 scaler) adds further compute-side overhead.

## 9. Work Distribution

`ttnn.split_work_to_cores(grid_size, total_row_tiles)` returns two groups with row-tile counts differing by at most 1. The factory walks both groups in order and assigns a per-core `(start_row_tile, num_row_tiles)` pair. Load balance is therefore optimal — at most one row-tile of imbalance across all cores regardless of `total_row_tiles`.

Tail behavior: when `total_row_tiles < num_cores` (e.g. `(1, 1, 32, 32)` → 1 row-tile total on an 8×8 grid), only one core does work and the remaining 63 are idle. This is unavoidable for the unit-of-work granularity defined by the design (1 row-tile = 1 work unit).

## 10. Optimization Opportunities

| # | Opportunity | Cost / Benefit |
|---|-------------|----------------|
| 1 | **Batched reads with single barrier per row-tile**: replace `noc_async_read_barrier()` per inner tile with one barrier after the `Wt` issues, allowing `Wt` reads to pipeline. | Reduces per-tile NoC stall on multi-Wt shapes; trivial code change. Compute is sequential within a row-tile (waits on full row before reduce), so this does not change the critical path for Wt ≤ 2 but helps for Wt ≥ 4. |
| 2 | **Padded write avoidance**: the output is allocated as a full tile per row but only column 0 is valid. Could the writer untilize into a 1-element-wide row-major buffer to avoid the 32× padding? This would change the output memory layout — not a free win, but worth measuring on the high-channel regime where output traffic is non-trivial. | High effort; only relevant when `total_row_tiles × 4096` is a measurable fraction of total bandwidth. |
| 3 | **Memory pressure on extreme Wt**: `cb_atan_tiles` is sized to `Wt` fp32 tiles = `Wt × 4096` B. For W = 8192 → Wt = 256 → 1 MB just for this CB on every core, which would not fit in 1.5 MB L1 once other CBs are accounted for. The current Phase-0 shape set caps Wt at 4 (W=128), so this is a future concern for refinements. | Will need a tile-block-streaming variant of the helper pair (or a streaming reduce) to lift the W ceiling. |
| 4 | **No multicast needed**: the scaler is constant-valued and computed locally per core. Multicasting would be more expensive (semaphores + 1 sender + N receivers) than the current local recompute. Not an opportunity, but worth noting that the trivial pattern is the right one. | — |

## 11. Summary

- **DRAM read amplification**: 1.0× (no duplication, perfect work split).
- **DRAM write amplification**: 32× over valid output bytes (col-0 padding intrinsic to REDUCE_ROW output layout).
- **NoC channel balance**: `Wt : 1` (read-leaning; mild for Wt=1–2, more significant for Wt=4+).
- **Cross-core duplication**: none in DRAM. Local scaler recompute on every core (≤ 32 fp32 writes per core; negligible).
- **Compute-to-data ratio**: compute-bound on atan SFPU in fp32/HiFi4 — DRAM traffic is well within steady-state NoC capacity.
- **Bottleneck**: SFPU atan throughput on the compute kernel. Reader/writer are well below their channel limits for the Phase-0 shape set.
