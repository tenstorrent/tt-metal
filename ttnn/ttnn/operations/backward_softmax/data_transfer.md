# Data Transfer Analysis: backward_softmax

## Algorithm Summary

`backward_softmax` is a two-pass streaming VJP. Each lane (a row of tiles for
`dim=-1`, a column of tiles for `dim=-2`) is processed end-to-end on a single
core. Both inputs are re-read from DRAM in pass 2; the output is written once.

Key transfer counts (per lane, `Wt` = tiles in W, `Ht` = tiles in H, with the
reduce dimension determining which is the "reduce-axis tile count" `R`):

| Phase | Source | Sink | Transactions per lane | Notes |
|-------|--------|------|------------------------|-------|
| Reader pass 1 (dy) | DRAM (`grad_output`) | `cb_grad_output` | `R` × `noc_async_read_tile` | NoC0, lockstep with y |
| Reader pass 1 (y) | DRAM (`output`) | `cb_output` | `R` × `noc_async_read_tile` | NoC0, lockstep with dy |
| Reader pass 2 (dy) | DRAM (`grad_output`) | `cb_grad_output` | `R` × `noc_async_read_tile` | re-read same tiles |
| Reader pass 2 (y) | DRAM (`output`) | `cb_output` | `R` × `noc_async_read_tile` | re-read same tiles |
| Writer | `cb_grad_input` | DRAM (`grad_input`) | `R` × `noc_async_write_tile` | NoC1, one tile at a time |
| Reader scaler (once per program) | local (`MEM_ZEROS_BASE`) | `cb_scaler` | 1 zero-fill + 1 `pack_tile`-style fill | Done by `calculate_and_prepare_reduce_scaler` at startup |

`R = Wt` for `dim=-1`, `R = Ht` for `dim=-2`. The reduce direction is the only
dimension along which a "lane" runs — the orthogonal-axis tile count is 1.

## Per-Lane Transfer Inventory

| Direction | Tensor | Bytes per lane | Reads | Writes |
|-----------|--------|----------------|-------|--------|
| DRAM → core | `grad_output` (dy, fp32) | `2 × R × 4096` | `2 × R` |  |
| DRAM → core | `output` (y, fp32) | `2 × R × 4096` | `2 × R` |  |
| core → DRAM | `grad_input` (dx, fp32) | `R × 4096` |  | `R` |
| **Per-lane total** | | **`5 × R × 4096`** = `20480 × R` B | `4R` reads | `R` writes |

The 2× factor on inputs is the two-pass nature of the algorithm. The output
is written exactly once.

## Per-Program Transfer Totals

Let `L = total_lanes`. For `dim=-1`, `L = N × C × Ht`; for `dim=-2`,
`L = N × C × Wt`. In both cases the sum-of-lane-tiles equals `N × C × Ht × Wt`.

| Quantity | Formula | Bytes |
|----------|---------|-------|
| DRAM read total | `2 × N × C × Ht × Wt × 4096 × 2` | `2 × footprint × 2` (both inputs, both passes) |
| DRAM write total | `N × C × Ht × Wt × 4096` | `1 × footprint` |
| **Total DRAM bandwidth** | **`5 × N × C × Ht × Wt × 4096`** | `5 × footprint` |

For shape `(2, 4, 64, 128)` (Ht=2, Wt=4, NC=8): footprint = `8 × 8 × 4096` =
**256 KiB** per tensor. Total DRAM bandwidth = `5 × 256 KiB` = **1.25 MiB**.

For transformer-typical `(1, 16, 512, 512)` (Ht=16, Wt=16, NC=16): footprint
= `16 × 256 × 4096` = **16 MiB** per tensor. Total DRAM bandwidth =
**80 MiB**. Compared to a fused softmax-backward that streams once per pass
this is the same 5× factor; if the inputs+output can stay in L1 across the
two passes, that 5× drops to 3× (read each input once, write output once).

## Transactions per Program

| Transaction | Count | NoC channel |
|-------------|-------|-------------|
| `noc_async_read_tile` (dy + y, both passes) | `4 × N × C × Ht × Wt` | NoC0 (NCRISC reader) |
| `noc_async_read_barrier` | one per inner reader iteration: `2 × N × C × Ht × Wt` | NoC0 |
| `noc_async_write_tile` (grad_input) | `N × C × Ht × Wt` | NoC1 (BRISC writer) |
| `noc_async_write_barrier` | one per writer iteration: same as above | NoC1 |
| Scaler `zero_tile` reads | 1 per program, broken into `tile_size / MEM_ZEROS_SIZE` chunks | NoC0 |

The reader uses `noc_async_read_barrier()` once per inner iteration (after
both dy and y reads) — this is acceptable because of the small per-tile
working set, but is a noticeable per-tile overhead at high tile counts. The
writer issues a barrier per tile.

## L1 Footprint per Core

Using `BLOCK_SIZE = B` (default = largest divisor of the reduce-axis tile
count, ≤ 8) and float32 tile size 4096 B:

| CB | Pages | Page size | Bytes per core |
|----|-------|-----------|-----------------|
| `cb_grad_output` | 2 | 4096 | 8 192 |
| `cb_output` | `2 × B` | 4096 | `8 192 × B` |
| `cb_scaler` | 1 | 2048 | 2 048 |
| `cb_grad_input` | 2 | 4096 | 8 192 |
| `cb_prod` | `2 × B` | 4096 | `8 192 × B` |
| `cb_sum` | 2 | 4096 | 8 192 |
| `cb_centered` | 2 × B | 4096 | `8 192 × B` |

Total per core: `26 624 + 24 576 × B` bytes. For `B = 8` (the maximum default):
**220 672 B** ≈ **216 KiB**. Well under the 1.5 MiB L1 budget per Tensix.

`cb_output` is sized `2 × BLOCK_SIZE` rather than the obvious `2` to avoid a
**deadlock** in pass 2 (sub consumes only dy, mul consumes only y; reader
pushes them in lockstep, so y must hold a full block while sub drains dy).
This is the largest single CB by L1 footprint and is essential.

## NoC Channel Balance

| NoC | User | Traffic per program (bytes) | Pattern |
|-----|------|------------------------------|---------|
| NoC0 | Reader (NCRISC) | `4 × footprint` | Burst reads, lockstep on dy/y per tile, two passes |
| NoC1 | Writer (BRISC) | `1 × footprint` | One write per tile, per-tile barrier |

The reader does **4× more DRAM traffic** than the writer. This is unbalanced
and reader-bound — the writer NoC is largely idle.

## Streaming Pattern Notes

- **Lockstep dy/y push**: the reader pushes one dy tile and one y tile per
  inner iteration. This is required for pass-1 mul (consumes both per-tile)
  and accepted for pass-2 (where sub consumes dy and mul consumes y at
  different cadences — `cb_output` sizing absorbs the impedance mismatch).
- **Double-buffered streaming CBs**: `cb_grad_output` and `cb_grad_input` are
  2-page double buffers — the reader/writer can pre-fetch one tile ahead.
- **Persistent scaler**: `cb_scaler` is filled once at program startup and
  never popped; the reduce LLK waits on it but does not consume it.
- **Persistent sum CB across pass-2 blocks**: `cb_sum` is filled once at the
  end of pass 1 and held with `WaitUpfrontNoPop` across all pass-2 blocks of
  the lane; an explicit `cb_pop_front(cb_sum, 1)` at the end of the lane
  releases it before the next lane.
- **No NoC barriers across CBs**: each CB is local to its producer/consumer
  pair; CB push/wait counts match per CB (verified in op_design.md).

## Cross-Core Projection (Multi-Core Refinement)

Phase 0 is single-core. The natural multi-core split (per `op_design.md` work
distribution table) hands each core `pages_per_core_g{1,2}` lanes via
`split_work_to_cores`. There is **no inter-core data exchange**: each core
owns its lanes end-to-end. Specifically:

- No multicast. Each core reads its own DRAM tiles independently.
- No semaphores. Each core's reader→compute→writer pipeline is local.
- No ring topology.
- **No cross-core duplication**. With `L` lanes split across `K` cores, each
  lane is read and written exactly once across the program (same as
  single-core). The L1 footprint per core stays the same; only the lane count
  per core drops.

DRAM bandwidth is unchanged — per-core throughput rises linearly until DRAM
becomes the bottleneck. With float32 tiles (4 KiB/tile) and Wormhole's DRAM
bandwidth at ~~150 GB/s~, even a small chip (~64 cores) saturates DRAM well
before all cores are busy on tile-aligned shapes.

## Bottleneck Analysis

| Regime | Bottleneck | Reasoning |
|--------|-----------|-----------|
| Tiny shapes (`Wt × Ht ≤ 4`) | Compute / dispatch | Per-program fixed costs dominate; DRAM and compute under-utilized. |
| Mid shapes (`Wt × Ht` ~ 16-64) | DRAM read | Reader does 4× DRAM read traffic vs writer's 1× write. fp32 tiles are 4 KiB, so streaming through DRAM at NoC0 line rate is the limit. |
| Large shapes (`Wt × Ht` ≥ 256) | DRAM read, multi-core | Same as mid, scaled out. The 5× footprint factor (2 reads × 2 inputs + 1 write) is intrinsic to the two-pass algorithm. |

A future single-pass refinement that **caches both inputs in L1** between the
two passes would cut DRAM traffic from 5× footprint to 3× (one read per
input, one write of the output), at the cost of `2 × R × 4096 ×
sizeof(lane)` of L1 per core. For typical reduction depths of 32-128 tiles,
this is 128 KiB - 512 KiB per core — feasible only when the lane is small
relative to L1.

## Recommendations

1. **Multi-core (Refinement 1)**: enable `split_work_to_cores` to scale to all
   available cores. No new data movement is introduced; this is pure
   parallelism over lanes.
2. **L1 caching of inputs (future)**: when the operation is moved to a perf
   path, consider buffering dy and y in L1 between passes for shapes whose
   per-lane footprint fits. Cuts DRAM read traffic by 50%.
3. **Writer pipelining**: the writer's per-tile `noc_async_write_barrier()`
   could be replaced with a per-block (BLOCK_SIZE-tile) barrier, halving the
   barrier count at no correctness cost.
4. **NoC0/NoC1 imbalance**: the reader does 4× the writer's traffic on NoC0.
   No mitigation in the current single-core layout; in multi-core, the
   imbalance is per-core but does not stack.
