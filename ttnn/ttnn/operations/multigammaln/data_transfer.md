# Data Transfer Analysis: multigammaln

Pure elementwise SFPU operation; one output tile per input tile, no reduction,
no broadcast, no cross-tile dependency. The DRAM traffic is bounded below by
1.0× the input size + 1.0× the output size.

## Tensor Inventory

| Tensor | Role | Shape | Pages | Page Size | Total Bytes | Memory |
|--------|------|-------|-------|-----------|-------------|--------|
| input  | Per-element argument `a` | rank-4 `(N, C, H, W)`, `H%32==0`, `W%32==0` | `total_tiles = N·C·H·W / 1024` | `tile_size(float32)` = 4096 B (includes tile header) | `total_tiles × 4096` B | DRAM (or L1) interleaved |
| output | `multigammaln(a, 4)` | same as input | `total_tiles` | 4096 B | `total_tiles × 4096` B | DRAM by default; L1 if caller passes `memory_config=L1_MEMORY_CONFIG` |

Per-iteration intermediates (L1 only, no DRAM traffic):

| CB | Index | Size | Purpose |
|----|-------|------|---------|
| `cb_input_tiles` | 0 | 2 × 4096 B (double buffer) | Reader → compute |
| `cb_lgamma_a` | 24 | 2 × 4096 B | sub-phase A out, sub-phase B in (offset 0.0) |
| `cb_lgamma_a_half` | 25 | 2 × 4096 B | offset 0.5 |
| `cb_lgamma_a_one` | 26 | 2 × 4096 B | offset 1.0 |
| `cb_lgamma_a_three_halves` | 27 | 2 × 4096 B | offset 1.5 |
| `cb_output_tiles` | 16 | 2 × 4096 B | Compute → writer |

Total per-core L1 footprint for CBs: `6 × 2 × 4096 = 49,152 B ≈ 48 KiB` — well under
the 1.5 MiB Tensix L1 budget.

## DRAM Read Profile

| Source | Tiles per core | Tiles total (all cores) | Transaction Size | Total Bytes | Notes |
|--------|---------------|-------------------------|------------------|-------------|-------|
| `input_tensor` | `pages_per_core_g1` or `pages_per_core_g2` | `total_tiles` | 4096 B (single tile) | `total_tiles × 4096` | `split_work_to_cores` assigns unique contiguous tile ranges — no overlap |

Reader kernel pattern (`multigammaln_reader.cpp:25–32`):

```cpp
for (uint32_t tile_id = start_tile_id; tile_id < end_tile_id; ++tile_id) {
    cb_reserve_back(cb_input_tiles, 1);
    noc_async_read_tile(tile_id, input_accessor, get_write_ptr(cb_input_tiles));
    noc_async_read_barrier();
    cb_push_back(cb_input_tiles, 1);
}
```

- One `noc_async_read_tile` per output tile.
- One barrier per tile — reads are serialized. Pipeline overlap relies on the
  double-buffered `cb_input_tiles` and the fact that the compute kernel does
  ~30 SFPU operations per tile before requesting the next one, so reader latency
  is hidden.
- TensorAccessor (not the deprecated InterleavedAddrGen) — pages resolve to
  banks via the standard round-robin (`page_id % num_banks`).
- **Read amplification**: 1.0× input size.

## DRAM Write Profile

| Destination | Tiles per core | Tiles total | Transaction Size | Total Bytes |
|-------------|---------------|-------------|------------------|-------------|
| `output_tensor` | per-core split | `total_tiles` | 4096 B (single tile) | `total_tiles × 4096` |

Writer kernel pattern (`multigammaln_writer.cpp:25–32`):

```cpp
for (uint32_t tile_id = start_tile_id; tile_id < end_tile_id; ++tile_id) {
    cb_wait_front(cb_output_tiles, 1);
    noc_async_write_tile(tile_id, output_accessor, get_read_ptr(cb_output_tiles));
    noc_async_write_barrier();
    cb_pop_front(cb_output_tiles, 1);
}
```

- One write per output tile, paired with a per-tile barrier.
- **Write amplification**: 1.0× output size.

## Cross-Core Data Duplication

| Tensor | Duplication | Details |
|--------|------------|---------|
| `input_tensor`  | None | Each core reads a unique, contiguous range of tile IDs assigned by `split_work_to_cores`. |
| `output_tensor` | None | Each core writes its own slice. |
| `cb_lgamma_*`, `cb_input_tiles`, `cb_output_tiles` | N/A | Intermediates — never leave the producing core's L1. |

**Multicast used**: No.
**Multicast opportunity**: None — there is no shared DRAM data to broadcast.

## NoC Channel Balance

| Channel | Direction | Bytes (full op) | Load |
|---------|-----------|-----------------|------|
| NoC0 | Read (DRAM → L1) | `total_tiles × 4096` | 50% |
| NoC1 | Write (L1 → DRAM) | `total_tiles × 4096` | 50% |

**Balance ratio**: 1:1.
**Assessment**: ideal — every reader byte is paired with an equal-sized writer byte
on the opposite NoC. The op is compute-bound (the SFPU lgamma recipe is ~30
operations per tile and runs four times); DRAM bandwidth is not the limiter.

## Work Distribution

| Metric | Value |
|--------|-------|
| Total cores | `min(grid_size, total_tiles)` (via `split_work_to_cores`) |
| Tiles per core (group 1) | `pages_per_core_g1` |
| Tiles per core (group 2) | `pages_per_core_g2` (= `pages_per_core_g1 - 1` when remainder > 0, else group 2 is empty) |
| Load imbalance | ≤ 1 tile across cores (the two-group split bound) |

Per-core RT args are populated by iterating `core_group_1` then `core_group_2`,
tracking a running `start_tile_id`. Reader, writer, and compute share the same
per-core tile range so they all process the same physical tiles in lockstep.

## Key Observations

1. **No data-movement waste.** Read amplification 1.0×, write amplification 1.0×.
   Every DRAM byte is touched exactly once.
2. **Balanced NoC usage** by construction — elementwise unary on interleaved
   tensors is the canonical 1:1 case.
3. **Compute-bound, not bandwidth-bound.** The per-tile SFPU work is large
   (the lgamma recipe runs four times per input tile, each ~30 SFPU operations).
   Reader/writer latency is fully hidden behind the compute kernel.
4. **Per-tile barriers in reader/writer** could be relaxed by batching
   `noc_async_read_tile` calls across multiple tiles and issuing a single
   barrier per batch. With the current double-buffer CB sizing (2 pages) the
   gain would be at most 1 tile of pipeline depth — marginal because compute
   dominates. Listed as a future refinement only if profiling shows DRAM stalls.
5. **L1 pressure is low** (~48 KiB of CBs per core). Plenty of headroom for
   future enlargements (e.g., larger intermediate CBs to enable cross-tile
   batching).
