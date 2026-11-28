# g6_sharded_to_interleaved changelog

## Phase 0: owner_writes, as designed

- One writer kernel per shard core (BRISC / NoC1) writes its resident shard rectangle straight to
  the interleaved pages: one NoC write per tile, zero reads, one barrier. `cb_input_shard` is
  zero-copy on the input shard. The design has no reader or compute kernel, so none exists.
- Knobs live in `Knobs` (`num_write_lanes=1`, `block_tile_rows/cols=None` = whole lane rectangle).
  The memoized descriptor is patched per call: output address (common RT) and shard CB binding.
- SUPPORTED: bf16 / TILE / width_sharded / dram. The bf8b, height-sharded and L1-output paths
  already pass bit-exact through `_dispatch` (impl debug test), so each refinement only widens
  SUPPORTED.

### Measurements (Blackhole p150b, 11x10 grid, device kernel duration, single fresh run each)

| Case | default (1 lane) | `num_write_lanes=2` | all writes stubbed | address math only |
|------|------------------|---------------------|--------------------|-------------------|
| p2 (1,5120) bf16 width -> DRAM, 32 cores | 1500, 1419 ns | 1561, 1524 ns | 590 ns | 641 ns |
| p1 (1,128,256,32) bf8b height -> L1, 103 cores | 2118, 2118 ns | 1879, 1964, 1917 ns | 807 ns | 919 ns |

Findings for the refinements:
- p2 moves 327,680 B in ~0.88 us of payload (~370 GB/s), close to the DRAM write ceiling; the second
  lane on NoC0 is slower, as the `noc_placement` catalog entry predicts for DRAM writes.
- p1 is issue/latency-bound, not bandwidth-bound: a second lane takes ~200 ns off (2118 -> ~1920 ns,
  under the 2.05 us incumbent). Per-page `get_noc_addr` costs ~11 ns/page (112 ns of 2118), so the
  incremental bank-walk lamp is worth at most ~5%.
- A per-core start-row stagger against L1 bank collisions measured slower (2282, 2360 ns) because
  of its per-row modulo; not kept.
- The lane default stays 1, as the design sets it: the descriptor carries no branch on buffer type,
  and the design asks for the lane lamp to be judged on trace-replay time (RB-053) once p1 is in
  SUPPORTED.
