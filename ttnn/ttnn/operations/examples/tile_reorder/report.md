# tile_reorder — measured report

WH B0 n150 numbers live in [`README.md`](README.md) (*When it helps / when it does not*).

## Blackhole p150b — bh-qb-15-special-adamjanovic-for-reservation-139410 (2026-10-19)

| stamp | value |
|---|---|
| arch / board | Blackhole p150b, compute grid 12×10 = 120 Tensix cores, 8 DRAM banks, AICLK 1350 MHz |
| commit | `f74ee7aad64` + arch-agnostic regimes tests (branch `adamjanovic/ab-tilize-run-scoreboard-catalog`), Release build |
| metric / method | `test_tile_reorder_regimes`: `DEVICE KERNEL DURATION [ns]`, baseline `scatter` (4 × 512 B writes + 4 write barriers per tile) vs lever `relocate` (1 × 2 KB write + 1 barrier per tile), interleaved A,B,A,B, median of 12 trials × 3 launches; full grid; input DRAM interleaved (`TensorMemoryLayout::INTERLEAVED`), bf16 |

### Regimes and shape sweep

`(2048, 1024)` = 2048 tiles = 64 tile-rows → ~17 tiles per Tensix core on 12×10 (one tile-row per Tensix core only
on the WH B0 8×8 grid). `(3840, 1024)` = 3840 tiles = 120 tile-rows → one tile-row (32 tiles) per Tensix core on
12×10, the BH equivalent of the WH B0 shape.

| case | shape | output | scatter median ns | relocate median ns | speedup | current test asserts |
|---|---|---|---|---|--:|---|
| helps_l1_interleaved_out | (3840, 1024) | L1 interleaved | 38446 | 25612 | **1.50×** | ≥ 1.20× |
| flat_l1_local_shard_out | (3840, 1024) | L1 `HEIGHT_SHARDED`, 1 tile-row per Tensix core | 25645 | 25448 | 1.01× | ≤ 1.05× |
| DRAM out at the WH shape | (2048, 1024) | DRAM interleaved | 21180 (21045–21293) | 21387 (20706–21955) | 0.99× | sweep / probe — not asserted |
| measure_dram_out (probe) | (3840, 1024) | DRAM interleaved | 48209 | 43962 | 1.10× | measure (logged only — verdict differs by arch) |
| L1 interleaved at the WH shape | (2048, 1024) | L1 interleaved | 18906 (18873–18987) | 11320 (11233–11361) | 1.67× | — |
| L1 sharded at the WH shape (shard not local) | (2048, 1024) | L1 `HEIGHT_SHARDED`, 64 tile-rows over 120 Tensix cores | 18946 (18920–18977) | 11286 (11231–11379) | 1.68× | — |

`(2048, 1024)` rows are a BH run of the regimes test at its earlier fixed shape (their test log is tagged "WH B0"
but the device was BH p150b); `(3840, 1024)` rows come from a probe of the same A/B at that shape.

### Reading it
- **DRAM output flips from helps (WH B0 n150 1.58×) to flat (0.99×).** With DRAM in and out, `(2048, 1024)`
  moves 8.4 MB; `scatter` already does it in ~21 µs, i.e. ~400 GB/s, the BH p150b DRAM ceiling, so the writer's
  4 barriers per tile are hidden behind DRAM bandwidth. At `(3840, 1024)` (15.7 MB) `scatter` is at ~326 GB/s
  and `relocate` gains only 1.10×.
- **Remote L1 output still helps (1.50–1.67×)**, as on WH B0 n150 (1.40×): the per-face write barrier, not
  DRAM bandwidth, is the cost whenever the destination is remote L1.
- **Local output is flat (1.01×) only when the shard really is local.** At the WH shape the 64 tile-rows do not
  map one per Tensix core on 12×10, the "local" shard becomes remote writes and coalescing gains 1.68×. Size
  shapes to the attached grid before reading a local-shard result.
