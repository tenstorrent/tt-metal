# tile_reorder

**Trick:** when the writer's barriers are on the critical path, coalesce into whole-page transfers and batch barriers — relocate whole tiles instead of writing sub-tile faces. Read on NoC0, write on NoC1. (On BH p150b a full-grid DRAM → DRAM move is already at the DRAM ceiling and does not gain; see below.)

**Measured on:** WH B0 n150 (regimes 2026-10-12) · BH p150b (`bh-qb-15-special-adamjanovic-for-reservation-139410`, 2026-10-19)

**Op:** `tile_reorder(t, method="relocate" | "scatter")` — reverses column-tile order. `relocate` = one whole-page write/tile; `scatter` = 4 faces + barrier each (baseline).

**Run:** `scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_tile_reorder.py` (correctness + `test_tile_reorder_regimes`)

## When it helps / when it does not
**Rule:** coalescing pays whenever the **writer's barrier count is on the critical path** — 1 × 2 KB write + 1
barrier per tile instead of 4 × 512 B + 4 barriers. That holds for a remote-L1 output on both boards and for a
DRAM output on **WH B0 n150 only**: on **BH p150b** a DRAM-in, DRAM-out move on the full grid already runs at the
~400 GB/s DRAM ceiling, so the writer's barriers are hidden and coalescing is flat. A destination local to the
writing Tensix core is flat on both boards.

**The shape is per board.** `(2048, 1024)` = 2048 bf16 tiles = 64 tile-rows is one tile-row (32 tiles) per Tensix
core on the WH B0 8×8 grid, but only ~17 tiles per Tensix core on the BH p150b 12×10 grid, where the "local shard"
output is then no longer local. BH p150b L1 rows therefore use `(3840, 1024)` = 120 tile-rows = one per Tensix core;
its DRAM row keeps `(2048, 1024)`. Input is DRAM interleaved (`TensorMemoryLayout::INTERLEAVED`) in every row; only
the **output** placement changes. Baseline = `scatter`, lever = `relocate`.

| Regime (same definition on every board) | WH B0 n150 · 8×8 = 64 Tensix cores | BH p150b · 12×10 = 120 Tensix cores |
|---|---|---|
| **Helps:** DRAM interleaved output, `(2048, 1024)` | 1 tile-row per Tensix core: 71 855 → 45 513 ns **1.58×** | ~17 tiles per Tensix core: 21 180 → 21 387 ns **0.99× — flat on BH** (DRAM-ceiling-bound) |
| DRAM interleaved output, 1 tile-row per Tensix core on BH, `(3840, 1024)` | not measured | 48 209 → 43 962 ns **1.10×** (small) |
| **Helps:** L1 interleaved output (remote L1 banks), 1 tile-row per Tensix core | `(2048, 1024)`: 43 773 → 31 317 ns **1.40×** | `(3840, 1024)`: 38 446 → 25 612 ns **1.50×** (`(2048, 1024)`: 18 906 → 11 320 ns 1.67×) |
| **Does not help:** L1 `HEIGHT_SHARDED` output, 1 tile-row per Tensix core matching the work split | `(2048, 1024)`: 23 666 → 23 515 ns **1.01×** | `(3840, 1024)`: 25 645 → 25 448 ns **1.01×** |
| Same output config at the wrong shape for the grid | — | `(2048, 1024)`, shard not local to the writer: 18 946 → 11 286 ns **1.68× — helps** |
| **Boundary** | helps for any non-local output (DRAM or remote L1) | helps for remote L1 only; DRAM output is flat at the DRAM ceiling |

**Why the boards differ:**
- **DRAM ceiling.** With DRAM in and out, `(2048, 1024)` moves 8.4 MB; BH p150b's `scatter` already does that in
  ~21 µs (~400 GB/s, the BH DRAM ceiling), so cutting the writer's barriers cannot help. On WH B0 n150 `scatter`
  takes ~72 µs (~117 GB/s) against a ~190 GB/s ceiling, and `relocate` closes most of that gap (~184 GB/s).
- **Grid size.** A shape sized to one tile-row per Tensix core on 8×8 is not on 12×10; size "local" shards to the
  attached grid or the local-shard row silently becomes a remote-write row (1.68×).

- **Helps** whenever the writer is on the critical path: the scatter writer waits for 4 serialized write round
  trips per tile instead of 1, so the time scales with the barrier count, not with the bytes. It holds for a
  remote **L1** target too (1.40× WH B0, 1.50× BH p150b), so the win is not specifically about DRAM bandwidth.
- **Does not help** when the destination pages are local to the writing Tensix core: each face write is a short
  NoC loopback, 4 of them fit inside the reader's one remote read per tile, and the reader sets the pace — or,
  on BH p150b, when DRAM bandwidth rather than the writer sets the pace.

Correction to the claims above (and to the "Measured win" wording in `master.md`): the gap is caused by the
**per-face write barrier** (4 round trips per tile), not mainly by "higher achieved DRAM bandwidth" for bigger
transactions — on WH B0 n150 it persists with an L1-interleaved output (1.36–1.49×) and with tiny moves. A small
tile count is **not** a no-gain regime on WH B0 n150: 2 tiles measured 2.24×, 64 tiles (1 tile per Tensix core)
1.50×, 16 384 tiles (256 tiles per Tensix core) 1.62× (probe, DRAM → DRAM, 12 trials). Coalescing only stopped
mattering there once the writes became local; on BH p150b it also stops for a full-grid DRAM → DRAM move.

*How measured:* `test_tile_reorder_regimes`, `DEVICE KERNEL DURATION [ns]`, baseline/lever interleaved A,B,A,B,
median of 12 trials × 3 launches.
- **WH B0 n150:** branch `adamjanovic/perf-example-improvements`, 2026-10-12, at `(2048, 1024)` (one tile-row per
  Tensix core on the 8×8 grid).
- **BH p150b:** the `(3840, 1024)` cells (one tile-row per Tensix core on 12×10, the shape the regimes test now
  sizes to this grid) come from a probe of the same A/B at that shape; all `(2048, 1024)` cells come from a BH run
  of the regimes test at its earlier fixed `(2048, 1024)` shape.
  `bh-qb-15-special-adamjanovic-for-reservation-139410`, 2026-10-19, commit `f74ee7aad64` ([`report.md`](report.md)).

Reproduce: `scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_tile_reorder.py::test_tile_reorder_regimes`
(same regimes on any board; asserts only what held on every measured board, `measure` regimes only log).
