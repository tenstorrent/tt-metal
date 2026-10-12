# tile_reorder

**Trick:** on a DRAM-bound move, coalesce into whole-page transfers and batch barriers — relocate whole tiles instead of writing sub-tile faces. Read on NoC0, write on NoC1.

**Op:** `tile_reorder(t, method="relocate" | "scatter")` — reverses column-tile order. `relocate` = one whole-page write/tile; `scatter` = 4 faces + barrier each (baseline).

**Run:** `scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_tile_reorder.py` (correctness + `test_tile_reorder_regimes`)

## When it helps / when it does not
Board: **Wormhole B0 n150** (8×8 Tensix grid, full grid used). Numbers come from `test_tile_reorder_regimes`:
`shape=(2048, 1024)` = 2048 bf16 tiles (2 KB each) = 32 tiles per Tensix core (one tile-row each), input DRAM
interleaved; only the **output** placement changes. Baseline = `scatter` (4 × 512 B writes + 4 write barriers
per tile), lever = `relocate` (1 × 2 KB write + 1 barrier per tile). Interleaved A,B,A,B, median of 12 trials × 3
launches, spread = min–max over trials.

| condition | output | baseline ns (scatter) | lever ns (relocate) | speedup | trials | board |
|---|---|---|---|---|---|---|
| **helps:** each write round trip goes to a remote DRAM bank | DRAM interleaved | 71 855 (71 006–72 857) | 45 513 (44 812–47 369) | **1.58×** | 12 | WH B0 n150 |
| **helps (not DRAM-bound):** remote L1 banks | L1 interleaved (`TensorMemoryLayout::INTERLEAVED`) | 43 773 (43 632–43 918) | 31 317 (30 664–32 152) | **1.40×** | 12 | WH B0 n150 |
| **does not help:** every output page is in the writer's own L1 | L1 `HEIGHT_SHARDED`, one tile-row per Tensix core, matching the work split | 23 666 (23 464–23 991) | 23 515 (23 357–23 755) | **1.01×** | 12 | WH B0 n150 |

- **Helps** whenever the writer is on the critical path: the scatter writer waits for 4 serialized write round
  trips per tile instead of 1, so the time scales with the barrier count, not with the bytes. This holds for a
  remote **L1** target too (1.40×), so the win is not specifically about DRAM bandwidth.
- **Does not help** when the destination pages are local to the writing Tensix core: each face write is a short
  NoC loopback, 4 of them fit inside the reader's one remote read per tile, and the reader sets the pace.

Correction to the claims above (and to the "Measured win" wording in `master.md`): the gap is caused by the
**per-face write barrier** (4 round trips per tile), not mainly by "higher achieved DRAM bandwidth" for bigger
transactions — it persists with an L1-interleaved output (1.36–1.49×) and with tiny moves. A small tile count is
**not** a no-gain regime: 2 tiles measured 2.24×, 64 tiles (1 tile/core) 1.50×, 16 384 tiles (256 tiles/core)
1.62× (probe, DRAM → DRAM, 12 trials). Coalescing only stopped mattering once the writes became local.
