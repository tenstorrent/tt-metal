# All-reduce ring transport - synchronization and route-cost ablation

**Problem type:** data movement below the ceiling: contention  ·  **Concepts:** neighbor synchronization and NoC route contention
**First profiled on:** `bgd-lab-t3003-special-mstaletovic-for-reservation-40918` · Wormhole B0 · 1000 MHz · 2026-07-10 · `5f0ad060667`
**Measured on:** WH B0 n150 (regimes, 2026-10-12) · BH p150b (`bh-qb-15-special-adamjanovic-for-reservation-139410`, 2026-10-19)

> Reading order: [`../master.md`](../master.md) → **this file** → run the CLI, then read the code if needed.

## What this isolates

Every active core belongs to a rectangular serpentine ring. All four variants perform the same final
local identity copy for a correctness check. Before that copy:

| Variant | Diagnostic work per round |
|---|---|
| `semaphore_only` | Repeat `group_size - 1` NoC0 neighbor semaphore increments and waits; no payload moves. |
| `payload_ring` | Forward an L1 payload on NoC0 through `group_size - 1` neighbor writes using two ping-pong scratch slots. |
| `semaphore_only_noc1` | Repeat the same semaphore sequence on NoC1. |
| `payload_ring_noc1` | Forward the same payload and use the same synchronization on NoC1. |

These variants intentionally measure components rather than equivalent algorithms. Their delta is
not a speedup claim; it estimates the payload/route cost added to the synchronization skeleton.

## Device result

On the Wormhole B0 box above, for six BF16 tiles (12 KiB per Tensix core), 64 Tensix cores in total,
and 100 rounds per launch (BH p150b numbers are in *When it helps* below):

| Placement | Group | NoC0 semaphore | NoC0 payload | NoC1 semaphore | NoC1 payload | NoC1 / NoC0 |
|---|---:|---:|---:|---:|---:|---:|
| half rows | 1x4 | 215.0 ns | 4267.7 ns | 349.0 ns | 10426.1 ns | 2.44x |
| whole rows | 1x8 | 393.8 ns | 4335.5 ns | 876.3 ns | 26300.1 ns | **6.07x** |
| whole columns | 8x1 | 410.0 ns | 4487.7 ns | 986.8 ns | 27566.4 ns | **6.14x** |
| two rows | 2x8 | 1346.4 ns | 47173.5 ns | 1376.9 ns | 48551.2 ns | 1.03x |

NoC0 routes in `+X` then `+Y`, while NoC1 routes in `-Y` then `-X`. The logical ring ordering favors
NoC0 for line groups: most NoC0 edges are one physical hop, but those same edges go almost all the
way around the torus on NoC1. The `2x8` serpentine contains one row in each horizontal direction,
so changing NoCs swaps which row is expensive and barely changes the aggregate. At 48 KiB, NoC1 is
8.10x slower for rows and 8.14x for columns. The full matrix is in [`report.md`](report.md).

## tt-npe cross-check

The included `tt_npe_ring_model.py` creates one simultaneous neighbor write per active core. It
models transport only, then multiplies the hop estimate by `group_size - 1`:

| Group | NoC | 12 KiB ring cycles | Congestion impact | Device payload ring |
|---:|---:|---:|---:|---:|
| 1x8 | 0 | 3066 | 0.0% | 4335.5 ns |
| 1x8 | 1 | **20097** | **84.7%** | **26300.1 ns** |
| 8x1 | 0 | 3066 | 0.0% | 4487.7 ns |
| 8x1 | 1 | **20097** | **84.7%** | **27566.4 ns** |
| 2x8 | 0 | 43065 | 84.7% | 47173.5 ns |
| 2x8 | 1 | 43065 | 84.7% | 48551.2 ns |

The model reproduces the geometry break closely. It does not model semaphore waits, compute, CB
backpressure, or dependencies between hops, so it is evidence about NoC routes rather than a full
kernel prediction. The script queries the selected device for its architecture, worker grid, and
logical-to-physical worker mapping; tile bytes come from `ttnn.tile_size()`.

```bash
cd /path/to/tt-npe
source ENV_SETUP
cd /path/to/tt-metal
python ttnn/ttnn/operations/examples/tensix_all_reduce_ring_transport/tt_npe_ring_model.py --noc both
```

## Run

```bash
python -m ttnn.operations.examples.tensix_all_reduce_ring_transport \
  --group-shape 2,8 --num-groups 4 --num-tiles 1 6 24 \
  --kernel-iters 100 --trials 5
```

The CLI also accepts `--variant` and `--report`. Correctness is the only test failure condition.

## When it helps / when it does not
**Rule:** put the payload ring on the NoC whose routing direction matches the ring order — for a 1-D line
ring ordered `+X` (or `+Y`) that is NoC0, and it wins on both boards (WH B0 n150 6.04×, BH p150b 3.94× on
`1x8`; up to 10.59× on BH p150b's full-width `1x12` line at 48 KiB per Tensix core). A 2-row serpentine ring
is flat on both boards (1.00–1.04×): the boundary is ring geometry, not group size or payload.

Baseline `payload_ring_noc1` → lever `payload_ring` (NoC0); bf16; payload per Tensix core in the row label
(6 tiles = 12 288 B); parentheses = min–max over the retained trials. Full-grid rows place as many groups as the grid holds, so the group shape differs per board.

| Regime (group × groups, payload per Tensix core) | WH B0 n150 · 8×8 = 64 Tensix cores | BH p150b · 12×10 = 120 Tensix cores |
|---|---|---|
| **Helps:** line ring `1x8` × 8 groups, 6 tiles | 26 373 (26 324–26 481) → 4 364 (4 362–4 380) ns **6.04×** | 10 308 (10 269–10 326) → 2 613 (2 612–2 615) ns **3.94×** (64 of 120 Tensix cores) |
| **Helps:** full-width rows, 1 tile (2 KiB) | `1x8` × 8: 5 106 → 1 974 ns **2.59×** | `1x12` × 10: 4 838 → 2 623 ns **1.84×** |
| **Helps:** full-width rows, 6 tiles (12 KiB) | `1x8` × 8: 26 300 → 4 336 ns **6.07×** | `1x12` × 10: 25 498 → 4 040 ns **6.31×** |
| **Helps:** full-width rows, 24 tiles (48 KiB) | `1x8` × 8: 103 488 → 12 777 ns **8.10×** | `1x12` × 10: 96 744 → 9 134 ns **10.59×** |
| **Helps:** full-height columns, 6 tiles | `8x1` × 8: 27 566 → 4 488 ns **6.14×** | `10x1` × 12: 15 993 → 2 980 ns **5.37×** |
| **Helps:** full-height columns, 24 tiles | `8x1` × 8: 105 408 → 12 942 ns **8.14×** | `10x1` × 12: 61 213 → 7 113 ns **8.61×** |
| **Helps:** half rows, 6 tiles | `1x4` × 16: 10 426 → 4 268 ns **2.44×** | `1x6` × 20: 10 734 → 2 930 ns **3.66×** |
| **Does not help:** 2-row serpentine `2x8` × 4 groups, 6 tiles | 48 920 (48 801–49 083) → 47 490 (47 263–47 808) ns **1.03×** | 19 554 (19 537–19 603) → 19 309 (19 268–19 367) ns **1.01×** |
| **Does not help:** full-width 2-row serpentine, 6 tiles | `2x8` × 4: 48 551 → 47 174 ns **1.03×** | `2x12` × 5: 48 455 → 48 561 ns **1.00×** |
| **Boundary** | line → helps at every payload measured (≥ 2.0×, `1x4` at 1 tile); serpentine → flat (1.01–1.04×) | same: line ≥ 1.7× (`1x6` at 1 tile); serpentine flat (1.00–1.01×) |

**Why the boards differ:**
- On the `1x8` regime case the NoC1 baseline is 2.6× faster on BH p150b than on WH B0 n150 (10 308 vs
  26 373 ns) while NoC0 is only 1.7× faster (2 613 vs 4 364 ns), so the gain drops to 3.94×. On BH p150b
  the eight `1x8` groups use columns 0–7 of a 12-column grid (one group per row); this example does not
  isolate why the NoC1 detour is relatively cheaper there.
- On the full-width BH p150b line (`1x12`, 12 Tensix cores per ring) the NoC1 detour is paid on every hop of a
  longer ring and the gain grows with payload faster than on WH B0 (6.31× → 10.59× from 12 to 48 KiB).
- Small payloads gain less on BH p150b (1 tile: 1.84× vs 2.59×): the fixed semaphore skeleton
  (`semaphore_only`, ~508 ns NoC0 / ~1 556 ns NoC1 on `1x12`) is a larger share of a 2 KiB ring.

On a line, the logical ring order follows NoC0's `+X` routing, so each forward is one physical hop,
while on NoC1 (`-X`) the same neighbor write wraps almost the whole row. A `2x8` (or `2x12`) serpentine has
one row running each way, so either NoC is wrong for one of the two rows and swapping NoCs just swaps
which row is slow (≤ 3% on both boards). Choose the NoC by checking the ring direction against the NoC's
routing direction, not by group size alone; a serpentine ring needs a different remedy (reorder or split
the ring), not a NoC switch. The WH B0 regime results match the Wormhole table above (6.07x / 1.03x at 100
rounds).

*How measured:*
- WH B0 n150: `test_tensix_all_reduce_ring_transport_regimes` (kernel-iters 20, interleaved A/B, median of 11
  retained trials; three runs, ranges in [`report.md`](report.md)), 2026-10-12. Full-grid rows from the
  2026-07-10 `test_tensix_all_reduce_ring_transport_device_perf` sweep (`bgd-lab-t3003`, kernel-iters 100, N=5).
- BH p150b: the same regimes test (`1x8` and `2x8` rows) plus
  `test_tensix_all_reduce_ring_transport_device_perf` (kernel-iters 100, N=5) for the full-grid rows;
  `bh-qb-15-special-adamjanovic-for-reservation-139410`, 2026-10-19.

Reproduce (same regimes on any board; asserts only what held on every measured board, `measure` regimes only log):
```bash
scripts/run_safe_pytest.sh --run-all tests/ttnn/unit_tests/operations/examples/test_tensix_all_reduce_ring_transport.py::test_tensix_all_reduce_ring_transport_regimes
```
