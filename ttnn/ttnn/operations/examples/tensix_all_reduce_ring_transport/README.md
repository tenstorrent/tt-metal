# All-reduce ring transport - synchronization and route-cost ablation

**Difficulty:** ⭐⭐⭐ T3  ·  **Concepts:** neighbor synchronization and NoC route contention
**First profiled on:** `bgd-lab-t3003-special-mstaletovic-for-reservation-40918` · Wormhole B0 · 1000 MHz · 2026-07-10 · `5f0ad060667`

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

For six BF16 tiles (12 KiB per core), 64 total cores, and 100 rounds per launch:

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
