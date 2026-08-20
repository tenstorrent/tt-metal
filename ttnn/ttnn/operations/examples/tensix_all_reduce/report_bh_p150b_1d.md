# Tensix rectangular all-reduce — Blackhole 1-D groups

box=`bh-49-special-mstaletovic-for-reservation-60064` · arch=BLACKHOLE · N=5 (median) · kernel-iters=10

Fills the gap between the two campaigns already in `README.md`: the Wormhole B0 pass (2026-07-10)
measured 1-D groups but no Blackhole; the Blackhole pass (2026-07-20, commit `c4d9dfbf075`) was
scoped to validating `tree_reduce_mcast`, which is meaningless on a 1-D group, so it measured
only `2x8`/`8x2`/`4x4`. Blackhole × 1-D was never covered.

Motivation: rms_norm `BLOCK_SHARDED` makes one grid ROW of the shard rectangle one reduction group,
so its combine groups are 1-D lines of 8 — exactly this topology.

## Full variant comparison at 8 tiles/core

| Placement | Group | Groups | Cores | Method | Median ns | Std / median | vs root |
|---|---:|---:|---:|---|---:|---:|---:|
| custom | 1x8 | 8 | 64 | reduce_root_mcast | 3189.6 | 0.1% | 1.00x |
| custom | 1x8 | 8 | 64 | **reduce_scatter_mcast** | **2299.8** | 0.2% | **1.39x** |
| custom | 1x8 | 8 | 64 | tree_reduce_mcast | 3188.4 | 0.1% | 1.00x |

`tree_reduce_mcast` lands within **0.04%** of `reduce_root_mcast`, empirically confirming the
README's claim that tree reduce collapses to the gather-to-root path on a 1-D group.

## Payload sweep — where reduce-scatter starts paying

| tiles/core | reduce_root_mcast | reduce_scatter_mcast | reduce-scatter vs root |
|---:|---:|---:|---:|
| 1 | 990.8 | 1329.9 | **0.75x** (loses) |
| 2 | 1317.4 | 1408.3 | **0.94x** (loses) |
| 4 | 1927.7 | 1549.9 | **1.24x** |
| 8 | 3189.6 | 2299.8 | **1.39x** |
| 16 | 5681.9 | 3527.3 | **1.61x** |

Crossover is between **2 and 4** tiles/core. Root scales roughly linearly with payload
(990 → 1317 → 1927 → 3189 → 5682); reduce-scatter scales much flatter (1329 → 1408 → 1549 → 2299
→ 3527), which is the parallelize-across-tile-indices effect earning its extra handoff.

Extending past the table (same box, `kernel-iters=10`, N=3): at **24** tiles/core reduce-scatter is
6949.3 ns vs root 8190.1 ns (**1.18x**), and by **32** tiles/core the advantage is gone — all three
reducers land within 0.03% of each other (10689 / 10691 / 10688 ns), i.e. the group is bandwidth-
bound and the work-distribution choice no longer matters.

## The split does not have to divide the group

`reduce_scatter_mcast` gives worker `i` the tile indices `i, i+W, i+2W, …` for `W` workers, so
whenever `num_tiles` is not a multiple of `W` some workers own one tile more than the rest. That
**ragged** case is the normal one — it is what an arbitrary `num_tiles` produces against a fixed
group — so the variant is built to take it rather than being restricted to even divisions.

Two properties carry it, and both are worth copying into any op that distributes a ragged split
over a CB:

- **The CB quantum is uniform, not per-worker.** The gather and partial CBs are sized at
  `max_assigned = ceil(num_tiles / W)`, and every worker pushes and pops that uniform amount with
  gather stride `contributor * max_assigned`, whatever its own share happens to be. A CB's capacity
  must be an exact multiple of its push/pop quantum, so a per-worker quantum would wrap at a
  different offset on every core and contributors would land on each other's slots. Holding the
  quantum uniform keeps each contributor's slot at a fixed address. The pad slots a short worker
  leaves are never read and never written back — they cost a little L1 and nothing else.
- **Every core is a worker, the root included.** `W = min(num_tiles, group_size)`: the root takes a
  share like everyone else and only then multicasts, writing its own partials to itself over NoC
  loopback so the sender's `wait_min` still counts one increment per worker per iteration.
  Reserving the root instead would idle `1/G` of the group through the whole gather/reduce phase,
  *and* leave `W` coprime with power-of-two tile counts — which is the more ragged split, not the
  less.

### Measured — ragged splits sit on the same curve as even ones

`1x8` × 8 groups, `kernel-iters=10`, this box. `W = 8`, so the split is ragged whenever
`num_tiles % 8 != 0`:

| tiles/core | split over the 8 workers | shape | median ns | N |
|---:|---|---|---:|---:|
| 12 | `2,2,2,2,1,1,1,1` | ragged | 2778.9 | 2 |
| 16 | `2` each | even | 3527.3 | 5 |
| 20 | `3,3,3,3,2,2,2,2` | ragged | 5238.1 | 2 |
| 24 | `3` each | even | 6949.3 | 3 |
| 32 | `4` each | even | 10689.2 | 2 |

The ragged points carry no measurable penalty: the straight line through the two *even* neighbours
16 and 24 predicts 5238.3 ns at 20 tiles, and the measured ragged value is 5238.1 ns. Read that as
"inside the run-to-run spread" (1–3% here) rather than as a precise coincidence — the point is that
a ragged split does not introduce a step in the curve.

Ragged group *widths* behave the same way. The grid-derived sweep at 16 tiles/core covers
`1x11` (16 over 11 workers → `2,2,2,2,2,1,1,1,1,1,1`), `10x1` (16 over 10) and `1x5` (16 over 5 →
`4,3,3,3,3`), alongside `2x11`, `10x2` and `5x5` where `W` caps at `num_tiles = 16` and the split is
even with the surplus cores idle. All six placements × all seven variants pass.

The 16-tile cell in the payload sweep above was left unmeasured in the first revision of this
report, pending this mechanism being pinned down; see `5ac64b6d1e2` and `01d578082e8` for the
history.

## Consequence for rms_norm R5

`reduce_scatter_mcast` is the only reducer that beats flat root on a 1-D group, and its **measured
winning region on this box is 4–24 tiles/core** — it loses below 4, where there are too few tiles to
spread, and by 32 the group is bandwidth-bound and all three reducers converge. rms_norm's
`MAX_GATHER_TILES = 64` caps `R <= 64/G = 8` for the `BLOCK_SHARDED` `G=8` case. Raising that cap to
reach `R=16` is worth doing: 16 tiles/core is where the pattern is fastest relative to root
(**1.61x**, its best ratio at any size measured), and `R` does not need to divide `G` to get there.
Past 24 tiles/core there is nothing left to win.
