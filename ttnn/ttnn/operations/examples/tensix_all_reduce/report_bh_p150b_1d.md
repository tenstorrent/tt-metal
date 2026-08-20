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

## RESOLVED defect: ragged CB quantum corrupted the gather above 8 tiles/core

**This section originally reported an open, undiagnosed defect. It is diagnosed and fixed; the text
below is kept because the failure mode generalizes to any ragged work split feeding a CB.**

The symptom was real. On the pre-fix code (`a3b4ce20bc9`) at `--num-tiles 16` on a `1x8` group,
the reducer fails the example's correctness assertion hard — not marginally:

```
Mismatched elements: 655360 / 1048576 (62.5%)
Greatest absolute difference: 28.90625 (up to 0.125 allowed)
```

8 tiles/core passed (2304.5 ns), 16 failed. That boundary is not a hardware limit and not a DEST
batching issue — it is the point at which the per-worker tile split first goes **ragged**.

**Root cause.** Worker `i` owns tiles `i, i+W, i+2W, …`, so its share is either `q` or `q+1` tiles.
The pre-fix kernel used each worker's *own* ragged `assigned` count as the CB push/pop quantum and
as the gather stride, while both CBs were allocated at the **uniform** `max_assigned`. A CB's
capacity must be an exact multiple of its push/pop quantum, so a short worker's ragged quantum
wraps illegally and overwrites another contributor's slot in the gather buffer. At 8 tiles over
7 workers the split is `2,1,1,1,1,1,1` and the wrap happened to be benign; at 16 over 7 it is
`3,3,2,2,2,2,2` and it corrupts most of the buffer — hence 62.5%, not a few LSBs.

**Fix** (in place since `5ac64b6d1e2`, the same commit that first published this report — which is
why the report's own "suspected but NOT verified" note is stale): every worker pushes and pops the
uniform `max_assigned`, and the gather stride is `contributor * max_assigned`, even when a worker's
real share is smaller. The pad slots are neither read nor written back, so they never reach the
output. The worker count also became `min(num_tiles, group_size)` — the root now takes a share
instead of being reserved — but that was a *utilization* change, not the correctness fix: reverting
only the worker count while keeping the uniform quantum still passes at 16 tiles.

**Verified fixed** on this box, current code:

| Config | Result |
|---|---|
| `1x8` × 8 groups, 12 / 16 / 20 / 24 / 32 tiles/core | PASS (12 and 20 are ragged over 8 workers; 16 / 24 / 32 divide evenly) |
| grid-derived sweep at 16 tiles/core — `1x11`, `10x1`, `1x5`, `2x11`, `10x2`, `5x5` × all 7 variants | PASS |

The 16-over-11, 16-over-5 and 16-over-25 splits in that sweep are all ragged, so the fix is
general, not tuned to the `1x8` case.

## Consequence for rms_norm R5

`reduce_scatter_mcast` is the only reducer that beats flat root on a 1-D group, and with the
ragged-quantum fix its **measured-correct region on this box is 4–24 tiles/core** (it loses below 4
and stops winning by 32). rms_norm's `MAX_GATHER_TILES = 64` caps `R <= 64/G = 8` for the
`BLOCK_SHARDED` `G=8` case. Raising that cap to reach `R=16` is now safe *and* worth it — 16
tiles/core is where the pattern is fastest relative to root (**1.61x**, its best ratio at any size
measured). Past 24 tiles/core there is nothing left to win.
