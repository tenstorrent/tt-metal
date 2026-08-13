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
| 16 | 5679.0 | — | **INCORRECT — see below** |

Crossover is between **2 and 4** tiles/core. Root scales roughly linearly with payload
(990 → 1317 → 1927 → 3189 → 5679); reduce-scatter scales much flatter (1329 → 1408 → 1549 → 2299),
which is the parallelize-across-tile-indices effect earning its extra handoff.

## DEFECT: `reduce_scatter_mcast` is numerically wrong at 16 tiles/core

At `--num-tiles 16` on a `1x8` group, `reduce_scatter_mcast` **fails the example's correctness
assertion** (`torch.testing.assert_close(rtol=0.04, atol=0.125)` — "Tensor-likes are not close").
Isolated per-variant at the same size:

- `reduce_root_mcast` — PASS, 5679.0 ns, 0.1% noise
- `reduce_scatter_mcast` — **FAIL**

So it is variant-specific, not a harness or capacity limit.

Why it went unnoticed: **every published measurement of this example uses 1 or 6 tiles/core.**
Wormhole ran 6, Blackhole ran 6 and 1. The variant was never exercised above 6, so the defect sits
entirely in unmeasured territory. 8 passes; 16 does not. The exact boundary between 8 and 16 is
not established here.

Suspected but NOT verified: worker assignment is `min(num_tiles, group_size - 1)` workers over
`num_tiles` tile indices, which does not divide evenly at 16 tiles / 7 workers. Could equally be a
DEST batching issue — the README notes the live output batch comes from the JIT-derived
`DEST_AUTO_LIMIT`. Needs a proper diagnosis before anyone adopts the pattern above 8 tiles.

## Consequence for rms_norm R5

`reduce_scatter_mcast` is the only reducer that beats flat root on a 1-D group, and its **safe,
measured region on this box is 4–8 tiles/core**. rms_norm's `MAX_GATHER_TILES = 64` caps
`R <= 64/G = 8` for the `BLOCK_SHARDED` `G=8` case — which happens to sit exactly at the top of the
proven-correct range. Raising that cap to reach `R=16` would have walked straight into the defect
above.
