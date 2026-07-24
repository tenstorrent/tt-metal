# dram_saturation — measured report

| stamp | value |
|---|---|
| box | `bgd-lab-16-special-dstoiljkovic-for-reservation-44175` |
| arch | Wormhole B0 (8×8 = 64 compute grid) |
| commit | `e1dc0e60d9a` |
| date | 2026-07-24 |
| metric | `DEVICE KERNEL DURATION [ns]`, read **in-process** (`ttnn.ReadDeviceProfiler` + `ttnn.get_latest_programs_perf_data`); achieved bandwidth = `2 × tensor_bytes / ns` (read + write) |
| method | 5 warmup + 20 timed launches per case, flush-bracketed; whole `variant × core-count` matrix in one device session |

> Numbers are illustrative of the *effect*, not a CI bound — single-box, single-arch.
> Re-run `python -m ttnn.operations.examples.dram_saturation` to measure your own sweep.
> A different arch (e.g. Blackhole) should be **appended** as a new block, not overwritten.

A pure DRAM→DRAM copy (reader on NoC0, writer on NoC1, **no compute**) of a fixed 2048×2048 bf16
tensor (4096 tiles, 8.4 MB; 16.8 MB moved read+write), swept over core count for two placements.
The kernels are byte-identical across every point — only the core count and the placement geometry
change — so the delta is purely work distribution + NoC contention.

## bfloat16, 2048×2048 interleaved DRAM, iters=1, trials=20

| variant | cores | ns/op | GB/s | GB/s/core |
|---|--:|--:|--:|--:|
| spread | 1 | 771739.1 | 21.7 | 21.7 |
| spread | 2 | 387236.3 | 43.3 | 21.7 |
| spread | 4 | 195716.4 | 85.7 | 21.4 |
| spread | 8 | 114706.4 | 146.3 | 18.3 |
| spread | 16 | 87895.0 | 190.9 | 11.9 |
| spread | 32 | 87139.9 | 192.5 | 6.0 |
| spread | 48 | 86081.6 | 194.9 | 4.1 |
| spread | 64 | 86556.1 | 193.8 | 3.0 |
| stacked | 1 | 771642.9 | 21.7 | 21.7 |
| stacked | 2 | 388728.1 | 43.2 | 21.6 |
| stacked | 4 | 258930.5 | 64.8 | 16.2 |
| stacked | 8 | 233727.9 | 71.8 | 9.0 |
| stacked | 16 | 128588.4 | 130.5 | 8.2 |
| stacked | 32 | 93433.4 | 179.6 | 5.6 |
| stacked | 48 | 87463.9 | 191.8 | 4.0 |
| stacked | 64 | 87431.3 | 191.9 | 3.0 |

### Reading it
- **The knee is the sweet spot.** `spread` scales linearly (~21.7 GB/s/core) to 4 cores, bends at
  8 (146.3), and is essentially saturated by **16 cores (190.9 GB/s)**. 32/48/64 cores land at
  192–195 GB/s — within noise of 16. The achieved ceiling for this pattern is ~195 GB/s; **16 cores
  reach 98% of it.**
- **Past the knee, cores are wasted.** 16 → 64 cores is **4× the cores for ~1.5% more bandwidth**;
  `GB/s/core` falls 21.7 → 3.0. On a real op those 48 cores would do more good on other work.
- **Placement moves the knee.** `stacked` piles the line onto one column's NoC links: it delivers
  only **71.8 GB/s at 8 cores** (spread: 146.3) and needs **~48 cores** to reach the ~192 GB/s that
  `spread` reaches at **16**. A well-placed 16-core copy beats a stacked 32-core one (190.9 vs 179.6).
- **No hard rollover on this shape.** `stacked` saturates late but does not get strictly *slower* as
  cores grow here — the over-subscription penalty is wasted cores, not a slowdown. A more contended
  transfer pattern can push the same mechanism into an actual rollover.

## The exploit — cap at the knee, free the rest

The test derives the sweet spot automatically (`sweet_spot_cores`: the smallest core count within 3% of
peak on the `spread` curve) and quantifies the win:

```
--- EXPLOIT: cap a DRAM-bound op at the bandwidth knee ---
sweet spot (spread, within 3% of peak):  16 cores @ 191.9 GB/s
full grid:                               64 cores @ 192.7 GB/s
=> same bandwidth at 4.0x fewer cores: 48 cores freed at +0.4% perf cost.
```

The exploit: a DRAM-bandwidth-bound op **saturates at ~16 well-placed cores**, so pinning it there
delivers full bandwidth on **1/4 of the grid** and leaves **48 cores free** — at ~0% cost to the op.
In a real kernel/model those freed cores are the resource: run other work on them, or cut power/heat.
`sweet_spot_cores()` is reusable on any measured `{cores: GB/s}` sweep.

**Takeaway:** for a DRAM-bandwidth-bound op, the fastest *and cheapest* configuration is the **minimum
well-placed cores that reach the bandwidth plateau** (~16 spread cores here), not the full grid. Adding
cores past the knee buys nothing and just spends a resource you could exploit; getting the placement
wrong (`stacked`) only pushes the knee to a higher, more wasteful core count.
