# Performance examples — catalog

Short, self-contained, **runnable** ops that each isolate one or two kernel-level performance
concepts and **measure** them on device (real `ns`, never a claimed speedup). Use them to learn a
pattern, then re-measure on your own shapes with each example's CLI / test.

**Reading order:** this file → the example's `README.md` → read the code and run the test only if
you need to. For the ⭐ Starter examples, the *gist* below is often enough to act immediately.

**Difficulty:**
- **⭐ T1 Starter** — one knob/placement decision, no kernel restructure. Actionable from the gist.
- **⭐⭐ T2 Intermediate** — a CB-sizing / transfer-shape / kernel change. Read the README.
- **⭐⭐⭐ T3 Advanced** — kernel restructure, overlap scheduling, mcast / semaphores. Read the code.

Every number below is stamped in that example's `report.md` with the box + arch it was measured
on. They are illustrative of the *effect*, not CI bounds.

---

## ⭐ T1 — [`reader_placement`](reader_placement/README.md)
**Concept:** core placement → DRAM-traffic (read+write) NoC contention.
**Situation:** you spread a *line* of reader/worker cores over an interleaved-DRAM tensor (grid-
filling copies, or the reader line of an mcast) and it is slower than it should be.
**Measured win:** placing the line as a **row** or **diagonal** instead of a **column** is
**~2.2–2.8× faster** (Wormhole B0, 4–8 cores); the gap grows with core count. A column line
saturates its shared NoC links and stops scaling.
**Gist:** a column line is what `split_work_to_cores(..., row_wise=False)` (the **default**) gives
you — pass **`row_wise=True`** to spread across the DRAM-facing axis. (Diagonal only beats row on
asymmetric grids like Blackhole.)

## ⭐⭐ T2 — [`double_buffer`](double_buffer/README.md)
**Concept:** keeping bytes in flight on the NoC for a DRAM reader→compute→writer pipeline, via three
levers — outstanding reads per barrier (`block`), double-buffered CBs, and transfer size (dtype).
**Situation:** you wrote the reader the obvious way — *read one tile, barrier, push, repeat* — with
one-tile CBs. It's correct but leaves the NoC mostly idle (latency-bound).
**Measured win:** on **1 core** (bf16), trap (`block=1`, single-buffered: **6.5 GB/s**) →
`block=4` + double-buffered = **2.78× (17.9 GB/s)** (Wormhole B0). The levers **compound**: batching
alone buys ~2× but saturates ~13 GB/s (can't overlap read+write); double buffering lifts it to the
single-core NoC limit. **Transfer size** sets the bandwidth ceiling: best GB/s scales ~linearly with
tile bytes (bfp8 9.8 → bf16 17.9 → fp32 31.7), but a smaller dtype moves less data so it wins on
wall time. **No gain once DRAM-bandwidth-bound** — 64 cores hit **190.8 GB/s** (≈DRAM peak) untuned.
**Gist:** never `read-one / barrier` — issue a **block** (~4–8) of async reads then **one** barrier,
and size each CB to `2 * block` tiles (double-buffered). Small sweet spot (~4–8); bigger wastes L1.
Use the smallest dtype your accuracy allows. Skip all of it if you're already bandwidth-bound (enough
cores) or compute-bound.

## ⭐⭐ T2 — [`tile_reorder`](tile_reorder/README.md)
**Concept:** transfer coalescing on a DRAM-bandwidth-bound move.
**Situation:** a whole-tile relocation (permute / transpose-of-tiles) written the generic way —
as many small sub-tile (face) writes with a barrier each.
**Measured win:** relocating each **whole 2 KB tile in one NoC write** is at least as fast as, and
on this move faster than, writing it as 4 × 512 B faces — bigger coalesced transactions hit higher
achieved DRAM bandwidth. Reader on NoC0, writer on NoC1 to overlap.
**Gist:** on a DRAM-bound move, move whole pages and batch barriers; don't scatter sub-tile faces.

## ⭐⭐ T2 — [`tensix_all_reduce_compute`](tensix_all_reduce_compute/README.md)
**Concept:** FPU destination reuse for a multi-block tile reduction already resident in L1.
**Situation:** a reducer copies each contributor into DST, repeatedly calls
`add_binary_tile_init()`, and uses one SFPU binary add per contributor.
**Measured win:** pairwise FPU `add_tiles(..., acc_to_dest=true)` with FP32 DST is **2.70× faster**
for 2 blocks and **5.92× faster** for 8 blocks (six tiles, one Wormhole B0 core). At 16 blocks it
is **6.75× faster** (**3.46 µs** versus **23.31 µs**).
**Gist:** initialize FPU add once per DST batch, pair source blocks, accumulate directly into DST,
and pack only the final sum. Seed DST with one copy only for an odd contributor count.

## ⭐⭐ T2 — [`compute_fusion`](compute_fusion/README.md)
**Concept:** fusing a small expression through DEST vs. computing it as separate helper calls that
round-trip each intermediate through an L1 circular buffer (single core, pure compute).
**Situation:** you built `exp(sqrt(x)+y)` / `sqrt(x)*b` / `1/rowsum(x)` the readable way — one helper
per op — and wonder whether fusing it into one pass (or using a reduce post-op) is worth it.
**Measured win (Wormhole B0, 1 core):** it depends entirely on **which engine consumes the
intermediate**. When the consumer is an **SFPU** op (reads DEST natively), fusion wins:
`exp(sqrt(x)+y)` **1.03–1.12×**, reduce+reciprocal post-op **1.01–1.07×**. When the consumer is an
**FPU** op, fusing via DEST-reuse **loses** (`sqrt(x)*b` at **0.94–1.02×**; isolating just the combine
step, dest-reuse is **0.82×** — the L1 round-trip is 1.22× *faster*), because the FPU wants operands
in source registers and DEST→src costs more than the pack+unpack it replaces. Doing a plain multiply
on the SFPU instead of the FPU is a **0.58×** loss. DEST-lane block size is a ~1–3% second-order knob.
**Gist:** fuse (keep intermediates in DEST) when the next op is **SFPU** — sqrt/exp/recip and reduce
post-ops. Do **not** reach for FPU dest-reuse just to "skip L1": for a single FPU binary, pack the
intermediate to L1 and let the unpacker feed it back. Never use the SFPU for what the FPU does.
Ships a `--microbench` mode (`DeviceZoneScopedN` per phase, per TRISC) that shows the mechanism at
engine granularity: the L1 round-trip surfaces as **unpack** cost; dest-reuse surfaces as extra
**math** cost; SFPU-mul is ~22k ns more math than FPU-mul.

## ⭐⭐ T2 — [`compute_block_size`](compute_block_size/README.md)
**Concept:** compute block / loop granularity — amortizing the fixed per-helper-call overhead
(phase-boundary data-format reconfig + LLK init/uninit + unpack/math/pack pipeline fill/drain) over
more tiles per call (single core, pure compute).
**Situation:** you built a row-parallel compute chain (here `out = (A + B) @ C`: tilize A, tilize B,
add, matmul, untilize) the readable way — loop over the M rows a tile-row at a time, running the whole
chain on each — and wonder whether doing more of M per pass is worth it.
**Measured win (Wormhole B0, 1 core):** doing the whole chain in **one pass** over M is **1.65×**
faster than tile-row-by-tile-row (17.4 µs vs 28.7 µs, M=256 K=128 N=128, bf16), identical math (PCC
0.99999). The curve is monotonic with diminishing returns (1.27× → 1.51× → 1.65× as the block
doubles) — the amortize-a-fixed-cost signature; ≈1.6 µs of pure overhead per extra pass. The win
**shrinks as the per-block payload grows** (wider N=256 → 1.40×) and **grows with the phase count**
(five reconfigs here). Costs L1: intermediate CBs scale with the block.
**Gist:** don't loop a row-parallel compute chain a tile-row at a time — run each helper on the whole
row-parallel block in one call (or the largest block your L1 budget allows). Every extra pass repays
the per-phase reconfig + init + pipeline fill/drain for no extra work. Biggest payoff on many-phase
chains (tilize/eltwise/matmul/untilize) with small per-call payloads; smaller once each call already
does a lot.
**Second lever (same mechanism, other side):** the helpers reconfig data formats at every phase
boundary by default; when the format never changes (all-bf16 chain) that reconfig is wasted MMIO.
Turning it off — keep the inits, drop the format reconfig — is correct (PCC unchanged) and up to
**1.19×** faster, largest where there are the most transitions. Compounds with block size to
**1.72×** (WH B0). Only safe when the dtype is genuinely constant across the boundary. See the
example's `report_reconfig_ablation.md`.

## ⭐⭐⭐ T3 — [`tensix_all_reduce_ring_transport`](tensix_all_reduce_ring_transport/README.md)
**Concepts:** neighbor semaphore cost and direction-sensitive NoC contention in serpentine rings.
**Situation:** a reduce-and-forward ring is much slower when a rectangular group spans two rows.
**Measured result:** for a 12 KiB payload on 64 Wormhole B0 cores, NoC0 forwards 8-core lines in
**4.34–4.49 µs**, while NoC1 takes **26.30–27.57 µs** (**6.07–6.14× slower**) because the fixed
ring order opposes NoC1 routing. A `2x8` serpentine costs **47.17 µs** on NoC0 and **48.55 µs** on
NoC1 because it contains equal traffic in both horizontal directions. tt-npe predicts the same
geometry reversal: **3,066 → 20,097 cycles** for lines and **43,065 cycles** on either NoC for
`2x8`.

## ⭐⭐⭐ T3 — [`tensix_all_reduce`](tensix_all_reduce/README.md)
**Concepts:** Tensix-to-Tensix collective topology and reduction work distribution.
**Situation:** every core in each rectangular L1-sharded group contributes the same tile block,
and every member needs the elementwise group sum.
**Measured result:** with FPU destination-reuse reduction, two-phase worker reduction beats ring
push by **4.64–4.73×** on 8-core lines and **6.48×** on a 16-core `2x8` group (**8.36 µs** versus
**54.18 µs**, 9.8% noise). On 4-core groups, root reduction is fastest at **4.00 µs** because the
extra two-phase handoff is not amortized.
