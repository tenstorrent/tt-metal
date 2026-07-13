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

## ⭐⭐ T2 — [`row_reduce_accumulate`](row_reduce_accumulate/README.md)
**Concept:** how to sum a **row of `W` tiles** for a mean (`REDUCE_ROW`) — fold the cross-tile sum into the
reduce, or do it separately (FPU `add_tiles` into DEST, or the packer's L1 accumulator) and finalize the
within-tile collapse on the **FPU reduce library or the SFPU** (`sfpu_reduce` in DEST) — measured on two
precision axes: **input dtype × accumulation dtype**, over three input distributions (single core, pure compute).
**Situation:** you wrote a row-mean as one reduce over the whole strip and it scales badly as the row widens,
because the reduce pays its per-tile datapath cost `W` times.
**Measured win (WH B0, 1 core, sweep 1→32 tiles = 32→1024 elements):** at narrow rows (1–2 tiles) the single
reduce (`reduce_fold`) is *fastest*; from **W≥4** the cheapest path is **pairwise `add_tiles(acc_to_dest)`
then one finalize reduce** (`dest_accum_pairs`) — **2.91×** at 32 tiles (bf16 input); `dest_accum` **1.84×**;
packer L1-accumulate (`l1_accum`) only ties the baseline (1.03×) at 32 tiles. fp32 input ≈ halves the win
(pairs 1.86×) since the add path unpacks 2× the bytes; `reduce_fold`'s cost is input-dtype-insensitive.
**Accuracy (error vs fp64 mean, swept over input distributions signal/uniform/positive):** the two precision
axes behave oppositely — bf16 **input** error *averages DOWN* with width (a wide mean washes out input
quantization: `reduce_fold` bf16-fp32 0.17→0.04 ULP, all methods stay sub-ULP; fp32 accumulation is ~exact),
while bf16 **accumulation** error *grows UP* with width. In bf16 accumulation on all-positive/`signal` data
`reduce_fold` is worst (**13.3 ULP** @ W=32 — the full running sum lives in one bf16 DEST), `dest_accum` 2.4,
`dest_accum_pairs` 1.4 (fewer rounding steps), `l1_accum` best at **0.24 ULP** (packer L1-acc is
**fp32-DEST-only**, so its finalize reduce stays fp32). On zero-mean `uniform` data every method keeps max-abs
tiny (~1e-3) — a near-zero mean has little to lose — so max-abs (not ULP, which is inflated near zero) is the
honest metric and the method choice barely matters there.
**SFPU vs FPU finalize:** doing the within-tile collapse on the SFPU in DEST (`sfpu_reduce` + a scalar-mul for
1/N) instead of the FPU reduce library reads DEST natively and skips the pack→L1→unpack round-trip, but is
**not faster** (the SFPU vector reduce costs more than the FPU matmul-reduce, just outweighing the saved
round-trip) — it buys **bf16 accuracy** instead (it collapses the columns in fp32 internally): `dest_accum_pairs_sfpu`
is ~2.85× and the most accurate bf16 DEST-add option.
**Odd tile count:** don't reach for a phantom zero CB to give the unpaired tile a partner — resolve parity at
the SEED (`copy_tile` one tile when odd, `add_tiles` the first pair when even) so the remainder is always even
and the pair loop needs no zero CB (fewer L1 CBs, no dataflow zero-fill, ~1–2% faster at odd widths, `W==1`
free). `copy_tile` is unary; only strict 1-tile-per-add needs the binary zero operand.
**Gist:** for a mean over a wide row of tiles, don't fold it into one reduce — accumulate the tiles first
(`add_tiles(acc_to_dest)`, two tiles per add, parity resolved at the seed) and reduce **once** at the
end (fastest AND the more accurate of the DEST-add methods). Keep the single reduce only for narrow rows
(≤2 tiles). bf16 *input* is nearly free for a wide mean; bf16 *accumulation* is what costs precision — use
fp32 DEST if it matters (packer L1-accumulate forces fp32 DEST regardless), or the SFPU finalize for a bit
more bf16 accuracy at equal speed.

## ⭐⭐ T2 — [`reduce_accumulate`](reduce_accumulate/README.md)
**Concept:** build a SUM/mean reduce as cross-tile **FPU `add_tiles` accumulate + within-tile SFPU
`sfpu_reduce` finalize** (SFPU reads DEST in place, no L1 round-trip), across all three reduce dims, vs the
standard reduce library (FPU matmul-with-ones) — with a dispatch that picks per (dim, width). Single core,
pure compute.
**Situation:** you reduce N tiles with the reduce library and wonder whether accumulating first + finalizing
on the SFPU is faster / more accurate, and whether it generalizes past width reductions.
**Measured win (WH B0, 1 core, N tiles reduced):** the fast path wins once there are enough tiles, and the
**crossover is dim-dependent** because the FPU REDUCE_COL datapath is cheaper than REDUCE_ROW: **row wins from
4t → 2.87× @32t; scalar from ~8t → 2.94×; col from ~8t → only 1.71×** (col benefits least). Below the
crossover the single matmul-reduce is faster, so `dispatch` (row≥4, col≥8, scalar≥8) falls back and is **never
slower than the library**. Accuracy: **equal in fp32, better in bf16** (the SFPU collapses columns in fp32
before one rounding — row/col bf16 ~3–5.5× lower error @32t); for **scalar the fast path is ~100× more
accurate even in fp32** — it multiplies by 1/N once vs the library's AVG-scalar applying a 1/√N scaler twice.
**Gist:** for a *wide* SUM/mean reduce, accumulate the tiles (pairwise `add_tiles`, copy-seed the odd one)
then finalize on the SFPU (`sfpu_reduce` + a `mul_unary_tile` 1/N) — it generalizes to row/col/scalar and is
markedly more accurate for scalar. But it's a **dispatched fast path, not a replacement**: it loses below the
(dim-dependent) crossover, benefits least on col, and the win is compute-only/single-core (most real reductions
are data-movement-bound, where it won't show).

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
