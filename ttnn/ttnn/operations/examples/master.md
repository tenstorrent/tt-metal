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

## ⭐⭐ T2 — [`noc_placement`](noc_placement/README.md)
**Concept:** two knobs for interleaved-DRAM NoC contention — core **placement** (column/row/diagonal)
and **NoC selection** (which NoC a read/write stream uses) — as a switchable placement × NoC × op matrix.
**Situation:** you spread a *line* of cores over an interleaved-DRAM tensor (grid-filling copies, or the
reader line of an mcast) and it is slower than it should be; and/or you are unsure whether reads/writes
belong on NoC0 vs NoC1.
**Measured win:** placing the line as a **row**/**diagonal** instead of a **column** is **~2.9× faster**
(WH B0, 8 cores); and **reads on NoC0 / writes on NoC1** (the default) is **2.5–4.8× faster** than the
reverse for those spread placements. Reads and writes are mirror images (read·NoC0 ≈ write·NoC1).
**Gist:** a column line is what `split_work_to_cores(..., row_wise=False)` (the **default**) gives you —
pass **`row_wise=True`** to spread across the DRAM-facing axis; keep reads on NoC0 / writes on NoC1
(the `ReaderConfigDescriptor`/`WriterConfigDescriptor` default). NoC0's east→south routing disperses
column-localized DRAM traffic; NoC1's north→west concentrates it. The `noc_placement_matrix.html`
report is reconstructed from code + tt-npe (`--report`). (Diagonal only beats row on asymmetric grids like Blackhole.)

## ⭐⭐ T2 — [`width_split`](width_split/README.md)
**Concept:** work distribution — splitting a **wide, short** tensor along its **width** (tile-columns)
to fill the grid, instead of along its height (tile-rows), which strands a one-tile-row-tall tensor
on a single core.
**Situation:** a wide, short tensor (few tile-rows, many tile-columns; extreme: one tile-row tall
like `[32, 8192]`) runs at single-core speed — the natural tile-row split has only `nt_h` rows to
hand out, so for `nt_h=1` one core does everything and the other 63 sit idle.
**Measured win:** width-splitting the `Wt` tiles across `min(Wt, grid)` cores is **up to ~7.8× faster**
than the single-core (tile-row-split) baseline (WH B0, 64-core grid, bf16, relu): 2.24× at Wt=8,
4.25× at Wt=32, 6.25× at Wt=64 (grid saturation), 7.76× at Wt=256. The only regime with no benefit is
`Wt=1` (a single tile-column — nothing to split).
**Gist:** when `Wt > nt_h` (especially `nt_h` small), do **not** split work by tile-rows — assign each
core a contiguous tile-**column** range, capped by a `WT_CHUNK` constant so per-core L1 stays bounded.
Same reader/compute/writer kernels; only the per-core `(start_page, num_pages)` and the active-core
count change.

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

## ⭐⭐ T2 — [`split_reader`](split_reader/README.md)
**Concept:** when a *single* data-movement RISC-V is the bottleneck — saturated **issuing** NoC read
transactions — split those independent reads across both data-movement RISC-Vs (NCRISC + BRISC) so
the issue work runs in parallel.
**Situation:** you have measured that one reader RISC-V is issue-bound: it is on the critical path
and its time goes to issuing NoC commands. The reads divide into independent ranges and the other
data-movement RISC-V has spare capacity.
**Does NOT solve:** this is *not* a general "make reads faster" trick — it does nothing unless a
data-movement RISC-V is itself the bottleneck. First confirm you are RISC-V-issue-bound.
**Gist:** split the disjoint reads across NCRISC and BRISC, preserving the second RISC-V's existing
role. The example shrinks the NoC transaction size only to *create* that bottleneck and expose the
effect (up to ~1.7×, Wormhole B0; see [`report.md`](split_reader/report.md)) — transaction size is
the knob, not the point.

## ⭐⭐ T2 — [`matmul_output_subblock`](matmul_output_subblock/README.md)
**Concept:** matmul output-subblock shape → SRC-register operand reuse (via the `matmul_block` helper).
**Situation:** you wrote a tiled matmul that produces **one output tile per block-matmul** (a `1×1` subblock), so every output tile re-loads both its A and B operand into the SRC registers; you wonder whether a bigger output subblock is worth it.
**Measured win:** grouping output tiles into a bigger subblock is **~1.46×** (Blackhole, 1 core, sharded L1, M=N=16, Kt=1). A **wide** subblock (`1×8`) loads one A-tile into SRC once and reuses it across 8 B-tiles; a **tall** one (`8×1`) reuses one B-tile across 8 A-rows. The win tracks subblock **size**, not shape — every 8-tile subblock (`1×8`,`8×1`,`2×4`,`4×2`) lands at 1.46×, wide (reuse A) and tall (reuse B) symmetric; the 4-tile `2×2` gets 1.40× (less amortization). Ceiling is the DEST budget (8 fp16 tiles).
**Gist:** make the output subblock as large as the DEST budget allows — it amortizes the SRC operand load across the block. Caveat: a real multi-K matmul (`Kt>1`) needs a 32-bit DEST (`fp32_dest_acc_en`), halving the budget to 4 tiles, so cap the subblock at 4 (e.g. `2×2`/`1×4`) for ~1.40×. Matters most for **short-K** matmuls (small contraction depth) where the contraction can't hide the per-tile operand-load overhead.

## ⭐⭐ T2 — [`tensix_all_reduce_compute`](tensix_all_reduce_compute/README.md)
**Concept:** FPU destination reuse for a multi-block tile reduction already resident in L1.
**Situation:** a reducer copies each contributor into DST, repeatedly calls
`add_binary_tile_init()`, and uses one SFPU binary add per contributor.
**Measured win:** pairwise FPU `add_tiles(..., acc_to_dest=true)` with FP32 DST is **2.70× faster**
for 2 blocks and **5.92× faster** for 8 blocks (six tiles, one Wormhole B0 core). At 16 blocks it
is **6.75× faster** (**3.46 µs** versus **23.31 µs**).
**Gist:** initialize FPU add once per DST batch, pair source blocks, accumulate directly into DST,
and pack only the final sum. Seed DST with one copy only for an odd contributor count.

## ⭐⭐ T2 — [`eltwise_l1_vs_dest_accumulate`](eltwise_l1_vs_dest_accumulate/README.md)
**Concept:** the accumulate mechanism in a reduction loop, ranked by how much L1 traffic the running accumulator pays — L1↔DEST round-trip vs. packer L1-accumulation vs. DEST-resident accumulation.
**Situation:** you build a running accumulator by summing a stream of tiles into it. Addition can happen at three distinct points in the pipeline — the **FPU adder** (`add_tiles(A, B)` → DEST, any FPU binary op), the **DEST accumulator** (`acc_to_dest` folds each FPU result into a held DEST tile), and the **L1 accumulator** (`pack_reconfig_l1_acc` folds DEST onto the resident L1 tile at pack). What you accumulate is incidental. The dominant cost is re-touching `acc` in L1 every step; the naive read-modify-write uses only the FPU adder, so `acc` round-trips L1 (unpack, add, pack) every step.
**Measured win:** three mechanisms, each stripping more accumulator L1 traffic (64 fp32 single-tile steps, one Blackhole core, sharded L1). Baseline `rmw` (**975 µs**) round-trips `acc` through unpack+add+pack every tile. `pack_l1_acc` (**192 µs, 5.09×**) reads two tiles per step, sums them in DEST, and lets the **packer** fold that onto `acc` in place — `acc` is only packed, never unpacked (B/2 steps). `dest_acc` (**92 µs, 10.59×**) keeps the running sum in a sticky DEST tile and touches L1 once, at the end — the upper bound.
**Gist:** the win tracks accumulator L1 traffic. Don't re-read `acc` every step: either **pack-L1-accumulate** onto it (`OutputLifecycle::L1AccumulationCallerManaged` — never unpacks `acc`), or if DEST is free keep the running sum **in DEST** (`BinaryFpu<…, DestAccumulation::Enabled>` → `OutputLifecycle::DestAccumulation`, one pack at the end). `dest_acc` is the upper bound — it camps DEST for the whole reduction, so it only applies when DEST isn't needed for per-step work; otherwise `pack_l1_acc` is the realistic win with `acc` resident in L1.
**Syntax — the part that is not guessable.** `DestAccumulation` lives on `BinaryFpu`, which takes TWO CB
inputs, so accumulating ONE stream looks impossible and the tempting fix is to pair every tile against a
zero tile. Don't: that zero tile has to be filled, and the fill is never free. Instead point BOTH operands
at the SAME CB and offset the second by half the run — the operands are the two **HALVES** of the stream
(`x[i] + x[i + N/2]`), not adjacent pairs — so N tiles fold in N/2 FPU steps with no identity operand:
```cpp
constexpr uint32_t N = B / 2;                       // B tiles to sum; the SECOND operand starts at N
ckl::eltwise_chain<ckl::SetupOwner::Caller>(
    ckl::EltwiseShape::tiles(N),
    ckl::BinaryFpu<cb_in, cb_in, Add, None, ..., D0,
                   OperandKind::Block, OperandKind::Block,
                   ckl::TileOffset::Unset, ckl::TileOffset::Set,
                   ckl::DestAccumulation::Enabled>{0, N},        // <- {A base, B base} IS the trick
    ckl::PackTile<cb_out, ckl::OutputLifecycle::DestAccumulation, ..., D0>{});
```
`tiles(...)` is one contiguous shape, so the accumulation scope is `WholeShape`/`Enabled` (`PerRow` is
rejected there); a 2D walk uses `grid(H, W)` with `TileOffset::Strided` + a `StridedTileRange{base,
row_stride}` per operand. **Odd N does not tile into halves.** For the specific case of summing tiles that
are ALREADY reduced (per-core partials from an earlier `REDUCE_ROW`), `reduce_helpers_compute.hpp` does it
for you — `ReduceAlgorithm::AccumulateViaAdd` + `ReduceWithinTile::Skip`.

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

## ⭐⭐ T2 — [`reduce_block`](reduce_block/README.md)
**Concept:** the accumulate + SFPU-finalize reduce, applied **per output tile** over a full 2-D `(Ht, Wt, NC)`
block (many output tiles), vs the matmul-reduce library — with a per-dim, per-width dispatch. Single core, pure
compute.
**Situation:** the fast reduce (accumulate the tiles, finalize once on the SFPU) is easy when a strip collapses
to *one* output tile; a real reduce runs over a 2-D block and emits *many* output tiles, and you wonder if the
fast path still wins there — and whether it clears the library's `REDUCE_COL` DST/chunk limit.
**Measured win (BH, 1 core, bf16 in / fp32 acc, reducing R tiles into one output):** the fast path becomes a
**loop over output tiles** (one DEST each → no chunk limit) and each output costs the same as the single-strip
reduce, so a block ≈ `out_tiles × single-output`. It wins past a dim-dependent crossover: **row from ~4 reduced
tiles → 5.35× @32t; scalar → 4.88× @32t; col → 3.37× @32t** (col wins least — the FPU `REDUCE_COL` datapath is
already cheap). Below the crossover the single matmul-reduce is faster, so `dispatch` (row≥4, col≥8, scalar≥8)
falls back and is **never slower than the library**. Accuracy: equal in fp32, lower error in bf16 (SFPU collapses
columns in fp32 before one rounding).
**Gist:** to reduce a wide 2-D block along one dim, don't fold it into the matmul-reduce — loop over the output
tiles, and for each accumulate its input subset (`add_tiles(acc_to_dest)`, one DEST) then finalize on the SFPU
(`sfpu_reduce` + `mul_unary_tile` 1/N). One DEST per output also sidesteps the `REDUCE_COL` DST/chunk limit. It's
a **dispatched fast path, not a replacement**: it loses below the (dim-dependent) crossover, wins least on col,
and the win is compute-only / single-core (most real reductions are data-movement-bound, where it won't show).
The same `AccumulateViaAdd` datapath also (correctness-validated in the bench, not part of the perf table)
handles **partial** (non-tile-aligned) row/col via a masked bcast-mul on the last tile, **streaming**
(`WaitAndPopPerTile` — DST is the accumulator, ~2 tiles resident), and **cross-call accumulate** (a raw
partial-sum CB tile folded into the pairwise add natively by parity — no `binary_dest_reuse` — finalized only on
the last chunk).
## ⭐⭐/⭐⭐⭐ T2/T3 — [`sfpu_tile_scope`](sfpu_tile_scope/README.md)
**Concept:** SFPU work-scoping — run only the 32-lane **vector ops** that cover the meaningful axis of a tile,
instead of the whole 32×32 tile. An SFPU vector op = 4 rows × 8 stride-2 columns; a tile = 32 vector ops (4
faces × 4 row-groups × 2 column parities), walked `[rg0-even, rg0-odd, rg1-even, …]`.
**Situation:** you apply `rsqrt`/`recip` (a norm denominator, softmax `1/rowsum`, any reduce-then-activate
epilogue) to a tile whose useful value lives on **one axis** — a per-row result in column 0, a per-column
result in row 0, or a scalar at `[0,0]` — but the SFPU runs all 32 vectors, most on lanes you never read.
**Measured win (BH, 1 core, sharded L1, isolated MATH-thread ns per SFPU call — copy+pack OUTSIDE the timed
`DeviceZoneScopedN`, only the SFPU on the math thread; cost is ~flat per vector op, ~24 ns rsqrt / ~28 ns
recip; zone unpack/pack ≈0 ns = proof of isolation):** the ladder is just vector count — rc=32 → r/c=16
(**1.98×**) → face=8 (**3.96×**) → face_iter1=1 (**26.5×**). The two axis-optimal tricks: **row-0 via
`ITERATIONS` alone** — `r_iter2` (`VectorMode::R` + `ITERATIONS=2`) = 4 vectors, **7.26× vs rc (rsqrt), 7.37×
(recip)**, ~3.7× vs the coarse `R`; **col-0 via an address stride (raw sfpi)** — `c_skip` (`VectorMode::C` +
even-parity `dst_reg+=2`) = 8 vectors, rsqrt **3.84× vs rc / clean 1.94× vs `c`** (same body). `r_iter2` (4)
beats `c_skip` (8): a row collapses to one row-group, a column still spans all 32 rows. Caveat: `recip`'s
`c_skip` (~10×) is confounded — reciprocal's fast path uses `SFPLOADMACRO` addressing that can't be strided, so
the skip forces a cheaper hand-written Newton body (ns/vector ≈11 vs 28); the clean pure-skip number is rsqrt's
1.94× over `c`. ISOLATION only — a full op's copy/pack/DRAM dilute it, a DM-bound op won't show it.
**Gist:** when a reduction has collapsed data onto one axis, scope the SFPU to match. **Row-0 result → the
`ITERATIONS` knob** (`VectorMode::R` + `ITERATIONS=2`): the row waste is the OUTER walk axis, so truncating
iterations keeps just the top row-group of both top faces (4 vectors). **Col-0 result → an even-parity address
stride** (raw sfpi `dst_reg += 2` inside a `VectorMode::C` body): the waste is column PARITY, the INNER walk
axis, which `ITERATIONS` can't isolate — you skip the odd-parity vectors (they never touch column 0), 8 vectors.
Coarser fallbacks: a half-tile result → `VectorMode::R`/`::C`; a scalar → `VectorMode::None` (+ `ITERATIONS=1`).
`recip_tile` takes `vector_mode` directly; `rsqrt_tile` hardcodes it (scope via the `SFPU_UNARY_CALL` macro);
neither exposes `ITERATIONS` or a parity stride, so `r_iter2`/`c_skip` need the underlying calls.

## ⭐⭐ T2 — [`mcast_topology`](mcast_topology/README.md)
**Concept:** the multicast topology a **2-D (block-sharded) work split** requires — two **1-D** mcast families (`Mcast1D(PerRow)` + `Mcast1D(PerColumn)`) — vs. every core re-reading its own operand slices from DRAM.
**Situation:** neither `M` nor `N` alone is long enough to fill the grid, so you split both at once (grid rows carry `M`, grid columns carry `N`). Core `(x=c, y=r)` then needs `A[M_r, :]` and `B[:, N_c]`. Written the obvious way every core fetches its own two slices, which is redundant *by construction* — all `Gc` cores in a row want the same A slice and all `Gr` cores in a column want the same B slice. This is exactly the point where "a 2-D split costs `P×` the operand traffic" gets written down as a reason to reject the 2-D split; it is only true if every core reads for itself.
**Measured result:** delivering the same operands to the same 64 cores is **1.91× faster** with the two 1-D mcasts (Blackhole, 11×10 grid, 8×8 split, `M=8t N=32t K=4t`; **8512 ns → 4450 ns**). DRAM tile-reads drop `1280 → 160` (**8×**). As in `shared_input_reuse`, the device-time win is *much smaller than the read-count reduction* — each line's sender reads its slice serially and the bytes still cross the NoC.
**Gist:** broadcast an operand along the axis it does **not** vary with. On a 2-D split that means **two** `Mcast1D` families on the same grid at disjoint `base_sem_id`s (0 and 2) — `PerRow` for the operand that is constant along a row, `PerColumn` for the one constant down a column — with each core a sender on one, both, or neither (four CT-specialized reader kernels, so every core hosts exactly one). **The naming inverts and this is the trap:** a 2-D work split needs **1-D** mcasts, while a 1-D work split (cut `M` only, every core needing all of `B`) is what needs a **2-D** mcast to a whole rectangle. "More sharded" means a *shorter, narrower* path per operand, not a bigger broadcast. Ordering is deadlock-free because every core completes its A phase before any core can block in its B phase.

## ⭐⭐⭐ T3 — [`shared_input_reuse`](shared_input_reuse/README.md)
**Concept:** redundant-DRAM-read elimination — stream a shared input once and NoC-multicast it (the `mcast_pipe` helper) vs. every core re-reading it from DRAM.
**Situation:** a grid of cores all need the **same** multi-MB input — a large shared matrix `[R, C]` (~2.4 MB) streamed in fixed-size chunks (larger than L1). Written the obvious way, every core streams the whole input from DRAM — `N×` the unique bytes.
**Measured result:** reading each chunk once on a top-left injector and `mcast_pipe`-broadcasting it to the other cores is **1.71×** faster than per-core DRAM reads (Blackhole, 22 cores = 2×11, shared input = 19×16×4 = 1216 tiles ≈ 2.4 MB bf16; **135 µs → 79 µs**). The device-time win is *smaller than the ~11× DRAM-read-**count** reduction* forwarding gives, because the single injector reads the input serially and the bytes still cross the NoC.
**Gist:** for a shared, re-read multi-MB input at grid scale, read each chunk once + `noc_async_write_multicast` to the rest (use the `mcast_pipe` `SenderPipe`/`ReceiverPipe` helper + `ttnn.Mcast2D` host wiring; sender-in-rect self-excludes; double-buffer so the injector prefetches while consumers drain). The win grows with core count and with more concurrent injectors (one per independent stream). Two orthogonal correctness notes it demonstrates: keep output ≪ input (write a tiny per-core reduction, not the block) so you measure the READ not the write; and accumulate bf16 data in **fp32** via `add_tiles(acc_to_dest)` (never `binary_dest_reuse`, which round-trips the sum through a bf16 Src register and saturates at 256).

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
**Concepts:** Tensix-to-Tensix collective topology and reduction work distribution (seven variants,
incl. two grid-hierarchy vs. flat-root reducers).
**Situation:** every core in each rectangular L1-sharded group contributes the same tile block,
and every member needs the elementwise group sum.
**Measured result:** with FPU destination-reuse reduction, reduce-scatter worker reduction beats ring
push by **4.64–4.73×** on 8-core lines and **6.48×** on a 16-core `2x8` group (**8.36 µs** versus
**54.18 µs**, 9.8% noise, WH B0). On 4-core groups, root reduction is fastest at **4.00 µs** because
the extra reduce-scatter handoff is not amortized. On **fully 2-D groups the best reducer is
regime-dependent** (Blackhole): under **grid-filling / multi-group NoC contention**,
`tree_reduce_mcast` (hierarchical reduce along one grid axis then the other: reduce x →
row-leaders, reduce y → root, mcast back) wins — **1.45–1.60×** over flat root, **3.58–3.88 µs**, and
the only steady variant (<1% vs 15–28% noise) because its per-axis fan-in stays small (`cols`, then
`rows`) and its traffic is localized. But in an **isolated single group with several tiles/core** the
tile-index `reduce_scatter_mcast` wins instead (**~2×** over root) by parallelizing across tiles;
at **1 tile/core** tree reduce wins again (low fan-in; reduce-scatter degenerates to one worker).
Rule of thumb: **tree reduce when the grid is busy or the payload is tiny; tile-index reduce-scatter
for an isolated, well-fed group.** On a 1-D group tree reduce collapses to the single root reduce.
**It is also robust to ragged splits, which is the normal case:** `num_tiles` need not divide the worker
count, and a ragged split measures on the same curve as an even one (a ragged 20-tiles-over-8 point lands
on the straight line through its even 16 and 24 neighbours, well inside run-to-run spread). Two properties
to copy when distributing any ragged split over a CB: size both CBs at the **uniform** `max_assigned =
ceil(num_tiles/W)` and have every worker push/pop that amount with gather stride `contributor *
max_assigned` rather than its own share — a CB's capacity must be an exact multiple of its push/pop
quantum, so a per-worker quantum wraps at a different offset on every core and contributors land on each
other's slots (the unread pad slots cost a little L1 and nothing else); and let **every** core work, root
included (`W = min(num_tiles, group_size)`, the root taking a share and only then multicasting), since
reserving the root idles `1/G` of the group for the whole gather/reduce phase and leaves `W` coprime with
power-of-two tile counts — a *more* ragged split, not less.
**L1 is the other axis, and the gap is bigger than the speed gap:** the root reducers **push** into the
root's gather buffer, so that buffer must be allocated identically on *every* core for the contributor's
local `get_write_ptr()` to resolve to the root's address — `G * T * P` per core, growing with group size.
Reduce-scatter **pulls** from each contributor's (already symmetric) shard into a buffer nobody else
addresses, so it lives only on the `W` workers and holds only that worker's `1/W` slice — `(G+1) * A * P
~= T * P`, essentially **independent of `G`**. Measured on Blackhole (1.5 MB L1, bf16, one group): flat
root tops out at **70** tiles/core on a `1x8` line and **36** on `4x4`, tree reduce at **64** on `4x4`,
reduce-scatter at **224** on both — **3.2×/6.2×** more payload. Past ~200 tiles/core the binding cost is
the `2*T*P` of the input and output shards, not the reducer. Headroom, not speed, is usually what rules
the root reducers out first.
**Push vs pull for the gather, measured both ways** (`reduce_scatter_push` vs `reduce_scatter_mcast`):
**no L1 difference and none required** — the gather buffer's size follows the *work* split (each worker
holds `G` copies of its own `1/W` slice), not the transfer direction; both top out at 224 tiles/core on
`1x8`. **It is not contention:** `--collect-noc-traces` + tt-npe gives a congestion cost of *exactly 0
cycles* for both directions at both sizes, avg link util ≤11%. A trace census shows why push loses —
byte-for-byte identical payload plus **512 extra `SEMAPHORE_INC`** (`G*W` per group), the handshake push
needs and pull does not (pull reads the immutable input, so no contributor-ready notification exists), so
push does strictly more NoC work. Measure with **`--kernel-iters 1`**: push has no consumer back-pressure,
so in-kernel repeats let contributors run ahead and inflate it. Corrected picture — push is monotone and
stable (~247 ns/tile, ≤1.5% noise); **pull is erratic above ~24 tiles/core** (36 t/c is 1.46× slower than
48 t/c, noise to 10.4%), and that instability, not a push advantage, is what looks like a crossover.
Prefer **pull** (less NoC work, no handshake, faster where it behaves); pick **push** when predictability
beats the median.
