# Performance examples — catalog

Short, self-contained, **runnable** ops that each isolate one kernel-level performance lever and
**measure** it on device. Every entry says when the lever helps **and when it does not**, with
real device-kernel ns for both, and the board they were measured on.

**How to use this catalog.** Read this index (it ends at `END OF INDEX`). Measure your op first:
stage breakdown (stub reads / writes / compute and time what is left), achieved GB/s vs the
DRAM/NoC ceiling, active Tensix cores vs the grid. Find your row below, then search the rest:
`grep -n "^## " master.md` lists every entry with its problem type; `grep -n -A10 "<example>"`
returns one whole entry. Details, CLI and code: the example's `README.md`.

| If you measure… | Problem type | Examples |
|---|---|---|
| Few Tensix cores active, or one core holds the whole time | **Grid not filled** | `width_split` |
| Reads/writes dominate, achieved GB/s far below the ceiling, few reads in flight per barrier | **Data movement below the ceiling: latency** | `double_buffer` |
| …and each page goes out as several small writes, one barrier each | **…: transfer size** | `tile_reorder` |
| …and one data-movement RISC-V's own time equals the kernel time and scales with the read count | **…: issue-bound** | `split_reader` |
| …and time changes with which cores / which NoC run the same bytes | **…: contention** | `noc_placement`, `tensix_all_reduce_ring_transport` |
| Many Tensix cores fetch the same bytes from DRAM | **Redundant reads** | `shared_input_reuse`, `mcast_topology` |
| Compute dominates and cost is per helper call, not per tile | **Compute-bound: fixed cost per call** | `compute_block_size` |
| …intermediates or an accumulator round-trip through L1 between steps | **Compute-bound: L1 round-trips** | `compute_fusion`, `eltwise_l1_vs_dest_accumulate` |
| …a reduction over many tiles per output | **Compute-bound: reduction** | `reduce_accumulate`, `row_reduce_accumulate`, `reduce_block`, `tensix_all_reduce_compute` |
| …SFPU work on lanes whose results are never read | **Compute-bound: wasted SFPU work** | `sfpu_tile_scope` |
| …matmul reloading SRC operands per output tile | **Compute-bound: SRC operand reload** | `matmul_output_subblock` |
| Tensix cores wait on each other (gather, reduce, ring) | **Cross-core communication** | `tensix_all_reduce` |

Each entry, in this order: `## <example> — <problem type>` · **Situation** · **How to recognize
it** · **Helps when** · **Does not help when** · **Gist** · **See also**. Numbers are device kernel
ns, baseline → lever, speedup = baseline / lever (< 1.00× = slower). Wormhole numbers (WH B0, n150)
come from each example's `test_<example>_regimes`, which asserts both sides; Blackhole (BH)
numbers are labelled. **Not covered yet** (no example — measure it yourself): fixed launch cost of
kernels with nothing to move; data already resident in L1 (CB aliasing); DRAM bank contention from
issue order; read-barrier granularity.

<!-- END OF INDEX -->

---

# Grid not filled

## width_split — grid not filled
**Situation:** a wide, short tensor (1 tile-row, Wt tile-columns) split by tile-rows, so one Tensix core does all Wt tiles.
**How to recognize it:** active Tensix cores = tile-rows (1) while Wt ≫ 1; time grows ~240 ns per tile above a ~1.8 µs launch floor.
**Helps when:** Wt ≥ 3 tile-columns: `[32,2048]` (Wt=64, 64 cores) 16 832 → 2 720 ns = **6.19×** (WH B0); Wt=3 1.30×, Wt=8 2.2×.
**Does not help when:** Wt ≤ 2: `[32,64]` 2 076 → 2 105 ns = **0.99×** (0.89–0.99×, WH B0) — the single core is already at the launch floor.
**Gist:** give each core a contiguous tile-column range over min(Wt, grid) cores (capped by a chunk constant for L1).
**See also:** `double_buffer` (per-core efficiency once the grid is full).

# Data movement below the ceiling

## double_buffer — data movement below the ceiling: latency
**Situation:** a reader that does *read one tile, barrier, push* into one-tile CBs.
**How to recognize it:** per-core DRAM rate ~6.5 GB/s (bf16) vs ~18 GB/s single-core ceiling, and cores × 6.5 GB/s well below the ~190 GB/s DRAM peak.
**Helps when:** cores × per-core rate ≪ DRAM peak (≲16 cores, bf16): 1 core `[512,512]` 155 933 → 59 156 ns = **2.64×**; 8 cores `[2048,2048]` 322 468 → 118 236 ns = **2.73×** (WH B0).
**Does not help when:** the naive reader already saturates DRAM: 64 cores `[2048,2048]` 87 884 → 89 425 ns = **0.98×** (190.9 GB/s untuned, WH B0). Sweep at 4096 tiles: 16 cores 1.97×, 32 cores 1.11×, 64 cores 0.98×.
**Gist:** issue ~4–8 async reads per barrier and size each CB to 2 × block; skip it once cores × naive rate reaches DRAM peak.
**See also:** `split_reader`, `width_split`.

## tile_reorder — data movement below the ceiling: transfer size
**Situation:** a whole-tile relocation whose writer sends each 2 KB tile as 4 × 512 B face writes with a barrier after each face.
**How to recognize it:** the writer RISC-V (BRISC) is on the critical path and its time scales with the write-barrier count, not with bytes; the gap persists with an L1 output, so it is not DRAM bandwidth.
**Helps when:** destination pages are remote to the writer: DRAM out `(2048,1024)` 71 855 → 45 513 ns = **1.58×**; L1-interleaved out 43 773 → 31 317 ns = **1.40×** (WH B0). Small moves gain too (2 tiles 2.24×).
**Does not help when:** every output page is in the writer's own L1 (L1 HEIGHT_SHARDED, one tile-row per core): 23 666 → 23 515 ns = **1.01×** (WH B0).
**Gist:** write whole pages with one barrier per page or batch whenever the target is remote — the win is fewer barrier round trips.
**See also:** `noc_placement`, `split_reader`.

## split_reader — data movement below the ceiling: issue-bound
**Situation:** one Tensix core gathers many small NoC reads issued from NCRISC alone while BRISC only writes.
**How to recognize it:** NCRISC lifetime ≈ device time and grows with the read count at fixed bytes (128 KiB: 32 B reads 146 µs vs 2048 B reads 24 µs).
**Helps when:** ≤ 256 B per NoC read: 32 B reads (4096 reads/launch) 147 274 → 79 968 ns = **1.84×** (WH B0).
**Does not help when:** no flat regime is reachable in this example — with ≥ 1 KiB reads the gain shrinks to **1.38×** (2048 B: 25 041 → 18 182 ns, WH B0); lowest seen 1.29×. The remaining gain is per-tile barrier latency, which batching reads (`double_buffer`) also removes.
**Gist:** split disjoint reads across NCRISC and BRISC only when the reader's lifetime scales with read count; otherwise batch reads per barrier first.
**See also:** `double_buffer`.

## noc_placement — data movement below the ceiling: contention
**Situation:** a line of Tensix cores streams interleaved-DRAM pages, laid out as a column (`split_work_to_cores(..., row_wise=False)`, the default) and/or with reads on NoC1.
**How to recognize it:** same bytes and kernel, but time changes a lot when the line moves to a row or the NoC flips; per-core time grows with outstanding reads on the line.
**Helps when:** ≥ 16 outstanding 2 KB reads on the line (8 cores × 16 per barrier): column → row 246 765 → 84 917 ns = **2.91×**; reads NoC1 → NoC0 278 002 → 85 066 ns = **3.27×** (WH B0).
**Does not help when:** latency-bound line (2 cores × 1 read per barrier): column → row 1 106 889 → 1 077 065 ns = **1.03×**; on a single core the default NoC pairing is **0.94×** (WH B0). Payload size is not the boundary (16 tiles still 1.51×); reads in flight are.
**Gist:** with ≥ ~16 pages in flight, use `row_wise=True` (row/diagonal) and keep reads on NoC0 / writes on NoC1; with 1 read per barrier fix latency first.
**See also:** `split_reader`, `mcast_topology`, `tile_reorder`.

## tensix_all_reduce_ring_transport — data movement below the ceiling: contention
**Situation:** a reduce-and-forward ring of neighbour L1 writes whose ring order runs against the chosen NoC's routing direction.
**How to recognize it:** the same ring payload on NoC0 vs NoC1 differs by a large factor on a line; the semaphore-only variants differ by only ~2×.
**Helps when:** ring along a line (all hops one direction), 12 KiB per hop: 1×8 × 8 groups NoC1 → NoC0 26 373 → 4 364 ns = **6.04×** (WH B0).
**Does not help when:** a 2-row serpentine (equal traffic both directions): 2×8 × 4 groups 48 920 → 47 490 ns = **1.03×** (WH B0).
**Gist:** on line rings keep ring order along the NoC's routing direction; on a serpentine, switching NoC does nothing — reorder or split the ring.
**See also:** `noc_placement`, `tensix_all_reduce`.

# Redundant reads

## shared_input_reuse — redundant reads
**Situation:** every core in a rectangle streams the same multi-MB DRAM input in chunks; lever: one injector reads each chunk once and multicasts it (`mcast_pipe`).
**How to recognize it:** N cores read identical page ids, so DRAM bytes = N × unique bytes; check the chunk size (bytes per multicast handshake).
**Helps when:** ≥ 64 tiles (≥ 128 KB) per chunk, 16 cores: 1216 tiles 255 592 → 191 246 ns = **1.34×**; 128-tile chunks 1.45× (WH B0). BH, 22 cores: 1.71×.
**Does not help when:** ≤ 16 tiles per chunk: 8-tile chunks 214 339 → 262 332 ns = **0.82×** (WH B0); 4-tile 0.72×, 16-tile 0.93×, 32-tile 1.04× — crossover ~32 tiles (64 KB) per chunk. Stream length is not the boundary.
**Gist:** multicast a shared stream only with chunks large enough to amortize the ack-then-broadcast handshake; otherwise enlarge chunks first, or keep per-core reads.
**See also:** `mcast_topology`.

## mcast_topology — redundant reads
**Situation:** a 2-D work split where every core in a row needs the same A slice and every core in a column the same B slice, each fetched from DRAM; lever: two 1-D multicasts (`Mcast1D` PerRow + PerColumn).
**How to recognize it:** count cores reading identical tiles — the line length is the DRAM-read reduction factor.
**Helps when:** full 8×8 grid (lines of 8): Mt=8 Nt=32 Kt=4 16 204 → 8 677 ns = **1.87×**; Kt=16 2.67×, Kt=64 3.47× (WH B0). BH 8×8: 1.91×.
**Does not help when:** lines of ≤ 4 cores: 4×4 4 186 → 4 056 ns = **1.03×**; 2×2 2 288 → 3 120 ns = **0.73×** (WH B0); 1×8 0.84×, 2×8 0.86×.
**Gist:** use two 1-D multicasts only when the split fills a big grid with several tiles per core; on ≤ 4×4 grids per-core reads match or beat them. A 2-D *work* split needs 1-D multicasts.
**See also:** `shared_input_reuse`.

# Compute-bound

## compute_block_size — compute-bound: fixed cost per call
**Situation:** a row-parallel compute chain (tilize, tilize, add, matmul, untilize) looped one tile-row at a time.
**How to recognize it:** time vs number of passes over M: ~1.5 µs per extra pass, flat across payload size.
**Helps when:** many passes, small payload each: M,K,N = 8,4,4 tiles 29 064 → 17 271 ns = **1.68×** (WH B0); 8,2,16 1.39×.
**Does not help when:** few passes that each carry tens of µs: 2,16,16 tiles 37 202 → 35 634 ns = **1.04×** (WH B0).
**Gist:** run each helper on the whole row-parallel block (or the largest L1 allows); also drop data-format reconfig at boundaries where the format is constant (up to 1.19×, see README).
**See also:** `compute_fusion`.

## compute_fusion — compute-bound: L1 round-trips
**Situation:** one helper call per op, each packing its result to a CB that the next op unpacks.
**How to recognize it:** the phase after the producer is unpack-bound although its math is cheap; check whether the consumer is SFPU or FPU.
**Helps when:** SFPU consumer, small work per call: `sfpu_chain` 4 tiles 9 674 → 8 632 ns = **1.12×** (WH B0); the win is ~1 µs per call, so it shrinks with tile count (~1.05× at 64 tiles).
**Does not help when:** FPU consumer: `fpu_sfpu` 64 tiles 77 734 → 84 105 ns (dest-reuse) = **0.92×** (WH B0) — the DEST → SRC move costs more than the pack + unpack it replaces.
**Gist:** keep intermediates in DEST when the next op is SFPU; for an FPU consumer pack to L1 and let the unpacker feed it back.
**See also:** `eltwise_l1_vs_dest_accumulate`, `sfpu_tile_scope`.

## eltwise_l1_vs_dest_accumulate — compute-bound: L1 round-trips
**Situation:** a running L1 accumulator rebuilt each step by unpacking it, adding on the FPU and packing it back (`rmw`).
**How to recognize it:** `rmw` grows ~370 ns per step while packer L1-accumulation (`pack_l1_acc`) costs ~90 ns per step.
**Helps when:** `rmw` → `pack_l1_acc` at every step count: 64 fp32 steps 23 784 → 5 769 ns = **4.12×**; 2 steps 2.03× (WH B0). BH 64 steps: 5.09×; DEST-resident `dest_acc` 10.59× (BH).
**Does not help when:** short loops for the DEST-resident version: 2 steps `pack_l1_acc` → `dest_acc` 367 → 370 ns = **0.99×** (WH B0); it grows to 1.09× at 4 steps and 1.30× at 64.
**Gist:** fold pairs into the L1 accumulator with `pack_l1_acc`; reserve DEST (`dest_acc`) only for long loops when DEST is free. Syntax for one stream: README.
**See also:** `compute_fusion`, `reduce_accumulate`.

## reduce_accumulate — compute-bound: reduction
**Situation:** a SUM/mean reduce of N tiles written with the reduce library (matmul-with-ones per tile); lever: pairwise `add_tiles` accumulate + one SFPU finalize.
**How to recognize it:** library time grows ~linearly per tile; the fast path has an ~800–900 ns floor plus a small slope.
**Helps when:** ≥ 8 tiles reduced per output: row 32 tiles 3 652 → 1 865 ns = **1.96×**, col 1.72×, scalar 2.93× (WH B0).
**Does not help when:** ≤ 4 tiles per output: row 1 tile 467 → 940 ns = **0.50×**, row 4 0.83×, col 4 0.81×, scalar 4 0.95× (WH B0). The example's `dispatch` threshold row ≥ 4 is stale on WH (0.83× at 4).
**Gist:** accumulate + SFPU finalize only when the reduce spans ~8+ tiles per output; below that use the library reduce.
**See also:** `row_reduce_accumulate`, `reduce_block`.

## row_reduce_accumulate — compute-bound: reduction
**Situation:** a row mean over W tiles written as one `reduce<SUM, REDUCE_ROW>` over the row; lever: pairwise `add_tiles` into DEST then one finalize reduce (`dest_accum_pairs`).
**How to recognize it:** the single reduce costs ~70–140 ns per tile; `dest_accum_pairs` has an ~800 ns floor plus ~30–60 ns per tile.
**Helps when:** rows ≥ 16 tiles (bf16 input) or ≥ 8 (fp32): 32 tiles fp32 4 478 → 2 850 ns = **1.57×**, bf16 2 419 → 1 803 ns = **1.34×** (WH B0).
**Does not help when:** ≤ 8 tiles (bf16): 1 tile 433 → 837 ns = **0.52×**, 2 tiles 0.60×, 8 tiles 0.87× (WH B0).
**Gist:** use `dest_accum_pairs` (parity resolved at the seed, no zero CB) for wide rows; keep the single reduce for narrow ones. bf16 accumulation, not bf16 input, costs precision (README).
**See also:** `reduce_accumulate`, `reduce_block`.

## reduce_block — compute-bound: reduction
**Situation:** a 2-D `(Ht, Wt, NC)` block reduced along one dim into many output tiles with the default `reduce_tile`; lever: `AccumulateViaAdd` per output tile.
**How to recognize it:** count reduced tiles per output tile (row Wt, col Ht, scalar Ht×Wt) — that, not block size, decides it.
**Helps when:** ≥ 8 reduced tiles per output: row `(2,32,1)` 6 977 → 3 343 ns = **2.09×**, col 1.88×, scalar `(4,8,1)` 3.04× (WH B0).
**Does not help when:** ≤ 4 per output: row `(4,2,1)` 1 307 → 2 569 ns = **0.51×**, row `(2,4,1)` 0.81×, col 0.90×, scalar `(2,2,1)` 1.02× (WH B0).
**Gist:** opt into `AccumulateViaAdd` only when each output reduces ~8+ tiles; one DEST per output also avoids the `REDUCE_COL` chunk limit.
**See also:** `reduce_accumulate`, `row_reduce_accumulate`.

## tensix_all_reduce_compute — compute-bound: reduction
**Situation:** a reducer that copies each contributor into DEST, re-runs `add_binary_tile_init()` and does one SFPU add per contributor.
**How to recognize it:** single-core reducer time grows ~linearly with (blocks − 1) × tiles with low unpack/pack counts.
**Helps when:** always, growing with blocks: 8 blocks × 6 tiles 11 178 → 1 899 ns = **5.89×**; 2 blocks 1.90×, 4 blocks 3.01× (WH B0).
**Does not help when:** no slower case exists in the measured range (≥ 2 blocks); the smallest gain is **1.38×** (2 blocks × 1 tile, one call: 783 → 568 ns, WH B0). FP32 vs BF16 DEST on the fast path is only 1.03×.
**Gist:** pairwise FPU `add_tiles(..., acc_to_dest=true)`, one init per DEST batch, pack once.
**See also:** `eltwise_l1_vs_dest_accumulate`, `tensix_all_reduce`.

## sfpu_tile_scope — compute-bound: wasted SFPU work
**Situation:** `rsqrt`/`recip` run on all 32 vector ops of a tile whose meaningful result lives on one row or column after a reduction.
**How to recognize it:** MATH-thread zone ns per SFPU call is flat per vector op (~35.6 ns rsqrt, ~17.6 ns recip on WH), so cost tracks the vector count.
**Helps when:** SFPU math dominates and the result is on row 0: `VectorMode::R` + `ITERATIONS=2` (4 vectors) rsqrt 1 139 → 156 ns = **7.30×**, recip 6.69× (math zone); whole op on 1 tile 2.22× / 1.61× (WH B0).
**Does not help when:** swapping scopes with the same vector count: `VectorMode::C` → `R` (16 vectors) 576 → 576 ns = **1.00×** (WH B0); at op level copy/pack dilute the gain.
**Gist:** scope the SFPU to the minimum vectors covering the result (row 0: `ITERATIONS=2`; col 0: even-parity stride, README).
**See also:** `reduce_accumulate`.

## matmul_output_subblock — compute-bound: SRC operand reload
**Situation:** `matmul_block` with a 1×1 output subblock, so every output tile reloads both operands into SRC.
**How to recognize it:** sweeping the subblock shape at fixed Mt/Nt/Kt drops time 1.5–1.8× at 8-tile subblocks.
**Helps when:** many output tiles per call: 16×16 output tiles, Kt=1, 1×1 → 2×4 24 312 → 13 905 ns = **1.75×**; tall 8×1 → wide 1×8 1.40× (WH B0). Still 1.66–1.69× at Kt=2–12.
**Does not help when:** only a handful of output tiles with short K: 2×2 output tiles, Kt=1, 1×1 → 2×2 670 → 656 ns = **1.02×** (WH B0); 2×2 at Kt=8 still 1.34×.
**Gist:** use the largest subblock that fits DEST (2×4, 4×2, 1×8 — avoid a tall 8×1 on WH); with fp32 DEST the budget halves to 4 tiles.
**See also:** `tile_reorder`.

# Cross-core communication

## tensix_all_reduce — cross-core communication
**Situation:** cores in an L1-sharded group all need the group sum, and one root core gathers and adds every contributor serially, then multicasts (`reduce_root_mcast`).
**How to recognize it:** the root alone is busy for (group size − 1) × tiles-per-core adds while the others idle.
**Helps when:** groups ≥ 8 cores and ≥ 6 tiles per core: `reduce_scatter_mcast` on 1×8 × 8 groups, 6 tiles 6 145 → 3 646 ns = **1.68×** (WH B0).
**Does not help when:** 4-core groups (3 976 → 4 663–5 120 ns = **0.78–0.85×**) or 1 tile per core (1 614 → 1 994 ns = **0.81×**) (WH B0) — keep root reduction there.
**Gist:** reduce-scatter for well-fed groups of 8+; root (or BH tree) reduction for small groups or tiny payloads; reduce-scatter also needs far less L1 per core (README).
**See also:** `tensix_all_reduce_compute`, `tensix_all_reduce_ring_transport`.
