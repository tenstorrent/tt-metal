# Performance examples — catalog

Short, self-contained, **runnable** ops that each isolate one kernel-level performance lever and
**measure** it on device. Every entry says when the lever helps **and when it does not**, with
real device-kernel ns for both, on **both** Wormhole B0 and Blackhole where measured.

**How to use this catalog.** Read this index (it ends at `END OF INDEX`). Find your board below and
note how it differs from the other. Measure your op first: stage breakdown (stub reads / writes /
compute and time what is left), achieved GB/s vs the DRAM/NoC ceiling, active Tensix cores vs the
grid. Find your row below, then search the rest: `grep -n "^## " master.md` lists every entry with its
problem type; `grep -n -A12 "<example>"` returns one whole entry; its **Code** line names the file and
functions where the lever lives. Details, the full regime × board table and CLI:
`examples/<example>/README.md`.

| If you measure… | Problem type | Examples |
|---|---|---|
| Few Tensix cores active, or one Tensix core holds the whole time | **Grid not filled** | [`width_split`](width_split/) |
| Reads/writes dominate, achieved GB/s far below the ceiling, few reads in flight per barrier | **Data movement below the ceiling: latency** | [`double_buffer`](double_buffer/), [`trid_double_issue`](trid_double_issue/) |
| …and each page goes out as several small writes, one barrier each | **…: transfer size** | [`tile_reorder`](tile_reorder/) |
| …and one data-movement RISC-V's own time equals the kernel time and scales with the read count | **…: issue-bound** | [`split_reader`](split_reader/) |
| …and time changes with which Tensix cores / which NoC run the same bytes, or with the order the Tensix cores walk the DRAM banks | **…: contention** | [`noc_placement`](noc_placement/), [`tensix_all_reduce_ring_transport`](tensix_all_reduce_ring_transport/), [`bank_stagger`](bank_stagger/) |
| Many Tensix cores fetch the same bytes from DRAM | **Redundant reads** | [`shared_input_reuse`](shared_input_reuse/), [`mcast_topology`](mcast_topology/) |
| Compute dominates and cost is per helper call, not per tile | **Compute-bound: fixed cost per call** | [`compute_block_size`](compute_block_size/) |
| …intermediates or an accumulator round-trip through L1 between steps | **Compute-bound: L1 round-trips** | [`compute_fusion`](compute_fusion/), [`eltwise_l1_vs_dest_accumulate`](eltwise_l1_vs_dest_accumulate/) |
| …a reduction over many tiles per output | **Compute-bound: reduction** | [`reduce_accumulate`](reduce_accumulate/), [`row_reduce_accumulate`](row_reduce_accumulate/), [`reduce_block`](reduce_block/), [`tensix_all_reduce_compute`](tensix_all_reduce_compute/) |
| …SFPU work on lanes whose results are never read | **Compute-bound: wasted SFPU work** | [`sfpu_tile_scope`](sfpu_tile_scope/) |
| …matmul reloading SRC operands per output tile | **Compute-bound: SRC operand reload** | [`matmul_output_subblock`](matmul_output_subblock/) |
| Tensix cores wait on each other (gather, reduce, ring), or a combine's time grows with the number of Tensix cores at fixed work per core | **Cross-core communication** | [`tensix_all_reduce`](tensix_all_reduce/), [`tensix_all_reduce_wave_pipeline`](tensix_all_reduce_wave_pipeline/) |
| Input and output already L1-sharded, but the work split ignores the shard spec: Tensix cores gather sticks from each other, and time grows with Tensix cores, not bytes | **Cross-core communication: resident shards** | [`resident_shard_locality`](resident_shard_locality/) |

## Wormhole vs Blackhole — read before trusting a number

Boards measured: **WH B0 n150** and **BH p150b** (plus older BH p150 / p100a / p300c runs where an
entry says so). Every number below names its board. A lever's *direction* is the same on both archs
for most examples, but the **boundary** (how many tiles, Tensix cores or steps before it pays) moves,
and for a few levers the verdict **flips**.

| | WH B0 n150 | BH p150b |
|---|---|---|
| Tensix compute grid | 8×8 = 64 Tensix cores | 12×10 = 120 Tensix cores (p100a / p300c: other grids — query `compute_with_storage_grid_size()`) |
| DRAM banks | 12 | 8 (p100a: 7) — interleaved page `p` lands in bank `p % 8` |
| AICLK | 1000 MHz | 1350 MHz |
| L1 per Tensix core | 1464 KB | 1536 KB |
| NoC flit / max single packet | 32 B / 8 KB | 64 B / 16 KB |
| DRAM alignment (NoC DRAM read alignment) | 32 B | **64 B** — row-major pages pad to 64 B |
| DST capacity | 16 tiles full, 8 half-sync bf16, 4 half-sync fp32 | same |
| Kernel launch floor (1-tile relu, 1 Tensix core) | ~1.8 µs | ~1.0 µs |
| Achieved DRAM read + write, full grid | ~190 GB/s | ~415–430 GB/s |

Where the boundary or verdict moves on BH p150b (everything else: same verdict, numbers in each entry):

| Example | WH B0 n150 | BH p150b |
|---|---|---|
| `width_split` | pays from Wt = 3 tile-columns | pays from Wt = 2 (lower launch floor) |
| `double_buffer` | flat at 64 Tensix cores (0.98×, DRAM saturated) | never flat: 1.07× at 120 Tensix cores (DRAM ceiling ~2× higher) |
| `tile_reorder` | DRAM output 1.58× | DRAM output **0.99×** (DRAM in + out at 120 Tensix cores is the ceiling); L1 output still 1.50× |
| `shared_input_reuse` | crossover ~32 tiles per chunk | crossover 8–16 tiles per chunk |
| `mcast_topology` | 4×4 grid flat (1.03×) | 4×4 grid loses (0.80×) |
| `matmul_output_subblock` | ~1.7× from Kt = 1 to 12; tall 8×1 slower than wide 1×8 | 1.46× at Kt = 1 → **1.10× at Kt = 4**; tall 8×1 = wide 1×8 |
| `eltwise_l1_vs_dest_accumulate` | DEST accumulation 1.30× at 64 steps | 2.01× at 64 steps |
| `reduce_accumulate` | all dims pay from ~8 tiles | row needs 16 tiles; col / scalar pay from 8 |
| `row_reduce_accumulate` | pays from 16 tiles (bf16 input) | pays from 8 tiles |
| `reduce_block` | all dims pay from ~8 tiles per output | row needs > 8; col / scalar pay from 4 |
| `compute_block_size` | flat at 2 passes (1.04×) | smallest gain 1.07×; no flat case fits L1 |
| `noc_placement` | single Tensix core loses 0.94× on the canonical NoC pairing | flat (1.02×) |

Kernel-writing differences that are not levers (code, not measurement): DRAM reads must be 64 B
aligned on BH (`NOC_DRAM_READ_ALIGNMENT_BYTES`, `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h`);
`reduce_uninit` and `untilize_uninit` take different arguments per arch (`tt_metal/hw/inc/api/compute/reduce.h`,
`untilize.h`); the matmul dynamic throttle and `skip_cmdbuf_chk` are BH-only
(`tt_metal/hw/inc/api/compute/matmul.h`, `tt_metal/hw/inc/api/dataflow/dataflow_api.h`). Shapes sized
for an 8×8 grid (64 tile-rows → one per Tensix core) do **not** map one-per-Tensix-core on 12×10.

Each entry, in this order: `## <example> — <problem type>` · **Situation** · **How to recognize
it** · **Helps when** · **Does not help when** · **Gist** · **Code** (where the lever lives) · **See also**.
Numbers are device kernel ns, baseline → lever, speedup = baseline / lever (< 1.00× = slower), each
labelled WH B0 (n150) or BH (p150b unless another board is named). Both boards' numbers come from
the example's `test_<example>_regimes`, which is arch-agnostic: it runs the same regimes on any board
and asserts only what held on every measured board; a regime whose verdict differs by arch is marked
`measure` and only logs its speedup, so the per-board verdicts live here and in the README. Sweep
points come from the README. `trid_double_issue` and `bank_stagger` have no regimes test and no WH measurement; `tensix_all_reduce_wave_pipeline`, `resident_shard_locality` and the `width_split` reduce-with-combine regimes have no WH measurement.
**Not covered yet** (no example — measure it yourself): fixed launch cost of kernels with nothing to
move.

<!-- END OF INDEX -->

---

# Grid not filled

## width_split — grid not filled
**Situation:** a short tensor split by tile-rows, so only R_t Tensix cores work (1 for a single tile-row), each over all Wt tile-columns; for a per-row reduce + apply (norm, softmax, centering) a width split also needs a per-row combine between the k Tensix cores sharing a row, and on `Layout::ROW_MAJOR` sticks it also cuts every stick into k smaller DRAM transfers.
**How to recognize it:** active Tensix cores = tile-rows R_t ≪ grid; time grows ~240 ns per tile above a ~1.8 µs launch floor (WH B0), ~160 ns (eltwise) / ~230 ns (reduce + apply) per tile above ~1.0 µs (BH). Reduce case: also check the baseline's DRAM rate, (in + out bytes) ÷ time. ROW_MAJOR: also count DRAM transfers per launch (2 · sticks · pieces per stick) × ~1 ns (BH) against bytes ÷ ~400 GB/s — pieces below ~400 B are count-bound.
**Helps when:** elementwise (no combine): WH B0 Wt ≥ 3 tile-columns, BH Wt ≥ 2. `[32,2048]` (Wt=64, 64 Tensix cores): WH B0 16 832 → 2 720 ns = **6.19×** · BH 11 092 → 1 536 ns = **7.22×**. Wt=3: WH 1.30× · BH 1.36×; Wt=8: WH 2.2× · BH 2.19×; BH Wt=256 on 120 Tensix cores 13.68×. Reduce + combine (BH p150b, 130 Tensix cores): R_t ≤ ~1/8 of the grid, k ≤ 8, ≥ 4 tiles per slice. R_t=1, Wt=32, k=8 (1 → 8 Tensix cores, 4 tiles per slice): 8 715 → 3 690 ns = **2.36×**; Wt=64, k=8: 3.45×; Wt=16, k=4: 1.71×; Wt=8, k=2: 1.17×. R_t=16 (16 of 130 Tensix cores): only 1.11–1.28×. ROW_MAJOR sticks (BH p150b): same R_t condition plus ≥ 1 KB sticks at R_t ≤ 4, ≥ 2 KB at R_t = 16, pieces ≥ 512 B (≥ 256 B at R_t ≤ 4). R_t=1, W=2048 (4 KB sticks), k=4 (1 KB pieces): 12 803 → 6 872 ns = **1.86×**; R_t=1, W=512, k=4: 1.15×; R_t=16, W=1024: 1.16–1.22×.
**Does not help when:** elementwise: WH B0 Wt ≤ 2: `[32,64]` 2 076 → 2 105 ns = **0.99×** (0.89–0.99×) — the single Tensix core is already at the launch floor. BH: only Wt = 1 (nothing to split, 1.00–1.02×); at Wt = 2 BH already gains 1 217 → 1 031 ns = **1.18×**. Reduce + combine (BH): tile-rows already on ~half the grid — R_t=64 on 64 of 130 Tensix cores, k=2 (8 tiles per slice): 13 581 → 13 759 ns = **0.98×** (baseline already at ~310 GB/s, ~75% of the DRAM ceiling; R_t=64–128: 0.86–1.09×); Wt=4 at any R_t, k: 0.69–0.95×; 1-tile slices: 0.52–0.97×; k ≥ 16 (the all-gather combine grows with k: ~0.5 µs at k=2, 1.5 µs at k=8, 4 µs at k=16). ROW_MAJOR (BH): `2048 × 256` (R_t=64, 512 B sticks → 256 B pieces, 64 → 128 Tensix cores): 8 566 → 13 032 ns = **0.66×** (2× the transfers at ~1 ns each); R_t=47: 0.69×; R_t ≥ 47 with pieces ≤ 1 KB: 0.38–1.02× (2 KB pieces 1.07–1.08×); sticks ≤ 512 B at any R_t: 0.38–1.02×.
**Gist:** give each Tensix core a contiguous tile-column range over min(Wt, grid) Tensix cores; for a row reduction split each row over k ≤ 8 neighbouring Tensix cores only while R_t·k ≤ grid and Wt/k ≥ 4 tiles, exchange 1-tile partials (faces 0 and 2 only) in one symmetric round, and check per-core time saved (~(Wt − Wt/k) × ~230 ns on BH) beats your combine, measured as T(k, Wt) − T(1, Wt/k). On ROW_MAJOR sticks a Tensix core still issues 32 reads + 32 writes per tile-row at any k, so the split only saves tilize/reduce/untilize work (~150 ns per tile on BH) while adding k-fold transfers to the grid: keep whole sticks unless R_t is small and the pieces stay ≥ 512 B.
**Code:** `examples/width_split/width_split.py` — `_num_cores_for()`, `_assign_pages()` (eltwise, host only); `reduce_layout()`, `create_reduce_program_descriptor()` + `kernels/ws_reduce_writer.cpp` (combine); `WS_ROW_MAJOR` paths in `kernels/ws_reduce_*.cpp` (stick pieces, tilize / untilize).
**See also:** `double_buffer` (per-core efficiency once the grid is full), `tensix_all_reduce` (combine topologies across many Tensix cores).

# Data movement below the ceiling

## double_buffer — data movement below the ceiling: latency
**Situation:** a reader that does *read one tile, barrier, push* into one-tile CBs.
**How to recognize it:** per-Tensix-core DRAM rate ~6.5 GB/s (WH B0) / ~8.8 GB/s (BH), bf16, vs a single-core ceiling of ~18 / ~26 GB/s, and Tensix cores × that rate well below the achieved DRAM ceiling (~190 GB/s WH B0, ~415–430 GB/s BH).
**Helps when:** Tensix cores × per-core rate ≪ DRAM ceiling. 1 Tensix core `[512,512]`: WH B0 155 933 → 59 156 ns = **2.64×** · BH 118 883 → 40 361 ns = **2.95×**. 8 Tensix cores `[2048,2048]`: WH B0 322 468 → 118 236 ns = **2.73×** · BH 238 575 → 86 108 ns = **2.77×**. At 4096 tiles: WH 16 Tensix cores 1.97×, 32 1.11× · BH 16 2.44×, 32 1.55×.
**Does not help when:** the naive reader already saturates DRAM. WH B0 64 Tensix cores `[2048,2048]` 87 884 → 89 425 ns = **0.98×** (190.9 GB/s untuned). BH never goes flat: 64 Tensix cores 1.08×, 120 Tensix cores 43 554 → 40 519 ns = **1.07×** (385 → 414 GB/s) — the BH DRAM ceiling is ~2× WH, so the gain shrinks but stays.
**Gist:** issue ~4–8 async reads per barrier and size each CB to 2 × block. On WH B0 skip it once Tensix cores × naive rate reaches DRAM peak; on BH keep it — it still adds ~7–9% at the full grid.
**Code:** `examples/double_buffer/double_buffer.py` — `_DEPTH`, `create_program_descriptor()` CB `total_size` (host only; kernels unchanged).
**See also:** `split_reader`, `width_split`.

## trid_double_issue — data movement below the ceiling: latency
**Situation:** a reader that already batches reads (a block, or several blocks, per barrier) but cannot widen the block — a shard size, a wide tensor or co-resident buffers cap L1 — and whose `noc_async_read_barrier()` drains every outstanding read, so the NoC idles once per block; lever: tag each block with a transaction id and retire only the oldest (`noc_async_read_barrier_with_trid`).
**How to recognize it:** reads are batched, yet single-Tensix-core GB/s sits well below the ~110–122 GB/s single-core ceiling (BH) and keeps rising with block size and with blocks issued per barrier (`ahead`) — depth is being bought with L1.
**Helps when:** small blocks at a fixed CB (`cb_blocks=6`), 1 Tensix core, bf16 `[512,512]`: block=2 best baseline (`ahead=3`) → trid ×4: BH bh-50 39.4 → 64.3 GB/s = **1.63×** · BH p150b 39.7 → 65.1 GB/s = **1.64×**; block=1 1.54×, block=4 1.36–1.37×, block=16 1.11–1.13× (GB/s, same bytes, so the ratio is the time ratio). Per Tensix core: block=1 still 1.48× on 6 Tensix cores. At a 96 KB CB 89.8 → 107.1 GB/s = 1.19× (BH).
**Does not help when:** L1 is free to widen the block: at a 768 KB CB best baseline 120.9 → best trid 122.2 GB/s = **1.01×**, and the trivial one-block loop (118.7 GB/s) is within 3% (BH). A shallow ring only ties a tuned baseline: block=4 `ahead=3` 62.2 → `trid ×2` 62.1 GB/s = **1.00×** (BH). WH B0 and the DRAM-saturated full grid: not measured.
**Gist:** tune `ahead` first (optimum ≈ `cb_blocks/2`; block=8 52.1 → 85.1 GB/s, BH) — it is the bigger win; add a trid ring of depth 3–4 only when L1 caps the block — it reaches a given GB/s with ~1.5–2× less L1, not a higher ceiling. Land each new block at `get_write_ptr() + in_flight` slots (README). Both archs have 16 NoC transaction ids.
**Code:** `examples/trid_double_issue/kernels/trid_reader.cpp` — `if constexpr (num_trids == 0)` (full barrier, `ahead` blocks per barrier) vs the trid ring (`dataflow_kernel_lib::set_read_trid` / `async_read_barrier_with_trid`); `trid_double_issue.py` `create_program_descriptor()` passes `num_trids`, `ahead`, `cb_blocks` (reader only; writer and CB size shared).
**See also:** `double_buffer`, `split_reader`.

## tile_reorder — data movement below the ceiling: transfer size
**Situation:** a whole-tile relocation whose writer sends each 2 KB tile as 4 × 512 B face writes with a barrier after each face.
**How to recognize it:** the writer RISC-V (BRISC) is on the critical path and its time scales with the write-barrier count, not with bytes; the gap persists with an L1 output, so it is not DRAM bandwidth.
**Helps when:** destination pages are remote to the writer and the op is not DRAM-bound. L1-interleaved output: WH B0 `(2048,1024)` 43 773 → 31 317 ns = **1.40×** · BH `(3840,1024)` 38 446 → 25 612 ns = **1.50×**. DRAM output on WH B0 `(2048,1024)` 71 855 → 45 513 ns = **1.58×**. Small moves gain too (2 tiles 2.24×, WH B0).
**Does not help when:** every output page is in the writer's own L1 (L1 `HEIGHT_SHARDED`, one tile-row per Tensix core): WH B0 `(2048,1024)` 23 666 → 23 515 ns = **1.01×** · BH `(3840,1024)` 25 645 → 25 448 ns = **1.01×**. On BH, also DRAM output: `(2048,1024)` 21 180 → 21 387 ns = **0.99×** — DRAM read + write at 120 Tensix cores is ~400 GB/s, the BH DRAM ceiling, so the writer's barriers are hidden (`(3840,1024)` 1.10×).
**Gist:** write whole pages with one barrier per page or batch whenever the target is remote — the win is fewer barrier round trips; on BH it shows only for L1 destinations or when DRAM is not saturated. Size shapes to the grid: a 64-tile-row shape is one tile-row per Tensix core on 8×8 but not on 12×10.
**Code:** `examples/tile_reorder/` — `_WRITER_SOURCE` picks `kernels/tile_reorder_writer_relocate.cpp` vs `tile_reorder_writer_scatter.cpp` (writer only; reader, CBs and work split shared).
**See also:** `noc_placement`, `split_reader`, `double_buffer` (BH DRAM ceiling).

## split_reader — data movement below the ceiling: issue-bound
**Situation:** one Tensix core gathers many small NoC reads issued from NCRISC alone while BRISC only writes.
**How to recognize it:** NCRISC lifetime ≈ device time and grows with the read count at fixed bytes (128 KiB: 32 B reads 146 µs vs 2048 B reads 24 µs on WH B0; 111 µs vs 19 µs on BH).
**Helps when:** ≤ 256 B per NoC read. 32 B reads (4096 reads/launch): WH B0 147 274 → 79 968 ns = **1.84×** · BH 111 870 → 58 811 ns = **1.90×**; BH 256 B 1.70×.
**Does not help when:** no flat regime is reachable in this example on either arch — with ≥ 1 KiB reads the gain shrinks to WH B0 **1.38×** (2048 B: 25 041 → 18 182 ns; lowest seen 1.29×) · BH **1.61×** (2048 B: 19 432 → 12 070 ns; 1.60× at 1 KiB). The remaining gain is per-tile barrier latency, which batching reads (`double_buffer`) also removes.
**Gist:** split disjoint reads across NCRISC and BRISC only when the reader's lifetime scales with read count; otherwise batch reads per barrier first.
**Code:** `examples/split_reader/split_reader.py` — `_create_program_descriptor()` (`make_reader(num_halves)`, `writer_compile_args`); `kernels/gather_writer.cpp` `if constexpr (split_reader)` (host and kernel).
**See also:** `double_buffer`.

## noc_placement — data movement below the ceiling: contention
**Situation:** a line or grid of Tensix cores streams interleaved-DRAM pages, laid out as a column (`split_work_to_cores(..., row_wise=False)`, the default), with reads on NoC1, or with every Tensix core on the same NoC pairing.
**How to recognize it:** same bytes and kernel, but time changes a lot when the line moves to a row or the NoC flips; per-Tensix-core time grows with outstanding reads on the line; a full-grid read+write copy runs below the DRAM ceiling (read+write bytes ÷ device kernel duration < ~400 GB/s on BH p150b).
**Helps when:** ≥ 16 outstanding 2 KB reads on the line (8 Tensix cores × 16 per barrier): column → row WH B0 246 765 → 84 917 ns = **2.91×** · BH 97 835 → 43 723 ns = **2.24×**; reads NoC1 → NoC0 WH B0 278 002 → 85 066 ns = **3.27×** · BH 107 508 → 43 806 ns = **2.45×**; whole grid (130 Tensix cores), 32 MB copy: 208 978 → 173 195 ns = **1.21×** (BH). A quarter of one stream's writes on the read NoC (`write_split=4`) only where the full-grid DRAM write stream sits below the ceiling: 16 MB L1→DRAM 60 141 → 53 772 ns = **1.12×** (BH). Reversed pairing (reads NoC1 / writes NoC0) on only the 3 worker columns nearest DRAM (`swap_columns=3`) while a full-grid copy is below the DRAM ceiling and bandwidth-bound: 8 MB read+write 24 675 → 21 474 ns = **1.15×**, 65 Tensix cores 67 MB **1.10×**, 34 MB at 64 pages per barrier 84 134 → 78 665 ns = **1.07×**, DRAM→L1 1.09× (BH).
**Does not help when:** latency-bound line (2 Tensix cores × 1 read per barrier): column → row WH B0 **1.03×** · BH **1.01×**. On a single Tensix core the default NoC pairing is WH B0 **0.94×** · BH **1.02–1.06×**. Payload size is not the boundary (16 tiles still 1.51×, WH B0); reads in flight are. Write split at the DRAM ceiling (whole grid, 32 MB, 388 GB/s): **1.02×**; where the NoC is the bound it loses (DRAM→L1 **0.88×**, 8-core row **0.79×**); `NOC_MODE::DM_DYNAMIC_NOC` alone costs 9–27% on 1–2 Tensix cores. Column swap on a long full-grid stream at the ceiling (176 MB, 408 GB/s): 432 281 → 429 853 ns = **1.01×**; at 1 page per barrier **1.00×**; 34 MB at 16 pages per barrier 1.03×; 6 of 13 columns swapped on that stream 0.97×; every column 0.62–0.86× (BH).
**Gist:** with ≥ ~16 pages in flight, use `row_wise=True` (row/diagonal) and keep reads on NoC0 / writes on NoC1 on most of the grid. Measured exception: on a full-grid copy below the DRAM ceiling, give the 2–4 worker columns nearest a DRAM column (ask the device: `get_optimal_dram_bank_to_logical_worker_assignment`) the reversed pairing; never more than ~a third of the grid. With 1 read per barrier fix latency first. Split writes across NoCs only for a full-grid write stream below the DRAM ceiling, a quarter at most.
**Code:** `examples/noc_placement/noc_placement.py` — `_ordered_cores()` (incl. `grid`), `_dm_config()` (NoC + `NOC_MODE`), `dram_adjacent_columns()` + the two kernel groups in `_copy()` (`swap_columns`); `kernels/copy_writer.cpp` `write_split` (every k-th page on the read NoC, two-NoC barrier).
**See also:** `split_reader`, `mcast_topology`, `tile_reorder`.

## tensix_all_reduce_ring_transport — data movement below the ceiling: contention
**Situation:** a reduce-and-forward ring of neighbour L1 writes whose ring order runs against the chosen NoC's routing direction.
**How to recognize it:** the same ring payload on NoC0 vs NoC1 differs by a large factor on a line; the semaphore-only variants differ by only ~2×.
**Helps when:** ring along a line (all hops one direction), 12 KiB per hop, 1×8 × 8 groups NoC1 → NoC0: WH B0 26 373 → 4 364 ns = **6.04×** · BH 10 308 → 2 613 ns = **3.94×**.
**Does not help when:** a 2-row serpentine (equal traffic both directions), 2×8 × 4 groups: WH B0 48 920 → 47 490 ns = **1.03×** · BH 19 554 → 19 309 ns = **1.01×**.
**Gist:** on line rings keep ring order along the NoC's routing direction; on a serpentine, switching NoC does nothing — reorder or split the ring.
**Code:** `examples/tensix_all_reduce_ring_transport/program_descriptor_with_inline_kernels.py` — `create_program_descriptor()` `use_noc1` branch (host only; same `_PAYLOAD_RING_KERNEL`).
**See also:** `noc_placement`, `tensix_all_reduce`.

## bank_stagger — data movement below the ceiling: contention
**Situation:** every Tensix core walks its DRAM data in the same order and the data lines up on the banks — a width-sharded DRAM source with one shard per bank, or an interleaved source split by width (Tensix cores share rows) or by height on Blackhole (rows start at `32·k`, `32·k % 8 == 0`) — so at any moment the whole grid reads one bank; lever: start each Tensix core's walk at a different block (`stagger_blocks`) or row (`stagger_rows`) and wrap.
**How to recognize it:** map each Tensix core's first page to its DRAM bank (shard → bank for width-sharded; row `r` → bank `r % num_banks` for interleaved): every Tensix core starts on the same bank and steps through the banks in lockstep. Same transactions and bytes; only the order differs.
**Helps when:** width-sharded source, few blocks per Tensix core, 8 blocks per core, 256 B reads: BH p300c (110 Tensix cores) `3520×1024` 101 361 → 77 087 ns = **1.32×** · BH p150b (120 Tensix cores) `3840×1024` 111 677 → 84 255 ns = **1.33×**; 1024 B reads 1.16× (p300c) / 1.19× (p150b).
**Does not help when:** interleaved source with several blocks per Tensix core: `stagger_rows`, `128×56320`, 4 blocks per core 66 462 → 66 040 ns = **1.01×** (BH p300c); at 1 block per core it is only 1.04–1.07× (~0.3–1.3 µs saved; p150b 1.04–1.06×). On the width-sharded source the gain shrinks as blocks per core grow: 16 blocks 1.19× / 1.12–1.13× (256 B / 1024 B) — do not extrapolate the 8-block numbers. WH B0 (12 DRAM banks): not measured.
**Gist:** start Tensix core k at block `k % n_w` (or row `k % 32`) and wrap; both are compile-time switches, so flip them per call and keep them where they pay — mostly width-sharded sources with few blocks per core. Writes into width-sharded or bank-aligned outputs are the same pattern (README).
**Code:** `examples/bank_stagger/kernels/bs_reader.cpp` — `if constexpr (stagger_blocks)` / `if constexpr (stagger_rows)` (and `stagger_blocks` in `bs_writer.cpp`); `bank_stagger.py` `create_program_descriptor()` sets `blk_rot` / `row_rot` per Tensix core (compute kernel unchanged).
**See also:** `noc_placement`.

# Redundant reads

## shared_input_reuse — redundant reads
**Situation:** many Tensix cores read the same DRAM bytes: (1) a multi-MB input every Tensix core streams in chunks, or (2) a small window (bias, scale) every Tensix core keeps resident in L1 while streaming its own rows. Lever: one injector reads once and multicasts; (1) chunk by chunk with an ack-then-broadcast handshake (`mcast_pipe`), (2) one-shot, a semaphore set per chunk and no handshake.
**How to recognize it:** N Tensix cores read identical page ids, so DRAM bytes = N × unique bytes. (1) Check the chunk size (bytes per handshake). (2) Compute r = window bytes ÷ x bytes per Tensix core, and check whether the per-core stream is DRAM-bound (bytes ÷ time near the ceiling).
**Helps when:** (1) chunks amortize the handshake. 1216-tile stream, 64-tile chunks: WH B0 (2×8 = 16 Tensix cores) 255 592 → 191 246 ns = **1.34×** · BH p150b (2×12 = 24 Tensix cores) 150 228 → 77 777 ns = **1.93×** · BH p150b 13×10 grid (2×13 = 26) 177 524 → 78 360 ns = **2.27×** (older BH P150 1.71×, P100a 1.82×); 16-tile chunks BH 1.32× / 2.03× (WH 0.93×). (2) r ≥ 0.5, BH p150b 13×10: 130 Tensix cores, 256 KB window, 65 MiB x (r = 0.5) 437 387 → 364 948 ns = **1.20×**; r = 1 **1.51×**; 64 KB window, 16 MiB **1.21×**; 16–64 Tensix cores at r ≥ 0.5 1.06–1.35×. Gain ≈ 1 + r/2 while DRAM-bound.
**Does not help when:** (1) small chunks: 8-tile WH B0 214 339 → 262 332 ns = **0.82×** · BH 24 cores 0.96× (26 cores 1.50×); 1-tile WH 0.70× · BH 0.64×. Crossover WH ~32 tiles (64 KB) per chunk, BH between 8 and 16 (24 cores) or 2 and 4 (26 cores, slower baseline). (2) r ≤ 0.125: 130 Tensix cores, 16 KB window 92 383 → 92 609 ns = **1.00×**; 32 Tensix cores, 256 KB window, 64 MiB 0.96–1.00×; windows ≤ 16 KB on ≥ 64 Tensix cores 0.96–1.03×. A per-chunk handshake on (2) costs ≈ 1.2 µs per chunk on 130 Tensix cores: 1-tile chunks 124 342 → 141 805 ns = **0.88×**.
**Gist:** (1) multicast a shared stream only with chunks large enough to amortize the handshake; otherwise enlarge the chunks first, or keep per-core reads. (2) Broadcast a resident window once, with large chunks and no handshake, when it is ≥ half of each Tensix core's own x bytes. Waiting for the whole window before streaming is free, because the first row needs all of it; each handshake round is not.
**Code:** `examples/shared_input_reuse/shared_input_reuse.py` — `create_program_descriptor()` (`per_core_dram` vs `ttnn.Mcast` path); `examples/shared_input_reuse/resident_window.py` — `create_window_program_descriptor()`, `_READER_KERNEL` `get_window` (`WINDOW_PER_CORE` / `WINDOW_ONESHOT` / `WINDOW_LOCKSTEP`).
**See also:** `mcast_topology`.

## mcast_topology — redundant reads
**Situation:** a 2-D work split where every Tensix core in a row needs the same A slice and every Tensix core in a column the same B slice, each fetched from DRAM; lever: two 1-D multicasts (`Mcast` channels grouped `RowMajor` + `ColumnMajor`).
**How to recognize it:** count Tensix cores reading identical tiles — the line length is the DRAM-read reduction factor.
**Helps when:** a big grid with several tiles per Tensix core. 8×8 (lines of 8), Mt=8 Nt=32 Kt=4: WH B0 16 204 → 8 677 ns = **1.87×** · BH 8 698 → 4 327 ns = **2.01×**; Kt=16 WH 2.67× · BH 3.27×; Kt=64 WH 3.47× · BH 3.90×. BH full 10×12 = 120 Tensix cores, Kt=4: 14 171 → 5 849 ns = **2.42×** (Kt=16 4.18×).
**Does not help when:** lines of ≤ 4 Tensix cores: 4×4 WH B0 4 186 → 4 056 ns = **1.03×** · BH 1 791 → 2 252 ns = **0.80×**; 2×2 WH 0.73× · BH 0.65×; 1×8 WH 0.84× · BH 0.75×; 2×8 WH 0.86× · BH 0.91×.
**Gist:** use two 1-D multicasts only when the split fills a big grid with several tiles per Tensix core; on ≤ 4×4 grids per-core reads match (WH) or beat (BH) them. A 2-D *work* split needs 1-D multicasts. These numbers are matmul operands fetched per output block; a one-shot broadcast of a small resident window (a bias, a scale) is measured in `shared_input_reuse` (resident window).
**Code:** `examples/mcast_topology/mcast_topology.py` — `_PER_CORE_DRAM_READER` vs `_1D_PAIR_READER`, `create_program_descriptor()` `mcast_1d_pair` branch (two `ttnn.Mcast` channels, `RowMajor` + `ColumnMajor`; work split and writer unchanged).
**See also:** `shared_input_reuse`.

# Compute-bound

## compute_block_size — compute-bound: fixed cost per call
**Situation:** a row-parallel compute chain (tilize, tilize, add, matmul, untilize) looped one tile-row at a time.
**How to recognize it:** time vs number of passes over M: ~1.5 µs (WH B0) / ~1.4 µs (BH) per extra pass, flat across payload size.
**Helps when:** many passes, small payload each: M,K,N = 8,4,4 tiles WH B0 29 064 → 17 271 ns = **1.68×** · BH 18 623 → 9 096 ns = **2.05×**; 8,2,16 1.39× (WH B0).
**Does not help when:** few passes that each carry a large payload: 2,16,16 tiles WH B0 37 202 → 35 634 ns = **1.04×**. BH has no flat case that fits L1: 2,16,16 is 19 976 → 18 693 ns = **1.07×** (the 1.35× faster AICLK halves the payload per pass while the fixed cost stays).
**Gist:** run each helper on the whole row-parallel block (or the largest L1 allows); also drop data-format reconfig at boundaries where the format is constant (up to 1.19×, see README).
**Code:** `examples/compute_block_size/program_descriptor_with_inline_kernels.py` — `block_rows_for()`, `create_program_descriptor()` (host only; `_COMPUTE_KERNEL` unchanged, block size is a compile-time arg).
**See also:** `compute_fusion`.

## compute_fusion — compute-bound: L1 round-trips
**Situation:** one helper call per op, each packing its result to a CB that the next op unpacks.
**How to recognize it:** the phase after the producer is unpack-bound although its math is cheap; check whether the consumer is SFPU or FPU.
**Helps when:** SFPU consumer, small work per call: `sfpu_chain` 4 tiles WH B0 9 674 → 8 632 ns = **1.12×** · BH 6 386 → 5 745 ns = **1.11×**; the win is ~0.6–1 µs per call, so it shrinks with tile count (~1.05× at 64 tiles, both). FPU → FPU chain (`fpu_chain`, `(a+b)*c`) at ≤ 2 tiles per call: BH 1 tile 349 → 314 ns = **1.11×**, 2 tiles 1.05×.
**Does not help when:** FPU consumer: `fpu_sfpu` 64 tiles (dest-reuse) WH B0 77 734 → 84 105 ns = **0.92×** · BH 54 023 → 58 328 ns = **0.93×** — the DEST → SRC move costs more than the pack + unpack it replaces. `fpu_chain` from 4 tiles per call (crossover 2–3, BH): 4 tiles 731 → 759 ns = **0.96×**, 64 tiles 5 908 → 8 226 ns = **0.72×** — ~40 ns per tile of DEST → SrcA move outweighs the second pass's ~80–120 ns fixed cost.
**Gist:** keep intermediates in DEST when the next op is SFPU; for an FPU consumer pack to L1 and let the unpacker feed it back, unless a call covers only 1–2 tiles (crossover ≈ the extra pass's fixed cost ÷ ~40 ns per tile, BH).
**Code:** `examples/compute_fusion/program_descriptor_with_inline_kernels.py` — `_SFPU_CHAIN_KERNEL`, `_FPU_SFPU_KERNEL`, `_FPU_CHAIN_KERNEL` `if constexpr (method == …)` branches, `_VARIANT_SPEC` (kernel only; host adds scratch CBs); work-per-call sweep `test_compute_fusion_per_call`.
**See also:** `eltwise_l1_vs_dest_accumulate`, `sfpu_tile_scope`.

## eltwise_l1_vs_dest_accumulate — compute-bound: L1 round-trips
**Situation:** a running L1 accumulator rebuilt each step by unpacking it, adding on the FPU and packing it back (`rmw`).
**How to recognize it:** `rmw` grows ~370 ns (WH B0) / ~150 ns (BH) per step while packer L1-accumulation (`pack_l1_acc`) costs ~90 / ~27 ns per step.
**Helps when:** `rmw` → `pack_l1_acc` at every step count: 64 fp32 steps WH B0 23 784 → 5 769 ns = **4.12×** · BH 9 772 → 1 857 ns = **5.26×**; 2 steps WH 2.03× · BH 2.25×. DEST-resident `dest_acc` over `pack_l1_acc` pays sooner on BH: 64 steps WH 1.30× · BH 1 857 → 922 ns = **2.01×**; 4 steps WH 1.09× · BH 1.15×.
**Does not help when:** short loops for the DEST-resident version: 2 steps `pack_l1_acc` → `dest_acc` WH B0 367 → 370 ns = **0.99×** · BH 155 → 155 ns = **1.00×**.
**Gist:** fold pairs into the L1 accumulator with `pack_l1_acc`; reserve DEST (`dest_acc`) for loops of ≥ 4 steps when DEST is free (WH needs longer loops for it to matter). Syntax for one stream: README.
**Code:** `examples/eltwise_l1_vs_dest_accumulate/eltwise_l1_vs_dest_accumulate.py` — `_ACCUMULATE_KERNEL` `if constexpr (method == …)` branches, `_METHOD` (kernel only; host unchanged).
**See also:** `compute_fusion`, `reduce_accumulate`.

## reduce_accumulate — compute-bound: reduction
**Situation:** a SUM/mean reduce of N tiles written with the reduce library (matmul-with-ones per tile); lever: pairwise `add_tiles` accumulate + one SFPU finalize.
**How to recognize it:** library time grows ~linearly per tile; the fast path has a ~800–900 ns (WH B0) / ~450–600 ns (BH) floor plus a small slope.
**Helps when:** WH B0 ≥ 8 tiles reduced per output; BH col / scalar ≥ 8, row ≥ 16. 32 tiles: row WH 3 652 → 1 865 ns = **1.96×** · BH 1 385 → 800 ns = **1.73×**; col WH 1.72× · BH **2.97×**; scalar WH 2.93× · BH 3.73×. BH 8 tiles: col 1.35×, scalar 1.48×; BH row 16 tiles 1.24×.
**Does not help when:** WH B0 ≤ 4 tiles: row 1 tile 467 → 940 ns = **0.50×**, row 4 0.83×, col 4 0.81×, scalar 4 0.95×. BH: row ≤ 8 tiles (8: 514 → 578 ns = **0.89×**; 4: 0.69×), col / scalar 4 tiles 0.96× / 0.97×. The example's `dispatch` threshold row ≥ 4 is stale on both archs (WH 0.83×, BH 0.69× at 4).
**Gist:** accumulate + SFPU finalize only when the reduce spans enough tiles per output (WH ~8; BH 8 for col / scalar, 16 for row); below that use the library reduce.
**Code:** `examples/reduce_accumulate/program_descriptor_with_inline_kernels.py` — `_HELPER_KERNEL` vs `_FAST_KERNEL`, `_resolve()` + `_DISPATCH_MIN` for `dispatch` (separate compute kernel per variant).
**See also:** `row_reduce_accumulate`, `reduce_block`.

## row_reduce_accumulate — compute-bound: reduction
**Situation:** a row mean over W tiles written as one `reduce<SUM, REDUCE_ROW>` over the row; lever: pairwise `add_tiles` into DEST then one finalize reduce (`dest_accum_pairs`).
**How to recognize it:** the single reduce costs ~70–140 ns (WH B0) / ~35–45 ns (BH) per tile; `dest_accum_pairs` has a ~800 ns (WH) / ~400 ns (BH) floor plus a small per-tile slope.
**Helps when:** WH B0 rows ≥ 16 tiles (bf16 input) or ≥ 8 (fp32); BH ≥ 8 tiles for both. 32 tiles: fp32 WH 4 478 → 2 850 ns = **1.57×** · BH 1 386 → 784 ns = **1.77×**; bf16 WH 2 419 → 1 803 ns = **1.34×** · BH 1 369 → 694 ns = **1.97×**. BH 8 tiles 1.05–1.06×, 16 tiles 1.37–1.46×.
**Does not help when:** narrow rows: bf16 1 tile WH B0 433 → 837 ns = **0.52×** · BH 240 → 413 ns = **0.58×**; 2 tiles WH 0.60× · BH 0.70×; 4 tiles BH 0.83×; 8 tiles WH 0.87× (BH already 1.06×).
**Gist:** use `dest_accum_pairs` (parity resolved at the seed, no zero CB) for wide rows; keep the single reduce for narrow ones. bf16 accumulation, not bf16 input, costs precision (README).
**Code:** `examples/row_reduce_accumulate/program_descriptor_with_inline_kernels.py` — `_COMPUTE_KERNEL` `if constexpr (method == …)` branches, `_METHOD_ID` (kernel; host only varies CBs via `_NEEDS_INTERM` / `_NEEDS_SCALER_CB` / `_NEEDS_ZERO`).
**See also:** `reduce_accumulate`, `reduce_block`.

## reduce_block — compute-bound: reduction
**Situation:** a 2-D `(Ht, Wt, NC)` block reduced along one dim into many output tiles with the default `reduce_tile`; lever: `AccumulateViaAdd` per output tile.
**How to recognize it:** count reduced tiles per output tile (row Wt, col Ht, scalar Ht×Wt) — that, not block size, decides it.
**Helps when:** WH B0 ≥ 8 reduced tiles per output; BH col / scalar ≥ 4, row > 8. Row `(2,32,1)` WH 6 977 → 3 343 ns = **2.09×** · BH 2 565 → 1 296 ns = **1.98×**; col 32 WH 1.88× · BH 3.64×; scalar `(4,8,1)` WH 3.04× · BH 4.12×. BH col `(4,2,1)` 738 → 663 ns = **1.11×**, scalar `(2,2,1)` 1.08×.
**Does not help when:** WH B0 ≤ 4 per output: row `(4,2,1)` 1 307 → 2 569 ns = **0.51×**, row `(2,4,1)` 0.81×, col 0.90×, scalar `(2,2,1)` 1.02×. BH: row ≤ 8 (`(2,8,1)` 825 → 876 ns = **0.94×**, 0.99× on a re-run — flat; 4 tiles 0.66×; 2 tiles 0.39×), col / scalar 2 tiles 0.78× / 0.76×.
**Gist:** opt into `AccumulateViaAdd` only when each output reduces enough tiles (WH ~8; BH 4 for col / scalar, > 8 for row); one DEST per output also avoids the `REDUCE_COL` chunk limit.
**Code:** `examples/reduce_block/program_descriptor_with_inline_kernels.py` — `_HELPER_BLOCK_KERNEL` `algo` arg (`ReduceAlgorithm::Auto` vs `AccumulateViaAdd`), `_FAST_BLOCK_KERNEL`, `_resolve()` (kernel; scaler kernel shared).
**See also:** `reduce_accumulate`, `row_reduce_accumulate`.

## tensix_all_reduce_compute — compute-bound: reduction
**Situation:** a reducer that copies each contributor into DEST, re-runs `add_binary_tile_init()` and does one SFPU add per contributor.
**How to recognize it:** single-Tensix-core reducer time grows ~linearly with (blocks − 1) × tiles with low unpack/pack counts.
**Helps when:** always, growing with blocks: 8 blocks × 6 tiles WH B0 11 178 → 1 899 ns = **5.89×** · BH 6 937 → 740 ns = **9.37×**; 2 blocks WH 1.90× · BH 3.91×, 4 blocks WH 3.01× · BH 6.74×.
**Does not help when:** no slower case exists in the measured range (≥ 2 blocks); the smallest gain is 2 blocks × 1 tile, one call: WH B0 783 → 568 ns = **1.38×** · BH 509 → 392 ns = **1.30×**. FP32 vs BF16 DEST on the fast path: WH 1.03× · BH 0.94× (fp32 DEST slightly slower on BH).
**Gist:** pairwise FPU `add_tiles(..., acc_to_dest=true)`, one init per DEST batch, pack once.
**Code:** `examples/tensix_all_reduce_compute/program_descriptor_with_inline_kernels.py` — `_SFPU_SERIAL_KERNEL` vs `_FPU_DEST_REUSE_KERNEL`, chosen in `create_program_descriptor()` (compute kernel only).
**See also:** `eltwise_l1_vs_dest_accumulate`, `tensix_all_reduce`.

## sfpu_tile_scope — compute-bound: wasted SFPU work
**Situation:** `rsqrt`/`recip` run on all 32 vector ops of a tile whose meaningful result lives on one row or column after a reduction — a REDUCE_ROW result read back through `BroadcastDim::Col` uses column 0 only.
**How to recognize it:** MATH-thread zone ns per SFPU call is flat per vector op (rsqrt ~35.6 ns WH B0 / ~25 ns BH; recip ~17.6 / ~6.5 ns), so cost tracks the vector count.
**Helps when:** SFPU math dominates. Row-0 result: `VectorMode::R` + `ITERATIONS=2` (4 vectors) rsqrt WH B0 1 139 → 156 ns = **7.30×** · BH 795.6 → 109.0 ns = **7.30×**; recip WH 6.69× · BH 5.46× (math zone); whole op on 1 tile WH 2.22× / 1.61× · BH 2.27× / 1.38×. Column-0 result: `c_skip` (`VectorMode::C` + even-parity stride `dst_reg += 2`, 8 vectors) rsqrt BH 748.2 → 194.9 ns = **3.84×** vs all 32 vectors, **1.94×** vs `VectorMode::C` (16 vectors, same body).
**Does not help when:** swapping scopes with the same vector count: `VectorMode::C` → `R` (16 vectors) WH B0 576 → 576 ns = **1.00×** · BH 402 → 402 ns = **1.00×** — this compares equal vector counts only; it says nothing against `c_skip` (8 vectors) or `r_iter2` (4).
**Gist:** scope the SFPU to the minimum vectors covering the result. Row 0: `ITERATIONS=2`. Column 0: one raw-sfpi pass over the 8 even-parity vectors of faces 0/2 (`dst_reg += 2` inside a `VectorMode::C` body). `rsqrt_tile` hardcodes its vector mode and exposes neither `ITERATIONS` nor a parity stride, so `r_iter2` / `c_skip` need the underlying calls (README).
**Code:** `examples/sfpu_tile_scope/program_descriptor_with_inline_kernels.py` — `_SCOPE` (`VectorMode`, `ITERATIONS` per variant), `_KERNEL` `kernel_main()` / `cskip()` (one kernel; compile-time args differ).
**See also:** `reduce_accumulate`.

## matmul_output_subblock — compute-bound: SRC operand reload
**Situation:** `matmul_block` with a 1×1 output subblock, so every output tile reloads both operands into SRC.
**How to recognize it:** sweeping the subblock shape at fixed Mt/Nt/Kt drops time 1.5–1.8× (WH B0) / ~1.46× (BH, Kt = 1 only) at 8-tile subblocks.
**Helps when:** many output tiles per call, and on BH short K. 16×16 output tiles, Kt=1, 1×1 → 2×4: WH B0 24 312 → 13 905 ns = **1.75×** · BH 10 429 → 7 161 ns = **1.46×**. WH holds 1.66–1.69× at Kt=2–12; BH drops to 1.17× at Kt=2 and **1.10×** at Kt=4 (29 238 → 26 654 ns). Tall 8×1 → wide 1×8: WH 1.40× · BH **1.00×** (equivalent on BH). BH 8×8 output tiles 1.34×.
**Does not help when:** only a handful of output tiles: 2×2 output tiles, Kt=1, 1×1 → 2×2 WH B0 670 → 656 ns = **1.02×** · BH 372 → 382 ns = **0.97×**; 2×2 at Kt=8 WH still 1.34× · BH 0.98×.
**Gist:** use the largest subblock that fits DEST (2×4, 4×2, 1×8 — avoid a tall 8×1 on WH; on BH orientation does not matter); with fp32 DEST the budget halves to 4 tiles. On BH expect ~1.1–1.2× once Kt ≥ 2, not the WH 1.7×.
**Code:** `examples/matmul_output_subblock/matmul_output_subblock.py` — `SUBBLOCK`, `create_program_descriptor()` (host only; `_MATMUL_KERNEL` unchanged, subblock shape is a compile-time arg).
**See also:** `tile_reorder`.

# Cross-core communication

## tensix_all_reduce — cross-core communication
**Situation:** Tensix cores in an L1-sharded group all need the group sum, and one root Tensix core gathers and adds every contributor serially, then multicasts (`reduce_root_mcast`).
**How to recognize it:** the root alone is busy for (group size − 1) × tiles-per-core adds while the others idle.
**Helps when:** groups ≥ 8 Tensix cores and enough tiles per core: `reduce_scatter_mcast` on 1×8 × 8 groups, 6 tiles WH B0 6 145 → 3 646 ns = **1.68×** · BH 2 592 → 1 817 ns = **1.43×**. BH 1-D crossover 2–4 tiles per core; 16 tiles 1.61×. Large payloads on groups of ≥ 2 rows: 4×4 32 tiles 19 897 → 4 930 ns = **4.04×**, 5×6 16 tiles **4.66×** (BH, isolated groups; 1.07–1.56× packed across the grid); lines of ≥ 10 Tensix cores need the push form (`reduce_scatter_push`): 1×13 32 tiles 17 272 → 12 771 ns = **1.35×**. Signal after `noc_async_writes_flushed()` instead of a write barrier (`*_flush` variants), 1–6 tiles per core: push 1 493 → 1 260 ns = **1.19×**, pull 1.03–1.08× (BH, 4–26 members).
**Does not help when:** 4-core groups WH B0 3 976 → 4 663–5 120 ns = **0.78–0.85×** · BH 2 086 → 3 812 ns = **0.55×**; 1 tile per core WH 1 614 → 1 994 ns = **0.81×** · BH 990 → 1 326 ns = **0.75×**; BH 6-core half-row groups: tree ≈ root, reduce-scatter 0.95× — keep root reduction there. Pull form on 1-D lines with large payloads (the "gone by 32 tiles" case): 1×13 16 tiles 8 916 → 10 821 ns = **0.82×**, 1×8 32–48 tiles packed 0.95–0.97×. Root reduction stops fitting past (G + 2) · tiles ≈ 670 tiles of L1 per Tensix core. Flush lever: null by 32 tiles per core (0.97–1.05×).
**Gist:** reduce-scatter for well-fed groups of 8+; root (or BH tree) reduction for small groups or tiny payloads; on lines past ~16 tiles per core use the push form; reduce-scatter also needs far less L1 per Tensix core (README). When a data write and its semaphore increment share NoC, VC and destination, flush rather than barrier before the increment (~0.1 µs per signal) and keep one barrier at kernel exit.
**Code:** `examples/tensix_all_reduce/program_descriptor_with_inline_kernels.py` — `_create_reduce_root_descriptor()` + `_REDUCE_ROOT_KERNEL` vs `_create_reduce_scatter_descriptor()` + `_REDUCE_SCATTER_KERNEL` (pull) / `_create_reduce_scatter_push_descriptor()` + `_REDUCE_SCATTER_PUSH_KERNEL`; flush lever `_SIGNAL_FENCE` (`fence_before_signal()`, `SIGNAL_AFTER_FLUSH` via `_signal_defines()`) in the `*_flush` variants (host and dataflow kernels).
**See also:** `tensix_all_reduce_compute`, `tensix_all_reduce_ring_transport`, `tensix_all_reduce_wave_pipeline`.

## tensix_all_reduce_wave_pipeline — cross-core communication
**Situation:** an L1-sharded group runs a reduce-scatter exchange between two compute steps (each member sends its row partials to the row's owner; the owner sums them and all-gathers the totals) once per block, so exchange and compute run back to back; lever: split the block's tile-rows into waves, so wave k's partials travel while wave k+1 is computed and wave k is applied as soon as its totals land.
**How to recognize it:** compute and exchange take about the same time: stubbing either alone barely moves the kernel time while stubbing both collapses it, and the compute step after the exchange waits on its result.
**Helps when:** each wave keeps ≥ 16 tiles of compute per Tensix core. 8×8 groups of 8 Tensix cores, 32 tile-rows × 4 tiles per Tensix core: 1 wave 22 007 → 4 waves 16 290 ns = **1.35×** (2 waves 1.23×, 8 waves 1.34×) · 8 tile-rows × 16 tiles: 13 897 → 12 031 ns = **1.16×** at 4 waves (BH p150b).
**Does not help when:** waves of ≤ 2 tiles per Tensix core: 8 tile-rows × 1 tile, 4 waves 5 484 → 5 972 ns = **0.92×**, 8 waves (1 tile) **0.65×** — each wave adds a semaphore round and too little compute to hide it. One tile-row per wave funnels the wave through one owner: 8 × 16 at 8 waves **1.03×** (vs 1.16× at 4). WH B0: not measured.
**Gist:** pipeline the exchange in ~4 waves per block, keeping ≥ 16 tiles and ≥ 2 tile-rows per wave per Tensix core; give each wave its own semaphore pair (one counter cannot tell wave k's arrivals from a faster member's wave k+1); drive the four exchange steps from a non-blocking poll loop, since any blocking wait serializes the waves again; order compute as all partials, then owned sums, then the apply step wave by wave. L1 cost: the one-shot's pages plus one semaphore pair per wave.
**Code:** `examples/tensix_all_reduce_wave_pipeline/program_descriptor_with_inline_kernels.py` — `_EXCHANGE_KERNEL` poll loop (`try_send` / `try_push_gather` / `try_allgather` / `try_push_result`), `_COMPUTE_KERNEL` wave order, `wave_plan()` (one kernel for both; `one_shot` = 1 wave).
**See also:** `tensix_all_reduce`, `tensix_all_reduce_ring_transport`.

## resident_shard_locality — cross-core communication
**Situation:** an L1-sharded ROW_MAJOR input (HEIGHT or BLOCK, bf16 or fp32) that the op tilizes → computes → untilizes into an L1-sharded output, with the work dealt out like an interleaved tensor (tile-rows of 32 sticks over the Tensix cores, each stick fetched from its owner through a sharded `TensorAccessor`); lever: make the work split the shard spec — each Tensix core copies its own sticks locally into a staging CB, one ≤ 32-tile column chunk at a time, padding a partial last tile-row and a partial last tile-column locally — and when the shard is whole tile-rows ≤ 32 tiles wide, alias the input/output CBs onto the shards.
**How to recognize it:** stub the reader's NoC reads — time collapses (60 → 24 µs, BH); the same op on 1 vs all Tensix cores with the same per-core shard grows ~11× instead of staying flat; a local path gated on shard height % 32, shard width % 32, or "the whole row fits in L1" (each gate sends real shards back to the gather).
**Helps when:** ≥ 3 sticks per Tensix core at any stick width, or any stick count at ≤ 2 KB sticks — for any shard height AND width. BH p150b, 13×10 (130 Tensix cores): HEIGHT [72, 544] 59 221 → 5 498 ns = **10.77×**; wide rows chunked into ≤ 32-tile column chunks: [24, 4064] (127 tiles per stick) 81 129 → 10 222 ns = **7.94×**, [17, 5120] (160 tiles) 132 890 → 11 976 ns = **11.10×**, using 384 KB of CBs per Tensix core where whole-row staging needs 1 524 / 1 920 KB. Partial-width BLOCK on the 10×13 grid (K = 13): [311, 320] **5.47×**, [221, 400] 55 783 → 9 850 ns = **5.66×**, [23, 848] **6.28×**, [16, 848] **4.25×**, fp32 [13, 632] 11 939 → 5 596 ns = **2.13×**. Grows with Tensix cores: 1.03× (1), 1.29× (2), 6.6× (13), 10.9× (130). Aliasing on top, whole-tile shards: [32, 1024] 4 247 → 2 720 ns = **1.56×** over local.
**Does not help when:** ≤ 2 sticks per Tensix core of ≥ 8 KB sticks — the boundary is sticks per Tensix core, not tiles per core or row width: local tilizes a whole tile-row (~68 ns per tile, BH) for 1–2 valid sticks while the re-split packs full tile-rows on fewer Tensix cores: [1, 8192] on 128 Tensix cores 12 074 → 17 423 ns = **0.69×**; [2, 8192] 0.92–0.95×, [1, 4096] 0.68×, [2, 4096] 0.98×; at 3 sticks local already wins (1.37–1.74×). WH B0: not measured.
**Gist:** the work split is the shard spec for any shard height and width; pad the last tile-row and tile-column locally. Chunk W into balanced ≤ 32-tile column chunks (chunks = ⌈W tiles / 32⌉, chunk = ⌈W tiles / chunks⌉; not "widest divisor ≤ 32", which turns 127 tiles into 1-tile chunks: 68.6 vs 10.2 µs) and stage only the current chunk, 2 deep — a local path that does not fit L1 means W is not chunked, never a reason to fall back to the re-split. A row reduction that needs two passes keeps only the tilized row resident (or re-reads the shard for the second pass) and stages the rest per chunk. Re-split only for ≤ 2 sticks of ≥ 8 KB per Tensix core; if it must stay, put the read NoC along the owner direction (1 grid row: 6.6× → 1.3×).
**Code:** `examples/resident_shard_locality/program_descriptor_with_inline_kernels.py` — `_LOCAL_READER` / `_LOCAL_WRITER` vs `_RESPLIT_READER` / `_RESPLIT_WRITER`, `chunk_plan()` / `local_l1_bytes()`, `create_program_descriptor()` (`resplit` unit deal-out; `local_alias` branch: `cb_descriptor_from_sharded_tensor` with tile-sized pages); `_COMPUTE` shared.
**See also:** `noc_placement` (NoC direction and contention), `tile_reorder` (L1-sharded destinations).
