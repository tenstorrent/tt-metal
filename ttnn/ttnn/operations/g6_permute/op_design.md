# Operation Design: g6_permute

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement — no compute kernel, no arithmetic, no compute_kernel_config |
| Goal | Reorder the dims of one device tensor as `torch.permute(x, dims)` in one device program and one pass over DRAM, output in the input's dtype, layout and placement. Phase 0 builds regime `rows_tile`: TILE bfloat16 DRAM-interleaved rank 4, dims (0, 2, 1, 3), the class of 2176 of the 2203 traced calls. |
| Math | `out[o_0..o_{r-1}] = x[i_0..i_{r-1}]` with `i_{dims[k]} = o_k`. Phase 0: `out[n, b, a, w] = x[n, a, b, w]` (rows of W move whole). |
| References | `eval/prompts/g6_permute.txt` (contract); `docs/explorations/g6_permute/{DOSSIER,math,tensix,risks}.md`; `eval/golden_tests/g6_permute/{feature_spec,helpers}.py`; `.claude/references/{blocking-model,l1-footprint-discipline,noc-alignment-in-tt-metal,ttnn-cb-memory-fundamentals,device-zone-scope-attribution}.md`; catalog `ttnn/ttnn/operations/examples/master.md` (entries in "Catalog entries used"); `l1_ledger.md` (this directory) |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| input_tensor | ttnn.Tensor | yes | on device, INTERLEAVED, rank 4 or 6 (Phase 0: rank 4 TILE bfloat16 DRAM) | — | buffer address: common RT arg of reader |
| dims | sequence of r ints, positional | yes | r entries in [-r, r-1]; after wrapping negative entries once (d -> d + r) a permutation of range(r) | — | canonical dims select the regime and the index map; part of the program cache key; never an RT arg |

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | Phase 0: `[N, A, B, W]` any N, A >= 1, B not a multiple of 32 ("h_non_aligned"), W a multiple of 32. Production: N = 1, W = 64, (A, B) in (1,16), (16,1), (6,16), (16,6), (1,8), (8,1), (6,8), (8,6) |
| Dtype | bfloat16 (Phase 0) |
| Layout | TILE (32 x 32 tiles of four 16 x 16 faces; each (n, a) image tile-padded independently: `tensor_b_tiles = ceil(B/32)` tile rows, `tensor_w_tiles = ceil(W/32)` tile columns) |
| Memory | INTERLEAVED, DRAM (Phase 0) |
| Padding | Rows `b >= B` of the last tile row of each image hold anything (garbage from a model producer). They are never read (see `load_rows_block`). |

### Output

| Property | Value |
|----------|-------|
| Shape | `[x.shape[d] for d in dims]`; Phase 0: `[N, B, A, W]`, padded per (n, b) image to `tensor_a_tiles = ceil(A/32)` tile rows |
| Dtype | input dtype (bfloat16) |
| Layout | input layout (TILE) |
| Memory | input memory config (DRAM INTERLEAVED), allocated by the entry point with `ttnn.allocate_tensor_on_device` |
| Padding | Rows `a >= A` of each output tile hold whatever the staging page held (unspecified, decision D7 of the contract). Never graded. |

## Blocking Model

Semantics: `.claude/references/blocking-model.md`. Everything below this section realizes these decisions.

### Index map of `rows_tile` (Phase 0)

| Quantity | Definition |
|----------|------------|
| tile byte offset of row `r` (0..31), column half `h` (0, 1) | `off(r, h) = (r div 16) * 1024 + h * 512 + (r mod 16) * 32`; a face-row segment is 32 B (16 bfloat16 values) |
| input page id of tile `(n, a, tb, tw)` | `((n * A + a) * tensor_b_tiles + tb) * tensor_w_tiles + tw` |
| output page id of tile `(n, b, ta, tw)` | `((n * B + b) * tensor_a_tiles + ta) * tensor_w_tiles + tw` |
| move | output tile `(n, b, ta, tw)` segment at `off(a mod 32, h)` <- input tile `(n, a, b div 32, tw)` segment at `off(b mod 32, h)`, for every live `a` in `[32*ta, min(32*ta + 32, A))` and `h` in {0, 1} |
| work unit `u` (one output tile), b fastest | `u = ((n * tensor_a_tiles + ta) * tensor_w_tiles + tw) * B + b`, `u` in `[0, total_units)`, `total_units = N * tensor_a_tiles * tensor_w_tiles * B` |
| block rule (host count and both kernels apply the same rule) | a block starts at the core's next unit `(n, ta, tw, b0)` and takes the following units while they stay in the core's range, keep `(n, ta, tw)`, keep `b div 32`, and number at most `block_b_rows`. Its actual extent is `block_b_count` (<= `block_b_rows`); its input rows are `[b0 mod 32, b0 mod 32 + block_b_count)` of the input tiles `(n, a, b0 div 32, tw)` |
| `a_live(ta)` | `min(32, A - 32*ta)` (live rows of output tile row `ta`, live input tiles per block) |

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| n (N, batch) | independent — copied index, `out[n] = x[n]` | `block_n` | 1 image. Two images share no input and no output tile, so batching them moves no transaction into another; the per-block fixed cost is already amortized along b (up to 32 output tiles per block) | host constant in the plan (`block_n = 1`), implied by the block rule's "keep n" | outermost digit of the unit order; units are split in contiguous balanced ranges over the grid | knob-turn |
| a (A, input dim 1 / output rows) | independent — no result spans a; each output row `(n, b, a)` is one input row | `block_a_rows` | the whole a-tile: `a_live(ta)` rows (<= 32), i.e. every live row of the output tile. Pinned by the mechanism cap "an output page is written whole by exactly one core" | derived: `a_tile_rows = min(32, A)` (host, CT arg) is the nominal slot count; `a_live(ta)` per block in-kernel | across a-tiles (`ta`) through the unit order; within an a-tile not split (deferred regime `rows_tile_split_a` splits it) | scheme-change (`rows_tile_split_a`: partial output-page writes) |
| b (B, input rows / output tile index) | independent — no result spans b; each output tile is one b | `block_b_rows` | `min(32, B, units_per_core_g1)`: the coarsest run that fits the core's whole assignment, capped at one b-tile. Qwen: 1 (each core owns one output tile, see Work Distribution) | host plan, CT arg; sizes `cb_output_staging` | fastest digit of the unit order: consecutive b of one `(n, ta, tw)` land on one core, so a core's b-run becomes one coalesced row-span read per input tile | knob-turn |
| w (W in tile columns, kept last dim) | independent — kept dim; a tile column moves with its rows | `block_w_tiles` | 1 tile column. Two tile columns share no transaction (different pages, different banks); the per-block fixed cost is amortized along b instead, where coarsening also coalesces reads | host constant in the plan (`block_w_tiles = 1`), implied by the block rule's "keep tw" | middle digit of the unit order | knob-turn |
| stage data: input row slots (a x b-run x 1 tw, per block) | independent along every axis it carries | slot count `a_tile_rows`, rows per slot `block_b_count` | as above | as above | owned by the core that owns the block; nothing is collected on a designated core | — |
| stage data: output staging (b-run x a rows x 1 tw, per block) | independent | `block_b_rows` pages | as above | as above | owned by the block's core; no coordinator role exists in this scheme | — |

No dependent axis exists (no reduction, scan or combine), so no split needs a combine. No reuse-shared axis exists and no split creates one:

| Operand-reuse check (operand x chosen split) | Varies along the split axis? | Verdict |
|---|---|---|
| x vs b-split (units b-fastest) | yes: a core's b-run selects rows `[b0 mod 32, ...)` of the input tiles, and two cores never read the same row bytes | not reuse-shared; no broadcast |
| x vs `(n, ta, tw)` digits | yes: each digit selects different input tiles | not reuse-shared |

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_input_rows` | `depth_in` (blocks in flight) | `min(2, max_blocks_per_core)`: 1 for every Qwen shape (one block per core), 2 when some core has two or more blocks | the reader loads block k+1 while the writer exchanges block k. A core with one block can never fill a second slot set |
| `cb_output_staging` | `depth_out` | 1 | the writer refills staging only after `noc_async_writes_flushed()` has drained the previous block's tile writes out of L1; with one block per core (every Qwen shape) there is no refill |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Blackhole DRAM read alignment: local and NoC address equal modulo 64 (`tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:379`) | input slot layout: a slot is a whole tile page and every row range lands at its own intra-tile offset. CB base is aligned to the DRAM alignment (`tt_metal/impl/program/program.cpp:2505-2509`), page 2048 B, so the slot base is 0 mod 64 | `dst = slot_base + off(...)` = the same offset as `src = page_addr + off(...)`; never compact rows | hang under watcher, silently shifted values without it |
| DRAM write / L1 alignment modulo 16 (`noc_parameters.h:375-380`) | exchange copy unit = one 32-B face-row segment at a multiple of 32 on both sides | none needed | — |
| one b-tile per block | `block_b_rows <= 32`, and a run never crosses `b div 32` | block rule clause "keep `b div 32`" | rows `b mod 32` of the wrong input tile, i.e. wrong values |
| whole output page per core | `block_a_rows` = all `a_live(ta)` rows of the a-tile | unit = whole output tile | two cores write one page, or rows of a page are missing |
| NoC packet size `NOC_MAX_BURST_SIZE` = 256 words x 64 B = 16384 B (`noc_parameters.h:287-290`) | single-packet transfers: tile write 2048 B, row-range reads <= 2048 B, copies 32 B; `set_read_state<32>` static_asserts this (`ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.inl:66`) | none needed | — |
| CB ring wrap (`ttnn-cb-memory-fundamentals.md` "CB Ring-Wrap Invariant") | push and pop quantum of `cb_input_rows` | always the nominal `a_tile_rows` pages per block, also when `a_live(ta) < a_tile_rows` (ragged last a-tile) | pointer walks past `fifo_limit` into neighbouring L1 on the second lap |
| profiler marker budget, 125 zone executions per RISC (`device-zone-scope-attribution.md` §7) | zones per block | per-step zones on the first 16 blocks of a core, one tail zone per role after that | zones after the budget silently vanish |

### Regimes

Minimum on the named boundary (DRAM): each input row crosses DRAM once and each output page crosses DRAM once.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `rows_tile` | **built** (Phase 0) | `layout == TILE and dtype == bfloat16 and rank == 4 and dims == (0, 2, 1, 3)`; Phase 0 SUPPORTED narrows it to buffer "dram" and alignment "h_non_aligned". The regime itself handles any alignment, because W padding columns travel with their rows into output padding columns | `block_n = 1` x `block_a_rows = a_live(ta)` (whole a-tile) x `block_b_count <= block_b_rows` (one b-tile) x `block_w_tiles = 1` | x: minimum, and below the page-granular figure: only live rows cross (padding rows never read). out: minimum at page granularity: each output tile written once, whole, so `(32 - a_live)/32` of its bytes are padding (structural for one write per page). L1-local: every live row is copied once more inside L1 as 2 segments of 32 B (structural: a tile-layout row transpose keeps no unit larger than a face row contiguous on both sides) | Per block: reader 1 `cb_reserve_back` + 1 read barrier + 1 `cb_push_back`; writer 1 `cb_wait_front` + 1 `writes_flushed` + 1 copy barrier + 1 `cb_pop_front`. Regime term: input read transactions per input tile fall from 2 per row (run of 1) to <= 4 per run, and to 1 x 1024 B or 1 x 2048 B when the run covers a full 16-row band or the full tile. Intended frequency: handshakes and barriers once per block; `set_read_state` once per writer kernel; write barrier once per writer kernel |
| `rows_tile_scatter` | **deferred**. Crossings are equal (1 per tensor). It saves only output padding bytes (303: 64 KB -> 12 KB written) at the price of `2*a_live` DRAM write transactions per output page instead of 1 (303: 32 -> 384), which is the sub-tile-write shape the `tile_reorder` entry measured slower than whole-page writes on a DRAM move. `rows_tile` serves every shape it would serve. Reachable: same reader, same unit split; the writer swaps exchange + whole-tile write for 32-B writes straight from the input slots (DRAM writes are mod-16 legal at any 32-B offset) and `cb_output_staging` disappears | same as `rows_tile` | out: live bytes only, `2 * N*A*B*tensor_w_tiles` writes of 32 B; no L1-local copies | same handshake terms; no exchange barrier |
| `rows_tile_split_a` | **deferred**. Traffic is no lower: same crossings, and DRAM write transactions rise from 1 to <= 4 per (output page, a-chunk). It buys occupancy only where `total_units < grid cores` (302 and 306: 2 output tiles, 16 and 8 input tiles), and `rows_tile` covers those shapes. Reachable: the unit becomes (output tile x a-chunk) with an `a_chunk_rows` knob, the writer writes row spans `[a0 mod 32, a1 mod 32)` (mod-16 legal), the reader and exchange are unchanged | `N*tensor_a_tiles*tensor_w_tiles*B < grid cores` | a-chunk x b-run rectangle | x minimum; out: each page written in pieces, live rows only | same as `rows_tile` |
| `rows_tile_reread` | **rejected** (dead end): each core reads whole input tiles for its b-run, so x crosses DRAM `b_live / block_b_count` times (303: 16x). Superseded everywhere by `rows_tile`'s row-span reads, which fill the grid with one crossing | same predicate | b-run x whole input tiles | x read up to 32x | — |
| `rows_tile_engine` | **rejected**: unpack -> DEST (`transpose_tile`, `reshuffle_rows_tile`) -> pack routes bfloat16 through the engine, which the contract forbids (bit identity incl. denormals and NaN, rule 1; risk R3). The cross-tile row move is not a DEST operation either: `reshuffle_rows_tile` moves rows inside one tile (`tt_metal/hw/inc/api/compute/reshuffle.h:31`). Superseded by NoC copies, which are bit-exact | same predicate | — | — | — |
| `rows_tile_bf8` | **deferred**: no production case (TARGET cartesian only). Outside Phase 0 SUPPORTED. Rows move whole, so the 16-value groups stay intact and the result is exact. Mantissa rows are 16-B atoms (NoC-movable); each group exponent is a 1-B atom below the 16-B NoC alignment, so exponent bytes need RISC-V byte moves. Reachable: `rows_tile` reader, unit split and staging; the exchange adds the exponent-byte moves | `TILE and bfloat8_b and rank 4 and dims (0,2,1,3)` | as `rows_tile` (1088-B pages) | as `rows_tile` | as `rows_tile` |
| `sticks_rm` | **deferred**: no production case; outside Phase 0 SUPPORTED. Each output page is one input page (rows move whole): page-to-page relocation, no exchange | `ROW_MAJOR and bfloat16 and rank 4 and dims (0,2,1,3)` | b-run of sticks | minimum, whole pages | handshakes |
| `blocks_rm` (FLUX p1, p2) | **deferred**: refinement (ROW_MAJOR, rank 6, w_non_aligned outside Phase 0 SUPPORTED). Topology: per (n, c, h-run): read the 64 input pages (w, i) of 4 B (2 values) for each h; build the 4 output rows (i, j) of 32 values per h with 2-B element moves on a RISC-V (below the 16-B NoC alignment, F2 of `math.md`); write whole 64-B output pages. Reachable: same backbone (plan, unit split, reader -> `cb_input_*` -> writer-side exchange -> staging -> whole-page writes) with its own kernel pair (BB-18) | `ROW_MAJOR and bfloat16 and rank 6 and dims (0,1,4,2,5,3)` | `(n, c, h-run)` with an `h` knob | x: every 4-B page sits in its own 64-B DRAM slot (`aligned_page_size` 64), so the DRAM-side floor is one 64-B burst per input page (`C*H*64` bursts; p1 32 MiB stored for 2 MiB logical), unless bank-major bulk reads trade the 16x bytes for fewer transactions. out: minimum, whole 64-B pages | per-block handshakes; with bank-major reads a longer h-run covers more pages per transaction |
| `nhwc_to_nchw_rm` (Panoptic p4) | **deferred**: refinement (ROW_MAJOR, l1, tile_aligned, dims (0,3,1,2) outside Phase 0 SUPPORTED). Topology: block = one h (256 input pages of 64 B = 16 KB) -> 32 whole output pages of 512 B; 32x32 element transposes by RISC-V (2-B atoms, F3 of `math.md`). Output pages in L1 count against 1347776 B per core | `ROW_MAJOR and bfloat16 and rank 4 and dims (0,3,1,2)` | `(n, h-run)` | minimum both sides (whole pages) | handshakes, whole-page writes |
| `nhwc_to_nchw_tile_bf8` (Panoptic p3) | **deferred**: refinement (bfloat8_b, l1, dims (0,3,1,2) outside Phase 0 SUPPORTED). Topology: exchange (hb, wb) of 32 input tiles <-> 32 output tiles. Per input tile a tile transpose on the compute engine regroups each output row's 16-value groups and the packer sets each group's shared exponent to its largest (meets the one-step gate and the exponent rule; bit identity is not required for bfloat8_b), followed by the `rows_tile` row exchange (rows of the transposed tile = output rows), whose mantissa rows move as 16-B atoms and exponents as 1-B RISC-V moves. Reachable: reuses the `rows_tile` exchange as its second half | `TILE and bfloat8_b and rank 4 and dims (0,3,1,2)` | exchange (hb, wb) | minimum both sides; one engine pass in L1 | handshakes; engine init once per block |

Regime selection function (host, pinned): `regime = rows_tile if layout == TILE and dtype == bfloat16 and rank == 4 and canonical_dims == (0, 2, 1, 3)`. Every other cell is refused by `validate()` in Phase 0, so no other row is reachable at run time. Regime-pinned test: `test_plan_reports_rows_tile_multicore` in the acceptance test.

### Traffic ranking

Rows-cells `R = N * A * B * tensor_w_tiles` (each 64 B = 2 face-row segments). Example 303 `(1, 6, 16, 64)`: 12 input tiles (24 KiB stored, 12 KiB live), 32 output tiles (64 KiB, 12 KiB live), R = 192.

| Candidate split | x DRAM crossings | out DRAM crossings | DRAM transactions (303) | L1-local | Cross-core | Max cores (303) |
|-----------------|------------------|--------------------|-------------------------|----------|------------|-----------------|
| S1 exchange per core: block = whole b-tile x whole a-tile (whole-band reads, whole-tile writes) | 1 (live bands) | 1 (whole tiles) | 12 reads of 1 KiB + 32 writes = 44 | 2R copies | none | `N*At*Bt*Wt` = 2 |
| **S2 b-split, units b-fastest (`rows_tile`, built)** | 1 (live rows) | 1 (whole tiles) | 384 reads of 32 B + 32 writes = 416, spread over 32 cores (13 per core) | 2R copies (12 per core) | none | `N*At*Wt*B` = 32 |
| S3 a-split (`rows_tile_split_a`) | 1 | 1 (row spans) | 12 reads + 384 writes of 32 B | 0 to 2R | none | up to R |
| S4 scatter (`rows_tile_scatter`) | 1 | 1 (live segments) | 384 reads + 384 writes of 32 B | 0 | none | 32 |
| S5 re-read (`rows_tile_reread`) | b_live / run (16) | 1 | 192 reads of 2 KiB + 32 writes | 2R | none | 32 |
| S6 dependent-axis split + combine | n/a: no dependent axis | | | | | |
| Broadcast of a reuse-shared operand | n/a: no operand is shared across any split | | | | | |

Ranking. On crossings, S1 = S2 = S3 = S4 (one per tensor) < S5 (rejected). The tie is broken by transfer shape: one whole-page write per output tile (the `tile_reorder` entry measured whole-page writes faster than sub-tile writes on a DRAM move) puts S1 and S2 ahead of S3 and S4. Occupancy then decides between S1 and S2 (spread first): S1 is S2 at `block_b_rows = b_live`, which the plan reaches automatically whenever a core's unit range covers a whole b-tile run. **Chosen: S2 (`rows_tile`).**

### Block schedule

```cpp
// per core: units [core_start_unit, core_start_unit + core_num_units), cut into blocks by the block rule
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_rows_block(block_idx);      // reader:  input rows of the block -> cb_input_rows
    exchange_rows_block(block_idx);  // writer:  cb_input_rows -> cb_output_staging (32-B segments)
    store_tiles_block(block_idx);    // writer:  cb_output_staging -> DRAM, whole tiles
}
```

| Block operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------------|-------------|--------------------|-----------------------------------|
| `load_rows_block` (reader, NCRISC, NoC0) | `a_live(ta)` input tiles x rows `[b0 mod 32, b0 mod 32 + block_b_count)` x 1 tile column | `depth_in` slot sets of `a_tile_rows` tile pages; nothing else | per block: 1 `cb_reserve_back(a_tile_rows)`, then <= 4 reads per live input tile issued back to back (one per contiguous byte range of the row run: two face-pair ranges per 16-row band, merged into one 1024-B range for a full band and one 2048-B range for the full tile), 1 read barrier, 1 `cb_push_back(a_tile_rows)`. The a-walk starts at `b0 mod a_live(ta)` and wraps (bank stagger). TensorAccessor built once per kernel |
| `exchange_rows_block` (writer, BRISC, NoC1) | `block_b_count` output tiles x `a_live(ta)` rows x 2 halves = `2 * a_live * block_b_count` copies of 32 B, each from slot `a - 32*ta` at `off(b mod 32, h)` to staging page `b - b0` at `off(a mod 32, h)` | the block's input slots (until the pop) and its staging pages | `set_read_state<32>` once per kernel (src coordinates and size fixed). Per block: 1 `cb_wait_front(a_tile_rows)`, 1 `noc_async_writes_flushed()` (blocks >= 2 only), the copies via `read_with_state` (3 register writes each, `noc_nonblocking_api.h:1498-1513`), 1 read barrier, 1 `cb_pop_front(a_tile_rows)` |
| `store_tiles_block` (writer) | `block_b_count` whole output tiles (2048 B each) | staging until flushed | per block: `block_b_count` single-packet writes, no barrier. Staging reuse is guarded by the next block's `writes_flushed`; 1 `noc_async_write_barrier()` per kernel at exit (BB-03) |

Stall-shadow check. The waits are: reader at its read barrier (DRAM latency); writer at `cb_wait_front` (on the reader), at the copy barrier (local NoC latency), and at the final write barrier.
- Single-block core (every Qwen shape): every later stage needs the awaited data. The copies need the rows, the writes need the copies. Nothing independent of the wait exists beyond page-id arithmetic, which is negligible.
- Multi-block core: the reader's load of block k+1 runs in the writer's block-k window (`depth_in = 2`).
- Algebraic permission: a permutation computes nothing, so moves commute and read, copy and write order are free. That is what licenses the rotated a-walk. No floating-point order exists, so nothing is gated on a precision baseline.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Grid synchronization | Qwen: each output tile is its own unit on its own core (2 to 32 cores, one block each, `2*a_live` reads of 32 B per core). Per-core fixed costs (launch, one DRAM read round trip, one write round trip) dominate a handful of transfers | Cap `num_cores` so a core's units cover a whole live b-run (`block_b_rows = b_live`: 1 whole-band read per input tile). 303: 2 cores x (12 reads, 192 copies, 16 writes) vs 32 x (12, 12, 1); 304 midpoint: 6 cores x 2 tiles |
| Overlap | Multi-block cores (non-production shapes): the writer's exchange (up to 2048 copies per 32-row block) dwarfs the reader's <= 32 reads, so `depth_in = 2` cannot hide it | `block_b_rows = 8` with `depth_in = 3` |
| Exchange owner (`split_reader` entry) | The writer issues every copy and every tile write; the reader idles after its barrier | Move the exchange to the reader (`cb_output_staging` becomes a reader -> writer CB, `cb_input_rows` reader-private) or split the copies by column half between the two RISCs, measured with the `PERM-WR-XCHG-ISSUE` vs `PERM-WR-XCHG-BARRIER` zones first |
| Bank stagger (`bank_stagger` entry) | Default rotates each block's a-walk start (`b0 mod a_live`). Up to 16 cores read row b of the same 6 to 16 input tiles at once | Plain a-walk from `a = 32*ta` |
| NoC placement (`noc_placement` entry) | Default: `split_work_to_cores(..., row_wise=True)`, reads on NoC0, writes and copies on NoC1 | Column placement (`row_wise=False`) and writer copies on NoC0 |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| DRAM x -> `cb_input_rows` | bfloat16 face-row bytes, unchanged | reader (NCRISC, NoC0): `noc_async_read(in_accessor.get_noc_addr(in_page, off), slot_base + off, len)` per contiguous range of the block's row run | same intra-tile offset on both ends (mod-64 rule). Input padding rows are never addressed |
| `cb_input_rows` -> `cb_output_staging` | 32-B face-row segments | writer (BRISC, NoC1): self-aimed single-packet reads, `dataflow_kernel_lib::set_read_state<32>` once + `read_with_state(noc, dst, src)` per segment | L1 mod-16 rule holds for any two 32-B-aligned offsets. Bit-exact: no engine involved |
| `cb_output_staging` -> DRAM out | whole 2048-B tiles | writer: `noc_async_write(staging_page, out_accessor.get_noc_addr(out_page), 2048)` | output padding rows carry stale staging bytes (allowed, D7) |
| Tensix <-> Tensix | — | none in any regime | no reuse-shared or dependent axis; no mcast, no semaphore |
| Compute engine | — | no compute kernel in `rows_tile` | bit identity (rule 1); TRISCs idle |

TARGET axes as blocking-scheme realizations (`feature_spec.py` TARGET has no sharded placement; "buffer" is dram | l1, both INTERLEAVED, so no physical shard exists to consume in place):

| TARGET value outside Phase 0 SUPPORTED | Regime | Class |
|-----------------------------------------|--------|-------|
| alignment "tile_aligned", "w_non_aligned" with dims (0,2,1,3) TILE | `rows_tile` (`b_live = 32`; W padding columns move with their rows into output padding columns) | knob-turn: SUPPORTED widening + tests |
| buffer "l1" with `rows_tile` | `rows_tile` (TensorAccessor addresses L1 banks) | knob-turn: placement; the ledger gains the output's own pages per core (`ceil(out_pages / L1 banks)` x 2048 B) against 1347776 B |
| dtype bfloat8_b with dims (0,2,1,3) | `rows_tile_bf8` | scheme addition (exponent-byte moves) |
| layout ROW_MAJOR with dims (0,2,1,3) | `sticks_rm` | scheme addition (own kernel pair) |
| rank 6, dims (0,1,4,2,5,3), ROW_MAJOR | `blocks_rm` | scheme addition (own kernel pair) |
| dims (0,3,1,2) ROW_MAJOR l1 / TILE bfloat8_b l1 | `nhwc_to_nchw_rm` / `nhwc_to_nchw_tile_bf8` | scheme addition (own kernel pairs; p3 reuses the `rows_tile` exchange) |

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one output tile `(n, b, ta, tw)`. Units are ordered b-fastest: `u = ((n * tensor_a_tiles + ta) * tensor_w_tiles + tw) * B + b` |
| Grid | `device.compute_with_storage_grid_size()` (runtime-derived, never inlined); `G = grid.x * grid.y` |
| Split | `(num_cores, all_cores, core_group_1, core_group_2, units_per_core_g1, units_per_core_g2) = ttnn.split_work_to_cores(grid, total_units, row_wise=True)`. Core k, in the row-wise order of `all_cores`, owns the contiguous range `[core_start_unit, core_start_unit + core_num_units)` |
| Per-core work | `core_num_units` output tiles, cut into blocks by the block rule; `num_blocks_this_core` follows from the rule. Reader and writer walk the same sequence from the same two RT args |
| Remainder | group 2 cores own `units_per_core_g2 = units_per_core_g1 - 1` units. A ragged b-run (B boundary, b-tile boundary or range end) is a shorter `block_b_count` with the nominal CB quantum. The ragged last a-tile (`a_live < a_tile_rows`) reads and copies only its live rows and pushes `a_tile_rows` |
| Geometry | `tensor_a_tiles = ceil(A/32)`, `tensor_b_tiles = ceil(B/32)`, `tensor_w_tiles = ceil(W/32)`, per image; never `floor`, never `//` on a flattened height |
| Qwen result | `total_units = 2B` (`N = 1`, `tensor_a_tiles = 1`, `tensor_w_tiles = 2`) <= 32 < G, so `num_cores = 2B` (2 to 32), one unit and one block per core, `block_b_rows = 1`, `depth_in = 1` |
| Core ranges (BB-17) | `all_cores` from the row-wise split is one rectangle of full rows plus at most one partial row: <= 2 ranges per kernel group for every production shape. Reader and writer each use one KernelDescriptor over `all_cores` |

Plan selection function (host, pinned; a pure function of shape, canonical dims and grid; the values are fixed per call site):

| Step | Rule |
|------|------|
| 1 | `regime` = selection function above (Phase 0: `rows_tile`) |
| 2 | `a_tile_rows = min(32, A)`; `total_units = N * tensor_a_tiles * tensor_w_tiles * B` |
| 3 | `split_work_to_cores(grid, total_units, row_wise=True)` gives `num_cores`, groups, `units_per_core_g1/g2` |
| 4 | `block_b_rows = min(32, B, units_per_core_g1)` (spread first, then the coarsest block within a core's assignment) |
| 5 | `max_blocks_per_core` = max over cores of the block-rule count of that core's range (host walk, O(blocks)); `depth_in = min(2, max_blocks_per_core)` |
| 6 | CB pages: `cb_input_rows = depth_in * a_tile_rows`, `cb_output_staging = block_b_rows`. Footprint `<= (2*32 + 32) * 2048 = 196608 B` < 1347776 B for every shape, so no host refusal is needed in Phase 0 (BB-05) |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_input_rows` | 0 | 2048 B (`ttnn.tile_size(bfloat16)`) | `depth_in * a_tile_rows` | spans a (one tile-page slot per live input tile of the a-tile); streams b (the run's rows sit inside each slot at their own offsets); streams w (1 column); streams n | bfloat16 (dataflow-only payload; no DEST) | reader | writer | `load_rows_block` -> `exchange_rows_block` of one block |
| `cb_output_staging` | 1 | 2048 B | `block_b_rows` (`depth_out = 1`) | spans b (one page per output tile of the run); spans a (the `a_live` rows inside each page); streams w, n | bfloat16 | writer | writer (private scratch: no handshake, base from `get_write_ptr` once) | `exchange_rows_block` -> `store_tiles_block` of one block |

Push count = wait count: `cb_input_rows` is pushed `a_tile_rows` per block by the reader and waited and popped `a_tile_rows` per block by the writer, the same block sequence on both. Capacity `depth_in * a_tile_rows` closes the ring. `cb_output_staging` has no handshake.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_rows_block` | `a_live` tiles x `block_b_count` rows x 1 column | raw dataflow (no helper reads row ranges of a DRAM tile) | DRAM x via TensorAccessor | `cb_input_rows`, `a_tile_rows` pages reserved -> pushed | slots hold the run's rows at their tile offsets; rows outside the run are stale |
| 2 | `exchange_rows_block` | `block_b_count` tiles x `a_live` rows x 2 halves | helper `local_copy_helpers_dataflow` | `cb_input_rows`, `a_tile_rows` pages waited, popped after the copy barrier | `cb_output_staging`, `block_b_count` pages written in place | staging pages hold rows `0..a_live-1`; the input slots are free for the reader |
| 3 | `store_tiles_block` | `block_b_count` whole tiles | raw dataflow | `cb_output_staging`, `block_b_count` pages | DRAM out via TensorAccessor | writes in flight until the next block's `writes_flushed` or the final barrier |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `load_rows_block` | raw_api | `cb_reserve_back` / `cb_push_back` / `get_write_ptr` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:406` / `:208` / `:325` | `(cb_input_rows, a_tile_rows)` | — | `cb_input_rows` | `a_tile_rows` (quantum), `depth_in` (capacity) |
| `load_rows_block` | raw_api | `TensorAccessor::get_noc_addr(page_id, offset)` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:165` | `in_page(n, a, b0 div 32, tw)`, `off` = range start | DRAM x | — | `block_b_count` sets the ranges |
| `load_rows_block` | raw_api | `noc_async_read` / `noc_async_read_barrier` | `dataflow_api.h:547` / `:1921` | `(src_noc, slot_base + off, len)`, `len` in {32*k, 1024, 2048} | DRAM x | `cb_input_rows` | — |
| `exchange_rows_block` | raw_api | `cb_wait_front` / `get_read_ptr` / `cb_pop_front` | `dataflow_api.h:479` / `:346` / `:260` | `(cb_input_rows, a_tile_rows)` | `cb_input_rows` | — | `a_tile_rows` |
| `exchange_rows_block` | helper | `dataflow_kernel_lib::set_read_state<32>(noc, src)` + `read_with_state(noc, dst, src)` | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:75`, `:95`; impl `.inl:62-77` | transfer_size 32; `dst = staging + (b-b0)*2048 + off(a mod 32, h)`, `src = slot + (a-32ta)*2048 + off(b mod 32, h)`. The per-call path rewrites both `NOC_RET_ADDR_LO` and `NOC_TARG_ADDR_LO` (`tt_metal/hw/inc/internal/tt-1xx/blackhole/noc_nonblocking_api.h:1508-1509`), so `src` may vary per call under one state; the "call set_read_state again" note at `.inl:60` is conservative | `cb_input_rows` | `cb_output_staging` | `block_b_count`, `a_live` (copy count) |
| `exchange_rows_block` | raw_api | `noc_async_read_barrier`, `noc_async_writes_flushed` | `dataflow_api.h:1921`, `:1980` | writer's NoC | — | — | — |
| `store_tiles_block` | raw_api | `noc_async_write` + `TensorAccessor::get_noc_addr` | `dataflow_api.h:876`, `tensor_accessor.h:165` | `(staging_page, out_page(n, b, ta, tw), 2048)` | `cb_output_staging` | DRAM out | `block_b_count` |
| kernel exit (writer) | raw_api | `noc_async_write_barrier` | `dataflow_api.h:1951` | — | — | — | — |
| host split | raw_api | `ttnn.split_work_to_cores(grid, units, row_wise=True)` | `ttnn/cpp/ttnn-nanobind/operations/core.cpp:470-475` | `total_units` | — | — | core count derives from the unit split |

Helpers considered and rejected:

| Raw entry | Candidate helper | File:Line of the mismatch | Reason |
|-----------|------------------|---------------------------|--------|
| `load_rows_block` reads | `local_copy_helpers_dataflow` | `local_copy_helpers_dataflow.inl:30-34` (`local_addr` aims the source at this core's own L1) | the source here is a DRAM bank page, not local L1 |
| `load_rows_block` reads | `tilize_helpers_dataflow` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp:47` (reads row-major sticks for tilization) | changes layout; input and output here are both tiles |
| `load_rows_block` reads | `tile_column_dataflow` | `ttnn/cpp/ttnn/kernel_lib/tile_column_dataflow.hpp:9`, `:23` (writes one tile column with RISC-V stores) | wrong axis (columns, not rows) and a RISC-V store path, not a NoC read |
| `store_tiles_block` | (no dataflow write helper in `kernel_lib`) | — | whole-page writes are the plain API |
| any phase | compute helpers `copy` (`ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp:113`), `transpose_tile` (`tt_metal/hw/inc/api/compute/transpose.h:120`), `reshuffle_rows_tile` (`tt_metal/hw/inc/api/compute/reshuffle.h:31`) | as cited | each routes bfloat16 through unpack/DEST/pack, which the contract forbids (rule 1, R3); `reshuffle_rows_tile` moves rows within one tile while this move crosses tiles |

## Host Contract

| Item | Decision |
|------|----------|
| Entry point | `g6_permute(input_tensor, dims)`; first line `validate(input_tensor, dims)` |
| `validate()` order | (0) dims structure: wrong length, entry outside `[-r, r-1]`, or not a permutation after wrapping raises **ValueError** before any tagging. (1) a non-INTERLEAVED input raises `UnsupportedAxisValue`. (2) scenario dict `{input_shape, dims, dtype, layout, buffer}` -> INPUT_TAGGERS. (3) SUPPORTED per axis -> `UnsupportedAxisValue`. (4) EXCLUSIONS (Phase 0: none) -> `ExcludedCell` |
| INPUT_TAGGERS | `dtype`, `layout`, `buffer` ("dram" / "l1" from `memory_config().buffer_type`), `rank` (`len(input_shape)`), `alignment` (W % 32 != 0 -> "w_non_aligned"; else H % 32 != 0 -> "h_non_aligned"; else "tile_aligned"; H = shape[-2], W = shape[-1]), `dims` (canonical tuple, negative entries wrapped once) |
| Phase 0 SUPPORTED (the implementer writes it) | dtype [bfloat16], layout [TILE], buffer ["dram"], rank [4], alignment ["h_non_aligned"], dims [(0, 2, 1, 3)]; EXCLUSIONS [] |
| Output | `ttnn.allocate_tensor_on_device(ttnn.Shape(out_shape), x.dtype, x.layout, device, x.memory_config())`, then one `ttnn.generic_op([x, out], descriptor)` |
| Plan report (BB-08, BB-21) | `g6_permute_plan(input_tensor, dims) -> Plan`, exported from `ttnn.operations.g6_permute`; validates first. Fields: `regime`, `num_cores`, `total_units`, `units_per_core_g1`, `units_per_core_g2`, `a_tile_rows`, `block_b_rows`, `depth_in`, `max_blocks_per_core`, `core_roles` (`{"x,y": "rows_tile reader+writer units [s, s+n)"}`, one entry per used core). The entry point builds its descriptor from the same Plan |
| Program and descriptor cache (BB-07, BB-16) | Key: input shape, canonical dims, dtype, layout, buffer type, memory layout, device grid. The ProgramDescriptor is built once per key. The only per-call values are the two buffer addresses, carried as common runtime args of the reader (input) and writer (output) and patched on a hit. Per-core RT args are `(core_start_unit, core_num_units)` only. The program allocates no tensor besides the output |
| Kernel text (BB-04, BB-18) | One reader and one writer source per regime. Shape constants, `a_tile_rows`, `block_b_rows`, `depth_in` are CT args, so no other regime's path is compiled into a `rows_tile` binary |
| Zones (BB-21) | `DeviceZoneScopedN` (a no-op without the profiler, `tt_metal/hw/inc/internal/profiler/kernel_profiler_streaming.hpp:569`). Reader: `PERM-RD-RESERVE`, `PERM-RD-ISSUE`, `PERM-RD-BARRIER`. Writer: `PERM-WR-WAIT`, `PERM-WR-FLUSH`, `PERM-WR-XCHG-ISSUE`, `PERM-WR-XCHG-BARRIER`, `PERM-WR-STORE-ISSUE`, `PERM-WR-FINAL-BARRIER`. Per-step zones on the first 16 blocks of a core, then `PERM-RD-TAIL` / `PERM-WR-TAIL` around the rest |
| NoC completion (BB-03) | reader: read barrier per block; writer: copy barrier per block and a write barrier at exit; no semaphores, no atomics |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Blackhole DRAM read mod-64 mismatch | 32-B reads of odd rows start at 32 mod 64 | rows land at their own tile offset in a tile-page slot; CB base and page size are multiples of 64; never compact slots, never read into the staging page |
| Input padding reaching a live output row (R4, poisoned-padding loose cases) | B of 32 rows live per input tile; garbage in a producer's padding | runs never extend past B; whole-band and whole-tile ranges are used only when every row in them is live |
| Rows straddling the 16-row band | a run `[r0, r1)` crossing row 16 is 4 ranges, not 2 | range rule per band; the (1, 33, 100, 512) acceptance shape exercises it |
| Ragged last a-tile | A = 40 or 33: `a_live = 8` or `1` in a-tile 1, yet the CB quantum is `a_tile_rows = 32` | push and pop the nominal `a_tile_rows`; read and copy only `a_live` slots |
| CB ring overshoot | a ragged quantum reaching `cb_push_back` corrupts the second lap | the quantum is a CT constant; ragged extents only change transfer counts |
| Staging overwritten while its tile writes are in flight | the writer refills staging on the next block | `noc_async_writes_flushed()` before each exchange after the first |
| Popping input slots before the copies land | the reader would overwrite rows still being read | copy barrier, then `cb_pop_front` |
| Writer reads on its own NoC | local copies are reads issued by BRISC on NoC1 | `set_read_state` and the barrier on the writer's NoC; `local_addr` indexes `my_x/my_y` by that NoC id (`.inl:30-34`) |
| Unit index decode cost | division on a data-movement RISC-V | decode `core_start_unit` once, then advance `(b, tw, ta, n)` with carries |
| bfloat16 bit identity incl. NaN payloads and denormals (R3) | any engine route changes bits | no compute kernel; every byte moves by NoC |
| Output padding is stale staging bytes | a reader of the output might expect zeros | contractually unspecified (D7); tests read logical values only |
| dims missing from the cache key (R10) | one shape, two permutations | canonical dims in both the descriptor key and the CT args |

## Structural impossibilities

None beyond `feature_spec.py` INVALID (`{dtype: bfloat8_b, layout: ROW_MAJOR}`).

## Catalog entries used

| Entry | Informed |
|-------|----------|
| `tile_reorder` | whole-page output writes (`store_tiles_block`); deferral of `rows_tile_scatter` |
| `double_buffer` | all reads of a block issued before one barrier; `depth_in = 2` for multi-block cores |
| `noc_placement` | `row_wise=True`; reads on NoC0, writes on NoC1; placement lamp |
| `bank_stagger` | rotated a-walk start; bank-stagger lamp |
| `split_reader` | exchange-owner lamp |
| `width_split` | split the axis with the most units (b-fastest output tiles) so a few a-tiles and tile columns still fill the grid |
