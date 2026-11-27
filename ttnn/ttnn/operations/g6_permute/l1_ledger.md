# L1 Ledger: g6_permute

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking decisions: `op_design.md` → Blocking Model. Built regimes: `rows_tile_scatter` (Phase 0), `blocks_rm_gather` (Refinement 1), `nhwc_tile_bfp8_transpose` (Refinement 2), `nhwc_rm_transpose` (Refinement 4), `blocks_rm_bank_coalesced` (Refinement 5). Every regime runs with buffer `"dram"` or `"l1"`; with `"l1"` the output's own pages on a core (`out_l1_bytes_per_core`) are subtracted from `L1_OP_BUDGET_BYTES` before the block or depth is sized.

## CB ledger (`rows_tile_scatter`)

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_input_tiles` | `min(input_depth, max_blocks_per_core) · block_in_tiles` pages of `in_page_bytes` (production: 1 page) | `min(input_depth, max_blocks_per_core) · block_in_tiles`: with ≥ 2 blocks per core the reader fills block `k+1` while the writer scatters block `k`, so both are live; capacity = live set | `{in_tile: spans → block_in_tiles · min(input_depth, max_blocks_per_core); r: spans → 32 (whole tile slot; the prefix fills rows_live(bt) of them); h: spans → 2; n, a, bt, wt: folded into in_tile}` | Float16_b — a dataflow-only payload; no compute kernel exists, so there is no DEST width to match (audit 2 does not apply) | reader | writer | whole program | Nothing to share with: it is the op's only buffer. No output CB exists — rows go from this slot straight to their output pages in DRAM (pack-into-destination, Rule 3.1); no second lifetime exists to alias |

## Symbol table

| Symbol | Bound | Established by |
|--------|-------|----------------|
| `in_page_bytes` | `input_tensor.buffer_page_size()` = 2048 | Phase 0 SUPPORTED dtype bfloat16, 32×32 tiles (validate) |
| `input_depth` | 2 | host constant |
| `max_core_in_tiles` | `ceil(tensor_in_tiles / num_cores)` = `tiles_g1` of `split_work_to_cores` | work split |
| `block_in_tiles` | `min(max_core_in_tiles, block_cap)`, ≥ 1 | Work Distribution formula (single source) |
| `max_blocks_per_core` | `ceil(max_core_in_tiles / block_in_tiles)` | derived |
| `block_cap` | `floor((L1_OP_BUDGET_BYTES − out_l1_bytes_per_core) / (input_depth · in_page_bytes))` = 329 in Phase 0 | mechanism cap (BB-05) |
| `L1_OP_BUDGET_BYTES` | 1347776 | model environment: L1 left per core, output's own L1 pages included |
| `out_l1_bytes_per_core` | 0 in Phase 0 (output in DRAM); `ceil(tensor_out_tiles / num_l1_banks) · out_page_bytes` for buffer `"l1"` | output placement |

## Total per-core footprint

`min(input_depth, max_blocks_per_core) · block_in_tiles · in_page_bytes + out_l1_bytes_per_core ≤ input_depth · block_cap · in_page_bytes + out_l1_bytes_per_core ≤ L1_OP_BUDGET_BYTES`.

The first term scales with `block_in_tiles` (block knob) and `input_depth` (depth knob); the second with the output placement only. Every production shape: `1 · 1 · 2048 = 2048` B per core.

## Data-movement budget (`rows_tile_scatter`)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `x` | 1 — the live prefix of each tile, one transaction per tile | each input tile belongs to exactly one core and is read once; its rows go straight to their destinations, so nothing is re-read | 0 |
| `out` | 1 — exactly the logical bytes, 32 B per transaction | each half-row is written once, from the slot that holds it; output padding rows are never written | 0 |

General totals per call: DRAM read `Σ_tiles prefix_bytes(rows_live(bt))`; DRAM write `tensor_n · tensor_a · tensor_b · tensor_w · 2` B in `2 · tensor_n · tensor_a · tensor_b · tensor_wt` writes; NoC cross-core 0; core-local copies 0.

| Case | Input `[N, A, B, W]` | Input tiles (= cores) | DRAM read (B) | DRAM write (B) | Write transactions |
|------|----------------------|-----------------------|---------------|----------------|--------------------|
| 301 | `[1, 1, 16, 64]` | 2 | 2048 | 2048 | 64 |
| 302 | `[1, 16, 1, 64]` | 32 | 17408 | 2048 | 64 |
| 303 | `[1, 6, 16, 64]` | 12 | 12288 | 12288 | 384 |
| 304 | `[1, 16, 6, 64]` | 32 | 22528 | 12288 | 384 |
| 305 | `[1, 1, 8, 64]` | 2 | 1536 | 1024 | 32 |
| 306 | `[1, 8, 1, 64]` | 16 | 8704 | 1024 | 32 |
| 307 | `[1, 6, 8, 64]` | 12 | 9216 | 6144 | 192 |
| 308 | `[1, 8, 6, 64]` | 16 | 11264 | 6144 | 192 |

Reads exceed the logical bytes only by padding rows inside a prefix (`(16 − B) · 32` B per tile for `B < 16`); writes equal them.

> Cheapest-traffic split considered: input tiles with rows scattered straight to the output (`rows_tile_scatter`) — x once (live prefix), out once (logical bytes), 0 cross-core. Implemented: the same split. The alternatives cost more on some tier: `rows_tile_group_assemble` writes whole output tiles (303: 65536 B against 12288 B), `rows_tile_gather` reads 2× through 64 B bounces, `rows_tile_reread` re-reads x up to 32×, `rows_tile_mcast` adds 32× cross-core bytes.


## CB ledger (`blocks_rm_gather`)

Unit `(n, c, h)`: `W·I` input pages of `J` values in, `I·J` output pages of `W` values out. `unit_slot_bytes = W·I·slot_bytes`, `unit_row_bytes = I·J·out_row_bytes`.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_input_slots` | `min(input_depth, max_blocks_per_core) · block_h` pages of `unit_slot_bytes` (FLUX_74: 2 × 4 × 4096 B) | with ≥ 2 blocks per core the reader fills block `k+1` while the writer gathers block `k`; capacity = live set | `{h: spans → block_h · depth; w: spans → W; i: spans → I; j: spans → J (one input page, in a slot of slot_bytes = the page's bank stride, so local ≡ remote mod 64); n, c: folded into the unit}` | Float16_b, dataflow-only payload | reader | writer | whole program | Cannot take the rows: the gather reads every slot of a unit while it writes the unit's rows, and the rows are 16× denser than the slots, so writing them in place overwrites unread values |
| `cb_output_rows` | `block_h` pages of `unit_row_bytes` (FLUX_74: 4 × 256 B), never pushed or popped | one block of gathered rows: the gather of block `k+1` starts only after `noc_async_writes_flushed` releases block `k`'s rows | `{h: spans → block_h; i, j: spans → I·J rows; w: spans → W values per row, padded to out_row_bytes (L1 alignment, a NoC write source)}` | Float16_b | writer (gather) | writer (NoC write source) | whole program | The writer's private staging area. Cannot be the output tensor: an output page is in DRAM. Cannot alias `cb_input_slots`: both are live during the gather |

### Symbol table (`blocks_rm_gather`)

| Symbol | Bound | Established by |
|--------|-------|----------------|
| `slot_bytes` | `input_tensor.buffer_aligned_page_size()` = 64 on DRAM (J ≤ 32) | DRAM read alignment (risk R7) |
| `out_row_bytes` | `align_up(W · 2, l1_alignment)` = 64 for W = 32 | NoC write source alignment |
| `block_h` | `min(units_per_core_g1, block_cap, ceil(BLOCK_PAGES / (W·I)))` with `BLOCK_PAGES = 256` (and `BLOCK_H_LIMIT` when set); FLUX_74 and FLUX_73: 4 | host plan, single source (Refinement 3: a small block keeps all but the last block's gather under the reads) |
| `block_cap` | `floor((L1_OP_BUDGET_BYTES − out_l1_bytes_per_core) / (input_depth · unit_slot_bytes + unit_row_bytes))` = 159 on FLUX | mechanism cap (BB-05); `RuntimeError` when < 1 |
| `input_depth` | `min(INPUT_DEPTH = 2, max_blocks_per_core)` | host constant |

### Total per-core footprint (`blocks_rm_gather`)

`depth · block_h · unit_slot_bytes + block_h · unit_row_bytes + out_l1_bytes_per_core ≤ L1_OP_BUDGET_BYTES`.
FLUX_74 and FLUX_73 (blocks of 4 units, depth 2): 32768 + 1024 = 33792 B per core. Before Refinement 3 the whole per-core assignment was one block (FLUX_74: 326400 B).

## Data-movement budget (`blocks_rm_gather`)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `x` | 1, one read per 4 B page into its own 64 B slot | each unit belongs to exactly one core; its pages are read once | 0 |
| `out` | 1, one 64 B write per output page | each output row is gathered once and written once | 0 |

| Case | Input pages (reads) | Output pages (writes) | Logical bytes each way | DRAM slot traffic on the read side |
|------|---------------------|-----------------------|------------------------|-------------------------------------|
| FLUX_73 `[1,32,64,32,2,2]` | 131072 × 4 B | 8192 × 64 B | 512 KiB | 131072 × 64 B = 8 MiB |
| FLUX_74 `[1,64,128,32,2,2]` | 524288 × 4 B | 32768 × 64 B | 2 MiB | 524288 × 64 B = 32 MiB |

Core-local: every value moves once by RISC-V loads and stores (the gather). The read side is bound by the count of DRAM requests, not their bytes: each DRAM port retires one 4 B read per ~17 ns (`changelog.md`, Refinements 1 and 3). On DRAM input the reader sends half of each bank's reads through the NoC1 port of that bank, so 16 ports serve the 524288 FLUX_74 requests instead of 8. The crossings per tensor are unchanged.


## CB ledger (`blocks_rm_bank_coalesced`)

Same cell, unit, split and writer as `blocks_rm_gather`; the host selects it for DRAM and L1 input (`READ_MODE = "auto"`, `COALESCED_BUFFERS`). The reader reads each bank's pages of a block in one transaction of whole slots into `cb_bank_staging`, then copies the payloads into `cb_input_slots` in page order. On DRAM output the writer stages a write group's rows bank by bank in `cb_output_rows` and writes each (run `i`, bank) at once.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_bank_staging` | `staging_depth` pages of `num_banks · staging_rounds · slot_bytes`, never pushed or popped (FLUX_74 DRAM: 1 × 8 × 32 × 64 B = 16384 B; L1: 1 × 110 × 20 × 16 B = 35200 B) | one block's whole slots, bank by bank, from the read until the copy | `{bank: spans → num_banks regions; round: spans → staging_rounds slots per region (a block that starts mid round needs one more); j: one page payload in a slot of slot_bytes, so local ≡ remote mod 64}` | Float16_b, dataflow-only | reader (NoC read) | reader (copy) | whole program | The reader's private landing area. Cannot be `cb_input_slots`: a slot is 16× a payload (DRAM), and the writer's gather wants page order at a word stride. Depth 1 is measured, not forced: with 2 stages the reads never pause and the NoC links into the far columns congest (FLUX_74: 116 µs against 89 µs) |
| `cb_input_slots` | `min(input_depth, max_blocks_per_core) · block_h` pages of `W·I·page_stride`, `page_stride = align_up(J·2, 4)` (FLUX_74 DRAM: 2 × 4 × 256 B) | as `blocks_rm_gather`, at the payload stride instead of the slot stride | as `blocks_rm_gather`, `j: one payload per page_stride` | Float16_b | reader (copy) | writer (gather) | whole program | As `blocks_rm_gather` |
| `cb_output_rows` (grouped writes) | `write_group_depth` pages of `I · num_banks · group_slots · out_slot_bytes`, never pushed or popped (FLUX_74 DRAM: 2 × 2 × 8 × 17 × 64 B = 34816 B) | the group being gathered plus, with depth 2, the group whose writes are draining | `{i: spans → I runs; bank: spans → num_banks regions; slot: group_slots per region, out_slot_bytes apart like the bank's pages; h, j: folded into the run (t = (h − h0)·J + j)}` | Float16_b | writer (gather) | writer (NoC write source) | whole program | The writer's private staging area. Cannot take the per-row layout: a (run, bank) write needs that bank's rows contiguous. Without grouped writes (L1 output, or `J ∤ num_banks`) it keeps the `blocks_rm_gather` layout |

### Symbol table (`blocks_rm_bank_coalesced`)

| Symbol | Bound | Established by |
|--------|-------|----------------|
| `block_h` | `min(units_per_core_g1, block_cap, ceil(COALESCED_BLOCK_PAGES[placement] / (W·I)))`, 256 pages on DRAM, 2048 in L1; FLUX_74: 4 (DRAM), 32 (L1) | host plan, single source |
| `staging_rounds` | `(first_bank_max + block_h·W·I − 1) // num_banks + 1`, `first_bank_max = 0` when `W·I` is a multiple of `num_banks`, else `num_banks − 1` | every block starts at a unit boundary |
| `staging_depth` | `min(STAGING_DEPTH = 1, max_blocks_per_core)` | host constant, measured |
| `write_group_units` | `min(WRITE_GROUP_UNITS = 64, H, units_per_core_g1)` on DRAM output with `J | num_banks`, else 0; lowered until `group_slots · out_slot_bytes ≤ 16384` (one NoC packet per write) | host plan |
| `group_slots` | `(num_banks − J + write_group_units·J − 1) // num_banks + 1` | a run starts at a multiple of `J`, so at most at bank `num_banks − J` |
| `write_group_depth` | `WRITE_GROUP_DEPTH = 2` when a core has more units than one group, else 1 | host constant |
| `block_cap` | `floor((budget − STAGING_DEPTH·2·num_banks·slot_bytes − write staging) / (input_depth·W·I·page_stride + unit_row_bytes + STAGING_DEPTH·W·I·slot_bytes))`, `unit_row_bytes = 0` with grouped writes; FLUX_74: 284 (DRAM), 739 (L1) | mechanism cap (BB-05) |

### Total per-core footprint (`blocks_rm_bank_coalesced`)

| Case | `cb_input_slots` | `cb_bank_staging` | `cb_output_rows` | Total |
|------|------------------|-------------------|------------------|-------|
| FLUX_74 DRAM | 2048 | 16384 | 34816 | 53248 B |
| FLUX_74 L1 | 16384 | 35200 | 8192 | 59776 B + the output's own L1 pages |
| FLUX_73 DRAM | 2048 | 16384 | 6144 | 24576 B |
| FLUX_73 L1 | 4864 | 22880 | 4864 | 32608 B + the output's own L1 pages |

## Data-movement budget (`blocks_rm_bank_coalesced`)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `x` | 1, whole 64 B slots: each bank's pages of a block in one read | each unit belongs to exactly one core; its slots are read once. The payload is 4 B of every 64 B slot, so the bytes are 16× the logical bytes, in `num_banks`–`2·num_banks` reads per block instead of one per page | 0 |
| `out` | 1, whole 64 B slots: each (run `i`, bank) of a write group in one write (DRAM output) | each output row is gathered once, into its bank's slot of the group, and written once | 0 |

| Case | Reads | Read bytes | Writes | Write bytes |
|------|-------|------------|--------|-------------|
| FLUX_74 DRAM | 19 blocks × 8–16 per core | 524288 × 64 B = 32 MiB | 2–3 groups × 2 runs × 8 banks = 32–48 per core, ≤ 1088 B each (was 300 × 64 B) | 2 MiB |
| FLUX_73 DRAM | 5 blocks × 8–16 per core | 131072 × 64 B = 8 MiB | 1–2 groups × 2 runs × 8 banks per core | 512 KiB |

Core-local: every payload moves twice by RISC-V loads and stores, the copy out of the staged slots and the gather. FLUX_74 DRAM reads alone take 71.2 µs of the 83.3 µs kernel (~508 GB/s, the DRAM peak), so the read bytes, not their count, bound it now.

## CB ledger (`nhwc_tile_bfp8_transpose`)

Group `(n, ht, wt, ct)`: 32 input tiles `(n, h, wt, ct)` → 32 output tiles `(n, c, ht, wt)`. `in_page_bytes = out_page_bytes = 1088` (bfloat8_b 32×32), `row_page_bytes = 2048` (32 rows of 32 bfloat16).

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_input_tiles` | `group_depth · 32` pages of `in_page_bytes` | the group being transposed plus, with `group_depth = 2`, the next group being read | `{h: spans → 32 per group · group_depth (pushed in chunks of read_chunk_tiles); w, c: spans → the whole tile; n, ht, wt, ct: folded into the group}` | Bfp8_b (input dtype) | reader | compute (transpose) | whole program | Cannot alias `cb_row_block`: the transpose reads one while it packs the other. Cannot be read in place by the tilize: tilize needs bfloat16 rows, and these are bfloat8_b tiles |
| `cb_row_block` | 32 pages of `row_page_bytes` | one group: page `h` = transposed input tile `h` untilized, i.e. row `h` of a `[32, 32·32]` row-major block | `{h: spans → 32 (one page per h); c: spans → 32 (64 B row chunks of a page); w: spans → 32 values per chunk}` | Float16_b — an intermediate with 16-bit DEST (`fp32_dest_acc_en = False`); bfloat8_b values are exact in bfloat16 | compute (pack untilize) | compute (tilize) | whole program | The row exchange itself: the tilize reads 32-column chunk `c` of every row to build output tile `c`. Cannot alias `cb_input_tiles` or `cb_output_tiles`: each is live with it in one of the two passes. Cannot be Bfp8_b: an untilized row is not a block-float page |
| `cb_output_tiles` | `group_depth · 32` pages of `out_page_bytes` | the group being written plus, with `group_depth = 2`, the next group being tilized | `{c: spans → 32 per group · group_depth (only c_live slots are written); h, w: spans → the whole tile}` | Bfp8_b (output dtype); the packer sets each 16-value group's exponent to its largest | compute (tilize + pack) | writer | whole program | The writer's NoC source. Cannot be the output tensor: output tiles `c` of one group live in 32 different banks |

### Symbol table (`nhwc_tile_bfp8_transpose`)

| Symbol | Bound | Established by |
|--------|-------|----------------|
| `GROUP_TILES` | 32 | the tilize width of the row exchange (one output tile per input-tile column) |
| `group_depth` | `min(GROUP_DEPTH = 2, groups_per_core_g1)`, lowered to 1 if two groups exceed the budget | host plan, single source |
| `read_chunk_tiles` | `READ_CHUNK_TILES = 8` (divisor of 32) | host constant; the DEST batch of the transpose is 8 tiles (16-bit half-sync) |
| footprint | `32 · (2 · group_depth · in_page_bytes + row_page_bytes)` | `_footprint()` |

### Total per-core footprint (`nhwc_tile_bfp8_transpose`)

`32 · (2 · group_depth · 1088 + 2048) + out_l1_bytes_per_core ≤ L1_OP_BUDGET_BYTES`; `RuntimeError` when one group does not fit.
Panoptic p3 `[1,128,256,32]` in L1 (32 groups, one per core, `group_depth = 1`): 135168 B of CBs + 10 × 1088 = 10880 B of output pages = 146048 B. Several groups per core (`group_depth = 2`): 204800 B of CBs.

## Data-movement budget (`nhwc_tile_bfp8_transpose`)

| Tensor | DRAM / L1-bank crossings | Why that many | Cross-core traffic added |
|--------|--------------------------|---------------|--------------------------|
| `x` | 1, one 1088 B read per live input tile | each group belongs to one core; its 32 tiles are read once | 0 |
| `out` | 1, one 1088 B write per live output tile | each output tile is packed once, by the group that owns all 32 of its rows | 0 |

p3: 1024 tile reads and 1024 tile writes, 1114112 B each way; no core-local NoC copies (the row exchange is the compute tilize of `cb_row_block`).


## CB ledger (`nhwc_rm_transpose`)

Unit `(n, h, wb)`: `width` input pages of `C` values (`width = block_w`, or `tail_w` for the last unit of a row) → `C` partial output rows of `width` values. `unit_slot_bytes = block_w · slot_bytes`, `row_bytes = align_up(block_w · 2, 16)`, `unit_row_bytes = C · row_bytes`. Same two CBs, indices and roles as `blocks_rm_gather` (shared reader).

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_input_slots` | `min(input_depth, max_blocks_per_core) · block_units` pages of `unit_slot_bytes` (p4: 2 × 4 × 4096 B) | with ≥ 2 blocks per core the reader fills block `k+1` while the writer transposes block `k`; capacity = live set. A block's pages are packed back to back (a tail unit leaves the block's end unused) | `{w: spans → block_units · block_w · depth (one input page per slot of slot_bytes = the page's bank stride); c: spans → C (the page); n, h: folded into the unit}` | Float16_b, dataflow-only payload | reader | writer | whole program | Cannot take the rows: the transpose reads all `width` slots of a unit while it writes the unit's `C` rows, so in-place rows overwrite unread values |
| `cb_output_rows` | `block_units` pages of `unit_row_bytes` (p4: 4 × 4096 B), never pushed or popped | one block of transposed rows: the transpose of block `k+1` starts only after `noc_async_writes_flushed` releases block `k`'s rows | `{c: spans → C rows; w: spans → block_w values per row, padded to row_bytes (a 16 B-aligned NoC write source)}` | Float16_b | writer (transpose) | writer (NoC write source) | whole program | The writer's private staging area. Cannot be the output tensor: the `C` rows of a unit live in `C` different pages and banks. Cannot alias `cb_input_slots`: both are live during the transpose |

### Symbol table (`nhwc_rm_transpose`)

| Symbol | Bound | Established by |
|--------|-------|----------------|
| `slot_bytes` | `input_tensor.buffer_aligned_page_size()` = 64 for C = 32 (L1 and DRAM) | read alignment (local ≡ remote mod 64 on DRAM, mod 16 on L1) |
| `block_w` | `W`, or a multiple of 8 below `W` (16 B write alignment of the offset `wb · block_w · 2`); `BLOCK_W` when set, else `_choose_block_w`, among the widths whose one unit fits the budget. p4: 64 | host plan, single source |
| `block_units` | `min(units_per_core_g1, block_cap, ceil(BLOCK_PAGES / block_w))`, `BLOCK_PAGES = 256`. p4: 4 | host plan, single source |
| `block_cap` | `floor((L1_OP_BUDGET_BYTES − out_l1_bytes_per_core) / (INPUT_DEPTH · unit_slot_bytes + unit_row_bytes))` | mechanism cap (BB-05); `RuntimeError` when no width fits one unit |
| `input_depth` | `min(INPUT_DEPTH = 2, max_blocks_per_core)` | host constant |
| `out_l1_bytes_per_core` | `ceil(N · C · H / num_l1_banks) · align_up(W · 2, 16)` in L1; p4: 38 × 512 = 19456 B | output placement |

### Total per-core footprint (`nhwc_rm_transpose`)

`input_depth · block_units · unit_slot_bytes + block_units · unit_row_bytes + out_l1_bytes_per_core ≤ L1_OP_BUDGET_BYTES` (reported as `l1_bytes_per_core` + `out_l1_bytes_per_core` by the plan).
Panoptic p4 `[1,128,256,32]` in L1: 32768 + 16384 B of CBs + 19456 B of output pages = 68608 B per core (L17 runs it with 1347776 B free).

## Data-movement budget (`nhwc_rm_transpose`)

| Tensor | DRAM / L1-bank crossings | Why that many | Cross-core traffic added |
|--------|--------------------------|---------------|--------------------------|
| `x` | 1, one read per `C·2` B input page into its own slot | each unit belongs to one core; its pages are read once | 0 |
| `out` | 1, `C` writes of `width · 2` B per unit | each output value is transposed once and written once, by the unit that owns its `w` | 0 |

p4 (`block_w` = 64): 32768 reads of 64 B and 16384 writes of 128 B, 2 MiB each way. Core-local: every value moves once by RISC-V loads and stores (the transpose).
