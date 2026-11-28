# Operation Design: g6_copy

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement: one reader and one writer dataflow kernel, **no compute kernel** |
| Goal | Copy every buffer page of a DRAM-interleaved source into the caller's DRAM-interleaved destination of identical spec, in one device program, at DRAM roofline, and return the destination object itself. Perf target (feature_spec `LOOSE_CASES[0]`): float32 TILE (16, 12, 128, 128) below 68.56 µs device kernel time (roofline 54.61 µs, 25,165,824 B across DRAM) |
| Math | `output_tensor[i] = input_tensor[i]` for every logical index i, realized as `dst_page[p] = src_page[p]` byte for byte for every buffer page p |
| References | Contract: the g6_copy spec (user prompt). `eval/golden_tests/g6_copy/feature_spec.py`, `helpers.py`, `test_regression.py`. `.claude/references/blocking-model.md`, `l1-footprint-discipline.md`, `noc-alignment-in-tt-metal.md`. `tt_metal/hw/inc/api/tensor/tensor_accessor.h` (interleaved contiguity API, :560-575). Catalog `ttnn/ttnn/operations/examples/master.md`: `read_coalescing`, `noc_placement`, `double_buffer`, `trid_double_issue`, `bank_stagger`, `tile_reorder`, `split_reader`. Device facts probed on this board (tt-probe, this run): grid 11 × 10 = 110 cores, DRAM grid 8 × 1 (8 banks), DRAM alignment 64 B, L1 alignment 16 B; DRAM-interleaved TensorAccessorArgs = 2 CT args, 0 common RT args. The exploration bundle `docs/explorations/g6_copy/` is not installed on this branch |

**Why a page copy is the element copy.** Source and destination have the same shape, dtype, layout
and memory config, hence the same page size, page count, page → (bank, offset) map and element →
(page, byte) map. Copying page p verbatim puts every logical element at its own position. A
whole-page copy never moves a byte across positions, so source tile padding lands only in
destination tile padding: no logical destination element depends on source padding, and the three
alignment tags need no algorithm of their own. A bfloat8_b tile (mantissas plus shared exponents,
one 1088 B page) moves as raw bytes, which is bit-identical and stronger than the one-step gate.
No value passes through an unpacker, DEST or a packer, so no representation can change.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | rank 1–4, on device, DRAM interleaved, a SUPPORTED cell | — | buffer address: reader common RT arg (patched per call) |
| `output_tensor` | `ttnn.Tensor` | yes | same logical shape, dtype, layout and memory_config as `input_tensor` | — | buffer address: writer common RT arg (patched per call) |

No other parameter: no dtype, memory_config or compute_kernel_config. No index axis exists, so no
canonicalization is needed.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `(d0, …, dn)`, rank 1–4 |
| Dtype | Phase 0: float32. Refinement 1: float32, bfloat16, bfloat8_b, int32, uint32 (= TARGET) |
| Layout | Phase 0: TILE. Refinement 1: TILE and ROW_MAJOR (= TARGET; ROW_MAJOR in bfloat16, int32, uint32 per the EXCLUSIONS / INVALID) |
| Memory | DRAM interleaved |

### Output

| Property | Value |
|----------|-------|
| Shape | equal to the input shape (validated, never derived) |
| Dtype | equal to the input dtype |
| Layout | equal to the input layout |
| Memory | DRAM interleaved. The object passed as `output_tensor`, on its own buffer; the op allocates nothing in DRAM |

## Blocking Model

### Page index space

| Term | Definition |
|------|-----------|
| `tensor_pages` | `input_tensor.buffer_num_pages()` (equal to the output's). TILE: `prod(batch dims) · ceil(H/32) · ceil(W/32)`, tile-padded per image (H = 1 for rank 1). ROW_MAJOR: number of rows. Never `floor(volume / 1024)` |
| `page_bytes` | `input_tensor.buffer_aligned_page_size()`. float32 / int32 / uint32 tile 4096, bfloat16 tile 2048, bfloat8_b tile 1088; ROW_MAJOR row `round_up(W · elem_bytes, 64)` (probed: a 4 B uint32 row is a 64 B page). This is the DRAM bank stride of a page |
| `num_banks` | DRAM banks of the interleaved allocator, 8 here. Read on device only, as `TensorAccessor::contiguous_page_stride()` (= `NUM_DRAM_BANKS`, tensor_accessor.h:562). **Host code never uses it** |
| bank run | Page p lives in bank `p mod num_banks` at slot `p / num_banks`; slots of one bank sit `page_bytes` apart. So pages `p, p + num_banks, p + 2·num_banks, …` form one contiguous byte span, at the same offsets in both tensors. `num_contiguous_pages(p, end)` (tensor_accessor.h:568) returns the length of that span below `end` |

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `batch` (TILE: d0..d(n-3); ROW_MAJOR: folded with rows) | independent: destination element at batch index b reads only the source element at b | folded into `page` | folded | — | through `page` | — |
| `tile_row` (`ceil(H/32)`, TILE) | independent: same reason | folded into `page` | folded | — | through `page` | — |
| `tile_col` (`ceil(W/32)`, TILE) | independent: same reason | folded into `page` | folded | — | through `page` | — |
| `page` (flat page id `0 … tensor_pages − 1`) | independent: destination page p depends only on source page p | `block_pages` | the whole per-core assignment, `core_pages_max`, for every golden INPUTS shape (one block per core); L1-clamped otherwise (see Regimes) | host `_plan()` in `g6_copy_program_descriptor.py`, from the module knobs `L1_BUDGET_BYTES`, `MAX_BLOCK_PAGES`, `STREAM_DEPTH_BLOCKS`. Passed once to both kernels (common RT arg); CB capacity, block count and last-block extent derive from it | contiguous page-id ranges, one per core, over `num_cores = min(tensor_pages, pool_cores)` cores in row-major order; `pool_cores` = full grid (`MAX_CORES = None`) | knob-turn |
| `intra_page` (32 × 32 elements of a tile; W elements of a row) | independent, but the page is the addressing atom of the interleaved map | none: the whole page | whole page | mechanism (page granularity of the interleaved map) | not split | scheme-change: the deferred `sub_page_rows` row |

Folding is honest: the copy treats batch, tile_row and tile_col identically, both buffers store them
as one page sequence with the same page → bank map, and page ids are preserved. A rank-4 block over
(batch, tile_row, tile_col) would express nothing a page range does not.

### Knob decisions — one row per axis

| Axis | Character | Block factor (Phase 0) | Core assignment | Buffer depth | Why |
|------|-----------|------------------------|-----------------|--------------|-----|
| `batch` | independent | folded into `page` | folded into `page` | n/a | Contiguous inside the page sequence; a copy has no per-batch structure |
| `tile_row` | independent | folded into `page` | folded into `page` | n/a | Same |
| `tile_col` | independent | folded into `page` | folded into `page` | n/a | Same |
| `page` | independent | `block_pages = BLOCK_BYTES // page_bytes` (16 KiB: 4 fp32 / 8 bf16 tiles) when the per-core assignment exceeds it, else the whole assignment (one block per core); L1-clamped. Refinement 2 (measured) | every core of the grid: `split_work_to_cores(pool, tensor_pages, row_wise=True)`, balanced to ±1 page; grid read at runtime from `compute_with_storage_grid_size()`; core count capped only by the `MAX_CORES` knob | `cb_depth_blocks` = 1 with one block per core, `STREAM_DEPTH_BLOCKS` (2) with more | Every byte crosses DRAM exactly once whatever the block size, so the block sets fixed costs **and** read/write overlap. The coarsest block (one per core) leaves the writer idle until the whole assignment has landed (reader done at 38.5 µs, writer at 67.2 µs on the flagged shape), so Refinement 2 departs from the coarsest-block default on measurement: 16 KiB blocks at depth 2 take (16, 12, 128, 128) fp32 from 68.0 to 61.6 µs and (8, 12, 128, 128) from 32.9 to 30.8 µs (median of 5). Shapes whose assignment fits one block (≤ 4 fp32 pages per core) keep one hand-off. A second slot with one block per core is dead L1 |
| `intra_page` | independent | whole page, not split | not assigned | n/a | A split moves the same bytes in more, smaller transactions (catalog `tile_reorder`: whole-page transfers beat sub-tile faces) |

**Intermediate data.** The scheme has one stage (move) and no intermediate result. The only internal
data is the staged block in L1, whose only axis is `page`, assigned above. There is no coordinator
core: every used core runs identical work on its own range, and no stage computes on one core while
the others wait.

**Interleaved default, in order.** (1) Spread `page` over the full grid (110 cores here).
(2) Take the whole assignment as one block when it fits `BLOCK_BYTES` (16 KiB); otherwise stream
16 KiB blocks through a depth-2 ring so reads overlap writes (Refinement 2, measured; `BLOCK_BYTES = None`
restores the coarsest-block default).

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_page_staging` | `cb_depth_blocks`: 1 on the `one_block` plan; `min(STREAM_DEPTH_BLOCKS, L1_BUDGET_BYTES // (block_pages · page_bytes))` on the `streamed` plan, `STREAM_DEPTH_BLOCKS = 2` (module constant) | 1 where the assignment fits one `BLOCK_BYTES` block; 2 otherwise (fp32 (16|8, 12, 128, 128), bf16 (1, 1, 32, 62080), (1, 1, 8448, 128), …). Depth 3 measured: 60.7 µs vs 61.6 µs on (16, 12, 128, 128) but 31.8 vs 30.8 µs on (8, 12, 128, 128), so 2 kept | With ≥ 2 blocks per core, the reader fills block k+1 while the writer drains block k (reads on NoC0 overlapping writes on NoC1). With one block there is nothing to overlap |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| DRAM bank contiguity (tensor_accessor.h:560-575) | pages per transfer (a bank run) | `run = accessor.num_contiguous_pages(p, block_end)`, with `block_end ≤ core end ≤ tensor_pages` | A transfer runs past this block's pages in the bank: the reader overwrites another block's L1, the writer **overwrites another core's pages or a neighbouring allocation** |
| Accessor page size vs bank stride (tensor_accessor.h:515-521, 564-571) | address stride inside a run | Build both accessors with the default page size (the CT `AlignedPageSize`); CB page size = `buffer_aligned_page_size()` | `num_contiguous_pages` returns 1 (correct but uncoalesced); an L1 stride ≠ DRAM stride scrambles pages |
| Blackhole DRAM read alignment: L1 address ≡ DRAM address mod 64 (`noc-alignment-in-tt-metal.md` §1) | L1 offset of every run | Every run starts at `block_base + m · page_bytes`; `page_bytes` is a multiple of 64 and CBs are placed at DRAM alignment (program.cpp:2505-2509) | Silent corruption (the watcher halts the core under `--dev`) |
| L1 left to the op: 1347776 B per core (BB-05) | `cb_depth_blocks · block_pages · page_bytes` | the `one_block` predicate and the `streamed` clamp (Regimes); host refusal when `page_bytes > L1_BUDGET_BYTES` | Launch refused, or the model's L1-resident tensors clobbered |
| `noc_async_read` / `noc_async_write` size (dataflow_api.h:547, 876; documented range 0..1 MB; `NOC_MAX_BURST_SIZE` = 256 × 64 = 16384 B, noc_parameters.h:289-290) | bytes per call | none needed: run bytes ≤ `ceil(block_len / nk) · page_bytes` ≤ block bytes ≤ `L1_BUDGET_BYTES`. The documented 0..1 MB is the L1 address range; the multi-burst path loops over 16 KB bursts for any length (`ncrisc_noc_fast_read_any_len`, dataflow_api_raw.h:73) | — |
| Runs per block | transfers per block | `nk = min(num_banks, block_len)` runs, never more | — |
| CB ring wrap (blocking-model §2) | pushed / popped page counts | push and pop the **nominal** `block_pages` every block, short last block included; capacity `cb_depth_blocks · block_pages` | the CB pointers miss `fifo_limit` and wrap mid-block: pages land outside the slot the writer reads |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `page_copy / one_block`: flat page ranges per core, coalesced bank runs, one CB, reader NCRISC/NoC0 + writer BRISC/NoC1 | **built** | `core_pages_max · page_bytes ≤ L1_BUDGET_BYTES` and `core_pages_max ≤ block_target` (`block_target = min(BLOCK_BYTES // page_bytes, MAX_BLOCK_PAGES)`, each term skipped when `None`). Small-assignment shapes | `block_pages = core_pages_max`, one block per core, `cb_depth_blocks = 1` | **Minimum** against the DRAM boundary: each source page crosses once (read), each destination page once (write); 0 cross-core bytes | Per core: one `cb_reserve_back` / `cb_push_back`, one `noc_async_read_barrier`, one `cb_wait_front` / `cb_pop_front`, one write flush, `nk ≤ 8` read calls and `nk` write calls. Intended once per core. No inits or reconfigs exist (no compute) |
| `page_copy / streamed`: same kernels, several blocks per core | **built** (same code path; only the knob values differ) | not `one_block`, and `page_bytes ≤ L1_BUDGET_BYTES`. The default for every shape whose per-core assignment exceeds 16 KiB (Refinement 2); the acceptance test also forces it with `MAX_BLOCK_PAGES` | `block_pages = max(1, min(block_target or ∞, L1_BUDGET_BYTES // (STREAM_DEPTH_BLOCKS · page_bytes)))`; `cb_depth_blocks` per the depth knob | Minimum, same as above | Same fixed costs, paid once per block: `ceil(core_pages / block_pages)` times per core |
| `sub_page_rows`: split `intra_page`, moving one ROW_MAJOR row in byte pieces | **deferred**: reachable only for ROW_MAJOR rows with `page_bytes > L1_BUDGET_BYTES` (> 336,944 int32/uint32 or > 673,888 bfloat16 elements per row). No golden INPUTS or production row comes near (longest 5120 B), and ROW_MAJOR is outside Phase 0 SUPPORTED. Reachable later: `load_block` / `store_block` already move byte spans at an address, and a piece is a span plus an offset | `page_bytes > L1_BUDGET_BYTES` | a byte span of one page | Minimum bytes, more transactions | — |
| `compute_passthrough`: reader → `copy_tile` through DEST → pack → writer | rejected: superseded by `page_copy`, and illegal | — | — | Same DRAM bytes plus a second CB and unpack/pack L1 traffic per page. **Changes values**: 16-bit DEST or the TF32 srcA path cuts float32 mantissas, int32/uint32 travel a float path, bfloat8_b is re-packed — each a contract violation | — |
| `single_core` | rejected: superseded by `page_copy` | `num_cores = 1` with `tensor_pages > 1` | whole tensor | Same bytes, one core's NoC ceiling (~122 GB/s single-core Blackhole copy, catalog `trid_double_issue`) against the 460.8 GB/s roofline | — |
| `single_risc_copy`: one kernel reads, barriers, writes | rejected: superseded by `page_copy` | — | — | Same bytes; the writes leave on the reader's NoC0, the slow mirror of the measured default (catalog `noc_placement`: write·NoC0 ≈ read·NoC1, 2.5–4.8× slower for spread placements), and the `streamed` plan loses read/write overlap | — |
| `dram_to_dram_direct` | rejected: not expressible | — | — | A Tensix NoC read lands only in the issuing core's L1 (dataflow_api.h:526-530), so an L1 hop is structurally required | — |
| `split_dependent_axis_combine` | rejected: not applicable | — | — | No dependent axis: no destination element depends on more than one source element | — |
| `reuse_broadcast` (multicast a shared operand) | rejected: not applicable | — | — | Operand-reuse check below: each source page feeds exactly one destination page on one core | — |
| `sharded_zero_copy` (CB backed on a sharded buffer) | rejected: out of contract | — | — | TARGET has no memory_layout axis; the contract fixes DRAM interleaved for both tensors | — |

**Regime-selection function** (host `_plan()`, the single source for the dispatch and for
`describe_plan()`):

```text
pool_cores     = grid.x · grid.y if MAX_CORES is None else min(MAX_CORES, grid.x · grid.y)
num_cores      = min(tensor_pages, pool_cores)
core_pages_max = ceil(tensor_pages / num_cores)
if page_bytes > L1_BUDGET_BYTES:                               -> refuse on host (BB-05; sub_page_rows deferred)
elif core_pages_max · page_bytes ≤ L1_BUDGET_BYTES
     and (block_target is None or core_pages_max ≤ block_target):   # block_target = min(BLOCK_BYTES // page_bytes, MAX_BLOCK_PAGES)
                                                               -> one_block: block_pages = core_pages_max, depth 1
else:                                                          -> streamed:  block_pages, depth as in the table
```

Since Refinement 2, shapes whose per-core assignment exceeds `BLOCK_BYTES` select `streamed` by default;
the rest stay on `one_block`. Forced-knob tests still pin `streamed` at odd block sizes (Work Distribution).

### Traffic ranking

| Candidate split | DRAM bytes (perf case) | Cross-core bytes | NoC calls per core (perf case, fp32) | Verdict |
|-----------------|------------------------|------------------|--------------------------------------|---------|
| A. `page`, flat contiguous ranges, coalesced bank runs | 12,582,912 read + 12,582,912 write (minimum) | 0 | 8 reads + 8 writes of 12–16 KB (one burst each) | **chosen** |
| B. `page`, bank-affine ranges (each core takes contiguous slots of one bank) | same | 0 | 1–2 reads + 1–2 writes of ≤ 112 KB (7 bursts each) | perf lamp (catalog `read_coalescing`: affinity after coalescing 0.91–1.28×, two longer cases regress) |
| C. `page`, flat ranges, one transfer per page | same | 0 | 28 + 28 of 4 KB | dominated by A (catalog `read_coalescing`: coalescing alone is the default) |
| D. `page`, round-robin page assignment | same | 0 | 28 + 28, no run longer than one page | dominated by A |
| E. `intra_page` split (faces, row pieces) | same | 0 | ≥ 4× A | rejected (catalog `tile_reorder`) |
| F. no split | same | 0 | 3072 + 3072 on one core | rejected on occupancy, not on traffic |
| G. dependent axis + combine | n/a: no dependent axis | — | — | not applicable |

Every candidate ties at the minimum: 2 × tensor bytes across DRAM, 0 cross-core. Transaction shape
decides among them. A issues `block_len / min(8, block_len)` times fewer calls than C (3.5× in the
perf case, 1× for blocks of ≤ 8 pages), each call one sequential span inside one bank, and each
core's calls spread across all 8 banks. The ratio grows with pages per core, which is where
coalescing matters most (64 B ROW_MAJOR rows). Occupancy (full grid in A) is accounted for
separately.

### Block schedule

```cpp
// Per core (RT): core_first_page, core_pages, walk_rotation. Common: buffer address, block_pages.
num_blocks_this_core = ceil(core_pages / block_pages);
last_block_pages     = core_pages - (num_blocks_this_core - 1) * block_pages;
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_block(block_idx);    // reader kernel
    store_block(block_idx);   // writer kernel
}
drain_writes();               // writer kernel, once
```

| Block operation | Kernel | Block shape | Acts on | Resident across it | Intended frequency of fixed-cost work |
|-----------------|--------|-------------|---------|--------------------|---------------------------------------|
| `load_block` | reader (NCRISC, NoC0) | `block_pages` pages of `page` (actual `block_len = last_block_pages` on the last block) | source DRAM → `cb_page_staging` | the block's pages stay in `cb_page_staging` until `store_block` pops them | `cb_reserve_back(block_pages)` once per block; `nk` `noc_async_read` calls (one per bank run); one `noc_async_read_barrier` per block; `cb_push_back(block_pages)` once per block, **nominal** count on a short last block |
| `store_block` | writer (BRISC, NoC1) | same | `cb_page_staging` → destination DRAM | — | `cb_wait_front(block_pages)` once per block; `nk` `noc_async_write` calls; one `noc_async_writes_flushed` per block before the pop (the slot may be refilled); `cb_pop_front(block_pages)` once per block |
| `drain_writes` | writer | — | — | — | one `noc_async_write_barrier` per kernel (BB-03) |

**Bank-run walk.** One header, `kernels/g6_copy_page_walk.hpp`, used by both kernels so the L1
layout of a block is identical on both sides. For a block of `block_len` pages starting at page
`block_first = core_first_page + block_idx · block_pages`, with `block_end = block_first + block_len`:

```text
nk = min(accessor.contiguous_page_stride(), block_len)
l1 = block_base                                   // get_write_ptr (reader) / get_read_ptr (writer)
for i in [0, nk):
    k   = (walk_rotation + i) mod nk
    p   = block_first + k
    run = accessor.num_contiguous_pages(p, block_end)      // pages p, p+nb, … < block_end
    transfer(accessor.get_noc_addr(p), l1, run · page_bytes)   // read: DRAM→L1; write: L1→DRAM
    l1 += run · page_bytes
```

Every page of the block falls in exactly one run (`q ≡ block_first + k (mod num_banks)` for exactly
one k < nk), so each page is read once and written once. The L1 position of a page follows walk
order, not page order; that is invisible to the result because the writer walks identically.
`walk_rotation` is a per-core RT arg: the core's linear index under the knob `BANK_STAGGER = True`
(the Phase 0 default, set by the implementer on measurement: (16, 12, 128, 128) fp32 one_block
74.7 → 67.6 µs, (8, 12, 128, 128) 34.5 → 32.6 µs, medians of 3), 0 with the knob off.

### Stall-shadow check

| Stage that waits | Waits on | Work independent of the wait | Decision |
|------------------|----------|------------------------------|----------|
| reader: `noc_async_read_barrier` | the DRAM reads of its own block | none on this core: every read of the block is already issued; with ≥ 2 blocks the next block's reads need a free slot, which depth 2 provides | nothing to fill |
| writer: `cb_wait_front(block_pages)` | the reader's push of the whole block | writing runs of the block that already landed | **not filled in Phase 0.** DRAM is the bottleneck and is busy with every core's reads during this wait, so the wait costs pipeline fill (one DRAM round trip per core) plus the read → write turnover. The reorder is a knob-turn (`MAX_BLOCK_PAGES`, `STREAM_DEPTH_BLOCKS`; CB capacity unchanged), recorded as the overlap lamp. Pages are independent, so no algebraic permission is needed; there is no arithmetic, so no floating-point order changes |
| writer: `noc_async_write_barrier` | acks of its own writes | none: last stage | — |

### Operand-reuse check

| Operand | Varies along the split axis (`page`)? | Reuse-shared by construction? |
|---------|--------------------------------------|-------------------------------|
| `input_tensor` | yes: each core reads only its own pages | no |
| `output_tensor` | yes: each core writes only its own pages | no |

No broadcast regime exists; `reuse_broadcast` is rejected as not applicable.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| overlap | With one block per core the writer starts only after the whole assignment (up to 28 pages, 112 KB) has landed; all cores' read and write phases roughly align, leaving a turnover bubble and a full-block drain at the end | `MAX_BLOCK_PAGES = ceil(core_pages_max / k)`, `STREAM_DEPTH_BLOCKS = k`, k ∈ {2, 4} (CB capacity unchanged, the reader never waits on reserve). Optionally per-block transaction ids (`dataflow_kernel_lib::set_read_trid` / `async_read_barrier_with_trid`, local_copy_helpers_dataflow.hpp:147, 158; catalog `trid_double_issue`) so block k+1's reads stay on the wire across block k's hand-off |
| grid synchronization | Small production cases spread thin: (1, 12, 128, 128) is 192 pages, 1–2 pages (4–8 KB) per core on 110 cores, where launch and one DRAM round trip per core may dominate | `MAX_CORES` so `core_pages ≥ 8` (24 cores for 192 pages, 48 for 384), against the full grid |
| bank stagger | In the perf case core c starts at page 28c, so `28c mod 8 ∈ {0, 4}`: every core issues its first run to bank 0 or bank 4 (catalog `bank_stagger`: 1.04–1.07× for one block per core on interleaved sources) | `BANK_STAGGER = True` (per-core `walk_rotation` = linear core index; a host-only change) |
| bank affinity | Each core spreads its runs over all 8 banks; giving each core contiguous slots of one bank concentrates its traffic on one DRAM endpoint and makes runs up to 7 bursts long | Candidate B of the traffic ranking: per-core ranges in bank-major order (catalog `read_coalescing` "bank_affinity", 0.91–1.28× after coalescing). Changes the host split and the walk's page enumeration, not the block schedule |

Not lamps here:
- **split_reader:** the reader issues ≤ 8 calls per block, so it is never issue-bound (catalog
  `split_reader`: "does nothing unless a data-movement RISC-V is itself the bottleneck").
- **NoC selection / placement:** reads on NoC0, writes on NoC1, `row_wise=True` placement is the
  measured fast default (catalog `noc_placement`).

**Catalog entries behind each knob.**

| Knob | Entries |
|------|---------|
| `block_pages` | `double_buffer` (issue a batch, then one barrier; never read-one/barrier), `trid_double_issue` ("at a large CB the trivial one-block loop is within 3% of the best number") |
| `cb_depth_blocks` | `double_buffer` ("no gain once DRAM-bandwidth-bound" with enough cores) |
| core assignment | `noc_placement` (`row_wise=True`), `bank_stagger`, `read_coalescing` (affinity) |
| transfer granularity | `read_coalescing` (coalesce contiguous pages into one call), `tile_reorder` (whole pages) |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| source DRAM → L1 | the tensor's own page bytes: fp32 / bf16 / bfp8 / int32 / uint32 tile, or a ROW_MAJOR row with its alignment tail | reader `noc_async_read` on NoC0, one call per bank run, at `src_accessor.get_noc_addr(p)` | no format conversion anywhere on the path |
| L1 staging | same bytes, `page_bytes` slots in walk order | `cb_page_staging`, reader → writer handshake | the CB `data_format` is the tensor dtype only because the descriptor requires one; no unpacker or packer touches it |
| L1 → destination DRAM | same bytes | writer `noc_async_write` on NoC1, one call per bank run, at `dst_accessor.get_noc_addr(p)` | destination page p receives exactly source page p's bytes; a ROW_MAJOR alignment tail lands inside the destination's own page slot |

- **Inter-Tensix:** none, and no deferred regime needs it.
- **Multi-device:** `generic_op` runs the same program SPMD on every device of the tensors' mesh
  (generic_op.cpp:13-23). Each device copies its own buffer; shapes are per device.

### TARGET axes as regime rows

| TARGET value | Cost class | What changes |
|--------------|-----------|--------------|
| dtype bfloat16 / bfloat8_b / int32 / uint32 (TILE) | knob-turn | nothing in the kernels: `page_bytes` comes from `buffer_aligned_page_size()` (2048 / 1088 / 4096 / 4096). Add the value to SUPPORTED and drop the matching EXCLUSIONS entry |
| layout ROW_MAJOR (bf16, int32, uint32) | knob-turn | page = row, `page_bytes` = aligned row (≥ 64 B); same kernels, same walk. Rows of 4–128 B move as 64–128 B aligned slots, coalesced along bank runs |
| alignment `w_non_aligned` / `h_non_aligned` | built in Phase 0 | whole-page copy; padding goes to padding |
| memory_layout | not a TARGET axis | — |

**Structural impossibilities:** none beyond `feature_spec.INVALID` (bfloat8_b × ROW_MAJOR).

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one block of `block_pages` consecutive page ids |
| Grid | `device.compute_with_storage_grid_size()` (11 × 10 = 110 here). Core pool: the grid when `MAX_CORES is None`, else `ttnn.num_cores_to_corerangeset(min(MAX_CORES, grid cores), grid, row_wise=True)`. Split: `ttnn.split_work_to_cores(pool, tensor_pages, row_wise=True)` |
| Per-core work | page range `[core_first_page, core_first_page + core_pages)`; `core_pages` ∈ {g1, g2} (differ by 1); `core_first_page` is the running sum over group-1 cores then group-2 cores in row-major order (`ttnn.corerange_to_cores(group, None, True)`) |
| Remainder | `split_work_to_cores` balances to ±1 page and per-core counts are RT args, so there is no tail core. `tensor_pages` comes from `buffer_num_pages()` (per-image ceil in TILE) |

| Phase 0 shape (fp32 TILE) | `tensor_pages` | `num_cores` | `core_pages` (value × cores) | plan | `block_pages` | runs per block | CB bytes per core |
|---------------------------|---------------|-------------|------------------------------|------|---------------|----------------|-------------------|
| (16, 12, 128, 128) | 3072 | 110 | 28 × 102, 27 × 8 | streamed (R2) | 4, depth 2 | 4 (4 KB) | 32,768 |
| (8, 12, 128, 128) | 1536 | 110 | 14 × 106, 13 × 4 | streamed (R2) | 4, depth 2 | 4 (4 KB) | 32,768 |
| (2, 12, 128, 128) | 384 | 110 | 4 × 54, 3 × 56 | one_block | 4 | 4 (4 KB) | 16,384 |
| (1, 12, 128, 128) | 192 | 110 | 2 × 82, 1 × 28 | one_block | 2 | 2 (4 KB) | 8,192 |
| (1, 1) | 1 | 1 | 1 × 1 | one_block | 1 | 1 | 4,096 |
| (1, 1, 1, 256) (h_non_aligned, test-only in fp32) | 8 | 8 | 1 × 8 | one_block | 1 | 1 | 4,096 |

Split figures for 3072 and 192 pages were probed on this board. Every other golden INPUTS shape
(bf16, bfp8, RM) has ≤ 1940 pages and also selects `one_block` (largest: bfloat16 (1, 1, 32, 62080),
1940 tiles → 18 × 2048 = 36,864 B per core).

- **Core ranges (BB-17):** one kernel per role over `all_cores`, per-core RT args. `all_cores` is the
  full grid or full rows plus one partial row: ≤ 2 rectangles for every shape.
- **Regime selection:** the function in Regimes. `describe_plan()` reports it (BB-08), and every
  production shape reports the default plan (`one_block` when the assignment fits `BLOCK_BYTES`, else `streamed`).
- **Regime-pinned tests:** the `streamed` plan (several blocks per core, ring wrap, ragged last
  block, blocks shorter than `num_banks`) and a capped grid are reached by no production shape, so the
  acceptance test forces them through the module knobs on production shapes, with and without
  `BANK_STAGGER`.

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|----------|
| `cb_page_staging` | 0 | `page_bytes` | `cb_depth_blocks · block_pages` | spans `page` (`block_pages × cb_depth_blocks`); batch / tile_row / tile_col folded into `page`; `intra_page` lies inside one page slot | the tensor dtype (never interpreted) | reader | writer | whole kernel |

The only CB and the program's whole L1 footprint. Full ledger: `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_block` | `block_pages` pages (actual `block_len` ≤ `block_pages`) | no: raw dataflow (API Mapping) | — (source DRAM through the TensorAccessor) | `cb_page_staging`, reserve and push `block_pages` | `block_pages` pushed; only the first `block_len` slots hold pages on a short last block |
| 2 | `store_block` | same | no: raw dataflow | `cb_page_staging`, wait `block_pages` | — (destination DRAM through the TensorAccessor) | `block_pages` popped after the flush |
| 3 | `drain_writes` | — | no | — | — | every write acked |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `load_block` | raw_api | `cb_reserve_back` | tt_metal/hw/inc/api/dataflow/dataflow_api.h:406 | `(cb_page_staging, block_pages)` | — | `cb_page_staging` | `block_pages` |
| `load_block` | raw_api | `get_write_ptr` | dataflow_api.h:325 | `(cb_page_staging)` | — | `cb_page_staging` | — |
| `load_block` | raw_api | `TensorAccessor` (interleaved), default page size | tt_metal/hw/inc/api/tensor/tensor_accessor.h:515-521 | `(TensorAccessorArgs<src CT offset>(), src_addr)` | — | — | — |
| `load_block` | raw_api | `TensorAccessor::get_noc_addr` | tensor_accessor.h:554 | `(p)` | — | — | — |
| `load_block` | raw_api | `contiguous_page_stride` / `num_contiguous_pages` | tensor_accessor.h:562, 568 | `()` → `num_banks`; `(p, block_end)` → run | — | — | run bound (`block_end`) |
| `load_block` | raw_api | `noc_async_read` | dataflow_api.h:547 | `(noc_addr, l1, run · page_bytes)` | — | `cb_page_staging` | run |
| `load_block` | raw_api | `noc_async_read_barrier` | dataflow_api.h:1921 | `()` | — | — | — |
| `load_block` | raw_api | `cb_push_back` | dataflow_api.h:208 | `(cb_page_staging, block_pages)` | — | `cb_page_staging` | `block_pages` |
| `store_block` | raw_api | `cb_wait_front` | dataflow_api.h:479 | `(cb_page_staging, block_pages)` | `cb_page_staging` | — | `block_pages` |
| `store_block` | raw_api | `get_read_ptr` | dataflow_api.h:346 | `(cb_page_staging)` | `cb_page_staging` | — | — |
| `store_block` | raw_api | destination `TensorAccessor`, `get_noc_addr`, contiguity API | tensor_accessor.h:515-521, 554, 562, 568 | same walk | — | — | run bound |
| `store_block` | raw_api | `noc_async_write` | dataflow_api.h:876 | `(l1, noc_addr, run · page_bytes)` | `cb_page_staging` | — | run |
| `store_block` | raw_api | `noc_async_writes_flushed` | dataflow_api.h:1980 | `()` | — | — | — |
| `store_block` | raw_api | `cb_pop_front` | dataflow_api.h:260 | `(cb_page_staging, block_pages)` | `cb_page_staging` | — | `block_pages` |
| `drain_writes` | raw_api | `noc_async_write_barrier` | dataflow_api.h:1951 | `()` | — | — | — |
| all (BB-21) | raw_api | `DeviceZoneScopedN` | tt_metal/hw/inc/api/debug/kernel_profiler.hpp:1012 (no-op in the plain build, :1047) | zone name | — | — | — |

**Helpers considered and rejected** (covers every `raw_api` row; there is no compute phase):

| Candidate | File:Line | Why it cannot express the operation |
|-----------|-----------|-------------------------------------|
| `compute_kernel_lib::copy` (CopyTile → PackTile) | ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp:109-113 | Routes every value through unpack → DEST → pack: with 16-bit DEST a float32 loses mantissa bits, int32/uint32 travel a float path, bfloat8_b is re-packed — each forbidden by the contract. The op has no compute phase; a compute kernel would add a CB and three threads for zero work |
| `dataflow_kernel_lib::read_with_state` / `set_read_state` | ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:75, 95; `.inl`:66 | Self-aimed L1 → L1 reads: the source is built by `local_addr()` (:57) and a `static_assert` caps `transfer_size ≤ NOC_MAX_BURST_SIZE` (.inl:66). They cannot source DRAM pages or a multi-burst run |
| `dataflow_kernel_lib::set_read_trid` / `async_read_barrier_with_trid` | local_copy_helpers_dataflow.hpp:147, 158 | Usable (they tag any read batch) but buy nothing with one block per core, where one barrier exists. They are the overlap lamp's mechanism, not Phase 0's |
| tilize / untilize / reduce / matmul / eltwise chain / mcast helpers | — | A copy has no layout change, reduction, product, element-wise math or multi-receiver transfer |

No compute kernel exists, so `compute_kernel_hw_startup()` does not apply.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Reader and writer walk a block differently | The L1 layout is walk order, not page order; a mismatch writes page p's bytes to page q | One shared walk header; both kernels get the same `block_pages`, `core_first_page`, `walk_rotation`; forced tests run `BANK_STAGGER = True` |
| A run overruns its block | Contiguity holds across the whole bank, not only inside the block | `num_contiguous_pages(p, block_end)` with the block's own end, never `tensor_pages` |
| Stale buffer address on a host-cache hit | The descriptor is cached per spec; buffer addresses change per call | Addresses live only in common RT arg 0 of each kernel and are patched in place on every call (Appendix B); `test_g6_copy_rebinds_buffers` |
| Returning `generic_op`'s wrapper instead of the caller's object | The test checks `out is output_tensor`; the model reads its own object | `return output_tensor` after the dispatch |
| Push count on a short last block | Per-core counts are ragged (28 vs 27) and the streamed plan has tails | Reader always pushes `block_pages`, writer waits on and pops `block_pages`; only the walk uses `block_len`. Capacity `cb_depth_blocks · block_pages` closes the ring |
| The writer pops before its data left L1 | On the streamed plan the reader refills the slot | `noc_async_writes_flushed()` before `cb_pop_front` |
| NoC writes in flight at kernel exit | Another program starts right after the call (BB-03) | `noc_async_write_barrier()` at the writer's end; the reader ends after its last read barrier |
| Any compute / unpack path | float32 bits, int32 values and bfp8 groups would change | No compute kernel; the CB format is never interpreted |
| Accessor built with a non-aligned page size | `num_contiguous_pages` returns 1 and an arithmetic stride goes wrong | Default page size (CT `AlignedPageSize`); CB page = `buffer_aligned_page_size()` |
| Accessor CT args collide with the op's CT args | Both are positional | Op CT args first (`cb_page_staging` index), `TensorAccessorArgs(...).get_compile_time_args()` appended last; the accessor's CT offset = number of op CT args. DRAM interleaved emits no common RT args (probed), so the op's common RT args start at 0 |
| Hard-coded 110 cores or 8 banks | This board's grid is overridden to 11 × 10; other boards differ | Grid from `compute_with_storage_grid_size()`; `num_banks` only on device via `contiguous_page_stride()` |
| Python cache keyed on a per-call wrapper | `tensor.device()` may return a fresh wrapper, missing the cache every call (BB-07) | Key on a stable device identity (`device.id()`), never `id()` of the wrapper |

## Appendix A — Entry point and registry contract

| Item | Specification |
|------|---------------|
| Files | `ttnn/ttnn/operations/g6_copy/__init__.py` (exports `g6_copy`, `INPUT_TAGGERS`, `SUPPORTED`, `EXCLUSIONS`, `validate`, `describe_plan`); `g6_copy.py` (registry, `validate`, entry point); `g6_copy_program_descriptor.py` (knob constants `L1_BUDGET_BYTES = 1347776`, `MAX_CORES = None`, `BLOCK_BYTES = 16384` (Refinement 2), `MAX_BLOCK_PAGES = None`, `STREAM_DEPTH_BLOCKS = 2`, `BANK_STAGGER = True` (measured lamp, see Bank-run walk); `_plan()`; descriptor builder and cache; `describe_plan()`); `kernels/g6_copy_reader.cpp`, `kernels/g6_copy_writer.cpp`, `kernels/g6_copy_page_walk.hpp` |
| `INPUT_TAGGERS` | `{"alignment": tag_alignment}`. On the shared shape: `W = shape[-1]`, `H = shape[-2]` (H = 1 for rank 1). `"w_non_aligned"` if `W % 32`, else `"h_non_aligned"` if `H % 32`, else `"tile_aligned"` |
| `SUPPORTED` | Refinement 1: `{"dtype": [ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.int32, ttnn.uint32], "layout": [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], "alignment": [...]}`; Phase 0 was `{"dtype": [ttnn.float32], "layout": [ttnn.TILE_LAYOUT], "alignment": ["tile_aligned", "w_non_aligned", "h_non_aligned"]}` |
| `EXCLUSIONS` | `[{"dtype": ttnn.float32, "layout": ttnn.ROW_MAJOR_LAYOUT}, {"dtype": ttnn.int32, "layout": ttnn.TILE_LAYOUT}, {"dtype": ttnn.uint32, "layout": ttnn.TILE_LAYOUT}]` |
| `validate(input_tensor, output_tensor)` | (1) Malformed-request checks, raising `ValueError`, in this order: both tensors on device (`ttnn.is_tensor_storage_on_device`); rank 1–4; shapes equal; dtypes equal; layouts equal; memory configs equal; memory DRAM interleaved. (2) Axes from `input_tensor`: dtype, layout, then the taggers. (3) SUPPORTED → `UnsupportedAxisValue`. (4) EXCLUSIONS → `ExcludedCell`. INVALID is never checked. Every message starts with `g6_copy:` (the acceptance test matches it) |
| `g6_copy(input_tensor, output_tensor)` | `validate(...)` is the first line. Then fetch the cached descriptor (build on a miss), patch the two buffer addresses, call `ttnn.generic_op([input_tensor, output_tensor], descriptor)` (output last), and `return output_tensor` — the caller's object. Exactly one device program per call; no host-side tensor transformation |

## Appendix B — Host program cache (BB-07, BB-16)

| Item | Specification |
|------|---------------|
| Key | `(device.id(), grid (x, y), tuple(shape), dtype, layout, tensor_pages, page_bytes, L1_BUDGET_BYTES, MAX_CORES, BLOCK_BYTES, MAX_BLOCK_PAGES, STREAM_DEPTH_BLOCKS, BANK_STAGGER)`. Knob constants are in the key so a perf pass or a forced-knob test builds a new entry instead of replaying a wrong program. Read them from the module at call time, never captured at import or as default arguments: the acceptance test overrides them with `monkeypatch.setattr` |
| Value | the built `ttnn.ProgramDescriptor` and the indices of the reader and writer in `descriptor.kernels` |
| Per call | only the two buffer addresses change: `descriptor.kernels[reader].common_runtime_args[0] = input_tensor.buffer_address()` and `descriptor.kernels[writer].common_runtime_args[0] = output_tensor.buffer_address()`. Probed on host this run: `pd.kernels[i]` is a reference into the descriptor and `common_runtime_args[j] = v` mutates it in place. Per-core RT args, `block_pages`, CT args and the CB size all derive from the key |
| Device-side cache | `generic_op` hashes RT-arg counts, not values (`compute_program_descriptor_hash`, generic_op_device_operation.cpp:78-147), so the compiled program is reused across calls and buffers; on a hit the ProgramDescriptor path re-applies the descriptor's runtime args (mesh_device_operation_adapter.hpp:345-348) |
| Trace (BB-16) | nothing is allocated per call; the first call (before capture) builds the descriptor and kernels; replays reuse the captured addresses |

## Appendix C — Plan and role reporting (BB-08, BB-21)

| Item | Specification |
|------|---------------|
| `describe_plan(input_tensor) -> dict` | host-only, deterministic, same `_plan()` as the dispatch. Keys: `regime` (`"page_copy"`), `plan` (`"one_block"` / `"streamed"`), `grid` (`(x, y)`), `num_cores`, `tensor_pages`, `page_bytes`, `core_pages` (list, row-major over used cores), `block_pages`, `cb_depth_blocks`, `cb_pages`, `l1_bytes_per_core` (= `cb_pages · page_bytes`), `bank_stagger`, `roles` (dict keyed `"(x,y)"` for every grid core: `"copy pages [a,b): reader NCRISC/NoC0 + writer BRISC/NoC1"` for used cores, `"idle"` otherwise) |
| Zones (BB-21) | `DeviceZoneScopedN`, one zone per wait and per piece of work, never shared. Reader: `G6C-RD-RESERVE` (wait), `G6C-RD-ISSUE` (work), `G6C-RD-BARRIER` (wait), `G6C-RD-PUSH` (work). Writer: `G6C-WR-WAIT` (wait), `G6C-WR-ISSUE` (work), `G6C-WR-FLUSH` (wait), `G6C-WR-POP` (work), `G6C-WR-DRAIN` (wait). No-op in the plain build |
| Kernel config (BB-04, BB-18) | two small dataflow kernels; RT args: 2 common + 3 per core per kernel; CT args: 1 + 2 accessor. Must fit the kernel-config region in the plain, zone and NoC-trace builds without a size-optimising build |

## Appendix D — Acceptance test coverage (BB-22)

| Test | Cases | Why it is required |
|------|-------|--------------------|
| `test_g6_copy_values` | the 5 float32 production shapes, plus (1, 1, 1, 256) in float32 | Production rows. (1, 1, 1, 256) is the only `h_non_aligned` and non-square float32 case; SUPPORTED claims `h_non_aligned`, and the native-handling rule requires it to run |
| `test_g6_copy_adversarial_bits` | (2, 12, 128, 128) with NaN payloads, ±inf, ±0, denormals, extremes | the contract's bit-identity gate |
| `test_g6_copy_poisoned_padding` | (1, 1) and (1, 1, 1, 256), source padding = 1000.0 | no logical destination element may depend on source padding |
| `test_g6_copy_rebinds_buffers` | (1, 12, 128, 128), two live tensor pairs | BB-07 (cache hit with new addresses writes its own destination only) and BB-06 (repeat is bit-identical) |
| `test_g6_copy_streamed_plan` | (16, 12, 128, 128) with `MAX_BLOCK_PAGES = 3` on the full grid, `BANK_STAGGER` off and on (blocks shorter than `num_banks`, 1-page runs, ragged 1-page last block); (1, 12, 128, 128) with `MAX_CORES = 3`, `MAX_BLOCK_PAGES = 11`, `BANK_STAGGER` on (64 pages per core: multi-page runs, a capped grid, a ragged 9-page last block) | pins the `streamed` plan: multi-block, ring wrap, ragged last block, `nk < num_banks` and `nk = num_banks`, rotated walk |
| `test_g6_copy_dtypes_layouts.py` (Refinement 1) | the non-float32 production rows (bf16 / bf8b TILE, bf16 / int32 / uint32 RM, incl. shortest RM row `(1,)`), adversarial bits per dtype, bf16 poisoned padding (1, 1, 1, 256), `describe_plan` per row | values graded by `helpers.TOLERANCES`; plan deterministic, full-grid split, L1 ≤ 1347776 |
| `test_g6_copy_plan` | the Phase 0 shapes | BB-08 (plan reported, deterministic, production shapes on the default plan) and BB-05 (L1 ≤ 1347776) |
| `test_g6_copy_rejects_*` | shape, dtype, layout and memory-config mismatches; rank 5; host tensor | the spec's Validation list |
| `test_g6_copy_trace_replay` (verifier) | (1, 12, 128, 128) and (16, 12, 128, 128), eager call then capture; replay into a zeroed destination, then replay after a new source is written in place | BB-16: a replay equals the eager call bit for bit and copies the source's current contents |
| `test_g6_copy_precision_baseline` (verifier) | (1, 1), (1, 12, 128, 128), (8, 12, 128, 128), (16, 12, 128, 128) | PCC, abs / relative-RMS error, ULP and got/true ratio recorded for the verification report; gated on bit equality |
