# Operation Design: g6_copy

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement: two dataflow kernels, **no compute kernel** |
| Goal | Copy every page of a DRAM-interleaved source into the caller's DRAM-interleaved destination of identical spec, in one device program at DRAM roofline, and return the destination object itself. Perf target (LOOSE_CASES[0]): float32 TILE (16, 12, 128, 128) < 68.56 µs device kernel time; roofline 54.61 µs. |
| Math | `output_tensor[i] = input_tensor[i]` for every logical index i, realized as `dst_page[p] = src_page[p]`, byte for byte, for every buffer page p |
| References | Contract: the g6_copy spec (user prompt). `eval/golden_tests/g6_copy/feature_spec.py`, `helpers.py`. `.claude/references/blocking-model.md`, `l1-footprint-discipline.md`, `noc-alignment-in-tt-metal.md`. `tt_metal/hw/inc/api/tensor/tensor_accessor.h` (interleaved contiguity API). Catalog `ttnn/ttnn/operations/examples/master.md`: `noc_placement`, `double_buffer`, `trid_double_issue`, `read_coalescing`, `bank_stagger`, `tile_reorder`, `split_reader`. The exploration bundle `docs/explorations/g6_copy/` is not installed on this branch. |

**Why a page copy is the element copy.** The two tensors have the same shape, dtype, layout and
memory config, so they have the same page size, page count, page → (bank, offset) map and element
→ (page, byte) map. Copying page p verbatim moves every logical element to its own position. A
whole-tile copy never moves a byte across positions, so the source's tile padding lands only in the
destination's tile padding. No logical destination element depends on source padding, and the three
alignment tags need no algorithm of their own. A bfloat8_b tile (mantissas and shared exponents in
one 1088 B page) moves as raw bytes, which is bit-identical and stronger than the one-step gate. No
value passes through an unpacker, DEST or a packer, so no representation change can happen.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | rank 1–4, on device, DRAM interleaved, a SUPPORTED cell | — | buffer address: reader common RT arg |
| `output_tensor` | `ttnn.Tensor` | yes | same logical shape, dtype, layout, memory_config as `input_tensor` | — | buffer address: writer common RT arg |

No other parameter. No dtype, memory_config or compute_kernel_config.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `(d0, …, dn)`, rank 1–4 |
| Dtype | Phase 0: float32. TARGET adds bfloat16, bfloat8_b, int32, uint32 |
| Layout | Phase 0: TILE. TARGET adds ROW_MAJOR (bfloat16, int32, uint32) |
| Memory | DRAM interleaved |

### Output

| Property | Value |
|----------|-------|
| Shape | equal to the input shape (validated, never derived) |
| Dtype | equal to the input dtype |
| Layout | equal to the input layout |
| Memory | DRAM interleaved. The object passed as `output_tensor`, on the same buffer; the op allocates nothing |

## Blocking Model

### Page index space

| Term | Definition |
|------|-----------|
| `tensor_pages` | `input_tensor.buffer_num_pages()` (equals the output's). TILE: `prod(batch dims) · ceil(H/32) · ceil(W/32)`, padded per image (H = 1 for rank 1). ROW_MAJOR: number of rows. Never `floor(volume / 1024)`. |
| `page_bytes` | `input_tensor.buffer_aligned_page_size()`. float32 tile 4096, bfloat16 tile 2048, bfloat8_b tile 1088, int32/uint32 tile 4096; ROW_MAJOR row `round_up(W · elem, 64)` on Blackhole (a 4 B uint32 row occupies 64 B). |
| `num_banks` | DRAM banks of the interleaved allocator. Read on device as `TensorAccessor::contiguous_page_stride()` (= firmware `NUM_DRAM_BANKS`, tensor_accessor.h:562); 8 on this board. **Host code never uses it.** |
| bank-major order `q` | Pages enumerated bank by bank. Page id `p = b + num_banks · j` (bank b, slot j) has `q = Σ_{b' < b} R_{b'} + j`, where `R_b = (tensor_pages − b + num_banks − 1) / num_banks` for `b < tensor_pages`, else 0. Within a bank, consecutive slots sit `page_bytes` apart (tensor_accessor.h:560-575), so **a q-range inside one bank is one contiguous byte span in DRAM**, at the same offsets in both tensors. |

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `batch` (d0..d_{n-3} TILE; d0..d_{n-2} RM) | independent: a destination element at batch index b reads only the source element at b | folded into `page` | (folded) | — | through `page` | — |
| `tile_row` (`ceil(H/32)`, TILE) | independent: same reason | folded into `page` | (folded) | — | through `page` | — |
| `tile_col` (`ceil(W/32)`, TILE) | independent: same reason | folded into `page` | (folded) | — | through `page` | — |
| `page` (bank-major q, `0 … tensor_pages−1`) | independent: destination page p depends only on source page p | `block_pages` | `min(core_pages_max, stream_block_cap)`. For every golden INPUTS shape this is the whole per-core assignment (one block per core) | host `_plan()` in `g6_copy_program_descriptor.py`, computed once per cache key from the module constants. Passed to both kernels as one common RT arg; CB size, trip counts and the last-block extent derive from it | contiguous q-ranges, one per core, over `num_cores = min(tensor_pages, max_cores)` cores in row-major order; `max_cores` = full grid in Phase 0 | knob-turn |
| `intra_page` (32×32 elements of a tile; W elements of a row) | independent, but the page is the interleaved addressing atom | none: the whole page | whole page | mechanism (page granularity of the interleaved map) | not split | scheme-change: the deferred `sub_page_rows` row |

Folding is honest here. The copy treats batch, tile_row and tile_col identically. Both buffers store
them as one page sequence with the same page → bank map, and the copy preserves page ids. A rank-4
block over (batch, tile_row, tile_col) would express nothing that a page range does not.

### Knob decisions — one row per axis

| Axis | Character | Block factor (Phase 0) | Core assignment | Buffer depth | Why |
|------|-----------|------------------------|-----------------|--------------|-----|
| `batch` | independent | folded into `page` | folded into `page` | n/a | Contiguous inside the page sequence; a copy has no per-batch structure |
| `tile_row` | independent | folded into `page` | folded into `page` | n/a | Same |
| `tile_col` | independent | folded into `page` | folded into `page` | n/a | Same |
| `page` | independent | `block_pages` = the whole per-core assignment, L1-clamped | every core of the grid (`MAX_CORES = None`): contiguous bank-major q-ranges, balanced to ±1 page, grid read at runtime | `cb_depth_blocks = min(CB_DEPTH_BLOCKS = 2, num_blocks_max, L1_BUDGET_BYTES // (block_pages · page_bytes))`, which is 1 for every golden shape | Every byte crosses DRAM exactly once whatever the block size, so the block only sets fixed costs. The coarsest block gives one hand-off, one read barrier and one write flush per core. Depth beyond the blocks that actually exist is dead L1 |
| `intra_page` | independent | whole page, not split | not assigned | n/a | A split moves the same bytes in more transactions (catalog `tile_reorder`: whole-page transfers beat sub-tile faces) |

**Intermediate data.** The scheme has one stage (move) and produces no intermediate result. The only
internal data is the staged block in L1, whose only axis is `page`, assigned above. There is no
coordinator core: every used core runs identical work on its own range, and no stage computes on
one core while others wait.

**Interleaved default, in order.** (1) Spread `page` over the full grid: 110 cores here, read from
`device.compute_with_storage_grid_size()`. (2) Take the coarsest block that fits the core's
assignment, which is all of it for every golden shape (largest: 28 fp32 pages = 114,688 B).

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_page_staging` | `cb_depth_blocks = min(CB_DEPTH_BLOCKS, num_blocks_max, L1_BUDGET_BYTES // (block_pages · page_bytes))`, with `CB_DEPTH_BLOCKS = 2` (module constant) | 1 for every golden INPUTS shape; 2 whenever a core holds more than one block | With more than one block, the reader fills block k+1 while the writer drains block k (read on NoC0 overlapping write on NoC1). With one block there is nothing to overlap, so a second slot would be dead L1 |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| DRAM bank contiguity (tensor_accessor.h:560-575) | pages per transfer (a *run*) | `run = min(pages left in this block, (tensor_pages − 1 − p) / num_banks + 1)`; when a bank's run ends, the cursor moves to the next bank's slot 0 | The transfer runs past this buffer's last page in the bank into another allocation: wrong values, and the writer **overwrites a neighbouring tensor** |
| TensorAccessor page stride (tensor_accessor.h:515-521, 565-572) | address stride inside a run | Build both accessors with the default page size (the args' aligned page size); CB page = `buffer_aligned_page_size()` | `num_contiguous_pages` returns 1, and an arithmetic run walks addresses at the wrong stride |
| Blackhole DRAM read alignment: L1 destination ≡ DRAM source mod 64 (`noc-alignment-in-tt-metal.md` line 43) | CB slot stride | Slot stride = `page_bytes`, a multiple of 64 for every TARGET dtype/layout. `allocate_dfb` places CBs at DRAM alignment (program.cpp:2505-2509) | Silent corruption (the watcher stops the core under `--dev`) |
| L1 left to the op: 1347776 B per core (BB-05) | `cb_depth_blocks · block_pages · page_bytes` | `stream_block_cap = max(1, L1_BUDGET_BYTES // (CB_DEPTH_BLOCKS · page_bytes))`; the depth clamp above; host refusal when `page_bytes > L1_BUDGET_BYTES` | Launch refused, or model-resident L1 tensors clobbered |
| `noc_async_read` / `noc_async_write` size (dataflow_api.h:547, 876; `NOC_MAX_BURST_SIZE` = 256 × 64 = 16384 B, noc_parameters.h:289-290) | bytes per call | ≤ block bytes ≤ `L1_BUDGET_BYTES / 2` = 673,888 B by the block clamp; each call splits itself into 16 KB bursts | None beyond the L1 range (the documented size range is the L1 range) |
| Runs per block | transfers per block | None needed: the walk loops over runs. There are ≤ 2 runs per block whenever `num_cores ≥ num_banks`, which covers every golden shape with more than one page | — |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `dram_page_stream`: bank-major q-ranges, coalesced bank-run transfers, one CB, reader NCRISC/NoC0 + writer BRISC/NoC1 | **built** | Both tensors DRAM interleaved with identical spec, and `page_bytes ≤ L1_BUDGET_BYTES`. Covers every TARGET cell and every golden INPUTS shape | `block_pages` pages of `page` in q order | **Minimum**, against the DRAM boundary: each source page crosses once (read), each destination page once (write); 0 cross-core bytes. Per block: one transfer call per bank run (≤ 2), split into 16 KB bursts | Per block: one `cb_reserve_back`/`cb_push_back` and one `cb_wait_front`/`cb_pop_front` pair, one read barrier (drains the core's read pipe), one write flush. Intended once per block, i.e. once per core for golden shapes. No inits or reconfigs exist (no compute) |
| `single_core` | rejected: superseded by `dram_page_stream` | `num_cores == 1` with `tensor_pages > 1` | whole tensor | Same bytes, but one core's NoC ceiling (~122 GB/s single-core Blackhole copy, catalog `trid_double_issue`) against the 460.8 GB/s roofline: ~4× slower on the perf case | — |
| `compute_passthrough`: reader → copy_tile through DEST → pack → writer | rejected: superseded by `dram_page_stream` | any | — | Same DRAM bytes, plus a second CB and unpack/pack L1 traffic on every page. It also **changes values**: 16-bit DEST or the TF32 srcA path cuts float32 mantissas, int32/uint32 go through a float path, and bfloat8_b is re-packed. That violates the contract | — |
| `dram_to_dram_direct` | rejected: not expressible | any | — | A Tensix NoC read lands only in the issuing core's L1 (dataflow_api.h:529-530), so an L1 hop is structurally required | — |
| `split_dependent_axis_combine` | rejected: not applicable | — | — | No dependent axis: no destination element depends on more than one source element, so there is no combine to build | — |
| `reuse_broadcast` (multicast a shared operand) | rejected: not applicable | — | — | Operand-reuse check (below): each source page feeds exactly one destination page on exactly one core, so nothing is re-read | — |
| `sharded_zero_copy` (CB backed on a sharded buffer) | rejected: out of contract | — | — | TARGET has no memory_layout axis; the contract fixes DRAM interleaved for both tensors | — |
| `sub_page_rows`: split `intra_page`, moving one ROW_MAJOR row in byte pieces | **deferred**: reachable only for ROW_MAJOR rows with `page_bytes > L1_BUDGET_BYTES` (> 336,944 int32/uint32 elements, > 673,888 bfloat16). No golden INPUTS shape or production call has such a row (the longest is 5120 B), and `dram_page_stream` serves every shape below it. Reachable later: `load_block`/`store_block` already move byte spans at an address, and a piece is a span plus an offset | `page_bytes > L1_BUDGET_BYTES` | a byte span of one page | Minimum bytes, more transactions | — |

**Single built regime: the proof obligation.**
- **Domain:** every TARGET cell (all five dtypes, both layouts, all three alignments), DRAM interleaved, with `page_bytes ≤ L1_BUDGET_BYTES`.
- **No other regime applies:** there is no dependent axis, no shared operand, no sharded placement, and no compute.
- **Minimum vs. actual:** each tensor crosses DRAM once (perf case: 12,582,912 B read + 12,582,912 B written = 25,165,824 B), and that is the actual.
- **What a larger block amortizes:** CB handshakes, the read-pipe drain at each barrier, and the write flush, intended once per core.

### Traffic ranking

| Candidate split | DRAM bytes (perf case) | Cross-core bytes | NoC transactions per core (perf case, fp32) | Verdict |
|-----------------|------------------------|------------------|---------------------------------------------|---------|
| A. `page`, bank-major contiguous q-ranges, coalesced bank runs | 12,582,912 read + 12,582,912 write (minimum) | 0 | ~7 read + ~7 write bursts of 16 KB, from ≤ 2 calls each | **chosen** |
| B. `page`, flat page-id contiguous ranges, one transfer per page | same | 0 | 28 + 28 transfers of 4 KB. Core c starts on bank (28·c) mod 8 ∈ {0, 4}, so all cores pile their first issues onto two banks (catalog `bank_stagger`) | perf-lamp alternative |
| C. `page`, round-robin page assignment | same | 0 | 28 + 28, with no coalescing possible | dominated by A |
| D. `intra_page` split (faces, partial rows) | same | 0 | ≥ 4× A | rejected (catalog `tile_reorder`) |
| E. no split | same | 0 | 3072 + 3072 on one core | rejected on occupancy, not on traffic |
| F. dependent axis + combine | n/a: no dependent axis | — | — | not applicable |

Every candidate ties at the minimum: 2 × tensor bytes across DRAM and 0 cross-core. Transaction shape
decides. A moves the same bytes in 4× fewer transactions for fp32 (8× for bf16, up to 256× for
64 B ROW_MAJOR rows). Each transaction is a sequential span inside one bank. Its bank load is
balanced by construction, because each bank's run is split among that bank's own ~`num_cores /
num_banks` cores, rather than depending on issue order. Occupancy (full grid in A) is accounted for
separately and does not enter this ranking.

### Block schedule

```cpp
// Per core: core_q_start, core_pages (per-core RT). Common: tensor_pages, block_pages.
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
| `load_block` | reader (NCRISC, NoC0) | `block_pages` pages of q (actual extent `last_block_pages` on the last block) | source DRAM → `cb_page_staging` | The block's pages stay in `cb_page_staging` until `store_block` pops them | `cb_reserve_back(block_pages)` once per block. One `noc_async_read` per bank run (≤ 2 per block for golden shapes). One `noc_async_read_barrier` per block. `cb_push_back(block_pages)` once per block, at the **nominal** count even on a short last block |
| `store_block` | writer (BRISC, NoC1) | same | `cb_page_staging` → destination DRAM | — | `cb_wait_front(block_pages)` once per block. One `noc_async_write` per bank run. One `noc_async_writes_flushed` per block, before the pop, because the slot may be refilled. `cb_pop_front(block_pages)` once per block |
| `drain_writes` | writer | — | — | — | One `noc_async_write_barrier` per kernel (BB-03) |

**Run walk.** Shared by both kernels through one header, `kernels/g6_copy_bank_walk.hpp`, so the
reader and writer cannot diverge.
- Once per kernel, locate the cursor from `core_q_start`: find bank b and slot j with
  `core_q_start = Σ_{b'<b} R_{b'} + j`. This takes ≤ `num_banks` iterations. Set `p = b + num_banks · j`.
- Per block of n pages: while `n > 0`:
  - `run = min(n, (tensor_pages − 1 − p) / num_banks + 1)`.
  - Issue one transfer of `run · page_bytes` bytes between `accessor.get_noc_addr(p)` and the current L1 slot, then advance the L1 pointer by `run · page_bytes`.
  - If the run ended the bank, move to the next bank's first page: `p = (p mod num_banks) + 1`. Otherwise `p += run · num_banks`.
  - `n −= run`.
- The cursor persists across blocks.
- Banks with no pages (`b ≥ tensor_pages`) are never reached, because q stops at `tensor_pages`.

### Stall-shadow check

| Stage that waits | Waits on | Work independent of the wait | Decision |
|------------------|----------|------------------------------|----------|
| reader: `noc_async_read_barrier` | the DRAM reads of its own block | None on this core. All reads of the block are already issued, and with more than one block the next block's reads need only a free slot, which depth 2 provides | Nothing to fill |
| writer: `cb_wait_front(block_pages)` | the reader's push of the whole block | Writing pages of the block that have already landed | **Not filled in Phase 0.** DRAM is the bottleneck, and it is busy with every core's reads during this wait. The wait therefore costs only pipeline fill (~one DRAM round trip per core) and the mid-op read → write turnover. The reorder is a pure knob-turn (`block_pages < core_pages`, `cb_depth_blocks = num_blocks_max`, CB capacity unchanged) and is recorded as the overlap lamp. It needs no algebraic permission (pages are independent) and has no floating-point-order impact (no arithmetic) |
| writer: `noc_async_write_barrier` | acks of its own writes | None: this is the last stage | — |

### Operand-reuse check

| Operand | Varies along the split axis (`page`)? | Reuse-shared by construction? |
|---------|--------------------------------------|-------------------------------|
| `input_tensor` | yes: each core reads only its own pages | no |
| `output_tensor` | yes: each core writes only its own pages | no |

No broadcast regime exists, which is why `reuse_broadcast` is rejected as not applicable.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| overlap | With one block per core, the writer starts only after the whole assignment (up to 28 pages, 112 KB) has landed. The read and write phases of all cores roughly align, leaving a DRAM bubble at the turnover and a full-block drain at the end | `block_pages = ceil(core_pages_max / k)`, k ∈ {2, 4}, with `cb_depth_blocks = num_blocks_max` (CB capacity unchanged, so the reader never waits on reserve). Optionally add per-block transaction-id waits (`dataflow_kernel_lib::set_read_trid` / `async_read_barrier_with_trid`, local_copy_helpers_dataflow.hpp:147, 158; catalog `trid_double_issue`) so block k+1's reads stay on the wire across block k's hand-off |
| grid synchronization | The small production cases spread thin. (1, 12, 128, 128) is 192 pages, i.e. 1–2 pages (4–8 KB) per core on 110 cores, where launch cost and one DRAM round trip per core may dominate | Cap `MAX_CORES` so `core_pages ≥ 8` (24 cores for 192 pages, 48 for 384) and compare with the full grid |
| transfer order / coalescing | Bank-major runs focus each core's traffic on one DRAM endpoint (~14 cores per bank) instead of spreading each core over all 8 banks. 4 KB pages are already large transactions, so coalescing may buy little | Flat page-id ranges with one transfer per page (candidate B), with and without bank stagger (start each core's walk at offset `core_idx mod num_banks` inside its range; catalog `bank_stagger`) |
| core ↔ bank placement | q-ranges map to cores in row-major order, so bank b's ~14 cores sit wherever row-major order puts them, not next to bank b's DRAM endpoint | Give bank b's q-range to the cores nearest its DRAM NoC endpoint (catalog `read_coalescing` "bank_affinity": 0.91–1.28× after coalescing) |

These are not lamps here:
- **split_reader:** the reader issues ≤ 2 calls per block, so it is never issue-bound (catalog `split_reader`: "does nothing unless a data-movement RISC-V is itself the bottleneck").
- **NoC selection:** reads on NoC0 and writes on NoC1 is the measured fast default (catalog `noc_placement`).

**Catalog entries behind each knob.**
- `block_pages`: `double_buffer` (issue a batch, then one barrier, never read-one/barrier) and `trid_double_issue` ("at a large CB the trivial one-block loop is within 3% of the best number").
- `cb_depth_blocks`: `double_buffer` ("no gain once DRAM-bandwidth-bound" with enough cores).
- Core assignment: `noc_placement` (`row_wise=True`), `bank_stagger`, `read_coalescing`.
- Transfer granularity: `tile_reorder`, `read_coalescing`.

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| source DRAM → L1 | The tensor's own page bytes: fp32 / bf16 / bfp8 / int32 / uint32 tile, or a ROW_MAJOR row with its alignment tail | reader `noc_async_read` on NoC0, one call per bank run, at address `src_accessor.get_noc_addr(p)` | No format conversion exists anywhere on the path |
| L1 staging | same bytes, in `page_bytes` slots | `cb_page_staging`, reader → writer handshake | The CB `data_format` is the tensor dtype only because the descriptor requires one; no unpacker or packer touches it |
| L1 → destination DRAM | same bytes | writer `noc_async_write` on NoC1, one call per bank run, at address `dst_accessor.get_noc_addr(p)` | Destination page p receives exactly source page p's bytes. A ROW_MAJOR alignment tail lands inside the destination's own page allocation |

- **Inter-Tensix:** none. No deferred regime needs it.
- **Multi-device:** `generic_op` runs the same program SPMD on every device of the tensors' mesh (generic_op.cpp:13-23). Each device copies its own buffer; shapes are per device.

### TARGET axes as regime rows

| TARGET value | Cost class | What changes |
|--------------|-----------|--------------|
| dtype bfloat16 / bfloat8_b / int32 / uint32 (TILE) | knob-turn | Nothing in the kernels: `page_bytes` comes from `buffer_aligned_page_size()` (2048 / 1088 / 4096 / 4096). Move the cell into SUPPORTED, or drop it from EXCLUSIONS |
| layout ROW_MAJOR (bf16, int32, uint32) | knob-turn | page = row; `page_bytes` = aligned row (≥ 64 B on Blackhole); same kernels, same walk. Rows of 4–128 B move as 64–128 B aligned slots, and bank runs coalesce up to 16 KB per burst |
| alignment `w_non_aligned` / `h_non_aligned` | built in Phase 0 | Whole-tile copy; padding goes to padding |
| memory_layout | not a TARGET axis | — |

**Structural impossibilities:** none beyond `feature_spec.INVALID` (bfloat8_b × ROW_MAJOR).

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one block of `block_pages` pages in bank-major order |
| Grid | `device.compute_with_storage_grid_size()` (11 × 10 = 110 on this board), capped by `MAX_CORES` (`None` = full grid). Core pool: `ttnn.num_cores_to_corerangeset(max_cores, grid, row_wise=True)`; split: `ttnn.split_work_to_cores(pool, tensor_pages, row_wise=True)` |
| Per-core work | q-range `[core_q_start, core_q_start + core_pages)`, with `core_pages ∈ {core_pages_g1, core_pages_g2}` (differing by 1) and `core_q_start` the running sum over used cores in row-major order |
| Remainder | `split_work_to_cores` balances to ±1 page, and the per-core count is a runtime arg, so there is no tail core. `tensor_pages` comes from `buffer_num_pages()` (per-image ceil in TILE) |

| Production shape (fp32 TILE) | `tensor_pages` | `num_cores` | `core_pages` (value × cores) | `block_pages` | CB bytes per core |
|------------------------------|---------------|-------------|------------------------------|---------------|-------------------|
| (16, 12, 128, 128) | 3072 | 110 | 28 × 102, 27 × 8 | 28 | 114,688 |
| (8, 12, 128, 128) | 1536 | 110 | 14 × 106, 13 × 4 | 14 | 57,344 |
| (2, 12, 128, 128) | 384 | 110 | 4 × 54, 3 × 56 | 4 | 16,384 |
| (1, 12, 128, 128) | 192 | 110 | 2 × 82, 1 × 28 | 2 | 8,192 |
| (1, 1) | 1 | 1 | 1 × 1 | 1 | 4,096 |

- **Core ranges:** one kernel per role over `all_cores`, with per-core RT args. `all_cores` is ≤ 2 rectangles (full rows plus one partial row) for any `num_cores` (work_split.cpp:105-122), which satisfies BB-17.
- **Regime selection:** there is one regime, so no selection function. The extent clamp is the only shape-dependent choice, and `describe_plan()` reports it (BB-08).
- **Forced-knob tests:** the multi-block path (`num_blocks_this_core > 1`, ring wrap, ragged last block) and the >2-runs-per-block walk are never reached by a production shape. The acceptance test therefore forces them by overriding the module knobs on production shapes.

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|----------|
| `cb_page_staging` | 0 | `page_bytes` | `cb_depth_blocks · block_pages` | spans `page` (`block_pages × cb_depth_blocks`); batch/tile_row/tile_col are folded into `page`; `intra_page` lies inside a page | input dtype (never interpreted) | reader | writer | whole kernel |

This is the only CB, and the program's whole L1 footprint. Full ledger: `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_block` | `block_pages` pages (actual n ≤ `block_pages`) | no: raw dataflow (see API Mapping) | — (source DRAM through the TensorAccessor) | `cb_page_staging`, reserve and push `block_pages` | `block_pages` pushed; only the first n are meaningful on a short last block |
| 2 | `store_block` | same | no: raw dataflow | `cb_page_staging`, wait `block_pages` | — (destination DRAM through the TensorAccessor) | `block_pages` popped after the flush |
| 3 | `drain_writes` | — | no | — | — | every write acked |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `load_block` | raw_api | `cb_reserve_back` | tt_metal/hw/inc/api/dataflow/dataflow_api.h:406 | `(cb_page_staging, block_pages)` | — | `cb_page_staging` | `block_pages` |
| `load_block` | raw_api | `get_write_ptr` | dataflow_api.h:325 | `(cb_page_staging)` | — | `cb_page_staging` | — |
| `load_block` | raw_api | `TensorAccessor` (interleaved), default page size | tt_metal/hw/inc/api/tensor/tensor_accessor.h:515-521 | `(TensorAccessorArgs<src CT offset>(), src_addr)` | — | — | — |
| `load_block` | raw_api | `TensorAccessor::get_noc_addr` | tensor_accessor.h:554 | `(p)` | — | — | — |
| `load_block` | raw_api | `TensorAccessor::contiguous_page_stride` / `num_contiguous_pages` | tensor_accessor.h:562, 568 | `()` → `num_banks`; `(p, tensor_pages)` → pages left in the bank | — | — | run bound |
| `load_block` | raw_api | `noc_async_read` | dataflow_api.h:547 | `(noc_addr, l1_slot, run · page_bytes)` | — | `cb_page_staging` | run |
| `load_block` | raw_api | `noc_async_read_barrier` | dataflow_api.h:1921 | `()` | — | — | — |
| `load_block` | raw_api | `cb_push_back` | dataflow_api.h:208 | `(cb_page_staging, block_pages)` | — | `cb_page_staging` | `block_pages` |
| `store_block` | raw_api | `cb_wait_front` | dataflow_api.h:479 | `(cb_page_staging, block_pages)` | `cb_page_staging` | — | `block_pages` |
| `store_block` | raw_api | `get_read_ptr` | dataflow_api.h:346 | `(cb_page_staging)` | `cb_page_staging` | — | — |
| `store_block` | raw_api | `TensorAccessor` / `get_noc_addr` / contiguity API | tensor_accessor.h:515-521, 554, 562, 568 | destination accessor, same walk | — | — | run bound |
| `store_block` | raw_api | `noc_async_write` | dataflow_api.h:876 | `(l1_slot, noc_addr, run · page_bytes)` | `cb_page_staging` | — | run |
| `store_block` | raw_api | `noc_async_writes_flushed` | dataflow_api.h:1980 | `()` | — | — | — |
| `store_block` | raw_api | `cb_pop_front` | dataflow_api.h:260 | `(cb_page_staging, block_pages)` | `cb_page_staging` | — | `block_pages` |
| `drain_writes` | raw_api | `noc_async_write_barrier` | dataflow_api.h:1951 | `()` | — | — | — |

**Helpers considered and rejected** (covers every `raw_api` row; there is no compute phase):

| Candidate | File:Line | Why it cannot express the operation |
|-----------|-----------|-------------------------------------|
| `compute_kernel_lib::copy` (CopyTile → PackTile) | ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp:109-113 | It routes every value through unpack → DEST → pack. With 16-bit DEST a float32 loses mantissa bits, int32/uint32 travel a float path, and bfloat8_b is re-packed; the contract forbids each of these. The op also has no compute phase, so a compute kernel would add a CB and three threads for zero work |
| `dataflow_kernel_lib::set_read_state` / `read_with_state` | ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:75, 95; `.inl`:66 | Self-aimed L1 → L1 reads: the source is built by `local_addr()` (:57), and a `static_assert` limits `transfer_size ≤ NOC_MAX_BURST_SIZE`. They cannot source DRAM pages or express a multi-burst bank run |
| `dataflow_kernel_lib::set_read_trid` / `async_read_barrier_with_trid` | local_copy_helpers_dataflow.hpp:147, 158 | Usable (they tag any read batch), but they buy nothing with one block per core, where only one barrier exists. They are the overlap lamp's mechanism, not Phase 0's |
| tilize / untilize / reduce / matmul / eltwise chain / mcast helpers | — | A copy has no tilize, reduction, product, element-wise math or multi-receiver transfer |

There is no compute kernel, so `compute_kernel_hw_startup()` does not apply.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| A bank run overruns its bank | Coalesced transfers are contiguous only inside one bank | Run clamp from Mechanism caps; when a bank's run ends, the cursor jumps to the next bank's slot 0. The forced-knob test with `MAX_CORES = 3` crosses banks mid-block |
| Stale buffer address on a host-cache hit | The descriptor is cached per shape, while buffer addresses change per call | Addresses live only in common RT args and are re-set on every call (Appendix B). `test_g6_copy_rebinds_buffers` writes two different destinations of one shape |
| Returning `generic_op`'s wrapper instead of the caller's object | The test checks `out is output_tensor` and the buffer address; the model reads its own object | `return output_tensor` after the dispatch |
| Push count on a short last block | Per-core counts are ragged (28 vs 27), and the multi-block path has tails | The reader always pushes `block_pages`; the writer waits on and pops `block_pages`. Only transfer sizes use n. Capacity `cb_depth_blocks · block_pages` closes the ring (blocking-model §2) |
| The writer pops before the data has left L1 | With more than one block, the reader refills the slot | `noc_async_writes_flushed()` before `cb_pop_front` |
| NoC writes in flight at kernel exit | Another program starts right after the call (BB-03) | `noc_async_write_barrier()` at the writer's end; the reader ends after its last read barrier |
| Any compute / unpack path | float32 bits, int32 values and bfp8 groups would change | No compute kernel; the CB format is never interpreted |
| The accessor is built with a non-aligned page size | The contiguity API returns 1, and an arithmetic stride is then wrong | Construct with the default page size; CB page = `buffer_aligned_page_size()` |
| TensorAccessorArgs emit common RT args | The kernels' own common RT args must not collide with the accessor's | Put the op's common args first (address, `tensor_pages`, `block_pages`) and give the accessor's CRTA offset after them. For DRAM interleaved the accessor is all compile-time today; verify that, and keep the offset correct if it changes |
| Hard-coded 110 cores or 8 banks | This board's grid is overridden to 11 × 10; other boards differ | Grid from `compute_with_storage_grid_size()`; `num_banks` only on device, via `contiguous_page_stride()` |
| The same tensor passed as source and destination | The contract does not forbid it | Each page is read and then rewritten by the same core with identical bytes, so the result is correct; not refused |
| `get_noc_addr` notes the source as read+write for op-to-op R/W inference (tensor_accessor.h:546-553) | It is an annotation for tooling | Harmless for values; no action |

## Appendix A — Entry point and registry contract

| Item | Specification |
|------|---------------|
| Files | `ttnn/ttnn/operations/g6_copy/__init__.py` (exports `g6_copy`, `INPUT_TAGGERS`, `SUPPORTED`, `EXCLUSIONS`, `describe_plan`); `g6_copy.py` (registry, `validate`, entry point); `g6_copy_program_descriptor.py` (knob constants `L1_BUDGET_BYTES = 1347776`, `MAX_CORES = None`, `CB_DEPTH_BLOCKS = 2`; `_plan()`; descriptor builder and cache; `describe_plan()`); `kernels/g6_copy_reader.cpp`, `kernels/g6_copy_writer.cpp`, `kernels/g6_copy_bank_walk.hpp` |
| `INPUT_TAGGERS` | `{"alignment": tag_alignment}`. On the shared shape, `W = shape[-1]` and `H = shape[-2]` (H = 1 for rank 1). Returns `"w_non_aligned"` if `W % 32`, else `"h_non_aligned"` if `H % 32`, else `"tile_aligned"` |
| `SUPPORTED` | `{"dtype": [ttnn.float32], "layout": [ttnn.TILE_LAYOUT], "alignment": ["tile_aligned", "w_non_aligned", "h_non_aligned"]}` |
| `EXCLUSIONS` | `[{"dtype": float32, "layout": ROW_MAJOR}, {"dtype": int32, "layout": TILE}, {"dtype": uint32, "layout": TILE}]` |
| `validate(input_tensor, output_tensor)` | (1) Malformed-request checks, raising `ValueError`: both tensors on device; rank 1–4; shapes equal; dtypes equal; layouts equal; memory configs equal; memory DRAM interleaved. (2) Axes from `input_tensor`: dtype, layout, then taggers. (3) SUPPORTED → `UnsupportedAxisValue`. (4) EXCLUSIONS → `ExcludedCell`. INVALID is never checked. Every message starts with `g6_copy:` (the acceptance test matches it) |
| `g6_copy(input_tensor, output_tensor)` | `validate(...)` is the first line. Then get the cached descriptor (or build it on a miss), set the two buffer addresses, call `ttnn.generic_op([input_tensor, output_tensor], descriptor)` with the output last, and `return output_tensor`, the caller's object. Exactly one device program per call; no host-side tensor transformation |

## Appendix B — Host program cache (BB-07, BB-16)

| Item | Specification |
|------|---------------|
| Key | `(tuple(shape), dtype, layout, memory_config, device id, grid (x, y), L1_BUDGET_BYTES, MAX_CORES, CB_DEPTH_BLOCKS)`. The knob constants are in the key so that changing one (perf pass, forced-knob test) builds a new entry instead of replaying a wrong program. Read them from the module at call time, never captured at import or as default arguments: the acceptance test overrides them with `monkeypatch.setattr` |
| Value | The built ProgramDescriptor, plus the handles needed to set the addresses |
| Per call | Only the source and destination buffer addresses change. They are the first common RT arg of the reader and of the writer respectively, the only values set on a hit. Per-core RT args (`core_q_start`, `core_pages`), `tensor_pages`, `block_pages`, CT args and the CB size all derive from the key |
| Patch reaches the dispatch | If `ProgramDescriptor.kernels` returns copies in Python, keep the `KernelDescriptor` objects and reassemble `ProgramDescriptor(kernels=…, cbs=…, semaphores=[])` from the cached parts each call, without recomputing the plan or args. `common_runtime_args` is read-write (program_descriptors.cpp:852) |
| Device-side cache | `generic_op` hashes the runtime-arg count, not the values (generic_op_device_operation.cpp:83-101), so the compiled program is reused across calls and buffers |
| Trace (BB-16) | Nothing is allocated per call. The first call (before capture) builds the descriptor; replays reuse the captured addresses |

## Appendix C — Plan and role reporting (BB-08, BB-21)

| Item | Specification |
|------|---------------|
| `describe_plan(input_tensor) -> dict` | Host-only and deterministic; uses the same `_plan()` as the dispatch. Keys: `regime` (`"dram_page_stream"`), `grid` (`(x, y)`), `num_cores`, `num_pages`, `page_bytes`, `core_pages` (list, row-major over used cores), `block_pages`, `cb_depth_blocks`, `cb_pages`, `l1_bytes_per_core` (= `cb_pages · page_bytes`), `roles` (dict keyed `"(x,y)"` for every grid core: `"copy q[a,b): reader NCRISC/NoC0 + writer BRISC/NoC1"` for used cores, `"idle"` otherwise) |
| Zones (BB-21) | `DeviceZoneScopedN` (kernel_profiler.hpp:1012; no-op in the plain build, :1047/:1136). Reader: `G6C-RD-RESERVE` (wait), `G6C-RD-ISSUE` (work), `G6C-RD-BARRIER` (wait), `G6C-RD-PUSH` (work). Writer: `G6C-WR-WAIT` (wait), `G6C-WR-ISSUE` (work), `G6C-WR-FLUSH` (wait), `G6C-WR-POP` (work), `G6C-WR-DRAIN` (wait). Every wait has its own zone and is never shared with work |

## Appendix D — Acceptance test coverage (BB-22)

| Test | Cases | Why it is required |
|------|-------|--------------------|
| `test_g6_copy_values` | the 5 float32 production shapes, plus (1, 1, 1, 256) in float32 | Production rows. (1, 1, 1, 256) is the only `h_non_aligned` float32 case and the non-square one; SUPPORTED claims `h_non_aligned`, and the native-handling rule requires it to run |
| `test_g6_copy_adversarial_bits` | (2, 12, 128, 128) with NaN payloads, ±inf, ±0, denormals, extremes | The contract's bit-identity gate |
| `test_g6_copy_poisoned_padding` | (1, 1) and (1, 1, 1, 256) with source padding = 1000.0 | No logical destination element may depend on source padding |
| `test_g6_copy_rebinds_buffers` | (1, 12, 128, 128): two tensor pairs alive at once (distinct addresses), each call checked to write its own destination and leave the other untouched, then the first pair again | BB-07 (cache hit with new addresses) and BB-06 (repeat is bit-identical) |
| `test_g6_copy_forced_multi_block` | (16, 12, 128, 128) with `block_pages` forced to 3 (full grid); (1, 12, 128, 128) with `MAX_CORES = 3` and `block_pages` forced to 5 | The only way to run the multi-block / ring-wrap / ragged-tail path and the multi-bank walk, which every shape beyond the L1 clamp or a capped grid reaches |
| `test_g6_copy_plan` | the Phase 0 shapes | BB-08 (plan reported, deterministic) and BB-05 (L1 ≤ 1347776) |
| `test_g6_copy_rejects_*` | shape, dtype, layout and memory-config mismatches; rank 5; host tensor | The spec's Validation list |
