# Operation Design: g6_concat

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (reader + writer, **no compute kernel**) |
| Goal | `torch.cat(tensors, dim)` on device tensors in one program: every logical input element read once from its own buffer, every output element written once, bit for bit. |
| Math | `d = dim mod r`; `o_k = n_0 + … + n_{k-1}`; `out[…, o_k + j, …] = x_k[…, j, …]`, `0 ≤ j < n_k` |
| References | `.claude/references/blocking-model.md`, `.claude/references/l1-footprint-discipline.md`, `.claude/references/noc-alignment-in-tt-metal.md`, `eval/golden_tests/g6_concat/{feature_spec,helpers,axes}.py`, `ttnn/ttnn/operations/examples/master.md` (catalog entries named in the Blocking Model), `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp`, `tt_metal/hw/inc/api/dataflow/dataflow_api.h`, `tt_metal/hw/inc/api/tensor/tensor_accessor.h` |

Notation used throughout:

| Symbol | Meaning |
|--------|---------|
| `K` | number of inputs, `1 ≤ K ≤ K_MAX` |
| `r`, `d` | rank (3 or 4); canonical dim `d = dim + r if dim < 0 else dim` |
| `s_k`, `n_k`, `N` | shape of input k; `n_k = s_k[d]`; `N = Σ n_k` |
| `o_k` | join offset of input k along `d`: `o_k = Σ_{i<k} n_i` |
| `ct(x)` | `ceil(x / 32)` — tile count of a dim, **per image** |
| `page_bytes` | tile size of the dtype: bf16 2048, fp32 4096, bf8_b 1088 |
| `rb` | bytes of one face row (16 elements): bf16 32, fp32 64 |
| `face_bytes` | `16 · rb` (bf16 512) |
| `out_imgs`, `out_Ht`, `out_Wt` | output: `Π out[0:r-2]`, `ct(out[r-2])`, `ct(out[r-1])` |
| `P_out` | output pages `= out_imgs · out_Ht · out_Wt = output.buffer_num_pages()` |
| `v(p)` | valid rows of output page `p`: `v_last = out[r-2] − 32·(out_Ht − 1)` if `p`'s tile row is `out_Ht − 1`, else 32 |
| `A_read(k)` | NoC read alignment of input k's buffer: DRAM 64 B on Blackhole (32 B Wormhole), L1 16 B |

Tile storage (needed for every row-granular transfer): four 16×16 faces, face-major, row-major
inside a face — face 0 rows 0–15 cols 0–15, face 1 rows 0–15 cols 16–31, face 2 rows 16–31
cols 0–15, face 3 rows 16–31 cols 16–31 (`ttnn/cpp/ttnn/kernel_lib/tile_column_dataflow.hpp:11`).
Row `y` of a tile is two contiguous `rb`-byte pieces at offsets
`face_off(2·(y/16) + fc) + (y mod 16)·rb`, `fc ∈ {0, 1}`; consecutive rows inside one face half
are contiguous.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `tensors` | `list[ttnn.Tensor]` (positional) | yes | `1 ≤ K ≤ K_MAX (64)`; rank 3/4; one dtype; one layout; equal sizes except dim `d`; interleaved | — | shapes / dtype / layout / placements are part of the program (cache key); base addresses are common RT args |
| `dim` | `int` (positional) | no | `[-r, r-1]`, canonicalized to `d` before any gate | `0` | host only (selects regime and address map) |
| `memory_config` | `ttnn.MemoryConfig` (keyword-only) | no | `None` or an INTERLEAVED config in DRAM or L1 | `None` → `ttnn.DRAM_MEMORY_CONFIG` | host (output allocation); writer accessor CT args |

No `compute_kernel_config`: nothing is computed.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | rank 3 or 4; all `K` inputs equal in every dim except `d`; second-to-last dim any size ≥ 1 (padding rows are not data) |
| Dtype | shared; Phase 0 `ttnn.bfloat16` (TARGET adds `float32`, `bfloat8_b`) |
| Layout | shared; Phase 0 `ttnn.TILE_LAYOUT` (TARGET adds `ROW_MAJOR_LAYOUT`) |
| Memory | interleaved, each input read where it lives; Phase 0 DRAM (TARGET adds L1 and mixed per input); sharded is refused |

### Output

| Property | Value |
|----------|-------|
| Shape | shape of `x_0` with `N` in dim `d` |
| Dtype | inputs' dtype |
| Layout | inputs' layout |
| Memory | INTERLEAVED in `memory_config.buffer_type`; DRAM interleaved when `memory_config is None` (also when the inputs live in L1). Padding rows of the last tile row of each image hold unspecified bytes. |

## Blocking Model

Semantics: `.claude/references/blocking-model.md`.

**The decision in one paragraph.** Concatenation has no dependent axis and no reuse-shared operand. Every
output element has exactly one source element, so every output page can be produced on its own.
The work is split by **output page**. Each core gets a contiguous range of the row-major output page
index, balanced to ±1 page across `min(P_out, grid)` cores (multi-core from Phase 0). Each core
walks its range in **blocks of `block_pages` consecutive output pages**. The default is the coarsest
block under an L1 cap (`BLOCK_BYTES_CAP = 128 KiB`, so 64 bf16 pages) or the core's whole range
when that is smaller. There is one CB between reader and writer, double-buffered.

Two regimes fill a block:
- `page_copy`: each output page is one source page.
- `row_gather`: used for `h` joins whose offsets are not multiples of 32. Each output page is
  assembled from row-range reads of the inputs whose rows land in it.

Both regimes use the same writer, the same work split and the same block knob.

### Axes

The tensor's own axes are named in the output's tile index space. The blocking axis is `out_page`,
the row-major flattening of all of them. The flattening is honest for two reasons. First, every
output page costs one independent block-level transfer set with no state shared between pages.
Second, consecutive output pages are consecutive page ids of the output buffer. Each operand is
fetched once per page however the run is cut, so the cut leaves the bytes unchanged
(`blocking-model.md` §4). The extent is therefore set by the **extent ranking**: fixed cost per
block against L1.

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `lead` — dims before `d` (outer join) / all dims before the last two (`h`, `w` join) | **independent** — output slice `i` is assembled from slice `i` of every input; nothing crosses slices | folded into `out_page` (the part of a `block_pages` run that lies on it) | — (folded) | `block_pages` | contiguous `out_page` ranges | knob-turn |
| `cat` — dim `d`, the joined axis (`N = Σ n_k`) | **independent** — each output index along `d` has one source `(k, j = idx − o_k)`. `page_copy` joins on page boundaries. `row_gather` assembles each output tile row from row-range reads. An input tile row straddles two output tile rows only at tile granularity, and the row-range reads remove that coupling, so no output page needs another page's state. | folded into `out_page` | — (folded) | `block_pages` | contiguous `out_page` ranges (cuts may fall anywhere, including inside one input or between output tile rows of a `row_gather` image) | knob-turn |
| `mid` — dims strictly between `d` and the last two (outer join of rank 4 with `d = 0`) | **independent** — same as `lead` | folded into `out_page` | — (folded) | `block_pages` | contiguous `out_page` ranges | knob-turn |
| `h_tiles` — tile rows of an image (`d ≠ r-2`) | **independent** — rows copy straight through | folded into `out_page` | — (folded) | `block_pages` | contiguous `out_page` ranges | knob-turn |
| `w_tiles` — tile columns (`d ≠ r-1`) | **independent** — columns copy straight through; in `row_gather` every column of an output tile row shares one run table | folded into `out_page` (innermost) | — (folded) | `block_pages` | contiguous `out_page` ranges (Qwen 79: ~15 contiguous tile columns per core) | knob-turn |
| `rows_in_tile` — the 32 rows inside one tile (sub-page) | **independent** — each logical row has one source row | rows per transfer piece: whole page (`v = 32`), the valid rows `[0, v)` when `v ≤ TRIM_ROWS_MAX`, `row_gather` pieces ≤ 16 rows (face half) | `TRIM_ROWS_MAX = 16` (bf16/fp32), `0` for bf8_b | host constant `TRIM_ROWS_MAX` → kernel CT arg | not assigned across cores: a page is never split between cores (a page is the NoC transfer unit of an interleaved buffer) | knob-turn (threshold) |
| **`out_page`** — flattened output page index `[0, P_out)` | **independent** (all of the above) | **`block_pages`** | `min(pages_per_core_g1, block_pages_cap)`, `block_pages_cap = max(1, BLOCK_BYTES_CAP // page_bytes)`, `BLOCK_BYTES_CAP = 128 KiB` (bf16 64, fp32 32, bf8_b 120). Qwen 79 → 15; p2–p4 → 64 | host constant `BLOCK_BYTES_CAP` in `g6_concat_program_descriptor.py`; `block_pages` computed once; CB capacities and the kernels' common RT arg derive from it | `ttnn.split_work_to_cores(grid, P_out, row_wise=True)`, contiguous ranges, `num_cores = min(P_out, grid.x·grid.y)`; grid from `device.compute_with_storage_grid_size()`, overridable by a `max_cores` host parameter | knob-turn |

No axis is **dependent**, and no axis is **reuse-shared**. Operand-reuse check: under the
`out_page` split, does any input stay the same along the split axis? No. Each input element feeds
exactly one output element, so no two cores read the same bytes, and nothing needs to be
broadcast or multicast.

Why the extent is not the full per-core range when that range exceeds the cap:
1. **L1 coexistence.** TARGET places inputs and the output in L1 (`input_buffer` l1/mixed,
   `output_buffer` l1). Their per-bank footprint sits in the same L1 as the CBs and is not under
   the op's control at call time.
2. **Measured plateau.** A DRAM-bound multi-core copy plateaus long before 128 KiB per block.
   `double_buffer` (catalog) is flat from block 4 to 64 tiles. `trid_double_issue` (Blackhole)
   reaches 97.8 GB/s per core at block 16. Saturating DRAM needs only about 4 GB/s per core
   (~461 GB/s over ~120 cores). The cap still leaves ≥ 32 pages per block, so it is coarse, not
   minimal.

The overlap lamp below names the alternative to measure.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_output_pages` | `OUTPUT_PAGES_DEPTH` (host constant) | 2 blocks | reader fetches/assembles block `i+1` while writer stores block `i` (catalog `double_buffer`: CB = 2 × block is the measured sweet spot) |
| `cb_gather_staging` (row_gather only) | structural, not a knob | 1 block (+1 slack page) | holds a block's misaligned pieces from the fetch phase to the place phase of the **same** block; a second staging block plus transaction ids would let `gather_fetch_block(i+1)` run under `gather_place_block(i)` — perf lamp, not Phase 0 |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| NoC DRAM read alignment — Blackhole compares source and destination **mod 64** (`noc_parameters.h:379`, `noc-alignment-in-tt-metal.md` §1); DRAM writes mod 16 (`:380`), L1 mod 16 (`:375`) | placement of a `row_gather` piece: bf16 face rows are 32 B, so a row shift `δ = d − s` that is odd puts source and destination 32 B apart mod 64 | per piece: **direct** read into the output slot when `(src_off − dst_off) mod A_read(k) == 0`. Otherwise read into the `cb_gather_staging` mirror slot at `dst_off + shift`, `shift = (src_off − dst_off) mod A_read(k)` (always 32 for bf16 on BH; never for fp32, whose `rb = 64`; never for L1 inputs), then a local L1→L1 copy (mod 16, always legal). This relies on CB bases being DRAM-aligned (`tt_metal/impl/program/program.cpp:2509`, allocated with `get_alignment(BufferType::DRAM)`) and on `page_bytes ≡ 0 mod 64` | silent corruption without the watcher; watcher halt with it |
| Face geometry (16-row face halves, face-major storage) | rows per contiguous transfer piece | a run piece never crosses row 16 on the source **or** the destination; split each run at both boundaries (≤ 3 pieces per run, × 2 face columns) | rows land in the wrong face; values silently wrong |
| Per-kernel runtime args `max_runtime_args = 341` words per core (`tt_metal/api/tt-metalium/kernel_types.hpp:50`), shared by unique and common args | `K` (two common RT words per input: base address, plus `C_k` for `page_copy` or `n_k` for `row_gather`) | `K ≤ K_MAX = 64`; host raises `RuntimeError` above it (production K ≤ 16) | refused launch |
| CB ring wrap — pointers reset only on an exact `fifo_limit` hit (`ttnn-cb-memory-fundamentals.md`, CB Ring-Wrap Invariant) | the last, ragged block of a core | always push and pop **`block_pages`** (nominal); transfer only `last_block_extent` pages | reader/writer pointer desync → hang or wrong pages |
| bf8_b tile = 64 B exponent header + mantissas (an exponent byte per face row) | `rows_in_tile` trimming and `row_gather` pieces | `TRIM_ROWS_MAX = 0` for bf8_b (whole pages only); `row_gather` + bf8_b is the deferred `row_gather_bfp8` regime | exponents not moved → wrong values |
| Interleaved accessor buffer type is a template parameter (`TensorAccessor<…, IsDram, …> : InterleavedAddrGen<IsDram>`, `tensor_accessor.h:493-521`) | number of distinct input accessors in one kernel | inputs share page size and layout. Per input the reader keeps a base address (common RT) and a buffer-type bit (CT mask), and rebases one DRAM accessor and one L1 accessor when the walk moves to input `k` | per-input CT accessor chains grow the kernel and cannot be indexed at runtime |

### Regimes

The boundary named for the minimum is the NoC edge of each tensor's buffer. Each input page holding
logical data is read once, and each output page is written once. With row trimming, only the valid
rows of a partially valid tile row move.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `page_copy` | **built** | `not row_gather`. This covers every outer join, every `w` join, and `h` joins whose offsets `o_1 … o_{K-1}` are all multiples of 32 (K = 1 included). | `block_pages` consecutive output pages; page `p` comes from exactly one input page via the `(O, C_k)` map (Work Distribution) | **Minimum.** Each input page once: one whole-page read, or for a page with `v(p) ≤ TRIM_ROWS_MAX` only rows `[0, v)` at identical offsets (2 reads; 1 when `v = 16`). Each output page once, trimmed the same way. No cross-core traffic. Reads on NOC0, writes on NOC1. Domain: 70 of the 73 Phase 0 INPUTS cells, and 153 of all 156 INPUTS cells. | Fixed cost per block, paid once each: one `noc_async_read_barrier`, one `cb_reserve_back`/`cb_push_back`, one `cb_wait_front`/`cb_pop_front`, one `noc_async_writes_flushed`. Per-input accessor rebase only when the walk crosses into the next input. Intended frequency: `ceil(pages_this_core / block_pages)` per core (p2: 12; p1 fp32: ~24; Qwen 79-class small outputs: 1). |
| `row_gather` | **built** | `layout == TILE ∧ d == r−2 ∧ ∃ k ∈ [1, K−1]: o_k mod 32 ≠ 0` | `block_pages` consecutive output pages. Each page is assembled from the **run table** of its output tile row `R`: for each input whose rows `[o_k, o_k+n_k)` meet `[32R, 32R+32)`, one run per input tile row met (source rows `[s0,s1)` → destination rows `[d0,d1)`), split at row 16 on both sides | **Inputs: exactly their logical rows, once.** Row-range reads move no padding row and read no row twice; an input tile straddling output rows `R` and `R+1` is read as two disjoint row ranges. **Output: each page once**, with the last tile row trimmed (Qwen 79: `v = 16`, one 1024 B write). **Added: one core-local L1→L1 hop** for the pieces whose shift is not read-aligned (bf16 on BH: odd `δ`, about half the rows of Qwen 79/89/143; zero for fp32 or L1 inputs). No cross-core traffic. Domain: Qwen 79, 89 and 143 (the 3 remaining INPUTS cells, all Phase 0) and every `h`-non-aligned join. | Per block, paid once each: two read barriers (after fetch, after place), one reserve/push on the reader, one wait/pop plus one flush on the writer. The run table is built once per output-row segment of the block and shared by all its columns. Intended frequency: one block per core for Qwen 79 (~15 pages per core). |
| `row_gather_whole_tile` | **rejected** — dead end; `row_gather` supersedes it | (same as `row_gather`) | input tile rows streamed whole, rows placed into a 2-row output ring | Reads whole input tiles, up to 32× the logical bytes for height-1 inputs: Qwen 79 reads 63.6 MB of input tiles against 2.0 MB of rows. A stateless split of output rows would also read each straddling tile twice, which violates read-once. Nothing built on it survives `row_gather`. | — |
| `uniform_row_runs` (one gather path for every shape) | **rejected** — `page_copy` supersedes it on its predicate | all | as `row_gather` | Same bytes as `page_copy` for page-aligned joins, but ≥ 2 transactions per full tile (two face columns per run piece) instead of 1 | — |
| `page_copy_row_major` | **deferred** — Phase 0 SUPPORTED is TILE only by the spec's contract. The structure keeps it reachable: same reader and writer, `page = stick` with the `aligned_page_size` stride, the same `(O, C_k)` map in stick units with H untiled, no trimming. Covers every production ROW_MAJOR case: rank-3 dim 0 (Qwen 28, 29, 33, 35, …) and rank-3 dim 1 (Qwen 77, 78, 141, 142, 159). | `layout == ROW_MAJOR ∧ d ≠ r−1` | `block_pages` sticks | Minimum (each stick once) | Same as `page_copy` |
| `stick_gather_row_major` | **deferred** — no INPUTS entry joins ROW_MAJOR tensors along the last dim, and BB-22 forbids testing it. Reachable through `row_gather`'s fetch/place structure, using byte ranges inside sticks instead of face-row pieces. | `layout == ROW_MAJOR ∧ d == r−1` | output sticks, each assembled from `K` byte ranges | Minimum, plus a bounce hop where stick byte offsets are not read-aligned | Same as `row_gather` |
| `row_gather_bfp8` | **deferred** — no INPUTS entry: the bf8_b cases (Panoptic 23/32/40/51) are tile-aligned `w` joins, which run as `page_copy`. The NoC moves 16 B-aligned chunks and cannot place the 1-byte face-row exponents, so this needs a byte-placement step in the place phase. The run table carries over unchanged. | `dtype == bfloat8_b ∧ row_gather predicate` | as `row_gather` | as `row_gather` | as `row_gather` |
| `read_once_multicast` | **rejected** — not applicable | — | — | No reuse-shared operand exists, so there is nothing to multicast | — |

**Regime-selection function** (host, exact; `select_regime` in the program-descriptor module):

```
select_regime(shapes, d, layout):
    r = len(shapes[0])
    if layout == TILE and d == r - 2:
        offsets = prefix sums of [s[d] for s in shapes]          # o_0 .. o_{K-1}
        if any(o % 32 != 0 for o in offsets[1:]): return "row_gather"
    return "page_copy"
```

An `h` join where only the **last** input is unaligned is `page_copy`: its padding rows become the
output's padding rows. Regime-pinned tests (acceptance test) cover both regimes. They cover each
`page_copy` address map (outer, `w`, `h`-aligned, `h` with only the last input unaligned). They
cover `row_gather` with height-1 inputs (Qwen 79/89/143) and with runs that straddle tile rows
and face halves.

### Traffic ranking

The ranking counts crossings per tier, not nanoseconds. It covers every candidate split, including
those not built.

| # | Candidate split | DRAM crossings (inputs / output) | Cross-core traffic | Notes | Verdict |
|---|-----------------|----------------------------------|--------------------|-------|---------|
| 1 | **Output-page ranges** (contiguous `out_page` per core) | 1× / 1× | none | balance ±1 page for any `K` and any `n_k`; the writer address map is shared by both regimes; output written in contiguous page order | **chosen** — it is the minimum |
| 2 | Input-page ranges (each core takes a contiguous range of one input's pages) | 1× / 1× | none | `page_copy`: ties #1 on bytes. `row_gather`: an output tile would mix rows owned by several cores, so several cores would make partial writes to one page, adding write transactions | tie for `page_copy`; #1 wins the tiebreak (one map, output contiguity); worse for `row_gather` |
| 3 | Core groups per input (cores ∝ `C_k`) | 1× / 1× | none | integer rounding of the group sizes unbalances unequal inputs (Qwen 18: 108 vs 84) and fails when `K` exceeds the core count | rejected — balance |
| 4 | Split the dependent axis and combine | — | — | no dependent axis exists | not applicable |
| 5 | Read once and multicast a shared operand | — | — | no reuse-shared operand exists | not applicable |

The chosen split is also the cheapest-traffic split. Occupancy follows from the same split:
`min(P_out, grid)` cores, so the shape never limits the core count below what the page count
allows.

**Stall-shadow check.** Each stage that waits, and what can run during the wait:

| Waiting stage | Waits for | Independent work that can fill it | Phase 0 |
|---------------|-----------|-----------------------------------|---------|
| writer `cb_wait_front` | reader's block | — (writer has nothing else) | filled from the other side: depth 2 lets the reader fetch block `i+1` during the writer's store of block `i` |
| reader `cb_reserve_back` | writer freeing a block | nothing; the reader is ahead | — |
| `row_gather` place phase | the fetch barrier of the same block | the **next block's fetch**. Its destination slots and staging are disjoint, every transfer is a pure copy, and copies of different bytes commute with no arithmetic order to preserve | not in Phase 0: needs a second staging block and per-batch transaction ids (`local_copy_helpers_dataflow.hpp:147,158`). Recorded as a perf lamp |
| writer `noc_async_writes_flushed` before pop | its own writes leaving L1 | nothing; short | — |

### Block schedule

A logical schedule, realized by two asynchronous kernels: reader on NCRISC/NOC0, writer on
BRISC/NOC1.

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    // reader (one of the two, selected by regime at program build):
    page_fetch_block(block_idx);          // page_copy
    gather_fetch_block(block_idx);        // row_gather, phase A
    gather_place_block(block_idx);        // row_gather, phase B
    // writer:
    store_block(block_idx);
}
```

| Operation | Kernel | Block shape acted on | Resident across it | Intended frequency of fixed costs |
|-----------|--------|----------------------|--------------------|-----------------------------------|
| `page_fetch_block` | reader | `block_pages` output pages, `last_block_extent` on the last block | the walk cursor `(o, k, c')` and the per-input accessor (rebased when `k` changes); nothing else | 1 `cb_reserve_back(block_pages)`, 1 read barrier, 1 `cb_push_back(block_pages)` per block; one page-sized read per page (whole page) or 2 row-range reads (trimmed page) |
| `gather_fetch_block` | reader | `block_pages` output pages split into output-row segments `(img, R, c_begin, c_end)` | the run table of the current segment, built once per segment and shared by all its columns; staging slots of this block | 1 `cb_reserve_back(block_pages)` per block. Per piece × face column × page: one DRAM read, direct or staged. 1 read barrier per block. The loop is ordered **piece-major, page-minor** so the source and destination offsets stay fixed across a page sweep |
| `gather_place_block` | reader | the staged pieces of the same block | output slots (being assembled) | one self-aimed local read per staged piece × face column × page; 1 read barrier then 1 `cb_push_back(block_pages)` per block |
| `store_block` | writer | `block_pages` output pages | — | 1 `cb_wait_front(block_pages)`; per page one whole-page write, or 2 row-range writes when `v(p) ≤ TRIM_ROWS_MAX` (1 when `v = 16`); 1 `noc_async_writes_flushed` and 1 `cb_pop_front(block_pages)` per block; 1 `noc_async_write_barrier` per kernel (BB-03) |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Overlap | `BLOCK_BYTES_CAP = 128 KiB` with depth 2 is chosen for L1 coexistence. On p1–p4 (49k–98k pages, DRAM-bound) a deeper pipeline may keep more transactions in flight per core. | depth 3–4 with transaction-id issue (`trid_double_issue`: 1.11–1.63× per core, optimum at `cb_blocks/2` ahead); `BLOCK_BYTES_CAP` ∈ {64, 256, 512} KiB on p1–p4 |
| Grid synchronization | Tiny outputs (FLUX 22/23: 8 pages; Qwen 104: 16; Qwen 74: 128) put one page per core, so the launch cost on every core dominates | a floor of 4–8 pages per core (fewer cores) on the ≤ 128-page cases |
| Bank stagger | When `pages_per_core` is a multiple of the DRAM bank count, every core starts on the same bank | rotate the in-block page issue order by `core_idx mod num_banks` (`bank_stagger`, Blackhole: 1.04–1.07× on interleaved) |
| Gather issue balance | Qwen 79 per core: the reader issues ~480 DRAM reads plus ~240 local copies, while the writer issues ~15 writes. Small transfers are issue-bound (`split_reader`: 32 B reads at ~36 ns each). | move `gather_place_block` to the writer RISC: staging becomes a reader→writer CB and the writer fixes up the pages it holds before storing them (`split_reader`: 1.21–1.74×) |
| Gather fetch/place overlap (stall shadow) | Phase A of block `i+1` could run under phase B of block `i` | double staging plus `set_read_trid` / `async_read_barrier_with_trid`; only matters when a core has > 1 `row_gather` block |
| Trim threshold | `TRIM_ROWS_MAX = 16` trades 2 transactions against ≥ 50% of a page's bytes. Shapes with `v` in 17–31 (Qwen 108/109: `v = 31`) move whole pages. | thresholds 8 and 24 on Qwen 80 (`v = 16`), 90 (`v = 2`), 144 (`v = 8`), 160 (`v = 1`), 25 (`v = 4`) |
| Core line shape | `split_work_to_cores(row_wise=True)` gives row lines (`noc_placement`: row beats column 2.91× on Wormhole); the diagonal is unmeasured on Blackhole | a diagonal core order on Blackhole |

Catalog entries that set knobs: `double_buffer` (depth 2, block ≥ 4–8 pages), `trid_double_issue`
(block coarseness and the overlap lamp), `noc_placement` (reads on NOC0 and writes on NOC1, which
is 2.5–4.8× better than the reverse; `row_wise=True`), `read_coalescing` (interleaved pages
cannot coalesce, so one transaction per page), `tile_reorder` (whole 2 KB page writes beat 4 face
writes, so trimming only when it saves ≥ 50%), `split_reader` (cost per small transaction, and the
gather issue-balance lamp), `bank_stagger` (stagger lamp), `width_split` (contiguous tile-column
ranges for one-tile-row outputs: Qwen 79's 1940 columns).

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| Inputs in their buffers | dtype tiles, face-major | interleaved; one accessor per buffer type rebased per input (base address = common RT arg, buffer type = CT mask bit) | Phase 0 all DRAM. L1/mixed is the same code with another mask bit. |
| `page_copy`: input page → `cb_output_pages` slot | whole tile, or rows `[0, v)` at identical offsets | `noc_async_read_page` (whole) / `noc_async_read` with `get_noc_addr(page, offset)` (rows), reader on NOC0 | zero row shift, so always alignment-legal |
| `row_gather` fetch: input rows → output slot (direct) or `cb_gather_staging` mirror (staged) | face-row pieces (`L·rb` bytes) | raw `noc_async_read`, reader on NOC0 | direct iff `(src_off − dst_off) mod A_read(k) == 0` |
| `row_gather` place: staging mirror → output slot | face-row pieces | self-aimed NoC read (`dataflow_kernel_lib::local_addr`), reader on NOC0 | L1→L1 mod 16: always legal |
| `cb_output_pages` → output | whole tile, or rows `[0, v)` at identical offsets | `noc_async_write_page` (whole) / `noc_async_write` (rows), writer on NOC1 | output padding rows are never written when trimmed (allowed) |

No compute kernel, so values never pass through DEST and nothing is converted. No Tensix-to-Tensix
communication exists, and no deferred regime needs any.

### TARGET axes seen through the regimes

| TARGET axis value | Phase 0 | Regime / change | Class |
|-------------------|---------|-----------------|-------|
| `dtype` bfloat16 | yes | both | — |
| `dtype` float32 | no | `page_bytes = 4096`, `rb = 64`: every `row_gather` piece is direct, so no staging CB is needed | knob-turn |
| `dtype` bfloat8_b | no | `page_copy` with `page_bytes = 1088` and `TRIM_ROWS_MAX = 0`, bit-exact because pages move whole; `row_gather_bfp8` deferred | knob-turn (page_copy) |
| `layout` ROW_MAJOR | no | `page_copy_row_major` (deferred row), `stick_gather_row_major` (deferred row) | knob-turn / scheme-change respectively |
| `concat_dim` outer / h / w | all three | `page_copy`; `h` with non-aligned offsets → `row_gather` | — |
| `alignment` tile_aligned / h_non_aligned | both | `page_copy` (trimmed last tile row) / `row_gather` for `h` joins | — |
| `input_buffer` l1 / mixed | no | per-input buffer-type mask bit; `A_read(k) = 16` for L1 inputs, so `row_gather` pieces are always direct | knob-turn |
| `output_buffer` l1 | no | writer accessor CT args from the output tensor; the BB-05 check adds the output's per-bank L1 bytes | knob-turn |

No placement value is sharded: the spec has every tensor interleaved, so the physical-shard path
does not arise. A sharded input or `memory_config` is refused on the host.

**Structural impossibilities.** None beyond the existing `INVALID`
(`{bfloat8_b, ROW_MAJOR}`).

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | a block of `block_pages` consecutive output pages; the split unit is one output page |
| Grid | `device.compute_with_storage_grid_size()` (full Tensix grid), capped by the `max_cores` host parameter (default: whole grid) |
| Core assignment | `(num_cores, all_cores, g1, g2, n1, n2) = ttnn.split_work_to_cores(grid, P_out, row_wise=True)`. Core `i` in row-major order owns `[start_i, start_i + count_i)`, `count_i ∈ {n1, n2}`. `all_cores` is ≤ 2 rectangles (full rows plus one partial row), and every kernel uses one descriptor over `all_cores` (BB-17). |
| Per-core work | `num_blocks_this_core = ceil(count_i / block_pages)`, `last_block_extent = count_i − (num_blocks_this_core − 1)·block_pages` |
| Remainder | balanced by `split_work_to_cores` (`n1 = n2 + 1`). Ragged last block: nominal push/pop, actual transfers. All tile counts are per image with `ceil`: `out_Ht = ct(out[r-2])`, `Ht_k = ct(n_k)` (h join) or `ct(H)`, never `floor(batch·H/32)` |

**`page_copy` address map** (host-derived, passed as common RT args):

| `d` | `O` (outer slices) | `C_k` (pages of input `k` per slice) |
|-----|--------------------|--------------------------------------|
| `d ≤ r−3` (outer) | `Π s[0:d]` | `n_k · Π s[d+1:r−2] · ct(H) · ct(W)` |
| `d = r−2` (h, aligned offsets) | `Π s[0:r−2]` | `ct(n_k) · ct(W)` (only the last input may be partial) |
| `d = r−1` (w) | `Π s[0:r−2] · ct(H)` | `ct(n_k)` |

`C = Σ C_k`, `Coff_k = Σ_{i<k} C_i`. Output page `p = o·C + c` with `Coff_k ≤ c < Coff_k + C_k`
reads input `k` page `o·C_k + (c − Coff_k)`. Each core receives its start cursor `(o, k, c')` as
per-core RT args, so the kernel never divides per page. It walks by incrementing: `c'` wraps at
`C_k`, then `k` wraps at `K`, then `o` advances. For every `page_copy` predicate, `v` of the input
page equals `v(p)` of the output page; the host asserts this.

**`row_gather` address map**: per input `o_k`, `n_k`, `Ht_k = ct(n_k)`; `imgs = Π s[0:r−2]`,
`Wt = ct(W)`. Output page `p` → `img = p / (out_Ht·out_Wt)`, `R = (p / out_Wt) mod out_Ht`,
`c = p mod out_Wt`. Input `k` tile row `t`, column `c` → input page `(img·Ht_k + t)·Wt + c`.

**`v(p)`** (both kernels, from output geometry only): tile row `R = (p / out_Wt) mod out_Ht`,
maintained incrementally; `v = v_last` if `R == out_Ht − 1` else 32.

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|----------|
| `cb_output_pages` | 0 | `page_bytes` (`output.buffer_page_size()`) | `OUTPUT_PAGES_DEPTH · block_pages` | spans `out_page` (`block_pages` per block) × depth; streams over the core's range; every tensor axis is folded into `out_page` | output dtype (bf16 Phase 0); format only labels bytes, nothing unpacks them | reader | writer | whole program |
| `cb_gather_staging` | 1 | `page_bytes` | `block_pages + 1` | spans `out_page` (one mirror slot per page of the block); +1 page of slack for the `shift` (≤ 32 B) past the last slot; allocated **only** in `row_gather` when `rb mod A_dram_read ≠ 0` (bf16 on Blackhole) | dtype | reader | reader (private scratch: written by the fetch phase, read by the place phase of the same block) | one block, fetch → place |

Push count equals wait count. The reader pushes `block_pages` per block and the writer waits and
pops `block_pages` per block, the ragged last block included. `cb_gather_staging` carries no
credit: it is ordered by the fetch barrier, which sits between phase A's writes into it and
phase B's reads from it, all on one thread.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `page_fetch_block` | `block_pages` output pages (`last_block_extent` actual on the last block) | no (dataflow; no helper covers a page copy, see API Mapping) | — (source: input buffers) | `cb_output_pages`, reserve/push `block_pages` | `block_pages` pages pushed |
| 2 | `gather_fetch_block` | `block_pages` output pages, by output-row segment | no | — (source: input buffers) | `cb_output_pages` reserved `block_pages` (not pushed); `cb_gather_staging` written in place | output slots partly assembled, staging filled |
| 3 | `gather_place_block` | staged pieces of the same block | `local_addr` from `local_copy_helpers_dataflow.hpp` for the self-aimed source | `cb_gather_staging` (scratch, no credit) | `cb_output_pages`, push `block_pages` | `block_pages` pages pushed |
| 4 | `store_block` | `block_pages` output pages | no | `cb_output_pages`, wait/pop `block_pages` | — (destination: output buffer) | `block_pages` freed after `noc_async_writes_flushed` |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| all | raw_api | `cb_reserve_back` / `cb_push_back` / `cb_wait_front` / `cb_pop_front` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:406` / `:208` / `:479` / `:260` | `(cb, block_pages)` | — | `cb_output_pages` | `block_pages` |
| all | raw_api | `get_write_ptr` / `get_read_ptr` | `dataflow_api.h:325` / `:346` | `(cb)` | — | — | — |
| `page_fetch_block` | raw_api | `noc_async_read_page<AddrGen>` | `dataflow_api.h:1243` | `(page_id, accessor_k, slot_addr)` | — | `cb_output_pages` | — |
| `page_fetch_block`, `gather_fetch_block` | raw_api | `TensorAccessor::get_noc_addr(page_id, offset)` (interleaved) | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:554` | `offset` = in-tile byte offset of the row piece | — | — | — |
| `page_fetch_block` (trimmed), `gather_fetch_block` | raw_api | `noc_async_read` | `dataflow_api.h:547` | `(src_noc_addr, dst_l1, L·rb)` | — | `cb_output_pages` / `cb_gather_staging` | — |
| `gather_place_block` | helper (address builder) + raw read | `dataflow_kernel_lib::local_addr(addr, noc_id)`, then a NoC read | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:57` | `noc_id` = the issuing NoC (`my_x/my_y` indexed by it, header warning at `:47-50`) | `cb_gather_staging` | `cb_output_pages` | — |
| fetch / place | raw_api | `noc_async_read_barrier` | `dataflow_api.h:1921` | once per phase per block | — | — | — |
| `store_block` | raw_api | `noc_async_write_page<AddrGen>` | `dataflow_api.h:1418` | `(page_id, out_accessor, slot_addr)` | `cb_output_pages` | — | — |
| `store_block` (trimmed) | raw_api | `noc_async_write` | `dataflow_api.h:876` | `(slot_addr + off, out_noc_addr + off, L·rb)` | `cb_output_pages` | — | — |
| `store_block` | raw_api | `noc_async_writes_flushed` / `noc_async_write_barrier` | `dataflow_api.h:1980` / `:1951` | flushed once per block before the pop; barrier once at kernel end | — | — | — |

**Helpers considered and rejected** (one entry per raw path):

| Raw path | Candidate helper | Mismatch (file:line) | Why it cannot express the shape |
|----------|------------------|----------------------|---------------------------------|
| every phase (no compute kernel) | `eltwise_convenience.hpp` `copy` / `eltwise_chain` `CopyTile` | they are compute-kernel helpers that unpack into DEST and pack back out (`ttnn/cpp/ttnn/kernel_lib/eltwise/`) | The op must not convert values. A DEST round trip is the conversion the Precision rule forbids (fp32 low bits under 16-bit DEST, bf8_b requantization). A compute kernel would also add a third kernel and a second CB per page for no work. |
| every phase | `tilize_helpers*` / `untilize_helpers*` | layout converters (`tilize_helpers_dataflow.hpp`, `untilize_helpers.hpp`) | Input and output layouts are identical, so there is nothing to convert. |
| `gather_place_block` | `set_read_state<transfer_size>` + `read_with_state` | `local_copy_helpers_dataflow.hpp:75,95`: `transfer_size` is a template parameter baked into the NoC state | Piece sizes are runtime values (`L·rb`, `L ∈ [1, 16]` per run). The helper's own `local_addr` source builder is used instead. |
| all fetch/store paths | `mcast/` (`McastArgs`, `ttnn.Mcast`) | `ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp` | No operand is shared between cores, so there is nothing to multicast. |
| reduce / matmul / bias / reblock helpers | — | — | No reduction, product or untilize exists in this op. |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Input padding reaches a logical row | Test inputs are zero-padded, so copying padding into a logical row passes the cartesian tests. Model producers do not zero padding. | `row_gather` reads only rows `[s0, s1)` of each input tile, inside that input's logical range. `page_copy` keeps every row at its own index (padding only reaches output padding). The acceptance test poisons padding with 1000 and NaN. |
| bf16 row shift on Blackhole | A 32 B face row is half of the 64 B DRAM read alignment. Odd-shift pieces are illegal as direct reads, and without the watcher they corrupt silently. | Per piece, `shift = (src_off − dst_off) mod A_read(k)`: direct when 0, otherwise the staging mirror at `+shift`, then a local copy. Assert once at kernel start (`--dev`) that `get_write_ptr(cb_output_pages) % 64 == 0` and `get_write_ptr(cb_gather_staging) % 64 == 0`. |
| Pieces crossing row 16 | Face halves split rows 0–15 from rows 16–31, and the source and destination split points differ by `δ` | Split each run at 16 on the source **and** on the destination: up to 3 pieces, each × 2 face columns. The acceptance test has runs with `δ = 31` and `δ = 5` across both halves. |
| Straddling input tile read twice, or rows dropped | An input tile row can meet two output tile rows (`row_gather`), and the two rows can sit in different blocks or on different cores | Every output tile row derives its runs from the prefix sums alone. The two parts are disjoint row ranges, so each element is read exactly once, whichever core owns each part. |
| Ragged last block / ring wrap | `count_i` is rarely a multiple of `block_pages` | Push and pop are always `block_pages`; only `last_block_extent` pages are transferred (blocking-model §2). |
| Variable `K`, per-input placement | Accessor buffer type is a compile-time parameter; `K` varies per call | One DRAM and one L1 accessor rebased per input. Base addresses are common RT args, refreshed on each call. Buffer type is a CT mask, which is part of the cache key. |
| Stale program on a cache hit (BB-07) | Same output shape from different splits (`64+192` vs `128+128`) or another placement builds a different program | The cache key includes every input shape, canonical `d`, dtype, layout, the buffer type of each input, the output buffer and layout, and the grid / `max_cores`. Only the address common RT args are refreshed on a hit (Host Entry Point). |
| Trimmed writes and stale output padding | The padding rows of the output's last tile row hold whatever was in DRAM or the CB slot | Allowed by the spec ("the output's padding may hold anything"); do **not** add a zero-fill (extra writes). Logical values stay deterministic (BB-06). |
| Writer returns with writes in flight (BB-03) | `noc_async_writes_flushed` only guarantees the data left L1 | One `noc_async_write_barrier` at the end of the writer; the reader's last action is a read barrier. |
| `my_x`/`my_y` of the wrong NoC in the self-aimed read | NoC 0 and NoC 1 use different coordinate spaces (`local_copy_helpers_dataflow.hpp:47-50`) | Pass the issuing NoC's id to `local_addr`. |
| bf8_b trimming | Exponents sit in a 64 B header, not next to the mantissas | `TRIM_ROWS_MAX = 0` for bf8_b. The Panoptic bf8_b cases have no partial tile rows anyway. |
| Per-image tile padding | Each image is padded on its own (Qwen 109: 31 rows → 1 tile row per image, 6 images) | `out_Ht = ct(out[r-2])` per image, `P_out = output.buffer_num_pages()`, `v(p)` from `R = (p / out_Wt) mod out_Ht`. |

## Host Entry Point

| Item | Decision |
|------|----------|
| Signature | `g6_concat(tensors, dim=0, *, memory_config=None)`; `tensors` and `dim` positional as in `torch.cat` |
| First line | `validate(tensors, dim, memory_config=memory_config)` |
| Structural refusals (in `validate`, before the support gate) | `ValueError`: `tensors` not a non-empty list, rank ∉ {3, 4}, ranks / dtypes / layouts differ, sizes differ outside `d`, `dim ∉ [−r, r−1]`, a sharded input or a non-INTERLEAVED `memory_config`. `RuntimeError`: `K > K_MAX`, or the BB-05 L1 total over 1347776 B (no golden shape comes near it). |
| Canonicalization | `d = dim + r if dim < 0 else dim` **before** any tagger or gate. `tag_concat_dim` returns `"w"` for `r−1`, `"h"` for `r−2`, and `"outer"` otherwise. `dim=-1` and `dim=3` of rank 4 are the same cell and the same program. |
| Scenario dict | built exactly as `eval/golden_tests/g6_concat/axes.py:scenario_of_call` builds it: `shapes`, raw `dim`, `dtype`, `layout`, `input_buffers` (`"l1"` iff `BufferType.L1`), `memory_config` (`"l1"` iff `memory_config.buffer_type == L1`, else `None`). Taggers are shared, so the runtime cell equals the golden cell. |
| Output | `ttnn.allocate_tensor_on_device(ttnn.Shape(out_shape), dtype, layout, device, memory_config or ttnn.DRAM_MEMORY_CONFIG)`, allocated on every call, which a trace captures (BB-16) |
| Dispatch | exactly one `ttnn.generic_op([*tensors, output], descriptor)`; no other device op before or after it (BB-01/BB-02) |
| Program cache (BB-07) | A module-level dict maps the key (above) to `(ProgramDescriptor, plan)`. The descriptor is built once per key with `custom_program_hash = hash(key)`. On a hit only the address common RT args (`K` input base addresses and the output base address) are rewritten. If nanobind properties return copies, keep the cache entry's `KernelDescriptor` objects and reassign `descriptor.kernels`. |
| Plan report (BB-08/BB-21) | `g6_concat_plan(tensors, dim=0, *, memory_config=None) -> dict` uses the same key and cache and dispatches nothing. It returns `regime`, `num_cores`, `core_ranges`, `block_pages`, `depth`, the CB bytes, the L1 total and the per-core role (`"page_copy reader+writer"` / `"row_gather reader+writer"` with its page range). |
| Profiler zones (BB-21) | `DeviceZoneScopedN` per block operation: reader `G6C-FETCH` (`page_fetch_block` / `gather_fetch_block`, barrier included), `G6C-PLACE`, `G6C-RESERVE` (the `cb_reserve_back` wait); writer `G6C-WAIT` (`cb_wait_front`), `G6C-STORE` (writes and flush), `G6C-DRAIN` (final barrier). Under the profiler build only; the plain build is unchanged. |
| Kernel size (BB-04/BB-18) | Each regime has its own reader source (`g6_concat_page_reader.cpp`, `g6_concat_gather_reader.cpp`); one shared writer. Shape-derived numbers are RT args, so one binary serves every shape per regime × dtype × placement mask. |
