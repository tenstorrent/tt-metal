# Operation Design: permute

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (no arithmetic; one regime uses the FPU only to transpose a tile) |
| Goal | `torch.permute` for a bfloat16 tensor of rank 3 or 4, row-major or tile layout, interleaved in DRAM or L1, any permutation and any alignment (PHASE0). It runs as **one** `ttnn.generic_op` dispatch per call. Which of three `ttnn.ProgramSpec`s it dispatches depends only on the layout and the canonical permutation. |
| Math | `output[i_0..i_{r-1}] = input[j_0..j_{r-1}]` with `j_{dims[k]} = i_k`. Every output element is one input element, bit for bit (R1). |
| References | `eval/specs/permute/spec.py` (the authority); `.claude/references/targets/quasar.md`; `.claude/references/backends/ttsim.md`; `.claude/references/quasar-library-gaps.md`; `.claude/references/quasar_op_examples/` (host-program pattern, thread plan, ring protocols); `tt_metal/impl/dataflow_buffer/DataflowBuffer.md` Part A + C2; `.claude/references/blocking-model.md`; `.claude/references/l1-footprint-discipline.md`; perf catalog `ttnn/ttnn/operations/examples/master.md` (entries `tile_reorder`, `double_buffer`, `noc_placement`, `split_reader`, `trid_double_issue`) |

## Target

| Fact | Value | Source |
|------|-------|--------|
| Target | `quasar` on the ttsim simulator | `.claude/references/targets/quasar.md` + `.claude/references/backends/ttsim.md` |
| Arch | quasar | target page |
| Compute grid | 8 × 4 = 32 clusters (probe 2026-11-21: `compute_with_storage_grid_size()` → `8-4`) | target page; probe |
| L1 per core | 4,194,304 B | target page |
| Unreserved budget | `ttnn.get_max_worker_l1_unreserved_size() - l1_reserve_bytes`. The query returns **3,963,328 B** on this target (probe 2026-11-21). With the Phase 0 reserve of 262,144 B and tensors in DRAM that leaves **3,701,184 B**. | probe; Mechanism caps |
| Host program | `ttnn.ProgramSpec` (Metal 2.0) + `ttnn.generic_op`, as in `.claude/references/quasar_op_examples/toy_thread_plan/toy_thread_plan_program_artifacts.py` | target page |
| Threads per core | 4 NEO compute engines, one compute thread each, so C ∈ {1,2,4}. 6 user data-movement harts, so R + W ≤ 6; on STRIDED rings R and W each divide C. 32 DFBs per cluster; a ring holds at most 1,048,560 B. 16 semaphores. Binaries: 48 KiB DM, 24 KiB compute. | target page *Cluster model* / *Thread plan* |
| DM core caches | Each DM core has a 4 KB write-back L1 D$ and a 128 KB L2 shared by the DM cores; both use 64 B lines (`tt_metal/hw/inc/internal/tt-2xx/risc_common.h:196-213`). A RISC-V store to L1 is not visible to the NoC until `flush_l2_cache_range` (`tt_metal/hw/inc/internal/tt-2xx/quasar/cache.h:28`). A RISC-V load of NoC-landed data needs `invalidate_l2_cache_range` first (`risc_common.h:282`). `invalidate_l1_cache()` is a no-op on Quasar DM (`risc_common.h:326`). | headers |
| Unavailable here | `ttnn.ProgramDescriptor` / Metal 1 kernels and the legacy C++ program factory (refused at kernel creation). bfloat8_b / bfloat4_b (no path; not in TARGET anyway). Library gaps that bind this design: `copy_tile` refuses any transpose (`tt_metal/hw/inc/api/compute/tile_move_copy.h:56-57`), so the eltwise chain's `CopyTile` cannot transpose. DPRINT from a compute kernel prints nothing. `fpu_math_fidelity` must be set on every compute kernel (ttsim). uint16 / uint32 are not buffer formats: a ring that carries uint32 later declares int32. An untagged NoC transfer after an implicit-sync program hangs, so every transfer of an explicit-sync kernel is tagged and every DM kernel hands its tag back. The eltwise helpers keep the previous pack target, so `pack_init` before a helper whose output ring changes; this design has one output ring per compute kernel, so the gap does not bite. `UnpackToDest` of a bfloat16 page into 32-bit DEST aborts ttsim; Phase 0 transposes bf16 with 16-bit DEST and no UnpackToDest. Reduce / matmul gaps do not apply: no reduce, no matmul. | target page *Not available*; `.claude/references/quasar-library-gaps.md` |

All worked numbers below are for this target: 32 clusters and a 3,963,328 B unreserved query. Every predicate the op evaluates calls the query.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | PHASE0: bfloat16; TILE or ROW_MAJOR; DRAM or L1 interleaved; rank 3 or 4 | — | — (tensor binding) |
| `dims` | `list` or `tuple` of `int` (R3) | yes | A permutation of `range(rank)`; entries in `[-rank, rank-1]`. Negative entries are read as `d + rank` before the support check (R2). | — | Host only: it becomes page strides, box extents and box order as RT args |

The entry point equals the stub `permute(input_tensor, dims) -> ttnn.Tensor`. No other public parameter. No compute config (the spec says the op computes nothing).

### Host-side contract (entry point, in order)

| Step | What | Rule |
|------|------|------|
| 1 | `dims = list(dims)` | R3 |
| 2 | Reject malformed `dims` with `ValueError`: wrong length (X1), an entry outside `[-rank, rank-1]` (X3), a repeated dimension after normalization (X2). | REJECTIONS |
| 3 | Normalize: `d = [x + rank if x < 0 else x for x in dims]`. | R2 (before the support check) |
| 4 | Derive axes: `INPUT_TAGGERS = AXES` from `eval.specs.permute.spec`, called on `(tuple(input_tensor.shape), tuple(d))`; `memory_config` / `sharding` from `eval.memory.memory_axes(input_tensor)`; plus `dtype`, `layout`. | R8, `op_template.py` |
| 5 | `validate()`: raise `NotImplementedError` for any axis value outside `SUPPORTED` (= PHASE0) or any EXCLUSIONS entry (none in Phase 0). | R6 |
| 6 | Canonicalize to rank 4: `k = 4 - rank`; `s = [1]*k + shape`; `d4 = list(range(k)) + [x + k for x in d]`. Output shape `t[i] = s[d4[i]]`; logical output shape = the spec's `output_shape`. | Rank 2 later reuses this unchanged |
| 7 | Allocate the output: `ttnn.allocate_tensor_on_device(ttnn.TensorSpec(out_shape, input.dtype, input.layout, memory_config=input.memory_config()), device)`. It is interleaved, in the input's buffer type. | R7 |
| 8 | `regime = select_regime(layout, d4)` (Regimes table). Build that regime's ProgramSpec and dispatch one `ttnn.generic_op([input, output], spec, run_args, tensor_indices)`. | One dispatch per call |

Identity `dims` is not special-cased. It lands in `page_gather` and the op returns a copy, which the spec's DECISIONS allow.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | rank 3 or 4 (PHASE0); any sizes; H (−2) and W (−1) need not be multiples of 32 |
| Dtype | bfloat16 (PHASE0) |
| Layout | ROW_MAJOR (a page is one row of W elements, `W·2` B) or TILE (a page is one 32×32 tile, 2048 B, face-major; padded per image to `ceil(H/32)·ceil(W/32)` tiles) |
| Memory | interleaved, DRAM or L1 |

### Output

| Property | Value |
|----------|-------|
| Shape | `[shape[d[i]] for i in range(rank)]` |
| Dtype | the input's |
| Layout | the input's. In TILE the output is padded for its own shape, and its padding is written as 0 by `staged_gather`. Under `page_gather` / `tile_transpose` the padding is the input's padding moved along with it, so no input padding lands on a logical output element (R4). |
| Memory | interleaved, in the input's buffer type (R7) |

## Blocking Model

Semantics: `.claude/references/blocking-model.md`. Index space: the rank-4 canonical input axes `lead0, lead1, h, w`, with sizes `s[0..3]`, canonical permutation `d[0..3]` and output sizes `t[i] = s[d[i]]`. The two input axes that become the output's last two are `a = d[2]` and `b = d[3]`.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `lead0` (input axis 0; the padded 1 for rank 3) | **independent**: each output element reads exactly one input element, so no result spans this axis | `page_gather` / `tile_transpose`: folded into the flat output-page run (`pages_this_core`). `staged_gather`: `gather_extent[0]` | Page regimes: the core's whole run from `split_work_to_cores`. `staged_gather`: 1, grown by the coarsen step (Work Distribution) while blocks ≥ `cores × gather_threads` and the footprint fits | Host: the split for page runs, `choose_gather_block()` for `gather_extent`. Passed to the kernels as RT args. | Spread over the full grid (`row_wise=True`) | knob-turn |
| `lead1` (input axis 1) | **independent** (same reason) | Same as `lead0`, `gather_extent[1]` | Same rule | Same | Same | knob-turn |
| `h` (input axis 2) | **independent** (same reason). Also an in-tile axis of the input in TILE. | Page regimes: inside the page (one tile = 32 rows) in TILE; folded into the page run in RM. `staged_gather`: `gather_extent[2]`. Its quantum is 32 in TILE (whole input tiles); in RM it is 32 only when `h == b`. | Page regimes: the page. `staged_gather`: one quantum (`min(s[2], 32)`) in TILE; in RM 1, or the whole axis if `h == b`. The occupancy / coarsen steps may change it. | `choose_gather_block()` | Spread over the grid (blocks) | knob-turn |
| `w` (input axis 3) | **independent** (same reason). It is inside the input page in both layouts. | Page regimes: inside the page (whole row or 32 columns). `staged_gather`: `gather_extent[3]`. Phase 0 reads whole input rows; in RM it is chunked only by the fit or occupancy step, in multiples of `align_in_elems`. | `staged_gather`: RM `s[3]` (whole row, so whole-page reads), TILE `min(s[3], 32)` | `choose_gather_block()` | Spread over the grid only via chunks when blocks are too few | knob-turn |

No axis is dependent: permute has no reduction or combine, so a dependent-axis split does not exist. No operand is **reuse-shared**. Under every split, each core reads a disjoint set of input pages, so no core re-reads bytes another core reads, and no broadcast row exists.

The operand-reuse check was run on every split. The single input varies along every split axis because blocks partition it.

Intermediate data: the `staged_gather` staging area and output slots have the same four axes and are **private to one worker thread**. No core or thread collects or redistributes for others, so no stage computes on one core while the rest wait.

**Rank 2** canonicalizes to `(1, 1, H, W)` by the same rule. It is a SUPPORTED-only refinement with no kernel change.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_pages` (`page_gather`) | `page_ring_depth` | 8 entries | Implicit sync splits the ring into txn-id halves (DataflowBuffer.md §"Implicit sync"), so 4 reads and 4 writes stay in flight per half. Catalog `double_buffer`: 4–8 in flight per barrier is the sweet spot, and RM pages can be as small as 2 B, where latency, not bandwidth, is the limit. |
| `cb_input_tiles` (`tile_transpose`) | `transpose_ring_blocks` × `transpose_block_tiles` × C | 2 × 4 × 1 = 8 entries | Double-buffers the compute block: the reader fills block k+1 while the NEO transposes block k. The capacity is a multiple of `transpose_block_tiles`, so a block never straddles the ring end. |
| `cb_output_tiles` (`tile_transpose`) | Same knobs | 8 entries | Double-buffers the pack against the writer |
| `scratch_gather` staging (`staged_gather`) | `staging_depth` | 1 block per worker thread | The overlap of reads and the shuffle comes from the thread plan: while one thread waits on its read barrier, the other five shuffle (Stall shadow). Depth 2 is a perf lamp. |
| `scratch_gather` output slots (`staged_gather`) | `out_slots` | 8 slots per worker thread | Output pages are written asynchronously from the slots, and the thread waits for its writes (`async_writes_flushed`) only once every 8 pages. Each slot is ≤ 1 tile or one output-row segment, which is small next to the staging. |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| NoC sub-page transfer alignment | RM `gather_extent[3]` (an input-row chunk) and RM `gather_extent[b]` (an output-row segment) | Each is either the whole axis or a multiple of `align_in_elems` / `align_out_elems`, where `align = (ttnn.get_dram_alignment() if buffer is DRAM else ttnn.get_l1_alignment()) // element_bytes` (bf16: 32 in DRAM, 8 in L1). So every sub-page offset `start·element_bytes` is buffer-aligned. | A transfer whose source and destination offsets disagree within the alignment moves the wrong bytes or faults. |
| Staging slot alignment | Every staged input page or row segment and every output slot | Slot stride `align_up(bytes, ALIGN_MAX)`, with `ALIGN_MAX = max(get_l1_alignment(), get_dram_alignment())` = 64 B | A misaligned landing slot breaks the NoC read. |
| DM cache line (64 B, write-back) | Each worker thread's scratch region and every output slot | Thread t's region starts at `t × per_thread_bytes`, with `per_thread_bytes` a multiple of 64. Staging is invalidated after the read barrier. Each slot is flushed before its NoC write and never invalidated. | One thread's invalidate discards another thread's dirty line, or the NoC sends stale bytes. Either way the output is silently wrong. |
| Implicit-sync transfer size = DFB entry size | `cb_pages` entry | `entry_size = align_up(page_bytes, get_l1_alignment())`. This is ≤ the buffer's aligned page slot, so the extra bytes stay inside the page's own slot. Implicit sync is used only where every entry is a whole page. | A write larger than the slot overwrites the next page. This is why `staged_gather`'s sub-page writes do not use a ring. |
| Ring capacity 1,048,560 B | `cb_pages` entry × depth | RM row bytes × 8 ≤ the ring cap, i.e. W ≤ 65,535 bf16 elements. Larger rows are refused by `validate()` as a `NotImplementedError` naming the cap. | Program build fails. |
| DEST capacity (16-bit DEST, half sync: 8 tiles) | `transpose_block_tiles` | `min(4, DEST_AUTO_LIMIT)` (`ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp`) | DEST overflow corrupts tiles. |
| `staged_gather` minimum block in TILE: one 32-quantum on each of `{h, w, a, b}` | The tile footprint `Π ceil(min(s_j,32)/32)` over in-box tiles, at most 32·32 tiles (2 MiB bf16) when both `a` and `b` are leading axes | Threads engaged per core: `min(gather_threads, floor(budget / min_per_thread_bytes))`. If that is 0, `validate()` refuses with `NotImplementedError`. Phase 0 bf16 always fits ≥ 1 thread: 2,113,536 B ≤ 3,701,184 B. | The scratchpad allocation exceeds L1 and the program fails to build. |
| L1-resident tensors share the program's L1 | `l1_reserve_bytes` | `l1_reserve_bytes = 262,144 + Σ_{L1 tensors} ceil(num_pages / num_l1_banks) × aligned_page_bytes`, with `num_l1_banks = ttnn.device.GetMemoryView(device, ttnn.BufferType.L1).num_banks` (`ttnn/cpp/ttnn-nanobind/device.cpp:617-625`) | The scratchpad region clashes with an L1 tensor bank at program creation. |
| DM kernel binary ≤ 48 KiB | — (not an extent) | Keep the gather kernel's per-element loop generic in layout via CT args, not unrolled per permutation | Kernel build fails. |

### Regimes

The minimum is stated against DRAM (or the L1 banks for an L1 tensor): each input byte crosses once and each output byte crosses once.

| Regime | Status | Predicate (on `layout`, canonical `d`) | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|------------------------------------------|-------|---------------------------|--------------------------|
| `page_gather` | **built** | `(layout == ROW_MAJOR and d[3] == 3)` or `(layout == TILE and d[2] == 2 and d[3] == 3)`. Every output page is exactly one input page. This covers `leading`, the identity, and `general` RM permutations that keep W last. | The core's run of `pages_this_core` consecutive output pages, streamed through `cb_pages` | **Minimum.** Each input page is read once and each output page written once, as whole-page transfers (catalog `tile_reorder`: whole pages, not faces). | Nothing per block beyond the ring handshakes, which implicit sync posts per txn-id half. The run is the whole core share, so program-level fixed costs (kernel launch, accessor setup) are paid once per core. |
| `tile_transpose` | **built** | `layout == TILE and d[2] == 3 and d[3] == 2` (`last_two`, plus `general` with H and W swapped and leading axes moved) | The run of `pages_this_core` output tiles. Compute works in blocks of `transpose_block_tiles`. | **Minimum.** Each input tile is read once and each output tile written once. Output tile `(o0,o1,ht,wt)` = transpose(input tile `(i0,i1,wt,ht)`). | `transpose_init` once per kernel. One DEST acquire / commit / pack window per `transpose_block_tiles` tiles instead of per tile. Ring handshakes per block. |
| `staged_gather` | **built** | Everything else: RM with `d[3] != 3`; TILE with `(d[2], d[3])` ∉ {(2,3), (3,2)}. Every output page draws on several input pages. | A box over the four input axes with extents `gather_extent[0..3]`. Quanta: TILE 32 on `{2, 3, a, b}`, 1 elsewhere; RM `align_in_elems` on axis 3 and `align_out_elems` on axis `b` (or the whole axis), 1 elsewhere. The input footprint is resident; output pages are streamed through the slots. | **Minimum in bytes.** The boxes partition the input, so each input page or row segment is read exactly once. Every output page lies in exactly one box: the box covers both output in-page axes in whole quanta, so each output page is written once. Above the minimum in **transactions** for tiny RM rows: each row is one NoC read however short. The input layout forces this. | One read barrier, one staging invalidate and one loop setup per block instead of per page. Fewer, longer bursts. The per-element shuffle cost does not depend on the extent. |
| `staged_gather_subtile` (face-granular boxes for the `a`, `b` both-leading TILE case) | **deferred**: the minimum TILE block, 32·32 tiles × 4 B = 4 MiB, exceeds the budget once float32 joins SUPPORTED. In PHASE0 (bf16) the minimum block is ≤ 2 MiB and always fits one thread, so `staged_gather` serves every PHASE0 cell. | Same predicate as `staged_gather`, plus `a ∉ {2,3} and b ∉ {2,3} and min_per_thread_bytes > budget` | The quantum on axes `h`, `w` drops from 32 to 16 (one face, 512 B contiguous for bf16, 1 KiB for fp32), read by face-offset sub-tile reads | Same as `staged_gather` (minimum) | Reachable: only `choose_gather_block()`'s quanta and the reader's per-page read size change. The box machinery, staging layout and shuffle are unchanged. |
| `cube_fpu_assist` (TILE general with one of `a`, `b` ∈ {h, w}: FPU-transpose the 32-tile cube on a NEO, then move whole 32 B tile-row chunks on the DM) | **deferred**: `staged_gather` already serves every shape it would serve. It adds a compute kernel and a second ring protocol to the regime, and its gain is confined to the shuffle stage. | Subset of `staged_gather`'s TILE predicate | The same 32-tile cube that `staged_gather` stages | Same (minimum) | Reachable: the cube is already `staged_gather`'s block. `shuffle_block` is one named stage that a NEO stage can replace. |
| `rm_plane_transpose_neo` (RM with W moving: tilize → transpose → untilize on a NEO instead of the DM shuffle) | **deferred**: `staged_gather` covers every RM shape it would serve, including non-32-aligned planes, which this path cannot handle without padding. Tilize/untilize on Quasar sit behind `ARCH_QUASAR` branches (`tt_metal/hw/inc/api/compute/tilize.h:445`, `pack_untilize.h:32`) that this op would be the first to rely on. | Subset of `staged_gather`'s RM predicate | `staged_gather`'s RM block | Same (minimum) | Reachable: the same stage replacement as above |
| `per_page_regather` (each output page reads every input page it needs, nothing resident) | **rejected**: superseded by `staged_gather`. It is a dead end that re-reads each input page up to 32× (a cube) or 1024× (both leading). | — | One output page | 32×–1024× the input crossings | — |
| `element_noc_gather` (one NoC transfer per element, straight into the output slot) | **rejected**: superseded by `staged_gather`. An element-sized transfer whose source and destination offsets differ within the 16 B / 64 B alignment is illegal, and one transaction per element is a dead end. | — | — | Minimum bytes, but 512–1024× the transactions | — |

#### Regime-selection function (pinned)

```
select_regime(layout, d):            # d = canonical rank-4 permutation
    if layout == ROW_MAJOR and d[3] == 3:                 return "page_gather"
    if layout == TILE and d[2] == 2 and d[3] == 3:        return "page_gather"
    if layout == TILE and d[2] == 3 and d[3] == 2:        return "tile_transpose"
    return "staged_gather"
```

It depends only on `(layout, d)`, never on the grid or on sizes, so a regime cannot be reachable on one device and not another. Regime-pinned tests are required (Work Distribution). How the PHASE0 cells land:

| Cell kind | RM | TILE |
|-----------|----|------|
| `leading` (incl. identity) | `page_gather` | `page_gather` |
| `last_two` | `staged_gather` | `tile_transpose` |
| `general`, W stays last (e.g. `(0,2,1,3)`) | `page_gather` | `staged_gather` |
| `general`, H↔W swapped plus leading moved (e.g. `(1,0,3,2)`) | `staged_gather` | `tile_transpose` |
| `general`, otherwise | `staged_gather` | `staged_gather` |

### Traffic ranking

| Candidate split | DRAM crossings (input / output) | Cross-core traffic | Verdict |
|-----------------|-------------------------------|--------------------|---------|
| Output-page runs, page regimes. Output-major boxes, `staged_gather`. | 1 / 1 | none | **Chosen.** Minimum bytes. Output-major order gives each core a contiguous output run and the writer a (start, count). |
| Input-page runs (each core scatters its input pages to wherever they go) | 1 / 1 | none | Tie in bytes. Not chosen: output-major keeps writes contiguous per core, and the output padding is produced where the output tile is assembled. |
| Split a dependent axis + combine | n/a | n/a | No dependent axis exists (no reduction). |
| Broadcast / multicast of a shared operand | n/a | n/a | No operand is read by more than one core. |
| `per_page_regather` | 32–1024 / 1 | none | Rejected (Regimes). |

The traffic minimum is reached by every built regime. What remains is extent ranking, i.e. per-block fixed cost, and the bound of each regime (Work Distribution).

**Stall shadow.** The only waiting stage is `staged_gather`'s read barrier before `shuffle_block`. Work independent of that wait is the shuffle and the writes of other blocks. The design schedules it there through the thread plan: six worker threads per cluster own disjoint blocks, so while one waits on its reads the others shuffle and write. The writes are asynchronous, so a thread's previous block's writes drain during its own next read. No floating-point order exists to preserve; this is a pure move. In `page_gather` and `tile_transpose` nothing waits except ring credits, which implicit sync posts.

### Block schedule

`page_gather` (reader → `cb_pages` → writer; no compute kernel):

```cpp
for (block_idx = 0; block_idx < num_blocks_this_core /* = 1: the core's page run */; ++block_idx) {
    load_block(block_idx);    // reader: for each output page p of the run, read input page src(p) into cb_pages
    store_block(block_idx);   // writer: write each cb_pages entry to output page p
}
```

`tile_transpose` (reader → `cb_input_tiles` → compute → `cb_output_tiles` → writer):

```cpp
for (block_idx = 0; block_idx < num_blocks_this_core /* = 1: the core's tile run */; ++block_idx) {
    load_block(block_idx);        // reader: input tile src(p) for each output tile p
    transpose_block(block_idx);   // compute: transpose_block_tiles tiles per DEST window, packed to cb_output_tiles
    store_block(block_idx);       // writer
}
```

`staged_gather` (one data-movement kernel, `gather_threads` threads, each a full worker; no rings):

```cpp
for (block_idx = thread_id; block_idx < num_blocks_this_core; block_idx += gather_threads) {
    gather_block(block_idx);    // read the box's input footprint into this thread's staging
    shuffle_block(block_idx);   // assemble each output page of the box into an output slot
    store_block(block_idx);     // write each slot to its output page (+ offset), out_slots in flight
}
```

| Operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------|-------------|--------------------|-----------------------------------|
| `load_block` (page regimes) | `pages_this_core` pages. The source page comes from an odometer over the output page grid `g[0..3]` with input-page strides `st[0..3]`: `src = Σ c_k·st[k]`. | `cb_pages` / `cb_input_tiles` window (8 entries) | Accessor setup once per kernel. One tagged implicit-sync read per page, no barrier or push. Tag handed back once at kernel end. |
| `transpose_block` | `transpose_block_tiles` tiles per window (the last window of a core may be shorter) | DEST window | `compute_kernel_hw_startup` + `transpose_init` once per kernel. Acquire/commit/wait/release, wait/pop and reserve/push once per window. |
| `store_block` (page regimes) | `pages_this_core` pages, consecutive output page ids from `start_page` | `cb_pages` / `cb_output_tiles` window | One tagged implicit-sync write per page. Tag handed back once at kernel end. |
| `gather_block` | Box `gather_extent[0..3]` → input footprint: RM `Π_{j≠3} e_j` row segments of `e_3` elements; TILE `Π_j ceil(e_j/q_j)` tiles over in-box tile coords (`q_j` = 32 on in-tile axes, 1 otherwise) | This thread's staging, for the whole of shuffle_block | Issue every read of the footprint (tagged trid 0), then **one** `async_read_barrier` and **one** `invalidate_l2_cache_range(staging, staging_bytes)` per block |
| `shuffle_block` | All output pages of the box. RM: output rows over the box's output axes 0..2, each a segment of `e_b` elements at element offset `start_b`. TILE: output tiles over the box's output tile grid, each 32×32 positions. | Staging (read only); `out_slots` slots | Loop bounds and the per-axis staging strides are set up once per block. Per output page: one slot fill and one `flush_l2_cache_range(slot)`. In TILE, positions outside the logical output are written as 0. |
| `store_block` (`staged_gather`) | Same pages as shuffle_block, interleaved with it page by page | Slots | One tagged `async_write` per page, at `(page_id, offset_bytes = start_b·element_bytes)` with size `e_b·element_bytes` in RM, a whole tile in TILE. One `async_writes_flushed` per `out_slots` pages before a slot is reused. One `async_write_barrier` at kernel end. |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| `page_gather` plan (1,0,1) | RM pages can be 2–100 B, so the op may be bound by transactions issued per hart, not bytes (catalog `split_reader`) | (2,0,2): two readers and two writers dealing pages round-robin over the STRIDED `cb_pages` |
| `page_ring_depth` 8 | Small RM pages may want more in flight; large ones less | 4 and 16 |
| `tile_transpose` plan (1,1,1) and `transpose_block_tiles` 4 | One FPU datacopy per tile is expected to sit on the DRAM floor (README: multiply-only at the floor with 1,1,1; blocks of 4/8 changed nothing). The transpose unpack may cost more than a multiply. | (1,2,1) and (2,2,2); `transpose_block_tiles` 1 and 8 |
| `gather_threads` 6 | Six harts share one 128 KB L2. The shuffle's strided staging reads may thrash it, and fewer threads may be as fast. | 4 and 2 threads |
| Overlap: `staging_depth` 1 | Each thread serializes read → barrier → shuffle. Threads hide each other's barriers only if enough blocks exist per core. | Depth 2 per thread (prefetch block k+1 on a second trid while shuffling block k), at half the per-thread block bytes |
| `out_slots` 8 | Tiny output rows (32 B) may want more writes in flight; tiles fewer | 2 and 32 |
| Grid synchronization (`staged_gather` on tiny tensors) | Cores-first dealing spreads e.g. 20 blocks over 20 clusters. Launch and per-cluster setup may dominate such small work. | Threads-first dealing: `ceil(n_blocks / gather_threads)` clusters with 6 blocks each |
| NoC for `staged_gather` writes | Every worker uses the reader config's NoC0 for both reads and writes. The default reads-NoC0 / writes-NoC1 split (catalog `noc_placement`) is not applied. | A second `Noc` object on NoC1 for the writes |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| `page_gather` read | Whole input page (RM row or tile) → `cb_pages` entry | `noc.async_read<NocOptions::TXN_ID>(acc_in, cb_pages, {.page_id = src}, {})`, implicit sync | The reader is the ring's only producer. Tag handed back at the end (`noc_async_read_set_trid(0, noc.get_noc_id())`). |
| `page_gather` write | `cb_pages` entry → output page | `noc.async_write<NocOptions::TXN_ID>(cb_pages, acc_out, {}, {.page_id = p})`, implicit sync | `noc_async_write_set_trid(0, …)` at the end |
| `tile_transpose` read | Input tile → `cb_input_tiles` | Implicit-sync read, as above | Same reader source as `page_gather`; only the strides `st[k]` differ |
| `tile_transpose` compute | Tile in `cb_input_tiles` → DEST (transposed) → `cb_output_tiles` | `transpose_init` / `transpose_tile` / `pack_tile` (raw API, see API Mapping), 16-bit DEST, `fpu_math_fidelity` set explicitly | A bf16 unpack → datacopy → pack is bit-exact |
| `tile_transpose` write | `cb_output_tiles` → output tile | Implicit-sync write | Same writer source as `page_gather` |
| `staged_gather` read | Input page or row segment → this thread's staging slot (64 B aligned) | `noc.async_read<NocOptions::TXN_ID>(acc_in, staging_slot, size, {.page_id, .offset_bytes}, {}, {.trid = 0})`, then `async_read_barrier<TXN_ID>({.trid = 0})` | No rings in this regime. Every transfer is tagged (target page *Ring protocols*). |
| `staged_gather` shuffle | Staging → output slot, element by element (16-bit moves for bf16, 32-bit later) | RISC-V loads/stores on the worker hart. `invalidate_l2_cache_range` on staging after the barrier; `flush_l2_cache_range` on each slot before its write. | Dtype-agnostic bit moves (R1). TILE in-tile offset of `(y, x)` = `((y>>4)·2 + (x>>4))·256 + (y&15)·16 + (x&15)` elements. |
| `staged_gather` write | Output slot → output page at `offset_bytes` | `noc.async_write<NocOptions::TXN_ID>(slot, acc_out, size, {}, {.page_id, .offset_bytes}, {.trid = 0})` | Several clusters write disjoint, buffer-aligned segments of the same RM output row. |

**TARGET's placement axis.** TARGET's `sharding` holds only `interleaved`, so no physical-shard scheme is owed. If a sharded placement were added, it would be placement, not a new algorithm: the boxes are the logical shards, and every axis is independent, so each would be a knob-turn. A shard would be read in place through a ring backed on the sharded buffer, never re-read over the NoC.

**TARGET values outside PHASE0, classified:**

| Value | Class | What changes |
|-------|-------|--------------|
| `dtype` float32 / int32 / uint32 | `page_gather`, `staged_gather`: knob-turn (`element_bytes` CT arg = 4; DFB `data_format_metadata` int32 for uint32). `tile_transpose`: knob-turn plus the 32-bit compute path. | 32-bit transpose on Quasar needs `UnpackToDestEn` with `enable_32_bit_dest=True` (`tt_metal/hw/inc/api/compute/transpose.h:72-84`). float32 TILE with both `a`, `b` leading and ≥ 32 needs the `staged_gather_subtile` row. |
| `rank` 2 | knob-turn (SUPPORTED only) | Canonicalization already pads to rank 4 |

**Structural impossibilities:** none to add. Every TARGET cell is buildable.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | `page_gather` / `tile_transpose`: one output page (RM row or tile). `staged_gather`: one box. |
| Grid | `device.compute_with_storage_grid_size()` (8×4 = 32 here) |
| Per-core work | Page regimes: `ttnn.split_work_to_cores(grid, n_out_pages, row_wise=True)` → `(start_page, pages_this_core)` per core. `staged_gather`: `ttnn.split_work_to_cores(grid, n_blocks, row_wise=True)` → `(start_block, num_blocks_this_core)`; thread t takes local blocks `t, t+gather_threads, …`. |
| Remainder | Handled by `split_work_to_cores`' two groups. Every count is alignment-aware and per image: RM `n_out_pages = t0·t1·t2`; TILE `n_out_pages = t0·t1·ceil(t2/32)·ceil(t3/32)`. Input tile grid `ceil(s2/32)`, `ceil(s3/32)` per image. Box counts are `Π_j ceil(s_j / gather_extent[j])`. Never `floor` or `//` on a tile count. |
| Bound | `page_gather`: **data movement**, zero compute per page. `tile_transpose`: **data movement**, one FPU datacopy-with-transpose per 2 KiB tile against a 2 KiB read and a 2 KiB write. `staged_gather`: **compute on the data-movement harts**: several RISC-V instructions per element against 2 B moved per element in page-sized bursts. |
| Thread plan | Host constant `THREAD_PLANS = {"page_gather": (1, 0, 1), "tile_transpose": (1, 1, 1), "staged_gather": GATHER_THREADS = 6}`. The page regimes are bound by data movement, so one reader and one writer (and one NEO for the transpose) reach the floor with implicit sync; the target page's *Choosing the plan* gives (1,1,1). `staged_gather` is bound by the hart's shuffle, so it engages all 6 user DM harts as independent workers in one DM kernel (`num_threads=6`, R + W = 6 ≤ 6). It has no compute kernel and no rings, so the STRIDED divisibility rules do not apply. `threads_engaged = min(6, floor(budget / min_per_thread_bytes))` (Mechanism caps). |
| Unit dealt to compute threads | `tile_transpose`: the ring entry itself (one tile), C = 1, so there is no round padding. `staged_gather`: no compute threads. Each **worker thread** is dealt whole boxes, and since threads share no ring, a short last round needs no padding. Dealing is cores-first because no padding exists to avoid (the target page's whole-rounds rule exists to bound padding). When `n_blocks < cores × 6`, some threads idle and no work is duplicated. |
| Tests the plan needs | (1) `test_every_thread_ran` (`staged_gather`). Use an internal entry (`_permute_into(input, dims, output)`, which `permute` calls) with an output pre-filled by a sentinel bit pattern. Pick a shape whose split gives every thread of every engaged cluster ≥ 1 block, e.g. RM `(2, 512, 1024)` dims `(0, 2, 1)` → 192 blocks on 32 × 6. Assert an exact match, with no sentinel left. Loops are sized from the CT `gather_threads`, not `get_num_threads()`, so a missing thread leaves sentinel pages. (2) The same shape of test for `tile_transpose` and `page_gather`. (3) In one device session, per regime: `ttnn.add` (or another regime's permute) before `permute`; and `permute` followed by `ttnn.to_layout(output, ROW_MAJOR_LAYOUT)` for TILE (`TILE_LAYOUT` for RM), checking both results. (4) Regime-pinned tests: one per regime asserting `select_regime(layout, d)` returns it and the result matches `torch.permute` exactly. Include a `staged_gather` TILE case with both `a` and `b` leading (`(2,3,4,5)`, dims `(3,2,1,0)`). |

### `choose_gather_block()` (host, `staged_gather` only; single source of `gather_extent[0..3]`)

Inputs: `s`, `d`, layout, `element_bytes`, `align_in_elems`, `align_out_elems`, `workers = n_cores × threads_engaged`, `budget_per_thread = align_down((unreserved − l1_reserve_bytes) / threads_engaged, 64)`.

| Step | Rule |
|------|------|
| 0. Quanta | TILE: `q_j = 32` for `j ∈ {2, 3, a, b}`, else 1. RM: `q_3 = align_in_elems`, `q_b = align_out_elems`, else 1. |
| 1. Start | TILE: `e_j = min(s_j, q_j)`. RM: `e_3 = s_3` (whole rows), `e_b = s_b` (whole output rows), other `e_j = 1`. |
| 2. Fit (mechanism cap) | While `per_thread_bytes(e) > budget_per_thread`: RM halves the longer of `e_3` / `e_b`, rounded up to its quantum (stop at the quantum). TILE is already at its minimum, so apply the thread clamp (Mechanism caps). |
| 3. Occupancy | While `n_blocks(e) < workers` and an RM chunkable axis (`3`, `b`) is above its quantum: for the longer of the two (ties: axis 3), `needed = ceil(workers / n_blocks(e))`; set `e = max(q, align_up(ceil(len / needed), q))`, len being that axis's current extent. Then the other axis. TILE does not split below its quanta. |
| 4. Coarsen | While `n_blocks(e) ≥ 2 × workers` and doubling still fits `budget_per_thread`: double the extent of the free axis that is innermost in output order (`d[3]`, `d[2]`, … order, skipping quantized axes), capped at `s_j`. Then grow a TILE quantized axis by +32. |
| 5. Box order | Block index = mixed radix over `ceil(s_j/e_j)` with significance in output-axis order `d[0]` (slowest) … `d[3]` (fastest). |

`per_thread_bytes(e)` = `align64(staging_bytes) + out_slots × align64(slot_bytes)`.
- RM: `staging_bytes = Π_{j≠3} e_j × align_up(e_3·element_bytes, 64)`, and `slot_bytes = e_b·element_bytes`.
- TILE: `staging_bytes = (Π_{j∈{0,1}} e_j) × ceil(e_2/32) × ceil(e_3/32) × tile_bytes`, and `slot_bytes = tile_bytes`.

Worked on this target. All are DRAM; `workers` = 192; budget 616,832 B per thread.

| Call | Regime | Box `e` | Blocks | Per-block footprint |
|------|--------|---------|--------|---------------------|
| RM `[1,16,20,20]` `(0,2,3,1)` | staged | `(1,16,1,20)` (b = 1, nothing to chunk) | 20 | 16 rows × 64 B staged → 20 output rows of 32 B |
| RM `[1,4,16,8400]` `(0,1,3,2)` | staged | `(1,1,16,192)` | 4 × 44 = 176 | 16 × 384 B staged → 192 output rows of 32 B |
| RM `[1,4,16,400]` `(0,1,3,2)` | staged | `(1,1,16,32)` | 4 × 13 = 52 | 16 × 64 B → 32 rows of 32 B |
| RM `[1,4,8400,1]` `(0,3,1,2)` | staged | `(1,1,192,1)` | 4 × 44 = 176 | 192 × 64 B slots → one 384 B output segment |
| TILE `[1,4,16,400]` `(0,1,3,2)` | tile_transpose | — | 52 tiles over 32 cores | — |
| TILE `[1,4,400,1]` `(0,3,1,2)` | staged | `(1,4,32,1)` | 13 | 4 tiles staged → 1 output tile |
| TILE `[1,400,144]` `(0,2,1)` | tile_transpose | — | 65 tiles over 32 cores | — |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Sync | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|------|----------|
| `cb_pages` | 0 | `align_up(page_bytes, get_l1_alignment())` (tile 2048; RM `align_up(W·2, 16)`) | `page_ring_depth` = 8 | Streams over the page run; the live set is the in-flight window, not the run | bf16 (input dtype) | reader | writer | implicit both sides | whole `page_gather` program |
| `cb_input_tiles` | 1 | tile bytes (2048) | `transpose_ring_blocks × transpose_block_tiles × C` = 2·4·1 = 8 | Spans one compute block (DEST window), double-buffered; streams over the run | bf16 | reader | compute | reader implicit; compute explicit (wait/pop) | whole `tile_transpose` program |
| `cb_output_tiles` | 2 | tile bytes | 2·4·1 = 8 | Same | bf16 | compute | writer | compute explicit (reserve/push); writer implicit | whole `tile_transpose` program |
| `scratch_gather` (a `ScratchpadSpec`, not a ring) | — | bytes | `size_per_node = threads_engaged × per_thread_bytes(gather_extent)` | Thread t's region spans the box's whole input footprint plus `out_slots` output pages | raw bytes (no metadata: DM only) | the worker thread that owns the region | the same thread (its NoC reads/writes and its loads/stores) | n/a (no credits). Explicit trid-0 barriers and cache flush/invalidate. | whole `staged_gather` program |

Each ring has exactly one producer kernel and one consumer kernel. The scratchpad is bound to one DM kernel; each thread touches only its own 64-B-aligned region.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_block` (page_gather / tile_transpose) | `pages_this_core` pages, odometer over `g[0..3]` / `st[0..3]` | no (dataflow) | input tensor (accessor) | `cb_pages` or `cb_input_tiles`, 1 entry per read | entries credited by the DM0 ISR |
| 2 | `transpose_block` | `transpose_block_tiles` tiles (last window ragged) | raw API (no transpose helper exists; see API Mapping) | `cb_input_tiles`: wait k, read in place, pop k | `cb_output_tiles`: reserve k, pack k, push k | — |
| 3 | `store_block` (page regimes) | `pages_this_core` pages from `start_page` | no (dataflow) | `cb_pages` / `cb_output_tiles`, 1 entry per write | output tensor | drained by the DM0 ISR |
| 4 | `gather_block` | box `gather_extent[0..3]` → its input footprint | no (dataflow) | input tensor (accessor) | `scratch_gather` staging of thread t | staging valid after barrier + invalidate |
| 5 | `shuffle_block` | all output pages of the box | no (custom block operation on the DM hart) | staging (read only) | `scratch_gather` slot `k mod out_slots` | slot flushed to L1 |
| 6 | `store_block` (staged_gather) | same pages, interleaved with #5 | no (dataflow) | slot | output tensor at `(page_id, offset_bytes)` | slot reusable after `async_writes_flushed` |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `load_block` | raw_api (dataflow) | `Noc::async_read` implicit-sync overload | `tt_metal/hw/inc/api/dataflow/noc.h:922-927` | `<NocOptions::TXN_ID>(acc_in, cb, {.page_id = src}, {})` | tensor | `cb_pages` / `cb_input_tiles` | `page_ring_depth` (ring capacity) |
| `store_block` (page) | raw_api (dataflow) | `Noc::async_write` implicit-sync overload | `noc.h:936-942` | `<NocOptions::TXN_ID>(cb, acc_out, {}, {.page_id = p})` | `cb_pages` / `cb_output_tiles` | tensor | — |
| tag hand-back | raw_api | `noc_async_read_set_trid` / `noc_async_write_set_trid` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:2558,2579` | `(0, noc.get_noc_id())` as the last statement of every DM kernel | — | — | — |
| `transpose_block` startup | raw_api | `compute_kernel_hw_startup` | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:113` | `(cb_input_tiles, cb_output_tiles)`, once | — | — | — |
| `transpose_block` | raw_api | `transpose_init`, `transpose_tile` / `transpose_block` | `tt_metal/hw/inc/api/compute/transpose.h:37,120,188` | `transpose_init(cb_input_tiles)` once. Per window: `tile_regs_acquire`; `transpose_block(cb_input_tiles, 0, 0, k)`; `tile_regs_commit`. | `cb_input_tiles` | DEST | `transpose_block_tiles` (= k) |
| `transpose_block` pack | raw_api | `tile_regs_wait`, `pack_tile` ×k, `tile_regs_release` | `tt_metal/hw/inc/api/compute/reg_api.h:50,63,96,106`; `tt_metal/hw/inc/api/compute/pack.h:87` | `pack_tile(i, cb_output_tiles)` | DEST | `cb_output_tiles` | `transpose_block_tiles` |
| `gather_block` reads | raw_api (dataflow) | `Noc::async_read` (sized, tagged) + `async_read_barrier` | `noc.h:252,739` | `<NocOptions::TXN_ID>(acc_in, staging_ptr, size, {.page_id, .offset_bytes}, {}, {.trid = 0})` | tensor | scratch staging | `gather_extent[*]` (footprint) |
| `gather_block` invalidate | raw_api | `invalidate_l2_cache_range` | `tt_metal/hw/inc/internal/tt-2xx/risc_common.h:282` | `(staging_base, staging_bytes)` once per block | — | — | — |
| `shuffle_block` | raw_api (RISC-V loads/stores) | `Scratchpad<T>` accessor; `addr_to_l1_ptr` | `tt_metal/hw/inc/api/scratchpad.h:42`; `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:30` | `T` = `uint16_t` (bf16) / `uint32_t` (4-byte dtypes) from the CT `element_bytes` | staging | slot | `gather_extent[*]` |
| `shuffle_block` flush | raw_api | `flush_l2_cache_range` | `tt_metal/hw/inc/internal/tt-2xx/quasar/cache.h:28` | `(slot, slot_bytes)` per output page | — | — | — |
| `store_block` (staged) | raw_api (dataflow) | `Noc::async_write` (sized, tagged) + `async_writes_flushed` + `async_write_barrier` | `noc.h:411,783,760` | `<NocOptions::TXN_ID>(slot_ptr, acc_out, size, {}, {.page_id, .offset_bytes}, {.trid = 0})` | slot | tensor | `out_slots` |

Helpers considered and rejected:

| Raw entry | Candidate helper | Mismatch (file:line) | Why it cannot express the shape |
|-----------|------------------|----------------------|---------------------------------|
| `transpose_block` | `compute_kernel_lib::eltwise_chain` / `copy` with `CopyTile` (`ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp`) | `tt_metal/hw/inc/api/compute/tile_move_copy.h:56-57`: `LLK_ASSERT(transpose == 0, "Transpose not supported on Quasar")` on the copy path `CopyTile` uses. `grep -i transpose ttnn/cpp/ttnn/kernel_lib/eltwise/` is empty: no chain element transposes. | The chain has no transpose element, and its copy path is barred from transposing on Quasar. The gap: a `Transpose<>` chain element would close it. |
| `transpose_block` | `reduce_helpers_compute.hpp` / `matmul_block_helpers.hpp` (the only kernel_lib headers mentioning transpose) | `matmul_block_helpers.hpp` does not compile on Quasar (library gaps). Reduce transposes only as part of a reduction. | Not a movement helper |
| `shuffle_block` | `tile_column_dataflow.hpp` `copy_tile_column` | `ttnn/cpp/ttnn/kernel_lib/tile_column_dataflow.hpp:91-92`: copies one column between tiles of one DFB's tile shape | One column of one tile pair, bound to a DFB. The shuffle gathers arbitrary per-element strides across a staged box and writes RM segments as well as tiles. Its face-offset rule (`:11-14`) is the one the shuffle uses. |
| `gather_block` / `store_block` | `local_copy_helpers_dataflow.hpp` (`set_read_trid`, `read_with_state`) | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:9-34`: L1 → L1 self-aimed copies | Our reads come from the tensor accessor (DRAM / remote L1), not local L1. Its trid batching is a later lamp (`staging_depth` 2). |
| `tilize` / `untilize` helpers | `tilize_helpers.hpp`, `untilize_helpers.hpp` | — | Not used: the op never changes layout. |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Stale or invisible bytes through the DM caches (`staged_gather`) | DM stores sit in a write-back L1 D$ and L2. NoC writes into staging do not update cached lines, and `invalidate_l1_cache()` is a no-op on Quasar. | `invalidate_l2_cache_range(staging)` after every read barrier; `flush_l2_cache_range(slot)` before every slot write; never invalidate a slot; 64-B-aligned per-thread regions. If stale reads persist, add `invalidate_l1_dcache(0)` at block start, after all slots are flushed, so there are no dirty lines to lose. This is a known gap: `invalidate_l2_cache_range` is the pad kernel's precedent (`ttnn/cpp/ttnn/operations/experimental/quasar/pad/device/kernels/dataflow/reader_pad_dims_rm_interleaved_v2.cpp:149-151`). |
| Untagged transfer after an implicit-sync program hangs, or `tile counter overflowed capacity` | Library gap; the page regimes use implicit sync | Every DM kernel ends with `noc_async_{read,write}_set_trid(0, noc.get_noc_id())`. `staged_gather` tags every transfer with trid 0. Session tests (Work Distribution (3)). |
| `transpose_tile` bf16 on Quasar unproven in a generated op | Quasar unpack-transpose path (`transpose.h:85-92`); exercised by LLK tests only | Regime-pinned test first. If it fails on ttsim, `staged_gather` is functionally complete for these cells (its TILE box degenerates to one tile when `{a, b} = {h, w}`). The fallback is a one-line change in `select_regime`, recorded in the changelog. |
| `fpu_math_fidelity` unset aborts ttsim | Library gap | `ComputeHardwareConfig(fpu_math_fidelity=ttnn.MathFidelity.LoFi, enable_32_bit_dest=False)`. Datacopy does no multiply, so LoFi is exact. |
| Several clusters write one RM output row | `staged_gather` chunks along `b` | Chunks are buffer-aligned and disjoint, written with exact sizes, never through an entry-sized implicit write. |
| Tiny RM pages (W = 1 → 2 B) | Each staged row takes a 64 B slot (32× L1 inflation), and each is one NoC transaction | The footprint uses `align_up(e_3·element_bytes, 64)`, so the fit step sees the inflation. The transaction count is the input layout's minimum. |
| TILE output padding | `staged_gather` assembles output tiles element by element | Positions outside `t[2]`, `t[3]` are written 0. Input padding is never read: only logical source coordinates are dereferenced (R4). |
| A ragged compute window straddling the ring end (`tile_transpose`) | The library gap "do not let a `copy_tile` window straddle the end of a ring" applies to indexed unpack | Ring capacity is a multiple of `transpose_block_tiles` and windows start block-aligned. The short window is a core's last, so it never wraps. |
| Ring entry larger than an RM page (`page_gather`) | Implicit sync moves `entry_size` bytes | `entry_size = align_up(page_bytes, l1_alignment)` ≤ the buffer's aligned page slot (DRAM 64 B, L1 16 B), so the extra bytes stay in the page's own padding. |
| uint32 later | Not a buffer format | That ring declares int32 (same width). `staged_gather`'s scratchpad carries no format. |
