# Operation Design: concat

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement. No compute kernel: every output byte is a NoC copy of an input byte |
| Goal | `torch.cat(tensors, dim)` for a list of tensors that share rank, dtype, layout and every size except `dim`. One `ttnn.generic_op` dispatch per call. Output is DRAM interleaved (R5). |
| Math | `out[..., o_k + i, ...] = tensors[k][..., i, ...]` along `dim`, with `o_k = Σ_{j<k} tensors[j].shape[dim]` |
| References | `eval/specs/concat/spec.py` (the authority); `.claude/references/targets/quasar.md`; `.claude/references/backends/ttsim.md`; `.claude/references/quasar_op_examples/` (`toy_thread_plan`: host style, implicit/explicit reader and writer); `tt_metal/impl/dataflow_buffer/DataflowBuffer.md` Part A; `ttnn/ttnn/operations/examples/master.md` (`double_buffer`, `tile_reorder`, `split_reader`, `trid_double_issue`, `noc_placement`, `bank_stagger`) |

## Target

| Fact | Value | Source |
|------|-------|--------|
| Target | `quasar` on the ttsim simulator | `.claude/references/targets/quasar.md` + `.claude/references/backends/ttsim.md` |
| Arch | quasar | target page |
| Compute grid | 8×4 = 32 clusters. Queried at run time with `device.compute_with_storage_grid_size()` | target page |
| L1 per core | 4,194,304 B (4 MiB), shared by the cluster's 4 NEOs and 8 DM cores | target page |
| Unreserved budget | `ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES`. The value on this target was not probed and is ≤ 4,194,304 B. This op uses 32 KiB per core at Phase 0, so the budget clamp never binds on this target | target page (query); value `unknown` |
| Host program | `ttnn.ProgramSpec` (Metal 2.0) + `ttnn.generic_op`. Reference: `.claude/references/quasar_op_examples/` (`toy_thread_plan/toy_thread_plan_program_artifacts.py`) | target page |
| Threads per core | 4 NEOs (C ∈ {1,2,4}); 6 user DM cores (R + W ≤ 6); on STRIDED rings R and W divide C. A work unit may hold DM kernels only (`tt_metal/impl/metal2_host_api/program_spec/validation/placement/work_unit.cpp:82-87,131-144`: at most one compute kernel, none required) | target page; validation code |
| Ring limits | ≤ 32 DFBs per cluster; one ring ≤ 1,048,560 B; 16 semaphores; DM kernel binary ≤ 48 KiB | target page, cluster model |
| NoC transfer alignment | 1 B for both L1 and DRAM (`tt_metal/hw/inc/internal/tt-2xx/quasar/noc/noc_parameters.h:486,490`). `L1_ALIGNMENT = 16` and `DRAM_ALIGNMENT = 64` constrain allocation, not transfers | NoC parameters header |
| Unavailable here | `ttnn.ProgramDescriptor` and Metal 1 program factories (refused at kernel creation, `tt_metal/impl/kernels/kernel.hpp`). `bfloat8_b` / `bfloat4_b` have no path on this part (out of SUPPORTED; refused in `validate()`). uint16 / uint32 are not DFB formats (`validation/resource/dfb.cpp:148`): a later dtype refinement omits `data_format_metadata` on the ring, which is legal because no compute kernel binds it (`dfb.cpp:272-281`), or carries uint16 as bfloat16 and uint32 as int32. Untagged NoC transfers after an implicit-sync program hang the program (library gaps): every explicit transfer is tagged, and every kernel hands its trid back. `DEVICE_PRINT` from compute prints nothing on ttsim, which does not affect this op (no compute kernel). Compute-helper gaps (reduce modes, `matmul_block`, `minmax`, `UnpackToDest`, eltwise pack target) do not apply: this op uses no compute helper. ALL-pattern DM consumer under implicit sync deadlocks (`tests/.../dataflow_buffer/dfb_test_common.hpp:343-352`), so the ring is STRIDED | target page; `.claude/references/quasar-library-gaps.md` |

The worked numbers below are for `quasar`: 32 clusters, bf16 tiles of 2,048 B. The op evaluates every predicate with the run-time queries.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `tensors` | `list[ttnn.Tensor]` | yes | length ≥ 1 (X4); Phase 0 accepts length ∈ {2, 3} | — | host. `count` is CT (number of tensor bindings) |
| `dim` | `int` | no | `[-ndim, ndim-1]` (X3). Canonicalised to `dim if dim < 0 else dim - ndim` **before** the support check (R2). Phase 0 accepts {-1, -2} | `0` | host. The concat axis (`concat_is_w = dim == -1`) is CT |

The entry point is `concat(tensors, dim=0)`, matching the specification stub exactly. `validate()` builds the axes from `(dim, *shapes)` with `INPUT_TAGGERS = AXES` (imported from `eval.specs.concat.spec`), plus `dtype`, `layout` and the two memory axes `eval.memory.memory_axes(t)` of **every** tensor (R6). The checks run in this order:

1. **Malformed calls** raise `ValueError` / `IndexError`:
   - an empty list (X4);
   - rank mismatch (X2);
   - `dim` out of range (X3);
   - any non-`dim` size mismatch (X1).
2. **A mixed list** raises `NotImplementedError`: dtype, layout or memory axes that differ between tensors, or tensors on different devices. None of these is a cell of TARGET.
3. **Support check** against SUPPORTED raises `NotImplementedError` (R4).

## Tensors

### Input

| Property | Requirement (Phase 0 = PHASE0) |
|----------|-------------|
| Shape | rank 3 or 4; equal sizes in all dims but `dim`; any alignment (H and W need not be multiples of 32) |
| Dtype | bfloat16 (all tensors the same) |
| Layout | TILE (32×32 tiles, four 16×16 faces) |
| Memory | DRAM or L1, interleaved (R3). An L1 input is read through its `TensorAccessor` exactly like a DRAM one |

### Output

| Property | Value |
|----------|-------|
| Shape | `shape_0` with `shape[dim] = Σ_k shape_k[dim]` (spec `output_shape`) |
| Dtype | the inputs' dtype |
| Layout | TILE |
| Memory | DRAM interleaved, always (R5): `ttnn.allocate_tensor_on_device(ttnn.Shape(out_shape), dtype, ttnn.TILE_LAYOUT, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)` |
| Padding | Elements of the output's tile padding are **not written** and their values are undefined. The specification says input padding is not output data (GUIDANCE, "Padding"). `ttnn.to_torch` strips the padding |

## Blocking Model

The op computes nothing, and **every output tile is a pure function of input bytes**: no output tile depends on another output tile. Every axis is therefore independent, and none is dependent. The single scheme-level question is **how an output tile gets its bytes**:

- **Seam on a tile boundary.** Every output tile is exactly one input page. This is the `page_copy` regime.
- **Seam not on a tile boundary.** An output tile is a set of byte **runs** from at most `count + 1` input pages, and one input page feeds two output tiles in disjoint parts. This is the `run_assembly` regime.

Both regimes read each run **straight from the input's DRAM/L1 page into the output ring entry** with a sub-page NoC read (`offset_bytes`, size). Quasar's NoC has 1-byte transfer alignment (`noc_parameters.h:486,490`). So no staging buffer and no CPU copy are needed, and no input byte crosses DRAM twice.

### Axes

Notation: `T = 32` (tile side), `F = 16` (face side), `eb = tile_bytes / 1024` (bytes per element; 2 for bf16). `D_k = shape_k[dim]` is piece `k`'s extent along `dim`, in elements. The piece offset `off_k = Σ_{j<k} D_j`, and `D_out = Σ_k D_k`. The concat tile axis has `L_out = ceil(D_out / T)` tiles, and `L_k = ceil(D_k / T)` per piece. The cross tile axis has `X = ceil(shape[cross] / T)` tiles, where `cross` is `-2` when `dim == -1` and `-1` when `dim == -2`. The number of images is `num_images = Π shape[:-2]` (N·C for rank 4, B for rank 3). Every tile count is `ceil` and per image: TILE layout pads each image independently.

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `out_tile` (the flat output page axis = `image` × `tile_row` × `tile_col` in output page order) | independent: each output tile is assembled from input bytes alone | `block_tiles`: output tiles per reader reservation (reserve → issue all runs → one barrier → push) | `min(BLOCK_TILES = 8, max_core_tiles, ring_cap_tiles, budget_tiles)` → 8 on every Phase 0 cell with ≥ 8 tiles per core; smaller cells clamp to their core share | host constant `BLOCK_TILES` → one CT arg `block_tiles`; ring capacity and loop bounds derive from it | contiguous ranges of `out_tile` over the full grid (`ttnn.split_work_to_cores(grid, total_out_tiles, row_wise=True)`); per-core `start_tile` and `num_tiles` are RT | knob-turn (raise `BLOCK_TILES`; see perf lamp) |
| `image` (leading dims, folded into `out_tile`) | independent: images never mix (concat is along H or W) | folded into `block_tiles` | — | — | through `out_tile` | — |
| `concat` (output tile rows if `dim=-2`, tile cols if `dim=-1`; `L_out` tiles) | independent: output tiles along it never combine. At a non-tile-aligned seam one source tile feeds two adjacent output tiles **in disjoint byte ranges**, so splitting here duplicates neither bytes nor work | folded into `block_tiles` | — | — | through `out_tile`: a core boundary may fall anywhere on it at no cost | — |
| `cross` (the other tile axis; `X` tiles) | independent: all pieces share it, and each output tile reads the same cross index of each piece | folded into `block_tiles` | — | — | through `out_tile` | — |
| `piece` (list index `k`, `count` values) | partition of the `concat` axis among inputs, not an output axis: every output tile touches the pieces that cover its 32-wide window | not split: whole extent | all `count` pieces are visible to every core | `count` CT; `piece_extent[k]` and `piece_offset[k]` are common RT arrays (sized `count`) | every core binds all `count` input accessors | count ∈ {1, 5}: knob-turn (count is the CT arg and the array size) |
| `in_tile` (32×32 elements inside one output tile: face × face-row × element) | intra-page: the tile is the atomic output page | whole tile, not split: a run never crosses a source page | runs per output tile: 1 (page copy); ≤ 2·(count+3) for `dim=-2`; ≤ 32·(count+3) for `dim=-1` with an element-granular seam | derived per tile by the run generator (below) | — | — |

**Why the three tile axes fold into one.** The op treats `image`, `tile_row` and `tile_col` identically: the same run generator serves every output tile, and the run *template* depends only on the tile's position along `concat`. The three are contiguous in output page order. So the block is honestly 1-D over `out_tile` (`blocking-model.md` §1). A core's assignment can start and end mid-line, mid-image or mid-seam with no halo and no re-read. That is why the split is over output tiles and not over lines: for example, `[1,1,32,4096]×2` along W is one tile-row of 256 tiles, and it still fills all 32 cores.

**What a block is here.** A block is `block_tiles` consecutive output pages in one ring reservation. The bytes moved are constant in `block_tiles`, because no operand is fetched per block. The extent therefore trades only the fixed per-block cost (one `reserve_back`, one `async_read_barrier` = one DRAM round-trip of latency, one `push_back`) against pipeline overlap with the writer. The coarsest block is *not* taken: one block per core would serialise the whole read against the whole write (overlap lamp). The catalog measured the sweet spot of reads per barrier at 4–8 with a ring of 2× block (`double_buffer`). On Quasar, blocks of 4 and 8 sit at the DRAM floor (`quasar_op_examples/README.md`, "What the timed simulator says"). `BLOCK_TILES = 8` is that measured point, recorded as a lamp.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_output_tiles` | `OUTPUT_DEPTH` (blocks resident) | 2 → capacity `2 · block_tiles` = 16 entries = 32 KiB (bf16) | The reader fills block i+1 while the writer drains block i. The capacity is a whole multiple of `block_tiles`, so every reservation of `block_tiles` entries is contiguous and never wraps. Implicit sync on the writer side gets 2 txn ids (`2·block_tiles % 2 == 0`) and so double-buffers its credits too |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| DFB ring size ≤ 1,048,560 B (target page, cluster model) | `block_tiles` | `ring_cap_tiles = floor(1,048,560 / (OUTPUT_DEPTH · tile_bytes))` = 255 for bf16; a named host constant (`RING_MAX_BYTES`; Python has no query) | ProgramSpec build error |
| L1 budget | `block_tiles` | `budget_tiles = floor((ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES) / (OUTPUT_DEPTH · tile_bytes))` | allocation failure |
| Core's own share | `block_tiles` | `≤ max_core_tiles` (largest per-core count of the split) | ring larger than any core uses: wasted L1 |
| Sub-page NoC read: `offset_bytes + size` must stay inside one page (`api/tensor/noc_traits.h:135-143`, `internal/tensor/generated_noc_addr.h:50-67`) | run length | the run generator splits at every multiple of `F = 16` in source coordinates, and so at every source tile boundary (32 = 2·16) | reads the next page's bytes into the run: **wrong values**, no error |
| Explicit reservation contiguity | `block_tiles` vs ring capacity | capacity = `OUTPUT_DEPTH · block_tiles` exactly. A core's ragged last block reserves and pushes only the tiles it fills, and that is the core's last reservation | a reservation that straddles the ring end writes past the ring: **corrupts L1** |
| `reserve_back(uint16_t)` (`dataflow_buffer.h:243`) | `block_tiles` | ≤ 255 already (ring cap) | — |
| Input accessor lookup by name (`tensor::get_token_if_present<"in_k">()`, `api/tensor/tensor_binding_token.h:177-199`) | `count` | `count ≤ MAX_INPUTS = 5` (TARGET's largest count); bindings `in_0 … in_{count-1}` | a missing accessor fails to compile |
| NoC transaction-id flow control (`noc.h:259-262`: a TXN_ID read waits for free transactions on its trid) | runs per block | none needed: the issue loop stalls itself, and correctness is unaffected | — |

### Regimes

The memory boundary is DRAM (or the input's L1 bank, for L1 inputs). The **minimum** is: each output page is written once, and each input byte that lands in the output is read once.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `page_copy` | built | `dim ∈ {-1,-2}` ∧ `∀ k < count-1: D_k % 32 == 0` (every seam on a tile boundary) | `block_tiles` output pages | **Minimum.** Output tile `p` along `concat` is page `p - tile_off_k` of the unique piece `k` with `tile_off_k ≤ p < tile_off_k + L_k` (`tile_off_k = off_k / 32`, exact because every earlier extent is aligned). The last piece's tile padding becomes output padding. 1 whole-page read + 1 whole-page write per output tile (`tile_reorder`: whole pages are the fastest transfer shape) | The reader uses implicit sync: one tagged read per entry, with no reserve, barrier or push in the kernel. A bigger ring only lengthens the ISR batches (fewer credit interrupts). The toy measured blocks of 1, 4 and 8 all at the floor. Fixed cost per call: program launch on each engaged core |
| `run_assembly` | built | `dim ∈ {-1,-2}` ∧ `¬page_copy` (some seam not on a tile boundary) | `block_tiles` output pages, each assembled from its runs | **Minimum in bytes:** only bytes that land in an output tile are read, and none twice. Input padding along `dim` is never read; cross-axis padding is read and lands in output padding, which is harmless and lets runs merge. **Above minimum in transactions:** `dim=-2` (row-granular, every run a multiple of 32 B at 32-B-aligned offsets): ≤ 2 runs per (segment, tile), typically 2–6 per tile. `dim=-1` with every seam a multiple of 16 columns (face-granular): 512 B runs, ≤ 8 per tile. `dim=-1` otherwise (element-granular): ~2 runs of ≤ 32 B per output face-row, ~64–160 per tile (issue-bound; see the `staged_shift` row and lamp) | Explicit-sync reader: one reserve, one barrier (≈ one DRAM round-trip of latency) and one push per block, amortised over `block_tiles` tiles. The run *generation* per tile is not per-block cost |
| `staged_shift` | deferred: `run_assembly` already serves every cell it would serve, and with fewer DRAM bytes. It pays off only if element-granular `dim=-1` seams prove issue-bound, and no YOLOv8 call reaches that predicate (the traced joins are `page_copy` or row-granular) | `dim == -1` ∧ `∃ k < count-1: off_{k+1} % 16 ≠ 0` | a line segment of `block_tiles` output tiles along `concat` | Fewer, larger transactions: whole source pages into an L1 staging ring (a halo of ≤ 1 page per piece per segment), then the DM hart copies element runs L1→L1 (or self-aimed reads, `kernel_lib/local_copy_helpers_dataflow.hpp:74-95`). DRAM bytes ≈ input pages of the segment (whole pages, more than `run_assembly`'s exact bytes); transactions drop from ~100 to ~2 per tile | Halo amortised over the segment; staging ring sized `≤ block_tiles + count + 1` pages. Reachable: the split, the writer and `cb_output_tiles` are unchanged; it adds one staging ring between reader and assembly and a CPU-copy step in the reader. Settle with one measurement: reader-issue time per output tile, `run_assembly` vs `staged_shift`, on `(4,8,32,47)+(…,100)+(…,50)` |
| `leading_dim_copy` | deferred: outside PHASE0 (`dim ∈ {-1,-2}`) | `dim ∈ {-3,-4}` | `block_tiles` output pages | **Minimum.** Joins along N or C move whole images (whole pages) always; it is `page_copy` with an image-level source map (output image → piece, local image) | Same as `page_copy`. Knob-turn: a new source-index map inside `page_copy`. No new kernel structure |
| `row_major_runs` | **built (Refinement 1)** as two host regimes, `row_major_copy` and `row_major_runs` | `layout == ROW_MAJOR`; `row_major_copy` when `dim == -2`, `row_major_runs` when `dim == -1` | `block_tiles` output sticks (default `default_block_pages`: as many sticks as `BLOCK_TILES` tiles hold bytes, so the single source stays `BLOCK_TILES`) | **Minimum.** `row_major_copy`: each output stick is one whole input stick (the `page_copy` shape in the stick domain): 1 read + 1 write per stick, implicit sync. `row_major_runs`: output stick `q` is input stick `q` of every piece side by side: `count` sub-page reads (`ext_k · elem_bytes` B from offset 0 into the entry at `off_k · elem_bytes`; Quasar's NoC moves any byte alignment) + 1 write per stick; explicit sync, one barrier per block | `row_major_runs`: one reserve, one barrier, one push per block of sticks. Ring entry = output stick rounded up to the L1 alignment (`page_geometry`), ≤ every input's and the output's aligned page, so whole-entry transfers stay inside their pages |
| `per_tile_full_reread` | rejected: a dead end, superseded by `run_assembly` | — | one output tile | Reads every contributing source tile whole into L1, then copies with the CPU: each straddled source tile crosses DRAM twice (2× on every shifted piece) plus CPU copies of every element | — |
| `neo_untilize_tilize` | rejected: superseded by `run_assembly` | — | a tile-row | Untilize on the NEOs, concat row-major sticks, tilize back: 2 compute passes, still byte copies for element shifts, and 32-bit dtypes would have to survive DEST (R1 risk) | — |

**Regime selection function (host, exact):** TILE: `regime = "page_copy" if all(shape_k[dim] % 32 == 0 for k in range(count - 1)) else "run_assembly"`. ROW_MAJOR (Refinement 1): `"row_major_copy" if dim == -2 else "row_major_runs"`. `dim` is the canonical negative index. The predicate depends on shapes only, never on the grid, so a cell lands in the same regime on every board. **Regime-pinned tests are required** (Work Distribution).

### Traffic ranking

| Candidate split | DRAM crossings | Cross-core traffic | Verdict |
|-----------------|----------------|--------------------|---------|
| Contiguous ranges of flat output tiles (`out_tile`) | each input byte that lands in the output once; each output page once | none | **chosen.** Fills the grid even for one-line shapes (`[1,32,4096]×2` W → 256 tiles; `[1,2,400]×2` H → 13 tiles) |
| Whole lines along `concat` per core (one line = one tile-row for W, one tile-col for H) | same bytes | none | ties in bytes, loses on occupancy: `[1,1,32,4096]×2` along W is 1 line, which means 1 core |
| Split by piece (core group per input) | same bytes | none | ties in bytes, but a seam tile needs bytes from two pieces: either two cores write one output page (two writers, one page) or the seam is re-read. Loses on structure |
| Dependent-axis split + combine | n/a: concat has no dependent axis (nothing is reduced, scanned or combined across tiles) | — | not applicable |

All splits that move no byte twice tie on bytes. The chosen one wins on occupancy, and occupancy never overrode the traffic ranking here. **Operand-reuse check:** under the `out_tile` split, no input operand is read by more than one core except at a core boundary that cuts a seam tile, where the two cores read *disjoint* runs of the same source page. No operand is reuse-shared, so no broadcast regime row exists. **Stall-shadow check:** the only wait is the reader's per-block barrier. What runs in its shadow is the writer draining the previous block (depth 2). A second block's reads could also be issued before the first barrier (two alternating user trids, `trid_double_issue`): recorded as a perf lamp, because it changes no structure.

### Block schedule

Logical schedule per core (`num_blocks_this_core = ceil(num_tiles / block_tiles)`, `last_block_extent = num_tiles - (num_blocks_this_core - 1)·block_tiles`):

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    gather_block(block_idx);   // reader: fill block_extent output pages of cb_output_tiles from input pages
    store_block(block_idx);    // writer: write those pages to the output tensor
}
```

| Operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------|-------------|--------------------|-----------------------------------|
| `gather_block` (`page_copy` form) | `block_extent` output pages | nothing but the ring | Accessor construction for all `count` inputs: once per kernel. `(img, tile_row, tile_col)` is decomposed from `start_tile` once per kernel and then incremented. Piece lookup is incremental (advance `k` when `p` passes `tile_off_k + L_k`). One tagged implicit read per page; no credit calls; trid hand-back once at kernel end |
| `gather_block` (`run_assembly` form) | `block_extent` output pages × their runs | nothing but the ring | Per block: `reserve_back(block_extent)`, then every run of every page, each a tagged (`trid 0`) sub-page read into `entry j` at `j·tile_bytes + dst_off`, then `async_read_barrier(trid 0)`, then `push_back(block_extent)`. The run template depends only on the tile's position `p` along `concat`. Generating it on the fly per tile is the default; caching templates per `p` is the implementer's choice |
| `store_block` | `block_extent` output pages | nothing | One tagged implicit write per page to page id `start_tile + i`; no credit calls; trid hand-back once at kernel end |

**The run generator** (the block operation's internal walk; semantics fixed here, realisation left to the implementer). Coordinates inside a tile: element `(r, c)` sits at byte `off(r,c) = ((r/16)·2 + c/16)·256·eb + ((r%16)·16 + c%16)·eb`.

1. **Pieces in the window.** For output tile position `p` along `concat`, the window is `[32p, 32p+32)`. Intersect it with each piece's `[off_k, off_k + D_k)`. A non-empty intersection `[a, b)` has piece-local coordinates `l = g - off_k`, source tile `j = l / 32`, source in-tile coordinate `v = l % 32` and destination coordinate `u = g - 32p`.
2. **Segments.** Split `[a, b)` into maximal segments on which neither `u` nor `v` crosses a multiple of 16.
3. **Runs per segment.** A segment has length `n` and starts at `(u0, v0)`.
   - `dim = -2` (rows): for each column half `h ∈ {0,1}` emit `src = (page(k, img, j, tile_col), off(v0, 16h))`, `dst = off(u0, 16h)`, `len = n·16·eb`.
   - `dim = -1` (cols): for each row `r ∈ [0, 32)` emit `src = (page(k, img, tile_row, j), off(r, v0))`, `dst = off(r, u0)`, `len = n·eb`.
4. **Emission order** (the order that lets whole faces and whole tiles merge):
   - `dim = -2`: segments, then halves.
   - `dim = -1`: row-half `rh ∈ {0,1}` outermost, then segments, then rows `16·rh … 16·rh+15`.
5. **Merge.** Merge a run into the previous one when both have the same source page and both `src_off` and `dst_off` continue the previous run. With this rule an aligned tile becomes 1 run of 2,048 B and full faces become 512 B or 1,024 B runs.

The source page id is `page(k, img, tr_k, tc_k) = img·Ht_k·Wt_k + tr_k·Wt_k + tc_k`. For `dim=-1`, `Ht_k = ceil(H/32)` and `Wt_k = L_k`. For `dim=-2`, `Ht_k = L_k` and `Wt_k = ceil(W/32)`.

Worked cases (YOLOv8's three traced joins, `quasar`, bf16):

| Call | Regime | Output tiles | Cores | Runs per tile |
|------|--------|--------------|-------|---------------|
| `[1,1,64,64]+[1,1,64,80]`, dim 3 | `page_copy` | 2·5 = 10 | 10 | 1 |
| `[1,64,144]+[1,16,144]+[1,4,144]`, dim 1 | `run_assembly` (seam at 80) | 3·5 = 15 | 15 | tile rows 0–1: 1 run (2,048 B). Tile row 2: 3 runs (1,024 B from piece 1; 2 × 128 B from piece 2) |
| `[1,2,400]+[1,2,400]`, dim 1 | `run_assembly` (seam at 2) | 1·13 = 13 | 13 | 4 runs of 64 B (faces 2–3 are output padding, not written) |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Overlap: `BLOCK_TILES = 8` | Bytes are constant in the extent. 8 is the catalog's WH/BH sweet spot (`double_buffer`) and the Quasar toy's floor, not a Quasar measurement of this op | `BLOCK_TILES ∈ {4, 32, whole core share}` on `(2,512,1024)×2` (64 tiles per core) |
| Grid synchronization | Small cells give each engaged core 1 tile (YOLOv8's calls: 10, 15 and 13 tiles), so program launch per core may dominate | engage `min(grid, ceil(total_out_tiles / MIN_TILES_PER_CORE))` cores with `MIN_TILES_PER_CORE ∈ {2, 4}` |
| Reader issue rate in element-granular `run_assembly` | ~64–160 small reads per tile can make the single reader issue-bound (`split_reader`) | thread plan (R = 2, W = 1): two reader threads on the STRIDED ring, tiles dealt round-robin; also the `staged_shift` regime row |
| Barrier per block (`run_assembly`) | One barrier per block drains all in-flight reads (`trid_double_issue`) | two user trids alternating per block: retire block i-1 after issuing block i |
| DRAM bank placement | Every core starts its range on consecutive pages; with 2 DRAM channels, bank alignment may collide (`bank_stagger`) | rotate each core's start within its range |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| Input DRAM/L1 pages → `cb_output_tiles` (`page_copy`) | whole bf16 tile pages | reader: `noc.async_read<NocOptions::TXN_ID>(acc_in_k, cb_output_tiles, {.page_id = src_page}, {})`: implicit sync, one entry per read | the DM0 ISR posts credits; kernel ends with `noc_async_read_set_trid(0, noc.get_noc_id())` |
| Input pages → byte runs inside `cb_output_tiles` entries (`run_assembly`) | sub-page byte ranges, 1-B aligned | reader: `reserve_back(block_extent)`; per run `noc.async_read<NocOptions::TXN_ID>(acc_in_k, cb_output_tiles, len, {.page_id, .offset_bytes = src_off}, {.offset_bytes = j·tile_bytes + dst_off}, {.trid = 0})`; `noc.async_read_barrier<NocOptions::TXN_ID>({.trid = 0})`; `push_back(block_extent)` | the ring is listed in the reader's `disable_dfb_implicit_sync_for` (the `toy_thread_plan` `_dm_config` pattern). A DFB destination offset is relative to the write pointer (`dataflow_buffer.h:607-610`) |
| `cb_output_tiles` → output DRAM | whole tile pages | writer: `noc.async_write<NocOptions::TXN_ID>(cb_output_tiles, acc_out, {}, {.page_id = start_tile + i})`: implicit sync | kernel ends with `noc_async_write_set_trid(0, noc.get_noc_id())`. An explicit producer facing an implicit consumer is supported per side (`tests/tt_metal/tt_metal/quasar_addrgen/test_tensor_accessor_addrgen.cpp:492-519`) |

**Placement axes as regimes.** TARGET's `sharding` axis is `["interleaved"]` only, so there is no physical-shard scheme to map. A sharded tensor is refused by name through `memory_axes` (R6). `memory_config ∈ {DRAM, L1}` changes nothing in the scheme: an L1 input is just another accessor source. Were sharding added later, cutting `out_tile` is independent on every axis, so a height/width/block shard of the output would be a knob-turn (a placement change). An input shard would be consumed in place through a ring backed on the sharded buffer, never re-read over the NoC.

**No inter-Tensix communication** in any regime, built or deferred.

**Host program shape.** One `ttnn.ProgramSpec`:

- **Kernels:** `reader` and `writer` only (no compute kernel). Both are DM `KernelSpec`s, `num_threads=1`. One `WorkUnitSpec` over the engaged cores.
- **Rings:** one `DataflowBufferSpec` `cb_output_tiles`.
- **Tensor parameters:** `in_0 … in_{count-1}`, `out`. The reader binds every input, with accessor names `in_k`. The writer binds `out`.
- **Dispatch:** `ttnn.generic_op([in_0, …, in_{count-1}, out], spec, run_args, {"in_0": 0, …, "out": count})`, with the output last (`ttnn/cpp/ttnn/operations/generic/generic_op_nanobind.cpp:33-47`).
- **Reader per regime:** a CT arg `regime` selects the reader's form, and the host sets the reader's `disable_dfb_implicit_sync_for=["cb_output_tiles"]` in `run_assembly` and `row_major_runs` only (`EXPLICIT_REGIMES`).

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one output tile (page) of the flat `out_tile` axis. A block is `block_tiles` of them |
| Grid | `device.compute_with_storage_grid_size()` (8×4 = 32 on `quasar`) |
| Per-core work | `ttnn.split_work_to_cores(grid, total_out_tiles, row_wise=True)`, where `total_out_tiles = num_images · Ht_out · Wt_out`, `Ht_out = ceil(H_out/32)` and `Wt_out = ceil(W_out/32)` (per image, `ceil`). Each core gets a contiguous `[start_tile, start_tile + num_tiles)`, as in `toy_thread_plan.split_tiles` |
| Remainder | `split_work_to_cores`' two groups (per-core counts differ by 1). The last block is ragged: `last_block_extent`; the reader reserves and pushes exactly that many, and the writer writes exactly `num_tiles` |
| Bound | **data movement.** Per output tile the op moves 2,048 B in and 2,048 B out and computes nothing; `run_assembly` adds run generation (a few integer operations per run) and more NoC issues, both data-movement-processor work |
| Thread plan | **(R, C, W) = (1, 0, 1)**: one reader thread, no compute kernel, one writer thread. Host constant `THREAD_PLAN = {"reader_threads": 1, "writer_threads": 1}`, passed as CT args (kernels size their loops from them). The page's rule for an op bound by data movement is one thread of each kind. A compute kernel here would only unpack and pack bytes: it adds a pass and buys nothing |
| Unit dealt to threads | the ring entry (one output tile). With R = W = 1 the dealing is trivial. A later R > 1 (lamp) deals tiles round-robin: reader thread `t` gathers tiles `t, t+R, …`, and with entry-sized units no round padding is needed |
| Tests the plan needs | (1) With one thread of each kind, the writer's page ids cover `[start_tile, start_tile + num_tiles)` exactly once per core; a correct full output shows every thread ran. (2) In one device session: the op after another program (run it twice in a row, and after a tt-metal op), and the op followed by `ttnn.to_layout(output, ttnn.ROW_MAJOR_LAYOUT)`, in both regimes (`page_copy` uses implicit reads; `run_assembly` uses explicit tagged reads) |
| Regime-pinned tests | `page_copy`: `(-1,(1,1,64,64),(1,1,64,80))`. `run_assembly` row-granular: `(-2,(1,64,144),(1,16,144),(1,4,144))` and `(-2,(1,2,400),(1,2,400))`. `run_assembly` face-granular (seam at a multiple of 16): `(-1,(1,1,32,48),(1,1,32,16))`. `run_assembly` element-granular: `(-1,(1,1,32,50),(1,1,32,14))` and `(-1,(4,8,32,47),(4,8,32,100),(4,8,32,50))`. Ragged last block: a core share not a multiple of `block_tiles` (e.g. `(1,(2,512,1024),(2,512,1024))`: 64 tiles per core at `BLOCK_TILES = 8` is exact, so also test `BLOCK_TILES = 5` through a test-only override). One test asserts which regime the host chose for each of these (the host exposes the selection function) |

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Sync | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|------|-----------|
| `cb_output_tiles` | 0 | one output page, `page_bytes` (`page_geometry`): TILE `out.buffer_page_size()` (2,048 B for bf16); ROW_MAJOR the output stick `W_out · elem_bytes` rounded up to `ttnn.get_l1_alignment()` | `OUTPUT_DEPTH · block_tiles` (= 16) | spans `out_tile` (`block_tiles` per block × depth); does not scale with `piece`, `in_tile`, `image`, `concat` or `cross` | output dtype (bfloat16). A DM-only ring: format metadata optional | reader | writer | reader: **implicit** in `page_copy`, **explicit** (reserve/push, every transfer tagged trid 0, opted out) in `run_assembly` and `row_major_runs`; implicit in `row_major_copy`. Writer: implicit | whole kernel |

One ring is the minimal inventory. Runs land directly in the destination entry (Rule 3, pattern 1: pack into the destination). The run lists are generated in registers and stored nowhere. There are no staging pages, scratchpad, scaler or intermediate buffers.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `gather_block` | `block_extent ≤ block_tiles` output pages | no (DM NoC API) | — (input tensors via `TensorAccessor` `in_0 … in_{count-1}`) | `cb_output_tiles`, `block_extent` entries | `block_extent` entries posted (by the ISR in `page_copy`; by `push_back` in `run_assembly`) |
| 2 | `store_block` | `block_extent` output pages | no (DM NoC API) | `cb_output_tiles`, `block_extent` entries | — (output tensor via `TensorAccessor` `out`) | entries acked by the ISR; the ring is balanced at kernel exit |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `gather_block` (`page_copy`) | raw_api | `Noc::async_read` implicit-DFB overload | `tt_metal/hw/inc/api/dataflow/noc.h:915-927` | `<NocOptions::TXN_ID>(acc_in_k, cb_output_tiles, {.page_id}, {})` | — | `cb_output_tiles` | none per call. The loop bound is `num_tiles` |
| `gather_block` (`run_assembly`) | raw_api | `DataflowBuffer::reserve_back` / `push_back` | `tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:243-244` | `block_extent` | — | `cb_output_tiles` | `block_tiles` |
| `gather_block` (`run_assembly`) | raw_api | `Noc::async_read` generic overload (sub-page, DFB destination + offset) | `noc.h:246-278`; DFB dst address `dataflow_buffer.h:607-610`; accessor src `{.page_id, .offset_bytes}` `api/tensor/noc_traits.h:135-143` | `<NocOptions::TXN_ID>(acc_in_k, cb_output_tiles, len, {.page_id, .offset_bytes = src_off}, {.offset_bytes = j·tile_bytes + dst_off}, {.trid = 0})` | — | `cb_output_tiles` | — |
| `gather_block` (`run_assembly`) | raw_api | `Noc::async_read_barrier` | `noc.h:739-744` | `<NocOptions::TXN_ID>({.trid = 0})` | — | — | once per block |
| `gather_block` (accessors) | raw_api | `tensor::get_token_if_present<"in_k">()` + `TensorAccessor` | `api/tensor/tensor_binding_token.h:177-199`; `tensor_accessor.h:124-128` | `k < count` (CT), dispatched on the runtime piece index | — | — | `count` |
| `store_block` | raw_api | `Noc::async_write` implicit-DFB overload | `noc.h:930-941` | `<NocOptions::TXN_ID>(cb_output_tiles, acc_out, {}, {.page_id = start_tile + i})` | `cb_output_tiles` | — | — |
| kernel exit (both) | raw_api | `noc_async_read_set_trid(0, noc.get_noc_id())` / `noc_async_write_set_trid(0, …)` | as used by `quasar_op_examples/toy_thread_plan/kernels/reader_implicit.cpp` and `writer_implicit.cpp` (last statement) | — | — | — | — |
| RT/CT args | raw_api | `get_arg(args::x)`; arrays `get_arg(args::piece_extent)[k]` | `tt_metal/hw/inc/experimental/kernel_args.h:77-183` | CT: `count`, `concat_is_w`, `regime`, `block_tiles`, `tile_bytes`, `reader_threads` / `writer_threads`. Common RT: `Ht_out`, `Wt_out` (and the arrays `piece_extent[count]`, `piece_offset[count]`). Per-core RT: `start_tile`, `num_tiles` | — | — | `block_tiles` (CT, single source `BLOCK_TILES`) |

**Helpers considered and rejected (applies to every raw_api row above).** The op has no compute phase, so no compute helper applies.

- `tilize_helpers.hpp` / `untilize_helpers.hpp`: they convert layout; nothing here changes layout, and the round trip is the rejected `neo_untilize_tilize` regime.
- `eltwise_convenience.hpp` `copy`: copies whole tiles through DEST and cannot shift an element inside a tile. A whole-tile copy is a NoC read already.
- `mcast/kernel/mcast_args.hpp`: no operand is reuse-shared, so there is nothing to multicast.
- `local_copy_helpers_dataflow.hpp:74-158` (self-aimed L1→L1 reads, `set_read_trid`, `async_read_barrier_with_trid`): L1→L1 only, and no run here starts in local L1 because runs come straight from DRAM/L1 input pages. It is the mechanism of the deferred `staged_shift` regime, and `set_read_trid` / `async_read_barrier_with_trid` (`:147,158`) are the mechanism of the trid lamp.
- `reduce_helpers_dataflow.hpp`: scaler setup, not applicable.
- `tile_column_dataflow.hpp` / `index_tile_dataflow.hpp`: produce index/column tiles; not a copy of tensor pages.

No `compute_kernel_hw_startup` is needed: there is no compute kernel.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Unaligned sub-page NoC reads on ttsim | Element-granular `dim=-1` runs have 2-byte-granular source/destination offsets that differ modulo 16. Older ports conservatively aligned offsets to 16/64 | Quasar declares 1-B NoC alignment (`noc_parameters.h:486,490`) and the sanitizer enforces the same. Row-granular and face-granular runs are 32-B aligned regardless. If ttsim rejects an element-granular run, only that predicate needs the `staged_shift` row. It is an escalation, not a silent change |
| Run crossing a page | A run that runs past `offset + size > tile_bytes` reads the neighbouring page silently | the run generator splits at every multiple of 16 in source coordinates (mechanism cap) |
| Input padding overwriting data | A non-last piece's tile padding along `dim` holds garbage. A page copy would paste it over the next piece's data | `page_copy` is selected only when every non-last extent is a multiple of 32. `run_assembly` reads only `[0, D_k)` along `dim` |
| Sticky trid | `page_copy`'s implicit reader and the implicit writer leave a ring's trid in the tag register. The explicit reader tags everything trid 0. A following tt-metal program (`ttnn.to_layout`) aborts with `tile counter overflowed capacity` | both kernels end with the set_trid(0) hand-back. Session tests in both regimes (Work Distribution) |
| Mixed ring protocols | The reader is explicit in one regime and implicit in the other; a stale host opt-out hangs | the host derives `disable_dfb_implicit_sync_for` from the same `regime` value that it passes as CT arg. A single source |
| Ragged last block | Pushing `block_tiles` when only `last_block_extent` were filled leaves the ring unbalanced, which hangs at exit (drain) | the reader pushes exactly `block_extent`. The ragged block is the core's last reservation, so contiguity holds |
| Variable input count | `tensor::in_3` does not compile when unbound | `get_token_if_present<"in_k">()` under `if constexpr (k < count)`, dispatching the run's piece index to the right accessor |
| Positive `dim` | INPUTS gives `dim=3` / `dim=1` (YOLOv8 calls). A literal gate fails them | canonicalise first (R2); tag with `AXES["dim"]` |
| Output placement | Inputs in L1 tempt `out` to follow them | the output is always DRAM interleaved (R5); the verifier reads `output.memory_config()` |
| Output padding undefined | `run_assembly` does not write output padding, so the ring's stale bytes land there | the spec excludes padding from output data; golden checks compare `to_torch` values. Document in the op's docstring |
| Program cache across shapes | Piece extents differ per call | extents and offsets are common RT args. CT holds only `count`, `concat_is_w`, `regime`, `block_tiles` and `tile_bytes` |

### Structural impossibilities

None beyond the specification's INVALID (`bfloat8_b` × ROW_MAJOR). `bfloat8_b` on Quasar is the target's own skip (`unbuildable on quasar`), not an INVALID entry.
