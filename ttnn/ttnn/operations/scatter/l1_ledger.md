# L1 Ledger: scatter

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking axes: `width` (face-columns),
`out_line` (output tile-rows along `dim`), `src_line` (source tile-rows along `dim`), `batch` (leading
images; extent 1 in Phase 0). Regime: `resident_line_block`. All buffers are per-cluster `ScratchpadSpec`s
bound to the one DM kernel `scatter` (6 threads); the program has no dataflow buffer and no compute kernel,
so the plan's compute count multiplies nothing.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `scratch_out_column` | `block_out_tile_rows · (block_width_faces // 2 + 1)` pages of `tile_bytes` | the whole block: every output tile-row of every tile-column the block touches, simultaneously (any source row may target any output row; the block is written out only after the last chunk) | `{out_line: spans → block_out_tile_rows (= tensor_out_tile_rows, regime), width: spans → block_width_faces // 2 + 1 tile-columns, src_line: none (written across all chunks, not sized by them), batch: spans → block_images = 1}` | bfloat16 bytes (none declared: DM-only payload, never through DEST) | `scatter` (NoC reads of `input`, then thread stores) | `scatter` (NoC writes to output) | `load_line_block` … `store_line_block` of one block | cannot share with the stages: concurrent (the scatter reads the stages while writing the slab). Capacity exceeds the live set by at most one tile-column (`// 2 + 1` is the worst case of a face range starting on an odd face) and by the unread half of a half-owned tile: mechanism reason — native tile layout keeps one element-offset formula and whole-tile transfers |
| `scratch_src_rows` | `block_src_tile_rows · (block_width_faces // 2 + 1) · src_stage_depth` pages of `tile_bytes` | one chunk of `src` tiles for the block's tile-columns (all threads read any row of it during the chunk) | `{src_line: spans → block_src_tile_rows, width: spans → block_width_faces // 2 + 1, out_line: none, batch: spans → 1}` | bfloat16 bytes (none declared) | `scatter` (NoC reads of `src`) | `scatter` (thread loads) | one chunk (`load_src_chunk` … end of `scatter_chunk`) | cannot share with the slab (concurrent, above) or with `scratch_index_rows` (same lifetime, both read per element); merging the two into one allocation saves nothing. `src` cannot land in the slab: its destination row is data. Same tile-column over-allocation reason as above |
| `scratch_index_rows` | `block_src_tile_rows · 32 · src_stage_depth` pages of `block_width_faces · 16 · 4` B | one chunk of index pieces (one per source row, the block's columns) | `{src_line: spans → 32 · block_src_tile_rows rows, width: spans → block_width_faces · 16 columns, out_line: none, batch: spans → 1}` | INT32 bytes (none declared: uint32/int32 metadata is refused) | `scatter` (NoC reads of `index`) | `scatter` (thread loads) | one chunk | cannot share with `scratch_src_rows` (same lifetime) or the slab (concurrent). Capacity = live set exactly, except the last chunk's rows past `Hs` (unused tail of a full-sized chunk; nominal size keeps one capacity for every chunk) |

### Implementation note (ttnn-implementer, Phase 0)

Every scratchpad is allocated with `CACHE_LINE_BYTES` (64 B) of slack, and the kernel starts its region at the
first 64 B line boundary inside it (`line_start()` in `kernels/scatter.cpp`). Reason: the scratchpad allocator
guarantees only L1 alignment (16 B), and the cache protocol invalidates the stage ranges at every chunk; were a
stage's first or last line shared with the slab, that invalidate would discard the slab's dirty stores (chunk
> 0). With each region on lines of its own, no `scoped_lock` acquire touches another region. Cost: 3 x 64 B per
cluster. A side effect: every NoC transfer's L1 end is 64 B-aligned. Capacities in the table above are the
usable regions; `scratchpad_sizes()` in `scatter_program_artifacts.py` adds the slack.

### Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `tile_bytes` | 2,048 (bf16 32×32); 4,096 for float32 later | input page size (`input.buffer_page_size()`) |
| `tensor_out_tile_rows = ceil(H/32)` | ≤ 1,901 on this target | `resident_line_fits`: `footprint(1,1) <= budget`; else refused (`banded_output_line` deferred) |
| `tensor_src_tile_rows = ceil(Hs/32)` | ≤ `tensor_out_tile_rows` | `index_form == "lines"` (`Hs <= H`) |
| `block_width_faces` | `1 <= wf <= max_cluster_faces = ceil(tensor_width_faces / clusters_engaged)` | `choose_blocking` |
| `block_src_tile_rows` | `1 <= sr <= tensor_src_tile_rows` | `choose_blocking` |
| `block_out_tile_rows` | `= tensor_out_tile_rows` | regime `resident_line_block` |
| `src_stage_depth` | 1 (Phase 0) | host constant `SRC_STAGE_DEPTH`; sizes the stages and reaches the kernel as CT arg `src_stage_depth`, which `static_assert`s 1 until the two-slot ping-pong is built (verifier fix) |
| `SCATTER_THREADS` | 6 (≤ `QUASAR_USER_DM_CORES_PER_NODE`) | host constant, validated by ProgramSpec |
| `CACHE_LINE_BYTES` | 64 (slack per scratchpad, see the note above) | host constant |
| `budget` | `ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES` = 3,963,328 − 65,536 = 3,897,792 B on this target | run-time query; `L1_RESERVE_BYTES` host knob |

### Total per-cluster footprint

```
footprint(wf, sr) = tensor_out_tile_rows · (wf//2 + 1) · tile_bytes          # scales with out_line (regime) and width
                  + sr · (wf//2 + 1) · tile_bytes · src_stage_depth          # scales with src chunk, width, depth
                  + sr · 32 · wf · 64 · src_stage_depth                      # scales with src chunk, width, depth
                  + 3 · CACHE_LINE_BYTES                                      # region-start slack (implementation)
                  <= budget
```

Worked on this target (Phase 0 values, one block and one chunk per cluster):

| Call | `tensor_out_tile_rows` | wf | sr | out column | src | index | total | of budget |
|------|-----|----|----|-----------|-----|-------|-------|-----------|
| traced `[70,512]`, Hs 64 | 3 | 1 | 2 | 6,144 | 4,096 | 4,096 | 14,336 B | 0.4 % |
| model `[2766,2560]`, Hs 2752 | 87 | 5 | 86 | 534,528 | 528,384 | 880,640 | 1,943,552 B | 49.9 % |
| `[1024,64]`, Hs 512 | 32 | 1 | 16 | 65,536 | 32,768 | 32,768 | 131,072 B | 3.4 % |
| `[64,8192]`, Hs 32 | 2 | 16 | 1 | 36,864 | 18,432 | 32,768 | 88,064 B | 2.3 % |

The inventory is three buffers and each is irreducible: the output line must be resident (dependent axis),
`src` and `index` must be in L1 for a DM hart to read them, and neither can land in the slab. No budget solve
engages on any INPUTS shape here; `choose_blocking` only clamps beyond them.

## Data-movement budget

Boundary: DRAM. Split: width (face-columns) across clusters; source rows across the threads of a cluster.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `input` | 1 (read) | each cluster reads only its own face-columns' tiles, once, into the resident slab | none |
| `src` | 1 (read) | each source chunk is read once per block; blocks own disjoint columns | none |
| `index` | 1 (read) | as `src`; one piece of `block_width_faces·64` B per source row per block | none |
| output | 1 (write) | the slab is written once per block | none |

Totals per tier:

| Call | DRAM read | DRAM write | NoC cross-core | Transactions per cluster |
|------|-----------|------------|----------------|--------------------------|
| traced `[70,512]`, Hs 64 | 98,304 (`input`) + 65,536 (`src`) + 131,072 (`index`) = 294,912 B | 98,304 B | 0 | 6 + 4 face reads, 64 index pieces of 64 B, 6 face writes |
| model `[2766,2560]`, Hs 2752 | 14,254,080 + 14,090,240 + 28,180,480 = 56,524,800 B | 14,254,080 B | 0 | ≈ 2 tile-columns whole + split ends: ~350 `input` and ~350 `src` transfers, 2,752 index pieces of 320 B, ~350 writes |

> Cheapest-traffic split considered: width (face-columns) across clusters — the minimum: every tensor
> crosses DRAM once, no cross-core traffic. Implemented: the same split. It is also the cheapest; the
> only costlier split with a use (`banded_output_line`, `index` + `src` re-read once per band) is a
> `deferred` regime for lines that do not fit, which no INPUTS shape reaches on this target.
