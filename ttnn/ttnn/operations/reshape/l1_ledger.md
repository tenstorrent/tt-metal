# L1 ledger: reshape

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Design: `op_design.md` (same directory). Target: `quasar` on ttsim; worked values use 32 clusters and the budget below. The two regimes are separate `ProgramSpec`s, so their buffers never coexist. No buffer passes through DEST (no compute kernel), so every page holds the tensor's own dtype bytes and audit 2 (page format vs DEST width) has nothing to check: `fp32_dest_acc_en` does not exist for this op.

Block axes (Blocking Model): regime A `page`; regime B `out_tile_row`, `tile_col`.

## Buffers

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `dfb_pages` (regime A) | `PAGE_RING_DEPTH × block_pages` = `2 × clamp(floor(BLOCK_BYTES_TARGET / entry_bytes), 1, max_core_pages)`, entry `entry_bytes = round_up(page_bytes, l1_alignment)` | `2 × block_pages`: one block being filled by the reader and one being drained by the writer (implicit sync's two txn-id batches) | `{page: spans → block_pages (× depth 2); streams → core_pages}` | the tensor dtype, raw bytes (bfloat16 in PHASE0) | reader | writer | the whole program | the only buffer of regime A: nothing to share with. Capacity equals the live set; the factor 2 is the double-buffering mechanism reason |
| `dfb_input_window` (regime B) | `input_window_depth × n_in_cap × block_tile_cols`, entry `page_bytes` (one tile); `input_window_depth = 2 if max_blocks_per_core > 1 else 1` | `n_in(block) × wb` tiles of the block being regrouped (≤ `n_in_cap × block_tile_cols`), plus the next block's window when depth is 2 | `{out_tile_row: spans → n_in_cap (input tile-rows feeding block_out_tile_rows output tile-rows); tile_col: spans → block_tile_cols; streams → the core's blocks}` | the tensor dtype, raw bytes | reader | relayout | the whole program; one window per block | cannot share with `dfb_output_tiles`: both are live at once (the relayout reads the window while it writes output tiles). Cannot be transformed in place: an output tile mixes rows of several input tiles, the output tile count differs from the input's, and the writer would be a second consumer of this ring. Capacity exceeds the block's real `n_in × wb` only for ragged blocks: the window is pushed and popped at its nominal count so that it never wraps (mechanism reason: the contiguity of `scoped_read_lock`) |
| `dfb_output_tiles` (regime B) | `OUTPUT_RING_DEPTH × output_quantum` = `2 × min(block_tile_cols, max(1, floor(BLOCK_BYTES_TARGET / page_bytes)))`, entry `page_bytes` | one tile being assembled + the tiles the writer has not yet sent | `{out_tile_row: streams → 1 tile at a time; tile_col: streams → up to output_quantum tiles in flight}` | the tensor dtype, raw bytes | relayout | writer | the whole program | it is the destination itself (the regroup packs straight into it: no scratch tile exists). Cannot alias the window (concurrent lifetime). Capacity exceeds the 1-tile live set to keep the writer's implicit-sync batches in flight (double buffering, txn-id batches need an even count) |

## Symbols

| Symbol | Bound | Predicate / source that establishes it |
|--------|-------|----------------------------------------|
| `page_bytes` | regime A row-major: `W · element_size` ≤ `NOC_MAX_BURST_SIZE` = 65,536 B; tile: 2,048 B (bf16), 4,096 B (fp32 / int32 / uint32) | `input.buffer_page_size()`; validate refuses row-major pages above `NOC_MAX_BURST_SIZE` (mechanism cap) |
| `entry_bytes` | `page_bytes ≤ entry_bytes ≤ page_bytes + 15` | `round_up(page_bytes, ttnn.get_l1_alignment())` |
| `BLOCK_BYTES_TARGET` | 32,768 B (constant knob) | host constant |
| `max_core_pages` | `ceil(tensor_pages / engaged_cores)` | `ttnn.split_work_to_cores` |
| `block_pages` | `1 ≤ block_pages ≤ max(1, 32,768 / entry_bytes)` | definition (a clamp) |
| `n_in_cap` | `≤ ceil(S / 32) + ceil(S / H_in) + 1` input tile-rows, `S = block_out_tile_rows · min(32, H_out)` logical rows per block (the tile-row ends crossed plus the plane ends crossed); 34 for one output tile-row at `H_in = 1` | exact maximum over the program's blocks, computed on host; held to the ring cap by the fit predicate |
| `block_tile_cols`, `block_out_tile_rows` | `≤ core_tile_cols`, `≤ core_out_tile_rows` | the core's assignment; fit predicate below |
| `max_blocks_per_core` | `ceil(core_tile_cols / block_tile_cols) · ceil(core_out_tile_rows / block_out_tile_rows)` | definition |
| `DFB_RING_MAX_BYTES` | 1,048,560 B per ring | target page *Cluster model* |
| `L1_RESERVE_BYTES` | 1,048,576 B (knob): headroom for L1-resident interleaved input and output tensors sharing the cluster's L1 with the rings | host constant |
| `budget` | `ttnn.get_max_worker_l1_unreserved_size() − L1_RESERVE_BYTES` = 3,963,328 − 1,048,576 = 2,914,752 B on quasar (probe) | run-time query; never a literal |

**Fit predicate (regime B)**, evaluated on host with the queries:

```
window_bytes(Bh, Wb) = input_window_depth(Bh, Wb) × n_in_cap(Bh) × Wb × page_bytes
output_bytes(Wb)     = OUTPUT_RING_DEPTH × min(Wb, max(1, BLOCK_BYTES_TARGET // page_bytes)) × page_bytes
fits(Bh, Wb)         = window_bytes ≤ DFB_RING_MAX_BYTES
                       and output_bytes ≤ DFB_RING_MAX_BYTES
                       and window_bytes + output_bytes ≤ budget
```

Search order: `Bh = core_out_tile_rows` with `Wb` from `core_tile_cols` down to 1 (no traffic cost), then `Wb = 1` with `Bh` downwards (each step re-reads at most one more boundary input tile-row). If `fits(1, 1)` fails, the op refuses with `NotImplementedError` (`n_in_cap` at `Bh = 1` is ≤ 34 tiles ≤ 139,264 B at 4 KiB tiles and always fits, so this cannot happen on this target). Regime A needs no search: its footprint is bounded by a constant.

## Total per-core footprint

| Regime | Expression | Scales with | Worked (quasar, bf16) |
|--------|------------|-------------|-----------------------|
| `page_identity` | `2 × block_pages × entry_bytes` ≤ `2 × max(BLOCK_BYTES_TARGET, entry_bytes)` | `block_pages` (the bytes-in-flight knob); nothing else | `(1,1,2048,256)→(1,4,512,256)` TILE: 2 × 16 × 2,048 = 64 KiB. `(1,64,8400)` RM: 2 × 1 × 16,800 = 33,600 B. `(1,1,1600,144)` RM: 2 × 50 × 288 = 28,800 B |
| `tile_row_regroup` | `input_window_depth × n_in_cap × block_tile_cols × page_bytes + 2 × output_quantum × page_bytes` | the window term with `block_out_tile_rows` (through `n_in_cap`) and with `block_tile_cols`; the output term with `block_tile_cols` up to `BLOCK_BYTES_TARGET` | `(1,40,40,512)→(1,1,1600,512)`: 1 × 40 × 1 × 2,048 + 2 × 1 × 2,048 = 86,016 B. `(1,64,8400)→(1,4,16,8400)`: 1 × 2 × 9 × 2,048 + 2 × 9 × 2,048 = 73,728 B. `(1,64,400)→(1,4,16,400)`: 2,048 + 4,096 = 6,144 B |

Every PHASE0 footprint is far below the budget: neither regime needs a smaller block than its whole per-core assignment on any INPUTS entry.

## Data-movement budget

Crossings of each tensor's memory boundary (DRAM, or the L1 banks for an L1 tensor).

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| input (regime A) | 1 | each core reads its contiguous page range once; nothing is re-read | none |
| output (regime A) | 1 | each page written once from the ring | none |
| input (regime B) | `Σ_blocks n_in(block) × wb / (tensor_in_tile_rows × tensor_w_tiles)` ≤ `(tensor_in_tile_rows + row_groups − 1) / tensor_in_tile_rows` < 2 with one block per core; exactly 1 when `row_groups = 1` or row-group boundaries fall on input tile-row boundaries | the whole per-core assignment is one block, so the window is read once per core; columns never overlap; only an input tile-row straddling a row-group boundary is read by both neighbours | none (the multicast that would replace the re-read is rejected: same payload plus a rendezvous) |
| output (regime B) | 1 | each output tile is assembled once in L1 and written once; it holds `tensor_out_tile_rows × tensor_w_tiles` tiles, more or fewer than the input as the padding moves | none |

Totals per tier: regime A moves `tensor_pages × page_bytes` in and the same out, with zero cross-core traffic. Regime B moves ≤ `(tensor_in_tile_rows + row_groups − 1) × tensor_w_tiles × page_bytes` in and `tensor_out_tile_rows × tensor_w_tiles × page_bytes` out, plus one L1-to-L1 copy of each output tile's valid rows inside the core, with zero cross-core traffic. Worked: `(1,64,400)→(1,4,16,400)` reads 26 tiles (the minimum) and writes 52. `(1,1,50,128)→(1,5,10,128)` reads 12 tiles against a minimum of 8 (`row_groups = 2`) and writes 20.

> Cheapest-traffic split considered: a view with no program (0 bytes on every tier). It is rejected by the run's one-dispatch rule, so it is not deferred. Among copying splits: regime A as implemented is the cheapest (1 crossing in, 1 out). For regime B the cheapest is `tile_col` only, which saves `(row_groups − 1) × tensor_w_tiles × page_bytes` of input reads (4 tiles on `(1,1,50,128)→(1,5,10,128)`; 0 on `(1,64,400)→(1,4,16,400)` and `(1,40,40,512)→(1,1,1600,512)`). Implemented: `tile_col` first, then `out_tile_row` in `row_groups ≤ tensor_in_tile_rows` groups. Not taken as the default because it engages only `min(tensor_w_tiles, 32)` clusters (4 of 32 for W = 128), and the cap keeps the extra reads below one input tile-row per group. It stays reachable as the `row_groups = 1` setting of the same split code (the grid-synchronization / duplication perf lamp).

## Implementation notes (ttnn-implementer, phase 0)

- Built as inventoried: `dfb_pages` (A), `dfb_input_window` + `dfb_output_tiles` (B); no buffer added, merged or removed. Sizes are computed in `reshape_program_artifacts.py` from the knobs named above (`BLOCK_BYTES_TARGET`, `PAGE_RING_DEPTH`, `OUTPUT_RING_DEPTH`, `L1_RESERVE_BYTES`, `DFB_RING_MAX_BYTES`), each defined once.
- Window entry layout: entry `(tr − window_first) · block_tile_cols + (w − w0)` — the row stride is the nominal `block_tile_cols`, not the block's real `wb`, so a ragged column block uses the same layout; it still fits because capacity per window is `n_in_cap × block_tile_cols`.
- Regime A rounds `block_pages` up to a multiple of `lcm(reader_threads, writer_threads)` so the implicit-sync batches divide over the threads of a non-default plan (no-op at the Phase 0 plan (1, 1, 1)).
- Data-movement budget is as stated: `test_engaged_clusters[tile_row_regroup]` asserts input tiles read == input tiles for `(1,40,40,512)→(1,1,1600,512)` (1 crossing).
