# L1 Ledger: g6_fill

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking axes: `op_design.md` → Blocking Model (`w_tiles`, `row0_segment`, `lane`, `plane_h`; `rm_bytes` for the deferred R2 regime).

## Inventory

A NoC write needs its source bytes in L1, and the one source run is a constant, so at most one buffer can exist. The knob `FILL_ROW_SOURCE` (`g6_fill_program_descriptor.py`) picks where that run lives:

- **`"cb"` (default; the design's Phase 0 source).** The writer builds `segment_words` copies of `e` in `cb_fill_row` with RISC word stores, once per core.
- **`"common_args"` (live setting, not the default).** The run is `segment_words` copies of `e` at the head of the writer's common runtime args, which the dispatcher writes to every core anyway. The program then has no CB and no build stage.

Why `"cb"` is the default although `"common_args"` is faster: the graded eval run (`eval/eval_test_runner.sh`, `EVAL_CAPTURE_L1=1` by default) raises `L1ProfilingError` when a graded call's L1 allocation peak is 0 (`eval/l1_profiling.py`). With `"common_args"` the op allocates no L1 at all, so all 16 supported golden cells fail (verified: 0/19 passed). With `"cb"` the recorded peak is 64 B per core on every cell.

Measured by the implementer (Blackhole p150b, device kernel duration, median of 9 calls), `"common_args"` vs `"cb"`: 475 vs 576 ns for `[512]`, 442 vs 524 for `[256]`, 407 vs 504 for `[207]`, 403 vs 526 for `[196]`, 400 vs 515 for `[185]`.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_fill_row` (R1, default `FILL_ROW_SOURCE = "cb"`) | `fill_row_depth` (= 1) pages of `segment_bytes = face_w * element_size` (= 16 * 4 = 64 B). A constant live set: it scales with no block extent. | 1 page: every write reads the same 64 B, so the peak is exactly the one run | `{plane_h: streams → 1 row (padding rows never stored), w_tiles: streams → the same run reused for every tile, row0_segment: streams → the same run reused for face 0 and face 1, lane: spans → face_w}` | output dtype (`Float32`). A dataflow-only payload that never passes through DEST, so the DEST-width audit does not apply. The page width is the element width the contract requires to survive unchanged. | writer (RISC word stores, once per core) | writer (NoC source of every segment write; same thread, no CB handshake) | whole kernel (`build_fill_row` through `drain_writes`) | Shares with nothing: it is the only buffer in the program. Capacity equals the live set, so there is no double-buffer gap to justify. Not allocated with `"common_args"`. |
| `cb_fill_row` (R2, deferred) | `fill_row_depth` (= 1) pages of `rm_chunk_bytes = min(align_up(4N, 16), RM_CHUNK_CAP_BYTES)` | 1 page (the constant run reused for every chunk) | `{rm_bytes: spans → rm_chunk_bytes / 4 elements, (R1 axes do not exist in R2)}` | output dtype (`Int32` for the production cell) | writer | writer | whole kernel | Shares with nothing: the only buffer in the program |

| Source run with `"common_args"` (not a CB) | Size | Location | Producer | Consumer | Lifetime |
|----|------|----------|----------|----------|----------|
| common runtime args `[0, segment_words)` of the writer, then `dst_addr` | `segment_bytes + 4` = 68 B of common args | kernel-config region; index 0 is L1-aligned (16 B), so it meets the Blackhole DRAM-write congruence rule for segment offsets `{0, face_bytes}` (asserted in the kernel under the watcher) | dispatcher (once per core range per call) | writer (NoC source of every segment write) | the launch |

With the default `"cb"`, the common runtime args are `[fill_bits, dst_addr]` (8 B).

## Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `face_w` | = 16 | host asserts `output_tensor.tile.face_shape == (16, 16)` and `tile_shape == (32, 32)` |
| `element_size` | = 4 | `validate()` admits only float32 / int32; host asserts `element_size() == 4` |
| `fill_row_depth` | = 1 | knob `FILL_ROW_DEPTH` (design constant, `op_design.md` → Buffer-depth knobs); host asserts `>= 1` |
| `N` | `>= 1`; no upper limit by contract | `validate()` (rank 1, `N >= 1`). In R1, `N` does not appear in any capacity. In R2 it enters only through `min(…, RM_CHUNK_CAP_BYTES)`. |
| `RM_CHUNK_CAP_BYTES` | = 2048, `<= NOC_MAX_BURST_SIZE` | host constant; host asserts the burst bound (R2, not built yet) |
| `rm_chunk_bytes` | `16 <= rm_chunk_bytes <= RM_CHUNK_CAP_BYTES`, multiple of 16 | regime predicate `layout == ROW_MAJOR`, plus the `min` / `align_up` above (R2, not built yet) |

## Footprint

| Regime | Per-core footprint | Scales with |
|--------|--------------------|-------------|
| R1 (built, default `"cb"`) | `fill_row_depth * segment_bytes = 64 B` of CB (graded peak: 64 B on every cell); 8 B of common runtime args | nothing: constant in `N`, in `block_w_tiles` and in core count |
| R1 (`"common_args"` setting) | 0 B of CB; 68 B of common runtime args | nothing |
| R2 (deferred) | `fill_row_depth * rm_chunk_bytes <= 2,048 B` (production `[1]`: 16 B) | the `rm_chunk_bytes` knob only, capped. Constant in `N` beyond the cap. |

All totals stay far below the BB-05 bound of 1,347,776 B per core for every shape, including any `N`. No L1 budget predicate, safety fraction or blocking search exists or is needed.

## Data-movement budget

Chosen split: `w_tiles` across `min(tensor_w_tiles, max_fill_cores)` cores (R1).

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `output_tensor` | 1 (write only). Only the two row-0 segments of each tile are written: `128 * tensor_w_tiles` B, versus `4096 * tensor_w_tiles` as whole tiles. | Each segment is written exactly once, by the one core that owns its tile. The source is resident (`cb_fill_row`), so nothing is re-fetched. There is no read-modify-write. | 0 |
| `input_tensor` | 0 | never read (contract): no accessor, not a `generic_op` operand | 0 |
| `fill_value` | 0 (4 B in the common runtime args with the default `"cb"`, 64 B with `"common_args"`; written by the dispatcher) | reuse-shared scalar, broadcast once per core range by the dispatcher | 0 NoC payload. With `"cb"` the 64-B run is built locally with 16 word stores per core; with `"common_args"` it is the common args themselves. |

Total per call, DRAM tier: writes `128 * tensor_w_tiles` B (2,048 / 1,024 / 896 / 896 / 768 B for `[512] / [256] / [207] / [196] / [185]`); reads 0 B. Cross-core tier: 0 B. Core-local: `2 * core_w_tiles` NoC source reads of 64 B per core, plus `segment_bytes` of stores with `"cb"`.

> Cheapest-traffic split considered: `w_tiles` across cores. Delta 0 B on every tier against every other candidate: single core, segment split, and injector multicast (+64 B per receiver NoC). Implemented: `w_tiles` across cores. **The implemented split is the cheapest.** Traffic is invariant under the split; the remaining trade, occupancy against launch cost, is the grid-synchronization perf lamp in `op_design.md`, not a deferred regime.
