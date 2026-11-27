# L1 Ledger: g6_fill

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking axes: `op_design.md` → Blocking Model (`w_tiles`, `row0_segment`, `lane`, `plane_h`; `rm_bytes` for the deferred R2 regime).

## Inventory

The op has one buffer. A NoC write needs its source bytes in L1, and the one source run is a constant, so no other buffer can exist. The only alternative to this buffer is the runtime-argument region, which is a perf lamp in `op_design.md` and would reduce the inventory to zero.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_fill_row` (R1, built) | `fill_row_depth` (= 1) pages of `segment_bytes = face_w * element_size` (= 16 * 4 = 64 B). A literal-constant live set: it scales with no block extent. | 1 page: every write reads the same 64 B, so the peak is exactly the one run | `{plane_h: streams → 1 row (padding rows never stored), w_tiles: streams → the same run reused for every tile, row0_segment: streams → the same run reused for face 0 and face 1, lane: spans → face_w}` | output dtype (`Float32`). A dataflow-only payload that never passes through DEST, so the DEST-width audit does not apply. The page width is the element width the contract requires to survive unchanged. | writer (RISC word stores, once per core) | writer (NoC source of every segment write; same thread, no CB handshake) | whole kernel (`build_fill_row` through `drain_writes`) | Shares with nothing: it is the only buffer in the program. Capacity equals the live set, so there is no double-buffer gap to justify. |
| `cb_fill_row` (R2, deferred) | `fill_row_depth` (= 1) pages of `rm_chunk_bytes = min(align_up(4N, 16), RM_CHUNK_CAP_BYTES)` | 1 page (the constant run reused for every chunk) | `{rm_bytes: spans → rm_chunk_bytes / 4 elements, (R1 axes do not exist in R2)}` | output dtype (`Int32` for the production cell) | writer | writer | whole kernel | Shares with nothing: the only buffer in the program |

## Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `face_w` | = 16 | host asserts `output_tensor.tile.face_shape == (16, 16)` and `tile_shape == (32, 32)` |
| `element_size` | = 4 | `validate()` admits only float32 / int32 |
| `fill_row_depth` | = 1 | design constant (`op_design.md` → Buffer-depth knobs) |
| `N` | `>= 1`; no upper limit by contract | `validate()` (rank 1, `N >= 1`). In R1, `N` does not appear in any capacity. In R2 it enters only through `min(…, RM_CHUNK_CAP_BYTES)`. |
| `RM_CHUNK_CAP_BYTES` | = 2048, `<= NOC_MAX_BURST_SIZE` | host constant; host asserts the burst bound |
| `rm_chunk_bytes` | `16 <= rm_chunk_bytes <= RM_CHUNK_CAP_BYTES`, multiple of 16 | regime predicate `layout == ROW_MAJOR`, plus the `min` / `align_up` above |

## Footprint

| Regime | Per-core footprint | Scales with |
|--------|--------------------|-------------|
| R1 (built) | `fill_row_depth * segment_bytes = 64 B` | nothing: constant in `N`, in `block_w_tiles` and in core count |
| R2 (deferred) | `fill_row_depth * rm_chunk_bytes <= 2,048 B` (production `[1]`: 16 B) | the `rm_chunk_bytes` knob only, capped. Constant in `N` beyond the cap. |

Both totals stay far below the BB-05 bound of 1,347,776 B per core for every shape, including any `N`. No L1 budget predicate, safety fraction or blocking search exists or is needed.

## Data-movement budget

Chosen split: `w_tiles` across `min(tensor_w_tiles, max_fill_cores)` cores (R1).

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `output_tensor` | 1 (write only). Only the two row-0 segments of each tile are written: `128 * tensor_w_tiles` B, versus `4096 * tensor_w_tiles` as whole tiles. | Each segment is written exactly once, by the one core that owns its tile. The source is resident (`cb_fill_row`), so nothing is re-fetched. There is no read-modify-write. | 0 |
| `input_tensor` | 0 | never read (contract): no accessor, not a `generic_op` operand | 0 |
| `fill_value` | 0 (4 B in the common runtime args, written by the dispatcher) | reuse-shared scalar, broadcast once per core range by the dispatcher | 0 NoC payload. The 64-B run is built locally with 16 word stores per core. |

Total per call, DRAM tier: writes `128 * tensor_w_tiles` B (2,048 / 1,024 / 896 / 896 / 768 B for `[512] / [256] / [207] / [196] / [185]`); reads 0 B. Cross-core tier: 0 B. Core-local: `segment_bytes` of stores plus `2 * core_w_tiles` NoC source reads of 64 B per core.

> Cheapest-traffic split considered: `w_tiles` across cores. Delta 0 B on every tier against every other candidate: single core, segment split, and injector multicast (+64 B per receiver NoC). Implemented: `w_tiles` across cores. **The implemented split is the cheapest.** Traffic is invariant under the split; the remaining trade, occupancy against launch cost, is the grid-synchronization perf lamp in `op_design.md`, not a deferred regime.
