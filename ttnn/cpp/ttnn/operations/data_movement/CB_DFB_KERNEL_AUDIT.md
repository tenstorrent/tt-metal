# CB→DFB Kernel Audit: `data_movement` (family-wide sweep)

**Date:** 2026-07-13
**Op root:** `ttnn/cpp/ttnn/operations/data_movement/`

**Scope:** All 194 device kernel files under `data_movement/*/device/kernels/` (37 ops: bcast, chunk, clone, common, concat, copy, expand, fill_pad, fill_rm, fold, gather, indexed_fill, moe_expert_token_remap, moe_routing_remap, move, narrow, non_zero_indices, pad, permute, repeat, repeat_interleave, reshape_on_device, reshape_view, roll, scatter, sharded, sharded_partial, slice, sort, split, squeeze, stack, tilize, tilize_with_val_padding, transpose, unsqueeze, untilize, untilize_with_unpadding, view). This is a family-wide litmus sweep, not a per-factory kernel-closure audit — it runs the [Step 4 classification scans](../../../../../CB_DFB_Check.md#step-4--classification-scans-on-scan_files-only) across every kernel in the tree.

## Overall verdict: YELLOW (one mechanical fix from GREEN)

**Summary:** Data movement is **overwhelmingly safe to port**. Across 194 kernel files, the litmus scans find **exactly one** illegal `LocalCBInterface` field access — a single `fifo_page_size` **read** in `reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp`. That is a **mechanical NEEDS-FIX** (getter exists: `get_entry_size()`), so it clears the GATE with a trivial one-line swap. There are **zero** `read_tile_value`/`get_tile_address`, **zero** `get_pointer_to_cb_data`, **zero** `get_cb_tiles_*_ptr`, and **zero** `fifo_wr_ptr`/`fifo_rd_ptr`/`push_back_hold`/`llk_push_pages` pointer-surgery hits. The remaining ~110 files use bare `get_read_ptr()`/`get_write_ptr()` only as L1/NoC byte addresses — already portable, do not churn.

## Scan results (whole tree)

| Litmus scan | Verdict weight | Hits |
|-------------|----------------|------|
| `get_local_cb_interface(...).<field>` | **GATE** | **1** (sharded reader, mechanical) |
| `get_cb_tiles_acked_ptr` / `get_cb_tiles_received_ptr` | silent-wrong | 0 |
| `read_tile_value` / `get_tile_address` | QUASAR-BLOCKED | 0 |
| `get_pointer_to_cb_data` | NEEDS-FIX → LTA | 0 |
| `fifo_wr_ptr` / `fifo_rd_ptr` / `push_back_hold` / `llk_push_pages` | ptr surgery (Class 2–5) | 0 |
| `fifo_page_size` / `fifo_num_pages` / `fifo_size` / `fifo_limit` | field read | 1 (same hit as GATE) |
| `get_read_ptr()` / `get_write_ptr()` | portable ptr use | ~110 files (WEIRD-OK / Portable — do not churn) |

## CB portability

Because the tree is clean, only the single offender needs a per-CB row. Every other op is canonical Class 1 linear FIFO and/or bare-pointer L1 addressing → **Portable** (mechanical `CircularBuffer` → `DataflowBuffer` rename) on both 1xx and 2xx.

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in1` (scratch) | 2/6 | `sharded/.../reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` | Portable | **NEEDS-FIX:** `get_local_cb_interface(cb_id_in1).fifo_page_size` (line 55) → `get_entry_size()` (getter exists). Used as private scratch region (`reserve_back(num_trids)` + per-slot `get_write_ptr()` offsets for the src→scratch→dest trid pipeline). Ideal end-state is `ScratchpadSpec`, but the getter swap alone clears the GATE and is safe to port. | Portable | same |
| all other CBs (36 ops) | 1 | `data_movement/**/kernels/**` | Portable | canonical linear FIFO + bare `get_read_ptr()`/`get_write_ptr()` L1/NoC addresses → mechanical `DataflowBuffer` rename | Portable | — |

## GATE hits (must be empty to merge)

- `sharded/device/kernels/dataflow/reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp:55` — `get_local_cb_interface(cb_id_in1).fifo_page_size` **read** — **mechanical**: → `get_entry_size()` (getter exists today, no runtime dependency). Clears the GATE with a one-line swap.

## Blocked on runtime (2xx rollup)

- (none) — no `read_tile_value` / `get_tile_address` / missing-getter dependencies anywhere in the tree.

## Recommended path

1. **One-line fix → GREEN:** swap `get_local_cb_interface(cb_id_in1).fifo_page_size` for `cb_in1.get_entry_size()` in the sharded reader. This clears the only GATE in the entire `data_movement` family.
2. **Everything else:** port freely. All other data_movement kernels are Class 1 linear FIFO or bare-pointer L1 addressing — mechanical `CircularBuffer` → `DataflowBuffer` with no field surgery, no runtime API dependency, and no LTA prerequisite.
3. **Optional hardening (not port-gating):** the sharded scratch `cb_in1` is really a private scratch region; a later refactor to `ScratchpadSpec` + semaphores would be cleaner than a scratch CB, but is not required for the port.
