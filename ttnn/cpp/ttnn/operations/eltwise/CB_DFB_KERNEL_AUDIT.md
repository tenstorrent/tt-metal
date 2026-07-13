# CB→DFB Kernel Audit: `eltwise` (family-wide sweep)

**Date:** 2026-07-13
**Op root:** `ttnn/cpp/ttnn/operations/eltwise/`

**Scope:** All 84 device kernel files under `eltwise/**/device/{kernels,kernels_ng}/` across the sub-families that own kernels — `binary` (2 compute + 1 reader), `binary_ng` (`kernels/` + `kernels_ng/`, 55 files), `ternary` (11 compute + 13 dataflow), `unary` (8 compute + 8 dataflow), and `unary_backward` (`gelu_bw`, `tanh_bw`). This is a family-wide litmus sweep (like the `data_movement` audit), not a per-factory kernel-closure audit — it runs the [Step 4 classification scans](../../../../../CB_DFB_Check.md#step-4--classification-scans-on-scan_files-only) across every kernel in the tree. The composite sub-families **`binary_backward`, `complex`, `complex_binary`, `complex_unary`, `complex_unary_backward`, `quantization`, `ternary_backward`, `unary_ng`** own **no device kernels** (they decompose into other eltwise/ttnn ops at the host level) and are therefore out of scope for a kernel audit.

## Overall verdict: YELLOW (four identical mechanical fixes from GREEN)

**Summary:** Eltwise is **overwhelmingly safe to port**. Across all 84 kernel files the litmus scans find **exactly four** `LocalCBInterface` field accesses — all the **same** `get_local_cb_interface(...).fifo_page_size` **read**, in the four `unary/` non-sharded reader/writer kernels. Every one is a **mechanical NEEDS-FIX** (getter exists: `get_entry_size()`), and each kernel already constructs a `DataflowBuffer` handle for the same CB, so the fix is a one-line swap that clears the GATE. There are **zero** `read_tile_value`/`get_tile_address`, **zero** `get_pointer_to_cb_data`, **zero** `get_cb_tiles_*_ptr`, and **zero** `fifo_wr_ptr`/`fifo_rd_ptr`/`push_back_hold`/`llk_push_pages` pointer-surgery hits. Every other CB is canonical Class 1 linear FIFO; the remaining `get_read_ptr()`/`get_write_ptr()` hits are bare L1 addresses (fill-tile macros, NoC read/write addresses) — already portable, do not churn.

## Scan results (whole tree)

| Litmus scan | Verdict weight | Hits |
|-------------|----------------|------|
| `get_local_cb_interface(...).<field>` | **GATE** | **4** (all `fifo_page_size` reads, mechanical) |
| `get_cb_tiles_acked_ptr` / `get_cb_tiles_received_ptr` | silent-wrong | 0 |
| `read_tile_value` / `get_tile_address` | QUASAR-BLOCKED | 0 |
| `get_pointer_to_cb_data` | NEEDS-FIX → LTA | 0 |
| `fifo_wr_ptr` / `fifo_rd_ptr` / `push_back_hold` / `llk_push_pages` | ptr surgery (Class 2–5) | 0 |
| `fifo_page_size` / `fifo_num_pages` / `fifo_size` / `fifo_limit` | field read | 4 (same hits as GATE) |
| `get_read_ptr()` / `get_write_ptr()` | portable ptr use | ~20 files (WEIRD-OK / Portable — do not churn) |

## CB portability

Because the tree is clean, only the four offending kernels need per-CB rows. Every other CB across `binary`, `binary_ng`, `ternary`, `unary`, and `unary_backward` is a canonical Class 1 linear FIFO (`reserve_back`/`push_back` on the producer, `wait_front`/`pop_front` on the consumer) and/or bare-pointer L1 addressing → **Portable** (mechanical `CircularBuffer` → `DataflowBuffer` rename) on both 1xx and 2xx.

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_id_in0` | 1 | `unary/.../reader_unary_interleaved_start_id.cpp` | Portable | **NEEDS-FIX:** `get_local_cb_interface(cb_id_in0).fifo_page_size` (line 20) → `dfb.get_entry_size()` (handle already built line 28). Canonical `reserve_back`→`async_read`→`push_back`. | Portable | same |
| `cb_id_src` | 1 | `unary/.../reader_unary.cpp` | Portable | **NEEDS-FIX:** `get_local_cb_interface(cb_id_src).fifo_page_size` (line 57) → `dfb_src.get_entry_size()` (handle line 18). Non-RM-interleaved branch only; canonical linear FIFO. | Portable | same |
| `cb_id_out` | 1 | `unary/.../writer_unary_interleaved_start_id.cpp` | Portable | **NEEDS-FIX:** `get_local_cb_interface(cb_id_out).fifo_page_size` (line 19) → `dfb.get_entry_size()` (handle line 22). Canonical `wait_front`→`async_write`→`pop_front`. | Portable | same |
| `cb_id_dst` | 1 | `unary/.../writer_unary.cpp` | Portable | **NEEDS-FIX:** `get_local_cb_interface(cb_id_dst).fifo_page_size` (line 60) → `dfb_dst.get_entry_size()` (handle line 18). Non-RM-interleaved branch only; canonical linear FIFO. | Portable | same |
| all binary CBs (`cb_id_in0/1`, `cb_id_out`) | 1 | `binary/device/kernels/**` | Portable | canonical linear FIFO → mechanical `DataflowBuffer` rename | Portable | — |
| all binary_ng CBs (`cb_src`, `cb_src_b`, `cb_out`, scalar CBs) | 1 | `binary_ng/device/{kernels,kernels_ng}/**` | Portable | canonical FIFO; `get_write_ptr()` only as fill-tile L1 dest / `get_read_ptr()` as NoC source — bare L1 address, no offset surgery | Portable | — |
| all ternary CBs (`cb_pred`, `cb_true`/`cb_tensor`, `cb_false`, `cb_a/b/c`, `cb_out`) | 1 | `ternary/device/kernels/**` | Portable | canonical FIFO; `get_write_ptr()` only as fill-tile L1 dest — bare L1 address | Portable | — |
| all unary compute + sharded/col/wh reader-writer CBs | 1 | `unary/device/kernels/**` (compute + other dataflow) | Portable | canonical linear FIFO + bare-pointer L1 addressing | Portable | — |
| all unary_backward CBs | 1 | `unary_backward/{gelu_bw,tanh_bw}/device/kernels/compute/**` | Portable | canonical compute FIFO consumer/producer | Portable | — |

## GATE hits (must be empty to merge)

- `unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp:20` — `get_local_cb_interface(cb_id_in0).fifo_page_size` **read** — **mechanical**: → `dfb.get_entry_size()` (getter exists today, `DataflowBuffer dfb(cb_id_in0)` already constructed on line 28).
- `unary/device/kernels/dataflow/reader_unary.cpp:57` — `get_local_cb_interface(cb_id_src).fifo_page_size` **read** — **mechanical**: → `dfb_src.get_entry_size()` (handle on line 18).
- `unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp:19` — `get_local_cb_interface(cb_id_out).fifo_page_size` **read** — **mechanical**: → `dfb.get_entry_size()` (handle on line 22).
- `unary/device/kernels/dataflow/writer_unary.cpp:60` — `get_local_cb_interface(cb_id_dst).fifo_page_size` **read** — **mechanical**: → `dfb_dst.get_entry_size()` (handle on line 18).

## Blocked on runtime (2xx rollup)

- (none) — no `read_tile_value` / `get_tile_address` / `get_pointer_to_cb_data` / missing-getter dependencies anywhere in the tree.

## Recommended path

1. **Four-line fix → GREEN:** swap `get_local_cb_interface(<cb>).fifo_page_size` for `<dfb>.get_entry_size()` in the four `unary/` reader/writer kernels. Each already holds the matching `DataflowBuffer` handle, so no new construction is needed. This clears every GATE in the entire `eltwise` family.
2. **Everything else:** port freely. All `binary`, `binary_ng` (incl. `kernels_ng`), `ternary`, `unary` compute/other-dataflow, and `unary_backward` kernels are Class 1 linear FIFO or bare-pointer L1 addressing — mechanical `CircularBuffer` → `DataflowBuffer` with no field surgery, no runtime API dependency, and no LTA prerequisite.
3. **Composite sub-families** (`binary_backward`, `complex*`, `quantization`, `ternary_backward`, `unary_ng`) own no device kernels — nothing to port at the kernel level; they inherit portability from the underlying eltwise/ttnn ops they compose.
