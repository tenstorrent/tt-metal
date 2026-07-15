# CB→DFB Kernel Audit: `all_gather_concat_heads_fused`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_concat_heads_fused/`

**Scope:** All in-scope device kernels under `device/kernels/`: `llama_all_gather_concat_reader.cpp`, `llama_all_gather_concat_writer.cpp`, `llama_concat_reader.cpp`, `tilize_compute.cpp`, `tilize_writer.cpp`.

## Overall verdict: GREEN

**Summary:** Fused all-gather + concat-heads (+ optional tilize). All Step-4 litmus scans return **zero** hits — no GATE, no silent-wrong, no `read_tile_value`/`get_tile_address`, no `get_pointer_to_cb_data`, no ptr surgery, no `fifo_*` field reads. CBs are canonical linear-FIFO input/output/q_out with a reserved fabric packet-header scratch CB; the concat writer uses bare `get_write_ptr()` + byte offsets for head placement (portable L1/NoC addressing, not field access). Mechanical rename.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in` / `cb_out` / `cb_q_out` (`cb_id_out`, `cb_id_q_out`) | 1 | `llama_all_gather_concat_reader.cpp`, `llama_all_gather_concat_writer.cpp`, `llama_concat_reader.cpp`, `tilize_writer.cpp` | Portable | canonical FIFO; concat via `get_write_ptr()` + byte offset (`cb_write_ptr_base`) = L1/NoC addr only | Portable | — |
| `cb_id` (tilize compute in/out) | 1 | `tilize_compute.cpp` | Portable | linear FIFO tilize stream | Portable | — |
| `reserved_packet_header_cb_id` / `cb_packet_header` | 1/6 | reader/writer kernels | Portable | fabric packet-header scratch; `reserve_back` + `get_write_ptr()` | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)

## Recommended path

Port freely — mechanical `CircularBuffer` → `DataflowBuffer` rename. No field surgery, no runtime API dependency, no LTA prerequisite.
