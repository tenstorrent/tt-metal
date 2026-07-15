# CB→DFB Kernel Audit: `deepseek_moe_fast_reduce_nc`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_moe_fast_reduce_nc/`

**Scope:** program factory → kernels: `device/kernels/deepseek_moe_fast_reduce_nc_reader.cpp`, `device/kernels/deepseek_moe_fast_reduce_nc_reduce.cpp`, `device/kernels/deepseek_moe_fast_reduce_nc_writer.cpp`. Include closure: `kernel_lib/l1_helpers.hpp`.

## Overall verdict: GREEN

**Summary:** Despite the "deepseek_moe" name this is **not** a MOE-gate firmware-reconfig op — it is a plain NC reduction (reader → reduce compute → writer). Litmus scans find **zero** `get_local_cb_interface`/`cb_interface.` access, **zero** `reconfig_cbs_for_mask`, **zero** `get_cb_tiles_*_ptr`, **zero** `read_tile_value`/`get_tile_address`, **zero** `get_pointer_to_cb_data`, and **zero** `fifo_*` surgery or field reads. All CBs are canonical Class 1 linear FIFO. Mechanical `CircularBuffer` → `DataflowBuffer` rename only.

## Scope notes

- Does **not** match the OUT-OF-SCOPE MOE-gate patterns — audited normally as a standard reduction.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_compute_input_0`, `cb_compute_input_1` | 1 | `*_reader.cpp`, `*_reduce.cpp` | Portable | input tiles + reduce scalar, linear FIFO → `DataflowBuffer` | Portable | — |
| `cb_compute_output` | 1 | `*_reduce.cpp`, `*_writer.cpp` | Portable | pack → output, `get_read_ptr()`/`get_write_ptr()` as L1/NoC addr only | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
