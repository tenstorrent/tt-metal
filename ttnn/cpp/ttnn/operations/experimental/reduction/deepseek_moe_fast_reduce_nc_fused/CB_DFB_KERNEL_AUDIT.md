# CB→DFB Kernel Audit: `deepseek_moe_fast_reduce_nc_fused`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_moe_fast_reduce_nc_fused/`

**Scope:** program factory → kernels: `device/kernels/deepseek_moe_fast_reduce_nc_fused_reader.cpp`, `device/kernels/deepseek_moe_fast_reduce_nc_fused_compute.cpp`. Include closure: `api/debug/dprint_pages.h` (debug only).

## Overall verdict: GREEN

**Summary:** Despite the "deepseek_moe" name this is **not** a MOE-gate firmware-reconfig op — it is a fused NC reduction that reads activations plus expert-index/mapping/score metadata and reduces (reader + compute). Litmus scans find **zero** `get_local_cb_interface`/`cb_interface.` access, **zero** `reconfig_cbs_for_mask`, **zero** `get_cb_tiles_*_ptr`, **zero** `read_tile_value`/`get_tile_address`, **zero** `get_pointer_to_cb_data`, and **zero** `fifo_*` surgery or field reads. All CBs are canonical Class 1 linear FIFO. Mechanical `CircularBuffer` → `DataflowBuffer` rename only.

## Scope notes

- Does **not** match the OUT-OF-SCOPE MOE-gate patterns — audited normally.

## CB portability

CBs collapsed by role (`_id` aliases are the CT-arg id of the same buffer).

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in_act`, `cb_in0`, `cb_in1` | 1 | `*_reader.cpp`, `*_compute.cpp` | Portable | activation inputs, linear FIFO → `DataflowBuffer` | Portable | — |
| `cb_scores`, `cb_scores_rm`, `cb_expert_indices`, `cb_expert_mapping` | 1 | `*_reader.cpp`, `*_compute.cpp` | Portable | routing metadata inputs, linear FIFO | Portable | — |
| `cb_out` | 1 | `*_compute.cpp` | Portable | pack → output | Portable | — |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
