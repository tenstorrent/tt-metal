# CB→DFB Kernel Audit: `deepseek_prefill/moe_hash_gate`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_hash_gate/`

**Scope:** `device/kernels/dataflow/{reader_moe_hash_gate.cpp, writer_moe_hash_gate.cpp}`, `device/kernels/compute/moe_hash_gate.cpp`

> **Note:** This op (`moe_hash_gate`) is an explicitly in-scope target of this audit and is distinct from the OUT-OF-SCOPE `deepseek_moe_gate` / `generalized_moe_gate` firmware-reconfig ops. It does **not** use `reconfig_cbs_for_mask` / `LocalCBInterface` rewrites.

## Overall verdict: GREEN

**Summary:** Zero litmus hits across all three kernels — no `get_local_cb_interface(...)` field access, no `read_tile_value`/`get_tile_address`, no `get_pointer_to_cb_data`, no `get_cb_tiles_*_ptr`, no ptr surgery, no `fifo_*` field reads. All CBs (score inputs, sigmoid/normalize/reduce intermediates, scalar broadcast CBs, output weights/indices, gather scratch) are canonical linear FIFO stages or bare-pointer L1 addressing. Mechanical `CircularBuffer` → `DataflowBuffer` rename; safe to port on both arches.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in_scores`, `cb_input_ids`, `cb_tid2eid_row` | 1 | `reader_*.cpp`, `compute_*.cpp` | Portable | linear FIFO inputs → `DataflowBuffer` | Portable | — |
| `cb_sigmoid_scores`, `cb_normalized_scores`, `cb_reduce_intermediate`, `cb_reciprocal_sums`, `cb_gathered_sigmoid` | 1 | `compute_*.cpp`, `writer_*.cpp` | Portable | intra-op linear FIFO intermediates | Portable | — |
| `cb_reduce_ones_scalar`, `cb_epsilon_scalar`, `cb_route_scale_scalar` | 1 | `compute_*.cpp`, `writer_*.cpp` | Portable | scalar broadcast CBs, linear FIFO | Portable | — |
| `cb_out_weights`, `cb_out_indices` | 1 | `compute_*.cpp`, `reader_*.cpp`, `writer_*.cpp` | Portable | pack → output, linear FIFO / bare L1 addr | Portable | — |
| `cb_padding_config` | 6 | `writer_moe_hash_gate.cpp` | Portable | config/scratch region, bare pointer only — autoportable | Portable | same |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
