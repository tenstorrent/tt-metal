# CB→DFB Kernel Audit: `deepseek_prefill/moe_grouped_topk`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_grouped_topk/`

**Scope:** `device/kernels/dataflow/{reader_moe_grouped_topk.cpp, writer_moe_grouped_topk.cpp, moe_gate_common_dataflow.hpp}`, `device/kernels/compute/{moe_grouped_topk.cpp, moe_gate_common_compute.hpp}`

> **Note:** This op (`moe_grouped_topk`) is an explicitly in-scope target of this audit and is distinct from the OUT-OF-SCOPE `deepseek_moe_gate` / `generalized_moe_gate` firmware-reconfig ops. It does **not** use `reconfig_cbs_for_mask` / `LocalCBInterface` rewrites.

## Overall verdict: GREEN

**Summary:** Zero litmus hits across all kernels and shared headers — no `get_local_cb_interface(...)` field access, no `read_tile_value`/`get_tile_address`, no `get_pointer_to_cb_data`, no `get_cb_tiles_*_ptr`, no ptr surgery, no `fifo_*` field reads. The op has a large intra-compute CB pipeline (score biasing, group sort/select, sigmoid/normalize/reduce, template index CBs, output weights/indices) plus scalar broadcast CBs — all canonical linear FIFO stages or bare-pointer L1 addressing. Mechanical `CircularBuffer` → `DataflowBuffer` rename; safe to port on both arches.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in_scores`, `cb_in_bias` | 1 | `reader_*.cpp`, `compute_*.cpp` | Portable | linear FIFO inputs → `DataflowBuffer` | Portable | — |
| `cb_sigmoid_scores`, `cb_biased_scores`, `cb_normalized_scores`, `cb_reduce_intermediate`, `cb_reciprocal_sums`, `cb_gathered_sigmoid`, `cb_group_summed_scores` | 1 | `compute_moe_grouped_topk.cpp` | Portable | intra-compute linear FIFO intermediates | Portable | — |
| `cb_sorted_group_scores`, `cb_sorted_group_order`, `cb_sorted_expert_indices_temp`, `cb_top_experts_per_group`, `cb_winning_group_scores`, `cb_winning_group_indices`, `cb_final_indices_transposed` | 1 | `compute_*.cpp`, `writer_*.cpp` | Portable | group sort/select linear FIFO stages | Portable | — |
| `cb_expert_index_template`, `cb_group_index_template` | 1 | `compute_*.cpp`, `writer_*.cpp` | Portable | index template CBs, linear FIFO / bare L1 addr | Portable | — |
| `cb_reduce_ones_scalar`, `cb_epsilon_scalar`, `cb_route_scale_scalar` | 1 | `compute_*.cpp`, `writer_*.cpp` | Portable | scalar broadcast CBs, linear FIFO | Portable | — |
| `cb_out_weights`, `cb_out_indices` | 1 | `compute_*.cpp`, `writer_*.cpp` | Portable | pack → output, linear FIFO / bare L1 addr | Portable | — |
| `cb_padding_config` | 6 | `writer_moe_grouped_topk.cpp` | Portable | config/scratch region, bare pointer only — autoportable | Portable | same |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
