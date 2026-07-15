# CB→DFB Kernel Audit: `deepseek_grouped_gate`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_grouped_gate/`

**Scope:** `deepseek_grouped_gate_program_factory` → kernels: `device/kernels/dataflow/reader_deepseek_grouped_gate.cpp`, `device/kernels/compute/deepseek_grouped_gate.cpp`, `device/kernels/dataflow/writer_deepseek_grouped_gate.cpp`. Include closure: cross-op donor `ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/topk_common_funcs.hpp` (topk sort helper), plus `kernel_lib/reduce_helpers_compute.hpp`, `kernel_lib/reduce_helpers_dataflow.hpp`.

## Overall verdict: GREEN

**Summary:** Despite the "deepseek" name this is **not** a MOE-gate firmware-reconfig op — it is a plain (if elaborate) grouped-gating / top-k reduction pipeline. Litmus scans over the full kernel closure (including the topk donor header) find **zero** `get_local_cb_interface`/`cb_interface.` field access, **zero** `reconfig_cbs_for_mask`, **zero** `get_cb_tiles_*_ptr`, **zero** `read_tile_value`/`get_tile_address`, **zero** `get_pointer_to_cb_data`, and **zero** `fifo_*` pointer surgery or field reads. All CBs are canonical Class 1 linear FIFO (reader→compute→writer). The writer's `get_write_ptr() + tile/face byte offset` calls are bare L1 addressing into the CB's own reserved region (WEIRD-OK / already portable). Mechanical `CircularBuffer` → `DataflowBuffer` rename only.

## Scope notes

- Does **not** match the OUT-OF-SCOPE MOE-gate patterns (no `reconfig_cbs_for_mask`, no `get_cb_tiles_acked_ptr`/`get_cb_tiles_received_ptr`, no `LocalCBInterface` rewrite). Audited normally.
- Donor `topk_common_funcs.hpp` is clean (sort/compare helpers only, no CB field access).

## CB portability

CBs collapsed by logical role (`_obj`/`_id` aliases are the same buffer's DataflowBuffer handle / CT-arg id).

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| input scores / bias (`cb_in_scores`, `cb_in_bias`, scalar/epsilon/route_scale CBs) | 1 | `reader_*.cpp`, `deepseek_grouped_gate.cpp` | Portable | linear FIFO inputs → `DataflowBuffer` | Portable | — |
| gate intermediates (`cb_sigmoid*`, `cb_biased*`, `cb_group_summed*`, `cb_reduce_intermediate`, `cb_reciprocal*`, `cb_normalized*`) | 1 | `deepseek_grouped_gate.cpp` | Portable | canonical `reserve/push` ↔ `wait/pop` | Portable | — |
| topk/sort intermediates (`cb_sorted_*`, `cb_winning_*`, `cb_top_experts*`, `cb_group_index*`, `cb_expert_index*`) | 1 | `deepseek_grouped_gate.cpp`, `topk_common_funcs.hpp` | Portable | linear FIFO, donor sort helper | Portable | — |
| outputs (`cb_out_weights`, `cb_out_indices`, `cb_top_experts`) | 1 | `writer_*.cpp` | Portable (workaround) | **undesirable but OK hack:** `get_write_ptr() + i*tile_size + face_line` byte offsets for face-layout NoC write into reserved region (bare L1 addressing, not `fifo_*` surgery) | Portable (workaround) | same |

## GATE hits (must be empty to merge)

- (none)

## Blocked on runtime (2xx rollup)

- (none)
