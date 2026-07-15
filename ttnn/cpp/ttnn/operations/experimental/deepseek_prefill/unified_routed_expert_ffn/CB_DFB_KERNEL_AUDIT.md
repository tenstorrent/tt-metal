# CB→DFB Kernel Audit: `unified_routed_expert_ffn` [fused_swiglu]

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn/`

**Scope:** In-scope device kernels under `device/kernels/` → `compute/fused_swiglu.cpp`, `dataflow/unified_routed_expert_ffn_reader.cpp`, `dataflow/unified_routed_expert_ffn_writer.cpp`.

## Overall verdict: YELLOW

**Summary:** The FFN matmul/SwiGLU pipeline CBs (`cb_in0_x`, `cb_in1_gate/up/down`, `cb_partials_*`, `cb_gate/up_intermed`, `cb_activated`, `cb_out`, `cb_in0_down_full`) are canonical Class 1 linear FIFOs → **Portable**. The offenders are the two per-expert **scratch/mailbox CBs** (`cb_counts_scratch`, `cb_idx_scratch`): the reader (BRISC) pushes a per-expert counts/idx tile, and `fused_swiglu.cpp:609-610` reads `get_local_cb_interface(cb).fifo_rd_ptr << 4` on the **UNPACK** thread to recover the L1 byte address, dereferences it as a scalar, and broadcasts the value to MATH/PACK via the inter-thread mailbox. This is a GATE field-read (Class 2 sync-free borrowed mailbox read). **1xx:** mechanical NEEDS-FIX — swap the `fifo_rd_ptr << 4` mailbox block for `cb.get_read_ptr()` → **Portable**. **2xx:** the scalar read needs `read_tile_value` on the DFB (in progress, Runtime team); LTA is insufficient because MATH/PACK cannot access `get_local_cb_interface` symbols and the value must be posted through the tile-read API → **Blocked (runtime)**. This is the recipe's fused_swiglu NEEDS-FIX case.

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_in0_x`, `cb_in1_gate`, `cb_in1_up`, `cb_in1_down` | 1 | `unified_routed_expert_ffn_reader.cpp`, `fused_swiglu.cpp` | Portable | matmul inputs, canonical `reserve/push`·`wait/pop` linear FIFO | Portable | — |
| `cb_in0_down_full` | 1 | `unified_routed_expert_ffn_reader.cpp`, `fused_swiglu.cpp` | Portable | down-proj activation input, linear FIFO | Portable | — |
| `cb_partials_gu`, `cb_partials_up`, `cb_partials_d` | 1 | `fused_swiglu.cpp` | Portable | matmul partials via canonical `pack_tile`/pop·reserve — no rd/wr ptr surgery | Portable | — |
| `cb_gate_intermed`, `cb_up_intermed`, `cb_activated` | 1 | `fused_swiglu.cpp` | Portable | SwiGLU intermediates, linear FIFO | Portable | — |
| `cb_out` | 1 | `fused_swiglu.cpp`, `unified_routed_expert_ffn_writer.cpp` | Portable | pack → output, `get_write_ptr()`/`get_read_ptr()` as L1/NoC addr only | Portable | — |
| `cb_start_scratch` | 2 | `unified_routed_expert_ffn_reader.cpp`, `unified_routed_expert_ffn_writer.cpp` | Portable | per-expert start offset scratch; canonical `reserve/push`·`wait/pop`, `get_read_ptr()` as L1 addr | Portable | — |
| **`cb_counts_scratch`** | **2** | **`fused_swiglu.cpp`** (reader/writer produce) | **Portable** | **NEEDS-FIX:** `fused_swiglu.cpp:609` `get_local_cb_interface(cb_counts_scratch).fifo_rd_ptr << 4` → `cb_counts_scratch.get_read_ptr()` (mailbox scalar source). Clears GATE. | **Blocked (runtime)** | scalar read on UNPACK broadcast to MATH/PACK — needs `read_tile_value` on DFB (in progress); LTA insufficient (MATH/PACK cannot access `get_local_cb_interface`, value must post via tile-read API) |
| **`cb_idx_scratch`** | **2** | **`fused_swiglu.cpp`** (reader produces) | **Portable** | **NEEDS-FIX:** `fused_swiglu.cpp:610` `get_local_cb_interface(cb_idx_scratch).fifo_rd_ptr << 4` → `cb_idx_scratch.get_read_ptr()`. Clears GATE. | **Blocked (runtime)** | same — needs `read_tile_value` on DFB (in progress) |

## GATE hits (must be empty to merge)

- `compute/fused_swiglu.cpp:609` — `get_local_cb_interface(cb_counts_scratch).fifo_rd_ptr` **read** (`<< 4` → L1 mailbox address) — **1xx mechanical:** → `cb_counts_scratch.get_read_ptr()`. **2xx:** requires `read_tile_value` on DFB (in progress).
- `compute/fused_swiglu.cpp:610` — `get_local_cb_interface(cb_idx_scratch).fifo_rd_ptr` **read** (`<< 4` → L1 mailbox address) — same fix path.

## Blocked on runtime (2xx rollup)

- `read_tile_value` on `DataflowBuffer` (Quasar, Runtime team — in progress) → `cb_counts_scratch`, `cb_idx_scratch`. UNPACK→mailbox→MATH/PACK scalar broadcast cannot use the raw `fifo_rd_ptr` address on 2xx; the tiled scalar read must go through the DFB read API.
