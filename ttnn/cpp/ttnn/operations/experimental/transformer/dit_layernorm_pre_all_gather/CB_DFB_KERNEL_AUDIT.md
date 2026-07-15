# CB→DFB Kernel Audit: `dit_layernorm_pre_all_gather`

**Date:** 2026-07-15
**Op root:** `ttnn/cpp/ttnn/operations/experimental/transformer/dit_layernorm_pre_all_gather/`

**Scope:** `device/kernels/compute/layernorm_pre_allgather_welford.cpp`, `device/kernels/dataflow/reader_layernorm_preallgather_dit.cpp`, `device/kernels/dataflow/writer_layernorm_preallgather_dit.cpp`. Donor include: `ttnn/operations/normalization/kernel_util/compute/memory.h` (Welford reciprocal-LUT pointer helper).

## Overall verdict: YELLOW

**Summary:** All dataflow/compute CBs are canonical Class 1 linear FIFOs **except** `cb_reciprocals`, which the Welford compute kernel reads sync-free via `kutil::compute::memory::get_pointer_to_cb_data<recip_lut_t>(cb_reciprocals, 0)` (`layernorm_pre_allgather_welford.cpp:40`). That helper (`kernel_util/compute/memory.h:31`) is `reinterpret_cast<To*>(get_tile_address(cb_id, tile_index))`. This is a **Class 6 sync-free borrowed read** → **Portable (prereq: LTA)** → op rollup **YELLOW**. It is **not** a GATE (`get_pointer_to_cb_data`/`get_tile_address` are sanctioned APIs, not `get_local_cb_interface` field access).

## CB portability

| CB | Class | Kernel(s) | 1xx status | 1xx notes | 2xx status | 2xx notes |
|----|-------|-----------|------------|-----------|------------|-----------|
| `cb_inp` (c_0) | 1 | `layernorm_pre_allgather_welford.cpp`, `reader_*` | Portable | input activations, linear FIFO → `DataflowBuffer` | Portable | — |
| `cb_x2` (c_1) | 1 | `layernorm_pre_allgather_welford.cpp` | Portable | E(x²) intermediate, canonical `reserve_back`/`pack_tile`/`push_back` + transpose stage | Portable | — |
| `cb_reduce` (c_1, reader) | 1 | `reader_layernorm_preallgather_dit.cpp` | Portable | reduce-scalar FIFO from reader | Portable | — |
| `cb_out` (c_14) | 1 | `layernorm_pre_allgather_welford.cpp`, `writer_*` | Portable | pack → stats output, `get_write_ptr()` as NoC addr | Portable | — |
| `cb_reciprocals` (c_2) | 6 | `layernorm_pre_allgather_welford.cpp` (via `memory.h`) | Portable (prereq: LTA) | sync-free borrowed reciprocal LUT → **LocalTensorAccessor** (replaces `get_pointer_to_cb_data`) | Portable (prereq: LTA) | same target; note: if `memory.h` is retained instead of migrating, the underlying `get_tile_address` is **QUASAR-BLOCKED** until the DFB read API lands (Runtime team, in progress). LTA migration clears both arches. |

## GATE hits (must be empty to merge)

- (none) — `get_pointer_to_cb_data`/`get_tile_address` are sanctioned APIs, not `get_local_cb_interface` field access.

## Blocked on runtime (2xx rollup)

- `cb_reciprocals` — targeting **LocalTensorAccessor** removes the runtime dependency entirely. If instead the shared `memory.h` path is kept on Quasar, it depends on `get_tile_address` / `read_tile_value` on `DataflowBuffer` (in-progress). Recommended: migrate the six-kernel `memory.h` Welford LUT family to LTA in the port.
