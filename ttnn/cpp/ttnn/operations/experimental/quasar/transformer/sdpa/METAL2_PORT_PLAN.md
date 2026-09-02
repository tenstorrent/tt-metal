# Port Plan — `experimental/quasar/transformer/sdpa` (SDPA + JointSDPA)

Port plan for the quasar SDPA fork, completing its Metal 2.0 port. Written during
inventory/planning; committed alongside the port for review.

> **Unusual shape — a port-*completion*, not a from-scratch port.** This op (fork
> `c1eaea9f196`) was already ~85% ported: the host factories and all six top-level kernel
> entry points are on Metal 2.0. The audit (`METAL2_PREPORT_AUDIT.md`, GREEN) and the gaps
> analysis (`METAL2_PORTING_STATE_GAPS.md`) found the remaining work confined to the **four
> shared kernel *helper* headers**. This plan covers only that work. **No host-side file is
> touched** — the `ProgramSpec` / `DataflowBufferSpec` / `TensorParameter` / `KernelSpec`
> wiring already exists and is unchanged, so all of the recipe's spec-construction,
> `hw_config`, and `opt_level` guidance is already satisfied and out of scope here.

## Legacy Inventory

### Factory shape
- Concept: **`ProgramSpecFactoryConcept`** (already realized — `MetalV2`). Not re-derived; inherited.
- Variants: two DeviceOperations sharing the kernel layer — `SDPADeviceOperation` /
  `SDPAProgramFactory` and `JointSDPADeviceOperation` / `JointSDPAProgramFactory`.
- Custom `compute_program_hash`: not touched by this port (kernel-only change).

### The three gaps (from `METAL2_PORTING_STATE_GAPS.md`) — the entire scope of this port

| Gap | File(s) | Whitelist rule | Change |
|---|---|---|---|
| **1 — CircularBuffer → DataflowBuffer** | all 4 helper headers | rule 1 (total CB→DFB) | 176 `CircularBuffer` → `DataflowBuffer`; `#include circular_buffer.h` → `dataflow_buffer.h` |
| **2 — raw cursor mutation** | `compute/compute_streaming.hpp:97` | rule 1 §D | `cb_push_back_hold_wr_ptr` raw `fifo_wr_ptr` writes → `DataflowBuffer::evil_set_write_ptr` |
| **3 — dead helper** | `dataflow/dataflow_common.hpp:77` | — | delete unreferenced `read_page_table_for_batch` |

### Files in scope (op's own directory — writeable surface)
- `device/kernels/dataflow/dataflow_common.hpp` (20 CB refs)
- `device/kernels/compute/compute_common.hpp` (78)
- `device/kernels/compute/compute_streaming.hpp` (75, incl. Gap 2)
- `device/kernels/dataflow/windowed_mask_gen.hpp` (3)

### Out of scope (do NOT touch)
- All host `.cpp` / `.hpp` (already Metal 2.0).
- The 6 pure-geometry headers included from main-tree `transformer/sdpa/device/kernels/`
  (`windowed_loop_geometry.hpp`, `q_chunk_remapping.hpp`, `chunked_prefill_utils.hpp`,
  `sdpa_streaming_qktv.hpp`, `sliding_window_geometry.hpp`, `sliding_window_work_plan.hpp`) —
  outside the op directory, no CB idioms, no port work.

## Planned changes (mechanical, per the kernel-side whitelist)

### Gap 1 — `CircularBuffer` → `DataflowBuffer`
Pure type-token swap. Verified safe:
- **Method surface**: the only methods called on these objects are `reserve_back` / `push_back`
  / `wait_front` / `pop_front` / `get_write_ptr` / `get_read_ptr` — all map 1:1 (whitelist §A/§C).
  No exotic CB-only methods (`get_tile_address`, `read_tile_value`, `scoped_lock`) are used.
- **Constructor**: helpers take `uint32_t cb_id`; `DataflowBuffer(uint16_t logical_dfb_id)` accepts
  it (established kernel_lib pattern, e.g. `kernel_lib/l1_helpers.hpp`, `softmax_large_tensor.cpp`).
- **`get_write_ptr()` semantics — checked.** `DataflowBuffer::get_write_ptr()` adds
  `L1_UNCACHED_OFFSET`, which is `MEM_L1_UNCACHED_BASE` **only** on `ARCH_QUASAR && COMPILE_FOR_DM`
  and **`0` on WH/BH and all TRISC**. So on the Gen1 target it equals the raw
  `CircularBuffer::get_write_ptr()` → **zero behavior change**; on Quasar DM it correctly returns
  the uncached alias for local L1 pointer deref (the exact Gen1/Gen2 divergence the audit flagged).
- **Include**: swap `api/dataflow/circular_buffer.h` → `api/dataflow/dataflow_buffer.h` in each
  of the 4 headers.

### Gap 2 — `cb_push_back_hold_wr_ptr` (§D cursor surgery)
Legacy mutates `LocalCBInterface` fields directly (forbidden). Faithful conversion, verified exact:
- `DataflowBuffer::evil_set_write_ptr(addr)` sets `local_dfb_interface_.fifo_wr_ptr = addr` **raw**
  (no shift, no offset), and the DFB's `local_dfb_interface_` is a **reference** initialized from
  `get_local_cb_interface(id)` — the *same* object the legacy code wrote. So reading the raw fields
  via the sanctioned `get_local_cb_interface(cb_id)` free function and redirecting only the *write*
  through `evil_set_write_ptr` is byte-exact.
- **Units stay raw**: keep `intf.fifo_page_size` / `fifo_limit` / `fifo_size` (raw 16B units) — do
  **not** use the §B getters (`get_entry_size()` etc.), which return bytes and would mismatch.
- **Quasar note**: `evil_set_*` is Gen1-only (unavailable on Quasar) → still Quasar-uplift debt
  after this fix (recorded in the report). This "hold-wr" trick needs a real refactor before Gen2;
  the port only makes it Metal-2.0-legal.

### Gap 3 — delete dead `read_page_table_for_batch`
Unreferenced (its one call site was inlined at `reader_interleaved.cpp:383`). Removing it drops the
last `TensorAccessorArgs` + raw-address form in the op. Zero-functional-change (dead code).

## Dropped Plumbing
| legacy | replacement |
|---|---|
| `CircularBuffer` (Device-2.0 wrapper) ×176 | `DataflowBuffer` |
| `#include "api/dataflow/circular_buffer.h"` ×4 | `#include "api/dataflow/dataflow_buffer.h"` |
| raw `intf.fifo_wr_ptr = …` ×2 | `dfb.evil_set_write_ptr(…)` |
| dead `read_page_table_for_batch` | (deleted) |

## Applied Patterns
- CB→DFB API whitelist §A (canonical FIFO), §C (public peeks), §D (cursor surgery → `evil_set_*`).
- Kernel-side whitelist rule 1 (total CB→DFB transition) and rule 7 (`get_tile_size(dfb::name)`
  constexpr token form — already present in the entry kernels; helpers keep raw-unit interface reads).

## Deferred / Flagged
- **JointSDPA has no runtime test** in the fork's test suite (only prefill SDPA). Build verifies its
  kernels compile against the converted headers; runtime coverage is a gap to note.
- **Host-side legality-check forcing is moot** for this port: no `ProgramSpec` is modified, so the
  validation surface is identical to the already-passing baseline. Watcher-on + the correctness
  tests are the relevant safety net for the kernel-side change. (Documented in the report.)
