# Metal 2.0 Port Plan — `experimental/quasar/transformer/sdpa_decode`

> **Scope note.** This is a **completion port**, not a from-scratch one. `transformer/sdpa_decode`
> was forked into this directory and ported to `ProgramSpecFactoryConcept` in commit `cafa17411f3`.
> The host factory, program spec, bindings, and the main kernels (`reader_decode_all.cpp`,
> `writer_decode_all.cpp`, `sdpa_flash_decode.cpp`, `dataflow_common.hpp`) were already correctly
> converted. This port closes the two gaps the post-port audit found (see
> `METAL2_PORT_COMPLIANCE_GAPS.md`): two vendored donor kernel headers that were left on the legacy
> `CircularBuffer` API. **All planning below concerns those two headers only.**

**Recipe docs:** `b3eb82ae3d2 2026-09-02 docs(metal_2.0): the conditional-binding pattern covers tensors and semaphores too`
**Audit docs (inherited):** `b3eb82ae3d2 2026-09-02 docs(metal_2.0): the conditional-binding pattern covers tensors and semaphores too`

## Legacy Inventory

- **Legacy factory shape:** the fork's source (`transformer/sdpa_decode`) is on the
  `ProgramDescriptorFactoryConcept` (`create_descriptor` → `ProgramDescriptor`). Already realized as
  `ProgramSpecFactoryConcept` in this fork by commit `cafa17411f3`; **not re-touched here.**
- **Custom `compute_program_hash`:** none. (Unchanged; not touched.)
- **Kernels in scope for this completion** (the two that were still on `CircularBuffer`):
  - `device/kernels/compute/compute_common.hpp` — compute helper library included by
    `sdpa_flash_decode.cpp`. **Unported:** 78 `CircularBuffer` sites, `#include "api/dataflow/circular_buffer.h"`
    (`:31`), 0 `DataflowBuffer`. Byte-identical to the legacy sdpa-prefill copy it was vendored from.
  - `device/kernels/dataflow/sdpa_dataflow_common.hpp` — mask/fill dataflow helper library included
    (via `dataflow_common.hpp`) by `reader_decode_all.cpp` / `writer_decode_all.cpp`. **Partially
    ported:** one Metal 2.0 `read_page_table_for_batch(DataflowBuffer&)` overload was already added,
    but 20 `CircularBuffer` sites remained and the stale `circular_buffer.h` include (`:15`) was kept
    alongside the added `dataflow_buffer.h` (`:16`).
- **CB / DFB APIs used by these two files (inventory drives the whitelist mapping):**
  - `compute_common.hpp`: canonical FIFO only on the buffer objects — `reserve_back` / `push_back` /
    `wait_front` / `pop_front`. Plus LLK free functions (`pack_tile`, `copy_tile`, `reduce_init`)
    that take a `uint32_t` CB id — those take the id directly and are **unchanged**.
  - `sdpa_dataflow_common.hpp`: FIFO (`reserve_back`/`push_back`/`wait_front`/`pop_front`) + public
    cursor peeks (`get_write_ptr`/`get_read_ptr`). All map 1:1 to `DataflowBuffer`.
  - **No** CircularBuffer-only constructs (`AddrSelector`, `CircularBufferView`, `use<...>`,
    `get_cb_tiles_acked_ptr`/`get_cb_tiles_received_ptr`, `get_local_cb_interface`, `evil_set_*`),
    and **no** metadata getters on the objects — verified by grep. So the swap is a pure type +
    name change with no method rewrites.
- **Shared kernels:** these two files are **vendored copies** local to this op (the fork relocated
  the whole op into the quasar tree). Editing them does not touch the prefill `sdpa` op, which keeps
  its own main-tree copies. No `_metal2` fork mechanism is involved.

## TTNN ProgramFactory

- **Concept (inherited from audit):** `ProgramSpecFactoryConcept`. Already realized; unchanged.
- **Custom `compute_program_hash`:** none.
- **Implementation notes:** no host-side work in this port — factory/spec/run-args are byte-unchanged.

## Planned Spec Shape

No change. Every `DataflowBufferSpec`, `TensorParameter`, `SemaphoreSpec`, `WorkUnitSpec`, and binding
was built by the prior port and is correct (multi-binding `c_16`, `q_in` self-loop, borrowed-DFB
outputs, Case-2 `get_bank_base_address` bridge). This completion touches **kernel source only**.

## Preserved Multiplicity

N/A — no `KernelSpec` changes.

## Dropped Plumbing

N/A for the host side (done previously). The only "drop" here is the stale
`#include "api/dataflow/circular_buffer.h"` in both files.

## Applied Patterns — the mechanical conversion

Follow **kernel-side whitelist rule 1** (CB→DFB total, `cb_*`→`dfb_*`, drop the `circular_buffer.h`
include) and the **CB→DFB API whitelist §A/§C** (FIFO + cursor peeks map 1:1). Match the convention
the already-ported sibling `dataflow_common.hpp` established (local `dfb`, params renamed `cb_*`→`dfb_*`).

Transformation applied to both files:
- `#include "api/dataflow/circular_buffer.h"` → dropped (`compute_common.hpp` gains
  `#include "api/dataflow/dataflow_buffer.h"`; `sdpa_dataflow_common.hpp` already had it).
- `CircularBuffer` → `DataflowBuffer` (the only functional change; FIFO/peek methods are identical).
- Object locals `cb_<x>` → `dfb_<x>`; bare local `cb` → `dfb`.
- CB-id params/NTTPs: `<x>_cb` → `<x>_dfb`, `cb_<x>` → `dfb_<x>`, and the two mid-token names
  `dst_cb_id` → `dst_dfb_id`, `page_table_cb_wr_ptr` → `page_table_dfb_wr_ptr`.
- Comment references to "CB" → "DFB" (rule 8 comment sweep).
- **Preserved, not buffers:** `reconfig_data_format_srcb`, `SrcB`, `srcB` (source-B register
  operands — they contain the letters "cb"/"cB" but are not circular buffers; the self-audit grep
  correctly ignores them, and so must the rename).

Rule 7 (metadata via object getter) does not apply — neither file reads tile/format metadata off the
buffer objects.

## Deferred / Flagged

- MLA (`flash_multi_latent_attention_decode`) compute path exists in the fork but has **no test** in
  the fork's suite — so it is exercised by construction (shared compute kernel) but not independently
  verified. Noted in the report.
- `rt_args_common.hpp:102` carried a stray "CB" comment (a leftover the prior port missed, not one of
  the two audited headers). Fixed here as part of completing the op's CB→DFB sweep; noted in the report.
