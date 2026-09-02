# Metal 2.0 Port Brief — `experimental/quasar/transformer/sdpa_decode`

> Feasibility audit cleared all gates. The full record is in `METAL2_PREPORT_AUDIT.md`.
>
> **This op is already ported** (commit `cafa17411f3`), but the landed port is **incomplete against
> this brief**: two vendored kernel headers remain on the legacy `CircularBuffer` API. This brief
> therefore doubles as the spec for the *remaining* port work. The concrete, file-scoped gap list is
> in **`METAL2_PORT_COMPLIANCE_GAPS.md`** — start there for the fix.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `b3eb82ae3d2 2026-09-02 docs(metal_2.0): the conditional-binding pattern covers tensors and semaphores too`

## TTNN factory analysis

The op ports to `ProgramSpecFactoryConcept`. Carry these forward:

- **Current concept (of the pristine reference):** `descriptor` (`create_descriptor` → `ProgramDescriptor`).
- **Op-owned tensors:** none.
- **Target concept:** `ProgramSpecFactoryConcept`.
- **Gate-cleared, confirmed absent:** a non-`none` `TensorParameter relaxation`; `get_dynamic_runtime_args`.
  (No custom hash, no `override_runtime_arguments`, no pybound `create_descriptor` — none of which gate.)

## Construct — to do

**Tensor bindings** (per binding):

- `q` — **Case 1** (DRAM `TensorAccessor`) in the interleaved path; **Case 2** (raw L1 base) in the
  HEIGHT_SHARDED non-MLA path → bind as `TensorParameter`, pull the base via `get_bank_base_address`,
  keep the raw arithmetic; **clean borrowed-DFB** in the MLA-local path. (Applied in the fork.)
- `output`, `cur_pos`, `page_table` — **clean** borrowed-memory DFB (`DataflowBufferSpec::borrowed_from`).

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** drop the redundant page-size arg at the two Class-2 sites (Q accessor;
shared sdpa donor page-table read). No `dynamic_tensor_shape`. (Applied in the fork.)

**CB endpoints:**
- `c_16` (`cb_out_o`/`cb_out_worker`) — set the multi-binding advanced option (tree reduction: writer
  P+C, compute P+C). (Applied — `device/sdpa_decode_program_factory.cpp:609`.)
- `q_in` — compute self-loop under `TILIZE_Q`. (Applied — `:640-641`.)
- Remaining CBs — plain 1P+1C. (Applied.)

**CircularBuffer → DataflowBuffer completeness (kernel-side whitelist rule 1) — NOT YET DONE.** The
transition is total: post-port, *no* `CircularBuffer` references may survive in the op directory.
Two vendored headers still fail this:
- `device/kernels/compute/compute_common.hpp` — **78** `CircularBuffer` sites, stale
  `#include "api/dataflow/circular_buffer.h"` (`:31`), zero `DataflowBuffer`.
- `device/kernels/dataflow/sdpa_dataflow_common.hpp` — **20** `CircularBuffer` sites, stale
  `#include "api/dataflow/circular_buffer.h"` (`:15`).

Convert both per rule 1 + the CB→DFB API whitelist. See `METAL2_PORT_COMPLIANCE_GAPS.md` for the
detailed treatment and the DFB-handle threading it implies at the call sites.

## Watch for

- **CB endpoints (multi-binding):** `c_16` — the second writer/compute endpoint is a genuine
  tree-reduction co-touch, not a stray; the flag is correct, don't try to relabel it to 1P+1C.
- **Cross-op / shared kernels:** `compute_common.hpp` and `sdpa_dataflow_common.hpp` are **vendored
  copies** of the sdpa-prefill donors, now op-owned — convert them in place (they are NOT shared;
  editing them does not affect prefill, which keeps its own main-tree copies). Do not treat them as
  off-limits shared code.
- **RTA varargs:** the data-indexed physical-core-coordinate arrays stay as varargs (correct).
- **`c_11` (`col_identity`):** dead in decode (writer fills, nothing consumes) — self-loop it; ops
  team confirms removal separately. Not a port blocker.
