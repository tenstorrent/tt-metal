# Metal 2.0 Audit Findings — `experimental/quasar/transformer/sdpa_decode`

> **Retrospective audit.** This op has *already* been forked and ported (commit `cafa17411f3`,
> "[Metal 2.0] Port `transformer/sdpa_decode` (#54249)"). This document is the standard feasibility
> audit — it records whether the op *could* be ported and what a compliant port looks like — run
> against the pristine `descriptor`-concept reference that still lives at
> `ttnn/cpp/ttnn/operations/transformer/sdpa_decode/` (the fork's source). The feasibility verdict is
> **GREEN** and matches the original pre-port audit. **The port that was actually landed is
> incomplete against that GREEN target** — two vendored kernel headers were left on the legacy
> CircularBuffer API. Those port-execution gaps are *not* feasibility findings; they are recorded
> separately in **`METAL2_PORT_COMPLIANCE_GAPS.md`** in this directory. Read that file for the "what's
> wrong with the port" answer; this file is the "was it portable, and how" record.

**Identifying section**

- **`SdpaDecodeDeviceOperation`** (single device-operation)
  - `SdpaDecodeProgramFactory` (`device/sdpa_decode_program_factory.cpp`) — one factory; paged / MLA /
    height-sharded / sliding-window / attention-sink / geometry-override are internal branches, not
    separate factories.
- Kernels the factory instantiates (`device/kernels/`):
  - `dataflow/reader_decode_all.cpp` → includes `dataflow/dataflow_common.hpp` → includes
    `dataflow/sdpa_dataflow_common.hpp`
  - `dataflow/writer_decode_all.cpp` → includes `dataflow/dataflow_common.hpp`
  - `compute/sdpa_flash_decode.cpp` → includes `compute/compute_common.hpp`
- `sdpa_dataflow_common.hpp` and `compute_common.hpp` are **vendored copies** of the sdpa-prefill
  donor headers (`transformer/sdpa/device/kernels/{dataflow/dataflow_common.hpp, compute/compute_common.hpp}`),
  copied into the op directory by the fork. Because they are now the op's own files and are
  transitively included by every instantiated kernel, they are **fully in audit/port scope**.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`. Per the user's
instruction, the `experimental/quasar` out-of-scope rule is waived for *this* op (it is a Metal 2.0
port that was relocated into the quasar tree, not a functional quasar port).

**Recipe docs:** `b3eb82ae3d2 2026-09-02 docs(metal_2.0): the conditional-binding pattern covers tensors and semaphores too`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/quasar/transformer/sdpa_decode` |
| **Overall (feasibility)** | **GREEN** — all gates clear (matches the original pre-port audit) |
| **Port-execution status** | **PORTED BUT NON-COMPLIANT** — see `METAL2_PORT_COMPLIANCE_GAPS.md` |
| **DOps / Factories** | `SdpaDecodeDeviceOperation` → `SdpaDecodeProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — kernels use `CircularBuffer`/`Noc`/`Semaphore<>`/`TensorAccessor` objects (Device 2.0 wrappers); only sanctioned `get_tile_size(cb)` free-function survives |
| *Prereqs* — Cross-op escapes | Ok — shared sdpa-donor headers were vendored into the op dir (now op-owned) |
| *Feature Support* — overall | **GREEN** (all Appendix A entries N/A) |
| *Feature Support* — GlobalCircularBuffer / GlobalSemaphore / `address_offset` | N/A — none present (verified on the pristine reference) |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** (per the original readiness-sheet lookup; **not re-fetched this session** — see Gate detail) |
| *TTNN Readiness* — Concept (of the source) | `descriptor` (pristine ref) → ported to `ProgramSpecFactoryConcept` (the fork) |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No |
| *TTNN Readiness* — `override_runtime_arguments` | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` |
| *Port work* — Offset base pointer | none — `Buffer*` BufferBinding delivery, no `->address()+offset` folds |
| *Port work* — Tensor bindings (per binding) | Q: Case 1 (DRAM `TensorAccessor`) / Case 2 (HEIGHT_SHARDED non-MLA raw L1 → `get_bank_base_address`) / clean borrowed-DFB (MLA-local); output/cur_pos/page_table borrowed-DFB |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) |
| *Port work* — TensorAccessor 3rd arg | two Class-2 sites — clean drops, no `dynamic_tensor_shape` |
| *Port work* — CB endpoints | `c_16` multi-binding (tree reduction); `q_in` self-loop; the rest 1P+1C |

## Result

**GREEN → the op is portable, and was ported.** The feasibility audit clears every gate; the target
is `ProgramSpecFactoryConcept`. **However, the landed port does not match this GREEN target** — the
vendored `compute_common.hpp` (unported, 78 `CircularBuffer` sites) and `sdpa_dataflow_common.hpp`
(partial port, 20 `CircularBuffer` sites) violate kernel-side whitelist rule 1 ("post-port, *no*
`CircularBuffer` references survive"). See **`METAL2_PORT_COMPLIANCE_GAPS.md`**. A brief
(`METAL2_PORT_BRIEF.md`) is emitted because the *feasibility* is GREEN; it doubles as the spec the
remaining port work must satisfy.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** GREEN. The pristine reference
  (`transformer/sdpa_decode/device/sdpa_decode_device_operation.hpp:95`,
  `sdpa_decode_program_factory.cpp:29`) is on the `descriptor` concept — a single `create_descriptor`
  returning a `ProgramDescriptor`, no custom hash, no `get_dynamic_runtime_args`, no
  `override_runtime_arguments`, no pybound `create_descriptor`, no op-owned tensors. The original
  readiness-sheet lookup recorded `Is able to port? = yes`. **Caveat:** the readiness sheet was *not*
  re-fetched in this session (the Drive connector authorizes only in an interactive main session, and
  the sheet is unchanged code for this op). The verdict rests on the prior recorded value plus a code
  cross-check that `Concept == descriptor` still holds. If a fresh gate value is needed, re-fetch per
  `analyses/ttnn_op_porting_readiness.md`.
- **Device 2.0 (every kernel used):** GREEN. Every instantiated kernel — and the two vendored donor
  headers — is on Device 2.0 idioms: `CircularBuffer` / `Noc` / `Semaphore<>` / `TensorAccessor`
  wrapper objects, no raw `noc_async_*` addr-gen, no `InterleavedAddrGen` family. `CircularBuffer` is
  the *Device 2.0* wrapper (it satisfies the Device 2.0 gate); converting it to `DataflowBuffer` is
  *Metal 2.0* port work, not a Device 2.0 blocker. This is exactly why the two vendored headers pass
  the gate yet still need the CB→DFB conversion (the gap doc).
- **Feature compatibility:** all Appendix A entries N/A.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | 0 hits on the pristine reference (no `remote_cb`/`.remote_index`/`global_circular_buffer`) |
  | CBDescriptor `address_offset` (non-zero) | N/A | 0 hits |
  | GlobalSemaphore | N/A | 0 hits |

- **CB endpoints (GATE-free):** `c_16` (`cb_out_o`/`cb_out_worker`) is genuine multi-binding — the
  tree reduction reuses one index bidirectionally (2P+2C on intermediate reduction nodes) → the port
  set `advanced_options.allow_instance_multi_binding = true` (`device/sdpa_decode_program_factory.cpp:609`).
  `q_in` is a compute self-loop under the `TILIZE_Q` path (`:640-641`). The remaining CBs are plain
  1P+1C. All correctly expressed in the landed factory.
- **Offset base pointers:** GREEN — no address RTA folds a host offset into a base. Delivery is the
  `Buffer*` BufferBinding pointer-patching form.
- **TensorAccessor 3rd argument:** GREEN — two Class-2 (redundant) sites (Q accessor; the page-table
  read in the shared sdpa donor), both clean drops; no `dynamic_tensor_shape`.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding): `q` — Case 1 (DRAM `TensorAccessor`) / Case 2 (HEIGHT_SHARDED
  non-MLA raw L1, bridged via `get_bank_base_address`, `device/kernels/dataflow/dataflow_common.hpp:2`)
  / clean (MLA-local borrowed DFB). `output`, `cur_pos`, `page_table` — clean borrowed-DFB.
- **TensorParameter relaxation:** none.
- **TensorAccessor 3rd arg:** drop the redundant page-size arg at the two Class-2 sites | done in the fork.
- **CB endpoints:** `c_16` multi-binding flag; `q_in` self-loop; rest 1P+1C — all applied in the fork.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding):** `c_16` tree-reduction (writer P+C, compute P+C) — flag set correctly.
- **Cross-op / shared kernels:** the sdpa-prefill donor helpers `compute_common.hpp` and
  `dataflow_common.hpp` were **vendored into this op's directory** rather than reused as `_metal2`
  forks (the fork relocated the whole op into the quasar tree). Because they are now op-owned, they
  must be fully CB→DFB converted — **this is where the landed port fell short** (see gaps doc).
- **RTA varargs:** the data-indexed physical-core-coordinate arrays are kept as varargs (correct).

## Team-only

- **Out-of-directory coupling:** after vendoring, the op owns all of its kernel sources. Remaining
  transitive includes to main-tree helpers (`q_chunk_remapping.hpp`, `chunked_prefill_utils.hpp`,
  `sliding_window_geometry.hpp`) are `CircularBuffer`-free (verified) — no coupling concern.
- **TTNN factory analysis:** `descriptor` concept, no op-owned tensors, no custom hash, no
  `override_runtime_arguments`, no pybound `create_descriptor` → target `ProgramSpecFactoryConcept`.

## Misc anomalies  *(team-only, non-gating)*

- `c_11` (`col_identity`) — filled by the writer, unconsumed in sdpa_decode (dead path inherited from
  sdpa-prefill's matmul-reduce). Flagged in the port PR for the ops team; not porter-actionable.
- `device/kernels/dataflow/dataflow_common.hpp:261,328` — pre-existing `// TODO: Make sure this is -inf`
  on `NEG_INF = 0xFF80FF80`; inherited from the legacy kernel, not introduced by the port.

## Questions for the user

1. **Readiness-sheet re-fetch:** the `Is able to port?` gate rests on the original lookup, not a
   fresh fetch (Drive connector is main-session-only and the op's TTNN shape is unchanged). Re-fetch
   if a current sheet value is required for sign-off.

## Recipe notes

- The recipe assumes the audit runs *before* the port, on legacy code. Here it ran *after* an
  already-landed (partial) port, at the user's request. The feasibility subjects were answered against
  the pristine `descriptor` reference; the port-execution compliance check (rule-1 CB→DFB
  completeness) has no home in `metal2_audit.md` — it belongs to the port recipe / `pass_procedure.md`
  — so it is reported in a separate `METAL2_PORT_COMPLIANCE_GAPS.md` rather than shoehorned into a
  feasibility gate. A short "post-port compliance re-audit" mode in the recipe would formalize this.
