# Metal 2.0 Audit Findings — `experimental/quasar/transformer/sdpa`

> **Non-standard use of this recipe (declared up front).** The audit recipe treats
> `experimental/quasar/**` as out of bounds and is written for a *pre-port* feasibility
> check on a *source* op. This run is different by explicit user request: it is a
> **retrospective feasibility + port-state audit of an already-ported fork**. Commit
> `c1eaea9f196` forked `operations/transformer/sdpa` (SDPA + JointSDPA factories only)
> into this directory and ported it; the main-tree original was restored to legacy.
> The twelve feasibility subjects below are assessed against the **fork's own code**
> and conclude **GREEN** — the op was, and is, portable. The separate question the
> user actually asked — *where the existing port is not yet Metal 2.0 compliant* — is
> not a feasibility gate and lives in **`METAL2_PORTING_STATE_GAPS.md`** (companion
> file). Read that file for the port-state findings; this file is the gate record.

- **`SDPADeviceOperation`**
  - `SDPAProgramFactory` (`device/sdpa_program_factory.cpp`)
- **`JointSDPADeviceOperation`**
  - `JointSDPAProgramFactory` (`device/joint_sdpa_program_factory.cpp`)

Both device-operations share the kernel utility layer (`dataflow_common.hpp`,
`compute_common.hpp`, `compute_streaming.hpp`, `windowed_mask_gen.hpp`) and are audited
together as one porting unit.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`, with
the fork caveat above.

**Recipe docs:** `385e3f7a90d 2026-09-02 docs(metal_2.0): the conditional-binding pattern covers tensors and semaphores too`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/quasar/transformer/sdpa` |
| **Overall (feasibility)** | **GREEN** — op is portable; port already performed (see port-state gaps below) |
| **DOps / Factories** | `SDPADeviceOperation` → `SDPAProgramFactory` · `JointSDPADeviceOperation` → `JointSDPAProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — `Noc` / `Semaphore` / Device-2.0 `CircularBuffer` wrapper throughout; no `InterleavedAddrGen`/`ShardedAddrGen`/raw-sem idioms |
| *Prereqs* — Cross-op escapes | Ok — 6 pure-geometry headers included from main-tree `transformer/sdpa/device/kernels/` (no CB / arg idioms); coupling note only |
| *Feature Support* — overall | **GREEN** (all Appendix A entries N/A) |
| *Feature Support* — GlobalCircularBuffer / GlobalSemaphore / `address_offset` | N/A / N/A / N/A |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** — fork is already on `ProgramSpecFactoryConcept`; assessed from code (fork not in readiness sheet — see Recipe notes) |
| *TTNN Readiness* — Concept (current) | `MetalV2` (already ported) — host uses `ProgramSpec` / `KernelSpec` / `DataflowBufferSpec` / `TensorParameter` / `SemaphoreSpec` |
| *TTNN Readiness* — Secretly SPMD | N/A (single-program `descriptor` lineage) |
| *TTNN Readiness* — Custom hash | Not a gate; not analysed (retrospective) |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No |
| *TTNN Readiness* — `override_runtime_arguments` | No — `ProgramSpecFactoryConcept` (not the Custom variant) |
| *TTNN Readiness* — Pybind `create_descriptor` | No — `sdpa_nanobind.cpp` binds no `create_descriptor` |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` (achieved) |
| *Port work* — Offset base pointer | none — all tensors flow through `TensorParameter`; no host-folded offset RTAs |
| *Port work* — Tensor bindings (per binding) | clean / Case 1 — all bound as `TensorParameter`, kernel builds `TensorAccessor(tensor::name)`; no address RTAs |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) |
| *Port work* — TensorAccessor 3rd arg | drop (Class 2) — one site (page-table page-size), already dropped in the ported reader |
| *Port work* — CB endpoints | self-loop / 1P+1C — resolved in the host `DataflowBufferSpec` group; no multi-binding, no dead CB |

## Result

**GREEN — feasibility clears every gate; brief issued (`METAL2_PORT_BRIEF.md`).** The op
was portable and has been ported (fork `c1eaea9f196`). **However, the existing port is
not yet fully Metal 2.0 compliant** — the kernel-side CB→DFB transition (kernel-side
whitelist rule 1) was completed for the six top-level kernel entry points but **not** for
the four shared kernel *helper* headers, which remain on the Device-2.0 `CircularBuffer`
wrapper (176 references). That is a **port-completeness** finding, not a feasibility gate;
it is detailed in **`METAL2_PORTING_STATE_GAPS.md`**. On Gen1 (WH/BH) the two idioms are
functional synonyms, so tests pass; on the fork's stated Gen2/Quasar target they diverge,
so the gap matters for the fork's purpose.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** GREEN. The fork's host factory is already
  `ProgramSpecFactoryConcept`-shaped — `Group<DataflowBufferSpec>`, `Group<TensorParameter>`,
  `KernelSpec` with named `SemaphoreSpec`s, no `create()` / `override_runtime_arguments()`.
  The readiness sheet carries rows for the main-tree `transformer/sdpa` op, not this fork;
  the source op's SDPA + JointSDPA factories were previously confirmed `Is able to port? = yes`.
  Assessed here from code because the fork is already `MetalV2` (see Recipe notes).
- **Device 2.0 (every kernel used):** GREEN. Every kernel — top-level entry points *and* the
  shared helper headers — uses Device-2.0 idioms: `Noc` (from `noc.h`), `Semaphore`, and the
  Device-2.0 `CircularBuffer` wrapper (from `api/dataflow/circular_buffer.h`). No
  `InterleavedAddrGen` / `ShardedAddrGen` / `InterleavedPow2AddrGen*`, no
  `get_noc_addr_from_bank_id`, no raw `noc_semaphore_*`. **Note the conceptual point:** the
  `CircularBuffer` wrapper that fails the *Metal 2.0* completeness check (gaps file) is itself
  a valid *Device 2.0* idiom, so it passes this gate. Device-2.0-complete, Metal-2.0-incomplete.

- **Feature compatibility:** every Appendix A entry scanned; none fire.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no `GlobalCircularBuffer` type, `global_circular_buffer` field, `remote_cb_*` / `remote_index` idiom |
  | CBDescriptor `address_offset` (non-zero) | N/A | no `address_offset` / `set_address_offset` anywhere |
  | GlobalSemaphore | N/A | no `GlobalSemaphore` type or `CreateGlobalSemaphore` |

- **CB endpoints (GATE-free):** GREEN — resolved in the ported host `DataflowBufferSpec`
  group (`sdpa_program_factory.cpp:630+`). Endpoint dispositions (self-loop for single-toucher
  scratch CBs, 1P+1C for reader→compute / compute→writer flows, conditional DFBs for the
  windowed / paged / sink / chunk-start paths) are already expressed. No multi-binding
  advanced-option flag, no dead CB. (Detail per binding is embodied in the shipped spec; not
  re-censused line-by-line for this retrospective.)
- **Offset base pointers:** GREEN — no address RTA folds a host-side offset into a base.
  Every tensor reaches the kernel through a `TensorParameter` binding
  (`sdpa_program_factory.cpp:776+`); the page-table read uses `page_id = batch_idx` against a
  clean `tensor::page_table` base. No `buffer()->address() + offset` fold. Type 4 (`narrow`)
  not used.
- **TensorAccessor 3rd argument:** GREEN — Class 2 (redundant, dropped). The one 3rd-arg site
  is the page-table page-size override. In the ported reader it is correctly **dropped**
  (`reader_interleaved.cpp:388` — `TensorAccessor(tensor::page_table)`, binding supplies the
  aligned page size). The dead helper `read_page_table_for_batch` in `dataflow_common.hpp:77`
  still carries the legacy 3-arg form, but it is unreferenced (see gaps file — dead code).

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding): `q_in`, `k_in`, `v_in`, `out`, `mask_in` (opt),
  `page_table` (opt), `attention_sink` (opt), `cu_window_seqlens` (opt), `windowed_q_token_offset`
  (opt) — all **Case 1** (via `TensorAccessor` / borrowed-DFB); host binds each as a
  `TensorParameter`. No Case 2.
- **TensorParameter relaxation:** none.
- **TensorAccessor 3rd arg:** drop the page-table page-size arg — **already done** in the port.
- **CB endpoints:** self-loop + 1P+1C + conditional DFBs — **already expressed** in the host spec.

## Heads-ups  *(mirrors the brief)*

- **Port-state compliance gaps (primary):** three distinct items in the four shared kernel
  helper headers — (1) 176 `CircularBuffer` refs + 4 `circular_buffer.h` includes (rule 1);
  (2) a raw `LocalCBInterface` cursor mutation in `cb_push_back_hold_wr_ptr`
  (`compute_streaming.hpp:97`) needing `evil_set_write_ptr` (whitelist §D — *not* a type swap,
  and Gen1-only); (3) a dead legacy helper. Full detail in **`METAL2_PORTING_STATE_GAPS.md`**.
- **Cross-op / shared kernels:** the fork `#include`s 6 headers directly from main-tree
  `transformer/sdpa/device/kernels/` (`windowed_loop_geometry.hpp`, `q_chunk_remapping.hpp`,
  `chunked_prefill_utils.hpp`, `sdpa_streaming_qktv.hpp`, `sliding_window_geometry.hpp`,
  `sliding_window_work_plan.hpp`). All are pure geometry/compute-plan headers with no CB or
  legacy-arg idioms → not a compliance issue, but a fork-hygiene / self-containedness note.
- **RTA varargs:** none observed — args are named (`get_arg(args::…)`).

## Team-only

- **Out-of-directory coupling:** the 6 cross-tree includes above are the only escapes; all
  resolve to pure-math main-tree headers (Device-2.0-agnostic). No donor kernel is instantiated
  by file path from outside the fork (kernels are the fork's own `_metal2`→plain-name copies).
- **TTNN factory analysis:** host is `ProgramSpecFactoryConcept`; no op-owned tensors, no pybound
  `create_descriptor`, no `get_dynamic_runtime_args`, no `override_runtime_arguments`. All
  consistent with the source op's prior GREEN audit.

## Misc anomalies

- **Dead helper carrying legacy idioms:** `read_page_table_for_batch`
  (`dataflow_common.hpp:77`) is unreferenced in the fork (the one call site was inlined at
  `reader_interleaved.cpp:383`). It still carries the legacy `TensorAccessorArgs` + raw-address +
  `CircularBuffer` form. Dead → drop candidate (routed in the gaps file, not the port gate).

## Per-DeviceOperation attribution

Findings are identical across `SDPADeviceOperation` and `JointSDPADeviceOperation` (shared
helper layer); no per-DOp divergence. The joint variant adds `joint_reader.cpp` /
`joint_writer.cpp` / `compute/joint_sdpa.cpp` entry points — all on `dfb::` tokens (0
`CircularBuffer`), same as the prefill entry points.

## Questions for the user

None blocking. (One judgment call surfaced in Recipe notes.)

## Recipe notes

- **Retrospective audit of a quasar fork — outside the recipe's nominal scope.** The recipe
  declares `experimental/quasar/**` out of bounds and is a pre-port instrument; this run
  audits an already-ported fork by explicit user request. The gate subjects still map cleanly
  (feasibility is a property of the op, not of when you ask), but two mechanical steps didn't
  fit: (1) the **readiness sheet** has no row for a quasar fork, so the factory-concept gate
  was read from the fork's code (already `MetalV2`) rather than the sheet — I judged fetching
  the sheet would add nothing since the fork is visibly ported and the source op's rows were
  previously confirmed `yes`; (2) the recipe has **no output shape for "already ported but
  incomplete."** The feasibility verdict is honestly GREEN, but that undersells the finding
  the user cares about, so I split the port-completeness findings into a third file
  (`METAL2_PORTING_STATE_GAPS.md`) rather than distorting a feasibility gate into a RED it
  isn't. A future recipe variant for *post-port compliance review* (measuring an existing port
  against the kernel-side whitelist rather than the feasibility gates) would have a natural
  home for these findings.
