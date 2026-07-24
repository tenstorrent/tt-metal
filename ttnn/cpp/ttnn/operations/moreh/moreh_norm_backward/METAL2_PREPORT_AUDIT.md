# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/moreh/moreh_norm_backward`

- **`MorehNormBackwardOperation`** (single DeviceOperation)
  - Single program factory: `create_descriptor(...)` in `device/moreh_norm_backward_program_factory.cpp` (returns `tt::tt_metal::ProgramDescriptor`)
  - Kernels (all op-owned, under `device/kernels/`):
    - `reader_moreh_norm_backward.cpp` (reader, DM)
    - `writer_moreh_norm_backward.cpp` (writer, DM)
    - `moreh_norm_backward_kernel.cpp` (compute)

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

> **Process caveat (read first):** the per-factory readiness sheet ("Operations analysis") **could not be fetched in this session** — the claude.ai Google Drive connector authorizes only in the main interactive session, and the download was blocked here (the exact subagent-OAuth wall `ttnn_op_porting_readiness.md` warns about); no local CSV fallback exists. The TTNN-factory-concept gate below is therefore resolved by the **full code-level cross-check** (all four shape conjuncts + the smuggled-pointer correctness signal), not by the sheet. The one axis code cannot derive — Diego's `Is safe to port?` expert call — must be confirmed against the sheet in the main session before the port starts. See *Questions for the user*. Every other gate is fully resolved by static analysis.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/moreh/moreh_norm_backward` |
| **Overall** | **GREEN** (pending readiness-sheet confirmation of `Is safe to port?` — see caveat) |
| **DOps / Factories** | `MorehNormBackwardOperation` → single `create_descriptor` factory (`moreh_norm_backward_program_factory.cpp`) |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** (own reader/writer/compute all on `Noc`/`DataflowBuffer`/`TensorAccessor`; one lib-team note below, non-gating) |
| *Prereqs* — Cross-op escapes | Ok (shared-lib `moreh_common.hpp` only; no cross-family/borrowed kernel files) |
| *Feature Support* — overall | **GREEN** (all Appendix A entries N/A) |
| *Feature Support* — Variadic-CTA | Ok (no runtime-varying CTA loop; no `std::vector<Tensor>` in `tensor_args_t`) |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** (by code cross-check; sheet unconfirmed — see caveat) |
| *TTNN Readiness* — Concept (current) | `descriptor` (`create_descriptor` returns `ProgramDescriptor`) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A (not a `WorkloadDescriptor` op) |
| *TTNN Readiness* — Is safe to port? | **Unconfirmed** (sheet not fetched) — code signal clean: no smuggled pointer (uses `Buffer*` binding form, not raw `->address()`) |
| *TTNN Readiness* — Custom hash | No (no `compute_program_hash` override in the device-op) |
| *TTNN Readiness* — Runtime-args update | No (no `get_dynamic_runtime_args` / `override_runtime_arguments`) |
| *TTNN Readiness* — Pybind `create_descriptor` | No (nanobind binds the `moreh_norm_backward` function only) |
| *TTNN Readiness* — Op-owned tensors | No (`descriptor` concept; no op-owned tensors) |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` |
| *Port work* — Offset base pointer | none (all bases clean; no host-folded offsets) |
| *Port work* — Tensor bindings (per binding) | Case 1 ×4 (`input`, `output`, `output_grad`, `input_grad`) |
| *Port work* — TensorParameter relaxation | none (no custom hash) |
| *Port work* — TensorAccessor 3rd arg | none (every `TensorAccessor` is 2-arg) |
| *Port work* — CB endpoints | legal / self-loop (all resolvable; no flags, no dead CBs) |

**Tensorless-dispatch check (orchestrator-requested):** **NOT a block.** `tensor_args_t` carries **three required** input tensors (`const Tensor& input`, `const Tensor& output`, `const Tensor& output_grad`) plus one optional (`const std::optional<Tensor>& input_grad`) — `device/moreh_norm_backward_device_operation.hpp:28-33`. The op can never be dispatched with an empty `tensor_args`; the factory already sources the device from a live tensor (`output_grad.device()`, `moreh_norm_backward_program_factory.cpp:70`). The MetalV2 factory adapter's tensor-sourced-MeshDevice requirement is satisfied.

## Result

**GREEN → brief issued.** All statically-auditable gates clear: Device 2.0 ✓, Feature compatibility ✓ (all N/A), Offset base pointers ✓, TensorAccessor 3rd arg ✓, and the TTNN-factory-concept shape cross-check ✓ (`descriptor`; custom-hash / runtime-args-update / pybind-`create_descriptor` all absent; smuggled-pointer signal clean). Target concept: `MetalV2FactoryConcept`.

**One open confirmation, not a gate failure:** the readiness sheet could not be fetched in this subagent session, so Diego's `Is safe to port?` expert axis is unconfirmed. Confirm it in the main session (fetch the sheet, verify `Is able to port? == yes`) before beginning the port. No portable-subset scoping is needed — the op is a single factory and is GREEN in full.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN by code cross-check** (sheet unfetchable this session — see caveat). Concept is `descriptor` (`create_descriptor` returns `ProgramDescriptor`, `moreh_norm_backward_program_factory.cpp:58`). All gate conjuncts confirmed absent from code: no `compute_program_hash` override, no `get_dynamic_runtime_args` / `override_runtime_arguments` (grep of the op dir returns nothing), no pybind `create_descriptor` (`moreh_norm_backward_nanobind.cpp:19` binds the `moreh_norm_backward` free function). The correctness axis is code-verifiable-clean here: the factory deliberately passes tensor **`Buffer*`** (not raw `->address()`) into RTAs (`moreh_norm_backward_program_factory.cpp:252-257,268`, with an explanatory comment), which is the framework's cache-hit-patched `BufferBinding` form — *not* a smuggled pointer. Route only if the sheet contradicts this (→ readiness-sheet owner).
- **Device 2.0 (every kernel used):** **GREEN.** The op's own DM kernels use `Noc` (`noc.async_read` / `async_read_barrier` / `async_write`), `DataflowBuffer` (`reserve_back`/`push_back`/`wait_front`/`pop_front`), and `TensorAccessor`; the compute kernel operates entirely on `DataflowBuffer` objects. The only CB-index free function in the op's own kernels is `get_tile_size(cb_id)` (reader `:101-103`, writer `:29`) — **sanctioned** (not a holdover). No `InterleavedAddrGen`/`ShardedAddrGen`/raw `noc_async_*`/manual CB-index management anywhere. See the lib-team note under *Team-only* for one shared-helper observation (non-gating).

- **Feature compatibility:** all Appendix A entries N/A.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | not used |
  | CBDescriptor `address_offset` (non-zero) | N/A | all CBs are plain tile CBs, no `address_offset` |
  | GlobalSemaphore | N/A | no semaphores of any kind |
  | Variable-count compile-time arguments (CTA varargs) | N/A | reader's only CTAs are `input_grad_rank` + three `TensorAccessorArgs`; the rank-bounded loops read **RTAs**, not CTAs; no `std::vector<Tensor>` in `tensor_args_t` |

- **CB endpoints (GATE-free):** all CBs resolve without a flag. Per-CB census (single config — the factory has no sharding/config branches; core-group split does not change endpoint roles):
  - `c_0` input, `c_1` output, `c_2` output_grad, `c_3` decimal — **plain 1:1**: reader produces (`reserve_back`/`push_back`, and `fill_cb_with_value` for `c_3`), compute consumes (`wait_front`/`pop_front`). Legal.
  - `c_16` input_grad (dx) — **plain 1:1**: compute produces (`mul_tiles_to_cb` → `dfb_dx_obj`), writer consumes. Legal.
  - `c_24`–`c_31` (8 intermediates: xpow/logx/exp_lxmd/correct_xpow/tmp4/tmp5/recip_ypow/sign) — **single toucher** (compute only, both fills and drains) → **self-loop** (bind the compute kernel PRODUCER and CONSUMER; legal on Gen1 for compute). No dead CBs (all 8 are used; several inside the `power_*`/`sign_*` helpers).
- **Offset base pointers:** **GREEN.** Every tensor base reaches the kernel as a clean `Buffer*` (`input.buffer()`, `output.buffer()`, `output_grad.buffer()`, `input_grad.buffer()`) with no host-side offset arithmetic. No Type 1/2 fold; no `narrow`/`address_offset`. All four hand off to TensorParameter analysis as clean bases.
- **TensorAccessor 3rd argument:** **GREEN.** Every `TensorAccessor` construction is 2-arg (`TensorAccessor(args, addr)`): reader `:86,89,92`, writer `:25`. No page-size override present, so nothing to classify or drop.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding) — all **Case 1** (fed into a `TensorAccessor`; no raw-pointer arithmetic):
  - `input` — reader `Buffer*` RTA → `TensorAccessor(input_args, input_addr)` (reader `:86`).
  - `output` — reader `Buffer*` RTA → `TensorAccessor(output_args, output_addr)` (reader `:89`).
  - `output_grad` — reader `Buffer*` RTA → `TensorAccessor(output_grad_args, output_grad_addr)` (reader `:92`).
  - `input_grad` — writer `Buffer*` RTA → `TensorAccessor(input_grad_args, input_grad_addr)` (writer `:25`).
  - Port action: express each as a `TensorParameter`/`TensorBinding`; kernel builds `TensorAccessor(tensor::name)`; the `Buffer*` RTA + `TensorAccessorArgs` CTAs both disappear.
- **TensorParameter relaxation:** none (no custom hash).
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** self-loop `c_24`–`c_31` (compute-only intermediates); all I/O CBs (`c_0`–`c_3`, `c_16`) are plain 1:1 — no assignment needed.

## Heads-ups  *(mirrors the brief)*

- **RTA varargs:** `reader_moreh_norm_backward.cpp` reads three consecutive **count-bounded vararg blocks** — `output_grad_dim` (`:52-55`), `input_grad_dim` (`:57-60`), `need_bcast_dim` (`:62-65`), each looped `for i < input_grad_rank`. `input_grad_rank` is a **CTA** (`:36`), but per the recipe a CTA-bounded loop still varies across instantiations → these must port as **RTA varargs**, not named args. The six leading reader scalars (`input_addr`, `output_addr`, `output_grad_addr`, `decimal`, `num_output_tiles`, `start_id`, `:43-50`) are fixed distinct fields → name them. The vararg blocks are at the tail (no trailing nameable scalars to rescue). Compute (5 RTAs `:16-20`) and writer (3 RTAs `:17-19`) are all fixed named scalars — ordinary port work.
- **Cross-op / shared kernels:** no borrowed kernel files (all three kernels are op-owned). Shared-header coupling only — see *Team-only*.

## Team-only

- **Out-of-directory coupling & donor shape:**
  - **Op-level roll-up:** ✓ clean. No function-call escape needs donor-side work; no file-path kernel instantiation from outside the op.
  - **Includes / donors:**
    | Op kernel | Donor include | Class | Status |
    |---|---|---|---|
    | reader | `ttnn/kernel/dataflow/moreh_common.hpp` | shared kernel pool (`ttnn/cpp/ttnn/kernel/`, class 3) | ✓ |
    | reader/writer | `api/dataflow/noc.h`, `api/dataflow/dataflow_buffer.h`, `api/tensor/noc_traits.h`, `api/dataflow/dataflow_api.h` | `tt_metal/*` HAL (class 1) | ✓ no concern |
    | compute | `ttnn/kernel/compute/moreh_common.hpp` | shared kernel pool (class 3) | ✓ |
    | compute | `api/dataflow/dataflow_buffer.h` | HAL (class 1) | ✓ |
  - **Per-call detail:** reader calls `fill_cb_with_value(DataflowBuffer, uint32_t, int32_t)` — Device 2.0 native (operates on a `DataflowBuffer` object). Compute calls `sign_tile_to_cb`, `power_tile_with_abs_x_to_cb`, `power_and_recip_tile_to_cb`, `mul_tiles_to_cb`, and the `mul_tiles*_with_dt` / `pack_tile_with_dt` init family — all take `DataflowBuffer` objects; no addr-gen, no `Semaphore`/`uint32_t sem`, no `CircularBuffer&` in the consumed signatures. All ✓.
  - **Port-together set:** the `moreh_common.hpp` headers (dataflow + compute) are shared across the whole `moreh` family. Their Metal 2.0 rewrite is a single shared change — but they are already `DataflowBuffer`-based, so no gate; noted so a planner sequences any shared-header token rewrite as one unit across moreh ops.
  - **Device 2.0 lib-team observation (non-gating):** `fill_cb_with_value` (`ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp:98-109`) does `get_dataformat(cb.get_id())` — a CB-index-keyed **format-metadata** free function, not on the explicitly-sanctioned short-list (`get_tile_size`, `get_local_cb_interface`) but of the same category (a metadata lookup the Metal 2.0 port moves onto the `DataflowBuffer` object per kernel-side whitelist rule 7, not a Device 2.0 boundary). Treated as non-gating. If the Device 2.0 team considers format free functions in-scope, this one line in lib-owned shared code is the only site; the op's own kernels have none.
- **Relaxation candidates:** none (no custom hash to mine).
- **TTNN factory analysis:** current concept `descriptor`; no op-owned tensors; no MeshWorkload; no pybind `create_descriptor`; no custom hash; no custom `override_runtime_arguments`. Target concept `MetalV2FactoryConcept`. (All confirmed from code; sheet confirmation pending per the caveat.)

## Misc anomalies  *(team-only, non-gating)*

- **`decimal` RTA is a float bit-cast** — `reader_rt_args.push_back(*reinterpret_cast<uint32_t*>(&decimal))` (`moreh_norm_backward_program_factory.cpp:258`); consumed on-device by `fill_cb_with_value(dfb_decimal, decimal)` (reader `:95`). This is a scalar value, not an address — not a TensorParameter/offset signal. Noted only because the raw-pointer-cast shape can read as an address at a glance; it is not.
- **`decimal_minus_one` / `p_minus_one_is_negative` split** — the factory computes `get_floored_p_and_decimal_and_p_is_negative(p - 1.0f)` (`:97-98`) but only `floored_p_minus_one` and `p_minus_one_is_negative` reach RTAs; `decimal_minus_one` is unused (the reader fills a single `decimal` CB used for both the `x^(p-1)` and `1/y^p` paths). Harmless dead local, not fed to any hash. Route to ops team if they want it cleaned.

## Questions for the user

1. **Readiness-sheet confirmation (the one open item):** the "Operations analysis" sheet could not be fetched in this subagent session (Google Drive connector authorizes only in the main interactive session; the download was classifier-blocked, and no local CSV exists). Please fetch it in the main session per `ttnn_op_porting_readiness.md` and confirm `moreh/moreh_norm_backward`'s row reads `Is able to port? == yes` and `Is safe to port? == yes`. The code cross-check clears all four shape conjuncts and shows no smuggled pointer, so the sheet is expected to agree — but the `Is safe to port?` expert axis is not derivable from code and should be confirmed before the port begins.

## Recipe notes

- **Subagent cannot fetch the readiness sheet.** The recipe mandates fetching the sheet "every run" and forbids delegating the fetch to a subagent (OAuth wall), but the audit itself is frequently launched *as* a subagent by an orchestrator (as here). The result is a hard, unavoidable conflict: the one authoritative gate input is unreachable from the very context the audit runs in. The recipe would benefit from an explicit fallback path for the subagent case — e.g. "record the code-level cross-check of all four shape conjuncts + the smuggled-pointer signal, mark the `safe` axis pending, and route sheet confirmation to the human," which is what this report did. Today the recipe only anticipates *conflict* or *missing-row* (both → "sheet broken → gate"), not *cannot-fetch-at-all*, leaving the auditor to improvise the disposition.
