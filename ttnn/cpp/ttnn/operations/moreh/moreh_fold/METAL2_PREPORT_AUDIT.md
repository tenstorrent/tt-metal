# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/moreh/moreh_fold`

Single device operation, single program factory:

- **`MorehFoldOperation`**
  - `MorehFoldOperation (single-descriptor)` (`device/fold_program_factory_rm.cpp`)
    - kernels: `device/kernels/reader_fold_rm.cpp`, `device/kernels/writer_fold_rm.cpp` (both op-owned, both referenced)

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/moreh/moreh_fold` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `MorehFoldOperation` → `MorehFoldOperation (single-descriptor)` (`fold_program_factory_rm.cpp`) |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — both own kernels are structurally Device 2.0 (`Noc`, `DataflowBuffer`, `TensorAccessor`, `CoreLocalMem`); no free-function holdovers |
| *Prereqs* — Cross-op escapes | Ok — kernels `#include` only `tt_metal` HAL/firmware headers; no donor coupling |
| *Feature Support* — overall | **GREEN** (all Appendix A entries N/A) |
| *Feature Support* — Variadic-CTA | Ok — fixed CTA count; no runtime-varying `get_compile_time_arg_val` loop |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** (readiness sheet; cross-check clean) |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Is safe to port? | Yes |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — Runtime-args update | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No (nanobind binds the `moreh_fold` free function only) |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` |
| *Port work* — Offset base pointer | none (clean bases; `Buffer*`-binding-form RTAs) |
| *Port work* — Tensor bindings (per binding) | `input` → Case 1 · `output` → Case 1 |
| *Port work* — TensorParameter relaxation | none (sheet); see 3rd-arg divergence note |
| *Port work* — TensorAccessor 3rd arg | **Class 2 drop** (both accessors) — diverges from dated triage doc (Class 1); non-gating either way |
| *Port work* — CB endpoints | self-loop (`c_0`, `c_1`) · legal 1:1 (`c_16`) |

**Tensorless-dispatch check (orchestrator ask):** **GREEN — not a blocker.** `tensor_args_t` carries `const Tensor& input` (required, non-optional) plus `const std::optional<Tensor>& output`. A required input tensor is always present at dispatch, so the MetalV2 factory adapter can source the `MeshDevice` from it. There is no tensorless / optional-only-output dispatch path. (`device/fold_device_operation.hpp:26-29`.)

## Result

**GREEN → brief issued.** Every gate clears: Device 2.0 ✓, Feature compatibility ✓, TTNN factory concept ✓ (`Is able to port? = yes`), Offset base pointers ✓, TensorAccessor 3rd argument ✓ (Class 2, non-gating). Single `descriptor` factory → target `MetalV2FactoryConcept`. Port work is routine: two Case-1 tensor bindings, a mechanical 3rd-arg drop on both accessors, two self-loop CBs, one legal 1:1 CB.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** Readiness sheet row `moreh/moreh_fold, MorehFoldOperation, MorehFoldOperation (single-descriptor)` reads `Is able to port? = yes`. Cross-check against code, all consistent:
  - `Concept = descriptor` — `create_descriptor()` returns `tt::tt_metal::ProgramDescriptor` (`fold_device_operation.hpp:34`, `fold_program_factory_rm.cpp:22`).
  - `Custom hash = no` — no `compute_program_hash` override in the op (grep clean).
  - `Runtime-args update = no` / `Override runtime args method = no` — no `get_dynamic_runtime_args` / `override_runtime_arguments` (grep clean).
  - `Pybind descriptor = no` — `fold_nanobind.cpp:19-32` binds the `ttnn::moreh_fold` free function via `ttnn::bind_function`; no `create_descriptor` / device-op `nb::class_`.
  - `Is safe to port? = yes`, `Smuggled pointer = no`.
- **Device 2.0 (every kernel used):** **GREEN.** Both kernels are structurally Device 2.0 across the board — `Noc noc`, `noc.async_read/async_write` + barriers, `DataflowBuffer` object (`reserve_back`/`push_back`/`wait_front`/`pop_front`/`get_write_ptr()`/`get_read_ptr()` all as methods on the object), `TensorAccessor` object, `CoreLocalMem<T>`, `UnicastEndpoint`. No raw `noc_async_read`, no `InterleavedAddrGen`/`ShardedAddrGen`, no CB-index free-function holdovers, no raw semaphore addresses. Files: `device/kernels/reader_fold_rm.cpp`, `device/kernels/writer_fold_rm.cpp`.
- **Feature compatibility:** every Appendix A entry is absent.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | plain `CBDescriptor`/`CBFormatDescriptor` only (`fold_program_factory_rm.cpp:89-123`); no `global_circular_buffer` field, no remote-CB idiom |
  | CBDescriptor `address_offset` (non-zero) | N/A | no `.address_offset` set on any `CBDescriptor` (default 0) |
  | GlobalSemaphore | N/A | op uses no semaphores at all |
  | Variable-count compile-time arguments (CTA varargs) | N/A | reader CTAs `{input_cb, output_cb, scratch_cb}` + `TensorAccessorArgs<3>`; writer CTAs `{output_cb}` + `TensorAccessorArgs<1>` — fixed count; `tensor_args_t` is a fixed 1-input (+ optional output) tuple, no `std::vector<Tensor>`; no runtime-varying `get_compile_time_arg_val(i)` loop |

- **CB endpoints (GATE-free):** classified per `(CB, config)`:
  - `input_cb` (`c_0`) — reader-only: `reserve_back`/`push_back`/`wait_front`/`pop_front`/`get_write_ptr`/`get_read_ptr` (`reader_fold_rm.cpp:52,90-123`). **1 toucher → self-loop.**
  - `scratch_cb` (`c_1`) — reader-only, raw peek (`scratch_dfb.get_write_ptr()` as `noc.async_read` dest + source addr; no FIFO ops) (`reader_fold_rm.cpp:54,101-116`). **1 toucher → self-loop.** *Config-dependent existence:* the CB is only allocated when `(src_is_dram && input_cb_page_size % dram_alignment != 0) || is_blackhole` (`fold_program_factory_rm.cpp:101`); disposition is self-loop whenever it exists.
  - `output_cb` (`c_16`) — reader is a locked producer (`reserve_back`/`push_back`, `reader_fold_rm.cpp:57,149`); writer is a locked consumer (`wait_front`/`pop_front`, `writer_fold_rm.cpp:30,33`). **2 touchers, 1 locked producer + 1 locked consumer → plain 1:1, legal.**
- **Offset base pointers:** **GREEN — no offset fold.** The reader RTA passes `input.buffer()` (the `Buffer*` itself, `Buffer*`-binding form) as arg 0 and the writer passes `output.buffer()` (`fold_program_factory_rm.cpp:174,197`) — never `->address() + <offset>`. No host-folded interior address anywhere. `moreh_fold` is **not** listed in the offset-base-pointer triage (`analyses/2026-07-19_offset_base_pointers.md`); my own scan confirms clean. Both addresses are clean bases → handed to TensorParameter analysis.
- **TensorAccessor 3rd argument:** **GREEN — Class 2 (redundant/inert), clean mechanical drop on both accessors.**
  - Reader `TensorAccessor(input_args, input_addr, input_cb_page_size)` (`reader_fold_rm.cpp:49`) where `input_cb_page_size = unit_size * input.logical_shape()[-1]` (`fold_program_factory_rm.cpp:69`).
  - Writer `TensorAccessor(output_args, output_addr, output_cb_page_size)` (`writer_fold_rm.cpp:24`) where the RTA supplies `aligned_output_cb_page_size = round_up_to_mul32(unit_size * output.logical_shape()[-1])` (`fold_program_factory_rm.cpp:74,197`).
  - **Q1 — sharded or interleaved?** Interleaved (ROW_MAJOR input enforced at `fold_device_operation.cpp:17`; default interleaved memory config; accessor built from `TensorAccessorArgs(buffer)`). Interleaved silently realigns → only *magnitude* matters.
  - **Q2 — correct or wrong magnitude?** Both values are the true logical page (`element_size × last_dim` = `buffer->page_size()` for row-major). **Correct magnitude.**
  - **Divergence from the dated triage doc (`analyses/2026-07-06_tensor_accessor_3rd_arg_triage.md:62`), noted as the recipe permits.** The doc lists `moreh_fold` as **Class 1 — Dynamic page size** (set `dynamic_tensor_shape`). I classify **Class 2**: Class 1 requires the page size to *vary across cache-reused shapes*, which requires the hash to exclude the width — a relaxation/custom hash. `moreh_fold` has **no custom hash** and the readiness sheet lists **`TensorParameter relaxation = none`**, so the default hash puts the full shape in the cache key: every distinct width is a cache *miss*, the page size is *constant per compiled program*, and the compile-time `AlignedPageSize` can never be stale. This is exactly the recipe's Class-2 "width hashed / constant per program → drop, do NOT set `dynamic_tensor_shape`" carve-out (audit §TensorAccessor 3rd argument; triage doc lines 111, 156 — the same reasoning that reclassifies `topk_router_gpt` / `sdpa_decode`). **Either classification is PORT WORK, not a gate**, so this does not affect the GREEN verdict — but the porter should drop the arg *without* adding a relaxation. See Questions.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding):
  - `input` — **Case 1** (via `TensorAccessor`). Legacy delivers the base as `input.buffer()` in reader RTA[0] (`Buffer*`-binding form; framework auto-registers a `BufferBinding`, patched on cache hit today), consumed as `input_addr` → `TensorAccessor(input_args, input_addr, …)` (`reader_fold_rm.cpp:16,49`). Port: express as a `TensorParameter`/`TensorBinding`; kernel builds `TensorAccessor(tensor::name)`; the RTA base + `TensorAccessorArgs<3>` CT plumbing disappear.
  - `output` — **Case 1** (via `TensorAccessor`). Legacy delivers `output.buffer()` in writer RTA[0], consumed as `output_addr` → `TensorAccessor(output_args, output_addr, …)` (`writer_fold_rm.cpp:13,24`). Same port shape.
- **TensorParameter relaxation:** **none** (per the readiness sheet). Do not add `dynamic_tensor_shape` — see the 3rd-arg divergence above and Questions.
- **TensorAccessor 3rd arg:** drop the redundant page-size arg at `reader_fold_rm.cpp:49` and `writer_fold_rm.cpp:24` (Class 2, pure no-op; Metal 2.0 supplies `aligned_page_size` implicitly). Also drop the now-unused `input_cb_page_size`/`output_cb_page_size` RTAs feeding them once the base moves to a binding.
- **CB endpoints:** self-loop `c_0` (all configs) · self-loop `c_1` (only in configs where it is allocated: DRAM-unaligned page, or Blackhole) · `c_16` is legal 1:1 (no action).

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none — no hidden second writer, no multi-reader; the census is 1 self-loop + 1 self-loop + 1 legal 1:1.
- **Cross-op / shared kernels:** none — both kernels are op-owned and `#include` only `tt_metal` HAL/firmware headers.
- **RTA varargs:** none — reader reads args 0-20 and writer reads args 0-3 as distinct fields at constant indices (nameable). No variable-count loop, no data-selected index.

## Team-only

- **Out-of-directory coupling & donor shape:** **✓ clean.** Kernel `#include`s (`api/dataflow/dataflow_api.h`, `api/dataflow/noc.h`, `api/dataflow/dataflow_buffer.h`, `api/dataflow/endpoints.h`, `api/core_local_mem.h`, `api/tensor/noc_traits.h`) all resolve to `tt_metal/*` HAL/firmware (donor class 1, no concern). No function-call escapes into other op families. The program factory instantiates only its own two kernels by file path (`fold_program_factory_rm.cpp:17-20`); no borrowed kernel files, so no port-together set.
- **TTNN factory analysis (sheet-derived, with evidence):** current concept `descriptor`; no op-owned tensors; no MeshWorkload; no pybind `create_descriptor`; no custom hash; no custom `override_runtime_arguments`. Target concept `MetalV2FactoryConcept`. Sheet `Op Classification = PD (pointer-patching)` reflects the `Buffer*`-binding-form RTAs (the framework's interim pointer-patching; superseded by the Metal 2.0 typed binding) — not a smuggled-pointer bug (`Smuggled pointer = no`, `Is safe to port? = yes`).

## Misc anomalies  *(team-only, non-gating)*

- `reader_fold_rm.cpp:15` — `int i{0};` declared and never used (dead local). Route to ops team; the port does not act on it.
- `reader_fold_rm.cpp:47-48` and `writer_fold_rm.cpp:22-23` — comment claims the 3rd page-size arg "overrides TensorAccessorArgs::AlignedPageSize, which may be stale on program cache hits." Given the op has **no** custom hash / relaxation, the compiled program's `AlignedPageSize` cannot be stale (each shape is a distinct cache key), so the override is inert defensive code, not a correctness fix. Misleading comment; note for the ops team.
- `reader_fold_rm.cpp:79,82` — `if (lh < 0 || …)` / `if (lw < 0 || …)` on `uint32_t` values (`lh`, `lw` are `uint32_t`), so the `< 0` half is always false (dead sub-condition). Harmless (the upper-bound half still guards), but a latent sign-type smell. Note for the ops team.

## Questions for the user  *(for the readiness-sheet / triage-doc owners)*

1. **3rd-arg triage vs. readiness sheet for `moreh_fold`:** the dated 3rd-arg triage (`analyses/2026-07-06_tensor_accessor_3rd_arg_triage.md:62`) lists `moreh_fold` as **Class 1 (Dynamic page size → set `dynamic_tensor_shape`)`, but the authoritative readiness sheet lists `TensorParameter relaxation = none` and `Custom hash = no`. Since a `dynamic_tensor_shape` relaxation requires a custom hash that excludes the width, and adding one would introduce cache reuse the legacy op does not have (a behavior change vs. the port's no-functional-change contract), I classified the site **Class 2 (drop, no relaxation)** per the recipe's "width-hashed / constant-per-program" carve-out. Please confirm the porter should **drop the 3rd arg without adding `dynamic_tensor_shape`** (my recommendation), and consider updating the triage doc — `moreh_fold` looks like the same over-flag the doc already retired for `topk_router_gpt` / `sdpa_decode`.

## Recipe notes

- The 3rd-arg triage doc's Class-1 list (line 91) and its own "width-hashed → Class 2, do NOT set `dynamic_tensor_shape`" carve-out (lines 111, 156) are in tension for `moreh_fold`: it is a row-major interleaved op *without* a relaxation/custom hash, so its width is effectively hashed and it belongs in the Class-2 carve-out, yet it is listed under Class 1. The audit recipe's "triage is a dated prior; classify from the two questions" instruction resolved this cleanly — flagging here only so the triage-doc owner can reconcile the two lists.
