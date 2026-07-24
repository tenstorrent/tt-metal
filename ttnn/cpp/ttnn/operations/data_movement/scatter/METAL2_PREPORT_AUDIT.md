# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/scatter`

One device operation, two program factories, all sharing a common kernel set:

- **`ScatterDeviceOperation`** (`device/scatter_device_operation.hpp` / `.cpp`)
  - `ScatterProgramFactory` (`device/scatter_program_factory.cpp`) — the general path (all supported dtypes)
  - `ScatterReduceBfloat16ProgramFactory` (`device/scatter_reduce_bfloat16_program_factory.cpp`) — selected only when a reduction is requested **and** the input dtype is `BFLOAT16` (`scatter_device_operation.cpp:15-22`); reduces in an fp32 scratch buffer before converting back to bf16

Both factories are on the `ProgramDescriptor` API (`create_descriptor` returning a `ProgramDescriptor`) and share the same host and kernel structure. Findings below apply to **both factories** unless a factory is named. The two are audited together because they share the device operation, the kernel-side common headers, and the addressing model (shared-code → shared port).

Host entry points `scatter` / `scatter_add` (`scatter.cpp`) and `tosa_scatter` (`tosa_scatter.cpp`) are thin wrappers that reshape/transpose inputs and dispatch into the same `ScatterDeviceOperation`; they add no separate device operation.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `44da718b06b 2026-07-23 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/scatter` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `ScatterDeviceOperation` → `ScatterProgramFactory`, `ScatterReduceBfloat16ProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — kernels use `Noc` / `DataflowBuffer` / `TensorAccessor` / `CoreLocalMem` / `UnicastEndpoint` object wrappers; no Device 1.0 idioms |
| *Prereqs* — Cross-op escapes | Ok — kernels `#include` only `tt_metal` LLK (`api/dataflow/*`, `api/core_local_mem.h`, `api/numeric/bfloat16.h`) plus in-op headers; no borrowed kernels |
| *Feature Support* — overall | **GREEN** — every Appendix A entry N/A |
| *Feature Support* — Variadic-CTA | Ok — all CTAs read at constexpr offsets; fixed 3-input `tensor_args_t` |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** (both factories) |
| *TTNN Readiness* — Concept (current) | `descriptor` (both factories) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Is safe to port? | Yes (both factories) |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — Runtime-args update | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` |
| *Port work* — Offset base pointer | none — clean bases |
| *Port work* — Tensor bindings (per binding) | `input`, `index`, `src`, `output` — all **Case 1** (via `TensorAccessor`) |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none — all accessors are 2-arg |
| *Port work* — CB endpoints | self-loop (`INPUT`/`INDEX`/`SRC`, plus `FP32_TEMP` in the reduce factory) · legal 1:1 (`DST`) |

**CB endpoints** are dispositions, not gates. Every scatter CB has a port-time resolution: a **self-loop** for the CBs a single kernel both fills and drains, and a plain **1:1** for the output CB (reader produces, writer consumes). Recorded per `(CB, config)` below.

## Result

**GREEN → brief issued.** All five gates clear for both factories: Device 2.0 ✓, Feature compatibility ✓, TTNN factory concept ✓ (`Is able to port? = yes`), Offset base pointers ✓, TensorAccessor 3rd argument ✓. `METAL2_PORT_BRIEF.md` is written alongside this report.

The op is unusually far along on the kernel side: its data-movement kernels are already written against Device 2.0 object wrappers, so the Metal 2.0 port is primarily a host-side rewrite (spec/binding wiring) plus swapping the numeric CB/DFB indices and RTA-delivered base addresses for named `dfb::*` / `tensor::*` bindings.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN** for both factories. The readiness sheet reports `Is able to port? = yes` for `ScatterProgramFactory` and `ScatterReduceBfloat16ProgramFactory`, with `Concept = descriptor`, `Custom hash = no`, `Runtime-args update = no`, `Override runtime args = no`, `Pybind descriptor = no`, `Smuggled pointer = no`, `Is safe to port? = yes`. Cross-check against the code confirms every cheaply-checkable column:
  - `Concept = descriptor` — both factories expose `create_descriptor(...)` returning a `ProgramDescriptor` (`scatter_program_factory.hpp:15-16`, `scatter_reduce_bfloat16_program_factory.hpp:15-16`); no mesh-workload return, no legacy `create()`/`override_runtime_arguments()`.
  - `Custom hash = no` — no `compute_program_hash` override anywhere in the op directory (grep clean).
  - `Runtime-args update = no` — no `get_dynamic_runtime_args` / `override_runtime_arguments` (grep clean).
  - `Pybind descriptor = no` — `scatter_nanobind.cpp` / `tosa_scatter_nanobind.cpp` bind only the host functions via `ttnn::bind_function<...>`; no `create_descriptor` / `nb::class_` of the device op.
  - Cross-column invariants hold (no `Runtime-args update` on a non-`descriptor`; no op-owned tensors on a `descriptor`). Sheet is internally consistent and agrees with the code.
- **Device 2.0 (every kernel used):** **GREEN.** The op instantiates four kernels, all owned by the op directory:
  - `device/kernels/dataflow/reader_scatter.cpp`, `writer_scatter.cpp` (general factory)
  - `device/kernels/dataflow/reader_bf16_reduction_scatter.cpp`, `writer_bf16_reduction_scatter.cpp` (reduce factory)
  - shared helpers `device/kernels/common.hpp`, `scatter_common.hpp`, `scatter_bf16_reduction_common.hpp`

  All four use Device 2.0 object wrappers throughout: `Noc noc;` with `noc.async_read` / `noc.async_write` + `UnicastEndpoint` + `CoreLocalMem` (`common.hpp:122-173`), `DataflowBuffer` objects with `.reserve_back` / `.push_back` / `.wait_front` / `.pop_front` / `.get_read_ptr` / `.get_write_ptr` / `.get_dataformat`, and `TensorAccessor(args, base_addr)`. No Device 1.0 idioms are present: no raw `noc_async_read`/`noc_async_write`, no `InterleavedAddrGen`/`ShardedAddrGen`/`InterleavedPow2AddrGen*`, no `cb_reserve_back`/`get_write_ptr(cb_id)` free-function pointer access, no `get_semaphore`/`noc_semaphore*`, no cursor mutators (`evil_set_*`).

  The only CB-index-keyed free function in use is `get_dataformat(ctas.<dfb>)`, used in a **compile-time** template-argument position — `std_type_t<get_dataformat(ctas.input_dfb)>` (`reader_scatter.cpp:120-121`, `reader_bf16_reduction_scatter.cpp:135-136`, `writer_scatter.cpp:19`, `writer_bf16_reduction_scatter.cpp:19`). This is **not** a Device 2.0 holdover: the Device 2.0 `CircularBuffer` wrapper's own `get_dataformat()` forwards to this same free function (`tt_metal/hw/inc/api/dataflow/circular_buffer.h:115` — `return ::get_dataformat(cb_id_);`), the exact structural relationship by which the audit sanctions `get_tile_size(cb_id)`. Device 2.0 itself uses the free function, so it is sanctioned here. Moving this metadata lookup onto the `DataflowBuffer` object is a **Metal 2.0** port-time change (kernel-side whitelist rule 7), not a Device 2.0 boundary. (See Recipe notes — the audit's explicitly-sanctioned list names only two functions.)

  | File | Line | Call | Wrapper in scope |
  |---|---|---|---|
  | (none — no Device 2.0 violations) | | | |

- **Feature compatibility:** every Appendix A entry scanned; none fire.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer`, no `.global_circular_buffer` field on any `CBDescriptor`, no `remote_cb`/`remote_index`; CBs built via plain `CBDescriptor{...}` literals (`scatter_program_factory.cpp:117-125`) |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `address_offset` set; CBs are plain-allocated (no `set_globally_allocated_address`), so offset defaults to 0 |
  | GlobalSemaphore | N/A | Op uses no semaphores at all (no `Semaphore`, no `GlobalSemaphore`) |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t` is a fixed 3-tensor `ScatterInputs`; every `get_compile_time_arg_val(...)` is read at a constexpr offset, and the `TensorAccessorArgs<...>` offset chain is compile-time-fixed. The variable-count loop that exists is over **runtime** args (see RTA varargs), not compile-time args |

- **CB endpoints (GATE-free):** every CB carries a port-time disposition; nothing here blocks a Gen1 port. Device 2.0 gate is GREEN, so the census keys on intact Device 2.0 idioms. Classified per `(CB, config)`:

  **`ScatterProgramFactory`** (CBs allocated at `scatter_program_factory.cpp:128-131`):
  - `INPUT` (`c_0`) — single toucher: the **reader** both fills it (`load_to_dfb` → `reserve_back`/`push_back`, `common.hpp:131,145`) and drains it (`input_dfb.wait_front`/`get_read_ptr`/`pop_front`, `reader_scatter.cpp:153,66,204`). The writer never touches it. → **self-loop**.
  - `INDEX` (`c_2`) — single toucher (reader fills + drains, `reader_scatter.cpp:185,198`). → **self-loop**.
  - `SRC` (`c_1`) — single toucher (reader fills + drains, `reader_scatter.cpp:186,197`). → **self-loop**.
  - `DST` (`c_3`) — two touchers, one locked producer + one locked consumer: the **reader** produces (`output_dfb.reserve_back`/`get_write_ptr`/`push_back`, `reader_scatter.cpp:154,19,203`); the **writer** consumes (`write_to_output` → `wait_front`/`get_read_ptr`/`pop_front`, `common.hpp:158,160,172`). → **plain 1:1, legal — no action**.

  **`ScatterReduceBfloat16ProgramFactory`** (CBs allocated at `scatter_reduce_bfloat16_program_factory.cpp:132-136`):
  - `INPUT`, `INDEX`, `SRC` — single toucher (reader fills + drains). → **self-loop**.
  - `FP32_TEMP` (`c_4`) — single toucher: the reduce reader alone reserves/writes/pushes/waits/reads/pops it as an fp32 scratch (`reader_bf16_reduction_scatter.cpp:170,220-221,226`). → **self-loop**.
  - `DST` (`c_3`) — reader produces (`copy_fp32_temp_to_output` + `output_dfb.reserve_back`/`push_back`, `reader_bf16_reduction_scatter.cpp:222,227`), writer consumes (`writer_bf16_reduction_scatter.cpp` → `common.hpp` `write_to_output`). → **plain 1:1, legal**.

  No dead CB, no hidden second writer, no multi-reader, no multi-binding. Config does not flip any disposition (the op rejects sharded inputs — `scatter_device_operation.cpp:45-47` — so there is a single interleaved configuration).

- **Offset base pointers:** **GREEN — clean bases.** Scatter is not in the offset-base-pointer triage doc (`2026-07-19_offset_base_pointers.md`); classified here by scan. Every tensor base address delivered to a kernel is a bare `buffer()->address()` with no host-side fold: the reader RTAs push `input_buffer`/`index_buffer`/`src_buffer` as `Buffer*` (`scatter_program_factory.cpp:163-165`; reduce `:168-170`); the writer RTA pushes `output_buffer` (`:181`; reduce `:186`). Row/chunk offsets are applied **on the device, after** the accessor produces a base NoC address — `addr_gtor.get_noc_addr(stick_id) + offset_bytes` (`common.hpp:132,159`) — so the value bound as the accessor base is always the clean tensor base. No Type 1 (raw offset arg), no Type 2 (accessor-fed offset), no Type 3 (`address_offset`), no Type 4 (`narrow`).

- **TensorAccessor 3rd argument:** **GREEN — no site.** Scatter is not in the 3rd-arg triage doc (`2026-07-06_tensor_accessor_3rd_arg_triage.md`); scanned directly. Every `TensorAccessor` construction passes exactly two arguments (args + base address): `TensorAccessor(ctas.input_args, input_buffer_address)` and siblings (`reader_scatter.cpp:116-118`, `writer_scatter.cpp:17`, `reader_bf16_reduction_scatter.cpp:131-133`, `writer_bf16_reduction_scatter.cpp:17`). No explicit page-size override anywhere, so there is nothing to classify or drop.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding) — **all Case 1** (via `TensorAccessor`), for both factories:
  - `input` — reader binding. Delivered today as a `Buffer*` RTA (`scatter_program_factory.cpp:163`), read as `get_arg_val<uint32_t>(0)` and fed straight into `TensorAccessor(ctas.input_args, input_buffer_address)` (`reader_scatter.cpp:103,116`). Port: express as `TensorParameter`/`TensorBinding`; kernel builds `TensorAccessor(tensor::input)`.
  - `index` — reader binding, same shape (`:164`, `reader_scatter.cpp:104,117`) → Case 1.
  - `src` — reader binding, same shape (`:165`, `reader_scatter.cpp:105,118`) → Case 1.
  - `output` — writer binding: `Buffer*` RTA (`scatter_program_factory.cpp:181`), read as `get_arg_val<uint32_t>(0)` → `TensorAccessor(ctas.output_args, output_buffer_address)` (`writer_scatter.cpp:13,17`) → Case 1.

  All four are the framework's `Buffer*`-binding (`BufferBinding`) interim shape — correct-on-cache-hit today, not the silent-stale hazard — and all four feed a `TensorAccessor`, so each ports to a typed `TensorParameter` with `TensorAccessor(tensor::name)` on the kernel side. The `TensorAccessorArgs` appended to the compile-time args (`scatter_program_factory.cpp:95-98`; reduce `:98-101`) come from the bindings after the port and drop out of the CTA list.
- **TensorParameter relaxation:** none (sheet: `none`; op carries no custom hash).
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** self-loop `INPUT`/`INDEX`/`SRC` (both factories) and `FP32_TEMP` (reduce factory); 1:1 legal `DST` (both factories). Single config.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none — no hidden second writer, no multi-reader; nothing needs the multi-binding flag.
- **Cross-op / shared kernels:** none. All four kernels and their common headers are owned by the scatter directory; the only cross-directory `#include`s are `tt_metal` LLK/HAL headers (`api/dataflow/dataflow_api.h`, `api/dataflow/noc.h`, `api/dataflow/dataflow_buffer.h`, `api/dataflow/endpoints.h`, `api/core_local_mem.h`, `api/numeric/bfloat16.h` — donor class 1, no concern). No file-path kernel instantiation from a shared pool. No port-together coupling.
- **RTA varargs:** the reader kernels read **two** genuine runtime-arg vararg blocks — the per-dimension shape extents of the input and index tensors — via `make_shape_array_from_runtime_args<N>` (`common.hpp:111-119`), where `N = ctas.input_rank - 1`. Call sites: `reader_scatter.cpp:125-126` (offsets `9` and `9+N`) and `reader_bf16_reduction_scatter.cpp:140-141`. The host emits them with a matching count-`(rank-1)` loop (`scatter_program_factory.cpp:172-177`; reduce `:177-182`). `N` is compile-time-fixed per instantiation but varies with tensor rank across instantiations, so there is no stable per-argument name — this is the CTA-bounded-loop vararg case: port both blocks with the kernel-side vararg mechanism, not by naming each element. The nine leading scalar args (offsets 0–8: the three `Buffer*` bindings, `stick_offset`, `sticks_per_core`, three chunk sizes, `reduction`) precede the vararg blocks and are cleanly nameable — no trailing-scalar-after-varargs pathology.

## Team-only

- **Out-of-directory coupling & donor shape:** ✓ clean. No function-call escape into another op's helpers and no file-path kernel borrows. Summary: every kernel `#include` resolves either inside the op directory or under `tt_metal/hw/inc/api/` (LLK/HAL, donor class 1). Per-call detail omitted (all rolls ✓). Borrowed kernel files: none — the op owns all four of its kernels.
- **Relaxation candidates** (mined from a custom hash): N/A — the op has no custom hash; the sheet lists `TensorParameter relaxation = none`.
- **TTNN factory analysis:** both factories are current concept `descriptor`, target `MetalV2FactoryConcept`, no op-owned tensors, no MeshWorkload, no pybind `create_descriptor`, no custom hash, no custom `override_runtime_arguments`. No gate conjunct is present. The port's TTNN ProgramFactory wiring targets the plain `MetalV2FactoryConcept`.

## Misc anomalies  *(team-only, non-gating)*

- **Dead buffer-address compile-time args — latent stale-pointer hazard, defused only by non-use.** Both factories bake the four tensor base addresses into the head of the compile-time-args list — `input_buffer->address()`, `index_buffer->address()`, `src_buffer->address()`, `output_buffer->address()` (`scatter_program_factory.cpp:78-81`; reduce `:80-83`). The kernels receive them as `ctas.input_tensor_addr` / `index_tensor_addr` / `source_tensor_addr` / `output_tensor_addr` (`scatter_common.hpp:18-21`; `scatter_bf16_reduction_common.hpp:18-21`) but **never read any of them** — the accessors use the RTA-delivered `Buffer*` addresses instead. A buffer address embedded in a compile-time arg is baked into the compiled kernel binary; on a program-cache hit with new tensor storage of the same shape, that baked value is stale. It is harmless here **only** because the fields are dead — but a future edit that reads any `ctas.*_tensor_addr` would silently mis-address on cache hits, with no assertion to catch it. Recommend the ops team drop these four compile-time args. (The port removes them naturally: addresses become `tensor::*` bindings and the CTA-baked copies disappear.)
- **Other dead compile-time args.** Also declared-but-never-read: `output_stick_size` (`scatter_common.hpp:29`) and the three `input_stick_size_bytes` / `index_stick_size_bytes` / `source_stick_size_bytes` fields (`:30-32`) — the kernels use `output_stick_size_bytes` and the element-count stick sizes, but not these. Cruft fed into the compiled program; safe to remove.

## Recipe notes

- **Device 2.0 sanctioned-free-function list is not exhaustive, and the sanctioning principle had to carry the call.** The audit's Device 2.0 Green bullet names exactly two sanctioned CB-index free functions (`get_tile_size(cb_id)`, `get_local_cb_interface(cb_id)`) and states the underlying principle ("if Device 2.0 allows the free function, so do we"). Scatter uses `get_dataformat(cb_id)` in a compile-time template-argument position. It is not on the named list, so a literal reading would flag it as a holdover (isolated, cheap — but still a gate). I resolved it GREEN on the principle: the Device 2.0 `CircularBuffer::get_dataformat()` forwards to `::get_dataformat(cb_id)` (`circular_buffer.h:115`), the identical structural argument the recipe uses to justify `get_tile_size`. Consider either (a) adding `get_dataformat(cb_id)` to the sanctioned list, or (b) restating the rule as "any CB-index free function that a Device 2.0 wrapper method forwards to is sanctioned" so future auditors do not have to reconstruct the parallel each time. This is exactly the kind of compile-time metadata lookup the breadcrumb says a Metal 2.0 port moves onto the object (whitelist rule 7).
