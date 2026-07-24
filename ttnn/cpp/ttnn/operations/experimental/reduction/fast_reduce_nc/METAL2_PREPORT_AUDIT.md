# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/reduction/fast_reduce_nc`

Single device operation, single program factory:

- **`FastReduceNCDeviceOperation`** (`ttnn::experimental::prim`)
  - `FastReduceNCProgramFactory` (`device/fast_reduce_nc_program_factory.cpp`)

Kernels referenced by the factory (all owned in-directory):
- `device/kernels/reader_reduce_nc.cpp` (reader / DM)
- `device/kernels/writer_reduce_nc.cpp` (writer / DM)
- `device/kernels/reduce_nc.cpp` (compute)

Donor include: `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp` (`dataflow_kernel_lib::prepare_zero_tile`), consumed by the reader.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `metal2_audit.md`.

**Recipe docs:** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/reduction/fast_reduce_nc` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `FastReduceNCDeviceOperation` → `FastReduceNCProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | Yes — reader/writer/compute + `l1_helpers.hpp` donor all Device 2.0 native |
| *Prereqs* — Cross-op escapes | Ok — one `kernel_lib` donor (`uint32_t cb_id` NTTP → ✓ OK); no borrowed kernel files |
| *Feature Support* — overall | GREEN |
| *Feature Support* — Variadic-CTA | Ok (N/A) |
| *TTNN Readiness* — `Is able to port?` (the gate) | Yes |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Is safe to port? | Yes |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — Runtime-args update | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` |
| *Port work* — Offset base pointer | none |
| *Port work* — Tensor bindings (per binding) | `input` Case 1 · `output` Case 1 |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none (no site passes a 3rd arg) |
| *Port work* — CB endpoints | c_0 / c_1 / c_16 legal 1:1 · **c_24 dead-CB drop** |

## Result

**GREEN → brief issued.** Every gate clears: Device 2.0 ✓, Feature compatibility ✓, TTNN factory concept ✓ (`Is able to port? = yes`), Offset base pointers ✓, TensorAccessor 3rd arg ✓. The op is a clean single-factory `descriptor` op targeting `MetalV2FactoryConcept`. Port work is light: two Case-1 tensor bindings and one confirmed dead-CB drop (`c_24`).

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** Readiness sheet row (`experimental/reduction/fast_reduce_nc`, `FastReduceNCDeviceOperation`, `FastReduceNCProgramFactory`) reads `Is able to port? = yes`. Cross-check against code — all consistent:
  - `Concept = descriptor` ✓ — `FastReduceNCProgramFactory::create_descriptor(...)` returns a `tt::tt_metal::ProgramDescriptor` (`device/fast_reduce_nc_program_factory.cpp:54`).
  - `Custom hash = no` ✓ — no `compute_program_hash` override on `FastReduceNCDeviceOperation` (`device/fast_reduce_nc_device_operation.hpp:16`, `.cpp` defines only `validate_on_program_cache_miss` / `compute_output_specs` / `create_output_tensors`).
  - `Runtime-args update = no` ✓ — no `get_dynamic_runtime_args` / `override_runtime_arguments`; args are set once in `create_descriptor`.
  - `Pybind descriptor = no` ✓ — `fast_reduce_nc_nanobind.cpp:34` binds a *function* (`bind_function<"fast_reduce_nc">`), not a `create_descriptor` / device-op class.
  - `Smuggled pointer = no`, `Is safe to port? = yes` — consistent: the factory delivers `Buffer*` handles (not raw `->address()` values) as RTAs (see Tensor bindings below).
  - `Op-owned tensors` blank → none (correct for a `descriptor` concept).
  No cross-column invariant violated. Target concept: `MetalV2FactoryConcept` (descriptor + no op-owned tensors).

- **Device 2.0 (every kernel used):** **GREEN.** All three op-owned kernels use Device 2.0 idioms end to end:
  - `reader_reduce_nc.cpp` — `Noc noc`, `CircularBuffer cb_in0_obj`, `noc.async_read(...)`, `TensorAccessor`. Free functions used are the **sanctioned** `get_tile_size(cb_id)` (`:40`) and standard `get_compile_time_arg_val` / `get_arg_val` — no holdovers.
  - `writer_reduce_nc.cpp` — `Noc`, `CircularBuffer cb_out_obj`, `noc.async_write(...)`, `TensorAccessor`, sanctioned `get_tile_size(cb_id)` (`:28`).
  - `reduce_nc.cpp` — `CircularBuffer` wrappers, `add_tiles` / `pack_tile` compute LLK. No raw idioms.
  - Donor `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp` (`prepare_zero_tile`) — `DataflowBuffer`, `Noc`, wrapper `dfb.get_tile_size()`, `noc.async_write_zeros`. Device 2.0 native.

  No `Noc`/`CircularBuffer`-object-in-scope + CB-index-keyed free-function holdovers found. No broad Device 1.0 idioms (no `InterleavedAddrGen` / raw `noc_async_read` / raw sem addresses).

- **Feature compatibility:** clean — every Appendix A entry is **N/A** (feature absent).

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | Only plain `CBDescriptor`/`CBFormatDescriptor` used; no `.global_circular_buffer` field, no `remote_*`/`CreateGlobalCircularBuffer`/`.remote_index`. |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `.address_offset` set on any of the 4 `CBDescriptor`s; no `set_address_offset` / 4-arg `UpdateDynamicCircularBufferAddress` / `cb_descriptor_from_sharded_tensor`. |
  | GlobalSemaphore | N/A | Op uses **no** semaphores at all (no `Semaphore`, no `GlobalSemaphore`). |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t` is a fixed `{input, preallocated_output}` (no `std::vector<Tensor>`). Kernels read CTAs only at **constexpr** indices (reader 0–2, writer 0–1, compute 0–2); no runtime-varying CTA index. Multi-dim reduce is handled host-side by looping `ttnn::prim::fast_reduce_nc` per dim (`fast_reduce_nc.cpp:40-52`), not by variadic CTAs. |

- **CB endpoints (GATE-free):** per-CB census below. Kernel CB indices are hardcoded constants (not config-dependent), so the census is config-independent for this op.

  | CB | Producers | Consumers | Verdict | Disposition |
  |---|---|---|---|---|
  | `c_0` (in0) | reader `reserve_back`/`push_back` (`reader_reduce_nc.cpp:66,80`) | compute `wait_front`/`pop_front` (`reduce_nc.cpp:41,45`) | 2 touchers: 1 locked producer + 1 locked consumer | plain **1:1 legal** |
  | `c_1` (in1/zero) | reader via `prepare_zero_tile<c_1>` → `reserve_back`/`push_back` (`l1_helpers.hpp:60,62`, called `reader_reduce_nc.cpp:38`) | compute `wait_front`/`pop_front` (`reduce_nc.cpp:30,57`) | 2 touchers: 1 producer + 1 consumer | plain **1:1 legal** |
  | `c_24` (intermed0) | — | — | **0 touchers — dead CB** | **dead-CB drop** |
  | `c_16` (out0) | compute `reserve_back`/`push_back` (`reduce_nc.cpp:48,53`) | writer `wait_front`/`pop_front` (`writer_reduce_nc.cpp:40,44`) | 2 touchers: 1 producer + 1 consumer | plain **1:1 legal** |

  **`c_24` dead-CB — confirmed drop.** The factory allocates `CBIndex::c_24` (`fast_reduce_nc_program_factory.cpp:177-185`, commented "accumulated sum"), but **no kernel references index 24**: grep of all three kernels finds zero occurrences, and the index is not threaded through any CTA (the reader/writer/compute CTA lists carry only `input_granularity`, `shard_factor`, `num_cores_to_be_used`, and per-group tile counts). The compute kernel accumulates in DST registers (`tile_regs_acquire` → `add_tiles` into `dst0` → `pack_tile` straight to `c_16`) and uses no intermediate CB. A dead DFB cannot be expressed in Metal 2.0 (no producer/consumer binding), so the porter **must drop the `CBDescriptor` at `:177-185`**; removing it changes only L1 footprint (zero behavior). Confidence high — single hardcoded-index config path, fully grepped.

- **Offset base pointers:** **GREEN — no fold.** The factory delivers memory objects as `Buffer*` handles, not `base + offset` addresses:
  - reader RTA position 0 = `input_buffer` (`= tensor_args.input.buffer()`, a `Buffer*`; `fast_reduce_nc_program_factory.cpp:310,327`).
  - writer RTA position 0 = `output_buffer` (`= tensor_return_value.buffer()`; `:311,337`).
  Kernel-side these arrive as clean `uint32_t` bases (`input_addr`, `output_addr`) fed directly to `TensorAccessor(tensor_args, input_addr)` / `TensorAccessor(tensor_args, output_addr)` — no host-side `->address() + offset` arithmetic, no interior/base+offset pointer. Not in the offset-base-pointer triage tables, and the scan confirms clean. No `ttnn::narrow` (Type 4) either.

- **TensorAccessor 3rd argument:** **GREEN — no site.** Both accessors are 2-arg: `TensorAccessor(tensor_args, input_addr)` (`reader_reduce_nc.cpp:42`) and `TensorAccessor(tensor_args, output_addr)` (`writer_reduce_nc.cpp:30`). No explicit page-size argument passed anywhere; nothing to drop or gate.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding):
  - `input` — **Case 1** (via `TensorAccessor`). Delivered as a `Buffer*` RTA (position 0 of the reader; framework `BufferBinding`, correct-on-cache-hit today), consumed as the base of `TensorAccessor(tensor_args, input_addr)` and accessed only through the accessor (`noc.async_read(tensor_accessor, ...)`). Port: express as a `TensorParameter` / `TensorBinding`; kernel builds `TensorAccessor(tensor::name)`; the RTA and the `TensorAccessorArgs(*input.buffer()).append_to(reader_compile_time_args)` plumbing (`fast_reduce_nc_program_factory.cpp:201`) disappear.
  - `output` — **Case 1** (via `TensorAccessor`). Same shape on the writer: `Buffer*` RTA position 0 → `TensorAccessor(tensor_args, output_addr)`, accessed only via `noc.async_write(cb_out_obj, tensor_accessor, ...)`. Port: bind as `TensorParameter`; kernel builds `TensorAccessor(tensor::name)`; RTA + `TensorAccessorArgs(*tensor_return_value.buffer()).append_to(writer_compile_time_args)` (`:204`) disappear.
  - Compute kernel (`reduce_nc.cpp`) — out of scope: consumes/produces CBs only, touches no tensor memory.
- **TensorParameter relaxation:** none (sheet: `TensorParameter relaxation = none`).
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** `c_0`, `c_1`, `c_16` all legal 1:1 (bind producer + consumer as-is). **Drop dead CB `c_24`** @ `fast_reduce_nc_program_factory.cpp:177-185`.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none — no hidden second writer (only `reserve_back`/`push_back` FIFO fills, no raw `get_write_ptr` co-fills), no multi-reader CB.
- **Cross-op / shared kernels:** the reader `#include`s `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp` (official shared `kernel_lib`). The consumed function `prepare_zero_tile<uint32_t dfb_id>()` takes the CB id as a `uint32_t` NTTP → shape `uint32_t cb_id` (✓ OK: `dfb::name`'s constexpr cast covers template-parameter position). No borrowed kernel *files* — the op instantiates only its own three kernels.
- **RTA varargs:** none — every RTA is read a fixed number of times at a distinct constexpr index (reader positions 0–6, writer 0–2). No loop-indexed / `arg_index++`-in-loop reads, no data-selected element.

## Team-only

- **Out-of-directory coupling & donor shape:**
  - **Op-level roll-up:** ✓ clean / workable. One donor (`kernel_lib`), classified ✓ OK; no ⭐/✗ blockers.

  | Op kernel | Donor file | Class | Shape | Status |
  |---|---|---|---|---|
  | `reader_reduce_nc.cpp` | `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp` | 2 — official shared `kernel_lib` | `prepare_zero_tile<uint32_t dfb_id>()` — `uint32_t cb_id` NTTP | ✓ OK |

  Other includes (`api/dataflow/dataflow_api.h`, `api/dataflow/noc.h`, `api/dataflow/circular_buffer.h`, `api/tensor/noc_traits.h`, and via the donor `api/dataflow/dataflow_buffer.h`, `api/dataflow/endpoints.h`, `api/core_local_mem.h`) resolve under `tt_metal/*` HAL/LLK — class 1, no concern. **Borrowed kernel files:** none (all three `.cpp` kernels are owned in-directory). `l1_helpers.hpp` is broadly shared across ops as a `kernel_lib` header; its Metal 2.0 rewrite (if any) is the lib team's, and the `uint32_t cb_id` shape ports without donor changes.
- **Relaxation candidates:** none mined (no custom hash on this op).
- **TTNN factory analysis:** concept `descriptor`; no op-owned tensors; no MeshWorkload; no pybind `create_descriptor`; no custom hash; no custom `override_runtime_arguments`; not secretly-SPMD (N/A — not a WorkloadDescriptor). Target concept `MetalV2FactoryConcept`.

## Misc anomalies  *(team-only, non-gating)*

- **Dead CB `c_24`** — `fast_reduce_nc_program_factory.cpp:177-185` allocates a 1-tile "accumulated sum" intermediate CB that no kernel touches (see CB endpoints). L1 waste; the port drops it. (Recorded as a port-work dead-CB drop above; noted here as the latent code issue that produced it.)
- **Unused `onetile` constants** — `reader_reduce_nc.cpp:32` declares `constexpr uint32_t onetile = 1;` that the reader never uses (the writer/compute uses of `onetile` are live). Harmless dead local; team-only, not porter work.

## Recipe notes

- The op delivers tensors via the **`Buffer*`-binding form** (`emplace_runtime_args({input_buffer, ...})` where `input_buffer` is a `Buffer*`), which the audit's TensorParameter-analysis "Detection — host side" section covers explicitly and routes to Case 1/2 by kernel use. This matched the readiness sheet's `Smuggled pointer = no` (the `Buffer*` form is the framework's sanctioned interim binding, not a raw `->address()` smuggle). The recipe handled this cleanly; no friction — noting only that this op is a clean worked example of the `Buffer*`-form → Case-1 path.
