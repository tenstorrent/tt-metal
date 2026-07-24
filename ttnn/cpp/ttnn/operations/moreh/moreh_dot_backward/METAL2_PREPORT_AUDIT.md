# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/moreh/moreh_dot_backward`

Single device operation, single program factory (a `create_descriptor` defined directly on the device-op):

- **`MorehDotBackwardOperation`**
  - `MorehDotBackwardOperation (single-descriptor)` — `create_descriptor` lives in `device/moreh_dot_backward_program_factory.cpp`; declared in `device/moreh_dot_backward_device_operation.hpp`.

Kernels (all op-owned, all referenced by the factory):
- `device/kernels/reader_moreh_dot_backward.cpp`
- `device/kernels/writer_moreh_dot_backward.cpp`
- `device/kernels/moreh_dot_backward.cpp` (compute)

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/moreh/moreh_dot_backward` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `MorehDotBackwardOperation` → `MorehDotBackwardOperation (single-descriptor)` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — reader/writer/compute all on Device 2.0 idioms (`Noc`, `DataflowBuffer`, `TensorAccessor`; `get_tile_size(cb_id)` is sanctioned) |
| *Prereqs* — Cross-op escapes | Ok — no out-of-directory kernel includes; all `#include`s are `tt_metal/*` HAL/LLK |
| *Feature Support* — overall | **GREEN** (all Appendix A entries N/A) |
| *Feature Support* — Variadic-CTA | Ok — CTAs fixed-count; kernels read `TensorAccessorArgs<N>()` at constexpr offsets |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Is safe to port? | Yes |
| *TTNN Readiness* — Custom hash | No (no `compute_program_hash` override) |
| *TTNN Readiness* — Runtime-args update | No (no `get_dynamic_runtime_args` / `override_runtime_arguments`) |
| *TTNN Readiness* — Pybind `create_descriptor` | No (nanobind binds the plain `ttnn::moreh_dot_backward` function) |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` |
| *Port work* — Offset base pointer | none (cleared) |
| *Port work* — Tensor bindings (per binding) | 5 × Case 1 (all via `TensorAccessor`) |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none present (no accessor passes a 3rd arg) |
| *Port work* — CB endpoints | all legal (5 CBs, single core, each 1P+1C) |
| **Tensorless-dispatch check** | **PASS** — `tensor_args_t` carries three mandatory `const Tensor&` inputs (`output_grad`, `input`, `other`), always present at dispatch; MeshDevice is always sourceable. Only the *outputs* are optional. |

## Result

**GREEN → brief issued.** Every gate clears: Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓. The op is on the `descriptor` concept and ports to `MetalV2FactoryConcept`. All five tensor bindings are Case 1 (routine `TensorAccessor` port work). No portable-subset scoping needed — the whole op is clear.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** GREEN. Readiness sheet row `moreh/moreh_dot_backward, MorehDotBackwardOperation, MorehDotBackwardOperation (single-descriptor)`: `Concept=descriptor`, `Custom hash=no`, `Runtime-args update=no`, `Override runtime args method?=no`, `Pybind descriptor=no`, `Smuggled pointer=no`, `Is safe to port?=yes`, `Is able to port?=yes`, `TensorParameter relaxation=none`, `Op-owned tensors?` blank, `Secretly SPMD` blank. Cross-check against code agrees on every cheaply-checkable column (concept from `create_descriptor` return type; no `compute_program_hash`; no `get_dynamic_runtime_args`/`override_runtime_arguments`; nanobind binds a plain function, not `create_descriptor`). No conflict — sheet trusted.
- **Device 2.0 (every kernel used):** GREEN.
  - Reader (`reader_moreh_dot_backward.cpp`): `Noc noc`, `noc.async_read(s0, dfb, ...)`, `DataflowBuffer dfb_in*`, `TensorAccessor`, `get_tile_size(cb_id)` (sanctioned). No Device 1.0 idioms.
  - Writer (`writer_moreh_dot_backward.cpp`): same Device 2.0 surface (`Noc`, `noc.async_write`, `DataflowBuffer`, `TensorAccessor`, `get_tile_size`).
  - Compute (`moreh_dot_backward.cpp`): `DataflowBuffer` wrappers + `init_bcast` / `mul_tiles_bcast` / `pack_tile` / `tile_regs_*`. No holdovers.
  - No `noc_async_read`, no `InterleavedAddrGen`/`ShardedAddrGen`, no raw sem addresses, no CB-index free-function holdovers (`get_read_ptr(cb_id)` etc.) anywhere.
- **Feature compatibility:** every Appendix A entry scanned. All N/A.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no `experimental::GlobalCircularBuffer`, no `.global_circular_buffer` field, no remote-CB idiom |
  | CBDescriptor `address_offset` (non-zero) | N/A | 5 `CBDescriptor`s, none set `address_offset` (default 0) |
  | GlobalSemaphore | N/A | op uses no semaphores at all |
  | Variable-count compile-time arguments (CTA varargs) | N/A | fixed-count CTAs; reader = 3 `TensorAccessorArgs`, writer = 2 CB indices + 2 `TensorAccessorArgs`; kernels read via constexpr `TensorAccessorArgs<N>()` / `get_compile_time_arg_val(0/1)` — no runtime-varying CTA index |

- **CB endpoints (GATE-free):** all 5 CBs legal — every CB has exactly one producer and one consumer on the single core `{0,0}`. No self-loop, multi-binding, or dead CB.

  | CB | Producer | Consumer | Verdict |
  |---|---|---|---|
  | `c_0` (in0, output_grad scalar) | reader `reserve_back`/`push_back` | compute `wait_front` (persistent, no pop) | 1P+1C |
  | `c_1` (in1, input) | reader `reserve_back`/`push_back` | compute `wait_front`/`pop_front` | 1P+1C |
  | `c_2` (in2, other) | reader `reserve_back`/`push_back` | compute `wait_front`/`pop_front` | 1P+1C |
  | `c_16` (out0, input_grad) | compute `push_back` | writer `wait_front`/`pop_front` | 1P+1C |
  | `c_17` (out1, other_grad) | compute `push_back` | writer `wait_front`/`pop_front` | 1P+1C |

  Config note: `has_input_grad` / `has_other_grad` gate whether a CB is exercised at all, but never change the producer/consumer *identity* — each CB stays 1P+1C in every config.
- **Offset base pointers:** GREEN. No address RTA folds a host offset into a base. The factory delivers tensor bases via the `Buffer*`-binding form — it pushes raw `Buffer*` pointers (`src0_buffer`/`src1_buffer`/`src2_buffer` and `dst0_buffer`/`dst1_buffer`) into the RTA lists, never `buffer()->address() + <offset>`. No `+ offset` arithmetic on any address. Op is absent from the offset triage doc (`2026-07-19_offset_base_pointers.md`) — consistent with a clean scan.
- **TensorAccessor 3rd argument:** GREEN. Every `TensorAccessor` construction is 2-arg (`TensorAccessor(src0_args, src0_addr)`, etc.) in both reader and writer. No page-size override anywhere. Op is absent from the 3rd-arg triage doc (`2026-07-06_tensor_accessor_3rd_arg_triage.md`).

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding, all Case 1 — via `TensorAccessor`):
  - `output_grad` → reader CB `c_0` accessor `s0` (`reader_moreh_dot_backward.cpp:28`). Delivered today as `Buffer* src0_buffer` in the reader RTA list (`..._program_factory.cpp:114`).
  - `input` → reader accessor `s1` (`reader_moreh_dot_backward.cpp:29`). RTA `Buffer* src1_buffer` (`..._program_factory.cpp:115`).
  - `other` → reader accessor `s2` (`reader_moreh_dot_backward.cpp:30`). RTA `Buffer* src2_buffer` (`..._program_factory.cpp:116`).
  - `input_grad` (optional output) → writer accessor `s0` (`writer_moreh_dot_backward.cpp:26`). RTA `Buffer* dst0_buffer` when present, `0u` when absent (`..._program_factory.cpp:154-158`).
  - `other_grad` (optional output) → writer accessor `s1` (`writer_moreh_dot_backward.cpp:28`). RTA `Buffer* dst1_buffer` when present, `0u` when absent (`..._program_factory.cpp:159-163`).
  - All five feed the base address into a `TensorAccessor` and do all memory access through it → **Case 1** (express each as a `TensorParameter`/`TensorBinding`; kernel builds `TensorAccessor(tensor::name)`; the `Buffer*` RTA + `TensorAccessorArgs(...).append_to(...)` CTA plumbing both disappear). The current `Buffer*`-binding form is patched by the framework on cache hits (correct-today), so this is routine port work, not a stale-pointer hazard.
- **TensorParameter relaxation:** none.
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** all legal — no dispositions to apply.

## Heads-ups  *(mirrors the brief)*

- **Optional-output conditional bindings.** `input_grad` (`c_16`) and `other_grad` (`c_17`) are optional (`std::optional<Tensor>`); when absent the factory pushes `0u` for the address and `TensorAccessorArgs(nullptr)`, and the kernels guard every access on `has_input_grad`/`has_other_grad` RTAs. The port must express these two output `TensorParameter`s as **conditionally bound** (bind when present; the kernel's `has_*_grad` guard stays). Both outputs can be simultaneously absent (op then does nothing) — inputs remain bound, so dispatch/MeshDevice sourcing is unaffected.
- **CB endpoints (multi-binding):** none.
- **Cross-op / shared kernels:** none — all three kernels are op-owned and file-path-instantiated from within the op directory; no borrowed kernel files, no shared-pool includes.
- **RTA varargs:** none. Reader RTAs (`has_input_grad`, `has_other_grad`, `src0_addr`, `src1_addr`, `src2_addr`, `num_tiles`, `start_id`), writer RTAs (`has_input_grad`, `has_other_grad`, `dst0_addr`, `dst1_addr`, `num_tiles`, `start_id`), and compute RTAs (`has_input_grad`, `has_other_grad`, `per_core_block_cnt`) are all fixed distinct named fields — port each as a named RTA.

## Team-only

- **Out-of-directory coupling & donor shape:** clean. Every kernel `#include` resolves to `tt_metal/*` HAL/LLK (`api/dataflow/dataflow_api.h`, `api/dataflow/noc.h`, `api/dataflow/dataflow_buffer.h`, `api/tensor/noc_traits.h`, `api/compute/bcast.h`) — donor class 1, no concern. No function-call escapes, no file-path kernel borrows. The host-side device-op includes `ttnn/operations/moreh/moreh_helper_functions.hpp`, but that is host code, not a kernel donor.
- **Relaxation candidates:** none (no custom hash to mine).
- **TTNN factory analysis:** current concept `descriptor`; target `MetalV2FactoryConcept`; no op-owned tensors; no MeshWorkload need; no pybind `create_descriptor`; no custom hash; no custom `override_runtime_arguments`. All gate conjuncts confirmed absent in code.

## Misc anomalies  *(team-only, non-gating)*

- **`start_id` RTA is always `0`.** Both reader (`..._program_factory.cpp:119`, arg index 6) and writer (`:165`, arg index 5) emplace `0u` for `start_id`; the kernels loop `start_id .. start_id+num_tiles`. Since this op runs on a single core with no work split, `start_id` is dead-but-harmless plumbing (always 0). Not porter-actionable; note for the ops team if a future cleanup pass wants to drop it.
- **Duplicated output plumbing / stale comment.** `tensor_args_t` carries a `std::vector<std::optional<Tensor>> output_tensors` alongside the historical comment at `moreh_dot_backward_device_operation.hpp:27` ("thanhnguyen's mistake"). `create_output_tensors` returns that vector verbatim, and `create_descriptor` reads outputs from `tensor_return_value` — consistent, but the vestigial-looking comment may confuse a reader. Cosmetic only.

## Recipe notes

- The op keeps `create_descriptor` in a file named `*_program_factory.cpp` but defines it as a static method on the `DeviceOperation` (no separate ProgramFactory struct). This is the "single-descriptor" shape the readiness sheet's `Factory (variant)` column already names, so no ambiguity — recorded here only because the audit's identifying-section examples assume a named factory class.
