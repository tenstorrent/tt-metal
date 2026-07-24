# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/moreh/moreh_dot`

Single device operation in this directory:

- **`MorehDotOperation`**
  - `MorehDotOperation (single-descriptor)` — factory `create_descriptor` in `device/moreh_dot_program_factory.cpp`; device-op scaffolding (`validate` / `compute_output_specs` / `create_output_tensors`) in `device/moreh_dot_device_operation.cpp`; declared in `device/moreh_dot_device_operation.hpp`.

Kernels (all owned by this op, all referenced by the factory):
- `device/kernels/reader_moreh_dot.cpp`
- `device/kernels/writer_moreh_dot.cpp`
- `device/kernels/moreh_dot.cpp` (compute)

No unreferenced kernel files. `moreh_dot_backward` is a **separate op** in its own directory (`moreh/moreh_dot_backward`) — out of scope here.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/moreh/moreh_dot` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `MorehDotOperation` → `MorehDotOperation (single-descriptor)` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — own kernels + `kernel_lib` donors all Device 2.0 |
| *Prereqs* — Cross-op escapes | Ok — `kernel_lib` shared helpers only (lib-team owned); no borrowed kernel files |
| *Feature Support* — overall | **GREEN** |
| *Feature Support* — Variadic-CTA | Ok (N/A) |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Is safe to port? | Yes |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — Runtime-args update | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` |
| *Port work* — Offset base pointer | none (clean bases) |
| *Port work* — Tensor bindings (per binding) | `input_a` Case 1 · `input_b` Case 1 · `output` Case 1 |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none (no 3rd arg present) |
| *Port work* — CB endpoints | 1:1 legal (`c_0`,`c_1`,`c_2`,`c_16`) · self-loop (`c_24`,`c_25`) |

## Result

**GREEN → brief issued.** Every gate clears: the op is a vanilla single-core `descriptor` factory (`Is able to port? == yes`, cross-check clean), all kernels (own + `kernel_lib` donors) are Device 2.0, no Appendix A feature is used, no offset-folded base pointer, no `TensorAccessor` 3rd-arg site. Port work is mechanical: bind three tensors (all Case 1 via `TensorAccessor`), self-loop two compute-internal intermediate CBs.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** Readiness sheet row `moreh/moreh_dot` / `MorehDotOperation (single-descriptor)`: `Concept = descriptor`, `Custom hash = no`, `Runtime-args update = no`, `Override runtime args method? = no`, `Pybind descriptor = no`, `Smuggled pointer = no`, `Is safe to port? = yes`, **`Is able to port? = yes`**. Cross-check against code:
  - `Concept = descriptor` — confirmed: `MorehDotOperation::create_descriptor(...)` returns a `ProgramDescriptor` (`device/moreh_dot_program_factory.cpp:22`).
  - `Custom hash = no` — confirmed: no `compute_program_hash` override in `device/moreh_dot_device_operation.hpp` / `.cpp`.
  - `Runtime-args update = no` — confirmed: no `get_dynamic_runtime_args` / `override_runtime_arguments` in the factory.
  - `Pybind descriptor = no` — confirmed: `moreh_dot_nanobind.cpp:20` binds a plain function (`ttnn::bind_function<"moreh_dot">(... &ttnn::moreh_dot ...)`), no `create_descriptor` / `nb::class_` of the device op.
  - `Op-owned tensors` blank — consistent with `descriptor` (cross-column invariant holds).
  - Sheet's `Factory definition path` / `Declared in` match the on-disk files.
- **Device 2.0 (every kernel used):** **GREEN.** All three own kernels use Device 2.0 idioms end-to-end — `Noc`, `DataflowBuffer`, `TensorAccessor`, `CoreLocalMem<uint16_t>`, `noc.async_read`/`noc.async_write`, `dfb.reserve_back`/`push_back`/`wait_front`/`pop_front`/`get_write_ptr()` (wrapper method). `get_tile_size(cb_id)` (reader `:87-88`, writer `:25`) is the **sanctioned** CB-index free function — not a violation. Donor `kernel_lib` helpers called from the kernels (`reduce_helpers_dataflow.hpp` + `.inl`, `reduce_helpers_compute.hpp`, and transitively `dfb_helpers_dataflow.hpp`, `l1_helpers.hpp`) are Device 2.0 (`DataflowBuffer`, `Noc`, `get_tile_size`, `addr_to_l1_ptr`; no `InterleavedAddrGen`/`ShardedAddrGen`, no raw `noc_async_*`, no CB-index free-function pointer holdovers). No violations to route.
- **Feature compatibility (Appendix A):**

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No global CB — all six CBs are plain `CBDescriptor` on a single core range. |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `CBDescriptor` sets `address_offset`; no borrowed-memory CB. |
  | GlobalSemaphore | N/A | Op uses no semaphores at all. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | CTAs are fixed: `TensorAccessorArgs` appends + one `cb_id` (writer). No `get_compile_time_arg_val(i)` loop; `tensor_args_t` carries fixed `Tensor` fields, no `std::vector<Tensor>`. |

  Clean scan — all `N/A`.
- **CB endpoints (GATE-free):** single core, node `{0,0}`. `c_0` (in0): reader FIFO-produces, compute FIFO-consumes → **1:1 legal**. `c_1` (in1): same → **1:1 legal**. `c_2` (scaler): reader produces via `calculate_and_prepare_reduce_scaler` (`reserve_back`/`push_back`), compute consumes (reduce helper waits; `dfb_c2.pop_front` at `moreh_dot.cpp:72`) → **1:1 legal**. `c_16` (out): compute produces (reduce output, last block), writer consumes → **1:1 legal**. `c_24` (im0): touched **only** by compute (`reserve_back`/`push_back` then read back as reduce input) → **self-loop**. `c_25` (im1): touched **only** by compute (reduce accumulation CB `Accumulate::at(c_25)` + non-last-block reduce output) → **self-loop**. Nothing gates.
- **Offset base pointers:** **GREEN.** The only address-bearing RTAs are `Buffer*` bindings — reader `emplace_runtime_args(core, {src0_buffer, src1_buffer, ...})` (`moreh_dot_program_factory.cpp:127`) and writer `{dst_buffer, ...}` (`:139`). Each delivers a clean base; the kernel reads it as `get_arg_val<uint32_t>(0/1)` and feeds it straight into a `TensorAccessor` constructor — no host-side `base + offset` fold. Not listed in the offset-base-pointer triage doc (`2026-07-19_offset_base_pointers.md`) — consistent. All bases hand off to TensorParameter analysis as clean.
- **TensorAccessor 3rd argument:** **GREEN.** Every `TensorAccessor` constructor uses **two** args only — reader `TensorAccessor(src0_args, src0_addr)` / `(src1_args, src1_addr)` (`reader_moreh_dot.cpp:80,82`), writer `TensorAccessor(dst_args, dst_addr)` (`writer_moreh_dot.cpp:21`). No explicit page-size 3rd arg anywhere. `moreh_dot` is not in the 3rd-arg triage doc (`2026-07-06_..._triage.md`) — only `moreh_fold` / `moreh_getitem` are, both other ops. Nothing to classify or drop.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding):
  - `input_a` (`src0_buffer`) — **Case 1**: `Buffer*` RTA (reader RTA idx 0), consumed as `TensorAccessor(src0_args, src0_addr)`.
  - `input_b` (`src1_buffer`) — **Case 1**: `Buffer*` RTA (reader RTA idx 1), consumed as `TensorAccessor(src1_args, src1_addr)`.
  - `output` (`dst_buffer`) — **Case 1**: `Buffer*` RTA (writer RTA idx 0), consumed as `TensorAccessor(dst_args, dst_addr)`.
  All Case 1: express as `TensorParameter` / `TensorBinding`; kernels build `TensorAccessor(tensor::name)`; the `Buffer*` RTA + `TensorAccessorArgs(...).append_to(...)` CTA plumbing both disappear.
- **TensorParameter relaxation:** none (sheet: `none`).
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** self-loop `c_24` (all configs) and `c_25` (all configs) — bind the compute kernel PRODUCER **and** CONSUMER. `c_0`/`c_1`/`c_2`/`c_16` are plain 1P+1C, no action.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none — no hidden second writer, no multi-reader. (The self-loops on `c_24`/`c_25` are in Port-work.)
- **Cross-op / shared kernels:** the op calls `ttnn/cpp/ttnn/kernel_lib` shared helpers (`reduce_helpers_dataflow` from the reader, `reduce_helpers_compute` from compute). Their Metal 2.0 CB→DFB rewrite is a `kernel_lib`-team concern, not this op's; the call shapes cross cleanly (see Team-only). No file-path instantiation of foreign kernel `.cpp`s — all three kernels are op-owned.
- **RTA varargs:** none — every kernel reads a fixed set of distinct RTA fields (reader idx 0–5, writer idx 0–2, compute idx 0). No counted/data-selected `get_arg_val` loop.

## Team-only

- **Out-of-directory coupling & donor shape:**
  - **Op-level roll-up:** `✓ clean`. Only `kernel_lib` (official shared library, class 2) function-call escapes; no cross-family donors; no borrowed kernel `.cpp` files.
  - **Summary table:**

    | Op kernel | Donor file | Class | Status |
    |---|---|---|---|
    | `reader_moreh_dot.cpp` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp` | shared-lib (`kernel_lib`) | ✓ |
    | `moreh_dot.cpp` (compute) | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` | shared-lib (`kernel_lib`) | ✓ |

  - **Per-call detail:** `calculate_and_prepare_reduce_scaler<cb_id_in2, PoolType::SUM, ReduceDim::REDUCE_ROW>()` (reader `:76`) takes the CB as a **template NTTP `uint32_t dfb_id`** → `uint32_t cb_id` shape, ✓ OK (`dfb::name`'s constexpr cast covers template-parameter position). `compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM, c_24, c_2, c_16/c_25, ...>(...)` (compute `:45,:57`) passes CB indices as template params → same ✓ OK shape. No `Semaphore`, no addr-gen, no `CircularBuffer&` in any donor signature.
  - **Borrowed kernel files (file-path instantiation):** none — the factory `CreateKernel`s only op-owned `.cpp` files.
  - *(Host-side only, not a kernel escape:* `device/moreh_dot_device_operation.cpp:9` includes `ttnn/operations/moreh/moreh_helper_functions.hpp` for `is_1d_tensor` in `validate` — no bearing on the kernel port.)*

## Misc anomalies  *(team-only, non-gating)*

- **Dead compute RTA.** `moreh_dot_program_factory.cpp:153` passes `KernelDescriptor::CoreRuntimeArgs{num_tiles, 1u}` to the compute kernel, but `device/kernels/moreh_dot.cpp:13` reads only `get_arg_val<uint32_t>(0)` (`per_core_block_cnt`). The second arg (`1u`, index 1) is never read — dead RTA. Not porter-actionable; routes to the ops team. (The port names `per_core_block_cnt` and simply won't carry the dead arg.)

## Recipe notes  *(none)*
