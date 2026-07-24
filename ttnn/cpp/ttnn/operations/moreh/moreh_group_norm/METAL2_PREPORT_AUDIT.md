# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/moreh/moreh_group_norm`

Single device operation, single program factory (a `create_descriptor` op — the `descriptor` concept). One internal cache-miss branch (`use_large_algorithm`) selects the small vs. large reader + compute kernels; both branches are the same op / same concept, not separate factory variants.

- **`MorehGroupNormOperation`**
  - `MorehGroupNormOperation (single-descriptor)` — `device/moreh_group_norm_program_factory.cpp` (`create_descriptor`)

**Kernels exercised** (audited per the follow-kernel-references rule):
- Local dataflow: `device/kernels/dataflow/reader_moreh_group_norm_small.cpp`, `reader_moreh_group_norm_large.cpp`, `writer_moreh_group_norm.cpp`
- Borrowed compute (in-family, from `moreh_layer_norm`): `moreh_layer_norm/device/kernels/moreh_layer_norm_small_kernel.cpp`, `moreh_layer_norm_large_kernel.cpp`
- Shared headers: `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp`, `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp`, `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp`

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/moreh/moreh_group_norm` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `MorehGroupNormOperation` → `MorehGroupNormOperation (single-descriptor)` |
| *Prereqs* — Device 2.0 (every kernel used) | Yes — reader/writer + borrowed layer_norm compute + shared headers all Device 2.0 |
| *Prereqs* — Cross-op escapes | Ok (in-family borrow + shared-lib; all Device 2.0) |
| *Feature Support* — overall | GREEN (all Appendix A entries N/A) |
| *Feature Support* — Variadic-CTA | Ok (all CTAs fixed-index; no runtime-count CTA loop; no `std::vector<Tensor>`) |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** (readiness sheet, row 377) |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Is safe to port? | Yes |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — Runtime-args update | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No (nanobind binds the plain `&ttnn::moreh_group_norm` function) |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` |
| *Port work* — Offset base pointer | none |
| *Port work* — Tensor bindings (per binding) | all Case 1 (`TensorAccessor`): input, gamma, beta, output, mean, rstd |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none (all sites 2-arg) |
| *Port work* — CB endpoints | 1:1 legal (I/O + params) · self-loop (intermediates c_24–c_31) |

**Tensorless-dispatch check (orchestrator-requested):** **Not a block.** `tensor_args_t::input` is a mandatory `const Tensor&` (`device/moreh_group_norm_device_operation.hpp:24`) — always present at dispatch. The MetalV2 factory adapter can source the `MeshDevice` from `input`; no tensorless / optional-only-output dispatch path exists. All other tensors (gamma, beta, output, mean, rstd) are optional, but `input` is unconditional.

## Result

**GREEN → brief issued.** Every gate clears: Device 2.0 ✓, Feature compatibility ✓, TTNN factory concept (`Is able to port? = yes`) ✓, Offset base pointers ✓, TensorAccessor 3rd arg ✓. Target concept `MetalV2FactoryConcept`. Port work is entirely mechanical: six Case-1 tensor bindings, self-loop the eight compute-internal intermediate CBs, nothing to relax or drop. `METAL2_PORT_BRIEF.md` issued.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** Readiness sheet row 377 (`moreh/moreh_group_norm`, `MorehGroupNormOperation (single-descriptor)`): `Is able to port? = yes`. Derivation columns all clear — `Concept = descriptor`, `Custom hash = no`, `Runtime-args update = no`, `Override runtime args method = no`, `Pybind descriptor = no`, `Smuggled pointer = no`, `Is safe to port? = yes`. Cross-check against code agrees on every cheaply-checkable column:
  - `Concept` — `create_descriptor()` returns a `ProgramDescriptor` (`device/moreh_group_norm_device_operation.hpp:35`, factory `:52`) → `descriptor`. ✓
  - `Custom hash` — no `compute_program_hash` override anywhere in the op (grep clean). ✓
  - `Runtime-args update` — no `get_dynamic_runtime_args` / `override_runtime_arguments` (grep clean). ✓
  - `Pybind descriptor` — `moreh_group_norm_nanobind.cpp:19` binds the plain host function `&ttnn::moreh_group_norm`, not a `create_descriptor`. ✓
  - `Op-owned tensors` — none (a `descriptor` op cannot carry them; `Op-owned tensors?` cell empty). ✓
  - Cross-column invariants hold (no `Runtime-args update` on a non-descriptor; no op-owned tensors on `descriptor`).
- **Device 2.0 (every kernel used):** **GREEN.** All kernels are structurally Device 2.0:
  - Local reader/writer use `Noc`, `DataflowBuffer` objects, `TensorAccessor`, `CoreLocalMem<T>`; sync via `dfb.reserve_back/push_back/wait_front/pop_front`; `dfb.get_write_ptr()/get_read_ptr()` methods (not free functions). The only CB-index free function is `get_tile_size(cb_id)` — **sanctioned** (kept by Device 2.0).
  - Borrowed compute kernels use `DataflowBuffer` objects for all FIFO sync and `_with_dt` inits / `pack_tile_with_dt(dst, dfb_obj)`; the tile-math primitives (`copy_tile`/`add_tiles`/`mul_tiles`/`mask_tile`/`reduce<...>`) take CB indices as is normal for the compute engine (not a DM idiom).
  - Shared headers (`moreh_common.hpp` dataflow + compute, `reduce_helpers_compute.hpp`) take `DataflowBuffer` and CB-index template params; no `InterleavedAddrGen` / `ShardedAddrGen` / raw `noc_async_read/write` / free-function `get_*_ptr(cb_id)` (grep clean).

- **Feature compatibility:** every Appendix A entry absent.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no `GlobalCircularBuffer` / `set_globally_allocated_address` |
  | CBDescriptor `address_offset` (non-zero) | N/A | all CBs created at base; no `address_offset` |
  | GlobalSemaphore | N/A | op declares no semaphores at all |
  | Variable-count compile-time arguments (CTA varargs) | N/A | all `get_compile_time_arg_val` are fixed indices (0–10); no runtime-count CTA loop; `tensor_args_t` carries no `std::vector<Tensor>` |

- **CB endpoints (GATE-free):** classified per CB, per node. No multi-binding, no dead CB.
  - **1:1 legal** (one locked producer + one locked consumer): `c_0` input (reader → compute), `c_1` scaler (reader → compute), `c_2` eps (reader → compute), `c_3` gamma (reader → compute), `c_4` beta (reader → compute), `c_5` mask_h (reader → compute), `c_6` mask_w (reader → compute), `c_16` output (compute → writer), `c_17` mean (compute → writer), `c_18` rstd (compute → writer). Reader's `dfb_gamma/beta.get_write_ptr()` and writer's `dfb_output/mean/rstd.get_read_ptr()` are raw peeks on the kernel's *own* producer/consumer binding — one toucher on that side, not a second.
  - **Self-loop** (single toucher — compute-only intermediates): `c_24` E[x], `c_25` x−E[x], `c_26` (x−E[x])², `c_27` Sum[(x−E[x])²], `c_28` Var[x], `c_29` 1/√(Var+eps), `c_30` gamma_beta, `c_31` Sum[x]. Bind the compute kernel PRODUCER and CONSUMER.
  - **Config-dependence:** the factory only pushes a CB when its tile count > 0 (`push_cb_if_nonzero`). So `c_3`/`c_4` exist only when gamma/beta present; `c_5`/`c_6` only when `do_mask_h`/`do_mask_w`; `c_30` only when gamma or beta present; `c_17`/`c_18` only when mean/rstd required. Disposition of each *present* CB is as above in every config.
- **Offset base pointers:** **GREEN.** No address RTA folds a host-side offset into its base. The factory passes tensor buffers as `Buffer*` (not `->address()` and not `base + offset`) via `emplace_runtime_args` (`program_factory.cpp:338-356`); `tile_offset` is a separate scalar arg, added to page indices on-device inside `TensorAccessor` page addressing — never folded into a base pointer. Op is not catalogued in `2026-07-19_offset_base_pointers.md` (no fold introduced since; my scan governs and is clean).
- **TensorAccessor 3rd argument:** **GREEN — N/A.** Every `TensorAccessor(...)` site is 2-arg `(args, addr)` (reader `:78/:82/:86`, writer `:37/:41/:45`); none passes an explicit page-size 3rd argument. Op absent from `2026-07-06_tensor_accessor_3rd_arg_triage.md`; scan confirms no site fires.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding) — all **Case 1** (`TensorAccessor`). Host delivers each as a `Buffer*` binding (framework `BufferBinding`, patched on cache hit — correct-today, not the stale-pointer hazard); the kernel feeds the received base into a `TensorAccessor`:
  - `input` — reader `TensorAccessor(input_args, input_addr)` → Case 1.
  - `gamma` — reader `TensorAccessor(gamma_args, gamma_addr)` → Case 1.
  - `beta` — reader `TensorAccessor(beta_args, beta_addr)` → Case 1.
  - `output` — writer `TensorAccessor(output_args, output_addr)` → Case 1.
  - `mean` — writer `TensorAccessor(mean_args, mean_addr)` → Case 1.
  - `rstd` — writer `TensorAccessor(rstd_args, rstd_addr)` → Case 1.
  Express each as a `TensorParameter` / `TensorBinding`; kernel builds `TensorAccessor(tensor::name)`; the `Buffer*` RTA + the `TensorAccessorArgs(...).append_to(...)` CTA plumbing both disappear.
- **TensorParameter relaxation:** none (`TensorParameter relaxation = none` on the sheet).
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** self-loop `c_24–c_31` (compute-only intermediates) · all I/O + param CBs are legal 1:1.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none — no hidden second writer, no multi-reader. The reader's gamma/beta raw writes and the writer's output/mean/rstd raw reads are peeks on their own bindings.
- **Cross-op / shared kernels:** the compute kernels are borrowed from **`moreh_layer_norm`** (in-family) and are **co-instantiated by `moreh_layer_norm` itself** → **port-together set `{moreh_group_norm, moreh_layer_norm}`** for `moreh_layer_norm_small_kernel.cpp` / `moreh_layer_norm_large_kernel.cpp`. Their Metal 2.0 (CB→DFB named-token) rewrite is one change both ops must adopt together. Shared headers `moreh_common.hpp` (`ttnn/cpp/ttnn/kernel/`) and `reduce_helpers_compute.hpp` (`ttnn/cpp/ttnn/kernel_lib/`) are shared-pool / lib-team owned.
- **RTA varargs:** none — reader and writer each read a fixed run of distinct fields via a top-of-kernel `i++` counter (nameable, non-signal); no loop-indexed or data-selected reads.

## Team-only

- **Out-of-directory coupling & donor shape:**
  - **Roll-up: ✓ clean.** All donor consumption is via `DataflowBuffer`-shaped helpers (Device 2.0 native) or shared-lib compute helpers.
  - **Function-call escapes:**
    | Op kernel | Donor include | Class | Shape / status |
    |---|---|---|---|
    | reader/writer | `ttnn/kernel/dataflow/moreh_common.hpp` | shared pool (`ttnn/cpp/ttnn/kernel/`) | helpers take `DataflowBuffer` → ✓ excellent |
    | compute | `ttnn/kernel/compute/moreh_common.hpp` | shared pool | `DataflowBuffer` / CB-index → ✓ |
    | compute | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` | kernel_lib | lib-team owned; CB-index template params → ✓ |
    | all | `api/dataflow/noc.h`, `dataflow_buffer.h`, `core_local_mem.h`, `tensor/noc_traits.h` | `tt_metal/*` HAL | no concern |
  - **Borrowed kernel files (file-path instantiation):** `moreh_layer_norm/device/kernels/moreh_layer_norm_{small,large}_kernel.cpp` — owned by `moreh_layer_norm` (in-family); also instantiated by `moreh_layer_norm`'s own factory → port-together set `{moreh_group_norm, moreh_layer_norm}` (above). The reader/writer kernel files are owned by this op.
- **TTNN factory analysis:** current concept `descriptor`; no op-owned tensors; no pybind `create_descriptor`; no custom hash; no custom `override_runtime_arguments`; not secretly-SPMD (N/A on `descriptor`). Target concept `MetalV2FactoryConcept`. `Smuggled pointer = no` (the factory deliberately uses `Buffer*` bindings, documented in the comments at `program_factory.cpp:308-337`).

## Misc anomalies  *(team-only, non-gating)*

- **`mean_memory_config` / `rstd_memory_config` are accepted but never used to place the tensors.** `operation_attributes_t` carries `mean_memory_config` and `rstd_memory_config` (`device_operation.hpp:18-19`, populated in `moreh_group_norm(...)` `:171`), but `compute_output_specs` builds the mean/rstd `TensorSpec` from `operation_attributes.memory_config` (the *output*'s config), not from `mean_memory_config`/`rstd_memory_config` (`device_operation.cpp:96,106`). The two dedicated configs are dead for tensor placement. Not porter work; route to the ops team.
- **`reader_moreh_group_norm_small.cpp` double-reserves the input CB.** The small reader calls `dfb_input.reserve_back(num_inner_tiles)` at `:95` and again inside the inner loop at `:97` before a single `push_back(num_inner_tiles)` at `:107`. The outer reserve looks redundant/suspicious (the large reader has no such pattern). Behavior is preserved today; note for the ops team, not the port.

## Recipe notes

- The audit doc says "Do NOT fetch or cross-check in a subagent — the Drive connector authorizes only in the main session," and the readiness-doc fetch procedure repeats "Do not delegate the fetch to a subagent … a spawned subagent hits the OAuth wall." This audit ran in an agent context, yet the `mcp__claude_ai_Google_Drive__download_file_content` call **succeeded** (row 377 fetched and cross-checked). Either the OAuth wall no longer applies to this agent type, or the warning is narrower than stated. Flagging so the maintainer can reconcile the guidance with observed behavior; the gate verdict here rests on a real sheet fetch, not a code-only derivation.
