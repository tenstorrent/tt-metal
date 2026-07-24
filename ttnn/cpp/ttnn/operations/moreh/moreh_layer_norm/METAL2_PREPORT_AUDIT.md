# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm`

Single device operation, single program factory:

- **`MorehLayerNormOperation`**
  - `ProgramFactory` (`device/moreh_layer_norm_program_factory.cpp`) — one `create_descriptor` returning a `ProgramDescriptor`; branches internally between a *small* and a *large* algorithm (chosen by L1-fit), each selecting its own reader + compute kernel. Both algorithms share the same one writer kernel.

**Kernels referenced** (all owned in-directory, `device/kernels/`):
- Small algo: `reader_moreh_layer_norm_small.cpp`, `moreh_layer_norm_small_kernel.cpp`
- Large algo: `reader_moreh_layer_norm_large.cpp`, `moreh_layer_norm_large_kernel.cpp`
- Both: `writer_moreh_layer_norm.cpp`

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `MorehLayerNormOperation` → `ProgramFactory` (small + large algorithm branches) |
| *Prereqs* — Device 2.0 (every kernel used) | Yes |
| *Prereqs* — Cross-op escapes | Ok (shared moreh headers + kernel_lib, all Device 2.0 native) |
| *Feature Support* — overall | GREEN |
| *Feature Support* — Variadic-CTA | Ok |
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
| *Port work* — Tensor bindings (per binding) | all Case 1 (via `TensorAccessor`) |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none (no site passes a 3rd arg) |
| *Port work* — CB endpoints | 1P+1C on the I/O CBs · self-loop on the 8 intermediates |
| **Tensorless-dispatch check** | GREEN — mandatory `input` tensor in `tensor_args_t`; MeshDevice always sourceable |

## Result

**GREEN → brief issued.** Every gate clears: Device 2.0 ✓, Feature compatibility ✓, TTNN factory concept ✓ (`Is able to port? == yes`, cross-check clean), Offset base pointers ✓, TensorAccessor 3rd argument ✓. The op is a single `descriptor`-concept factory with no op-owned tensors → target `MetalV2FactoryConcept`. Port work is entirely routine: six tensor bindings, all Case 1; CB endpoints resolve by self-loop / 1P+1C with no multi-binding and no dead CBs.

**Tensorless-dispatch (framework block check):** GREEN. `tensor_args_t` carries `const Tensor& input` — a **mandatory, non-optional** input (`device/moreh_layer_norm_device_operation.hpp:23`). Every dispatch has at least this tensor, so the MetalV2 factory adapter can always source the `MeshDevice` from `tensor_args`. This op is **not** a tensorless / optional-only-output dispatch, so the adapter's `TT_FATAL`-on-no-tensor path cannot fire. Not a block.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** GREEN. Readiness sheet row `moreh/moreh_layer_norm , MorehLayerNormOperation , ProgramFactory`: `Is able to port? == yes`, `Concept == descriptor`, `Custom hash == no`, `Runtime-args update == no`, `Override runtime args method == no`, `Pybind descriptor == no`, `Smuggled pointer == no`, `Is safe to port? == yes`, `TensorParameter relaxation == none`, `Op-owned tensors? == (blank)`. Cross-check against code — all confirmed:
  - `Concept == descriptor`: `ProgramFactory::create_descriptor(...)` returns `tt::tt_metal::ProgramDescriptor` (`device/moreh_layer_norm_device_operation.hpp:36-40`, `device/moreh_layer_norm_program_factory.cpp:28`).
  - `Custom hash == no`: no `compute_program_hash` override anywhere in the op (grep clean).
  - `Runtime-args update == no`: no `get_dynamic_runtime_args` / `override_runtime_arguments` (grep clean).
  - `Pybind descriptor == no`: `moreh_layer_norm_nanobind.cpp` binds only the op function via `ttnn::bind_function`; no `create_descriptor` / `nb::class_` of the device op.
  - Op-owned tensors: `create_descriptor` returns a plain `ProgramDescriptor` (no `buffers` vector) → none. Consistent with the `descriptor` concept.
  - Cross-column invariants hold (Runtime-args update `no` on a `descriptor` concept; no op-owned tensors on a `descriptor` concept).
  - **Note (informational, not a conflict):** the two `moreh/moreh_layer_norm_backward` rows (rows 381/382) are a **separate op directory** and out of scope here.
- **Device 2.0 (every kernel used):** GREEN. All five referenced kernels are pervasively Device 2.0 native — `Noc noc; noc.async_read/async_write(...)`, `DataflowBuffer` objects with `reserve_back` / `push_back` / `wait_front` / `pop_front` / `get_read_ptr()` methods, `TensorAccessor(args, addr)`, `CoreLocalMem<...>`. No Device-1.0 idioms (`InterleavedAddrGen` / `ShardedAddrGen` / raw `noc_async_read` / manual CB-index management / raw sem addresses) in the op's kernels or its donor headers.
  - CB-index free functions present: `get_tile_size(cb_id)` (explicitly sanctioned) and `get_dataformat(cb_id)` (reader_small:32, reader_large:32). `get_dataformat(cb_id)` is a **tile/format-metadata lookup** in the same family the Metal 2.0 port relocates onto the `DataflowBuffer` object via **kernel-side whitelist rule 7** — `DataflowBuffer::get_dataformat()` exists (`tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:241`), exactly as for `get_tile_size` (line 167). Device 2.0's own `CircularBuffer` wrapper keeps `get_dataformat()` forwarding to the free function (`circular_buffer.h:115`), so it is sanctioned at this stage. **Not a Device 2.0 gate** — the porter handles it at port time (and in fact the value feeds a dead variable — see Misc anomalies). Flagged in Recipe notes because the sanctioned-list wording names only `get_tile_size` / `get_local_cb_interface`.
- **Feature compatibility:** every Appendix A entry scanned; none in use.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no `GlobalCircularBuffer` / global-cb usage |
  | CBDescriptor `address_offset` (non-zero) | N/A | all CBs pushed at base; no `address_offset` set (factory builds `CBDescriptor` with `total_size` / `format_descriptors` only, `program_factory.cpp:184`) |
  | GlobalSemaphore | N/A | no semaphores of any kind in this op |
  | Variable-count compile-time arguments (CTA varargs) | N/A | all CTAs are fixed-count; `TensorAccessorArgs<...>` compile-time offsets are statically chained, not runtime-counted |

- **CB endpoints (GATE-free):** all CBs either legal 1:1 or self-loop; no multi-binding, no dead CBs. Census per node (reader + writer over `all_cores`; the two compute `KernelDescriptor`s share the same source over **disjoint** core ranges `core_group_1` / `core_group_2`, so each node sees exactly **one** compute instance — the demoting-per-group shape, ordinary 1:1, not a two-toucher assignment):
  - **1P+1C (legal FIFO, 2 touchers):** `c_0` input (reader P → compute C), `c_1` scaler / `c_2` eps (reader fills via `fill_cb_with_value` reserve/push → compute C), `c_3` gamma / `c_4` beta (reader P → compute C, present-only), `c_5` mask_h / `c_6` mask_w (reader `generate_mask_*` P → compute C, present-only), `c_16` output (compute P → writer C), `c_17` mean / `c_18` rstd (compute P → writer C, present-only).
  - **Self-loop (1 toucher = compute, both produce & consume):** `c_24` E[x] (also reused as `cb_tmp`), `c_25` x-E[x], `c_26` (x-E[x])², `c_27` Sum[(x-E[x])²], `c_28` Var[x], `c_29` 1/sqrt(Var+eps), `c_30` gamma·+beta, `c_31` Sum[x].
  - **No dead CBs:** `push_cb` skips zero-size CBs (`program_factory.cpp:180-183`) and sizes are gated on `*_has_value` / `do_mask_*`, so a CB is allocated iff a kernel touches it. Present/absent optional CBs (gamma/beta/mask/mean/rstd) flip allocation with config but never produce an allocated-yet-untouched CB. `c_0`/`c_25`/`c_26` change tile counts between small/large algorithm but endpoints are unchanged.
- **Offset base pointers:** GREEN. All tensor addresses reach the kernels as clean `Buffer*` bindings (BufferBinding form) with **no** host-folded `+ offset`. `tile_offset` is passed as a **separate scalar RTA** (`program_factory.cpp:383,393`) and consumed in-kernel purely as a page-index term (`page_id = ... + tile_offset`), routed through the `TensorAccessor`, not added to a base address. No Type 1/2/3/4 site. (Not in the offset-base-pointer triage doc; scan confirms clean.)
- **TensorAccessor 3rd argument:** GREEN. Every `TensorAccessor` construction is the 2-arg form `TensorAccessor(args, addr)` — reader: input/gamma/beta (`reader_*:39,43,48`); writer: output/mean/rstd (`writer:121,124,127`). No site passes an explicit page-size 3rd argument. (Not in the 3rd-arg triage doc; scan confirms clean.)

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding) — all **Case 1** (via `TensorAccessor`), delivered today as `Buffer*` BufferBindings in the RTAs, consumed in-kernel by `TensorAccessor(args, addr)`:
  - `input` (c_0) — reader, `TensorAccessor(input_args, input_addr)`.
  - `gamma` (c_3) — reader, present-only (`GAMMA_HAS_VALUE`).
  - `beta` (c_4) — reader, present-only (`BETA_HAS_VALUE`).
  - `output` (c_16) — writer, `TensorAccessor(output_args, output_addr)`.
  - `mean` (c_17) — writer, present-only.
  - `rstd` (c_18) — writer, present-only.
  - Compute kernels are **out of scope** for this subject (they only produce/consume CBs, never touch tensor memory).
- **TensorParameter relaxation:** none.
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** self-loop `c_24..c_31`; bind 1P+1C on `c_0,c_1,c_2,c_3,c_4,c_5,c_6,c_16,c_17,c_18`.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none — no hidden second writers (no raw `get_write_ptr` co-fills; the only raw pointer access is the writer's own `dfb.get_read_ptr()` peek on its consumer binding in `writer:34`), no multi-reader CBs.
- **Cross-op / shared kernels:**
  - `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp` (shared kernel pool) — `fill_cb_with_value`, `generate_mask_h`, `generate_mask_w`, `get_tilized_idx`; all take `DataflowBuffer` / plain-uint args (Device 2.0 native).
  - `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` (shared kernel pool) — the `*_with_dt` tile-op wrappers; take `DataflowBuffer` args (Device 2.0 native).
  - `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` (official shared kernel lib) — `compute_kernel_lib::reduce<...>`; lib team owns.
  - These are `#include` (function-call) escapes into shared header pools; the Metal 2.0 CB→DFB rewrite of these headers is a **shared rewrite** — port every moreh op that includes them as one unit. No kernel `.cpp` is file-path-instantiated from another op (the op owns all five of its kernel sources).
- **RTA varargs:** none. Both readers and the writer read a **fixed run** of distinct named scalars (`get_arg_val<uint32_t>(i++)` at the top / fixed constant indices) — the non-signal named-arg case, ordinary port work.

## Team-only

- **Out-of-directory coupling & donor shape:**
  - Roll-up: **✓ clean** (all donor functions are Device 2.0 native; no `CircularBuffer&`, no addr-gen, no semaphore-shape donors).

  | Op kernel | Donor file | Class | Shape | Status |
  |---|---|---|---|---|
  | reader_small/large | `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp` | shared pool (`ttnn/cpp/ttnn/kernel/`) | `DataflowBuffer` + plain-uint args | ✓ |
  | reader_small/large, writer | `api/dataflow/*.h`, `api/tensor/noc_traits.h`, `api/core_local_mem.h` | `tt_metal/*` | HAL/LLK | ✓ (no concern) |
  | compute_small/large | `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` | shared pool | `DataflowBuffer` args | ✓ |
  | compute_small/large | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` | official kernel lib | `compute_kernel_lib::reduce<CB...>` (CB-index NTTPs) | ✓ (lib team) |

  - **Borrowed kernel files (file-path instantiation):** none — all five kernel `.cpp` sources are owned in `device/kernels/`.
- **TTNN factory analysis (sheet-derived facts, cross-checked):** current concept `descriptor`; no op-owned tensors; no MeshWorkload; no pybind `create_descriptor`; no custom hash; no custom `override_runtime_arguments`; `Is safe to port? == yes` (no smuggled pointer). Target concept `MetalV2FactoryConcept`.

## Misc anomalies  *(team-only, non-gating)*

- **Dead variable `input_data_format`** — `reader_moreh_layer_norm_small.cpp:32` and `reader_moreh_layer_norm_large.cpp:32`: `const auto input_data_format = get_dataformat(cb_id_input);` is computed and never used. Removing it also removes the sole non-`get_tile_size` CB-index free-function call. (The port will relocate remaining metadata lookups onto the DFB object regardless; deleting the dead line is the cleaner ops-team fix.)
- **Dead variable `offs`** — `reader_moreh_layer_norm_small.cpp:71,126`: `offs` is declared and incremented (`offs += num_inner`) but never read (the small reader addresses via `tile_offset + outer_idx * num_inner + inner_idx`). Dead in the small reader only; the large reader (`reader_..._large.cpp`) genuinely uses `offs`.
- **`onetile` constant** — declared in both readers (`reader_small:70`, `reader_large:71`) but not obviously consumed there; minor dead constant.

## Recipe notes

- **Device 2.0 sanctioned-free-function list is under-specified for the metadata family.** The Device 2.0 Green bullet names only `get_tile_size(cb_id)` and `get_local_cb_interface(cb_id)` as sanctioned CB-index free functions, but the same subject's breadcrumb says the Metal 2.0 `DataflowBuffer` "exposes a **full** tile/format metadata accessor set ... so a Metal 2.0 port moves **these lookups** onto the object (kernel-side whitelist rule 7)." `get_dataformat(cb_id)` is squarely in that family (`DataflowBuffer::get_dataformat()` at `dataflow_buffer.h:241`; Device 2.0 `CircularBuffer::get_dataformat()` forwards to the free function at `circular_buffer.h:115`), yet it is not on the explicit sanctioned list — forcing an auditor judgment call on whether it gates. It was resolved GREEN here (same reasoning as `get_tile_size`), but the list should either enumerate the metadata accessors or state that the whole "tile/format metadata accessor set" is sanctioned-at-Device-2.0 / handled by whitelist rule 7.
