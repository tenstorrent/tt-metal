# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/split`

Single device operation, single program factory, both kernels owned by the op:

- **`SplitDeviceOperation`** (`ttnn::prim`; declared in `device/split_device_operation.hpp`)
  - `SplitProgramFactory` — `create_descriptor` in `device/split_program_factory.cpp` (declared in `device/split_program_factory.hpp`)
  - Kernels (both referenced by the factory, both in the op's own directory):
    - reader — `device/kernels/dataflow/reader_tm_tile_layout_split_two_chunks.cpp`
    - writer — `device/kernels/dataflow/writer_split_n_chunks_tile.cpp`

No unreferenced/dead kernel files in the directory.

**Two-backend composite — scope note.** The host-facing `ttnn::split` (`split.cpp`) is a dispatcher with two backends: the native TILE device op (`ttnn::prim::split` → `SplitDeviceOperation`, **in scope here**), and a fallback of N independent `ttnn::slice` calls (`detail::split_with_slice_impl`, used for ROW_MAJOR / unequal / non-last-dim / batch>1-for-N>2 cases). **Only the native device op is audited here.** The slice-fallback path executes a *different* op (`ttnn::slice`) with its own audit; note that per the offset-base-pointer triage `slice`'s RM path is itself an offset-base gate, so porting `SplitDeviceOperation` neither ports nor unblocks the slice-fallback path.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `44da718b06b 2026-07-23 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/split` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `SplitDeviceOperation` → `SplitProgramFactory` (single factory) |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — both kernels fully object-oriented Device 2.0 (`Noc`, `DataflowBuffer`, `TensorAccessor`); no free-function CB-index holdovers, no Device 1.0 idioms |
| *Prereqs* — Cross-op escapes | **Ok** — both kernels owned by the op; every `#include` is `api/*` (LLK/HAL) |
| *Feature Support* — overall | **GREEN** (all Appendix A entries N/A) |
| *Feature Support* — Variadic-CTA | **Ok** — kernels read CTAs only at constexpr offsets 0–4; no runtime-varying CTA index |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A (concept is `descriptor`) |
| *TTNN Readiness* — Is safe to port? | Yes |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — Runtime-args update | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` (no op-owned tensors) |
| *Port work* — Offset base pointer | none (clean bases + separate page-index scalars) |
| *Port work* — Tensor bindings (per binding) | input `in0`: Case 1 · outputs (N chunks): Case 1 |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none (neither accessor passes a 3rd arg) |
| *Port work* — CB endpoints | legal (1 producer + 1 consumer on `src0` CB 0) |

**CB endpoints** are dispositions, not gates. Split has one CB with a clean 1-producer / 1-consumer census on every node — no self-loop, assignment, flag, or drop required.

## Result

**GREEN → brief issued.** Every gate clears: Device 2.0 ✓ · Feature compatibility ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd argument ✓. The port targets `MetalV2FactoryConcept`. `METAL2_PORT_BRIEF.md` is written alongside this file.

**One prominent port-shape consideration (not a gate):** the op produces a **runtime-variable number of output tensors** (`N = num_splits`), and each core writes exactly one of them — delivered today as a per-core `Buffer*` runtime arg (`output_buffers[chunk_id]`). The port must express N output `TensorBinding`s, each bound to its chunk's core group. `N` is folded into the program hash (an `operation_attributes` field), so it is **fixed per compiled program** — not a within-program runtime-varying count. This is not an Appendix A feature (variable-count *output* bindings are unlisted → supported per the recipe's operating principle), and the readiness sheet independently rates the op `Is able to port? = yes`. It is called out here and in the brief because it is the non-mechanical shape of this port. See *Questions* and *Recipe notes*.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** The readiness sheet (Diego's *"Operations analysis"*, fetched fresh this run) has exactly one row for `data_movement/split`, and `Is able to port?` = `yes`. Every conjunct in the derivation passes: `Is safe to port?` = `yes`, `Custom hash` = `no`, `Runtime-args update` = `no`, `Pybind descriptor` = `no`, `Concept` = `descriptor`. Cross-check clean (all cheaply-checkable columns confirmed against the code):
  - `Concept` = `descriptor` — `SplitProgramFactory::create_descriptor(...)` returns `tt::tt_metal::ProgramDescriptor` (`split_program_factory.cpp:74`, `split_program_factory.hpp:14`).
  - `Custom hash` = `no` — no `compute_program_hash` override anywhere in the op directory.
  - `Runtime-args update` = `no` — no `get_dynamic_runtime_args` / `override_runtime_arguments` in the factory.
  - `Pybind descriptor` = `no` — `split_nanobind.cpp` binds only the free function `ttnn::split` via `bind_function`; no `nb::class_` of the device op, no `create_descriptor` binding.
  - Cross-column invariants hold (`Runtime-args update` = `no`; `Op-owned tensors?` empty, consistent with a `descriptor` concept). No sheet/code conflict; op is present (not a spreadsheet-broken case).
  - Sheet also records `Op Classification` = `PD (pointer-patching)`, `Smuggled pointer` = `no`, `TensorParameter relaxation` = `none` — all consistent with the per-binding analysis below.

- **Device 2.0 (every kernel used):** **GREEN.** Both kernels are structurally Device 2.0 (indeed already on the kernel-side `DataflowBuffer` object) with no holdovers:
  - `Noc noc;` object + `noc.async_read` / `noc.async_write` / `noc.async_read_barrier` / `noc.async_write_barrier` — no raw `noc_async_*`.
  - `DataflowBuffer dfb(id);` object + `.reserve_back` / `.push_back` / `.wait_front` / `.pop_front` / `.get_entry_size` — no free-function CB-index calls (`get_read_ptr(cb_id)`, `get_write_ptr(cb_id)`, `get_tile_size(cb_id)`, `cb_reserve_back(cb_id, …)`), and no sanctioned-holdover forms either.
  - `TensorAccessor(args, addr)` for addressing — no `InterleavedAddrGen` / `ShardedAddrGen` / `InterleavedAddrGenFast` / `InterleavedPow2AddrGen*`, no raw bank-address arithmetic.
  - No semaphores.
  - Includes are all `api/*` (LLK/HAL) plus `tensix_types.h` and `stdint.h`.

- **Feature compatibility:** every Appendix A entry, in order. All **N/A** (no entry's recognition signals fire).

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type/alias, no `CreateGlobalCircularBuffer`, no `.global_circular_buffer` field on the `CBDescriptor`, no `remote_cb` / `.remote_index` idiom, no `num_global_cb_receivers`. The single `CBDescriptor` (`split_program_factory.cpp:164`) is a plain double-buffered CB. |
  | CBDescriptor `address_offset` (non-zero) | N/A | The `CBDescriptor` does not set `.address_offset` (defaults to 0); no `set_address_offset`, no 4-arg `UpdateDynamicCircularBufferAddress`, no `cb_descriptor_from_sharded_tensor`. |
  | GlobalSemaphore | N/A | The op uses no semaphores of any kind (no `GlobalSemaphore`, no plain `Semaphore`/`CreateSemaphore`). |
  | Variable-count compile-time arguments (CTA varargs) | N/A | Op-level cue absent: `tensor_args_t` (`SplitInputs`) is a single fixed input `Tensor`, not a variable-count container. Kernel-level decider absent: both kernels read `get_compile_time_arg_val` only at constexpr offsets 0–4 (fixed 5-arg schema + `TensorAccessorArgs<5>`), never at a runtime-varying index. The op's variability is in its *outputs* (`tensor_return_value_t = std::vector<Tensor>`), which ride per-core `Buffer*` RTAs and the core-grid layout — not CTAs. This is the matmul-style N/A case, not the concat-style RED. |

- **CB endpoints (GATE-free):** **legal.** One CB in the whole factory: `src0` at `buffer_index = 0`, `total_size = 2 * single_tile_size` (`split_program_factory.cpp:162-172`). Census per node: the reader instance is the **locked producer** (`dfb_in0.reserve_back` / `push_back`), the writer instance is the **locked consumer** (`dfb_out.wait_front` / `pop_front`); both kernels run over the same `all_cores` range and both bind DFB id `0`. That is a plain **1 producer + 1 consumer** on every node → legal 1:1, no port-time disposition. Config-invariant: `num_splits` changes the core-grid dimensions but never the CB's endpoint structure.

- **Offset base pointers:** **GREEN.** `split` is absent from the Type-1/Type-2 tables of `2026-07-19_offset_base_pointers.md`, and my own scan confirms no offset is folded into any base. Both address args are delivered as the **`Buffer*`-binding form** (the pointer object itself, not `->address()`), and the per-tile offsets ride *separate* scalar args used as `page_id`, never folded into the base:
  - reader RTA (`split_program_factory.cpp:63`): `{reader_core_id, in0_buffer, 0}` — `in0_buffer` is a clean `Buffer*`; `reader_core_id` is a page index consumed as `.page_id` in the kernel.
  - writer RTA (`split_program_factory.cpp:65`): `{writer_core_id, output_buffers[chunk_id]}` — `output_buffers[chunk_id]` is a clean `Buffer*`; `writer_core_id` is a page index consumed as `.page_id`.
  This is exactly the split-out shape a Type-1 fix produces (clean base + separate offset scalar), so the bindings drop straight into ordinary TensorParameter port work.

- **TensorAccessor 3rd argument:** **GREEN.** `split` is absent from the `2026-07-06_tensor_accessor_3rd_arg_triage.md` table, and my own read confirms neither accessor passes a 3rd argument: reader `TensorAccessor(in0_tensor_args, in0_tensor_addr)` (`reader_tm_tile_layout_split_two_chunks.cpp:37`) and writer `TensorAccessor(out_tensor_args, out_tensor_addr)` (`writer_split_n_chunks_tile.cpp:37`) are both 2-arg. Subject does not fire; nothing to drop.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding):
  - `in0` (input, reader) — **Case 1** (via `TensorAccessor`). Base delivered today as a per-core `Buffer*` RTA (same value on every core); kernel feeds it into `TensorAccessor(in0_tensor_args, in0_tensor_addr)` and reads exclusively through the accessor.
  - output chunks (writer) — **Case 1** (via `TensorAccessor`), **N = num_splits distinct bindings**. Each core's output base is delivered today as a per-core `Buffer*` RTA `output_buffers[chunk_id]` (differs per chunk group); kernel feeds it into `TensorAccessor(out_tensor_args, out_tensor_addr)`. Port binds each output `TensorParameter` to its chunk's core group (see the port-shape consideration in Result / Questions).
- **TensorParameter relaxation:** none (sheet `TensorParameter relaxation` = `none`; no custom hash).
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** all legal — the single `src0` CB is a clean 1-producer / 1-consumer FIFO.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none — the only CB is a plain 1:1 FIFO.
- **Cross-op / shared kernels:** none — both kernels are owned by the op; no file-path borrows, no cross-family `#include`s. No port-together coupling.
- **RTA varargs:** none — every RTA is read at a fixed offset (reader 0–2, writer 0–1); no counted loop, no data-selected index.
- **Variable output-tensor binding (the port's non-mechanical shape):** N output `TensorBinding`s, one per chunk core-group, replacing the per-core `output_buffers[chunk_id]` `Buffer*` RTA. Confirm the binding strategy before construction (see *Questions*).

## Team-only

- **Out-of-directory coupling & donor shape:** **✓ clean.** Neither kernel `#include`s anything outside the op directory beyond `api/*` (LLK/HAL/firmware, class 1 — no concern) and `tensix_types.h` / `stdint.h`. No function-call escapes, no file-path kernel instantiation of borrowed kernels, no shared-pool kernels. No donor shape table needed. (The op-level composite `split.cpp` calls into `ttnn::slice` / `ttnn::reshape_on_device` / `ttnn::view` on its *fallback* path, but those are host-side op dispatches on a separate code path, not kernel donors of `SplitDeviceOperation` — see the scope note at the top.)
- **TTNN factory analysis (sheet-derived facts, with evidence):**
  - Current concept: `descriptor` (`create_descriptor` → `ProgramDescriptor`). Target concept: `MetalV2FactoryConcept`.
  - Op-owned tensors: none (sheet empty; consistent with `descriptor`). No MeshWorkload need.
  - `Op Classification` = `PD (pointer-patching)`: the op's tensor addresses ride the framework's `Buffer*`-binding / cache-hit-patching mechanism (both `in0_buffer` and `output_buffers[chunk_id]` are `Buffer*` RTAs). This is correct-on-cache-hit today and is superseded by the Metal 2.0 typed `TensorBinding` at port time.
  - `Smuggled pointer` = `no`: confirmed — no raw `buffer->address()` rides an un-annotated RTA/CRTA; addresses are delivered via the auto-registered `Buffer*` form.
  - Custom hash: no. Custom `override_runtime_arguments`: no. Pybind `create_descriptor`: no. Runtime-args update: no.
- **Relaxation candidates:** none (no custom hash to mine).

## Misc anomalies  *(team-only, non-gating; route to the ops team; the port does not act on these)*

- **Dead RTA — `split_last_dim`.** The reader reads `bool split_last_dim = (bool)get_arg_val<uint32_t>(2);` (`reader_tm_tile_layout_split_two_chunks.cpp:22`) but never uses it, and the factory always passes it as literal `0` (`split_program_factory.cpp:63`). A dead runtime arg on both sides.
- **Vestigial multi-tensor scaffolding in the reader.** `constexpr uint32_t out_num_tensors = 1;` (`reader_tm_tile_layout_split_two_chunks.cpp:32`) with the outer `for (out_tensor_id …)` loop that always runs once, plus `tensor_stride` / `tensor_stride_cum` (lines 43-44, 76) which contribute a constant `0` to `tile_id` across the single iteration. Leftover from an earlier design where one reader instance produced multiple output tensors' worth; inert today (each core now handles exactly one chunk). Harmless but dead.
- **Stale kernel filename.** `reader_tm_tile_layout_split_two_chunks.cpp` reads as a 2-chunk-specific reader, but it is the generalized N-chunk reader (chunk count arrives via CTAs / grid layout). The writer counterpart is already named `writer_split_n_chunks_tile.cpp`. Filename only; no behavioral effect.

## Questions for the user  *(non-blocking — port-strategy confirmation, not a gate)*

1. **Multi-output per-core tensor binding.** The writer is a single `KernelDescriptor` over `all_cores`, and each core writes to one of `N = num_splits` output tensors, selected per-core via the `output_buffers[chunk_id]` `Buffer*` RTA (`split_program_factory.cpp:65`). The Metal 2.0 port needs to bind N output `TensorParameter`s and have each core's writer instance address the output for its chunk group. `N` is fixed per compiled program (it is hashed via `operation_attributes.num_splits`), so this is a fixed-count-per-program binding, not a within-program runtime-varying count — and it is not an Appendix A feature, so it does not gate. The recipe does not explicitly work an example of *variable-count output* bindings bound per-core-subset (its variable-count discussion is about *input* tensors under CTA varargs). Please confirm the intended shape — bind each output tensor to its chunk's `CoreRange` sub-region within the one writer `KernelSpec` — so the porter builds toward it directly rather than discovering it mid-port.

## Recipe notes  *(friction with the audit recipe itself)*

- **Variable-count *output* tensors are not addressed by any subject.** The CTA-varargs Appendix A entry and its false-positive guard are framed entirely around *input* tensors (`tensor_args_t` carrying `std::vector<Tensor>`). Split's variability is in `tensor_return_value_t` (`std::vector<Tensor>` outputs), each bound per-core. No subject or Appendix A entry names this shape. I resolved it as *supported, non-gating port work* per the operating principle ("a construct not listed in Appendix A is supported") reinforced by the readiness sheet's `Is able to port? = yes`, and surfaced it as a port-shape heads-up plus a Question. A one-line note in either TensorParameter analysis or the TTNN porting shape subject on how per-core-subset multi-output bindings map to Metal 2.0 would remove the judgment call for the next auditor of a multi-output data-movement op.
- **Kernels already on the Metal 2.0 `DataflowBuffer` object at the Device 2.0 gate.** Both kernels use `DataflowBuffer` (the Metal 2.0 kernel-side object per the recipe's Device-2.0-gate breadcrumb) rather than the Device 2.0 `CircularBuffer` wrapper the migration guide's examples use. This is *ahead of* the Device 2.0 baseline and still cleanly passes the gate (object-oriented, no free-function holdovers), but the recipe's Device 2.0 recognition signals are written around `CircularBuffer` idioms; a reader matching literally on `CircularBuffer` might hesitate on a kernel that has skipped straight to `DataflowBuffer`. Worth a note that a `DataflowBuffer`-based kernel is Device-2.0-clean by construction (it is strictly ahead), so it passes the gate even though it does not match the `CircularBuffer`-shaped examples.
