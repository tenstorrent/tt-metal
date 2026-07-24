# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/clone`

Single device operation, single program factory (one `create_descriptor` with config-dependent branches):

- **`CloneOperation`**
  - `create_descriptor` (`clone_program_factory.cpp`) — the readiness sheet's `CloneOperation (single-descriptor)` factory. One `ProgramDescriptor` builder with four code-paths selected at build time: {tilized, row-major} × {interleaved, sharded}, plus an optional compute kernel when `convert_dtype`.

**Referenced kernels** (all owned by clone, under `device/kernels/`):

| Path selector | Reader | Writer | Compute (only if `convert_dtype`) |
|---|---|---|---|
| tilized · interleaved | `read_kernel.cpp` | `write_kernel.cpp` | `compute_kernel.cpp` |
| row-major · interleaved | `read_kernel_rm.cpp` | `write_kernel_rm.cpp` | `compute_kernel.cpp` |
| tilized · sharded | `read_kernel_sharded.cpp` | `write_kernel_sharded.cpp` | `compute_kernel.cpp` |
| row-major · sharded | `read_kernel_rm_sharded.cpp` | `write_kernel_rm_sharded.cpp` | `compute_kernel.cpp` |

No unreferenced kernel files in the directory. No cross-op / donor kernels — all nine kernel files are clone-owned.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `4af0db51e3c 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/clone` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `CloneOperation` → `create_descriptor` (single-descriptor, four config branches) |
| *Prereqs* — Device 2.0 (every kernel used) | Yes (all kernels use `Noc` / `DataflowBuffer` / `TensorAccessor` / `UnicastEndpoint`; only CB-index free fn is sanctioned `get_tile_size`) |
| *Prereqs* — Cross-op escapes | Ok — no out-of-directory op coupling; all kernels clone-owned; all `#include`s resolve to `tt_metal/hw/inc/api/*` (LLK/HAL) |
| *Feature Support* — overall | GREEN |
| *Feature Support* — Variadic-CTA | Ok (single input tensor; CTAs read at constexpr offsets only) |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Is safe to port? | Yes |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — Runtime-args update | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` |
| *Port work* — Offset base pointer | none (clean bases; `Buffer*`-binding RTAs, no host-folded offset) |
| *Port work* — Tensor bindings (per binding) | `input`, `output`: Case 1 (interleaved) / Case 2 (sharded) — per-config split |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none (no accessor passes a 3rd arg) |
| *Port work* — CB endpoints | legal (every CB is 1P+1C on every node, every config) |

**CB endpoints** are dispositions, not gates. For clone, every CB is a genuine 1-producer/1-consumer FIFO on every node in every config — no self-loop, 1P+1C assignment, multi-binding flag, or dead-CB drop is needed.

## Result

**GREEN → brief issued.** `CloneOperation` is on the `descriptor` concept, is Device 2.0 compliant across all four config branches, uses no Appendix A feature, folds no offset into any base pointer, and passes no `TensorAccessor` 3rd argument. The readiness sheet reports `Is able to port? = yes` and the cheap cross-check agrees on every column. The port is straightforward tensor-binding work: two bindings (`input`, `output`), each Case 1 on the interleaved paths and Case 2 (raw-pointer bridge) on the sharded paths.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** Sheet row `data_movement/clone,CloneOperation,CloneOperation (single-descriptor)` → `Is able to port? = yes`. Cross-check against code, all consistent:
  - `Concept = descriptor` ✓ — `create_descriptor()` returns `tt::tt_metal::ProgramDescriptor` (`clone_device_operation.hpp:29`).
  - `Custom hash = no` ✓ — no `compute_program_hash` override anywhere in the op (grep clean).
  - `Runtime-args update = no` ✓ — no `get_dynamic_runtime_args` / `override_runtime_arguments` (grep clean).
  - `Pybind descriptor = no` ✓ — `clone_nanobind.cpp` binds only the `clone` free function via `ttnn::bind_function<"clone">`; no `create_descriptor` / device-op class binding.
  - `Is safe to port? = yes`, `Smuggled pointer = no` (readiness-sheet owner's correctness axis — trusted, not re-derived).
  - Cross-column invariants OK: `Op-owned tensors?` blank (consistent with a `descriptor` concept), `Runtime-args update = no`.

- **Device 2.0 (every kernel used):** **GREEN.** All nine kernels are structurally Device 2.0 — in fact they already use the Metal 2.0 kernel-side `DataflowBuffer` object. Idioms observed: `Noc` object (`async_read`/`async_write`/`*_barrier`), `DataflowBuffer` FIFO ops (`reserve_back`/`push_back`/`wait_front`/`pop_front`), `TensorAccessor` (interleaved paths), `UnicastEndpoint` (sharded paths). The only CB-index free function is `get_tile_size(cb_id)` — **sanctioned** by the Green bullet (the Device 2.0 migration guide uses it in its own migrated examples, `device_api_migration_guide.md:605,630`); not a holdover. The compute kernel's `unary_op_init_common` / `copy_tile` / `pack_tile` take a `cb_id` but are **compute-domain LLK**, outside the Device-2.0 data-movement surface — not a violation (and the compute kernel's own FIFO ops use `DataflowBuffer`). No `InterleavedAddrGen` / `ShardedAddrGen` / raw `noc_async_*` / manual CB-index management anywhere.

- **Feature compatibility:** every Appendix A entry scanned; all absent.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, no `.global_circular_buffer` field, no `remote_cb`/`remote_index` idiom. CBs are plain `CBDescriptor`s (`clone_program_factory.cpp:103,117`). |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `.address_offset` set on any `CBDescriptor`; no `set_address_offset` / 4-arg `UpdateDynamicCircularBufferAddress` / `cb_descriptor_from_sharded_tensor`. |
  | GlobalSemaphore | N/A | Op uses no semaphores at all. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t` carries a single `const Tensor& input` — no `std::vector<Tensor>`. All kernels read CTAs at constexpr offsets (`get_compile_time_arg_val(0)`, `TensorAccessorArgs<1>`/`<2>`); no runtime-varying CTA index. |

- **CB endpoints (GATE-free):** **✓ legal** in all four configs. Census per CB, per node:
  - `src_cb` (`c_4`), when `convert_dtype = false` (`dst_cb_id == src_cb_id`): reader FIFO-produces, writer FIFO-consumes → **1P+1C, legal**. (No compute kernel in this config.)
  - `src_cb` (`c_4`), when `convert_dtype = true`: reader FIFO-produces, compute FIFO-consumes → **1P+1C, legal**.
  - `dst_cb` (`c_20`), only when `convert_dtype = true`: compute FIFO-produces, writer FIFO-consumes → **1P+1C, legal**.
  - Holds on every node: reader/writer run on `all_cores`; compute runs on `core_group_1` ∪ `core_group_2` = `all_cores`, so each node sees exactly one producer + one consumer per CB. No raw-pointer co-fillers, no hidden second writers, no multi-reader shapes, no dead CBs.

- **Offset base pointers:** **GREEN.** No address RTA folds a host-side offset into a base. The factory delivers both tensors as `Buffer*` via `emplace_runtime_args(core, {input_buffer, ...})` / `{output_buffer, ...}` (`clone_program_factory.cpp:231-243`) — the `Buffer*`-binding (`BufferBinding`) form, **not** `buffer->address()` and never `address() + offset`. `start_id` is a separate page/stick-index scalar RTA added on-device in the loop (`start_id + num_tiles`), not folded into an address; the sharded kernels compute their local L1 walk on-device (`local_l1_read_addr += tile_size`). Clone is in neither the offset-base-pointer triage table nor (by scan) an offset-bearing op. All bases clean → handed to TensorParameter analysis as ordinary port work.

- **TensorAccessor 3rd argument:** **GREEN.** No `TensorAccessor` construction passes a 3rd (page-size) argument. Interleaved kernels use the 2-arg `TensorAccessor(args, base_addr)` (`read_kernel.cpp:20`, `read_kernel_rm.cpp:21`, `write_kernel.cpp:20`, `write_kernel_rm.cpp:21`); sharded kernels use no `TensorAccessor` at all. Clone is absent from the 3rd-arg triage table, consistent with the scan. Nothing to classify.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding, per config — the classification splits by factory branch):
  - **`input`** — **Case 1** (via `TensorAccessor`) on the interleaved paths (`read_kernel.cpp`, `read_kernel_rm.cpp`): base fed into `TensorAccessor(args, input_buffer_address)`. **Case 2** (raw pointer) on the sharded paths (`read_kernel_sharded.cpp`, `read_kernel_rm_sharded.cpp`): base used directly as `.addr` in a local `noc.async_read(UnicastEndpoint{}, …, {.addr = local_l1_read_addr})`, no accessor.
  - **`output`** — **Case 1** on the interleaved paths (`write_kernel.cpp`, `write_kernel_rm.cpp`). **Case 2** on the sharded paths (`write_kernel_sharded.cpp`, `write_kernel_rm_sharded.cpp`).
  - Delivery today is the `Buffer*`-binding form (`emplace_runtime_args` with `Buffer*`) — correct-on-cache-hit under the framework's interim `BufferBinding` patching, **not** the silent-wrong `->address()`-in-RTA hazard. The port replaces it with a typed `TensorParameter` / `TensorBinding` either way.
- **TensorParameter relaxation:** none (sheet `none`; no custom hash).
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** all legal — no disposition needed.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none — every CB is a plain 1P+1C FIFO.
- **Cross-op / shared kernels:** none — clone owns all nine kernel files; no file-path instantiation from a shared pool, no function-call escape. No port-together coupling.
- **RTA varargs:** none — every kernel reads its RTAs as a fixed set of distinct fields at constant indices (`get_arg_val<uint32_t>(0..3)`); no counted loop over `arg_index++`, no data-selected index.

## Team-only

- **Out-of-directory coupling & donor shape:** **✓ clean.** Every kernel `#include` resolves to `tt_metal/hw/inc/api/*` (`api/dataflow/*`, `api/compute/*`, `api/tensor/noc_traits.h`) — the LLK/HAL/firmware class (no concern). No `ttnn/cpp/ttnn/kernel_lib`, `kernel/`, `kernel_helper_functions/`, in-family, or cross-family includes. No `KernelDescriptor::kernel_source` points outside the clone directory (all nine paths are `.../clone/device/kernels/*.cpp`). No function-call escapes; no borrowed kernel files. Summary table and per-call detail omitted (all rolls ✓).
- **Relaxation candidates:** none — no custom hash to mine.
- **TTNN factory analysis:** current concept `descriptor`; no op-owned tensors; no MeshWorkload; no pybind `create_descriptor`; no custom hash; no custom `override_runtime_arguments`. Target concept `MetalV2FactoryConcept`.

## Misc anomalies  *(team-only, non-gating)*

- **Dead compile-time arg on the row-major *interleaved* paths.** The factory sets `reader_ct_args = {src_cb_id, input_unit_size}` and `writer_ct_args = {dst_cb_id, output_unit_size}` (`clone_program_factory.cpp:140,142`), i.e. the unit/stick size occupies compile-time-arg index 1. But `read_kernel_rm.cpp:17` / `write_kernel_rm.cpp:17` declare `TensorAccessorArgs<2>()`, so the accessor consumes CTAs from index 2 onward and **CTA index 1 is never read** — the kernels instead take the stick size from RTA index 1 (`get_arg_val<uint32_t>(1)`). The value is duplicated (CTA and RTA carry the same `input_unit_size`/`output_unit_size`) and the CTA copy is dead. Harmless (functionally inert; the RTA copy drives the kernel), but it needlessly widens the compile-time key. The tilized interleaved paths do not have this gap (`reader_ct_args = {src_cb_id}`, `TensorAccessorArgs<1>`). Routes to the ops team; not porter work.

## Questions for the user

*(none)*

## Recipe notes

- The readiness sheet lists clone as a single factory row (`CloneOperation (single-descriptor)`), but the one `create_descriptor` fans out into four build-time code-paths ({tilized, RM} × {interleaved, sharded}) whose kernel *source files* differ, and the TensorParameter classification splits across them (Case 1 interleaved vs. Case 2 sharded). The recipe's per-factory framing handled this cleanly via the "classification can vary per factory / per config" guidance in [TensorParameter analysis](#) and [Code-path scope]; recording here only that "one sheet factory row" did not mean "one code-path" for this op — a reader cross-referencing the sheet's single row against four kernel-file pairs might briefly expect a mismatch. No friction with the gate logic itself.
