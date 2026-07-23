# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/examples/example`

- **`ExampleDeviceOperation`**
  - `SingleCore` (`device/single_core_program_factory.cpp`) — pins work to a 1×1 grid
  - `MultiCore` (`device/multi_core_program_factory.cpp`) — distributes tiles across the storage grid

Both factories are `create_descriptor()` → `ProgramDescriptor` (the `descriptor` concept). They share identical structure: two CBs (`c_0` input, `c_2` output) and three **cross-family donor** kernels instantiated by file path from `eltwise/unary` (see below) — the op owns none of the kernels it runs.

**Unreferenced kernel files in the op directory (out of scope).** The op's own `device/kernels/` tree contains `compute/eltwise_sfpu.cpp` and `dataflow/{blank,reader_binary_diff_lengths,reader_unary,writer_unary}.cpp`. No factory references any of them — both factories instantiate the `eltwise/unary` copies by their repo-root path strings. These local files are dead code and were not audited.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `metal2_audit.md`.

**Recipe docs:** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/examples/example` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `ExampleDeviceOperation` → `SingleCore`, `MultiCore` |
| *Prereqs* — Device 2.0 (every kernel used) | Yes — all 3 donor kernels are Device 2.0 (`Noc` / `DataflowBuffer` / `TensorAccessor`; only sanctioned `get_local_cb_interface` free fn) |
| *Prereqs* — Cross-op escapes | Ok — 3 file-path kernel borrows from `eltwise/unary`; no function-call escapes (all `#include`s are `api/*` LLK) |
| *Feature Support* — overall | GREEN |
| *Feature Support* — Variadic-CTA | Ok (N/A) |
| *TTNN Readiness* — `Is able to port?` (the gate) | Yes (both factories) |
| *TTNN Readiness* — Concept (current) | `descriptor` (both factories) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Is safe to port? | Yes |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — Runtime-args update | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` |
| *Port work* — Offset base pointer | none (cleared) |
| *Port work* — Tensor bindings (per binding) | `input_tensor` Case 1 · `output_tensor` Case 1 |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none (no site passes a 3rd arg) |
| *Port work* — CB endpoints | legal (both CBs 1P+1C on every node/config) |

## Result

**GREEN → brief issued.** Every gate clears: Device 2.0 ✓, Feature compatibility ✓, TTNN factory concept ✓ (both factories `Is able to port? = yes`), Offset base pointers ✓, TensorAccessor 3rd arg ✓. The port is a clean, canonical `descriptor` → `MetalV2FactoryConcept` port: two interleaved tensor bindings (both Case 1), two 1:1 CBs, no relaxations, no gates. This is the tutorial example op and behaves like one — no anomalies of substance. `METAL2_PORT_BRIEF.md` is written alongside this file.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** GREEN. Readiness sheet rows for `examples/example` / `ExampleDeviceOperation` — both `SingleCore` and `MultiCore` — read `Is able to port? = yes`, `Concept = descriptor`, `Custom hash = no`, `Runtime-args update = no`, `Override runtime args? = no`, `Pybind descriptor = no`, `Smuggled pointer = no`, `Is safe to port? = yes`, `TensorParameter relaxation = none`, `Op-owned tensors? = (blank/none)`. Cross-check against the code agrees on every cheaply-checkable column: both factories define `create_descriptor()` returning `ProgramDescriptor` (`descriptor`); no `compute_program_hash` override in the device-op; no `get_dynamic_runtime_args` / `override_runtime_arguments` (the one textual hit is a comment in `device/example_device_operation.hpp:65`); `example_nanobind.cpp` binds only the free function `composite_example`, no `create_descriptor` binding. No conflict.
- **Device 2.0 (every kernel used):** GREEN. The op runs three donor kernels, all from `eltwise/unary`, all structurally Device 2.0:

  | Kernel file | Idioms observed | Verdict |
  |---|---|---|
  | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` | `Noc noc`, `DataflowBuffer dfb`, `TensorAccessor(src_args, src_addr)`, `noc.async_read`/`async_read_barrier`, `dfb.reserve_back/push_back`; `get_local_cb_interface(cb_id_in0).fifo_page_size` (sanctioned free fn) | Device 2.0 ✓ |
  | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | `Noc noc`, `DataflowBuffer dfb`, `TensorAccessor(dst_args, dst_addr)`, `noc.async_write`/`async_writes_flushed`/`async_write_barrier`, `dfb.wait_front/pop_front`; `get_local_cb_interface(...)` (sanctioned) | Device 2.0 ✓ |
  | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` | `DataflowBuffer dfb_in/dfb_out`, `wait_front`/`reserve_back`/`pop_front`/`push_back`, compute LLK (`copy_tile`/`pack_tile`/`tile_regs_*`) | Device 2.0 ✓ |

  No legacy Device 1.0 idioms found (no raw `noc_async_read`, no `InterleavedAddrGen`/`ShardedAddrGen`, no raw sem addresses, no CB-index-keyed `get_read_ptr(cb_id)`/`get_write_ptr(cb_id)` holdovers). `get_local_cb_interface(cb_id)` is explicitly sanctioned by the Green bullet and does not knock the op out of Green.

- **Feature compatibility:** every Appendix A entry scanned; none fires.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, no `CreateGlobalCircularBuffer`, no `CBDescriptor.global_circular_buffer` field set, no `remote_index`/remote-CB idiom. Plain `CBDescriptor` literals only. |
  | CBDescriptor `address_offset` (non-zero) | N/A | Neither `CBDescriptor` literal sets `.address_offset`; both default to 0. |
  | GlobalSemaphore | N/A | Op uses no semaphores at all. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t` carries a single fixed `const Tensor&`; no `std::vector<Tensor>`. Kernels read CTAs only at fixed constexpr offsets (writer: `get_compile_time_arg_val(0)` + `TensorAccessorArgs<1>`; reader: `TensorAccessorArgs<0>`). No runtime-varying CTA index. |

- **CB endpoints (GATE-free):** all legal — every CB is a plain 1-producer / 1-consumer FIFO on every node, in both configs. Census per `(CB, config)`:
  - **`c_0` (`src0_cb_index`, input)** — reader FIFO-produces (`dfb.reserve_back`/`push_back`); compute FIFO-consumes (`dfb_in.wait_front`/`pop_front`). 2 touchers = 1 locked producer + 1 locked consumer → **1:1 legal**. No raw co-fillers, no third toucher.
  - **`c_2` (`output_cb_index`, output)** — compute FIFO-produces (`dfb_out.reserve_back`/`push_back`); writer FIFO-consumes (`dfb.wait_front`/`pop_front`). → **1:1 legal**.
  - Config check: in `SingleCore` (grid `{1,1}`) compute runs on `core_group_1` and reader/writer on `all_cores` — the single active node runs all three, so both CBs are 1:1 there. In `MultiCore` all three kernels span `all_cores`, so every active node runs reader+compute+writer → both CBs 1:1. No self-loops, no multi-binding, no dead CBs under any config.

- **Offset base pointers:** GREEN — no address RTA folds a host-side offset. Both factories deliver the tensor address via the **`Buffer*`-binding form**: `reader_desc.emplace_runtime_args(core, {src_buffer, num_tiles_per_core, num_tiles_written})` and `writer_desc.emplace_runtime_args(core, {dst_buffer, ...})` push the raw `Buffer*` object (not `->address()`, and with no `+ offset` arithmetic anywhere). No Type 1 / Type 2 fold; no `ttnn::narrow`. Clean bases handed to TensorParameter analysis. (`single_core_program_factory.cpp:111,113`, `multi_core_program_factory.cpp:109,111`.)

- **TensorAccessor 3rd argument:** GREEN — no site passes a 3rd argument. Both accessors are 2-arg: reader `TensorAccessor(src_args, src_addr)` (`reader_unary_interleaved_start_id.cpp:25`), writer `TensorAccessor(dst_args, dst_addr)` (`writer_unary_interleaved_start_id.cpp:31`). Subject does not fire.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding):
  - **`input_tensor`** → **Case 1** (via `TensorAccessor`). Delivered as `Buffer* src_buffer` in the reader RTA (position 0); reader consumes it as `src_addr = get_arg_val<uint32_t>(0)` and feeds it straight into `TensorAccessor(src_args, src_addr)` — all memory access is through the accessor. Express as a `TensorParameter` / `TensorBinding`; kernel builds `TensorAccessor(tensor::name)`; the RTA address and its `TensorAccessorArgs` CTA plumbing both disappear.
  - **`output_tensor`** → **Case 1** (via `TensorAccessor`). Delivered as `Buffer* dst_buffer` in the writer RTA (position 0); writer consumes it as `dst_addr = get_arg_val<uint32_t>(0)` and feeds it into `TensorAccessor(dst_args, dst_addr)`. Same treatment.
  - *Note:* the `Buffer*` delivery form is the framework's interim binding-injection hack — patched on cache hits today, so **not** the silent-wrong stale-pointer hazard. This is routine Case-1 port work, not a correctness fix. The compute kernel touches only CBs (no tensor memory) → out of scope for this subject.
- **TensorParameter relaxation:** none.
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** all legal — `c_0` and `c_2` are 1P+1C on every node in both configs. No self-loop, 1P+1C-assignment, multi-binding flag, or dead-CB drop needed.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none — no borrowed-memory tensor-view CBs, no dual-instance work-split, no hidden second writer.
- **Cross-op / shared kernels:** all three kernels are **file-path borrows from `eltwise/unary`** (see Team-only for the inventory). The reader/writer are broadly-shared library-grade kernels; their CB↔DFB / named-token rewrite is a single change that every borrowing op must adopt together. This op consumes them unmodified, so nothing to do here beyond awareness — but the port must not fork these kernel files.
- **RTA varargs:** none. Every RTA is a fixed distinct field read at a constant index (reader/writer: `src_addr`/`dst_addr`(0), `num_pages`(1), `start_id`(2); compute: `num_tiles`(0)) → all nameable, no vararg block.

## Team-only

- **Out-of-directory coupling & donor shape:**
  - **Op-level roll-up: ✓ clean.** No function-call escapes — every kernel `#include` resolves to `api/*` (tt_metal LLK/HAL, donor class 1, no concern). The only coupling is file-path kernel instantiation.
  - **Borrowed kernel files (file-path instantiation)** — the op owns *none* of its runtime kernels:

    | Kernel file (borrowed) | Owning family | Also used by |
    |---|---|---|
    | `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` | `eltwise/unary` | broadly shared (many eltwise/DM ops use the interleaved unary reader) |
    | `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | `eltwise/unary` | broadly shared |
    | `eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp` | `eltwise/unary` | shared within the eltwise/unary family |

    **Port-together set:** these kernels sit at the center of the eltwise/unary port unit; their Metal 2.0 rewrite (CB→DFB, named-token bindings) is one change all co-borrowers adopt together. Sequence this example op's port with the eltwise/unary shared-kernel rewrite, or port it to consume the already-rewritten kernels — do not fork local copies.
- **Relaxation candidates:** none (no custom hash to mine).
- **TTNN factory analysis:** `descriptor` concept, both factories; `Op-owned tensors? = no`; no MeshWorkload; no pybind `create_descriptor`; no custom hash; no custom `override_runtime_arguments`; no smuggled pointer; `Is safe to port? = yes`. Target concept **`MetalV2FactoryConcept`** (no op-owned tensors).

## Misc anomalies  *(team-only, non-gating)*

- **Unused operation attribute.** `operation_attributes_t::some_other_attribute` (`device/example_device_operation.hpp:22`) is set to `42` in `ttnn::prim::example` (`device/example_device_operation.cpp:42`) but never read by either factory or `select_program_factory` (only `attribute` is read). Dead; not hashed (no custom hash). Illustrative-only in a tutorial op — flag for awareness, no action.
- **Dead kernel files in the op directory.** `device/kernels/compute/eltwise_sfpu.cpp` and `device/kernels/dataflow/{blank,reader_binary_diff_lengths,reader_unary,writer_unary}.cpp` are unreferenced (both factories point at the `eltwise/unary` copies). Likely leftover tutorial scaffolding. Not audited; route to the ops team if cleanup is desired.

## Recipe notes

- **Example/tutorial op vs. "spreadsheet-broken" routing.** This is the canonical tutorial op and it *is* present in the readiness sheet with clean `yes` rows, so no conflict arose. Worth noting for future auditors that a documentation/example op can legitimately appear in the sheet; the "no row → spreadsheet broken → gate" rule did not need to fire here.
