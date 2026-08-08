# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding`

Single device operation sharing the directory:

- **`TilizeWithValPaddingDeviceOperation`** (`device/tilize_with_val_padding_device_operation.{hpp,cpp}`)
  - `TilizeWithValPaddingSingleCoreFactory` (`device/factories/tilize_with_val_padding_single_core_program_factory.cpp`)
  - `TilizeWithValPaddingMultiCoreDefaultFactory` (`device/factories/tilize_with_val_padding_multi_core_default_program_factory.cpp`)
  - `TilizeWithValPaddingMultiCoreBlockInterleavedFactory` (`device/factories/tilize_with_val_padding_multi_core_block_interleaved_program_factory.cpp`)
  - `TilizeWithValPaddingMultiCoreShardedFactory` (`device/factories/tilize_with_val_padding_multi_core_sharded_program_factory.cpp`)

All four factories are on the `ProgramDescriptor` API (each exposes `create_descriptor() -> ProgramDescriptor`). One `DeviceOperation`, four factories sharing a device-op + a common helper (`detail::get_packed_value`) — audited together as one porting unit.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `metal2_audit.md`.

**Recipe docs:** provenance could not be pinned — `git log -1 … docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/` printed nothing (the `metal_2.0` doc tree is not present in this checkout; the recipe was supplied as a standalone file `/localdev/edwinlee/metal2_audit.md`). Repo HEAD at audit time: `033960ede6d 2026-07-23`.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `TilizeWithValPaddingDeviceOperation` → SingleCore, MultiCoreDefault, MultiCoreBlockInterleaved, MultiCoreSharded |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** (all own + donor kernels Device-2.0 native) |
| *Prereqs* — Cross-op escapes | Ok (all donor kernels/helpers Device-2.0 compliant) |
| *Feature Support* — overall | **GREEN** (all Appendix A entries N/A) |
| *Feature Support* — GlobalCircularBuffer / address_offset / GlobalSemaphore / CTA-varargs | N/A / N/A / N/A / N/A |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** (all 4 factories) |
| *TTNN Readiness* — Concept (current) | `descriptor` (all 4) |
| *TTNN Readiness* — Secretly SPMD | N/A (no `WorkloadDescriptor` factory) |
| *TTNN Readiness* — Is safe to port? | Yes (sheet, all 4) |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — Runtime-args update | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` |
| *Port work* — Offset base pointer | none (cleared) |
| *Port work* — Tensor bindings (per binding) | Case 1 (interleaved in/out) · clean borrowed-DFB (sharded in/out) |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none (all accessors 2-arg) |
| *Port work* — CB endpoints | legal 1:1 + self-loops (per-`(CB,config)` below) |

## Result

**GREEN → brief issued.** All five gate-bearing subjects clear on every factory: Device 2.0 (all kernels native), Feature compatibility (no Appendix A feature in use), TTNN factory concept (`descriptor`, `Is able to port? == yes`), Offset base pointers (no host-folded offset), TensorAccessor 3rd argument (no 3rd-arg site). `METAL2_PORT_BRIEF.md` is emitted alongside this report.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN — `yes` for all 4 factories.** Verdict derived from `Is safe to port? == yes` AND `Concept == descriptor` AND `Custom hash == no` AND `Runtime-args update == no` AND `Pybind descriptor == no`.
  - *Concept* — cross-checked from code: every factory defines `create_descriptor(...)` returning `tt::tt_metal::ProgramDescriptor` (e.g. `tilize_with_val_padding_single_core_program_factory.cpp:22`). Sheet agrees (`descriptor`).
  - *Custom hash* — `No`. No `compute_program_hash` override anywhere in the op (grep clean; default framework hash via `ttnn::device_operation::launch`).
  - *Runtime-args update* — `No`. No `override_runtime_arguments` / `get_dynamic_runtime_args` in any factory.
  - *Pybind `create_descriptor`* — `No`. `tilize_with_val_padding_nanobind.cpp` binds only the top-level host functions `tilize_with_val_padding` / `tilize_with_zero_padding`; no `create_descriptor` binding.
  - *Op-owned tensors* — `No` (a `descriptor`-concept op has none; consistent).
  - *Is safe to port?* — `yes` (readiness-sheet owner's judgment; not re-derived). Consistent with the code: interleaved factories deliver tensor bases via the framework's `Buffer*`-binding form (patched on cache hits), and sharded factories via borrowed-memory CBs — neither is a smuggled raw-address RTA.
  - **Readiness rows** (`data_movement/tilize_with_val_padding`, `TilizeWithValPaddingDeviceOperation`), branch `edwinlee/DFB_Audits`, commit `cc7f53177e6f`:

    | Factory (variant) | Concept | Is safe to port |
    |---|---|---|
    | `TilizeWithValPaddingSingleCoreFactory` | descriptor | yes |
    | `TilizeWithValPaddingMultiCoreDefaultFactory` | descriptor | yes |
    | `TilizeWithValPaddingMultiCoreBlockInterleavedFactory` | descriptor | yes |
    | `TilizeWithValPaddingMultiCoreShardedFactory` | descriptor | yes |

- **Device 2.0 (every kernel used):** **GREEN.** Every kernel the op instantiates — own readers, donor writers, compute, and the one donor helper — uses Device-2.0 idioms (`Noc`, `DataflowBuffer`, `TensorAccessor`, `CoreLocalMem`, `UnicastEndpoint`), with only *sanctioned* free-function lookups. No legacy Device-1.0 idiom (`InterleavedAddrGen`, `ShardedAddrGen`, raw `noc_async_read/write`, `get_noc_addr_from_bank_id`, raw sem addresses) appears in any referenced kernel.

  | Kernel | Owner | Role | Device-2.0 evidence |
  |---|---|---|---|
  | `…/tilize_with_val_padding/device/kernels/dataflow/reader_unary_pad_dims_split_rows.cpp` | this op | reader (single-core) | `Noc`/`DataflowBuffer`/`TensorAccessor` (`:44,46,47`) |
  | `…/reader_unary_pad_dims_split_rows_multicore.cpp` | this op | reader (default) | `Noc`/`DataflowBuffer`/`TensorAccessor` (`:84,85,86`) |
  | `…/reader_unary_pad_multicore_both_dims.cpp` | this op | reader (block-interleaved) | `Noc`/`DataflowBuffer`/`TensorAccessor`/`tt_memmove(noc,…)` (`:91,92,93`) |
  | `…/reader_unary_pad_height_width_sharded.cpp` | this op | reader (sharded) | `Noc`/`DataflowBuffer`/`UnicastEndpoint`/`CoreLocalMem` (`:26-29,41`) |
  | `…/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | eltwise/unary (donor) | writer (single-core, default) | `Noc`/`DataflowBuffer`/`TensorAccessor`; `get_local_cb_interface(cb).fifo_page_size` (`:19`) — sanctioned |
  | `…/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_wh.cpp` | eltwise/unary (donor) | writer (block-interleaved) | `Noc`/`DataflowBuffer`/`TensorAccessor`; `get_tile_size(cb)` (`:24`) — sanctioned |
  | `…/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp` | data_movement/sharded (in-family) | writer (sharded) | `DataflowBuffer` wait_front/pop_front only (`:14-19`) |
  | `ttnn/cpp/ttnn/kernel/compute/tilize.cpp` | shared pool `ttnn/cpp/ttnn/kernel/` | compute (single/default/sharded) | `api/compute/tilize.h` + `compute_kernel_lib::tilize` (`:7,23`) |
  | `…/data_movement/tilize/device/kernels/compute/tilize_wh.cpp` | data_movement/tilize (in-family) | compute (block-interleaved) | `api/compute/tilize.h` + `compute_kernel_lib::tilize` (`:7,23`) |
  | `…/data_movement/common/kernels/common.hpp` — `tt_memmove` | data_movement/common (in-family) | helper (called by block-interleaved reader) | Noc-first overload (`:88-146`) is `noc.async_read/write` + `UnicastEndpoint` + `CoreLocalMem` |

  Note: `get_local_cb_interface(cb_id)` and `get_tile_size(cb_id)` are on the recipe's **sanctioned** CB-index free-function list (Device 2.0 keeps them), so they do not knock the writers out of Green. The block-interleaved reader calls the **non-deprecated** `Noc`-first `tt_memmove` overload (`common.hpp:89`), not the deprecated no-`noc` one (`:150`).

- **Feature compatibility:** every Appendix A entry scanned; none fires.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no `GlobalCircularBuffer` type, no `.global_circular_buffer` field, no `remote_index`/`remote_cb`/`CreateGlobalCircularBuffer` |
  | CBDescriptor `address_offset` (non-zero) | N/A | no `address_offset`/`set_address_offset`; the sharded factory uses `CBDescriptor.buffer = …` (borrowed memory, a mechanical translation), which is not this feature |
  | GlobalSemaphore | N/A | no `GlobalSemaphore`/`CreateGlobalSemaphore`; op uses no semaphores at all |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t = Tensor` (single fixed input); no kernel reads `get_compile_time_arg_val` at a runtime-varying index |

- **CB endpoints (GATE-free):** every CB is legal `1:1` or resolves via self-loop; no dead CB, no multi-binding. Per-`(CB, config)` census below (Port-work summary).
- **Offset base pointers:** **GREEN — cleared.** No address RTA folds a host-side offset into its base. The interleaved factories pass the tensor object as a `Buffer*` slot (`emplace_runtime_args({src0_buffer, …})` / `{dst_buffer, …}`) — a clean base with no `+ offset` arithmetic; the kernel reads it verbatim as `src_addr`/`dst_addr` (`get_arg_val<uint32_t>(0)`) and feeds a `TensorAccessor`. The sharded factory passes no address RTA at all (tensor bases ride borrowed-memory CBs). (Offset-base triage doc `2026-07-19_offset_base_pointers.md` not present in checkout; scan performed directly on every address RTA.)
- **TensorAccessor 3rd argument:** **GREEN — N/A.** No accessor passes a 3rd (page-size) argument; every construction is 2-arg `TensorAccessor(args, addr)` (5 sites: the 3 own readers + the 2 eltwise/unary writers). (3rd-arg triage doc `2026-07-06_tensor_accessor_3rd_arg_triage.md` not present in checkout; scan performed directly.)

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding, per factory):
  - **SingleCore / MultiCoreDefault / MultiCoreBlockInterleaved** — `input` **Case 1** (`Buffer*` RTA → kernel `src_addr` → `TensorAccessor`); `output` **Case 1** (`Buffer*` RTA → writer `dst_addr` → `TensorAccessor`).
  - **MultiCoreSharded** — `input` **clean** (borrowed-memory DFB `c_1`, `cb_src0.buffer = a.buffer()`, `…sharded_program_factory.cpp:75`); `output` **clean** (borrowed-memory DFB `c_16`, `cb_output.buffer = dst_buffer`, `:111`). Bind both via `DataflowBufferSpec::borrowed_from`.
  - Delivery detail: interleaved bases arrive via the framework's `Buffer*`-binding form (auto-registered `BufferBinding`, patched on cache hits) — routine port work, not a correctness hazard. All are fed into a `TensorAccessor` → Case 1 (express as `TensorParameter`; the `Buffer*` RTA slot + the `TensorAccessorArgs(*buf).append_to(...)` CTA plumbing both disappear). No Case 2 (no raw-pointer address arithmetic).
- **TensorParameter relaxation:** none.
- **TensorAccessor 3rd arg:** none.
- **CB endpoints** (per `(CB, config)`):

  | Factory | CB | Touchers | Disposition |
  |---|---|---|---|
  | SingleCore | `c_0` (in) | reader P / compute C | legal 1:1 |
  | SingleCore | `c_16` (out) | compute P / writer C | legal 1:1 |
  | MultiCoreDefault | `c_0` (in) | reader P / compute C | legal 1:1 |
  | MultiCoreDefault | `c_16` (out) | compute P / writer C | legal 1:1 |
  | MultiCoreBlockInterleaved | `c_1` (temp stage) | reader only | **self-loop** |
  | MultiCoreBlockInterleaved | `c_0` (in) | reader P / compute C | legal 1:1 |
  | MultiCoreBlockInterleaved | `c_16` (out) | compute P / writer C | legal 1:1 |
  | MultiCoreSharded | `c_1` (borrowed input) | reader only | **self-loop** (+ borrowed_from input buffer) |
  | MultiCoreSharded | `c_0` (staging) | reader P / compute C | legal 1:1 |
  | MultiCoreSharded | `c_2` (pad scratch) | reader only | **self-loop** |
  | MultiCoreSharded | `c_16` (borrowed output) | compute P / writer C | legal 1:1 (+ borrowed_from output buffer) |

  Self-loop CBs are single-toucher scratch/staging buffers: block-interleaved `c_1` (`reader_unary_pad_multicore_both_dims.cpp:96-99`), sharded `c_1` (borrowed input, `reader_unary_pad_height_width_sharded.cpp:31,35`) and sharded `c_2` (pad, `:33,37`). No dead CB, no multi-binding.

## Heads-ups  *(mirrors the brief)*

- **RTA varargs (FYI-P):** `reader_unary_pad_dims_split_rows_multicore.cpp:143-169` (MultiCoreDefault factory) — the per-core reader consumes a **variable-count block-representation stream** via a running `rt_arg_idx` advanced inside a loop bounded by the runtime arg `n_block_reps` (`get_arg_val<uint32_t>(4)`). This is a genuine loop-indexed RTA vararg block (recognition shape (a)); port it via the kernel-side vararg mechanism, not by naming each element. The host side that writes this stream is the RTA loop at `…multicore_program_factory.cpp:193-229`. The other three readers read only fixed-index / re-read fields → nameable, ordinary port work.
- **Cross-op / shared kernels (FYI-P):** the op instantiates several kernels it does not own; each shared file is a single Metal 2.0 rewrite that every co-borrower must adopt together (port-together set):
  - `eltwise/unary/…/writer_unary_interleaved_start_id.cpp` and `…_wh.cpp` — broadly-shared writers (used across eltwise/unary and multiple data_movement tilize/untilize ops).
  - `data_movement/sharded/…/writer_unary_sharded.cpp` — broadly-shared sharded writer.
  - `ttnn/cpp/ttnn/kernel/compute/tilize.cpp` — shared compute pool.
  - `data_movement/tilize/device/kernels/compute/tilize_wh.cpp` — in-family (sibling `tilize` op) compute.
  - `data_movement/common/kernels/common.hpp` (`tt_memmove`) — broadly-shared kernel-lib helper, function-call escape (Device-2.0 native).

## Team-only

- **Out-of-directory coupling & donor shape.** Roll-up: **✓ clean** — every function-call escape resolves to a Device-2.0-native shape.
  - **Function-call escape:** the block-interleaved reader `#include`s `cpp/ttnn/operations/data_movement/common/kernels/common.hpp` and calls `tt::data_movement::common::tt_memmove<false,false,true,0>(noc, …)`. Signature takes a leading `Noc` (`common.hpp:89`) → ✓ Device-2.0 native (no `CircularBuffer&`, no old addr-gen, no bare `sem_id`/`sem_addr`). The other own readers `#include` only `api/*` (LLK/HAL — no concern).
  - **File-path kernel instantiation (borrowed kernels):**

    | Kernel file | Owning family / pool | Broadly shared? |
    |---|---|---|
    | `eltwise/unary/…/writer_unary_interleaved_start_id.cpp` | eltwise/unary | yes (eltwise/unary + tilize/untilize family) |
    | `eltwise/unary/…/writer_unary_interleaved_start_id_wh.cpp` | eltwise/unary | yes |
    | `data_movement/sharded/…/writer_unary_sharded.cpp` | data_movement/sharded | yes (sharded writers) |
    | `ttnn/cpp/ttnn/kernel/compute/tilize.cpp` | shared pool `ttnn/cpp/ttnn/kernel/` | yes (tilize family) |
    | `data_movement/tilize/…/compute/tilize_wh.cpp` | data_movement/tilize (in-family) | yes (sibling `tilize` op) |

    None gates (all Device-2.0 compliant); coupling is a port-together sequencing note.
- **TTNN factory analysis (sheet-derived + `file:line` evidence):** target concept `MetalV2FactoryConcept` for all 4 factories (from `Concept == descriptor` + `Op-owned tensors == no`). Confirmed absent: custom hash, custom `override_runtime_arguments`, pybind `create_descriptor`, op-owned tensors.
- **Sibling / twin ops (context, not a finding):** the readiness sheet also lists `data_movement/tilize` (`TilizeDeviceOperation`, all-`descriptor`, safe) and an already-partly-`MetalV2` twin `experimental/quasar/tilize_with_val_padding`. The block-interleaved compute kernel is borrowed from the `data_movement/tilize` sibling.

## Misc anomalies  *(team-only, non-gating)*

- **Dead compute-arg read gap (cosmetic).** `reader_unary_pad_dims_split_rows_multicore.cpp` declares `aligned_page_size` as reader CTA index 5 in the factory (`…multi_core_default_program_factory.cpp:105-112`) but the kernel reads CTAs 0,1,2,3,4,6 and skips index 5 (`:67-72`) — the aligned-page-size slot is passed but unused by this kernel. Not a correctness issue (the index is still consumed positionally by `TensorAccessorArgs<7>`), just an unused arg. Route to the ops team; the port does not act on it.
- **Comment/dtype drift.** Several factory comments say "Assuming bfloat16 dataformat" (e.g. `…single_core_program_factory.cpp:59-60`) although the op now supports fp32/int32/uint32/uint16/fp8. Cosmetic; `element_size()` is used correctly.

## Recipe notes

- **Readiness sheet variant.** The Drive connector surfaced the **"DFB / Metal 2.0 Porting"** sheets (`Metal 2.0 _ Quasar Testing - DFB_Metal 2.0 Porting.csv`, id `1_cwuQCZST_J3RdTShCR4RSknhDLa3V6U`), which carry `Concept` and `Is safe to port` but **not** the fuller "Operations analysis" sheet's composite `Is able to port?` column nor the explicit `Custom hash` / `Runtime-args update` / `Pybind descriptor` columns the recipe's gate-derivation formula references by header name. I used the DFB sheet for `Concept` + `Is safe to port` and derived the remaining gate conjuncts directly from the op's code (grep-clean for custom hash / override-RTA / pybind-`create_descriptor`). Worth confirming whether the DFB sheet *is* the intended readiness source for this workflow or whether a separate "Operations analysis" sheet should be located.
- **Missing local doc tree.** The `metal_2.0` doc tree, the `analyses/` triage docs (offset-base-pointers, 3rd-arg), and `ttnn_op_porting_readiness.md` are not present in this checkout — the recipe arrived as a standalone file. The dated triage analyses could therefore not be used as priors; both scans (offset base, 3rd arg) were run directly on the code, which for this op is unambiguous. Provenance could not be pinned per the recipe's `git log` procedure (printed nothing).
