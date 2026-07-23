# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/embedding`

> Note on the target: this audit was requested against `ttnn/ttnn/operations/embedding.py`, which is only the Python golden-function shim (`attach_golden_function` plus an `EmbeddingsType` re-export). The portable op is the C++ device operation under `ttnn/cpp/ttnn/operations/embedding/`, which is what this report audits.

- **`EmbeddingsDeviceOperation`** (`ttnn::prim`, new-style device operation via `ttnn::device_operation::launch`)
  - `EmbeddingsRMProgramFactory` (`device/embeddings_rm_program_factory.cpp`) — row-major output; selected when input is row-major and `tilized == false`
  - `EmbeddingsFusedProgramFactory` (`device/embeddings_fused_program_factory.cpp`) — tilized output; selected when input is row-major and `tilized == true`
  - `EmbeddingsTilizedIndicesProgramFactory` (`device/embeddings_tilized_indices_program_factory.cpp`) — selected when the index/input tensor is `TILE_LAYOUT`

Factory selection lives in `device/embedding_device_operation.cpp:17-26`. There is a single DeviceOperation, so this is one combined report; findings are attributed per factory where they differ.

**Kernels in scope** (every kernel any factory references by `kernel_source`):

Own kernels (`device/kernels/`):
- `dataflow/embeddings.cpp` — RM reader
- `dataflow/embeddings_rm_writer_chunked.cpp` — RM writer, chunked path
- `dataflow/embeddings_tilize.cpp` — fused reader
- `dataflow/embedding_ind_tilized.cpp` — tilized-indices reader
- `dataflow/embeddings_common.hpp` — shared reader header (included by all three readers)
- `compute/tilize_chunked.cpp` — fused compute, chunked path

Donor / shared kernels (instantiated by file path; audited equally):
- `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp` — shared pool; RM non-chunked writer and tilized-indices writer
- `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/compute/tilize.cpp` — cross-family donor; fused non-chunked compute
- `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` — cross-family donor; fused non-sharded writer

No unreferenced (dead) kernel files exist in the op directory: every kernel file present is referenced by a factory.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `d28425ca5cf 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

**Readiness data:** "Operations analysis" sheet, this op's three factory rows, obtained by manual paste (the Google Drive MCP connector was not loadable in the audit session; see Recipe notes). Reflects the sheet as read on 2026-07-23.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/embedding` |
| **Overall** | **GREEN** — every gate cleared. `METAL2_PORT_BRIEF.md` issued alongside this report. |
| **DOps / Factories** | `EmbeddingsDeviceOperation` → `EmbeddingsRMProgramFactory`, `EmbeddingsFusedProgramFactory`, `EmbeddingsTilizedIndicesProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes (GREEN)** — all 9 kernels (6 own + 3 donor) are structurally Device 2.0; no holdovers |
| *Prereqs* — Cross-op escapes | Ok — function-call escapes reach only `tt_metal/*` LLK and `kernel_lib/`; 3 file-path borrowed kernels induce port-together coupling (team-only) |
| *Feature Support* — overall | **GREEN** — every Appendix A entry N/A |
| *Feature Support* — Variadic-CTA | Ok (N/A) — fixed tensor count, all CTAs read at constexpr offsets |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** — all three factory rows read `yes` (the gate cell). Cross-check matches code |
| *TTNN Readiness* — Concept (current) | `descriptor` (all three factories return `ProgramDescriptor` via `create_descriptor`; sheet agrees) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A (not a `WorkloadDescriptor` op) |
| *TTNN Readiness* — Is safe to port? | Yes (all three factories; sheet `Smuggled pointer == no`, matches the code's `Buffer*`-binding form) |
| *TTNN Readiness* — Custom hash | No (sheet + code; no `compute_program_hash` override anywhere in the op) |
| *TTNN Readiness* — Runtime-args update | `get_dynamic_runtime_args`: No (sheet + code). `PD override_runtime_args (?)`: sheet reads **yes on TilizedIndices** but code shows no such hook — non-gating discrepancy, see Gate detail |
| *TTNN Readiness* — Pybind `create_descriptor` | No (sheet + code; `embedding_nanobind.cpp` binds only `ttnn::embedding` via `bind_function`) |
| *TTNN Readiness* — Op-owned tensors | No (`descriptor` concept cannot carry them; factories return only `.cbs` / `.kernels`) |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` (no op-owned tensors) |
| *Port work* — Offset base pointer | none (GREEN) — no host-side fold; `Buffer*`-binding form + separate scalar offsets. One kernel-side heads-up (fused weights accessor, sharded output) |
| *Port work* — Tensor bindings (per binding) | input: Case 1 · weights: Case 1 · output: Case 1 (interleaved) / clean borrowed-DFB (sharded) |
| *Port work* — TensorParameter relaxation | none (no custom hash) |
| *Port work* — TensorAccessor 3rd arg | drop (Class 2) at one site: `embeddings_rm_writer_chunked.cpp:26` |
| *Port work* — CB endpoints | all legal (1:1) or self-loop; no multi-binding, no dead CB |

## Result

**GREEN → brief issued.** Every gate is cleared:

- **Device 2.0** ✓ — all 9 kernels (6 own + 3 donor) structurally Device 2.0; no holdovers.
- **Feature compatibility** ✓ — every Appendix A entry N/A.
- **TTNN factory concept** ✓ — all three factory rows read `Is able to port? == yes` (the gate cell); the cross-check of every documented conjunct matches the code.
- **Offset base pointers** ✓ — no host-side fold; `Buffer*`-binding form with separate scalar offsets.
- **TensorAccessor 3rd argument** ✓ — the single 3rd-arg site is Class 2 (redundant), a drop, not a gate.

`METAL2_PORT_BRIEF.md` is written alongside this report.

**One non-gating discrepancy, routed to the readiness-sheet owner (does not block the port):** the sheet's `Runtime-args update (?) (PD override_runtime_args)` column reads `yes` for `EmbeddingsTilizedIndicesProgramFactory` (the other two factories read `no`), but the code shows no `override_runtime_arguments` hook (nor any runtime-arg update mechanism) in that factory or any other — each factory exposes only `create_descriptor`. See Gate detail for why this does not gate.

No code-scoped subset distinction applies: all three factories clear, so the whole op is portable.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN — all three factory rows read `yes`.** The readiness sheet has one row per factory (`EmbeddingsFusedProgramFactory`, `EmbeddingsRMProgramFactory`, `EmbeddingsTilizedIndicesProgramFactory`), all under `EmbeddingsDeviceOperation`, all `Concept == descriptor`, all `Is able to port? == yes`. Each factory is a struct with a single static `create_descriptor(...)` returning `tt::tt_metal::ProgramDescriptor` (`embeddings_rm_program_factory.hpp:12-16`, `embeddings_fused_program_factory.hpp:12-16`, `embeddings_tilized_indices_program_factory.hpp:12-16`). Cross-check of the cheaply-checkable columns against code (all consistent):
  - `Concept == descriptor` — confirmed (returns `ProgramDescriptor`, not a mesh-workload; not `MetalV2`).
  - `Custom hash == no` — confirmed; no `compute_program_hash` / `compute_descriptor_program_hash` anywhere in the op.
  - `Runtime-args update (get_dynamic_runtime_args) == no` — confirmed; no such hook. (This is the column the gate formula uses.)
  - `Pybind descriptor == no` — confirmed; `embedding_nanobind.cpp:40-52` binds only `&ttnn::embedding`.
  - `Is safe to port? == yes`, `Smuggled pointer == no` — the sheet owner's correctness axis; consistent with the code's clean `Buffer*`-binding form (no `->address()` smuggling), and the sheet's own op classification "PD (pointer-patching)".
  - `Op-owned tensors? == (none)` — consistent with `descriptor` (the factories populate only `desc.cbs` and `desc.kernels`); no cross-column invariant is violated.
  - `TensorParameter relaxation == none` — consistent with the absent custom hash.

  **Non-gating code-vs-sheet discrepancy (routed to the readiness-sheet owner).** The sheet carries a second, newer runtime-args column, `Runtime-args update (?) (PD override_runtime_args)`, distinct from the `get_dynamic_runtime_args` column the gate formula uses. It reads `yes` for `EmbeddingsTilizedIndicesProgramFactory` and `no` for the other two. The code does not support this: no factory (the tilized-indices one included) has any runtime-arg override/update hook — each exposes only `create_descriptor`, and a whole-op grep for `override_runtime_arguments` / `get_dynamic_runtime_args` returns nothing. This does not gate, for three reasons: (1) the column is self-flagged uncertain by its `(?)` name; (2) it is not a conjunct of the recipe's documented `Is able to port?` formula, which uses the `get_dynamic_runtime_args` column (`no`); and (3) it does not change the sheet's own derived verdict, which is `yes`. Both the authoritative gate cell and the code agree the op is portable, so RED-ing here would be a too-conservative misroute. Routed to the readiness-sheet owner to reconcile (most likely a false positive from an automated classification pass on this one factory).

- **Device 2.0 (every kernel used):** **GREEN.** Every one of the 9 kernels is structurally Device 2.0: each data-movement kernel uses the `Noc` object (`noc.async_read` / `noc.async_write` / `*_barrier`), kernel-side `CircularBuffer` / `DataflowBuffer` wrapper objects with method-form access (`cb.reserve_back()`, `cb.get_write_ptr()`, `cb.push_back()`, `cb.wait_front()`, `cb.get_read_ptr()`, `cb.pop_front()`), `TensorAccessor` / `TensorAccessorArgs`, `CoreLocalMem`, and `UnicastEndpoint`. No `InterleavedAddrGen` / `ShardedAddrGen` / `InterleavedAddrGenFast` / `InterleavedPow2AddrGen*`, no raw `noc_async_read` / `noc_async_write`, no free-function `cb_reserve_back(cb_id)` / `get_write_ptr(cb_id)` holdovers. The compute kernels (`tilize_chunked.cpp`, donor `tilize.cpp`) are pure CB-to-CB tilize via `compute_kernel_lib::tilize` and `compute_kernel_hw_startup`, which is compute-side, not data-movement. The only free function keyed on a CB index is `get_local_cb_interface(cb_id_out)` at `writer_unary_interleaved_start_id.cpp:19`, which is **explicitly sanctioned** by the gate and is not a holdover.

  No violations table (no violations).

- **Feature compatibility:** every Appendix A entry, in order. No entry's recognition signals fire.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, no `CreateGlobalCircularBuffer`, no `.global_circular_buffer` field, no `remote_cb` / `.remote_index(` / `remote_circular_buffer.h` idiom. Output-sharded CBs use plain borrowed memory (`CBDescriptor.buffer = out_buffer`), which is the ordinary borrowed-memory pattern, not a GCB. |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `.address_offset`, no `set_address_offset`, no 4-arg `UpdateDynamicCircularBufferAddress`, no `cb_descriptor_from_sharded_tensor`. Borrowed-memory CBs are attached at base (`.buffer = out_buffer` with default zero offset). |
  | GlobalSemaphore | N/A | No `GlobalSemaphore`, no `CreateGlobalSemaphore`. The op uses **no semaphores of any kind**. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t` (`EmbeddingInputs`) is a fixed set (input, weight, optional output) — no `std::vector<Tensor>`. All kernels read CTAs at constexpr indices and via `TensorAccessorArgs<N>()`; no `get_compile_time_arg_val(i)` with a runtime-varying index. |

- **CB endpoints (GATE-free):** every CB is either a legal 1:1 or carries a port-time disposition; no multi-binding, no dead CB. The census was taken per CB, per node, per factory, per config. Device 2.0 idioms are intact, so this scan is reliable (not deferred).

  | Factory / config | CB | Touchers | Disposition |
  |---|---|---|---|
  | RM, interleaved output | c_0 (out) | reader `embeddings.cpp` produces; writer (chunked or stick-layout) consumes | 1:1 legal |
  | RM, HEIGHT_SHARDED output | c_0 (out, borrowed `.buffer=out_buffer`) | reader produces; no writer created | **self-loop** (single toucher, borrowed memory) |
  | RM, all | c_1 (index scratch) | reader only: `reserve_back(1)` + trailing `push_back(1)`, no consumer | **self-loop** (single toucher) |
  | RM, PADDED/BINARY | c_2 (weight cache) | reader only (`prepare_local_cache`) | **self-loop** (single toucher) |
  | Fused, all | c_0 (weights) | reader `embeddings_tilize.cpp` produces; compute consumes | 1:1 legal |
  | Fused, all | c_1 (index scratch) | reader only | **self-loop** (single toucher) |
  | Fused, interleaved output | c_2 (out) | compute produces; writer `writer_unary_interleaved_start_id.cpp` consumes | 1:1 legal |
  | Fused, sharded output | c_2 (out, borrowed `.buffer=out_buffer`) | compute produces; no writer created | **self-loop** (single toucher, borrowed memory) |
  | Fused, PADDED/BINARY | c_3 (weight cache) | reader only | **self-loop** (single toucher) |
  | Tilized-indices, all | c_0 (weights, reused as output CB: `output_cb_index = src0_cb_index`) | reader `embedding_ind_tilized.cpp` produces; writer consumes | 1:1 legal |
  | Tilized-indices, all | c_1 (index scratch) | reader only | **self-loop** (single toucher) |
  | Tilized-indices, PADDED/BINARY | c_2 (weight cache) | reader only | **self-loop** (single toucher) |

  No hidden second writer exists (the op uses no semaphores, so there is no semaphore-gated raw co-fill), no split-reader, and no dual-instance work-split (the fused factory's two compute descriptors cover *disjoint* core groups, so each node sees one compute instance — an ordinary 1:1, not a co-touch). Nothing here blocks a Gen1 port.

- **Offset base pointers:** **GREEN.** No address RTA folds a host-side offset into its base. The op uses the `Buffer*`-binding form throughout (it pushes `Buffer*` objects — `a_buffer`, `weights_buffer`, `output_buffer` — into the RT-arg lists; e.g. `embeddings_rm_program_factory.cpp:264-265,280`), never `buffer()->address()`. There is no `->address()` call anywhere in the op's host code, and no imperative `SetRuntimeArgs`. Every offset (batch index, byte offset, tile index, column offset, weight column-block offset) is passed as a *separate scalar* RTA, which is exactly the clean base-plus-separate-offset shape. This matches `embedding`'s absence from the offset-base-pointer triage tables (`2026-07-19_offset_base_pointers.md`).

  One kernel-side subtlety is surfaced as a heads-up (it is not this gate, because there is no host fold and no lost offset): see Heads-ups → "Fused weights accessor base offset."

- **TensorAccessor 3rd argument:** **GREEN gate** (no Class 3/4/Special). Exactly one site passes a 3rd argument:
  - `embeddings_rm_writer_chunked.cpp:26` — `TensorAccessor(dst0_args, dst_addr, output_page_size)`. Sharded-or-interleaved: this writer runs only on the RM non-sharded path (`use_chunked = !output_sharded && ...`), so the output accessor is **interleaved**. Magnitude: `output_page_size = output.padded_shape()[-1] * output_element_size_bytes` is the true logical page magnitude. An interleaved accessor realigns the passed value up to the allocator alignment, so a correct-magnitude value is inert. **Class 2 (redundant / inert) → PORT WORK: drop the arg** (a pure no-op). This agrees with `2026-07-06_tensor_accessor_3rd_arg_triage.md`, which lists `embeddings_rm_writer_chunked` as Class 2 (interleaved, correct-magnitude-but-unaligned, realigned). The op has no custom hash, so every distinct output shape recompiles; the framework's implicit `aligned_page_size` is always correct and this is not a Class 1 dynamic-page case. All other accessors in the op are 2-argument.

## Port-work summary  *(mirrors what a brief would carry once the TTNN gate clears)*

- **Tensor bindings** (per binding):
  - **input** (index tensor `a`) — **Case 1** in all three readers: the `Buffer*` base is fed into `TensorAccessor(input_args, input_buffer_src_addr)` and all access goes through the accessor (`embeddings.cpp:39`, `embeddings_tilize.cpp:35`, `embedding_ind_tilized.cpp:35`). Express as a `TensorParameter` / `TensorBinding`; kernel builds `TensorAccessor(tensor::input)`.
  - **weights** — **Case 1** in all three readers (`embeddings.cpp:40`, `embeddings_tilize.cpp:36`, `embedding_ind_tilized.cpp:36`). In the fused reader the accessor base is `weight_buffer_src_addr + weight_offset`; see the Heads-up for how the port must handle `weight_offset` under a base-only binding.
  - **output** — **Case 1** on interleaved / non-sharded output (reached via the writer's `TensorAccessor`: `embeddings_rm_writer_chunked.cpp:26`, `writer_unary_stick_layout_interleaved_start_id.cpp:20`, `writer_unary_interleaved_start_id.cpp:31`). **clean (borrowed-memory DFB)** on sharded output, where the CB is backed directly by the output buffer (`CBDescriptor.buffer = out_buffer`) and no writer runs — port via `DataflowBufferSpec::borrowed_from`. This split is per factory / per config; the tilized-indices factory has no sharded path, so its output is always Case 1.
  - No Case 2 (raw-pointer) bindings anywhere: every tensor touch goes through a `TensorAccessor`.
- **TensorParameter relaxation:** none. There is no custom hash, so no relaxation applies.
- **TensorAccessor 3rd arg:** drop the redundant page-size arg at `embeddings_rm_writer_chunked.cpp:26` (RM factory only). No `dynamic_tensor_shape` needed (Class 2, not Class 1).
- **CB endpoints:** self-loop the single-toucher CBs (all `c_1` index scratch CBs; all weight-cache CBs `c_2`/`c_3`; the borrowed-memory output CBs on sharded configs). Bind 1P+1C on the 1:1 CBs (which the ops already fix). No multi-binding flag, no dead-CB drop.

## Heads-ups  *(mirrors what a brief would carry)*

- **Fused weights accessor base offset (fused factory, sharded output).** `embeddings_tilize.cpp:36` constructs `TensorAccessor(weights_args, weight_buffer_src_addr + weight_offset)`. `weight_offset` is RTA arg 4, passed by the factory as a *separate scalar* (`embeddings_fused_program_factory.cpp:325`); it is non-zero **only** when the output is block/width-sharded (the host advances it by `weight_block_size` per core in the sharded branch, `embeddings_fused_program_factory.cpp:340-344`) and is `0` for interleaved output. Because the offset arrives as a live, separate RTA (not a host-folded base), it is *not* the Offset-base-pointer gate and it is not lost at the binding boundary. But under a `tensor::weights` binding the accessor base is fixed to the buffer base, so the port cannot add `weight_offset` to the base. The auditor-resolved fix: **route `weight_offset` into the accessor read's `offset_bytes`** (fold it together with the existing `weight_chunk_offset` that `read_token_async` already passes as `offset_bytes`). This is arithmetically identical to the current base shift — the accessor's page-to-bank mapping depends only on `page_id` and the aligned page size, so a flat byte addend produces the same final address whether applied to the base or to `offset_bytes`, and `weight_offset` stays within one weight page. A naive Case-1 port that just writes `TensorAccessor(tensor::weights)` would silently drop `weight_offset` and mis-address sharded output, so this must not be missed.
- **CB endpoints (multi-binding shapes to watch):** none. No hidden second writer, no multi-reader, no dual-instance work-split.
- **Cross-op / shared kernels:** three kernels are instantiated by file path and are shared beyond this op; their Metal 2.0 rewrite must be coordinated across all co-borrowers (see Team-only → Out-of-directory coupling for the port-together sets). Device 2.0 for all three is already GREEN.
- **RTA varargs:** none. Every kernel reads its runtime args at fixed indices (the `PADDED` pad-token is a single fixed trailing arg, not a loop-indexed or data-selected read). All args port to named runtime args.

## Team-only

- **Out-of-directory coupling & donor shape.**
  - **Op-level roll-up: `✓ clean`** for function-call escapes. The `#include`s outside the op directory resolve only to `tt_metal/*` LLK headers (`api/dataflow/*`, `api/compute/*`, `api/core_local_mem.h`, `api/tensor/noc_traits.h`, `api/debug/dprint.h`) and to `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` (the official shared kernel library, lib-team-owned). The only cross-boundary function calls are `compute_kernel_lib::tilize<...>` and `compute_kernel_hw_startup(...)` (compute-side, no resource-handle shapes to translate). No op's kernels call another op family's helper functions.
  - **Borrowed kernel files (file-path instantiation).** These do not gate but induce port-the-family-together coupling — the shared kernel's Metal 2.0 rewrite (CB to DFB, named-token bindings) is a single change every co-borrower must adopt together:

    | Borrowed kernel | Owning pool / family | Also instantiated by (port-together set) |
    |---|---|---|
    | `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp` | shared pool `ttnn/cpp/ttnn/kernel/dataflow/` | `data_movement/concat`, `data_movement/slice`, `embedding` |
    | `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/compute/tilize.cpp` | `data_movement/tilize` | `data_movement/tilize`, `tilize_with_val_padding`, `untilize`, `untilize_with_unpadding`, `moreh/moreh_getitem`, `pool/upsample`, `sliding_window/halo`, `experimental/deepseek_prefill/combine`, `experimental/quasar/tilize_with_val_padding`, `embedding` |
    | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | `eltwise/unary` | broadly shared — ~22 op families (matmul, kv_cache, reduction, many `data_movement/*` and `eltwise/*`, `embedding`, and more) |

- **Relaxation candidates (mined from a custom hash):** none — the op has no custom hash.
- **TTNN factory analysis (sheet-derived facts, with code evidence):** current concept `descriptor` (all three factories); no op-owned tensors; no pybind `create_descriptor`; no custom hash; no `get_dynamic_runtime_args` hook; `Is safe to port? == yes` and `Smuggled pointer == no` (sheet owner's calls). Target concept `MetalV2FactoryConcept`. Sheet `Model` column: `llama` (the op is exercised by llama models — informational). The one discrepancy is the `PD override_runtime_args (?)` column on the tilized-indices factory (sheet `yes`, code shows no such hook); see Gate detail.

## Misc anomalies  *(team-only, non-gating, not porter-actionable)*

- **Likely latent bug: wrong pad-token arg index in the tilized-indices reader.** `embedding_ind_tilized.cpp:42` calls `prepare_local_cache(..., /*pad_token_arg_idx=*/6)`, but this reader has seven base runtime args (0-6), and its factory places the pad token at arg **7** (`embeddings_tilized_indices_program_factory.cpp:208-218`). Arg 6 is `starting_index` (`col_offset % FACE_HEIGHT`, `embedding_ind_tilized.cpp:23`). So on the `PADDED` + TILE-layout-input path, `prepare_local_cache` reads `starting_index` as the pad token instead of the real pad token. The other two readers get this right: both have six base args (0-5) with the pad token at arg 6, matching the `pad_token_arg_idx = 6` they pass (`embeddings.cpp` and `embeddings_tilize.cpp`). The path is reachable by construction (selection routes any TILE-layout index tensor to this factory regardless of `embeddings_type`, and validation does not forbid `PADDED` there), though it may be untested. Routes to the ops team.
- **Dead compile-time arg in the stick-layout writer path.** The RM non-chunked writer and the tilized-indices writer both pass `output_page_size` as CTA index 1 (`embeddings_rm_program_factory.cpp:230-231`, `embeddings_tilized_indices_program_factory.cpp:169`), but `writer_unary_stick_layout_interleaved_start_id.cpp` reads `cb_id_out0` at CTA 0 and its accessor args from `TensorAccessorArgs<2>()`, never reading CTA 1; the page size it uses comes from RTA 1 (`stick_size`). So CTA 1 is a passed-but-unread arg for these callers. It sits in a shared donor kernel, so the intent may be historical; noted for the ops team, not for the port diff.
- **Dead macro.** `embedding_device_operation.cpp:13` defines `#define RISC_CORES_PER_TENSIX 2`, which is never used anywhere in the op.
- **Vestigial debug include.** `embedding_ind_tilized.cpp:11` includes `api/debug/dprint.h` but the kernel contains no `DPRINT` usage.

## Questions for the user

1. **Fused sharded-output weight offset (confirm the port approach):** the Heads-up proposes moving `weight_offset` from the fused weights-accessor base into the read's `offset_bytes` when porting (arithmetically identical, keeps the base clean for the binding). Please confirm this is acceptable as ordinary port work, or whether the framework owner wants to weigh in given it is a kernel-side offset relocation rather than a pure token swap. (Non-blocking: the brief is issued regardless; this is a confirmation, not a gate.)

## Recipe notes

- **Separate-offset arg fed to an accessor base is a gap between two subjects.** The recipe's Offset-base-pointer gate keys on a *host-side* fold (`buffer()->address() + offset` in the factory), and the TensorParameter-analysis Case 1 assumes the accessor base is exactly the clean buffer base. The fused reader here is neither: the host passes a clean `Buffer*` base and a *separate* scalar `weight_offset`, and the kernel adds them when constructing the accessor (`TensorAccessor(args, base + weight_offset)`). By the literal Offset-base recognition this is correctly GREEN (no fold, offset not lost), but a naive Case-1 port would still drop the offset. It would help to have the recipe explicitly name this "separate-offset arg added to an accessor base" shape under Case 1 (resolution: relocate the offset to the accessor read's `offset_bytes`), so a future auditor/porter is not left to infer that the clean-base Case-1 wording does not cover it. Cited: `metal2_audit.md` TensorParameter analysis, Case 1; Offset base pointers, Type 2 recognition.
- **Unreachable readiness sheet (vs. a broken one), and a working fallback.** The recipe handles a sheet that is present-but-wrong or missing-a-row ("spreadsheet is broken" GATE), and `ttnn_op_porting_readiness.md` covers connector-authorization troubleshooting, but neither states the intended *verdict* when the sheet cannot be fetched at all. In this audit the Google Drive MCP connector was not loadable (its tool was absent from the session's registry even after the human authorized it mid-session, because MCP tools are enumerated at session start; a direct CSV-export `WebFetch` returned HTTP 401 on the restricted sheet). The fallback that worked was the human pasting this op's three factory rows into chat, which the auditor then cross-checked by header name exactly as if fetched. Suggest the recipe name "human pastes the row(s)" as an explicit sanctioned fallback for a non-interactive or unauthorized session, and state that all other subjects should still be completed when the code is settled. Cited: `metal2_audit.md` TTNN factory concept prerequisite, "Fetch and locate"; `ttnn_op_porting_readiness.md` Troubleshooting.
- **The sheet has two runtime-args-update columns; the recipe/legend describe one.** The live sheet carries both `Runtime-args update (get_dynamic_runtime_args)` and a newer `Runtime-args update (?) (PD override_runtime_args)`, but the recipe's `Is able to port?` derivation and the `ttnn_op_porting_readiness.md` legend mention only a single "Runtime-args update" column (the `get_dynamic_runtime_args` one). This forced a judgment call here: the `(?)` column read `yes` on one factory while the code showed no override hook, and the recipe gives no guidance on whether that auxiliary, self-flagged-uncertain column should be cross-checked or can gate. This audit treated it as non-gating (not a formula conjunct; verdict unaffected) and routed the discrepancy to the sheet owner. A line in the recipe clarifying which runtime-args column feeds the gate, and how to treat a `(?)`-marked column, would remove the ambiguity. Cited: `metal2_audit.md` TTNN factory concept prerequisite, the `Is able to port?` derivation; `ttnn_op_porting_readiness.md` "Reading the CSV".
