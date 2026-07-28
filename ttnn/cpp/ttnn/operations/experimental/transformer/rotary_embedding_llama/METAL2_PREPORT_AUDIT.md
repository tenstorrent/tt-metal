# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama`

One DeviceOperation sharing all three program factories (shared kernels; audited together as a single unit):

- **`RotaryEmbeddingLlamaDeviceOperation`**
  - `RotaryEmbeddingLlamaMultiCore` (`rotary_embedding_llama_multi_core_program_factory.cpp`) — interleaved prefill
  - `RotaryEmbeddingLlamaMultiCorePrefillSharded` (`rotary_embedding_llama_multi_core_prefill_sharded_program_factory.cpp`) — prefill with sharded cos/sin/trans_mat (hybrid borrowed-memory + reload)
  - `RotaryEmbeddingLlamaMultiCoreSharded` (`rotary_embedding_llama_sharded_program_factory.cpp`) — fully-sharded decode

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `156b384a2cf 2026-07-28 docs(metal_2.0): put experimental/quasar out of bounds for production ports`

Kernels referenced (all owned by this op, all Device 2.0 idioms):
- `device/kernels/dataflow/reader_rotary_embedding_llama_interleaved_start_id.cpp` (factory 1)
- `device/kernels/dataflow/reader_rotary_embedding_llama_prefill_sharded.cpp` (factory 2)
- `device/kernels/dataflow/writer_rotary_embedding_llama_interleaved_start_id.cpp` (factories 1 & 2)
- `device/kernels/compute/rotary_embedding_llama.cpp` (factories 1 & 2)
- `device/kernels/compute/rotary_embedding_llama_sharded.cpp` (factory 3)

No unreferenced kernel files in the directory.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `RotaryEmbeddingLlamaDeviceOperation` → `RotaryEmbeddingLlamaMultiCore`, `RotaryEmbeddingLlamaMultiCorePrefillSharded`, `RotaryEmbeddingLlamaMultiCoreSharded` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all 5 kernels are structurally Device 2.0 (`Noc`, `CircularBuffer` wrappers, `TensorAccessor`); only free function is sanctioned `get_tile_size(cb_id)` |
| *Prereqs* — Cross-op escapes | **Ok** — op owns every kernel; all `#include`s are `api/*` (tt_metal LLK/HAL) |
| *Feature Support* — overall | **GREEN** (all Appendix A entries N/A) |
| *Feature Support* — Variadic-CTA | Ok — all CTAs read at constexpr indices |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** — all three factory rows |
| *TTNN Readiness* — Concept (current) | `descriptor` (all three) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Is safe to port? | Yes (all three) |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No |
| *TTNN Readiness* — `override_runtime_arguments` | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` (no op-owned tensors) |
| *Port work* — Offset base pointer | none |
| *Port work* — Tensor bindings (per binding) | clean (borrowed-DFB) + Case 1 (`TensorAccessor`); config/factory-dependent — see below |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none (no site passes a 3rd argument) |
| *Port work* — CB endpoints | self-loop + legal 1:1 only; no multi-binding, no dead CBs |

**CB endpoints** are dispositions, not gates. Every out-of-window CB here resolves to a **self-loop** (single toucher) — no 1P+1C assignments, no multi-binding flags, no dead-CB drops are needed. Dispositions recorded per `(CB, config)` below.

## Result

**GREEN → brief issued.** All five gate-bearing subjects clear: Device 2.0 ✓ · Feature compatibility ✓ · TTNN factory concept ✓ (all three factories, `Is able to port? == yes`) · Offset base pointers ✓ · TensorAccessor 3rd arg ✓. The op is a modern `descriptor`-concept TMP op that already uses `Buffer*`-binding + `TensorAccessorArgs` on its dataflow path and `CBDescriptor::buffer` borrowed-memory on its sharded paths — no blockers. Port work is confined to expressing tensor bindings and dropping the interim `Buffer*`/RTA plumbing; see `METAL2_PORT_BRIEF.md`.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN** — the readiness sheet's `Is able to port?` == `yes` for all three factory rows, and the lightweight cross-check is clean:
  - `Concept == descriptor` for all three — confirmed: each factory defines `static ProgramDescriptor create_descriptor(...)` (`*_program_factory.hpp:19/19/22`), no `create()`/`override_runtime_arguments()`, no mesh-workload return.
  - `Custom hash == no` — confirmed: no `compute_program_hash` override in the device-op (grep clean).
  - `get_dynamic_runtime_args == no` — confirmed: no such hook (grep clean).
  - `override_runtime_arguments == no` — confirmed: no such method (grep clean).
  - `Pybind descriptor == no` — confirmed: `rotary_embedding_llama_nanobind.cpp:18` binds the plain function `ttnn::experimental::rotary_embedding_llama`, no `nb::class_` of the device op / `create_descriptor`.
  - `Op-owned tensors == no`, `Secretly SPMD == N/A` — consistent with `descriptor` concept.
  - **Factory-set match:** exactly three sheet rows ↔ three code factories, one-to-one; no phantom or missing row.
  - Cross-column invariants hold (no `get_dynamic_runtime_args` on a non-descriptor; no op-owned tensors on a `descriptor`).
- **Device 2.0 (every kernel used):** **GREEN.** All five kernels are structurally Device 2.0:
  - `Noc noc;` object + `noc.async_read` / `noc.async_write` / `noc.async_read_barrier` / `noc.async_write_barrier` (dataflow kernels).
  - `CircularBuffer cb_x(cb_id);` wrapper objects; all FIFO ops and pointer peeks go through wrapper methods (`.reserve_back`, `.push_back`, `.wait_front`, `.pop_front`, `.get_write_ptr()`, `.get_read_ptr()`). The `get_*_ptr(` grep hits are all wrapper-method calls, **not** CB-index free-function holdovers.
  - `TensorAccessor` / `TensorAccessorArgs<N>` for all DRAM access; `CoreLocalMem<uint32_t>` for L1.
  - Only CB-index free function used is `get_tile_size(cb_id)` (reader `:46,49,52,70`; writer `:36,37`; prefill reader `:48,74,85,86,142,145`) — **sanctioned** by the Device 2.0 Green bullet; not a holdover.
  - No `noc_async_read`/`noc_async_write` free functions, no `InterleavedAddrGen`/`ShardedAddrGen`, no raw semaphore addresses, no `get_noc_addr_from_bank_id`.

  | File | Line | Call | Wrapper in scope |
  |---|---|---|---|
  | *(none — no Device 2.0 violations)* | — | — | — |

- **Feature compatibility:** every Appendix A entry is **N/A** — none present.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, no `CreateGlobalCircularBuffer`, no `remote_index`/`remote_cb`, no `CBDescriptor.global_circular_buffer` field set. CBs use plain `.buffer =` (borrowed-memory), which is the supported path. |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `.address_offset`, `set_address_offset`, 4-arg `UpdateDynamicCircularBufferAddress`, or `cb_descriptor_from_sharded_tensor` (grep clean). |
  | GlobalSemaphore | N/A | Op uses **no** semaphores at all (grep clean). |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t` is a fixed 4-tensor struct; every `get_compile_time_arg_val(N)` reads a constexpr literal index; `TensorAccessorArgs<N>` offsets are constexpr. No runtime-varying CTA loop. |

- **CB endpoints (GATE-free):** every CB is legal 1:1 or a **self-loop**; nothing blocks a Gen1 port. Dispositions per `(CB, config)`:

  **`RotaryEmbeddingLlamaMultiCore` (interleaved):** kernels = reader + writer + compute.
  - `c_0` input, `c_1` cos, `c_2` sin, `c_3` trans_mat → **legal 1:1** (reader produces, compute consumes).
  - `c_16` output → **legal 1:1** (compute produces, writer consumes).
  - `c_24` rotated-interm, `c_25` cos-interm, `c_26` sin-interm → **self-loop** (compute is the only toucher: reserve/push then wait/pop).
  - `c_27` zero → **self-loop** (writer is the only toucher: reserve + raw zero-write via `get_write_ptr` + push, then wait/read/pop; `writer:43-79`).

  **`RotaryEmbeddingLlamaMultiCorePrefillSharded`:** kernels = reader + writer + compute. Same shape as interleaved, with config-dependent borrowed-memory binding on cos/sin/trans_mat:
  - `c_0` input → **legal 1:1** (reader→compute; never borrowed here).
  - `c_1` cos, `c_2` sin → **legal 1:1** (reader produces, compute consumes) in every config; backed by `CBDescriptor::buffer` (borrowed) in the sharded fast path (`:175,186`) and plain FIFO otherwise. In the partial-shard-grid case a second CBDescriptor with the same `buffer_index` covers the remaining cores (`:190-211`) — same node-level census.
  - `c_3` trans_mat → **legal 1:1** (reader→compute); borrowed (`.buffer`, `:260`) in the `trans_mat_use_global_cb` path.
  - `c_24/c_25/c_26` interm → **self-loop** (compute); `c_16` output → **legal 1:1** (compute→writer); `c_27` zero → **self-loop** (writer).

  **`RotaryEmbeddingLlamaMultiCoreSharded` (decode):** kernel = compute only. All eight CBs (`c_0`,`c_1`,`c_2`,`c_3`,`c_24`,`c_25`,`c_26`,`c_16`) are touched by the single compute kernel as both producer and consumer → **self-loop** for every CB. `c_0`/`c_1`/`c_2`/`c_3`/`c_16` are borrowed-memory (`.buffer =`, `:87,99,111,125,171`); the self-loop is legal on Gen1 for a compute kernel.

- **Offset base pointers:** **GREEN** — no address RTA folds a host-side offset into a base. The op never calls `buffer()->address()` (grep clean); the dataflow factories push raw `Buffer*` objects into runtime args (the framework-patched `BufferBinding` interim form: `multi_core:337-338`, `prefill_sharded:472-475`) and build `TensorAccessorArgs` in CTAs — clean bases, no `base + offset` arithmetic. Not in the offset-base-pointers triage doc (consistent).

- **TensorAccessor 3rd argument:** **GREEN** — no accessor passes a 3rd argument. Every construction is the two-arg `TensorAccessor(args, addr)` (reader `:47,50,53,55`; prefill reader `:49,69,87,88,143,146`; writer `:38`). The 3rd-arg triage doc lists the unrelated generic ops `rotary_embedding` / `rotary_embedding_hf`, **not** `rotary_embedding_llama` — and my scan confirms no 3rd-arg site exists here. Subject does not fire.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding, per factory):
  - **`RotaryEmbeddingLlamaMultiCore` (interleaved):** `input`, `cos`, `sin`, `trans_mat`, `output` → all **Case 1** (`TensorAccessor`). Addresses arrive today as `Buffer*` RTAs (`:337-338`) fed to `TensorAccessor(args, addr)` in the kernels; the port replaces these with `TensorParameter`/`TensorBinding` and `TensorAccessor(tensor::name)`, dropping the `Buffer*` RTAs and the `TensorAccessorArgs` CTA plumbing.
  - **`RotaryEmbeddingLlamaMultiCoreSharded` (decode):** `input`, `cos`, `sin`, `trans_mat`, `output` → all **clean (borrowed-DFB)**. Each binds via `CBDescriptor::buffer`; the port expresses them with `DataflowBufferSpec::borrowed_from`. No `TensorAccessor`, no RTAs.
  - **`RotaryEmbeddingLlamaMultiCorePrefillSharded` (hybrid, config-dependent):** `input` → **Case 1**; `output` → **Case 1**; `cos`/`sin` → **clean (borrowed-DFB)** in the sharded fast path, **Case 1** in the reload/interleaved path; `trans_mat` → **clean (borrowed-DFB)** in the global-CB path, **Case 1** otherwise. The port's binding for cos/sin/trans_mat must support both shapes (borrowed-memory on the shard grid + accessor-read fallback), matching the existing same-`buffer_index` merged-CB construct.
  - No **Case 2** (raw-pointer) bindings anywhere — every non-clean access flows through a `TensorAccessor`.
- **TensorParameter relaxation:** none (sheet `none`; no custom hash).
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** self-loop `c_24/c_25/c_26` (compute) and `c_27` (writer) in factories 1 & 2; **all** CBs self-loop in factory 3 (decode); everything else legal 1:1. No 1P+1C assignments, no multi-binding flags, no dead-CB drops.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none — no hidden second writer, no multi-reader, no ≥3-toucher CB in any factory/config. (All non-1:1 CBs are single-toucher self-loops.)
- **Cross-op / shared kernels:** none — the op owns all five kernels; every `#include` resolves to `api/*` (tt_metal LLK/HAL). No borrowed kernel files, no `_metal2` fork to reuse or create.
- **Prefill-sharded merged CBs:** the prefill-sharded factory emits **multiple `CBDescriptor`s sharing one `buffer_index`** over disjoint core ranges (shard-grid cores get a borrowed-memory CB; remaining cores get a small non-borrowed CB) for `c_1`/`c_2` (`:167-211`) and `c_3` (`:247-288`). The port must preserve this per-core-range split when expressing the DFB.
- **RTA varargs:** none — every kernel reads a fixed run of named args via a top-of-kernel `argrt++` counter (reader 8, writer 5, compute 4); all nameable, no loop-indexed or data-selected reads.

## Team-only

- **Out-of-directory coupling & donor shape:** roll-up **✓ clean.** No function-call escapes and no file-path kernel instantiation outside the op. All kernel `#include`s are donor class 1 (`tt_metal/*` via `api/*`): `api/dataflow/{dataflow_api,noc,circular_buffer}.h`, `api/core_local_mem.h`, `api/tensor/noc_traits.h`, `api/compute/*`. No summary table / per-call detail needed (all ✓).
- **Relaxation candidates** (mined from a custom hash on a gated op): N/A — no custom hash, op not gated.
- **TTNN factory analysis:** current concept `descriptor` (all three factories); no op-owned tensors; no MeshWorkload; no pybind `create_descriptor`; no custom hash; no `get_dynamic_runtime_args`; no `override_runtime_arguments`; `Is safe to port? == yes` (no smuggled pointer). Target concept `MetalV2FactoryConcept`. `Op Classification` on the sheet is `PD (pointer-patching)` — the `Buffer*`-binding interim form, consistent with the dataflow factories' RTA shape.

## Misc anomalies  *(team-only, non-gating)*

- **Stale struct comment** — `multi_core_program_factory.cpp:295` comments the per-core args as `{start_batch, end_batch, start_seq, end_seq, active}`, but the `CoreArgs` struct (`:296-301`) has only four fields; there is no `active` field. Cosmetic comment drift; the code skips idle cores via the `continue` at `:312`. Route to ops team; the port does not act on it.
- **Redundant `matmul_init`** — `compute/rotary_embedding_llama_sharded.cpp` calls `matmul_init(in_cb, trans_mat_cb)` both once before the loop (`:47`) and again on every iteration inside it (`:77`). The pre-loop call is redundant. Harmless; not port work.

## Recipe notes

- **Hybrid per-factory / per-config binding classification.** The prefill-sharded factory is the one place the recipe's "clean vs Case 1" split does not resolve to a single verdict for a binding — `cos`/`sin`/`trans_mat` are borrowed-memory DFBs in one config and `TensorAccessor` reads in another *within the same factory* (not just across sibling factories, which the [TensorParameter analysis](metal2_audit.md) granularity note anticipates). The report captures this as a per-config split under one factory; flagging in case the recipe wants an explicit "config-varying binding within a single factory" idiom rather than leaning on the per-factory attribution mechanism.
