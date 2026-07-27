# Metal 2.0 Audit Findings — `experimental/transformer/rotary_embedding_llama_fused_qk`

- **`RotaryEmbeddingLlamaFusedQKDeviceOperation`**
  - `RotaryEmbeddingLlamaFusedQKProgramFactory` (`rotary_embedding_llama_fused_qk_program_factory.cpp`)
    - one factory, one `create_descriptor`; selects one of two compute-kernel **source files** at runtime via `row_major_QK` (tiled: `device/kernels/compute/rotary_embedding_llama_sharded.cpp`; row-major: `.../rotary_embedding_llama_sharded_row_major.cpp`). Same CB layout, same descriptor shape — one porting unit.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama_fused_qk` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `RotaryEmbeddingLlamaFusedQKDeviceOperation` → `RotaryEmbeddingLlamaFusedQKProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — one compute kernel (2 source variants); `CircularBuffer` wrappers throughout; no DM kernels |
| *Prereqs* — Cross-op escapes | Ok — LLK includes only; op owns both kernel sources |
| *Feature Support* — overall | **GREEN** |
| *Feature Support* — Variadic-CTA | Ok — CTAs read at fixed constexpr offsets `0..12`; `tensor_args_t` is a fixed 5-tensor struct |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Is safe to port? | Yes |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — Runtime-args update | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No (nanobind uses `ttnn::bind_function`) |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` |
| *Port work* — Offset base pointer | none |
| *Port work* — Tensor bindings (per binding) | clean (all 7 are borrowed-memory DFB) |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | N/A — no `TensorAccessor` in the op |
| *Port work* — CB endpoints | self-loop (all 10 CBs — single toucher) |

**CB endpoints** are dispositions, not gates. Every CB here is touched by exactly one kernel (the sole compute kernel), so every CB is a single-ended **self-loop** — no multi-binding, no dead CB, no 1P+1C.

## Result

**GREEN → brief issued.** All five gates clear:

- **Device 2.0** — the op has no data-movement kernels; its only kernels are the compute kernel's two source variants, both structurally Device 2.0 (`CircularBuffer` wrappers for every FIFO op, `api/dataflow/circular_buffer.h`, no legacy Device-1.0 idioms, no CB-index free-function holdovers).
- **Feature compatibility** — all Appendix A entries `N/A`.
- **TTNN factory concept** — readiness sheet `Is able to port? = yes`; concept `descriptor`; cross-check clean.
- **Offset base pointers** — no address RTAs at all; nothing to fold.
- **TensorAccessor 3rd arg** — no `TensorAccessor` in the op.

No blockers; no subset scoping needed. Target concept `MetalV2FactoryConcept`.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** Readiness sheet row (op `experimental/transformer/rotary_embedding_llama_fused_qk`, DOp `RotaryEmbeddingLlamaFusedQKDeviceOperation`, factory `RotaryEmbeddingLlamaFusedQKProgramFactory`): `Is able to port? = yes`. Derivation, all satisfied: `Is safe to port? = yes` · `Custom hash = no` · `Runtime-args update (get_dynamic_runtime_args) = no` · `Runtime-args update (PD override_runtime_args) = no` · `Pybind descriptor = no` · `Concept = descriptor`. Cross-check against the code — all confirmed:
  - `Concept = descriptor` — `create_descriptor(...)` returns a `ProgramDescriptor` (`rotary_embedding_llama_fused_qk_program_factory.cpp:18`, `.hpp:18`).
  - `Custom hash = no` — no `compute_program_hash` override anywhere in the op (grep clean).
  - `Runtime-args update = no` — no `get_dynamic_runtime_args` / `override_runtime_arguments` (grep clean; the DeviceOperation defines only `validate_on_program_cache_miss`, `compute_output_specs`, `create_output_tensors`).
  - `Pybind descriptor = no` — `rotary_embedding_llama_fused_qk_nanobind.cpp:18` binds the public op via `ttnn::bind_function`, not a `create_descriptor` nanobind.
  - `Smuggled pointer = no` — no `->address()` in the op (grep clean).
  - Cross-column invariants hold (no runtime-args-update / op-owned tensors on a `descriptor` row). Sheet internally consistent.

- **Device 2.0 (every kernel used):** **GREEN.** The factory instantiates a single `KernelDescriptor` — a compute kernel (`rotary_embedding_llama_fused_qk_program_factory.cpp:238-264`), from one of two source files chosen by `row_major_QK` (`.cpp:231-236`). There are **no reader/writer/dataflow kernels**: all tensor I/O is via borrowed-memory CBs bound through `CBDescriptor::buffer`. Both compute sources:
  - manage every CB FIFO through the Device-2.0 `CircularBuffer` wrapper object (`in_cb_obj.reserve_back/push_back/wait_front/pop_front`, etc. — `rotary_embedding_llama_sharded.cpp:59-66,73-129`; `..._row_major.cpp:59-63,70-121`);
  - include the wrapper header `api/dataflow/circular_buffer.h` (`:12` in both);
  - contain **no** legacy Device-1.0 data-movement idioms (no `noc_async_read/write`, `get_noc_addr*`, `InterleavedAddrGen`/`ShardedAddrGen`, raw `get_read_ptr`/`get_write_ptr`, `cb_reserve_back`/`cb_push_back`/`cb_wait_front`/`cb_pop_front` free functions, or raw sem addresses — grep clean);
  - the only non-object CB references are `cb_id` (`uint32_t`) passed to **compute-math LLK primitives** (`compute_kernel_hw_startup`, `matmul_init`, `matmul_tiles`, `binary_op_init_common`, `mul_tiles`, `mul_tiles_bcast`, `add_tiles`, `pack_tile`). These are the compute API surface, not data-movement free functions; Device 2.0 defines no wrapper-method replacement for them, so they are not holdovers and do not affect this gate.

- **Feature compatibility:** all Appendix A entries `N/A`.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no `GlobalCircularBuffer` type, no `.global_circular_buffer` field, no `remote_index`/`remote_cb`, no 4-arg `CreateCircularBuffer` (grep clean) |
  | CBDescriptor `address_offset` (non-zero) | N/A | no `address_offset` set anywhere; all `CBDescriptor`s omit the field (default 0) — grep clean |
  | GlobalSemaphore | N/A | no `GlobalSemaphore`; the op declares no semaphores at all |
  | Variable-count compile-time arguments (CTA varargs) | N/A | kernels read `get_compile_time_arg_val(0..12)` at fixed constexpr offsets; no runtime-varying CTA index; `tensor_args_t` is a fixed 5-tensor struct, not a `std::vector<Tensor>` |

- **CB endpoints (GATE-free):** all 10 CBs are single-toucher → **self-loop**. There is exactly one kernel in the program (the compute kernel), so no CB can have more than one distinct toucher per node — multi-binding and hidden-second-writer are structurally impossible here, and there are no dead CBs (every `buffer_index` is referenced by the kernel). Census (per node, one config family; disposition uniform):
  - `c_0` q_input (borrowed) — compute self-loops via `in_cb` on q-cores → **self-loop**
  - `c_1` k_input (borrowed) — compute self-loops via `in_cb` on k-cores → **self-loop**
  - `c_2` cos (borrowed, read-only) — compute reads via `mul_tiles`/`mul_tiles_bcast` → **self-loop** (may bind consumer-only)
  - `c_3` sin (borrowed, read-only) — compute reads → **self-loop** (may bind consumer-only)
  - `c_4` trans_mat (borrowed, read-only) — compute reads via `matmul_tiles` → **self-loop** (may bind consumer-only)
  - `c_16` q_output (borrowed) — compute produces via `out_cb` on q-cores → **self-loop**
  - `c_17` k_output (borrowed) — compute produces via `out_cb` on k-cores → **self-loop**
  - `c_24` rotated_input_interm (local) — compute produces+consumes → **self-loop**
  - `c_25` cos_interm (local) — compute produces+consumes → **self-loop**
  - `c_26` sin_interm (local) — compute produces+consumes → **self-loop**

  Per-node runtime nuance (not a finding): the input/output CBs (`c_0`/`c_1`, `c_16`/`c_17`) are allocated over `all_cores_bb` but each is exercised only on its q- or k-core subset (the kernel's `is_q` runtime arg selects the branch). The `buffer_index` is still referenced by the kernel, so none is a dead CB; the self-loop binding is valid over the CB's full core range and inert where the branch isn't taken — identical to legacy behavior.

- **Offset base pointers:** **GREEN.** No address RTA folds a host offset into a base — the op passes **no** `->address()` through any runtime arg (grep clean; the sharded tensors reach the kernel through `CBDescriptor::buffer` borrowed-memory bindings, not address RTAs). Not present in the offset-base-pointer triage analysis (`2026-07-19_offset_base_pointers.md`), consistent with the scan. No Type 1/2/3/4 site.

- **TensorAccessor 3rd argument:** **N/A.** The op constructs no `TensorAccessor` (grep clean) — there are no dataflow kernels and no accessor-based addressing. The 3rd-arg triage (`2026-07-06_tensor_accessor_3rd_arg_triage.md`) lists the sibling ops `rotary_embedding` and `rotary_embedding_hf` (both Class 2), but **not** this fused-QK op; the code confirms no accessor site exists here, so nothing to classify or drop.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding): all seven tensors are **clean** (borrowed-memory DFB) — bound through `CBDescriptor::buffer` and touched by the compute kernel via CB FIFO / compute-math reads, never through an `->address()` RTA or a `TensorAccessor`. Port via `DataflowBufferSpec::borrowed_from`. Bindings:
  - `q_input` → `c_0` (`buffer = q_src_buffer`, `program_factory.cpp:103`) — clean
  - `k_input` → `c_1` (`buffer = k_src_buffer`, `:115`) — clean
  - `cos` → `c_2` (`buffer = cos_buffer`, `:127`) — clean
  - `sin` → `c_3` (`buffer = sin_buffer`, `:139`) — clean
  - `trans_mat` → `c_4` (`buffer = trans_mat_buffer`, `:153`) — clean
  - `q_output` → `c_16` (`buffer = q_dst_buffer`, `:199`) — clean
  - `k_output` → `c_17` (`buffer = k_dst_buffer`, `:210`) — clean
- **TensorParameter relaxation:** none (sheet `TensorParameter relaxation = none`; no custom hash).
- **TensorAccessor 3rd arg:** none — no accessor in the op.
- **CB endpoints:** self-loop all 10 CBs (single toucher). The 3 local interm CBs (`c_24`/`c_25`/`c_26`) are ordinary local DFBs; the 7 tensor CBs are borrowed-memory DFBs (`borrowed_from`). No dead-CB drop, no multi-binding flag, no 1P+1C.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none — a single-kernel program cannot produce a multi-binding or a hidden second writer.
- **Cross-op / shared kernels:** none. Both kernel sources live in the op's own `device/kernels/compute/` and are instantiated by file path; no borrowed kernel files, no shared-pool instantiation.
- **RTA varargs:** none. The sole runtime arg is a single fixed scalar `is_q = get_arg_val<uint32_t>(0)` (`rotary_embedding_llama_sharded.cpp:29`, `..._row_major.cpp:29`), set per core in `program_factory.cpp:257-262`. Nameable (e.g. `is_q`); no loop-indexed or data-selected read.

## Team-only

- **Out-of-directory coupling & donor shape:** `✓ clean`. Function-call escapes: none — both compute kernels `#include` only `api/compute/*` (LLK compute API) and `api/dataflow/circular_buffer.h` (the `CircularBuffer` wrapper), all `tt_metal/*` class-1 headers with no concern. Borrowed kernel files: none — the op owns both kernel sources and instantiates them from its own directory by file path (`program_factory.cpp:231-240`). No port-together coupling induced.
- **Relaxation candidates (mined from a custom hash):** none — the op has no custom hash.
- **TTNN factory analysis:** `descriptor` concept, single program, no MeshWorkload, no op-owned tensors, no pybound `create_descriptor`, no custom hash, no custom `override_runtime_arguments`. Target concept `MetalV2FactoryConcept`. (Sheet-derived facts all cross-checked against the code above.)

## Misc anomalies  *(team-only, non-gating)*

- **Unused `CircularBuffer` wrapper objects in the tiled kernel.** `rotary_embedding_llama_sharded.cpp:61-63` declares `cos_cb_obj`, `sin_cb_obj`, `trans_mat_cb_obj` but never calls a method on any of them (the compute-math primitives take the raw `cb_id` instead). Dead locals; harmless. The row-major variant does not declare them.
- **Commented-out `has_work` early return.** Both kernels carry a commented-out early-return guard (`rotary_embedding_llama_sharded.cpp:24-28`, `..._row_major.cpp:24-28`) with a note that TRISC2 exceeds code size by 4 B with the profiler on. Context only — not porter work; the compute kernel currently runs on every core in `all_cores_bb` regardless of q/k assignment (the factory comment at `program_factory.cpp:249-251` notes this is tolerated since it's compute-only).
- **`batch_per_core` hardcoded to 1** (`program_factory.cpp:80`) with a `TODO` to generalize. Not a bug — a documented current limitation, matched by the `seq_len == 1` decode-mode validation (`device_operation.cpp:52-54`).

## Recipe notes

- **Device 2.0 gate on a compute-only op.** The gate's prose is framed around data-movement kernels ("Device 2.0 **Data Movement** migration"), but the pass condition is "every kernel this op exercises." This op exercises only compute kernels and owns no DM kernels, so the gate reduces to: are the compute kernels' *CB-management* idioms on the `CircularBuffer` wrapper (yes), and do they avoid legacy DM free functions (yes)? The recipe handles this correctly via the sanctioned-free-function note, but a one-line acknowledgement that a compute-only op is a valid (and clean) shape for this gate — where the compute-math LLK primitives inherently take `cb_id` and have no wrapper form — would save the next auditor a moment of doubt about whether `matmul_tiles(cb_id, ...)` is a holdover. (It is not: it's the compute API surface, not a DM idiom.)
