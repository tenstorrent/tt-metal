# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/copy/typecast`

One device operation shares the directory:

- **`TypecastDeviceOperation`** (`device/typecast_device_op.{hpp,cpp}`)
  - `TypecastProgramFactory` (`device/typecast_program_factory.cpp`) — interleaved / tiled path, and the non-optimized-sharded fallback (`select_program_factory` returns it for sharded inputs that fail `can_use_sharded_optimized_factory`)
  - `TypecastSubgridProgramFactory` (`device/typecast_program_factory.cpp`) — `sub_core_grids`, tiled
  - `TypecastShardedProgramFactory` (`device/typecast_sharded_program_factory.cpp`) — L1-sharded optimized (borrowed-memory CBs)
  - `TypecastRowMajorChunkedProgramFactory` (`device/typecast_rm_chunked_program_factory.cpp`) — ROW_MAJOR chunked DRAM path

All four are one porting unit (shared compute kernel + shared device-op); findings are attributed per-factory where they differ. A separate `experimental/quasar/typecast/` op exists but is a different architecture (Gen2) and out of scope.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `metal2_audit.md`.

**Recipe docs:** provenance could not be pinned — `git log -1 -- docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/` produced no output (this checkout is not a tracked doc-branch checkout). Audit ran against the standalone recipe at `/localdev/edwinlee/metal2_audit.md`.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/copy/typecast` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `TypecastDeviceOperation` → `TypecastProgramFactory`, `TypecastSubgridProgramFactory`, `TypecastShardedProgramFactory`, `TypecastRowMajorChunkedProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** (own + eltwise/unary donor kernels all Device 2.0) |
| *Prereqs* — Cross-op escapes | Ok (3 eltwise/unary donor kernels, file-path instantiated; port-together coupling) |
| *Feature Support* — overall | **GREEN** (all Appendix A entries N/A) |
| *Feature Support* — Variadic-CTA | Ok |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** (all 4 factories) |
| *TTNN Readiness* — Concept (current) | `descriptor` (all 4 factories) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Is safe to port? | Yes |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — Runtime-args update | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` (no op-owned tensors) |
| *Port work* — Offset base pointer | none |
| *Port work* — Tensor bindings (per binding) | Case 1 (interleaved/subgrid/rm_chunked in+out) · clean/borrowed (sharded in+out) |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none (no 3-arg accessor sites) |
| *Port work* — CB endpoints | all legal 1:1, except sharded `c_2` = self-loop (single toucher); sharded `c_0`/`c_2` are borrowed-memory |

**CB endpoints** are dispositions, not gates. Every CB is either a legal 1:1 FIFO or carries a port-time resolution; nothing here blocks the port.

## Result

**GREEN → brief issued.** All five gate-bearing subjects pass:

- **Device 2.0** ✓ — every kernel the op exercises (its own compute + RM-chunked reader/writer, and the three borrowed eltwise/unary dataflow kernels) is on Device 2.0 idioms.
- **Feature compatibility** ✓ — no `GlobalCircularBuffer`, `GlobalSemaphore`, non-zero `address_offset`, or CTA-varargs.
- **TTNN factory concept** ✓ — the readiness sheet marks `Is able to port? = yes` for all four `descriptor`-concept factories; the code cross-check agrees on every column.
- **Offset base pointers** ✓ — no host-folded offset base; the only pointer args are clean `Buffer*` bindings.
- **TensorAccessor 3rd argument** ✓ — no accessor passes a page-size 3rd argument.

Port work is routine: three factories bind interleaved tensors via `TensorAccessor` (Case 1), and the sharded factory uses borrowed-memory DFBs (`borrowed_from`) with one self-loop CB. See `METAL2_PORT_BRIEF.md`.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** From the *TTNN Operations analysis* sheet (Drive id `1KUMj8SyBGlNMZlLFgs1MbAZlO2g6EoUc4KaxSlcy8jw`, owner dgomez, modified 2026-07-23), the four `copy/typecast` rows are identical:

  | Factory | Concept | Custom hash | RT-args update (get_dynamic) | RT-args update (PD override) | Pybind descriptor | Smuggled ptr | Is safe to port? | **Is able to port?** | Relaxation | Op-owned tensors |
  |---|---|---|---|---|---|---|---|---|---|---|
  | `TypecastProgramFactory` | descriptor | no | no | no | no | no | yes | **yes** | none | no |
  | `TypecastSubgridProgramFactory` | descriptor | no | no | no | no | no | yes | **yes** | none | no |
  | `TypecastShardedProgramFactory` | descriptor | no | no | no | no | no | yes | **yes** | none | no |
  | `TypecastRowMajorChunkedProgramFactory` | descriptor | no | no | no | no | no | yes | **yes** | none | no |

  Op Classification: `PD (pointer-patching)` — matches the interim `Buffer*`→`BufferBinding` pointer-patching the factories use. Cross-check (code side, per the trust-but-verify rule):
  - `Concept == descriptor` — confirmed: each factory defines `static ProgramDescriptor create_descriptor(...)` (`typecast_program_factory.hpp:13,18`, `typecast_sharded_program_factory.hpp:13`, `typecast_rm_chunked_program_factory.hpp:13`); no mesh-workload return, no `create()/override_runtime_arguments()`.
  - `Custom hash == no` — confirmed: no `compute_program_hash` override in `typecast_device_op.{hpp,cpp}`.
  - `Runtime-args update == no` — confirmed: no `get_dynamic_runtime_args` / `override_runtime_arguments` anywhere in the op.
  - `Pybind descriptor == no` — confirmed: `ttnn/cpp/ttnn-nanobind/operations/copy.cpp` binds `typecast` via `ttnn::bind_function<"typecast">` (a plain function), with no `create_descriptor` / `nb::class_` of the device op.
  - `Op-owned tensors == no` — consistent with the `descriptor` concept (a descriptor factory cannot carry op-owned tensors). The sharded factory's `CBDescriptor.buffer = input.buffer()/output.buffer()` (`typecast_sharded_program_factory.cpp:94,111`) is a **borrowed-memory** CB, not an op-owned tensor.
  - Cross-column invariants hold (descriptor + no runtime-args-update + no op-owned tensors). No conflict → sheet trusted.

- **Device 2.0 (every kernel used):** **GREEN.** No violations. Kernels exercised, all on Device 2.0 object idioms (`Noc`, `CircularBuffer`/`DataflowBuffer` wrappers, `TensorAccessor`, `CoreLocalMem`):

  | Kernel | Owner | Used by | Idioms |
  |---|---|---|---|
  | `device/kernels/compute/eltwise_typecast.cpp` | typecast | all 4 | `CircularBuffer cb_in/cb_out` + `.wait_front/.pop_front/.reserve_back/.push_back` methods. `copy_tile/pack_tile/init_sfpu` take a CB *index* — these are compute-LLK APIs, not data-movement holdovers; not a violation. |
  | `device/kernels/dataflow/reader_typecast_rm_chunked.cpp` | typecast | rm_chunked | `Noc`, `CircularBuffer` + `.get_write_ptr()` method (not the free function), `TensorAccessor`, `CoreLocalMem`, `noc.async_read/async_read_barrier`. |
  | `device/kernels/dataflow/writer_typecast_rm_chunked.cpp` | typecast | rm_chunked | `Noc`, `CircularBuffer` + `.get_read_ptr()` method, `TensorAccessor`, `noc.async_write/async_writes_flushed/async_write_barrier`. |
  | `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` | eltwise/unary (donor) | interleaved, subgrid | `Noc`, `DataflowBuffer dfb`, `TensorAccessor`, `get_local_cb_interface(cb).fifo_page_size` (**sanctioned** free function). |
  | `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | eltwise/unary (donor) | interleaved, subgrid | `Noc`, `DataflowBuffer dfb`, `TensorAccessor`, `get_local_cb_interface(cb).fifo_page_size` (sanctioned). |
  | `eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp` | eltwise/unary (donor) | sharded | `DataflowBuffer dfb` + `.push_back`. |

- **Feature compatibility:** every Appendix A entry scanned against host + kernel + descriptor code. All absent → all `N/A`.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, no `CBDescriptor.global_circular_buffer` field set, no `remote_index`/remote-CB idioms. The sharded factory's `.buffer = …` is plain borrowed-memory, not a GCB. |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `.address_offset` set on any `CBDescriptor` (defaults to 0). The `.offset_bytes` in the RM-chunked kernels' `noc.async_read/async_write` page args (`reader_typecast_rm_chunked.cpp:45,62`) is a kernel-side NoC page offset — the explicit false-positive-guard case, not this field. |
  | GlobalSemaphore | N/A | The op uses no semaphores at all. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t` (`TypecastInputs`) is fixed (`input` + optional `preallocated_output`), no `std::vector<Tensor>`. Kernels read CTAs at constexpr offsets; `TensorAccessorArgs<N>()` is constexpr. No runtime-varying CTA index. |

- **CB endpoints (GATE-free):** per `(CB, factory)`, censused per node. All Device 2.0 gates GREEN, so the scan keys on intact idioms.
  - **`TypecastProgramFactory`** (`c_0` in, `c_2` out — plain, not borrowed): `c_0` produced by the interleaved reader (`dfb.reserve_back/push_back`), consumed by compute (`cb_in.wait_front/pop_front`) → **1 producer + 1 consumer, legal**. `c_2` produced by compute (`cb_out.reserve_back/push_back`), consumed by the interleaved writer (`dfb.wait_front/pop_front`) → **legal 1:1**.
  - **`TypecastSubgridProgramFactory`** — identical topology to `TypecastProgramFactory` → both CBs **legal 1:1**.
  - **`TypecastRowMajorChunkedProgramFactory`** — `c_0` produced by `reader_typecast_rm_chunked`, consumed by compute → **legal 1:1**; `c_2` produced by compute, consumed by `writer_typecast_rm_chunked` → **legal 1:1**.
  - **`TypecastShardedProgramFactory`** (borrowed-memory CBs):
    - `c_0` (`borrowed_from input.buffer()`): the sharded reader does `dfb.push_back(num_tiles_per_core)` (locked producer — signals the pre-populated borrowed buffer), compute consumes (`wait_front/pop_front`, locked consumer) → **1P+1C, legal** (bind `borrowed_from input.buffer()`).
    - `c_2` (`borrowed_from output.buffer()`): only compute touches it (`cb_out.reserve_back/push_back`); there is **no writer kernel** in this factory (nothing drains it — the borrowed output buffer *is* the result). Single toucher → **self-loop** (bind compute PRODUCER **and** CONSUMER, `borrowed_from output.buffer()`).

- **Offset base pointers:** **GREEN.** Every address-carrying RTA resolved to a clean base. The interleaved / subgrid / rm-chunked factories push the `Buffer*` object itself (`src_buffer` / `dst_buffer`) as RTA[0] — the `Buffer*`-binding (`BufferBinding`) form, not `buffer()->address()` and no host arithmetic (`typecast_program_factory.cpp:177-178,324-325`; `typecast_rm_chunked_program_factory.cpp:261-262`). The RM-chunked kernels compute `byte_offset = chunk_idx * full_chunk_size_bytes` *on-device* and pass it as `.offset_bytes` on a `TensorAccessor` whose base is the clean RTA address (`reader_typecast_rm_chunked.cpp:30,40-46`) — a kernel-side page offset, not a host-folded pointer. No Type 1/2/3/4. (Triage doc `2026-07-19_offset_base_pointers.md` not present in this checkout; a dated prior only — own scan is authoritative and clean.)

- **TensorAccessor 3rd argument:** **GREEN.** No accessor anywhere passes the optional 3rd (page-size) argument. All constructions are 2-arg: `TensorAccessor(src_args, src_addr)` (`reader_typecast_rm_chunked.cpp:30`, `writer_typecast_rm_chunked.cpp:30`, `reader_unary_interleaved_start_id.cpp`, `writer_unary_interleaved_start_id.cpp`); the sharded reader constructs no accessor. Nothing to classify. (Triage doc `2026-07-06_tensor_accessor_3rd_arg_triage.md` not present; dated prior only — own scan authoritative.)

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding, per factory):
  - `TypecastProgramFactory` — **input** Case 1 (via `TensorAccessor`); **output** Case 1. `Buffer*` delivered via `emplace_runtime_args` (`typecast_program_factory.cpp:177-178`); the interleaved reader/writer build `TensorAccessor(src_args, src_addr)` / `(dst_args, dst_addr)` and access only through it.
  - `TypecastSubgridProgramFactory` — **input** Case 1; **output** Case 1 (`typecast_program_factory.cpp:324-325`; same interleaved kernels).
  - `TypecastRowMajorChunkedProgramFactory` — **input** Case 1; **output** Case 1 (`typecast_rm_chunked_program_factory.cpp:261-262`; own RM-chunked kernels build the accessors).
  - `TypecastShardedProgramFactory` — **input** clean (borrowed-memory DFB, `borrowed_from input.buffer()`); **output** clean (borrowed-memory DFB, `borrowed_from output.buffer()`). No address RTA, no `TensorAccessor`.
  - *Per-binding split:* the same logical input/output tensor is Case 1 on the three interleaved-style factories but clean/borrowed on the sharded factory — expected, record per factory.
- **TensorParameter relaxation:** none (all four rows `relaxation = none`).
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** interleaved / subgrid / rm-chunked — all `c_0`/`c_2` legal 1:1, no action. Sharded — `c_0` bind `borrowed_from input.buffer()` (1P+1C); `c_2` bind `borrowed_from output.buffer()` **self-loop** (compute is the only toucher).

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none — no hidden second writer, no multi-reader, no ≥3-toucher CB in any factory.
- **Cross-op / shared kernels:** three eltwise/unary dataflow kernels are file-path-instantiated (cross-family donors): `reader_unary_interleaved_start_id.cpp` + `writer_unary_interleaved_start_id.cpp` (interleaved + subgrid factories) and `reader_unary_sharded.cpp` (sharded factory). Their Metal 2.0 CB→DFB / named-token rewrite is a **single shared rewrite** — every co-borrower (eltwise/unary ops and typecast) must adopt it together, or the first op to migrate in isolation breaks the others. Port the shared kernel as one unit. (These kernels are already Device 2.0 and use `DataflowBuffer`/`TensorAccessor`, so they cross cleanly; the coupling is a sequencing concern, not a blocker.)
- **RTA varargs:** none — every kernel reads its RTAs as a fixed run of distinct fields at constant indices (`reader/writer_*` args 0/1/2; sharded reader arg 0). Port them as named args.

## Team-only

- **Out-of-directory coupling & donor shape:**
  - *Op-level roll-up:* **✓ clean.** No function-call escape — every typecast-owned kernel `#include`s only `tt_metal/*` LLK/HAL headers (`api/dataflow/*`, `api/compute/*`, `api/tensor/*`, `api/core_local_mem.h`). Coupling is purely file-path kernel instantiation.
  - *Borrowed kernel files (file-path instantiation):*

    | Kernel file | Owning family | Instantiated by (this op) | Shared? |
    |---|---|---|---|
    | `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` | eltwise/unary | `TypecastProgramFactory`, `TypecastSubgridProgramFactory` | broadly shared across eltwise/unary ops |
    | `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | eltwise/unary | `TypecastProgramFactory`, `TypecastSubgridProgramFactory` | broadly shared across eltwise/unary ops |
    | `eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp` | eltwise/unary | `TypecastShardedProgramFactory` | broadly shared across eltwise/unary ops |

    Port-together set: typecast + the eltwise/unary family that co-owns these three kernels. (Exact co-borrower list not enumerated this run — the coupling and its resolution — one shared rewrite — are unchanged by the precise membership.)
- **Relaxation candidates (mined from a custom hash):** none — the op has no custom hash.
- **TTNN factory analysis:** all sheet gate-conjuncts confirmed absent (custom hash / PD override-runtime-args / pybind `create_descriptor` / genuine multi-program). Op-owned tensors: none. Current concept `descriptor` on all four factories → target `MetalV2FactoryConcept`.

## Misc anomalies  *(team-only, non-gating)*

- **`device/kernels/compute/eltwise_typecast.cpp:31` — `TYPECAST_LLK_INIT()` invoked inside the innermost per-tile loop** (re-runs every tile, before each `TYPECAST_LLK(0)`), rather than once before the loop. Appears redundant (typecast LLK init typically needs to run only once per configuration); a possible minor per-tile overhead. Not a correctness issue and not port work — flagged for the ops team to confirm whether the per-tile re-init is intentional. (Noticed incidentally; the port carries the kernel through unchanged.)

## Recipe notes

- The two dated triage docs referenced by the *Offset base pointers* and *TensorAccessor 3rd argument* subjects (`analyses/2026-07-19_offset_base_pointers.md`, `analyses/2026-07-06_tensor_accessor_3rd_arg_triage.md`) and the readiness-sheet fetch-procedure doc (`analyses/ttnn_op_porting_readiness.md`) are **not present in this repo checkout**, and the recipe was supplied as a standalone file (`/localdev/edwinlee/metal2_audit.md`) outside its doc tree. The recipe frames those triage docs as optional dated priors, so their absence did not impede the audit (own scan is authoritative for both subjects, and both came back clean). The readiness sheet itself was fetched successfully from Drive. Noting the missing local docs in case the auditor is expected to have the full `metal_2.0/analyses/` tree on disk.
- The provenance `git log` command (against `docs/source/tt-metalium/.../metal_2.0/`) produced no output in this checkout, so the recipe version could not be pinned per the recipe's instruction; recorded as such in the header.
- The Google Drive connector was transiently disconnected mid-audit and had to be reconnected before the readiness sheet could be fetched — worth noting since the readiness-sheet lookup is a hard dependency of the *TTNN factory concept* gate and cannot be substituted by the code cross-check alone (the `Is safe to port?` axis is the sheet owner's judgment).
