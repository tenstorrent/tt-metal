# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads`

One device operation, one program factory, two mutually-exclusive configs inside it:

- **`NLPConcatHeadsDeviceOperation`** (`device/nlp_concat_heads_device_operation.{hpp,cpp}`)
  - `NLPConcatHeadsProgramFactory` (`device/nlp_concat_heads_program_factory.cpp`) — a single
    `create_descriptor` that branches on `in_sharded` into two entirely different kernel sets:
    - **config `INTERLEAVED`** (`in_sharded == false`) — op-private reader + donor writer, DRAM/L1 interleaved I/O.
    - **config `SHARDED`** (`in_sharded == true`) — one op-private kernel instantiated **twice**
      (Reader-config + Writer-config), both borrowed-memory CBs, no NoC-remote traffic.

Kernels referenced by the factory (all in scope):

| Kernel | Owner | Used by config |
|---|---|---|
| `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads.cpp` | this op (private) | INTERLEAVED |
| `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads_sharded.cpp` | this op (private) | SHARDED (×2 instances) |
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | `eltwise/unary` (donor) | INTERLEAVED |

No unreferenced kernel files in the op directory. The op has no compute kernel and no semaphores.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `bcf38615192 2026-08-03 docs(metal_2.0): add the op-porting recipe set`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `NLPConcatHeadsDeviceOperation` → `NLPConcatHeadsProgramFactory` (single factory, two internal configs) |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all three kernels fully Device 2.0 (`Noc`, `CircularBuffer`/`DataflowBuffer`, `TensorAccessor`, `CoreLocalMem`, `UnicastEndpoint`); only sanctioned free functions remain |
| *Prereqs* — Cross-op escapes | Ok — no function-call escape; one file-path borrow (`writer_unary_interleaved_start_id.cpp`) |
| *Feature Support* — overall | **GREEN** — every Appendix A entry `N/A` |
| *Feature Support* — Variadic-CTA | Ok — all CTAs read at literal constexpr indices |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Is safe to port? | Yes |
| *TTNN Readiness* — Custom hash | No (confirmed: no `compute_program_hash` in the op dir) |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No (confirmed: hook absent from the device-op) |
| *TTNN Readiness* — `override_runtime_arguments` | No (confirmed: method absent) |
| *TTNN Readiness* — Pybind `create_descriptor` | No (confirmed: `nlp_concat_heads_nanobind.cpp` has no `create_descriptor` binding) |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` |
| *Port work* — Offset base pointer | none — no address RTA folds a host-side offset into its base |
| *Port work* — Tensor bindings (per binding) | INTERLEAVED: `input` Case 1, `output` Case 1 · SHARDED: `input` clean (borrowed DFB), `output` clean (borrowed DFB) |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none — no accessor in this op passes a 3rd argument |
| *Port work* — CB endpoints | INTERLEAVED: `cb_src0` legal 1:1 · SHARDED: `cb_src0` **multi-binding flag**, `cb_out0` **multi-binding flag** |

**CB endpoints** are dispositions, not gates. Record the disposition per `(CB, config)`; the same CB's
disposition flips between this op's two configs (`cb_src0` is a legal 1:1 FIFO under INTERLEAVED and a
two-locked-producer borrowed DFB under SHARDED).

## Result

**GREEN → brief issued.** Every gate clears: Device 2.0 ✓ · Feature compatibility ✓ · TTNN factory
concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd argument ✓. Both configs are portable — no
subset scoping needed.

The port is a small one by volume (three kernels, no compute, no semaphores, four RTAs at most), but
carries two non-obvious construction items the porter must not miss:

1. The **SHARDED config's two same-source kernel instances are both locked FIFO producers** on both
   borrowed DFBs (each calls `reserve_back` on `cb_in0` *and* `cb_out0`), so the census cannot fold to
   1P+1C and the multi-binding advanced option is required on both. See *CB endpoints* below.
2. A **pre-existing latent hole in the SHARDED config** — `in_sharded && !out_sharded` is a
   validation-reachable combination for which the factory never creates `cb_out0` (index 16) even
   though the kernel unconditionally binds it. Legacy tolerates this silently; Metal 2.0 will not.
   See *Misc anomalies* — the porter needs a decision here, and it is an ops-team question, not a
   porter judgment call.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** The readiness sheet's row
  (`experimental/transformer/nlp_concat_heads` / `NLPConcatHeadsDeviceOperation` /
  `NLPConcatHeadsProgramFactory`) reads `Is able to port? = yes`, with `Concept = descriptor`,
  `Custom hash = no`, `Runtime-args update (get_dynamic_runtime_args) = no`,
  `Override runtime args method? (PD only) = no`, `Pybind descriptor = no`,
  `Smuggled pointer = no`, `Is safe to port? = yes`, `TensorParameter relaxation = none`,
  `Porting Target = ProgramSpecFactoryConcept`.

  Cross-check against the code — **clean, no conflict**:

  | Column | Sheet | Code evidence |
  |---|---|---|
  | `Concept` | `descriptor` | `NLPConcatHeadsProgramFactory::create_descriptor(...) -> ProgramDescriptor`, `nlp_concat_heads_program_factory.hpp:15-16` |
  | `Custom hash` | `no` | no `compute_program_hash` anywhere under the op directory |
  | `get_dynamic_runtime_args` | `no` | hook absent from `NLPConcatHeadsDeviceOperation` (`nlp_concat_heads_device_operation.hpp:19-31` declares only `validate_on_program_cache_miss` / `compute_output_specs` / `create_output_tensors`) |
  | `override_runtime_arguments` | `no` | method absent (same declaration list) |
  | `Pybind descriptor` | `no` | `nlp_concat_heads_nanobind.cpp` contains no `create_descriptor` binding |
  | `Op-owned tensors?` | (blank) | `create_descriptor` returns a bare `ProgramDescriptor`; no `buffers` vector exists on it |
  | Factory-set match | 1 row | exactly one factory in the code (`program_factory_t = std::variant<NLPConcatHeadsProgramFactory>`, `nlp_concat_heads_device_operation.hpp:24`) ↔ exactly one sheet row |

  Cross-column invariants hold (`descriptor` row with no op-owned tensors; no `get_dynamic_runtime_args`
  on a non-legacy concept).

- **Device 2.0 (every kernel used):** **GREEN.** All three kernels are structurally Device 2.0 — no
  `noc_async_read`/`noc_async_write` free calls, no `InterleavedAddrGen` / `ShardedAddrGen` /
  `InterleavedAddrGenFast` / `InterleavedPow2AddrGen*`, no raw semaphore addresses, no manual CB index
  management.

  | Kernel | Device 2.0 surface in use |
  |---|---|
  | `reader_tm_tile_layout_nlp_concat_heads.cpp` | `Noc`, `noc.async_read(...)` / `async_read_barrier()`, `TensorAccessorArgs<4>` + `TensorAccessor`, `CircularBuffer` (`reserve_back` / `push_back` / `get_write_ptr` methods), `CoreLocalMem<uint32_t>` |
  | `reader_tm_tile_layout_nlp_concat_heads_sharded.cpp` | `Noc`, `noc.get_noc_id()`, `UnicastEndpoint` + `noc_traits` src args, `CircularBuffer` (`reserve_back` / `get_read_ptr` / `get_write_ptr` methods), `CoreLocalMem<uint32_t>` |
  | `writer_unary_interleaved_start_id.cpp` (donor, `eltwise/unary`) | `Noc`, `noc.async_write(dfb, s, ...)` / `async_writes_flushed()` / `async_write_barrier()`, `DataflowBuffer` (`wait_front` / `pop_front`), `TensorAccessorArgs<1>` + `TensorAccessor` |

  Free functions remaining, all **sanctioned** (explicitly kept by Device 2.0 — not holdovers, no table
  row owed):

  | File | Line | Call | Status |
  |---|---|---|---|
  | `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads.cpp` | 30 | `get_tile_size(cb_id_in0)` | sanctioned |
  | `device/kernels/dataflow/reader_tm_tile_layout_nlp_concat_heads_sharded.cpp` | 30 | `get_tile_size(cb_id_in0)` | sanctioned (and the result is dead — see *Misc anomalies*) |
  | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | 19 | `get_local_cb_interface(cb_id_out).fifo_page_size` | sanctioned |

  One construct worth naming explicitly so a later reader does not re-litigate it: the SHARDED kernel's
  self-loopback read uses the hardware globals `my_x[noc_id]` / `my_y[noc_id]`
  (`reader_tm_tile_layout_nlp_concat_heads_sharded.cpp:39-40`) to fill a `UnicastEndpoint`'s
  `{.noc_x, .noc_y, .addr}` src args. This is **not** a Device 1.0 holdover: the coordinates are
  hardware state, not an addressing API, and they are consumed *through* the Device 2.0 `UnicastEndpoint`
  wrapper (`tt_metal/hw/inc/api/dataflow/endpoints.h`), which is precisely the migrated shape the
  Device 2.0 guide prescribes for remote/loopback unicast.

- **Feature compatibility:** every Appendix A entry scanned; none fires.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, no `CreateGlobalCircularBuffer`, no `global_circular_buffer` field on either `CBDescriptor` literal, no `remote_index` / `remote_cb_*` idiom, no `<tt-metalium/global_circular_buffer.hpp>` include. The two `CBDescriptor`s set `.buffer` (`nlp_concat_heads_program_factory.cpp:150`, `:163`) — that is the plain borrowed-memory pattern, a mechanical porting-recipe translation via `DataflowBufferSpec::borrowed_from`, explicitly *not* this entry. |
  | CBDescriptor `address_offset` (non-zero) | N/A | Neither `CBDescriptor` sets `.address_offset` (default 0); no `set_address_offset`, no 4-arg `UpdateDynamicCircularBufferAddress`, no `cb_descriptor_from_sharded_tensor`. |
  | GlobalSemaphore | N/A | The op declares no semaphore of any kind — no `Semaphore`, no `CreateSemaphore`, no `GlobalSemaphore`, no `<tt-metalium/global_semaphore.hpp>`. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | Op-level cue absent (`tensor_args_t = Tensor` — exactly one input tensor, `nlp_concat_heads_device_operation.hpp:21`). Kernel-level decider absent: every `get_compile_time_arg_val` call in all three kernels uses a **literal** index (sharded kernel 0–5, interleaved reader 0–3, donor writer 0), each bound to a `constexpr` — no runtime-varying CTA index anywhere. |

- **CB endpoints (GATE-free):** census per `(CB, config)`, per node. Two CB indices exist:
  `src0_cb_index = 0` and `out_cb_index = 16` (`nlp_concat_heads_program_factory.cpp:82`). No dead CB —
  every allocated index is referenced by a bound kernel in the config that allocates it.

  **Config INTERLEAVED** — `cb_src0` only (index 16 is not allocated; `out_sharded` is forced false by
  `validate_on_program_cache_miss` line 53-56, which requires an INTERLEAVED output whenever the input
  is not sharded).

  | CB | Toucher | How it touches | Role |
  |---|---|---|---|
  | `cb_src0` (0) | reader `reader_tm_tile_layout_nlp_concat_heads.cpp` | `reserve_back(1)` @ :47, `push_back(1)` @ :59, `get_write_ptr()` @ :41 | **locked producer** |
  | `cb_src0` (0) | writer `writer_unary_interleaved_start_id.cpp` (`cb_id_out` CTA = `src0_cb_index` = 0, `nlp_concat_heads_program_factory.cpp:118`) | `wait_front(1)` @ :40, `pop_front(1)` @ :43, `noc.async_write(dfb, ...)` @ :41 | **locked consumer** |

  → 2 touchers, ≤1 locked producer and ≤1 locked consumer ⇒ **plain 1:1, legal. No action.**
  (The producer's `get_write_ptr()` peek at line 41 is covered by its own PRODUCER binding — it is not
  a third endpoint.)

  **Config SHARDED** — both CBs allocated as borrowed-memory (`.buffer = in0_buffer` @ :150,
  `.buffer = out_buffer` @ :163). One kernel source
  (`reader_tm_tile_layout_nlp_concat_heads_sharded.cpp`) instantiated **twice** over the same
  `all_cores` — `reader_desc` with `ReaderConfigDescriptor{}` @ :101 and `writer_desc` with
  `WriterConfigDescriptor{}` @ :109 — differing only in config and in the three-element RTA that
  splits the head range (`nheads_first_risc` / `nheads_second_risc`, :169-187). This is the
  dual-instance work-split face, so **every node has two co-resident touchers of each CB**.

  | CB | Toucher | How it touches | Role |
  |---|---|---|---|
  | `cb_src0` (0) | instance A (Reader config) | `cb_in0.reserve_back(block_size)` @ :35, `cb_in0.get_read_ptr()` @ :43 | **locked producer** |
  | `cb_src0` (0) | instance B (Writer config) | same two calls (same source) | **locked producer** |
  | `cb_out0` (16) | instance A (Reader config) | `cb_out0.reserve_back(block_size)` @ :36, `cb_out0.get_write_ptr()` @ :44 | **locked producer** |
  | `cb_out0` (16) | instance B (Writer config) | same two calls (same source) | **locked producer** |

  → **2 locked producers on each CB, on every node ⇒ multi-binding advanced option on both DFBs.**

  This is deliberately *not* the usual dual-instance-work-split outcome, and the divergence is worth
  stating precisely because it inverts the recipe's default expectation for face (c). The recipe's
  default is 1P+1C, which applies when both co-touchers are *sync-free* raw peeks. Here they are not:
  each instance issues a genuine FIFO `reserve_back` on **both** CBs (lines 35 and 36), which locks it
  to PRODUCER on both, and there is no consumer anywhere — nothing ever `wait_front`s either CB, and
  the matching `push_back` is commented out at :62. The census therefore cannot be relabelled into
  1P+1C, and the multi-binding option is the correct (and only) resolution under the classification
  table.

  For the record, both `reserve_back` calls are *functionally* no-ops: `block_size` (CTA 5) equals
  `num_blocks_per_core_group_1 * in0_HtWt`, which is exactly the tile count both CBs are sized to
  (`per_tensor_tiles` recomputed for the shard at :58, `cb_out_num_tiles = per_tensor_tiles` at :154),
  so each reserves the whole buffer and can never block. Line 35 even carries the author's own comment
  `// Redundant`. That makes the shape *morally* sync-free — but removing a `reserve_back` is a
  functional kernel change, off the port's kernel-side whitelist, so the port must take the census as
  written. Flagged as a recipe note below.

- **Offset base pointers:** **GREEN.** The op is not listed in
  `analyses/2026-07-19_offset_base_pointers.md`, and an independent scan of every address-carrying arg
  confirms no fold:

  - INTERLEAVED reader RTA[0] — `in0_buffer`, a **bare `Buffer*`** pushed into the RTA list
    (`nlp_concat_heads_program_factory.cpp:201`). No arithmetic; the framework supplies the base.
  - INTERLEAVED writer RTA[0] — `out_buffer`, likewise a bare `Buffer*` (`:210`). No arithmetic.
  - SHARDED config — **no address arg at all.** The two nonzero RTAs that *look* like offsets,
    `nheads_first_risc * in0_HtWt * single_tile_size` and `nheads_first_risc * in0_w_tiles *
    single_tile_size` (`:185-186`), are pure scalar byte offsets. They are added on-device to a
    **CB-derived** base — `cb_in0.get_read_ptr() + start_read_offset_bytes` @ :43 and
    `cb_out0.get_write_ptr() + start_write_offset_bytes` @ :44 — never to a tensor buffer address.
    This is the already-split-out shape (separate base + separate offset, summed in the kernel), i.e.
    the Type-1 *fix*, not the Type-1 defect.

  No Type 1, no Type 2. Type 3 (`address_offset`) is `N/A` per Appendix A above; no `ttnn::narrow` or
  interior-base `MeshBuffer::create` (Type 4) anywhere in the op.

- **TensorAccessor 3rd argument:** **GREEN — the subject does not fire.** The op is not listed in
  `analyses/2026-07-06_tensor_accessor_3rd_arg_triage.md`, and the independent syntactic scan finds
  exactly two `TensorAccessor` constructions in the op's kernel set, both **two-argument**:
  `TensorAccessor(in0_args, in0_tensor_addr)` (`reader_tm_tile_layout_nlp_concat_heads.cpp:31`) and
  `TensorAccessor(dst_args, dst_addr)` (`writer_unary_interleaved_start_id.cpp:31`). No explicit
  page-size override exists, so there is nothing to classify or drop.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding — classification splits by config within the single factory):
  - `input` — **INTERLEAVED: Case 1.** Delivered as a `Buffer*` in reader RTA[0]
    (`nlp_concat_heads_program_factory.cpp:201`) → read as `get_arg_val<uint32_t>(0)` →
    `TensorAccessor(in0_args, in0_tensor_addr)` (`reader_tm_tile_layout_nlp_concat_heads.cpp:17,31`).
    Express as a `TensorParameter`; the kernel builds `TensorAccessor(tensor::input)` and both the
    address RTA and the `TensorAccessorArgs<4>` CTA block disappear.
  - `output` — **INTERLEAVED: Case 1.** Delivered as a `Buffer*` in writer RTA[0] (`:210`) →
    `TensorAccessor(dst_args, dst_addr)` (`writer_unary_interleaved_start_id.cpp:13,31`). Same
    treatment.
  - `input` — **SHARDED: clean.** Borrowed-memory DFB read: `cb_src0` is backed by `in0_buffer`
    (`:150`) and the kernel reads it via `cb_in0.get_read_ptr()`
    (`reader_tm_tile_layout_nlp_concat_heads_sharded.cpp:43`). Port as
    `DataflowBufferSpec::borrowed_from(tensor::input)`; no accessor, no address arg.
  - `output` — **SHARDED: clean.** Borrowed-memory DFB write: `cb_out0` is backed by `out_buffer`
    (`:163`), written via `cb_out0.get_write_ptr()` (`...sharded.cpp:44`). Port as
    `borrowed_from(tensor::output)`.

  Neither `Buffer*` RTA is the silent-wrong hazard — the framework registers them as `BufferBinding`s
  and patches them on cache hits today — but both are pointer arguments the port replaces with a typed
  binding, so they are enumerated as routine port work.

- **TensorParameter relaxation:** none (sheet: `none`; no custom hash exists to cross-check against).
- **TensorAccessor 3rd arg:** none — no site.
- **CB endpoints:**
  - `(cb_src0, INTERLEAVED)` — legal 1:1, no action.
  - `(cb_src0, SHARDED)` — **set the multi-binding advanced option** (2 locked producers/node).
  - `(cb_out0, SHARDED)` — **set the multi-binding advanced option** (2 locked producers/node).
  - No dead CB.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** `(cb_src0, SHARDED)` and `(cb_out0, SHARDED)`. The
  hidden-second-writer hunt was run and found **no** semaphore-gated raw co-fill — the op has no
  semaphores at all, and the two co-touchers are the two instances of one kernel source, fully visible
  at `nlp_concat_heads_program_factory.cpp:95-109`. The multi-binding verdict here comes from the
  *FIFO-role* rule (two `reserve_back`ers), not from a hidden endpoint.
- **Cross-op / shared kernels:** the INTERLEAVED writer is a file-path borrow of
  `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`,
  owned by `eltwise/unary` and instantiated by ~35 non-quasar factories. **No `_metal2` fork exists
  beside it** — this port creates the first one, at
  `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp`.
  See the caution in *Team-only* below about the misplaced typecast copy.
- **RTA varargs:** none. Every RTA in all three kernels is read at a literal index as a distinct field
  (interleaved reader 0–3, donor writer 0–2, sharded kernel 0–2). No counted loop over `get_arg_val`,
  no `arg_index++`, no data-selected index. All become named args.

## Team-only

- **Out-of-directory coupling & donor shape:**

  **Op-level roll-up: ✓ clean.** No function-call escape of any kind — every `#include` in all three
  kernels resolves under `tt_metal/hw/inc/api/` (donor class 1: LLK/HAL/firmware, no concern). The op's
  kernels call no helper owned by another op, so the by-shape per-call table has no rows and the
  per-call detail section is omitted.

  | Op kernel | `#include` | Resolved donor | Class |
  |---|---|---|---|
  | `reader_tm_tile_layout_nlp_concat_heads.cpp` | `api/dataflow/dataflow_api.h`, `api/dataflow/noc.h`, `api/dataflow/circular_buffer.h`, `api/core_local_mem.h`, `api/tensor/noc_traits.h` | `tt_metal/hw/inc/api/` | 1 — no concern |
  | `reader_tm_tile_layout_nlp_concat_heads_sharded.cpp` | `api/dataflow/dataflow_api.h`, `api/dataflow/noc.h`, `api/dataflow/circular_buffer.h`, `api/dataflow/endpoints.h`, `api/core_local_mem.h` | `tt_metal/hw/inc/api/` | 1 — no concern |
  | `writer_unary_interleaved_start_id.cpp` | `api/dataflow/dataflow_api.h`, `api/dataflow/noc.h`, `api/dataflow/dataflow_buffer.h`, `api/tensor/noc_traits.h` | `tt_metal/hw/inc/api/` | 1 — no concern |

  **Borrowed kernel files (file-path instantiation):**

  | Kernel file | Owning family | Sharing | `_metal2` fork beside it? |
  |---|---|---|---|
  | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | `eltwise/unary` | broadly shared — ~35 non-quasar factories (`data_movement/{concat,copy,tilize,transpose,slice,permute,reshape_on_device,bcast}`, `reduction/{generic,prod}`, `matmul`, `embedding`, `kv_cache`, `eltwise/unary_backward`, `experimental/matmul/attn_matmul`, `experimental/transformer/nlp_concat_heads_boltz`, `examples/example`, …) | **No.** `ls` of `eltwise/unary/device/kernels/dataflow/` shows no `*_metal2*` file. |

  The op's other two kernels are private — a repo-wide grep finds
  `reader_tm_tile_layout_nlp_concat_heads.cpp` and
  `reader_tm_tile_layout_nlp_concat_heads_sharded.cpp` referenced only by this op's own factory, so
  they carry no coordination cost and can be converted in place.

  **Caution — a near-identical fork exists, in the wrong directory.** A landed, non-quasar Metal 2.0
  fork of this exact donor sits at
  `ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp`
  (from `cbde3d44ff3`, *[Cleanup] Port Typecast to Metal 2.0 (#51397)*). Its header comment even names
  the donor. It is **not** a fork the locational rung-1 test can find, because it lives in typecast's
  own tree rather than beside its donor — the same misplacement that commit `5ecda11bb71` corrected for
  `transpose_wh` on the current branch. Two consequences for planning: (a) the ~35-factory sunset list
  above is now served by *two* divergent Metal 2.0 entry points unless the typecast copy is eventually
  relocated/merged, and (b) any future auditor running the locational test on this donor will keep
  reporting "no fork" while a usable one exists. Worth an ops/Metal-2.0-track ticket to relocate the
  typecast fork to `eltwise/unary/device/kernels/dataflow/`.

- **Relaxation candidates** (mined from a custom hash): none — the op has no custom hash, so there is
  nothing to mine.

- **TTNN factory analysis:** all sheet-derived facts confirmed against the code in the *Gate detail*
  cross-check table above. Summary: concept `descriptor`; **no** op-owned tensors (the returned
  `ProgramDescriptor` has no `buffers` vector); **no** MeshWorkload need (single `create_descriptor`,
  `Execution Model = SPMD`); **no** pybind of factory/device-op internals (`nlp_concat_heads_nanobind.cpp`
  binds only the user-facing `ttnn::experimental::nlp_concat_heads` entry point); **no** custom hash;
  **no** `get_dynamic_runtime_args`; **no** `override_runtime_arguments`. Target concept:
  `ProgramSpecFactoryConcept`, plain (no op-owned tensors).

## Misc anomalies  *(team-only, non-gating)*

1. **Reachable config with a missing output CB — the one that matters.**
   `nlp_concat_heads_device_operation.cpp:48-51` permits `in_sharded == true` with any output layout
   other than `HEIGHT_SHARDED`, so **INTERLEAVED output on a sharded input passes validation**, and
   `compute_output_specs` (`:73-88`) builds a well-formed interleaved output spec for it. But
   `nlp_concat_heads_program_factory.cpp:153` allocates `cb_out0` (index 16) **only when
   `out_sharded`**, while the SHARDED branch's kernel unconditionally does
   `CircularBuffer cb_out0(cb_id_out0); cb_out0.reserve_back(block_size)` and writes through
   `cb_out0.get_write_ptr()` (`...sharded.cpp:33,36,44`). In that combination the kernel operates on a
   CB that was never created — undefined behaviour today, silently. No test covers it:
   `test_sharded.py::test_sharded_concat_heads` parametrizes only `[True, True]` and `[False, False]`
   (`tests/tt_eager/python_api_testing/unit_testing/misc/test_sharded.py:1342`), and
   `test_nlp_concat_heads.py` is interleaved-only. Routed to the ops team; the honest fix is a
   `TT_FATAL` requiring a sharded output whenever the input is sharded (or an
   interleaved-output code path). **This also blocks a clean port construction** — Metal 2.0 requires
   the `dfb::out` binding the kernel names to exist, so the porter cannot simply mirror the
   conditional. Flagged to the porter as a question rather than a decision to make alone.
2. **Dead local + dead `get_tile_size` call.**
   `reader_tm_tile_layout_nlp_concat_heads_sharded.cpp:30` computes
   `const uint32_t single_tile_size_bytes = get_tile_size(cb_id_in0);` and never uses it — the loops
   stride by `head_dim_size_bytes` and `out_row_size_bytes` (CTAs 3 and 4). Removable; not port work.
3. **Dead `grid_to_cores` result in the SHARDED path.** `nlp_concat_heads_program_factory.cpp:167`
   computes `const auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major)`
   unconditionally, but `cores` is consumed only in the `else` (INTERLEAVED) branch at `:191`. In that
   branch `row_major` is always `false` (it is assigned only inside the `if (in_sharded)` block at
   `:59`), so the `row_major` variable has **no effect on any code path** — the shard orientation it
   reads is silently discarded. Benign today (see next item) but misleading.
4. **Hardcoded `row_wise=true` against a possibly COL_MAJOR shard.**
   `nlp_concat_heads_program_factory.cpp:173` iterates
   `corerange_to_cores(all_cores, num_cores, /*row_wise=*/true)` while the shard may be `COL_MAJOR`
   (the sharded test uses `ShardOrientation.COL_MAJOR`). Harmless **only** because every core in the
   sharded branch receives byte-identical RTAs, so iteration order cannot matter — but the invariant is
   undocumented and would break the moment the args become per-core.
5. **Stale model-specific comment.** `nlp_concat_heads_program_factory.cpp:36` —
   `uint32_t per_tensor_tiles = ashape[1] * ashape[3] / TILE_WIDTH;  // 142` — the `// 142` is a
   leftover Falcon-7B-specific value, not a general invariant; `per_tensor_tiles` is recomputed for the
   sharded case at `:58` anyway.
6. **Commented-out `push_back`.** `reader_tm_tile_layout_nlp_concat_heads_sharded.cpp:62` —
   `// cb_out0.push_back(block_size);`. Consistent with the CB being a borrowed output shard that
   nothing drains, but it leaves the FIFO half-driven (`reserve_back` without `push_back`), which is
   what forces the multi-binding disposition above. Ops team may prefer to drop both FIFO calls; that
   would reduce the census to two role-free touchers and hence to a plain 1P+1C.

## Recipe notes

1. **The classification table has no row for a *vestigial* FIFO op.** `CB endpoints` says flatly that
   "a kernel that FIFO-produces (`reserve_back`/`push_back`) is a **locked producer**", and separately
   that a raw peek is role-free. This op sits between the two: both co-touchers call `reserve_back` on
   a borrowed CB sized so the reservation is unconditionally satisfiable, with no `push_back`
   (commented out) and no consumer anywhere. Semantically it is exactly the sync-free dual-instance
   work-split the recipe's face (c) describes and expects to resolve as 1P+1C; syntactically the
   `reserve_back` locks both to PRODUCER and forces the multi-binding flag. I followed the rule as
   written (multi-binding) because the alternative requires the auditor to reason about whether a FIFO
   call can block — but the recipe might usefully say whether a provably-non-blocking `reserve_back`
   with no matching `push_back` should still lock the role, since the two readings produce different
   port artifacts (an advanced-option flag versus none) and the "reach for the flag only as a last
   resort" guidance pulls the other way.
2. **Column-name drift in the readiness sheet.** The audit and `ttnn_op_porting_readiness.md` both name
   the column `Override runtime args method? (PD and legacy)`; the live sheet's header reads
   `Override runtime args method? (PD only)`. Header-name lookup still resolves it unambiguously, and
   the standing rule is that existing names never change — so this looks like the docs quoting a name
   that was later edited, rather than a sheet break. Not treated as a spreadsheet-broken conflict.
   Worth reconciling in the docs.
3. **Config-scoped findings inside a *single* factory.** The recipe consistently scopes per-factory
   (`Classify per instantiation`, `Code-path scope`, the per-factory readiness rows). This op has one
   factory whose `create_descriptor` branches into two kernel sets that share no kernel, no CB layout,
   and no binding classification — so nearly every subject here needed a per-*config* rather than
   per-*factory* answer (bindings: Case 1 vs clean; CB endpoints: legal vs multi-binding). The report
   template accommodates this fine, but the vocabulary ("factory subset", "per-factory verdict")
   assumes the split is at factory granularity. A sentence acknowledging the intra-factory branch case
   would save the next auditor a judgment call.
