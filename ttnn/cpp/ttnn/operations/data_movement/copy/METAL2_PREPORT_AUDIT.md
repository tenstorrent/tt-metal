# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/copy`

**Device operations & program factories in this directory:**

- **`CopyDeviceOperation`** (`device/copy_device_operation.hpp` / `.cpp`)
  - `SameMemoryConfig` (`device/copy_same_memory_config_program_factory.cpp`)
  - `DefaultRowMajor` (`device/copy_default_row_major_program_factory.cpp`)
  - `DefaultTilized` (`device/copy_default_tilized_program_factory.cpp`)

One DeviceOperation, three factories, all sharing the same `CopyParams` / `CopyInputs` types — audited together as one unit. All kernel files under `device/kernels/` are referenced (no unreferenced/dead kernel files). The op's host entry points (`ttnn::copy`, `ttnn::assign`) live in `copy.cpp`; `copy_nanobind.cpp` binds only the free functions `copy` / `assign` (no `create_descriptor` binding).

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `c16f21b8cb6 2026-08-18 docs(metal_2.0): unpack_modes -- the trigger is the buffer format, not the dtypes`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/copy` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `CopyDeviceOperation` → `SameMemoryConfig`, `DefaultRowMajor`, `DefaultTilized` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all 12 referenced kernels (6 own + 6 donor/shared) are Device 2.0 compliant |
| *Prereqs* — Cross-op escapes | Ok — one in-family function-call escape (`tt_memmove`, takes `Noc`); several file-path borrows (all Device 2.0) |
| *Feature Support* — overall | **GREEN** (all Appendix A entries N/A) |
| *Feature Support* — GlobalCircularBuffer / `address_offset` / GlobalSemaphore | N/A / N/A / N/A |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** — all 3 factory rows `yes`; primary cross-check clean |
| *TTNN Readiness* — Concept (current) | `descriptor` (all 3 factories) |
| *TTNN Readiness* — Secretly SPMD | N/A (`Concept == descriptor`, not `WorkloadDescriptor`) |
| *TTNN Readiness* — Custom hash | No (sheet `no`; no `compute_program_hash` in code) |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No (sheet `no`; absent in device-op) |
| *TTNN Readiness* — `override_runtime_arguments` | No (sheet `no`; absent in device-op) |
| *TTNN Readiness* — Pybind `create_descriptor` | No (sheet `no`; nanobind binds only `copy`/`assign`) |
| *TTNN Readiness* — Op-owned tensors | No (sheet blank; `descriptor` concept) |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` (sheet `Porting Target`; `Override runtime args? == no`) |
| *Port work* — Offset base pointer | none — no `->address()` fold; factories deliver bases via `Buffer*` bindings |
| *Port work* — Tensor bindings (per binding) | all **Case 1** (`Buffer*` base → `TensorAccessor`) |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) |
| *Port work* — TensorAccessor 3rd arg | none — no accessor in any kernel passes a 3rd argument |
| *Port work* — CB endpoints | all legal 1:1, except `DefaultRowMajor` c_0 → **self-loop** |

**CB endpoints** are dispositions, not gates: every out-of-window CB has a port-time resolution. Here the only out-of-window case is one single-toucher scratchpad CB (self-loop). See [CB endpoints](#cb-endpoints).

## Result

**GREEN → brief issued.** Every gate clears: Device 2.0 ✓, Feature compatibility ✓, TTNN factory concept ✓ (all 3 factories `Is able to port? == yes`), Offset base pointers ✓, TensorAccessor 3rd arg ✓ (N/A). The op is a clean `descriptor`-concept op with three factories, all targeting `ProgramSpecFactoryConcept`. Port work is routine: express the `Buffer*` runtime-arg bindings as `TensorParameter` / `TensorBinding` (all Case 1), self-loop one scratchpad CB, and reuse/create the shared-kernel `_metal2` forks. `METAL2_PORT_BRIEF.md` is written.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** The readiness sheet ("Operations analysis", fetched this session) carries three rows for `data_movement/copy` — one per factory — all with `Is able to port? == yes`. Primary-column cross-check against the code is clean:
  - `Concept == descriptor` — confirmed: each factory defines `create_descriptor()` returning a `ProgramDescriptor` (`copy_device_operation.hpp:24-42`).
  - `Custom hash == no` — confirmed: no `compute_program_hash` override anywhere in the op.
  - `Runtime-args update (get_dynamic_runtime_args) == no` — confirmed: no such hook on the device-op.
  - `Override runtime args method? == no` — confirmed: no `override_runtime_arguments`. → target concept is plain `ProgramSpecFactoryConcept`.
  - `Pybind descriptor == no` — confirmed: `copy_nanobind.cpp` binds only `copy`/`assign` free functions, no `create_descriptor`.
  - `Smuggled pointer == no`, `TensorParameter relaxation == none`, `Known op issues` empty, `Op-owned tensors?` blank.
  - **Factory-set match:** sheet rows {`SameMemoryConfig`, `DefaultRowMajor`, `DefaultTilized`} map one-to-one to the code's three factories — no phantom/missing rows.
  - No cross-column invariant is violated. (`Op Classification == "PD Op (pointer-patching)"` and `Pointer patching perf issue? == "OK"` reflect the interim `Buffer*`-`BufferBinding` delivery — correct-on-cache-hit, not a block.)

- **Device 2.0 (every kernel used):** **GREEN.** All twelve referenced kernels use Device 2.0 idioms throughout (`Noc`, `DataflowBuffer` / `CircularBuffer` wrapper objects, `TensorAccessor`, `CoreLocalMem`, `noc.async_read`/`async_write`). No `InterleavedAddrGen` / `ShardedAddrGen` / raw `noc_async_*` / manual CB-index management anywhere.

  Only free functions taking a `uint32_t` CB index that appear are the **sanctioned** ones — not violations:

  | Sanctioned free function | Site |
  |---|---|
  | `get_tile_size(cb_id)` | `device/kernels/reader_unary_start_id.cpp:22`, `device/kernels/writer_unary_start_id.cpp:25` |
  | `get_local_cb_interface(cb_id).fifo_page_size` | `eltwise/unary/.../reader_unary_interleaved_start_id.cpp:25`, `.../writer_unary_interleaved_start_id.cpp:24` |

  In-family helper `tt::data_movement::common::tt_memmove(Noc noc, …)` (`data_movement/common/kernels/common.hpp:143`) takes a `Noc` object — Device 2.0 native.

  Kernels audited (all Device 2.0):
  - *Own* (`copy/device/kernels/`): `reader_unary_start_id.cpp`, `writer_unary_start_id.cpp`, `reader_unary_stick_start_id.cpp`, `writer_unary_stick_start_id.cpp`, `redistribute_pages_row_major_reader.cpp`, `redistribute_pages_row_major_writer.cpp`.
  - *Shared pool* (`ttnn/cpp/ttnn/kernel/`): `dataflow/reader_unary_stick_layout_interleaved_start_id.cpp`, `dataflow/writer_unary_stick_layout_interleaved_start_id.cpp`, `compute/eltwise_copy.cpp`.
  - *Cross-family* (`eltwise/unary/`): `reader_unary_interleaved_start_id.cpp`, `writer_unary_interleaved_start_id.cpp`.
  - *In-family* (`data_movement/sharded/`): `device/kernels/compute/eltwise_copy.cpp`.

- **Feature compatibility:** all Appendix A entries **N/A** — no signal fires anywhere in host code, factories, or kernels.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, no `remote_cb`/`remote_index`, no `.global_circular_buffer` field. CBs are plain `CBDescriptor`s. |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `address_offset` set on any `CBDescriptor`; no `set_address_offset` / 4-arg `UpdateDynamicCircularBufferAddress` / `cb_descriptor_from_sharded_tensor`. |
  | GlobalSemaphore | N/A | No `GlobalSemaphore` type or `CreateGlobalSemaphore`. The op uses no semaphores at all. |

- **CB endpoints (GATE-free):** every CB is legal 1:1 except one single-toucher scratchpad (self-loop). Device 2.0 gate is GREEN, so idioms are intact and this census is trustworthy. No dead CBs, no multi-binding, no hidden second writer (actively checked: the redistribute reader's `get_write_ptr()` writes are all to CBs it owns; no second kernel co-fills any CB). Per `(CB, config)`:

  | Factory | CB | Config | Producer | Consumer | Disposition |
  |---|---|---|---|---|---|
  | `SameMemoryConfig` | c_0 | tilized, no convert | reader | writer | legal 1:1 |
  | `SameMemoryConfig` | c_0 | tilized, convert-dtype | reader | compute (`kernel/compute/eltwise_copy.cpp`) | legal 1:1 |
  | `SameMemoryConfig` | c_0 | row-major (sharded/interleaved) | reader | writer | legal 1:1 |
  | `SameMemoryConfig` | c_16 | tilized, convert-dtype only | compute | writer | legal 1:1 |
  | `DefaultRowMajor` | c_0 | all | reader (scratchpad — `reserve_back`/`get_write_ptr`/`push_back`/`wait_front`/`pop_front`, `redistribute_pages_row_major_reader.cpp:42,61,183-185`) | *(same kernel)* | **self-loop** — one toucher; the writer never references c_0 |
  | `DefaultRowMajor` | c_1 | all | reader | writer | legal 1:1 |
  | `DefaultTilized` | c_0 | no convert | reader | writer | legal 1:1 |
  | `DefaultTilized` | c_0 | convert-dtype | reader | compute (`sharded/.../eltwise_copy.cpp`) | legal 1:1 |
  | `DefaultTilized` | c_16 | convert-dtype only | compute | writer | legal 1:1 |

  (`SameMemoryConfig`/`DefaultTilized` dtype-conversion path exists only for TILE layout — the op forbids RM dtype conversion, `copy_device_operation.cpp:134-135`.)

- **Offset base pointers:** **GREEN** — no address RTA folds a host-side offset into its base. The op contains **no** `buffer()->address()` expression at all. Each factory delivers the tensor base by pushing the `Buffer*` itself into the runtime-arg list (`Buffer*`-`BufferBinding` form): `copy_same_memory_config_program_factory.cpp:169,173,180,186`; `copy_default_row_major_program_factory.cpp:172-173`; `copy_default_tilized_program_factory.cpp:144-145`. No `base + offset` arithmetic; no Type-1/Type-2/Type-3/Type-4 pattern. Clean bases hand off to TensorParameter analysis below.

- **TensorAccessor 3rd argument:** **N/A** — no accessor in the op passes a 3rd argument. Every `TensorAccessor(...)` construction across all ten data-movement kernels is the 2-argument form `TensorAccessor(args, base_addr)` (verified by scan). The subject never fires. *(No sites — distinct from "sites found and classified redundant".)*

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding, per factory) — all **Case 1** (`Buffer*` base delivered via runtime arg → consumed through `TensorAccessor`). None are raw-pointer (Case 2) and none are borrowed-memory DFB (no `set_globally_allocated_address` anywhere). Routine port work: replace each `Buffer*` runtime-arg with a `TensorParameter` / `TensorBinding`; kernels build `TensorAccessor(tensor::name)` and the `TensorAccessorArgs(...).append_to(...)` CTA plumbing disappears.

  | Factory | Binding | Host delivery | Kernel use | Case |
  |---|---|---|---|---|
  | `SameMemoryConfig` | input (`src_buffer`) | RTA `Buffer*` @ `:169`/`:180` | `TensorAccessor(src_args, src_addr)` (reader) | 1 |
  | `SameMemoryConfig` | output (`dst_buffer`) | RTA `Buffer*` @ `:173`/`:186` | `TensorAccessor(dst_args, dst_addr)` (writer) | 1 |
  | `DefaultRowMajor` | input (`input.buffer()`) | RTA `Buffer*` @ `:172` + CTA `TensorAccessorArgs` @ `:136` | `TensorAccessor(src_args, src_addr)` @ reader `:38` | 1 |
  | `DefaultRowMajor` | output (`output.buffer()`) | RTA `Buffer*` @ `:173` + CTA `TensorAccessorArgs` @ `:148` | `TensorAccessor(dst_args, dst_addr)` @ writer `:31` | 1 |
  | `DefaultTilized` | input (`input.buffer()`) | RTA `Buffer*` @ `:144` + CTA `TensorAccessorArgs` @ `:105` | `TensorAccessor(src_args, src_addr)` (donor reader) | 1 |
  | `DefaultTilized` | output (`output.buffer()`) | RTA `Buffer*` @ `:145` + CTA `TensorAccessorArgs` @ `:110` | `TensorAccessor(dst_args, dst_addr)` (donor writer) | 1 |

- **TensorParameter relaxation:** `none` (sheet, all 3 factories) — no relaxation to apply.
- **TensorAccessor 3rd arg:** none — nothing to drop.
- **CB endpoints:** self-loop `DefaultRowMajor` c_0 (all configs); all other CBs legal 1:1.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none — no multi-binding, no hidden second writer.
- **Cross-op / shared kernels (file-path borrows):**

  | Borrowed kernel | Owner / pool | `_metal2` fork? | Other binding ops (sunset list — *not* authorization to convert in place) |
  |---|---|---|---|
  | `ttnn/cpp/ttnn/kernel/dataflow/reader_unary_stick_layout_interleaved_start_id.cpp` | shared pool (`kernel/`) | **no fork** — copy's port creates the first | `embedding` (rm + tilized-indices factories), `data_movement/concat` |
  | `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp` | shared pool (`kernel/`) | **no fork** — copy's port creates the first | `embedding`, `data_movement/concat` (co-borrowers overlap with the reader) |
  | `ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp` | shared pool (`kernel/`) | **no fork** — copy's port creates the first | `sharded/sharded_to_interleaved`, `sharded_partial/sharded_to_interleaved_partial`, `untilize_with_unpadding` |
  | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` | cross-family (`eltwise/unary`) | **fork EXISTS** (`reader_unary_interleaved_start_id_metal2.cpp`) — bind it, don't re-fork | broadly shared (dozens: `reduction/generic`, `concat`, `pad`, `untilize*`, `tilize*`, `kv_cache`, examples, …) |
  | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | cross-family (`eltwise/unary`) | **fork EXISTS** (`writer_unary_interleaved_start_id_metal2.cpp`) — bind it, don't re-fork | broadly shared (as above) |
  | `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/compute/eltwise_copy.cpp` | in-family (`data_movement/sharded`) | **no fork** — copy's port creates the first | `sharded/interleaved_to_sharded`, `sharded_partial/interleaved_to_sharded_partial` |

  `DefaultRowMajor` borrows no kernels (uses only the op's own `redistribute_pages_*`). Function-call escape: `redistribute_pages_row_major_reader.cpp:11` includes `data_movement/common/kernels/common.hpp` and calls `tt_memmove(Noc, …)` — in-family, Device 2.0 native, does not gate (see donor-shape detail below).

- **RTA varargs:** none. Every kernel reads a fixed set of distinct runtime args by constant index (`src_addr`, `stick_size`, `num_sticks`, `start_id`, `num_shards`, etc.) — no variable-count loop reading RTAs, no data-selected element. The `for (k < num_shards)` loops in the stick kernels read no RTAs inside the loop.

## Team-only

- **Out-of-directory coupling & donor shape (full inventory):**
  - **Roll-up: ✓ clean.** No ✗/⚠/⭐ function-call shapes. The only cross-directory function-call escape is `tt_memmove` (in-family `data_movement/common`), whose signature takes `Noc noc` — a Device 2.0-native shape that bridges cleanly. All other kernel includes resolve to `tt_metal/*` (LLK/HAL: `api/dataflow/*`, `api/compute/*`, `api/tensor/*`, `api/core_local_mem.h`) — no concern.
  - **File-path kernel instantiation:** six borrowed kernels (table above). This induces cross-op coordination cost (shared-kernel `_metal2` fork discipline) but does not gate. Two of the six (the `eltwise/unary` interleaved reader/writer) already have `_metal2` forks to bind; the other four need first forks created by copy's port (or a co-borrower's, whichever ports first).
- **TTNN factory analysis (sheet-derived, cross-checked):** concept `descriptor` (all factories); no custom hash; no `get_dynamic_runtime_args`; no `override_runtime_arguments`; no pybound `create_descriptor`; no op-owned tensors; target `ProgramSpecFactoryConcept`. `Execution Model == SPMD` on the sheet but `Concept == descriptor` (not `WorkloadDescriptor`) — the single-program descriptor path, no MeshWorkload artifact.

## Misc anomalies  *(team-only, non-gating)*

- **Dead compile-time arg:** `redistribute_pages_row_major_reader.cpp:24` declares `constexpr uint32_t num_output_pages_in_row = get_compile_time_arg_val(2);` but never uses it in the kernel body (only `num_input_pages_in_row` is used, at `:70`). The factory still emits it at `copy_default_row_major_program_factory.cpp:127`. Harmless (a `constexpr` the compiler drops), but it is a dead CTA the ops team may wish to prune. Not porter work.

## Questions for the user

*(none)*

## Recipe notes

*(none — the recipe covered every case cleanly. A `descriptor` op that delivers bases exclusively via the `Buffer*`-`BufferBinding` form, with zero `->address()` sites, is squarely in scope: Offset base pointers reads GREEN by absence, and TensorParameter analysis classifies each `Buffer*` binding Case 1 by kernel use, exactly as the [Buffer*-binding form](#) bullet prescribes.)*
