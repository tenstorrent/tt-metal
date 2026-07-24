# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/repeat`

Device operations and program factories in this directory:

- **`RepeatDeviceOperation`** (`device/repeat_device_operation.hpp` / `.cpp`)
  - `RepeatProgramFactoryLastDim` (`device/repeat_program_factory_last_dim.cpp`)
  - `RepeatProgramFactoryHigherDim` (`device/repeat_program_factory_higher_dim.cpp`)

Single DeviceOperation, so this is one combined (unbundled) report. `select_program_factory` routes on the operation attributes: a positive `m_tile_page_size_bytes` picks the higher-dim factory in its tile-native mode, `m_is_last_dim` picks the last-dim factory, and everything else picks the higher-dim factory in its row-major mode. Each factory then chooses its kernel and circular-buffer set from the runtime sharding state of the input and output buffers.

Kernels referenced by the factories (all five are live; none are unreferenced):

| Factory | Config | Kernel |
|---|---|---|
| HigherDim | tile-native (`m_tile_page_size_bytes > 0`) | `device/kernels/repeat_higher_dim_tile.cpp` |
| HigherDim | row-major, src or dst sharded | `device/kernels/repeat_higher_dim_rm_sharded.cpp` |
| HigherDim | row-major, interleaved | `device/kernels/repeat_higher_dim_rm_interleaved.cpp` |
| LastDim | row-major, src or dst sharded | `device/kernels/repeat_last_dim_rm_sharded.cpp` |
| LastDim | row-major, interleaved | `device/kernels/repeat_last_dim_rm_interleaved.cpp` |

The higher-level `ttnn::repeat` host entry point in `repeat.cpp` composes the primitive over several *other* ops (`view`, `to_layout`, `sharded_to_interleaved`, `interleaved_to_sharded`, `zeros`). Those are separate ops with their own audits and are out of scope here. This audit covers the `ttnn::prim::repeat` / `ttnn::prim::repeat_tile` primitive and its two factories only.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `44da718b06b 2026-07-23 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/repeat` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `RepeatDeviceOperation` → `RepeatProgramFactoryLastDim`, `RepeatProgramFactoryHigherDim` |
| *Prereqs* — Device 2.0 (every kernel used) | Yes (own kernels + in-family `common.hpp` donor helpers all Device 2.0) |
| *Prereqs* — Cross-op escapes | Ok (one in-family header escape; no cross-family donors; no borrowed kernel files) |
| *Feature Support* — overall | GREEN |
| *Feature Support* — Variadic-CTA | Ok (fixed-count CTAs, single input tensor) |
| *TTNN Readiness* — `Is able to port?` (the gate) | Yes (both factories) |
| *TTNN Readiness* — Concept (current) | `descriptor` (both factories) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Is safe to port? | Yes (both factories; `Smuggled pointer = no`) |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — Runtime-args update | No |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` (both factories, no op-owned tensors) |
| *Port work* — Offset base pointer | none (GATE cleared) |
| *Port work* — Tensor bindings (per binding) | input → Case 1 · output → Case 1 (both via `TensorAccessor`) |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none present (GATE cleared) |
| *Port work* — CB endpoints | self-loop (every CB, every config) |

## Result

**GREEN → brief issued.** Both program factories clear every gate. Device 2.0 is complete for the op's own kernels and for the in-family `data_movement/common` helpers they call; no Appendix A feature is in use; and the readiness sheet reports `Is able to port? = yes` for both factories, which the code cross-check confirms. The two gate-bearing porter subjects (Offset base pointers, TensorAccessor 3rd argument) are both clear: no address arithmetic is folded into any base pointer, and no `TensorAccessor` passes a third page-size argument.

Port work is routine: bind the input and output tensors as `TensorParameter` / `TensorBinding` (both are consumed through a `TensorAccessor`, so both are Case 1), self-loop each single-toucher circular buffer, and drop the now-redundant address runtime args and `TensorAccessorArgs` plumbing. See `METAL2_PORT_BRIEF.md`.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN** for both factories. The readiness sheet ("Operations analysis", fetched fresh this run) has one row per factory, both reading `Is able to port? = yes`. Cross-check against the code is clean on every cheaply-checkable column:

  | Column | Sheet (both rows) | Code cross-check |
  |---|---|---|
  | `Concept` | `descriptor` | `create_descriptor()` returning `ProgramDescriptor` in both factory `.cpp` files — confirmed |
  | `Custom hash (compute_program_hash)` | `no` | no `compute_program_hash` override in the device op — confirmed by grep |
  | `Runtime-args update (get_dynamic_runtime_args)` | `no` | no `get_dynamic_runtime_args` in either factory — confirmed by grep |
  | `Override runtime args? (PD or legacy)` | `no` | no `override_runtime_arguments` — confirmed by grep |
  | `Pybind descriptor (nb::class_ of device op)` | `no` | `repeat_nanobind.cpp` binds only the `ttnn::repeat` free function; no `create_descriptor` binding, no `nb::class_` of the device op — confirmed |
  | `Smuggled pointer (raw buffer addr in RTA/CRTA)` | `no` | buffers ride the `Buffer*` binding channel, not `->address()` in a runtime arg (see Tensor bindings below) — confirmed |
  | `Is safe to port?` | `yes` | (readiness-sheet owner's correctness axis; not re-derived) |

  Cross-column invariants hold: `Runtime-args update = no` is legal on a `descriptor` concept, and `Op-owned tensors?` is blank (a `descriptor` concept cannot carry op-owned tensors). Target concept per [TTNN porting shape]: `descriptor` with no op-owned tensors maps to `MetalV2FactoryConcept`.

- **Device 2.0 (every kernel used):** **GREEN.** No violations. All five of the op's kernels are structurally Device 2.0: they use the `Noc` object (`noc.async_read`, `noc.async_write`, `noc.async_read_barrier`, `noc.async_write_barrier`), the `DataflowBuffer` object with method-style `reserve_back` / `get_write_ptr` / `push_back`, `TensorAccessor`, and `CoreLocalMem`. There are no `InterleavedAddrGen` / `ShardedAddrGen` idioms, no raw `noc_async_read(addr, ...)` free calls, no raw semaphore addresses, and no CB-index-keyed free-function holdovers (`get_write_ptr(cb_id)` and friends). The only free functions the kernels call are the in-family shared helpers in `data_movement/common/kernels/common.hpp`, which are themselves Device 2.0 (see Out-of-directory coupling): `tt_memmove(Noc, ...)`, `noc_async_read_sharded(Noc, ..., AddrGenType tensor, ...)`, `noc_async_write_sharded(Noc, ...)`, and `align_address`. In every call site the repeat kernels use the non-deprecated `Noc`-first overload, not the deprecated address-only overload. The sanctioned `get_tile_size(cb_id)` / `get_local_cb_interface(cb_id)` free functions do not appear at all.

- **Feature compatibility:** **GREEN.** Every Appendix A entry is absent (`N/A`). No entry's recognition signals fire anywhere in the host code, factories, descriptors, or kernels.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no `GlobalCircularBuffer` type, no `CreateGlobalCircularBuffer`, no `remote_cb` / `remote_index` idiom, no `.global_circular_buffer` field set on any `CBDescriptor` |
  | CBDescriptor `address_offset` (non-zero) | N/A | no `.address_offset` field is set on any `CBDescriptor` literal (both factories set only `total_size`, `core_ranges`, `format_descriptors`), no `set_address_offset`, no 4-arg `UpdateDynamicCircularBufferAddress`, no `cb_descriptor_from_sharded_tensor` |
  | GlobalSemaphore | N/A | the op uses no semaphores at all |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t` is a single named `Tensor` (`RepeatInputs`), not a variable-count container; kernels read compile-time args only at fixed constexpr offsets plus fixed `TensorAccessorArgs<N, M>()` positions; no `get_compile_time_arg_val(i)` under a runtime-varying index |

- **CB endpoints (GATE-free):** every circular buffer is **single-toucher → self-loop**, in every config. Each program instantiates exactly one kernel (a single reader that performs the full read-repeat-write); that kernel is the sole toucher of every CB in its program, so each CB binds the one kernel as both PRODUCER and CONSUMER. There are no dead CBs, no multi-binding CBs, and no hidden second writers (there are no semaphores to coordinate one). Per `(CB, config)`:

  | Factory | Config | CBs (buffer_index) | Disposition |
  |---|---|---|---|
  | HigherDim | tile-native | `0` (src0) | self-loop |
  | HigherDim | RM sharded | `0` (src0) | self-loop |
  | HigherDim | RM interleaved | `0` (src0), `1` (src1, alignment scratchpad) | self-loop (each) |
  | LastDim | RM sharded | `0` (src0) | self-loop |
  | LastDim | RM interleaved | `0` (src0), `1` (src1, alignment scratchpad) | self-loop (each) |

  The second CB (index `1`) exists only in the interleaved row-major configs, where `needs_alignment_cb` is true; it is a write-alignment scratchpad. Nothing here blocks a Gen1 port.

- **Offset base pointers:** **GREEN — cleared.** No address runtime arg folds a host-side offset into a base pointer. Both factories pass the raw `Buffer*` objects (`src_buffer`, `dst_buffer`) into `emplace_runtime_args`; the framework converts these to clean base addresses on the binding channel. There is no `buffer()->address() + <offset>` expression anywhere in the factories. On the device side, the kernels compute page *indices* (`read_offset`, `write_offset`, `.page_id`) and within-page byte offsets that ride the NoC transfer descriptor or a separate helper argument; none of these are folded into the accessor's base. In particular `repeat_last_dim_rm_sharded.cpp:48` passes the per-replica write offset `k * original_page_size_bytes` as a *separate* argument to `noc_async_write_sharded`, keeping the accessor base clean (the already-split-out shape the triage doc calls a Type-1 non-issue). `repeat` appears in none of the Type 1/2/4 tables of the offset-base-pointer triage analysis (`analyses/2026-07-19_offset_base_pointers.md`, a dated prior), and this independent scan of every address arg agrees: no fold present.

- **TensorAccessor 3rd argument:** **GREEN — cleared.** The subject does not fire: all ten `TensorAccessor(...)` constructions across the five kernels are the two-argument form `TensorAccessor(args, addr)`. No site passes an explicit page-size third argument, so there is nothing to classify or drop. `repeat` appears in none of the rows of the 3rd-arg triage analysis (`analyses/2026-07-06_tensor_accessor_3rd_arg_triage.md`, a dated prior), consistent with this scan.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding, identical across both factories and all five kernels):
  - **input** — **Case 1** (via `TensorAccessor`). The kernel receives the base as `src_addr = get_arg_val<uint32_t>(0)` (delivered today by the `Buffer*` binding of `src_buffer`) and constructs `TensorAccessor(src_args, src_addr)`; all reads go through that accessor (`s.get_noc_addr(...)`, `noc.async_read(s, ...)`, or `noc_async_read_sharded(noc, cb_slot, s, ...)`).
  - **output** — **Case 1** (via `TensorAccessor`). Same shape via `dst_addr = get_arg_val<uint32_t>(1)` → `TensorAccessor(dst_args, dst_addr)`; all writes go through that accessor.
  - Port action for both: express as `TensorParameter` / `TensorBinding`; the kernel builds `TensorAccessor(tensor::name)`; the address runtime arg and the `TensorAccessorArgs(...).append_to(...)` plumbing both disappear. No borrowed-memory (clean) bindings exist; the CBs are internal scratchpads, not tensor-backed.
- **TensorParameter relaxation:** **none** (sheet column `TensorParameter relaxation = none`; the op has no custom hash).
- **TensorAccessor 3rd arg:** **none** (no third-argument sites present).
- **CB endpoints:** **self-loop** on every CB in every config (table above). No 1P+1C assignments, no multi-binding flags, no dead-CB drops.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none. Every CB is single-toucher; there are no split-reader, dual-instance-work-split, or hidden-co-fill shapes (each program has exactly one kernel, and the op uses no semaphores).
- **Cross-op / shared kernels:** the op owns all five of its kernel `.cpp` files (no file-path instantiation of a borrowed kernel). The kernels do `#include` one in-family shared header, `ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp`, for `tt_memmove`, `noc_async_read_sharded`, `noc_async_write_sharded`, `align_address`, and the `MASK_*` / `OFFSET_*` alignment constants. This is a function-call escape within the same op family (`data_movement`); it does not gate the Metal 2.0 syntax rewrite, but the family should be ported as a unit. The donor signatures translate cleanly to Metal 2.0 tokens (see Team-only).
- **RTA varargs:** none. Every runtime arg is read at a fixed distinct index and names a single field (`src_addr`, `dst_addr`, the per-dim start/end bounds, `repetitions` / `num_repeats`, `nop`). There is no count-bounded or data-selected `get_arg_val` loop. The `TensorAccessorArgs` common runtime args are framework-managed and are subsumed by the tensor binding at port time. The porter names each runtime arg.

## Team-only

- **Out-of-directory coupling & donor shape:**
  - **Op-level roll-up:** ✓ clean. One in-family function-call escape; no cross-family donors; no file-path (borrowed) kernel instantiation.
  - **Summary table** (one row per op-kernel → donor-file pair):

    | Op kernel(s) | Donor file | Class | Status |
    |---|---|---|---|
    | all five repeat kernels | `ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp` | in-family shared (same op family) | ✓ |

  - **Per-call detail** (functions the repeat kernels actually call, by handle shape):

    | Function | Signature handles | Shape | Status |
    |---|---|---|---|
    | `noc_async_read_sharded` / `noc_async_write_sharded` | `Noc noc`, `uint32_t l1_addr` (a raw CB-slot address), `AddrGenType tensor` (a `TensorAccessor`) | `Noc` native + plain address + `TensorAccessor` (Shape 1) | ✓ excellent — porter passes `TensorAccessor(tensor::name)` and the DataflowBuffer write pointer |
    | `tt_memmove` | `Noc noc`, two `uint32_t` SRAM addresses, `uint32_t bytes` | `Noc` native + plain addresses | ✓ excellent |
    | `align_address` | `uint32_t address`, `uint64_t mask` | pure arithmetic | ✓ excellent |

    None of these donor functions takes a `uint32_t sem_id` / `sem_addr`, a `CircularBuffer&`, a `TensorAccessorArgs<N>`, or a legacy addr-gen type, so none of the flagged/blocked shapes apply. `common.hpp` also carries deprecated address-only overloads of `tt_memmove` and the sharded helpers, but the repeat kernels do not use them.

  - **Borrowed kernel files (file-path instantiation):** none. Every `kernel_source` string points into the op's own `device/kernels/` directory.

- **TTNN factory analysis** (sheet-derived facts, code-confirmed):
  - **Current concept:** `descriptor` (both factories).
  - **Op-owned tensors:** none (sheet blank; a `descriptor` concept cannot carry them).
  - **MeshWorkload need:** none (not a `WorkloadDescriptor` op).
  - **Pybind `create_descriptor`:** absent (`repeat_nanobind.cpp` binds only the `ttnn::repeat` free function).
  - **Other risky pybind:** none.
  - **Custom hash:** absent.
  - **Custom `override_runtime_arguments`:** absent.
  - **Target concept:** `MetalV2FactoryConcept` (no op-owned tensors), for both factories.

## Misc anomalies  *(team-only, non-gating; the port does not act on these)*

- **Operator-precedence defect in the last-dim CB size expression.** In `repeat_program_factory_last_dim.cpp:53-57`, `cb_size_bytes` is written as:

  ```cpp
  const uint32_t cb_size_bytes = READ_ALIGNMENT * 2 + (source_page_size_bytes & 0xF) == 0 ? source_page_size_bytes
                                 : (source_page_size_bytes & 0x7) == 0                    ? source_page_size_bytes * 2
                                 : ...
  ```

  C++ binds `*` and `+` tighter than `==`, so the first condition parses as `((READ_ALIGNMENT * 2 + (source_page_size_bytes & 0xF)) == 0)`, i.e. `((128 + (spb & 0xF)) == 0)`, which is never true. Two consequences: (1) the intended "multiple of 16 → `source_page_size_bytes`" branch is unreachable, so a multiple-of-16 page falls through to the "multiple of 8" branch and gets `source_page_size_bytes * 2`; and (2) the `READ_ALIGNMENT * 2` (128-byte) alignment headroom that the kernel comment in `repeat_last_dim_rm_interleaved.cpp:29-34` documents for each CB page is dropped from every branch. Computed sizes fall below the documented intent for most page sizes (for example `spb = 16` yields 32 bytes rather than the intended 144, and `spb = 64` yields 128 rather than 192). The analogous higher-dim factory computes the size correctly as `(READ_ALIGNMENT * 2) + page_size_bytes` (`repeat_program_factory_higher_dim.cpp:66`). Whether the row-major-interleaved staging buffer is actually under-allocated relative to the kernel's read-alignment, page-doubling, and write-alignment needs should be confirmed by the ops team. This is pre-existing host logic; the port would carry the expression into `DataflowBufferSpec::total_size` verbatim (zero functional change), so any fix belongs on the ops track, not in the port diff.

## Recipe notes

- **Readiness-sheet column split vs. the recipe's cross-check.** The current "Operations analysis" sheet has *two* runtime-args columns: `Runtime-args update (get_dynamic_runtime_args)` and a separate `Override runtime args? (PD or legacy)`. The recipe's cross-check bullet folds both `get_dynamic_runtime_args` and `override_runtime_arguments` under the single name `Runtime-args update`, and the gate-derivation formula names only `Runtime-args update`. Both columns are `no` for this op, so there was no ambiguity to resolve here, but a future auditor mapping the grep results to sheet columns may briefly wonder which column `override_runtime_arguments` feeds. A one-line note in the cross-check bullet acknowledging the sheet's two-column split would remove that friction.
