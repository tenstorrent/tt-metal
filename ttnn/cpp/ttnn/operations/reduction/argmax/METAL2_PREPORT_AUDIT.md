# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/reduction/argmax`

The directory holds **two** DeviceOperations, both reachable from the single user-facing
`ttnn::argmax` facade (`argmax.cpp`), which routes between them by dim/dtype/layout:

- **`ArgMaxDeviceOperation`** (`device/argmax_device_operation.{hpp,cpp}`) — `descriptor` concept
  - `ArgMaxSingleCoreProgramFactory` (`device/argmax_single_core_program_factory.cpp`)
    - RM input → `kernels/reader_argmax_interleaved.cpp`
    - TILE input, dim == W → `kernels/reader_argmax_tile_layout.cpp`
    - TILE input, dim == H → `kernels/reader_argmax_tile_layout_h.cpp`
  - `ArgMaxMultiCoreProgramFactory` (`device/argmax_multi_core_program_factory.cpp`)
    - `kernels/reader_argmax_interleaved_multicore.cpp`
- **`ArgMaxNCDeviceOperation`** (`device/argmax_nc_device_operation.{hpp,cpp}`) — **`legacy device-op`** concept
  - `ArgMaxNCProgramFactory` (`device/argmax_nc_program_factory.cpp`)
    - `kernels/reader_argmax_nc.cpp`, `kernels/argmax_nc_compute.cpp`, `kernels/writer_argmax_nc.cpp`

**Bundling rationale.** The two DeviceOperations share **no** factories and **no** kernels
(the NC kernels include none of `argmax_common.hpp` / `argmax_tile_layout.hpp` /
`argmax_tile_h_col.hpp`), so on the recipe's shared-code test they are *independent* and the
recipe would have them audited separately. They are reported together anyway because the audit's
output filename is **op-directory-scoped** — two independent DeviceOperations in one directory
cannot each own a `METAL2_PREPORT_AUDIT.md`. Findings are attributed per DeviceOperation
throughout, and a *Per-DeviceOperation attribution* section is provided. See *Recipe notes*.

**Kernel-file census.** All seven kernel `.cpp` files in `device/kernels/` are referenced by a
factory; none are dead. The three `.hpp` files in that directory are headers included by the
`ArgMaxDeviceOperation` readers only.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `b73b958088a 2026-08-24 docs(metal_2.0): a run in flight freezes the kernel sources`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/reduction/argmax` |
| **Overall** | **RED at op level; subset `ArgMaxDeviceOperation` (single-core + multi-core factories) is clear** |
| **DOps / Factories** | `ArgMaxDeviceOperation` → `ArgMaxSingleCoreProgramFactory`, `ArgMaxMultiCoreProgramFactory` · `ArgMaxNCDeviceOperation` → `ArgMaxNCProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all 7 kernels structurally Device 2.0 (`Noc`, `CircularBuffer` / `DataflowBuffer`, `Semaphore<>`, `UnicastEndpoint`, `TensorAccessor`). One judgment call recorded below (`get_dataformat(cb_id)`) |
| *Prereqs* — Cross-op escapes | **Ok** — every kernel `#include` resolves to `tt_metal/*` (`api/…`, `tt-metalium/constants.hpp`) or an in-directory header. Zero borrowed kernel files. One *reverse* coupling reported (a gtest consumer) |
| *Feature Support* — overall | **GREEN** — no Appendix A entry fires |
| *Feature Support* — Variadic-CTA | Ok (no variable-index compile-time-arg reads anywhere) |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Per-factory split.** `ArgMaxSingleCoreProgramFactory` = `yes` · `ArgMaxMultiCoreProgramFactory` = `yes` · **`ArgMaxNCProgramFactory` = `no`**, attributed to `Concept` == `legacy device-op` |
| *TTNN Readiness* — Concept (current) | `descriptor` (both `ArgMaxDeviceOperation` factories) · **`legacy device-op`** (`ArgMaxNCProgramFactory`) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A — no factory is `WorkloadDescriptor` |
| *TTNN Readiness* — Custom hash | **No** — sheet `no` on all three rows; `grep` for `compute_program_hash` / `attribute_values` / `to_hash` over the op directory returns nothing |
| *TTNN Readiness* — `get_dynamic_runtime_args` | **No** — sheet `no` on all three rows; no hook on either device-op |
| *TTNN Readiness* — `override_runtime_arguments` | **No** on the two `descriptor` factories. `ArgMaxNCProgramFactory` defines one (`argmax_nc_program_factory.cpp:218`) but as part of the **legacy** concept signature — the sheet correctly records that column as `n/a` there, and it gates via `Concept`, not via this column |
| *TTNN Readiness* — Pybind `create_descriptor` | **No** — `argmax_nanobind.cpp` binds only the user-facing `ttnn::argmax`; no `create_descriptor` binding, no `nb::class_` of a device op |
| *TTNN Readiness* — Op-owned tensors | **No** (`no` / blank on all rows; no `WorkloadDescriptor`, no `buffers` vector) |
| *TTNN Readiness* — Target concept | **`ProgramSpecFactoryConcept`** (for the clean `ArgMaxDeviceOperation` subset) |
| *Port work* — Offset base pointer | **none** — every address argument is a clean base; the multi-core factory already passes its byte offset as a *separate* RTA |
| *Port work* — Tensor bindings (per binding) | `input` **Case 1**, `output` **Case 1** — in both clean factories, all three single-core configs |
| *TTNN Readiness* — TensorParameter relaxation | **`none`** on all three rows → clears |
| *Port work* — TensorAccessor 3rd arg | **none — no accessor in the op passes a 3rd argument** (all 10 construction sites are 2-arg) |
| *Port work* — CB endpoints | **self-loop** on all 6 CBs of the clean subset (every CB has exactly one, sync-free, toucher per node). `ArgMaxNCProgramFactory`'s 2 CBs are plain 1:1 — recorded but deferred, see below |

**CB endpoints** are dispositions, not gates: every out-of-window CB has a port-time resolution
— a **self-loop** (one toucher: single-ended / sync-free), a **1P+1C assignment** (two touchers),
the **multi-binding advanced-option flag** (genuine ≥2 of a role on a node the census cannot
relabel), or a **dead-CB drop** (zero endpoints). Dispositions are recorded per `(CB, config)`.

## Result

**RED at op level; subset `ArgMaxDeviceOperation` (both factories) is clear.**

The single blocker is the **TTNN factory concept** gate on `ArgMaxNCProgramFactory`: that factory
is still on the legacy imperative `host_api.hpp` builder (`Program`, `CreateCircularBuffer`,
`CreateKernel`, `SetRuntimeArgs`, `CachedProgram` + `override_runtime_arguments`), so its concept
is `legacy device-op` and the readiness sheet's `Is able to port?` cell reads `no`. This is the
**expected** outcome for an op still on the legacy API, not an alarm — it unblocks when
`ArgMaxNCDeviceOperation`'s `ProgramDescriptor` migration lands, a separate ongoing effort owned by
the **TTNN / PD-migration team**. Routed there.

**Every other gate is clear, for both DeviceOperations** — Device 2.0 (all 7 kernels, including
the NC ones), feature compatibility (no Appendix A entry fires), offset base pointers (no
host-folded offset anywhere), and the `TensorAccessor` 3rd argument (no site exists). So when the
PD migration lands, `ArgMaxNCProgramFactory` should re-audit to GREEN with nothing else to clear.

Because a clean factory subset survives, a **porter brief is issued** — scoped to
`ArgMaxDeviceOperation`'s two factories only (`METAL2_PORT_BRIEF.md`).

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **per-factory split.**

  | Factory | `Concept` | `Is able to port?` | Verdict |
  |---|---|---|---|
  | `ArgMaxSingleCoreProgramFactory` | `descriptor` | `yes` | **cleared** |
  | `ArgMaxMultiCoreProgramFactory` | `descriptor` | `yes` | **cleared** |
  | `ArgMaxNCProgramFactory` | `legacy device-op` | **`no`** | **GATE** |

  The `no` is **fully attributed**: the blocking column is `Concept`, value `legacy device-op`.
  Route: **TTNN / PD-migration team**. This is a separate ongoing effort and the expected outcome
  for a legacy op; the Metal 2.0 gate lifts when that op's `ProgramDescriptor` migration lands.
  Every other blocking column on that row is benign — `Runtime-args update
  (get_dynamic_runtime_args)` = `no`, `Smuggled pointer` = `no`, `TensorParameter relaxation` =
  `none`, `Known op issues` = *(empty)*, `Op-owned tensors?` = `no`.

  **Cross-check (trust, but verify) — clean on every checkable column:**

  | Column | Sheet | Code evidence | Match |
  |---|---|---|---|
  | `Concept` (ArgMax ×2) | `descriptor` | `create_descriptor()` returning `ProgramDescriptor` @ `argmax_device_operation.hpp:17,22`; bodies @ `argmax_single_core_program_factory.cpp:116`, `argmax_multi_core_program_factory.cpp:159` | ✓ |
  | `Concept` (NC) | `legacy device-op` | `create()` + `override_runtime_arguments()` on a `CachedProgram` @ `argmax_nc_device_operation.hpp:41-50`; imperative body @ `argmax_nc_program_factory.cpp:47,218` | ✓ |
  | `Custom hash` / `Backdoor custom hash` | `no` / `no` | no `compute_program_hash`, `attribute_values`, or `to_hash` in the op directory | ✓ |
  | `Runtime-args update (get_dynamic_runtime_args)` | `no` (all 3) | no such hook on either DeviceOperation | ✓ |
  | `Override runtime args method?` | `no`, `no`, `n/a` | absent on both `descriptor` factories; present on `ArgMaxNCProgramFactory` @ `argmax_nc_program_factory.cpp:218` — the *legacy*-concept signature, so `n/a` is the correct value per the recipe's name-collision rule | ✓ |
  | `Pybind descriptor` | `no` | `argmax_nanobind.cpp` binds only the user-facing op; no `create_descriptor` | ✓ |
  | `Smuggled pointer` | `no` | the `descriptor` factories pass tensors via the framework-registered `emplace_runtime_args(core, {input, output})` MeshTensor form, not a raw `->address()` (see *Tensor bindings*) | ✓ |
  | `Op-owned tensors?` | `no` / blank | no `WorkloadDescriptor`, no `buffers` vector | ✓ |
  | **Factory-set match** | 3 rows | 3 factories in code, one-to-one, no phantom and no missing row | ✓ |

  **Cross-column invariants:** `get_dynamic_runtime_args` == `no` everywhere (so the
  "`yes` only on `descriptor`/`WorkloadDescriptor`" invariant is vacuously held);
  `Op-owned tensors?` is never `yes`, so the "`yes` only on `WorkloadDescriptor`" invariant holds.
  No conflict, no violated invariant → **the sheet is sound for this op.**

  *Informational, non-gating, worth carrying:* `Op Classification` reads
  `PD Op (pointer-patching)` for the two `descriptor` rows with `Pointer patching perf issue?` =
  `OK` — consistent with the `emplace_runtime_args(core, {input, output})` binding form these
  factories use, which the Metal 2.0 typed binding supersedes.

- **Device 2.0 (every kernel used): GREEN — all 7 kernels.**

  Every data-movement kernel is on the Device 2.0 object surface: `Noc` for all NoC traffic,
  `CircularBuffer` / `DataflowBuffer` wrappers for CB access, `Semaphore<>` +
  `UnicastEndpoint` for the multi-core handshake, `TensorAccessor` for tensor addressing. Zero
  Device 1.0 idioms across the directory: no `noc_async_read` / `noc_async_write`, no
  `cb_reserve_back` / `cb_push_back` / `cb_wait_front` / `cb_pop_front` free functions, no
  `get_write_ptr(cb)` / `get_read_ptr(cb)` free functions, no `InterleavedAddrGen` /
  `ShardedAddrGen` / `InterleavedAddrGenFast` / `InterleavedPow2AddrGen*`, no raw semaphore
  addresses, no `get_noc_addr_from_bank_id`, no `evil_set_*_ptr`. Confirmed for the NC kernels too,
  even though that factory is gated elsewhere.

  **The one judgment call — `get_dataformat(<cb_idx>)`, and why it is *not* scored a holdover.**
  Four kernels query the buffer's data format through the CB-index free function, at a
  `constexpr` site whose value is then used as a template argument:

  | File | Line | Call | Wrapper in scope |
  |---|---|---|---|
  | `device/kernels/reader_argmax_interleaved.cpp` | `54` | `get_dataformat(src_cb_idx)` | `CircularBuffer src_cb` |
  | `device/kernels/reader_argmax_interleaved_multicore.cpp` | `317` | `get_dataformat(src_cb_idx)` | `CircularBuffer src_cb` |
  | `device/kernels/reader_argmax_interleaved_multicore.cpp` | `334` | `get_dataformat(red_vals_cb_idx)` | `CircularBuffer red_val_cb` |
  | `device/kernels/reader_argmax_tile_layout.cpp` | `63` | `get_dataformat(src_dfb_idx)` | `DataflowBuffer src_dfb` |
  | `device/kernels/reader_argmax_tile_layout_h.cpp` | `53` | `get_dataformat(src_dfb_idx)` | `DataflowBuffer src_dfb` |

  `get_dataformat` is **not** on the recipe's sanctioned list (which names only
  `get_tile_size(cb_id)` and `get_local_cb_interface(cb_id)`), and a wrapper object *is* in scope
  at each site — so it matches the holdover *shape*. It is nonetheless **not** a holdover, because
  the second half of the recipe's own holdover criterion is unmet: **no usable wrapper-method
  replacement exists.** `CircularBuffer::get_dataformat()` is a plain `const` member
  (`tt_metal/hw/inc/api/dataflow/circular_buffer.h:113`), *not* `constexpr`, and the
  `CircularBuffer` object is not a constant expression either — while the free function *is*
  `constexpr` (`tt_metal/hw/inc/api/dataflow/dataflow_api.h:300`) and every one of these five sites
  feeds a `constexpr DataFormat` that is consumed as a non-type template parameter
  (`get_default_value<fmt>()`, `compare_values<fmt>(…)`, `process_input_tile<T, fmt>(…)`,
  `find_argmax_from_intermediate_outputs<n, fmt>(…)`, plus a `static_assert` @
  `reader_argmax_interleaved_multicore.cpp:341`). Substituting the member would not compile. The
  lookup is also the same *class* of thing as the sanctioned `get_tile_size(cb_id)` — a
  compile-time CB metadata query, not a data-movement operation — and the recipe's own breadcrumb
  states these metadata lookups move onto the object at **port** time, which is exactly the
  resolution path here: `DataflowBuffer::get_dataformat()` **is** `constexpr`
  (`tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:279`), so the Metal 2.0 port can move them
  where Device 2.0 could not. Recorded as PORT WORK in the brief, not as a Device 2.0 gate.
  Flagged in *Recipe notes* as a sanctioned-list gap.

  The two kernels that still `#include "api/dataflow/circular_buffer.h"`
  (`reader_argmax_interleaved.cpp:7`, `reader_argmax_interleaved_multicore.cpp:8`) do so
  legitimately — they genuinely use the `CircularBuffer` wrapper and its
  `use<CircularBuffer::AddrSelector::WRITE_PTR>` view. This is *not* the stale-include idiom;
  the include goes away as part of the Metal 2.0 CB→DFB swap, not as a Device 2.0 fix.

- **Feature compatibility:** every Appendix A entry, in order. A clean scan is all-`N/A`.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | **N/A** | No `GlobalCircularBuffer` type, no `CreateGlobalCircularBuffer`, no `global_circular_buffer.hpp` include, no `CBDescriptor::global_circular_buffer` field set, no `experimental::CreateCircularBuffer(…, global_cb)` 4-arg form, no `CircularBufferConfig::remote_index()`, no `remote_cb_*` identifiers, no `remote_circular_buffer.h`. The multi-core factory's cross-core traffic is *not* a remote CB — it is a plain `noc.async_write` to an explicit `{.noc_x, .noc_y, .addr}` on the reducer core through a `UnicastEndpoint` (`reader_argmax_interleaved_multicore.cpp:416-427, 467-478`), using ordinary per-core CBs at both ends. |
  | CBDescriptor `address_offset` (non-zero) | **N/A** | No `.address_offset` on any `CBDescriptor`, no `set_address_offset`, no 4-arg `UpdateDynamicCircularBufferAddress`, no `cb_descriptor_from_sharded_tensor`. All five `CBDescriptor` literals (`argmax_single_core_program_factory.cpp:145,156`; `argmax_multi_core_program_factory.cpp:218,231,244,257,270`) leave the field at its default zero. |
  | GlobalSemaphore | **N/A** | No `GlobalSemaphore` type, no `CreateGlobalSemaphore`, no `global_semaphore.hpp` include. The multi-core factory's two semaphores are plain `SemaphoreDescriptor`s (`argmax_multi_core_program_factory.cpp:303,309`), consumed kernel-side as `Semaphore<>` (`reader_argmax_interleaved_multicore.cpp:347-348`) — the regular path, supported as `SemaphoreSpec`. |

- **CB endpoints (GATE-free):** census per `(CB, config)`, per node. **Every CB in the clean
  subset has exactly one toucher per node, and that toucher is sync-free (raw
  `get_write_ptr()` peek, no FIFO ops at all) → self-loop across the board.**

  `ArgMaxSingleCoreProgramFactory` — one reader kernel on the single node in every config:

  | CB | Config | Touchers on a node | Verdict | Disposition |
  |---|---|---|---|---|
  | `c_0` src | RM (`reader_argmax_interleaved.cpp`) | 1 — reader, raw `src_cb.get_write_ptr()` @ `:53` + `noc.async_read` destination @ `:67`; no FIFO ops | single-ended / sync-free | **self-loop** |
  | `c_1` dst | RM | 1 — reader, raw `dst_cb.get_write_ptr()` @ `:57`, written via `out_idxs[]` and drained by `noc.async_write(use<WRITE_PTR>(dst_cb), …)` @ `:86,99` | single-ended / sync-free | **self-loop** |
  | `c_0` src | TILE-W (`reader_argmax_tile_layout.cpp`) | 1 — raw `src_dfb.get_write_ptr()` @ `:62` + `noc.async_read` destination @ `:136` | single-ended / sync-free | **self-loop** |
  | `c_1` dst | TILE-W | 1 — raw `dst_dfb.get_write_ptr()` @ `:66`, then written/read through `CoreLocalMem<uint32_t>` in `write_to_output` (`argmax_tile_layout.hpp:329`) | single-ended / sync-free | **self-loop** |
  | `c_0` src | TILE-H (`reader_argmax_tile_layout_h.cpp`) | 1 — raw `src_dfb.get_write_ptr()` @ `:52` + `noc.async_read` @ `:104` | single-ended / sync-free | **self-loop** |
  | `c_1` dst | TILE-H | 1 — raw `dst_dfb.get_write_ptr()` @ `:55`, same `CoreLocalMem` path | single-ended / sync-free | **self-loop** |

  `ArgMaxMultiCoreProgramFactory` — `reader_argmax_interleaved_multicore.cpp`, one instance per
  node (see the note below on the two same-source `KernelDescriptor`s):

  | CB | Config | Touchers on a node | Verdict | Disposition |
  |---|---|---|---|---|
  | `c_0` src | both (cores0 spec; cores1 spec when `num_cores1 > 0`) | 1 — raw `src_cb.get_write_ptr()` @ `:318` + `noc.async_read` destination @ `:58` | single-ended / sync-free | **self-loop** |
  | `c_1` dst | both | 1 — raw `dst_cb.get_write_ptr()` @ `:322` + `noc.async_write(use<WRITE_PTR>(dst_cb), …)` @ `:450,498`. *Functionally* only the reducer core fills it, but the binding is taken unconditionally on every node | single-ended / sync-free | **self-loop** |
  | `c_2` red_idxs | both | 1 — raw `red_idx_cb.get_write_ptr()` @ `:326`, written locally @ `:116` and drained @ `:416,467` | single-ended / sync-free | **self-loop** |
  | `c_3` red_vals | both | 1 — raw `red_val_cb.get_write_ptr()` @ `:333`, written locally @ `:117` and drained @ `:422,473` | single-ended / sync-free | **self-loop** |

  **The remote NoC writes do not add touchers.** In the multi-core kernel a worker writes its
  partials into the *reducer's* L1 (`{.noc_x = reduce_core_x, .noc_y = reduce_core_y,
  .addr = red_idx_cb_local_addr}`). That destination is a bare NoC address, not a DFB binding —
  the writing kernel binds only its **own** node's `c_2` / `c_3` (as the `use<WRITE_PTR>` source).
  So each node's census stays at 1, and there is no hidden second writer on any CB. Actively
  scanned for face (a): no kernel writes another kernel's CB via `get_write_ptr()` /
  `fifo_wr_ptr` on the same node, and the two semaphores gate core-to-core ordering, not an
  intra-node co-fill.

  **The two same-source `KernelDescriptor`s are the disjoint-node shape, not a dual-instance
  work-split.** `argmax_multi_core_program_factory.cpp:376` and `:408` push the same
  `kernel_source` twice, but over **`cores0`** and **`cores1`** respectively — disjoint core sets,
  so each node sees exactly one instance and every CB stays at 1 toucher. This is the
  demoting-per-group-CTA shape, not the both-instances-on-every-node split that produces
  two touchers.

  **No dead CB.** Every allocated `buffer_index` is referenced by its bound kernel in every config
  — verified by resolving each index through its CTA to a wrapper construction and at least one
  access. Note the near-miss: `c_1` in the multi-core factory is *documented* as
  "only used in the reduction core" (`reader_argmax_interleaved_multicore.cpp:239`), but
  `dst_cb.get_write_ptr()` @ `:322` runs unconditionally on every node, so it is live everywhere
  and **must not** be dropped or made conditional.

  **`ArgMaxNCProgramFactory` (gated — recorded, then deferred).** Its two CBs are textbook plain
  1:1 and need no action if that factory ever ports as-is: `c_0` — reader is a locked producer
  (`dfb.reserve_back`/`push_back` @ `reader_argmax_nc.cpp:57,60`), compute is a locked consumer
  (`dfb_val_obj.wait_front`/`pop_front` @ `argmax_nc_compute.cpp:82,84,92,94`); `c_16` — compute
  is a locked producer (`:112,116`), writer a locked consumer (`writer_argmax_nc.cpp:97,100`).
  This census is **provisional**: the `ProgramDescriptor` migration will rewrite that factory, so
  re-derive it at re-audit rather than trusting these rows.

- **Offset base pointers:** **GREEN — no address argument folds a host-side offset into its base,
  in any factory.** Every address site resolved to its host computation:

  | Site | Host expression | Kernel consumption | Verdict |
  |---|---|---|---|
  | `argmax_single_core_program_factory.cpp:205` | `emplace_runtime_args(core, {input, output})` — MeshTensor bindings, base only, **no arithmetic** | fed to `TensorAccessor(args, base)` @ `reader_argmax_interleaved.cpp:45,46` / `reader_argmax_tile_layout.cpp:52,53` / `reader_argmax_tile_layout_h.cpp:47,48` | clean base |
  | `argmax_multi_core_program_factory.cpp:394,424` | `{input, output, …}` — MeshTensor bindings, base only | `TensorAccessor(args, base)` @ `reader_argmax_interleaved_multicore.cpp:306,307` | clean base |
  | `argmax_nc_program_factory.cpp:199` | bare `input.buffer()->address()`, **no arithmetic** | `TensorAccessor(args, addr)` @ `reader_argmax_nc.cpp:46` | clean base |
  | `argmax_nc_program_factory.cpp:207` | bare `output.buffer()->address()`, **no arithmetic** | `TensorAccessor(args, addr)` @ `writer_argmax_nc.cpp:89` | clean base |

  **The multi-core factory is the case worth stating explicitly, because it looks like a Type 1
  and is not.** It *does* compute a per-core byte offset — `i * src_read_size0` and
  `src_offset1 + (i * src_read_size1)` (`argmax_multi_core_program_factory.cpp:399,429`) — but
  passes it as a **separate RTA at index 3**, never folded into the base. The kernel consumes it
  as the accessor's per-read `{.offset_bytes = src_offset}` argument
  (`reader_argmax_interleaved_multicore.cpp:62`) on top of a clean-base accessor. That is exactly
  the already-split-out shape the recipe calls GREEN, and it drops straight into ordinary
  tensor-binding port work. Likewise `red_dim_offset` (RTA 4) is an element index, not an address.

  Type 3 (`address_offset`) is absent — see the Appendix A row. Type 4 (`ttnn::narrow` /
  interior-base `MeshBuffer::create`) is absent. `reduction/argmax` appears in **neither** table
  of `2026-07-19_offset_base_pointers.md`, which agrees with this scan (the "no fold, not in the
  tables → clean" outcome).

- **TensorAccessor 3rd argument:** **N/A — no accessor in the op passes a 3rd argument.** All ten
  construction sites are the 2-arg `TensorAccessor(args, base_addr)` form
  (`reader_argmax_interleaved.cpp:45,46`; `reader_argmax_interleaved_multicore.cpp:306,307`;
  `reader_argmax_tile_layout.cpp:52,53`; `reader_argmax_tile_layout_h.cpp:47,48`;
  `reader_argmax_nc.cpp:46`; `writer_argmax_nc.cpp:89`). The subject never fires, so there is
  nothing to classify — this is *no sites*, not *sites found and judged redundant*. Consistent
  with `reduction/argmax` being absent from `2026-07-06_tensor_accessor_3rd_arg_triage.md`.

## Port-work summary  *(mirrors the brief)*

Scoped to the clean subset — `ArgMaxDeviceOperation`'s two factories.

- **Tensor bindings** (per binding, both factories, all three single-core configs):
  - `input` — **Case 1** (via `TensorAccessor`).
  - `output` — **Case 1** (via `TensorAccessor`).

  Delivery today is the `Buffer*`-binding form in its MeshTensor spelling —
  `emplace_runtime_args(core, {input, output})` (`argmax_single_core_program_factory.cpp:205`;
  `argmax_multi_core_program_factory.cpp:394,424`) — which the framework auto-registers as
  `BufferBinding`s and patches on cache hits. So this is **routine port work, not a correctness
  hazard**: it is already correct-on-cache-hit today, and the Metal 2.0 typed binding supersedes
  the mechanism. The kernel side then loses both `TensorAccessorArgs(...).append_to(...)` calls
  (`argmax_single_core_program_factory.cpp:182,183`;
  `argmax_multi_core_program_factory.cpp:373,374`) and the paired
  `TensorAccessorArgs<N>()` / `next_compile_time_args_offset()` CTA plumbing.

- **TensorParameter relaxation:** **none** (sheet `none` on both clean rows).
- **TensorAccessor 3rd arg:** none — no site exists.
- **CB endpoints:** **self-loop** on all six: `(c_0, RM)`, `(c_1, RM)`, `(c_0, TILE-W)`,
  `(c_1, TILE-W)`, `(c_0, TILE-H)`, `(c_1, TILE-H)` for the single-core factory; and
  `(c_0, multicore)`, `(c_1, multicore)`, `(c_2, multicore)`, `(c_3, multicore)`. No 1P+1C
  assignment, no multi-binding flag, no dead-CB drop, no conditional DFB.
- **`get_dataformat(<cb_idx>)` → DFB member:** move the five sites listed in the Device 2.0
  block onto the bound object (`DataflowBuffer::get_dataformat()` is `constexpr`, so the NTTP
  uses survive). Port work by the recipe's own breadcrumb, not a Device 2.0 fix.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** **none.** No CB in the clean subset reaches
  two touchers on a node, so nothing needs the multi-binding advanced option. The two
  same-source `KernelDescriptor`s in the multi-core factory are the *disjoint-node* shape
  (`cores0` / `cores1`), not the dual-instance work-split — do not read them as two touchers.
- **Two DFB specs sharing one buffer index over disjoint core ranges.** The multi-core factory
  declares `c_0` **twice** — `argmax_multi_core_program_factory.cpp:218` over `cores0` with
  `total_size = round_up_to_mul32(red_dim_units0 * input_unit_size)`, and `:231` over `cores1`
  with the `red_dim_units1` size — and on the default (no `sub_core_grids`) path those two sizes
  genuinely **differ**, because `split_work_to_cores` hands the two groups different per-core
  block counts. The port must reproduce this as two `DataflowBufferSpec`s at the same index over
  disjoint core ranges, keeping the second one conditional on `num_cores1 > 0`.
- **The multi-core kernel requires `c_2` / `c_3` to land at the *same* L1 address on every
  node.** A worker computes the reducer's destination address from its **own** base pointer —
  `red_idx_cb_local_addr = red_idx_cb.get_write_ptr() + core_id * red_idx_size_per_core`, then
  sends to `{.noc_x = reduce_core_x, .noc_y = reduce_core_y, .addr = red_idx_cb_local_addr}`
  (`reader_argmax_interleaved_multicore.cpp:326-339, 416-427, 467-478`). Legacy honours that
  assumption: `ProgramImpl::allocate_circular_buffers` assigns **one** address per CB object,
  taken as the max region-end across all the core ranges it spans
  (`tt_metal/impl/program/program.cpp:1719-1751`), so a CB over `all_cores` is uniform even
  though the `c_0` specs differ in size between the groups. Confirm the Metal 2.0 DFB allocator
  keeps the same property — if a DFB were placed per-core-range instead, the cross-core writes
  would silently land at the wrong offset. Numerically wrong, with nothing to flag it.
- **Cross-op / shared kernels:** **no borrowed kernel files** — the op owns all seven, and no
  other *op* instantiates any of them. But there is one **reverse** coupling the port must not
  trip over: `tests/ttnn/unit_tests/gtests/test_generic_op.cpp:126` file-path-instantiates
  `device/kernels/reader_argmax_interleaved.cpp` from a **hand-built `ProgramDescriptor`**, with
  the CTA layout (indices 0-7 plus two `TensorAccessorArgs` blocks) and the RTA pair
  (`src_buffer->address()`, `dst_buffer->address()`) hardcoded in the test at `:105-121`. A
  Metal 2.0 rewrite of that kernel changes exactly that contract, and the `generic_op` /
  `ProgramDescriptor` entry point cannot supply Metal 2.0 named bindings — so the test breaks.
  No `_metal2` fork exists beside any argmax kernel today. Resolve deliberately (fork the kernel,
  or update the test) — see *Questions*.
- **RTA varargs:** **none.** Every kernel reads a fixed set of args at constant indices — 2
  (single-core readers), 7 (multi-core reader), 7 / 3 (NC reader / writer) — with no counted
  loop, no `arg_index++` inside a loop, and no data-selected index. Ordinary named-arg port work.
  **No CTA varargs either:** every `get_compile_time_arg_val` call in the directory uses a literal
  constant index; there is no variable-index compile-time-arg read anywhere.
- **`experimental/quasar/reduction/` exists — stay out of it.** There is a quasar copy of the
  reduction family. It is a deliberately hacky shortcut port, not a precedent, not a naming
  source, and not a fork to reuse. Do not read it and do not let it into the port diff.
- **Two kernels are already partly on the Metal 2.0-era object.**
  `reader_argmax_tile_layout.cpp` and `reader_argmax_tile_layout_h.cpp` already hold
  `DataflowBuffer` (not `CircularBuffer`) at `:58,59` and `:51,54`, while
  `reader_argmax_interleaved.cpp` and `reader_argmax_interleaved_multicore.cpp` are still on
  `CircularBuffer` + the `use<CircularBuffer::AddrSelector::WRITE_PTR>` view. So the port's
  kernel work is asymmetric: for the two TILE readers it is a binding-layer change
  (name the DFB, drop the CTA index) rather than an object swap; for the two RM readers it is
  the full CB→DFB swap, including replacing the `use<…>` views and dropping
  `#include "api/dataflow/circular_buffer.h"`.
- **`constexpr`-vs-`const` decides token form at every DFB site.** All four kernels take their
  buffer index from a `constexpr uint32_t` CTA and then build a *non*-`constexpr` wrapper object
  (e.g. `constexpr uint32_t src_dfb_idx = get_compile_time_arg_val(0);` then
  `DataflowBuffer src_dfb(src_dfb_idx);`). The `get_dataformat` results, by contrast, must stay
  compile-time constants (they are NTTPs). Decide token-form vs member-getter per site rather
  than uniformly.

## Team-only

- **Out-of-directory coupling & donor shape:** **roll-up ✓ clean.** No per-call detail table is
  owed — there are no ⚠ / ✗ / ⭐ entries.

  | Op kernel | Donor file | Class | Status |
  |---|---|---|---|
  | all 7 | `tt_metal/hw/inc/api/**` (`dataflow_api.h`, `noc.h`, `circular_buffer.h`, `dataflow_buffer.h`, `noc_semaphore.h`, `endpoints.h`, `core_local_mem.h`, `tensor/tensor_accessor.h`, `tensor/noc_traits.h`, `numeric/*`, `debug/*`, `compute/*`) | 1 — `tt_metal/*` LLK/HAL | ✓ no concern |
  | `argmax_tile_layout.hpp` | `tt-metalium/constants.hpp` | 1 — `tt_metal/*` | ✓ no concern |
  | 4 readers | `argmax_common.hpp`, `argmax_tile_layout.hpp`, `argmax_tile_h_col.hpp` | in-directory (not an escape) | ✓ |

  No include resolves into `ttnn/cpp/ttnn/kernel_lib/`, `ttnn/cpp/ttnn/kernel/`,
  `ttnn/cpp/ttnn/operations/kernel_helper_functions/`, another op family, or another op in
  `reduction/`. **Borrowed kernel files: none** — every `kernel_source` string in all three
  factories points inside `reduction/argmax/device/kernels/`. Host-side, the two `descriptor`
  factories include the in-family `ttnn/operations/reduction/reduce_op_validation.hpp` for
  `validate_reduce_op_program_grid` / `validate_reduce_op_tensor` — host validation, outside the
  kernel-coupling model and unaffected by the port.

  **Reverse (consumer-side) coupling — the recipe has no slot for this, so it is recorded here
  as well as in the brief:** `tests/ttnn/unit_tests/gtests/test_generic_op.cpp:126` is an
  out-of-directory consumer that file-path-instantiates `reader_argmax_interleaved.cpp` against a
  hardcoded CTA/RTA contract. Not a donor problem and not a gate, but it is coupling the port
  will break, and it is invisible to a borrowed-kernel inventory that only looks outward.

- **Relaxation candidates:** none to mine — the op declares no custom `compute_program_hash` and
  no backdoor `attribute_values` / `to_hash`, so there is no hash to read dependencies out of.
  `TensorParameter relaxation` is `none` on all three rows.

- **TTNN factory analysis:** op-owned tensors — none (no `WorkloadDescriptor`, no `buffers`
  vector). MeshWorkload need — none; `Execution Model` is `SPMD` on all three rows and neither
  device-op builds a workload. Pybind `create_descriptor` — absent; `argmax_nanobind.cpp` binds
  only the user-facing `ttnn::argmax` (no `nb::class_` of a device op, so **no user-visible API
  deletion** falls out of this port). Other risky pybind — none. Custom hash — absent, so the
  default attribute hash applies (see *Misc anomalies* for one forced attribute that still feeds
  it). `get_dynamic_runtime_args` — absent on both device-ops. `override_runtime_arguments` —
  only on `ArgMaxNCProgramFactory` (`argmax_nc_program_factory.cpp:218`), as the legacy-concept
  signature, so the clean subset targets the **base** `ProgramSpecFactoryConcept`, not the custom
  one. Target concept for the clean subset: **`ProgramSpecFactoryConcept`**, matching the sheet's
  `Porting Target` cell.

## Misc anomalies  *(team-only, non-gating; route to the ops team — the port does not act on these)*

1. **Dead CTA — `src_page_size` in the multi-core reader.** The factory passes `src_page_size`
   at compile-time index 4 (`argmax_multi_core_program_factory.cpp:346`), but the kernel reads
   indices 0,1,2,3 then jumps to 5 (`reader_argmax_interleaved_multicore.cpp:249`) — index 4 is
   never read. The kernel sizes its reads from the `src_read_size` RTA instead. Harmless; it does
   cost a CTA slot and misleads anyone counting offsets by hand.
2. **Dead CTA — `dst_page_size` in both TILE readers.** `get_ctime_args_single_core` emits
   `dst_page_size` at index 3 for the TILE branch too
   (`argmax_single_core_program_factory.cpp:91`), but `reader_argmax_tile_layout.cpp` and
   `reader_argmax_tile_layout_h.cpp` both skip index 3 (they read 0,1,2 then 4…) and derive the
   write size from `output_page_elements * sizeof(uint32_t)` in `write_to_output`
   (`argmax_tile_layout.hpp:344`). Live only on the RM branch, where
   `reader_argmax_interleaved.cpp:23` does read it. Sharing one CTA builder across two kernel
   families is what leaves the hole.
3. **A core index narrowed through `bool` — latent, currently inert.**
   `reader_argmax_interleaved_multicore.cpp:273` reads
   `constexpr uint32_t reduce_core_id = (bool)get_compile_time_arg_val(13);`. CTA 13 is the
   reducer's **core index**, not a flag. It is 0 today, so the cast is a no-op — but the factory
   explicitly anticipates changing it: *"We can do perf optimization by tuning this in the
   future"* (`argmax_multi_core_program_factory.cpp:284`). The moment anyone sets it to a
   non-zero index the `bool` cast collapses it to 1, and both the `is_reduce_core` test
   (`:304`) and the worker-skip test (`:415`) silently pick the wrong core. Cheap to fix now,
   expensive to debug later.
4. **Dummy core-range CTAs for the single-group case.** When `all_cores.size() == 1`,
   `argmax_multi_core_program_factory.cpp:290` substitutes
   `CoreRange(CoreCoord(0,0), CoreCoord(0,0))` for the absent second group, so CTAs 20-23
   (`start_core_[xy]1` / `end_core_[xy]1`) describe core (0,0) rather than "no group". Inert
   because every use is guarded by `num_cores1 > 0` (`reader_argmax_interleaved_multicore.cpp:378,
   522`), but it hands the kernel plausible-looking coordinates for a group that does not exist.
5. **A forced attribute that still participates in the hash.** `ArgmaxParams::output_dtype` is
   validated to be exactly `UINT32` (`argmax_device_operation.cpp:125`) and the facade only ever
   passes `DataType::UINT32` (`argmax.cpp`, both `prim::argmax` call sites). It is therefore a
   constant that still rides the default attribute hash and the pybind surface. Not a
   portability question — the port leaves hashing alone — but it is dead configurability.
6. **`ArgMaxNCProgramFactory` reads `input.device()` rather than the output's.**
   `argmax_nc_program_factory.cpp:49` takes the device from the input tensor while the two
   `descriptor` factories use `&output.mutable_device()`
   (`argmax_single_core_program_factory.cpp:124`, `argmax_multi_core_program_factory.cpp:184`).
   Equivalent in practice; noted only because the PD migration of that factory will have to pick
   one, and the `descriptor` siblings set the house style.

## Per-DeviceOperation attribution

| Field | `ArgMaxDeviceOperation` | `ArgMaxNCDeviceOperation` |
|---|---|---|
| **Factories** | `ArgMaxSingleCoreProgramFactory`, `ArgMaxMultiCoreProgramFactory` | `ArgMaxNCProgramFactory` |
| **Overall** | **GREEN** — every gate cleared | **RED** |
| *Prereqs* — Device 2.0 | Yes (4 kernels) | **Yes** (3 kernels) — not the blocker |
| *Prereqs* — Cross-op escapes | Ok — no borrowed kernels; one reverse coupling (a gtest consumer of `reader_argmax_interleaved.cpp`) | Ok — no borrowed kernels, no external consumer |
| *Feature Support* | GREEN (all Appendix A `N/A`) | GREEN (all Appendix A `N/A`) |
| *TTNN Readiness* — `Is able to port?` | `yes` (both rows) | **`no`** — `Concept` == `legacy device-op` → **TTNN / PD-migration team** |
| *TTNN Readiness* — Concept (current) | `descriptor` | `legacy device-op` |
| *TTNN Readiness* — Custom hash / `get_dynamic_runtime_args` / Pybind descriptor / Op-owned tensors | No / No / No / No | No / No / No / No |
| *TTNN Readiness* — `override_runtime_arguments` | No | Yes @ `argmax_nc_program_factory.cpp:218` — the *legacy* signature, so it does not select the custom concept |
| *TTNN Readiness* — relaxation | `none` | `none` |
| *TTNN Readiness* — Target concept | `ProgramSpecFactoryConcept` | `ProgramSpecFactoryConcept` (per the sheet's `Porting Target`), reachable only after the PD migration |
| *Port work* — Offset base pointer | none | none |
| *Port work* — Tensor bindings | `input` Case 1, `output` Case 1 | *(informational subject skipped — see below)* |
| *Port work* — TensorAccessor 3rd arg | none (no site) | none (no site) |
| *Port work* — CB endpoints | self-loop ×6 | *(informational subject skipped; a provisional 1:1 census is noted in Gate detail for context only)* |
| **Brief issued?** | **Yes** — `METAL2_PORT_BRIEF.md` covers this DeviceOperation only | No |

**Skipped informational subjects for `ArgMaxNCDeviceOperation`** — recorded so the omission is
never mistaken for a clean result. The blocker clears on the **op-code side** (a
`ProgramDescriptor` migration rewrites the very factory these subjects would describe), so the
**Red** outcome scoping rule says defer them to the re-audit:

- **TTNN porting shape** — `skipped — whole-DeviceOperation RED, no portable subset; re-audit on unblock`
- **TensorParameter relaxations** — *(read anyway from the sheet: `none`)*; candidate mining `skipped — no custom hash to mine`
- **TensorParameter analysis** — `skipped — whole-DeviceOperation RED, no portable subset; re-audit on unblock`
- **CB endpoints** — `skipped — whole-DeviceOperation RED, no portable subset; re-audit on unblock` (a provisional census appears in Gate detail as context, not as a finding)
- **RTA varargs** — `skipped as a finding — whole-DeviceOperation RED`; scanned incidentally while checking the gates and clean (fixed indices only)
- **Out-of-directory coupling** — `skipped as a full inventory — whole-DeviceOperation RED`; the Device 2.0 gate confirmed no donor kernels exist, so nothing gate-relevant is deferred
- **Incidental anomalies** — opportunistic only; item 6 above was noticed while working the gates

The four **gate-bearing** subjects were run in full for this DeviceOperation, and all four are
clear — so the PD migration is the *only* thing standing between it and a GREEN re-audit.

## Questions for the user

1. **How should the port handle the gtest that consumes `reader_argmax_interleaved.cpp`?**
   `tests/ttnn/unit_tests/gtests/test_generic_op.cpp:126` builds its own `ProgramDescriptor`
   around that kernel with the CTA/RTA contract hardcoded at `:105-121`. A Metal 2.0 rewrite
   changes that contract, and the `generic_op` / `ProgramDescriptor` entry point cannot supply
   Metal 2.0 named bindings, so the test cannot simply be re-pointed. Two paths: **(a)** create
   `reader_argmax_interleaved_metal2.cpp` beside the original per the shared-kernel fork
   convention and leave the test on the legacy copy (the copy is then sunset when the test is
   retired or reworked), or **(b)** rewrite the kernel in place and migrate the test off
   `generic_op`. (a) keeps the port's blast radius inside the op; (b) avoids a duplicated kernel
   but pulls a test rewrite into the port diff. Which do you want?
2. **Is `ArgMaxNCDeviceOperation`'s `ProgramDescriptor` migration scheduled?** It is the sole
   blocker, and every other gate is already clear for it — so it should re-audit to GREEN with
   nothing further to fix. Worth confirming with the PD-migration team whether it is queued, so
   the two halves of `ttnn::argmax` can converge rather than one shipping ported and the other
   waiting indefinitely.
3. **Should the clean subset be ported now, or held until the whole op can go together?**
   `ArgMaxDeviceOperation` (both factories) is portable today. Porting it alone leaves
   `ttnn::argmax` split across a Metal 2.0 device-op and a legacy one, which is functionally fine
   but means two rounds of review on one user-facing op.

## Recipe notes

1. **`get_dataformat(cb_id)` should join the sanctioned free-function list — or the list should
   say what to do with a near-miss.** The Device 2.0 Green bullet sanctions exactly
   `get_tile_size(cb_id)` and `get_local_cb_interface(cb_id)`, and insists *"the list is the whole
   test."* `get_dataformat(cb_id)` is the same *kind* of thing — a compile-time CB metadata query,
   not a data-movement operation — appears in five places in this op, and its
   `CircularBuffer::get_dataformat()` counterpart **cannot substitute** at any of them, because
   the member is not `constexpr` (`circular_buffer.h:113`) while the free function is
   (`dataflow_api.h:300`) and every site feeds an NTTP. I read the Red bullet's own conjunct
   (*"**and** a wrapper-method replacement exists"*) as unmet and scored it GREEN, but that is a
   judgment the auditor should not have to make on a hard prereq gate. Please either add
   `get_dataformat` (and plausibly `get_tile_hw`) to the sanctioned list, or add a sentence saying
   that a free function whose only wrapper counterpart is non-`constexpr` is not a holdover at a
   `constexpr` site. Note the asymmetry that makes this specifically a *port*-stage fix:
   `DataflowBuffer::get_dataformat()` **is** `constexpr` (`dataflow_buffer.h:279`), so the Metal
   2.0 port can move these lookups onto the object where Device 2.0 could not — which is exactly
   what the Green bullet's breadcrumb already implies for the two sanctioned names.

2. **Two *independent* DeviceOperations in one op directory have nowhere to go.** The Scope rule
   says to audit device-operations together when they share factories or kernels and *"audit each
   separately"* when they are independent. `ArgMaxDeviceOperation` and `ArgMaxNCDeviceOperation`
   share neither — but `METAL2_PREPORT_AUDIT.md` is **op-directory-scoped**, so "separately"
   has no filename to land in. I bundled them and leaned on *Per-DeviceOperation attribution*,
   which worked well, but the recipe should say so explicitly: either name a filename convention
   for the independent case (`METAL2_PREPORT_AUDIT_<DeviceOperation>.md`?) or state that
   co-directory device-ops are always bundled with per-DOp attribution regardless of the
   shared-code test. (Here the shared *facade* — one `ttnn::argmax` dispatching to both — is
   arguably the stronger reason to bundle than the shared-code test the rule actually asks about.)

3. **"Clean factory subset" is written as if the subset is always within one DeviceOperation.**
   The Code-path scope rule and the `RED at op level; subset <X> is clear` formula read
   factory-scoped (*"a single factory's `if (use_width_sharding)` branch"*). Here the clean subset
   is a **whole DeviceOperation** and the gated remainder is a *different* DeviceOperation. The
   rule generalizes fine and I applied it that way — brief issued, scoped to the clean
   DeviceOperation — but one clause acknowledging the DeviceOperation-granular subset would save
   the next auditor the inference, especially since the brief-vs-no-brief decision hangs on it.

4. **Out-of-directory coupling has no slot for a *consumer* of the op's own kernel.** The subject
   inventories what the op borrows — outward escapes and borrowed kernel files. It has no place
   for the mirror case found here: an out-of-directory consumer that file-path-instantiates one of
   *this* op's kernels against a hardcoded CTA/RTA contract
   (`tests/ttnn/unit_tests/gtests/test_generic_op.cpp:126`). That is real, port-breaking coupling
   with exactly the same shape as the shared-kernel problem the `_metal2` fork convention exists
   to solve, and it is invisible to an inventory that only looks outward. It also cannot be
   re-pointed the usual way, since `generic_op`/`ProgramDescriptor` cannot feed a Metal 2.0
   `KernelSpec`. Suggest adding a "who else instantiates *my* kernels" grep to the borrowed-kernel
   step — it is one `grep` for the kernel paths — and saying whether a **test** consumer justifies
   a fork or a test rewrite.

5. **Minor: the *Red* scoping rule's clear-side question is easy to answer per-subject but is
   asked per-op.** For `ArgMaxNCDeviceOperation` the blocker clears on the op-code side (skip the
   seven), while for the bundled report the clean subset needs all seven *for the other
   DeviceOperation*. So the same audit both skips and runs the informational subjects, which the
   one-line disclosure format does not quite anticipate. I disclosed per-subject inside the
   *Per-DeviceOperation attribution* section; a sentence blessing that placement would help.
