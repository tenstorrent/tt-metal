# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/move`

- **`MoveDeviceOperation`**
  - `MoveProgramFactory` (`move_program_factory.cpp`) — the `MULTI_CORE` strategy. **Owns no kernels of its own**: `create_descriptor` delegates verbatim to `CopyDeviceOperation::SameMemoryConfig::create_descriptor`, so the kernels this factory instantiates are copy's.
  - `MoveOverlapProgramFactory` (`move_overlap_program_factory.cpp`) — the `MULTI_CORE_OVERLAP` strategy.
  - `MoveShardedProgramFactory` (`move_sharded_program_factory.cpp`) — the `MULTI_CORE_SHARDED` strategy.

**Kernels referenced (audited across directory boundaries):**

| Factory / config | Kernel | Owner |
|---|---|---|
| Overlap (tilized) | `move/device/kernels/dataflow/move_interleaved_with_overlap.cpp` | move (own) |
| Overlap (row-major) | `move/device/kernels/dataflow/move_stick_layout_interleaved_with_overlap.cpp` | move (own) |
| Sharded | `move/device/kernels/dataflow/reader_unary_local_l1_copy_backwards.cpp` | move (own) |
| MULTI_CORE (tilized) | `copy/device/kernels/reader_unary_start_id.cpp`, `copy/device/kernels/writer_unary_start_id.cpp` | copy (donor) |
| MULTI_CORE (row-major interleaved) | `ttnn/cpp/ttnn/kernel/dataflow/reader_unary_stick_layout_interleaved_start_id.cpp`, `.../writer_unary_stick_layout_interleaved_start_id.cpp` | shared kernel pool (donor) |

Move validates input dtype == output dtype, and routes sharded inputs to the sharded factory, so on the `MULTI_CORE` path copy's `SameMemoryConfig` always runs with `convert_dtype == false` and `sharded == false`. Consequently copy's dtype-convert compute kernel (`eltwise_copy.cpp`) and copy's sharded row-major kernels (`reader_unary_stick_start_id.cpp` / `writer_unary_stick_start_id.cpp`) are **not reached by move** and are out of scope here.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `29e0fc7e341 2026-07-23 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/move` |
| **Overall** | **RED** |
| **DOps / Factories** | `MoveDeviceOperation` → `MoveProgramFactory`, `MoveOverlapProgramFactory`, `MoveShardedProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **No (RED)** — donor `reader_unary_stick_layout_interleaved_start_id.cpp` is broadly Device 1.0; routed to the Device 2.0 track. Affects `MoveProgramFactory` row-major path only. |
| *Prereqs* — Cross-op escapes | Issue — `MoveProgramFactory` delegates its whole descriptor to copy's `SameMemoryConfig`; row-major path instantiates a shared-pool kernel (port-together coupling) |
| *Feature Support* — overall | GREEN (no Appendix A feature in use) |
| *Feature Support* — Variadic-CTA | Ok |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Mixed** — `MoveProgramFactory` yes · `MoveOverlapProgramFactory` yes · **`MoveShardedProgramFactory` no** |
| *TTNN Readiness* — Concept (current) | `descriptor` (all three factories) |
| *TTNN Readiness* — Secretly SPMD | N/A (no `WorkloadDescriptor`) |
| *TTNN Readiness* — Is safe to port? | `MoveShardedProgramFactory` **No** (→ readiness-sheet owner); others Yes |
| *TTNN Readiness* — Custom hash | No (confirmed: no `compute_program_hash` in the op) |
| *TTNN Readiness* — Runtime-args update | **Yes on `MoveShardedProgramFactory`** (gate). See cross-check note: a DeviceOperation-level `override_runtime_arguments` actually covers all three strategies. |
| *TTNN Readiness* — Pybind `create_descriptor` | No (confirmed: no descriptor binding in `move_nanobind.cpp`) |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` (for factories that clear) |
| *Port work* — Offset base pointer | none — every address binding is a clean base |
| *Port work* — Tensor bindings (per binding) | Case 1 (via `TensorAccessor`) on interleaved factories; clean (borrowed-memory DFB) on the sharded factory |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none — no accessor passes a 3rd argument |
| *Port work* — CB endpoints | self-loop (single-toucher CBs) / plain 1:1 (copy reader+writer) — all resolvable, none gate |

**CB endpoints** are dispositions, not gates: every out-of-window CB has a port-time resolution. Recorded per `(CB, config)` below.

## Result

**RED at op level.** Two independent blockers, in different factories, routed to different teams:

1. **Device 2.0 (prerequisite) — `MoveProgramFactory` row-major path.** The `MULTI_CORE` strategy delegates its entire descriptor to `CopyDeviceOperation::SameMemoryConfig`, which for a **row-major interleaved** tensor instantiates the shared-pool donor kernel `ttnn/cpp/ttnn/kernel/dataflow/reader_unary_stick_layout_interleaved_start_id.cpp`. That kernel is broadly Device 1.0 (raw `noc_async_read` / `noc_async_read_barrier`, and CB-index free functions `cb_reserve_back` / `get_write_ptr(cb_id)` / `cb_push_back` with no wrapper object in scope). Device 2.0 migration of this donor is a hard prerequisite. → **Device 2.0 migration team.**

2. **TTNN factory concept (prerequisite) — `MoveShardedProgramFactory`.** The readiness sheet's `Is able to port? = no` for this factory, on two failing conjuncts: `Is safe to port? = no` (→ **readiness-sheet owner**) and `Runtime-args update = yes` (→ **TTNN / ProgramDescriptor-migration team**; the gate lifts when the Metal 2.0 runtime-args-update infrastructure ships). The root cause is visible in the code: the sharded reader's `move_chunk_size_bytes` runtime arg is derived from the *difference of the input and output buffer base addresses* (`move_sharded_program_factory.cpp:52`), a placement-dependent scalar that must be recomputed every dispatch — which is exactly why the op carries a custom `override_runtime_arguments` and why the migration is flagged not-safe.

**Both blockers are external-team prerequisites with clear paths forward; neither is a permanent wall.** Once the donor kernel's Device 2.0 migration lands and the sharded factory's runtime-args-update infra + correctness reconciliation complete, the op should be re-audited (the code is small and stable, so a clean re-audit is cheap).

**Clean factory (for the record): `MoveOverlapProgramFactory` clears every gate.** Its two kernels are Device 2.0 compliant, its readiness-sheet verdict is `yes`, it uses no Appendix A feature, and it has no offset-base-pointer or 3rd-arg issue. `MoveProgramFactory`'s *tilized* path is likewise gate-clean, but the factory as a whole is blocked because its row-major sibling path (same runtime-selected `create_descriptor`) instantiates the Device-1.0 donor.

**A subset (overlap-only) port is *not* recommended, and no brief is issued.** The three factories share a single `MoveDeviceOperation` with one `override_runtime_arguments` method that switches on strategy and serves all three. Porting `MoveOverlapProgramFactory` to `MetalV2FactoryConcept` in isolation, while the sharded and multi-core factories remain on the `ProgramDescriptor` path (and keep needing that shared override hook), is a mixed-concept device-op with real coupling risk for marginal value (one of three runtime-selected strategies). The higher-value path is to clear both prerequisites and port all three factories together. This is a judgement call on the [Code-path scope](#result) rule — see *Recipe notes*.

## Gate detail

- **TTNN factory concept (`Is able to port?`):**
  - `MoveProgramFactory` → **GREEN** (sheet `yes`; cross-check clean: concept `descriptor`, no custom hash, no pybind `create_descriptor`).
  - `MoveOverlapProgramFactory` → **GREEN** (sheet `yes`; cross-check clean).
  - `MoveShardedProgramFactory` → **RED**. Sheet `Is able to port? = no`. Failing conjuncts: `Is safe to port? = no` → **readiness-sheet owner** (the correctness call is theirs; the placement-dependent `move_chunk_size_bytes` RTA is the likely trigger). `Runtime-args update = yes` → **TTNN / PD-migration team** (gate lifts when runtime-args-update infra ships). `Smuggled pointer` is `no` on the sheet, so the not-safe verdict is Diego's broader judgment, not a raw-pointer flag.
  - **Cross-check note (`Runtime-args update`) — surfaced, not treated as sheet-broken.** The sheet marks `Runtime-args update = yes` only for `MoveShardedProgramFactory` (and leaves the `PD override_runtime_args` column blank for all move rows). The code, however, defines a single `MoveDeviceOperation::override_runtime_arguments` (`move_sharded_program_factory.cpp:121`) that switches on strategy and therefore fires for **all three** factories. My read: the sheet's finer per-factory classification is *correct* — only the sharded factory genuinely depends on dynamic runtime args (the address-difference `move_chunk_size_bytes`); for the overlap and multi-core factories every scalar RTA is deterministic from the cache-miss inputs and the only per-dispatch-varying quantity (buffer addresses) rides auto-patched `BufferBinding`s, so their `override_runtime_arguments` is a redundant re-application. I therefore keep the two `yes` verdicts as cleared rather than flipping them to "sheet broken." Flagged as a Question for the readiness-sheet owner to confirm the intended per-factory semantics of the column.

- **Device 2.0 (every kernel used):** **RED** on one donor kernel; all other kernels the op uses are compliant.

  | File | Line | Call | Wrapper in scope |
  |---|---|---|---|
  | `ttnn/cpp/ttnn/kernel/dataflow/reader_unary_stick_layout_interleaved_start_id.cpp` | 26 | `cb_reserve_back(cb_id_in0, 1)` | none |
  | `ttnn/cpp/ttnn/kernel/dataflow/reader_unary_stick_layout_interleaved_start_id.cpp` | 27 | `get_write_ptr(cb_id_in0)` | none |
  | `ttnn/cpp/ttnn/kernel/dataflow/reader_unary_stick_layout_interleaved_start_id.cpp` | 29 | `noc_async_read(src_noc_addr, l1_write_addr, stick_size)` | none |
  | `ttnn/cpp/ttnn/kernel/dataflow/reader_unary_stick_layout_interleaved_start_id.cpp` | 30 | `noc_async_read_barrier()` | none |
  | `ttnn/cpp/ttnn/kernel/dataflow/reader_unary_stick_layout_interleaved_start_id.cpp` | 31 | `cb_push_back(cb_id_in0, 1)` | none |

  **Scope of incompleteness: broad Device 1.0, not isolated holdovers.** No `Noc` or `DataflowBuffer`/`CircularBuffer` wrapper object is ever constructed in this kernel; the CB is managed entirely through CB-index free functions and the NoC through raw `noc_async_*` calls. (Its addressing is via `TensorAccessor`, which is Device 2.0, but the CB + NoC surface needs a full migration.) The owning "family" is the **shared kernel pool `ttnn/cpp/ttnn/kernel/dataflow/`**; migrating it is a single shared-pool change that benefits every consumer (copy, and any other op instantiating it). The sibling writer `writer_unary_stick_layout_interleaved_start_id.cpp` is already Device 2.0 compliant (`Noc`, `CircularBuffer`, `TensorAccessor`), so only the reader is outstanding.

  All move-owned kernels and the tilized copy donors are compliant: `move_interleaved_with_overlap.cpp`, `move_stick_layout_interleaved_with_overlap.cpp`, `reader_unary_local_l1_copy_backwards.cpp`, `copy/.../reader_unary_start_id.cpp`, `copy/.../writer_unary_start_id.cpp` all use `Noc` / `DataflowBuffer` / `Semaphore<>` / `TensorAccessor` / `CoreLocalMem` / `UnicastEndpoint`. The free function `get_tile_size(cb_id)` (`move_interleaved_with_overlap.cpp:53`, `reader_unary_start_id.cpp:22`, `writer_unary_start_id.cpp:25`) is **sanctioned** and not flagged.

- **Feature compatibility:** every Appendix A entry scanned; none in use.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | Plain `CBDescriptor`s only; no `.global_circular_buffer` field, no `remote_index`/`remote_cb` idiom. The sharded factory's `.buffer = src_buffer/dst_buffer` is the ordinary borrowed-memory pattern (a mechanical port-recipe translation), not a GCB. |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `.address_offset` set anywhere; no `set_address_offset` / 4-arg `UpdateDynamicCircularBufferAddress` / `cb_descriptor_from_sharded_tensor`. |
  | GlobalSemaphore | N/A | Overlap uses a plain `SemaphoreDescriptor` + kernel-side `Semaphore<>`; no `GlobalSemaphore` type or `CreateGlobalSemaphore`. |
  | Variable-count compile-time arguments (CTA varargs) | N/A | `tensor_args_t` is a fixed pair (`input_tensor`, `output_tensor`); no kernel reads `get_compile_time_arg_val` at a runtime-varying index. |

- **Offset base pointers:** **GREEN.** No address RTA folds a host-side offset into its base. The interleaved factories deliver clean bases via `Buffer*` bindings (overlap: `move_overlap_program_factory.cpp:169-170`; copy: `copy_same_memory_config_program_factory.cpp:179,183,198,204`), consumed through `TensorAccessor`. The sharded factory passes no buffer address at all — the kernel reads tensor memory through borrowed-memory DFB pointers. The `move_chunk_size_bytes = output_buffer_address - input_buffer_address` scalar (`move_sharded_program_factory.cpp:52`) is a *byte count* (a difference of two addresses), not a base pointer handed to a kernel or accessor, so it is **not** a Type 1/2 offset-base-pointer fold; it is the dynamic-runtime-args driver captured under the TTNN concept gate above. (No entry in the offset-base-pointer triage doc for `move`; scan is clean regardless.)

- **TensorAccessor 3rd argument:** **GREEN / N/A.** No `TensorAccessor` in any referenced kernel passes a 3rd (page-size) argument. All construct the 2-arg form `TensorAccessor(args, addr)` (`move_interleaved_with_overlap.cpp:55-56`, `move_stick_layout_interleaved_with_overlap.cpp:49-50`, `reader_unary_start_id.cpp:26`, `writer_unary_start_id.cpp:29`, `reader_unary_stick_layout_interleaved_start_id.cpp:17`, `writer_unary_stick_layout_interleaved_start_id.cpp:20`).

- **CB endpoints (GATE-free):** every CB resolves at port time; nothing here blocks. See Port-work summary.

## Port-work summary  *(informational — see Recipe notes on why this is included despite a RED)*

Per-binding / per-CB detail for the factories, so a re-audit or an eventual all-clear port has it in hand.

- **Tensor bindings** (per binding):
  - **Overlap factory** — `src` (RTA slot 0, `Buffer*`) and `dst` (RTA slot 1, `Buffer*`): both **Case 1** (fed into `TensorAccessor(src_args, src_addr)` / `(dst_args, dst_addr)`; express as `TensorParameter`/`TensorBinding`, kernel builds `TensorAccessor(tensor::name)`, the `Buffer*` RTA + its `TensorAccessorArgs` CTAs disappear).
  - **MULTI_CORE factory (via copy)** — reader `src` and writer `dst` (`Buffer*` RTAs): both **Case 1** (via `TensorAccessor`). *Note:* this factory's port is a port of copy's `SameMemoryConfig` (shared with the copy op).
  - **Sharded factory** — `src` (CB `c_0`) and `dst` (CB `c_1`): both **clean** (borrowed-memory DFB reads via `.buffer = src_buffer/dst_buffer` + kernel `get_read_ptr()`/`get_write_ptr()`; the causal-link gate applies — port via `DataflowBufferSpec::borrowed_from`, not Case 1/2).
- **TensorParameter relaxation:** none (sheet `none` on every factory; no custom hash to reconcile).
- **TensorAccessor 3rd arg:** none.
- **CB endpoints** (per `(CB, config)`):
  - Overlap `cb_index 0` (tilized *and* row-major configs): **single toucher** (one reader kernel over `all_cores` doing both the fill and the drain) → **self-loop** (bind the one kernel PRODUCER and CONSUMER).
  - MULTI_CORE (copy) `c_0` (tilized *and* row-major-interleaved): **two touchers** — reader FIFO-produces, writer FIFO-consumes → **plain 1:1** (one locked producer + one locked consumer; no flag). No dtype-convert path for move, so no `c_16` and no third (compute) toucher.
  - Sharded `c_0` and `c_1`: each **single toucher** (the one reader kernel raw-reads `c_0` and raw-writes `c_1`) → **self-loop** each.
  - No dead CBs; no multi-binding.

## Heads-ups

- **Cross-op / shared kernels (port-together coupling):**
  - `MoveProgramFactory` **owns no kernels** — `create_descriptor` returns `CopyDeviceOperation::SameMemoryConfig::create_descriptor(...)` verbatim (`move_program_factory.cpp:25`). Porting move's `MULTI_CORE` path *is* porting copy's `SameMemoryConfig` factory, which the **copy** op also uses. These form a **port-together set**: `{move MULTI_CORE, copy SameMemoryConfig}` must adopt the Metal 2.0 rewrite together.
  - The row-major-interleaved donor kernels live in the shared pool `ttnn/cpp/ttnn/kernel/dataflow/` (`reader_unary_stick_layout_interleaved_start_id.cpp`, `writer_unary_stick_layout_interleaved_start_id.cpp`) and are broadly shared. Their Metal 2.0 rewrite is one shared change across every consumer.
- **Shared `override_runtime_arguments` coupling:** a single `MoveDeviceOperation::override_runtime_arguments` serves all three strategies. Any partial (single-factory) port must reckon with this shared hook remaining in place for the unported factories — a reason the audit recommends porting all three factories together rather than a subset.

## Team-only

### Out-of-directory coupling & donor shape

Two escape types are present.

**Descriptor-level delegation (host side):** `MoveProgramFactory::create_descriptor` calls `CopyDeviceOperation::SameMemoryConfig::create_descriptor` (include: `move_program_factory.cpp:7-8`). This is a whole-descriptor delegation, not a kernel-function escape — move's `MULTI_CORE` strategy has no descriptor logic of its own.

**Function-call escapes (kernel side):** none. The referenced kernels `#include` only `api/*` headers (LLK/HAL/firmware, donor class 1 — no concern). No kernel calls another op's helper function.

**Borrowed / file-path-instantiated kernels:**

| Kernel file | Owner / pool | Also used by | Shape |
|---|---|---|---|
| `copy/device/kernels/reader_unary_start_id.cpp` | copy (in-family) | copy | ✓ Device 2.0, `TensorAccessor` I/O |
| `copy/device/kernels/writer_unary_start_id.cpp` | copy (in-family) | copy | ✓ Device 2.0, `TensorAccessor` I/O |
| `ttnn/cpp/ttnn/kernel/dataflow/reader_unary_stick_layout_interleaved_start_id.cpp` | shared pool | copy + others (broadly shared) | ⭐ **Device 1.0** — donor-side Device 2.0 gate (see Gate detail) |
| `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp` | shared pool | copy + others (broadly shared) | ✓ Device 2.0 |

Roll-up: **⭐ blocked** — the one Device-1.0 shared-pool reader sequence-blocks move's `MULTI_CORE` row-major path (the gate judgment is in Gate detail → Device 2.0).

### TTNN factory analysis (sheet-derived facts + cross-check)

- **Concept:** `descriptor` on all three factories (confirmed: each factory's `create_descriptor` returns a `ProgramDescriptor`).
- **Custom hash:** No — confirmed no `compute_program_hash` anywhere in the op directory.
- **Runtime-args update:** sheet `yes` for `MoveShardedProgramFactory` only; code has a DeviceOperation-level `override_runtime_arguments` covering all three (cross-check note above).
- **Pybind `create_descriptor`:** No — `move_nanobind.cpp` binds only the user-facing `ttnn::move`; no device-op/descriptor class binding.
- **Op-owned tensors:** No (no `WorkloadDescriptor`; `descriptor` concept cannot carry them).
- **Is safe to port?:** `no` on `MoveShardedProgramFactory` (readiness-sheet owner's correctness call), `yes` on the other two.
- **Target concept** (for factories that clear): `MetalV2FactoryConcept`, no op-owned tensors.

## Misc anomalies

- **Placement-dependent RTA in the sharded factory.** `move_chunk_size_bytes = output_buffer_address - input_buffer_address` (`move_sharded_program_factory.cpp:52`) encodes the gap between the (freed) input allocation and the reallocated output as a runtime scalar, and the kernel copies backward in chunks of that size to safely shift overlapping in-place shard data. Correct, but it makes the program's runtime args a function of allocator placement — the reason the factory needs `override_runtime_arguments` and is flagged not-safe-to-port. Team-only context for the readiness-sheet owner; not a porter action.
- **Readiness sheet vs. copy code (out of move's scope, noted in passing).** The sheet marks copy's `DefaultRowMajor` and `DefaultTilized` factories `Custom hash = yes`, but `grep` finds no `compute_program_hash` anywhere in the copy op directory. This does not touch move (move uses only copy's `SameMemoryConfig`, which is `Custom hash = no` and code-confirmed), but it is worth the copy op's own auditor reconciling.

## Questions for the user

1. **`Runtime-args update` column semantics:** The sheet marks `Runtime-args update = yes` only for `MoveShardedProgramFactory`, yet the code's `override_runtime_arguments` hook (`move_sharded_program_factory.cpp:121`) fires for all three strategies. I treated the sheet's finer per-factory classification as authoritative (only sharded genuinely depends on dynamic RTAs). Please confirm with the readiness-sheet owner that this is the intended reading, and that the overlap / multi-core `override_runtime_arguments` is considered redundant re-application (not a gate).
2. **Subset port:** I recommend against an overlap-only subset port (shared `override_runtime_arguments`, mixed-concept device-op, one-of-three value) and against issuing a brief, favoring an all-three port after both prerequisites clear. Confirm you agree, or say if you want a subset brief for `MoveOverlapProgramFactory` regardless.

## Recipe notes

- **Config-scoped gate vs. shared-DeviceOperation coupling.** The recipe's Code-path scope rule (in `audit/metal2_audit.md`) says a config-scoped GATE with a surviving clean factory subset "still issues a brief for the clean subset." Here a clean factory (`MoveOverlapProgramFactory`) technically survives, but the three factories share one `MoveDeviceOperation` (single `override_runtime_arguments`, single `select_program_factory`), so a single-factory port yields a mixed-concept device-op with real coupling for marginal value. The recipe's GCB example (matmul) assumes independent factories; it does not address a shared override hook across a runtime-selected factory set. I judged "no practically-portable subset → whole-op RED, no brief" and documented the reasoning in Result. Flagging in case the maintainer wants the rule to speak to shared-DeviceOperation coupling.
- **Delegating factory (`MoveProgramFactory`) owns no kernels.** A factory whose `create_descriptor` delegates wholesale to another op's factory is a shape the recipe's per-factory model handles implicitly (follow kernel references), but it is worth an explicit line: the Device 2.0 gate and CB-endpoint census for such a factory are entirely the donor factory's, and the port-together coupling is total (porting move's `MULTI_CORE` = porting copy's `SameMemoryConfig`).
- **Included the informational (Port-work / CB endpoints) sections despite a whole-op RED.** The Red-outcome scoping rule says to skip the purely-informational subjects on a no-portable-subset RED. I included compact versions anyway because this op is small and stable (three short factories, four CBs total), so the census is cheap, unlikely to go stale, and directly useful to the eventual re-audit/all-clear port. Noting the deviation explicitly per the rule's spirit.
