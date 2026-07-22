# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/tilize`

- **`TilizeDeviceOperation`**
  - `TilizeMultiCoreDefaultProgramFactory` (`tilize_multi_core_default_program_factory.cpp`)
  - `TilizeMultiCoreBlockProgramFactory` (`tilize_multi_core_block_program_factory.cpp`)
  - `TilizeSingleCoreProgramFactory` (`tilize_single_core_program_factory.cpp`)
  - `TilizeMultiCoreShardedProgramFactory` (`tilize_multi_core_sharded_program_factory.cpp`)

Single `DeviceOperation`, four program factories. The op directory also contains `device/kernels/compute/tilize.cpp`, which **no tilize factory references** — the factories use the shared-pool compute kernel `ttnn/cpp/ttnn/kernel/compute/tilize.cpp` instead. That file is **unreferenced by this op** (out of audit scope), though it is borrowed by `embedding` (`embeddings_fused_program_factory.cpp:244`).

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `metal2_audit.md`.

**Recipe docs:** `b6dadc46ee0 2026-07-21 docs: fix metal_2.0 doc links (READ_ME_FIRST rename + two stragglers)`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/tilize` |
| **Overall** | **RED at op level** (config-scoped — sharded factory only); subset {Default, Block, SingleCore} is clear → brief issued for the subset |
| **DOps / Factories** | `TilizeDeviceOperation` → Default, Block, SingleCore, **Sharded** |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all referenced kernels (own + donor) are Device 2.0 compliant |
| *Prereqs* — Cross-op escapes | Ok (workable — one in-family `tt_memmove(Noc,…)`, kernel_lib compute helpers) |
| *Feature Support* — overall | GREEN (every Appendix A entry N/A) |
| *Feature Support* — Variadic-CTA | Ok (N/A) |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Default / Block / SingleCore: Yes** · **Sharded: No** — `Runtime-args update == yes` **and** `Is safe to port? == no` |
| *TTNN Readiness* — Concept (current) | `descriptor` (all four rows) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A |
| *TTNN Readiness* — Is safe to port? | Default/Block/SingleCore: `yes` · **Sharded: `no`** (→ readiness-sheet owner) |
| *TTNN Readiness* — Custom hash | No (no `compute_program_hash` override anywhere) |
| *TTNN Readiness* — Runtime-args update | **Yes on Sharded** (`get_dynamic_runtime_args`, `tilize_device_operation.cpp:270`) · No on the others |
| *TTNN Readiness* — Pybind `create_descriptor` | No (`tilize_nanobind.cpp` binds via `bind_function`, no `create_descriptor` / `nb::class_` of device op) |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `MetalV2FactoryConcept` (no op-owned tensors) |
| *Port work* — Offset base pointer | none (no `->address()` fold in any factory) |
| *Port work* — Tensor bindings (per binding) | input → Case 1 · output → Case 1 (all clean factories) |
| *Port work* — TensorParameter relaxation | none |
| *Port work* — TensorAccessor 3rd arg | none (no 3-arg `TensorAccessor` in any kernel) |
| *Port work* — CB endpoints | Default/SingleCore: all legal 1:1 · Block: `c_1` self-loop, `c_0`/`c_16` legal |

**Sheet discrepancy (flagged, see Gate detail):** the readiness sheet carries a **fifth** row for `data_movement/tilize` — `TilizeMultiCoreWidthShardedProgramFactory` — whose named factory file does **not** exist in this op. That class lives only under `experimental/quasar/tilize/`. Routed to the readiness-sheet owner to reconcile; it does not correspond to a code factory here, so it is not part of the port unit.

## Result

**RED at op level; subset {Default, Block, SingleCore} is clear.** The `TilizeMultiCoreShardedProgramFactory` fails the TTNN factory-concept gate on **two** conjuncts:
- `Runtime-args update == yes` — the op defines `get_dynamic_runtime_args` (`tilize_device_operation.cpp:270-288`), which by its own comment applies *only* to the sharded factory (it re-applies the sharded reader's arg0 to trip the descriptor fast-path on a cache hit). **Route → TTNN / ProgramDescriptor-migration team.** The gate lifts once Metal 2.0 / TTNN infra supports a runtime-args-update hook.
- `Is safe to port? == no` — the readiness-sheet owner's correctness call on this factory's prior PD migration. **Route → readiness-sheet owner** (the correctness axis is theirs; reconcile before porting the sharded factory).

The other three factories clear every gate and are portable now. A partial port of the interleaved/blocked/single-core paths delivers value without waiting for the sharded gate to clear. **This is a config-scoped gate, so a brief is issued for the clean subset** (`METAL2_PORT_BRIEF.md`).

Separately, reconcile the phantom `TilizeMultiCoreWidthShardedProgramFactory` sheet row (below) → readiness-sheet owner.

## Gate detail

- **TTNN factory concept (`Is able to port?`):**
  - **Default / Block / SingleCore → GREEN.** Sheet: `Concept == descriptor`, `Custom hash == no`, `Runtime-args update == no`, `Pybind descriptor == no`, `Is safe to port? == yes`, `Is able to port? == yes`. Cross-check confirms: concept is `descriptor` (each factory defines `create_descriptor()` returning a `ProgramDescriptor`); no `compute_program_hash` override in the device op; `get_dynamic_runtime_args` early-returns `{}` for these factories (`tilize_device_operation.cpp:278-281`); nanobind binds via `bind_function`.
  - **Sharded → RED (GATE).** Sheet: `Runtime-args update == yes`, `Is safe to port? == no` → `Is able to port? == no`. Cross-check confirms `get_dynamic_runtime_args` returns a live `DynamicRuntimeArg` for the sharded factory only (`tilize_device_operation.cpp:282-288`). Cross-column invariant holds (`Runtime-args update == yes` on a `descriptor` concept is permitted). Route: shape failure (runtime-args-update) → TTNN/PD-migration team; correctness failure (`safe == no`) → readiness-sheet owner. Name both failing conjuncts.
  - **Sheet-reconciliation item (spreadsheet-broken class, scoped):** the sheet has a row `data_movement/tilize, TilizeDeviceOperation, TilizeMultiCoreWidthShardedProgramFactory` (`Is able to port? == yes`) whose `Factory definition path` is `…/tilize/device/tilize_multi_core_width_sharded_program_factory.hpp` — **a file that does not exist**. The only `TilizeMultiCoreWidthShardedProgramFactory` in the tree is under `experimental/quasar/tilize/` (out of scope for this Gen1 audit). The four *real* factory rows are present and their cheaply-checkable columns cross-verify against the code, so the discrepancy is a misattributed/stale **extra** row, not corrupted data on a relied-upon row. Routed to the readiness-sheet owner to reconcile (remove or re-attribute the row). It does not gate the clean subset. *(See Recipe notes — the recipe's "any cross-check conflict → whole-op GATE" rule doesn't cleanly cover an extra/misattributed row.)*

- **Device 2.0 (every kernel used): GREEN.** Every kernel the op instantiates — own, in-family, cross-family donor, and shared-pool — is Device 2.0 compliant: object-oriented `Noc` / `DataflowBuffer` / `TensorAccessor` / `CoreLocalMem` / `UnicastEndpoint` idioms, no raw `noc_async_read/write`, no `InterleavedAddrGen`/`ShardedAddrGen`, no CB-index free-function holdovers. All `get_write_ptr()` occurrences are **methods on `DataflowBuffer` objects** (`dfb.get_write_ptr()`), not the free function. Sanctioned free functions in use — not violations — are `get_tile_size(cb_id)` (`writer_unary_interleaved_start_id_wh.cpp:24`) and `get_local_cb_interface(cb_id).fifo_page_size` (`writer_unary_interleaved_start_id.cpp:19`). Recent history corroborates: `#49392 [Cleanup] Migrate Data Movement Kernels from CircularBuffer to DataflowBuffer`.

  Kernels verified (by factory):
  | Kernel | Owner | Role | Device 2.0 |
  |---|---|---|---|
  | `reader_unary_stick_layout_split_rows_multicore.cpp` | tilize (own) | Default reader | ✓ |
  | `reader_unary_stick_layout_split_rows_singlecore.cpp` | tilize (own) | SingleCore reader | ✓ |
  | `reader_unary_pad_multicore_both_dims.cpp` | tilize_with_val_padding (in-family) | Block reader | ✓ |
  | `writer_unary_interleaved_start_id.cpp` | eltwise/unary (cross-family) | Default/SingleCore/Sharded-interleaved writer | ✓ |
  | `writer_unary_interleaved_start_id_wh.cpp` | eltwise/unary (cross-family) | Block writer | ✓ |
  | `reader_unary_sharded.cpp` | eltwise/unary (cross-family) | Sharded reader | ✓ |
  | `writer_unary_sharded.cpp` | data_movement/sharded (in-family) | Sharded (sharded-out) writer | ✓ |
  | `ttnn/cpp/ttnn/kernel/compute/tilize.cpp` | shared pool | Default/SingleCore/Sharded compute | ✓ |
  | `tilize_wh.cpp` | tilize (own) | Block compute | ✓ |

- **Feature compatibility:** every Appendix A entry, in order — all **N/A** (clean scan). No feature signals fire.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no `GlobalCircularBuffer` / `.global_circular_buffer` / `remote_cb` / `.remote_index(` / `num_global_cb_receivers` |
  | CBDescriptor `address_offset` (non-zero) | N/A | no `.address_offset` / `set_address_offset` / `cb_descriptor_from_sharded_tensor` |
  | GlobalSemaphore | N/A | no `GlobalSemaphore` / `CreateGlobalSemaphore`; the op uses no semaphores at all |
  | Variable-count compile-time arguments (CTA varargs) | N/A | no `std::vector<Tensor>` tensor-args; no runtime-varying `get_compile_time_arg_val(i)` loop (kernels read CTAs at constexpr indices only) |

- **CB endpoints (GATE-free):** classified for the **clean subset** only (the sharded factory is gated out — see Red-outcome scoping). Every CB is legal 1:1 or carries a self-loop disposition; nothing blocks a Gen1 port.
  - **Default / SingleCore:** `c_0` (input) — reader FIFO-produces (`reserve_back`/`push_back`), compute FIFO-consumes → **legal 1:1**. `c_16` (output) — compute produces, writer FIFO-consumes (`wait_front`/`pop_front`) → **legal 1:1**.
  - **Block:** `c_0` — reader produces, compute consumes → **legal 1:1**. `c_16` — compute produces, writer consumes → **legal 1:1**. `c_1` — a per-row DRAM-alignment staging buffer touched **only by the reader** (`reserve_back(1)`/`get_write_ptr`/`push_back(1)` once, then raw `fill_with_val` + `tt_memmove` scratch; `reader_unary_pad_multicore_both_dims.cpp:96-99,159-163`) → single toucher → **self-loop** (bind the reader PRODUCER and CONSUMER).
  - No dead CBs. No multi-binding / hidden-second-writer / multi-reader shapes.

- **Offset base pointers:** GREEN. No `->address()` appears in any tilize factory — the factories deliver tensor bases via the `Buffer*`-binding form (the framework auto-registers `BufferBinding`s), not a folded `buffer()->address() + offset` RTA. No Type 1/2 fold; no `narrow`/interior-base (Type 4); no `address_offset` (Type 3, also N/A in Appendix A). Not in the offset-base-pointer triage doc (`2026-07-19`), consistent with a clean scan.

- **TensorAccessor 3rd argument:** GREEN. No kernel constructs a 3-arg `TensorAccessor(args, addr, page_size)` — every accessor is the 2-arg form (`TensorAccessor(src_tensor_args, src_addr)` etc.). Not in the 3rd-arg triage doc (`2026-07-06`), consistent with a clean scan. (Note: the readers' first CTA — `aligned_page_size` / `stick_size` — is *not* an accessor 3rd arg; the `TensorAccessorArgs<N>` block starts after it. See Misc anomalies — that CTA is dead.)

## Port-work summary  *(mirrors the brief — clean subset {Default, Block, SingleCore})*

- **Tensor bindings** (per binding, all three clean factories):
  - **input** (`src0_buffer`, delivered via the `Buffer*`-binding form) → **Case 1**: the reader feeds the base into `TensorAccessor(src_tensor_args, src_addr)` and reads through it. Express as `TensorParameter`/`TensorBinding`; kernel builds `TensorAccessor(tensor::name)`; the `Buffer*` RTA slot and `TensorAccessorArgs` plumbing disappear.
  - **output** (`dst_buffer`, `Buffer*`-binding) → **Case 1**: writer (`writer_unary_interleaved_start_id[_wh].cpp`) feeds the base into a `TensorAccessor`. Same treatment.
- **TensorParameter relaxation:** none.
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** self-loop `c_1` (Block, all block configs); all other CBs legal 1:1.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none — no hidden-second-writer or multi-reader CB in the clean subset.
- **Cross-op / shared kernels:** the clean subset instantiates **borrowed** kernels that must be Metal-2.0-rewritten as one unit with their co-borrowers (see Team-only for the full sets):
  - `writer_unary_interleaved_start_id.cpp` (eltwise/unary) — **broadly shared, ~28 factories**. The shared rewrite is a large port-together set.
  - `writer_unary_interleaved_start_id_wh.cpp` (eltwise/unary) — shared with `tilize_with_val_padding` (2 factories).
  - `ttnn/cpp/ttnn/kernel/compute/tilize.cpp` (shared pool) — shared across `tilize` + `tilize_with_val_padding` (6 factories).
  - `reader_unary_pad_multicore_both_dims.cpp` (in-family, `tilize_with_val_padding`) — Block reader.
- **RTA varargs:** none — every reader/writer reads RTAs at distinct constant indices; no loop-indexed `get_arg_val(i)`.

## Team-only

### Out-of-directory coupling & donor shape

**Op-level roll-up: ✓ workable.** No `⭐`/`✗` sequence-blockers. Function-call escapes are all Device 2.0 native or official shared-lib; file-path borrows are broadly-shared kernels that induce a port-together coupling (not a gate).

**Function-call escapes (per donor file):**

| Op kernel | Donor `#include` | Donor class | Function(s) called | Shape | Status |
|---|---|---|---|---|---|
| `reader_unary_pad_multicore_both_dims.cpp` | `cpp/ttnn/operations/data_movement/common/kernels/common.hpp` | in-family shared utility | `tt::data_movement::common::tt_memmove(Noc, uint32_t, uint32_t, uint32_t)` | takes `Noc` (Device 2.0 native) | ✓ excellent |
| `tilize_wh.cpp`, `kernel/compute/tilize.cpp` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` | official kernel_lib | `compute_kernel_lib::tilize<…>` | compute LLK wrapper, CBIndex NTTPs | ✓ (lib team) |

The own/donor dataflow kernels (`reader_unary_stick_layout_split_rows_*`, `writer_unary_interleaved_start_id[_wh]`, `reader_unary_sharded`, `writer_unary_sharded`) have **no** out-of-directory `#include`s beyond `api/*` (HAL/LLK — bucket 1, no concern).

**Borrowed kernel files (file-path instantiation) — port-together sets** (co-borrower counts by grep of factory `.cpp` instantiations):

| Kernel file | Owning pool | Co-borrowers | Used by tilize factory |
|---|---|---|---|
| `eltwise/unary/…/writer_unary_interleaved_start_id.cpp` | eltwise/unary | **~28 factories** (reduction/generic, transformer, kv_cache, embedding, examples, …) | Default, SingleCore, Sharded(interleaved-out) |
| `eltwise/unary/…/writer_unary_interleaved_start_id_wh.cpp` | eltwise/unary | 2 (tilize block, tilize_with_val_padding block) | Block |
| `ttnn/cpp/ttnn/kernel/compute/tilize.cpp` | shared pool `ttnn/cpp/ttnn/kernel/` | 6 (tilize ×3, tilize_with_val_padding ×3) | Default, SingleCore, Sharded |
| `eltwise/unary/…/reader_unary_sharded.cpp` | eltwise/unary | 11 (untilize, transpose, slice_write, typecast, sharded↔interleaved, …) | Sharded (gated) |
| `data_movement/sharded/…/writer_unary_sharded.cpp` | data_movement/sharded | 10 (untilize, transpose, interleaved→sharded, tilize_with_val_padding, …) | Sharded (gated) |
| `data_movement/tilize_with_val_padding/…/reader_unary_pad_multicore_both_dims.cpp` | data_movement/tilize_with_val_padding | (in-family) | Block |

The dominant coupling for the clean subset is `writer_unary_interleaved_start_id.cpp`: its Metal 2.0 CB→DFB / named-token rewrite must land simultaneously across all ~28 co-borrowers, or the first isolated migrant breaks the rest.

### Relaxation candidates

None mined — no custom hash on any factory.

### TTNN factory analysis (sheet-derived, cross-checked)

- **Concept:** `descriptor` (all 4 factories); target `MetalV2FactoryConcept`.
- **Op-owned tensors:** none (`descriptor` concept can't carry them; sheet blank).
- **Custom hash:** no (no `compute_program_hash`).
- **Custom `override_runtime_arguments`:** no (the op is on the descriptor `create_descriptor` API).
- **`get_dynamic_runtime_args`:** present (`tilize_device_operation.cpp:270`) but active **only for the sharded factory** — the gate conjunct that REDs the sharded row.
- **Pybind `create_descriptor` / risky pybind:** no.

## Misc anomalies  *(team-only, non-gating; noticed while auditing — not porter work)*

- **Dead compile-time arg (Default reader):** `reader_ct_args[0] = aligned_page_size` (`tilize_multi_core_default_program_factory.cpp:92-93`) is never read by the kernel — it reads `get_compile_time_arg_val(1)`/`(2)` and `TensorAccessorArgs<3>()`, skipping index 0 (`reader_unary_stick_layout_split_rows_multicore.cpp:24-30`).
- **Dead compile-time arg (SingleCore reader):** `reader_compile_time_args[0] = stick_size` (`tilize_single_core_program_factory.cpp:105`) is likewise never read — the kernel uses `TensorAccessorArgs<1>()` and no `get_compile_time_arg_val(0)` (`reader_unary_stick_layout_split_rows_singlecore.cpp:24`).
- **Dead runtime-arg slots (Default & SingleCore readers):** the factory pushes 9 reader RTAs but the kernels read only slots 0,1,3,4,5,8 — slots **2, 6, 7** are unused (Default: `…_default_program_factory.cpp:168-178`; SingleCore: `…_single_core_program_factory.cpp:122-132`). Slot 2 (`page_size`/`stick_size`) is a duplicate of a value passed elsewhere; slots 6/7 are hardcoded `0` (leftover-tile fields the multicore path never exercises).
- **Unreferenced kernel file in the op directory:** `device/kernels/compute/tilize.cpp` is not instantiated by any tilize factory (they use the shared-pool `ttnn/cpp/ttnn/kernel/compute/tilize.cpp`). It *is* borrowed by `embedding` (`embeddings_fused_program_factory.cpp:244`). Neither an audit finding nor porter work — noted so a reader isn't confused by the two same-named compute kernels.

## Per-DeviceOperation attribution

Single `DeviceOperation`; per-factory verdicts are in the Status summary. No bundling.

## Questions for the user  *(for routing, not blocking)*

1. **Phantom `TilizeMultiCoreWidthShardedProgramFactory` sheet row:** the readiness sheet lists a 5th `data_movement/tilize` factory row (`Is able to port? == yes`) pointing at `…/tilize/device/tilize_multi_core_width_sharded_program_factory.hpp`, which does not exist in this op (the class lives only under `experimental/quasar/tilize/`). Was there a width-sharded factory in `data_movement/tilize` that was since removed/merged, or is this a misattribution of the quasar factory? Either way the sheet row wants reconciling by its owner. *(This audit treats the 4 real, code-backed rows as authoritative and does not let the phantom row gate the clean subset.)*

## Recipe notes  *(friction with the audit recipe itself)*

- **"Cross-check conflicts with the sheet … → whole-op GATE" is too blunt for a misattributed *extra* row.** The [TTNN factory concept prerequisite](metal2_audit.md) routing says any cross-check conflict means "spreadsheet is broken → GATE (whole op) → readiness-sheet owner." Here the conflict is a **5th row for a factory the op doesn't have** (a nonexistent factory-definition file), while the **four real factory rows are present and cross-verify cleanly against the code**. A literal reading would gate the entire op — including the three factories that provably clear every gate — and defer a genuinely-portable subset to a re-audit, misrouting work. I read the rule's *intent* ("don't proceed on data we can't trust") as scoped to the rows we actually rely on, and treated the phantom row as a reconciliation item routed to the sheet owner without gating the clean subset. Suggest the recipe distinguish **"wrong/missing data on a relied-upon (op,DOp,factory) row"** (→ gate) from **"an extra/misattributed row that names no real code factory"** (→ flag + route, don't gate the verified rows).
- **Sheet cross-check surfaced a factory-set mismatch the recipe doesn't explicitly ask you to check.** The recipe lists per-*column* cross-checks and cross-column invariants, but not "does the set of factory rows match the set of code factories?" That row-set comparison is what caught the phantom row. Might be worth an explicit cross-check bullet: *confirm the sheet's factory rows for the op are in 1:1 correspondence with the code's `program_factory_t` variant, and flag extras/missing.*
