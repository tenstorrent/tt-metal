# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/tilize`

Single device operation, six program factories:

- **`TilizeDeviceOperation`** (`device/tilize_device_operation.{hpp,cpp}`, types in `device/tilize_device_operation_types.hpp`)
  - `TilizeMultiCoreDefaultProgramFactory` (`tilize_multi_core_default_program_factory.cpp`)
  - `TilizeSingleCoreProgramFactory` (`tilize_single_core_program_factory.cpp`)
  - `TilizeMultiCoreShardedProgramFactory` (`tilize_multi_core_sharded_program_factory.cpp`)
  - `TilizeMultiCoreShardedRetileProgramFactory` (`tilize_multi_core_sharded_retile_program_factory.cpp`)
  - `TilizeMultiCoreRetileProgramFactory` (`tilize_multi_core_retile_program_factory.cpp`)
  - `TilizeMultiCoreBlockProgramFactory` (`tilize_multi_core_block_program_factory.cpp`) — **blocked, see Result**

**Unreferenced kernel (out of scope):** `device/kernels/compute/tilize.cpp` is referenced by **no** factory — every compute path uses the shared-pool `ttnn/cpp/ttnn/kernel/compute/tilize.cpp` or the op-local `retile.cpp` / `tilize_wh.cpp`. Dead file in the op directory; its contents were not audited.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `7d5ddd43e0e 2026-08-27 docs(metal_2.0): a run in flight freezes the kernel sources`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/tilize` |
| **Overall** | **RED at op level; clean 5-factory subset is portable** |
| **DOps / Factories** | `TilizeDeviceOperation` → Default, SingleCore, Sharded, ShardedRetile, Retile *(portable)* · Block *(blocked)* |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes (GREEN)** — all 11 referenced kernels Device 2.0 compliant |
| *Prereqs* — Cross-op escapes | Ok — all cross-op donors are Device 2.0; all shared donors already have `_metal2` forks |
| *Feature Support* — overall | **GREEN** — no Appendix A feature in use |
| *Feature Support* — GlobalCircularBuffer / address_offset / GlobalSemaphore | N/A / N/A / N/A |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Block = No** (`Known op issues` = "Per-node CB size" → ops team) · other 5 = **Yes** |
| *TTNN Readiness* — Concept (current) | `descriptor` (all six) |
| *TTNN Readiness* — Secretly SPMD | N/A (not `WorkloadDescriptor`) |
| *TTNN Readiness* — Custom hash | No |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No |
| *TTNN Readiness* — `override_runtime_arguments` | **Yes** (all six) — selects `CustomProgramSpecFactoryConcept`; see below |
| *TTNN Readiness* — Pybind `create_descriptor` | No |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | **`CustomProgramSpecFactoryConcept`** (all six; sheet `Porting Target` agrees) |
| *Port work* — Offset base pointer | **none (GREEN)** — every address fold is a clean `->address()` |
| *Port work* — Tensor bindings (per binding) | Case 1 (interleaved I/O) + clean (borrowed-memory sharded) — see Port-work summary |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) — all six |
| *Port work* — TensorAccessor 3rd arg | **none** — no accessor passes a 3rd argument |
| *Port work* — CB endpoints | legal 1:1 + self-loop (retile intermediates); no multi-binding, no dead CBs |

## Result

**RED at op level; subset {Default, SingleCore, Sharded, ShardedRetile, Retile} is clear** — a **brief is issued for that 5-factory subset**.

The **`TilizeMultiCoreBlockProgramFactory`** is blocked: the readiness sheet marks it `Is able to port? = no` with `Known op issues = "Per-node CB size"`, routed to the **ops team**. This is a config-scoped gate (per Code-path scope) — every other gate is GREEN, and the other five factories all read `Is able to port? = yes`. The Block factory is reachable only via `select_program_factory` for (a) Blackhole non-sharded UINT8 inputs, (b) `!enough_space_height` non-sharded inputs, and (c) the wide-tensor `compute_ncores_wh` heuristic; the port keeps the legacy path for those cases until the block issue clears.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** RED at op level, clean subset survives.
  - **Block = `no`**, attributed to the `Known op issues` column, value **"Per-node CB size"** (verbatim). Grounded in code: `push_cb_pair(desc, core_ranges, …, num_tiles, …)` (`tilize_multi_core_block_program_factory.cpp:28`, pushing c_0 / c_1 / c_16) is invoked **up to four times over disjoint core-ranges** (`core_range`, `cliff_col_row_core_range`, `cliff_row_core_range`, `cliff_col_core_range`; lines 158-209) with **different tile counts** (`single_sub_block_size` vs `single_block_size_cliff_row`), so the same `buffer_index` is allocated at **different `total_size` / `page_size` on different nodes**.

    **Why Metal 2.0 cannot express this today (mechanism):** `DataflowBufferSpec` (`tt_metal/api/tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp:79`) carries a single `entry_size` + `num_entries` per named DFB (`unique_id`) and **no `core_range` field** — a local DFB's placement is derived from the binding kernels, and its size is one value (runtime-overridable only as a single per-execution value, not per node). A named DFB is therefore uniform-sized across every node it lands on. Faithfully porting the Block factory's per-group sizes would require restructuring each CB into several distinct named DFB specs bound per-core-group kernel instance — a genuine refactor, not the mechanical CB→DFB swap the port is scoped to. This is what the "Per-node CB size" issue tracks. Routed to the **ops team / framework** to resolve before this factory ports (clears when the framework grows per-node DFB sizing, or the factory is refactored to uniform / per-group named DFBs). Gate cleared for the other five factories.

    **Note on the readiness-sheet state (2026-08-27):** the derived `Is able to port?` cell for this row was manually flipped to `yes`, but the `Known op issues` cell it derives from **still reads "Per-node CB size"** — an internal inconsistency (derived verdict vs. its unresolved blocking input). Per the recipe this is routed to the readiness-sheet owner to reconcile; it does **not** clear the gate, and the code + DFB-spec mechanism above confirm the block is real regardless of the summary cell.
  - Cross-check clean: `Concept` = `descriptor` (all six confirmed — `create_descriptor` returning `ProgramDescriptor`); `Custom hash` = no (no `compute_program_hash` / `attribute_values` / `to_hash` in the op); `get_dynamic_runtime_args` = no (no hook on the device-op); `override_runtime_arguments` = yes (all six define it); `Pybind descriptor` = no (`tilize_nanobind.cpp` binds no `create_descriptor`). Factory-set match: 6 sheet rows ↔ 6 code factories, one-to-one. No cross-column invariant violations.

- **Device 2.0 (every kernel used):** **GREEN.** All 11 referenced kernels are Device 2.0 compliant — no `InterleavedAddrGen` / `ShardedAddrGen` / raw `noc_async_*` / manual CB-index management anywhere. Data movement is expressed through `TensorAccessor` (`s.get_noc_addr(...)`) and `DataflowBuffer` wrappers. The only CB-index free functions present are **sanctioned**: `get_tile_size(cb_id)` (`writer_unary_interleaved_start_id_wh.cpp:24`, `reader_unary_start_id.cpp:25`) and `get_local_cb_interface(cb_id)` (`retile.cpp:108,119`, `writer_unary_interleaved_start_id.cpp:27`). No holdovers.

  Referenced kernels (all Device 2.0):

  | Kernel | Owner | Used by (subset) |
  |---|---|---|
  | `tilize/device/kernels/dataflow/reader_unary_stick_layout_split_rows_multicore.cpp` | op-owned | Default |
  | `tilize/device/kernels/dataflow/reader_unary_stick_layout_split_rows_singlecore.cpp` | op-owned | SingleCore |
  | `tilize/device/kernels/compute/retile.cpp` | op-owned | Retile, ShardedRetile |
  | `ttnn/cpp/ttnn/kernel/compute/tilize.cpp` | shared pool (`kernel/`) | Default, SingleCore, Sharded |
  | `eltwise/unary/.../writer_unary_interleaved_start_id.cpp` | eltwise/unary donor | Default, SingleCore, Sharded, ShardedRetile, Retile |
  | `untilize/.../reader_unary_start_id.cpp` | data_movement/untilize donor | Retile |
  | `eltwise/unary/.../reader_unary_sharded.cpp` | eltwise/unary donor | Sharded, ShardedRetile |
  | `sharded/.../writer_unary_sharded.cpp` | data_movement/sharded donor | Sharded, ShardedRetile |
  | `tilize/device/kernels/compute/tilize_wh.cpp` | op-owned | *Block only (excluded)* |
  | `tilize_with_val_padding/.../reader_unary_pad_multicore_both_dims.cpp` | tilize_with_val_padding donor | *Block only (excluded)* |
  | `eltwise/unary/.../writer_unary_interleaved_start_id_wh.cpp` | eltwise/unary donor | *Block only (excluded)* |

- **Feature compatibility:** **GREEN** — clean scan, all-N/A.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | no `GlobalCircularBuffer` / `remote_cb` / `.remote_index` / `global_circular_buffer` field |
  | CBDescriptor `address_offset` (non-zero) | N/A | sharded factories use buffer-backed CBs (`cb.buffer = …`) with **zero** offset; no `.address_offset` / `set_address_offset` anywhere |
  | GlobalSemaphore | N/A | no `GlobalSemaphore` / `CreateGlobalSemaphore`; no semaphores in any referenced kernel |

- **CB endpoints (GATE-free):** all CBs are legal 1:1 or self-loop; no dead CBs, no multi-binding. Detail per factory in Port-work summary. **Not deferred** — the Device 2.0 gate is GREEN, so the idioms the scan keys on are intact.

- **Offset base pointers:** **GREEN.** Every device-pointer that reaches a kernel is a clean base. Address delivery uses the `Buffer*`-binding form in `emplace_runtime_args` (framework auto-registers `BufferBinding`s) plus a manual slot-0 re-point in each `override_runtime_arguments` via `patch_tilize_kernel_slot0(program, k, buffer->address())` — a bare `->address()`, no `+ offset` arithmetic anywhere in the op. No Type 1/2/3/4.

- **TensorAccessor 3rd argument:** **N/A** — no accessor in the op passes a 3rd argument. Every construction is the 2-arg `TensorAccessor(args, addr)` form (`reader_unary_stick_layout_split_rows_multicore.cpp:32`, `…singlecore.cpp:26`, `reader_unary_pad_multicore_both_dims.cpp:34`, `writer_unary_interleaved_start_id.cpp:39`, `writer_unary_interleaved_start_id_wh.cpp:26`, `reader_unary_start_id.cpp:28`). The sharded kernels use no `TensorAccessor` at all (borrowed-memory). Subject never fires.

## Port-work summary  *(mirrors the brief — 5-factory subset)*

- **Tensor bindings** (per binding, per factory):
  - **Default:** `input` Case 1 (fed to `TensorAccessor` in `reader_unary_stick_layout_split_rows_multicore.cpp`) · `output` Case 1 (`writer_unary_interleaved_start_id.cpp`).
  - **SingleCore:** `input` Case 1 (`…singlecore.cpp`) · `output` Case 1 (`writer_unary_interleaved_start_id.cpp`).
  - **Retile:** `input` Case 1 (`untilize/reader_unary_start_id.cpp`) · `output` Case 1 (`writer_unary_interleaved_start_id.cpp`).
  - **Sharded:** `input` **clean** (borrowed-memory CB `cb_src0.buffer = src_buffer`; `reader_unary_sharded.cpp` only `push_back`s) · `output` **clean** for sharded output (`cb_output.buffer = dst_buffer`; `writer_unary_sharded.cpp`) / **Case 1** for INTERLEAVED output (local CB drained by `writer_unary_interleaved_start_id.cpp` via `TensorAccessor`). Per-config split.
  - **ShardedRetile:** same shape as Sharded — `input` clean (borrowed) · `output` clean (borrowed) / Case 1 (interleaved output).
  - Note: the Case-1 bindings today ride the `Buffer*`-binding form **plus** the manual `patch_tilize_kernel_slot0` slot-0 re-point in `override_runtime_arguments`. The Metal 2.0 port replaces **both** with a typed `TensorParameter`/`TensorBinding` (framework refreshes on cache hit) — the `Buffer*` arg, the `TensorAccessorArgs` plumbing, and the slot-0 patch all disappear.
- **TensorParameter relaxation:** `none` (all factories) — clears.
- **TensorAccessor 3rd arg:** none.
- **CB endpoints** (per factory; classify per `(CB, config)`):
  - **Default:** `c_0` **1:1** (reader PRODUCER, compute CONSUMER) · `c_16` **1:1** (compute PRODUCER, writer CONSUMER). Compute is emitted as two same-source kernels over **disjoint** `core_range` (full) and `core_range_cliff` (cliff) node sets — each node sees one compute instance → ordinary 1:1 (the demoting-per-group shape, **not** multi-binding).
  - **SingleCore:** `c_0` **1:1** · `c_16` **1:1** (single core).
  - **Retile:** `c_0` **1:1** (reader P, compute C) · `c_1` (`mid_cb`) **self-loop** (compute both produces — `untilize` + `fill_zeros_pages` — and consumes — `mid.wait_front`/`pop_front`) · `c_2` (`mid_view_cb`) **self-loop** (compute-only; no FIFO producer, read cursor driven by `get_local_cb_interface(mid_view_cb).fifo_rd_ptr = …` then consumed by `tilize_block`) · `c_16` **1:1** (compute P, writer C).
  - **Sharded:** `c_0` **1:1** (reader P, compute C; borrowed-memory) · `c_16` **1:1** (compute P, writer C; borrowed for sharded output, local for interleaved output).
  - **ShardedRetile:** `c_0` **1:1** (borrowed) · `mid_cb` **self-loop** · `mid_view_cb` **self-loop** · `c_16` **1:1** (borrowed / local).

## Heads-ups  *(mirrors the brief)*

- **Aliased intermediate CB — two format descriptors on one allocation (Retile / ShardedRetile).** `mid_cb` (`c_1`) and `mid_view_cb` (`c_2`) are two `CBFormatDescriptor`s inside a **single** `CBDescriptor` (`tilize_multi_core_retile_program_factory.cpp:136-153`): one L1 region, two buffer indices, different tile/face geometry (input-tile shape for the untilize producer, output-tile shape for the tilize consumer). The kernel drives `c_2`'s read pointer manually into the shared region. This is **not** an Appendix A feature (no `address_offset`, no GlobalCircularBuffer) so it does not gate, but it is the one genuinely non-boilerplate construct in the port — the porter must confirm a single `DataflowBufferSpec` can carry two format descriptors sharing one allocation (or express the alias another way). See *Recipe notes*.
- **Cross-op / shared kernels — all shared donors already have `_metal2` forks (reuse, don't re-fork):**
  - `ttnn/cpp/ttnn/kernel/compute/tilize.cpp` → `…tilize_metal2.cpp` **exists**
  - `eltwise/unary/.../writer_unary_interleaved_start_id.cpp` → `…_metal2.cpp` **exists**
  - `untilize/.../reader_unary_start_id.cpp` → `…_metal2.cpp` **exists**
  - `eltwise/unary/.../reader_unary_sharded.cpp` → `…_metal2.cpp` **exists** (fork note in file header)
  - `sharded/.../writer_unary_sharded.cpp` → `…_metal2.cpp` **exists** (sunset tracked in issue #52228)
  - Op-owned kernels (`reader_unary_stick_layout_split_rows_multicore.cpp`, `…singlecore.cpp`, `retile.cpp`) are ported in place — no fork needed.
- **RTA varargs:** none — every kernel reads a fixed set of named args; no loop-indexed / `arg_index++`-in-loop / data-selected reads.

## Team-only

- **Out-of-directory coupling & donor shape.** Roll-up: **✓ clean** for the 5-factory subset.
  - *File-path kernel instantiation (borrowed kernel files):* the five shared donors listed under Heads-ups. Each has an existing `_metal2` fork → the port binds the fork; this op is one consumer on each fork's **sunset list** (coordination/retire list, **not** authorization to convert the kernel in place). `writer_unary_sharded.cpp` sunset is tracked in issue #52228.
  - *Function-call escape:* `retile.cpp` `#include`s `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp` and `untilize_helpers.hpp` — the official shared kernel library (`kernel_lib/`), lib-team-handled, no concern. No other cross-dir function-call escapes in the subset kernels.
  - *Block factory (excluded) donors, for planning:* `tilize_with_val_padding/.../reader_unary_pad_multicore_both_dims.cpp` and `eltwise/unary/.../writer_unary_interleaved_start_id_wh.cpp` (both Device 2.0). Not relevant until the "Per-node CB size" issue clears.
- **TTNN factory analysis (sheet-derived, `file:line` confirmed):** Concept `descriptor` (all six) → target `CustomProgramSpecFactoryConcept` (Override runtime args = yes; sheet `Porting Target` agrees). No op-owned tensors, no custom hash, no `get_dynamic_runtime_args`, no pybound `create_descriptor`. `override_runtime_arguments` sites (translate into a `ProgramRunArgs`-returning method per the port recipe): `tilize_multi_core_default_program_factory.cpp:236`, `tilize_single_core_program_factory.cpp` (override method), `tilize_multi_core_sharded_program_factory.cpp:165`, `tilize_multi_core_sharded_retile_program_factory.cpp`, `tilize_multi_core_retile_program_factory.cpp`. All route through the shared `patch_tilize_kernel_slot0` helper (`tilize_device_operation.cpp:372`); the sharded/sharded-retile factories additionally rebuild borrowed-CB base addresses via a throwaway `cb_addr_only` `ProgramDescriptor` + `apply_descriptor_runtime_args` (`tilize_multi_core_sharded_program_factory.cpp:180-183`) — this is the borrowed-memory address refresh, which the port expresses as `DataflowBufferSpec::borrowed_from`.

## Misc anomalies  *(team-only, non-gating)*

- **Dead kernel file:** `device/kernels/compute/tilize.cpp` is referenced by no factory (every compute path uses the shared-pool `ttnn/cpp/ttnn/kernel/compute/tilize.cpp`). Candidate for deletion — route to ops team; the port does not act on it.

## Per-DeviceOperation attribution

Single DeviceOperation (`TilizeDeviceOperation`); per-factory status captured in the Status summary and Result. Portable subset: Default, SingleCore, Sharded, ShardedRetile, Retile. Blocked: Block (Per-node CB size).

## Recipe notes

- **Aliased-intermediate CB (two `CBFormatDescriptor`s in one `CBDescriptor`).** The audit recipe's CB-endpoints subject counts endpoints per buffer_index and the feature list (Appendix A) does not mention a multi-format-descriptor / aliased CB, so the recipe leaves it as "supported, mechanical." In practice this is the highest-uncertainty item of the port (retile's `c_1`/`c_2` share one L1 allocation with different tile geometry, and the kernel drives `c_2`'s read pointer by hand into the shared region). It would help if the port recipe (or a shared reference) stated explicitly how a `DataflowBufferSpec` expresses two views over one allocation, so the porter isn't the first to discover whether it's expressible. Flagging here rather than gating, per the "not in Appendix A ⇒ supported" rule.
