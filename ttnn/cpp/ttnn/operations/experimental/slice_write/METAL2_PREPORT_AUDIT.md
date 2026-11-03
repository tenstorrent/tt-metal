# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/slice_write`

- **`SliceWriteDeviceOperation`** (`device/slice_write_device_operation.{hpp,cpp}`), one device-op with three factories:
  - `SliceWriteRMInterleavedProgramFactory` (`device/slice_write_rm_interleaved_program_factory.cpp`). Used when the input is not sharded.
  - `SliceWriteRMShardedInputProgramFactory` (`device/slice_write_rm_sharded_input_program_factory.cpp`). Used for a sharded ROW_MAJOR input (height or block sharded).
  - `SliceWriteTiledShardedInputProgramFactory` (`device/slice_write_tiled_sharded_input_program_factory.cpp`). Used for a sharded TILE input.
- Kernels bound:
  - Own: `device/kernels/dataflow/slice_write_reader_interleaved.cpp` (RMInterleaved), `slice_write_writer_interleaved_strided.cpp` (RMInterleaved), `slice_write_writer_interleaved.cpp` (RMSharded **and** TiledSharded, so it is shared within the op).
  - Borrowed: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp` (RMSharded, TiledSharded).
  - Every kernel file in the op directory is referenced; there are no unreferenced kernel files.
- Selection logic: `slice_write_device_operation.cpp:16-38`. The composite caller `slice_write.cpp:38` rejects sharded outputs, so every factory writes to an **interleaved** output.

**Scope:** TTNN op, Gen1 (WH/BH) target, within the scope of `audit/metal2_audit.md`.

**Recipe docs:** `c41f7f4ef0a 2026-10-29 docs(metal_2.0): DFBAccessor -> DFBBindingToken; drop the dead `.id` form`

**Readiness-sheet provenance:** the launching (main) session fetched the *"Operations analysis"* sheet (file ID `1KUMj8SyBGlNMZlLFgs1MbAZlO2g6EoUc4KaxSlcy8jw`) on 2026-11-03. It used the Google Drive MCP `download_file_content` as `text/csv` and decoded the result into `docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/analyses/ttnn_op_porting_readiness.csv`. This audit (a subagent) read that local copy and did not fetch the sheet. All three factory rows were found (CSV lines 303–305), and columns were resolved by header name.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/slice_write` |
| **Overall** | **RED** |
| **DOps / Factories** | `SliceWriteDeviceOperation` → `SliceWriteRMInterleavedProgramFactory`, `SliceWriteRMShardedInputProgramFactory`, `SliceWriteTiledShardedInputProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes.** No violations in the 3 own kernels or the borrowed `reader_unary_sharded.cpp` |
| *Prereqs* — Cross-op escapes | Ok. One borrowed kernel file, and a `_metal2` fork of it already exists. The kernel `#include` donors are all Device 2.0 shapes |
| *Feature Support* — overall | GREEN (no gate fired) |
| *Feature Support* — Variadic-CTA | Ok (none) |
| *TTNN Readiness* — `Is able to port?` (the gate) | Sheet: `yes (with PD step)` on all 3 rows. **RED: spreadsheet is broken.** The `Concept` column conflicts with the code: the sheet says `legacy device-op`, but all 3 factories are `descriptor`. `Custom hash` and `Override runtime args method?` also conflict. Routed to the readiness-sheet owner |
| *TTNN Readiness* — Concept (current) | Sheet: `legacy device-op`. **Code: `descriptor`** on all 3 factories (`create_descriptor()` returns `ProgramDescriptor`, with no `MeshCoordinate` parameter) |
| *TTNN Readiness* — Execution Model | `SPMD` (sheet). The concept is not mesh-shaped, so `Secretly SPMD Workload?` does not apply |
| *TTNN Readiness* — Custom hash | Sheet: `no`. **Code: Yes.** `SliceWriteDeviceOperation::compute_program_hash` at `device/slice_write_device_operation.cpp:86-97`. Not a gate; the port leaves it intact |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No (matches sheet) |
| *TTNN Readiness* — `override_runtime_arguments` | Sheet: `n/a` on all 3. **Code: Yes** on RMInterleaved (`slice_write_rm_interleaved_program_factory.cpp:342`) and RMSharded (`slice_write_rm_sharded_input_program_factory.cpp:353`). TiledSharded has none. Not a gate |
| *TTNN Readiness* — Pybind `create_descriptor` | No (matches sheet). `slice_write_nanobind.cpp` binds only the composite function |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | Sheet `Porting Target`: `ProgramSpecFactoryConcept` (all 3 rows). This is derived from the stale `Override` column; see Gate detail |
| *Port work* — Offset base pointer | **GATE → ops team + framework/Audrey (Type 2, accessor-fed; flag early)** on `SliceWriteRMShardedInputProgramFactory`, writer RTA[0]. RMInterleaved and TiledSharded are clean |
| *Port work* — Tensor bindings (per binding) | RMInterleaved: `input` Case 1, `output` Case 1 · RMSharded: `input` clean (borrowed DFB), `output` gated (Type 2) · TiledSharded: `input` clean (borrowed DFB), `output` Case 1 |
| *TTNN Readiness* — TensorParameter relaxation | `none` on all 3 rows (clears) |
| *Port work* — TensorAccessor 3rd arg | Drop (Class 2) at 4 accessor sites in 3 kernel files. No site gates |
| *Port work* — CB endpoints | Legal 1P+1C (RMInterleaved `c_0`, RMSharded `c_0`, TiledSharded `c_0`), plus a self-loop on RMInterleaved `c_1` under `LAST_DIM_STRIDED`. `c_1` is already allocated only conditionally on the host side. No multi-binding, no dead CB |

## Result

**RED.** Two gates are blocked:

1. **TTNN factory concept: the readiness sheet is broken for this op.** It is routed to the **readiness-sheet owner** (Diego). The sheet still lists all three factories as `legacy device-op`. The op was ported to `ProgramDescriptor` in `79df5deebb9` (2026-09-30, #57550). The custom hash and the two `override_runtime_arguments` hooks arrived with `caa236078c8` (2026-10-05, #57195). This blocker is cleared **outside the op code**: the sheet needs reconciling, and the op needs no change.
2. **Offset base pointer, Type 2 (accessor-fed), on `SliceWriteRMShardedInputProgramFactory` only.** It is routed to the **ops team + framework/Audrey (flag early)**. The writer's output base folds `+ output_start[-1]·elem_size + width_offset` into the address that feeds a `TensorAccessor`. The fold is non-zero for BLOCK_SHARDED inputs.

**Subset:** `RED at op level; no portable subset today.` Gate 1 covers all three factory rows. Once the sheet is reconciled, and if its re-derived `Is able to port?` then reads `yes`, the subset **`SliceWriteRMInterleavedProgramFactory` + `SliceWriteTiledShardedInputProgramFactory` is clear** of every audit-side gate (Device 2.0, features, offset base pointers, 3rd-arg). `SliceWriteRMShardedInputProgramFactory` would still wait on the ops team. A subset port would have to fork `slice_write_writer_interleaved.cpp`, because it is shared within the op with the gated RMSharded factory. See Heads-ups.

**Scoping call (informational subjects):** I ran them. The primary blocker (gate 1) clears **elsewhere**, so a re-audit would read the same code and today's detail stays valid. Gate 2 is op-code-side but scoped to one factory. The detail below still applies to the RMInterleaved + TiledSharded subset in full. For RMSharded it applies everywhere except the gated output binding.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **RED: spreadsheet is broken.** It is routed to the readiness-sheet owner to reconcile. Below is the cross-check against code at `8d3610f3fdb`, one row per factory (CSV lines 303/304/305):

  | Column (by header) | Sheet (all 3 rows unless noted) | Code | Verdict |
  |---|---|---|---|
  | `Concept` | `legacy device-op` | `create_descriptor()` → `ProgramDescriptor` on all 3 (`rm_interleaved…cpp:199`, `rm_sharded_input…cpp:231`, `tiled_sharded_input…cpp:215`). No `create()` or `MeshCoordinate` overload exists | **Conflict**: the code is `descriptor` |
  | `Custom hash  (compute_program_hash)` | `no` | `compute_program_hash` declared at `device/slice_write_device_operation.hpp:35` and defined at `.cpp:86-97`. It keys RMInterleaved on `factory.index(), slice_start[-1], step, tensor_args`, and the sharded factories on `factory.index(), args, tensor_args`. It was removed in `1ba8d48697f` (2026-06-23) and re-added in `caa236078c8` (2026-10-05) | **Conflict** (not a pybound op, so the rename caveat does not apply) |
  | `Runtime-args update  (get_dynamic_runtime_args)` | `no` | No hook on the device-op | Match |
  | `Override runtime args method? (PD only)` | `n/a` | RMInterleaved: yes (`rm_interleaved…hpp:21`, `.cpp:342-395`). RMSharded: yes (`rm_sharded_input…hpp:19`, `.cpp:353-375`). TiledSharded: none (so it should read `no` for a PD op) | **Conflict** (follows from the stale `Concept`) |
  | `Pybind descriptor  (nb::class_ of device op)` | `no` | `slice_write_nanobind.cpp:51-58` binds only `ttnn::experimental::slice_write` | Match |
  | Factory-set | 3 rows | 3 factories in `program_factory_t` (`slice_write_device_operation.hpp:26-29`) | Match |
  | Cross-column invariants | — | No `get_dynamic_runtime_args` on a legacy row; no op-owned tensors | Hold |

  The primary-column conflicts are the trigger. `Is able to port?` = `yes (with PD step)` would normally route to the legacy-API short-circuit (PD port first). Its premise, that the op is not on the `ProgramDescriptor` API, is contradicted by the code, so I routed this as **spreadsheet-broken**, not as a legacy RED (see Recipe notes). The derived cells `Op Classification` = `Legacy Op` and `Porting Target` = `ProgramSpecFactoryConcept` rest on the stale columns. With `override_runtime_arguments` present on RMInterleaved and RMSharded, I would expect the reconciled `Porting Target` for those two to read `CustomProgramSpecFactoryConcept`. That expectation is a question for the owner, not a re-derivation. `Known op issues` is empty; `TensorParameter relaxation` is `none`; `Diego validation` is `yes`. **Path forward:** the sheet owner updates the three rows, then this op gets a cheap re-audit of this subject only. The rest of this report stays valid.

- **Device 2.0 (every kernel used):** **GREEN.** All four kernels use the Device 2.0 surface:
  - `Noc` objects and `noc.async_read/async_write` with a `TensorAccessor` or `CircularBuffer` endpoint.
  - `CircularBuffer` via the pool alias `experimental::CB`, using its `reserve_back`/`push_back`/`wait_front`/`pop_front`/`get_read_ptr()`/`get_write_ptr()` methods.
  - `DataflowBuffer` in `reader_unary_sharded.cpp`.

  There is no raw `noc_async_*`, no addrgen, no CB-index free function (`get_write_ptr(cb_id)` etc.), and no semaphores. The one donor call, `tt::data_movement::common::tt_memmove(Noc, uint32_t, uint32_t, uint32_t)` (`slice_write_reader_interleaved.cpp:43`), is itself on `Noc` (`data_movement/common/kernels/common.hpp:217`). `get_arg_addr(9)` (writers) is RTA plumbing, not a data-movement idiom; see RTA varargs.

  | File | Line | Call | Wrapper in scope |
  |---|---|---|---|
  | — | — | none | — |

- **Feature compatibility:** GREEN (no gate fired).

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, `.global_circular_buffer` field, or `remote_*` idiom anywhere in the op, its kernels, or `padded_slice/device/slice_cb_descriptor.hpp` |
  | CBDescriptor `address_offset` (non-zero) | N/A | `make_slice_cb_descriptor` (`padded_slice/device/slice_cb_descriptor.hpp:14-31`) never sets `address_offset`. `UpdateDynamicCircularBufferAddress(program, cb->id(), input_buffer)` (`rm_sharded_input…cpp:52`) is the 3-arg, no-offset form |
  | GlobalSemaphore | N/A | No semaphores of any kind |

- **CB endpoints (GATE-free):** every CB is legal or self-loop. The census is in Port-work. There is no multi-binding and no dead CB.

- **Offset base pointers:** **RED on `SliceWriteRMShardedInputProgramFactory`, Type 2 (accessor-fed), routed to ops team + framework/Audrey, flag early.**
  - Arg: **writer RTA[0], output base** (`slice_write_writer_interleaved.cpp:11` `dst_addr`).
  - Host expression: `output_buffer->address() + (output_tensor_start[-1] * output_tensor.element_size()) + width_offset` (`rm_sharded_input…cpp:30-47`). It is emitted at `:198-199` and re-emitted on cache hits by `override_runtime_arguments` at `:371-373`.
  - `width_offset = core_w_index * input_row_size_bytes`, where `core_w_index` is non-zero only for **BLOCK_SHARDED** inputs (`:39-45`). `output_tensor_start[-1]` is `TT_FATAL`'d to 0 (`:138-141`). So a HEIGHT_SHARDED input folds offset 0 (a bare base), and BLOCK_SHARDED folds a real sub-page column offset.
  - Consumption: the folded address is the `TensorAccessor` **base**: `TensorAccessor(dst_args, dst_addr, output_stick_size)` (`slice_write_writer_interleaved.cpp:49`). That makes it Type 2, not Type 1.
  - This matches the dated triage (`2026-07-19_offset_base_pointers.md`, Type 2 table: "BLOCK_SHARDED only; HEIGHT_SHARDED folds offset 0"). The fold is still present, so the doc is not stale for this op.
  - Observation for the ops team (not a prescription): the offset is a byte offset *within* each output page (row). The sibling writer `slice_write_writer_interleaved_strided.cpp:139` already addresses a sub-page start via the per-page `{.page_id, .offset_bytes = page_begins_offset}` NoC argument with a clean base. Any change to `slice_write_writer_interleaved.cpp` also reaches TiledSharded, which shares that kernel.
  - **Clean:**
    - RMInterleaved reader RTA[0] (`src0_buffer` as `Buffer*`, `rm_interleaved…cpp:31-43,321`) and writer RTA[0] (`dst_buffer` as `Buffer*`, `:322`). The override re-patches bare `->address()` at `:392-393`. The last-dim start is already split out as the CTA `begins_bytes` and added kernel-side.
    - TiledSharded writer RTA[0] (`dst_buffer` as `Buffer*`, `tiled_sharded_input…cpp:32-44,330`). Its per-core width offset rides the tile-index `start_id` (`:163,168`), as the triage predicts for tiled variants.
  - Factory-scoped: RMSharded is RED; RMInterleaved and TiledSharded are clear.

- **TensorAccessor 3rd argument:** **GREEN.** Every site is **Class 2** (redundant: correct magnitude on an interleaved accessor). Both questions per site:

  | Site | Accessor | Sharded? | 3rd-arg value (host) | True page | Class |
  |---|---|---|---|---|---|
  | `slice_write_reader_interleaved.cpp:25` (RMInterleaved) | input | interleaved (factory picked only when `!input.is_sharded()`, `slice_write_device_operation.cpp:24-36`) | RTA[1] `stick_size` = `input_padded[-1]·elem` (`rm_interleaved…cpp:66,184`) | RM row = same | 2 |
  | `slice_write_writer_interleaved_strided.cpp:66` (RMInterleaved) | output | interleaved (`slice_write.cpp:38`) | RTA[1] `output_stick_size` = `output_padded[-1]·elem` (`:65,129`) | RM row = same | 2 |
  | `slice_write_writer_interleaved.cpp:49` (RMSharded) | output | interleaved | RTA[1] = `output_logical[-1]·elem` (`rm_sharded_input…cpp:87,145`) | RM row (logical = padded last dim for RM) | 2 |
  | `slice_write_writer_interleaved.cpp:49` (TiledSharded) | output | interleaved | RTA[1] = `tt::tile_size(fmt)` (`tiled_sharded_input…cpp:62,132`); dtype equality is FATAL'd (`:237`) | tile size = same | 2 |

  **Disagreement with the dated triage** (`2026-07-06_tensor_accessor_3rd_arg_triage.md`, which lists `slice_write` as Class 1, dynamic page size). The program hash now includes the full `tensor_args` in **every** factory branch (`slice_write_device_operation.cpp:86-97`). A `Tensor` hashes via its `tensor_spec` (`ttnn/api/ttnn/tensor/tensor.hpp:252-253`), so a cached program never sees a different row width and the page size cannot vary across cache hits. That agrees with the sheet's `TensorParameter relaxation` = `none`. I treat the triage row as stale, so no `dynamic_tensor_shape` is needed. The kernel comments claiming the page size "may be stale on program cache hits" (`slice_write_reader_interleaved.cpp:23-24`, and the same comment in both writers) are out of date.

## Port-work summary  *(mirrors the brief; no brief issued, since the op is RED)*

- **Tensor bindings** (per binding):
  - RMInterleaved `input`: **Case 1**. Buffer* RTA[0] → `TensorAccessor` at `slice_write_reader_interleaved.cpp:25`.
  - RMInterleaved `output`: **Case 1**. Buffer* RTA[0] → `TensorAccessor` at `slice_write_writer_interleaved_strided.cpp:66`. The kernel also reads back output rows through it under `LAST_DIM_STRIDED` (`:91`).
  - RMSharded `input`: **clean**. Borrowed-memory CB `c_0` (`rm_sharded_input…cpp:299-305`); the override refreshes it at `:364`.
  - RMSharded `output`: **gated (Type 2)**. It would become Case 1 once the ops team splits the offset.
  - TiledSharded `input`: **clean**. Borrowed-memory CB `c_0` (`tiled_sharded_input…cpp:289-295`).
  - TiledSharded `output`: **Case 1**. Buffer* RTA[0] → `TensorAccessor` at `slice_write_writer_interleaved.cpp:49`.
  - Op roll-up: ⚠ port work. All are `Buffer*` deliveries, already patched on cache hit today, so they are routine and carry no stale-pointer hazard.
- **TensorParameter relaxation:** `none` (all 3 rows).
- **TensorAccessor 3rd arg:** drop the 3rd constructor argument at `slice_write_reader_interleaved.cpp:25`, `slice_write_writer_interleaved_strided.cpp:66`, and `slice_write_writer_interleaved.cpp:49`. **Only the accessor argument drops. The RTA itself survives as a named arg**, because each kernel also uses it as a transfer size:
  - reader `stick_size` at `:38,47`
  - strided writer `output_stick_size` at `:67,91,92,131,132`
  - writer `output_stick_size` at `:50`
- **CB endpoints** (per `(CB, config)`, per node):
  - RMInterleaved `c_0` (all configs): **1P+1C, legal**. The reader is the FIFO producer (`reserve_back`/`push_back` at `:33,54`; its own `get_write_ptr()` peek under `LAST_DIM` at `:42` is covered by the PRODUCER binding). The strided writer is the FIFO consumer (`wait_front`/`pop_front` at `:109,161`; `get_read_ptr()` peek at `:122`).
  - RMInterleaved `c_1` under `LAST_DIM_STRIDED` (`stride[-1] != 1`): **self-loop**. The strided writer is the sole toucher (`reserve_back`/`push_back`/`wait_front`/`pop_front` at `:80,105,111,165`; `get_read_ptr()` at `:120`), and the reader never touches it. With `stride[-1] == 1` the CB is not allocated (`rm_interleaved…cpp:267-274` already conditions it on the host side), so the DFB spec stays conditional on the same predicate. See Heads-ups for the kernel-side wrinkle.
  - RMSharded `c_0` and TiledSharded `c_0` (borrowed): **1P+1C, legal**. `reader_unary_sharded.cpp` does `push_back` (locked producer); `slice_write_writer_interleaved.cpp` does `wait_front`/`pop_front` (`:58,95`, locked consumer).

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none. I hunted for hidden second writers: no kernel raw-writes a CB it does not FIFO-own, and there are no semaphores.
- **Strided writer constructs `c_1` unconditionally:** `slice_write_writer_interleaved_strided.cpp:71` `experimental::CB cb_out(cb_id_out)` sits outside `#ifdef LAST_DIM_STRIDED`, and its CTA index (`:59`, host `rm_interleaved…cpp:280`) is always emitted. Under legacy, an unused handle to an unallocated CB index is harmless. Under Metal 2.0, a `DataflowBuffer` built from a `dfb::` token for a DFB that isn't declared in that config has no token to bind to. The porter will need to guard the construction under the same `#ifdef`, which is a kernel change outside the pure-syntax whitelist (see Questions).
- **Cross-op / shared kernels:**
  - `reader_unary_sharded.cpp` (eltwise/unary, borrowed by RMSharded + TiledSharded).
    - A **`_metal2` fork already exists** beside it: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded_metal2.cpp`. **Bind it; don't re-fork.** It fits: binding `dfb::in`, named RTA `args::num_tiles_per_core`, and no `#ifdef`s. The slice_write reader RTA[0] (`num_sticks_per_core` / `num_tiles_this_core`) maps onto `num_tiles_per_core`.
    - **Sunset list (not authorization to convert in place).** Remaining legacy binders of the original: slice_write RMSharded + TiledSharded, `data_movement/sharded_partial/sharded_to_interleaved_partial`, and `data_movement/untilize` (`untilize_multi_core_input_and_output_nd_shard_type_and_shard_spec_identical_program_factory.cpp:102`).
  - `slice_write_writer_interleaved.cpp` is **shared within the op** by RMSharded and TiledSharded. A subset port that leaves RMSharded behind takes the intra-op fork rung (`slice_write_writer_interleaved_metal2.cpp` beside it).
  - Host-side `make_slice_cb_descriptor` (`experimental/padded_slice/device/slice_cb_descriptor.hpp`) is shared with `padded_slice`. Stop calling it, but don't delete it (padded_slice still uses it).
- **RTA varargs:**
  - `slice_write_writer_interleaved_strided.cpp:33-36` and `slice_write_writer_interleaved.cpp:40-42` read the per-dim arrays `num_unpadded_sticks`, `num_padded_sticks`, `id_per_dim` (and `rev_stride` in the strided writer) as **one RTA vararg block**: `get_arg_addr(9)`, then pointer arithmetic by `num_dims` (RTA[4], runtime). The block is 3·`num_dims` (writer) or 4·`num_dims` (strided) words. These are genuine RTA varargs, because the count is runtime-varying.
  - **Wrinkle:** both kernels **mutate the block in place** (`id_per_dim[j]++` / `= 0` through a `volatile` pointer into RTA L1: `slice_write_writer_interleaved.cpp:85,87`, strided `:151,153`). `get_vararg(i)` is value-only (`tt_metal/jit_build/genfiles.cpp:577-580`), and no vararg-address accessor exists. The port therefore needs a local copy of at least `id_per_dim`, which is a kernel change beyond the syntax swap (see Questions). The strided writer already makes such a local copy for its read-back (`:84-88`, `max_num_dims = 8`).
  - **Name the trailing scalar:** `padding_width_ntiles` (`slice_write_writer_interleaved.cpp:22`, `get_arg_val(21)`, TiledSharded only under `UNPAD_INPUT_WIDTH`, pushed at `tiled_sharded_input…cpp:204`) is a distinct field after the vararg block. Name it. Its hard-coded index 21 only holds because sharded inputs are rank 4 (`slice_write_device_operation.cpp:80-83`).
  - Fixed-position RTAs 0–8 in all three own kernels, and RTA[0] in the reader, are ordinary named-arg work.
- **Pool convenience header:** all three own kernels include `ttnn/operations/pool/device/kernels/experimental_device_api.hpp`. It supplies the `experimental::CB` = `CircularBuffer` alias **and** is the kernels' only source of `api/dataflow/noc.h` / `endpoints.h`, since `dataflow_api.h` does not include them. Swap `experimental::CB` → `DataflowBuffer`, but **keep the include**, or `Noc` disappears. The `circular_buffer.h` it pulls in lives in pool's header, not the op directory.
- **`override_runtime_arguments` translation:** RMInterleaved's override (`rm_interleaved…cpp:342-395`) re-derives all writer RTAs, because the custom hash keys this factory on `slice_start[-1]` only. The outer-dim start offsets live in RTAs (`start_id`, `id_per_dim`), which the porter must refresh on a hit. RMSharded's override (`rm_sharded_input…cpp:353-375`) only refreshes the CB address and the folded writer base. After the ops-team split, that override may reduce to binding refresh. Its target concept follows the reconciled sheet.

## Team-only

- **Out-of-directory coupling & donor shape:** roll-up **✓ clean**.

  | Op kernel | Donor | Class | Functions used → shape | Status |
  |---|---|---|---|---|
  | all 3 own kernels | `api/dataflow/dataflow_api.h` | tt_metal | — | ✓ |
  | all 3 own kernels | `ttnn/operations/pool/device/kernels/experimental_device_api.hpp` | cross-family (pool) | type alias `experimental::CB` only; no donor function called | ✓ (alias, not a `CircularBuffer&` donor signature) |
  | `slice_write_reader_interleaved.cpp` | `ttnn/operations/data_movement/common/kernels/common.hpp` | cross-family (data_movement) | `tt_memmove<…>(Noc, uint32_t dst_l1, uint32_t src_l1, uint32_t bytes)`: Device 2.0 `Noc` + raw L1 addresses; no resource-handle shape | ✓ |
  | `reader_unary_sharded.cpp` (borrowed) | `api/dataflow/dataflow_api.h`, `dataflow_buffer.h`, `api/debug/dprint.h` | tt_metal | — | ✓ |

  Borrowed kernel files: `eltwise/unary/.../reader_unary_sharded.cpp`. It is broadly shared, its `_metal2` fork exists, and it has 9 non-quasar fork binders (e.g. `transpose_wh_sharded`, `sharded_to_interleaved`, `tilize` sharded, `untilize` multi-core, `typecast` sharded). The legacy sunset list is in Heads-ups.

  Host-side out-of-directory helpers (not kernel coupling):
  - `padded_slice/device/slice_cb_descriptor.hpp`
  - `padded_slice/device/padded_slice_utils.hpp` (`get_num_cores_channels_from_sharded_tensor`)
  - `data_movement/slice/device/slice_device_operation.hpp` (`get_rm_start_offset`, `get_tiled_start_offset`)
- **Relaxation candidates:** none. The custom hash keeps the full `tensor_args` in every branch, so it shows no reduced tensor-property dependence.
- **TTNN factory analysis:**
  - Current concept: `descriptor` (all 3).
  - Op-owned tensors: none.
  - Execution model: SPMD.
  - Pybind `create_descriptor`: none.
  - Custom hash: `slice_write_device_operation.cpp:86-97`.
  - `get_dynamic_runtime_args`: none.
  - `override_runtime_arguments`: RMInterleaved `.cpp:342` and RMSharded `.cpp:353`.
  - Target concept per sheet: `ProgramSpecFactoryConcept`, pending reconciliation.

## Misc anomalies  *(team-only, non-gating)*

- `slice_write_rm_interleaved_program_factory.cpp:102-118`: four debug strings (`unpadded_sticks_str`, `padded_sticks_str`, `accumulated_str`, `rev_stride_str`) are built on every call (including every cache hit via the override) and never logged. This is dead work.
- `slice_write_rm_interleaved_program_factory.cpp:182-190`: reader RTA[7] (`0`) is never read; the reader reads RTAs 0–6 only. This is a dead RTA.
- `slice_write_tiled_sharded_input_program_factory.cpp:279`: `num_tiles_channel_per_core = shard_spec.shape[1] / TILE_HEIGHT` divides a **width** by `TILE_HEIGHT`. It is harmless today, since both are 32.
- `slice_write_writer_interleaved.cpp:22`: `get_arg_val(21)` hard-codes the offset `9 + 3·num_dims` for rank 4. It is correct only because sharded inputs are validated as rank 4.
- `SliceWriteRMShardedInputProgramFactory` never checks the output layout. A sharded RM input paired with a TILE interleaved output reaches it: the composite's `rm_only_not_sharded` (`slice_write.cpp:31-32`) is false whenever the input is sharded. It would then write row sticks into a tiled buffer. `validate_on_program_cache_miss` has no output-layout check. The tiled factory checks this at `:247-254`.
- Kernel comments about a page size "stale on program cache hits" are obsolete (see the 3rd-arg detail).
- Comment at `rm_interleaved…cpp:174` / `rm_sharded_input…cpp:215` / `tiled_sharded_input…cpp:198` (`addr_offset = 5; // output buffer addr, output_row_size_bytes, …`) misdescribes the slots being written (start_id, counts).

## Per-DeviceOperation attribution

There is a single DeviceOperation. Per-factory status:

| Factory | Sheet gate | Offset base ptr | 3rd arg | Device 2.0 | Features | Net |
|---|---|---|---|---|---|---|
| `SliceWriteRMInterleavedProgramFactory` | RED (sheet broken) | clean | Class 2 (drop) | ✓ | N/A | RED: clears with the sheet fix |
| `SliceWriteRMShardedInputProgramFactory` | RED (sheet broken) | **RED: Type 2** | Class 2 (drop) | ✓ | N/A | RED: sheet fix **and** ops-team split |
| `SliceWriteTiledShardedInputProgramFactory` | RED (sheet broken) | clean | Class 2 (drop) | ✓ | N/A | RED: clears with the sheet fix |

## Questions for the user

1. **Reconcile the readiness sheet (owner: Diego).** For rows 303–305 (`experimental/slice_write`):
   - `Concept` → `descriptor`.
   - `Custom hash` → `yes`.
   - `Override runtime args method?` → `yes` for RMInterleaved and RMSharded, `no` for TiledSharded.
   - Re-derive `Is able to port?` and `Porting Target`. I would expect `CustomProgramSpecFactoryConcept` for the two factories with overrides, but that is the owner's call.

   Evidence: commits `79df5deebb9` (PD port, 2026-09-30) and `caa236078c8` (hash + overrides, 2026-10-05).
2. **`Smuggled pointer` column, RMSharded.** The RMSharded writer RTA[0] is a raw `uint32_t` address, not a `Buffer*` (`rm_sharded_input…cpp:198-199`). The factory's own `override_runtime_arguments` (`:371-373`) refreshes it, so it is not stale on cache hits. The sheet reads `no`. Does the sheet's definition intend to exclude override-patched raw addresses? This is not a cross-check trigger; I am raising it only for consistency.
3. **Kernel-side edits the port will need that the whitelist does not obviously cover.** Both are under Heads-ups. May the porter make them, or should they land on the ops track first?
   - (a) Guard `experimental::CB cb_out(cb_id_out)` in `slice_write_writer_interleaved_strided.cpp:71` under `#ifdef LAST_DIM_STRIDED`, so the `dfb::` token exists only when `c_1` is declared.
   - (b) Replace in-place mutation of the RTA-resident `id_per_dim` (and the pointer-walked arrays) with a local copy, because `get_vararg` is value-only.
4. **Subset appetite.** After the sheet fix, would you take a RMInterleaved + TiledSharded subset port (with an intra-op `_metal2` fork of `slice_write_writer_interleaved.cpp`)? Or would you rather wait for the ops-team Type-2 split on RMSharded and port all three together?

## Recipe notes

- **Legacy short-circuit vs spreadsheet-broken precedence.** When `Is able to port?` = `yes (with PD step)` but the code is already `descriptor`, the Routing list offers two applicable rows: "`yes (with PD step)` → GATE … apply the legacy-API short-circuit" and "A primary-column conflict … → spreadsheet is broken". The recipe doesn't say which wins. I chose spreadsheet-broken, because the short-circuit's premise (op not on PD) is contradicted by code evidence. That choice also decided whether the seven informational subjects run: the short-circuit skips them, but "clears elsewhere" runs them. A one-line precedence rule would help, since recently PD-migrated ops will hit this until the sheet catches up.
- **Subagent fetch rule vs this launch.** The recipe says "Do not fetch or cross-check in a subagent … the cross-check needs the op's code in hand." Here the main session fetched the sheet and this subagent did the cross-check with the code in hand, which seems to meet the intent. The wording reads as forbidding the cross-check in a subagent outright.
- **RTA varargs: in-place mutation of the arg region.** The subject covers *reading* varargs. It doesn't anticipate kernels that take `get_arg_addr(n)` pointers into the RTA block and **write** through them (scratch state). Since `get_vararg(i)` is value-only and there is no vararg-address accessor, such kernels need a local copy at port time. That falls outside the kernel-side whitelist, and the recipe gives no ruling.
- **Unconditionally-constructed CB handle for a conditionally-allocated CB.** The CB endpoints census counts touchers, and a constructor-only handle with no access isn't one. But the construction alone breaks under Metal 2.0's token model when the DFB is conditional. It might deserve a recognition line (in CB endpoints, or under "Anything else").
- **CircularBuffer type and `noc.h` reaching a kernel via a cross-family convenience header** (`pool/.../experimental_device_api.hpp`, included by 38 kernels). The whitelist's "port adds exactly two headers; the now-unused `CircularBuffer` include drops" assumes a direct `circular_buffer.h` include. A note that such convenience headers stay, because they carry `noc.h`, would save porters a wrong turn.
- **3rd-arg triage vs relaxation column.** The dated triage lists `slice_write` as Class 1 (needs `dynamic_tensor_shape`), while the sheet's relaxation column says `none`. Had I followed the triage, the Class 1 → relaxation cross-reference would have collided with a forbidden self-derived relaxation. My own read (the hash includes the full tensor specs, so the page size is fixed per cached program) resolves it to Class 2. A sentence noting that a Class-1 triage row on an op with relaxation `none` should be re-checked against the current hash key would make this explicit.
