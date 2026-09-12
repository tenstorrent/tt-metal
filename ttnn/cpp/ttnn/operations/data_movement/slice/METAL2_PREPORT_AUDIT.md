# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/data_movement/slice`

One device operation, five program factories:

- **`ttnn::prim::SliceDeviceOperation`**
  - `SliceRmProgramFactory` (`device/slice_program_factory_rm.cpp`)
  - `SliceRmShardedProgramFactory` (`device/slice_program_factory_rm_sharded.cpp`)
  - `SliceRmStrideProgramFactory` (`device/slice_program_factory_rm_stride.cpp`)
  - `SliceTileProgramFactory` (`device/slice_program_factory_tile.cpp`)
  - `SliceTileTensorArgsProgramFactory` (`device/slice_program_factory_tile_tensor_args.cpp`)

Ten kernel files are referenced by these factories, nine of them slice-owned and one borrowed from `eltwise/unary`. Two kernel files in the op's directory are referenced by **no** factory and are therefore out of scope; they are listed under *Misc anomalies* so a reader does not mistake them for live code.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `metal2_audit.md`.

**Recipe docs:** `c07a9a48c61 2026-09-12 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/data_movement/slice` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `SliceDeviceOperation` → `SliceRmProgramFactory`, `SliceRmShardedProgramFactory`, `SliceRmStrideProgramFactory`, `SliceTileProgramFactory`, `SliceTileTensorArgsProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes** — all ten referenced kernels plus the one donor kernel are Device 2.0 throughout |
| *Prereqs* — Cross-op escapes | Ok — one in-family header (`data_movement/common/kernels/common.hpp`), all handle shapes ✓ |
| *Feature Support* — overall | **GREEN** (every Appendix A entry N/A) |
| *Feature Support* — Variadic-CTA | Ok — no kernel reads a compile-time arg at a varying index (see *Recipe notes* about this row) |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes**, on all five factory rows |
| *TTNN Readiness* — Concept (current) | `descriptor` (all five rows) |
| *TTNN Readiness* — Secretly SPMD (WorkloadDescriptor only) | N/A — concept is `descriptor`, not `WorkloadDescriptor` |
| *TTNN Readiness* — Custom hash | Yes (not a gate; port leaves it intact): `device/slice_device_operation.cpp:348` |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No — the device operation declares no such hook |
| *TTNN Readiness* — `override_runtime_arguments` | Yes (not a gate; selects `CustomProgramSpecFactoryConcept`): one per factory, all delegating to `patch_slice_program_addresses` (`device/slice_program_factory_rm.cpp:425`, `device/slice_program_factory_rm_sharded.cpp:418`, `device/slice_program_factory_rm_stride.cpp:178`, `device/slice_program_factory_tile.cpp:189`, `device/slice_program_factory_tile_tensor_args.cpp:195`) |
| *TTNN Readiness* — Pybind `create_descriptor` | Yes (not a gate; port deletes the binding): `slice_nanobind.cpp:167-179` |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | **`CustomProgramSpecFactoryConcept`** (all five factories) |
| *Port work* — Offset base pointer | none — every address argument is a clean base |
| *Port work* — Tensor bindings (per binding) | 10 × Case 1 · 2 × clean (borrowed-memory DFB) · 0 × Case 2 |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) on all five rows |
| *Port work* — TensorAccessor 3rd arg | none — no accessor in the op passes a 3rd argument |
| *Port work* — CB endpoints | 4 CBs plain 1:1 · 3 CBs self-loop · no dead CB · no multi-binding · no conditional DFB |

**CB endpoints** are dispositions, not gates. Every out-of-window CB here is a one-toucher single-ended case resolved by a self-loop; nothing needs the multi-binding advanced option and nothing is dropped.

## Result

**GREEN → brief issued.** All five gate-bearing subjects clear. The porter brief is at `METAL2_PORT_BRIEF.md` in this directory.

Two dated triage analyses list slice as blocked; **both are stale for this op** and my own scan is the source of truth per the recipe's contract for those documents. The ops team has already done the refactors each one asked for:

- The **offset-base-pointer** triage (`analyses/2026-07-19_offset_base_pointers.md`, line 63) lists `slice_program_factory_rm.cpp` reader RTA[0] as a **Type 2** fold, `input->address() + begins_bytes − misalignment`. That fold no longer exists. The factory now emits a bare `Buffer*` binding as the accessor base and passes `begins_bytes − misalignment` as its own scalar argument, which the kernel applies per read. This is the recipe's third reconciliation outcome (*no fold, op in the tables → the doc is stale → GREEN*).
- The **TensorAccessor 3rd-argument** triage (`analyses/2026-07-06_tensor_accessor_3rd_arg_triage.md`, lines 75 and 139) lists slice as Class 1 with a Special sub-page-base-offset note. No accessor anywhere in the op passes a 3rd argument today, and the sub-page offset rides the read call rather than the accessor's base.

Both documents should be corrected by their owner; details in *Gate detail* below.

## Gate detail

- **TTNN factory concept (`Is able to port?`): GREEN.** The readiness sheet (fetched fresh this session) carries five `data_movement/slice` rows, one per factory, and all five read `Is able to port?` == `yes`. The lightweight cross-check came back clean on every column:

  | Column | Sheet value | Code evidence |
  |---|---|---|
  | `Concept` | `descriptor` (×5) | each factory defines `create_descriptor(...)` returning a `tt::tt_metal::ProgramDescriptor` — e.g. `device/slice_program_factory_rm.hpp:26-27` |
  | `Custom hash (compute_program_hash)` | `yes` (×5) | `SliceDeviceOperation::compute_program_hash` at `device/slice_device_operation.cpp:348-432` |
  | `Backdoor custom hash` | `no` (×5) | no `attribute_values` / `to_hash` anywhere in the op |
  | `Runtime-args update (get_dynamic_runtime_args)` | `no` (×5) | no such hook on the device operation (`device/slice_device_operation.hpp:31-55`) |
  | `Override runtime args method? (PD only)` | `yes` (×5) | one `override_runtime_arguments` per factory, sites in the status table above |
  | `Pybind descriptor` | `PR` (×5) | `SliceTileProgramFactory::create_descriptor` is pybound at `slice_nanobind.cpp:167-179`; `SliceParams`, `SliceInputs` and `SliceDeviceOperation` are also exposed at `slice_nanobind.cpp:138-166` |
  | `Smuggled pointer` | `no` (×5) | confirmed — see the *Buffer\*-binding form* note under Tensor bindings |
  | `Op-owned tensors?` | blank (×5) | `create_descriptor` returns a `ProgramDescriptor`, which cannot carry them |
  | `Known op issues` | blank (×5) | — |
  | Factory-set match | 5 sheet rows | 5 alternatives in `SliceDeviceOperation::program_factory_t` (`device/slice_device_operation.hpp:36-41`); one-to-one, no phantom and no missing row |

  Cross-column invariants hold: `get_dynamic_runtime_args` is `no` throughout, and no `descriptor` row claims op-owned tensors.

  One naming trap worth recording, because it looks like a `get_dynamic_runtime_args` hit and is not: `slice_tile_dynamic_args` (`device/slice_program_factory_tile.hpp:30-36`, defined at `device/slice_program_factory_tile.cpp:198`) returns a `std::vector<tt::tt_metal::DynamicRuntimeArg>`. It is a plain free function that the factories' own `override_runtime_arguments` calls; it is not the deprecated device-operation hook, and the sheet's `no` is correct.

- **Device 2.0 (every kernel used): GREEN.** Every kernel the op instantiates is structurally Device 2.0: `Noc` for all transfers, `DataflowBuffer` objects for every circular buffer, `CoreLocalMem` / `UnicastEndpoint` where a raw local address is addressed, and `TensorAccessor` for tensor memory. Every `get_write_ptr()` / `get_read_ptr()` in the op is a **method call on a `DataflowBuffer` object**, not a CB-index free function, so there are no isolated holdovers either.

  The one free function taking a CB index in the whole set is in the borrowed donor kernel:

  | File | Line | Call | Wrapper in scope |
  |---|---|---|---|
  | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | 27 | `get_local_cb_interface(cb_id_out).fifo_page_size` | `DataflowBuffer dfb(cb_id_out)` constructed at line 30 |

  `get_local_cb_interface(cb_id)` is on the recipe's **sanctioned** list, and the list is the whole test regardless of what object is in scope. This is **not** a violation and does not gate. (At port time the equivalent lookup moves onto the object, as slice's own copy of that kernel already does — `device/kernels/dataflow/writer_unary_interleaved_start_id.cpp:26` uses `dfb_out.get_entry_size()`.)

  No kernel uses `InterleavedAddrGen`, `ShardedAddrGen`, `InterleavedAddrGenFast`, `InterleavedPow2AddrGen*`, raw `noc_async_read` / `noc_async_write`, raw semaphore addresses, or `cb_reserve_back` / `cb_push_back` / `cb_wait_front` / `cb_pop_front`.

- **Feature compatibility: GREEN.** Every Appendix A entry is absent from the op.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, no `CreateGlobalCircularBuffer`, no `global_circular_buffer` field on any `CBDescriptor`, no `remote_index` / `remote_cb` / `remote_circular_buffer.h` idiom anywhere in the op. The seven `CBDescriptor` literals across the five factories set only `total_size`, `core_ranges`, `format_descriptors` and (in the sharded factory) `buffer`. |
  | CBDescriptor `address_offset` (non-zero) | N/A | No `CBDescriptor` in the op sets `address_offset`; no `set_address_offset`, no `UpdateDynamicCircularBufferAddress` of either arity, no `cb_descriptor_from_sharded_tensor` call. The two Buffer-backed CBs in `SliceRmShardedProgramFactory` (`device/slice_program_factory_rm_sharded.cpp:291-312`) are the plain borrowed-memory pattern at offset zero, which is a mechanical porting-recipe translation via `DataflowBufferSpec::borrowed_from`, not this entry. |
  | GlobalSemaphore | N/A | The op contains no semaphore of any kind — the string `semaphore` does not appear in any file under the op directory. |

- **CB endpoints (GATE-free): all resolved.** Seven CBs across the five factories; the factory selection is the config axis, and within a factory no census flips, so each row below is a complete `(CB, config)` disposition. No semaphores exist in the op, so the hidden-second-writer face cannot apply; I also found no `evil_set_write_ptr` / `evil_set_read_ptr` cursor driver and no kernel raw-writing a CB it does not also bind.

  | Factory | CB | Touchers on a node | Verdict | Resolution |
  |---|---|---|---|---|
  | `SliceRmProgramFactory` | `src0_cb_index` = `c_0` | reader `reserve_back`/`push_back` (locked producer) + writer `wait_front`/`pop_front` (locked consumer) | plain 1:1 | legal — bind PRODUCER + CONSUMER as they are |
  | `SliceRmShardedProgramFactory` | `src0_cb_index` = `c_0`, borrowed from `input.buffer()` | reader only, and only a raw peek (`dfb_in.get_write_ptr()`, `device/kernels/dataflow/slice_reader_unary_unpad_dims_rm_sharded.cpp:41`) — role-free | single-ended / sync-free | **self-loop** |
  | `SliceRmShardedProgramFactory` | `c_16`, borrowed from `output.buffer()` | reader only (`reserve_back`/`push_back` plus a raw peek) — locked producer, one toucher | single-ended | **self-loop** |
  | `SliceRmStrideProgramFactory` | `in_cb` = `c_0` | reader producer + writer consumer | plain 1:1 | legal |
  | `SliceTileProgramFactory` | `src0_cb_index` = `c_0` | reader producer + writer consumer | plain 1:1 | legal |
  | `SliceTileTensorArgsProgramFactory` | `src0_cb_index` = `c_0` | reader producer + donor writer consumer | plain 1:1 | legal |
  | `SliceTileTensorArgsProgramFactory` | `tensor_cb_index` = `c_1` | reader only, running the full `reserve_back` → `push_back` → `wait_front` → `pop_front` handshake against itself twice (`device/kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp:52-83`) | single-ended | **self-loop** |

  `SliceRmShardedProgramFactory` builds only one kernel, so its two CBs cannot have a second toucher; I confirmed the factory's `desc.kernels` receives just `reader_desc` (`device/slice_program_factory_rm_sharded.cpp:348`). No zero-endpoint CB exists: every allocated `buffer_index` is referenced by a bound kernel, either through a compile-time argument, a named compile-time argument, or a hardcoded `constexpr` in the kernel.

- **Offset base pointers: GREEN.** I resolved every address argument in the op back to its host computation. All of them are bare buffer bases delivered as `Buffer*` bindings; none folds host arithmetic into the address.

  The one site the triage document names deserves its own paragraph, because the fix is what makes this row green. `SliceRmProgramFactory` pushes `src0_buffer` unmodified as reader argument 0 (`device/slice_program_factory_rm.cpp:406`) and `dst_buffer` unmodified as writer argument 0 (`device/slice_program_factory_rm.cpp:414`). The W-begin byte shift now travels as a separate scalar, `begins_bytes - misalignment`, built at `device/slice_program_factory_rm.cpp:99` and read by the kernel as `src_offset_bytes` at `device/kernels/dataflow/slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp:29`. The kernel constructs its accessor on the unshifted base (`...:40`) and hands the shift to each read as the `offset` argument of `noc_async_read_sharded` (`...:67` and `...:98`). The in-file comment at `...:38-39` states the invariant the split exists to satisfy. That is exactly the shape the offset-base wall requires, so slice's row in `analyses/2026-07-19_offset_base_pointers.md` is stale and should be removed or marked resolved by that document's owner.

  The other three RM factories are equally clean: `SliceRmStrideProgramFactory` pushes `input_buffer` / `output_buffer` verbatim (`device/slice_program_factory_rm_stride.cpp:128`, `:136`, `:147`, `:160`), and the two tile factories pass a **tile-index scalar** (`start_id`) computed by `get_tiled_start_offset`, not an address (`device/slice_program_factory_tile.cpp:97`, `:125`). The five `->address()` calls in the op all sit inside `patch_slice_program_addresses` (`device/slice_program_factory_rm_sharded.cpp:384`, `:392`, `:398`, `:401`, `:402`) and are plain bases on the cache-hit path. Neither `ttnn::narrow` nor a `MeshBuffer::create(..., parent_base + offset)` interior-base view appears anywhere in the op, so Type 4 does not arise either.

- **TensorAccessor 3rd argument: N/A.** No accessor in the op passes a 3rd argument — all fourteen `TensorAccessor(...)` constructions in the slice directory, and the one in the borrowed donor kernel, use the two-argument `(args, addr)` form. The subject never fires, so there is nothing to classify.

  This contradicts `analyses/2026-07-06_tensor_accessor_3rd_arg_triage.md`, which lists `slice` (interleaved RM path) as Class 1 with a Special sub-page-base-offset note. The ops team has since refactored both halves. The page size now comes from the compile-time word that `TensorAccessorArgs` bakes in, and the host pins the equivalence with `check_accessor_page_size` (`device/slice_program_factory_rm.cpp:292-308`, called at `:337-340`), which fails loudly rather than striding by a wrong page when a caller reaches the factory without going through `ttnn::slice`'s resharding guard. The reasoning is written out at `device/slice_program_factory_rm.cpp:283-291`. The Special sub-page-base-offset concern is the same thing resolved by the offset split described in the previous bullet: the byte offset now rides each read's `offset_bytes` field, which the binding model expresses directly. That document's owner should update slice's rows.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding, per factory). Every one of the ten non-clean bindings is **Case 1** — the kernel feeds the base straight into a `TensorAccessor` and does all its addressing through it. There is no Case 2 anywhere in the op: no kernel does hand-rolled bank arithmetic on a tensor base.

  | Factory | Binding | Delivery today | Case |
  |---|---|---|---|
  | `SliceRmProgramFactory` | `input` | reader RTA 0, `Buffer*` binding (`device/slice_program_factory_rm.cpp:406`) | **Case 1** |
  | `SliceRmProgramFactory` | `output` | writer RTA 0, `Buffer*` binding (`device/slice_program_factory_rm.cpp:414`) | **Case 1** |
  | `SliceRmShardedProgramFactory` | `input` | borrowed-memory DFB, `CBDescriptor::buffer` (`device/slice_program_factory_rm_sharded.cpp:299`) | **clean** |
  | `SliceRmShardedProgramFactory` | `output` | borrowed-memory DFB, `CBDescriptor::buffer` (`device/slice_program_factory_rm_sharded.cpp:311`) | **clean** |
  | `SliceRmStrideProgramFactory` | `input` | reader RTA 0, `Buffer*` binding (`device/slice_program_factory_rm_stride.cpp:128`, `:147`) | **Case 1** |
  | `SliceRmStrideProgramFactory` | `output` | writer RTA 0, `Buffer*` binding (`device/slice_program_factory_rm_stride.cpp:136`, `:160`) | **Case 1** |
  | `SliceTileProgramFactory` | `input` | reader **CRTA** 0, `Buffer*` binding (`device/slice_program_factory_tile.cpp:143`) | **Case 1** |
  | `SliceTileProgramFactory` | `output` | writer RTA 0, `Buffer*` binding (`device/slice_program_factory_tile.cpp:180`) | **Case 1** |
  | `SliceTileTensorArgsProgramFactory` | `input` | reader CRTA 0, `Buffer*` binding (`device/slice_program_factory_tile_tensor_args.cpp:182`) | **Case 1** |
  | `SliceTileTensorArgsProgramFactory` | `start_tensor` | reader CRTA 1, `Buffer*` binding (`device/slice_program_factory_tile_tensor_args.cpp:183`) | **Case 1** |
  | `SliceTileTensorArgsProgramFactory` | `end_tensor` | reader CRTA 2, `Buffer*` binding (`device/slice_program_factory_tile_tensor_args.cpp:184`) | **Case 1** |
  | `SliceTileTensorArgsProgramFactory` | `output` | writer RTA 0, `Buffer*` binding (`device/slice_program_factory_tile_tensor_args.cpp:151`, `:168`) | **Case 1** |

  **All ten Case-1 bindings arrive in the `Buffer*`-binding form**, not as an `->address()` expression in an argument list. That form is the framework's interim hack: it registers a `BufferBinding` that is patched on cache hits, so it is **correct today** and is not the silent-wrong hazard the subject exists to catch. Enumerated here because each one still becomes a `TensorParameter` / `TensorBinding` in the port, and the kernel-side `TensorAccessorArgs<N>()` plus the raw `uint32_t` base both disappear with it. Op-level roll-up: **⚠ port work**.

- **TensorParameter relaxation:** `none` — the sheet reads `none` on all five factory rows, so the port applies no relaxation and no analysis document is required.

- **TensorAccessor 3rd arg:** none — no accessor in the op passes one.

- **CB endpoints:** self-loop `SliceRmShardedProgramFactory / c_0`, `SliceRmShardedProgramFactory / c_16`, `SliceTileTensorArgsProgramFactory / c_1`. All four remaining CBs are plain 1:1 and need no action. No dead-CB drop, no multi-binding flag, no conditional DFB.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none. No CB in the op has more than two touchers on a node, and no two kernels lock the same FIFO role.

- **Cross-op / shared kernels.** One borrowed kernel file: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`, instantiated by `SliceTileTensorArgsProgramFactory` (`device/slice_program_factory_tile_tensor_args.cpp:133`). It is broadly shared — thirteen other factories instantiate the same file. **A `_metal2` fork already exists beside it**, at `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp`, outside `experimental/quasar/`, so this port binds the existing fork rather than creating one. The donor file's own header comment (lines 5-11) points at issue #52228 for the consumer list and sunset plan.

- **A host-side coupling the two escape categories do not cover — read this one.** `ttnn/cpp/ttnn/operations/ccl/mesh_partition/device/mesh_partition_program_factory.cpp` drives slice's factories directly rather than going through `ttnn::prim::slice`. It calls `SliceOp::validate_on_program_cache_miss` and `SliceOp::select_program_factory` at `:126-127`, `Factory::create_descriptor` at `:131`, and `ttnn::prim::patch_slice_program_addresses` at `:155`, and it stores a `SliceDeviceOperation::program_factory_t` in its own `shared_variables_t`. A Metal 2.0 port changes every one of those entry points, so MeshPartition has to move in the same change or it will not compile. `device/slice_device_operation.hpp:71-78` records the shared-slot-layout intent behind `patch_slice_program_addresses`, and `device/slice_program_factory_rm.cpp:290-291` notes that MeshPartition bypasses `ttnn::slice`'s resharding guard.

- **The pybound `create_descriptor` is a user-visible deletion.** `slice_nanobind.cpp:167-179` exposes `SliceTileProgramFactory.create_descriptor` to Python, alongside `SliceParams`, `SliceInputs` and two `SliceDeviceOperation` statics at `:138-166`. The port deletes the `create_descriptor` binding; the sheet's `Pybind descriptor` cell reads `PR`, which suggests the removal is already in flight. The other bindings in `bind_slice_descriptor` do not touch a factory entry point and are unaffected.

- **RTA and CRTA varargs.** Six of the ten referenced kernels read a genuinely variable-count block and must use the vararg mechanism rather than named arguments. In each case the scalars **before** the block are ordinary named arguments; only the counted block is a vararg.

  | Kernel | Site | Block(s) | Count driven by |
  |---|---|---|---|
  | `slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp` | `:31-33` | `num_unpadded_sticks`, `num_padded_sticks`, `id_per_dim` — three `num_dims`-long RTA blocks, indexed by the loop variable at `:76-84` and `:106-114` | RTA 3 (`num_dims`) |
  | `slice_reader_unary_unpad_dims_rm_sharded.cpp` | `:26-30` | `read_noc_x` / `read_noc_y` (interleaved, 2 per source core), `num_stick_chunks`, then `chunk_start_id` / `chunk_num_sticks` (2 per chunk) | RTA 0 (`num_cores_read`), and a per-core chunk count that is itself runtime data |
  | `reader_unary_unpad_dims_interleaved_start_id.cpp` | `:17-18` (CRTA), `:23` (RTA) | CRTA: `num_unpadded_tiles` + `num_padded_tiles`, each `num_dims` long. RTA: `id_per_dim`, `num_dims` long | CTA 0 (`num_dims`) |
  | `reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp` | `:25-26`, `:91-92` (CRTA), `:31` (RTA) | CRTA: `num_unpadded_tiles`, `num_padded_tiles`, `input_shape_args` — three `num_dims`-long blocks. RTA: `id_per_dim` | CTA 2 (`num_dims`) |
  | `reader_multicore_slice_nd.cpp` | `:73-87` | five `tensor_rank`-long RTA blocks: `input_dims`, `output_dims`, `slice_starts`, `slice_ends`, `slice_steps` | RTA 1 (`tensor_rank`) |
  | `writer_multicore_slice_nd.cpp` | `:73` | one `tensor_rank`-long RTA block: `output_dims` | RTA 1 (`tensor_rank`) |

  A CTA-bounded count is still a vararg — the block's length varies across instantiations, so there is no stable per-argument name to attach.

  The four kernels **not** in the table read a fixed set of arguments and get names: `slice_writer_unary_stick_layout_interleaved_start_id.cpp` (ten constant indices), `writer_unary_interleaved_start_id.cpp` in both its slice-owned and its borrowed copy (three constant indices), and the two 4D stride kernels `reader_multicore_slice_4d.cpp` / `writer_multicore_slice_4d.cpp`. The last two use a running `rt_args_idx++` at the top of the kernel, which is a fixed run of distinct fields, not a loop — the recipe's explicit non-signal. Name each of those.

- **No CTA varargs.** Every `get_compile_time_arg_val` in the op and in the donor kernel uses a literal constant index. Nothing here needs `KernelAdvancedOptions::compile_time_varargs`.

## Team-only

### Out-of-directory coupling & donor shape

**Op-level roll-up: ✓ clean.** Every kernel include resolves either into `tt_metal/*` (the `api/dataflow/*`, `api/tensor/*`, `api/core_local_mem.h` headers — donor class 1, no concern) or to a single in-family shared header. No cross-family function-call escape exists, and the only `uint32_t`-shaped handle in any donor signature is a plain L1 address, not a semaphore or CB identifier.

**Summary table** — one row per (op kernel, donor file):

| Op kernel | Donor file | Donor class | Status |
|---|---|---|---|
| `reader_multicore_slice_4d.cpp` | `ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp` | 5 — in-family shared | ✓ |
| `reader_multicore_slice_nd.cpp` | same | 5 | ✓ |
| `writer_multicore_slice_4d.cpp` | same | 5 | ✓ |
| `writer_multicore_slice_nd.cpp` | same | 5 | ✓ |
| `slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp` | same | 5 | ✓ |
| `slice_writer_unary_stick_layout_interleaved_start_id.cpp` | same | 5 | ✓ |
| `reader_unary_unpad_dims_interleaved_start_id.cpp` | *(none)* | — | ✓ |
| `reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp` | *(none)* | — | ✓ |
| `slice_reader_unary_unpad_dims_rm_sharded.cpp` | *(none)* | — | ✓ |
| `writer_unary_interleaved_start_id.cpp` (slice-owned) | *(none)* | — | ✓ |
| `eltwise/unary/.../writer_unary_interleaved_start_id.cpp` (borrowed, see below) | *(none)* | — | ✓ |

**Per-call detail.** Three public functions are called across the whole op, all from the one in-family header:

| Function | Signature shape | Verdict |
|---|---|---|
| `noc_async_read_sharded(Noc, uint32_t l1_addr, AddrGenType tensor, uint32_t src_id, uint32_t offset, uint32_t size)` (`common.hpp:394`) | `AddrGenType` is instantiated with `TensorAccessor<DSpec>` at every call site — **Shape 1** | ✓ excellent — the porter constructs `TensorAccessor(tensor::name)` and passes it |
| `noc_async_write_sharded(Noc, uint32_t l1_addr, AddrGenType tensor, uint32_t dest_id, uint32_t offset, uint32_t size)` (`common.hpp:344`) | same — **Shape 1** | ✓ excellent |
| `tt_memmove<guaranteed_16B_aligned, copy_async, use_read_datamover, max_transfer_size>(Noc, uint32_t dst_l1_addr, uint32_t src_l1_addr, uint32_t bytes)` (`common.hpp:162`) | plain L1 addresses, no resource handle | ✓ — nothing to bridge |

  Every call site uses the **leading-`Noc`** overload; none uses the `[[deprecated]]` no-`Noc` forms at `common.hpp:229`, `:383` and `:432`. No donor function takes a `uint32_t sem_id`, a semaphore address, a `TensorAccessorArgs<N>`, a tensor CTA offset as a template parameter, an old-style addr-gen, a `CircularBuffer`, or a `DataflowBuffer` — so none of the table's ⚠ / ✗ / ⭐ rows fire.

**Borrowed kernel files (file-path instantiation).** Exactly one:

| Kernel file | Owning family | Also instantiated by | `_metal2` fork beside it? |
|---|---|---|---|
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | `eltwise/unary` (shared writer pool) | broadly shared — thirteen other factories, including `data_movement/concat`, `data_movement/reshape_on_device`, four `data_movement/tilize` factories, `embedding`, `copy/typecast`, `eltwise/unary_backward/tanh_bw`, `experimental/matmul/attn_matmul`, `experimental/transformer/nlp_concat_heads_boltz`, and the two `examples/example` factories | **Yes** — `writer_unary_interleaved_start_id_metal2.cpp`, same directory, outside `experimental/quasar/` |

  The consumer list is a **sunset list**, not an authorization to convert the donor file in place. The fork already exists, so this port binds it and adds slice to the set of ops whose migration eventually retires the legacy copy (issue #52228).

  Slice's own `device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` is a separate, slice-owned file with the same basename, derived from the donor but reading its DFB index through `get_named_compile_time_arg_val("dfb_id_out")` so the fusion infrastructure can remap it. `SliceTileProgramFactory` instantiates the slice-owned copy; `SliceTileTensorArgsProgramFactory` instantiates the donor. The duplicate basename is a real trip hazard when reading the two factories side by side.

### Relaxation candidates

**FALLIBLE — candidates to verify; default strict. The ops team owns the real analysis.** The readiness sheet's authoritative `TensorParameter relaxation` column reads `none`, so nothing below reaches the porter.

- The custom hash keys on the **full** `TensorSpec` of the input, the output, and each optional tensor — logical shape, padded shape, rank, layout, dtype and memory config (`device/slice_device_operation.cpp:369-429`). Two of the conjuncts carry their own written justification that is specific to the legacy binding model, and both are worth re-examining once typed bindings refresh on a cache hit:
  - the end tensor's memory config, hashed because "its memory config picks the bank table and cannot be refreshed on a hit" (`device/slice_device_operation.cpp:391-392`);
  - the preallocated output's spec, hashed because "every factory bakes a `TensorAccessorArgs` for that destination buffer into its writer's compile-time args" (`device/slice_device_operation.cpp:415-418`).
- The sheet carries a separate informational column, `Provisional relaxation finding (Edwin)`, which reads **`needs fix, then none`** for `SliceRmProgramFactory` and `SliceTileProgramFactory` and is blank for the other three. It is not the gate column and the audit recipe does not cover it, so I have not acted on it; see *Questions for the user*.

### TTNN factory analysis

- **Current concept:** `descriptor` on all five factories, each via a `create_descriptor` returning a `ProgramDescriptor`. No `create_workload_descriptor`, no mesh-workload return.
- **Op-owned tensors:** none. The sheet's cell is blank and the concept structurally cannot carry them.
- **MeshWorkload need:** none. `Execution Model` reads `SPMD` and the concept is `descriptor`, so the `WorkloadDescriptor` escape and its "secretly SPMD" question do not arise.
- **Custom hash:** `SliceDeviceOperation::compute_program_hash` at `device/slice_device_operation.cpp:348`. Not a gate; the port **leaves it exactly as it is**. Its stated reason (`:350-353`) is that the default hash distributes small-integer shape sequences poorly, causing false cache hits.
- **`get_dynamic_runtime_args`:** absent. Confirmed against the device-operation declaration.
- **`override_runtime_arguments`:** present on every factory; each is a one-line delegation to `ttnn::prim::patch_slice_program_addresses` (`device/slice_program_factory_rm_sharded.cpp:357-416`). That function is where the whole cache-hit refresh lives, and it is the thing the porter translates into a `ProgramRunArgs`. It carries three distinct shapes: a CB-address-only patch for the sharded factory (`:365-371`), a slot-0 address patch guarded by a non-zero check for the RM factories (`:374-392`), and a `DynamicRuntimeArg` re-emission of both the addresses and the per-core scalars for the two tile factories (`:396-412`). The per-core scalar re-emission exists because those scalars are hash-excluded and a divergent-partition cache hit would otherwise leave the writer at `num_pages = 0`, producing all-zero output (issue #52651, noted at `device/slice_program_factory_tile.hpp:29`).
- **Pybind `create_descriptor`:** `slice_nanobind.cpp:170`. Not a gate; the port deletes it. Sheet cell reads `PR`.
- **Other pybind of internals:** `SliceParams` and `SliceInputs` are exposed as classes and `SliceDeviceOperation::create_output_tensors` / `compute_output_specs` as statics (`slice_nanobind.cpp:138-166`). None is a factory entry point, so none is affected by the port.
- **Target concept:** **`CustomProgramSpecFactoryConcept`**, driven by `Override runtime args method?` == `yes`. The sheet's own `Porting Target` column independently reads `CustomProgramSpecFactoryConcept` on all five rows, which agrees.

## Misc anomalies  *(team-only, non-gating)*

These route to the ops team. The port does not act on any of them.

- **Two unreferenced kernel files in the op's directory.** `device/kernels/dataflow/strided_slice_reader_rm_interleaved_nd.cpp` and `device/kernels/dataflow/strided_slice_writer_rm_interleaved.cpp` are instantiated by no factory in the repository — `SliceRmStrideProgramFactory` uses the `*_multicore_slice_4d` / `*_multicore_slice_nd` pair instead (`device/slice_program_factory_rm_stride.cpp:44-54`). Both unreferenced files also address DFB index 24, which no slice factory allocates, so they could not run against any current program. Their contents are out of audit scope.
- **Dead runtime arguments in the 4D stride kernels.** `reader_multicore_slice_4d.cpp:60-62` reads `output_h`, `output_d` and `output_n` (RTA slots 7, 8, 9) and never uses them. `writer_multicore_slice_4d.cpp:54` reads `tensor_rank` (slot 1) and `:56-58` reads `output_h`, `output_d`, `output_n` (slots 3, 4, 5), none of which is used. The host still emits all of them (`device/slice_program_factory_rm_stride.cpp:128-144`).
- **A dead compile-time argument in all four stride kernels.** `compile_time_element_size` is declared from compile-time argument index 1 and never read: `reader_multicore_slice_4d.cpp:81`, `writer_multicore_slice_4d.cpp:65`, `reader_multicore_slice_nd.cpp:67`, `writer_multicore_slice_nd.cpp:66`. The value it carries also rides a runtime argument, which is the copy the kernels actually use.
- **Two dead locals.** `output_bytes_per_row` is computed and never used at `reader_multicore_slice_4d.cpp:86` and `reader_multicore_slice_nd.cpp:91`.
- **A stray post-increment.** `writer_multicore_slice_nd.cpp:73` writes `get_arg_addr(rt_args_idx++)`; `rt_args_idx` is never read again. Harmless, but it reads as if a second block were meant to follow.
- **Non-uniform writer argument 0 across the two tile factories.** `SliceTileProgramFactory` emits a literal `0u` in writer slot 0 for no-op cores (`device/slice_program_factory_tile.cpp:176`) while binding `dst_buffer` on active cores (`:180`), whereas `SliceTileTensorArgsProgramFactory` binds `dst_buffer` on every core including no-op ones and documents why (`device/slice_program_factory_tile_tensor_args.cpp:148-151`). The zero-slot skip in `patch_slot0` (`device/slice_program_factory_rm_sharded.cpp:373-383`) exists to accommodate the first shape. Under a typed binding the distinction disappears, so this is worth cleaning up on the ops track rather than carrying forward.

## Per-DeviceOperation attribution

Not applicable — the directory holds one device operation, `ttnn::prim::SliceDeviceOperation`. Findings that differ between its five factories are attributed per factory in the tables above.

## Questions for the user

1. **The sheet's provisional relaxation note.** The readiness sheet's `Provisional relaxation finding (Edwin)` column reads **`needs fix, then none`** for `SliceRmProgramFactory` and `SliceTileProgramFactory`, while the authoritative `TensorParameter relaxation` column reads `none` on all five rows. The audit recipe names only the authoritative column, so I gated on that and this audit is GREEN. If the provisional note is tracking an op fix that has to land first, the gate ought to be expressed through a column the recipe reads — either `TensorParameter relaxation` or `Known op issues`, which is blank on every slice row. Worth confirming with the readiness-sheet owner before the port starts.

2. **`Diego validation` reads `no` on all five rows.** The column is not one the audit recipe reads, and I have not treated it as a signal. Flagging it only in case it means these rows have not been reviewed yet, which would weaken the prior the five `yes` verdicts carry.

3. **MeshPartition must move with the port.** `ttnn/cpp/ttnn/operations/ccl/mesh_partition/` calls slice's `create_descriptor`, `select_program_factory` and `patch_slice_program_addresses` directly and will not compile after the port. Should the porter update MeshPartition in the same change, or is that a separate, coordinated piece of work? The answer changes the size of the port diff substantially.

## Recipe notes

- **The status-summary template has a `Variadic-CTA` row with no Appendix A entry behind it.** The template at `metal2_audit.md:575` asks for `*Feature Support* — Variadic-CTA | Ok / Unsupported`, but Appendix A carries only three entries — GlobalCircularBuffer, `address_offset`, GlobalSemaphore — and none of them is about variadic compile-time arguments. Meanwhile the *RTA varargs* subject states positively that CTA varargs "don't gate either" and port onto `KernelAdvancedOptions::compile_time_varargs`. So the row asks a gate-shaped question about a construct the recipe elsewhere says is supported. I filled it `Ok` and explained; the row should either be dropped or given a matching Appendix A entry.

- **The two dated triage documents both list this op, and both are stale — which made this the most expensive part of the audit.** The recipe is clear that my own scan wins over a dated document, and it even warns that "the slice family is the catalogued example, which is exactly why it is the one most likely to have been fixed since." That warning was correct and useful. What cost time was that the *same underlying refactor* clears rows in **both** documents: the byte-offset split that removes slice from the offset-base-pointer tables is also what dissolves the "Special — sub-page base offset" note in the 3rd-argument triage. The 3rd-argument document itself says that note is "a *2nd-arg* concern, separate from the page-size 3rd arg", which is accurate but means a Special entry is parked in a subject that cannot resolve it. Consider either moving that note to the offset-base-pointer document, or having the *TensorAccessor 3rd argument* subject say explicitly that a Special entry describing a base offset is adjudicated by the *Offset base pointers* subject and not re-adjudicated here.

- **A host-side coupling category is missing.** *Out-of-directory coupling* defines exactly two escape types: a kernel `#include`-and-call, and a program factory instantiating another op's kernel `.cpp`. Slice has a third kind that neither covers and that matters more than either: another op's C++ host code (`ccl/mesh_partition`) calls slice's `create_descriptor`, `select_program_factory` and a shared address-patching helper directly, so a Metal 2.0 port of slice breaks that op's compilation. I reported it under the brief's open "anything else" bullet, which works, but a consumer-of-the-factory-entry-point shape seems common enough — any op that borrows another's factory rather than its kernels — to deserve a named slot in the subject.

- **Two kernel files in this op share a basename with a donor, which the borrowed-kernel inventory does not anticipate.** `device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (slice-owned) and `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (borrowed) are both instantiated by this op, by different factories. A file-path inventory keyed on the basename would silently merge them. Worth a sentence in the *Borrowed kernel files* step telling the auditor to key on the full path.

- **The `experimental/quasar/` warning paid for itself.** A quasar copy of this op exists at `ttnn/cpp/ttnn/operations/experimental/quasar/slice`, and the 3rd-argument triage document has a `quasar/slice` row sitting two lines above the mainline `slice` row, so a grep for "slice" in that document surfaces the quasar row alongside the real one. The recipe's instruction to warn the porter off the directory is carried into the brief.
