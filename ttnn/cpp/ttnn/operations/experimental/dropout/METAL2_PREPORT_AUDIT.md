# Metal 2.0 Audit Findings — `experimental/dropout`

- **`DropoutDeviceOperation`** (`device/dropout_device_operation.hpp:20`, variant at `:25`). `select_program_factory` (`device/dropout_device_operation.cpp:14-20`) chooses between the two factories based on `use_per_device_seed`:
  - `DropoutProgramFactory` (`device/dropout_program_factory.{hpp,cpp}`) is used when `use_per_device_seed == false`. It builds one program and stamps it across the mesh.
  - `DropoutMeshWorkloadFactory` (same files) is used when `use_per_device_seed == true`, **which is the default** at both the Python and C++ entry points (`dropout_nanobind.cpp:70`, `dropout.hpp:20`). Its `create_descriptor` takes a `mesh_dispatch_coordinate` (`dropout_program_factory.hpp:32-36`) and delegates the build to `DropoutProgramFactory::create_descriptor` with a per-device seed (`dropout_program_factory.cpp:342-351`).
- Kernels. The op owns all three, and no other op instantiates them:
  - `device/kernels/dataflow/reader_dropout_interleaved_start_id.cpp` (reader)
  - `device/kernels/dataflow/writer_dropout_interleaved_start_id.cpp` (writer)
  - `device/kernels/compute/dropout_kernel.cpp` (compute). It is instantiated twice, as `compute_group_1` and `compute_group_2`, over the disjoint core groups from `split_work_to_cores`.

**Scope:** TTNN op, Gen1 (WH/BH) target. Within the scope of `audit/metal2_audit.md`.

**Recipe docs:** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state` (branch `akertesz/op-porting-recipe`, tip at audit time)

**Code audited:** `origin/main` @ `1bb6ea7a04e` (2026-10-13). **Readiness sheet:** fetched 2026-10-13 in this session; rows 187-188.

**Revised 2026-10-14:** added *Precedent: the paged_cache port*, recorded the invoker's choice of a complete port, and cross-referenced the Recipe notes against the paged_cache port report. The recipe branch tip was re-checked and is unchanged. The dropout op, `operation_concepts.hpp`, `mesh_device_operation_adapter.hpp` and `paged_fill_cache_program_factory.*` are identical at `origin/main` @ `3c202f9944d`, so every cited line holds there too.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/dropout/` |
| **Overall** | **RED.** Blocked only by the TTNN factory concept gate, because the readiness sheet is stale for this op. |
| **DOps / Factories** | `DropoutDeviceOperation` → `DropoutProgramFactory`, `DropoutMeshWorkloadFactory` |
| *Prereqs*: Device 2.0 (every kernel used) | **Yes.** Reader and writer are fully Device 2.0. Compute has no data-movement idioms, and its CB sync already goes through `DataflowBuffer` inside `kernel_lib`. |
| *Prereqs*: Cross-op escapes | Ok. The only out-of-directory dependency is `ttnn/cpp/ttnn/kernel_lib/eltwise`, called with the `uint32_t cb_id` shape (✓). There are no borrowed kernel files. |
| *Feature Support*: overall | GREEN. No Appendix A entry fires. |
| *Feature Support*: Variadic-CTA | Ok. None. |
| *TTNN Readiness*: `Is able to port?` (the gate) | **No, on both rows. Spreadsheet is broken:** the primary columns `Runtime-args update (get_dynamic_runtime_args)` and `Override runtime args method?` conflict with the code (see Gate detail). |
| *TTNN Readiness*: Concept (current) | `descriptor` on both factories. Matches the code. |
| *TTNN Readiness*: Secretly SPMD (WorkloadDescriptor only) | N/A (not `WorkloadDescriptor`) |
| *TTNN Readiness*: Custom hash | No `compute_program_hash`. **Backdoor** `attribute_values()` excludes `seed`: `device/dropout_device_operation_types.hpp:28-32`. Not a gate; the port leaves it intact. |
| *TTNN Readiness*: `get_dynamic_runtime_args` | **No in the code** (removed by #50341). The sheet says `yes` → conflict. |
| *TTNN Readiness*: `override_runtime_arguments` | **Yes in the code**, on both factories: `DropoutProgramFactory` at `dropout_program_factory.hpp:19-24` / `.cpp:353-405`; `DropoutMeshWorkloadFactory` at `.hpp:40-45` / `.cpp:407-415`, which delegates. The sheet says `no` → conflict. |
| *TTNN Readiness*: Pybind `create_descriptor` | No. `dropout_nanobind.cpp:61-72` binds only the `ttnn.experimental.dropout` function. |
| *TTNN Readiness*: Op-owned tensors | No |
| *TTNN Readiness*: Target concept | `DropoutProgramFactory` → **`CustomProgramSpecFactoryConcept`**. `DropoutMeshWorkloadFactory` → **`MeshWorkloadSpecFactoryConcept`** (header lookup; per-device program variation). The mesh target is **outside the port procedure's stated coverage** but has an in-tree precedent: `experimental/paged_cache`, #56022. See TTNN factory analysis and *Precedent: the paged_cache port*. |
| *Port work*: Offset base pointer | none |
| *Port work*: Tensor bindings (per binding) | `input` Case 1, `output` Case 1 (`Buffer*`-binding delivery, consumed via `TensorAccessor`) |
| *TTNN Readiness*: TensorParameter relaxation | `none` (clears) |
| *Port work*: TensorAccessor 3rd arg | none. No accessor passes a 3rd argument. |
| *Port work*: CB endpoints | legal. Both CBs are 1P+1C on every node, in every config. |

## Result

**RED: blocked on the TTNN factory concept gate (spreadsheet broken).** Routed to the **readiness-sheet owner** (Diego, `dgomez@tenstorrent.com`) to reconcile.

In plain terms: both readiness-sheet rows still describe the op as it was **before** PR #50341, *"[ttnn] experimental/dropout: replace get_dynamic_runtime_args with override_runtime_arguments"* (merged 2026-08-19, authored by the sheet owner). The sheet still records the deprecated `get_dynamic_runtime_args` hook (`yes`) and no `override_runtime_arguments` (`no`). Both are now the other way round in the code. The sheet's `Is able to port? = no` and `Op Classification = Broken Op` are therefore computed from stale inputs.

Every other gate-bearing subject is clean: Device 2.0 ✓, Feature compatibility ✓ (all N/A), Offset base pointers ✓, TensorAccessor 3rd arg N/A.

**RED at op level; no portable subset today.** Both factory rows are gated by the same sheet conflict. **Expected path after the sheet is reconciled** (the verdict cell is the sheet's call, not this audit's):

- `DropoutProgramFactory` (`use_per_device_seed=False`) is a normal single-program port to `CustomProgramSpecFactoryConcept`.
- `DropoutMeshWorkloadFactory` (`use_per_device_seed=True`, the **default** path) has genuine per-device run-arg variation. Its Metal 2.0 target is `MeshWorkloadSpecFactoryConcept`. The port procedure (`metal2_port.md` §What this procedure covers) handles only the two single-program concepts, so a porter following it literally would stop at this factory even with a `yes` on its row.

**Port scope chosen by the invoker (2026-10-14): a complete port of both factories, following the `experimental/paged_cache` port (#56022) as the precedent for the mesh factory.** That port landed the same two-factory shape on main, taking the mesh variants beyond the recipe's stated coverage on explicit invoker authorisation. This audit's verdict is unchanged by that decision: the sheet gate stays RED until the sheet owner reconciles rows 187-188. Starting the port before then is a separate, explicit invoker override. See *Precedent: the paged_cache port* below.

**Informational subjects were run in full anyway.** Under the scoping rule's exception, this RED clears **without touching the op's code**: the sheet owner updates two cells and the derived verdict is recomputed. A re-audit would read the same code, so today's detail stays valid.

## Gate detail

- **TTNN factory concept (`Is able to port?`): RED, spreadsheet broken.** The routing rule's "primary-column conflict" trigger fires on two columns, identically on both rows (187 `DropoutMeshWorkloadFactory` and 188 `DropoutProgramFactory`):

  | Column | Sheet (verbatim) | Code @ `1bb6ea7a04e` | Evidence |
  |---|---|---|---|
  | `Runtime-args update (get_dynamic_runtime_args)` | `yes` | **absent** | `grep -rn get_dynamic_runtime_args` over the op directory finds only a comment, `dropout_program_factory.hpp:18` (*"supersedes get_dynamic_runtime_args and resolve_bindings"*). The `DropoutDeviceOperation` struct (`dropout_device_operation.hpp:20-34`) declares no such hook. #50341's diff deletes `DropoutDeviceOperation::get_dynamic_runtime_args`. |
  | `Override runtime args method? (PD only)` | `no` | **present on both factories** | `dropout_program_factory.hpp:19-24` and `:40-45`; bodies at `dropout_program_factory.cpp:353-405` and `:407-415`. The descriptor adapter detects it via `has_override_runtime_arguments()` (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp`, the "override_runtime_arguments … descriptor-era analog" block). |

  - The other primary columns cross-check clean:
    - `Concept = descriptor`: both factories have `create_descriptor` returning `ProgramDescriptor`.
    - `Custom hash (compute_program_hash) = no`: none.
    - `Backdoor custom hash = yes`: `attribute_values()` at `dropout_device_operation_types.hpp:28-32`.
    - `Pybind descriptor = no`: `dropout_nanobind.cpp:61-72`.
  - **Factory-set match:** the code has exactly the two factories the sheet lists (`dropout_device_operation.hpp:25`). There are no phantom or missing rows.
  - **Cross-column invariants:** the sheet is internally consistent (`get_dynamic_runtime_args=yes` on a `descriptor` concept is allowed; the two runtime-args columns are not both `yes`). It is stale, not self-contradictory.
  - **Blocking attribution (for the reconciler):**
    - The only blocking signal visible on either row is the stale `get_dynamic_runtime_args = yes`, which per the audit's blocking table also produces `Op Classification = Broken Op`.
    - `Known op issues` is empty.
    - `TensorParameter relaxation = none` (clears).
    - `Smuggled pointer = no`.
    - `Diego validation = no` on both rows, which may mean the rows were never re-validated after #50341.
  - **Path forward:** the sheet owner refreshes rows 187-188 to `get_dynamic_runtime_args = no` and `Override runtime args method? = yes`, then re-derives `Is able to port?` / `Op Classification` / `Porting Target`. Re-run this audit when the cell changes. See also *Questions* (1) on `Execution Model` for the mesh factory.
- **Device 2.0 (every kernel used): GREEN.**
  - **Reader** (`reader_dropout_interleaved_start_id.cpp`) uses `Noc noc` (`:13`), `TensorAccessorArgs<1>()` + `TensorAccessor(src_args, src_addr)` (`:20,:24`), `CircularBuffer cb_in0` with `.reserve_back/.get_write_ptr/.push_back` methods (`:26,:36-40`), and `noc.async_read(s, CoreLocalMem<uint32_t>(…), …, {.page_id = i}, {})` + `noc.async_read_barrier()` (`:38-39`).
  - **Writer** (`writer_dropout_interleaved_start_id.cpp`) uses the mirror image (`:13,:19,:21,:29,:38-42`).
  - Both match the migration guide's *Example 1: Tile Read Kernel* (migrated form) almost line for line.
  - `get_tile_size(cb_id_in0)` (reader `:38`) and `get_tile_size(cb_id_out)` (writer `:40`) are **sanctioned** free functions and not flagged.
  - **Compute** (`dropout_kernel.cpp`) has no data-movement idioms. Its CB FIFO sync happens inside `compute_kernel_lib::eltwise_chain`, which already routes `wait_front/pop_front/reserve_back/push_back` through `DataflowBuffer(cb)` (`ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.inl:20,2421-2442`).
  - No holdovers, no `InterleavedAddrGen*` / `ShardedAddrGen`, and no raw semaphore addresses.
- **Feature compatibility: GREEN. No gate fired.**

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, no `global_circular_buffer` field, no `remote_index`, no `CreateCircularBuffer(…, global_cb)`. |
  | CBDescriptor `address_offset` (non-zero) | N/A | Both `CBDescriptor`s are built in `create_circular_buffer` (`dropout_program_factory.cpp:69-77`) with no `.address_offset` and no `.buffer` (program-local CBs). |
  | GlobalSemaphore | N/A | No `GlobalSemaphore`, no `CreateGlobalSemaphore`, no semaphores at all. |

- **CB endpoints (gate-free): legal.**
  - `c_0` (input): the reader is the locked producer and compute is the locked consumer → 1:1.
  - `c_2` (output): compute is the locked producer and the writer is the locked consumer → 1:1.
  - This holds on every node and in every config. Details are under Port-work summary.
- **Offset base pointers: GREEN.**
  - Both address RTAs are clean bases. The reader gets `{src_buffer, num_tiles, tile_offset}` and the writer gets `{dst_buffer, num_tiles, tile_offset}`. Both use the `Buffer*`-binding form (`dropout_program_factory.cpp:226,229`); the tile offset travels as a separate scalar and is consumed as a `TensorAccessor` page index (`{.page_id = i}`).
  - On a cache hit the override writes the bare `buffer()->address()` (`:375-376`). There is no `+ offset` fold anywhere.
  - The op is not in `2026-07-19_offset_base_pointers.md`, and the scan agrees: no fold, op not listed → clean.
- **TensorAccessor 3rd argument: N/A.** No accessor in the op passes a 3rd argument (reader `:24`, writer `:29` are both two-argument). The op is not in `2026-07-06_tensor_accessor_3rd_arg_triage.md`.

## Port-work summary  *(mirrors the brief, which is not issued on RED)*

- **Tensor bindings (per binding):**
  - `input` is **Case 1**. It is delivered as a `Buffer*` RTA (`dropout_program_factory.cpp:226`), so it is correct on a cache hit today and not the silent-wrong hazard. The reader feeds it to `TensorAccessor(src_args, src_addr)` (reader `:15,:24`). Its `TensorAccessorArgs` CTAs are appended at `dropout_program_factory.cpp:278-279`.
  - `output` is **Case 1**: writer `:14,:29`, factory `:229,:281-283`.
  - Both factories are the same here, since the mesh factory delegates its build.
- **TensorParameter relaxation:** none (`none` on both rows). Keep strict.
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:** all legal. Census per node:
  - The reader covers `all_cores`. Exactly one of `compute_group_1` / `compute_group_2` covers each node (disjoint groups from `split_work_to_cores`, `dropout_program_factory.cpp:151-164`). The writer covers `all_cores`.
  - `c_0` (`kSrc0CbIndex`, `:24`, 2 tiles, `:270`): the reader FIFO-produces (`reserve_back`/`push_back`, plus a `get_write_ptr` peek covered by its producer binding). Compute FIFO-consumes (`CopyTile<input(c_0, PerTile, PerTile, …)>`) → **1P+1C, legal**.
  - `c_2` (`kOutputCbIndex`, `:25`, 2 tiles, `:272`): compute FIFO-produces (`PackTile<output(c_2, PerBlockSize, PerBlockSize, …)>`). The writer FIFO-consumes (`wait_front`/`pop_front`, plus a `get_read_ptr` peek) → **1P+1C, legal**.
  - The factory sets no kernel defines, so there is no config flip. The `#ifdef OUT_SHARDED` / `#ifdef BACKWARDS` branches are never compiled (see Misc anomalies). There are no dead CBs.

## Heads-ups  *(mirrors the brief, which is not issued on RED)*

- **CB endpoints (multi-binding shapes to watch):** none.
- **Cross-op / shared kernels:** none. All three kernels live in the op directory and nothing else instantiates them (repo-wide grep for the three kernel paths finds only `dropout_program_factory.cpp:17-22`). No `_metal2` fork question arises.
- **RTA varargs:** none.
  - Reader and writer each read three fixed named RTAs: `src_addr`/`dst_addr`, `num_tiles`, `start_id` (reader `:15-17`, writer `:14-16`).
  - Compute reads one, `seed` (`dropout_kernel.cpp:18`).
  - CTAs are fixed and nameable: compute `per_core_block_cnt`, `per_core_block_dim`, `int_probability`, `int_scale_factor` (`:13-16`); reader/writer `cb_id` at index 0, then `TensorAccessorArgs<1>`.
- **Anything else the porter needs:**
  - **Target concept differs per factory, and the port is a complete one.** `DropoutProgramFactory` → `CustomProgramSpecFactoryConcept`; translate `override_runtime_arguments` (`dropout_program_factory.cpp:353-405`) into a `ProgramRunArgs`-returning override. `DropoutMeshWorkloadFactory` → `MeshWorkloadSpecFactoryConcept`. The recipe's stop rule covers this target, but the invoker has chosen to port both factories (2026-10-14), following *Precedent: the paged_cache port* below. Record that authorisation in the port plan and report, as paged_cache did.
    - The two factories share code. The mesh factory's `create_descriptor` calls `DropoutProgramFactory::create_descriptor` (`:350`), and its override calls `DropoutProgramFactory::override_runtime_arguments` (`:413-414`). Both convert in the same change, so keep that sharing, the way paged_cache's mesh factories call their single-device sibling's build helper.
  - **The override's refresh set is inherited, not re-decided.** On every hit the ported-from override re-applies:
    - the `src`/`dst` addresses (`:375-376`). These must become `tensor_args` entries for **both** io `TensorParameter`s, not RTA values.
    - per-node `num_tiles` and `start_id` for reader and writer (`:392-399`). The comment at `:378-379` says these are "rewritten anyway".
    - the compute `seed` (`:403`). On the plain factory it is `operation_attributes.seed`.

    Reproduce exactly that set.
  - **In-place aliasing is a tested path.** `test_dropout.py:49` calls `dropout(tensor, …, output_tensor=tensor)`, so the input and output `TensorParameter`s bind the same tensor. The factory comments call this case out (`dropout_program_factory.cpp:223-225,372-374`). Keep both `TensorArgument`s sourced from their own tensor (input from `tensor_args.input`, output from `tensor_return_value`), as the legacy override does.
  - **Compute is a same-source, disjoint-node work split.** There are two compute `KernelDescriptor`s, one per core group, differing only in the `per_core_block_cnt` CTA (`dropout_program_factory.cpp:299-322`). Port them as **two `KernelSpec`s in two `WorkUnitSpec`s**, both binding the same DFBs. Do **not** demote `per_core_block_cnt` to an RTA (port_patterns.md, *Anti-pattern: Demoting per-group CTA to RTA*). Group 2 is conditional on `core_group_2` being non-empty (`:312`).
  - **The compute kernel hardcodes its CB ids** as `constexpr auto dfb_input_id = tt::CBIndex::c_0; dfb_output_id = tt::CBIndex::c_2;` (`dropout_kernel.cpp:20-21`). They match the factory's `kSrc0CbIndex`/`kOutputCbIndex` only by convention, not by CTA. Both are `constexpr` and feed `ckl::input(...)` / `ckl::output(...)` in NTTP position (`:28-37`) and `compute_kernel_hw_startup` (`:23`). That is the `uint32_t cb_id` shape `dfb::name`'s constexpr cast covers in template position.
  - **Compute config is fixed:** `HiFi4`, `fp32_dest_acc_en=false`, `dst_full_sync_en=false`, `math_approx_mode=false` (`dropout_program_factory.cpp:131-136,306`). Carry it verbatim, and set an explicit `opt_level`: a `ComputeConfigDescriptor` with no `opt_level` resolves to `O3`, while Metal 2.0 defaults to `O2`. paged_cache hit the same case and set `KernelBuildOptLevel::O3` explicitly (e.g. `paged_update_cache_program_factory.cpp:501` on main).
  - **Tests for verification.** `tests/ttnn/unit_tests/operations/data_movement/test_dropout.py` covers both cache-hit override paths: `test_dropout_seed_distinguishes_cache_entries` (`:74-118`), parametrized over `use_per_device_seed` and in-place, and `test_dropout_cache_hit_rederives_buffer_address` (`:121`). On a single device the mesh is 1×1, so these exercise the mesh factory's build and override but not seed *differences* across devices. Cross-device behaviour is only covered by `tt-train/tests/core/mesh_1x2_utils_test.cpp:231`, which needs two devices. paged_cache had the same gap (its report, Open items #4).

## Precedent: the paged_cache port

`experimental/paged_cache` was ported in #56022 ("[Cleanup] Metal 2.0 port: experimental/paged_cache", Frank Plavec, merged 2026-09-11). It ran on this same recipe: its port report pins `f9451e2a21d`, whose `metal_2.0` docs are byte-identical to `4bd4bf42bfe`. It is the in-tree model for porting `DropoutMeshWorkloadFactory`. Its plan, report, brief and audit were never merged; they live at commit `dce271e7bbfa` under `ttnn/cpp/ttnn/operations/experimental/paged_cache/` (`METAL2_PORT_PLAN.md`, `METAL2_PORT_REPORT.md`).

**Why it is the right precedent.** Each paged_cache device-op has the same two-factory shape as dropout: a single-device factory with a PD `override_runtime_arguments`, and a `*MeshWorkloadFactory` whose `create_descriptor` takes a coordinate and delegates to its sibling. The closest analog is `PagedFillCacheMeshWorkloadFactory`. It emits a program for **every** coordinate and varies **one run-arg value** per coordinate (`noop`), exactly as dropout varies `seed`. The other three paged_cache mesh factories instead omit excluded coordinates; dropout has no such idiom.

**How paged_fill_cache is structured on main** (`ttnn/cpp/ttnn/operations/experimental/paged_cache/device/fill_cache/paged_fill_cache_program_factory.{hpp,cpp}`, unchanged between `1bb6ea7a04e` and `3c202f9944d`):

| Piece | paged_fill_cache (`.cpp`) | dropout equivalent |
|---|---|---|
| Shared spec + run-args builder, parameterized by the per-coordinate value | `build_paged_fill_cache_artifacts(attrs, tensor_args, noop)` (`:117`) | today's `DropoutProgramFactory::create_descriptor` body (`dropout_program_factory.cpp:241-340`), parameterized by `seed` |
| Shared cache-hit run args for one coordinate | `paged_fill_cache_run_args_for_coord(attrs, tensor_args, coord)` (`:519-547`) | today's override body (`:353-405`); the seed derivation at `:368-370` already takes the coordinate |
| Single-device `create_program_artifacts` | `:551-557`, builds with `noop(attrs, nullopt)` | builds with `args.seed` |
| Single-device override | `:559-565`, `std::optional<MeshCoordinate>` form | same |
| Mesh `create_mesh_workload_artifacts` | `:567-589`: loops `tensor_coords.coords()`, one `PerCoordProgram` per coordinate with `range = MeshCoordinateRange(coord)` | loop the coordinates, build with `per_device_seed(args, coord, input)` (`dropout_program_factory.cpp:33-40`) |
| Mesh override | `:591-601`, `const MeshCoordinateRange&` form; uses `range.start_coord()` because every range is one coordinate | same, with the seed taken from `range.start_coord()` |

**The four decisions paged_cache had to make without recipe guidance** (its report, Handoff #1(b)), applied to dropout:

1. **Program granularity:** one program per coordinate, reproducing what the ported-from descriptor adapter builds (`mesh_device_operation_adapter.hpp:608-612`). Ranges wider than one device would share a seed and change behaviour.
2. **Omitted vs. no-op coordinates:** not applicable. Dropout builds a real program for every coordinate in `tensor_coords`, so it cannot hit paged_cache's zero-programs error (its report, Handoff #2).
3. **How `tensor_coords` bounds the ranges:** iterate `tensor_coords.coords()` exactly, as the descriptor path does today.
4. **Mapping the override's range to a coordinate:** `range.start_coord()`, valid because every emitted range is a single coordinate. Say so in a comment, as paged_fill_cache does (`:596-597`).

**Other lessons from the paged_cache report that apply here:**
- **Kernels convert in place; no `_metal2` fork** (its Friction #4). A fork is only needed when some binder of a kernel won't convert in the same change. Both dropout factories convert together, and no other op binds these kernels.
- **Cache-hit tensor-arg completeness:** on both spec concepts the override must return a `TensorArgument` for **every** io `TensorParameter` on every dispatch. The framework refreshes none of them for you.
- **Compute `hw_config`:** paged_cache built `ComputeGen1Config` directly instead of routing through `to_compute_hardware_config`, which would have started honouring fields the legacy config ignored (its Successes, `hw_config` bullet). Dropout's `ComputeConfigDescriptor` (`dropout_program_factory.cpp:131-136`) sets `HiFi4` / `fp32_dest_acc_en=false` / `dst_full_sync_en=false` / `math_approx_mode=false`; check each against the `ComputeGen1Config` defaults the same way.
- **Verification:** force `skip_validation=false` in the Metal 2.0 host-API entry points and prove the legality checks ran (its *Legality checks were provably live*). Run the op's tests with `TT_METAL_WATCHER` before and after, then revert the scaffolding.
- **Stop rule vs. disagree-with-the-audit:** paged_cache's audit named the wrong concept, so the port was a *disagree* case. Here the audit names `MeshWorkloadSpecFactoryConcept` itself, so the recipe's stop rule applies literally, and proceeding rests on the invoker's explicit decision (recorded above). That decision is what the port plan should cite.

## Team-only

- **Out-of-directory coupling and donor shape: ✓ clean.**
  - The reader and writer include only `tt_metal/*` headers (class 1): `api/dataflow/{dataflow_api,noc,circular_buffer}.h`, `api/core_local_mem.h`, `api/tensor/noc_traits.h`.
  - Compute includes `api/compute/compute_kernel_hw_startup.h` (class 1) and `ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp` + `kernel_lib/eltwise/unary/scalar.hpp` (class 2, shared kernel lib).

  | Op kernel | Donor file | Functions called | Shape | Status |
  |---|---|---|---|---|
  | `dropout_kernel.cpp` | `kernel_lib/eltwise/api/chain.hpp` | `eltwise_chain`, `CopyTile<input(cb_id, …)>`, `PackTile<output(cb_id, …)>`, `IterationShape::tiles` | `uint32_t cb_id` (constexpr, NTTP position via `InputSpec`/`OutputSpec`, `chain.hpp:318-392`) | ✓ |
  | `dropout_kernel.cpp` | `kernel_lib/eltwise/unary/scalar.hpp` | `Dropout<Dst::D0>{prob, scale, seed}` | scalars only | ✓ |

  - Per-call detail is omitted because every row is ✓.
  - Borrowed kernel files: none.
- **Relaxation candidates (FALLIBLE: candidates to verify; default strict).** A comment at `dropout_device_operation_types.hpp:35-39` says the input used to be hashed by **volume only** before the current full-`TensorSpec` reflection hash. That suggests the program depends on tile count, not shape, which makes it a candidate for a future shape relaxation. The ops team owns the real analysis. Note also that the sheet's `Formerly custom hashed?` cell is blank for both rows.
- **TTNN factory analysis:**
  - **Current concept:** `descriptor` on both, via `create_descriptor` → `ProgramDescriptor`. `DropoutMeshWorkloadFactory` uses the **4-argument per-coordinate form** (`mesh_dispatch_coordinate`). The descriptor adapter therefore builds **one Program per mesh coordinate** (`mesh_device_operation_adapter.hpp:608-612`), and each program differs only in its compute `seed` RTA = `seed + device_id` (`dropout_program_factory.cpp:33-40`). The header TODO (`dropout_program_factory.hpp:28-31`) states the intent to collapse this to one program "when heterogeneous runtime arguments are supported".
  - **Why the mesh factory cannot target a single-program spec concept:** the `ProgramSpecFactoryConcept` / `CustomProgramSpecFactoryConcept` adapters call `create_program_artifacts` **once** and map that one `ProgramArtifacts` and its `run_params` onto every occupied range (`mesh_device_operation_adapter.hpp:984-991`).
    - On a cache miss, every device in a range would get the same seed.
    - On a hit, the custom adapter calls the override once per range with `coordinate_range.start_coord()` (`:1033-1039`), so devices in a range still share one seed.
    - The adapter's own comment says so: *"per-coordinate run args must come from a MeshWorkloadSpecFactoryConcept factory's create_mesh_workload_artifacts"* (`:1017-1021`).
    - Header lookup (`grep -n "concept \|create_" ttnn/api/ttnn/operation_concepts.hpp`): **`MeshWorkloadSpecFactoryConcept`** (`operation_concepts.hpp:117-132`) admits "a ProgramSpec and ProgramRunArgs per coordinate range" with an optional range-scoped `override_runtime_arguments`. It landed in #54988 (2026-09-04). In-tree adopters: the four `experimental/paged_cache` mesh factories, ported from PD per-coordinate factories in #56022 (the precedent above), and seven `experimental/kda` factories, which were written natively on Metal 2.0 and moved to this concept in #56632.
    - **Target: `MeshWorkloadSpecFactoryConcept`.** The port procedure does not cover it; paged_cache shows the shape a port takes.
    - This is **genuine per-coordinate variation**, not the "legacy MeshWorkload as resource workaround" shape: there are no op-owned tensors, and what varies is a run-arg *value*.
  - **Op-owned tensors:** none.
  - **Pybind `create_descriptor`:** none. Other risky pybind: none (only `bind_function`).
  - **Custom hash:** backdoor only. `DropoutParams::attribute_values()` hashes `output_dtype`, `output_memory_config`, `use_per_device_seed`, `prob`, `scale` and **excludes `seed`** (`dropout_device_operation_types.hpp:28-32`). `DropoutInputs` uses default reflection (full `TensorSpec`). The exclusion is consistent with the override re-applying `seed` on every hit. Leave it intact.
  - **`get_dynamic_runtime_args`:** none (removed by #50341).
  - **`override_runtime_arguments`:** present on both factories (`void`-returning, PD form). See the cites above.

## Misc anomalies  *(team-only, non-gating)*

- **Dead preprocessor branches.**
  - The reader and writer carry `#ifdef BACKWARDS` (reader `:29`, writer `:31`) and the writer carries `#ifdef OUT_SHARDED` (`:23`). The factory never sets kernel defines (`create_reader_kernel` / `create_writer_kernel`, `dropout_program_factory.cpp:83-114`), so these branches never compile.
  - The `OUT_SHARDED` branch would `wait_front(num_tiles)` with no `pop_front`.
  - The port must carry them unchanged; removal is an ops-team cleanup.
- **Sharded-input validation gap** (noticed, not exercised by any test found). `validate_on_program_cache_miss` checks `TILE` + `INTERLEAVED` only for non-sharded input (`dropout_device_operation.cpp:54-66`), and `compute_output_specs` keeps the input's layout for a sharded output (`:93-95`). The factory, however, always moves `tile_size`-byte pages by page index with tile-count work splitting (`dropout_program_factory.cpp:151-153`, reader `:38`, writer `:40`). A sharded **ROW_MAJOR** input appears to pass validation and then be processed as if it were tiled. Unverified on hardware.
- **Correlated per-device seeds.** `per_device_seed = seed + device_id` (`dropout_program_factory.cpp:38-39`), so a call with seed `s` on device `k` draws the same mask as seed `s+k` on device 0. Callers that step seeds by 1 per call get masks that repeat across devices and steps (e.g. `tt-train/tests/ttnn_fixed/dropout_op_test.cpp:104` uses `dropout_seed + i`). Possibly intended; worth a look by the ops team.
- **Stale TODO** at `dropout_program_factory.hpp:28-31`, which mentions `AdaptedCachedMeshWorkload` / `CachedMeshWorkload`. Cosmetic.

## Questions for the user

1. **`Execution Model = SPMD` on the `DropoutMeshWorkloadFactory` row (sheet row 187).** That factory builds one program per coordinate with a per-device seed RTA (`dropout_program_factory.hpp:32-36`, `mesh_device_operation_adapter.hpp:608-612`), which maps to `MeshWorkloadSpecFactoryConcept` rather than a single stamped spec. This column is not on the audit's cross-check list, so it is a question for the sheet owner, not a conflict claim. Should it read as per-device / multi-program? The four paged_cache mesh rows (201, 203, 205, 207) carry the same value (see Recipe notes), so the answer probably applies sheet-wide.
2. ~~**Is a `DropoutProgramFactory`-only subset port wanted?**~~ **Answered by the invoker (2026-10-14):** no. Port both factories completely, following the paged_cache precedent.

## Recipe notes

> **These notes repeat open asks from the paged_cache port.** Its port report (`ttnn/cpp/ttnn/operations/experimental/paged_cache/METAL2_PORT_REPORT.md` at commit `dce271e7bbfa`, PR #56022, merged 2026-09-11) raised the first three items below under *Handoff points #1* and *Friction #2*, against a recipe copy (`f9451e2a21d`) whose `metal_2.0` docs are byte-identical to the one this audit ran on. The recipe branch `akertesz/op-porting-recipe` has not moved since (tip `4bd4bf42bfe`, re-checked 2026-10-14; no newer `metal_2.0` docs exist on any `akertesz/*` branch, and none on `main`). So dropout is the second op to hit these gaps, not the first.

- **`ttnn_factory.md` §Multi-program is stale.** It says *"As of 2026-08-31 there was none"* (no Metal 2.0 per-coordinate concept). `MeshWorkloadSpecFactoryConcept` landed in #54988 on 2026-09-04, a day after the recipe tip. `main` now has the concept (`operation_concepts.hpp:117-132`), its own adapter (`mesh_device_operation_adapter.hpp:1044+`), and eleven in-tree adopters (four `paged_cache`, seven `kda`). The recipe's own lookup instruction worked; only the dated sentence is stale.
- **The TTNN porting shape mapping misses per-coordinate `descriptor` factories.** The audit maps `descriptor` → `ProgramSpecFactoryConcept` / `CustomProgramSpecFactoryConcept` by the override column alone. `ttnn_factory.md` §The cache-hit contract adds that on the custom concept *"Per-device values are expressible — that is what `coord` is for."*
  - That is not true for a cache **miss**: the custom adapter inherits the base miss path, which applies one `run_params` to every range (`mesh_device_operation_adapter.hpp:984-991`). The adapter comment at `:1017-1021` says per-coordinate run args must come from `MeshWorkloadSpecFactoryConcept`.
  - A `descriptor` factory whose `create_descriptor` takes a `mesh_dispatch_coordinate` (4-argument form) is a per-coordinate op, and it has a `descriptor` concept on the sheet. Applied literally, the current mapping would send `DropoutMeshWorkloadFactory` to `CustomProgramSpecFactoryConcept` and silently lose the per-device seed.
  - **paged_cache hit exactly this.** Its audit named `CustomProgramSpecFactoryConcept` for all four mesh factories, and only the porter caught it. Its report shows the failure would have been silent: excluded chips would have written to the cache (Handoff #1). Its ask (a) proposed the same rule as this note: *"a `descriptor` factory whose `create_descriptor` takes a `std::optional<MeshCoordinate>` and varies per coordinate is a mesh-workload target, not a single-program one."* This audit applied that rule by hand, so dropout's audit names the right target. Without it, the recipe would have reproduced paged_cache's mis-targeting.
  - Suggested fix (unchanged): add that recognition line to TTNN porting shape, routing to the ttnn_factory §Multi-program lookup, and qualify the `coord` sentence in §The cache-hit contract.
- **No procedure exists for the mesh-workload target.** The port recipe stops a porter at `MeshWorkloadSpecFactoryConcept` (`metal2_port.md` §What this procedure covers). paged_cache's ask (b) noted that a procedure would have settled the four decisions its port made unaided: program granularity, omitted vs. no-op coordinates, how `tensor_coords` bounds the emitted ranges, and how the override's range maps to a coordinate. Dropout needs the same four decisions (worked out in *Precedent: the paged_cache port*). Two ops now share the per-coordinate-descriptor → mesh-spec translation, which argues for writing it into the recipe rather than leaving each porter to rediscover it.
- **The stop rule and the disagree-with-the-audit rule overlap** (paged_cache report, Friction #2). paged_cache proceeded because its brief named a *supported* concept that the porter found wrong, which the recipe treats as disagree-and-surface. Dropout's audit names the mesh concept *directly*, so the stop rule fires literally, and proceeding here rests only on an invoker override. The recipe should say whether an audit that correctly names a mesh-workload target may proceed on invoker authorisation, as it effectively did for paged_cache.
- **The sheet-side ask from paged_cache is also still open** (paged_cache report, Handoff #1(a): *"Correct the sheet's `Porting Target` for these four rows"*). In the copy of the sheet fetched for this audit (2026-10-13), rows 201-208 (`experimental/paged_cache`) still read:
  - `Concept = descriptor` on all eight, a month after #56022 put them on Metal 2.0 (`MetalV2` would be expected);
  - `Porting Target = CustomProgramSpecFactoryConcept` on all four `*MeshWorkloadFactory` rows, the target the port report showed to be wrong;
  - `Execution Model = SPMD` on those four mesh rows.

  This is an observation for the recipe and sheet owners, not part of dropout's verdict; cross-row analysis is outside this audit. It bears on dropout in two ways. Dropout's mesh row carries the same `Execution Model = SPMD` (see *Questions* 1), which suggests a sheet-wide derivation that doesn't see the 4-argument `create_descriptor`. And once rows 187-188 are reconciled, the dropout mesh row's derived `Porting Target` (currently `(N/A)`) may come out as `CustomProgramSpecFactoryConcept`, as paged_cache's did. A porter should not take it from there: this audit's target, `MeshWorkloadSpecFactoryConcept`, stands.
- **`ttnn_op_porting_readiness.md` fetch target.** It says to decode the CSV into `docs/.../metal_2.0/analyses/` *of your checkout*. When auditing `main` code from a worktree that does not carry the doc branch, that folder does not exist. This audit decoded into a session scratch directory instead (still ephemeral, never committed). It may be worth saying explicitly that any ephemeral location is fine.
- **Primary-column conflict caused by a sheet-owner PR.** The conflicting change (#50341) was authored by the sheet owner, and the sheet's own `Diego validation` column reads `no` for both rows. The audit's "spreadsheet broken" routing still applies cleanly. The `Diego validation` column is not described in `ttnn_op_porting_readiness.md`; if it means "row not yet re-validated after a code change", it would be a cheap pre-check for exactly this staleness.
