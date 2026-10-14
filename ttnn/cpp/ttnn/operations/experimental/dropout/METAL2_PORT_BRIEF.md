# Metal 2.0 Port Brief — `experimental/dropout`

> **Issued on an invoker override, not on a GREEN audit.** The audit (`METAL2_PREPORT_AUDIT.md`) is RED on exactly one gate: the TTNN factory concept gate, because readiness-sheet rows 187-188 predate #50341 and still record `get_dynamic_runtime_args = yes` / `Override runtime args method? = no`. The code contradicts both cells. Every code-side gate passes. On 2026-10-14 the invoker (iwrosz) chose to proceed with the port while the sheet owner is asked to reconcile the rows, and chose a **complete** port of both factories. The mesh factory follows the `experimental/paged_cache` port (#56022) as precedent. This brief is the porter's actionable input under that decision; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✗ (stale sheet; invoker override) · Offset base pointers ✓ · TensorAccessor 3rd arg ✓ (no sites)

**Recipe docs:** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state` *(carry this line into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`). The two factories land on **different** concepts.

- **Current concept:** `descriptor` on both factories.
  - `DropoutProgramFactory`: `create_descriptor(args, tensor_args, output)` (`device/dropout_program_factory.hpp:13-14`).
  - `DropoutMeshWorkloadFactory`: the 4-argument per-coordinate form `create_descriptor(args, tensor_args, output, mesh_dispatch_coordinate)` (`:32-36`). The descriptor adapter builds one Program per mesh coordinate for it (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:608-612`), each differing only in the compute `seed` runtime arg.
- **Op-owned tensors:** none.
- **Target concepts:**
  - `DropoutProgramFactory` → **`CustomProgramSpecFactoryConcept`**. It has a PD `override_runtime_arguments` (`device/dropout_program_factory.cpp:353-405`); translate it into one returning `ProgramRunArgs`.
  - `DropoutMeshWorkloadFactory` → **`MeshWorkloadSpecFactoryConcept`** (`ttnn/api/ttnn/operation_concepts.hpp:117-132`). Implement `create_mesh_workload_artifacts(attrs, tensor_args, output, const MeshCoordinateRangeSet& tensor_coords)` and the range-scoped `override_runtime_arguments(attrs, tensor_args, output, const MeshCoordinateRange& range)`. This target is outside `metal2_port.md` §What this procedure covers; the invoker authorised it. Follow `PagedFillCacheMeshWorkloadFactory` on `main` (see *Watch for*).
- **Gate-cleared, confirmed absent:** a `TensorParameter relaxation` that is neither `none` nor an analysis pointer (the cell is `none`) · `get_dynamic_runtime_args` (removed by #50341; the sheet is stale on this).
- **Present, not gates:**
  - Backdoor custom hash: `DropoutParams::attribute_values()` excludes `seed` (`device/dropout_device_operation_types.hpp:28-32`). Leave it exactly as it is.
  - No pybound `create_descriptor`: `dropout_nanobind.cpp:61-72` binds only `ttnn.experimental.dropout`, so the port removes no pybind surface.

## Construct — to do

**Tensor bindings** (per binding):

- `input` — **Case 1** (via `TensorAccessor`). The reader receives it as a `Buffer*` runtime arg (`dropout_program_factory.cpp:226`) and builds `TensorAccessor(src_args, src_addr)` (reader `:15,:20,:24`). Express it as a `TensorParameter` / `TensorBinding`; the kernel uses `TensorAccessor(tensor::<name>)`. The `src_addr` runtime arg and its `TensorAccessorArgs` compile-time plumbing (`dropout_program_factory.cpp:278-279`, reader `:20`) both drop.
- `output` — **Case 1**, same shape: writer `:14,:19,:29`, factory `:229,:281-283`.

**TensorParameter relaxation:** `none`. Keep strict.

**TensorAccessor 3rd arg:** none. No accessor passes one.

**CB endpoints:** all legal 1:1; no self-loop, no multi-binding, no dead CB, no conditional DFB.
- `c_0` (input, 2 tiles): reader PRODUCER, compute CONSUMER.
- `c_2` (output, 2 tiles): compute PRODUCER, writer CONSUMER.
- Each node runs exactly one of the two compute instances (disjoint core groups), so each compute `KernelSpec` binds both DFBs in the same roles.

## Watch for

- **Precedent for the mesh factory: `experimental/paged_cache`, #56022.** The exact analog is `PagedFillCacheMeshWorkloadFactory` (`ttnn/cpp/ttnn/operations/experimental/paged_cache/device/fill_cache/paged_fill_cache_program_factory.cpp` on `main`). It emits a program for every coordinate and varies one run-arg value per coordinate (`noop`), as dropout varies `seed`.
  - Mesh `create_mesh_workload_artifacts` (`:567-589`): loop `tensor_coords.coords()`, one `PerCoordProgram` per coordinate with `range = MeshCoordinateRange(coord)`, built by the shared single-device builder with the per-coordinate value.
  - Mesh override (`:591-601`): one call per range; every range is one coordinate, so `range.start_coord()` is the coordinate.
  - For dropout, the per-coordinate value is `per_device_seed(args, coord, input)` (`dropout_program_factory.cpp:33-40`), which the cache-miss build and the override must both use, as today.
  - Its unmerged port report and plan (commit `dce271e7bbfa`, `ttnn/cpp/ttnn/operations/experimental/paged_cache/METAL2_PORT_{REPORT,PLAN}.md`) record the four decisions the recipe doesn't cover; the audit's *Precedent* section maps each to dropout.
- **Keep the factory sharing.** Both factories convert in this change. The mesh factory builds through `DropoutProgramFactory`'s builder (today `:350`) and patches through its override logic (today `:413-414`). Keep that sharing rather than duplicating the build.
- **Translate the override with fidelity.** The ported-from override re-applies, on every hit:
  - the `src`/`dst` addresses (`:375-376`) → `tensor_args` entries for **both** `TensorParameter`s, never runtime-arg values;
  - per-node `num_tiles` and `start_id` for reader and writer (`:392-399`; "rewritten anyway", `:378-379`);
  - the compute `seed` (`:403`), with the `use_per_device_seed` branch at `:368-370`.

  Nothing more. Its `TT_FATAL` (`:401`) comes across with it.
- **In-place aliasing is a tested path** (`test_dropout.py:49`, `output_tensor=tensor`). The input and output `TensorParameter`s bind the same tensor. Source each `TensorArgument` from its own tensor, as the legacy override does (`:372-376`).
- **Compute is a same-source, disjoint-node work split** (`dropout_program_factory.cpp:299-322`). Two compute `KernelSpec`s in two `WorkUnitSpec`s (reader and writer in both), differing only in `per_core_block_cnt`. Do not demote it to a runtime arg. Group 2 exists only when `core_group_2` is non-empty (`:312`).
- **Compute CB ids are hardcoded** (`dropout_kernel.cpp:20-21`: `tt::CBIndex::c_0` / `c_2`), matching the factory's `kSrc0CbIndex` / `kOutputCbIndex` by convention. They feed `ckl::input(...)` / `ckl::output(...)` in template-argument position and `compute_kernel_hw_startup`. Pass `dfb::` tokens directly; the constexpr conversion covers template position.
- **`get_tile_size(cb_id)` in reader `:38` and writer `:40`** is an inline call argument, not a `constexpr` declaration, so rule 7 moves it to the DFB member getter.
- **Compute config is Style B** (a literal `ComputeConfigDescriptor`, `:131-136`): `HiFi4`, `fp32_dest_acc_en=false`, `dst_full_sync_en=false`, `math_approx_mode=false`. No `opt_level` is set anywhere in the factory, so compute resolves to `O3` today; set `KernelBuildOptLevel::O3` explicitly on **both** compute `KernelSpec`s. DM kernels use bare `ReaderConfigDescriptor{}` / `WriterConfigDescriptor{}`, i.e. the stock reader/writer triples.
- **Dead `#ifdef BACKWARDS` / `#ifdef OUT_SHARDED` branches** (reader `:29`, writer `:23,:31`): the factory sets no defines, so they never compile. Convert their `CircularBuffer` calls mechanically and keep them; their removal is an ops-team cleanup.
- **Tests:** the confirmed baseline is `tests/ttnn/unit_tests/operations/data_movement/test_dropout.py`, `tt-train/tests/ttnn_fixed/dropout_op_test.cpp` (`DropoutTest.*`), and `tt-train/tests/core/mesh_1x2_utils_test.cpp` (`Mesh1x2UtilsTest.DropoutDifferentSeed`, 2 devices). The last one is the only check that per-device seeds differ across devices.
