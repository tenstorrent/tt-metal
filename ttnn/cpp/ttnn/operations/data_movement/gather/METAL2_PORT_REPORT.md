# Metal 2.0 Port Report: `ttnn/cpp/ttnn/operations/data_movement/gather`

Links in this file resolve from the op directory, against the ported code. Legacy code is referenced as plain `path:line` at commit `e9385a86c5c` (the pre-port tree).

## Outcome

**PORTED**: all seven factories, in both device operations, are on `ProgramSpecFactoryConcept`.

- `GatherDeviceOperation`: `SingleRowSingleCore`, `SingleRowMultiCore`, `RmSingleRowSingleCore`, `RmSingleRowMultiCore` ([device/gather_program_factory.cpp:47-972](device/gather_program_factory.cpp#L47-L972)).
- `GatherCodegenDeviceOperation`: `GatherCodegenProgramFactoryInterleaved`, `GatherCodegenProgramFactoryTiled`, `GatherCodegenProgramFactoryStreaming` ([codegen/gather_codegen_program_factory.cpp:259-785](codegen/gather_codegen_program_factory.cpp#L259-L785)).
- All 14 kernels converted in place. No shared kernel, no fork.

Test result: the confirmed test set gives the same result before and after the port: 288 passed and 1 expected failure (`test_gather_codegen_trace_capture_under_pressure[warm_then_pressure]`, issue #46533), and the gtest `CodegenGatesUnallocatedFixture.GatesAnswerForDeallocatedTensor` passes. Both runs had Watcher on and the Metal 2.0 legality checks forced on, cache-hit checks included. The deepseek mesh and trace test, run on the ported code only, also passes (2 of 2) on a 1x2 n300 mesh under the same conditions. Details under Test coverage notes.

## Provenance

- **Recipe docs (this port):** `553eadc5591 2026-10-29 docs(metal_2.0): per-coordinate ports onto MeshWorkloadSpecFactoryConcept; legacy rows RED`
- **Audit docs (inherited):** `553eadc5591 2026-10-29 docs(metal_2.0): per-coordinate ports onto MeshWorkloadSpecFactoryConcept; legacy rows RED`

## User overrides (from the audit; both applied)

1. **Codegen readiness rows.** The readiness sheet has no rows for `GatherCodegenDeviceOperation`. On the user's instruction (2026-10-29) the audit treated its three factories as present and ready, with target `ProgramSpecFactoryConcept`. The port follows that.
2. **Device 2.0 fixes in the port.** On the user's instruction (2026-10-29), the port fixed the 6 Device 2.0 holdovers in the row-major kernels: `noc,` inserted as the first argument of each call to the deprecated `tt::data_movement::common::noc_async_read_sharded` / `noc_async_write_sharded` overloads. These are changes outside the kernel-side whitelist, made by user direction. They are in their own commit `e9385a86c5c`, before any Metal 2.0 change, and the commit holds only the 6 insertions. The sites, at their post-port lines:
   - [device/kernels/dataflow/gather_reader_rm_single_row_single_core.cpp:48-54](device/kernels/dataflow/gather_reader_rm_single_row_single_core.cpp#L48-L54) (`noc_async_read_sharded`)
   - [device/kernels/dataflow/gather_writer_rm_single_row_single_core.cpp:44-50](device/kernels/dataflow/gather_writer_rm_single_row_single_core.cpp#L44-L50) (`noc_async_read_sharded`)
   - [device/kernels/dataflow/gather_writer_rm_single_row_single_core.cpp:55-61](device/kernels/dataflow/gather_writer_rm_single_row_single_core.cpp#L55-L61) (`noc_async_write_sharded`)
   - [device/kernels/dataflow/gather_reader_rm_single_row_multi_core.cpp:50-56](device/kernels/dataflow/gather_reader_rm_single_row_multi_core.cpp#L50-L56) (`noc_async_read_sharded`)
   - [device/kernels/dataflow/gather_writer_rm_single_row_multi_core.cpp:47-53](device/kernels/dataflow/gather_writer_rm_single_row_multi_core.cpp#L47-L53) (`noc_async_read_sharded`)
   - [device/kernels/dataflow/gather_writer_rm_single_row_multi_core.cpp:58-64](device/kernels/dataflow/gather_writer_rm_single_row_multi_core.cpp#L58-L64) (`noc_async_write_sharded`)

## TTNN ProgramFactory

### Concept realized

`ProgramSpecFactoryConcept` for all seven factories, as the audit chose. Each factory defines only `create_program_artifacts` and returns a `ProgramArtifacts` with an empty `op_owned_tensors`. No factory has an `override_runtime_arguments`, so on a cache hit the framework refreshes the three tensor bindings and nothing else, which matches the legacy descriptor path (it patched only the `Buffer*` runtime args).

### Device-op-class edits

- Pybind entry points removed: none.
- Custom `compute_program_hash`: left intact at [codegen/gather_codegen_device_operation.cpp:55-69](codegen/gather_codegen_device_operation.cpp#L55-L69). The port does not change that file.
- The four native factory structs are declared inside `GatherDeviceOperation`. Their declarations changed from `create_descriptor` to `create_program_artifacts` ([device/gather_device_operation.hpp:24-43](device/gather_device_operation.hpp#L24-L43)), and the header now includes `ttnn/metal_v2_artifacts.hpp` in place of `tt-metalium/program_descriptors.hpp`. Nothing else in the device-operation class changed.

### Open items

- Relaxation candidates: none. Every kernel takes its tensor extents as compile-time args, so none would tolerate a relaxed `TensorSpec` match.
- No capability gap and no friction with the concept fit.

## Handoff points

None. Nothing in the port needed a change outside the op directory, no out-of-op call site needed a `sem::` or `tensor::` handle, and no pybind surface was removed. The doc gaps found during the port are under Friction; the follow-ups for the op owners are under Open items.

## Successes

- **Endpoint procedure** ([patterns catalog, endpoint-assignment procedure](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-two-toucher-dfb--assign-1p1c-dual-instance-work-split)). Re-deriving the endpoints from the kernel bodies gave the same result as the brief in all seven factories: input and output are plain 1:1 pairs, and the index DFB is touched only by the reader, so it is a self-loop ([device/gather_program_factory.cpp:150-197](device/gather_program_factory.cpp#L150-L197) is the first of seven such reader specs). The brief's note that the row-major readers pop the index DFB without a `wait_front` kept the kernel logic unchanged.
- **`constexpr` metadata** ([CB→DFB whitelist §A](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/cb_dfb_api_whitelist.md#tile--format-metadata-jit-descriptors)). The rule "the legacy declaration is the entire test" settled all 32 metadata declarations without judgment calls: each stays `constexpr` and passes the `dfb::` token to the same free function.
- **Third accessor argument** ([recipe, kernel-side whitelist rule 3](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#kernel-side-whitelist)). The brief's classification held up when checked against the code: the interleaved accessor never reads its page-size member (`noc_async_read_sharded` takes the interleaved branch), and for sharded row-major tensors the binding's aligned page equals the legacy per-shard value because validation rejects unaligned shard rows.
- **Forcing the legality checks and proving it** ([recipe](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#ensure-the-metal-20-host-side-legality-checks-are-enabled)). The markers showed that both translation units were rebuilt, and they exposed the grep gap listed under Friction. TTNN calls `UpdateTensorArgs` with `skip_validation = !validate_program_args`, and `validate_program_args` defaults to false, so without the force the cache-hit `TensorSpec` checks would not have run.
- **Guard count check** ([recipe self-audit](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#anti-pattern-self-audit)). The per-file `TT_FATAL` / `TT_ASSERT` / `TT_THROW` counts over the op directory are unchanged (5 files; the factory files hold no guards).

## Friction

### Gaps

1. **The legality-check grep misses the spec-side gate.** `port/metal2_port.md` says to force every function that `grep -n 'bool skip_validation' tt_metal/impl/metal2_host_api/*.cpp` names. That glob does not recurse, and `BuildProgramFromSpec`, `MakeProgramFromSpec`, `MakeMeshWorkloadFromSpecs` and `MakeMeshWorkloadFromSpec` are in `tt_metal/impl/metal2_host_api/program_spec/program_spec.cpp`. A porter who follows the step literally forces only `program_run_args.cpp` and then finds one marker where the recipe expects two. Fix: `grep -rn 'bool skip_validation' tt_metal/impl/metal2_host_api/`, which finds 9 functions in 2 files today.
2. **Stale DM config names in the recipe.** Hardware configuration shows `create_reader_datamovement_config(device->arch())` and `DataMovementGen1Config{...}`. In this tree the helper is `ttnn::create_reader_datamovement_config(bool disable_dfb_implicit_sync_for_all = false)` ([../../core/data_movement_kernel/datamovement_kernel_config.hpp:24-38](../../core/data_movement_kernel/datamovement_kernel_config.hpp#L24-L38)), and the Gen1 struct is `DataMovementHardwareConfig::DataMovement1XXConfig` (field `config_1xx`). Passing `device->arch()` would not compile. The migration guide's `CreateReaderGen1DataMovementConfig()` is `CreateReaderDataMovementConfig()` in the header.
3. **Token type name.** The recipe and the whitelist call the kernel-side token `DFBAccessor` (`DFBAccessor::operator uint32_t()`); the type is `DFBBindingToken` (`tt_metal/hw/inc/api/dataflow/dfb_binding_token.h`).
4. **`get_dataformat(dfb::x)` changes its data source.** Whitelist §A says to keep the free function and pass the token. For `get_tile_size` and `get_tile_hw` that resolves to the legacy `int32_t` overload, which reads the JIT descriptor arrays. For `get_dataformat` it resolves to the token overload in `dfb_binding_token.h` (an exact match beats the conversion), which reads the host-side `data_format_metadata` and fails constant evaluation when that field is unset. In gather the value is the same (the factory sets the field to the legacy CB format) and the local is unused, but the whitelist should say this.
5. **The `cb` self-audit cannot reach zero over the op directory.** The recipe expects zero hits for `grep -rnE '[Cc][Bb]_|_[Cc][Bb]\b|\b[Cc][Bb]\b|\bCB[A-Z]' <op-dir>`. Here the remaining hits are of three kinds, none a leftover DFB name:
   - files the port may not edit (device-operation side and docs), listed under Open items;
   - the helper `gather_output_cb_tiles`, kept because off-limits files name it (Open items);
   - comments that describe the runtime allocator's circular-buffer region, which on Gen1 still holds the DFBs and is checked by `Program::validate_circular_buffer_region` ([codegen/gather_codegen_program_factory.cpp:42-74](codegen/gather_codegen_program_factory.cpp#L42-L74)).
   Suggest scoping the grep to the files the port changes and naming which kinds of hit are acceptable.
6. **The scaffolding check matches the recipe's own text.** The self-audit runs `git diff "$BASE" | grep -nE 'METAL2_CHECKS_FORCED|DO NOT COMMIT'` and expects no output. On a branch that also carries the `docs/.../metal_2.0/` commits, as this one does, the recipe's own description of the scaffolding produces 6 hits. Scoping the diff to the code (`git diff "$BASE" -- ttnn/ tt_metal/`) gives 0 here.

### Confusion

7. **Idle-core branches in the brief.** The brief said to drop the address slots "in both branches" and keep the branch comments. After the address and page-size slots go, the two branches emit identical values in all four native factories (the idle branch runs only when the work count is 0), and the comments describe `Buffer*` bindings that no longer exist. The port emits the args once without the branch ([device/gather_program_factory.cpp:247-265](device/gather_program_factory.cpp#L247-L265) and the three other native factories). This is the one place the port departs from the brief's wording; the plan records it under Deferred / Flagged.
8. **Unity builds are off in the recommended local build.** `./build_metal.sh -e` sets `CMAKE_EXPORT_COMPILE_COMMANDS`, and `cmake/project_options.cmake:58-61` then disables unity builds. So the local build cannot catch the anonymous-namespace collisions that the patterns catalog's unity-build entry warns about; only CI can. The port avoids the risk by declaring all spec names inside each factory function.
9. **Removing an include that the port made unused exposed a transitive dependency.** `ttnn/operations/data_movement/common/common.hpp` was included only for `per_shard_page_size_bytes`, but `tt::div_up` also reached the native factory through it. The factory now includes `tt-metalium/math.hpp` directly ([device/gather_program_factory.cpp:13](device/gather_program_factory.cpp#L13)).
10. **What the markers prove.** In this test set both markers fire hundreds of times even when no gather spec is built (605 each in the baseline run, where gather was still on the descriptor API), because other ops on the gather path (transpose and its neighbors) are already on Metal 2.0. A marker proves that a translation unit was rebuilt, not that the op under test went through it. Comparing the counts does show it: 898 per file after the port against 605 before, so 293 gather spec builds went through the forced checks.

## Open items for downstream

### Shared kernel touches

None. Each of the 14 kernels is bound by exactly one factory of this op, no other op binds any of them, and no `_metal2` fork exists or was created.

### Records the brief asked for

- **Dropped runtime-arg slots** (row-major page sizes; each fed only the accessor's 3rd argument):
  - `RmSingleRowSingleCore`: reader RTA 3 `input_index_per_shard_page_size_bytes`; writer RTA 4 `input_per_shard_page_size_bytes` and RTA 5 `output_per_shard_page_size_bytes`. Host values and emission were `device/gather_program_factory.cpp:480-510` at `e9385a86c5c`.
  - `RmSingleRowMultiCore`: reader RTA 4; writer RTA 5 and RTA 6. Host values and emission were `device/gather_program_factory.cpp:644-678` at `e9385a86c5c`.
  - With these gone, `using ttnn::operations::data_movement::per_shard_page_size_bytes;` and the `common.hpp` include were removed from the native factory.
- **Dropped 3rd accessor arguments, codegen** (9 sites): the `constexpr ..._tile_bytes` locals stay, since each is also a transfer size or SRAM offset step.
- **`constexpr` token-form metadata sites** (Gen1-only; Quasar-uplift debt, since a DFB token's `uint32_t` conversion is documented as Gen1-only):
  - [device/kernels/dataflow/gather_reader_single_row_single_core.cpp:132-141](device/kernels/dataflow/gather_reader_single_row_single_core.cpp#L132-L141)
  - [device/kernels/dataflow/gather_writer_single_row_single_core.cpp:52-53](device/kernels/dataflow/gather_writer_single_row_single_core.cpp#L52-L53)
  - [device/kernels/dataflow/gather_reader_single_row_multi_core.cpp:72-81](device/kernels/dataflow/gather_reader_single_row_multi_core.cpp#L72-L81)
  - [device/kernels/dataflow/gather_writer_single_row_multi_core.cpp:44-45](device/kernels/dataflow/gather_writer_single_row_multi_core.cpp#L44-L45)
  - [codegen/kernels/gather_reader.cpp:43-54](codegen/kernels/gather_reader.cpp#L43-L54)
  - [codegen/kernels/gather_writer.cpp:41-45](codegen/kernels/gather_writer.cpp#L41-L45)
  - [codegen/kernels/gather_reader_tiled.cpp:55-65](codegen/kernels/gather_reader_tiled.cpp#L55-L65)
  - [codegen/kernels/gather_writer_tiled.cpp:47-50](codegen/kernels/gather_writer_tiled.cpp#L47-L50)
  - [codegen/kernels/gather_reader_streaming.cpp:53-64](codegen/kernels/gather_reader_streaming.cpp#L53-L64)
  - [codegen/kernels/gather_writer_streaming.cpp:42-46](codegen/kernels/gather_writer_streaming.cpp#L42-L46)
- **DM self-loops** (Quasar-uplift debt): the index DFB on the reader of every factory, seven in total.
- **Retained varargs:** none. **Dead CBs dropped:** none. **Guards lost with a deleted subject:** none.

### Follow-ups for the op owners (not fixed: outside the port's writeable surface)

- **Comments that name the replaced CB constructs**, now stale:
  - [codegen/gather_codegen_supported.cpp:21](codegen/gather_codegen_supported.cpp#L21) and [codegen/gather_codegen_supported.cpp:55](codegen/gather_codegen_supported.cpp#L55) name `make_tile_cb()` (now `make_tile_dfb()`) and `CBFormatDescriptor::tile` (now the unset `DataflowBufferSpec::tile_format_metadata`).
  - [codegen/gather_codegen_device_operation.cpp:172-173](codegen/gather_codegen_device_operation.cpp#L172-L173) names `make_tile_cb`.
  - Other comments that call this op's buffers CBs, which are now DFBs: [codegen/gather_codegen_device_operation.cpp:17](codegen/gather_codegen_device_operation.cpp#L17), [:40](codegen/gather_codegen_device_operation.cpp#L40), [:58](codegen/gather_codegen_device_operation.cpp#L58), [:83](codegen/gather_codegen_device_operation.cpp#L83) (inside a `TT_FATAL` message), [:188](codegen/gather_codegen_device_operation.cpp#L188); [codegen/gather_codegen_supported.cpp:36](codegen/gather_codegen_supported.cpp#L36) and [:109-110](codegen/gather_codegen_supported.cpp#L109-L110).
  - [docs/Gather.md:171-173](docs/Gather.md#L171-L173), [docs/Gather.md:208-210](docs/Gather.md#L208-L210) and [docs/Gather.md:320-328](docs/Gather.md#L320-L328) describe a `CircularBufferConfig` setup that was already out of date before the port.
- **`gather_output_cb_tiles` keeps its name** ([codegen/gather_codegen_program_factory.hpp:34-41](codegen/gather_codegen_program_factory.hpp#L34-L41)). The routing side and the tests name it ([codegen/gather_codegen_supported.cpp:110](codegen/gather_codegen_supported.cpp#L110), `tests/ttnn/unit_tests/operations/data_movement/test_gather.py:378-409`), so a rename belongs with those files. The kernel variable `OUT_CB_DEPTH` became `OUT_DFB_DEPTH`.
- **Unused include:** [codegen/gather_codegen_device_operation.hpp:12](codegen/gather_codegen_device_operation.hpp#L12) still includes `tt-metalium/program_descriptors.hpp`; my guess is that nothing in that header needs it any more.
- **CRTA candidates, not converted:** `tile_width` and `tile_height` hold the same value on every node in the TILE and codegen readers. Moving them to common runtime args changes dispatch, so it is a separate change.
- **Unused args kept** (audit, Misc anomalies): `input_index_tensor_is_dram`, `input_tensor_is_dram`, `output_tensor_is_dram`, `compute_with_storage_grid_size_x` / `_y`, `Ht` in the single-core TILE kernels, `H` / `W_input` / `W_index` and `core_id` in some row-major kernels. They stay as named args, because dropping a compile-time arg changes how many kernel binaries the JIT builds.

### Test coverage notes

- Test set, confirmed with the user: `tests/ttnn/unit_tests/operations/data_movement/{test_gather.py, test_tosa_gather.py}`, `tests/ttnn/nightly/unit_tests/operations/data_movement/{test_gather.py, test_gather_codegen_routing.py, test_gather_invalid_indices.py, test_universal_input_tm_gather.py}`, `tests/ttnn/docs_examples/test_data_movement_examples.py::test_gather` (289 tests), and the gtest `CodegenGatesUnallocatedFixture.*` in `unit_tests_ttnn`.
- Baseline (pre-port tree `e9385a86c5c`, same build flags, same forced checks): 288 passed, 1 xfailed, gtest passed. Post-port: 288 passed, 1 xfailed (the same test), gtest passed. The 14 expected `TT_FATAL` lines of the negative tests are the same in both runs; Watcher reported nothing.
- Every factory ran on the ported code: all 14 gather kernels have JIT cache files written during the post-port run (`~/.cache/tt-metal-cache/*/kernels/<kernel>/`).
- Cache hits with new tensor addresses ran with the forced `UpdateTensorArgs` checks. Codegen: `test_gather_invalid_indices.py` (keeps earlier allocations alive, so a stale binding would show) and `test_gather_codegen_routing.py::test_gather_codegen_program_cache_hit`. Native: the second forced-native call on a new tensor in `test_gather_codegen_routing.py:105-111`.
- Mesh and trace: `models/demos/deepseek_v3/tests/unit/test_gather.py` ran after the port only, with `MESH_DEVICE=N300` (a 1x2 mesh on one n300 board, fabric `FABRIC_1D`), Watcher on and the same forced checks. Result: 2 passed (trace on and trace off), and the outputs of both devices match torch. Watcher reported nothing.
  - Both cases select `GatherCodegenProgramFactoryInterleaved` ([codegen/gather_codegen_program_factory.cpp:259-431](codegen/gather_codegen_program_factory.cpp#L259-L431)). Its two kernels, [codegen/kernels/gather_reader.cpp](codegen/kernels/gather_reader.cpp#L1-L178) and [codegen/kernels/gather_writer.cpp](codegen/kernels/gather_writer.cpp#L1-L112), are the only gather kernels compiled during the run.
  - The trace-on case calls gather twice: once to compile, then once inside the trace capture. The trace-off case calls gather once. Both cases log 3 `BuildProgramFromSpec` markers, so the second gather call of the trace-on case built no spec. That call was a program-cache hit, it went through the forced `UpdateTensorArgs` checks, and the trace replayed it.
- The forced-legality changes in `tt_metal/impl/metal2_host_api/` were in the builds for the baseline run, the post-port run and the deepseek run, and are not in any commit.
