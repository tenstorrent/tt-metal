# Port Report — `data_movement/reshape_on_device`

## Outcome

**`PORTED`.** Both factories of `ReshapeDeviceOperation` now satisfy `ProgramSpecFactoryConcept`: `ReshapeRMProgramFactory` and `ReshapeTileProgramFactory`. Nothing is left on the legacy concept.

Across the 11 pytest groups (the tests that reach the op's kernels plus the tests of its neighbours) the result sets are identical before and after the port: 588 passed, 318 skipped, 0 failed on both sides, on a Wormhole n150 with Watcher on. One configuration could not run here: the Blackhole-only scratch path (see *Verification* and *Open items*). It was compiled, and run in an emulation on Wormhole, but it has not run on Blackhole silicon.

The port is seven files (`+341 / -224`): the two factory `.cpp` / `.hpp` pairs and the three op-owned kernels. The tile writer is the existing `_metal2` fork of `eltwise/unary`'s kernel; no kernel was forked or edited outside this directory.

## Provenance

- **Recipe docs (this port):** `960cab58ff2f 2026-11-24 Add reshape_on_device TensorParameter relaxation analysis`
  (verbatim output of the recipe's provenance command). That is this branch's own analysis commit, the last one to touch `docs/.../metal_2.0/`, which means it does not name a recipe version. The recipe was read in place. The six documents this port followed are byte-identical to upstream `akertesz/op-porting-recipe` at `c41f7f4ef0a0` (2026-10-29): port recipe `aad3eedddf74`, audit `6435c8fcae15`, port patterns `7ba67547a9d7`, TTNN factory `bd5f68d742c7`, CB→DFB whitelist `798733a9c6c9`, migration guide `009627b34a8d`. `git diff --stat origin/akertesz/op-porting-recipe HEAD -- <metal_2.0 dir>` lists only the added relaxation analysis.
- **Audit docs (inherited):** ``547b694ae6f8 2026-11-23 docs(metal_2.0): DFBAccessor -> DFBBindingToken; drop the dead `.id` form`` (copied from the brief; after this branch's rebase the same commit is `df870cca23d`).
- **Code base:** `origin/main` `8294ab326a5`, plus the branch's recipe, analysis and audit commits (`4571c481b8f` before the port).

## TTNN ProgramFactory

### Concept realized

`ProgramSpecFactoryConcept` for both factories, as the audit decided. Neither factory has an `override_runtime_arguments`. A cache hit therefore refreshes the tensor bindings and nothing else. One spec is stamped across the mesh. Each factory has one `WorkUnitSpec`, `main`.

| factory | DFB spec | entry size × entries | reader binding | writer binding | tensor bindings |
|---|---|---|---|---|---|
| RM | `src0` | `new_stick_size` × `num_pages * max_page_size / new_stick_size` | PRODUCER, `dfb::in0` | CONSUMER, `dfb::out0` | reader `tensor::src` (input), writer `tensor::dst` (output) |
| Tile | `in0` | `single_tile_size` × 2 | PRODUCER, `dfb::in0` | CONSUMER, `dfb::out` (the fork's name) | reader `tensor::src0` (input), writer fork `tensor::dst` (output) |
| Tile, conditional | `scratch` | `alignment` × 1 | reader PRODUCER and CONSUMER, `dfb::intermed` (self-loop) | none | none |

The `scratch` DFB, its binding and the reader's `WIDE_ALIGNMENT_SCRATCH` define all follow one host condition, `alignment > FACE_WIDTH * element_size`. The kernel `#ifdef`s its handle and the one ternary that reads it; `MISALIGNED` stays a `constexpr` built from the `ALIGNMENT` and `ELEMENT_SIZE` CTAs.

Both kernels of each factory use the default reader and writer hardware config helpers. No `opt_level` is set: the legacy descriptors resolved to O2 and so does Metal 2.0. No compute kernel exists.

### Device-op-class edits

- Pybind entry points removed: none. `reshape_nanobind.cpp` binds only the user-facing function.
- Custom `compute_program_hash`: **left intact** at `device/reshape_op.cpp:111-127`. `reshape_op.cpp`, `reshape_op.hpp`, `reshape_device_operation_types.hpp` and `reshape.cpp` are unchanged. The `TT_FATAL` census agrees (`reshape_op.cpp` 11 = 11, `reshape.cpp` 4 = 4).
- Guard deleted with its subject: `reshape_tile_program_factory.cpp` had `TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!")`. It guarded the raw `Buffer*` that the output `TensorBinding` replaced. It is the only entry in the census whose count dropped (1 → 0). Both `TT_FATAL`s of the RM factory are kept as they were.

### Relaxation declaration

Transcribed from the relaxation analysis for this op: the **input** `TensorParameter` of both factories declares `{.match_padded_shape_only = true, .relax_tensor_layout_alignment = true}`; the **output** stays an exact match; `dynamic_tensor_shape` is not used. The analysis's six validity checks were run against this tree before the declaration was written, and all six pass:

1. The key pins the input as the declaration matches it: `compute_program_hash` keys the input with `hash_tensorspec_with_relaxation` and the same two flags. In `tt_metal/impl/metal2_host_api/tensor_spec_relaxations.cpp` both the hash and the match pin dtype, page config (with `Tile`), memory config and padded shape, and free logical shape and alignment.
2. The output spec is a pure function of the key (`compute_output_specs`; the op has no preallocated output).
3. No logical-shape or alignment read in `device/` or in any kernel: the only hit for `alignment` is `hal::get_dram_alignment()` / `hal::get_l1_alignment()`, a buffer-type alignment.
4. Still the base concept: no `override_runtime_arguments`, `get_dynamic_runtime_args` or `validate_on_program_cache_hit` in the op directory.
5. Validation reads only pinned terms. `handle_mesh_adapter_cache_hit` (`ttnn/api/ttnn/device_operation.hpp:374-378`) runs the miss validator on every hit when no hit validator exists.
6. Both factories are interleaved-only and chosen by the input layout (`reshape_op.cpp:21-27`, `:43-48`).

Main moved by 84 commits after the audited base (`e39e42718ca`). The op directory and `tensor_spec_relaxations.cpp` are unchanged since. The spec headers moved from `tensor/spec/` to `memory/tensor/spec/` byte-for-byte (the old paths are now forwarding headers), and `device_operation.hpp` gained only a miss-validator call on its skip-launch paths.

The tile factory reads the buffer type for its alignment choice from `input_tensor.memory_config().buffer_type()` instead of the raw `Buffer*`. The validator requires an interleaved input, for which the two are the same value.

## Handoff points

- **Readiness-sheet relaxation cell.** Both sheet rows read `Padded Shape + Alignment` at audit time, which is outside the vocabulary the audit accepts (`none` / `dynamic (see analysis)` / `legality (see analysis)`). The audit cleared GREEN on the user's direction to read the cell as `legality (see analysis)`, and this port proceeded on that clearance. The cell itself still needs the sheet owner or the ops team to change it (the analysis document says it should read `legality (see analysis)`), or the recipe vocabulary needs to cover the value. I did not re-fetch the sheet in this session.

No other handoff: no kernel-lib gap, no boundary-rule violation, no framework gap, no removed pybind surface.

## Successes

- **The audit's "unguarded scratch handle" warning and the conditional-bindings pattern fixed the one structural trap.** The tile reader built its scratch handle in every configuration, while the factory builds the scratch buffer in one family of configurations. Following *Conditional / optional resource bindings* (promote the gate to a define, share one condition between binding and define) gave a kernel that compiles with and without the token. Both variants were built. In the normal Wormhole build none of the 4 compiled tile-reader variants has an `intermed` token in its generated bindings header. In the forced build exactly 1 of 4 has the self-loop token, the BFLOAT16 DRAM variant where the scratch DFB applies.
- **Shared kernel, rung 1, fit table.** The brief's fit table for `writer_unary_interleaved_start_id_metal2.cpp` was exact (`dfb::out`, `tensor::dst`, named RTAs `num_pages` and `start_id`, no defines needed). No fork was created or edited.
- **The hit-path warning in the brief paid off.** It said the input relaxation is only proven live with `validate_program_args` on. Experiment E1 and E2 below show precisely that: the same defect fails with the knob and passes silently without it.

## Friction

- **Recipe: header paths and namespaces are stale.** The recipe points at `tt_metal/api/tt-metalium/experimental/metal2_host_api/` for the spec headers and writes `tt::tt_metal::experimental::ProgramSpec` / `ProgramRunArgs`. On this tree the spec headers are under `tt_metal/api/tt-metalium/workload/specification/` and `workload/run_args/`, the names are in `tt::tt_metal`, and only `advanced_options.hpp` is still under `experimental/metal2_host_api/`. The recipe's relative links to those headers do not resolve. `ttnn/api/ttnn/metal_v2_artifacts.hpp` is the header for `ProgramArtifacts`.
- **Recipe: `create_reader_datamovement_config(device->arch())` is stale.** The helper in `ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp` takes no architecture; it takes an optional `disable_dfb_implicit_sync_for_all` flag. The default call is right for a default reader or writer. Its resolved values were checked against the legacy lowering in `tt_metal/impl/program/program.cpp` and `kernel_types.cpp`: reader RISCV_1 / NOC_0 / dedicated, writer RISCV_0 / NOC_1 / dedicated, both O2.
- **Recipe: the `skip_validation` forcing step cannot be applied.** `grep -n 'bool skip_validation' tt_metal/impl/metal2_host_api/*.cpp` finds only the deprecated `MergeProgramRunArgs(..., bool skip_validation)` overload (`program_run_args.cpp:2028`). The gates are `ValidationMode` parameters. The TTNN adapter validates every cache miss unconditionally, and `validate_program_args` (default `false`, `ttnn/api/ttnn/config.hpp:31`) gates only the hit path (`mesh_device_operation_adapter.hpp:54-57`). Nothing in `tt_metal/impl` was edited. Every verification run used `TTNN_CONFIG_OVERRIDES='{"validate_program_args": true}'`, and liveness is shown by the experiments below.
- **Audit: its baseline test selection missed the RM-path tests.** `-k "reshape_on_device or on_device"` on `test_reshape.py` selects 17 cases. Sixteen are `test_reshape_in_4D_on_device`, which calls the `ttnn.reshape` view and never reaches this op. The five tests that do call `ttnn.reshape_on_device` on a row-major tensor (`test_reshape_cw_div2_rm`, `test_reshape_cw_mul2_rm`, `test_reshape_hw_div2_rm`, `test_reshape_hw_mul2_rm`, `test_reshape_hw_rm_with_program_cache`) have no "on_device" in their names and were not selected. A name filter is not a reachability filter. This port's test set includes them, and proves reachability by the kernels a fresh-cache run compiles.
- **Brief versus recipe name sweep.** The brief names the RM runtime argument `num_sticks_per_cb_push`, the kernels' own local. The recipe's self-audit expects zero `cb` hits in the op directory. The port therefore uses `num_sticks_per_dfb_push` and renames the matching host variables and kernel locals (`cb_write_offset`, `cb_read_offset`, `cb_data_format`, `num_*_per_cb_push`).
- **Recipe provenance command on a branch that carries an analysis doc.** It prints the analysis commit, not the recipe version. Comparing blobs with upstream, as done above, pins the version.
- **Sweeps have no recipe guidance, and the tooling has two traps.** `sweeps_runner.py` stops at once unless `ARCH_NAME=wormhole_b0` (or `blackhole`) is set. `sweeps_parameter_generator.py` takes one module and overwrites `generation_manifest.json` each run. Generating three modules in a row leaves a manifest that lists only the last, and the other two sweeps then execute zero vectors and still exit 0. Generate and run per module, and check the executed-vector count.
- **IDE noise.** The editor's language server has no include paths for this repository and reported dozens of errors on every file, including on stale copies of the old content. None of them was used; the build is the check.

## Open items for downstream

- **Shared kernel touches.**
  - `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (the tile writer): rung 1, **reused the existing `_metal2` fork** beside it (`writer_unary_interleaved_start_id_metal2.cpp`); no new file; the legacy original already carries the pointer comment and was not touched. Remaining unmigrated consumers of the legacy copy under `ttnn/cpp`: `eltwise/unary_backward/device/unary_backward_program_factory.cpp:343`, `examples/example/device/multi_core_program_factory.cpp:81`, `examples/example/device/single_core_program_factory.cpp:83`, plus the test and model binders the audit lists (tracking issue #52228). `matmul/` and `slice/` have private copies and are not consumers.
  - The op's three own kernels have no other binder in the repository (filename census, hits checked by bound path). They were converted in place.
- **Quasar uplift (not authored, as the recipe says).**
  - The tile reader's `scratch` DFB is a DM self-loop. It is legal on Gen1, and Gen2 rejects DM self-loops.
  - The RM reader and writer move sticks: explicit `reserve_back` / `push_back` around several sub-entry NoC transfers at non-zero offsets inside the DFB. The comment on `create_reader_datamovement_config` in `datamovement_kernel_config.hpp` says that DM kernels doing many sub-tile ("stick") transfers stall Quasar's implicit credit accounting, which is what its `disable_dfb_implicit_sync_for_all` flag is for. The port sets no opt-out, because the recipe leaves that to the Quasar pass.
- **Audit anomalies, routed and re-confirmed at their current lines; none touched by the port.**
  - `device/reshape_op.cpp:51-53`: stale comment, "the tile is absent from compute_program_hash". Since #63798 the key hashes the page config, which carries the `Tile`. The 32×32 rejection under it is still right.
  - `device/reshape_op.cpp:81-86`: the row-major path has no input/output padded-volume check, unlike the tile path (`:72-80`, #52828).
  - `reshape.cpp:28-33`: `TT_ASSERT` compiles out of release builds. The logical-volume check in `manual_insertion` is therefore a no-op there.
  - `reshape.cpp:59`: the comment says "Will do a tensor copy" where the code returns a view.
  - `reshape_nanobind.cpp:37`: the docstring says the X dimension of input and output must match, but equal-X is routed to the view path before the device op runs.
  - `data_movement/sources.cmake:270,272`: `reshape_on_device/reshape_nanobind.cpp` is listed twice.
- **Test coverage notes.**
  - The Blackhole-only path (scratch DFB plus the volatile byte-copy loop) has not run on Blackhole silicon.
  - Of the audit's test list, the selected `test_grid_sample.py` cases, `test_split.py`, `test_universal_input_tm_split.py`, `test_zero_divisor_guards.py`, `test_maxpool2d_shard_grid.py`, `test_comparison_mode_context.py` and the three sweeps do **not** reach the op's kernels on this machine (see the table). They are non-regression runs only. In particular nothing run here reaches the batched last-dim two-chunk tiled split in `split.cpp:43,52`, one of the op's three C++ callers.
  - The model-traced sweeps' master configuration file is not in the repository. Only their one-vector sample suites exist.
  - No gtest names the op. Two CCL gtests contain "reshape" in the mesh-partition sense and need a mesh.
  - Eleven files under `models/experimental/` call `ttnn.reshape_on_device`; none was run (no weights, legacy code).
  - Row-major relaxation (input logical shape or alignment differing at the same padded shape) has no Python route. The row-major declaration is therefore inert in tests, as the analysis predicted.
- **Not measured:** device and host performance. The recipe has no perf step. The device kernels read the same values from the same L1 slots (named arguments are fixed offsets), and the TensorAccessor configuration is unchanged. No device-time change is expected; a ttperf before/after on the three shapes is cheap if wanted.

## Verification

All runs: Wormhole n150, `TT_METAL_WATCHER=10`, `TTNN_CONFIG_OVERRIDES='{"validate_program_args": true}'` (and `enable_fast_runtime_mode: false` for the one comparison-mode test), a fresh JIT cache per group. The legacy baseline was taken first, on the unmodified tree and its existing build, before any kernel or factory edit. The port run used the staged, pre-commit-clean tree and the build made from it. One comment-only trim in the tile factory came after that run; the build was repeated on the final text and the smoke re-run on it (22 + 13 + 1 + 2 cases pass: the g01 selection, tt_eager `test_reshape.py`, the end-to-end case, and the two `test_fold_op.py` cases that call the op).

### Before and after

| group | selection | legacy | port | op kernels compiled (RM reader, RM writer, tile reader) |
|---|---|---|---|---|
| g01 | `base_functionality/test_reshape.py`: the five RM tests, `test_reshape_on_device_tile_same_padded_shape_different_alignment`, `test_reshape_in_4D_on_device` | 22 pass | 22 pass | yes, yes, yes |
| g02 | `tt_eager/.../misc/test_reshape.py`, in full | 13 pass | 13 pass | yes, yes, yes |
| g03 | `operations/data_movement/test_zero_divisor_guards.py` | 9 pass | 9 pass | no |
| g04 | `operations/pool/test_maxpool2d_shard_grid.py` | 16 pass | 16 pass | no |
| g05 | `base_functionality/test_comparison_mode_context.py -k test_fold_legacy_transpose_with_asymmetric_padding_in_comparison_mode` | 1 pass | 1 pass | no |
| g06 | `end_to_end_tests/test_unit_ops.py` | 1 pass | 1 pass | tile reader only |
| g07 | nightly `test_universal_input_tm_fold.py` | 97 pass, 1 skip | 97 pass, 1 skip | RM reader and writer |
| g08 | nightly `test_universal_input_tm_split.py` | 88 pass | 88 pass | no |
| g09 | nightly `test_split.py` | 102 pass, 56 skip | 102 pass, 56 skip | no |
| g10 | nightly `pool/test_grid_sample.py -k test_grid_sample_oft` | 7 pass, 9 skip | 7 pass, 9 skip | no |
| g11 | `conv/data_movement/test_fold_op.py`, in full | 232 pass, 252 skip | 232 pass, 252 skip | RM reader and writer |
| sweeps | `split_pytorch2` nightly (40), `split_model_traced` sample (1), `fold_model_traced` sample (1) | 42 pass | 42 pass | no |

Totals for the 11 pytest groups: 588 passed, 318 skipped, 0 failed, on both sides. For every group the sorted `PASSED` / `FAILED` / `ERROR` / `XFAIL` / `XPASS` lines and the aggregated `SKIPPED` lines are identical. The op's three kernels compile in exactly the same groups on both sides. In the three tile-path groups (g01, g02, g06) the legacy run compiled `writer_unary_interleaved_start_id` and the port compiled `writer_unary_interleaved_start_id_metal2`, which is the intended switch. The sweep vectors are identical across the two runs by `input_hash` (42 of 42), every status is `pass`, and only the start and end timestamps differ.

The set of tests that actually exercises this op's kernels is therefore g01, g02, g06, g07 and g11. The rest guard neighbouring behaviour.

### Legality checks and the relaxation, proven live

Done with temporary env-switched edits in the two factories (removed again before the commit). Each ran on a fresh cache with Watcher on.

| id | change | knob | result | evidence |
|---|---|---|---|---|
| E0 | none (control) | on | pass | |
| E1 | input relaxations dropped (exact match) | on | **fail** on the second dispatch | `program_run_args.cpp:305: runtime_spec == expected_spec`, "TensorArgument for binding 'input' supplied a MeshTensor whose TensorSpec does not match the binding's declared spec" |
| E2 | same defect as E1 | off | pass | the hit-path check is skipped by default. The default configuration would not show this defect |
| E3a | tile reader's `in0` PRODUCER binding dropped | off | fail at the cache miss | `collect_metadata.cpp:92: !endpoint_info.producers.empty()`, "DFB 'in0' has no producer" |
| E3b | RM reader's `src0` PRODUCER binding dropped | off | fail at the cache miss | same check, "DFB 'src0' has no producer" |

E1 against E2 shows both that the input relaxation is load-bearing and that the knob is live; E3 shows the miss-time validator is live in these binaries. After the experiments the staged files restored the factories and kernel, the binary was rebuilt, and the switch string occurs 0 times in it. That binary passed the same smoke (22 + 13 + 1 cases), and setting the old switch on it changes nothing (1 pass).

### The Blackhole-only configuration, emulated on Wormhole

The scratch DFB and the byte-copy loop run only for a Blackhole BFLOAT16 DRAM input. To exercise them here, a temporary switch forced the DRAM alignment to 64 (Blackhole's value), which makes the factory declare the scratch DFB and the kernel take `MISALIGNED`.

- The generated `kernel_bindings_generated.h` of the tile reader contains the self-loop token `intermed` in 1 of its 4 compiled variants (BFLOAT16, DRAM), and in none of the 4 variants of the normal build.
- The tile tests passed under the forced build: tt_eager `test_reshape.py` 13 of 13 (its value checks compare against torch), three selected g01 cases (the tile cache test and two RM tests), and the end-to-end case.
- A temporary DPRINT in the kernel (one run, Watcher off) showed both branches of the misaligned configuration executing on core (0,0): the scratch byte-copy path (at least 512 hits) and the direct read.

This is an emulation. It does not prove Blackhole behaviour (NoC alignment rules, timing, the Blackhole DRAM read path), it forces only BFLOAT16 DRAM, and the DPRINT count is a floor. The byte-copy logic itself is untouched by the port. What changed is how the scratch handle is obtained and gated, together with the host condition and the binding, and the two variants exercised exactly those. A Blackhole run of g01 and g02 is the remaining check.

### Static self-audit

All sweeps are over the 14 `.cpp` / `.hpp` files of the op directory (2 factories, 3 kernels). Four of the patterns were shown to hit in the legacy files before the port: `emplace_runtime_args` (4 hits in the legacy RM factory), `Buffer*` (4, legacy tile factory), `TensorAccessorArgs` (1, legacy tile reader) and the `cb`-name regex (13, the two legacy RM kernels).

| check | hits |
|---|---|
| `buffer()->address()`, `->address()`, `emplace_runtime_args`, `Buffer*` in the factories | 0 |
| `CBIndex`, `CBDescriptor`, `CBFormatDescriptor`, `format_descriptors`, `.cbs`, `CircularBuffer` | 0 |
| `TensorAccessorArgs`, `get_compile_time_arg_val`, `get_arg_val`, `get_common_arg_val` in the kernels | 0 |
| the recipe's `cb`-name regex over all `.cpp` / `.hpp` | 0 |
| `allow_instance_multi_binding`, varargs (`get_vararg`, `num_runtime_varargs`, `compile_time_varargs`) | 0 |
| positional `compile_time_args` (all three sites use `{{"name", value}}`) | 0 |
| `skip_validation`, `METAL2_CHECKS_FORCED`, `DO NOT COMMIT` in the diff outside `docs/` (the 7 changed C++ files, the brief and the audit) | 0 |
| `tt_metal/` files in the diff | 0 |
| `.md` citations from the 7 changed C++ files | 0 |
| `TT_FATAL` / `TT_ASSERT` / `TT_THROW` census against `origin/main` | one legitimate loss (above) |
| `opt_level` / compute `KernelSpec`s | none set; no compute kernel |

Other checks: the four bound kernel sources (three own, one fork) all exist and each has an install rule in the regenerated `cmake_install.cmake` (4 of 4; a nonexistent path returns 0). No build-file edit is needed. `pre-commit` (clang-format, whitespace, the descriptor-factory checkers) passes on the final text; its only change was wrapping the long writer `.source` path.
