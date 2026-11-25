# Port Report — eltwise/unary_backward

## Outcome

**`PORTED`** — `UnaryBackwardProgramFactory`, the op's only factory, moved from `ProgramDescriptorFactoryConcept` to
`ProgramSpecFactoryConcept`. All three configurations of the factory (ALIAS, ADDRESS-TILE, ADDRESS-RM) are ported, with
every compute kernel the factory can select (41 of 41) and the two row-major dataflow kernels. One kernel fork was created
(`eltwise/binary/.../reader_binary_interleaved_start_id_metal2.cpp`); one existing fork was reused. The device operation,
the composite layer, the nanobind file and the custom hash are untouched. Results are identical to the legacy build on the
same checkout (see *Verification*).

Invoker decisions applied (answers to the audit's two open questions): **Q1, option (b)** — a `.def` op type that lands
mid-port is converted with the same edit when its kernel is a clone, and listed here. None landed during the port (still 36 `.def` files and 41 kernels at the end; #59374, #59369, #57589 and #59336 were all still open). The later rebase onto `origin/main` `22e8a7c341a` brought in #59374 (logiteps): `eltwise_bw_logiteps.cpp` and `logiteps_bw.def`. The tree now has 37 `.def` files and 42 compute kernels. The new kernel is a clone of `eltwise_bw_sigmoid_bf16.cpp` and got the same edit (a normalized diff against it differs in two lines, which are the op's own math structure). It has not been compiled or run on a device. #59369, #57589 and #59336 had not landed. **Q2** — the implicit-sync opt-out is left out:
every DM kernel uses the default `ttnn::create_reader_datamovement_config()` / `ttnn::create_writer_datamovement_config()`.

## Provenance

- **Recipe docs (this port):** `32fff42bf79 2026-11-21 Add unary_backward TensorParameter relaxation analysis`
  *(verbatim output of the provenance command at the branch tip. It names the relaxation-analysis commit, which is now the only
  file left under `metal_2.0/` because the recipe commits were stripped from the branch before the PR, so it does not pin the recipe at all. That commit has since been removed from this branch, and the analysis doc now lives on the recipe branch (see *Relaxation analysis* below). The recipe the port followed is upstream **`akertesz/op-porting-recipe` at `c41f7f4ef0a`** (2026-10-29,
  `docs(metal_2.0): DFBAccessor -> DFBBindingToken; drop the dead .id form`), the newest upstream commit touching `metal_2.0/`
  (checked through the GitHub API on 2026-11-21, and it is the tip of the local `origin/akertesz/op-porting-recipe`). To re-check it,
  after `git fetch origin akertesz/op-porting-recipe`, `git rev-parse
  c41f7f4ef0a:docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai` must print an id starting `3137c0f8edb5`. Blobs at that
  commit, under `docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/`: `ai/port/metal2_port.md` `aad3eedddf74`,
  `ai/audit/metal2_audit.md` `6435c8fcae15`, `ai/shared/ttnn_factory.md` `bd5f68d742c7`, `ai/shared/port_patterns.md` `7ba67547a9d7`,
  `ai/shared/migration_guide.md` `009627b34a8d`, `ai/shared/cb_dfb_api_whitelist.md` `798733a9c6c9`. While the port was built and
  verified, this branch carried those files in place and its `ai/` tree was identical to that commit's.)*
- **Audit docs (inherited):** `32fff42bf79 2026-11-21 Add unary_backward TensorParameter relaxation analysis` *(the same commit; the
  audit and the brief recorded it as `0b4911ccc4f6` before the recipe commits were stripped and it was rebased)*
- **Relaxation analysis:** it is not part of this branch. It lives on `akertesz/op-porting-recipe` at `docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/analyses/relaxations/eltwise_unary_backward.md`, blob `83ff1bbed224`. The audit validated blob `f0c9ad2ebc69`. The two differ on line 15 only, where two links became relative. Read it with `git show origin/akertesz/op-porting-recipe:<path>`.

## Verification

Hardware: one Wormhole n150. Every run below used `TT_METAL_WATCHER=10` and
`TTNN_CONFIG_OVERRIDES='{"validate_program_args": true}'`, with a fresh `TT_METAL_CACHE` per run. The legacy baseline ran on
the build already on disk, whose sources equal the checkout (no source file was newer than the `.so`, and `HEAD` had not moved
since the build), before any edit.

| What | Legacy baseline | Port |
|---|---|---|
| Main pytest list, 280 node ids built once and replayed from a file: whole files `test_unary_backward_device_op.py` (44), `test_backward_gelu.py` (65), `test_tanh_bw_ulp.py` (54), `test_tanh_bw_fp32_ulp.py` (17); the first test function of each of the 36 `.def` ops' nightly files, first and last case (72); every 5th case of `test_backward_memory_config.py` (28) | 252 passed, 24 skipped, 4 xfailed | 252 passed, 24 skipped, 4 xfailed |
| Comparison-mode cases (`test_unary_backward_with_unit_grad_in_comparison_mode`, need `enable_fast_runtime_mode` off) | 8 passed | 8 passed |
| ULP gtests (`TanhBwUlpTest`, `GeluBwUlpTest`, `GeluBwMainUlpTest`, `unit_tests_ttnn`) | 31 passed | 31 passed |

- **Same cases, same results.** The sorted per-case result lines of the legacy and port logs differ in **0** lines (256 result
  lines each, plus 24 skipped cases with identical skip reasons); the gtest `OK` sets are identical.
- **All 41 compute kernels ran.** From the fresh JIT cache directories, the union of `eltwise_bw_*` kernel directories across the
  port's runs is 43: all 41 of this op plus `eltwise_bw_mul` and `eltwise_bw_sub`, which are `binary_backward`'s. The first list
  reached 37 of the 41. The other four need a call at the "fitted" scalar value or a non-bf16 operand (`softplus_bw` at beta 1 /
  threshold 20, `hardshrink_bw` at lambd 0.5, `leaky_relu_bw` at slope 0.01, and `eltwise_bw_tanh.cpp`, which only non-bf16
  `tanh_bw` reaches). They were run separately: 68 extra cases (the `…_default` and
  `…_runs_the_fused_program_only_where_it_applies` tests of the three scalar ops) gave 67 passed, 1 xfailed, and a direct probe
  of `tanh_bw` on float32 and bfloat8_b against torch autograd gave PCC 1.000000 (max abs error 8.1e-4) and 0.999921. Neither
  of those four was run on the legacy build, so they have no legacy twin; they pass on correctness alone. The legacy tanh
  fp32 ULP tests skip on Wormhole (#57509), and the sampled list missed the tests that do reach `eltwise_bw_tanh.cpp`
  (`test_bw_tanh_mixed_operand_dtypes` and three `…_runs_the_fused_program_only_where_it_applies` cases in `test_backward_tanh.py`);
  the full run below covers them.
- **The two cache-hit tests you named pass with the knob on:** `test_sigmoid_bw_program_cache_same_padded_shape` and
  `test_tanh_bw_cache_hit_same_padded_shape_different_logical_shape`.
- **The legality checks are live in the binary under test (negative controls).** Temporary, env-switched defects in the factory
  (removed before the commit; the final binary contains no trace of them):
  - Exact `TensorParameter` relaxations (`{}`), knob on: both named tests **fail**, with `TT_FATAL
    program_run_args.cpp:305 runtime_spec == expected_spec — TensorArgument for binding 'grad_output' supplied a MeshTensor whose
    TensorSpec does not match the binding's declared spec`. The same defect with the knob **off** passes both tests silently.
  - A compute kernel stripped of its `grad_input` PRODUCER binding, knob off: the cache **miss** fails with `TT_FATAL
    collect_metadata.cpp:92 !endpoint_info.producers.empty() — DFB 'grad_input' has no producer`. The miss path validates
    regardless of the knob.
- **Empty operands behave as before.** Zero-volume tile and row-major inputs return an empty output on legacy and on the port
  alike (4 probe cases, identical).
- **Static self-audit** (denominators: factory pair 2 files, 57 op code files, 45 bound kernels in or beside the op):
  buffer address in run args **0** hits / `emplace_runtime_args` **0** / `Buffer*` **0**; `TensorAccessorArgs` in code **0** (two stale
  comments only, in the off-limits device op, see *Open items*); positional `compile_time_args` **0** (the one CTA is named);
  `get_vararg` family **0**; `allow_instance_multi_binding` **0**; `TT_FATAL` / `TT_ASSERT` / `TT_THROW` per-file counts identical to
  the base in all 5 files that have them; `grep opt_level` on the factory prints exactly one line, `O3`, for the one compute
  `KernelSpec`; DM reader and writer resolve to RISCV_1/NOC_0 and RISCV_0/NOC_1, both dedicated, which is what the helpers
  build (read from the legacy lowering and `kernel_types.cpp`); `cb`-shaped names in the op's code files: **1** hit, a comment in
  the off-limits device op (down from 254 hits in 46 files at the base); diff scope: no file under `tt_metal/` and no `METAL2_CHECKS_FORCED` /
  `DO NOT COMMIT` marker; install coverage: **45/45** bound kernel sources are matched by a regenerated install rule (41
  compute, 2 RM, the new fork, the reused fork), checked in `build_Release/ttnn/cpp/.../cmake_install.cmake`.
- **Not measured:** device or host performance. The kernels, compile options (O3 compute, O2 DM) and hardware configs are the
  legacy ones, but no `ttperf` comparison was taken.
- **Final run, on the tree that is committed.** The formatting hook rewrote three files (line wrapping only), so the binary was
  rebuilt and the whole set replayed: main list 252 passed / 24 skipped / 4 xfailed, comparison-mode 8 passed, gtests 31 passed,
  again **0** differing result lines against the legacy logs; the 68 extra cases gave 67 passed, 1 xfailed, the probes were
  unchanged, and all 41 kernels were compiled again. The two named tests passed in it.
- **Drift.** The branch base is `5340d88ced5`; local `origin/main` was 5 commits ahead, and no commit on `main` touched the op
  directory between the base and the end of the port (checked through the GitHub API as well, 2026-11-21 ~13:20 UTC). The checkout was not rebased then. The branch was rebased onto `origin/main` `22e8a7c341a` afterwards (see *Outcome*). The tests in this report ran before that rebase.
- **Full run of every test that reaches this op's kernels** (2026-11-21, after the commit). The run above sampled; this one ran
  the whole files. Same build (the 13:25 build, whose code equals `HEAD`: the working tree then differed from the commit only in six
  comment lines with an unchanged line count) and same settings: `TT_METAL_WATCHER=10`, `validate_program_args` on, one fresh
  `TT_METAL_CACHE`, serial on the n150, one pytest process per file. Only the comparison-mode group also had
  `"enable_fast_runtime_mode": false`. It took 17.5 minutes. **There is no legacy baseline for it**: the legacy tree was not built, so
  these are absolute pass/fail results, and every skip and xfail below is declared by the test code itself.

  | Group | Files | Cases | Passed | Failed | Skipped | XFailed |
  |---|---|---|---|---|---|---|
  | 1. Nightly per-op files | 38 | 1,769 | 1,747 | **0** | 6 | 16 |
  | 2. Unit test files | 8 | 390 | 366 | **0** | 24 | 0 |
  | 3. This op's cases in shared files | 3 | 107 | 98 | **0** | 9 | 0 |
  | 4. C++ gtests (`unit_tests_ttnn`, 5 suites) | — | 46 | 46 | **0** | 0 | — |

  Every file's passed + failed + skipped + xfailed equals the number of ids fed to it, and the counts match the ones you gave.
  - **Group 1, passed/cases per file:** abs 37/37; acos 37/37; asin 37/37; asinh 37/37; atanh 38/38; cosh 43/45; elu 53/53; erf 37/37;
    erfinv 37/37; exp2 37/37; expm1 39/39; gelu 61/65; hardshrink 52/53; hardsigmoid 36/37; hardswish 40/40; hardtanh 42/43;
    hyperbolic_range 48/48; leaky_relu 53/53; log 40/40; log10 37/37; log1p 37/37; log2 37/37; log_sigmoid 49/49; logit 37/37;
    mvlgamma 31/37; reciprocal 40/40; relu 47/49; relu6 89/91; rsqrt 44/44; selu 37/37; sigmoid 37/37; silu 44/44; sinh 43/45;
    softplus 134/134; softshrink 67/68; sqrt 44/44; square 37/37; tanh 52/52. The 6 skips: `sinh` ×2 and `cosh` ×2 ("Unsupported on WH
    and BH"), `relu` ×1 and `relu6` ×1 (#31406, bfloat16 loses a NaN operand). The 16 xfails: 12 are `…_runs_the_fused_program_only_where_it_applies`
    cases ("main's composite differs from torch here on wormhole_b0": `mixed_layouts` for relu, relu6, hardtanh, hardsigmoid,
    hardshrink and softshrink, and six `multigammaln` calls), which exercise the composite path, not the fused kernel; the other
    4 are `gelu` special values (fp32 tanh ±inf produce NaN; bf16 NaN returns 1.0, tt-llk #1701).
  - **Group 2, passed/cases:** `test_unary_backward_device_op.py` 44/44; `test_tanh_bw_ulp.py` 47/54; `test_tanh_bw_fp32_ulp.py` **0/17**;
    `test_gelu_bw_ulp.py` 63/63; `test_gelu_bw_main_ulp.py` 34/34; `test_erfinv_reciprocal_bw.py` 22/22; `test_rsqrt_bw_zero_sign.py`
    20/20; `test_backward_memory_config.py` 136/136 (all cases). The 24 skips are all the Wormhole tanh-derivative issue (#57509):
    all 17 of the fp32 file ("only fixed on Blackhole") and 7 of the bf16 file (a sign-set NaN from `sfpi::abs`).
  - **Group 3, passed/cases:** comparison-mode 18/18; `test_backward.py -k test_atanh` 3/12 (the 9 skips are "Skipped due to hardware
    restriction in storing nan"); `docs_examples/test_backward_examples.py` 77/77.
  - **Group 4:** `TanhBwUlpTest` (9), `GeluBwUlpTest` (11), `GeluBwMainUlpTest` (11), `GeluBwPolyTest` (7), `GeluBwMainPolyTest` (8):
    46 of 46 passed.
  - **Kernel coverage in this run's fresh cache:** all 41 compute kernels and the four dataflow kernels (both forks and both RM kernels)
    were compiled. The non-bf16 `eltwise_bw_tanh.cpp` was first compiled during `test_backward_tanh.py`, through
    `test_bw_tanh_mixed_operand_dtypes` (fp32/fp32, bf16/fp32, fp32/bf16, two shapes: 6 cases) and the `row_major`, `float32` and
    `input_grad_float32` cases of `test_tanh_bw_runs_the_fused_program_only_where_it_applies`, all passed. So that kernel has tests
    behind it after all, in addition to the direct probe; only the fp32 ULP file, which would check its accuracy, is skipped on
    Wormhole.

## TTNN ProgramFactory

### Concept realized

`ProgramSpecFactoryConcept`, as the audit decided. There was no `override_runtime_arguments` to translate. `create_program_artifacts`
(`device/unary_backward_program_factory.cpp:48`) builds one spec for the mesh; a cache hit refreshes the three tensor bindings
and nothing else, which is what the legacy descriptor adapter did for this op (`Buffer*` RTAs patched on every hit). Every other
value stays baked at the miss, and the custom hash pins what those values depend on.

### Device-op-class edits

- Pybind entry points removed: **none**. `unary_backward_nanobind.cpp` has no `create_descriptor` binding and is unchanged.
- Custom `compute_program_hash`: **left intact** at `device/unary_backward_device_operation.cpp:273-300`. No file of the device
  operation, the composite (`unary_backward.cpp`) or the nanobind module changed.

### Open items

- The relaxation is declared once, as `tensor_relaxations` (`.cpp:626`), on all three `TensorParameter`s
  (`match_padded_shape_only` + `relax_tensor_layout_alignment`), as the analysis doc requires. Its six validity checks were run by
  the audit on this tree and the op's files did not change afterwards. The check that a hit's tensor still matches the
  declared spec runs only with `validate_program_args`; the negative control above shows a wrong relaxation passes silently
  without it.
- The geometry computations (ALIAS test, row-major paging, work split) stay on the `ttnn::Tensor` accessors exactly as they
  were. The three `MeshTensor`s are extracted once (`.cpp:56-58`) and used for the parameter specs and the arguments only. See
  *Friction* G7.

## Handoff points

- **Port capitulation:** none.
- **Boundary-rule assumption violations:** none. `dfb::name` converts to `uint32_t` at the LLK and kernel-lib call sites;
  `tensor::name` is used only in `TensorAccessor(...)` constructions inside the op's own kernels and the two forks.
- **Kernel-lib gaps:** none. `ckl::input(...)` and `ckl::output(...)` take the token through its `uint32_t` conversion.
- **Framework gaps:** none that bit the port.
- **Removed pybind surface:** none.

## Successes

- **[Conditional / optional resource bindings](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-conditional--optional-resource-bindings).**
  The shared reader and writer already `#ifdef`-gated every accessor on `IN0_SHARDED` / `IN1_SHARDED` / `OUT_SHARDED`. Binding the
  tensors only where the define is unset (`.cpp:398-451`) therefore changed no kernel logic, and the host and the kernel share one
  condition.
- **[Porting a shared kernel](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-porting-a-shared-kernel).**
  The census (a factory binds the binary reader, a legacy generic-op gtest binds it, no fork exists) put the reader on rung 2
  and the writer on rung 1 with no ambiguity. The writer fork's vocabulary fitted this factory exactly, so it was bound
  without an edit. The new fork differs from its original only by the mechanical conversion (verified by a line diff).
- **[Pass DFB handles directly to LLKs](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers).**
  `dfb::grad_out` etc. flow straight into `compute_kernel_hw_startup` and `ckl::input(...)` / `ckl::output(...)`; no shim was
  needed. The edit is identical for all 41 compute kernels, so one script made it, and a normalized diff proves the 36 `.def`
  kernels are still clones of each other (36/36).
- **The relaxation analysis and the brief.** The per-slot declaration was a transcription, not a judgment. The two named
  cache-hit tests plus the negative control are what show it is both sufficient and checked.
- **Self-loop for the RM scratch DFBs.** The one-toucher census gave the binding shape directly; both pass validation on Gen1.

## Friction

### Gaps

- **G1. The legality-forcing step cannot be applied as written.** *Ensure the Metal 2.0 host-side legality checks are enabled* says
  to find the gates with `grep -n 'bool skip_validation' tt_metal/impl/metal2_host_api/*.cpp`. On this tree that finds one hit,
  the deprecated `MergeProgramRunArgs(..., bool skip_validation)` overload (`program_run_args.cpp:2028`). The real gates are
  `ValidationMode` parameters on `Program::set_program_run_args`, `update_program_run_args` and `update_tensor_args`. TTNN's
  adapter always validates on a cache miss and gates only the hit path on `validate_program_args`
  (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:53-56`, `:1330-1336`). So there is nothing to force, and no marker to
  add. I followed the invoker's instruction and used `TTNN_CONFIG_OVERRIDES='{"validate_program_args": true}'` for every run,
  proved it live (it reads `True` in the test environment and `False` without), and proved the checks live with the negative
  controls above. The section needs rewriting around that setting.
- **G2. Header paths and namespace are stale** in the recipe and the migration guide. They cite
  `tt-metalium/experimental/metal2_host_api/{program_spec,kernel_spec,dataflow_buffer_spec,program,program_run_args}.hpp` and the
  `tt::tt_metal::experimental` namespace. The headers are `tt-metalium/workload/specification/*.hpp` (including
  `tensor_parameter.hpp`, `scratchpad_spec.hpp`, `tensor_spec_relaxations.hpp`) and `tt-metalium/workload/run_args/program_run_args.hpp`,
  in `tt::tt_metal`. Only `advanced_options.hpp`, `prefetcher_pipe_parameter.hpp` and `sram_arena_parameter.hpp` remain under
  `experimental/metal2_host_api/`. The invoker supplied the right paths.
- **G3. Hardware-config names differ from the recipe's.** The recipe shows `create_reader_datamovement_config(device->arch())`,
  `ComputeGen1Config` and `DataMovementGen1Config`. The TTNN helpers take no architecture (only
  `bool disable_dfb_implicit_sync_for_all = false`), the compute type is `ComputeHardwareConfig` with an optional `config_1xx`, and the
  DM type is `DataMovementHardwareConfig::DataMovement1XXConfig`. The recipe's defaults table is right.
- **G4. The brief and the whitelist disagree on one getter.** The brief suggests `get_entry_size()` for the reader's two
  `get_tile_size(cb_id)` calls. The whitelist (§A) and kernel-side rule 7 map `get_tile_size(cb_id)` to the member
  `get_tile_size()`, which reads the same `unpack_tile_size` array the free function does (`dataflow_buffer.h:276-281`,
  `dataflow_api.h:282-290`); `get_entry_size()` (§B) is the mapping for `fifo_page_size`. The two values are equal for this op,
  but the fork is shared and `get_tile_size()` is exact for every consumer, so the fork uses it
  (`reader_binary_interleaved_start_id_metal2.cpp`, the two `*_tile_bytes` lines).
- **G5. The test-confirmation checkpoint assumes an interactive invoker.** *Locate and confirm the op's tests* asks to present the
  found set and wait. The session was non-interactive, so the brief's item 10 (device-op tests, gelu, tanh ULP, a few per-op
  nightly files) was taken as the confirmed baseline. The wider set found (grouped by tree) is listed under *Open items*.
- **G6. The `cb`-leftover sweep cannot reach zero in a correctly scoped port.** It scans the whole op directory, but the
  device-operation files are off-limits, and one of them carries a `CB` in a comment (`unary_backward_device_operation.cpp:291`).
  The sweep should say "in the files the port may touch".

### Confusion

- **G7. "Work with the `MeshTensor` throughout" versus "don't refactor adjacent code".** The TTNN integration doc says to extract the
  `MeshTensor` at the top of the factory and use it throughout; the scope discipline forbids refactoring factory code beyond the API
  renames. This factory's geometry lives in lambdas over `ttnn::Tensor` (`memory_config()`, `padded_shape()`, `mesh_buffer()`,
  `physical_volume()`). I kept those as they were and extracted the `MeshTensor`s once for the specs and arguments. If the intended
  reading is the broader one, that is a refactor for a separate change.
- **A second kernel-entry form is documented in `kernel_args.h`** (`TT_KERNEL`, "the JIT generates `kernel_main()` from its
  signature"). The recipe and whitelist only describe `kernel_main()` with `get_arg(args::…)`, which is what the port uses. A
  sentence on which form is intended would help.

## Open items for downstream

- **Shared kernel touches.**
  - `ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp` — (a) kernel path as
    given; (b) rung 2, **created the fork** `reader_binary_interleaved_start_id_metal2.cpp` beside it, and the pointer comment landed in
    the original (the only edit there); binding vocabulary: `dfb::in0`, `dfb::in1`, `tensor::src0`, `tensor::src1`, named CTA
    `block_or_width_sharded`, named RTAs `num_tiles`, `start_id`, `block_height`, `block_width`, `num_cores_y`, defines
    `IN0_SHARDED`, `IN1_SHARDED`; (c) remaining unmigrated consumers of the original: `tests/ttnn/unit_tests/gtests/test_generic_op.cpp:437`
    (legacy generic-op API). No other op factory binds it. The writer fork cites a sunset issue (#52228); I found none for this one,
    so the sunset list lives here.
  - `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` — (b) rung 1, **reused**
    `writer_unary_interleaved_start_id_metal2.cpp` (no new file, original untouched); 26 other op factories bind the fork;
    remaining consumers of the legacy original: `test_generic_op.cpp:894,1013`, `test_exp_approx_disable_sfploadmacro.py:86`,
    `test_generic_op_host_time.py:68`.
- **The op-type set is open, and the kernel contract changed.** Four open PRs each add one `.def` file and one compute kernel
  (#59374 logiteps, #59369 acosh, #57589 softsign, #59336 erfc). After this port a new kernel must be written in Metal 2.0 form:
  `dfb::grad_out`, `dfb::input`, `dfb::grad_in` tokens (with the three `DataflowBuffer` objects in that order) and
  `get_arg(args::per_core_tile_cnt)`. The comment new authors copy, `device/unary_backward_op_utils.hpp:19-21`, still describes the
  legacy form (`c_0` / `c_1` / `c_2`, positional RTA 0). It sits outside the factory body, so it was not edited.
- **Stale legacy wording in files the port may not touch:** `device/unary_backward_device_operation.cpp:106` ("aliases each
  circular buffer"), `:283` and `:291-292` (`TensorAccessorArgs`, "destination CB"), `unary_backward.cpp:674` ("circular
  buffers to the shards"). The new fork keeps its original header comment, which still mentions `CircularBufferConfigs` and a
  `CBIndex::c_2` use that the file never had; it was preserved rather than deleted.
- **RTAs that are really CRTAs** (same value on every node; left as RTAs, as the recipe requires): in both RM kernels
  `row_elements`, `chunk_elements`, `chunks_per_row`, `rows_per_tile`, `total_rows` and every paging field; in the borrowed reader
  `block_height`, `block_width` and `num_cores_y`. `block_height`, `block_width`, `num_cores_y` and the CTA `block_or_width_sharded`
  are dead for this op (read only under `if constexpr (block_or_width_sharded)`) and live for the kernel's other binder.
- **Gen2 notes.** The two RM scratch DFBs (`reader_scratch`, `writer_scratch`, only under `rm_blocked`) are DM self-loops, legal on
  Gen1 and rejected on Gen2; they are what the post-port `sync_free_dfbs` pass turns into `Scratchpad`s. The `dfb::name` to `uint32_t`
  conversions at `compute_kernel_hw_startup` and `ckl::input` / `ckl::output` (41 kernels) are the Gen1-only bridge.
- **The 36 `.def` kernels assume BF16 TILE operands, and only the composite enforces it** (`generated_bf16_kernel_applies`,
  `unary_backward.cpp:46-79`). The device operation accepts BF16, FP32, BFP8 and BFP4 for every op type and builds the fp32-DEST
  and mixed-format settings for them. Every call site is gated, so there is no live bug (audit finding, unchanged).
- **Test coverage.** The per-op nightly files reach the three scalar-parameter kernels only through their `…_default` and
  `…_runs_the_fused_program_only_where_it_applies` tests, and `eltwise_bw_tanh.cpp` only through the fp32, row-major and
  mixed-dtype cases of `test_backward_tanh.py`; the fp32 tanh ULP tests that would check its accuracy skip on Wormhole (#57509). The first
  verification run sampled; the full-run block under *Verification* later ran this op's 38 nightly files, the whole unit files,
  `test_backward_memory_config.py`, the op's comparison-mode cases and the docs examples in full. What was still not run: the
  other 65 nightly `test_backward_*.py` files (other ops), the `*_bw_device_perf.py` files under `…/eltwise/perf/`, the remaining
  cases of `tests/ttnn/unit_tests/base_functionality/test_comparison_mode_context.py` (290 in the file; 18 run), and the sweeps under
  `tests/sweep_framework/sweeps/eltwise/unary_backward/` (left to CI).
- **Doc evolution:** G1 (rewrite the legality-forcing section around `ValidationMode` and `validate_program_args`), G2 and G3 (paths,
  namespace, helper and type names), G6 (scope the sweep).
