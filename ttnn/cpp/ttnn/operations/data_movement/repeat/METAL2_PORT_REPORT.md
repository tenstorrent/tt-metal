# Metal 2.0 Port Report — `data_movement/repeat`

## Outcome

**`PORTED`** — both program factories (`RepeatProgramFactoryLastDim`, `RepeatProgramFactoryHigherDim`) and
all five kernels converted to Metal 2.0 (`MetalV2FactoryConcept`). No capitulation; no construct fell
outside the host-side scope or the kernel-side whitelist.

**Verification (with the environment corrected — see Friction):**
- **Build:** `./build_metal.sh` green (host factories compile clean under warnings-as-errors).
- **Tests:** `test_universal_input_tm_repeat.py` + `test_repeat.py` → **194 passed, 242 skipped, 1 failed**.
  The single failure — `test_repeat_explicit_grid_edge_cases[tile_block_3x2_irregular_H]` — is **not** a
  port regression: `ttnn.repeat(...)` itself succeeds, and the failure is in the test's post-op
  `result.cpu().to(ROW_MAJOR_LAYOUT)` verification step, which hits
  `TT_FATAL @ tt_metal/impl/tensor/tensor_apis.cpp:1177: unpad: sharded host tensors are not supported`
  (a shared-utility limitation on unpadding an irregular 65-row block-sharded host tensor, out of the op's
  scope). **Confirmed pre-existing:** the identical case fails identically on the untouched
  `ops2_0_baseline` checkout (legacy repeat), same TT_FATAL and traceback. The port diff is confined to the
  9 ported files (5 kernels + 2 factory `.cpp` + 2 factory `.hpp`); `compute_output_specs`,
  `create_output_tensors`, and the composite are untouched, so the output tensor geometry the test converts
  is byte-identical pre- and post-port.
- **Anti-pattern self-audit:** clean (see the checklist result under Successes).

## Provenance

- **Recipe docs (this port):** `44da718b06b 2026-07-23 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`
- **Audit docs (inherited):** `44da718b06b 2026-07-23 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## TTNN ProgramFactory

### Concept realized

`MetalV2FactoryConcept` for both factories, as the audit chose. Each factory's `create_descriptor` (returning
`tt::tt_metal::ProgramDescriptor`) became `create_program_artifacts` (returning
`ttnn::device_operation::ProgramArtifacts` = `ProgramSpec` + `ProgramRunArgs`, no op-owned tensors). Both
factories flip together, and the framework dispatches per-factory by concept, so the device-op variant is
fully on Metal 2.0.

### Device-op-class edits

- **Custom `compute_program_hash` deleted:** none. The device op never defined one (default reflection hash).
- **Pybind entry points removed:** none. `repeat_nanobind.cpp` binds only the `ttnn::repeat` free function
  (no `nb::class_` of the device op, no `create_descriptor` pybind), so the disappearance of
  `create_descriptor` forces no pybind cleanup. The device-operation class
  (`repeat_device_operation.*`) was **not** touched.

### Open items

- **`dynamic_tensor_shape` relaxation candidate (not applied).** Both factories legacy-build accessor args
  with `tensor_accessor::ArgConfig::RuntimeTensorShape`, which the migration guide's `TensorParameter`
  pre-flight maps to `advanced_options.dynamic_tensor_shape = true`. The port keeps the **strict default**
  (audit: relaxation = none), which is correct here (see Friction). A later pass *could* set
  `dynamic_tensor_shape = true` on the INPUT/OUTPUT `TensorParameter`s to widen program-cache equivalence
  (one cached program serving multiple shapes), if the caching-path implications are judged worth it. Not a
  port-time call.

## Handoff points

None. No capitulation, no boundary-rule assumption violations (no out-of-op call site needed `sem::` /
`tensor::`), no kernel-lib gaps (the in-family `common.hpp` helpers took the Metal 2.0 tokens unchanged), no
framework gaps, no removed pybind surface.

## Successes

- **[Sync-free / single-ended CB → self-loop DFB](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb)**
  fit every DFB in the op exactly. Each kernel is the sole toucher of its buffer(s) and uses them as
  scratch (`reserve_back` / `get_write_ptr` / `push_back`, then base-pointer access; no `wait_front` /
  `pop_front`). Binding the reader PRODUCER + CONSUMER with one shared accessor name satisfied the
  validator's ≥1-of-each rule with the kernel body untouched — matching the pattern's promise for a Gen1
  DM self-loop. (`repeat_program_factory_higher_dim.cpp:104-113`, `repeat_program_factory_last_dim.cpp`.)
- **`create_reader_datamovement_config(device->arch())`** cleanly reproduced the legacy
  `ReaderConfigDescriptor{}` (reader default RISCV_1/NOC_0/DM_DEDICATED_NOC) for the single DM kernel of
  each program, with the Gen2 branch supplied for free — no arch branch in the factory.
- **`AddRuntimeArgsForNode`** let the legacy node-first per-core loop stay verbatim (only the
  `emplace_runtime_args(core, {...})` call swapped for the helper), avoiding a risky loop-inversion into
  name-first form on top of an already-large rewrite.
- **[Unity-build hygiene](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-unity-build-hygiene-for-anonymous-namespace-symbols)**
  warning fired correctly: `data_movement` is a unity build, so declaring the `KernelSpecName` / `DFBSpecName`
  / `TensorParamName` constants function-local (not at anon-namespace scope) avoided cross-factory duplicate
  symbols before they could happen.
- **Strict `TensorParameter` binding validated across the full sharded matrix.** The decision to keep the
  strict default (no `dynamic_tensor_shape`, following the audit against the migration guide's
  `RuntimeTensorShape` pre-flight) held up empirically: the 194 passing tests include height/width/block
  sharded, interleaved, TILE and ROW_MAJOR, DRAM and SRAM, and fp32/bf16/bf8_b/int32/uint32/uint16 across
  both factories and all five kernels, plus the program-cache-hit tests (`test_pc_repeat`,
  `test_pc_with_different_shapes_in_sequence`) that exercise `UpdateTensorArgs` on the strict binding. No
  spec-validation or numeric failure anywhere — confirming strict is correct here, and the migration
  guide's "RuntimeTensorShape ⇒ likely dynamic_tensor_shape" heuristic is optional, not required, when
  TTNN's cache is spec-keyed.
- **Anti-pattern self-audit — clean.** Grep of the op directory post-port: no `tensor.buffer()->address()`,
  no surviving `TensorAccessorArgs<N>()` / `get_arg_val` / `get_compile_time_arg_val`, no magic CB indices
  in CTAs (all → DFB bindings), no `.id` extraction / temp DFB wrappers, no `ProducerOf`/`ConsumerOf`
  factories, no `allow_instance_multi_binding` (every DFB is a one-toucher self-loop, never stacked with a
  flag), all CTAs named, no varargs, no CB/`CBDescriptor`/`ProgramDescriptor`/`create_descriptor` residue.
  `hw_config` reproduces the legacy `ReaderConfigDescriptor{}` reader default via
  `create_reader_datamovement_config`; no compute kernels, so no compute-config precision knobs to diff.

## Friction

### Gaps / Confusion

- **Reading a compile-time `is_dram` off the Metal 2.0 `TensorAccessor`.** The two interleaved kernels
  compute alignment masks from `src_args.is_dram`, where `src_args` was the legacy `TensorAccessorArgs`
  (dropped in the port). The recipe/whitelist cover dropping `TensorAccessorArgs` and constructing
  `TensorAccessor(tensor::name)`, but do not spell out how to recover a compile-time `is_dram` afterward.
  Resolved with `decltype(s)::is_dram` (the accessor's `static constexpr bool is_dram`), matching the
  `Reader::DSpec::is_dram` idiom in `experimental/conv3d/device/kernels/reader_vol2col.cpp`. A one-line note
  in the whitelist (accessor-property getters like `is_dram` survive on the object) would remove the guess.
  (`repeat_higher_dim_rm_interleaved.cpp:50-52`, `repeat_last_dim_rm_interleaved.cpp:65-67`.)

- **`ArgConfig::RuntimeTensorShape` vs. the audit's "relaxation = none".** The migration guide's
  `TensorParameter` pre-flight says: grep kernels for `ArgConfig::Runtime`; if `RuntimeTensorShape` appears,
  the faithful port likely sets `dynamic_tensor_shape = true`. The audit, meanwhile, recorded relaxation =
  none. These *appear* to conflict. Resolution (following the audit + the recipe's strict-default bias):
  keep strict, because it is provably correct here and relaxing is the unsafe direction. Why strict is
  correct: TTNN keys its program cache on tensor spec, so each distinct spec re-runs the factory and bakes
  the right shape into the kernel's CTAs at program creation; and the repeat kernels compute offsets and
  transfer sizes from their own `original_page_size_bytes` CTA, never from the accessor's dynamic page size
  (`get_aligned_page_size()` is not called). So strict vs. `dynamic_tensor_shape` differ only in program-
  cache granularity, never in numerics. A sentence in either the audit sheet or the migration-guide
  pre-flight reconciling "legacy used RuntimeTensorShape" with "relaxation = none" (i.e. that TTNN's
  spec-keyed cache makes the relaxation optional, not required) would remove the apparent contradiction.

- **Stale `build_Release` CMakeCache (environment, not port).** This checkout's `build_Release/` was copied
  from the sibling `ops2_0_baseline` checkout, so its `CMakeCache.txt` hardcoded baseline's absolute source
  path; `./build_metal.sh` failed at CMake *configure* ("source ... does not match the source ... used to
  generate cache") before compiling anything. Resolved by removing this checkout's `build_Release/` (a real
  directory, not a symlink — baseline and the shared `~/.cache/tt-metal-cache` left untouched) and
  reconfiguring, which forces a full cold rebuild. Not a Metal 2.0 concern; noted so the next porter
  recognizes a copied-workspace build dir.

- **Copied venv resolved `ttnn` to the sibling baseline checkout (environment, not port) — the subtle
  one.** This checkout's `python_env` was likewise copied from `ops2_0_baseline`, and its
  `ttnn-custom.pth` + editable-install finder hardcode baseline's paths, so `import ttnn` resolved to
  `…/ops2_0_baseline/…/ttnn/ttnn/__init__.py` — i.e. the whole test run exercised **baseline's** framework
  (whose repeat factory is still the legacy `create_descriptor`) while the JIT read *this* checkout's
  Metal 2.0 kernel *source* (relative kernel paths resolve against the CWD). The mismatch surfaced as a
  kernel JIT compile failure in all 175 device tests: `'args'/'dfb'/'tensor' not declared`, because the
  baseline (legacy) build path emits `-DKERNEL_COMPILE_TIME_ARGS=...` and does **not** inject
  `kernel_args_generated.h` / `kernel_bindings_generated.h` (those come only from
  `MakeProgramFromSpec`, `genfiles.cpp` gating on `is_metal2_kernel()`). The tell in the failing JIT
  command line was the include paths pointing at `ops2_0_baseline` while the kernel `.cpp` pointed at
  `ops2_0_repeat`. Fixed by running pytest with `PYTHONPATH=<repeat>/ttnn:<repeat>:<repeat>/tools` and
  `TT_METAL_HOME=<repeat>` so `ttnn` resolves to this checkout's freshly-built module. This is exactly the
  "PYTHONPATH must be `$(pwd)` from inside your clone, not a path copied from someone else's instructions"
  trap the workspace-setup doc warns about — worth a louder callout, since the symptom (a Metal 2.0 kernel
  compile error) masquerades as a port bug rather than an environment misconfiguration.

## Open items for downstream

- **Cross-op kernel touches:** none. All five kernel sources live in the op's own `device/kernels/`. The
  in-family shared header `data_movement/common/kernels/common.hpp` was consulted but **not** modified; its
  `noc_async_read_sharded` / `noc_async_write_sharded` / `tt_memmove` / `align_address` helpers accept the
  Metal 2.0 `TensorAccessor` (by value) and the DFB write pointer (a raw SRAM address) unchanged. If that
  header is later rewritten for Metal 2.0, port the whole `data_movement` family as one unit.
- **Relaxation candidate:** the `dynamic_tensor_shape` opportunity on the INPUT/OUTPUT `TensorParameter`s
  (see TTNN ProgramFactory → Open items).
- **Pre-existing host-logic defect (routed to ops team, not the port):** the operator-precedence bug in the
  last-dim `cb_size_bytes` expression (`repeat_program_factory_last_dim.cpp:53-57`), flagged by the audit's
  Misc-anomalies section. The port carries the expression verbatim into `DataflowBufferSpec::entry_size`
  (zero functional change). A fix belongs on the ops track, not the port diff.
- **Pre-existing test failure (framework, not the port):**
  `test_universal_input_tm_repeat.py::test_repeat_explicit_grid_edge_cases[tile_block_3x2_irregular_H]`
  fails in its post-op `to(ROW_MAJOR_LAYOUT)` verification with
  `TT_FATAL @ tensor_apis.cpp:1177: unpad: sharded host tensors are not supported`, for an irregular 65-row
  block-sharded (3x2 grid) output. Confirmed to fail identically on the untouched `ops2_0_baseline` legacy
  op, so it is not a Metal 2.0 regression. Owner: the `unpad`/`unpad_from_tile` utility team (or the test
  author, if the test should skip this irregular-shard geometry until the utility supports it). Out of the
  repeat op's scope.
- **Test coverage note:** no dedicated C++ gtest exists for `repeat` (the `*Repeat*` gtest matches are
  incidental uses in async-runtime / tensor-stream tests). The no-regression baseline is the two pytests
  confirmed with the invoker: `tests/ttnn/unit_tests/operations/data_movement/test_repeat.py` and
  `tests/ttnn/nightly/unit_tests/operations/data_movement/test_universal_input_tm_repeat.py`.
