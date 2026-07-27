# Port Report — moreh_sum_backward

## Outcome

**PORTED** — the single `MorehSumBackwardOperation` factory converted from the legacy
`ProgramDescriptor` (`descriptor`) concept to `MetalV2FactoryConcept`
(`create_program_artifacts`). Build/test verification is the orchestrator's (this porter did not
build or run tests, per orchestration constraints).

## Provenance

- **Recipe docs (this port):** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`
- **Audit docs (inherited):** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## Files created / modified

- `device/moreh_sum_backward_device_operation.hpp` — replaced the direct `create_descriptor`
  (→ `ProgramDescriptor`) method with a **nested `ProgramFactory` struct** whose static
  `create_program_artifacts` (→ `ttnn::device_operation::ProgramArtifacts`) is referenced by a new
  `using program_factory_t = std::variant<ProgramFactory>;`. Replaced the
  `<tt-metalium/program_descriptors.hpp>` include with `"ttnn/metal_v2_artifacts.hpp"` and added
  `<variant>`. The `program_factory_t` variant is **required** for `MetalV2FactoryConcept` detection
  and generated-header injection (see Friction → root cause).
- `device/moreh_sum_backward_program_factory.cpp` — rewrote the factory body from
  `ProgramDescriptor` (`CBDescriptor` / `KernelDescriptor` / buffer-address RTAs / `TensorAccessorArgs`
  CTAs) to a `ProgramSpec` + `ProgramRunArgs`. Two host helper functions (`get_tensor_dim`,
  `get_output_grad_shape`) and all parameter/work-split math are unchanged.
- `device/kernels/reader_moreh_sum_backward.cpp` — CTA `input_grad_rank` → `get_arg(args::…)`;
  fixed RTAs → named `get_arg(args::num_output_tiles/start_id)`; the three variable-length per-dim
  blocks → positional `get_vararg(i)`; `TensorAccessorArgs<1>()` + addr RTA →
  `TensorAccessor(tensor::output_grad)`; DFB objects from `dfb::in0`/`dfb::in1`;
  `get_tile_size(cb_id)` → `dfb.get_tile_size()`. Added `experimental/kernel_args.h`.
- `device/kernels/writer_moreh_sum_backward.cpp` — same treatment: named RTAs
  `num_tiles`/`start_id`; `TensorAccessorArgs<0>()` + addr RTA → `TensorAccessor(tensor::input_grad)`;
  DFB from `dfb::out`; `get_tile_size(cb_id)` → `dfb.get_tile_size()`. Added `experimental/kernel_args.h`.
- `device/kernels/moreh_sum_backward.cpp` (compute) — CTAs → `get_arg(args::…)`; DFB objects from
  `dfb::in0`/`dfb::in1`/`dfb::out0`; all LLK/kernel-lib call sites (`binary_op_init_common`,
  `add_bcast_*_init_short`, `add_tiles_bcast_*`, `copy_tile_to_dst_init_short`, `copy_tile`,
  `pack_tile`) now take `dfb::` handles via the implicit `DFBAccessor → uint32_t` conversion.
  Added `experimental/kernel_args.h`.
- `METAL2_PORT_PLAN.md`, `METAL2_PORT_REPORT.md` — created (this port's artifacts). Audit brief and
  full audit (`METAL2_PORT_BRIEF.md`, `METAL2_PREPORT_AUDIT.md`) were pre-existing inputs.

## Port summary

Host-side factory rewrite, as the audit predicted (the op's kernels were already on the Metal-2.0
kernel-side idioms — `DataflowBuffer`, `Noc`, `TensorAccessor`). Structure:

- **3 DataflowBufferSpecs** `c0_in` (2 entries), `c1_zero` (1 entry), `c16_out` (2 entries), each
  carrying `data_format_metadata` (all compute-bound).
- **2 TensorParameters** `output_grad`, `input_grad`, replacing the two `Buffer*` RTAs. Kernels now
  build `TensorAccessor(tensor::name)`.
- **4 KernelSpecs** — `reader`, `writer`, `compute_group_1`, `compute_group_2` (the last only when
  `core_group_2` is non-empty). The two compute KernelSpecs preserve the legacy two-`KernelDescriptor`
  work-split multiplicity, differing only in the `num_output_tiles` CTA.
- **WorkUnitSpecs** — `group1` {reader, writer, compute_group_1} @ core_group_1;
  `group2` {reader, writer, compute_group_2} @ core_group_2 (only if non-empty). reader/writer are in
  both work units (effective placement = core_group_1 ∪ core_group_2 = all_cores); each compute group
  is co-located with reader/writer per node. compute_group_1/2 bind the shared DFB endpoints over
  disjoint node sets (legal multi-KernelSpec-on-one-endpoint; no `allow_instance_multi_binding`).
  This matches the proven `moreh_group_norm` layout.
- **KernelRunArgs** — one entry each for reader and writer; the two compute kernels are arg-less
  (compile-time args only) and get **no** `KernelRunArgs` entry (matches `moreh_group_norm`).
- **DFB endpoints** all natural 1P+1C per node: reader PRODUCER of c0_in/c1_zero; compute CONSUMER of
  both + PRODUCER of c16_out; writer CONSUMER of c16_out.
- **Runtime varargs** — the three per-dim blocks (`output_grad_dim`, `input_grad_dim`,
  `need_bcast_dim`, each length `input_grad_rank`) are passed as positional runtime varargs
  (`reader.advanced_options.num_runtime_varargs = 3 * input_grad_rank`; per-node
  `runtime_varargs`), matching recipe RTA-varargs shape (a). Value is core-independent, built once.

### `hw_config`
- reader = `ttnn::create_reader_datamovement_config(arch)`, writer =
  `ttnn::create_writer_datamovement_config(arch)` (legacy used default `ReaderConfigDescriptor{}` /
  `WriterConfigDescriptor{}`).
- compute = `ComputeGen1Config` built directly (Style B — legacy set a Metal `ComputeConfigDescriptor`
  from the resolved scalars): `fpu_math_fidelity = math_fidelity`,
  `sfpu_precision_mode = math_approx_mode ? Approximate : Precise`,
  `enable_32_bit_dest = fp32_dest_acc_en`, `double_buffer_dest = !dst_full_sync_en`. `FP32_DEST_ACC_EN`
  define preserved. `packer_l1_acc` is resolved but was unused by the legacy compute config — left
  unused. `unpack_modes`: legacy set none (all default `UnpackToSrc`); an explicit `UnpackToSrc`
  entry is added for the two consumed DFBs (`c0_in`, `c1_zero`) **only** when the data format is
  Float32 and `enable_32_bit_dest` is true, because the Metal 2.0 validator requires an explicit
  entry in exactly that case — faithful to the legacy default (see Successes).

## TTNN ProgramFactory

### Concept realized
`MetalV2FactoryConcept`, as the audit chose. Realized through a nested `ProgramFactory` struct +
`using program_factory_t = std::variant<ProgramFactory>;` (not a direct method on the op struct —
see Friction → root cause).

### Device-op-class edits
- Custom `compute_program_hash` deleted: none (op had none).
- Pybind entry points removed: none (`moreh_sum_backward_nanobind.cpp` binds only the
  `moreh_sum_backward` function; no `create_descriptor` pybind).

### Open items
- No tensor-matching relaxation applied (kept strict). No relaxation candidate noticed.

## Test command(s)  *(verification is the orchestrator's)*

The op has no C++ gtest. Python coverage (no-regression baseline) is the backward tests in
`tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_sum.py`:

```bash
pytest tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_sum.py \
  -k "backward" -v
```

This selects `test_moreh_sum_backward`, `test_moreh_sum_backward_wo_input_grad`,
`test_moreh_sum_backward_enable_cache` (program-cache hot path — exercises tensor-arg re-binding on
cache hit), and `test_moreh_sum_backward_fp32_dest_acc` (exercises the fp32 / `unpack_modes` path).
All should pass unchanged. To run the whole file:

```bash
pytest tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_sum.py -v
```

## Handoff points

None. The port stayed entirely within the op directory. The shared moreh helper headers
(`ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp`, `.../compute/moreh_common.hpp`) were already on
Metal-2.0 idioms and were not touched.

## Successes

- **Newly-required `unpack_modes` entry (recipe §Hardware configuration → compute → item 3).** The
  recipe's warning that the Metal 2.0 validator requires an explicit `unpack_modes` entry for a
  Float32 DFB consumed with `enable_32_bit_dest = true` — where legacy silently defaulted — fired
  here. Without it the `fp32_dest_acc` test path would have hit a validator `TT_FATAL`. Added
  explicit `UnpackToSrc` (the faithful legacy default) for the two consumed DFBs, guarded on
  `Float32 && fp32_dest_acc_en`. Applied at `device/moreh_sum_backward_program_factory.cpp` (compute
  config block).
- **RTA varargs guidance (kernel-side whitelist rule 4 + brief).** The brief flagged the three
  variable-count per-dim RTA blocks as vararg-shape; the host `num_runtime_varargs` /
  `runtime_varargs` mechanism plus kernel-side `get_vararg(i)` mapped them cleanly without trying to
  name each element.

## Friction

### Gaps
- **ROOT CAUSE of the first-pass on-device failure: `create_program_artifacts` MUST live on a
  `program_factory_t` variant, not directly on the op struct.** My first pass put
  `create_program_artifacts` as a direct static method on `MorehSumBackwardOperation` (mirroring how
  the legacy `create_descriptor` sat directly on the struct). It **built cleanly** but failed all 75
  on-device tests with JIT errors (`'args'/'dfb'/'tensor' has not been declared`,
  `get_arg`/`get_vararg` unresolved) — the build-injected generated headers
  (`kernel_args_generated.h` / `kernel_bindings_generated.h`) were never emitted for the kernels.
  Reason (`ttnn/api/ttnn/operation_concepts.hpp`): the direct-method shortcut `HasDirectDescriptor`
  (line 97) is defined **only** for `create_descriptor`; there is no `HasDirectProgramArtifacts`.
  `MetalV2FactoryConcept` is detected only via a `program_factory_t` variant whose alternative has
  `create_program_artifacts` (lines 91, 149, 164). Without the variant, the framework's MetalV2
  adapter never registered the factory as Metal 2.0, so it never drove generated-header injection.
  **Fix:** nested `ProgramFactory` struct + `using program_factory_t = std::variant<ProgramFactory>;`
  (single-alternative variant → framework auto-selects it, no `select_program_factory` needed).
  Diagnosed by comparison with the same-session `moreh_group_norm` port, which uses exactly this
  shape and passes.
  **Doc gap:** the recipe / `ttnn_factory.md` show `create_program_artifacts` as a bare static and
  never state that a `program_factory_t` variant is mandatory for a TTNN op — the `HasDirect*`
  shortcut applying only to `create_descriptor` is a silent trap: it builds green and fails only
  on-device. Worth an explicit "wrap it in `program_factory_t`" instruction in `ttnn_factory.md`.
- **`KernelRunArgs` for arg-less kernels — resolved.** `program_run_args.hpp` says "A KernelRunArgs
  must be specified for ALL kernels," but the recipe says an entry "may be omitted" when a kernel has
  no RTAs. `moreh_group_norm` omits run-args for its arg-less compute kernels and passes on-device,
  so omission is correct; I omit them. The header comment overstates the requirement and should be
  reconciled with the recipe.

### Confusion
- **Style A vs Style B for the compute config.** The op resolves a TTNN config via
  `get_compute_kernel_config_args` (recipe's Style A trigger) but then builds a *Metal*
  `ComputeConfigDescriptor` directly from the destructured scalars (Style B shape). I treated it as
  Style B (build `ComputeGen1Config` directly, per-field), which is the most literal 1:1 with what
  legacy actually constructed and avoids depending on whether `to_compute_hardware_config` reproduces
  the exact same resolved values. Worth a recipe note on the "resolves-then-hand-builds-Metal-config"
  hybrid, which is common in the moreh family.

## Open items for downstream

- **Cross-op kernel touches:** none. All three kernels are op-owned; no fork needed.
- **Sibling moreh ops:** the moreh family shares `moreh_common.hpp` (dataflow + compute), already
  Metal-2.0-flavored. Sibling moreh backward ops with the same reader/writer/compute skeleton
  (per-dim vararg reader, `fill_cb_with_value` zero tile, bcast compute) will port with this exact
  shape.
- **Test coverage note:** the op has no C++ gtest and its Python coverage lives under
  `tests/ttnn/nightly/...` (not `tests/ttnn/unit_tests/...`); an auditor guessing the standard
  `unit_tests/operations/moreh/` path would miss it.
- **`writer_moreh_sum_backward.cpp` include:** the writer no longer uses anything from
  `ttnn/kernel/dataflow/moreh_common.hpp` (its former `ArgFetcher` / `get_tile_size` uses are gone),
  but the include was kept to avoid a transitive-include build risk on a header that is out of the
  CB-sweep's scope. A later cleanup could drop it.
