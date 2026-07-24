# Metal 2.0 Port Report — `ttnn/cpp/ttnn/operations/moreh/moreh_dot`

## Outcome

**PORTED** — the single `MorehDotOperation` factory (reader + writer + compute) converted from the
legacy `ProgramDescriptor` (`HasDirectDescriptor` / `create_descriptor`) API to
`MetalV2FactoryConcept` (`create_program_artifacts`). Build + test verification is the
orchestrator's (this porter did not build or run tests, per orchestration constraints).

## Provenance

- **Recipe docs (this port):** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`
- **Audit docs (inherited):** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## TTNN ProgramFactory

### Concept realized
`MetalV2FactoryConcept`, as the audit specified. The op previously satisfied `HasDirectDescriptor`
(a bare `create_descriptor` static method on the device-op struct, no `program_factory_t`). Metal 2.0
requires the factory to live in a `program_factory_t` variant, so the port introduced a nested
`struct MorehDotOperation::ProgramFactory` with a static `create_program_artifacts`, plus
`using program_factory_t = std::variant<ProgramFactory>;`. No custom `select_program_factory` (single
alternative → framework default). This is a structural device-op-class change forced by the concept
switch, not a freelance edit — the `validate` / `compute_output_specs` / `create_output_tensors`
scaffolding is untouched.

### Device-op-class edits
- Custom `compute_program_hash` deleted: none (op already used the default reflection-based hash).
- Pybind entry points removed: none (`moreh_dot_nanobind.cpp` binds a plain function via
  `ttnn::bind_function<"moreh_dot">`; no `create_descriptor` / device-op class was ever pybound).
- `create_descriptor` removed and replaced by `ProgramFactory::create_program_artifacts` (the port
  itself); `program_factory_t` variant added. Header include `<tt-metalium/program_descriptors.hpp>`
  replaced by `ttnn/metal_v2_artifacts.hpp`.

### Open items
- **RTA→CRTA cleanup candidate (not applied — dispatch-semantics change, out of scope):** the reader's
  `start_id` (always 0), `mask_h`, `mask_w`, the writer's `num_tiles` (always 1) and `start_id` (0), and
  the compute's `per_core_block_cnt` are all single-node constants. On a single-core op they could be
  common runtime args (or even CTAs), but the port preserves them as named RTAs to stay a pure syntax
  swap. Flagging for a later cleanup pass.
- **TensorParameter relaxation:** none applicable; strict spec matching kept (the kernels are not
  shape-agnostic in a way that requires `dynamic_tensor_shape`, and the audit flagged no relaxation).

## Handoff points

None. The port stayed entirely within the op directory; no kernel-lib / LLK / framework change was
needed. The two `kernel_lib` helpers the kernels call (`dataflow_kernel_lib::calculate_and_prepare_reduce_scaler`
in the reader, `compute_kernel_lib::reduce` in compute) take their CB argument as a `uint32_t` template
NTTP; the `dfb::name` constexpr implicit conversion crosses cleanly at both sites, so neither helper was
touched (they are `kernel_lib`-team owned).

## Successes

- **Self-loop pattern (`port_patterns.md` §Sync-free and single-ended CBs → self-loop DFB).** The two
  compute-internal intermediate CBs `c_24`/`c_25` (im0/im1) are touched only by the compute kernel. The
  one-toucher self-loop resolution (bind compute as both PRODUCER and CONSUMER, shared accessor name)
  applied exactly as documented — `moreh_dot_program_factory.cpp` compute `dfb_bindings`. No multi-binding
  flag needed, matching the audit's brief.
- **Compute hw_config Style A (`metal2_port.md` §Hardware configuration).** `to_compute_hardware_config`
  translated the legacy `ComputeConfigDescriptor` (fed by `get_compute_kernel_config_args`) 1:1 including
  the `dst_full_sync_en → double_buffer_dest` inversion; the moreh_dot inputs are bf16/bfp8 (never
  Float32), so no `unpack_modes` entry is required and the default (empty) matches legacy.
- **DFB metadata via object (whitelist rule 7).** `get_tile_size(cb_id)` in the reader/writer became
  `dfb_in0.get_tile_size()` / `dfb_out.get_tile_size()` — the cb-id is gone, so the object getter is the
  only correct form.

## Friction

- **Gaps:** The reference port called out in the recipe (accumulation, branch
  `akertesz/porting-experiment-accumulation-jun10`) still uses the older method name `create_program_spec`
  and the `TensorArgument{std::cref(...)}` style. The current concept requires `create_program_artifacts`
  and the recipe now says to pass the `MeshTensor` directly (no `std::cref`). The reference is stale vs the
  recipe on both points — following the recipe over the reference resolved it, but a fresher reference
  would remove the ambiguity.
- **Confusion:** The op is a `HasDirectDescriptor` op (no `program_factory_t`). The recipe/ttnn_factory doc
  describe the `program_factory_t` variant world but don't explicitly walk through converting a
  *bare-`create_descriptor`* op into one. The accumulation reference (which does have a `program_factory_t`)
  was what confirmed the target shape. Worth a one-line note in `ttnn_factory.md` that `HasDirectDescriptor`
  ops must grow a nested factory struct + `program_factory_t` variant during the port.

## Open items for downstream

- **Dead compute RTA (ops team, not porter-actionable).** Legacy `moreh_dot_program_factory.cpp:153` passed
  `CoreRuntimeArgs{num_tiles, 1u}` to the compute kernel, but `moreh_dot.cpp` only read index 0
  (`per_core_block_cnt`). The `1u` at index 1 was never read. The port names only `per_core_block_cnt` and
  simply does not carry the dead arg — zero functional change. (Pre-recorded in the audit's Misc anomalies.)
- **Cross-op kernel touches:** none — all three kernels are op-owned; `moreh_dot_backward` is a separate op
  with its own kernel sources.
- **Test coverage note:** primary coverage is `tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_dot.py`
  (nightly tree, not the usual `unit_tests/operations/moreh/`). There is also a graph-capture gtest,
  `tests/ttnn/unit_tests/gtests/test_graph_capture_arguments_morehdot.cpp`. The pytest skips bfloat8_b
  ("not supported in the kernel") and int32 inputs are cast to bf16, so effective correctness coverage is
  bf16.
