# Metal 2.0 Port Report — moreh_dot_backward

## Outcome

**PORTED** — the single factory (`MorehDotBackwardOperation::ProgramFactory`, the whole op) converted
to `MetalV2FactoryConcept` / `create_program_artifacts`. Build and test verification are the
orchestrator's (this porter did not build or run tests, per orchestration constraints).

## Provenance

- **Recipe docs (this port):** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`
- **Audit docs (inherited):** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## TTNN ProgramFactory

### Concept realized
`MetalV2FactoryConcept` — as the audit chose. The op was on the `HasDirectDescriptor` shape
(`create_descriptor` directly on the device-op, no `program_factory_t`). Because `MetalV2FactoryConcept`
requires `create_program_artifacts` on a factory referenced by a `program_factory_t` variant (and a
device-op with `create_program_artifacts` but no `program_factory_t` does **not** satisfy
`DeviceOperationConcept` — see `ttnn/api/ttnn/operation_concepts.hpp`), the port introduced:
- a nested `struct ProgramFactory` with `create_program_artifacts` (device_operation.hpp),
- `using program_factory_t = std::variant<ProgramFactory>;`,
- `static program_factory_t select_program_factory(...)` returning `ProgramFactory{}` (device_operation.cpp).

This is the forced concept-migration wiring, not a freelance device-op edit. `select_program_factory` was
added explicitly even though a single-type variant auto-returns, to remove any ambiguity.

### Device-op-class edits
- Custom `compute_program_hash` deleted: none (op had none).
- Pybind entry points removed: none — nanobind binds the plain `ttnn::moreh_dot_backward` function, not a
  factory entry point; no pybind surface changed.

### Open items
- Relaxation candidates: none identified.
- The device-op's `tensor_args_t::output_tensors` carries a vestigial comment ("thanhnguyen's mistake",
  `moreh_dot_backward_device_operation.hpp:27`); cosmetic, op-level, left untouched per scope discipline.

## Handoff points

None. No capitulation, no cross-op kernel touches, no `sem::`/`tensor::` boundary violations, no
kernel-lib gaps, no framework gaps.

## Successes

- **Rule 6 (conditional bindings) fired correctly.** `input_grad` / `other_grad` are `std::optional`
  outputs; an absent output has no MeshTensor to bind. The audit/brief said to bind them conditionally
  and keep the `has_*_grad` guards. Rule 6's mechanics resolved the tension: a conditionally-bound tensor
  token (`tensor::s0` / `tensor::s1`) does not exist when the output is absent, so the runtime guard
  cannot gate its *reference*. The condition moved to writer `compiler_options.defines`
  (`HAS_INPUT_GRAD` / `HAS_OTHER_GRAD`) and the writer `#ifdef`-gates the accessor construction and each
  write block (`writer_moreh_dot_backward.cpp`). The `out0`/`out1` DFBs stay bound 1P+1C unconditionally
  (the sanctioned "declare the conditional-side endpoint unconditionally" shape), matching the audit's
  "each CB stays 1P+1C in every config."
- **Kernel-side whitelist rule 7 (`get_tile_size(cb_id)` → object getter).** Used `dfb.get_entry_size()`
  in the reader/writer (the documented DM-kernel-safe getter; the DFB `entry_size` equals the legacy
  per-CB tile bytes). `get_entry_size()` is available on DM builds, unlike the descriptor-gated
  `get_tile_size()` member.

## Friction

### Gaps
- **Recipe says "keep the has_*_grad runtime guards" (brief) but rule 6 forces the writer's guard to a
  compile-time define.** The brief's "Watch for" says to keep the `has_input_grad`/`has_other_grad`
  runtime guards in *all* affected kernels, yet a conditionally-bound tensor accessor cannot be referenced
  under a runtime guard (the token is absent when unbound). Resolution: reader and compute keep their
  runtime `has_*_grad` RTA guards (their bindings are unconditional), while the **writer** drops those two
  RTAs and uses the `HAS_*_GRAD` compile defines (its tensor bindings are conditional). The brief and
  rule 6 are reconcilable but the brief's phrasing ("keep the runtime guards") reads as if it applies to
  the writer too — a doc clarification opportunity: for a kernel that *only* touches a conditionally-bound
  resource behind the guard, the guard becomes the compile define.

### Confusion
- **The branch's transpose "port" (commit `68f020d`) is a misleading reference.** Its factories still
  return `ProgramDescriptor` from `create_descriptor`, not `ProgramArtifacts` from
  `create_program_artifacts`, despite the commit message "…subset to MetalV2FactoryConcept". Following the
  recipe's explicit "don't lean on already-ported ops" guidance, I used the authoritative
  `migration_guide.md` / `ttnn_factory.md` examples plus the quasar `fold`/`binary_ng` factories for the
  concrete API surface (header paths, `Group`/`Table` types, `MakeRuntimeArgsForSingleNode`,
  `DFBBinding`/`TensorBinding` field names). Noting it so the next porter isn't misled by the transpose
  diff.

## Open items for downstream

- **Cross-op kernel touches:** none — all three kernels are op-owned and edited in place.
- **`start_id` RTA is always `0`** (reader + writer). Ported faithfully as an RTA per the brief; a future
  cleanup could drop it (single-core op, no work split). Not porter-actionable.
- **Compute-config style:** the op sets a bare Metal `ComputeConfigDescriptor{}` (Style B, all defaults),
  ported to `ComputeGen1Config{}` (defaults coincide: HiFi4, Precise, `enable_32_bit_dest=false`,
  `double_buffer_dest=true`). No `unpack_modes` needed (bf16/bf8 inputs, not Float32).

## Files created / modified

- `device/moreh_dot_backward_device_operation.hpp` — replaced the direct `create_descriptor` with a nested
  `ProgramFactory::create_program_artifacts`, added `program_factory_t` + `select_program_factory` decl;
  swapped `program_descriptors.hpp` include for `ttnn/metal_v2_artifacts.hpp`, added `<variant>`.
- `device/moreh_dot_backward_device_operation.cpp` — added `select_program_factory` returning `ProgramFactory{}`.
- `device/moreh_dot_backward_program_factory.cpp` — rewrote as `create_program_artifacts`: 5 DataflowBufferSpecs,
  5 TensorParameters (2 conditional), 3 KernelSpecs with DFB/tensor bindings, 1 WorkUnitSpec, ProgramRunArgs.
- `device/kernels/reader_moreh_dot_backward.cpp` — named args, `tensor::s0/s1/s2`, `dfb::in0/in1/in2`,
  `get_entry_size()`; dropped buffer-address RTAs + `TensorAccessorArgs`.
- `device/kernels/writer_moreh_dot_backward.cpp` — named args, `dfb::out0/out1`, `get_entry_size()`;
  conditional tensor accessors `#ifdef`-gated on `HAS_INPUT_GRAD`/`HAS_OTHER_GRAD`; dropped CB-index CTAs +
  buffer-address RTAs + `TensorAccessorArgs` + the two `has_*_grad` RTAs.
- `device/kernels/moreh_dot_backward.cpp` — named args; `tt::CBIndex::c_*` → `dfb::in0/in1/in2/out0/out1`
  at LLK call sites and DataflowBuffer constructions.
- `METAL2_PORT_PLAN.md`, `METAL2_PORT_REPORT.md` — new artifacts (committed alongside the audit brief/report).
