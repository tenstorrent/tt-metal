# Metal 2.0 Port Report — `ttnn/cpp/ttnn/operations/moreh/moreh_sgd`

## Outcome

**PORTED.** The single program factory (`create_descriptor` → `create_program_artifacts`) is converted
to `MetalV2FactoryConcept`, together with all three kernel entry points (reader / writer / compute).
Build and test verification are the **orchestrator's** (this session did not build or run tests, per
the orchestration constraints).

## Provenance

- **Recipe docs (this port):** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`
- **Audit docs (inherited):** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## Files created / modified

Created:
- `METAL2_PORT_PLAN.md` — inventory + spec plan + the intermediate-DFB conditionality decision.
- `METAL2_PORT_REPORT.md` — this report.

Modified (all under the op directory):
- `device/moreh_sgd_device_operation.hpp` — replaced the `HasDirectDescriptor` shape with the Metal 2.0
  wiring: added `#include "ttnn/metal_v2_artifacts.hpp"` + `<variant>`, dropped
  `<tt-metalium/program_descriptors.hpp>`, removed the `create_descriptor` declaration, added the
  nested `struct MorehSgdProgramFactory { create_program_artifacts(...); }`,
  `using program_factory_t = std::variant<MorehSgdProgramFactory>`, and a `select_program_factory`
  declaration.
- `device/moreh_sgd_device_operation.cpp` — added the `select_program_factory` definition (returns
  `MorehSgdProgramFactory{}`). Nothing else in the device-op class changed.
- `device/moreh_sgd_program_factory.cpp` — full rewrite of the factory body to build a `ProgramSpec` +
  `ProgramRunArgs` (`ProgramArtifacts`). Named DFB/tensor/kernel resources replace magic CB indices and
  buffer-address RTAs; `TensorBinding`s replace the `TensorAccessorArgs` plumbing; two per-core-group
  compute `KernelSpec`s preserve the work-split CTA multiplicity; optional momentum DFBs/tensors are
  conditionally bound; `unpack_modes` added for the FP32 path.
- `device/kernels/reader_moreh_sgd.cpp` — buffer-address RTAs + `TensorAccessorArgs` dropped for
  `TensorAccessor(tensor::…)`; CB indices → `dfb::…`; scalar RTAs → `get_arg(args::…)`;
  `get_tile_size(cb)` → `dfb.get_tile_size()`; added `experimental/kernel_args.h`.
- `device/kernels/writer_moreh_sgd.cpp` — same class of changes as the reader.
- `device/kernels/moreh_sgd.cpp` (compute) — CB indices → `dfb::…` (including the `cb_grad_tmp` /
  `cb_momentum_tmp` selection variables, which now hold `dfb::…` ids and reconstruct via
  `DataflowBuffer(uint16_t)`); `get_compile_time_arg_val(0)` → `get_arg(args::num_tiles)`;
  `binary_op_init_common` takes `dfb::…`; the two momentum `DataflowBuffer` constructions are newly
  `#ifdef`-gated to match the conditional host binding; added `experimental/kernel_args.h`.

`METAL2_PREPORT_AUDIT.md` and `METAL2_PORT_BRIEF.md` are pre-existing audit inputs (committed
alongside).

## Port summary

`moreh_sgd` is a single-program `descriptor`-concept op (SGD optimizer step). The reader streams
`param_in` / `grad` / (optional) `momentum_buffer_in` tiles into DFBs and fills a 5-entry scalar-args
DFB with the hyperparameters; the compute kernel does the SGD update through four intermediate DFBs;
the writer streams `param_out` / (optional) `momentum_buffer_out` back out. Work is split across the
core grid: reader/writer run on all cores, compute is split into two per-core-group `KernelSpec`s.

Metal 2.0 shape:
- **Factory concept:** `MetalV2FactoryConcept`. The op was `HasDirectDescriptor` (create_descriptor on
  the op struct, no `program_factory_t`); since the framework only detects `create_program_artifacts`
  as a `program_factory_t` variant alternative, the port introduces the nested factory struct + a
  single-alternative variant + `select_program_factory` (same wiring the sibling moreh ports adopted).
- **10 DFBs** (1:1 with legacy CBs). `param_in`/`grad`/`param_out`/`scalar_args`/`tmp1`–`tmp4`
  unconditional; `momentum_in` (c_2) and `momentum_out` (c_17) conditional.
- **5 tensor parameters** (all Case 1, via `TensorAccessor`): `param_in`/`grad`/`param_out`
  unconditional; `momentum_in`/`momentum_out` conditional (a `TensorArgument` is mandatory for every
  declared `TensorParameter`, so the optionals are declared only when their tensor exists).
- **Preserved work-split multiplicity:** two compute `KernelSpec`s over disjoint core groups, each with
  its own live per-group CTA `num_tiles` (the compute kernel reads it as its loop count — not demoted
  to an RTA).
- **Self-loop** intermediates `tmp1`–`tmp4` (compute bound both PRODUCER and CONSUMER).
- **Two work units** (`group_1`/`group_2`), reader/writer in both (their union = all cores); a single
  work unit when `core_group_2` is empty.
- **No** custom hash, **no** pybind removal (nanobind binds `&ttnn::moreh_sgd`), **no** op-owned
  tensors, **no** semaphores.

### Custom `compute_program_hash`
None — nothing to delete.

### Device-op-class edits
- Custom `compute_program_hash` deleted: none.
- Pybind entry points removed: none (nanobind binds the op, not `create_descriptor`).
- Wiring edits (forced by `MetalV2FactoryConcept` detection): nested factory struct +
  `program_factory_t` + `select_program_factory` (device-op header/cpp). These are the standard
  `HasDirectDescriptor` → variant conversion, not a behavioral device-op change.

## Test command(s)  *(build + run are the orchestrator's)*

Build (orchestrator):
```
./build_metal.sh --build-tests
```

Tests — the op's only coverage is a single nightly pytest (no C++ gtests exist for moreh_sgd). The full
parametrization exercises every compile-define path (momentum {0, 7.7}, dampening {0, 0.5},
weight_decay {0, 2.2}, nesterov {T,F}, momentum_initialized {T,F}) and both `fp32_dest_acc_en` values
— so it covers the conditional momentum bindings and the FP32 `unpack_modes` path:
```
pytest tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_sgd.py -v
```
All cases passing pre-port should continue to pass post-port (no behavior change intended).

## Handoff points

None. No change was needed outside the op directory; no `sem::`/`tensor::` boundary violation; no
kernel-lib gap; no framework gap.

## Successes

- **Kernel rule 6 (conditional / optional bindings)** steered the momentum DFBs correctly. c_2/c_17
  cross kernels (reader↔compute, compute↔writer), so an ungated binding on one side with a gated
  counterpart would fail the validator's ≥1-producer/≥1-consumer check. Gating the DFB spec + both
  endpoint bindings + tensor parameter + `#ifdef` object construction under a single condition
  (`moreh_sgd_program_factory.cpp` `bind_momentum_in` / `bind_momentum_out`;
  `moreh_sgd.cpp:11-18`) keeps host and kernel consistent.
- **Preserved-multiplicity anti-pattern warning** fired correctly: the sibling `moreh_abs_pow` port
  collapsed its two per-group compute descriptors into one KernelSpec, which is *only* valid because
  its CTA was dead. moreh_sgd's compute genuinely reads the per-group CTA as its loop count
  (`moreh_sgd.cpp:37` `get_arg(args::num_tiles)`), so the two KernelSpecs are kept — a CTA→RTA
  demotion would have been the forbidden anti-pattern.
- **CB→DFB whitelist rule 7** (`get_tile_size(cb)` → `dfb.get_tile_size()`) applied cleanly in the
  reader/writer even though the audit had called those free-function calls "sanctioned" for Device 2.0
  (the cb id is gone in Metal 2.0, so the object getter is the correct form).

## Friction

### Gaps
- **The recipe/`ttnn_factory.md` do not spell out the `HasDirectDescriptor` → `program_factory_t`
  wiring.** `MetalV2FactoryConcept` is *not* detected directly on the op struct (framework
  `mesh_device_operation_adapter` / `operation_concepts.hpp:97,164`); a bare
  `create_program_artifacts` on `MorehSgdOperation` would silently not be selected. The required shape
  (nested factory struct + single-alternative `program_factory_t` variant + `select_program_factory`)
  is documented only in sibling moreh ports' `METAL2_PORT_*.md`, not in the recipe. A short note in
  `ttnn_factory.md`'s "Device-operation-class edits" would save every direct-descriptor porter a
  detour.

### Confusion
- **Brief vs. self-loop L1 for the four intermediates.** The brief asked to "classify per (CB, config)
  and self-loop where live" — implying per-config gating of `tmp1`–`tmp4`. But (a) legacy allocates
  all four CBs unconditionally and the compute kernel constructs all four wrappers unconditionally, so
  gating deviates from legacy L1 usage rather than matching it; (b) a self-loop supplies both endpoints
  from the single compute kernel, so an untouched-in-this-config intermediate still validates; and
  (c) gating would need compound `#ifdef`s over the `cb_grad_tmp`/`cb_momentum_tmp` selection logic.
  I kept them unconditional (see the plan's "Design decision" section). This is the faithful,
  lower-risk reading, but the brief's wording pushed the other way — a recipe example distinguishing
  "cross-kernel conditional DFB (must gate)" from "compute-internal self-loop the kernel constructs
  unconditionally (bind unconditionally)" would remove the ambiguity.

## Open items for downstream

- **`unpack_modes` for the FP32 path is a best-effort read of the validator rule.** Under
  `fp32_dest_acc_en` the intermediate DFBs (`scalar_args`, `tmp1`–`tmp4`) are Float32 and bound as
  compute CONSUMER, so I added explicit `UnpackMode::UnpackToSrc` entries for all five (legacy set no
  `unpack_to_dest_mode`). If the validator only requires entries for DFBs the *compiled* binary
  actually unpacks (vs. all consumer-bound Float32 DFBs), some entries are harmlessly redundant. The
  `fp32_dest_acc_en=True` test case will confirm; flagging in case the requirement is narrower than
  "every consumer-bound 32-bit DFB."
- **RTA → CRTA candidate (not converted — would change dispatch semantics).** The reader's `lr`,
  `momentum`, `dampening`, `weight_decay`, `one` RTAs carry the *same* value on every core (they are
  scalar hyperparameters). They are really common runtime args. Left as per-node RTAs per the recipe
  (RTA→CRTA is a separate cleanup, not port work).
- **Cross-op kernel touches:** none. The kernels `#include` the shared
  `ttnn/cpp/ttnn/kernel/{dataflow,compute}/moreh_common.hpp` helper pools (function-call escapes,
  already `DataflowBuffer`-native), which were left untouched. A future Metal 2.0 rewrite of any
  `moreh_common` helper is a single shared change across every moreh op that uses it — coordinate as
  one unit (per the brief's port-together note).
