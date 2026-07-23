# Metal 2.0 Port Report — `data_movement/moe_expert_token_remap`

## Outcome

**`CAPITULATED`** — grounded stop at the planning step. The op's single factory (`Multicore`) requires **per-mesh-coordinate program specialization** (a device-position value, `flat_mesh_idx`, baked into a reader compile-time argument), and the audit's chosen target concept `MetalV2FactoryConcept` **cannot express per-coordinate variation** — its adapter calls the factory once and stamps one artifact identically across every mesh coordinate. No code was changed; the diff is this report plus `METAL2_PORT_PLAN.md`. Full analysis in the plan's [Deferred / Flagged](METAL2_PORT_PLAN.md) section and in [Handoff points](#handoff-points) below.

This is a framework capability gap, not a porter workaround target. Per the recipe, a grounded capitulation is a success-tier deliverable: the framework's calibration depends on knowing where its assumptions break, and this op is a clean data point for the per-coord-variation gate.

## Provenance

- **Recipe docs (this port):** `05554b94288 2026-07-23 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`
- **Audit docs (inherited):** `05554b94288 2026-07-23 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## TTNN ProgramFactory

### Concept realized
**None.** The audit chose `MetalV2FactoryConcept`; the port did **not** realize it. During planning the porter found the concept cannot express this op's per-mesh-coordinate program specialization, disagreed with the audit's choice, and (per the recipe) stopped and surfaced the disagreement to the invoker rather than unilaterally substituting a different concept or forcing a wrong port. The correct home for this op is the not-yet-implemented `MeshWorkloadSpecFactoryConcept` (the multi-program artifact concept named in [`metal_v2_artifacts.hpp:15-19`](../../../../../../ttnn/api/ttnn/metal_v2_artifacts.hpp#L15-L19)); until it lands, the op stays on its current per-coord `descriptor` adapter.

### Device-op-class edits
- Custom `compute_program_hash` deleted: none (the op already uses the default hash).
- Pybind entry points removed: none (no `create_descriptor` pybind exists; the module binds only the free function `moe_expert_token_remap` — [moe_expert_token_remap_nanobind.cpp:45-56](moe_expert_token_remap_nanobind.cpp#L45-L56)).

No device-op-class edits were made. The port stopped before construction, so nothing outside the factory body was touched.

### Open items
- The op needs a concept that supports per-mesh-coordinate specialization (`MeshWorkloadSpecFactoryConcept`, or equivalent). See [Handoff points](#handoff-points).
- TensorParameter relaxation candidates: none identified (no custom hash to mine, and the port did not proceed to a point where kernel tolerance could be assessed).

## Handoff points

### 1. Port capitulation — `MetalV2FactoryConcept` cannot carry a per-mesh-coordinate compile-time value

- **Op / factory:** `ttnn/cpp/ttnn/operations/data_movement/moe_expert_token_remap`, `MoeExpertTokenRemapDeviceOperation::Multicore` (the op's only factory).
- **The construct that could not convert:** reader compile-time-arg slot 8, `flat_mesh_idx` — the device's linearized mesh coordinate — emitted at [factory:150-151, 164](device/moe_expert_token_remap_program_factory.cpp#L150-L164). It is consumed kernel-side as a **non-type template parameter** (`get_device_expert_indices<linearized_mesh_coord, ...>`, [reader:100-101](../../ccl/all_to_all_combine/device/kernels/dataflow/reader_all_to_all_combine.cpp#L100-L101)), where it selects which experts are local to this device (`mapping_ptr[DeviceIdx] == 1u`, [reader:47](../../ccl/all_to_all_combine/device/kernels/dataflow/reader_all_to_all_combine.cpp#L47)). Its value is distinct on every device (0..N-1).
- **Why mechanical conversion failed:** `MetalV2FactoryConcept::create_program_artifacts` has a fixed 3-arg signature with **no mesh-coordinate parameter** ([operation_concepts.hpp:90-92](../../../../../../ttnn/api/ttnn/operation_concepts.hpp#L90-L92)); the adapter calls it once and stamps the identical spec + run-args onto every coordinate range ([mesh_device_operation_adapter.hpp:884-910](../../../../../../ttnn/api/ttnn/mesh_device_operation_adapter.hpp#L884-L910)). The legacy `descriptor` adapter, by contrast, iterates **per coordinate** precisely because `create_descriptor` accepts `mesh_dispatch_coordinate` ([mesh_device_operation_adapter.hpp:595-599](../../../../../../ttnn/api/ttnn/mesh_device_operation_adapter.hpp#L595-L599)). So the per-device value has a home under the legacy concept and no home under the target concept. No run-arg can substitute (run-args are keyed by node — a core within a device — not by mesh coordinate, and the same run-args are stamped everywhere).
- **The off-rules change that would have been needed:** either (a) a framework concept whose factory entry point is called per mesh coordinate (the planned `MeshWorkloadSpecFactoryConcept`), or (b) a per-coordinate override hook on the artifact. Neither exists today. A porter-level workaround (baking one value, or smuggling the index through a run-arg / manufactured tensor) would silently corrupt 7 of 8 devices on the op's real (multi-device) use.
- **Owner:** Metal 2.0 framework team (factory-concept / adapter owners).
- **Audit-gate feedback:** the [TTNN factory feasibility gate](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/ttnn_factory.md) already lists "per-coord variation" as a BLOCKED (RED) case. The audit missed it here because it reasoned only about the *return type* ("plain `ProgramDescriptor`, not a `WorkloadDescriptor`, so single-program") and did not test whether the `mesh_dispatch_coordinate`-driven CTA makes the compiled program vary per coordinate. See [Friction](#friction) for the suggested gate refinement.

### 2. Cross-op shared reader kernel (informational — made moot by the capitulation)

- **Kernel:** `ttnn/cpp/ttnn/operations/ccl/all_to_all_combine/device/kernels/dataflow/reader_all_to_all_combine.cpp`, instantiated by file path here and co-borrowed by `all_to_all_combine` itself ([all_to_all_combine_program_factory.cpp:250](../../ccl/all_to_all_combine/device/all_to_all_combine_program_factory.cpp#L250)).
- Had the port proceeded, this kernel's Metal 2.0 rewrite would have required the **fork** path (a `reader_all_to_all_combine_metal2.cpp` copy inside this op's directory), because `all_to_all_combine` is not being ported and an in-place rewrite would break it. No fork was created — the port stopped first. Recorded so the next porter of either op sees the coupling. **Port-together set: {`ccl/all_to_all_combine`, `data_movement/moe_expert_token_remap`}.** Note that `all_to_all_combine` is itself a CCL op and its own portability (fabric, global semaphores) has not been assessed here.

## Successes

- **The "stop and report" operating posture (recipe [§Read this first](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md) / [§When the discipline doesn't fit](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#when-the-discipline-doesnt-fit)) fired exactly as designed.** The natural pull was to mechanically translate the factory (it is otherwise a textbook clean port: 5 Case-1 tensor bindings, 3 self-loop + 3 legal-1:1 DFBs, no varargs, no semaphores). The recipe's repeated insistence that a grounded stop is success-tier is what made pausing on the one anomalous CTA the right move instead of powering through.
- **The recipe's directive to plan the "Dropped Plumbing" exhaustively caught the blocker.** Enumerating every CTA's replacement primitive (recipe [Dropped Plumbing](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#dropped-plumbing)) is what surfaced that reader CTA slot 8 has no replacement — it would have been trivial to miss if CTAs were translated by reflex.
- **The CB-endpoint census verified cleanly against the brief** (recipe's "verify, don't transcribe" directive): re-deriving all six CB dispositions from the kernel bodies produced exactly the brief's 3-self-loop / 3-legal-1:1 result.

## Friction

### Gaps
- **The audit's TTNN-factory gate under-specifies the per-coord-variation check.** [`ttnn_factory.md`](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/ttnn_factory.md) frames the feasibility question as single- vs multi-program and leans on "is it a `WorkloadDescriptor`?" as the signal. But a plain `ProgramDescriptor` op can *still* be per-coordinate-specialized — via a `create_descriptor` that takes `mesh_dispatch_coordinate` and bakes the coordinate into a CTA (this op). The gate needs an explicit sub-check: **"Does `create_descriptor` take a `mesh_dispatch_coordinate` and use it to vary any CTA / kernel selection? If yes -> per-coord variation -> BLOCKED, regardless of the return type."** A grep signal for the auditor: `mesh_dispatch_coordinate` used anywhere other than a `.value_or({0,0})` that is then discarded. This op would have been caught pre-port by that check.
- **The recipe has no worked example of the per-coord-variation capitulation.** [§When the discipline doesn't fit](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#when-the-discipline-doesnt-fit) and the ttnn_factory gate both name the case abstractly ("CCL-style"), but a concrete "CTA baked from `mesh_dispatch_coordinate`" example would make it recognizable at a glance. This op is a candidate example.

### Confusion
- **The brief and audit both stated "single-program, not a `WorkloadDescriptor`" as if that settled portability.** It reads as reassuring ("nothing exotic here") and it took reading the adapter source ([mesh_device_operation_adapter.hpp](../../../../../../ttnn/api/ttnn/mesh_device_operation_adapter.hpp), the `descriptor` per-coord loop vs the MetalV2 single-stamp loop) to see that "single ProgramDescriptor per coord" and "single spec stamped identically" are different things — and that this op depends on the difference. The migration guide / ttnn_factory docs describe the MetalV2 adapter as "stamps one spec everywhere" but do not contrast it against the legacy descriptor adapter's per-coord loop, which is the comparison that makes the gap legible.

## Open items for downstream

- **Framework:** implement (or expose) a per-mesh-coordinate factory concept (`MeshWorkloadSpecFactoryConcept`) so this op — and other `mesh_dispatch_coordinate`-specialized `descriptor` ops — can port. This op is a minimal, non-CCL (no fabric, no global semaphores) test case for that concept: everything except the per-coord CTA is a trivial DM port.
- **Audit process:** add the `mesh_dispatch_coordinate` -> CTA per-coord-variation sub-check to the TTNN-factory gate (see [Friction](#friction)); re-audit this op under the refined gate (expected result: RED / BLOCKED-on-framework).
- **Cross-op kernel:** the `reader_all_to_all_combine.cpp` fork coupling with `all_to_all_combine` (Handoff point 2) remains for whoever eventually ports either op.
- **Test coverage / hardware:** the op's sole test, `tests/nightly/t3000/ccl/test_moe_expert_token_remap.py`, requires a **T3000 (2x4 = 8-device) mesh**; the bench available during this port had a single n150 (1 device), so the no-regression baseline could not be exercised here. Any future port attempt must be verified on a T3000 — and note that the per-coord bug described above is invisible on a 1-device mesh (where `flat_mesh_idx` is trivially 0), so a single-device smoke test would give false confidence.
- **Ops-team items the audit flagged (unchanged, not port-related), repeated here so they aren't lost:** `std::ceil` over integer division is a no-op in [device_operation.cpp:78-80](device/moe_expert_token_remap_device_operation.cpp#L78-L80); the writer consumer reads via `get_write_ptr()` at [writer:65-67](device/kernels/dataflow/writer_moe_expert_token_remap.cpp#L65-L67); kernels placed over full `total_cores` but RTAs set only on the `split_work_to_cores` subset ([factory:205-241](device/moe_expert_token_remap_program_factory.cpp#L205-L241)). None are port concerns.
