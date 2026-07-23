# Metal 2.0 Port Report — `data_movement/moe_routing_remap`

## Outcome

**CAPITULATED.** The single `SingleCore` factory cannot be ported to `MetalV2FactoryConcept`: the op requires **per-mesh-coordinate** runtime-arg variation (`device_weights_count_offset`), which the single-program factory concept cannot express. No code was constructed or changed — this is a grounded stop discovered at the "Plan the spec" gate, surfaced to the invoker, and confirmed by both a framework read and a test/usage survey. Blocked pending framework work (a multi-program Metal 2.0 factory concept). This is a success-tier outcome per the recipe's operating posture, not a failed port.

## Provenance

- **Recipe docs (this port):** `597581e6151 2026-07-23 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`
- **Audit docs (inherited):** `597581e6151 2026-07-23 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper` *(copied from `METAL2_PORT_BRIEF.md`)*

## TTNN ProgramFactory

### Concept realized
**None — the audit's `MetalV2FactoryConcept` choice does not fit this op, and I stopped rather than override it.** Per the recipe's instruction ("if you find yourself disagreeing with the audit's choice, stop and surface the disagreement to the invoker — do not unilaterally override"), I surfaced the disagreement to the invoker before writing anything; the invoker directed the documented capitulation. The concept mismatch is a genuine framework capability gap (below), not a re-derivation of the concept choice on stylistic grounds.

### Device-op-class edits
- Custom `compute_program_hash` deleted: none (the op has no custom hash).
- Pybind entry points removed: none (`create_descriptor` is not pybound; the nanobind file binds only the host op).

*No files were modified — the port did not reach construction.*

### Open items
- When a multi-program Metal 2.0 factory concept lands (the framework's own "future `MeshWorkloadSpecFactoryConcept`"), this op is a clean candidate. `METAL2_PORT_PLAN.md` carries the fully worked-out spec design (kernels, three DFBs including the `c_2` self-loop, two Case-1 tensor parameters, single work unit) — a head start for that porter. The only piece that needs the new concept is emitting the per-coordinate `device_weights_count_offset`.
- No tensor-parameter relaxation candidates were identified (strict spec matching is correct here).

## Handoff points

### 1. Port capitulation — per-mesh-coordinate runtime-arg variation is not expressible on `MetalV2FactoryConcept`

- **Op / factory:** `ttnn/cpp/ttnn/operations/data_movement/moe_routing_remap`, `MoeRoutingRemapDeviceOperation::SingleCore` (the op's only factory).
- **The construct that cannot convert:** the reader runtime arg `device_weights_count_offset` = `mesh_coordinate[cluster_axis] * non_zero_per_device`, computed per mesh coordinate and baked into each coordinate's program ([device/moe_routing_remap_program_factory.cpp#L15-L25](device/moe_routing_remap_program_factory.cpp#L15-L25), [L146-L151](device/moe_routing_remap_program_factory.cpp#L146-L151)), and consumed by the reader kernel as a loop skip-count ([device/kernels/dataflow/reader_moe_routing_remap.cpp#L50-L62](device/kernels/dataflow/reader_moe_routing_remap.cpp#L50-L62)). The legacy `create_descriptor` takes `const std::optional<ttnn::MeshCoordinate>&` ([device/moe_routing_remap_device_operation.hpp#L45-L49](device/moe_routing_remap_device_operation.hpp#L45-L49)), and the framework's descriptor adapter dispatches **one program per mesh coordinate** with a distinct coordinate ([`mesh_device_operation_adapter.hpp#L595-L599`](../../../../../../ttnn/api/ttnn/mesh_device_operation_adapter.hpp#L595-L599)).
- **Why mechanical conversion fails:**
  - `MetalV2FactoryConcept::create_program_artifacts(attributes, tensor_args, tensor_return_value)` receives **no mesh coordinate** ([`operation_concepts.hpp#L90-L92`](../../../../../../ttnn/api/ttnn/operation_concepts.hpp#L90-L92)) and returns a **single** `ProgramArtifacts` (one `ProgramSpec` + one `ProgramRunArgs`) ([`metal_v2_artifacts.hpp#L23-L33`](../../../../../../ttnn/api/ttnn/metal_v2_artifacts.hpp#L23-L33)).
  - The `MetalV2MeshWorkloadFactoryAdapter` calls the factory **once** and stamps the same spec and the *same* `run_params` onto **every** coordinate range ([`mesh_device_operation_adapter.hpp#L884`](../../../../../../ttnn/api/ttnn/mesh_device_operation_adapter.hpp#L884), [L904-L909](../../../../../../ttnn/api/ttnn/mesh_device_operation_adapter.hpp#L904-L909)). The cache-hit path refreshes only tensor bindings ([L919-L943](../../../../../../ttnn/api/ttnn/mesh_device_operation_adapter.hpp#L919-L943)).
  - `ProgramRunArgs` keys runtime args by `NodeCoord` ([`program_run_args.hpp#L69`](../../../../../../tt_metal/api/tt-metalium/experimental/metal2_host_api/program_run_args.hpp#L69)), and `NodeCoord = CoreCoord` — a Tensix core within *one* chip ([`node_coord.hpp#L29`](../../../../../../tt_metal/api/tt-metalium/experimental/metal2_host_api/node_coord.hpp#L29)). There is no mesh-coordinate axis, and all work here is on core `{0,0}` of every chip.
  - Net: a single-program port hands every device *device-0's* offset. It is not merely a latent risk — it produces wrong output on every execution (see the usage survey below).
  - No in-scope repair exists: computing the offset device-side relocates host computation into the kernel (off-whitelist) and the kernel has no mesh-coordinate source; the input is replicated across the mesh by contract, so the variation cannot be pushed into tensor data; the op has a single factory (the atomic unit), so there is no partial port that builds.
- **The framework change that would unblock it (sketch):** a multi-program Metal 2.0 factory concept — the "future `MeshWorkloadSpecFactoryConcept`" the framework names ([`metal_v2_artifacts.hpp#L21-L22`](../../../../../../ttnn/api/ttnn/metal_v2_artifacts.hpp#L21-L22)) — that returns a per-coordinate artifact (an `unordered_map<MeshCoordinateRange, ProgramSpec>` plus per-range `ProgramRunArgs`) and lowers through the **already-existing** host API `MakeMeshWorkloadFromSpecs` ([`program.hpp#L48-L51`](../../../../../../tt_metal/api/tt-metalium/experimental/metal2_host_api/program.hpp#L48-L51)). The per-coordinate capability exists at the Metal 2.0 host-API layer today; only the TTNN factory-concept bridge (a coordinate-aware `create_program_artifacts` and an adapter that stamps per range) is missing.
- **Owner:** Metal 2.0 framework / TTNN device-operation team.

### 2. Audit gap — the TTNN-factory feasibility gate should have fired RED for this op

- The audit classified the op **GREEN** on `MetalV2FactoryConcept` ([`METAL2_PREPORT_AUDIT.md`](METAL2_PREPORT_AUDIT.md), TTNN Readiness rows), treating the mesh-coord-aware `create_descriptor` as a portable "`descriptor`-with-per-coord-dispatch" variant and assuming the porter could "wire the per-coord dispatch through `MetalV2FactoryConcept` correctly" (audit "Mesh-coord-aware `create_descriptor`" and "Recipe notes" sections).
- That wiring does not exist (Handoff 1). This is the `ttnn_factory.md` feasibility gate's **"multi-program / per-coord variation → BLOCKED"** case ([`ttnn_factory.md`](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/ttnn_factory.md#feasibility-gate)). The audit's "morally single-program" heads-up does **not** apply, because the per-coordinate difference here is a non-tensor runtime scalar, not "only the tensor data differs."
- **Recommendation:** correct the audit for this op to RED (blocked on the multi-program concept), and tighten the audit gate so a per-coordinate `create_descriptor` whose per-coordinate payload is **anything other than tensor identity** (here, an RTA scalar) is classified BLOCKED on the single-program concept. The audit's own "Recipe notes" already flagged uncertainty about "how per-coord dispatch maps onto `MetalV2FactoryConcept`" — the concrete answer is "it does not map on the single-program concept."
- **Owner:** audit-doc maintainers.

## Successes

- **The `ttnn_factory.md` feasibility gate named exactly this failure.** Its "multi-program / per-coord variation → BLOCKED" bullet and the "legacy `MeshWorkload` is not automatically multi-program" heads-up ([`ttnn_factory.md` — Feasibility gate](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/ttnn_factory.md#feasibility-gate)) gave a precise vocabulary to recognize and characterize the blocker rather than improvising around it.
- **The recipe's "stop and surface an audit disagreement" instruction fired correctly.** The [Plan-the-spec / TTNN-ProgramFactory guidance](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#ttnn-programfactory) steered me to surface the disagreement to the invoker instead of either overriding the audit or shipping a knowingly-broken single-program port. The operating-posture framing of a grounded stop as success-tier ([recipe — Operating posture](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#read-this-first)) made the capitulation the clear correct action.

## Friction

### Gaps
- **The recipe assumes per-coord dispatch maps onto `MetalV2FactoryConcept`, but does not say how — because it cannot.** The audit's "Recipe notes" asked for "a one-line acknowledgement ... about how per-coord dispatch maps onto `MetalV2FactoryConcept`." Investigating, the answer is that a per-coordinate payload that is *not* tensor identity is **BLOCKED** on the single-program concept. Neither the recipe nor `ttnn_factory.md` states this crisply for the *auditor* — the "morally single-program" heads-up reads as an escape hatch that a reader can over-apply. Suggested fix: in `ttnn_factory.md`'s feasibility gate, add an explicit test — "does any per-coordinate payload beyond tensor identity (an RTA scalar, a CTA, a define) vary across mesh coordinates? → BLOCKED on the single-program concept" — and cross-link it from the audit's concept-classification bullets.

### Confusion
- **`create_program_artifacts` gives no signal that it is coordinate-blind.** The signature simply omits a coordinate, which is easy to read past. A one-line comment on the concept (or on `ProgramArtifacts`) stating "this artifact is stamped identically across all mesh coordinates; per-coordinate variation is not supported until `MeshWorkloadSpecFactoryConcept`" would make the constraint legible at the point of use. (The `ProgramArtifacts` header does say a "future" concept will handle multi-program ops — but from the *porter's* side, mid-construction, the constraint isn't obvious.)

## Open items for downstream

- **Cross-op kernel touches:** none — no code was modified, and the op owns both its kernels.
- **Incidental defects noticed during inventory (out of scope for a port; route to the ops team, not this diff):**
  - Operator-precedence bug in `validate_on_program_cache_miss` ([device/moe_routing_remap_device_operation.cpp#L37-L39](device/moe_routing_remap_device_operation.cpp#L37-L39)): `expert_parallel_size == (cluster_axis == 0) ? num_cols() : num_rows()` parses as `(expert_parallel_size == (cluster_axis == 0)) ? ...`, so the intended cluster-size check effectively never fires. In practice the only size enforcement is the Python test's own `pytest.skip` guard. (Also flagged by the audit's Misc anomalies.)
  - `cluster_axis` docstring contradicts the code: the nanobind docstring says "0: columns, 1: rows" ([moe_routing_remap_nanobind.cpp#L42](moe_routing_remap_nanobind.cpp#L42)), but the kernel, the test reference, and the validate error message all use axis 0 = rows.
  - Unused cross-family include `ttnn/operations/ccl/common/kernels/moe_utils.hpp` in both kernels ([reader L9](device/kernels/dataflow/reader_moe_routing_remap.cpp#L9), [writer L10](device/kernels/dataflow/writer_moe_routing_remap.cpp#L10)) — no symbol referenced; removable, and it is the op's only cross-family coupling.
- **Test coverage note:** the op's sole test, [`tests/nightly/t3000/ccl/test_moe_routing_remap.py`](../../../../../../tests/nightly/t3000/ccl/test_moe_routing_remap.py#L65-L108), runs **only** on a real 2×4 (8-device) mesh with `expert_parallel_size` ∈ {2, 4} (both > 1); its skip guard admits `ep=2, cluster_axis=0` and `ep=4, cluster_axis=1`. The one production caller, gpt-oss decode ([`models/demos/gpt_oss/tt/experts/decode.py#L60-L62`](../../../../../../models/demos/gpt_oss/tt/experts/decode.py#L60-L62)), uses `expert_parallel_size=4, cluster_axis=0` under `if ep > 1`. So the per-coordinate offset is load-bearing in 100% of executions — confirming the blocker is real in practice, not merely in principle. There is no single-device coverage that would let a single-program port pass even accidentally.
