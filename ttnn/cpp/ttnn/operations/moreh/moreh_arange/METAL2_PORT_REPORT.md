# Port Report — moreh/moreh_arange

## Outcome

**`CAPITULATED`** — grounded stop on a **framework** limitation discovered at on-device
verification: the MetalV2 factory adapter cannot dispatch a **tensorless** op (an op with no
input tensor whose only tensor is a freshly-created output). moreh_arange is exactly such an
op. This is the same wall that keeps `debug` / ApplyDeviceDelay RED on MetalV2.

**No regression.** All Metal 2.0 code changes were reverted; the op's four code files
(`device/moreh_arange_device_operation.hpp`, `device/moreh_arange_program_factory.cpp`, and
both `device/kernels/writer_moreh_arange*.cpp`) are restored **byte-for-byte** to their
pre-port legacy `create_descriptor` / `ProgramDescriptor` form (verified identical to the
port commit's parent `13f38de2bbd`). The op keeps working exactly as before.

A grounded capitulation is a success-tier deliverable (recipe §When the discipline doesn't
fit): it gives the framework team a precise, reproducible signal of where the MetalV2 adapter
assumption breaks, rather than a workaround buried in a diff.

## What was attempted and why it failed

The port was fully implemented and **built**; the orchestrator ran it on-device: **8/9
tests passed**. The single failure was the create-output path (no optional output tensor
passed):

```
test_arange[tilized=True-dtype=None-optional_output=False-start_end_step=[0, 32, 1]]
TT_FATAL @ ttnn/api/ttnn/mesh_device_operation_adapter.hpp:879: first_tensor.has_value()
MetalV2 factory adapter requires at least one Tensor in tensor_args to source the MeshDevice
```

**Root cause (framework, out of the porter's scope).** In the MetalV2 adapter's
`create_mesh_workload` (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:867-881`) the
MeshDevice is sourced *exclusively* from `tensor_args`:

```cpp
auto first_tensor = ttsl::reflection::get_first_object_of_type<tt::tt_metal::Tensor>(tensor_args);
TT_FATAL(first_tensor.has_value(),
    "MetalV2 factory adapter requires at least one Tensor in tensor_args to source the MeshDevice");
auto* mesh_device = first_tensor.value().device();
```

There is **no** alternative device-sourcing hook: the adapter ignores
`tensor_return_value` and `operation_attributes` (including moreh_arange's own
`operation_attributes.mesh_device`) for this purpose. The adapter's own comment records the
assumption: *"Op factories satisfying this concept are tensor-driven, so first_tensor is
always populated for current callers."*

moreh_arange breaks that assumption:
- `tensor_args_t` is `{ const std::optional<Tensor>& output }` — no input tensor.
- On the **optional-output** path the caller's output lands in `tensor_args` → device sourced
  fine → the 8 passing cases.
- On the **create-output** path `tensor_args.output` is `std::nullopt`; the output is created
  by `create_output_tensors` and delivered as `tensor_return_value`. `tensor_args` holds no
  device tensor → the `TT_FATAL` fires.

Notably, binding resolution is *not* the blocker: `collect_mesh_tensors`
(`mesh_device_operation_adapter.hpp:822-831`) enumerates **both** `tensor_args` *and*
`tensor_return_value`, so the output `TensorBinding` resolves correctly. The **only** gap is
device sourcing at line 876.

**No supported in-op-dir fix exists.** `tensor_args` is constructed in the device-op dispatch
layer (`ttnn::prim::moreh_arange`, `device/moreh_arange_device_operation.cpp`) from the
caller's arguments — *before* `create_output_tensors` runs — and that layer plus the adapter
are outside the porter's writeable surface (recipe §Scope discipline). There is no way, from
within the op directory, to seed a device tensor into `tensor_args` on the create path, and
fabricating one would be exactly the kind of dispatch-corrupting hack the recipe forbids.
Reverting to the legacy `create_descriptor` (whose descriptor adapter path sources the device
differently and handles output-creation correctly) is the only no-regression outcome.

## Provenance

- **Recipe docs (this port):** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`
- **Audit docs (inherited):** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

## TTNN ProgramFactory

### Concept realized
None — reverted to the legacy `descriptor` concept (direct `create_descriptor` on
`MorehArangeOperation`). The MetalV2 port was implemented (nested `MorehArangeProgramFactory`
+ `program_factory_t`) and worked for the optional-output path, but is not viable for the
create-output path pending the framework fix below, so it was fully reverted.

### Device-op-class edits
None survive (all reverted). No custom hash existed; no pybind surface changed.

### Open items
See Handoff points.

## Handoff points

### Framework: MetalV2 adapter tensorless-dispatch block (BLOCKER)
- **Owner:** Metal 2.0 / device-operation framework team.
- **Site:** `ttnn/api/ttnn/mesh_device_operation_adapter.hpp:867-881`
  (`MetalV2MeshWorkloadFactoryAdapter::create_mesh_workload`).
- **Problem:** the MetalV2 adapter sources the MeshDevice only from `tensor_args`
  (`get_first_object_of_type<Tensor>(tensor_args)` + hard `TT_FATAL`). Ops with no input
  tensor whose output is framework-created (delivered as `tensor_return_value`) hit the
  `TT_FATAL` on the create-output path. moreh_arange is the concrete example; `debug` /
  ApplyDeviceDelay is the previously-known instance.
- **What the port needed:** a supported device-sourcing fallback for tensorless dispatch —
  e.g. consult `tensor_return_value` (already enumerated by `collect_mesh_tensors` at
  line 829) when `tensor_args` has no device tensor, or source from an explicit device on
  `operation_attributes` where the op provides one (moreh_arange carries
  `operation_attributes.mesh_device`). Any of these would unblock the port with no change to
  the op-directory code that was written.
- **Reproduce:** `pytest tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_arange.py -k "not optional_output"` (the create-output cases) against a MetalV2 build (8/9 → the create-output case fatals).

### Audit gap (route back to the auditor)
- The audit (`METAL2_PREPORT_AUDIT.md`) cleared GREEN without flagging the empty-`tensor_args`
  path. The op's `tensor_args_t` carries only a `std::optional<Tensor>` output and no input
  tensor, so on the create-output dispatch `tensor_args` holds no device tensor — the exact
  condition the MetalV2 adapter `TT_FATAL`s on. The audit's TTNN-factory gate should include a
  **tensorless-dispatch check**: "does a device tensor always reach `tensor_args`, or can the
  op run with an empty/optional-only `tensor_args` (output-creating, no input)?" If the latter,
  the op is BLOCKED on the same framework limitation as `debug`. Recommend the audit treat this
  as RED (or explicitly gated) until the adapter fix lands. `METAL2_PREPORT_AUDIT.md` and
  `METAL2_PORT_BRIEF.md` are retained unchanged as the inputs this port ran against.

## Successes

- **Recipe off-ramp calibration.** The build-then-fail-on-device sequence is exactly the
  "the audit cleared the features known at audit time, but the port discovered something it
  missed" case the recipe anticipates. The port implementation itself was mechanically clean
  (self-loop DFB, TensorParameter, named args all landed and passed 8/9); the block is purely
  the framework's tensorless-dispatch assumption. Stopping and reverting rather than hacking a
  fake `tensor_args` tensor is the prescribed response.

## Friction

### Gaps
- **Audit had no tensorless-dispatch gate.** The single most actionable finding: the audit
  should catch an op whose `tensor_args` can be device-tensor-empty at dispatch, because the
  MetalV2 adapter hard-requires a `tensor_args` tensor for device sourcing. See the audit-gap
  handoff above.
- **No direct `create_program_artifacts` path for a direct-descriptor op** (secondary, still
  true). The framework's `resolve_program_factory` only wraps `create_descriptor`; a legacy
  direct-descriptor op must be restructured into a `program_factory_t` variant. Not the
  blocker here, but worth a one-line note in `ttnn_factory.md`.

### Confusion
- The op *appears* portable and even builds and mostly runs — 8/9 green — which masks the
  block until the create-output path is exercised. A porter without the `optional_output=False`
  case in the baseline could ship a silently-regressing partial port. Reinforces the recipe's
  insistence on confirming the *complete* test set before relying on it.

## Open items for downstream

- **Cross-op kernel touches:** none — both writer kernels are op-owned; nothing was forked or
  edited outside the op directory.
- **Re-attempt trigger:** once the MetalV2 adapter gains a tensorless device-sourcing path
  (handoff above), this op ports cleanly — the `METAL2_PORT_PLAN.md` spec shape
  (1 TensorParameter, 1 self-looped scratch DFB, 1 writer KernelSpec source-selected by
  `untilize_out`, per-node RTAs) was validated to build and to pass every case *except* the
  device-sourcing one. The implemented port can be restored wholesale at that point.

## Test command(s) — for the orchestrator

No C++ gtest exists. Python coverage (the no-regression baseline), which now passes on the
reverted legacy code:

```bash
pytest tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_arange.py -v
```
