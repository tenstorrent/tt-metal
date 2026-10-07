# Metal 2.0 Port Brief — `examples/example_multiple_return`

> Audit cleared all gates, one of them by user waiver (see below). This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ (user waiver) · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `4bd4bf42bfe 2026-09-03 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state` *(carry this line into the port report's Provenance section)*

**Waiver.** The TTNN factory concept gate was RED because the readiness sheet is broken for this op. Its row predates PD batch #57409 (`f5093e705ae`): it lists `Concept` = `legacy device-op` and a factory `SingleCore`, but #57409 deleted `SingleCore` and moved the op to a direct `create_descriptor` on the device-op. That was the only RED. The sheet's own cell reads `yes (with PD step)`, and the primary cross-check is otherwise clean (no custom hash, no `get_dynamic_runtime_args`, no `override_runtime_arguments`, no pybound descriptor, no smuggled pointer, relaxation `none`). The user waived the gate on 2026-10-07 ("Okay waive it if it's just the stale row"). Record the waiver in the port report's Provenance section. The sheet row still needs refreshing by its owner.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `ProgramSpecFactoryConcept`. Carry them forward:

- **Current concept:** `descriptor`, in the **direct-descriptor shape**: `create_descriptor` is a static member of `ExampleMultipleReturnDeviceOperation` (`device/example_multiple_return_device_operation.hpp:72`) with no `program_factory_t`. Its body is in `device/single_core_program_factory.cpp:12`.
- **Op-owned tensors:** none.
- **Target concept:** `ProgramSpecFactoryConcept`.
- **Forced device-op edit (`ttnn_factory.md` exception 3):** add `struct ExampleMultipleReturnProgramFactory { static ProgramArtifacts create_program_artifacts(...); };` and `using program_factory_t = std::variant<ExampleMultipleReturnProgramFactory>;`, replacing the direct `create_descriptor`. The `DirectDescriptorFactory` shim has no spec counterpart. Record it under Handoff points.
- **Gate-cleared, confirmed absent:** a `TensorParameter relaxation` that is neither `none` nor an analysis pointer · `get_dynamic_runtime_args`. (Also absent here: custom hash, `override_runtime_arguments`, pybound `create_descriptor`.)

## Construct — to do

**Tensor bindings** (per binding; all are `Buffer*` RTAs today, `single_core_program_factory.cpp:117-118`):

- `src` (input): **Case 1** → bind as `TensorParameter` / `TensorBinding`. The reader fork already uses `TensorAccessor(tensor::src)`.
- `dst1` (output 0, **optional**): **Case 1** → `TensorAccessor(tensor::dst1)` in the writer.
- `dst2` (output 1, **optional**): **Case 1** → `TensorAccessor(tensor::dst2)` in the writer.
- **Optional outputs → conditional binding** (`port_patterns.md`, *Conditional / optional resource bindings*). Today the writer uses the raw address as a presence flag: `if (dst_addr1 != 0)` / `if (dst_addr2 != 0)` (`device/kernels/writer_multiple.cpp:34,39`). The host passes a null `Buffer*` and `TensorAccessorArgs(nullptr)` for an absent output (`single_core_program_factory.cpp:82-83,118`). After the port:
  - Bind `dst1` / `dst2` only when the output is present.
  - Emit a matching define per output on the writer's `compiler_options.defines`.
  - `#ifdef`-gate each accessor and its write block in the kernel.
  - Presence comes from `return_output1` / `return_output2` in `operation_attributes_t`, which the default hash covers, so a compile-time gate preserves behaviour.
  - Drop the `dst_addr1` / `dst_addr2` RTAs and both `TensorAccessorArgs` appends.

**TensorParameter relaxation:** `none`.

**TensorAccessor 3rd arg:** none. No accessor passes one.

**CB endpoints:** all legal. Both CBs are plain 1P+1C on the single core:
- `c_0`: reader produces, compute consumes.
- `c_2`: compute produces, writer consumes.

## Watch for

- **CB endpoints (multi-binding):** none.
- **Cross-op / shared kernels:**
  - **Reader:** `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`. A `_metal2` fork **already exists**: `reader_unary_interleaved_start_id_metal2.cpp`. Bind it and adopt its names (`dfb::in`, `tensor::src`, `args::num_pages`, `args::start_id`). Do not set `BACKWARDS`, and don't re-fork. Other legacy binders (**sunset list, not authorization to convert the kernel in place**): `examples/example` (single- and multi-core), `experimental/transformer/nlp_create_qkv_heads_falcon7b`, `reduction/topk` (`topk_route_prep_program_factory.cpp`).
  - **Compute:** `eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp`. **No fork yet**, so this port creates `eltwise_sfpu_metal2.cpp` beside the original and adds the pointer comment to the original (rung 2).
    - Name its bindings from the kernel's vocabulary (e.g. `dfb::in` / `dfb::out`), not this op's locals.
    - Keep `#ifdef SFPU_OP_CHAIN_0`, which this op leaves undefined.
    - `num_tiles` becomes a named RTA.
    - Other binders (**sunset list, not authorization**): `eltwise/unary` (`unary_program_factory.cpp:414` via `unary_op_utils.cpp:1212`, its default compute kernel), `examples/example` (single- and multi-core), and `tests/ttnn/unit_tests/gtests/test_generic_op.cpp:284`.
- **RTA varargs:** none. Name every RTA:
  - writer: `num_tiles`, `start_id` (the address RTAs go away)
  - reader: per the fork
  - compute: `num_tiles`
- **Writer is op-owned:** `device/kernels/writer_multiple.cpp` is bound only by this op. Convert it in place, no fork. It is already Device 2.0 (`Noc`, `CircularBuffer`), so the port is a binding-layer change: `CircularBuffer cb_out(cb_id_out)` (CTA 0) becomes a `dfb::` token, and the accessors move to `tensor::` tokens.
- **Compute `opt_level`:** unset on the compute descriptor (`single_core_program_factory.cpp:94-101`), so legacy resolves it to `O3`. Set it explicitly on the spec. HiFi4 and `math_approx_mode = false` carry over.
- **Test:** `tests/ttnn/unit_tests/operations/debug/test_examples.py` (`test_composite_example_multiple_return`, `..._program_cache`). It covers all three presence combinations, runs on one core, and fits the 8×8 Wormhole.
