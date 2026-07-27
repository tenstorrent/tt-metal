# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/moreh/moreh_dot_backward`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper` *(carry this line into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `MetalV2FactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` (single `create_descriptor` on `MorehDotBackwardOperation`, in `device/moreh_dot_backward_program_factory.cpp`).
- **Op-owned tensors:** none. (The two optional outputs are user-supplied preallocated tensors carried in `tensor_args_t::output_tensors`, not op-owned buffers.)
- **Target concept:** `MetalV2FactoryConcept`.
- **Gate-cleared, confirmed absent** (each would have blocked this brief): custom hash · custom `override_runtime_arguments` · pybind `create_descriptor` · other migration-risky pybind. All `no` on the readiness sheet and confirmed in code.

## Construct — to do

**Tensor bindings** (per binding — all **Case 1**, via `TensorAccessor`):

- `output_grad` — reader accessor `s0` → express as `TensorParameter`/`TensorBinding`; kernel uses `TensorAccessor(tensor::name)`. Drops `Buffer* src0_buffer` RTA + its `TensorAccessorArgs(src0_buffer).append_to(...)` CTA.
- `input` — reader accessor `s1` → same. Drops `src1_buffer` RTA + CTA.
- `other` — reader accessor `s2` → same. Drops `src2_buffer` RTA + CTA.
- `input_grad` (optional output) — writer accessor `s0` → same, **conditionally bound** (see Watch for). Drops `dst0_buffer` RTA + CTA.
- `other_grad` (optional output) — writer accessor `s1` → same, **conditionally bound**. Drops `dst1_buffer` RTA + CTA.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — no accessor passes a page-size override; nothing to drop.

**CB endpoints:** all legal (5 CBs, single core `{0,0}`, each 1P+1C). No self-loop, no multi-binding flag, no dead-CB drop.
- `c_0`: reader → compute · `c_1`: reader → compute · `c_2`: reader → compute · `c_16`: compute → writer · `c_17`: compute → writer.

## Watch for

- **Optional outputs must be conditionally bound.** `input_grad` (`c_16`) and `other_grad` (`c_17`) are `std::optional<Tensor>`. Today the factory pushes `0u` for the address and `TensorAccessorArgs(nullptr)` when absent, and the kernels guard all access on the `has_input_grad` / `has_other_grad` RTAs. Preserve that: emit the output `TensorParameter`/`TensorBinding` only when the tensor is present, and keep the `has_*_grad` runtime guards in the writer and compute kernels. Both outputs may be absent at once (the op is then a no-op); the three inputs stay bound, so dispatch is fine.
- **CB endpoints (multi-binding):** none.
- **Cross-op / shared kernels:** none — all three kernels are op-owned; no shared-kernel port-together coupling.
- **RTA varargs:** none — name every RTA. Reader: `has_input_grad`, `has_other_grad`, then the three tensor bases become bindings, plus `num_tiles`, `start_id`. Writer: `has_input_grad`, `has_other_grad`, two bases become bindings, plus `num_tiles`, `start_id`. Compute: `has_input_grad`, `has_other_grad`, `per_core_block_cnt`. (`start_id` is always `0` — port it faithfully as-is; it is not this brief's job to remove it.)
