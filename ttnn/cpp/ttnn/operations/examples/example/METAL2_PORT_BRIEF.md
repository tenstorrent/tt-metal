# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/examples/example`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper` *(carry this line into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `MetalV2FactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — both factories (`SingleCore`, `MultiCore`) define `create_descriptor()` → `ProgramDescriptor`.
- **Op-owned tensors:** none.
- **Target concept:** `MetalV2FactoryConcept`.
- **Gate-cleared, confirmed absent** (each would have blocked the brief): custom hash · custom `override_runtime_arguments` · pybind `create_descriptor` · smuggled pointer · other migration-risky pybind. All `no` on this op.

The two factories are structurally identical (same two CBs, same three kernels, same RTA loop) and differ only in core-grid selection — `SingleCore` fixes a `{1,1}` grid, `MultiCore` uses `device->compute_with_storage_grid_size()`. Port them the same way.

## Construct — to do

**Tensor bindings** (per binding):

- **`input_tensor`** — **Case 1** (via `TensorAccessor`) → express as a `TensorParameter` / `TensorBinding`. In the reader kernel, replace `src_addr = get_arg_val<uint32_t>(0)` + `TensorAccessor(src_args, src_addr)` with `TensorAccessor(tensor::name)`. The `src_buffer` RTA element and the reader's `TensorAccessorArgs<0>` CTA plumbing (built in the factory via `TensorAccessorArgs(*src_buffer).append_to(...)`) both disappear.
- **`output_tensor`** — **Case 1** (via `TensorAccessor`) → same treatment. Writer: replace `dst_addr = get_arg_val<uint32_t>(0)` + `TensorAccessor(dst_args, dst_addr)` with `TensorAccessor(tensor::name)`; drop the `dst_buffer` RTA element and the writer's `TensorAccessorArgs<1>` CTA (currently `writer_compile_time_args = {output_cb_index}` followed by `TensorAccessorArgs(*dst_buffer).append_to(...)`).

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — no accessor site passes a 3rd argument.

**CB endpoints:** all legal. `c_0` (input) is reader→compute 1P+1C; `c_2` (output) is compute→writer 1P+1C, on every node in both configs. No self-loop, no multi-binding flag, no dead-CB drop.

**Remaining RTAs** (routine named args, not this audit's concern but noted for wiring): after the address bindings move to `TensorParameter`s, the reader/writer keep `num_pages` and `start_id` (currently `num_tiles_per_core`, `num_tiles_written`) as named RTAs; the compute kernel keeps `num_tiles`.

## Watch for

- **CB endpoints (multi-binding):** none.
- **Cross-op / shared kernels:** all three kernels are borrowed by file path from `eltwise/unary` — `reader_unary_interleaved_start_id.cpp`, `writer_unary_interleaved_start_id.cpp`, `eltwise_sfpu.cpp`. They are shared, library-grade kernels. Their CB→DFB / named-token rewrite is a **single change adopted by every borrowing op together** — do **not** fork local copies into this op's directory, and coordinate with the eltwise/unary shared-kernel port. (The op's own `device/kernels/` tree holds *unreferenced* stale copies; ignore them — the factories point at the `eltwise/unary` paths.)
- **RTA varargs:** none — every RTA is a nameable fixed field.
