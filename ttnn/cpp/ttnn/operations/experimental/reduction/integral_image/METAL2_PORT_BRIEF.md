# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/experimental/reduction/integral_image`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper` *(carry this line into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `MetalV2FactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` (single `create_descriptor()` returning a `ProgramDescriptor`, `intimg_program_factory.cpp:67`).
- **Op-owned tensors:** none — carried natively by the target concept.
- **Target concept:** `MetalV2FactoryConcept`.
- **Gate-cleared, confirmed absent** (each would have blocked the brief): custom hash · custom `override_runtime_arguments` · pybind `create_descriptor` · smuggled pointer — all `no` on the readiness sheet and confirmed in code.

One DeviceOperation, one factory, one config (interleaved, fixed 2×4 core grid `CORES_X=2 × CORES_Y=4`, `intimg_program_factory.cpp:64-65,92`). Op owns all three kernels.

## Construct — to do

**Tensor bindings** (per binding):

- **input** — **Case 1** (via `TensorAccessor`) → express as `TensorParameter` / `TensorBinding`; reader builds `TensorAccessor(tensor::input)` instead of `TensorAccessor(ctas.input_args, input_base_addr)` (`intimg_reader.cpp:53`). The `{src_buffer}` RTA (`intimg_program_factory.cpp:143-147`) and the `TensorAccessorArgs(src_buffer).append_to(dataflow_compile_time_args)` (`:133`) both disappear.
- **output** — **Case 1** (via `TensorAccessor`) → express as `TensorParameter` / `TensorBinding`; writer builds `TensorAccessor(tensor::output)` instead of `TensorAccessor(ctas.output_args, output_base_addr)` (`intimg_writer.cpp:64`). Note the writer uses this **one** accessor for both writing output (`write_to_dram`) and reading the upper block back for cross-row propagation (`receive_upper_block` → `load_from_dram`, `intimg_writer.cpp:31`) — a single binding covers both. Drop the `{dst_buffer}` RTA (`:167-171`) and `TensorAccessorArgs(dst_buffer).append_to(...)` (`:134`).

Both are delivered today as `Buffer*`-binding RTAs (correct-on-cache-hit, not a stale-pointer hazard) — routine, low-risk.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — both accessors are already 2-arg.

**CB endpoints:**

- **Self-loop** (one toucher — compute both produces and consumes; bind the compute kernel PRODUCER **and** CONSUMER): `ACC` (idx 2), `CUMSUM_STAGE_0` (idx 3), `CUMSUM_STAGE_1` (idx 4), `CUMSUM_STAGE_2` (idx 5), `AXIS_2_BUFFER` (idx 7).
- **Legal 1P+1C, no action:** `START` (idx 0, reader→compute), `INPUT` (idx 1, reader→compute), `OUTPUT` (idx 6, compute→writer), `AXIS_3_BUFFER` (idx 8, writer→compute).

No dead CBs, no multi-binding, no per-config flips (single config).

**Kernel-side metadata move:** `get_dataformat(ctas.input_cb)` (`intimg_reader.cpp:52`, a `constexpr` type-selection) → move onto the DFB object (`dfb::input.get_dataformat()`) per kernel-side whitelist rule 7.

## Watch for

- **CB endpoints (multi-binding):** none. The writer reading output-tensor memory back into `AXIS_3_BUFFER` is a `TensorAccessor` tensor read (Case-1 output binding), not a raw co-fill of a CB — no hidden second writer to hunt.
- **Cross-op / shared kernels:** none — op owns all three kernels; no borrowed kernel files, no out-of-directory includes. No port-together coupling.
- **RTA varargs:** none — each dataflow kernel reads exactly one RTA (the base address at index 0); name it directly.
