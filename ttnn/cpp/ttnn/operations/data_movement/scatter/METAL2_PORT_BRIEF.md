# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/data_movement/scatter`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `44da718b06b 2026-07-23 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper` *(carry this line into the port report's Provenance section)*

**Port unit:** one device operation, two factories — `ScatterDeviceOperation` → `ScatterProgramFactory` and `ScatterReduceBfloat16ProgramFactory`. They share the device operation, the kernel common headers, and the addressing model, so port them together. The reduce factory differs only by adding a fifth CB (`FP32_TEMP`) and doing bf16↔fp32 conversion; the binding/endpoint structure is otherwise identical. The kernels are already on Device 2.0 object wrappers, so the bulk of the work is host-side spec/binding wiring plus replacing numeric CB/DFB indices and RTA-delivered base addresses with named `dfb::*` / `tensor::*` bindings.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); both factories port to `MetalV2FactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` (both factories)
- **Op-owned tensors:** none
- **Target concept:** `MetalV2FactoryConcept`
- **Gate-cleared, confirmed absent** (each would have blocked the brief): custom hash · custom `override_runtime_arguments` · pybind `create_descriptor` · migration-risky pybind. All `no`.

## Construct — to do

**Tensor bindings** (per binding) — all **Case 1** (via `TensorAccessor`), both factories:

- `input` — **Case 1** → express as `TensorParameter` / `TensorBinding`; kernel builds `TensorAccessor(tensor::input)` in place of `TensorAccessor(ctas.input_args, input_buffer_address)` (`reader_scatter.cpp:116`, `reader_bf16_reduction_scatter.cpp:131`). The `Buffer*` reader RTA (`scatter_program_factory.cpp:163`) and the appended `TensorAccessorArgs(*input_buffer)` (`:95`) both disappear.
- `index` — **Case 1** → `TensorAccessor(tensor::index)` (`reader_scatter.cpp:117`). Drop the `Buffer*` RTA (`:164`) and `TensorAccessorArgs` (`:96`).
- `src` — **Case 1** → `TensorAccessor(tensor::src)` (`reader_scatter.cpp:118`). Drop the `Buffer*` RTA (`:165`) and `TensorAccessorArgs` (`:97`).
- `output` — **Case 1** (writer) → `TensorAccessor(tensor::output)` (`writer_scatter.cpp:17`, `writer_bf16_reduction_scatter.cpp:17`). Drop the `Buffer*` writer RTA (`:181`) and `TensorAccessorArgs` (`:98`).

While binding the addresses, **remove the four dead buffer-address compile-time args** (`scatter_program_factory.cpp:78-81`; reduce `:80-83`) and their unused `ctas.*_tensor_addr` fields (`scatter_common.hpp:18-21`, `scatter_bf16_reduction_common.hpp:18-21`). They are never read by any kernel; keeping a `buffer->address()` in a compile-time arg is a stale-address trap on cache hits (see the audit's Misc anomalies). This is a clean removal, not a behavior change.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — every accessor is already 2-arg.

**CB endpoints:**

- **`ScatterProgramFactory`:** self-loop `INPUT` (`c_0`), `INDEX` (`c_2`), `SRC` (`c_1`) — each is touched by the reader alone (it both fills and drains them); bind the reader as PRODUCER **and** CONSUMER. Bind `DST` (`c_3`) as a normal 1:1 — reader PRODUCER, writer CONSUMER (`reader_scatter.cpp:154,203` produce; `writer_scatter.cpp` via `common.hpp` `write_to_output` consumes).
- **`ScatterReduceBfloat16ProgramFactory`:** self-loop `INPUT`, `INDEX`, `SRC`, and `FP32_TEMP` (`c_4`, the fp32 scratch — reduce reader alone touches it). Bind `DST` 1:1 (reader PRODUCER, writer CONSUMER).
- Single interleaved config (the op rejects sharded inputs), so no per-config disposition flips.

## Watch for

- **CB endpoints (multi-binding):** none — no hidden second writer, no multi-reader. Every CB is a self-loop or a plain 1:1; do not reach for the multi-binding advanced option.
- **Cross-op / shared kernels:** none — the op owns all four kernels and their common headers; no shared-kernel port-together coupling.
- **RTA varargs:** the reader kernels read two runtime-arg vararg blocks — input-tensor and index-tensor per-dimension shape extents — via `make_shape_array_from_runtime_args<N>` with `N = rank-1` (`reader_scatter.cpp:125-126`, `reader_bf16_reduction_scatter.cpp:140-141`; helper `common.hpp:111-119`; host emit `scatter_program_factory.cpp:172-177`, reduce `:177-182`). `N` varies with tensor rank across instantiations, so these have no stable per-element names — port them with the kernel-side **vararg** mechanism (whitelist rule 4), not by naming each. The nine leading scalar reader args (offsets 0–8) precede the vararg blocks and stay named.
