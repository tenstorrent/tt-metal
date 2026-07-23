# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/data_movement/fill_rm`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `7ca84865be5 2026-07-23 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper` *(carry this line into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `MetalV2FactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — `FillRMProgramFactory::create_descriptor()` returns `ProgramDescriptor` (single factory, single kernel).
- **Op-owned tensors:** none — carried natively by the target concept if there were any; there are none.
- **Target concept:** `MetalV2FactoryConcept`.
- **Gate-cleared, confirmed absent** (each would have blocked the brief): custom hash · custom `override_runtime_arguments` · pybind `create_descriptor` — all `no` on this op — plus other migration-risky pybind (none; `Is safe to port? = yes`, no `warning`).

## Construct — to do

**Tensor bindings** (per binding):

- **output** — **Case 1** (via `TensorAccessor`) → express as `TensorParameter` / `TensorBinding`. Today it arrives as a `Buffer* dst_buffer` in RTA slot 0 (`device/fill_rm_program_factory.cpp:82-91`) and the kernel feeds it into `TensorAccessor(dst_args, dst_addr, …)` (`device/kernels/dataflow/fill_rm_interleaved.cpp:31`). In the port the kernel builds `TensorAccessor(tensor::name)`; the `Buffer*` RTA (`dst_addr = get_arg_val<uint32_t>(0)`, kernel `:19`) **and** the `TensorAccessorArgs(*dst_buffer)` CTA block (`device/fill_rm_program_factory.cpp:69-70`, kernel `:28`) both disappear.
- **input** (`tensor_args.input`, the "any" metadata tensor) — **not a binding.** Used host-side only for `dtype()` / `device()` (`device/fill_rm_device_operation.cpp:33-34,57`); the kernel never touches it. No `TensorParameter`, no work item.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** drop the redundant page-size arg `W << 1` @ `device/kernels/dataflow/fill_rm_interleaved.cpp:31` (Class 2 — interleaved + correct magnitude; Metal 2.0 supplies `aligned_page_size` implicitly). **Do not** set `dynamic_tensor_shape` — this is not a Class-1 dynamic page (`W` is hashed, so it never goes stale on a cache hit; the kernel's `:29-30` "stale on cache hit" comment does not apply to this op). Delete that comment along with the arg.

**CB endpoints:** self-loop both CBs — bind the single reader kernel as **both** PRODUCER and CONSUMER on `CB buffer_index 0` (`dfb_in0`) and `CB buffer_index 1` (`dfb_in1`). Each is single-toucher (one kernel FIFO-produces, raw-writes via `get_write_ptr()`, and reads as the `noc.async_write` source). Single config — the factory is 1-core (`CoreRange{{0,0},{0,0}}`) and interleaved-only, so the disposition does not flip.

## Watch for

- **CB endpoints (multi-binding):** none — both CBs are single-toucher self-loops; no hidden 2nd writer, no multi-reader.
- **Cross-op / shared kernels:** none — op owns its only kernel; all includes are `tt_metal/hw/inc/api/*` (LLK/HAL). No port-together coupling.
- **RTA varargs:** none — all 8 RTAs are fixed constexpr-index reads (`fill_rm_interleaved.cpp:19-26`); name each (`NC`, `H`, `W`, `fillH`, `fillW`, `val_hi`, `val_lo`; slot 0 becomes the output tensor-binding base). Prefer named RTAs; do not reach for the vararg mechanism.
