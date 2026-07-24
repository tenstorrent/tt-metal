# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/experimental/reduction/fast_reduce_nc`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper` *(carry this line into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `MetalV2FactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — `FastReduceNCProgramFactory::create_descriptor(...)` returns a `ProgramDescriptor`.
- **Op-owned tensors:** none.
- **Target concept:** `MetalV2FactoryConcept`.
- **Gate-cleared, confirmed absent** (each would have blocked the brief): custom hash · custom `override_runtime_arguments` · pybind `create_descriptor` · other migration-risky pybind. All `no` on this op (readiness sheet + code cross-check).

Single DeviceOperation (`FastReduceNCDeviceOperation`), single factory (`FastReduceNCProgramFactory`), three op-owned kernels (`reader_reduce_nc.cpp`, `writer_reduce_nc.cpp`, `reduce_nc.cpp`).

## Construct — to do

**Tensor bindings** (per binding):

- `input` — **Case 1** (via `TensorAccessor`) → express as `TensorParameter` / `TensorBinding`. The kernel builds `TensorAccessor(tensor::name)` instead of `TensorAccessor(tensor_args, input_addr)`. Delete the `Buffer*` RTA (reader arg position 0, `fast_reduce_nc_program_factory.cpp:327`) and the `TensorAccessorArgs(*tensor_args.input.buffer()).append_to(reader_compile_time_args)` plumbing (`:201`); the reader's `TensorAccessorArgs<3>()` CTA base (`reader_reduce_nc.cpp:21`) is replaced by the bound-tensor accessor.
- `output` — **Case 1** (via `TensorAccessor`) → express as `TensorParameter` / `TensorBinding`. The kernel builds `TensorAccessor(tensor::name)` instead of `TensorAccessor(tensor_args, output_addr)`. Delete the `Buffer*` RTA (writer arg position 0, `:337`) and `TensorAccessorArgs(*tensor_return_value.buffer()).append_to(writer_compile_time_args)` (`:204`); the writer's `TensorAccessorArgs<2>()` CTA base (`writer_reduce_nc.cpp:15`) is replaced.

(The compute kernel `reduce_nc.cpp` touches no tensor memory — CB-only — so no binding work there.)

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — both accessors are already 2-arg; nothing to drop.

**CB endpoints:**
- `c_0` (in0), `c_1` (in1/zero), `c_16` (out0) — all plain 1:1 (one producer + one consumer). Bind producer/consumer as-is; no self-loop, no flag.
- **Drop dead CB `c_24`** — `fast_reduce_nc_program_factory.cpp:177-185`. No kernel references index 24 (compute accumulates in DST registers, not through an intermediate CB); it carries no CTA. A bindingless DFB can't exist in Metal 2.0, so remove the `CBDescriptor`. Zero behavioral change (L1 footprint only).

## Watch for

- **CB endpoints (multi-binding):** none — no hidden second writer (no raw `get_write_ptr` co-fills; all fills are FIFO `reserve_back`/`push_back`), no multi-reader CB.
- **Cross-op / shared kernels:** the reader `#include`s `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp` and calls `prepare_zero_tile<c_1>()` — the donor takes the CB id as a `uint32_t` NTTP, which `dfb::name`'s constexpr cast satisfies. Pass the DFB token; no donor-side change needed. `l1_helpers.hpp` is a broadly shared `kernel_lib` header — do not modify it as part of this port.
- **RTA varargs:** none — name every RTA (reader: `num_input_tiles`, `id_range_length`, `start_id`, `dim`, `reduce_tile_size`, `inner_tile_size`; writer: `id_range_length`, `start_id`). All are fixed distinct fields read at constexpr indices; prefer named args.
