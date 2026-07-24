# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/experimental/plusone`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper` *(carry this line into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `MetalV2FactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` (`PlusOneProgramFactory::create_descriptor()` → `ProgramDescriptor`)
- **Op-owned tensors:** none
- **Target concept:** `MetalV2FactoryConcept`
- **Gate-cleared, confirmed absent** (each would have blocked the brief): custom hash · custom `override_runtime_arguments` · pybind `create_descriptor` — all `no`; plus no other migration-risky pybind (`plusone_nanobind.cpp` binds only the free function `plus_one`).

## Construct — to do

**Tensor bindings** (one binding — the input tensor; delivered today as a `Buffer*` RTA via `emplace_runtime_args(core, {src_buffer})`, `device/plusone_program_factory.cpp:85`). Classification splits by config:

- **input — interleaved / DRAM config → Case 1** (via `TensorAccessor`): express as `TensorParameter` / `TensorBinding`; the kernel builds `TensorAccessor(tensor::name)` in place of `TensorAccessor(s0_args, src_addr)` (`reader_plusone_interleaved.cpp:25-26`). The `Buffer*` RTA (`get_arg_val<uint32_t>(0)`) and the `TensorAccessorArgs` compile-time plumbing both disappear.
- **input — sharded (L1) config → clean (borrowed-memory DFB):** the CB `c_0` is `.buffer = src_buffer` (`device/plusone_program_factory.cpp:65`); the kernel operates in place on that borrowed memory (accessor path is skipped when `src0_is_dram` is false). Port via `DataflowBufferSpec::borrowed_from` on the same `TensorParameter` — no `TensorAccessor` needed on this path.

**TensorParameter relaxation:** none

**TensorAccessor 3rd arg:** none (the accessor is 2-arg)

**CB endpoints:** self-loop `c_0` — one toucher (the reader raw-peeks `get_write_ptr()`, `reader_plusone_interleaved.cpp:31`). Bind the reader as **both** PRODUCER and CONSUMER. Applies to both the interleaved (plain scratch) and sharded (borrowed-memory) configs.

## Watch for

- **CB endpoints (multi-binding):** none — single kernel, single toucher.
- **Cross-op / shared kernels:** none — the sole kernel is op-owned; port is self-contained.
- **RTA varargs:** none — the only RTA is the single fixed scalar `src_addr`.
- **Preserve the interleaved-in-L1 behavior as-is** (see the audit's Misc anomalies): the DMA in/out is gated on `src0_is_dram`. Do not "fix" the uninitialized-scratch case — zero functional change; that is an ops-team concern, not port work.
