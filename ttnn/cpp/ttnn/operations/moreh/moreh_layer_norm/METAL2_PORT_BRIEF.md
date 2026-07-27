# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper` *(carry this line into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `MetalV2FactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` (one `create_descriptor` returning a `ProgramDescriptor`; small/large algorithm branches inside it).
- **Op-owned tensors:** none — carried by neither; the target concept needs no op-owned-tensor support.
- **Target concept:** `MetalV2FactoryConcept`.
- **Gate-cleared, confirmed absent** (each would have blocked the brief): custom hash · custom `override_runtime_arguments` · pybind `create_descriptor` — all `no` — plus no migration-risky pybind (`Is safe to port? == yes`, no smuggled pointer).
- **Dispatch always has a tensor:** `tensor_args_t.input` is a mandatory `const Tensor&` — the MetalV2 adapter can always source the MeshDevice; no tensorless-dispatch concern.

## Construct — to do

**Tensor bindings** (per binding) — all **Case 1** (via `TensorAccessor`). Today each address is delivered as a `Buffer*` in the RTA list (`emplace_runtime_args`, `program_factory.cpp:381-395`) and consumed by `TensorAccessor(args, addr)`. Express each as a `TensorParameter` / `TensorBinding`; the kernel builds `TensorAccessor(tensor::name)`, and the `Buffer*` RTA + its `TensorAccessorArgs(...).append_to(...)` compile-time plumbing (`program_factory.cpp:218-228`) both disappear:

- `input` (c_0) — reader (`reader_moreh_layer_norm_{small,large}.cpp:39`).
- `gamma` (c_3) — reader, present-only under `GAMMA_HAS_VALUE` (`:43`).
- `beta` (c_4) — reader, present-only under `BETA_HAS_VALUE` (`:48`).
- `output` (c_16) — writer (`writer_moreh_layer_norm.cpp:121`).
- `mean` (c_17) — writer, present-only (`:124`).
- `rstd` (c_18) — writer, present-only (`:127`).

Compute kernels touch no tensor memory (CB-only) — no bindings there.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — every accessor is the 2-arg form; nothing to drop.

**CB endpoints:**
- **Self-loop** (single toucher = the compute kernel, produces and consumes): `c_24` E[x] (also reused as `cb_tmp`), `c_25` x-E[x], `c_26` (x-E[x])², `c_27` Sum[(x-E[x])²], `c_28` Var[x], `c_29` 1/sqrt(Var+eps), `c_30` gamma·+beta, `c_31` Sum[x].
- **1P+1C** (plain legal FIFO, one producer + one consumer): `c_0` (reader→compute), `c_1`/`c_2` (reader-fill→compute), `c_3`/`c_4` (reader→compute), `c_5`/`c_6` (reader→compute), `c_16` (compute→writer), `c_17`/`c_18` (compute→writer).
- No multi-binding flag anywhere; no dead CBs to drop.
- Config note: gamma/beta/mask_h/mask_w/mean/rstd CBs exist only when the corresponding option is present (`push_cb` skips zero-size, `program_factory.cpp:180-183`); `c_0`/`c_25`/`c_26` change tile counts between small/large algorithm but endpoints do not change.

## Watch for

- **CB endpoints (multi-binding):** none — no hidden second writers, no multi-reader CBs. The one raw `get_read_ptr()` (`writer:34`) is the writer peeking its own consumer binding on `c_output`, not a separate endpoint.
- **Cross-op / shared kernels:** three shared header pools are `#include`d — `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp`, `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp`, and `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp`. Their CB→DFB rewrite is shared across every moreh op that includes them — port the shared header change as one unit, not per-op. No kernel `.cpp` is borrowed by file path (the op owns all five kernel sources).
- **RTA varargs:** none — read each RTA as a named scalar (fixed run of `get_arg_val<uint32_t>(i++)` in the readers, fixed indices in the writer).
- **Dead code (optional cleanup, not required for the port):** `input_data_format` (`reader_{small,large}:32`) and `offs` in the small reader (`:71,126`) are unused; the metadata-lookup relocation (whitelist rule 7) will touch the former.
