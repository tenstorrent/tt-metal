# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/moreh/moreh_sgd`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ (code cross-check; confirm the readiness-sheet row — see the audit's Questions) · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper` *(carry this line into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `MetalV2FactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` (single `create_descriptor` returning a `ProgramDescriptor`, `device/moreh_sgd_program_factory.cpp:25`).
- **Op-owned tensors:** none — outputs (`param_out`, `momentum_buffer_out`) are ordinary returned device tensors bound as output `TensorParameter`s.
- **Target concept:** `MetalV2FactoryConcept`.
- **Gate-cleared, confirmed absent** (each would have blocked the brief): custom hash · custom `override_runtime_arguments` · pybind `create_descriptor` · other migration-risky pybind. All `no`.

## Construct — to do

**Tensor bindings** (per binding — all Case 1, via `TensorAccessor`):

- `param_in` (CB `c_0`) — express as `TensorParameter` / `TensorBinding`; kernel builds `TensorAccessor(tensor::param_in)`. Drops the `param_in.buffer()` RTA (`moreh_sgd_program_factory.cpp:268`) and the reader's `TensorAccessorArgs<0>` plumbing.
- `grad` (CB `c_1`) — same; drops the `grad.buffer()` RTA and `grad_args`.
- `momentum_buffer_in` (CB `c_2`, **optional**) — bind only under the `MOMENTUM && MOMENTUM_INITIALIZED` path (kernel gates the accessor with those defines, `reader_moreh_sgd.cpp:47-50`). Delivered today as `momentum_in_buf` (`Buffer*` or `nullptr`).
- `param_out` (CB `c_16`) — output binding; drops `param_out.buffer()` RTA and the writer's `TensorAccessorArgs<0>`.
- `momentum_buffer_out` (CB `c_17`, **optional**) — output binding gated by the `MOMENTUM` define (`writer_moreh_sgd.cpp:32-35`); delivered today as `momentum_out_buf`.

All five ride the sanctioned `Buffer*`-binding form today (`moreh_sgd_program_factory.cpp:252-279`) — correct-on-cache-hit, not a stale-pointer hazard; the port replaces them with typed bindings. The scalar RTAs (`num_tiles`, `tile_offset`, `lr`, `momentum`, `dampening`, `weight_decay`, `one`) become named runtime args.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — no accessor passes a page-size argument.

**CB endpoints:**

- Legal 1:1 (no action): `c_0` param_in, `c_1` grad, `c_2` momentum_in (reader→compute); `c_16` param_out, `c_17` momentum_out (compute→writer); `c_24` scalar_args (reader→compute).
- **Self-loop** (single toucher = compute; bind compute PRODUCER **and** CONSUMER): `c_25` tmp1, `c_26` tmp2, `c_27` tmp3, `c_28` tmp4.
- **Per-config nuance:** the four intermediates are touched only under specific compile-define configs — under the minimal config (`weight_decay==0 && momentum==0`) only `c_27` is exercised; `c_25`/`c_26`/`c_28` see zero touches. The compute kernel currently constructs all four `DataflowBuffer` wrappers unconditionally (`moreh_sgd.cpp:23-30`). Classify per `(CB, config)` and self-loop where live; keep the binding set consistent with the compute kernel's actual per-config touches.

## Watch for

- **CB endpoints (multi-binding):** none — no hidden 2nd writer, no multi-reader.
- **Cross-op / shared kernels:** the kernels call the shared `ttnn/cpp/ttnn/kernel/{dataflow,compute}/moreh_common.hpp` helpers (already `DataflowBuffer`-native). No file-path kernel borrows. A Metal 2.0 rewrite of any moreh_common helper is a shared rewrite — port it as one unit across every moreh op that uses it; do not diverge the helper for moreh_sgd alone.
- **RTA varargs:** none — name each RTA (fixed run read via a top-of-kernel `i++` counter, not a vararg loop).
- **Optional plumbing:** keep the `MOMENTUM` / `MOMENTUM_INITIALIZED` / `WEIGHT_DECAY` / `NESTEROV` / `FP32_DEST_ACC_EN` define conditionality intact across host defines, tensor bindings, and CB usage.
