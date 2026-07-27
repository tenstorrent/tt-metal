# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/moreh/moreh_matmul`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper` *(carry this line into the port report's Provenance section)*

Single DeviceOperation (`MorehMatmulOperation`), single factory (`MultiCoreProgramFactory`, `device/moreh_matmul_program_factory.cpp`). Three op-owned kernels: `reader_moreh_matmul.cpp`, `writer_moreh_matmul.cpp`, `moreh_matmul.cpp` (compute).

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `MetalV2FactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` (`create_descriptor` returns `ProgramDescriptor`).
- **Op-owned tensors:** none.
- **Target concept:** `MetalV2FactoryConcept` (plain — no op-owned tensors).
- **Gate-cleared, confirmed absent** (each would have blocked the brief): custom hash · custom `override_runtime_arguments` / `get_dynamic_runtime_args` · pybind `create_descriptor` — all `no`, on both the readiness sheet and the code.

## Construct — to do

**Tensor bindings** (per binding) — all four are **Case 1** (base fed straight into a `TensorAccessor`; today delivered via the `Buffer*`-binding RTA form):

- `input` — **Case 1** → express as `TensorParameter` / `TensorBinding`; reader uses `TensorAccessor(tensor::input)`. Drop the `input_buf` RTA push (`program_factory.cpp:481`) and the `TensorAccessorArgs(input.buffer()).append_to(reader_compile_time_args)` CTA plumbing (`:325`).
- `other` — **Case 1** → `TensorAccessor(tensor::other)`. Drop `other_buf` RTA (`:482`) and `TensorAccessorArgs(other.buffer())...` (`:326`).
- `bias` — **Case 1** (FUSE_BIAS path only) → `TensorAccessor(tensor::bias)`. Drop `bias_buf` RTA (`:494`) and `TensorAccessorArgs(bias->buffer())...` (`:331`). Keep the `#ifdef FUSE_BIAS` gating.
- `output` — **Case 1** → `TensorAccessor(tensor::output)`. Drop `output_buf` RTA (`:499`) and `TensorAccessorArgs(output.buffer())...` (`:336`).

Kernel side today already builds `TensorAccessor(args, addr)` (reader `:90-94`, writer `:21`); swap the `(args, addr)` pair for the `tensor::name` binding token. All memory access is already through the accessor + `Noc` / `DataflowBuffer` — no raw pointer arithmetic to preserve.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — every accessor is 2-arg; nothing to drop.

**CB endpoints:**
- **Self-loop** these four compute-internal intermediates (single toucher = the compute kernel, which both produces and consumes them): `c_24` (im0, matmul reload), `c_25` (im1, input transpose), `c_26` (im2, other transpose), `c_27` (im3, bias-add temp). Bind the compute kernel as both PRODUCER and CONSUMER.
- All other CBs are ordinary 1P+1C FIFOs — no action:
  - `c_0`/`c_1`/`c_2`/`c_3`/`c_4`: reader PRODUCER → compute CONSUMER.
  - `c_16` (out0): compute PRODUCER → writer CONSUMER.
- No dead CB, no multi-binding flag.

## Watch for

- **CB endpoints (multi-binding):** none. Reader and writer are distinct kernel sources; the two compute instances (`core_group_1` / `core_group_2`) cover disjoint cores, so no node has two touchers of one role. No hidden co-fill to hunt.
- **Cross-op / shared kernels:** the op owns all three kernels, but they `#include` the shared moreh pool headers `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp` and `.../compute/moreh_common.hpp`. Only `ArgFetcher` and `generate_mask_tiles(DataflowBuffer,...)` are used — both already Device 2.0 (take `DataflowBuffer` / `cb_id`), so no rewrite is needed there. When you self-loop `c_2`/`c_3`, note `generate_mask_tiles` receives the same DFB the compute kernel reads — bind accordingly.
- **RTA varargs:** none — prefer named RTAs. The reader's five 8-element arrays and compute's one 8-element `output_stride` are read via an `ArgFetcher` `arg_idx++` run bounded by the literal `MAX_NUM_DIMENSIONS = 8` (fixed across all instantiations) → name them as fixed fields/arrays, not varargs.
