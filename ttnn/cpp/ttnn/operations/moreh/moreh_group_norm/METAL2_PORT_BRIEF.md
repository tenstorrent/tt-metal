# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/moreh/moreh_group_norm`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper` *(carry this line into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `MetalV2FactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` (single `create_descriptor` returning a `ProgramDescriptor`; one internal `use_large_algorithm` branch selecting small vs. large kernels — same op, same concept)
- **Op-owned tensors:** none (carried natively by the target concept if any — none here)
- **Target concept:** `MetalV2FactoryConcept`
- **Gate-cleared, confirmed absent** (each would have blocked the brief): custom hash · custom `override_runtime_arguments` · pybind `create_descriptor` · smuggled pointer / migration-risky pybind. All `no` on this op.

## Construct — to do

**Tensor bindings** (per binding) — all **Case 1** (via `TensorAccessor`). Today the factory delivers each as a `Buffer*` runtime-arg binding (`program_factory.cpp:338-356`) and the kernel feeds the received base into a `TensorAccessor`. Replace each with a typed `TensorParameter` / `TensorBinding`; the kernel builds `TensorAccessor(tensor::name)`, and both the `Buffer*` RTA and the `TensorAccessorArgs(...).append_to(...)` CTA plumbing (`program_factory.cpp:217-219, 231-233`) disappear:

- `input` — **Case 1** → `TensorParameter`; kernel uses `TensorAccessor(tensor::input)`.
- `gamma` — **Case 1** (optional) → `TensorParameter`; `TensorAccessor(tensor::gamma)`.
- `beta` — **Case 1** (optional) → `TensorParameter`; `TensorAccessor(tensor::beta)`.
- `output` — **Case 1** → `TensorParameter`; `TensorAccessor(tensor::output)`.
- `mean` — **Case 1** (optional) → `TensorParameter`; `TensorAccessor(tensor::mean)`.
- `rstd` — **Case 1** (optional) → `TensorParameter`; `TensorAccessor(tensor::rstd)`.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — every accessor is already 2-arg.

**CB endpoints:**
- **Self-loop** the compute-only intermediates (single toucher each): `c_24` E[x] · `c_25` x−E[x] · `c_26` (x−E[x])² · `c_27` Sum[(x−E[x])²] · `c_28` Var[x] · `c_29` 1/√(Var+eps) · `c_30` gamma_beta · `c_31` Sum[x] — bind the compute kernel both PRODUCER and CONSUMER.
- **Legal 1:1** (no action): `c_0` input, `c_1` scaler, `c_2` eps, `c_3` gamma, `c_4` beta, `c_5` mask_h, `c_6` mask_w, `c_16` output, `c_17` mean, `c_18` rstd.
- Only push a DFB when its tile count > 0 (mirror `push_cb_if_nonzero`): `c_3`/`c_4` gated on gamma/beta, `c_5`/`c_6` on mask, `c_30` on gamma||beta, `c_17`/`c_18` on required mean/rstd. Disposition per present CB is unchanged across configs.

## Watch for

- **CB endpoints (multi-binding):** none. Reader's gamma/beta `get_write_ptr()` writes and writer's output/mean/rstd `get_read_ptr()` reads are peeks on the kernel's own binding — do not count them as extra touchers or reach for the multi-binding flag.
- **Cross-op / shared kernels:** the compute kernels `moreh_layer_norm_{small,large}_kernel.cpp` are **shared with `moreh_layer_norm`** (in-family). Their CB→DFB / named-token rewrite is a single change both ops must adopt together — **port `{moreh_group_norm, moreh_layer_norm}` as one unit** for these files, or the co-borrower breaks. Shared headers `moreh_common.hpp` and `reduce_helpers_compute.hpp` are shared-pool / lib-team surfaces.
- **RTA varargs:** none — name every runtime arg (reader and writer each read a fixed sequential field set via a top-of-kernel `i++` counter).
