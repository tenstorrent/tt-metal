# Metal 2.0 Port Brief — `moreh/moreh_mean_backward`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper` *(carry this line into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `MetalV2FactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` (single program factory, `MorehMeanBackwardOperation (single-descriptor)`)
- **Op-owned tensors:** none (`descriptor` concept carries none)
- **Target concept:** `MetalV2FactoryConcept`
- **Gate-cleared, confirmed absent** (each would have blocked the brief): custom hash · custom `override_runtime_arguments` · pybind `create_descriptor` · smuggled pointer / migration-risky pybind — all `no`, `Is safe to port? = yes`.

## Construct — to do

**Tensor bindings** (per binding — both **Case 1**, via `TensorAccessor`):

- `output_grad` — **Case 1**. Delivered today as `output_grad.buffer()` (`Buffer*` overload) in the reader RTA (`program_factory.cpp:251`), consumed via `TensorAccessor(output_grad_args, output_grad_addr)` (`reader:92`). Express as `TensorParameter`/`TensorBinding`; the kernel builds `TensorAccessor(tensor::name)`; drop the `Buffer*` RTA and the `TensorAccessorArgs<1>()` CTA (`reader:37`). (The `Buffer*` form is the correct-on-cache-hit interim binding, deliberately chosen for AdamW-style loops — see `program_factory.cpp:245-249`; the typed binding supersedes it.)
- `input_grad` / output — **Case 1**. Same `Buffer*` shape (`program_factory.cpp:267`), consumed via `TensorAccessor(input_grad_args, input_grad_addr)` (`writer:21`). Same treatment; drop `TensorAccessorArgs<0>()` (`writer:11`).

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none (both accessor sites are 2-arg).

**CB endpoints:**

- Self-loop `c_24` (intermed) — one toucher (compute both fills and drains it, `compute:52/58/63/75`): bind the compute kernel PRODUCER **and** CONSUMER.
- All legal 1P+1C, no action: `c_0` (reader→compute), `c_1` (reader→compute), `c_2` (reader→compute scalar operand), `c_16` (compute→writer).

## Watch for

- **CB endpoints (multi-binding):** none — no hidden second writer, no multi-reader; the two compute instances cover disjoint core groups (one per node), not the same nodes.
- **Cross-op / shared kernels:** the reader/writer/compute pull helpers from `ttnn/cpp/ttnn/kernel/{dataflow,compute}/moreh_common.hpp` (shared moreh pool, already `DataflowBuffer`-typed). If the port rewrites those helper signatures to named tokens, that is a moreh-family-wide change — port the shared helper as one unit across its co-borrowers, don't fork it for this op alone.
- **RTA varargs (reader):** the three per-dimension blocks `output_grad_dim` / `input_grad_dim` / `need_bcast_dim` (`reader:48-60`) are counted by `input_grad_rank` (a CTA) and vary across instantiations — port them via the kernel-side vararg mechanism (prefer named RTAs elsewhere, but these blocks cannot be named). The four leading reader scalars and all three writer args are ordinary named RTAs.
