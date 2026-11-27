# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/experimental/quasar/rotate`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.
>
> **Port together with `experimental/quasar/grid_sample`** (user instruction). Nothing under the main tree (`pool/`) may change.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorParameter Case 2 ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `9df4aa1e011 2026-11-27 docs(metal_2.0): fix-ups from an adversarial review of the trial-port fixes` *(carry this line into the port report's Provenance section)*

## TTNN factory analysis

- **Current concept:** `descriptor` (both factories)
- **Op-owned tensors:** none
- **Target concept:** `ProgramSpecFactoryConcept` (both factories)
- **Gate-cleared, confirmed absent:** a non-clearing relaxation · `get_dynamic_runtime_args`.

## Construct — to do

**Tensor bindings:**
- `input` — **Case 1** (nearest reader, bilinear reader; RTA 0) → `TensorAccessor(tensor::input)`.
- `output` — **Case 1** (nearest writer every config; bilinear writer when output interleaved). When sharded it also backs the output DFB (`borrowed_from`).

**TensorParameter relaxation:** `none`

**TensorAccessor 3rd arg:** none

**CB endpoints:**
- self-loop: nearest `fill` (reader), bilinear `fill` (reader), bilinear `out` when sharded (compute).
- legal 1:1: nearest `out` (reader → writer), bilinear `in` / `scalar` (reader → compute), bilinear `out` interleaved (compute → writer).
- **drop dead CB:** nearest sharded input CB `rotate_nearest_program_factory.cpp:169-182` (no kernel references its index, in any config; no CTA carries it).

## Watch for

- **Cross-op / shared kernels:** bilinear binds grid_sample's `writer_grid_sample_interleaved.cpp` (converted by the joint port; bind its Metal 2.0 names) and the grid_sample-local `compute_pool_2d.cpp` copy (replaces the main-tree path at `rotate_bilinear_program_factory.cpp:294`). Its reader includes `grid_sample_reader_common.hpp`.
- **Indexed-collection args:** none.
- **Two compute KernelSpecs** (interleaved core groups) — keep both; CTA 3 differs.
- **Name args that are read but unused** (nearest reader `input_batch`, `num_cb_pages`; nearest writer `num_cb_pages`).
- **Compute config Style B:** HiFi4, fp32 off, approx off, `double_buffer_dest` true (dst_full_sync unset), O3.
