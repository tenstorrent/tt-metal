# Port Report — SamplingProgramFactory (Metal 2.0)

**Status: Draft port — structural changes complete, needs build iteration.** The factory now satisfies `ProgramSpecFactoryConcept` (returns `ProgramArtifacts`). All three kernel sources under `device/kernels/` were converted: positional CTAs → named (`get_arg(args::name)`), CB indices → `dfb::name`, `CircularBuffer` → `DataflowBuffer`, `TensorAccessorArgs<N>()` → `TensorAccessor(ta::name)`. The port has NOT been built or tested in this session.

## Handoff points

None — no out-of-op call sites need `sem::` or `ta::` handles. The op uses no semaphores, and `ta::` is consumed entirely within its own dataflow kernels.

## Successes

### Clean preserved-multiplicity collapse via named RTA
The legacy factory created N separate writer + N separate compute `KernelDescriptor`s (where N = num_cores, typically 32 users). The compute kernels were identical across cores (no per-core CTA variation) — pure legacy redundancy. The writer's only per-core variation was a single `core_id` CTA used as a runtime array index, with no compile-time loop unrolling impact. The port collapses both to single `KernelSpec`s and promotes `core_id` to a per-node named RTA. The [Demoting per-group CTA to RTA](metal2_port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta) anti-pattern doesn't apply here because the legacy kernel's `core_id` usage is purely runtime (index calculation, conditional branch on `>= FACE_WIDTH`), not loop-unrolling.

### No borrowed-memory CBs, no semaphores → simple spec shape
Sampling is the cleanest possible Metal 2.0 port: 16 standard DFBs (entry_size × num_entries copied from the legacy `total_size` math), 6 TensorParameters mapped 1:1 from input tensors, single WorkUnitSpec on the full grid. No `borrowed_from`, no `SemaphoreSpec`, no `is_pre_all_gather`-style variant branching.

## Friction

### Confusion — `DFBAccessor::operator uint32_t()` constexpr-convertibility for templated CB args
The legacy compute kernel passes CB indices as non-type template parameters (e.g., `top_k<input_cb_index, ...>()`, `reduce_c<..., scaler_max_cb_index, ...>()`). After the port, these template args are `dfb::name`-initialized `constexpr uint32_t` values. The migration guide confirms `dfb::name` is constexpr-convertible to `uint32_t`, so the values resolve at compile time. No code change needed beyond replacing `get_compile_time_arg_val(N)` with `dfb::name` at the declaration site.

## Open items for downstream

### Build & test verification (sampling)
1. `cmake --build build_Release --target ttnncpp -j 8`
2. `pytest tests/ttnn/unit_tests/operations/reduction/test_sampling.py -x -v`

### Cross-op kernel touches
None — every kernel `source` is under the op's own directory.

### Pool deferral
`ttnn/cpp/ttnn/operations/pool/generic/` was scoped out of this session. It uses `create_workload_descriptor` with workload-scoped MeshBuffers (sliding-window halo lookup table + avg-pool scalar config tensor uploaded on the host). Metal 2.0's `ProgramSpecMeshWorkloadFactoryAdapter` explicitly TODOs this case ("support op-owned resource tensors — the prepare_resources analog from the descriptor adapter"). Port is blocked until that framework support lands.
