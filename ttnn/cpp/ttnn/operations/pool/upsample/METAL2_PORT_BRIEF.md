# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/pool/upsample`

> Audit cleared all gates **for a subset of this op's factories**. `UpsampleBilinearProgramFactory` is blocked on the Device 2.0 prerequisite and is **excluded** from this brief — see `METAL2_PREPORT_AUDIT.md` for its findings and status. This brief covers the three clean factories only:
>
> - `UpsampleMultiCoreInterleavedProgramFactory`
> - `UpsampleMultiCoreShardedProgramFactory` (`WorkloadDescriptor`, op-owned config tensor)
> - `UpsampleNearestFloatProgramFactory`
>
> Do not port `UpsampleBilinearProgramFactory` alongside these — it has no brief. When the Device 2.0 team clears the finding in `bilinear.cpp`, this op is re-audited for that factory alone.

**Gates cleared (for the three factories above):** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `a21c8f3f324 2026-07-28 docs(metal_2.0): put experimental/quasar out of bounds for production ports`

## TTNN factory analysis

All three factories port to `MetalV2FactoryConcept`.

- **`UpsampleMultiCoreInterleavedProgramFactory`** — current concept: `descriptor`. Op-owned tensors: none. Target: `MetalV2FactoryConcept`.
- **`UpsampleNearestFloatProgramFactory`** — current concept: `descriptor`. Op-owned tensors: none. Target: `MetalV2FactoryConcept`.
- **`UpsampleMultiCoreShardedProgramFactory`** — current concept: `WorkloadDescriptor`, secretly SPMD (single structurally-identical program replicated across mesh coords — see `create_workload_descriptor`, `device/upsample_program_factory_multicore_sharded.cpp:409-478`). Op-owned tensors: **yes** — the per-core halo/replication config tensor (`config_tensor_owner`, stashed in `WorkloadDescriptor::buffers`, `device/upsample_program_factory_multicore_sharded.cpp:457-461`), carried natively by the target concept. Target: `MetalV2FactoryConcept` with op-owned tensors.
- **Gate-cleared, confirmed absent** (all three factories): custom hash · `get_dynamic_runtime_args` · `override_runtime_arguments` · pybind `create_descriptor` · other migration-risky pybind. All `no`.

## Construct — to do

### `UpsampleMultiCoreInterleavedProgramFactory`

**Tensor bindings:**
- `input` — **Case 1** (via `TensorAccessor`, `Buffer*`-binding form today) → express as `TensorParameter`/`TensorBinding`; reader kernel (`reader_upsample_unary_stick_layout_interleaved_start_id.cpp:18-19`) uses `TensorAccessor(tensor::name)`.
- `output` — **Case 1** → same treatment; writer kernel (`writer_upsample_interleaved.cpp:31-32`).

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — no site passes a 3rd argument.

**CB endpoints:**
- Row-major path: 1 CB (`src0_cb_index`, reused as output) — already legal (1,1): reader produces, writer consumes.
- Tiled path: 2 CBs — `src0_cb_index` legal (1,1) (reader produces / compute consumes), `output_cb_index` legal (1,1) (compute produces / writer consumes). No self-loop or 1P+1C assignment needed on this factory.

### `UpsampleNearestFloatProgramFactory`

**Tensor bindings:**
- `input` — **Case 1** (`reader_upsample_nearest_float.cpp:30-31`).
- `output` — **Case 1** (`writer_upsample_nearest_float.cpp:22-23`).

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none.

**CB endpoints:** 1 CB (`output_cb_index`) — already legal (1,1): reader produces, writer consumes.

### `UpsampleMultiCoreShardedProgramFactory`

**Tensor bindings:**
- `input` — **clean** (borrowed-memory DFB: `CBDescriptor.buffer = input.buffer()` on `in_cb`, `device/upsample_program_factory_multicore_sharded.cpp:312-321`); port via `DataflowBufferSpec::borrowed_from`.
- `output` — **clean** (borrowed-memory DFB on `out_cb`, lines 328-337).
- `config` (op-owned) — **clean** (borrowed-memory DFB on `config_cb`, lines 356-366); the op-owned buffer itself is carried by the `WorkloadDescriptor`/`MetalV2FactoryConcept`, not by an RTA.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — this factory doesn't use `TensorAccessor` at all (raw NoC addressing from the config lookup table).

**CB endpoints:** all three CBs are the **dual-instance work-split** shape — the same kernel source (`writer_upsample_multi_core_sharded.cpp`) instantiated twice (reader + writer `KernelDescriptor`s, differing only by the `is_reader` CT arg and `ReaderConfigDescriptor`/`WriterConfigDescriptor`) over the same `cores_with_work`:
- `in_cb` — **assign 1P+1C** (both instances raw-peek via `get_read_ptr()`, sync-free).
- `config_cb` — **assign 1P+1C** (both instances raw-peek via `get_read_ptr()`, sync-free).
- `out_cb` — **assign 1P+1C** (both instances raw-write disjoint offset ranges via `noc.async_read(..., out_dfb, ..., {.offset_bytes=...})`; output is resident, nothing drains it — same shape as the `reshard` reference example in the patterns catalog).

No multi-binding advanced option needed anywhere in this factory; no dead CBs.

## Watch for

- **CB endpoints (multi-binding):** none — no CB in any of these three factories needs the multi-binding advanced option.
- **Cross-op / shared kernels:** `UpsampleMultiCoreInterleavedProgramFactory`'s tiled path instantiates `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize.cpp` by file path. A `_metal2` fork **already exists** beside it at `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_metal2.cpp` — **bind that fork, don't re-fork it.** Other ops binding the legacy (pre-fork) file — a **sunset list, not authorization to convert it in place**: `data_movement/untilize`'s own four factories, and `data_movement/untilize_with_unpadding`'s three factories.
- **RTA varargs:** none — every kernel in these three factories reads a small, fixed set of RTAs at fixed positions.
