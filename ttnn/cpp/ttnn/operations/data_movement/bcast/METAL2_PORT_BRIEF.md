# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/data_movement/bcast`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `e9e376712e5 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper` *(carry this line into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `MetalV2FactoryConcept`. One `BcastDeviceOperation` with five program factories, all sharing the `bcast_h.cpp`-family compute and the same three-tensor binding set (`input_a`, `input_b`, `output`).

- **Current concept:** `descriptor` (all 5 factories: `BcastMultiCoreH`, `BcastMultiCoreW`, `BcastMultiCoreHW`, `BcastShardedH`, `BcastShardedHOptimised`).
- **Op-owned tensors:** none.
- **Target concept:** `MetalV2FactoryConcept`.
- **Gate-cleared, confirmed absent** (each would have blocked this brief): custom hash · custom `override_runtime_arguments` · pybind `create_descriptor` · other migration-risky pybind. All `no`.

## Construct — to do

**Tensor bindings** (per binding; the classification differs by factory/config, so bind per factory):

- **`input_b` (src1)** — **Case 1** (via `TensorAccessor`) in **every** factory → express as `TensorParameter`/`TensorBinding`; kernel builds `TensorAccessor(tensor::name)`. The legacy `Buffer*`-in-RTA (`src1_buffer` / `b.buffer()`) plus `TensorAccessorArgs` plumbing both disappear.
- **`input_a` (src0):**
  - **Case 1** (via `TensorAccessor`) in `BcastMultiCoreH`, `BcastMultiCoreW`, and `BcastMultiCoreHW` interleaved (no `IN0_SHARDED`).
  - **clean / borrowed-memory DFB** in `BcastShardedH`, `BcastShardedHOptimised`, and `BcastMultiCoreHW` `IN0_SHARDED` (CB `c_0` has `.buffer = src0_buffer`) → port via `DataflowBufferSpec::borrowed_from` the `input_a` `TensorParameter`.
- **`output` (dst):**
  - **Case 1** (via `TensorAccessor`) in `BcastMultiCoreH`, `BcastMultiCoreW`, and `BcastMultiCoreHW` interleaved.
  - **clean / borrowed-memory DFB** in `BcastShardedH`, `BcastShardedHOptimised`, and `BcastMultiCoreHW` `OUT_SHARDED` (CB `c_16` has `.buffer = dst_buffer`) → port via `DataflowBufferSpec::borrowed_from`.

No Case 2 (raw-pointer) bindings — do not reach for the `get_bank_base_address` bridge anywhere in this op.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — every `TensorAccessor` is already the 2-arg form. Nothing to drop.

**CB endpoints:**
- `BcastShardedHProgramFactory` and `BcastShardedHOptimisedProgramFactory`: **self-loop the output CB `c_16`** (single toucher — compute produces into the resident borrowed-memory output, nothing drains it; bind compute both PRODUCER and CONSUMER, legal on Gen1 for compute).
- All other CBs in all factories are legal 1:1 — including the HW borrowed-memory configs (`IN0_SHARDED`: reader-producer + compute-consumer on `c_0`; `OUT_SHARDED`: compute-producer + donor-writer-consumer on `c_16`). No multi-binding flag anywhere.

## Watch for

- **CB endpoints (multi-binding):** none — no hidden second writer, no multi-reader, no dual-instance work-split. Do not set the multi-binding advanced option anywhere in this op.
- **Cross-op / shared kernels:** the HW factory instantiates the cross-family donor writer `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (owned by `eltwise/unary`, borrowed by ~46 factories tree-wide). Its Metal 2.0 rewrite is a **single shared change every borrower must adopt together** — do not rewrite it in isolation for bcast. A `writer_unary_interleaved_start_id_metal2.cpp` already exists in the `experimental/quasar/` tree; coordinate with that pattern (likely a parallel `_metal2` file) rather than editing the shared writer in place.
- **RTA varargs:** none — all runtime args are fixed-index distinct fields; name each in `runtime_arg_schema`. (Note the dead RTAs/CTAs listed in the audit's Misc anomalies — do **not** carry them into the port; they route to the ops team, not the port diff.)
