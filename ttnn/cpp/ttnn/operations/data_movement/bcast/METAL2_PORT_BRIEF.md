# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/data_movement/bcast`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `8086bd9df7d 2026-08-07 docs(metal_2.0): add the fake-FIFO DM self-loop recipe, hardened by a cold run` *(carry this line into the port report's Provenance section)*

> **Scope of this port: the `BcastMultiCoreHW` factory only.** The other four factories (`BcastMultiCoreH`, `BcastMultiCoreW`, `BcastShardedH`, `BcastShardedHOptimised`) are **already ported** (`create_program_artifacts`). HW is the fifth and last. It was previously held back by its borrowed cross-family writer; that is now a rung-1 fork reuse (see *Watch for*), so the port can proceed. The whole-op binding detail below is retained for context, but only the HW rows are new work.

> **✅ PORT STATUS: HW PORTED and verified (interleaved).** All 5 factories now on `MetalV2FactoryConcept`. Writer → rung-1 reuse of `writer_unary_interleaved_start_id_metal2.cpp`; compute → rung-2 fork `bcast_hw_metal2.cpp` (legacy `bcast_hw.cpp` retained for `rotate_half`). See `METAL2_PORT_REPORT.md` for the full outcome, verification, and sunset lists.

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
- **Cross-op / shared kernels — the HW factory's two shared kernels (this is the crux of the HW port):**
  1. **Donor writer** `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` (borrowed, cross-family; 32 legacy binders). **Rung 1 — REUSE the existing fork** `writer_unary_interleaved_start_id_metal2.cpp` (beside the original). Point the writer `KernelSpec::source` at it and adopt **its** interface (do **not** rename to bcast's locals):
     - DFB: bind `c_16` (output) as `dfb::out`, **CONSUMER**.
     - Tensor: bind the output `TensorParameter` as `tensor::dst`.
     - Named args: `num_pages` (← legacy `num_tensor_tiles_per_core`), `start_id` (← legacy `num_tiles_read`).
     - Defines: emit `OUT_SHARDED` when the output is sharded (matches the legacy `writer_defines`); `BACKWARDS` stays off for bcast.
     - The fork reads page size from `dfb.get_entry_size()`, so the legacy writer CTAs (`output_cb_index` + `TensorAccessorArgs`) **dissolve** — don't carry them over.
     - The fork **already has consumers** (`copy/typecast`, `gelu_backward`) → it is **read-only** to you: do not edit it, do not fork it. bcast's call site fits it exactly, so no change is needed; if you think it doesn't fit, re-derive from the legacy factory and, if the need is real, **stop and report** (per `Caution: Porting a shared kernel`).
  2. **Compute** `bcast_hw.cpp` — bcast-owned but **LENT** to `experimental/transformer/rotate_half` (on the legacy device-op concept, not migrating soon). **No fork exists yet → Rung 2 — CREATE `bcast_hw_metal2.cpp`** beside the original in bcast's own compute dir. Convert the copy (named `dfb::`/`args::` bindings), point the HW compute `KernelSpec::source` at it, and **leave `bcast_hw.cpp` untouched** except for the standard pointer comment — it must keep serving `rotate_half`. Name the fork's bindings for the kernel's role (`dfb::in0`/`dfb::in1`/`dfb::out`), consistent with the sibling `bcast_h.cpp`/`bcast_w.cpp` Metal 2.0 kernels. Record `rotate_half` as the still-unmigrated consumer (sunset list) in the port report.
  - The HW **reader** `reader_bcast_hw_interleaved_partitioned.cpp` is bcast-only → convert in place (no fork).
- **RTA varargs:** none — all runtime args are fixed-index distinct fields; name each in `runtime_arg_schema`. (Note the dead RTAs/CTAs listed in the audit's Misc anomalies — do **not** carry them into the port; they route to the ops team, not the port diff.)
