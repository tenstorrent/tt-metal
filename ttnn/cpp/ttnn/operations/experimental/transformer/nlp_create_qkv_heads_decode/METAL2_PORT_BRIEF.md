# Metal 2.0 Port Brief — `experimental/transformer/nlp_create_qkv_heads_decode`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `de19c9df758 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper`

Three factories under one `DeviceOperation`, selected by input layout: **Interleaved** (non-sharded input),
**Sharded** (width-sharded), **ShardedSubcoregrid** (width-sharded on sub-core-grids). Port them together —
they share the `q/k/v` output structure and the reader+writer dual-instance shape.

## TTNN factory analysis
- **Current concept:** `descriptor` (all three factories)
- **Op-owned tensors:** none
- **Target concept:** `MetalV2FactoryConcept`
- **Gate-cleared, confirmed absent:** custom hash · custom `override_runtime_arguments` · pybind `create_descriptor` · migration-risky pybind. All `no`.

## Construct — to do

**Tensor bindings** (per binding; input classification varies per factory):

- **q/k/v output** — audit disposition: **clean (borrowed-memory DFB)** → `DataflowBufferSpec::borrowed_from`
  the output tensor.
  > **PORT NOTE:** on the Sharded/Subcoregrid factories this was **changed to a Case-2 `TensorParameter`**
  > (`get_bank_base_address`) to work around a framework bug (borrowed DFBs corrupt in multi-work-unit
  > programs — the `!overlap` layout needs two work units). Interleaved keeps borrowed DFBs (single WU).
  > See `METAL2_PORT_REPORT.md`.
- **input_tensor** — **Interleaved: Case 1** (`TensorAccessor(tensor::name)`); **Sharded/Subcoregrid:
  Case 2** (raw base via `get_bank_base_address`, keep the hand-rolled `UnicastEndpoint` reads unchanged).
- **batch_offset** (Sharded + Subcoregrid, when provided) — **Case 1**; gated by `use_batch_offset`.

**TensorParameter relaxation:** none. **TensorAccessor 3rd arg:** none.

**CB endpoints:**
- Output CBs `c_16`/`c_17`/`c_18`: dual-instance → **1P+1C** (interleaved; on sharded/subcoregrid these
  became TensorParameters per the port note above).
- Interleaved scratch `c_0`/`c_1`: **self-loop** (aligned path only).
- Subcoregrid batch-offset `c_15`/`c_14`: **self-loop** each.
- Sharded batch-offset `c_14`/`c_15`: **reconcile** — the writer is never switched to `c_14` (probable
  bug; subcoregrid switches correctly). *(PORT: confirmed a bug; one-line switch applied.)*

## Watch for
- **CB endpoints:** the only multi-binding-looking CB is sharded `c_15` (from the probable bug); resolve
  via Questions #1, don't set the flag.
- **Cross-op / shared kernels:** the interleaved kernel calls `tt::data_movement::common::tt_memmove(noc, …)`;
  Device 2.0 native, ports as-is, not this op's to rewrite.
- **RTA varargs:** the sharded and subcoregrid kernels read the input-core NoC-coordinate arrays as
  variable-count RTA blocks — port as RTA varargs, not per-element named args. The interleaved kernel has
  no varargs.
