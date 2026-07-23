# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/copy/typecast`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** provenance could not be pinned (`git log` on the metal_2.0 docs path was empty; recipe supplied standalone at `/localdev/edwinlee/metal2_audit.md`). *Carry this note into the port report's Provenance section.*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`). One `DeviceOperation` (`TypecastDeviceOperation`) with four factories, all porting to `MetalV2FactoryConcept`:

- **Current concept:** `descriptor` — all four factories (`TypecastProgramFactory`, `TypecastSubgridProgramFactory`, `TypecastShardedProgramFactory`, `TypecastRowMajorChunkedProgramFactory`), each a `static ProgramDescriptor create_descriptor(...)`.
- **Op-owned tensors:** none.
- **Target concept:** `MetalV2FactoryConcept` (no op-owned tensors).
- **Gate-cleared, confirmed absent:** custom hash · custom `override_runtime_arguments` · pybind `create_descriptor` — all `no` on the readiness sheet and in code. Op is classified `PD (pointer-patching)`: the factories deliver tensor bases as `Buffer*` runtime args (`BufferBinding` auto-patched), which the typed `TensorParameter` bindings supersede.

## Construct — to do

**Tensor bindings** (per binding, per factory — the same tensor differs by factory, so bind per factory):

- `TypecastProgramFactory` (interleaved / tiled + non-optimized-sharded fallback):
  - **input** — **Case 1** (via `TensorAccessor`) → express as `TensorParameter` / `TensorBinding`; the reader (`reader_unary_interleaved_start_id.cpp`) builds `TensorAccessor(tensor::name)` instead of `TensorAccessor(src_args, src_addr)`. The `src_buffer` RTA[0] and its `TensorAccessorArgs` CTAs both disappear.
  - **output** — **Case 1** → same treatment for the writer (`writer_unary_interleaved_start_id.cpp`); drop the `dst_buffer` RTA[0] and `TensorAccessorArgs`.
- `TypecastSubgridProgramFactory` (tiled, `sub_core_grids`) — **input** Case 1, **output** Case 1; identical interleaved reader/writer kernels to `TypecastProgramFactory`.
- `TypecastRowMajorChunkedProgramFactory` — **input** Case 1, **output** Case 1; the RM-chunked reader/writer (`reader_typecast_rm_chunked.cpp` / `writer_typecast_rm_chunked.cpp`) build `TensorAccessor(tensor::name)`. Keep the kernel-side `.offset_bytes` page arithmetic exactly as-is — the base is clean; only the accessor construction changes.
- `TypecastShardedProgramFactory` — **input** clean (borrowed-memory DFB) → bind `DataflowBufferSpec::borrowed_from(input.buffer())` for `c_0`; **output** clean (borrowed-memory DFB) → `borrowed_from(output.buffer())` for `c_2`. No `TensorAccessor`, no address RTA to remove.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — no accessor passes a page-size argument, nothing to drop.

**CB endpoints:**
- `TypecastProgramFactory`, `TypecastSubgridProgramFactory`, `TypecastRowMajorChunkedProgramFactory` — `c_0` and `c_2` are plain 1-producer/1-consumer FIFOs; bind normally, no special action.
- `TypecastShardedProgramFactory`:
  - `c_0` — **1P+1C**: sharded reader is the producer (`dfb.push_back`), compute the consumer; bind `borrowed_from(input.buffer())`.
  - `c_2` — **self-loop**: compute is the only toucher (no writer kernel drains it; the borrowed output buffer is the result). Bind compute as **both** PRODUCER and CONSUMER, `borrowed_from(output.buffer())`.

## Watch for

- **CB endpoints (multi-binding):** none — no hidden second writer, no multi-reader CB, no ≥3-toucher CB. The only non-1:1 disposition is the sharded `c_2` self-loop above.
- **Cross-op / shared kernels:** three eltwise/unary dataflow kernels are borrowed by file path — `reader_unary_interleaved_start_id.cpp`, `writer_unary_interleaved_start_id.cpp` (interleaved + subgrid), `reader_unary_sharded.cpp` (sharded). Their CB→DFB / named-token rewrite is **one shared change** across every op that instantiates them (the eltwise/unary family + typecast); port the shared kernel as a single unit so co-borrowers don't break. They are already Device 2.0 (`DataflowBuffer`/`TensorAccessor`), so the rewrite is clean — this is a sequencing caution, not a blocker.
- **RTA varargs:** none — name every runtime arg (readers/writers: `src_addr`/`dst_addr`, `num_pages`/`num_rows`, `start_id`/`start_row_id`; sharded reader: `num_tiles_per_core`).
