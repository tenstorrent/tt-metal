# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/data_movement/permute`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `2a53d817976 2026-07-24 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper` *(carry this line into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `MetalV2FactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` (all five factories: `MultiCoreRowInvariant`, `MultiCoreBlockedGeneric`, `MultiCoreTileInvariant`, `MultiCoreTileRowInvariant`, `MultiCoreTiledGeneric`).
- **Op-owned tensors:** none.
- **Target concept:** `MetalV2FactoryConcept` (no op-owned tensors).
- **Gate-cleared, confirmed absent** (each would have blocked the brief): custom hash · custom `override_runtime_arguments` · pybind `create_descriptor` — all `no`.

## Construct — to do

**Tensor bindings** (per binding — identical shape in every factory):

- `input_tensor` (`src_buffer`) — **Case 1** (via `TensorAccessor`) → express as `TensorParameter` / `TensorBinding`; kernel builds `TensorAccessor(tensor::name)`. Remove the `Buffer*` RTA slot (`emplace_runtime_args(core, {src_buffer, …})`) and the `TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args)` CTA plumbing — the binding supplies both.
- `output_tensor` (`dst_buffer`) — **Case 1**, symmetric. Remove the `Buffer*` RTA slot and the `TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args)` plumbing.

Kernels already build `TensorAccessor(args, addr)` from the delivered base and do all access through it (`noc.async_read(s, …)`, `s.get_noc_addr(…)`, `noc_async_read_sharded(noc, …, s0, …)`), so the kernel-side change is the mechanical `args, addr` → `tensor::name` swap. No raw-pointer (Case 2) bridges; no borrowed-memory DFBs.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — every accessor is already the 2-arg form.

**CB endpoints:**

- **Self-loop** these two intermediate CBs (single toucher — produced *and* consumed by the compute kernel alone): bind PRODUCER **and** CONSUMER.
  - `c_1` (tilize CB) in `MultiCoreBlockedGeneric` — compute kernel `transpose_xw_rm_single_tile_size.cpp`.
  - `c_1` (tilize CB) in `MultiCoreTiledGeneric` — compute kernel `transpose_xw_tiled.cpp`.
- All other CBs are legal 1P+1C FIFOs — bind the FIFO producer PRODUCER and the FIFO consumer CONSUMER, no special action. Per config:
  - `MultiCoreRowInvariant`: `c_0`.
  - `MultiCoreBlockedGeneric`: `c_0`, `c_2`.
  - `MultiCoreTileInvariant`: `c_0` (both configs), `c_16` (swap-HW only).
  - `MultiCoreTileRowInvariant`: `c_0`, `c_1` (padding config only), `c_16` (swap-HW only).
  - `MultiCoreTiledGeneric`: `c_0`, `c_2`, `c_3` (y-padding config only).

## Watch for

- **CB endpoints (multi-binding):** none — no hidden second writer / multi-reader / ≥3-toucher CB. The only non-1P+1C CBs are the two self-loops above.
- **Cross-op / shared kernels:** three donor kernels are instantiated by file-path; port each shared kernel's Metal 2.0 rewrite as one unit across all co-borrowers, not just for permute:
  - `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` — broadly shared unary writer (used by `MultiCoreTileInvariant`).
  - `data_movement/transpose/device/kernels/compute/transpose_wh.cpp` — shared with transpose (swap-HW compute).
  - `data_movement/transpose/device/kernels/dataflow/reader_unary_transpose_hc_interleaved_tiled_padding_aware.cpp` — shared with transpose (`MultiCoreTileRowInvariant` reader).
- **RTA varargs:** the RM and tiled reader/writer kernels read rank-length shape / permutation / stride arrays in a count-bounded loop (count = tensor rank, a CTA that varies per instantiation). Port these via the kernel-side vararg mechanism — do **not** try to name each element. Keep the per-core scalar prefixes (`src_addr`/`dst_addr`, `start`, `end`, padding-tile indices) as ordinary named RTAs. Sites: `writer_permute_interleaved_rm_row_invariant.cpp:28`, `reader_permute_interleaved_rm_blocked_generic.cpp:37`, `writer_permute_interleaved_rm_blocked_generic.cpp:60`, `reader_permute_interleaved_tiled_invariant.cpp:32`, `writer_permute_interleaved_tiled_row_invariant.cpp:84`, `reader_permute_interleaved_tiled_generic.cpp:93`, `writer_permute_interleaved_tiled_generic.cpp:94`.
