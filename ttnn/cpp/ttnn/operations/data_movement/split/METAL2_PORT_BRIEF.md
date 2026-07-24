# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/data_movement/split`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `44da718b06b 2026-07-23 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper` *(carry this line into the port report's Provenance section)*

**Unit of work:** one device operation (`SplitDeviceOperation`), one program factory (`SplitProgramFactory`, `create_descriptor` in `device/split_program_factory.cpp`), two op-owned kernels (`device/kernels/dataflow/reader_tm_tile_layout_split_two_chunks.cpp`, `device/kernels/dataflow/writer_split_n_chunks_tile.cpp`).

**Scope reminder:** only the native TILE device op is in scope. The host-facing `ttnn::split` (`split.cpp`) also has a slice-fallback backend (`split_with_slice_impl` → `ttnn::slice`); that path runs a different op and is *not* part of this port.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `MetalV2FactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` (`create_descriptor` returns `tt::tt_metal::ProgramDescriptor`).
- **Op-owned tensors:** none — carried by neither concept; nothing to wire.
- **Target concept:** `MetalV2FactoryConcept`.
- **Gate-cleared, confirmed absent** (each would have blocked this brief): custom hash · custom `override_runtime_arguments` · pybind `create_descriptor` · runtime-args update · smuggled pointer — all `no` on the readiness sheet and confirmed in code. Sheet `Op Classification` = `PD (pointer-patching)`: today's tensor addresses ride the framework's `Buffer*`-binding / cache-hit-patch mechanism; your typed `TensorBinding`s supersede it.

## Construct — to do

**Tensor bindings** (per binding):

- `in0` (input, read by the reader) — **Case 1** (via `TensorAccessor`) → express as one `TensorParameter` / `TensorBinding`; the reader builds `TensorAccessor(tensor::name)` instead of `TensorAccessor(in0_tensor_args, in0_tensor_addr)`. The current `in0_buffer` `Buffer*` RTA (arg 1) and its `TensorAccessorArgs<5>` plumbing both disappear; the per-core tile offset `reader_core_id` (arg 0) stays as a named runtime arg (it is a page index, not an address).
- **output chunks** (written by the writer) — **Case 1** (via `TensorAccessor`), **N = `num_splits` distinct bindings**. Bind each output tensor as its own `TensorParameter`, scoped to that chunk's `CoreRange` group, so each core's writer instance addresses the correct output. The current per-core `output_buffers[chunk_id]` `Buffer*` RTA (arg 1) disappears; the per-core tile offset `writer_core_id` (arg 0) stays as a named runtime arg. **Confirm the multi-output binding shape first — see Watch for.**

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — both accessors are already 2-arg; nothing to drop.

**CB endpoints:** all legal. The single `src0` CB (`buffer_index = 0`, 2-tile double buffer) is a clean 1-producer (reader) / 1-consumer (writer) FIFO on every node → one `DataflowBufferSpec` bound PRODUCER on the reader and CONSUMER on the writer. No self-loop, no 1P+1C assignment, no multi-binding flag, no dead-CB drop.

**Kernel-side swaps (both kernels are already Device 2.0 / on the `DataflowBuffer` object):** replace the integer DFB id (`DataflowBuffer dfb_in0(0)` / `DataflowBuffer dfb_out(0)`) with the `dfb::name` binding token, and the RTA-fed `TensorAccessor(args, addr)` with `TensorAccessor(tensor::name)`. Per the kernel-side whitelist these are syntax swaps only — the `Noc` calls, FIFO ops, and loop structure stay as-is. (Optional cleanup you may fold in per the whitelist: `dfb.get_entry_size()` already reads tile metadata off the object, so no change needed there.)

## Watch for

- **CB endpoints (multi-binding):** none — the only CB is a plain 1:1 FIFO; no hidden second writer, no multi-reader.
- **Variable-count output bindings (the one non-mechanical part of this port):** the writer is a single kernel over `all_cores`, but each core writes one of `N = num_splits` outputs (today selected by the per-core `output_buffers[chunk_id]` `Buffer*` RTA, `split_program_factory.cpp:65`; the chunk-to-core-group mapping is built in `setup_runtime`). `N` is fixed per compiled program (hashed via `operation_attributes.num_splits`), so this is a fixed-count-per-program set of bindings, not a within-program runtime-varying count — it does **not** gate. Bind each output `TensorParameter` to its chunk's `CoreRange` sub-region within the one writer `KernelSpec`. **Confirm this binding strategy with the user/framework before building** (the recipe works examples of variable-count *input* tensors, not per-core-subset multi-*output* bindings) so you build toward it directly.
- **Cross-op / shared kernels:** none — both kernels are owned by the op; no port-together coupling.
- **RTA varargs:** none — name every runtime arg. Reader: `reader_core_id` (page offset, arg 0); drop the dead `split_last_dim` (arg 2, unused and always 0) rather than naming it. Writer: `writer_core_id` (page offset, arg 0). The compile-time args (per-core tile counts and strides, 5 each) are all fixed-offset — name each.
