# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/embedding`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `d28425ca5cf 2026-07-22 docs(metal_2.0): route Gen1 porters away from the Quasar-uplift audit helper` *(carry this line into the port report's Provenance section)*

**Scope:** one DeviceOperation, `EmbeddingsDeviceOperation`, with three `descriptor`-concept factories, all portable:
- `EmbeddingsRMProgramFactory` (`device/embeddings_rm_program_factory.cpp`) — row-major output
- `EmbeddingsFusedProgramFactory` (`device/embeddings_fused_program_factory.cpp`) — tilized output
- `EmbeddingsTilizedIndicesProgramFactory` (`device/embeddings_tilized_indices_program_factory.cpp`) — TILE-layout index input

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (see `ttnn_factory.md`); the op ports to `MetalV2FactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` (all three factories return `tt::tt_metal::ProgramDescriptor` from a single static `create_descriptor(...)`).
- **Op-owned tensors:** none (the `descriptor` concept does not carry them; factories populate only `desc.cbs` and `desc.kernels`).
- **Target concept:** `MetalV2FactoryConcept` (no op-owned tensors).
- **Gate-cleared, confirmed absent** (each would have blocked the brief): custom hash (no `compute_program_hash`), `get_dynamic_runtime_args`, pybind `create_descriptor` — all absent in both sheet and code. The op has no `override_runtime_arguments` hook in any factory (the readiness sheet's `PD override_runtime_args (?)` column flags the tilized-indices factory, but no such hook exists in the code; treated as a non-gating sheet discrepancy in the audit, nothing for you to port).

## Construct — to do

**Tensor bindings** (per binding). Every tensor reaches its kernel through the `Buffer*`-binding form today (the factory pushes `Buffer*` objects into the RT-arg lists, and the kernel receives a `uint32_t` base and feeds it to a `TensorAccessor`); express each as a `TensorParameter` / `TensorBinding` and construct `TensorAccessor(tensor::name)`:

- **`input`** (index tensor) — **Case 1** in all three readers (`embeddings.cpp:39`, `embeddings_tilize.cpp:35`, `embedding_ind_tilized.cpp:35`). Straightforward: bind, build `TensorAccessor(tensor::input)`, drop the address RTA and its `TensorAccessorArgs` CTA plumbing.
- **`weights`** — **Case 1** in all three readers (`embeddings.cpp:40`, `embeddings_tilize.cpp:36`, `embedding_ind_tilized.cpp:36`). Same treatment, with one subtlety in the fused reader (below).
- **`output`** — split by config:
  - **interleaved / non-sharded output** — **Case 1**, reached via the writer's `TensorAccessor` (`embeddings_rm_writer_chunked.cpp:26`, `writer_unary_stick_layout_interleaved_start_id.cpp:20`, `writer_unary_interleaved_start_id.cpp:31`). Bind and build `TensorAccessor(tensor::output)`.
  - **sharded output** (RM `HEIGHT_SHARDED`; fused block/width-sharded) — **clean borrowed-memory DFB**: the output CB is backed directly by the output buffer (`CBDescriptor.buffer = out_buffer`) and no writer runs; the producer writes into it. Port via `DataflowBufferSpec::borrowed_from` the output `TensorParameter`. The tilized-indices factory has no sharded path, so its output is always Case 1.

  No Case 2 (raw-pointer) bindings exist anywhere: every tensor touch goes through a `TensorAccessor`, so there is no `get_bank_base_address` bridge work.

  > **Fused weights accessor base offset — do not drop it.** `embeddings_tilize.cpp:36` builds `TensorAccessor(weights_args, weight_buffer_src_addr + weight_offset)`. `weight_offset` is RTA arg 4, supplied by the factory as a *separate scalar* (`embeddings_fused_program_factory.cpp:325`); it is non-zero only for block/width-sharded output (`embeddings_fused_program_factory.cpp:340-344`) and zero otherwise. Under a `tensor::weights` binding the accessor base is fixed to the buffer base, so you cannot add `weight_offset` to the base. **Fold `weight_offset` into the accessor read's `offset_bytes`** instead (add it to the `weight_chunk_offset` that `read_token_async` already passes as `offset_bytes` in `embeddings_common.hpp:82`). This is arithmetically identical to the current base shift (the accessor's page-to-bank mapping depends only on `page_id` and the aligned page size; a flat byte addend lands the same whether applied to the base or to `offset_bytes`, and `weight_offset` stays within one weight page). A naive `TensorAccessor(tensor::weights)` that drops `weight_offset` will silently mis-address sharded fused output. The RM and tilized-indices readers pass a clean weights base (no such offset), so this applies to the fused factory only. See the audit's open question: confirm this relocation with the framework owner if you want a second opinion before applying it.

**TensorParameter relaxation:** none (no custom hash, so nothing to relax).

**TensorAccessor 3rd arg:** drop the redundant page-size argument at `embeddings_rm_writer_chunked.cpp:26` (`TensorAccessor(dst0_args, dst_addr, output_page_size)` becomes 2-arg). This is Class 2 (interleaved, correct-magnitude, realigned by the addrgen), a pure no-op drop. Do **not** set `dynamic_tensor_shape` (it is not a Class 1 dynamic-page case). RM factory only; every other accessor in the op is already 2-arg.

**CB endpoints:** classify per `(CB, config)`; all resolve at port time, none gate.
- **Self-loop** (single toucher: bind the one kernel PRODUCER and CONSUMER):
  - every index scratch CB `c_1` (all three factories) — the reader `reserve_back(1)`s it as a scratch buffer and `push_back(1)`s once at the end, with no consumer.
  - every weight-cache CB (RM `c_2`, fused `c_3`, tilized-indices `c_2`) on the `PADDED` / `BINARY` paths — touched only by the reader via `prepare_local_cache`.
  - the borrowed-memory output CB on sharded configs (RM `c_0`, fused `c_2`) — one toucher (the producer), no writer.
- **Plain 1:1** (bind one PRODUCER + one CONSUMER; roles already fixed): the output CB on interleaved configs (RM `c_0` reader→writer; fused `c_2` compute→writer; tilized-indices `c_0` reader→writer, where `c_0` doubles as weights-in and output), and the fused weights CB `c_0` (reader→compute).
- **Multi-binding advanced option:** not needed anywhere. No hidden second writer (the op uses no semaphores), no multi-reader, no dual-instance work-split (the fused factory's two compute descriptors cover disjoint core groups, so each node sees one instance).
- **Dead-CB drop:** none.

## Watch for

- **Fused weights accessor base offset (the load-bearing one):** repeated here so it is not missed — relocate `weight_offset` into `offset_bytes`, do not drop it (see Construct → weights). Only bites the fused factory, sharded output.
- **CB endpoints (multi-binding):** none to hunt. No semaphore-gated raw co-fill and no multi-reader in this op.
- **Cross-op / shared kernels — port the shared kernel as one unit.** Three kernels are instantiated by file path and shared beyond this op; each one's Metal 2.0 rewrite (CB to DFB, named-token bindings) must be adopted by every co-borrower in the same change, or the co-borrowers break the instant one op migrates alone:
  - `ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp` (shared pool) — also used by `data_movement/concat` and `data_movement/slice`.
  - `ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/compute/tilize.cpp` — used by `tilize`, `tilize_with_val_padding`, `untilize`, `untilize_with_unpadding`, `moreh/moreh_getitem`, `pool/upsample`, `sliding_window/halo`, `deepseek_prefill/combine`, `quasar/tilize_with_val_padding`, and this op.
  - `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` — broadly shared (~22 op families).

  All three are already Device 2.0 compliant, so no Device 2.0 blocker; the coupling is about coordinating the Metal 2.0 syntax rewrite. (`tilize.cpp` and `writer_unary_interleaved_start_id.cpp` are used only on the fused factory's non-chunked / non-sharded paths; `writer_unary_stick_layout_interleaved_start_id.cpp` on the RM non-chunked path and the tilized-indices path.)
- **RTA varargs:** none. Every kernel reads its runtime args at fixed indices (the `PADDED` pad token is a single fixed trailing arg). Port all runtime args as named args.
