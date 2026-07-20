# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/data_movement/sharded/reshard`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `6a16e3bf8d8 2026-07-20 recipe: reframe migration-guide 'not yet available' as surface-maturation` *(carry this line into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the op ports to `MetalV2FactoryConcept`. Carry them forward:

- **Current concept:** `descriptor` — all 8 factory instantiations (`ReshardSameWidthFactory<true/false>`, `ReshardSameHeightFactory<true/false>`, `ReshardGenericFactory`, `NdReshardCopyPagesFactory`, `NdReshardCopyLocalShardFactory<true/false>`), each a single `create_descriptor()` → `ProgramDescriptor`.
- **Op-owned tensors:** none.
- **Target concept:** `MetalV2FactoryConcept` (no op-owned tensors).
- **Gate-cleared, confirmed absent:** custom hash · custom `override_runtime_arguments` / runtime-args update · pybind `create_descriptor` · smuggled pointer (`Is safe to port? = yes`). All `no` on this op.

**Note on kernels:** every referenced kernel is *already* on Device-2.0 idioms (`Noc`, `DataflowBuffer` methods, `TensorAccessor`, `CoreLocalMem`, `AllocatorBank`, `UnicastEndpoint`). The port is a syntax rewrite (CB→DFB spec, named-token bindings) — no Device-2.0 cleanup is owed.

## Construct — to do

**Tensor bindings** (per factory, per binding). Every base is already delivered via a `Buffer*` binding or a borrowed-memory CB — never a raw `->address()`-in-RTA value — so none is the fast-path-cache stale-pointer hazard; the port simply moves each to a typed `TensorParameter`/`TensorBinding`.

- `NdReshardCopyPagesFactory` — **input** & **output**: **Case 1** (via `TensorAccessor`) → express each as `TensorParameter`/`TensorBinding`; kernel builds `TensorAccessor(tensor::name)`. The CTA `TensorAccessorArgs` plumbing (`input_accessor_args.get_compile_time_args()`, `output_accessor_args`) and the `emplace_common_runtime_args({buffer})` base both disappear.
- `NdReshardCopyLocalShardFactory<true/false>` — **input** & **output**: **Case 1** (via `TensorAccessor`), same as above. Two DM kernels (BRISC + NCRISC) share the source `nd_reshard_copy_local_shards.cpp`.
- `ReshardGenericFactory` — **input**: **Case 2** (raw pointer) → bind the input tensor, pull the base via `get_bank_base_address`; the kernel's raw `noc.async_read({.noc_x,.noc_y,.addr = input_shard_addr + off})` walk stays unchanged. **output**: **clean** — borrowed-memory DFB (c_16 currently `cb.buffer = output_buffer`) → `DataflowBufferSpec::borrowed_from` the output tensor.
- `ReshardSameWidthFactory<true/false>` — **remote** tensor: **Case 2** (raw; `AllocatorBank` + `bank_id` + base) → bind + `get_bank_base_address` bridge, raw walk unchanged. **local** tensor: **clean** — borrowed DFB (c_0 `cb.buffer = local_buffer`) → `borrowed_from`.
- `ReshardSameHeightFactory<true/false>` — **remote**: **Case 2** (bridge). **local**: **clean** — borrowed DFB (c_0) → `borrowed_from`.

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none — all accessor constructions are already 2-arg.

**CB endpoints:**

- `NdReshardCopyLocalShardFactory<*>`: **no CB** — nothing to build.
- `NdReshardCopyPagesFactory` c_0: **legal (1, 1)** staging DFB — reader is FIFO producer, writer is FIFO consumer. Plain `DataflowBufferSpec` (not borrowed).
- `ReshardGenericFactory` c_16: borrowed-from output → **set the multi-binding advanced option** (2 raw writers/node: the one source runs as both Reader-config and Writer-config).
- `ReshardSameWidthFactory<true>` c_0: borrowed-from local → **multi-binding** (2 writers/node). c_1 scratch (only when *unaligned && local_is_output*): plain scratch DFB, **multi-binding** (each of the 2 instances writes+reads it).
- `ReshardSameWidthFactory<false>` c_0: borrowed-from local → **multi-binding** (2 readers/node). No scratch CB.
- `ReshardSameHeightFactory<true>` c_0: borrowed-from local → **multi-binding** (2 writers/node).
- `ReshardSameHeightFactory<false>` c_0: borrowed-from local → **multi-binding** (2 readers/node).

## Watch for

- **CB endpoints (multi-binding):** the multi-binding here is the *visible* face — one kernel source instantiated twice (Reader-config + Writer-config) on the same core range to split work across BRISC/NCRISC, so each borrowed CB has two same-kind endpoints per node. No hidden semaphore-gated second writer exists (the op uses no semaphores), so set the flag from the census — no hunt needed beyond confirming the two instances.
- **Cross-op / shared kernels:** the 6 legacy kernels live in the in-family parent pool `data_movement/sharded/device/kernels/dataflow/` (not the op's own dir). Their Metal 2.0 rewrite is a shared rewrite, but no other op instantiates these exact paths, so the port-together set is just this op. Rewrite them in place.
- **RTA varargs:** all 6 legacy kernels read a variable-length RTA block via a loop-indexed read (`get_arg_val(arg_index++)` loops in `reshard_reader.cpp`/`reshard_reader_diff_width.cpp`; `args = get_arg_addr(N)` + `args[args_idx++]` loops in the 4 same_width/same_height kernels). Port these as varargs — do not try to name each. The 3 ND kernels read only fixed-index args (name those normally).
