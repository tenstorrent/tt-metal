# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** ProgramDescriptor ✓ · Device 2.0 ✓ · Features ✓

**Recipe docs:** `776151aeca8 2026-06-24 docs(metal2): clarify SPSC face-(b) producer-as-consumer, aliased-vs-same-FIFO, no-portable-subset` *(carry this line into the port report's Provenance section)*

One device operation, three factories: `RotaryEmbeddingLlamaMultiCore` (prefill interleaved), `RotaryEmbeddingLlamaMultiCorePrefillSharded` (prefill, cos/sin and/or trans_mat HEIGHT_SHARDED), `RotaryEmbeddingLlamaMultiCoreSharded` (decode, fully HEIGHT_SHARDED). All five kernels are owned in-directory.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `port_op_to_metal2_ttnn_factory.md`); the op ports to `MetalV2FactoryConcept`. Carry them forward:

- **Op-owned tensors:** none (only the declared output via `create_output_tensors`).
- **MeshWorkload:** not needed (single `ProgramDescriptor` per factory; no cross-program/cross-device coordination).
- **Pybind `create_descriptor`:** none (`rotary_embedding_llama_nanobind.cpp` binds only the user-facing op via `bind_function`).
- **Other risky pybind:** none.
- **Custom `override_runtime_arguments`:** none.

## Construct — to do

**Tensor bindings** (per binding; classification varies per factory — honor the split):

- **MultiCore (interleaved):** `input`, `cos`, `sin`, `trans_mat`, `output` — all **Case 1** (via `TensorAccessor`). Today the factory passes each base address as a `Buffer*` runtime arg (`..._multi_core_program_factory.cpp:337-338`) and the kernel builds `TensorAccessor(args, addr)` (`reader_...interleaved...:47-55`, `writer...:38`). Express each as a `TensorParameter`/`TensorBinding`; kernel uses `TensorAccessor(ta::name)`; drop the RTA address + `TensorAccessorArgs` CTA plumbing.
- **PrefillSharded:** `input`, `output` — **Case 1** (always TensorAccessor). `cos`, `sin`, `trans_mat` — **per-config split**: **clean (borrowed-memory DFB)** on the sharded fast path (`.buffer=` set; kernel reads via CB FIFO), **Case 1** on the reload path (`COS_SIN_SHARDED_RELOAD==1` / non-global trans_mat → `TensorAccessor`). Port the clean branch via `borrowed_from`, the reload branch via `TensorParameter`.
- **Sharded (decode):** `input`, `cos`, `sin`, `trans_mat`, `output` — all **clean (borrowed-memory DFB)**. Every CB is `.buffer=<buffer>`; the compute-only kernel reads/writes through CB FIFO ops. Port all via `DataflowBufferSpec::borrowed_from`.

No Case 2 (raw-pointer) bindings, so no compute-kernel `TensorBinding` blocker.

**Custom hash:** delete custom `compute_program_hash` → default (sanctioned exception). It is at `device/rotary_embedding_llama_device_operation.cpp:228-232` (declaration `...hpp:29`); the body already reproduces the default `hash_operation<...>`, so deletion is behavior-preserving.

## Watch for

- **Notable constructs:** borrowed-mem DFB → port via `DataflowBufferSpec::borrowed_from`. Sites: Sharded factory `rotary_embedding_llama_sharded_program_factory.cpp:87,99,111,125,171`; PrefillSharded fast paths `rotary_embedding_llama_multi_core_prefill_sharded_program_factory.cpp:175,186,260`. No aliased CB, no dynamic TA, no semaphores.
- **Single-ended CB (`(1,0)`):** the decode factory output CB `c_16` (`.buffer = dst_buffer`, `..._sharded_program_factory.cpp:163-172`) is producer-only — the compute kernel `push_back`es but nothing `wait_front`/`pop_front`s it (`rotary_embedding_llama_sharded.cpp:67,118`). Apply the sanctioned **self-loop workaround** (fabricate the missing consumer side). Does not block.
- **Cross-op / shared kernels:** none — port the op alone; no port-together set.
- **RTA varargs:** none — every kernel reads a fixed, statically-known count of RTAs. Use named RTAs.
- **Dead CBs:** none.
