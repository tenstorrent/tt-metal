# Pre-port audit: `ttnn/cpp/ttnn/operations/normalization/layernorm/device`

- **`LayerNormDeviceOperation`** (in `layernorm_device_operation.{hpp,cpp}`)
  - `LayerNormMultiCoreProgramFactory` (interleaved input; `layernorm_op_multi_core.cpp`)
  - `LayerNormShardedProgramFactory` (sharded input; `layernorm_op_multi_core_sharded.cpp` + `sharded_layernorm_factory_helpers.{hpp,cpp}`)

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `port_op_to_metal2_audit.md`.

## Result

**GREEN** — both factories are eligible for Metal 2.0 port. No UNSUPPORTED feature signals fire. The user has scoped this audit to **port the interleaved factory only**; the sharded factory remains on `ProgramDescriptor` for a later session. Handoff to the port recipe is appropriate after explicit user go-ahead.

## Porting prerequisites

### ProgramDescriptor API: **GREEN**

Both factories implement `create_descriptor(...) -> tt::tt_metal::ProgramDescriptor` and use `KernelDescriptor`, `CBDescriptor`, `SemaphoreDescriptor`, `CBFormatDescriptor`, `ReaderConfigDescriptor`, `WriterConfigDescriptor`, `ComputeConfigDescriptor`. No `host_api.hpp`-style imperative `CreateProgram` / `CreateKernel` / `CreateCircularBuffer` / `CreateSemaphore` / `SetRuntimeArgs` calls in either factory.

### Device 2.0 DM: **GREEN**

All referenced dataflow kernels use Device 2.0 wrappers:
- `Noc noc;` instead of raw `noc_async_read`/`noc_async_write` free functions.
- `CircularBuffer cb_xxx(cb_id);` instead of raw `cb_reserve_back(cb_id, n)` / `cb_push_back(cb_id, n)` free functions.

Audited kernels:
- Interleaved readers: `reader_unary_interleaved_ln.cpp`, `reader_unary_interleaved_ln_large_tensor.cpp`, `reader_unary_interleaved_ln_large_tensor_welford.cpp`, `reader_unary_interleaved_ln_rm_gb.cpp`.
- Interleaved writers: `writer_unary_interleaved_start_id_blocked.cpp`, `writer_unary_interleaved_start_id_blocked_rm_output.cpp`.
- Sharded readers/writers: `reader_mcast_sender_unary_sharded_ln*.cpp`, `reader_mcast_receiver_unary_sharded_ln*.cpp`, `writer_unary_sharded_ln*.cpp`.

No isolated CB-index-keyed free-function holdovers observed.

### TensorAccessor usage: **GREEN**

- **Interleaved factory** — all tensor accesses use `TensorAccessor` with the standard `TensorAccessorArgs<N>()` plumbing pattern. Per-kernel CTA offsets are computed via `.next_compile_time_args_offset()` chaining. Host code uses `tt::tt_metal::TensorAccessorArgs(buffer).append_to(cta_vec)`.
- **Sharded factory** — sharded reader/writer kernels access tensor data through borrowed-memory CBs (CB 0/in0 backed by the input shard buffer, CB 1/in1 backed by the residual shard buffer when present, CB 7/stats backed by the stats shard, CB 17/out_reshard backed by the output buffer for resharding). Causal-link gate applies: the lack of `TensorAccessor` is by design — the borrowed-memory CB **is** the access mechanism. Translates to Metal 2.0 via `DataflowBufferSpec::borrowed_from`. The gamma/beta DRAM tensors (when present) are accessed via `TensorAccessor` in the sender-side reader kernels.

## Feature compatibility check

| Feature | Status | Notes |
|---|---|---|
| GlobalCircularBuffer | N/A | No `experimental::GlobalCircularBuffer` references; no `.global_circular_buffer` field set on any `CBDescriptor`. |
| Dynamic CircularBuffer (CB on borrowed memory) | GREEN (LANDED) | Used in both factories. Translates to `DataflowBufferSpec::borrowed_from`. |
| CBDescriptor `address_offset` (non-zero) | N/A | No `.address_offset` writes in either factory. |
| Aliased Circular Buffers | N/A | Every `format_descriptors` initializer is single-element; no aliased CBs. |
| GlobalSemaphore | N/A | No `GlobalSemaphore` references; the sharded factory uses plain `SemaphoreDescriptor`. |
| Non-zero semaphore initial value | GREEN | Sharded factory creates 3 semaphores all with `initial_value = 0`. Interleaved factory uses no semaphores. |
| `ArgConfig::Runtime*` tensor-accessor flavors | GREEN | No `ArgConfig::Runtime` token anywhere under the op's directory. |
| `UpdateCircularBuffer*` | GREEN | No `UpdateCircularBufferTotalSize` / `UpdateCircularBufferPageSize` calls. |

### Dynamic CircularBuffer (CB on borrowed memory): GREEN (LANDED)

**Signal:** `CBDescriptor::buffer` set to a non-null `Buffer*` in three sites:

- `device/layernorm_op_multi_core.cpp:734` — `recip_cb_desc.buffer = recip_tensor.value().buffer();` (Welford reciprocal LUT, CB 25). Interleaved factory.
- `device/sharded_layernorm_factory_helpers.cpp:964` — generic `cb_desc.buffer = buffer;` in the `make_cb_descriptor` helper; called for CB 0 / CB 1 / CB 14 with the input / residual / a-input shard buffers (lines 975, 985, 993).
- `device/sharded_layernorm_factory_helpers.cpp:1135` — `recip_cb_desc.buffer = cb_config.recip_buffer;` (Welford reciprocal LUT, CB 25). Sharded factory.
- `device/sharded_layernorm_factory_helpers.cpp:1148` — `stats_cb_desc.buffer = cb_config.stats_buffer;` (post-all-gather stats, CB 7). Sharded factory.
- (And in the sharded-only resharding case, CB 17 carries the output buffer — same path via `make_cb_descriptor` helper.)

**Expected resolution:** No port gate. Each affected `DataflowBufferSpec` declares `borrowed_from = <tensor_parameter_name>` where the named `TensorParameter` is the tensor whose `Buffer*` was the legacy `CBDescriptor::buffer` value. The kernel-side code that read from the borrowed-memory CB continues to work via the DFB wrapper.

## Path forward

GREEN — proceed with the port. Per user direction, the port covers only `LayerNormMultiCoreProgramFactory` (interleaved) in this session; `LayerNormShardedProgramFactory` stays on the legacy `ProgramDescriptor` API.

Cross-factory considerations to keep in mind:
- `select_program_factory` returns a `std::variant<LayerNormMultiCoreProgramFactory, LayerNormShardedProgramFactory>`. Mixing factory concepts within one `program_factory_t` variant is supported by the framework's `AllFactoriesValid` check, which accepts a mix of concepts across alternatives. Translation: porting only the interleaved factory's alternative while the sharded alternative stays on `ProgramDescriptorFactoryConcept` is mechanically fine.
- The interleaved factory's borrowed-memory CB (Welford reciprocal LUT) translates to a `DataflowBufferSpec` with `borrowed_from = RECIP_TENSOR` and the corresponding `TensorParameter` declared in `ProgramSpec::tensor_parameters` only when `use_welford` is true.

## Questions for the user

None — proceeding with the port per the chosen scope.
