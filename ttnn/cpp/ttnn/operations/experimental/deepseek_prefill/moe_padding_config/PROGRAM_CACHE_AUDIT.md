# Program Cache Audit — `experimental/deepseek_prefill/moe_padding_config`

Audit of `MoePaddingConfigDeviceOperation::compute_program_hash` against the framework default
("hash everything") key.

| | |
|---|---|
| Device operation | `ttnn::operations::experimental::deepseek_prefill::moe_padding_config::MoePaddingConfigDeviceOperation` (`device/moe_padding_config_device_operation.hpp:21`) |
| Custom hash | `device/moe_padding_config_device_operation.cpp:114-128` |
| `operation_attributes_t` | `tokens_per_chip`, `pad_side`, `cluster_axis` (all `uint32_t`) |
| `tensor_args_t` | `config` (in-place output), `actual_start`, `actual_end` |
| Program factories | one: `ProgramFactory::create_descriptor` (`ProgramDescriptor`-based), wrapped by `MeshWorkloadFactory` |
| `override_runtime_arguments` | **Yes**, at the mesh-workload level (`device/moe_padding_config_device_operation.cpp:220-241`) |
| `get_dynamic_runtime_args` | **No** |
| Cache-hit patch mechanism | **Op-owned workload-level override**, which internally delegates to the descriptor adapter's **buffer-binding fast path** and then hand-patches two common runtime args |

## Cache-hit patch mechanism

This op is the only one of the four `deepseek_prefill` ops audited here that uses a
`ProgramDescriptor` factory, and its cache-hit path is a hybrid worth spelling out precisely.

The framework's cache-hit dispatcher prefers a workload factory's `apply_descriptor` and otherwise
calls its `override_runtime_arguments`:

```279:285:ttnn/api/ttnn/device_operation.hpp
        if constexpr (requires { &WorkloadFactory::apply_descriptor; }) {
            WorkloadFactory::apply_descriptor(
                cached_mesh_workload, operation_attributes, tensor_args, tensor_return_value);
        } else {
            WorkloadFactory::override_runtime_arguments(
                cached_mesh_workload, operation_attributes, tensor_args, tensor_return_value);
        }
```

`MeshWorkloadFactory` declares no `apply_descriptor`, so the op's own hook runs on every hit. That
hook does two things:

```220:241:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_padding_config/device/moe_padding_config_device_operation.cpp
void MoePaddingConfigDeviceOperation::MeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& args,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    // Default adapter behaviour: patch operand buffer-binding addresses on cache hits.
    descriptor_adapter_t::apply_descriptor(cached_workload, args, tensor_args, output);
    // The metadata addresses are raw scalars in common runtime args, which the buffer-binding fast
    // path does not refresh — patch them on every cached program or the kernel would keep reading a
    // stale (possibly freed) address.
    constexpr uint32_t kWriterKernelHandle = 0;  // the only kernel pushed in create_descriptor
    const uint32_t start_addr = tensor_args.actual_start.buffer()->address();
    const uint32_t end_addr = tensor_args.actual_end.buffer()->address();
    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& writer_common = GetCommonRuntimeArgs(program, kWriterKernelHandle);
        TT_FATAL(
            kArgActualEndAddr < writer_common.size(),
            "moe_padding_config writer is missing its per-call common runtime args");
        writer_common[kArgActualStartAddr] = start_addr;
        writer_common[kArgActualEndAddr] = end_addr;
    }
}
```

The inner `descriptor_adapter_t::apply_descriptor` is instantiated on `ProgramFactory`, which
declares no `override_runtime_arguments` of its own, so inside the adapter the *fast path* is the
one that runs — the factory registered a `Buffer*` in `emplace_runtime_args`, so
`resolved_bindings.rt_args` is non-empty:

```726:731:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
                    if (!sv.resolved_bindings.rt_args.empty() ||
                        (!dynamic_args.empty() && !sv.resolved_bindings.empty())) {
                        auto collected =
                            collect_tensor_buffers(tensor_args, tensor_return_value, sv.workload_descriptor);
                        tt::tt_metal::apply_resolved_bindings(program, sv.resolved_bindings, collected.buffers);
                        tt::tt_metal::apply_dynamic_runtime_args(program, dynamic_args);
```

**Consequence for this audit.** Exactly three things are refreshed on a hit: the `config` buffer
address (via the `Buffer*` binding), and common runtime args `[3]` and `[4]` (the two metadata
tensor addresses). Everything else — common args `[0..2]` (`my_sp_coord`, `sp_factor`,
`tokens_per_chip`), the whole compile-time arg vector, both CB page sizes, and the single-core
`CoreRangeSet` — is frozen at the first miss and must be a pure function of the hashed set.

**In-place aliasing.** `config` is simultaneously a `tensor_args_t` member and the
`tensor_return_value_t` (`compute_output_specs` and `create_output_tensors` both return it,
`device/moe_padding_config_device_operation.cpp:102-112`), so its `Buffer*` appears once in the
input region and once in the output region of `collect_tensor_buffers`. That is the *safe* alias
case, not the `matmul(X, X)` bail case, so `resolve_bindings` does not return empty and
`allow_inplace_output_tensor_alias` is not needed:

```109:114:tt_metal/api/tt-metalium/experimental/program_descriptor_patching.hpp
//   - the SAME buffer appearing twice WITHIN the inputs (e.g. matmul(X, X)) is ambiguous —
//     a future call with distinct same-shape tensors would miscompute — so we bail to the
//     slow path.
//   - an OUTPUT buffer (from the output/workload region) that aliases an INPUT buffer (an
//     in-place op writing back into its input) is safe: every binding for that buffer resolves
//     to the one shared address, correct on every dispatch — so we keep the fast path.
```

**Which validator runs on a cache hit.** Several verdicts below rest on a `TT_FATAL` rather than on
the hash, so it matters exactly which validator executes on the offending second call. The
dispatcher runs one, not both:

```262:266:ttnn/api/ttnn/device_operation.hpp
    if constexpr (HasValidateOnProgramCacheHit<mesh_device_operation_t>) {
        mesh_device_operation_t::validate_on_program_cache_hit(operation_attributes, tensor_args);
    } else {
        mesh_device_operation_t::validate_on_program_cache_miss(operation_attributes, tensor_args);
    }
```

This op **defines** `validate_on_program_cache_hit`, so it takes the first branch and the hit
validator *replaces* the miss validator on every hit. That is a hazard in general — by existing, a
narrow hit validator silently disables every check the miss validator performs — but here the two
are wrappers around one shared checker, so the hit path drops nothing:

```92:100:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_padding_config/device/moe_padding_config_device_operation.cpp
void MoePaddingConfigDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_runtime_args(args, tensor_args);
}

void MoePaddingConfigDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_runtime_args(args, tensor_args);
}
```

Every `TT_FATAL` in `validate_runtime_args`
(`device/moe_padding_config_device_operation.cpp:42-83`) therefore runs on hits as well as misses,
which is what licenses the "VALID — pinned by validation" verdicts in items 4 and 8 and what would
make the recommended `buffer_type` guard an adequate fix for the bug in item 3.

## Baseline: what the default hash would cover

`hash_objects_with_default_seed(type_hash<MoePaddingConfigDeviceOperation>, attrs, tensor_args)`
would walk reflection over both structs:

| Source | Fields |
|---|---|
| `operation_attributes` | `tokens_per_chip`, `pad_side`, `cluster_axis` |
| `config` | storage kind; `logical_shape`; `dtype`; `page_config`; `memory_config`; `alignment` |
| `actual_start` | storage kind; `logical_shape`; `dtype`; `page_config`; `memory_config`; `alignment` |
| `actual_end` | storage kind; `logical_shape`; `dtype`; `page_config`; `memory_config`; `alignment` |

The mesh coordinates are appended by the framework for both the default and the custom path
(`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:989-992`), so they are never an omission.

## What the custom hash covers

```114:128:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_padding_config/device/moe_padding_config_device_operation.cpp
ttsl::hash::hash_t MoePaddingConfigDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // The per-chunk values are NEVER hashed: they are read on-device from the metadata tensors, whose
    // raw addresses live in common runtime args refreshed by override_runtime_arguments. That is the
    // whole point — one cached program serves every chunk, so it can be captured once and replayed.
    const auto& config = tensor_args.config;
    return tt::tt_metal::operation::hash_operation<MoePaddingConfigDeviceOperation>(
        args.tokens_per_chip,
        args.pad_side,
        args.cluster_axis,
        config.dtype(),
        config.layout(),
        config.memory_config(),
        config.padded_shape());
}
```

All three `operation_attributes` are kept. `config` is decomposed selectively. **`actual_start` and
`actual_end` contribute nothing at all** — not their shapes, not their dtypes, not their memory
configs.

## Omitted parameters

### 1. The per-chunk position values inside `actual_start` / `actual_end`

**Verdict: VALID — invariant.** This is the design, and it is the correct one.

These are the moving index this op exists to consume: the absolute KV position of a chunk's first
real token and one past its last. They advance every chunk. They are deliberately *not* host-side
scalars — they live in two 1-element `uint32` device tensors, and the kernel reads element `[0]` of
each at dispatch time:

```73:81:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_padding_config/device/kernels/dataflow/writer_moe_padding_config.cpp
    const auto s_start = TensorAccessor(meta_args, actual_start_addr);
    noc.async_read(s_start, cb_meta, kMetadataReadBytes, {.page_id = 0}, {.offset_bytes = 0});
    noc.async_read_barrier();
    // The metadata tensors sit at FIXED DRAM addresses reused every chunk, so the RISC data cache may
    // still hold the previous chunk's value for this L1 line (the barrier orders the DMA; volatile
    // still reads through the cache). Force a refetch, else a stale read silently produces the prior
    // chunk's config.
    invalidate_l1_cache();
    const uint32_t actual_start = CoreLocalMem<volatile uint32_t>(cb_meta.get_write_ptr())[0];
```

Because the value never enters a runtime arg or a compile-time arg, there is no stale slot to
patch. The whole rotation computation — `boundary_slab`, `boundary_chip`, `boundary_offset`,
`local_real_tokens` — is done on device from those two reads
(`device/kernels/dataflow/writer_moe_padding_config.cpp:98-119`). One cached program is correct for
every chunk, which is exactly what makes the op trace-capturable. The unit test asserts this
directly: four chunks with different `(actual_start, actual_isl)` must produce exactly one program
cache entry (`models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_moe_padding_config.py:153-194`).

The default hash would not have caught this either — the *values* inside a device tensor are not
reflected. There is no correctness exposure here at all; the value was never hashable.

### 2. `actual_start.buffer()->address()` and `actual_end.buffer()->address()`

**Verdict: VALID — patched.**

These are raw `uint32_t` scalars smuggled into common runtime args at indices 3 and 4:

```194:201:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_padding_config/device/moe_padding_config_device_operation.cpp
    writer_kernel.emplace_common_runtime_args({
        my_sp_coord,
        sp_factor,
        args.tokens_per_chip,
        tensor_args.actual_start.buffer()->address(),  // smuggled-rta-ok: 1-element metadata tensor DRAM
                                                       // addr; read on-device (trace-safe, unhashed)
        tensor_args.actual_end.buffer()->address(),    // smuggled-rta-ok: as above
    });
```

The buffer-binding fast path would not touch them — they are values, not declared bindings, and
`resolve_bindings` "does not infer addresses by scanning arg values". That is precisely why the op
carries its own `override_runtime_arguments`, and it re-applies both every hit
(`device/moe_padding_config_device_operation.cpp:238-239`, quoted above). The slot indices are
named constants shared between the create path and the override path
(`device/moe_padding_config_device_operation.cpp:35-36`), so the two cannot drift, and the override
guards the write with a `TT_FATAL` on the arg-vector length.

This is the one non-address... strictly, it *is* an address, but delivered as a scalar rather than a
binding, and it is the classic incomplete-override trap. Here the override is complete for it.

### 3. `actual_start` / `actual_end` `memory_config` (specifically `buffer_type`)

**Verdict: BUG.**

The `buffer_type` of the metadata tensors reaches a compile-time arg, is refreshed on no path, is
absent from the hash, and — the deciding point — the bad configuration is reachable through the
public API without violating any enforced constraint. `validate_meta` checks storage type,
allocation, dtype, layout, element count, shardedness and device — and not `buffer_type`.

The metadata tensors' accessor is a *compile-time* arg vector, built from `actual_start`'s buffer:

```179:183:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_padding_config/device/moe_padding_config_device_operation.cpp
    // Compile args: [0]=cb_out, [1]=cb_meta, [2]=pad_side, [3..]=config accessor, then ONE metadata
    // accessor (both 1-element tensors share an identical layout, so one accessor serves both reads).
    KernelDescriptor::CompileTimeArgs writer_compile_args = {kOutCbIndex, kMetaCbIndex, args.pad_side};
    TensorAccessorArgs(config.buffer()).append_to(writer_compile_args);
    TensorAccessorArgs(tensor_args.actual_start.buffer()).append_to(writer_compile_args);
```

For a non-sharded buffer `TensorAccessorArgs::append_to` emits two words — the raw `ArgsConfig`
bitset (which carries the `IsDram` bit) and `aligned_page_size`:

```194:198:tt_metal/impl/buffers/tensor_accessor_args.cpp
    if (args_config_.test(tensor_accessor::ArgConfig::Sharded)) {
        CMAKE_UNIQUE_NAMESPACE::append_sharded_args(*buffer_, args_config_, compile_time_args, /* is_runtime */ false);
    } else {
        compile_time_args.push_back(args_config_.raw());
        auto aligned_page_size = buffer_ ? buffer_->aligned_page_size() : 0;
```

Each `append_to` on a non-sharded buffer therefore contributes exactly two words, so the writer's
compile-time vector is `[0]=cb_out`, `[1]=cb_meta`, `[2]=pad_side`, `[3..4]`=the `config` accessor,
`[5..6]`=the metadata accessor — where **`[5]` is the `ArgsConfig` bitset carrying the `IsDram`
bit** and `[6]` is the metadata buffer's `aligned_page_size`. The kernel binds them at exactly those
offsets:

```57:58:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_padding_config/device/kernels/dataflow/writer_moe_padding_config.cpp
    constexpr auto config_args = TensorAccessorArgs<3>();
    constexpr auto meta_args = TensorAccessorArgs<config_args.next_compile_time_args_offset()>();
```

`validate_runtime_args` pins the metadata tensors' storage type, dtype, layout, element count,
shardedness and device, but **not** their `buffer_type`
(`device/moe_padding_config_device_operation.cpp:66-80`). The hash records nothing at all from these
two tensors — not even their presence, since they are non-optional
(`device/moe_padding_config_device_operation.cpp:114-128`). Compile-time args are baked into the
cached `Program` and are refreshed by nothing on any path, in any of the three cache-hit modes.

**Two-call reproduction.**

- **Call 1.** `ttnn.moe_padding_config(config, actual_start, actual_end, tokens_per_chip=T,
  pad_side=P, cluster_axis=A)` with `actual_start` and `actual_end` built with
  `memory_config=ttnn.DRAM_MEMORY_CONFIG`. Cache miss; the program is built and cached. Writer
  compile-time arg `[5]` has the `IsDram` bit set, `[6]` is the DRAM aligned page size.
- **Call 2.** The same `config` tensor and the same `T` / `P` / `A`, but with `actual_start` and
  `actual_end` allocated with `memory_config=ttnn.L1_MEMORY_CONFIG`. They are still device tensors,
  still `uint32`, still ROW_MAJOR, still one element, still unsharded, still on the same device, so
  `validate_meta` passes on both the miss and hit paths. Nothing else about the call differs.
- **Hash outcome.** The custom hash reads only `args.tokens_per_chip`, `args.pad_side`,
  `args.cluster_axis` and four fields of `config`, so the two calls hash identically. Call 2 is a
  cache hit.
- **Stale slot.** Writer compile-time arg `[5]`, the metadata `ArgsConfig` bitset. It still says
  `IsDram` while `actual_start`/`actual_end` now live in L1. Compile-time args are baked into the
  cached `Program`; `override_runtime_arguments` refreshes common runtime args `[3]` and `[4]` (the
  addresses) and nothing else, so the mismatch survives.
- **Observable symptom.** `TensorAccessor(meta_args, actual_start_addr)`
  (`device/kernels/dataflow/writer_moe_padding_config.cpp:73, 83`) resolves the freshly patched L1
  address through the DRAM bank table. The NoC read lands on an unrelated DRAM location, so
  `actual_start` and `actual_end` are garbage and the op silently writes a wrong
  `local_real_tokens` into the padding config. Downstream `moe_grouped_topk` and dispatch consume it
  as valid, so the failure surfaces as wrong MoE routing rather than as a crash — silent data
  corruption with no cache miss to hint at the cause. The stale `aligned_page_size` in `[6]` is
  *not* part of the symptom: the kernel only ever reads `page_id = 0`
  (`device/kernels/dataflow/writer_moe_padding_config.cpp:74, 84`), whose offset within its bank is
  zero regardless of the page size. The `IsDram` bit alone carries the fault.

**Severity and likelihood.** This is latent in every current caller, but being latent is not a
defence.
The documented API contract does pin DRAM
(`moe_padding_config_nanobind.cpp:44-48` — "1-element uint32 DRAM tensor"), and every in-tree
producer of these tensors complies. The production path allocates them in
`TtPrefillRuntime._meta1_dev`:

```329:338:models/demos/deepseek_v3_d_p/tt/tt_prefill_runtime.py
    def _meta1_dev(self, val: int) -> ttnn.Tensor:
        """One persistent 1-element uint32 replicated-DRAM metadata scalar (captured address)."""
        return ttnn.from_torch(
            torch.tensor([val], dtype=torch.int64).reshape(1, 1, 1, 1),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
```

and the unit test's `_meta1` helper does the same
(`models/demos/deepseek_v3_d_p/tests/op_unit_tests/test_moe_padding_config.py:60-68`).

But a docstring is not enforcement. Nothing rejects an L1 metadata tensor at any layer, and the op
is bound to Python with `noconvert()` tensor args
(`moe_padding_config_nanobind.cpp:59-61`), so a caller passing `ttnn.L1_MEMORY_CONFIG` reaches the
kernel unmodified rather than being silently converted. The bad configuration is therefore reachable
through the public API without violating an enforced constraint, which is what makes this a BUG and
not a caveat. The narrow in-tree producer set means the likelihood of hitting it today is low; it
says nothing about whether the defect exists.

The guard that closes it: one `TT_FATAL` inside `validate_meta` requiring
`meta.memory_config().buffer_type() == BufferType::DRAM`. Because
`validate_on_program_cache_hit` re-runs the same checker
(`device/moe_padding_config_device_operation.cpp:97-100`), that guard would run on the offending
second call and not merely on the first — which is what would make validation a legitimate
substitute for hashing here. Placement matters *for this op specifically*: because it defines a hit
validator, that validator replaces the miss validator on hits (see "Which validator runs on a cache
hit" above), so a guard added only to `validate_on_program_cache_miss` would not fix this. Adding it
to the shared `validate_runtime_args` checker, as recommendation 1 proposes, puts it on both paths.

This defect is not unique to this op. The sibling `update_padded_kv_cache` has the materially
identical structure — its metadata tensor's `buffer_type` reaches a writer `TensorAccessorArgs`
compile-time arg, its hash records only the tensor's presence, and its `validate_meta` pins storage
type, dtype, layout, element count, shardedness and device but not `buffer_type` — and that op's
audit grades it a BUG. The fix belongs at the family level (see recommendation 1).

### 4. `actual_start` / `actual_end` dtype, layout, shape, shardedness, storage kind

**Verdict: VALID — pinned by validation.**

```66:80:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_padding_config/device/moe_padding_config_device_operation.cpp
    auto validate_meta = [&config](const Tensor& meta, const char* name) {
        TT_FATAL(meta.storage_type() == StorageType::DEVICE, "metadata tensor {} must be on device", name);
        TT_FATAL(meta.buffer() != nullptr, "metadata tensor {} must be allocated", name);
        TT_FATAL(meta.dtype() == DataType::UINT32, "metadata tensor {} must be UINT32", name);
        TT_FATAL(meta.layout() == Layout::ROW_MAJOR, "metadata tensor {} must be ROW_MAJOR", name);
        TT_FATAL(
            meta.logical_volume() == 1,
            "metadata tensor {} must be a single element (got {})",
            name,
            meta.logical_volume());
        TT_FATAL(!meta.is_sharded(), "metadata tensor {} must not be sharded", name);
        // The kernel resolves meta.buffer()->address() against config.device(); a tensor on a
        // different mesh device would bake the wrong address and fail obscurely on device.
        TT_FATAL(meta.device() == config.device(), "metadata tensor {} must be on the same device as config", name);
    };
```

Each of these carries no information across admissible calls. Crucially, this checker is invoked
from *both* validators, so the pinning survives cache hits:

```97:100:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_padding_config/device/moe_padding_config_device_operation.cpp
void MoePaddingConfigDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_runtime_args(args, tensor_args);
}
```

Note `logical_volume() == 1` pins the *element count* but not the rank or the padded row width. That
residual does not matter: for a non-sharded accessor only `aligned_page_size` varies with it, and the
kernel reads only page 0 at offset 0.

### 5. `config.logical_shape()` — `padded_shape()` is hashed instead

**Verdict: VALID — relaxation win.**

The factory never reads the config's logical shape. The only shape-derived quantity it consumes is
the page size:

```153:155:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_padding_config/device/moe_padding_config_device_operation.cpp
    // Write the config row a full page at a time (the row may be padded up to the buffer's aligned
    // page size); the kernel zeroes the slot first so the padding bytes are deterministic.
    const uint32_t out_page_size = config.buffer()->aligned_page_size();
```

`padded_shape` is exactly the projection the buffer's page size is computed from, so hashing it
instead of `logical_shape` is both sufficient and slightly more permissive. Since validation only
requires `config.logical_shape()[-1] >= 2`
(`device/moe_padding_config_device_operation.cpp:59-62`), two callers with different logical widths
that pad to the same physical row legitimately share one program instead of forcing a recompile.

### 6. `config.tensor_spec().page_config()` beyond `layout()`

**Verdict: VALID — unused.**

`layout()` collapses `PageConfig` to `ROW_MAJOR` vs `TILE`, discarding the `Tile` shape and its
face/transpose configuration. Validation pins `config.layout() == Layout::ROW_MAJOR`
(`device/moe_padding_config_device_operation.cpp:58`), and a row-major tensor's page size does not
consult the tile at all. Neither the factory nor the kernel references
`config.tensor_spec().tile()`. Two calls differing only in the (unused) tile descriptor produce a
byte-identical descriptor. Item 11 records the search that backs that claim.

### 7. `config.tensor_layout().get_alignment()`

**Verdict: VALID — unused.**

Unlike the two `per_token_cast_*` siblings, this op has no residual alignment exposure, because it
hashes `padded_shape()` rather than `logical_shape()`. `Buffer::aligned_page_size()` is

```656:658:tt_metal/impl/buffers/buffer.cpp
uint32_t Buffer::alignment() const { return allocator_->get_alignment(this->buffer_type()); }

DeviceAddr Buffer::aligned_page_size() const { return align(page_size(), this->alignment()); }
```

that is, `align(page_size, allocator_alignment(buffer_type))`. `page_size` is a function of
`padded_shape` and `dtype` (both hashed), and the allocator alignment is a device constant selected
by `buffer_type`, which lives inside the hashed `memory_config`. The `TensorLayout::Alignment`
influences the result only *through* `padded_shape`, which is hashed directly. So the omission is
total, not partial.

### 8. `config.storage` variant kind (device vs host)

**Verdict: VALID — pinned by validation.**

```55:56:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_padding_config/device/moe_padding_config_device_operation.cpp
    TT_FATAL(config.storage_type() == StorageType::DEVICE, "config must be on device");
    TT_FATAL(config.buffer() != nullptr, "config must be allocated");
```

### 9. `config.buffer()->address()`

**Verdict: VALID — patched, and required.**

The factory deliberately passes the buffer as a binding rather than as a raw address, so the inner
fast path patches it:

```203:205:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_padding_config/device/moe_padding_config_device_operation.cpp
    // The config buffer is passed as a Buffer* binding (not a raw address) so cache hits take the fast
    // path that patches its address and skips create_descriptor.
    writer_kernel.emplace_runtime_args(core, {config.buffer()});
```

### 10. `my_sp_coord` and `sp_factor` — common args `[0]` and `[1]`, set only at create time

**Verdict: VALID — invariant.**

These are the classic "set in the create path, never re-set in the override" args, so they deserve
explicit proof rather than assumption.

```144:147:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_padding_config/device/moe_padding_config_device_operation.cpp
    const auto& mesh_view = device->get_view();
    const uint32_t sp_factor = (args.cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();
    const uint32_t my_sp_coord =
        ::ttnn::ccl::get_linearized_index_from_physical_coord(config, coord, args.cluster_axis);
```

- `sp_factor` is `num_rows()`/`num_cols()` of the mesh view, selected by the hashed `cluster_axis`.
  The mesh shape is fixed for the device the cache belongs to.
- `my_sp_coord`: because validation forces `cluster_axis ∈ {0, 1}`
  (`device/moe_padding_config_device_operation.cpp:45`), the `cluster_axis.has_value()` branch of
  `get_linearized_index_from_physical_coord` is always taken, and it returns
  `physical_coord[cluster_axis]` — nothing else (`ttnn/cpp/ttnn/operations/ccl/ccl_common.cpp:195-209`).
  The coordinate is per-program (one program per `MeshCoordinateRange`) and the framework appends
  the tensor coordinate set to the hash for custom-hash ops too.

`tokens_per_chip` (common arg `[2]`) and `pad_side` (compile-time arg `[2]`) are hashed directly.

### 11. Tile geometry — the tile-awareness check

**Verdict: VALID — unused.** This check was performed rather than skipped: the op does no host-side
tile math in any form, so neither the hardcoded-32x32 hazard nor its mirror image applies. Item 6
records the `page_config` omission; this subsection records the search behind it.

Neither the device operation (which holds `create_descriptor`), nor the writer kernel, calls
`tt::tile_size(...)`, `tensor_spec().tile()`, `get_tile_shape()` or `get_face_shape()`, and neither
uses `tt::constants::TILE_HW` / `TILE_WIDTH` / `TILE_HEIGHT` to convert a shape into a tile count.
`device/moe_padding_config_device_operation.cpp:20` does pull the constants into scope with
`using namespace tt::constants;`, but nothing in the file reads one — the include and the
using-directive are vestigial.

The program has exactly one shape-derived size, and it is a page size rather than a tile count:

```153:167:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_padding_config/device/moe_padding_config_device_operation.cpp
    // Write the config row a full page at a time (the row may be padded up to the buffer's aligned
    // page size); the kernel zeroes the slot first so the padding bytes are deterministic.
    const uint32_t out_page_size = config.buffer()->aligned_page_size();

    tt::tt_metal::ProgramDescriptor desc;

    desc.cbs.push_back(CBDescriptor{
        .total_size = out_page_size,
        .core_ranges = single_core,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = kOutCbIndex,
            .data_format = tt::DataFormat::UInt32,
            .page_size = out_page_size,
        }}},
    });
```

That `aligned_page_size` cannot smuggle in tile geometry, because `page_size` is tile-derived only on
the `TilePageConfig` branch:

```128:136:tt_metal/impl/tensor/spec/layout/page_config.cpp
size_t get_page_size_bytes_tile(const TilePageConfig& config, const Shape2D& page_shape, DataType dtype) {
    const auto tiles_count =
        page_shape.height() / config.tile.get_height() * page_shape.width() / config.tile.get_width();
    return tiles_count * config.tile.get_tile_size(datatype_to_dataformat_converter(dtype));
}

size_t get_page_size_bytes_rm(const RowMajorPageConfig&, const Shape2D& page_shape, DataType dtype) {
    return page_shape.height() * page_shape.width() * rm_element_size_bytes(dtype);
}
```

and `config` is pinned to ROW_MAJOR (`device/moe_padding_config_device_operation.cpp:58`), as are
both metadata tensors (`70`). The second CB is a fixed `kMetadataBytes`
(`device/moe_padding_config_device_operation.cpp:169-177`), and the core grid is the literal single
core `{0, 0}` (`150-151`) — not a tile-derived work split. The kernel reads its page size back from
the CB interface at runtime rather than recomputing it
(`device/kernels/dataflow/writer_moe_padding_config.cpp:123`).

## Keys the custom hash adds beyond the default

- `config.padded_shape()` — not in the default key (the default hashes `logical_shape` and lets
  `padded_shape` be a derived value). Adding it is what makes dropping `logical_shape` and
  `alignment` safe simultaneously.

## Framework side effect of having a custom hash

```1012:1014:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
        if constexpr (requires { DeviceOperation::compute_program_hash(attrs, tensor_args); }) {
            return key;  // custom hash -> opt out beyond the op-identity prefix
        } else {
```

Defining `compute_program_hash` degrades `ProgramCacheKey::canonical` to just the op type name, so
a 64-bit hash collision between two different `moe_padding_config` configurations resolves to a
wrong hit rather than a rebuild. That is inherent to every custom-hash op, but it raises the cost of
the BUG in item 3.

## Summary

| Omitted vs. default | Used by program? | Patched on hit? | Verdict |
|---|---|---|---|
| `actual_start`/`actual_end` element values (the per-chunk position) | Yes, read on device | n/a — never enters an arg | VALID — invariant |
| `actual_start`/`actual_end` buffer addresses | Yes (common args 3, 4) | Yes (op's own override) | VALID — patched |
| `actual_start`/`actual_end` `memory_config` (`buffer_type`) | Yes (writer compile-time arg `[5]`, the `IsDram` bit) | No | **BUG** |
| `actual_start`/`actual_end` dtype, layout, volume, shardedness, storage | Only via pinned values | n/a | VALID — pinned by validation (both paths) |
| `config.logical_shape` | No (`padded_shape` used) | n/a | VALID — relaxation win |
| `config.page_config` beyond `layout()` | No | n/a | VALID — unused |
| `config.tensor_layout.alignment` | Only via hashed `padded_shape` | n/a | VALID — unused |
| `config.storage` kind | n/a | n/a | VALID — pinned by validation (both paths) |
| `config` buffer address | Yes | Yes (`Buffer*` binding) | VALID — patched |
| `my_sp_coord`, `sp_factor` (create-only common args) | Yes | No | VALID — invariant |
| Tile geometry (no host-side tile math) | No — op has none | n/a | VALID — unused |

**One program-cache correctness bug found: item 3, the metadata tensors' `buffer_type`.** It reaches
the writer's compile-time arg `[5]` as the accessor's `IsDram` bit, it is refreshed on no cache-hit
path, it is absent from the hash, and an L1-allocated `actual_start`/`actual_end` passes every
`TT_FATAL` the op has. Two calls that differ only in that placement share a cache entry, and the
second one resolves an L1 address through the DRAM bank table, silently producing a wrong
`local_real_tokens`. It is latent in every current caller — the documented contract says DRAM and
all in-tree producers comply — but nothing enforces it, so the configuration is reachable through
the public API and the defect is real rather than hypothetical.

Every other frozen slot is clean. Both CB page sizes, the single-core range, common args `[0..2]`,
and the remaining compile-time args are pure functions of
{`tokens_per_chip`, `pad_side`, `cluster_axis`, `config.dtype`, `config.layout`,
`config.memory_config`, `config.padded_shape`} plus the mesh coordinate the framework appends and
device-fixed constants.

The tile check (item 11) was performed and found nothing to adjudicate: the op does no host-side tile
math, so it can neither bake a 32x32 assumption into a program for a non-32x32 tensor nor vary its
program with a tile that is absent from the key. The only shape-derived size in the descriptor is a
row-major page size, which does not consult the `Tile`.

The headline design point is worth stating plainly: the per-chunk position, which is exactly the
kind of value that goes stale in a cached KV-cache-adjacent op, is not merely omitted-and-patched
here — it never reaches the host dispatch path at all. That is a stronger guarantee than patching,
and it is what makes the op trace-safe.

## Recommendations

1. **Fix the bug in item 3.** Add
   `TT_FATAL(meta.memory_config().buffer_type() == BufferType::DRAM, ...)` to `validate_meta`
   (`device/moe_padding_config_device_operation.cpp:66-80`). Because both validators are wrappers
   around that same `validate_runtime_args` checker
   (`device/moe_padding_config_device_operation.cpp:92-100`), the guard runs on the offending second
   call and not merely on the first, which is what makes validation a sound substitute for hashing
   here. The alternative — folding `actual_start.memory_config()` into `compute_program_hash` — also
   works but costs a cache entry per placement for no benefit, since the op only supports DRAM
   anyway.

   Treat this as a family-wide fix rather than a local one. The sibling `update_padded_kv_cache` has
   the materially identical defect (metadata `buffer_type` reaching a writer `TensorAccessorArgs`
   compile-time arg, hashed only for presence, and a `validate_meta` that pins storage type, dtype,
   layout, element count, shardedness and device but not `buffer_type`), and its audit recommends the
   same `TT_FATAL`. Landing both in one change keeps the two ops from drifting apart again.
2. Independently of the cache, require the two metadata tensors to share a memory config. Only
   *one* accessor is built, from `actual_start`, and it is reused for both reads
   (`device/moe_padding_config_device_operation.cpp:183`, and `meta_args` in the kernel at
   `device/kernels/dataflow/writer_moe_padding_config.cpp:58`). A mismatched pair is wrong on the
   very first call, not just on a cache hit.
3. Run this op's unit tests under `-DTT_DESCRIPTOR_PATCHING_PARITY_CHECK`. The op takes the
   descriptor fast path internally, so `assert_fastpath_parity` will diff the patched program
   against a full rebuild and catch any future common-runtime-arg added to `create_descriptor`
   without a matching line in `override_runtime_arguments`
   (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:732-747`). The existing cache-entry-count
   assertion in `test_moe_padding_config.py` is a good complement but only proves reuse, not that
   reuse is correct.
