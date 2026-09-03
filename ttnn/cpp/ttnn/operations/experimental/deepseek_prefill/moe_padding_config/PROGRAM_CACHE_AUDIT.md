# Program Cache Audit — `experimental/deepseek_prefill/moe_padding_config`

Audit of `MoePaddingConfigDeviceOperation::compute_program_hash` against the framework default
("hash everything") key.

| | |
|---|---|
| Device operation | `ttnn::operations::experimental::deepseek_prefill::moe_padding_config::MoePaddingConfigDeviceOperation` (`device/moe_padding_config_device_operation.hpp:21`) |
| Custom hash | `device/moe_padding_config_device_operation.cpp:123-140` |
| `operation_attributes_t` | `tokens_per_chip`, `pad_side`, `cluster_axis` (all `uint32_t`) |
| `tensor_args_t` | `config` (in-place output), `actual_start`, `actual_end` |
| Program factories | one: `ProgramFactory::create_descriptor` (`ProgramDescriptor`-based), wrapped by `MeshWorkloadFactory` |
| `override_runtime_arguments` | **Yes**, at the mesh-workload level (`device/moe_padding_config_device_operation.cpp:232-253`) |
| `get_dynamic_runtime_args` | **No** |
| Cache-hit patch mechanism | **Op-owned workload-level override**, which internally delegates to the descriptor adapter's **buffer-binding fast path** and then hand-patches two common runtime args |

## Post-fix status — commit fab067a

**Verdict: CLEAR — no relaxations.** The single program-cache bug this document recorded (item 3, the
metadata tensors' `buffer_type` reaching the writer's compile-time accessor args unhashed) is closed,
and every remaining omission is category 3: either pinned by a `TT_FATAL` that runs on the hit path,
or functionally determined by a hashed term. Nothing the key drops can vary in a way that reaches the
compiled program.

What `fab067a` changed in this op:

- Added `tensor_args.actual_start.memory_config()` to `compute_program_hash`
  (`device/moe_padding_config_device_operation.cpp:139`), which keys the `IsDram` bit and
  `aligned_page_size` that the single metadata `TensorAccessorArgs` bakes into writer compile-time args
  `[5..6]`.
- Added `TT_FATAL(actual_start.memory_config() == actual_end.memory_config(), ...)` to the shared
  `validate_runtime_args` checker (`device/moe_padding_config_device_operation.cpp:83-91`). Placement
  matters here more than for most ops: this op **defines** `validate_on_program_cache_hit`
  (`:106-109`), which *replaces* the miss validator on hits, so a guard added only to the miss
  validator would not run on the offending second call. Both validators are thin wrappers around
  `validate_runtime_args` (`:101-109`), so this guard runs on both paths.
- Together those two changes are a stronger fix than this document recommended. Recommendation 1
  proposed a DRAM-only `TT_FATAL`; what landed instead *hashes* the placement and pins the two
  metadata tensors to agree with each other. That keeps L1 metadata tensors admissible (they simply
  key a different program) while closing the one-accessor-for-two-tensors hazard that recommendation 2
  raised separately. Both recommendations are satisfied by the pair.

What remains open:

- **Nothing on the cache axis.** Recommendation 3 (a run under
  `-DTT_DESCRIPTOR_PATCHING_PARITY_CHECK`, which would catch a future common runtime arg added to
  `create_descriptor` without a matching line in `override_runtime_arguments`) was not done and remains
  a worthwhile regression net.
- The op's documented API contract still says DRAM
  (`moe_padding_config_nanobind.cpp:44-48`) while the code now accepts either placement and keys it.
  That is a docstring/behaviour mismatch, not a defect.

**Metal 2.0 port: clear, and it needs no `TensorSpecRelaxations` flag.** The key drops
`config.logical_shape` in favour of `config.padded_shape()`, which would ordinarily map onto
`match_padded_shape_only`
(`tt_metal/api/tt-metalium/experimental/metal2_host_api/tensor_spec_relaxations.hpp:41,49`;
`pertinent_fields` → `PertinentFields{.padded_shape = true}` at
`tt_metal/impl/metal2_host_api/tensor_spec_relaxations.cpp:77-79`). Here that relaxation is
**vacuous**: `config` is pinned to `Layout::ROW_MAJOR` and non-sharded
(`device/moe_padding_config_device_operation.cpp:58`, `:63`), and a row-major interleaved tensor's
default `Alignment` is `{1}` (`tt_metal/impl/tensor/spec/layout/page_config.cpp:43-57`), so
`padded_shape ≡ logical_shape` for every admissible input a caller would normally build. The two keys
then define the same equivalence class, and a default-constructed `TensorSpecRelaxations` — exact
match — reproduces this document's key as-is; no flag needs to be declared. The relaxation only
becomes observable if a caller hands in a row-major tensor with an explicit non-default `Alignment`,
and even then it stays sound because the only shape-derived quantity the program uses is the padded
row's page size (item 5). If that case is ever wanted, `match_padded_shape_only` is the exact
expression of it. The same reasoning covers the metadata tensors, which are pinned to ROW_MAJOR,
unsharded and one-element (`:66-80`).

## Cache-hit patch mechanism

This op is the only one of the four `deepseek_prefill` ops audited here that uses a
`ProgramDescriptor` factory, and its cache-hit path is a hybrid worth spelling out precisely.

The framework's cache-hit dispatcher prefers a workload factory's `apply_descriptor` and otherwise
calls its `override_runtime_arguments`:

```282:288:ttnn/api/ttnn/device_operation.hpp
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

```232:253:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_padding_config/device/moe_padding_config_device_operation.cpp
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
Post-`fab067a` it is: the one term that was not, the metadata accessor's placement in compile-time args
`[5..6]`, is now keyed (`device/moe_padding_config_device_operation.cpp:139`).

**In-place aliasing.** `config` is simultaneously a `tensor_args_t` member and the
`tensor_return_value_t` (`compute_output_specs` and `create_output_tensors` both return it,
`device/moe_padding_config_device_operation.cpp:111-121`), so its `Buffer*` appears once in the
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

```265:269:ttnn/api/ttnn/device_operation.hpp
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

```101:109:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_padding_config/device/moe_padding_config_device_operation.cpp
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
(`device/moe_padding_config_device_operation.cpp:42-92`) therefore runs on hits as well as misses,
which is what licenses the "VALID — pinned by validation" verdicts in items 4 and 8, and what makes
`fab067a`'s new metadata memory-config equality check (`:83-91`) effective on the offending second
call rather than only on the first. Had that guard been added to `validate_on_program_cache_miss`
instead of to the shared checker, it would have been inert on hits.

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

```123:140:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_padding_config/device/moe_padding_config_device_operation.cpp
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
        config.padded_shape(),
        // The metadata accessor is baked into the writer's compile-time args, so the bank table it
        // selects cannot be refreshed on a cache hit; validate_runtime_args pins actual_end to match.
        tensor_args.actual_start.memory_config());
}
```

All three `operation_attributes` are kept. `config` is decomposed selectively. From the metadata
tensors, only `actual_start.memory_config()` survives — `fab067a`'s addition, and it stands in for
`actual_end`'s as well because the two are now required to be equal
(`device/moe_padding_config_device_operation.cpp:83-91`). Pre-fix **`actual_start` and `actual_end`
contributed nothing at all** — not their shapes, not their dtypes, not their memory configs — which is
what item 3 was about. Their shapes and dtypes are still absent, and correctly so (item 4).

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

```206:213:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_padding_config/device/moe_padding_config_device_operation.cpp
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
(`device/moe_padding_config_device_operation.cpp:250-251`, quoted above). The slot indices are
named constants shared between the create path and the override path
(`device/moe_padding_config_device_operation.cpp:35-36`), so the two cannot drift, and the override
guards the write with a `TT_FATAL` on the arg-vector length.

This is the one non-address... strictly, it *is* an address, but delivered as a scalar rather than a
binding, and it is the classic incomplete-override trap. Here the override is complete for it.

### 3. `actual_start` / `actual_end` `memory_config` (specifically `buffer_type`)

**Verdict: RESOLVED by fab067a** (was BUG).

`fab067a` closed this by hashing `tensor_args.actual_start.memory_config()`
(`device/moe_padding_config_device_operation.cpp:139`) — which is the second of the two routes this
document offered, not the `TT_FATAL` it recommended. That choice is the better one on inspection: it
keys the placement instead of forbidding it, so an L1 metadata tensor now compiles its own program
rather than being rejected, and the `IsDram` bit in compile-time arg `[5]` is a function of a hashed
term. The commit also added the guard that makes hashing `actual_start` alone sufficient:
`TT_FATAL(actual_start.memory_config() == actual_end.memory_config(), ...)` in the shared
`validate_runtime_args` (`:83-91`), which — because both validators wrap that checker (`:101-109`) —
runs on hits as well as misses. Without it, hashing only `actual_start` would have left `actual_end`'s
placement unkeyed while one accessor served both reads; that was the separate defect recorded as
recommendation 2, and it is now closed by the same line.

Pre-fix: the `buffer_type` of the metadata tensors reaches a compile-time arg, is refreshed on no path,
is absent from the hash, and — the deciding point — the bad configuration was reachable through the
public API without violating any enforced constraint. `validate_meta` checks storage type,
allocation, dtype, layout, element count, shardedness and device — and not `buffer_type`.

The metadata tensors' accessor is a *compile-time* arg vector, built from `actual_start`'s buffer:

```191:195:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_padding_config/device/moe_padding_config_device_operation.cpp
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

Pre-fix, `validate_runtime_args` pinned the metadata tensors' storage type, dtype, layout, element
count, shardedness and device, but **not** their `buffer_type`
(`device/moe_padding_config_device_operation.cpp:66-80`), and the hash recorded nothing at all from
these two tensors — not even their presence, since they are non-optional. Compile-time args are baked
into the cached `Program` and are refreshed by nothing on any path, in any of the three cache-hit
modes.

**Pre-fix two-call reproduction.**

- **Call 1.** `ttnn.moe_padding_config(config, actual_start, actual_end, tokens_per_chip=T,
  pad_side=P, cluster_axis=A)` with `actual_start` and `actual_end` built with
  `memory_config=ttnn.DRAM_MEMORY_CONFIG`. Cache miss; the program is built and cached. Writer
  compile-time arg `[5]` has the `IsDram` bit set, `[6]` is the DRAM aligned page size.
- **Call 2.** The same `config` tensor and the same `T` / `P` / `A`, but with `actual_start` and
  `actual_end` allocated with `memory_config=ttnn.L1_MEMORY_CONFIG`. They are still device tensors,
  still `uint32`, still ROW_MAJOR, still one element, still unsharded, still on the same device, so
  `validate_meta` passes on both the miss and hit paths. Nothing else about the call differs.
- **Hash outcome.** Pre-fix the custom hash read only `args.tokens_per_chip`, `args.pad_side`,
  `args.cluster_axis` and four fields of `config`, so the two calls hashed identically and call 2 was a
  cache hit. Post-`fab067a` `actual_start.memory_config()` is in the key, so call 2 misses and gets a
  program compiled for L1 metadata.
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

**Severity and likelihood.** This was latent in every current caller, but being latent is not a
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

But a docstring is not enforcement. Nothing rejected an L1 metadata tensor at any layer, and the op
is bound to Python with `noconvert()` tensor args
(`moe_padding_config_nanobind.cpp:59-61`), so a caller passing `ttnn.L1_MEMORY_CONFIG` reached the
kernel unmodified rather than being silently converted. The bad configuration was therefore reachable
through the public API without violating an enforced constraint, which is what made this a BUG and
not a caveat. The narrow in-tree producer set meant the likelihood of hitting it was low; it
said nothing about whether the defect existed. Note that the docstring still says DRAM while the code
now accepts and keys either placement — worth reconciling, though no longer a correctness matter.

The route this document proposed was a `TT_FATAL` inside `validate_meta` requiring
`meta.memory_config().buffer_type() == BufferType::DRAM`. `fab067a` took the other route and hashed
the placement instead (`:139`). Either would have worked, and the placement argument applies equally
to both: because this op defines a hit validator, that validator replaces the miss validator on hits
(see "Which validator runs on a cache hit" above), so a guard added only to
`validate_on_program_cache_miss` would not have fixed it. The equality guard that *did* land went into
the shared `validate_runtime_args` checker (`:83-91`), which both validators wrap (`:101-109`), so it
is live on both paths.

This defect was not unique to this op. The sibling `update_padded_kv_cache` has the materially
identical structure — its metadata tensor's `buffer_type` reaches a writer `TensorAccessorArgs`
compile-time arg, its hash records only the tensor's presence, and its `validate_meta` pins storage
type, dtype, layout, element count, shardedness and device but not `buffer_type` — and that op's
audit grades it a BUG. `fab067a` touched that op too, so its own audit is the place to check whether
the family-level fix landed there as well.

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

```106:109:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_padding_config/device/moe_padding_config_device_operation.cpp
void MoePaddingConfigDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_runtime_args(args, tensor_args);
}
```

Note `logical_volume() == 1` pins the *element count* but not the rank or the padded row width. That
residual does not matter: for a non-sharded accessor only `aligned_page_size` varies with it, and the
kernel reads only page 0 at offset 0. Since `fab067a` the metadata pair also has to agree on
`memory_config` (`:83-91`), which is what lets the hash carry `actual_start`'s alone; that check sits
in the shared checker, so it too is enforced on hits.

### 5. `config.logical_shape()` — `padded_shape()` is hashed instead

**Verdict: VALID — relaxation win.**

The factory never reads the config's logical shape. The only shape-derived quantity it consumes is
the page size:

```165:167:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_padding_config/device/moe_padding_config_device_operation.cpp
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

```215:217:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_padding_config/device/moe_padding_config_device_operation.cpp
    // The config buffer is passed as a Buffer* binding (not a raw address) so cache hits take the fast
    // path that patches its address and skips create_descriptor.
    writer_kernel.emplace_runtime_args(core, {config.buffer()});
```

### 10. `my_sp_coord` and `sp_factor` — common args `[0]` and `[1]`, set only at create time

**Verdict: VALID — invariant.**

These are the classic "set in the create path, never re-set in the override" args, so they deserve
explicit proof rather than assumption.

```156:159:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_padding_config/device/moe_padding_config_device_operation.cpp
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

```165:179:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_padding_config/device/moe_padding_config_device_operation.cpp
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
(`device/moe_padding_config_device_operation.cpp:181-189`), and the core grid is the literal single
core `{0, 0}` (`162-163`) — not a tile-derived work split. The kernel reads its page size back from
the CB interface at runtime rather than recomputing it
(`device/kernels/dataflow/writer_moe_padding_config.cpp:123`).

## Keys the custom hash adds beyond the default

- `config.padded_shape()` — not in the default key (the default hashes `logical_shape` and lets
  `padded_shape` be a derived value). Adding it is what makes dropping `logical_shape` and
  `alignment` safe simultaneously. For this op the substitution is vacuous in practice, since `config`
  is pinned to row-major unsharded and therefore pads to its logical shape; see the Metal 2.0 note in
  the post-fix section.

`actual_start.memory_config()` (`:139`) is not an addition beyond the default — the default key would
hash both metadata tensors in full — but it is what closed item 3.

## Framework side effect of having a custom hash

```1012:1014:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
        if constexpr (requires { DeviceOperation::compute_program_hash(attrs, tensor_args); }) {
            return key;  // custom hash -> opt out beyond the op-identity prefix
        } else {
```

Defining `compute_program_hash` degrades `ProgramCacheKey::canonical` to just the op type name, so
a 64-bit hash collision between two different `moe_padding_config` configurations resolves to a
wrong hit rather than a rebuild. That is inherent to every custom-hash op. Pre-fix it raised the cost
of the BUG in item 3; with the key now complete, it is the op's only remaining second-order exposure
and is not a reason to withhold the CLEAR verdict.

## Summary

| Omitted vs. default | Used by program? | Patched on hit? | Verdict |
|---|---|---|---|
| `actual_start`/`actual_end` element values (the per-chunk position) | Yes, read on device | n/a — never enters an arg | VALID — invariant |
| `actual_start`/`actual_end` buffer addresses | Yes (common args 3, 4) | Yes (op's own override) | VALID — patched |
| `actual_start`/`actual_end` `memory_config` (`buffer_type`) | Yes (writer compile-time arg `[5]`, the `IsDram` bit) | No — but `actual_start`'s is now hashed, and `actual_end` is pinned equal to it on both validator paths | **RESOLVED by fab067a** (was BUG) |
| `actual_start`/`actual_end` dtype, layout, volume, shardedness, storage | Only via pinned values | n/a | VALID — pinned by validation (both paths) |
| `config.logical_shape` | No (`padded_shape` used) | n/a | VALID — relaxation win |
| `config.page_config` beyond `layout()` | No | n/a | VALID — unused |
| `config.tensor_layout.alignment` | Only via hashed `padded_shape` | n/a | VALID — unused |
| `config.storage` kind | n/a | n/a | VALID — pinned by validation (both paths) |
| `config` buffer address | Yes | Yes (`Buffer*` binding) | VALID — patched |
| `my_sp_coord`, `sp_factor` (create-only common args) | Yes | No | VALID — invariant |
| Tile geometry (no host-side tile math) | No — op has none | n/a | VALID — unused |

**Post-`fab067a`: zero program-cache correctness bugs.** The one this audit found — item 3, the
metadata tensors' `buffer_type` — is closed. It reached the writer's compile-time arg `[5]` as the
accessor's `IsDram` bit, was refreshed on no cache-hit path, was absent from the hash, and an
L1-allocated `actual_start`/`actual_end` passed every `TT_FATAL` the op had, so two calls differing
only in that placement shared a cache entry and the second resolved an L1 address through the DRAM
bank table. `fab067a` keyed `actual_start.memory_config()` (`:139`) and required `actual_end` to match
it in the shared, hit-reachable checker (`:83-91`), which closes both the cache defect and the
one-accessor-for-two-tensors hazard in a single stroke.

Every frozen slot is now clean. Both CB page sizes, the single-core range, common args `[0..2]`,
and every compile-time arg — including the metadata accessor's — are pure functions of
{`tokens_per_chip`, `pad_side`, `cluster_axis`, `config.dtype`, `config.layout`,
`config.memory_config`, `config.padded_shape`, `actual_start.memory_config`} plus the mesh coordinate
the framework appends and device-fixed constants.

The tile check (item 11) was performed and found nothing to adjudicate: the op does no host-side tile
math, so it can neither bake a 32x32 assumption into a program for a non-32x32 tensor nor vary its
program with a tile that is absent from the key. The only shape-derived size in the descriptor is a
row-major page size, which does not consult the `Tile`.

The headline design point is worth stating plainly: the per-chunk position, which is exactly the
kind of value that goes stale in a cached KV-cache-adjacent op, is not merely omitted-and-patched
here — it never reaches the host dispatch path at all. That is a stronger guarantee than patching,
and it is what makes the op trace-safe.

## Recommendations

Items 1 and 2 were implemented by `fab067a`, though item 1 by the other of the two routes it offered;
3 remains open.

1. ~~**Fix the bug in item 3.** Add
   `TT_FATAL(meta.memory_config().buffer_type() == BufferType::DRAM, ...)` to `validate_meta`
   (`device/moe_padding_config_device_operation.cpp:66-80`).~~ **Done differently, and better.**
   `fab067a` took the alternative this item listed — folding `actual_start.memory_config()` into
   `compute_program_hash` (`:139`) — rather than the DRAM-only `TT_FATAL`. The objection raised here
   against that route ("costs a cache entry per placement for no benefit, since the op only supports
   DRAM anyway") turns out to be the weaker argument: hashing keeps L1 metadata tensors working
   instead of rejecting a configuration the kernels handle correctly once compiled for it, and it does
   not depend on the docstring's DRAM claim staying true. Note the placement point still applies to any
   guard added here: because this op defines a hit validator, checks must go in the shared
   `validate_runtime_args` (`:42-92`), which both validators wrap (`:101-109`) — and that is where the
   accompanying equality check landed.

   The family-wide framing was right. The sibling `update_padded_kv_cache` had the materially
   identical defect (metadata `buffer_type` reaching a writer `TensorAccessorArgs` compile-time arg,
   hashed only for presence, and a `validate_meta` that pins storage type, dtype, layout, element
   count, shardedness and device but not `buffer_type`), and `fab067a` touched that op in the same
   change; see its audit for what landed there.
2. ~~Independently of the cache, require the two metadata tensors to share a memory config. Only
   *one* accessor is built, from `actual_start`, and it is reused for both reads
   (`device/moe_padding_config_device_operation.cpp:195`, and `meta_args` in the kernel at
   `device/kernels/dataflow/writer_moe_padding_config.cpp:58`). A mismatched pair is wrong on the
   very first call, not just on a cache hit.~~ **Done** (`:83-91`), in the shared checker so it holds on
   both paths. This is also what makes hashing `actual_start`'s config alone sufficient in item 1.
3. **Open.** Run this op's unit tests under `-DTT_DESCRIPTOR_PATCHING_PARITY_CHECK`. The op takes the
   descriptor fast path internally, so `assert_fastpath_parity` will diff the patched program
   against a full rebuild and catch any future common-runtime-arg added to `create_descriptor`
   without a matching line in `override_runtime_arguments`
   (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:732-747`). The existing cache-entry-count
   assertion in `test_moe_padding_config.py` is a good complement but only proves reuse, not that
   reuse is correct.
