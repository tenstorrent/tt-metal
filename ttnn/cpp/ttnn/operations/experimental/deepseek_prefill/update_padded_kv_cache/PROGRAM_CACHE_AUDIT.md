# Program Cache Audit — `experimental/deepseek_prefill/update_padded_kv_cache`

Audit of
`ttnn::operations::experimental::deepseek_prefill::update_padded_kv_cache::UpdatePaddedKvCacheDeviceOperation::compute_program_hash`
against the framework default ("hash everything") key.

| | |
|---|---|
| Device operation | `UpdatePaddedKvCacheDeviceOperation` (`device/update_padded_kv_cache_device_operation.hpp:21`) |
| Custom hash | `device/update_padded_kv_cache_device_operation.cpp:208-234` |
| `operation_attributes_t` | `slot_idx`, `kv_actual_global`, `layer_idx`, `num_layers`, `cluster_axis` |
| `tensor_args_t` | `cache`, `input`, `std::optional<Tensor> slot_idx`, `std::optional<Tensor> kv_actual_global` |
| Program factories | one: `ProgramFactory::create_descriptor` (`ProgramDescriptor`-based), wrapped by `MeshWorkloadFactory` |
| `override_runtime_arguments` | **Yes**, on `MeshWorkloadFactory` (`device/update_padded_kv_cache_device_operation.cpp:439-464`) |
| `get_dynamic_runtime_args` | **No** |
| `validate_on_program_cache_hit` | **Yes** (`device/update_padded_kv_cache_device_operation.cpp:189-194`) — so it *replaces* the miss validator on hits rather than supplementing it |
| Validator actually run on a hit | `validate_runtime_args` only (`:57-126`); everything in `validate_on_program_cache_miss` before its delegation at `:186` is skipped |
| Cache-hit patch mechanism | **Op-owned override** at the device-operation level, implemented internally as the framework **buffer-binding fast path** plus a hand-written common-runtime-arg patch |
| In-place | Yes — `create_output_tensors` returns the `cache` tensor itself |

## Cache-hit patch mechanism

Two layers have to be read together here, and the op's classification depends on both.

**Outer layer.** `select_program_factory` always returns `MeshWorkloadFactory`
(`device/update_padded_kv_cache_device_operation.cpp:130-133`), and that factory defines
`override_runtime_arguments` but not `apply_descriptor`. The framework's cache-hit dispatcher
therefore calls the op's own hook on every hit:

```279:285:ttnn/api/ttnn/device_operation.hpp
        if constexpr (requires { &WorkloadFactory::apply_descriptor; }) {
            WorkloadFactory::apply_descriptor(
                cached_mesh_workload, operation_attributes, tensor_args, tensor_return_value);
        } else {
            WorkloadFactory::override_runtime_arguments(
                cached_mesh_workload, operation_attributes, tensor_args, tensor_return_value);
        }
```

**Inner layer.** The op's override immediately delegates to the descriptor adapter and then patches
by hand:

```439:463:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/update_padded_kv_cache_device_operation.cpp
void UpdatePaddedKvCacheDeviceOperation::MeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& args,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    // Default adapter behaviour: patch operand buffer-binding addresses on cache hits.
    descriptor_adapter_t::apply_descriptor(cached_workload, args, tensor_args, output);
    ...
    const bool has_metadata = tensor_args.slot_idx.has_value();
    const uint32_t arg8 = has_metadata ? tensor_args.slot_idx->buffer()->address() : args.slot_idx;
    const uint32_t arg9 = has_metadata ? tensor_args.kv_actual_global->buffer()->address() : args.kv_actual_global;
    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& writer_common = GetCommonRuntimeArgs(program, kWriterKernelHandle);
        TT_FATAL(
            kArg9 < writer_common.size(), "update_padded_kv_cache writer is missing its per-call common runtime args");
        writer_common[kArg8] = arg8;
        writer_common[kArg9] = arg9;
    }
}
```

`descriptor_adapter_t` is `DescriptorMeshWorkloadAdapter<ProgramFactory>` parameterised on
`DescriptorAdapterOperation`, a minimal four-typedef helper
(`device/update_padded_kv_cache_device_operation.hpp:65-78`). Neither that helper nor `ProgramFactory`
declares `override_runtime_arguments` or `get_dynamic_runtime_args`, so
`DescriptorMeshWorkloadAdapter::has_override_runtime_arguments()` is false for the *inner* adapter and
its `apply_descriptor` lands in the buffer-binding branch:

```726:731:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
                    if (!sv.resolved_bindings.rt_args.empty() ||
                        (!dynamic_args.empty() && !sv.resolved_bindings.empty())) {
                        auto collected =
                            collect_tensor_buffers(tensor_args, tensor_return_value, sv.workload_descriptor);
                        tt::tt_metal::apply_resolved_bindings(program, sv.resolved_bindings, collected.buffers);
                        tt::tt_metal::apply_dynamic_runtime_args(program, dynamic_args);
```

`sv.resolved_bindings.rt_args` is non-empty because both kernels register their address slot as a
`Buffer*` (`create_descriptor:416,420`), so the inner path is the fast path, never the slow-path
rebuild. `resolve_bindings` does **not** bail on this op's in-place aliasing: `cache.buffer()` appears
once in the input region and once in the output region, and an output-region entry that aliases an
input is explicitly skipped rather than treated as an ambiguous duplicate
(`tt_metal/impl/program/program_descriptor_patching.cpp:90-94`).

**Obligation on the hash.** On a hit, exactly three things get refreshed: the `input` and `cache`
buffer addresses (via `resolved_bindings`), and writer common runtime args 8 and 9. Everything else —
common args 0-7, both kernels' per-core runtime args, every compile-time arg, the CB page sizes and
`total_size`, and the core ranges — is frozen at the first miss. So every one of those must be a pure
function of the hashed set.

Note also that having `override_runtime_arguments` on the *outer* factory does **not** make
`resolve_bindings` unnecessary here, because the op delegates to the inner adapter rather than
re-deriving addresses itself. The aliasing safety comes from `resolve_bindings`'s output-region skip,
not from the mode-A bypass described for ops that hand-roll their whole cache-hit path.

### Which validator runs on a cache hit

The dispatcher runs exactly one validator on a hit, and which one is chosen has the opposite effect
from the intuitive reading:

```262:266:ttnn/api/ttnn/device_operation.hpp
    if constexpr (HasValidateOnProgramCacheHit<mesh_device_operation_t>) {
        mesh_device_operation_t::validate_on_program_cache_hit(operation_attributes, tensor_args);
    } else {
        mesh_device_operation_t::validate_on_program_cache_miss(operation_attributes, tensor_args);
    }
```

An op that defines no hit validator gets the miss validator substituted on hits, so all of its pins
hold. **This op defines one**, so the miss validator does not run on a hit at all — and this op's hit
validator is a single delegation:

```189:194:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/update_padded_kv_cache_device_operation.cpp
void UpdatePaddedKvCacheDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // Re-run the non-hashed structural checks on every hit. slot_idx and kv_actual_global are no longer
    // host attributes -- they live in the metadata tensor and are validated host-side by the caller.
    validate_runtime_args(args, tensor_args);
}
```

The miss validator also ends by delegating to `validate_runtime_args` (`:186`), so the two paths differ
by exactly the checks the miss validator performs *before* that delegation — lines 140 through 182. The
hit path therefore loses all of the following:

- `cache.storage_type() == DEVICE` and `input.storage_type() == DEVICE` (`:140-141`).
- `cache.dtype() == input.dtype()` (`:142`).
- `cache.layout() == input.layout()`, the TILE-or-ROW_MAJOR gate, and the block-float and FP8_E4M3
  layout gates (`:149-156`).
- The rank-4 checks, the head-dim and num-heads equalities (`:163-166`).
- The seq tile-alignment and `cache_seq % input_seq` checks (`:173-175`).
- `num_layers > 0` and `cache_shape[0] % num_layers == 0` (`:177-182`).

What *does* run on both paths is `validate_runtime_args` (`:57-126`): the `cluster_axis` and
`layer_idx` range checks, the 2D-mesh check, the paired-optional check, the whole `validate_meta`
lambda for both metadata tensors (`:85-100`, including `!meta.is_sharded()` at `:94`), and the
scalar-path `slot_idx` / `kv_actual_global` value checks.

This is the reason omissions 4 and 8 below are graded `CAVEAT — pinned only on the miss path` rather
than `VALID — pinned by validation`: their `TT_FATAL`s sit in the block the hit path skips. It is also
the reason every guard recommended at the end of this document is specified to go into
`validate_runtime_args` — a guard added to `validate_on_program_cache_miss` would never run on the
offending second call, which is the only call that matters for a cache bug.

**Which of the dropped checks are actually reachable.** The list above is the mechanical diff, but most
of those checks constrain values that are themselves in the cache key, and a miss-only pin on a *hashed*
value cannot be evaded: any call carrying a new value of that parameter misses, and the miss validator
runs and rejects it there. Filtering the list against
`compute_program_hash:223-233` leaves only three lines that a hit can actually reach:

| Dropped check | Constrains | In the key? | Reachable on a hit? |
|---|---|---|---|
| `storage_type() == DEVICE` ×2 (`:140-141`) | storage variant kind | No | **Yes** |
| `cache.dtype() == input.dtype()` (`:142`) | `cache.dtype()` | No (only `input.dtype()` is) | **Yes** |
| `cache.layout() == input.layout()` (`:149`) | `cache.layout()` | No (only `input.layout()` is) | **Yes** |
| TILE-or-ROW_MAJOR, block-float and FP8 gates (`:150-156`) | `input.layout()`, `input.dtype()` | Yes, both | No |
| Rank-4, head-dim, num-heads (`:163-166`) | both padded shapes | Yes, both | No |
| Seq alignment, `cache_seq % input_seq` (`:173-175`) | both padded shapes | Yes, both | No |
| `num_layers > 0`, batch divisibility (`:177-182`) | `num_layers`, `cache.padded_shape()` | Yes, both | No |

So the practical loss is `:140-141`, `:142` and `:149` — four `TT_FATAL`s, not forty lines. That matters
for the recommendations at the end of this document: the hit path is the fast path, so every check moved
onto it is paid on every dispatch, and the right move is these four (or, as argued there, just the two
at `:142` and `:149`) rather than the whole block.

## Baseline: what the default hash would cover

`hash_objects_with_default_seed(type_hash<UpdatePaddedKvCacheDeviceOperation>, attrs, tensor_args)`
walks reflection, giving:

| Source | Fields |
|---|---|
| `operation_attributes` | `slot_idx`, `kv_actual_global`, `layer_idx`, `num_layers`, `cluster_axis` |
| `cache` | storage variant kind, `logical_shape`, `dtype`, `page_config`, `memory_config`, `alignment` |
| `input` | storage variant kind, `logical_shape`, `dtype`, `page_config`, `memory_config`, `alignment` |
| `slot_idx` (optional tensor) | engaged/disengaged, plus the same six fields when engaged |
| `kv_actual_global` (optional tensor) | engaged/disengaged, plus the same six fields when engaged |

`padded_shape` is not in the default key directly — it is a derivation of `logical_shape`,
`page_config` and `alignment`. The mesh coordinates are appended by the framework
(`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:989-992`) on both the default and custom paths, so
`my_sp_coord` is never an omission.

## What the custom hash covers

```223:233:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/update_padded_kv_cache_device_operation.cpp
    return tt::tt_metal::operation::hash_operation<UpdatePaddedKvCacheDeviceOperation>(
        tensor_args.slot_idx.has_value(),
        args.layer_idx,
        args.num_layers,
        args.cluster_axis,
        input.dtype(),
        input.layout(),  // TILE vs ROW_MAJOR drives the page-unit math; must not collide
        input.memory_config(),
        input.padded_shape(),
        cache.memory_config(),
        cache.padded_shape());
```

Ten values. The two per-request scalars are dropped, the two metadata tensors collapse to a single
`has_value()` bit, and both real operands are decomposed selectively — with the notable asymmetry that
`input` contributes `dtype` and `layout` while `cache` contributes neither.

## Omitted parameters

### 1. `operation_attributes.kv_actual_global`

**Verdict: VALID — patched.**

This is the per-step index the task brief flags as the classic hazard: the prior valid global KV length
in tokens, which advances on every prefill chunk. It reaches the writer as common runtime arg 9 on the
scalar path:

```384:395:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/update_padded_kv_cache_device_operation.cpp
        writer_kernel.emplace_common_runtime_args({
            my_sp_coord,
            sp_factor,
            input_Ht,
            args.layer_idx,
            args.num_layers,
            Wt,
            cache_HtWt,
            cache_CHtWt,
            args.slot_idx,
            args.kv_actual_global,
        });
```

and is re-applied on every hit at `override_runtime_arguments:456,462`. The kernel derives the entire
write offset from it on-device — no host-computed offset is baked anywhere:

```108:122:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/kernels/dataflow/writer_update_padded_kv_cache.cpp
    const uint32_t chunk_global_t = sp_factor * chunk_local_t;
    const uint32_t boundary_slab_idx = kv_actual_global_t / chunk_global_t;
    const uint32_t boundary_chip = (kv_actual_global_t / chunk_local_t) % sp_factor;
    const uint32_t boundary_offset_t = kv_actual_global_t % chunk_local_t;

    // From the current slab base, chips before the boundary advance a full slab, the boundary chip
    // advances by its pad offset, and chips after it stay at the base.
    const uint32_t update_idxt =
        boundary_slab_idx * chunk_local_t +
        (my_sp_coord < boundary_chip ? chunk_local_t : (my_sp_coord == boundary_chip ? boundary_offset_t : 0));

    const uint32_t input_Ht = chunk_local_t;
    const uint32_t start_idx = batch_idx * cache_CHtWt + update_idxt * Wt;
```

This is the omitted-and-patched pattern done correctly. The work split (`num_blocks_of_work`, per-core
`num_blocks_per_core`, `num_blocks_written`) is a function of `input_shape[1] * input_Ht` and the
compute grid only (`create_descriptor:293-297`), so a changing `kv_actual_global` never shifts core
membership — the frozen per-core args stay correct.

On the metadata path the value is not a host scalar at all: the writer NoC-reads element [0] of a
1-element uint32 tensor (`writer_update_padded_kv_cache.cpp:74-92`). There the omission is trivially
correct — it is device data, and only the tensor's address (common arg 8/9) needs patching, which the
override does.

Because the value is not hashed, `validate_on_program_cache_hit` re-runs the range and alignment
checks on every dispatch rather than only on a miss:

```103:124:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/update_padded_kv_cache_device_operation.cpp
    if (!tensor_args.slot_idx.has_value()) {
        // The writer divides kv_actual_global by TILE_HEIGHT to get its tile offset, so it must be aligned.
        TT_FATAL(
            args.kv_actual_global % TILE_HEIGHT == 0,
            "kv_actual_global ({}) must be tile-aligned (a multiple of {})",
            args.kv_actual_global,
            TILE_HEIGHT);
        const uint32_t num_slots = cache.padded_shape()[0] / args.num_layers;
        TT_FATAL(args.slot_idx < num_slots, "slot_idx ({}) out of range for num_slots ({})", args.slot_idx, num_slots);

        // This chunk is written at a per-chip offset derived from kv_actual_global; the prior valid KV
        // plus this chunk must fit the global cache capacity (sp_factor slabs of cache_seq tokens each),
        // else the write spills past the cache. sp_factor = mesh extent along cluster_axis.
        const uint32_t sp_factor = (args.cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();
        const uint32_t chunk_global_tokens = sp_factor * tensor_args.input.padded_shape()[-2];
        const uint32_t global_cache_capacity = sp_factor * cache.padded_shape()[-2];
        TT_FATAL(
            args.kv_actual_global + chunk_global_tokens <= global_cache_capacity,
            "kv_actual_global ({}) + chunk_global ({}) would overflow global cache capacity ({})",
            args.kv_actual_global,
            chunk_global_tokens,
            global_cache_capacity);
    }
```

That is the right structure: the hash relaxation is paid for with a per-hit validator.

### 2. `operation_attributes.slot_idx`

**Verdict: VALID — patched.**

Same mechanism as omission 1, one arg over: created at `create_descriptor:393`, re-applied at
`override_runtime_arguments:455,461`, consumed on-device only as
`batch_idx = slot_idx * num_layers + layer_idx` (`writer_update_padded_kv_cache.cpp:101`), which feeds
the page index and nothing structural. `num_layers` and `layer_idx` are both hashed, so the
linearisation itself is pinned; only the free variable is omitted. Range-checked on every hit at
`device/update_padded_kv_cache_device_operation.cpp:110-111`.

### 3. `cache.logical_shape()` and `input.logical_shape()` — replaced by `padded_shape()`

**Verdict: VALID — relaxation win.**

`create_descriptor` reads padded shapes exclusively:

```251:282:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/update_padded_kv_cache_device_operation.cpp
    const auto& cache_shape = cache.padded_shape();
    const auto& input_shape = input.padded_shape();

    const tt::DataFormat data_format = datatype_to_dataformat_converter(input.dtype());
    ...
    if (is_row_major) {
        // ROW_MAJOR: page = one token row; use the buffer's aligned page size (handles row padding).
        single_page_size = cache.buffer()->aligned_page_size();
        Wt = 1;
        input_Ht = input_shape[-2];
        cache_HtWt = cache_shape[-2];
        writer_tile_height = 1;
    } else {
        single_page_size = tt::tile_size(data_format);
        Wt = cache_shape[-1] / TILE_WIDTH;
        input_Ht = input_shape[-2] / TILE_HEIGHT;
        cache_HtWt = cache_shape[-2] * Wt / TILE_HEIGHT;
        writer_tile_height = TILE_HEIGHT;
    }
    const uint32_t cache_CHtWt = cache_shape[1] * cache_HtWt;
```

The op is a page copy; the kernels address pages, and the page grid is the padded shape. Two prefill
chunks whose logical sequence lengths differ but pad to the same tile-aligned padded shape correctly
share one program, which the default hash would have forced apart. Since the output is the cache
tensor itself (`create_output_tensors:202-206`), there is no freshly-derived output spec that could go
stale from this relaxation.

### 4. `cache.dtype()` and `cache.layout()`

**Verdict: CAVEAT.** They are pinned to the hashed `input` values by `TT_FATAL`, but only on the
cache-*miss* path.

Both are genuinely consumed. `cache.layout()` selects the ROW_MAJOR branch above via the *input's*
layout, and `cache.dtype()` + `cache.layout()` together determine `cache.buffer()->aligned_page_size()`,
which becomes a writer **compile-time** arg through the tensor accessor:

```345:351:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/update_padded_kv_cache_device_operation.cpp
    KernelDescriptor::CompileTimeArgs writer_compile_args = {
        kSrcCbIndex, static_cast<uint32_t>(has_metadata), has_metadata ? kMetaCbIndex : 0u, writer_tile_height};
    TensorAccessorArgs(cache.buffer()).append_to(writer_compile_args);
    if (has_metadata) {
        // One accessor reused for both 1-element tensors (identical layout).
        TensorAccessorArgs(tensor_args.slot_idx->buffer()).append_to(writer_compile_args);
    }
```

`TensorAccessorArgs::append_to` emits the args-config word and the aligned page size as compile-time
args for a non-sharded buffer (`tt_metal/impl/buffers/tensor_accessor_args.cpp:196-205`), and
compile-time args are baked into the cached `Program` — nothing on the hit path refreshes them.

What makes this safe *today* is the pair of consistency checks:

```142:150:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/update_padded_kv_cache_device_operation.cpp
    TT_FATAL(cache.dtype() == input.dtype(), "cache and input dtype must match");
    ...
    TT_FATAL(cache.layout() == input.layout(), "cache and input layout must match");
    TT_FATAL(input.layout() == Layout::TILE || input.layout() == Layout::ROW_MAJOR, "layout must be TILE or ROW_MAJOR");
```

Given those, `cache.dtype()` and `cache.layout()` carry no information beyond the hashed
`input.dtype()` / `input.layout()`. The unchecked assumption is that they hold *on a cache hit* — and
they are not re-asserted there. `validate_on_program_cache_hit` calls only `validate_runtime_args`
(`device/update_padded_kv_cache_device_operation.cpp:189-194`), which does not contain either check.

What would break it: call 1 with `cache` and `input` both `BFLOAT16`, TILE, `cache` padded
`[1,1,128,128]`, `input` padded `[1,1,32,128]`, same memory configs — compiles and caches. Call 2 with
an identical `input` but a `cache` of dtype `BFLOAT8_B` at the same padded shape and memory config.
The hash is byte-identical (`cache.dtype()` is not in it), so the cache hits, the hit validator does
not catch the mismatch, and the writer runs with a compile-time page size of 2048 bytes against a
buffer whose pages are 1088 bytes — the KV cache is overwritten at the wrong stride. On a miss that
same call would have been rejected loudly by line 142.

The guard that closes it is one line: move the two `TT_FATAL`s from
`validate_on_program_cache_miss` into `validate_runtime_args` so they run on both paths. Adding
`cache.dtype()` / `cache.layout()` to the hash would also close it but is strictly worse — it grows the
key with values that are constrained to be redundant.

### 5. The metadata tensors' specs — only `slot_idx.has_value()` is hashed

**Verdict: BUG.** The metadata tensor's memory space and page size become writer compile-time args and
are neither hashed nor patchable.

On the metadata path the op appends a second `TensorAccessorArgs` built from
`tensor_args.slot_idx->buffer()` (line 350, quoted above). The emitted compile-time args are
`args_config.raw()` — which carries the `IsDram` and `Sharded` bits
(`tt_metal/impl/buffers/tensor_accessor_args.cpp:153-157`) — followed by
`buffer_->aligned_page_size()` (`:197-205`). Both depend on the metadata tensor's `memory_config`
and `alignment`. The hash contains only `tensor_args.slot_idx.has_value()`.

Validation pins several properties of these tensors, and unlike omission 4 these checks *do* run on
every hit (`validate_on_program_cache_hit` → `validate_runtime_args`):

```84:101:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/update_padded_kv_cache_device_operation.cpp
    if (tensor_args.slot_idx.has_value()) {
        auto validate_meta = [&cache](const Tensor& meta, const char* name) {
            TT_FATAL(meta.storage_type() == StorageType::DEVICE, "metadata tensor {} must be on device", name);
            TT_FATAL(meta.dtype() == DataType::UINT32, "metadata tensor {} must be UINT32", name);
            TT_FATAL(meta.layout() == Layout::ROW_MAJOR, "metadata tensor {} must be ROW_MAJOR", name);
            TT_FATAL(
                meta.logical_volume() == 1,
                "metadata tensor {} must be a single element (got {})",
                name,
                meta.logical_volume());
            TT_FATAL(!meta.is_sharded(), "metadata tensor {} must not be sharded", name);
            // The writer resolves meta.buffer()->address() against cache.device(); a tensor on a
            // different mesh device would bake the wrong address and fail obscurely downstream.
            TT_FATAL(meta.device() == cache.device(), "metadata tensor {} must be on the same device as cache", name);
        };
        validate_meta(tensor_args.slot_idx.value(), "slot_idx");
        validate_meta(tensor_args.kv_actual_global.value(), "kv_actual_global");
    }
```

Dtype, layout, element count and shardedness are all pinned. **Buffer type is not.** An interleaved
L1 metadata tensor passes every one of these checks.

Two-call reproduction:

- **Call 1:** `update_padded_kv_cache(cache, input, slot_idx_t, kv_t, 0, 0, layer_idx=0, num_layers=61,
  cluster_axis=1)` with `slot_idx_t` and `kv_t` allocated `MemoryConfig{INTERLEAVED, DRAM}`. Miss.
  The writer compiles with the metadata accessor's `IsDram` bit set and the DRAM-aligned page size.
- **Call 2:** identical, except `slot_idx_t` and `kv_t` are allocated
  `MemoryConfig{INTERLEAVED, L1}`. The hash is unchanged — only `has_value()` participates — so this
  is a cache hit. `override_runtime_arguments` patches common args 8/9 to the new addresses, which is
  all it can do; the compile-time accessor config is baked.
- **Stale slot:** the metadata `TensorAccessorArgs<kMetaArgsOffset>` compile-time block in
  `writer_update_padded_kv_cache.cpp:72-73`. The kernel resolves the L1 address through DRAM banking.
- **Symptom:** `slot_idx` and `kv_actual_global` are read as garbage, so
  `batch_idx = slot_idx * num_layers + layer_idx` and `update_idxt` point somewhere arbitrary in the
  cache. The chunk is written over another user's or another layer's KV, silently. No PCC check on
  this op's own output would catch it — the op returns the cache handle unchanged.

The same omission also makes the miss path fragile in a way worth recording: line 350 builds **one**
accessor from `slot_idx` and the writer uses it for both reads
(`writer_update_padded_kv_cache.cpp:79,88`), but nothing asserts that
`tensor_args.kv_actual_global` has the same buffer type and page size. A DRAM `slot_idx` paired with an
L1 `kv_actual_global` is wrong even on a fresh compile.

**This omission is family-wide.** `zero_padded_kv_cache` has the identical construction (its
`create_descriptor` appends `TensorAccessorArgs(tensor_args.slot_idx->buffer())` to both the reader and
writer compile args, hashing only `has_value()`), and its metadata validator is a strict subset of this
one — it omits even the `!is_sharded()` check. By contrast the third member of this family,
`rotary_embedding_indexed`, *does* hash `metadata->memory_config()` and `metadata->padded_shape()`
alongside `metadata.has_value()`. That asymmetry is strong evidence the omission here is an oversight
rather than a deliberate relaxation, and it means the fix should be applied across the family rather
than patched here alone.

### 6. `page_config` (the `Tile`) of `cache` and `input` — the unguarded 32x32 assumption

**Verdict: BUG.** The op accepts `Layout::TILE`, computes all of its tile geometry from the
architectural 32x32 constants rather than the tensor's actual `Tile`, validates nothing about the tile,
and does not hash `page_config`. A non-32x32 tensor therefore does not even get a freshly-built wrong
program — it silently inherits the cache entry built for a 32x32 tensor of the same padded shape.

Non-32x32 tiles are a supported TTNN configuration, so this is reachable rather than hypothetical.

**The factory is entirely 32x32-hardcoded.** The TILE branch derives every page-unit quantity from
`tt::tile_size` (which returns the byte size of a 32x32 tile, not `tile.get_tile_size(format)`) and from
bare `TILE_WIDTH`/`TILE_HEIGHT`:

```275:281:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/update_padded_kv_cache_device_operation.cpp
    } else {
        single_page_size = tt::tile_size(data_format);
        Wt = cache_shape[-1] / TILE_WIDTH;
        input_Ht = input_shape[-2] / TILE_HEIGHT;
        cache_HtWt = cache_shape[-2] * Wt / TILE_HEIGHT;
        writer_tile_height = TILE_HEIGHT;
    }
```

There is no `tensor_spec().tile()` read and no tile-geometry `TT_FATAL` anywhere in the op directory.
The seq-alignment checks that do exist are shape checks against the same architectural constant, not
tile checks, so they do not incidentally pin the geometry:

```170:174:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/update_padded_kv_cache_device_operation.cpp
    // Seq / offset arithmetic stays tile-granular (multiples of 32) in BOTH layouts: the writer's
    // update_idxt boundary math counts tile-rows even when ROW_MAJOR makes each page a single token
    // row, so input/cache seq must be 32-aligned regardless of layout.
    TT_FATAL(input_seq % TILE_HEIGHT == 0, "input seq dim ({}) must be tile-aligned", input_seq);
    TT_FATAL(cache_seq % TILE_HEIGHT == 0, "cache seq dim ({}) must be tile-aligned", cache_seq);
```

A padded sequence dimension of 32 satisfies both regardless of whether that is one 32-row tile or two
16-row tiles.

**This op is structurally worse than its two siblings, because `writer_tile_height` is a
compile-time arg.** Line 280 above assigns `writer_tile_height = TILE_HEIGHT`, and line 346 pushes it
into the writer's compile-time argument list at index 3:

```345:347:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/update_padded_kv_cache_device_operation.cpp
    KernelDescriptor::CompileTimeArgs writer_compile_args = {
        kSrcCbIndex, static_cast<uint32_t>(has_metadata), has_metadata ? kMetaCbIndex : 0u, writer_tile_height};
    TensorAccessorArgs(cache.buffer()).append_to(writer_compile_args);
```

where the kernel consumes it as a `constexpr` and uses it as the divisor that converts the per-request
token count into the page-row unit:

```49:51:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/kernels/dataflow/writer_update_padded_kv_cache.cpp
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr uint32_t tile_height = get_compile_time_arg_val(3);
    constexpr auto cache_args = TensorAccessorArgs<4>();
```

```96:97:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/kernels/dataflow/writer_update_padded_kv_cache.cpp
        slot_idx = get_common_arg_val<uint32_t>(8);
        kv_actual_global_t = get_common_arg_val<uint32_t>(9) / tile_height;
```

That matters because compile-time args are baked into the cached `Program` and are refreshed by no
cache-hit path at all — not the buffer-binding fast path, not the op's own
`override_runtime_arguments`, and not even the mode-C slow-path rebuild
(`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:748-753` re-applies runtime args only). A stale
runtime arg could at least in principle be patched by extending the override; a stale compile-time arg
can only be fixed by hashing the value or by rejecting the input.

**Two-call reproduction.**

- **Call 1:** `cache` and `input` both `BFLOAT16`, `Layout::TILE`, `Tile{32, 32}`, interleaved DRAM,
  `cache` padded `[1, 1, 128, 128]` and `input` padded `[1, 1, 32, 128]`;
  `layer_idx=0, num_layers=61, cluster_axis=1`, scalar path. Miss; the program compiles with
  `writer_tile_height = 32`, `Wt = 4`, `input_Ht = 1`, `cache_HtWt = 16`, a cache accessor whose
  compile-time `aligned_page_size` is 2048 bytes, and a source CB of `2 * 2048` bytes.
- **Call 2:** identical in every hashed respect — same dtypes, same `Layout::TILE`, same memory
  configs, same padded shapes — but both tensors carry `Tile{16, 32}`. The `Tile` lives inside
  `page_config`, which this hash does not include
  (`compute_program_hash:223-233` keeps `input.dtype/layout/memory_config/padded_shape` and
  `cache.memory_config/padded_shape` only), so the key is byte-identical and the cache hits.
  `validate_on_program_cache_hit` runs `validate_runtime_args`, which contains no tile check.
- **Stale slots:** writer compile-time arg 3 (`writer_tile_height`) stays 32 where the tensor's rows
  per page is 16; the writer's and reader's `TensorAccessorArgs` compile-time `aligned_page_size`
  stays 2048 where the real page is 1024 bytes; the source CB's `page_size` and `total_size`
  (`create_descriptor:302-310`) stay sized for 2048-byte pages; and common args 2, 5, 6, 7
  (`input_Ht`, `Wt`, `cache_HtWt`, `cache_CHtWt`) all remain the values computed by dividing by 32.
- **Symptom:** two independent corruptions compound. The writer computes
  `kv_actual_global_t = kv_actual_global / 32` instead of `/ 16`, so `update_idxt` is half the correct
  page-row offset and the chunk is written at the wrong sequence position. Each page copy then moves
  2048 bytes into a 1024-byte page (`noc.async_write(cb, s, page_bytes, {}, {.page_id = i})` at
  `writer_update_padded_kv_cache.cpp:134`, with `page_bytes` read from the stale CB), overrunning into
  the following page. The result is silent KV-cache corruption plus an out-of-bounds DRAM write past
  the last page, with no cache miss anywhere to hint at the cause.

**This defect is family-wide.** `zero_padded_kv_cache` has the identical shape — `tt::tile_size` at its
lines 315 and 317, `Wt`/`cache_H_pages` via bare `TILE_WIDTH`/`TILE_HEIGHT` at 251-252, and no tile
guard. `rotary_embedding_indexed` hardcodes 32x32 just as thoroughly (five `tt::tile_size` calls and
four bare-constant tile-count conversions) and also has no guard, yet its verdict is only CAVEAT —
purely because it dispatches through Metal 2.0 `UpdateProgramRunArgs`, whose exact `TensorSpec`
equality check covers `page_config` and therefore throws on the mismatched second call instead of
executing it. This op goes through the descriptor buffer-binding fast path
(`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:726-731`), which performs no spec comparison of any
kind, so the same source-level mistake degrades from a loud rejection to silent data loss. The
difference in outcome is entirely a property of the cache-hit mechanism the op was built on, not of the
op's own code quality.

**The guard.** The fix is small and makes omitting `page_config` correct by construction — the same
shape as the check the non-partial `interleaved_to_sharded` already carries:

```94:98:ttnn/cpp/ttnn/operations/data_movement/sharded/interleaved_to_sharded/device/interleaved_to_sharded_op.cpp
    if (input_tensor.layout() == Layout::TILE) {
        auto tile = input_tensor.tensor_spec().tile();
        if (tile.get_height() != tt::constants::TILE_HEIGHT || tile.get_width() != tt::constants::TILE_WIDTH) {
            return {false, fmt::format("interleaved_to_sharded requires standard 32x32 tiles, got {}x{}", tile.get_height(), tile.get_width())};
        }
```

Making the factory genuinely tile-aware instead (reading `tile().get_tile_shape()` and
`tile.get_tile_size(format)`) is also valid, but then the program provably varies with `Tile` and
`page_config` must be added to `compute_program_hash` in the same change.

### 7. `alignment` of `cache` and `input`

**Verdict: CAVEAT.** Not read as a tensor property, but it moves `aligned_page_size`, which is a
compile-time arg.

`Buffer::aligned_page_size()` is a function of the page size and the buffer alignment, and it appears
in three baked places — the input accessor (line 328), the cache accessor (line 347), and the CB
total size:

```302:310:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/update_padded_kv_cache_device_operation.cpp
    desc.cbs.push_back(CBDescriptor{
        .total_size = kNumInputPagesDoubleBuffered * single_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = kSrcCbIndex,
            .data_format = data_format,
            .page_size = single_page_size,
        }}},
    });
```

CB sizes are baked into the cached `Program` and are not refreshed on a hit. The hashed
{`dtype`, `layout`, `memory_config`, `padded_shape`} pins all of this for any tensor built the ordinary
way, because the alignment is then the canonical one for the buffer type. The residual is a
`TensorLayout` constructed with an explicit non-canonical `Alignment` that leaves the hashed four
unchanged. Nothing in this op's call path produces that today, and no validation rejects it — but
unlike the tile in omission 6, no supported TTNN configuration reaches it either, which is why this
half stays a caveat.

### 8. `cache.storage` and `input.storage` variant kind

**Verdict: CAVEAT — pinned only on the miss path.** The pin exists, but it is one of the checks the
hit validator drops.

```140:141:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/update_padded_kv_cache_device_operation.cpp
    TT_FATAL(cache.storage_type() == StorageType::DEVICE, "cache must be on device");
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "input must be on device");
```

Both `TT_FATAL`s sit above the `validate_runtime_args` delegation at `:186`, so under the dispatcher
branch quoted in `## Cache-hit patch mechanism` they run on the first call and never again. This is the
same structural weakness as omission 4, and the two must be graded consistently: a pin that lives only
in the miss validator of an op that defines its own hit validator is at most a caveat.

The severity is nonetheless much lower than omission 4's. A host-storage tensor has no `buffer()`, so
the hit path faults immediately in `collect_tensor_buffers` when it tries to collect an address, rather
than executing a stale program against a valid-looking buffer. The outcome is a crash on the offending
call, not silent corruption on a later one.

That difference in kind is why this document deliberately does **not** recommend closing this one. The
fix would be to move both `TT_FATAL`s into `validate_runtime_args`, which would upgrade the verdict to
`VALID — pinned by validation` — but `validate_runtime_args` runs on the cache-hit path, which is the
fast path, so those two `storage_type()` queries would be paid on every decode step of every layer for
the life of the process. The only thing they would buy is a clearer error message in front of a fault
that already happens on the same call. The right disposition is to leave this as a recorded caveat; see
recommendation 3.

### 9. Buffer addresses of `cache` and `input`

**Verdict: VALID — patched, and required.** Addresses must never be hashed. Both are declared as
`Buffer*` bindings, which is what puts the inner adapter on the fast path:

```411:423:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/update_padded_kv_cache_device_operation.cpp
    for (uint32_t i = 0; i < num_cores; ++i) {
        const CoreCoord& core = cores.at(i);
        const uint32_t num_blocks_per_core = (i < g1_numcores) ? num_blocks_per_core_g1 : num_blocks_per_core_g2;

        // Reader: (src_addr, num_tiles, src_start_tile_id)
        reader_kernel.emplace_runtime_args(core, {src_buffer, num_blocks_per_core * Wt, num_blocks_written * Wt});

        // Writer: (dst_addr, num_pages, core_blocks_written) — kernel derives update_idxt + head
        // offset from the slot_idx/kv_actual_global it reads from the metadata tensors.
        writer_kernel.emplace_runtime_args(core, {dst_buffer, num_blocks_per_core * Wt, num_blocks_written});
```

Regarding in-place aliasing specifically: because the cache is both `tensor_args.cache` (input region)
and the value returned by `create_output_tensors` (output region), it appears twice in
`collect_tensor_buffers`. That is the safe in-place case, explicitly skipped by the resolver rather than
treated as the ambiguous `matmul(X, X)` duplicate:

```90:94:tt_metal/impl/program/program_descriptor_patching.cpp
            const bool is_input = i < num_input_buffers;
            // An output/workload buffer that aliases an input is the safe in-place case — skip it.
            if (!is_input && input_buffers.contains(buf)) {
                continue;
            }
```

So the op never bails to an empty `ResolvedBindings`, and the whole class of in-place aliasing concern
is resolved by the framework here. The metadata tensors deliberately do *not* use `Buffer*` bindings —
their addresses ride in common args 8/9 (`create_descriptor:368-382`), which is why the op has to patch
them by hand.

### 10. `my_sp_coord` / `sp_factor` (derived, not attributes)

**Verdict: VALID — invariant.** `sp_factor` is the mesh extent along `cluster_axis` (hashed) and
`my_sp_coord` is derived from the dispatch coordinate (`create_descriptor:286-288`). Coordinates are
folded into the key by the framework for both the default and custom paths
(`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:989-992`), and the program cache is per-device, so a
program can never be reused at a different coordinate.

## Keys the custom hash adds beyond the default

- `input.padded_shape()` and `cache.padded_shape()` — derivations in the default key, promoted to
  first-class here. This is precisely what makes dropping both `logical_shape`s safe.
- `input.layout()` — a collapse of `page_config` to `ROW_MAJOR`/`TILE`, which is the only distinction
  the page-unit branch cares about.
- `tensor_args.slot_idx.has_value()` — a lossy projection of the optional tensors, not an addition; see
  omission 5.

## Framework side effect of having a custom hash

Defining `compute_program_hash` opts this op out of attribute-level collision resolution:

```1012:1014:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
        if constexpr (requires { DeviceOperation::compute_program_hash(attrs, tensor_args); }) {
            return key;  // custom hash -> opt out beyond the op-identity prefix
        } else {
```

`ProgramCacheKey::canonical` degrades to the op type name, so a 64-bit collision between two different
KV-cache configurations resolves to a wrong hit rather than a rebuild. For an op that mutates a shared
KV cache in place, a wrong hit is silent multi-user data corruption rather than a wrong tensor, which
raises the cost of every gap above.

## Summary

| Omitted vs. default | Used by program? | Patched on hit? | Verdict |
|---|---|---|---|
| `attrs.kv_actual_global` | Yes (writer common arg 9) | Yes (override) | VALID — patched |
| `attrs.slot_idx` | Yes (writer common arg 8) | Yes (override) | VALID — patched |
| `cache.logical_shape`, `input.logical_shape` | No (padded shapes used) | n/a | VALID — relaxation win |
| `cache.dtype`, `cache.layout` | Yes (accessor page size, CB) | No | CAVEAT — pinned only on the miss path |
| metadata tensors' `memory_config` / `alignment` | Yes (accessor compile-time args) | No (compile-time) | BUG |
| `page_config` (`Tile`) of cache/input | Yes (`writer_tile_height` compile-time arg, CB page size, tile counts) | No (compile-time) | BUG |
| `alignment` of cache/input | Only via `aligned_page_size` | No | CAVEAT |
| `cache.storage`, `input.storage` kind | n/a | n/a | CAVEAT — pinned only on the miss path |
| `cache` / `input` buffer addresses | Yes | Yes (`resolved_bindings`) | VALID — patched, required |
| `my_sp_coord` / `sp_factor` | Yes (common args 0-1) | n/a (coordinate hashed) | VALID — invariant |

**Two program-cache bugs were found.** The two per-request scalars — the values one would most expect
to be mishandled in an in-place KV-cache op — are handled correctly: omitted from the hash, re-applied
on every hit, and re-validated on every hit. The defects are elsewhere, and both are compile-time-arg
defects, which is the one category no cache-hit path can repair.

The first is the optional metadata tensors, reduced to a single `has_value()` bit while their buffer
type and aligned page size are baked into the writer's compile-time `TensorAccessorArgs`. A caller who
allocates the metadata tensors in L1 after a DRAM-allocated first call gets a silent wrong hit and
corrupts the KV cache at an arbitrary offset.

The second is the unguarded 32x32 tile assumption (omission 6). The op accepts `Layout::TILE`, derives
`single_page_size`, `Wt`, `input_Ht`, `cache_HtWt` and `writer_tile_height` from `tt::tile_size` and
bare `TILE_WIDTH`/`TILE_HEIGHT` (`create_descriptor:275-281`), never reads the tensor's actual `Tile`,
never validates it, and does not hash `page_config`. `writer_tile_height` is a *compile-time* arg
(`create_descriptor:345-346`), which makes this structurally worse here than a runtime-arg-only
exposure: a stale runtime arg could at least be patched by extending the override, whereas a stale
compile-time arg can only be fixed by hashing the value or rejecting the input. A `Tile{16, 32}` call
following an otherwise-identical `Tile{32, 32}` call hits the cache, writes at half the correct
sequence offset, and copies 2048-byte pages into 1024-byte pages. The same defect is present in
`zero_padded_kv_cache` and `rotary_embedding_indexed`; it is only in `rotary_embedding_indexed` that
the Metal 2.0 dispatch path turns it into a loud throw rather than silent corruption.

A third, lower-severity finding is structural and affects several rows above: because the op defines
`validate_on_program_cache_hit`, that validator *replaces* the miss validator on hits rather than
supplementing it, and this op's hit validator delegates to `validate_runtime_args` and does nothing
else. Everything the miss validator checks before its own delegation at `:186` is therefore absent on
the hit path — the `cache`-vs-`input` dtype and layout consistency checks that make omitting
`cache.dtype()` and `cache.layout()` sound (omission 4), the two `storage_type() == DEVICE` pins
(omission 8), and the layout, rank, shape-equality, seq-alignment and `num_layers` checks. A narrow
hit validator is a hazard rather than a safeguard here: by existing, it disables all of them.

Filtered for reachability, though, the practical loss is small: most of those checks constrain values
that are in the cache key, and a miss-only pin on a hashed value cannot be evaded, because any call
carrying a new value of it misses and meets the pin there. Only four lines survive the filter —
`:140-141`, `:142` and `:149` — and only two of those (`:142` and `:149`) fail silently rather than as
a crash. Those two are the ones recommendation 3 moves into `validate_runtime_args`; the storage pair
is left as a recorded caveat, because the hit path is the fast path and a per-dispatch check is not
worth buying a better error message in front of a fault that already occurs.

## Recommendations

**Every guard below names the function it must go into, and for this op that function is almost always
`validate_runtime_args`.** Because the op defines `validate_on_program_cache_hit`, the miss validator is
skipped entirely on a hit; a guard placed in `validate_on_program_cache_miss` would not run on the
offending second call, which is the only call a cache bug reaches. `validate_runtime_args` is the right
home for all of them because both validators delegate to it (`:186`, `:193`), so one placement covers
both paths.

**And every guard below is priced.** The cache-hit path is the fast path — it is what the program cache
exists to make cheap — so a `TT_FATAL` added to `validate_runtime_args` is paid on every dispatch for
the life of the process. That is the cost side of every recommendation here, and it is why this document
recommends moving *specific* pins rather than the whole miss-time block, and why one of the regraded
rows is deliberately left as a documented caveat rather than fixed.

There are two distinct ways to close a miss-only pin in this op, and they are not interchangeable:

- **Targeted (recommended):** move the specific `TT_FATAL`s into `validate_runtime_args`. Adds only
  those comparisons per dispatch. This is what recommendations 3 and 4 mean.
- **Wholesale (alternative):** delete `validate_on_program_cache_hit` entirely, putting the op on the
  dispatcher's substitution branch so the full miss validator runs on every hit. Simplest and safest,
  and it can never be silently regressed by someone adding a check to the wrong function — but it puts
  all of `:137-182` on the hot path: two `storage_type()` calls, four layout/dtype gates, two rank
  queries, five shape comparisons and two arithmetic divisibility checks, on every single decode step.
  For this op that is poor value, because the reachability table in `## Cache-hit patch mechanism`
  shows only four of those lines can be reached on a hit at all. Prefer the targeted move.

1. Hash the metadata tensors' specs. The minimal fix mirrors what `rotary_embedding_indexed` already
   does: add `tensor_args.slot_idx->memory_config()` and `tensor_args.kv_actual_global->memory_config()`
   (with neutral defaults on the scalar path) to `compute_program_hash`. This is a family-wide gap —
   apply the same change to `zero_padded_kv_cache`, whose metadata validator is even thinner.
2. Independently, add `TT_FATAL(meta.buffer()->buffer_type() == BufferType::DRAM, ...)` to the
   `validate_meta` lambda, and assert that the `slot_idx` and `kv_actual_global` tensors share a buffer
   type and aligned page size — the writer reuses one accessor for both
   (`create_descriptor:348-351`) and nothing currently enforces that they are interchangeable, which is
   a correctness gap even on a cache miss.
   **Target function:** `validate_meta` already lives inside `validate_runtime_args` (`:85-98`), so it
   is on the hit path and is the correct home as-is. This is worth stating because the equivalent guard
   must *not* go into `validate_on_program_cache_miss`: the defect it closes is a wrong *hit*, and a
   miss-only guard would pass the first call and never see the second.
   **Per-dispatch cost:** two extra checks per metadata tensor, and only on the metadata path — the
   lambda is already called there, so this adds to an existing cost rather than creating one. Worth it:
   unlike the storage rows below, the defect these close is the silent wrong hit that is this
   document's headline BUG.
3. **Move exactly two `TT_FATAL`s into `validate_runtime_args`:** `cache.dtype() == input.dtype()`
   (`:142`) and `cache.layout() == input.layout()` (`:149`). These are the pins behind omission 4, and
   they are the only dropped checks in this op whose absence produces *silent* misbehaviour: both
   constrain a `cache` property that is not in the cache key against an `input` property that is, so a
   mismatched second call hits and executes a program built for the wrong cache page size. Two scalar
   enum comparisons per dispatch is the right price for closing a silent-corruption path, and it is
   cheaper and more targeted than adding the two values to the hash. This upgrades omission 4 to
   `VALID — pinned by validation`.

   **Do not also move the two `storage_type() == StorageType::DEVICE` checks (`:140-141`), and leave
   omission 8 as a documented CAVEAT.** They are reachable on a hit, so the regrade stands — but the
   failure they prevent is not silent. A host-storage tensor has no device buffer, so the hit path
   faults in `collect_tensor_buffers` the moment it tries to collect an address; the caller gets a
   crash on the offending call either way, and the only thing the moved check buys is a better error
   message. That is not worth two `storage_type()` queries on every decode step of every layer. The
   correct disposition is the one this document already gives it: a CAVEAT recorded with its reasoning,
   not a guard. If the diagnostic quality is judged worth the cost later, the move is one line each and
   the reasoning above is what should be revisited.
4. Reject a non-32x32 `Tile` on the TILE path, closing omission 6. Assert
   `cache.tensor_spec().tile().get_height() == TILE_HEIGHT` and the same for `get_width()`, on `cache`
   and `input`, in the same shape as the `interleaved_to_sharded` guard quoted in omission 6.
   **Target function:** `validate_runtime_args`, not `validate_on_program_cache_miss`. The reproduction
   in omission 6 is a *hit*, so a guard in the miss validator would let the first `Tile{32,32}` call
   through and then not run at all on the `Tile{16,32}` call that corrupts the cache. Placing it in
   `validate_runtime_args` covers the miss path too, via the delegation at `:186`.
   **Per-dispatch cost:** two `uint32_t` comparisons against constants, on two tensors. This is the
   clearest case in the document of a check worth its price — it closes a BUG with silent
   KV-cache corruption and an out-of-bounds DRAM write as its symptom.
   This is minimal and makes omitting `page_config` correct by construction. The alternative — making
   the factory tile-aware via `tile.get_tile_size(data_format)` and `tile().get_tile_shape()`, and
   setting `writer_tile_height = tile.get_height()` — requires adding `page_config` to
   `compute_program_hash` in the same change, because the program would then provably vary with `Tile`.
   This is a family-wide gap: apply the same guard to `zero_padded_kv_cache` and
   `rotary_embedding_indexed`.
5. Run this op's tests under `-DTT_DESCRIPTOR_PATCHING_PARITY_CHECK`. Note that the parity oracle only
   covers runtime args and CB addresses
   (`tt_metal/api/tt-metalium/experimental/program_descriptor_patching.hpp:176-186`), so it will *not*
   catch the compile-time-arg defect in omission 5 — that one needs the hash fix or the validation
   guard. It will, however, catch any future regression in the common-arg patch.
6. The override hard-codes `kWriterKernelHandle = 1` on the assumption that kernel handles follow
   descriptor push order (`create_descriptor:425-426`). That holds today because
   `ProgramImpl` creates kernels in `descriptor.kernels` order
   (`tt_metal/impl/program/program.cpp:402-466`), but it is an implicit coupling between two functions
   several hundred lines apart. Deriving the handle from the descriptor, or at least asserting the
   kernel count, would make it robust.
