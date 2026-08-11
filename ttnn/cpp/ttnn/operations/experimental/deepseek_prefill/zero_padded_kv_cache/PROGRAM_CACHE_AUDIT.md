# Program Cache Audit — `experimental/deepseek_prefill/zero_padded_kv_cache`

Audit of
`ttnn::operations::experimental::deepseek_prefill::zero_padded_kv_cache::ZeroPaddedKvCacheDeviceOperation::compute_program_hash`
against the framework default ("hash everything") key.

| | |
|---|---|
| Device operation | `ZeroPaddedKvCacheDeviceOperation` (`device/zero_padded_kv_cache_device_operation.hpp:21`) |
| Custom hash | `device/zero_padded_kv_cache_device_operation.cpp:212-231` |
| `operation_attributes_t` | `slot_idx`, `valid_global`, `chunk_size_global`, `pad_align`, `layer_idx`, `num_layers`, `cluster_axis` |
| `tensor_args_t` | `cache`, `std::optional<Tensor> slot_idx`, `std::optional<Tensor> valid_global` |
| Program factories | one: `ProgramFactory::create_descriptor` (`ProgramDescriptor`-based), with two internal layout branches — a dataflow-only ROW_MAJOR writer, and a TILE reader/compute/writer trio |
| `override_runtime_arguments` | **Yes**, on `MeshWorkloadFactory` (`device/zero_padded_kv_cache_device_operation.cpp:409-446`) |
| `get_dynamic_runtime_args` | **No** |
| `validate_on_program_cache_hit` | **Yes** (`device/zero_padded_kv_cache_device_operation.cpp:197-200`) — so it *replaces* the miss validator on hits rather than supplementing it |
| Validator actually run on a hit | `validate_runtime_args` only (`:66-145`); everything in `validate_on_program_cache_miss` before its delegation at `:194` is skipped |
| Cache-hit patch mechanism | **Op-owned override** at the device-operation level, implemented internally as the framework **buffer-binding fast path** plus a hand-written common-runtime-arg patch |
| In-place | Yes — `create_output_tensors` returns the `cache` tensor itself; there is no separate input |

## Cache-hit patch mechanism

The op sits at the junction of two mechanisms and both have to be read to classify it.

**Outer layer.** `select_program_factory` always returns `MeshWorkloadFactory`
(`device/zero_padded_kv_cache_device_operation.cpp:149-152`). That factory defines
`override_runtime_arguments` and not `apply_descriptor`, so the framework's cache-hit dispatcher hands
control to the op on every hit:

```279:285:ttnn/api/ttnn/device_operation.hpp
        if constexpr (requires { &WorkloadFactory::apply_descriptor; }) {
            WorkloadFactory::apply_descriptor(
                cached_mesh_workload, operation_attributes, tensor_args, tensor_return_value);
        } else {
            WorkloadFactory::override_runtime_arguments(
                cached_mesh_workload, operation_attributes, tensor_args, tensor_return_value);
        }
```

**Inner layer.** The override delegates address patching to the descriptor adapter and then writes the
per-call scalars itself:

```409:445:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/zero_padded_kv_cache/device/zero_padded_kv_cache_device_operation.cpp
void ZeroPaddedKvCacheDeviceOperation::MeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& args,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    descriptor_adapter_t::apply_descriptor(cached_workload, args, tensor_args, output);
    ...
    if (tensor_args.slot_idx.has_value()) {
        const uint32_t slot_idx_addr = static_cast<uint32_t>(tensor_args.slot_idx->buffer()->address());
        const uint32_t valid_global_addr = static_cast<uint32_t>(tensor_args.valid_global->buffer()->address());
        for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
            for (uint32_t kernel_handle : {0u, 2u}) {  // reader, writer (metadata path is TILE-only)
                auto& common = GetCommonRuntimeArgs(program, kernel_handle);
                TT_FATAL(
                    kValidGlobalAddrCommonArgIdx < common.size(),
                    "zero_padded_kv_cache kernel missing the metadata-tensor addr common args");
                common[kSlotIdxAddrCommonArgIdx] = slot_idx_addr;
                common[kValidGlobalAddrCommonArgIdx] = valid_global_addr;
            }
        }
    } else {
        const uint32_t num_kernels = tensor_args.cache.layout() == Layout::ROW_MAJOR ? 1u : 3u;
        for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
            for (uint32_t kernel_handle = 0; kernel_handle < num_kernels; ++kernel_handle) {
                auto& common = GetCommonRuntimeArgs(program, kernel_handle);
                TT_FATAL(
                    kSlotIdxCommonArgIdx < common.size(), "zero_padded_kv_cache kernel missing per-call common args");
                common[kValidGlobalCommonArgIdx] = args.valid_global;
                common[kSlotIdxCommonArgIdx] = args.slot_idx;
            }
        }
    }
}
```

`descriptor_adapter_t` is `DescriptorMeshWorkloadAdapter<ProgramFactory>` over
`DescriptorAdapterOperation`, a four-typedef helper (`device/zero_padded_kv_cache_device_operation.hpp:70-82`).
Neither the helper nor `ProgramFactory` declares `override_runtime_arguments` or
`get_dynamic_runtime_args`, so the inner adapter's `has_override_runtime_arguments()` is false and its
`apply_descriptor` takes the buffer-binding branch:

```726:731:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
                    if (!sv.resolved_bindings.rt_args.empty() ||
                        (!dynamic_args.empty() && !sv.resolved_bindings.empty())) {
                        auto collected =
                            collect_tensor_buffers(tensor_args, tensor_return_value, sv.workload_descriptor);
                        tt::tt_metal::apply_resolved_bindings(program, sv.resolved_bindings, collected.buffers);
                        tt::tt_metal::apply_dynamic_runtime_args(program, dynamic_args);
```

`rt_args` is non-empty because every kernel registers the cache as a `Buffer*` binding
(`create_descriptor:310,360,393`), so the slow-path rebuild is never taken.

**Obligation on the hash.** A cache hit refreshes exactly: the cache buffer address on all kernels; and
common args 3/9 (scalar path) or 10/11 (metadata path) on the kernels the override enumerates.
Everything else — common args 0, 1, 2, 4, 5, 6, 7, 8, every compile-time arg, all four CB page sizes
and totals, the single-core `CoreRangeSet`, and which of the two layout branches was compiled — is
frozen at the first miss and must be a function of the hashed set.

Two consequences of that worth stating up front, because they are correct decisions rather than
omissions: `chunk_size_global` and `pad_align` are per-request-looking values that end up in common
args 2 and 4 (`create_descriptor:273-286`) and are **not** patched by the override, so hashing them is
mandatory — and the op does hash both. Similarly `layer_idx` and `num_layers` land in common args 5/6
unpatched, and both are hashed.

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
validator is a bare delegation, with not even a comment:

```197:200:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/zero_padded_kv_cache/device/zero_padded_kv_cache_device_operation.cpp
void ZeroPaddedKvCacheDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_runtime_args(args, tensor_args);
}
```

The miss validator also ends by delegating to `validate_runtime_args` (`:194`), so the two paths differ
by exactly the checks the miss validator performs *before* that delegation — lines 157 through 193. The
hit path therefore loses all of the following:

- `cache.storage_type() == DEVICE` (`:157`).
- `cache.buffer()->buffer_type() == BufferType::DRAM` (`:158`). This one is not load-bearing for the
  cache key, since `cache.memory_config()` is hashed and carries the buffer type — but it is worth
  noting that the op's one explicit buffer-type pin is also its one pin that never runs twice.
- The TILE-or-ROW_MAJOR gate and the ROW_MAJOR dtype and FP8_E4M3 layout gates (`:159-170`).
- The metadata-path-is-TILE-only guard (`:174-179`). This is the sharpest loss in the list: it is the
  check that keeps the metadata path off the ROW_MAJOR program, and it is absent on every hit.
- The rank-4 check, the num-heads-is-1 check, `num_layers > 0`, the batch-divisibility check, and the
  `layer_idx < num_layers` range check (`:181-193`).

What *does* run on both paths is `validate_runtime_args` (`:66-145`): the `cluster_axis` check, the
paired-optional check, the `validate_meta` lambda for both metadata tensors (`:83-93`), the scalar-path
`slot_idx` range check, the 2D-mesh check, and the whole `chunk_size_global` / `pad_align` /
`chunk_local` / capacity block. Note that this op's `validate_meta` is *thinner* than
`update_padded_kv_cache`'s — it omits the `!meta.is_sharded()` check that its sibling has — which is
the gap called out in omission 4.

This is the reason omission 7 below is graded `CAVEAT — pinned only on the miss path` rather than
`VALID — pinned by validation`. It is also the reason every guard recommended at the end of this
document is specified to go into `validate_runtime_args` — a guard added to
`validate_on_program_cache_miss` would never run on the offending second call, which is the only call
that matters for a cache bug.

**Which of the dropped checks are actually reachable — for this op, almost none.** The list above is
the mechanical diff, but a miss-only pin on a value that is itself in the cache key cannot be evaded:
any call carrying a new value of that parameter misses, and the miss validator runs and rejects it
there. This op's hash is unusually broad — it covers `cache.dtype()`, `cache.layout()`,
`cache.memory_config()`, `cache.padded_shape()`, `layer_idx`, `num_layers`, `chunk_size_global`,
`pad_align` and `slot_idx.has_value()` (`compute_program_hash:220-230`) — so filtering the dropped list
against it leaves exactly one line:

| Dropped check | Constrains | In the key? | Reachable on a hit? |
|---|---|---|---|
| `cache.storage_type() == DEVICE` (`:157`) | storage variant kind | No | **Yes** |
| `cache.buffer()->buffer_type() == DRAM` (`:158`) | buffer type | Yes, inside `cache.memory_config()` | No |
| Layout gate, ROW_MAJOR dtype gate, FP8 layout gate (`:159-170`) | `cache.layout()`, `cache.dtype()` | Yes, both | No |
| Metadata-path-is-TILE-only (`:174-179`) | `cache.layout()` and `slot_idx.has_value()` | Yes, both | No |
| Rank-4, num-heads-is-1 (`:181-182`) | `cache.padded_shape()` | Yes | No |
| `num_layers > 0`, batch divisibility, `layer_idx < num_layers` (`:183-193`) | `num_layers`, `layer_idx`, `cache.padded_shape()` | Yes, all three | No |

The metadata-path-is-TILE-only guard deserves a specific note, because it looks like the most dangerous
loss and is not one. Both values it constrains — `cache.layout()` and `slot_idx.has_value()` — are in
the key, so the combination (ROW_MAJOR, metadata) has its own cache slot. Its first occurrence is
necessarily a miss, the full miss validator runs, and the call is rejected there. It is impossible to
reach a hit on that key without having already been rejected on it. The same argument disposes of the
kernel-handle coupling in the override: the override's `{0u, 2u}` metadata-path selection rests on this
guard, and the guard is effectively enforced despite living in the miss validator.

So the only genuinely reachable dropped check in this op is the single `storage_type()` pin at `:157`,
and its failure mode is a crash rather than silent corruption. That conclusion drives the
recommendations at the end of this document.

## Baseline: what the default hash would cover

`hash_objects_with_default_seed(type_hash<ZeroPaddedKvCacheDeviceOperation>, attrs, tensor_args)`
walks reflection, giving:

| Source | Fields |
|---|---|
| `operation_attributes` | `slot_idx`, `valid_global`, `chunk_size_global`, `pad_align`, `layer_idx`, `num_layers`, `cluster_axis` |
| `cache` | storage variant kind, `logical_shape`, `dtype`, `page_config`, `memory_config`, `alignment` |
| `slot_idx` (optional tensor) | engaged/disengaged, plus the same six fields when engaged |
| `valid_global` (optional tensor) | engaged/disengaged, plus the same six fields when engaged |

`padded_shape` is not directly in the default key — it is derived from `logical_shape`, `page_config`
and `alignment`. Mesh coordinates are folded in by the framework on both paths
(`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:989-992`).

## What the custom hash covers

```220:230:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/zero_padded_kv_cache/device/zero_padded_kv_cache_device_operation.cpp
    return tt::tt_metal::operation::hash_operation<ZeroPaddedKvCacheDeviceOperation>(
        tensor_args.slot_idx.has_value(),
        args.layer_idx,
        args.num_layers,
        args.cluster_axis,
        args.chunk_size_global,
        args.pad_align,
        cache.dtype(),
        cache.layout(),
        cache.memory_config(),
        cache.padded_shape());
```

Five of the seven attributes are kept; the two per-request scalars are dropped. The cache is
decomposed selectively into four of its six default components. The two optional metadata tensors
collapse to a single `has_value()` bit.

## Omitted parameters

### 1. `operation_attributes.valid_global`

**Verdict: VALID — patched.**

This is the moving index the task brief warns about: the number of real tokens written so far, which
advances on every prefill chunk and defines where the pad window `[valid_global, ceil_pad_align(valid_global))`
starts. It is a host scalar on the scalar path, placed in common arg 3 at build time:

```273:286:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/zero_padded_kv_cache/device/zero_padded_kv_cache_device_operation.cpp
    const std::vector<uint32_t> common_runtime_args = {
        my_sp_coord,
        sp_factor,
        chunk_local,
        args.valid_global,
        args.pad_align,
        args.layer_idx,
        args.num_layers,
        Wt,
        cache_CH_pages,
        args.slot_idx,
        slot_idx_addr,
        valid_global_addr,
    };
```

and re-applied on every hit at `override_runtime_arguments:441`, across all three TILE kernels or the
single ROW_MAJOR writer. All of the window arithmetic is done on-device from that arg — nothing
derived from it is baked:

```23:44:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/zero_padded_kv_cache/device/kernels/zero_padded_kv_cache_common.hpp
inline ZeroPadTokenRange zero_pad_compute_token_range(uint32_t valid_global) {
    const uint32_t my = get_common_arg_val<uint32_t>(0);
    const uint32_t sp = get_common_arg_val<uint32_t>(1);
    const uint32_t chunk_local = get_common_arg_val<uint32_t>(2);
    const uint32_t pad_align = get_common_arg_val<uint32_t>(4);

    const uint32_t pad_end = ((valid_global + pad_align - 1) / pad_align) * pad_align;
    if (pad_end == valid_global) {
        return {0, 0};
    }

    const uint32_t chunk_global = sp * chunk_local;
    const uint32_t slab = valid_global / chunk_global;
    const uint32_t chip_begin = slab * chunk_global + my * chunk_local;
    const uint32_t chip_end = chip_begin + chunk_local;
    const uint32_t begin = valid_global > chip_begin ? valid_global : chip_begin;
    const uint32_t end = pad_end < chip_end ? pad_end : chip_end;
    if (begin >= end) {
        return {0, 0};
    }

    return {end - begin, slab * chunk_local + begin - chip_begin};
}
```

Crucially, the amount of *work* also varies with `valid_global` (`w.count`, `w.first_partial`,
`w.row_start`), and the design absorbs that on-device rather than in the work split: the program always
runs on exactly one core (`create_descriptor:261`), the reader unconditionally pushes `Wt` source tiles
and a mask (`reader_zero_padded_kv_cache.cpp:71-95`), the compute unconditionally processes `Wt` tiles
(`zero_padded_kv_cache.cpp:33-46`), and the writer discards them when there is nothing to write back
(`writer_zero_padded_kv_cache.cpp:62-73`). That unconditional CB protocol is what makes the omission
safe — with a `valid_global`-dependent core split or a conditional CB push, the frozen per-core args
would go stale.

On the metadata path `valid_global` arrives as device data instead: the reader and writer NoC-read
element [0] of a 1-element uint32 tensor (`reader_zero_padded_kv_cache.cpp:56-59`,
`writer_zero_padded_kv_cache.cpp:49-52`), and only the tensor's address needs patching, which the
override does at line 431.

Because the value is not hashed, the capacity check re-runs on every hit rather than only on a miss —
`validate_on_program_cache_hit` calls `validate_runtime_args`
(`device/zero_padded_kv_cache_device_operation.cpp:197-200`), which contains:

```138:144:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/zero_padded_kv_cache/device/zero_padded_kv_cache_device_operation.cpp
    if (!tensor_args.valid_global.has_value()) {
        TT_FATAL(
            args.valid_global <= global_capacity,
            "valid_global ({}) exceeds cache capacity ({})",
            args.valid_global,
            global_capacity);
    }
```

### 2. `operation_attributes.slot_idx`

**Verdict: VALID — patched.**

Same mechanism, common arg 9: written at `create_descriptor:283`, re-applied at
`override_runtime_arguments:442`, consumed on-device only as the batch-slot linearisation
`(slot * num_layers + layer) * cache_CH_pages`
(`device/kernels/zero_padded_kv_cache_common.hpp:66`, and its ROW_MAJOR counterpart at `:103`). Both
`num_layers` and `layer_idx` are hashed, so only the free variable is omitted, and it is range-checked
on every hit:

```100:103:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/zero_padded_kv_cache/device/zero_padded_kv_cache_device_operation.cpp
    if (!tensor_args.slot_idx.has_value()) {
        const uint32_t num_slots = cache.padded_shape()[0] / args.num_layers;
        TT_FATAL(args.slot_idx < num_slots, "slot_idx ({}) out of range for num_slots ({})", args.slot_idx, num_slots);
    }
```

### 3. `cache.logical_shape()` — replaced by `padded_shape()`

**Verdict: VALID — relaxation win.**

The factory reads only the padded shape:

```250:253:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/zero_padded_kv_cache/device/zero_padded_kv_cache_device_operation.cpp
    const bool is_row_major = cache.layout() == Layout::ROW_MAJOR;
    const uint32_t Wt = is_row_major ? 1 : cache_shape[-1] / TILE_WIDTH;
    const uint32_t cache_H_pages = is_row_major ? cache_shape[-2] : cache_shape[-2] * Wt / TILE_HEIGHT;
    const uint32_t cache_CH_pages = cache_shape[1] * cache_H_pages;
```

and the kernels address the cache in pages, which is exactly the padded-shape grid. Since the op is
in-place (`compute_output_specs` returns `cache.tensor_spec()` at line 202-205) there is no derived
output spec that could inherit a stale logical shape. Two callers whose KV caches differ only in an
unpadded logical sequence length correctly share one program; the default hash would have forced a
recompile for no reason.

### 4. The metadata tensors' specs — only `slot_idx.has_value()` is hashed

**Verdict: BUG.** The metadata tensors' memory space, shardedness and aligned page size become
reader-and-writer compile-time args and are neither hashed nor patchable.

On the metadata path the op appends a second tensor accessor to two kernels:

```346:357:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/zero_padded_kv_cache/device/zero_padded_kv_cache_device_operation.cpp
    reader.compile_time_args = {
        kSrcCbIndex,
        kMaskCbIndex,
        cache_tile_size,
        static_cast<uint32_t>(has_metadata),
        has_metadata ? kMetaCbIndex : 0u};
    TensorAccessorArgs(cache.buffer()).append_to(reader.compile_time_args);
    if (has_metadata) {
        // One metadata accessor, reused for both 1-element tensors (identical layout); the kernel reads
        // each from its own DRAM address (common args 10/11).
        TensorAccessorArgs(tensor_args.slot_idx->buffer()).append_to(reader.compile_time_args);
    }
```

(and the identical block for the writer at `:386-390`). For a non-sharded buffer,
`TensorAccessorArgs::append_to` emits the args-config word — which carries the `IsDram` and `Sharded`
bits (`tt_metal/impl/buffers/tensor_accessor_args.cpp:153-157`) — and `aligned_page_size`
(`:196-205`). For a sharded buffer it emits an entirely different, longer block. All of that is
compile-time state baked into the cached `Program`; no cache-hit path can refresh it. The hash contains
only `tensor_args.slot_idx.has_value()`.

The metadata validator does run on every hit (`validate_on_program_cache_hit` →
`validate_runtime_args`), but it pins less than it needs to:

```82:96:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/zero_padded_kv_cache/device/zero_padded_kv_cache_device_operation.cpp
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
            TT_FATAL(meta.device() == cache.device(), "metadata tensor {} must be on the same device as cache", name);
        };
        validate_meta(tensor_args.slot_idx.value(), "slot_idx");
        validate_meta(tensor_args.valid_global.value(), "valid_global");
    }
```

Dtype, layout, element count and device are pinned. **Buffer type is not, and neither is shardedness.**
An interleaved-L1 or a height-sharded single-element uint32 tensor passes all five checks.

Two-call reproduction:

- **Call 1:** `zero_padded_kv_cache(cache, slot_idx_t, valid_global_t, 0, layer_idx=0, num_layers=61,
  0, chunk_size_global=1024, cluster_axis=1, pad_align=128)` with both metadata tensors allocated
  `MemoryConfig{INTERLEAVED, DRAM}`. Miss; the reader and writer compile with the metadata accessor's
  `IsDram` bit set and the DRAM-aligned page size.
- **Call 2:** identical, except the metadata tensors are allocated `MemoryConfig{INTERLEAVED, L1}` (a
  natural thing to do for a 4-byte value the host rewrites every chunk). The hash is unchanged — only
  `has_value()` participates — so this is a cache hit. The override patches common args 10/11 to the
  new addresses, which is all it is able to do.
- **Stale slot:** the metadata `TensorAccessorArgs<kMetaArgsOffset>` compile-time block referenced at
  `reader_zero_padded_kv_cache.cpp:46-47` and `writer_zero_padded_kv_cache.cpp:39-40`. The kernels
  resolve an L1 address through DRAM banking.
- **Symptom:** `slot` and `valid_global` are read as garbage on-device. `batch_page_base` points at an
  arbitrary batch slot and `zero_pad_compute_token_range` computes an arbitrary window, so the writer
  zeroes pages belonging to some other user or layer. The KV cache is silently destroyed in a region
  the op was never asked to touch, and the op's return value (the cache handle) looks fine.

The shardedness gap is worse still: a sharded metadata tensor changes the *number* of compile-time args
the accessor emits, so on a hit the kernel's fixed `TensorAccessorArgs<5>` / `kMetaArgsOffset` decoding
would misparse the cache accessor's own arguments too.

**This omission is family-wide.** `update_padded_kv_cache` has the same construction — one
`TensorAccessorArgs` built from `slot_idx->buffer()` appended to the writer's compile args, with only
`has_value()` hashed. Its validator is a strict superset of this one: it also asserts
`!meta.is_sharded()`. The third member of the family, `rotary_embedding_indexed`, hashes
`metadata->memory_config()` and `metadata->padded_shape()` in addition to `has_value()` and so does not
have the gap at all. Because two of the three ops share it and the third shows the intended shape, the
fix belongs at the family level, not in this file alone.

A second, miss-path-only problem falls out of the same code: the comment at line 354-355 asserts the
two metadata tensors have "identical layout" and reuses one accessor for both reads, but nothing
validates that `valid_global`'s buffer matches `slot_idx`'s. A DRAM `slot_idx` paired with an L1
`valid_global` is wrong even on a fresh compile.

### 5. `cache.page_config()` (the `Tile`) — the unguarded 32x32 assumption

**Verdict: BUG.** `cache.layout()` is hashed, which collapses `PageConfig` to `ROW_MAJOR` vs `TILE`
and discards the tile shape. The op accepts `Layout::TILE`, computes every page-unit quantity from the
architectural 32x32 constants rather than the cache's actual `Tile`, validates nothing about the tile
geometry, and does not hash `page_config`. A non-32x32 cache therefore does not even get a freshly-built
wrong program — it silently inherits the cache entry built for a 32x32 cache of the same padded shape.

Non-32x32 tiles are a supported TTNN configuration, so this is reachable rather than hypothetical.

**The factory is entirely 32x32-hardcoded.** The page-count derivation uses bare `TILE_WIDTH` and
`TILE_HEIGHT` rather than `cache.tensor_spec().tile().get_tile_shape()`:

```249:253:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/zero_padded_kv_cache/device/zero_padded_kv_cache_device_operation.cpp
    const tt::DataFormat cache_format = datatype_to_dataformat_converter(cache.dtype());
    const bool is_row_major = cache.layout() == Layout::ROW_MAJOR;
    const uint32_t Wt = is_row_major ? 1 : cache_shape[-1] / TILE_WIDTH;
    const uint32_t cache_H_pages = is_row_major ? cache_shape[-2] : cache_shape[-2] * Wt / TILE_HEIGHT;
    const uint32_t cache_CH_pages = cache_shape[1] * cache_H_pages;
```

and the byte sizes come from `tt::tile_size`, which returns the size of a 32x32 tile, not
`tile.get_tile_size(format)`:

```315:330:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/zero_padded_kv_cache/device/zero_padded_kv_cache_device_operation.cpp
    const uint32_t cache_tile_size = tt::tile_size(cache_format);
    const tt::DataFormat mask_format = tt::DataFormat::Float16_b;
    const uint32_t mask_tile_size = tt::tile_size(mask_format);

    // CBs: src (partial tile read), mask (bf16 row-mask), out (masked partial), zero (write scratch).
    auto add_cb = [&](uint32_t index, tt::DataFormat fmt, uint32_t page, uint32_t npages) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = npages * page,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{.buffer_index = index, .data_format = fmt, .page_size = page}}},
        });
    };
    add_cb(kSrcCbIndex, cache_format, cache_tile_size, Wt);
    add_cb(kMaskCbIndex, mask_format, mask_tile_size, 1);
    add_cb(kOutCbIndex, cache_format, cache_tile_size, Wt);
    add_cb(kZeroCbIndex, cache_format, cache_tile_size, 1);
```

`cache_tile_size` is not only a CB page size — it is compile-time arg index 2 of both the reader
(`create_descriptor:346-351`) and the writer (`:379-384`), and `TensorAccessorArgs(cache.buffer())`
(lines 307, 352, 385) separately emits the *buffer's* real `aligned_page_size`. Under a non-32x32 tile
those two disagree, in a program nothing can rebuild.

And the reader hard-codes the 32x32 four-face layout when building the row mask:

```86:94:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/zero_padded_kv_cache/device/kernels/dataflow/reader_zero_padded_kv_cache.cpp
    for (uint32_t face = 0; face < 4; ++face) {
        const uint32_t row_base = (face >= 2) ? 16u : 0u;  // faces 0,1 -> rows 0-15; 2,3 -> rows 16-31
        for (uint32_t fr = 0; fr < 16; ++fr) {
            const uint16_t val = ((row_base + fr) < rs) ? kBf16One : 0u;
            for (uint32_t fc = 0; fc < 16; ++fc) {
                m[face * 256 + fr * 16 + fc] = val;
            }
        }
    }
```

as does the shared header (`device/kernels/zero_padded_kv_cache_common.hpp:72`,
`constexpr uint32_t tile_height = 32;`).

**Nothing validates the tile geometry.** There is no `tensor_spec().tile()` read and no tile-geometry
`TT_FATAL` anywhere in the op directory. The one alignment check that exists is a check on
`chunk_local` against the same architectural constant, not on the tile:

```123:127:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/zero_padded_kv_cache/device/zero_padded_kv_cache_device_operation.cpp
    TT_FATAL(
        chunk_local % TILE_HEIGHT == 0,
        "chunk_local ({}) must be tile-aligned (multiple of {})",
        chunk_local,
        TILE_HEIGHT);
```

A `chunk_local` of 32 satisfies it regardless of whether that is one 32-row tile or two 16-row tiles.

**Two-call reproduction.**

- **Call 1:** `cache` `BFLOAT16`, `Layout::TILE`, `Tile{32, 32}`, interleaved DRAM, padded
  `[1, 1, 128, 128]`; `layer_idx=0, num_layers=61, cluster_axis=1, chunk_size_global=128, pad_align=32`,
  scalar path. Miss; the program compiles with `Wt = 4`, `cache_H_pages = 16`, `cache_CH_pages = 16`,
  `cache_tile_size = 2048`, four CBs paged at 2048 bytes, and reader/writer compile-time arg 2 = 2048.
- **Call 2:** identical in every hashed respect — same dtype, same `Layout::TILE`, same memory config,
  same padded shape, same attributes — but the cache carries `Tile{16, 32}`. The `Tile` lives inside
  `page_config`, and   `compute_program_hash` hashes only `cache.dtype/layout/memory_config/padded_shape`
  (`compute_program_hash:227-230`), so the key is byte-identical and the cache hits.
  `validate_on_program_cache_hit` runs `validate_runtime_args`, which contains no tile check.
- **Stale slots:** reader compile-time arg 2 and writer compile-time arg 2 (`cache_tile_size`) stay
  2048 where the real page is 1024 bytes; the `src`, `out` and `zero` CBs stay paged and sized at 2048
  bytes; common args 7 and 8 (`Wt`, `cache_CH_pages`) stay at the values produced by dividing by 32,
  which is half the true page count; and the cache `TensorAccessorArgs`' compile-time
  `aligned_page_size` stays 2048, disagreeing with the buffer it is now pointed at.
- **Symptom:** the writer indexes pages by a stale `cache_CH_pages`, so the pad window lands at the
  wrong sequence position within the slab, and every page write moves 2048 bytes into a 1024-byte page,
  running over into the following page. Because this op's whole purpose is to *zero* a region, the
  observable result is that live KV data outside the pad window is destroyed — the most damaging
  possible failure for an in-place cache op, and it happens with no cache miss to hint at the cause.

**This defect is family-wide.** `update_padded_kv_cache` has the identical shape — `tt::tile_size` at
its line 276, `Wt`/`input_Ht`/`cache_HtWt` via bare constants at 277-279, `writer_tile_height =
TILE_HEIGHT` at 280, and no tile guard. `rotary_embedding_indexed` hardcodes 32x32 just as thoroughly
(five `tt::tile_size` calls and four bare-constant tile-count conversions) and also has no guard, yet
its verdict is only CAVEAT — purely because it dispatches through Metal 2.0 `UpdateProgramRunArgs`,
whose exact `TensorSpec` equality check covers `page_config` and therefore throws on the mismatched
second call instead of executing it. This op goes through the descriptor buffer-binding fast path
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

Making the factory tile-aware instead is a much larger change here than in the two siblings, because
the 32x32 assumption also lives in the kernel sources (the four-face mask loop above and
`tile_height = 32` in the shared header) — and it would additionally require adding `page_config` to
`compute_program_hash`, since the program would then provably vary with `Tile`.

### 6. `cache.tensor_layout().get_alignment()`

**Verdict: CAVEAT.** Not read as a tensor property, but it moves `aligned_page_size`, which is a
compile-time arg.

Alignment enters through the `TensorAccessorArgs(cache.buffer())` door at lines 307, 352 and 385, plus
the ROW_MAJOR `row_page_size` at line 291, which sizes the zero CB and is a compile-time arg to the
ROW_MAJOR writer (line 306). Compile-time args and CB sizes are baked into the cached `Program` and are
refreshed by no cache-hit path.

This is safe today because every caller builds the cache with the canonical alignment for its buffer
type, which the hashed {`dtype`, `layout`, `memory_config`, `padded_shape`} then fully determines. What
would break it is a `TensorLayout` constructed with an explicit non-canonical `Alignment` that leaves
those four unchanged. Unlike the tile in omission 5, no supported TTNN configuration reaches that
today, which is why this half stays a caveat.

### 7. `cache.storage` variant kind

**Verdict: CAVEAT — pinned only on the miss path.** The pin exists, but it is one of the checks the
hit validator drops.

```157:158:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/zero_padded_kv_cache/device/zero_padded_kv_cache_device_operation.cpp
    TT_FATAL(cache.storage_type() == StorageType::DEVICE, "cache must be on device");
    TT_FATAL(cache.buffer()->buffer_type() == BufferType::DRAM, "zero_padded_kv_cache requires a DRAM-backed cache");
```

The cache is constrained to a single storage kind *and* a single buffer type, so neither carries
information on the first call. But both `TT_FATAL`s sit above the `validate_runtime_args` delegation at
`:194`, so under the dispatcher branch quoted in `## Cache-hit patch mechanism` they run once and never
again.

The severity is low. A host-storage cache has no `buffer()`, so the hit path faults immediately in
`collect_tensor_buffers` when it tries to collect an address rather than executing a stale program; and
the buffer type is independently carried by the hashed `cache.memory_config()`, so a DRAM-to-L1 change
produces a genuine cache miss rather than a wrong hit. The grade is a caveat on structural grounds
rather than a live hazard — but it must be graded the same way as omission 5's alignment half and as
`update_padded_kv_cache`'s identical storage row, because the underlying weakness is the same.

This document deliberately does **not** recommend closing it. The fix would be to move `:157` into
`validate_runtime_args`, upgrading the verdict to `VALID — pinned by validation` — but
`validate_runtime_args` runs on the cache-hit path, which is the fast path, so that `storage_type()`
query would be paid on every dispatch for the life of the process. What it buys is a clearer error
message in front of a fault that already occurs on the same call. That is not a good trade; see
recommendation 4.

The observation this subsection previously rested on still holds and is worth keeping: the contrast
with omission 4 is that the op knows how to pin a buffer type when it wants to — it just never applied
the same discipline to the metadata tensors, where the buffer type genuinely does reach a baked
compile-time arg. That contrast is now sharper, not weaker: the one buffer-type pin the op does have is
also the one that stops running after the first call.

### 8. Buffer address of `cache`

**Verdict: VALID — patched, and required.** Registered as a `Buffer*` binding on every kernel —
`create_descriptor:310` (ROW_MAJOR writer), `:360` (reader), `:393` (writer) — which is what puts the
inner adapter on the fast path. The compute kernel takes a literal `0u` placeholder
(`:370`) because it reads no addresses.

On in-place aliasing specifically: the cache is simultaneously `tensor_args.cache` (input region) and
the value `create_output_tensors` returns (output region), so it appears twice in
`collect_tensor_buffers`. The resolver treats that as the safe in-place case rather than the ambiguous
`matmul(X, X)` duplicate, and does not bail:

```90:94:tt_metal/impl/program/program_descriptor_patching.cpp
            const bool is_input = i < num_input_buffers;
            // An output/workload buffer that aliases an input is the safe in-place case — skip it.
            if (!is_input && input_buffers.contains(buf)) {
                continue;
            }
```

Worth being precise about, since the op has `override_runtime_arguments`: that hook is on the *outer*
`MeshWorkloadFactory`, and it delegates addresses to the inner adapter rather than re-deriving them.
So this op does **not** bypass `resolve_bindings` the way a hand-rolled mode-A op does; its aliasing
safety comes from the resolver's output-region skip quoted above. The metadata tensors deliberately do
not use `Buffer*` bindings — their addresses ride in common args 10/11
(`create_descriptor:245-247,284-285`), which is exactly why the op must patch them by hand.

### 9. `my_sp_coord`, `sp_factor`, `chunk_local` (derived, not attributes)

**Verdict: VALID — invariant.** `sp_factor` is the mesh extent along the hashed `cluster_axis`,
`chunk_local = chunk_size_global / sp_factor` is derived from a hashed attribute, and `my_sp_coord`
comes from the dispatch coordinate (`create_descriptor:255-258`). Coordinates are folded into the key
by the framework for both the default and custom paths
(`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:989-992`), and the program cache is per-device, so a
program can never be reused at a different mesh position. These three sit in unpatched common args
0/1/2, so this argument is load-bearing rather than incidental.

## Keys the custom hash adds beyond the default

- `cache.padded_shape()` — a derivation in the default key, promoted to first-class. This is what makes
  dropping `cache.logical_shape()` safe.
- `cache.layout()` — a lossy projection of `page_config`; see omission 5.
- `tensor_args.slot_idx.has_value()` — a lossy projection of the two optional tensors, not an addition;
  see omission 4.

## Framework side effect of having a custom hash

Defining `compute_program_hash` opts this op out of attribute-level collision resolution:

```1012:1014:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
        if constexpr (requires { DeviceOperation::compute_program_hash(attrs, tensor_args); }) {
            return key;  // custom hash -> opt out beyond the op-identity prefix
        } else {
```

`ProgramCacheKey::canonical` degrades to the op type name, so a 64-bit collision between two different
configurations resolves to a wrong hit instead of a rebuild. This op's whole purpose is to destroy data
in a shared KV cache at a computed offset, so a wrong hit means zeroing the wrong region — the least
observable and most damaging failure mode available. That raises the cost of every gap above.

## Summary

| Omitted vs. default | Used by program? | Patched on hit? | Verdict |
|---|---|---|---|
| `attrs.valid_global` | Yes (common arg 3) | Yes (override) | VALID — patched |
| `attrs.slot_idx` | Yes (common arg 9) | Yes (override) | VALID — patched |
| `cache.logical_shape` | No (padded shape used) | n/a | VALID — relaxation win |
| metadata tensors' `memory_config` / shardedness / `alignment` | Yes (accessor compile-time args) | No (compile-time) | BUG |
| `cache.page_config` (`Tile`) | Yes (`cache_tile_size` compile-time arg, CB page sizes, `Wt` / `cache_CH_pages`) | No (compile-time) | BUG |
| `cache.alignment` | Only via `aligned_page_size` | No | CAVEAT |
| `cache.storage` kind | n/a | n/a | CAVEAT — pinned only on the miss path |
| `cache` buffer address | Yes | Yes (`resolved_bindings`) | VALID — patched, required |
| `my_sp_coord`, `sp_factor`, `chunk_local` | Yes (common args 0-2) | n/a (coordinate / hashed attrs) | VALID — invariant |

**Two program-cache bugs were found.** The two moving per-request indices behave correctly:
`valid_global` and `slot_idx` are omitted from the hash, re-applied to every kernel on every hit, and
re-validated on every hit, and the design deliberately keeps all `valid_global`-dependent work off the
host by fixing the program to one core with an unconditional CB protocol. Likewise the genuinely
structural padding parameters — `chunk_size_global` and `pad_align`, which land in unpatched common args
and would be the obvious place for this class of bug — are correctly hashed. Both defects are
compile-time-arg defects, the one category no cache-hit path can repair.

The first is the optional metadata tensors: they contribute only a `has_value()` bit to the key, yet
their buffer type, shardedness and aligned page size are compiled into the reader's and writer's
`TensorAccessorArgs`. Passing L1-allocated (or sharded) metadata tensors after a DRAM-allocated first
call produces a silent wrong hit in which the kernels read garbage indices and zero an arbitrary region
of another user's KV cache.

The second is the unguarded 32x32 tile assumption (omission 5). The op accepts `Layout::TILE`, derives
`cache_tile_size` from `tt::tile_size` and `Wt`/`cache_H_pages` from bare `TILE_WIDTH`/`TILE_HEIGHT`
(`create_descriptor:251-252`, `:315-317`), never reads the cache's actual `Tile`, never validates it,
and does not hash `page_config`. `cache_tile_size` is compile-time arg 2 of both the reader and the
writer as well as the page size of three CBs, and the reader's row-mask builder assumes the 32x32
four-face layout outright. A `Tile{16, 32}` call following an otherwise-identical `Tile{32, 32}` call
hits the cache, zeroes the wrong page range, and overruns each 1024-byte page with a 2048-byte write —
destroying live KV data in an op whose only job is to zero a bounded window. The same defect is present
in `update_padded_kv_cache` and `rotary_embedding_indexed`; it is only in `rotary_embedding_indexed`
that the Metal 2.0 dispatch path turns it into a loud throw rather than silent corruption.

A third, lower-severity finding is structural: because the op defines `validate_on_program_cache_hit`,
that validator *replaces* the miss validator on hits rather than supplementing it, and this op's hit
validator is a bare delegation to `validate_runtime_args`. Everything the miss validator checks before
its own delegation at `:194` is therefore absent on the hit path — the `storage_type()` and DRAM
buffer-type pins behind omission 7, the layout and dtype gates, the metadata-path-is-TILE-only guard,
and the rank, num-heads and `num_layers` checks. A narrow hit validator is a hazard rather than a
safeguard in general: by existing, it disables everything above it.

In this op's case, however, the practical damage is close to nil, and that is worth stating as
plainly as the hazard itself. This op hashes an unusually broad set — `cache.dtype()`, `cache.layout()`,
`cache.memory_config()`, `cache.padded_shape()`, `layer_idx`, `num_layers`, `chunk_size_global`,
`pad_align` and `slot_idx.has_value()` — and a miss-only pin on a hashed value cannot be evaded, because
any call carrying a new value of it misses and meets the pin there. Filtering the drop set against the
key leaves exactly one reachable check, the `storage_type()` pin at `:157`, and its failure mode is a
crash rather than silent corruption. The reachability table in `## Cache-hit patch mechanism` works
through this line by line. The finding is therefore a structural caveat to record rather than a defect
to fix, and recommendation 4 explains why moving the pin is not worth its per-dispatch cost.

## Recommendations

**Every guard below names the function it must go into, and for this op that function is always
`validate_runtime_args`.** Because the op defines `validate_on_program_cache_hit`, the miss validator is
skipped entirely on a hit; a guard placed in `validate_on_program_cache_miss` would not run on the
offending second call, which is the only call a cache bug reaches. `validate_runtime_args` is the right
home for all of them because both validators delegate to it (`:194`, `:199`), so one placement covers
both paths.

**And every guard below is priced.** The cache-hit path is the fast path — it is what the program cache
exists to make cheap — so a `TT_FATAL` added to `validate_runtime_args` is paid on every dispatch for
the life of the process. That is why only one new check is recommended here, and why the op's single
reachable miss-only pin is deliberately left as a documented caveat rather than fixed.

There are two distinct ways to close a miss-only pin in this op, and they are not interchangeable:

- **Targeted (recommended):** move the specific `TT_FATAL` into `validate_runtime_args`. Adds only that
  check per dispatch.
- **Wholesale (alternative):** delete `validate_on_program_cache_hit` entirely, putting the op on the
  dispatcher's substitution branch so the full miss validator runs on every hit. Simplest and safest,
  and immune to someone later adding a check to the wrong function — but it puts all of `:156-193` on
  the hot path: a `storage_type()` query, a buffer-type dereference, five layout/dtype gates, two shape
  queries and two divisibility checks, on every prefill step. For this op that is especially poor value,
  because the reachability table in `## Cache-hit patch mechanism` shows that exactly one of those lines
  can be reached on a hit at all, and its failure mode is already a crash. Prefer the targeted approach,
  which here means adding the tile guard and nothing else.

1. Hash the metadata tensors' specs. Mirror what `rotary_embedding_indexed` already does: add
   `tensor_args.slot_idx->memory_config()` and `tensor_args.valid_global->memory_config()` (with
   neutral defaults on the scalar path) to `compute_program_hash`. This is a family-wide gap — apply the
   same change to `update_padded_kv_cache`.
2. Bring this op's `validate_meta` up to `update_padded_kv_cache`'s: add the missing
   `TT_FATAL(!meta.is_sharded(), ...)`. Then go further in both ops and add
   `TT_FATAL(meta.buffer()->buffer_type() == BufferType::DRAM, ...)`, plus an assertion that the
   `slot_idx` and `valid_global` tensors share a buffer type and aligned page size — the kernels reuse
   one accessor for both reads (`create_descriptor:353-357`, `:386-390`) and nothing enforces that they
   are interchangeable, which is wrong even on a cache miss.
   **Target function:** the `validate_meta` lambda already lives inside `validate_runtime_args`
   (`:83-93`), so it is on the hit path and is the correct home as-is. This is worth stating because the
   equivalent guard must *not* go into `validate_on_program_cache_miss` — that is where this op's
   existing `buffer_type() == DRAM` pin for the *cache* lives (`:158`), and that pin consequently stops
   running after the first call. Repeating that placement for the metadata tensors would produce a guard
   that passes the DRAM-allocated first call and is then absent on the L1-allocated second call, which
   is precisely the wrong hit described in omission 4.
   **Per-dispatch cost:** two or three extra checks per metadata tensor, and only on the metadata path
   — the lambda is already called there, so this adds to an existing cost rather than creating one.
   Worth it: unlike omission 7 below, the defect these close is the silent wrong hit that is this
   document's headline BUG.
3. Reject a non-32x32 `Tile` on the TILE path, closing omission 5. Assert
   `cache.tensor_spec().tile().get_height() == TILE_HEIGHT` and the same for `get_width()`, in the same
   shape as the `interleaved_to_sharded` guard quoted in omission 5.
   **Target function:** `validate_runtime_args`, not `validate_on_program_cache_miss`. The reproduction
   in omission 5 is a *hit*, so a guard in the miss validator would let the first `Tile{32,32}` call
   through and then not run at all on the `Tile{16,32}` call that destroys live KV data. Placing it in
   `validate_runtime_args` covers the miss path too, via the delegation at `:194`.
   **Per-dispatch cost:** two `uint32_t` comparisons against constants. This is the only new hit-path
   check this document recommends, and it is the one clearly worth its price — it closes a BUG whose
   symptom is destroying live KV data in an op whose only job is to zero a bounded window.
   The mask builder in `reader_zero_padded_kv_cache.cpp:86-94` and `tile_height = 32` in
   `device/kernels/zero_padded_kv_cache_common.hpp:72` already assume 32x32 unconditionally, so the
   guard only makes an existing assumption explicit — and it makes omitting `page_config` correct by
   construction. Making the op genuinely tile-aware instead is a much larger change (it reaches into
   the kernel sources) and would require adding `page_config` to the hash in the same commit. This is a
   family-wide gap: apply the same guard to `update_padded_kv_cache` and `rotary_embedding_indexed`.
4. **Leave omission 7 as a documented CAVEAT — do not move `:157` onto the hit path.** This is a
   deliberate non-recommendation, recorded so it is not mistaken for an oversight.
   `cache.storage_type() == StorageType::DEVICE` is the op's one miss-only pin that a hit can actually
   reach, and moving it into `validate_runtime_args` would upgrade the verdict to
   `VALID — pinned by validation`. But the failure it prevents is not silent: a host-storage cache has
   no device buffer, so the hit path faults in `collect_tensor_buffers` on the same call. The moved
   check would cost a `storage_type()` query on every dispatch, for the life of the process, and would
   buy only a clearer message in front of a fault that already happens. The caveat as recorded in
   omission 7 is the correct disposition.

   Two related checks explicitly do **not** need moving, contrary to how the mechanical drop set reads:
   the DRAM buffer-type pin at `:158` is subsumed by the hashed `cache.memory_config()`, and the
   metadata-path-is-TILE-only guard at `:174-179` constrains `cache.layout()` and
   `slot_idx.has_value()`, both of which are hashed — so a (ROW_MAJOR, metadata) call has its own cache
   slot, misses on first occurrence, and is rejected by the miss validator there. Neither is reachable
   on a hit, so neither is worth its per-dispatch cost.
5. Run this op's tests under `-DTT_DESCRIPTOR_PATCHING_PARITY_CHECK`. The oracle covers runtime args and
   CB addresses only (`tt_metal/api/tt-metalium/experimental/program_descriptor_patching.hpp:176-186`),
   so it will not catch the compile-time-arg defect in omission 4 — but it will catch any regression in
   the hand-written common-arg patch, which currently has three separate index constants
   (`kValidGlobalCommonArgIdx`, `kSlotIdxCommonArgIdx`, and the pair of address indices) that must stay
   in sync with `create_descriptor`'s vector literal and with three kernels' `get_common_arg_val` calls.
6. The override selects kernels by raw handle — `{0u, 2u}` on the metadata path and `0..num_kernels` on
   the scalar path, with `num_kernels` re-derived from `cache.layout()`
   (`override_runtime_arguments:425,435`). That works because kernel handles follow descriptor push
   order (`tt_metal/impl/program/program.cpp:402-466`) and because the metadata path is validated
   TILE-only (`validate_on_program_cache_miss:174-179`), but it is a three-way coupling between the
   override, `create_descriptor`'s push order, and a layout guard in a third function. The third leg
   survives the cache-hit analysis, which is worth recording because it initially looks as though it
   does not: the TILE-only guard lives in the miss validator and so does not run on the hit at which the
   override selects kernels, but both values it constrains are hashed, so a (ROW_MAJOR, metadata) call
   cannot reach a hit without having first missed and been rejected. The coupling is sound; it is just
   fragile to read. Deriving the kernel set from the cached descriptor, or naming the handles in one
   place, would make the arrangement robust without depending on that argument.
