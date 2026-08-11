# Program Cache Audit — `experimental/deepseek_prefill/pack_scaled_fp8_kv_cache`

Audit of `PackScaledFp8KvCacheDeviceOperation::compute_program_hash` against the framework default
("hash everything") key.

| | |
|---|---|
| Device operation | `ttnn::experimental::prim::pack_scaled_fp8_kv_cache::PackScaledFp8KvCacheDeviceOperation` (`device/pack_scaled_fp8_kv_cache_device_operation.hpp:13`) |
| Custom hash | `device/pack_scaled_fp8_kv_cache_device_operation.cpp:100-108` |
| `operation_attributes_t` | `PackScaledFp8KvCacheParams` — `output_memory_config` (single field) |
| `tensor_args_t` | `PackScaledFp8KvCacheInputs` — `latent`, `scales`, `rope` |
| Program factories | one: `PackScaledFp8KvCacheProgramFactory` (legacy `CachedProgram` / `create` + `override_runtime_arguments`) |
| `override_runtime_arguments` | **Yes** (`device/pack_scaled_fp8_kv_cache_program_factory.cpp:75-87`) |
| `get_dynamic_runtime_args` | **No** |
| Cache-hit patch mechanism | **Op-owned override** — the factory re-derives per-dispatch state itself; `resolve_bindings` is never involved |

## Cache-hit patch mechanism

This is a `CachedProgram`-style factory, so the mesh adapter wraps it and calls the factory's
`override_runtime_arguments` once per program on every hit:

```285:304:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
        static void override_runtime_arguments(
            cached_mesh_workload_t& cached_workload,
            const operation_attributes_t& attrs,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value) {
            ProgramFactory program_factory;

            for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
                auto& shared_variables = cached_workload.shared_variables.at(coordinate_range);

                mesh_device_operation_utils::apply_override_runtime_arguments(
                    program_factory,
                    program,
                    shared_variables,
                    attrs,
                    *(coordinate_range.begin()),
                    tensor_args,
                    tensor_return_value);
            }
        }
```

Because the op supplies `override_runtime_arguments`, the descriptor-era binding machinery does not
apply: there is no `resolve_bindings` call, no `ResolvedBindings`, no `allow_inplace_output_tensor_alias`
decision, and no possibility of a bail-to-slow-path. The op owns its cache-hit re-derivation
entirely. This op also allocates a fresh output tensor rather than writing in place
(`device/pack_scaled_fp8_kv_cache_device_operation.cpp:95-98`), so there is no input/output aliasing
to reason about in the first place.

**Obligation on the hash.** "Has an override" is necessary but not sufficient — the override must
re-apply *every* arg that depends on an omitted parameter, and this one is deliberately partial:

```75:87:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/pack_scaled_fp8_kv_cache/device/pack_scaled_fp8_kv_cache_program_factory.cpp
void PackScaledFp8KvCacheProgramFactory::override_runtime_arguments(
    cached_program_t& cached,
    const PackScaledFp8KvCacheParams&,
    const PackScaledFp8KvCacheInputs& args,
    Tensor& output) {
    for (const auto& core : cached.shared_variables.cores) {
        auto& runtime_args = tt::tt_metal::GetRuntimeArgs(cached.program, cached.shared_variables.kernel_id, core);
        runtime_args[0] = args.latent.buffer()->address();
        runtime_args[1] = args.scales.buffer()->address();
        runtime_args[2] = args.rope.buffer()->address();
        runtime_args[3] = output.buffer()->address();
    }
}
```

The kernel takes six runtime args. Slots `[0..3]` are the four buffer addresses and are re-applied.
Slots `[4]` (`start_row`) and `[5]` (`num_rows`) are set only in `create` and are never re-set:

```59:69:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/pack_scaled_fp8_kv_cache/device/pack_scaled_fp8_kv_cache_program_factory.cpp
        SetRuntimeArgs(
            program,
            kernel_id,
            core,
            {latent_buffer->address(),
             scale_buffer->address(),
             rope_buffer->address(),
             output_buffer->address(),
             start_row,
             core_rows});
        start_row += core_rows;
```

So `start_row`, `core_rows`, the entire compile-time arg vector, the CB size, and the core grid are
frozen at the first miss and must be pure functions of the hashed set. The rest of this audit checks
exactly that.

**Which validator runs on a cache hit.** Most verdicts below rest on a `TT_FATAL` rather than on the
hash, so it matters exactly which validator executes on the offending second call. The dispatcher
runs one, not both:

```262:266:ttnn/api/ttnn/device_operation.hpp
    if constexpr (HasValidateOnProgramCacheHit<mesh_device_operation_t>) {
        mesh_device_operation_t::validate_on_program_cache_hit(operation_attributes, tensor_args);
    } else {
        mesh_device_operation_t::validate_on_program_cache_miss(operation_attributes, tensor_args);
    }
```

This op **defines** `validate_on_program_cache_hit`, so it takes the first branch and the hit
validator *replaces* the miss validator on every hit. That is a hazard in general — a hit validator
that pins less than the miss validator silently loses the difference on exactly the calls that reuse
a cached program — but here it is a total delegation, so the hit path drops nothing:

```73:76:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/pack_scaled_fp8_kv_cache/device/pack_scaled_fp8_kv_cache_device_operation.cpp
void PackScaledFp8KvCacheDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    validate_on_program_cache_miss(attrs, args);
}
```

Every `TT_FATAL` reached from `validate_on_program_cache_miss` therefore runs on hits too, which is
what licenses the "VALID — pinned by validation" verdicts in items 1 through 4.

## Baseline: what the default hash would cover

`hash_objects_with_default_seed(type_hash<PackScaledFp8KvCacheDeviceOperation>, attrs, tensor_args)`
would produce:

| Source | Fields |
|---|---|
| `operation_attributes` | `output_memory_config` |
| `latent` | storage kind; `logical_shape`; `dtype`; `page_config`; `memory_config`; `alignment` |
| `scales` | storage kind; `logical_shape`; `dtype`; `page_config`; `memory_config`; `alignment` |
| `rope` | storage kind; `logical_shape`; `dtype`; `page_config`; `memory_config`; `alignment` |

The output tensor is created by the op (`create_output_tensors`), not passed in, so it is outside
both keys; its spec is a pure function of `latent.logical_shape()` and `attrs.output_memory_config`
(`device/pack_scaled_fp8_kv_cache_device_operation.cpp:78-93`).

## What the custom hash covers

```100:108:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/pack_scaled_fp8_kv_cache/device/pack_scaled_fp8_kv_cache_device_operation.cpp
ttsl::hash::hash_t PackScaledFp8KvCacheDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    return tt::tt_metal::operation::hash_operation<PackScaledFp8KvCacheDeviceOperation>(
        attrs,
        args.latent.memory_config(),
        args.scales.memory_config(),
        args.rope.memory_config(),
        args.latent.logical_shape());
}
```

`attrs` is passed whole, so `output_memory_config` is fully covered. Of the three input tensors,
only the memory configs survive, plus `latent`'s logical shape. Everything else about all three
tensors — dtypes, layouts, alignments, storage kinds, and both `scales.logical_shape()` and
`rope.logical_shape()` — is dropped.

## Omitted parameters

### 1. `scales.logical_shape()` and `rope.logical_shape()`

**Verdict: VALID — pinned by validation.**

Both are total functions of the hashed `latent.logical_shape()`. The last dim of each is fixed to a
compile-time constant, and the leading dims are forced to match `latent`:

```26:35:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/pack_scaled_fp8_kv_cache/device/pack_scaled_fp8_kv_cache_device_operation.cpp
void validate_input(const Tensor& tensor, const char* name, tt::tt_metal::DataType dtype, uint32_t width) {
    TT_FATAL(tensor.storage_type() == ttnn::StorageType::DEVICE, "{} must be on device", name);
    TT_FATAL(tensor.buffer() != nullptr, "{} must have a buffer", name);
    TT_FATAL(tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR, "{} must be ROW_MAJOR", name);
    TT_FATAL(is_dram_interleaved(tensor.memory_config()), "{} must be DRAM interleaved", name);
    TT_FATAL(tensor.dtype() == dtype, "{} has the wrong dtype", name);
    TT_FATAL(!tensor.logical_shape().empty(), "{} must have at least one dimension", name);
    TT_FATAL(
        tensor.logical_shape()[-1] == width, "{} last dim must be {}, got {}", name, width, tensor.logical_shape()[-1]);
}
```

```58:69:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/pack_scaled_fp8_kv_cache/device/pack_scaled_fp8_kv_cache_device_operation.cpp
    const auto& shape = args.latent.logical_shape();
    TT_FATAL(
        shape.size() == args.scales.logical_shape().size() && shape.size() == args.rope.logical_shape().size(),
        "all inputs must have the same rank");
    uint64_t rows = 1;
    for (size_t dim = 0; dim + 1 < shape.size(); ++dim) {
        TT_FATAL(
            shape[dim] == args.scales.logical_shape()[dim] && shape[dim] == args.rope.logical_shape()[dim],
            "all inputs must have identical leading shapes");
        rows *= static_cast<uint64_t>(shape[dim]);
        TT_FATAL(rows <= std::numeric_limits<uint32_t>::max(), "folded row count exceeds uint32_t");
    }
```

`validate_input` is called with the fixed widths `LATENT_WIDTH`, `SCALE_WIDTH`, `ROPE_WIDTH`
(`device/pack_scaled_fp8_kv_cache_device_operation.cpp:46-50`), which are `constexpr` in
`pack_scaled_fp8_kv_cache.hpp:11-13`. Given rank equality, identical leading dims, and a pinned last
dim, `scales.logical_shape()` and `rope.logical_shape()` are uniquely determined by
`latent.logical_shape()` and carry no additional information.

This pinning holds on cache hits as well as misses, which matters:

```73:76:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/pack_scaled_fp8_kv_cache/device/pack_scaled_fp8_kv_cache_device_operation.cpp
void PackScaledFp8KvCacheDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    validate_on_program_cache_miss(attrs, args);
}
```

Without that, a validation-based argument would be worth nothing on the hit path — an omitted value
would only be checked the one time the program was built.

### 2. `latent.dtype()`, `scales.dtype()`, `rope.dtype()`

**Verdict: VALID — pinned by validation.**

Each is fixed to a single value by the `validate_input` call site:
`FP8_E4M3` for `latent`, `FLOAT32` for `scales`, `BFLOAT16` for `rope`
(`device/pack_scaled_fp8_kv_cache_device_operation.cpp:46-50`, checked at line 31 of the helper
quoted above). Note the factory does not use `element_size()` anywhere; the per-field byte counts
are computed from the `constexpr` widths, not from the dtypes:

```41:46:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/pack_scaled_fp8_kv_cache/device/pack_scaled_fp8_kv_cache_program_factory.cpp
    std::vector<uint32_t> compile_args = {
        cb_scratch, packed::LATENT_WIDTH, packed::SCALE_WIDTH * sizeof(float), packed::ROPE_WIDTH * sizeof(uint16_t)};
    TensorAccessorArgs(latent_buffer).append_to(compile_args);
    TensorAccessorArgs(scale_buffer).append_to(compile_args);
    TensorAccessorArgs(rope_buffer).append_to(compile_args);
    TensorAccessorArgs(output_buffer).append_to(compile_args);
```

So the dtypes reach the program only indirectly, via each buffer's page size inside
`TensorAccessorArgs` — and they are pinned, so that path is closed.

### 3. `latent`/`scales`/`rope` `page_config` (i.e. `layout()` and `Tile`)

**Verdict: VALID — pinned by validation.** All three are forced to `Layout::ROW_MAJOR`
(`device/pack_scaled_fp8_kv_cache_device_operation.cpp:29`). For a row-major tensor the `Tile`
descriptor does not participate in the page size, and neither the factory nor the kernel reads
`tensor_spec().tile()`. There is no tile-layout program variant at all — `select_program_factory`
returns the single factory unconditionally
(`device/pack_scaled_fp8_kv_cache_device_operation.cpp:39-42`). Item 10 records the search that
backs the "neither reads the tile" claim.

### 4. `latent`/`scales`/`rope` storage variant kind

**Verdict: VALID — pinned by validation.** `TT_FATAL(tensor.storage_type() == StorageType::DEVICE, ...)`
in `validate_input`. Constant across every admissible call.

### 5. `latent`/`scales`/`rope` `TensorLayout::Alignment`

**Verdict: CAVEAT.**

This is the one omission with a residual path into a frozen compile-time arg. For a non-sharded
buffer `TensorAccessorArgs::append_to` emits the config bitset and the aligned page size:

```194:198:tt_metal/impl/buffers/tensor_accessor_args.cpp
    if (args_config_.test(tensor_accessor::ArgConfig::Sharded)) {
        CMAKE_UNIQUE_NAMESPACE::append_sharded_args(*buffer_, args_config_, compile_time_args, /* is_runtime */ false);
    } else {
        compile_time_args.push_back(args_config_.raw());
        auto aligned_page_size = buffer_ ? buffer_->aligned_page_size() : 0;
```

and

```656:658:tt_metal/impl/buffers/buffer.cpp
uint32_t Buffer::alignment() const { return allocator_->get_alignment(this->buffer_type()); }

DeviceAddr Buffer::aligned_page_size() const { return align(page_size(), this->alignment()); }
```

`Buffer::alignment()` is an allocator constant selected by `buffer_type` (hashed, and pinned to DRAM
by `is_dram_interleaved`), so the rounding step is covered. `page_size()`, however, derives from the
*padded* shape, and the padded shape is a function of the logical shape **and** the
`TensorLayout::Alignment`. The hash keeps `latent.logical_shape()` but not any tensor's `Alignment`.

What would break it: a `latent` (or `scales`/`rope`) tensor constructed with a non-canonical
`Alignment` that widens the padded last dim while leaving the logical last dim at the validated
value. `validate_input` checks `logical_shape()[-1]`, never the padded width, so such a tensor is
accepted. Two calls differing only in that alignment collide in the hash, and the second inherits
the first's `aligned_page_size` compile-time arg — the kernel then strides through DRAM pages at the
wrong pitch and packs rows from the wrong offsets.

Nothing in the codebase produces such a tensor today: every caller reaches this op through
`ttnn.to_layout(..., ROW_MAJOR_LAYOUT)` outputs and the `per_token_cast_to_fp8` outputs
(`models/demos/deepseek_v3_d_p/utils/kv_cache_utils.py:167-177`), all built with canonical
alignments. The guard that closes it is either hashing `latent.padded_shape()` alongside the logical
shape (cheapest, and consistent with what the sibling `moe_padding_config` does), or a `TT_FATAL`
in `validate_input` asserting `tensor.padded_shape()[-1] == width`.

### 6. Buffer addresses (omitted by the default hash too)

**Verdict: VALID — patched, and required.** All four addresses (three inputs plus the op-allocated
output) are re-applied on every hit at
`device/pack_scaled_fp8_kv_cache_program_factory.cpp:82-85`, quoted above. Addresses must never be
hashed; that is the point of the cache.

### 7. `start_row` / `core_rows` — runtime args set only in `create`

**Verdict: VALID — invariant** (not an independent parameter; fully determined by the hashed set).

Not an omitted *parameter* strictly, but the classic incomplete-override failure mode, so it is
worth proving rather than assuming. Both derive from `rows` and the compute grid:

```26:30:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/pack_scaled_fp8_kv_cache/device/pack_scaled_fp8_kv_cache_program_factory.cpp
    const uint32_t rows = args.latent.logical_volume() / packed::LATENT_WIDTH;

    Program program;
    const auto grid = args.latent.device()->compute_with_storage_grid_size();
    auto [num_cores, all_cores, group_1, group_2, rows_group_1, rows_group_2] = split_work_to_cores(grid, rows);
```

`logical_volume()` is a pure function of `logical_shape()` (hashed), `LATENT_WIDTH` is `constexpr`,
and `compute_with_storage_grid_size()` is a device property — the program cache is per-device, so it
already partitions on it. `split_work_to_cores` is deterministic in `(grid, rows)`. Therefore
`num_cores`, `all_cores`, the per-core `core_rows` and the running `start_row` are all uniquely
determined by the hashed key. The same argument covers the core grid baked into the cached
`Program`, which `override_runtime_arguments` could not refresh even if it wanted to.

### 8. No position / sequence-offset / batch-index parameter exists

**Verdict: n/a — worth recording explicitly.**

The op name suggests an in-place KV-cache update at a decode position, which is the shape of
parameter that most often goes stale behind a program-cache hit. That is not what this op does. It
is a pure packer: it reads row `r` of each of `latent`, `scales`, `rope` and writes the three fields
contiguously into row `r` of a **newly allocated** output tensor, with no offset argument anywhere:

```37:47:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/pack_scaled_fp8_kv_cache/device/kernels/dataflow/pack_scaled_fp8_kv_cache.cpp
    for (uint32_t row = start_row; row < start_row + num_rows; ++row) {
        scratch.reserve_back(1);
        noc.async_read(latent, scratch, latent_bytes, {.page_id = row}, {.offset_bytes = 0});
        noc.async_read_barrier();
        noc.async_write(
            use<CircularBuffer::AddrSelector::WRITE_PTR>(scratch),
            output,
            latent_bytes,
            {.offset_bytes = 0},
            {.page_id = row});
        noc.async_write_barrier();
```

The `operation_attributes_t` has exactly one field (`output_memory_config`), so there is no
candidate index to omit. The actual write of the packed rows into a cache slot at a position is done
downstream by a different op; the caller here just receives the packed tensor
(`models/demos/deepseek_v3_d_p/utils/kv_cache_utils.py:177-192`).

### 9. FP8 scale values

**Verdict: VALID — invariant.** The per-token scales arrive as `scales`, a `FLOAT32` device tensor,
and the kernel copies its bytes verbatim into the packed row's scale field
(`device/kernels/dataflow/pack_scaled_fp8_kv_cache.cpp:49-57`). No scale value is read host-side, so
none can reach a runtime or compile-time arg. This is data, not structure, and correctly absent from
both the default key and this one. The structural counterpart — the number of scales per row — is
the `constexpr SCALE_WIDTH`, which is baked into the compile-time args and cannot vary at all.

### 10. Tile geometry — the tile-awareness check

**Verdict: VALID — unused.** This check was performed rather than skipped: there is no host-side tile
math anywhere in this op, so neither form of the tile hazard applies. Item 3 records the omission
itself; this subsection records the search that justifies it.

Neither the device operation, the program factory nor the kernel calls `tt::tile_size(...)`,
`tensor_spec().tile()`, `get_tile_shape()` or `get_face_shape()`, and none of them uses
`tt::constants::TILE_HW` / `TILE_WIDTH` / `TILE_HEIGHT` to convert a shape into a tile count. The
factory includes `<tt-metalium/constants.hpp>`
(`device/pack_scaled_fp8_kv_cache_program_factory.cpp:6`) but reads nothing from it. Every size in
the program comes from the fixed packed-row constants instead:

```33:46:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/pack_scaled_fp8_kv_cache/device/pack_scaled_fp8_kv_cache_program_factory.cpp
    constexpr uint32_t cb_scratch = CBIndex::c_0;
    constexpr uint32_t scratch_bytes = packed::LATENT_WIDTH;
    CreateCircularBuffer(
        program,
        all_cores,
        CircularBufferConfig(scratch_bytes, {{cb_scratch, DataFormat::UInt8}})
            .set_page_size(cb_scratch, scratch_bytes));

    std::vector<uint32_t> compile_args = {
        cb_scratch, packed::LATENT_WIDTH, packed::SCALE_WIDTH * sizeof(float), packed::ROPE_WIDTH * sizeof(uint16_t)};
    TensorAccessorArgs(latent_buffer).append_to(compile_args);
    TensorAccessorArgs(scale_buffer).append_to(compile_args);
    TensorAccessorArgs(rope_buffer).append_to(compile_args);
    TensorAccessorArgs(output_buffer).append_to(compile_args);
```

The single CB is one `LATENT_WIDTH`-byte `UInt8` page, and the work split is over `rows`, computed
as `latent.logical_volume() / packed::LATENT_WIDTH`
(`device/pack_scaled_fp8_kv_cache_program_factory.cpp:26, 29-30`) — a row count, never a tile count.

**The `aligned_page_size` compile-time arg does not carry tile geometry either**, which is the one
place it could have slipped in indirectly (see the CAVEAT in item 5, where that same arg is the
carrier for the *alignment* gap). `Buffer::aligned_page_size()` rounds `page_size()`, and `page_size`
is tile-derived only on the `TilePageConfig` branch:

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

All three inputs are pinned to ROW_MAJOR by `validate_input`
(`device/pack_scaled_fp8_kv_cache_device_operation.cpp:29`) and the output is constructed with
`PageConfig(Layout::ROW_MAJOR)` explicitly
(`device/pack_scaled_fp8_kv_cache_device_operation.cpp:87-92`), so all four accessors take the
row-major branch, where the page size is `height * width * element_size` with no `Tile` involved.
The alignment CAVEAT in item 5 therefore stands exactly as written — it is an alignment gap, not a
tile gap.

## Keys the custom hash adds beyond the default

None. The custom key is a strict subset of the default key's fields (the default would additionally
cover the storage kinds, dtypes, page configs, alignments, and the two derived shapes).

## Framework side effect of having a custom hash

```1012:1014:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
        if constexpr (requires { DeviceOperation::compute_program_hash(attrs, tensor_args); }) {
            return key;  // custom hash -> opt out beyond the op-identity prefix
        } else {
```

Because this op defines `compute_program_hash`, `ProgramCacheKey::canonical` degrades to just the op
type name, so a 64-bit collision between two different configurations becomes a wrong hit rather
than a rebuild. Inherent to every custom-hash op, but it raises the stakes on the alignment CAVEAT.

## Summary

| Omitted param | Used by program? | Patched on hit? | Verdict |
|---|---|---|---|
| `scales.logical_shape`, `rope.logical_shape` | Only via pinned equalities | n/a | VALID — pinned by validation |
| `latent`/`scales`/`rope` `dtype` | Only via buffer page size | n/a | VALID — pinned by validation |
| `latent`/`scales`/`rope` `page_config` | No | n/a | VALID — pinned by validation |
| `latent`/`scales`/`rope` storage kind | n/a | n/a | VALID — pinned by validation |
| `latent`/`scales`/`rope` `alignment` | Yes (`aligned_page_size` compile-time arg) | No | CAVEAT — unpinned canonical-alignment assumption |
| Buffer addresses (4) | Yes | Yes (override slots 0-3) | VALID — patched |
| `start_row` / `core_rows` (create-only rt args) | Yes | No | VALID — invariant (functions of the hashed set) |
| FP8 scale values | Yes, as data | n/a | VALID — invariant |
| Tile geometry (no host-side tile math) | No — op has none | n/a | VALID — unused |

**No program-cache correctness bug found.** The override is partial by design, but the two runtime
args it leaves alone (`start_row`, `core_rows`) and every compile-time arg are pure functions of
{`output_memory_config`, `latent.memory_config`, `scales.memory_config`, `rope.memory_config`,
`latent.logical_shape`} plus values pinned by a validator that runs on hits as well as misses, plus
device-fixed constants. The one live assumption is that no caller supplies a non-canonically-aligned
input (item 5).

The tile check (item 10) was performed and found nothing to adjudicate: this op does no host-side
tile math at all, so it can neither hardcode a 32x32 tile against a non-32x32 tensor nor vary its
program with a tile that is missing from the key. In particular the `aligned_page_size` compile-time
arg that carries the item-5 CAVEAT is fed by the row-major page-size path, which does not read the
`Tile`.

Worth noting relative to the family: the sibling `per_token_cast_back` closes the same alignment
question by hashing each input's whole `TensorSpec` rather than cherry-picking fields. Two of the
four `deepseek_prefill` ops audited here (this one and `per_token_cast_to_fp8`) share the identical
gap — they hash `logical_shape` but consume an `aligned_page_size` derived from the padded shape.
Since it is a family pattern rather than a one-off, the fix belongs at the family level.

## Recommendations

1. Close the alignment CAVEAT by adding `args.latent.padded_shape()` to `compute_program_hash`, or —
   better for the family — switch to hashing `args.latent.tensor_spec()` and dropping the three
   separate `memory_config()` calls, matching what `per_token_cast_back` already does. The cost is
   nil: every field a `TensorSpec` adds is already pinned by `validate_input`, so no additional cache
   entries can result for legal inputs.
2. `scales` and `rope` need no hash contribution at all beyond their memory configs, and even those
   are pinned to DRAM interleaved by `is_dram_interleaved`. If the alignment fix in (1) is applied to
   `latent`, apply it to those two as well rather than leaving three tensors hashed at three
   different levels of detail — the asymmetry is what makes this hash hard to review.
3. Run the op's unit test
   (`tests/ttnn/nightly/unit_tests/operations/experimental/deepseek_prefill/test_pack_scaled_fp8_kv_cache.py`)
   under `-DTT_DESCRIPTOR_PATCHING_PARITY_CHECK` if/when this factory migrates to the descriptor
   path. It is not covered today: `assert_fastpath_parity` is wired into the descriptor adapter only
   (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:679-693`), and this op still uses the legacy
   `CachedProgram` factory, so an incomplete `override_runtime_arguments` here would not be caught by
   that oracle. A cheap substitute is a test that calls the op twice with different row counts on the
   same program-cache-enabled device and checks both results.
