# Program Cache Audit — `data_movement/sharded_partial/interleaved_to_sharded_partial`

Audit of `ttnn::prim::InterleavedToShardedPartialDeviceOperation::compute_program_hash` against
the framework default ("hash everything") key.

| | |
|---|---|
| Device operation | `ttnn::prim::InterleavedToShardedPartialDeviceOperation` (`device/interleaved_to_sharded_partial_op.hpp:14`) |
| Custom hash | `device/interleaved_to_sharded_partial_op.cpp:77-92` |
| `operation_attributes_t` | `InterleavedToShardedPartialParams` — `grid_size`, `shard_spec`, `num_slices`, `slice_index`, `output_mem_config`, `output_dtype` |
| `tensor_args_t` | `Tensor` (the input tensor itself, not a wrapper struct) |
| Program factories | one: `InterleavedToShardedPartialProgramFactory` (`ProgramDescriptor`-based) |
| `override_runtime_arguments` | **Yes**, on the device operation (`device/interleaved_to_sharded_partial_program_factory.cpp:434`). Hand-written targeted patch, not a rebuild. |
| `get_dynamic_runtime_args` | **No** — no such member exists (see CSV note below) |
| Own cache-hit validator | No — the framework substitutes `validate_on_program_cache_miss` |
| Cache-hit patch mechanism | **Op-owned cache-hit re-derivation** (mode A) |

## Cache-hit patch mechanism

The op defines `override_runtime_arguments`, so the adapter selects the op-owned branch and
bypasses `resolve_bindings` and `get_dynamic_runtime_args` entirely. Because the hook lives on the
device operation rather than the factory struct, the `else` arm at line 671 is the one taken:

```657:678:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
                } else if constexpr (has_override_runtime_arguments()) {
                    // ProgramDescriptor variant, op owns its cache-hit re-derivation (the descriptor-era
                    // override_runtime_arguments()): re-apply ALL per-dispatch state — every runtime arg
                    // AND every tensor-backed CB address — for the current tensors.  No resolve_bindings
                    // (address inference) and no get_dynamic; correct by construction for in-place,
                    // mixed-aliasing, and work-set shifts. Prefer the factory's hook; fall back to the
                    // DeviceOperation for direct ops that predate the factory-struct shape.
                    if constexpr (factory_has_override_runtime_arguments()) {
                        DescriptorFactory::override_runtime_arguments(
                            program,
                            attrs,
                            tensor_args,
                            tensor_return_value,
                            std::optional<ttnn::MeshCoordinate>(coordinate_range.start_coord()));
                    } else {
                        DeviceOperation::override_runtime_arguments(
                            program,
                            attrs,
                            tensor_args,
                            tensor_return_value,
                            std::optional<ttnn::MeshCoordinate>(coordinate_range.start_coord()));
                    }
```

Mode A's "re-apply ALL per-dispatch state" guarantee is a description of what a *correct* override
does, not something the framework enforces. This override is deliberately **partial** — it patches
two runtime-arg slots per core plus the output CB address, and nothing else:

```440:472:ttnn/cpp/ttnn/operations/data_movement/sharded_partial/interleaved_to_sharded_partial/device/interleaved_to_sharded_partial_program_factory.cpp
    // Cache-hit fast path. Only TILE layout is supported (validate_on_program_cache_miss); the static
    // work-split (shard extents, curr_idx, num_units) is pinned by the hashed shape/shard-spec. The only
    // per-dispatch values are the source/output buffer addresses and the slice_index-dependent
    // starting_idx_h. Patch just those slots in place instead of rebuilding the whole descriptor (O(1)
    // per core rather than O(num_cores) descriptor work). Replaces get_dynamic_runtime_args.
    const uint32_t starting_idx_h = operations::data_movement::detail::calculate_starting_idx_h(
        input_tensor, operation_attributes.num_slices, operation_attributes.slice_index);
    const uint32_t src_addr = input_tensor.buffer()->address();
    auto* dst_buffer = output.buffer();
    const bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;

    const auto& shard_spec = operation_attributes.shard_spec;
    const bool rm_orientation = shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    const auto cores = corerange_to_cores(shard_spec.grid, std::nullopt, rm_orientation);

    // Kernel push order in create_descriptor: reader (0), writer (1)[, compute (2)].
    // Reader TILE RT args: {src, shard_h, shard_w, padded_offset, num_units_offset, units_per_shard,
    //   curr_idx_h+curr_idx_w, starting_idx_h}; the DRAM-output writer mirrors src->dst at 0 and
    //   starting_idx_h at 7. The sharded-output writer carries neither (address rides on the output CB).
    constexpr uint32_t kReaderKernelIdx = 0;
    constexpr uint32_t kWriterKernelIdx = 1;
    constexpr uint32_t kBufferAddrArgIdx = 0;
    constexpr uint32_t kStartingIdxHArgIdx = 7;
    for (const auto& core : cores) {
        auto& reader_rt = tt::tt_metal::GetRuntimeArgs(program, kReaderKernelIdx, core);
        reader_rt[kBufferAddrArgIdx] = src_addr;
        reader_rt[kStartingIdxHArgIdx] = starting_idx_h;
        if (dst_is_dram) {
            auto& writer_rt = tt::tt_metal::GetRuntimeArgs(program, kWriterKernelIdx, core);
            writer_rt[kBufferAddrArgIdx] = dst_buffer->address();
            writer_rt[kStartingIdxHArgIdx] = starting_idx_h;
        }
    }
```

The comment states the load-bearing assumption explicitly: *"the static work-split (shard extents,
curr_idx, num_units) is pinned by the hashed shape/shard-spec"*. **There is no hashed shape.**
`compute_program_hash` contains no shape term of any kind (quoted below). That single false
premise is the root of findings #2 and #4.

The resulting obligation on the hash is the strictest of any op in this audit set:

- everything affecting a compile-time arg, kernel source, CB size/format or core range must be
  hashed (mode A never refreshes those), **and**
- every runtime arg *other than* reader arg 0/7 and writer arg 0/7 must also be hashed, because
  the override leaves them frozen at the values baked in at the first miss.

Two secondary observations on the patch site, both benign. It writes `reader_rt[7]`
unconditionally, which is only a valid index on the TILE path — the row-major reader has ten args
and index 7 is `aligned_shard_width`
(`interleaved_to_sharded_partial_program_factory.cpp:388`). Row-major is rejected by
`validate_on_program_cache_miss` (line 26), and that validator also runs on hits — see "Which
validator runs on a cache hit" below — so the index is safe. And the override
enumerates cores with `corerange_to_cores(shard_spec.grid, std::nullopt, rm_orientation)` from the
attributes, which is textually the same expression `create_descriptor` uses at line 245 (via
`output.shard_spec()`, itself built from `operation_attributes.shard_spec` in
`compute_output_specs`, `interleaved_to_sharded_partial_op.cpp:60-63`), so the two core orderings
cannot drift.

**CSV correction.** The CSV records `get_dynamic_runtime_args = Y`. No such member exists:
`interleaved_to_sharded_partial_op.hpp:20-41` declares only `validate_on_program_cache_miss`,
`compute_output_specs`, `create_output_tensors`, `compute_program_hash` and
`override_runtime_arguments`. The factory comment says so directly — "Replaces
get_dynamic_runtime_args" — and the hash comment's claim that `slice_index` "is re-applied on
every cache hit via get_dynamic_runtime_args" names the wrong mechanism. The outcome for
`slice_index` is unaffected, but the CSV column and both comments are wrong.

A second internal inconsistency worth noting: the header comment on the override promises a
different implementation than the one that exists.

```32:41:ttnn/cpp/ttnn/operations/data_movement/sharded_partial/interleaved_to_sharded_partial/device/interleaved_to_sharded_partial_op.hpp
    // slice_index feeds only the runtime read-offset starting_idx_h and is excluded from the program
    // hash, so cache hits for a different slice must re-derive the per-dispatch args (otherwise the
    // reader/writer args baked at first miss stay frozen at the first slice). Re-run create_descriptor
    // (single source of truth) and re-apply its per-core args + tensor-backed CB addresses.
    static void override_runtime_arguments(
```

"Re-run `create_descriptor` (single source of truth) and re-apply its per-core args" describes a
full mode-A re-derivation. The implementation does not call `create_descriptor` at all. Had it
done what the header says, finding #2 would not exist, because a rebuild recomputes every
shape-derived runtime arg.

## Which validator runs on a cache hit

This op defines **no** `validate_on_program_cache_hit` — `interleaved_to_sharded_partial_op.hpp:20`
declares only `validate_on_program_cache_miss`. It therefore takes the favourable branch of the
dispatcher, which substitutes the miss validator on every hit:

```262:266:ttnn/api/ttnn/device_operation.hpp
    if constexpr (HasValidateOnProgramCacheHit<mesh_device_operation_t>) {
        mesh_device_operation_t::validate_on_program_cache_hit(operation_attributes, tensor_args);
    } else {
        mesh_device_operation_t::validate_on_program_cache_miss(operation_attributes, tensor_args);
    }
```

Every `TT_FATAL` in the miss validator is consequently a live constraint on hits. Two verdicts
below depend directly on this branch: #1, where the TILE-only requirement is what makes the
override's unconditional `reader_rt[7]` write a valid index on the hit path, and #7, where the
device-storage requirement is what makes the storage kind carry no information. Verdict #5 relies
on it in the negative direction — the miss validator runs on hits and still contains no tile
check, so there is nothing to pin the tile geometry on either path.

Had this op defined even a narrow hit validator, both of the pinned verdicts would degrade to
"pinned only on the miss path", and the `reader_rt[7]` write would become an unguarded
out-of-bounds hazard on the row-major path rather than a benign one.

## Baseline: what the default hash would cover

`tensor_args_t` is a bare `Tensor`, so reflection decomposes it directly:

| Source | Fields |
|---|---|
| `operation_attributes` | `grid_size`, `shard_spec`, `num_slices`, `slice_index`, `output_mem_config`, `output_dtype` |
| `input_tensor.storage` | storage variant kind |
| `input_tensor.tensor_spec` | `logical_shape`, and `tensor_layout` = { `dtype`, `page_config`, `memory_config`, `alignment` } |

## What the custom hash covers

```77:92:ttnn/cpp/ttnn/operations/data_movement/sharded_partial/interleaved_to_sharded_partial/device/interleaved_to_sharded_partial_op.cpp
ttsl::hash::hash_t InterleavedToShardedPartialDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const Tensor& input_tensor) {
    // slice_index is deliberately excluded from the key: it only feeds the runtime read-offset
    // starting_idx_h (same program structure for every slice of a given num_slices), and it is
    // re-applied on every cache hit via get_dynamic_runtime_args. Keying on it would rebuild the
    // program for each slice of a partial-slicing loop. num_slices -- which drives the work split --
    // stays keyed.
    return tt::tt_metal::operation::hash_operation<InterleavedToShardedPartialDeviceOperation>(
        operation_attributes.grid_size,
        operation_attributes.shard_spec,
        operation_attributes.num_slices,
        operation_attributes.output_mem_config,
        operation_attributes.output_dtype,
        input_tensor.dtype(),
        input_tensor.layout());
}
```

Five of six attributes are kept; `slice_index` is the documented omission. From the input tensor
only `dtype()` and `layout()` survive — **no shape, no memory config, no tile**.

**Comparison against the sibling op.** The non-partial `interleaved_to_sharded` hashes
`input_tensor.memory_config()` *and* `input_tensor.padded_shape()` on top of dtype and layout
(`sharded/interleaved_to_sharded/device/interleaved_to_sharded_op.cpp:141-152`). This op runs a
near-identical factory with both of those terms missing. The two ops should agree; the partial one
is the outlier.

## Omitted parameters

### 1. `slice_index` — the declared intentional omission

**Verdict: VALID — patched.**

This is the textbook case for omit-and-patch. It feeds exactly one derived value:

```240:241:ttnn/cpp/ttnn/operations/data_movement/sharded_partial/interleaved_to_sharded_partial/device/interleaved_to_sharded_partial_program_factory.cpp
    uint32_t starting_idx_h =
        operations::data_movement::detail::calculate_starting_idx_h(input, num_slices, slice_index);
```

which lands in reader arg 7 (line 291) and, when the output is in DRAM, writer arg 7 (line 305).
The override recomputes it with the same helper and writes both slots on every core. Nothing else
in `create_descriptor` reads `slice_index`, and the helper is a pure function that produces a tile
offset, never a structural quantity:

```16:28:ttnn/cpp/ttnn/operations/data_movement/sharded/sharded_common.cpp
uint32_t calculate_starting_idx_h(const Tensor& tensor, uint32_t num_slices, uint32_t slice_index) {
    if (num_slices <= 1) {
        return 0;
    }

    uint32_t num_tiles_height = tensor.physical_volume() / tensor.padded_shape()[-1] / tt::constants::TILE_HEIGHT;
    uint32_t num_tiles_width = tensor.padded_shape()[-1] / tt::constants::TILE_WIDTH;
    uint32_t total_num_tiles = num_tiles_height * num_tiles_width;

    uint32_t num_tiles_per_slice = total_num_tiles / num_slices;
    uint32_t starting_tile_in_slice = num_tiles_per_slice * slice_index;
    return starting_tile_in_slice;
}
```

The relaxation is real and is the whole point of the op: a loop over
`slice_index = 0..num_slices-1` compiles one program instead of `num_slices`. `num_slices` itself
does change the work split (line 103) and is correctly kept in the key. The range invariant
`0 <= slice_index < num_slices` is re-checked on every hit
(`interleaved_to_sharded_partial_op.cpp:21-25`) through the validator fallback, so an out-of-range
index cannot ride a cache hit into a bad offset.

### 2. `input_tensor.logical_shape()` / `padded_shape()` — no shape term in the key at all

**Verdict: BUG.**

The TILE branch derives the entire per-core work split from the input shape:

```100:110:ttnn/cpp/ttnn/operations/data_movement/sharded_partial/interleaved_to_sharded_partial/device/interleaved_to_sharded_partial_program_factory.cpp
        num_units_per_row = input.padded_shape()[-1] / TILE_WIDTH;
        num_units_offset = num_units_per_row;
        uint32_t num_units_height =
            static_cast<uint32_t>(input.physical_volume() / input.padded_shape()[-1] / TILE_HEIGHT / num_slices);
        num_units_per_shard_height_last =
            num_units_per_shard_height -
            (tt::round_up(num_units_height, num_units_per_shard_height) - num_units_height);
        num_units_per_shard_width_last =
            num_units_per_shard_width -
            (tt::round_up(num_units_per_row, num_units_per_shard_width) - num_units_per_row);
        padded_offset_bytes = (num_units_per_shard_width - num_units_per_shard_width_last) * input_unit_size;
```

`num_units_per_shard_height_last` becomes the last core's `shard_height`, and hence its
`curr_num_units_per_shard`:

```252:255:ttnn/cpp/ttnn/operations/data_movement/sharded_partial/interleaved_to_sharded_partial/device/interleaved_to_sharded_partial_program_factory.cpp
            if (shard_strategy == TensorMemoryLayout::HEIGHT_SHARDED) {
                if (core == end_core) {
                    shard_height = num_units_per_shard_height_last;
                }
            } else if (shard_strategy == TensorMemoryLayout::WIDTH_SHARDED) {
```

```280:292:ttnn/cpp/ttnn/operations/data_movement/sharded_partial/interleaved_to_sharded_partial/device/interleaved_to_sharded_partial_program_factory.cpp
            curr_num_units_per_shard = shard_height * num_units_per_shard_width;

            // Reader run-time args: arg 0 is the source-buffer base address (binding).
            KernelDescriptor::RTArgList reader_rt;
            reader_rt.push_back(src_buffer);
            reader_rt.push_back(shard_height);
            reader_rt.push_back(shard_width);
            reader_rt.push_back(padded_offset);
            reader_rt.push_back(num_units_offset);
            reader_rt.push_back(curr_num_units_per_shard);
            reader_rt.push_back(curr_idx_h + curr_idx_w);
            reader_rt.push_back(starting_idx_h);
            reader_desc.emplace_runtime_args(core, reader_rt);
```

So the shape reaches reader args **1, 2, 3, 4, 5 and 6** (and the same values in the DRAM writer's
args 1-6, lines 297-309, plus the compute kernel's single arg at line 421). The override patches
only args 0 and 7. Every one of those six slots is frozen at the first miss, and none of the
attributes that *are* hashed determines them.

It is worth being precise about why the hashed `shard_spec` does not rescue this. `shard_spec`
fixes `num_units_per_shard_height` and `num_units_per_shard_width` (lines 97-98) — the *nominal*
shard extents. What it cannot fix is `num_units_height`, the actual tile-row count of the input,
which is what decides how much the final shard is truncated. Two inputs whose heights round up to
the same number of shards share a shard spec but not a truncation.

**Reproduction** (reachable through the public `ttnn.interleaved_to_sharded_partial`, TILE
layout, `bfloat16`, DRAM-interleaved input, `shard_scheme=HEIGHT_SHARDED`,
`shard_shape=[64, 128]`, `grid=CoreCoord(8, 8)`, `num_slices=1`, `slice_index=0`):

- **Call 1**: input padded shape `[1, 1, 128, 128]`. The wrapper computes
  `total_height = 128`, `num_cores = div_up(128, 64) = 2`
  (`interleaved_to_sharded_partial.cpp:32-46`), giving a 2-core `grid_set`. In the factory,
  `num_units_height = 128*128/128/32/1 = 4`, `num_units_per_shard_height = 2`, so
  `num_units_per_shard_height_last = 2 - (round_up(4,2) - 4) = 2`. Both cores read a full 2x4-tile
  shard.
- **Call 2**: input padded shape `[1, 1, 96, 128]`, everything else identical.
  `total_height = 96`, `num_cores = div_up(96, 64) = 2` — the **same** `grid_set`, hence the same
  `ShardSpec`. But `num_units_height = 3`, so
  `num_units_per_shard_height_last = 2 - (round_up(3,2) - 3) = 1`: the end core should read a
  half-height shard.

Every hashed term is bit-identical across the two calls (`grid_size` `(8,8)`, the `ShardSpec`,
`num_slices = 1`, `output_mem_config` `{HEIGHT_SHARDED, L1}`, `output_dtype`, input `dtype`, input
`layout`), so call 2 hits call 1's program. The end core keeps `shard_height = 2` and
`curr_num_units_per_shard = 8`, and starts at `curr_idx_h + curr_idx_w = 8`. The input holds only
`3 * 4 = 12` tiles, so the reader issues NOC reads for tiles 8 through 15 — **four tiles past the
end of the input buffer** — and the second output shard is filled with whatever follows the
allocation in DRAM. There is no cache miss, no assertion, and `validate_on_program_cache_miss`
checks only `total_height % num_slices == 0` (lines 27-29), never that the shard grid matches the
input extent.

The width dimension collides the same way: `num_units_per_row` (reader arg 4, and the driver of
the `curr_idx_w`/`curr_idx_h` walk at lines 315-319) is `padded_shape[-1] / TILE_WIDTH` and is
completely absent from the key, so a WIDTH_SHARDED call whose width changes without changing
`div_up(total_width, shard_shape[1])` produces the same collision on `num_units_offset` and
`padded_offset_bytes`.

The fix is to hash `input_tensor.padded_shape()`, matching the non-partial sibling. Hashing the
padded rather than the logical shape preserves the relaxation described in #3.

### 3. `input_tensor.logical_shape()` specifically, as distinct from `padded_shape()`

**Verdict: VALID — relaxation win** (conditional on #2 being fixed by hashing `padded_shape`).

Listing this separately because the right fix for #2 is not "hash the tensor". Nothing in the
factory or in `compute_output_specs` reads the logical shape — the TILE branch above uses only
`padded_shape()[-1]` and `physical_volume()`, and the output spec is built from the padded shape:

```49:69:ttnn/cpp/ttnn/operations/data_movement/sharded_partial/interleaved_to_sharded_partial/device/interleaved_to_sharded_partial_op.cpp
tt::tt_metal::TensorSpec InterleavedToShardedPartialDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const Tensor& input_tensor) {
    auto shape = input_tensor.padded_shape();

    uint32_t total_height = input_tensor.physical_volume() / shape[-1];
    uint32_t new_height = total_height / operation_attributes.num_slices;

    shape[0] = 1;
    shape[1] = 1;
    shape[2] = new_height;

    auto mem_config = MemoryConfig(
        operation_attributes.output_mem_config.memory_layout(),
        operation_attributes.output_mem_config.buffer_type(),
        operation_attributes.shard_spec);

    return tt::tt_metal::TensorSpec(
        shape,
        tt::tt_metal::TensorLayout(
            operation_attributes.output_dtype, tt::tt_metal::PageConfig(input_tensor.layout()), mem_config));
}
```

So `[1,1,33,128]` and `[1,1,64,128]`, which pad to the same tile grid, correctly share one program
and one output spec; the default hash would force a needless recompile. The factory also collapses
the leading dimensions into `physical_volume()`, so `[1,1,256,128]` and `[1,2,128,128]` are
genuinely the same program — hashing the full padded shape still separates them, a missed
relaxation rather than a correctness issue.

### 4. `input_tensor.memory_config()` — the input's buffer type is not hashed

**Verdict: BUG.**

Validation constrains the input to be interleaved but says nothing about where it lives:

```34:36:ttnn/cpp/ttnn/operations/data_movement/sharded_partial/interleaved_to_sharded_partial/device/interleaved_to_sharded_partial_op.cpp
    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Input tensor must be Interleaved");
```

DRAM-interleaved and L1-interleaved inputs are both admissible and hash identically. The source
buffer's placement is baked into the reader's **compile-time** args:

```191:197:ttnn/cpp/ttnn/operations/data_movement/sharded_partial/interleaved_to_sharded_partial/device/interleaved_to_sharded_partial_program_factory.cpp
    if (input.layout() == Layout::TILE) {
        std::vector<uint32_t> reader_compile_time_args = {input_cb_index, all_cores.num_cores()};
        tt::tt_metal::TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);
        reader_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/"
            "reader_unary_sharded_blocks_interleaved_start_id.cpp";
        reader_desc.compile_time_args = std::move(reader_compile_time_args);
```

`TensorAccessorArgs` encodes placement as an explicit config bit read from the buffer:

```146:157:tt_metal/impl/buffers/tensor_accessor_args.cpp
void TensorAccessorArgs::update_args_config() {
    if (!buffer_) {
        args_config_ = tensor_accessor::ArgConfig::None;
        return;
    }

    if (buffer_->buffer_distribution_spec().has_value()) {
        args_config_.set(tensor_accessor::ArgConfig::Sharded);
    } else {
        args_config_ = tensor_accessor::ArgConfig::None;
    }
    args_config_.set(tensor_accessor::ArgConfig::IsDram, buffer_->is_dram());
```

Compile-time args are baked into the cached `Program` and are never refreshed on a hit, in mode A
or any other mode.

**Reproduction**: call 1 with a TILE `bfloat16` **DRAM-interleaved** input; call 2 with the
same-shaped, same-dtype tensor in **L1 interleaved**, all attributes identical. The hash is
unchanged, so call 2 hits. The override patches reader arg 0 to the L1 buffer's address, but the
compiled `TensorAccessor` still resolves pages through the DRAM bank table: the reader issues DRAM
NOC reads at an offset derived from an L1 address, the L1 input is never touched, and the output
shards contain unrelated memory.

The same omission has two further structural effects on the TILE path, either of which is
independently sufficient to make this a BUG. `src_is_dram` gates whether the scratch CB is created
at all:

```168:184:ttnn/cpp/ttnn/operations/data_movement/sharded_partial/interleaved_to_sharded_partial/device/interleaved_to_sharded_partial_program_factory.cpp
    uint32_t dram_alignment = hal::get_dram_alignment();
    uint32_t l1_alignment = hal::get_l1_alignment();
    uint32_t num_trids = 4;
    if ((src_is_dram && (input_unit_size % dram_alignment != 0)) || is_blackhole || keep_l1_aligned) {
        // scratchpad going to be used to align DRAM (64B) to L1 (16B)
        // This is done to mitigate the alignment issues.
        // See issue #34414.
        uint32_t scratch_cb_page_size = tt::align(input_unit_size + dram_alignment, dram_alignment);
        push_i2s_partial_cb_pair(
            desc,
            scratch_cb_index,
            input_cb_data_format,
            num_trids * scratch_cb_page_size,
            scratch_cb_page_size,
            all_cores,
            /*bound_buffer=*/nullptr);
    }
```

and when the data format is converted, the input CB's page size comes from
`src_buffer->alignment()`, which differs between DRAM and L1:

```143:155:ttnn/cpp/ttnn/operations/data_movement/sharded_partial/interleaved_to_sharded_partial/device/interleaved_to_sharded_partial_program_factory.cpp
    if (convert_df) {
        out_cb_index = tt::CBIndex::c_16;
        uint32_t input_page_size = tt::align(input_unit_size, src_buffer->alignment());
        // Non-globally-allocated input CB (interleaved input streamed via reader).
        push_i2s_partial_cb_pair(
            desc,
            input_cb_index,
            input_cb_data_format,
            num_input_units * input_page_size,
            input_page_size,
            all_cores,
            /*bound_buffer=*/nullptr);
    }
```

CB count, CB total size and CB page size are all part of the cached `Program`.

The fix is one line: add `input_tensor.memory_config()` to the hash, as the non-partial sibling
already does.

### 5. `input_tensor.tensor_spec().page_config()` — only `layout()` is hashed, and the factory hardcodes 32x32

**Verdict: BUG.**

`layout()` collapses `PageConfig` to `ROW_MAJOR` vs `TILE`, discarding the `Tile` shape. All three
conditions for the unguarded-tile bug class hold.

The op accepts `Layout::TILE` — indeed it accepts nothing else
(`interleaved_to_sharded_partial_op.cpp:26`). The factory then computes every tile quantity from
the architectural constants rather than the tensor's own tile:

```88:103:ttnn/cpp/ttnn/operations/data_movement/sharded_partial/interleaved_to_sharded_partial/device/interleaved_to_sharded_partial_program_factory.cpp
    if (input.layout() == Layout::TILE) {
        input_unit_size = tt::tile_size(input_cb_data_format);
        output_unit_size = tt::tile_size(output_cb_data_format);
        TT_FATAL(
            shard_spec.shape[0] % TILE_HEIGHT == 0 && shard_spec.shape[1] % TILE_WIDTH == 0,
            "Shard shape {} must be tile {}x{} sized!",
            shard_spec.shape,
            TILE_HEIGHT,
            TILE_WIDTH);
        num_units_per_shard_height = shard_spec.shape[0] / TILE_HEIGHT;
        num_units_per_shard_width = shard_spec.shape[1] / TILE_WIDTH;
        num_units_per_shard = num_units_per_shard_height * num_units_per_shard_width;
        num_units_per_row = input.padded_shape()[-1] / TILE_WIDTH;
        num_units_offset = num_units_per_row;
        uint32_t num_units_height =
            static_cast<uint32_t>(input.physical_volume() / input.padded_shape()[-1] / TILE_HEIGHT / num_slices);
```

`tt::tile_size(format)` returns the byte size of a 32x32 tile — the tile-aware call is
`tile.get_tile_size(format)` — and lines 97-103 convert shapes to tile counts with bare
`TILE_HEIGHT`/`TILE_WIDTH`. `calculate_starting_idx_h` does the same
(`sharded_common.cpp:21-22`). The `TT_FATAL` at lines 91-96 checks the *shard shape* against the
constants; it says nothing about the tensor's `Tile`, and `validate_on_program_cache_miss` has no
tile check either. Nothing in the op reads `tensor_spec().tile()`.

Because the factory hardcodes 32x32 *and* `page_config` is unhashed, the two defects compound: a
`Tile{16, 32}` input does not even get a freshly-built wrong program. It silently inherits the
32x32 entry built for a same-dtype, same-shard-spec 32x32 tensor, with `input_unit_size`, the CB
page sizes and every per-core tile count computed for the wrong tile geometry.

The non-partial sibling already carries the guard this op needs:

```94:98:ttnn/cpp/ttnn/operations/data_movement/sharded/interleaved_to_sharded/device/interleaved_to_sharded_op.cpp
    if (input_tensor.layout() == Layout::TILE) {
        auto tile = input_tensor.tensor_spec().tile();
        if (tile.get_height() != tt::constants::TILE_HEIGHT || tile.get_width() != tt::constants::TILE_WIDTH) {
            return {false, fmt::format("interleaved_to_sharded requires standard 32x32 tiles, got {}x{}", tile.get_height(), tile.get_width())};
        }
```

Adding it makes the `page_config` omission correct by construction. Making the factory genuinely
tile-aware instead would require adding `page_config` to the hash in the same change.

### 6. `input_tensor.tensor_layout().get_alignment()`

**Verdict: CAVEAT.**

The factory never reads `Alignment` directly. For a tile-layout tensor it reaches the program only
through `padded_shape` and `physical_volume` — both of which are the subject of finding #2 — and
through the buffer page size, which depends on `memory_config` (finding #4). Once `padded_shape`
and `memory_config` are hashed, the omission of `Alignment` becomes genuinely safe, since a tile
tensor's alignment is constrained to multiples of the tile dimensions (`validate_alignment_tile`,
`tt_metal/impl/tensor/spec/layout/page_config.cpp:59-75`) and contributes nothing beyond those two
hashed derivations.

This stays a CAVEAT rather than being upgraded, and the reason is reachability rather than
enforcement. There is no configuration in which the `Alignment` omission produces a wrong hit that
is not already the reproduction given for #2 or #4: every route by which a different alignment
changes the program passes through the padded shape, the physical volume or the page size. Marking
it a fourth BUG would count one defect three times. Unlike the corresponding entry in the
non-partial `interleaved_to_sharded` audit, `Alignment` is not the *enabler* of anything here —
findings #2 and #4 reproduce with entirely default alignments.

### 7. `input_tensor.storage` variant kind (device vs host)

**Verdict: VALID — pinned by validation.**

```31:32:ttnn/cpp/ttnn/operations/data_movement/sharded_partial/interleaved_to_sharded_partial/device/interleaved_to_sharded_partial_op.cpp
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to shard need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to shard need to be allocated in buffers on device!");
```

Constant across every admissible call, and re-checked on hits through the substitution branch
quoted under "Which validator runs on a cache hit".

### 8. Buffer addresses (omitted by the default hash as well)

**Verdict: VALID — patched.**

Addresses reach the program by two routes and the override covers both. A DRAM output rides on
writer arg 0; an L1 sharded output rides on the output CB's `.buffer` binding (line 166), which
the override refreshes with a synthetic address-only descriptor:

```474:487:ttnn/cpp/ttnn/operations/data_movement/sharded_partial/interleaved_to_sharded_partial/device/interleaved_to_sharded_partial_program_factory.cpp
    // Sharded (non-DRAM) output binds its buffer to the output CB rather than a writer RT arg, so refresh
    // that CB's base address in place. create_descriptor pushes the (unbound) input CB first only when
    // converting data formats; mirror that ordering positionally -- apply_descriptor_runtime_args maps
    // desc.cbs[i] to program CB i and updates only the entries that carry a buffer.
    if (!dst_is_dram) {
        const bool convert_df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype()) !=
                                tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
        ProgramDescriptor cb_addr_only;
        if (convert_df) {
            cb_addr_only.cbs.emplace_back();  // input CB placeholder (unbound; address unchanged)
        }
        cb_addr_only.cbs.push_back(CBDescriptor{.buffer = dst_buffer});
        tt::tt_metal::apply_descriptor_runtime_args(program, cb_addr_only);  // override-rebuild-ok: cb-addr-only
    }
```

Both sides recompute `convert_df` from the same two dtypes, so the positional CB mapping cannot
drift. Note that the scratch CB (line 176) is pushed *after* the output CB and carries no buffer,
so it does not disturb the indices the override relies on.

## Keys the custom hash adds beyond the default

None. Unusually for a `SELECTIVE` op, this hash is a strict subset of the default key: every term
it contains (`grid_size`, `shard_spec`, `num_slices`, `output_mem_config`, `output_dtype`, input
`dtype`, input `layout`) is either an attribute the default also hashes or a projection of a tensor
field the default hashes in full. It adds no `padded_shape` term of the kind that makes the
non-partial sibling's relaxation safe. That is precisely why findings #2 and #5 are bugs rather
than relaxations: the op took the relaxation without adding the compensating key.

## Framework side effect of having a custom hash

```1012:1014:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
        if constexpr (requires { DeviceOperation::compute_program_hash(attrs, tensor_args); }) {
            return key;  // custom hash -> opt out beyond the op-identity prefix
        } else {
```

`ProgramCacheKey::canonical` degrades to the op type name alone, so even a genuine 64-bit
collision between two distinct configurations resolves to a wrong hit rather than a rebuild. That
compounds the findings above, which are exact-key collisions rather than hash collisions and
therefore occur with probability 1.

## Summary

| Omitted vs. default | Used by program? | Patched on hit? | Verdict |
|---|---|---|---|
| `slice_index` | Yes — `starting_idx_h` (reader arg 7, writer arg 7) | Yes (override) | VALID — patched |
| `input.padded_shape` / `physical_volume` | Yes — reader args 1-6, writer args 1-6, compute arg 0 | **No** | **BUG** |
| `input.logical_shape` (beyond the padded shape) | No | n/a | VALID — relaxation win |
| `input.memory_config()` | Yes — `TensorAccessorArgs` compile-time args, scratch-CB existence, input CB page size | **No** (compile-time args and CBs are never refreshed) | **BUG** |
| `input.page_config` (`Tile`) | Yes — `tt::tile_size`, bare `TILE_HEIGHT`/`TILE_WIDTH`, unguarded | **No** | **BUG** |
| `input.tensor_layout.alignment` | Only via `padded_shape` and page size | No | CAVEAT — no reachability independent of #2/#4 |
| `input.storage` kind | n/a | n/a | VALID — pinned by validation |
| Buffer addresses | Yes | Yes (override: rt args + output CB) | VALID — patched |

**Three program-cache correctness bugs were found, and they share one root cause: the override
justifies its narrowness by appealing to a hash term that does not exist.**

The comment guarding the two-slot patch reads:

```440:444:ttnn/cpp/ttnn/operations/data_movement/sharded_partial/interleaved_to_sharded_partial/device/interleaved_to_sharded_partial_program_factory.cpp
    // Cache-hit fast path. Only TILE layout is supported (validate_on_program_cache_miss); the static
    // work-split (shard extents, curr_idx, num_units) is pinned by the hashed shape/shard-spec. The only
    // per-dispatch values are the source/output buffer addresses and the slice_index-dependent
    // starting_idx_h. Patch just those slots in place instead of rebuilding the whole descriptor (O(1)
    // per core rather than O(num_cores) descriptor work). Replaces get_dynamic_runtime_args.
```

The hash it appeals to contains no shape term of any kind:

```84:92:ttnn/cpp/ttnn/operations/data_movement/sharded_partial/interleaved_to_sharded_partial/device/interleaved_to_sharded_partial_op.cpp
    return tt::tt_metal::operation::hash_operation<InterleavedToShardedPartialDeviceOperation>(
        operation_attributes.grid_size,
        operation_attributes.shard_spec,
        operation_attributes.num_slices,
        operation_attributes.output_mem_config,
        operation_attributes.output_dtype,
        input_tensor.dtype(),
        input_tensor.layout());
}
```

"Pinned by the hashed shape/shard-spec" is half right. The shard spec is hashed; the shape is not,
in either its logical or its padded form, and neither is the input's `memory_config`. The work
split the comment calls static is computed from `padded_shape()` and `physical_volume()`
(`interleaved_to_sharded_partial_program_factory.cpp:100-110`) — values that can change freely
without changing the key. The header comment compounds this by describing the override as
re-running `create_descriptor`, which would in fact have made the claim true; the implementation
patches four slots and never calls it.

So the op combines the weakest hash of the pair — weaker than the non-partial
`interleaved_to_sharded`, which hashes both `padded_shape` and `memory_config` — with the narrowest
cache-hit patch, and every bug below sits in the gap between what the comment assumes and what the
key carries. Finding #2 is reachable from the public Python API with ordinary tile-aligned shapes
and produces an out-of-bounds NOC read; #4 produces reads against the wrong memory space; #5 is the
unguarded 32x32 assumption.

By contrast, the `slice_index` omission — the one the op deliberately made and documented — is
correct, and the set of slots the override patches (reader 0/7, writer 0/7, output CB) exactly
matches the set of values that depend on `slice_index` and on buffer addresses.

Two CSV corrections. `get_dynamic_runtime_args` should be **N**: no such member exists, and
`override_runtime_arguments` supersedes it. And `own_hit_validator = N` understates the situation,
since the dispatcher substitutes `validate_on_program_cache_miss` on hits
(`ttnn/api/ttnn/device_operation.hpp:262-266`), which verdicts #1, #5 (layout premise) and #7 rely
on.

## Recommendations

1. Add `input_tensor.padded_shape()` to `compute_program_hash`. This closes finding #2 and makes
   the override's stated premise true. Hash the padded shape rather than the logical shape so the
   relaxation in #3 is preserved.
2. Add `input_tensor.memory_config()` to `compute_program_hash`, closing finding #4. Together with
   (1) this makes the key identical in coverage to the non-partial
   `interleaved_to_sharded`, which is the right target given the two factories are near-copies.
3. Add the equivalent of the 32x32 tile check at `interleaved_to_sharded_op.cpp:94-98` to
   `validate_on_program_cache_miss`, as a `TT_FATAL` to match this op's validator style. This
   closes finding #5.
4. Fix the three stale comments: the hash comment naming `get_dynamic_runtime_args`
   (`interleaved_to_sharded_partial_op.cpp:79-83`), the header comment claiming the override
   re-runs `create_descriptor` (`interleaved_to_sharded_partial_op.hpp:32-35`), and the override's
   claim that the work split is pinned by a hashed shape
   (`interleaved_to_sharded_partial_program_factory.cpp:440-444`). Each currently asserts a
   safety property the code does not have.
5. The override depends on hardcoded arg indices (0 and 7), kernel indices (0 and 1) and CB
   ordering that must stay in lockstep with `create_descriptor`. Extract the reader/writer arg
   layout into named constants shared by both functions so a future arg insertion cannot silently
   shift `starting_idx_h` off slot 7.
6. Run this op's tests once under `-DTT_DESCRIPTOR_PATCHING_PARITY_CHECK`. The parity oracle is
   wired into the mode-A branch (`mesh_device_operation_adapter.hpp:679-693`) and rebuilds the
   descriptor as a reference, which would flag finding #2 directly as a mismatch on reader args
   1-6.
