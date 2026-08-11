# Program Cache Audit — `data_movement/slice`

Audit of `ttnn::prim::SliceDeviceOperation::compute_program_hash` against the framework default
("hash everything") key.

| | |
|---|---|
| Device operation | `ttnn::prim::SliceDeviceOperation` (`device/slice_device_operation.hpp:33`) |
| Custom hash | `device/slice_device_operation.cpp:298-350` |
| `operation_attributes_t` | `SliceParams` — `slice_start`, `slice_end`, `step`, `output_mem_config`, `use_tensor_args`, `slice_dim`, `num_devices`, `sub_core_grids` |
| `tensor_args_t` | `SliceInputs` — `input`, `start_tensor`, `end_tensor`, `preallocated_output` |
| Program factories | five: `SliceRmProgramFactory`, `SliceRmShardedProgramFactory`, `SliceRmStrideProgramFactory`, `SliceTileProgramFactory`, `SliceTileTensorArgsProgramFactory` |
| `override_runtime_arguments` | **Yes** (`device/slice_program_factory_rm_sharded.cpp:339`) |
| `get_dynamic_runtime_args` | **No** — the member does not exist (see CSV note) |
| Own cache-hit validator | No — the framework substitutes `validate_on_program_cache_miss` |
| Cache-hit patch mechanism | **Op-owned cache-hit re-derivation** (mode A), full descriptor rebuild for four factories and a CB-address-only patch for the fifth |

## Cache-hit patch mechanism

The op defines `override_runtime_arguments` on the device operation, so the adapter takes the
op-owned branch and bypasses `resolve_bindings` and `get_dynamic_runtime_args`:

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

Unlike the two `interleaved_to_sharded` ops, this override is a genuine full re-derivation for
four of the five factories:

```339:365:ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_program_factory_rm_sharded.cpp
void SliceDeviceOperation::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    const auto factory = select_program_factory(operation_attributes, tensor_args);

    // Height-sharded RM reader args depend only on shapes/slice_start/shard specs, all cache-keyed, so
    // on a hit the only thing that changes is the two CB addresses. Patch those in O(1) instead of
    // rebuilding all per-core args (which scaled host cost with the core grid). CBs: src0, then c_16.
    if (std::holds_alternative<SliceRmShardedProgramFactory>(factory)) {
        tt::tt_metal::ProgramDescriptor cb_addr_only;
        cb_addr_only.cbs.push_back(tt::tt_metal::CBDescriptor{.buffer = tensor_args.input.buffer()});
        cb_addr_only.cbs.push_back(tt::tt_metal::CBDescriptor{.buffer = tensor_return_value.buffer()});
        tt::tt_metal::apply_descriptor_runtime_args(program, cb_addr_only);
        return;
    }

    // Other factories bake buffer addresses into their runtime args, so re-derive and re-apply.
    auto desc = std::visit(
        [&](auto&& f) {
            return std::decay_t<decltype(f)>::create_descriptor(operation_attributes, tensor_args, tensor_return_value);
        },
        factory);
    tt::tt_metal::apply_descriptor_runtime_args(program, desc);
}
```

Two details make this the strongest cache-hit posture of the four ops in this audit set. First,
the rebuild path re-runs `create_descriptor` through the same `select_program_factory` the miss
path used, so there is a single source of truth and no hand-maintained arg-index table to drift.
Second, `apply_descriptor_runtime_args` refreshes **common** runtime args as well as per-core ones,
which matters because both tile factories carry the source, start and end buffer addresses in
common args rather than per-core args:

```193:204:tt_metal/impl/program/program_descriptors.cpp
        if (!kernel.common_runtime_args.empty()) {
            // Cannot use SetCommonRuntimeArgs here — it calls
            // Kernel::set_common_runtime_args which has a TT_FATAL requiring
            // common_runtime_args_ to be empty.  On cache hits the program is
            // reused, so the args are already populated from the initial
            // create().  Update in-place instead (same pattern used for
            // per-core runtime_args above).
            auto& common_args = GetCommonRuntimeArgs(program, k);
            for (uint32_t i = 0; i < static_cast<uint32_t>(kernel.common_runtime_args.size()); ++i) {
                common_args[i] = kernel.common_runtime_args[i];
            }
        }
```

The `SliceRmShardedProgramFactory` shortcut is sound. Every runtime arg that factory emits derives
from `input.padded_shape()`, `output.padded_shape()`, `args.slice_start`, the input shard spec
(`slice_program_factory_rm_sharded.cpp:224`), the output shard spec (line 244) and the device's
logical-to-physical core mapping (line 156) — the first four are all hashed (input `padded_shape`
and `memory_config`, `slice_start`, and the output spec's `memory_config` respectively), and the
last is fixed for a given device. So the two CB addresses really are the only per-dispatch values.

**The obligation this leaves on the hash** is therefore narrow but absolute: because mode A never
refreshes compile-time args, kernel sources, CB sizes/formats, semaphores or core ranges,
everything feeding those must be hashed. Runtime args are fully covered by the rebuild and impose
no hash obligation at all (except on the RM-sharded path, which is discharged above).

**CSV correction.** The CSV records `get_dynamic_runtime_args = Y`. The op has no such member
(`slice_device_operation.hpp:45-65` declares `select_program_factory`,
`validate_on_program_cache_miss`, `compute_output_specs`, `create_output_tensors`,
`compute_program_hash`, `create_op_performance_model` and `override_runtime_arguments`). What
exists is a leftover free function:

```386:392:ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_program_factory_rm.cpp
std::vector<tt::tt_metal::DynamicRuntimeArg> slice_rm_reader_dynamic_args(
    const SliceParams& args, const SliceInputs& tensor_args, const Tensor& output) {
    // Reader arg 0 holds the aligned source base (input buffer address + a hash-constant byte offset).
    // The offset is baked into the cached descriptor; only the buffer address changes per dispatch, so
    // re-emit the full value on every cache hit. The work-split (and thus the active-core set) derives
    // only from hashed shapes/grids, so it is identical on every hit — no freeze from a growing set.
    const auto& input = tensor_args.input;
```

A repo-wide search for `slice_rm_reader_dynamic_args` returns only its declaration
(`slice_program_factory_rm.hpp:31`) and this definition — it is called from nowhere. It is dead
code left from an earlier design, and even if it were wired up the adapter would ignore it,
because mode A bypasses `get_dynamic_runtime_args` entirely. Three comments in
`slice_program_factory_rm.cpp` (lines 34-40 and 103-104) still describe reader arg 0 as riding on
`get_dynamic_runtime_args`; in fact it rides on the full rebuild. The behaviour is correct either
way, but the CSV entry and the comments are wrong.

## Which validator runs on a cache hit

This op defines **no** `validate_on_program_cache_hit` — `slice_device_operation.hpp:47` declares
only `validate_on_program_cache_miss`. It therefore takes the favourable branch of the dispatcher,
which substitutes the miss validator on every hit:

```262:266:ttnn/api/ttnn/device_operation.hpp
    if constexpr (HasValidateOnProgramCacheHit<mesh_device_operation_t>) {
        mesh_device_operation_t::validate_on_program_cache_hit(operation_attributes, tensor_args);
    } else {
        mesh_device_operation_t::validate_on_program_cache_miss(operation_attributes, tensor_args);
    }
```

Every `TT_FATAL` in the miss validator is therefore live on hits. Verdict #4 below depends on this
branch directly: the device-storage requirements on the input and the start tensor are what make
the storage kinds carry no information, and they live only in the miss validator.

The substitution cuts both ways for this op, and finding #3 is where it shows. The
preallocated-output check at `slice_device_operation.cpp:135-143` also runs on every hit — it is
simply too narrow to pin anything but the shape, so the hit path inherits its gap rather than
being rescued by it. Had `slice` defined a narrow hit validator instead, verdict #4 would degrade
to "pinned only on the miss path" as well.

## Baseline: what the default hash would cover

| Source | Fields |
|---|---|
| `operation_attributes` | `slice_start`, `slice_end`, `step`, `output_mem_config`, `use_tensor_args`, `slice_dim`, `num_devices`, `sub_core_grids` |
| `tensor_args.input` | storage kind, `logical_shape`, `dtype`, `page_config`, `memory_config`, `alignment` |
| `tensor_args.start_tensor` | engaged-ness, then the same six fields |
| `tensor_args.end_tensor` | engaged-ness, then the same six fields |
| `tensor_args.preallocated_output` | engaged-ness, then the same six fields |

## What the custom hash covers

```298:350:ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_device_operation.cpp
ttsl::hash::hash_t SliceDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Default hash has weak distribution for small-integer shape sequences (boost::hash_combine
    // style), causing false cache hits when shapes differ but collide. Mix in full input/output
    // specs and slice params so any two invocations with different core-grid sizes get distinct
    // keys. Same pattern as concat fix in PR #45144 (issue #47602).
    auto factory = select_program_factory(operation_attributes, tensor_args);
    auto hash = tt::tt_metal::operation::hash_operation<SliceDeviceOperation>(
        operation_attributes.slice_start,
        operation_attributes.slice_end,
        operation_attributes.step,
        operation_attributes.use_tensor_args,
        operation_attributes.slice_dim,
        operation_attributes.num_devices,
        operation_attributes.output_mem_config,
        operation_attributes.sub_core_grids,
        factory.index(),
        tensor_args.start_tensor.has_value());

    const auto& input = tensor_args.input;
    hash = ttsl::hash::hash_objects(
        hash,
        input.logical_shape().rank(),
        input.logical_shape(),
        input.padded_shape(),
        input.layout(),
        input.dtype(),
        input.memory_config());

    if (tensor_args.start_tensor.has_value()) {
        const auto& st = tensor_args.start_tensor.value();
        hash = ttsl::hash::hash_objects(
            hash,
            st.logical_shape().rank(),
            st.logical_shape(),
            st.padded_shape(),
            st.layout(),
            st.dtype(),
            st.memory_config());
    }

    const auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    hash = ttsl::hash::hash_objects(
        hash,
        output_spec.logical_shape().rank(),
        output_spec.logical_shape(),
        output_spec.padded_shape(),
        output_spec.layout(),
        output_spec.data_type(),
        output_spec.memory_config());

    return hash;
}
```

This is a *widening* custom hash, not a narrowing one. Every operation attribute is kept, the
input contributes five of its six default fields plus `padded_shape`, and the hash additionally
folds in the selected factory index and the full computed output spec — neither of which the
default key contains.

### `slice_start` / `slice_end` / `step`: all three hashed, none patched, all structural

All three are keyed at lines 306-308, and none is treated as a dynamic runtime arg. That is the
correct choice, because each one changes program structure and mode A would not repair any of
them:

- **`slice_end` and `step`** determine the output shape via
  `(end - start + step - 1) / step` (`slice_device_operation.cpp:212-217`), which sets the total
  work unit count and hence the `split_work_to_cores` result — the **core ranges** and per-group
  work counts. Core ranges are baked into the cached `Program`
  (`slice_program_factory_tile.cpp:26-32`, `slice_program_factory_rm.cpp:298-304`).
- **`step`** additionally selects the factory: any non-unit step routes to
  `SliceRmStrideProgramFactory` (`slice_device_operation.cpp:289-291`), a different kernel source
  entirely, and it is emitted directly into per-core runtime args
  (`slice_program_factory_rm_stride.cpp:128-130`).
- **`slice_start`** feeds `get_tiled_start_offset` / `get_rm_start_offset` (runtime, and so
  patchable), but it also feeds the row-major **CB page size** through the misalignment
  computation:

```217:232:ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_program_factory_rm.cpp
    uint32_t begins_bytes = output_tensor_start[-1] * input.element_size();
    uint32_t misalignment = begins_bytes % src_buffer_alignment;

    if (misalignment != 0) {
        alignment *= 2;
    }
    const uint32_t unpadded_row_size_bytes = output.padded_shape()[-1] * input.element_size();
    const uint32_t stick_size_aligned = tt::round_up(unpadded_row_size_bytes, alignment);

    const uint32_t l1_budget = ttnn::operations::data_movement::get_max_l1_space(input);

    SliceCbSizing s{
        .cb_page_size = stick_size_aligned,
        .num_read_per_barrier = 0,
        .misalignment = misalignment,
        .chunking = {stick_size_aligned, 1, stick_size_aligned},
    };
```

  A misaligned `slice_start[-1]` doubles the alignment and can flip the sub-row chunking decision
  at line 235, changing `cb_page_size` and the CB `total_size` at
  `slice_program_factory_rm.cpp:319-327`. CB sizing is structural. So `slice_start` must be hashed
  even though most of its influence is on runtime args, and the factory comment at lines 314-315
  says exactly that.

So the answer to "which of start/end/step are hashed, which are patched, which must be hashed" is:
all three are hashed, none is patched as a dynamic arg, and all three must be hashed. The op gets
this right.

## Omitted parameters

### 1. `tensor_args.end_tensor` — omitted entirely, including its engaged-ness

**Verdict: BUG.**

The hash folds in `tensor_args.start_tensor.has_value()` (line 315) and, when engaged, the start
tensor's full spec (lines 327-337). There is no corresponding term for `end_tensor` anywhere in
`compute_program_hash`. But `SliceTileTensorArgsProgramFactory` consumes it, and specifically
consumes its buffer at **compile time**:

```78:85:ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_program_factory_tile_tensor_args.cpp
    std::vector<uint32_t> reader_compile_time_args = {
        src0_cb_index, tensor_cb_index, num_dims, tile_width, tile_height};
    TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*start_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*end_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {src0_cb_index};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);
```

`TensorAccessorArgs` encodes the buffer's memory space as a compile-time config bit:

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

The end tensor is caller-supplied and its memory config is unconstrained: `slice.cpp:471-473`
requires only that it be on device, and its rank be 1 (`slice.cpp:432-435`). Nothing pins its
buffer type.

**Reproduction** (through `ttnn.slice` with tensor-valued bounds, i.e. the `use_tensor_args` path,
TILE layout, unit step, with `slice_dim` and `num_devices` supplied):

- **Call 1**: `start_tensor` in DRAM, `end_tensor` in **DRAM**. The reader compiles with
  `ArgConfig::IsDram` set for the end accessor.
- **Call 2**: identical input, identical `start_tensor`, `slice_dim`, `num_devices` and memory
  config, but `end_tensor` in **L1**. Every hashed term is unchanged — the end tensor contributes
  nothing to the key — so this is a cache hit.

The override rebuilds the descriptor and refreshes common runtime arg 2 to the L1 end buffer's
address (`slice_program_factory_tile_tensor_args.cpp:182`), but the compiled `TensorAccessor` at
`reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp:45` still resolves through the DRAM
bank table. The kernel reads the slice bounds from the wrong memory space and then slices with
whatever integers it found, producing a wrong-shaped read pattern with no error.

The asymmetry with `start_tensor` — which is hashed in full three lines earlier — makes this look
like a simple oversight rather than a deliberate relaxation. The fix is to mirror the
`start_tensor` block for `end_tensor`, including its `has_value()`.

### 2. `input.tensor_spec().page_config()` — only `layout()` is hashed, and the tile factories disagree about tiles

**Verdict: BUG.**

`layout()` collapses `PageConfig` to `ROW_MAJOR` vs `TILE`, discarding the `Tile` shape. `slice`
accepts `Layout::TILE` (`slice_device_operation.cpp:102-105`), and this op is the "mixed" case:
one tile factory is tile-aware and the other is not, so the omission is a bug under *both* halves
of the rule.

**The tile-aware half.** `SliceTileTensorArgsProgramFactory` reads the real tile shape and passes
it into the reader's compile-time args:

```73:82:ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_program_factory_tile_tensor_args.cpp
    std::uint32_t num_dims = static_cast<std::uint32_t>(input_tensor.padded_shape().rank());
    auto tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    uint32_t tile_width = tile_shape[1];
    uint32_t tile_height = tile_shape[0];

    std::vector<uint32_t> reader_compile_time_args = {
        src0_cb_index, tensor_cb_index, num_dims, tile_width, tile_height};
    TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*start_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*end_buffer).append_to(reader_compile_time_args);
```

The generated program therefore provably varies with `Tile`, and `Tile` is not in the key. Two
calls that differ only in tile geometry — say `Tile{32, 32}` versus `Tile{16, 32}` on the same
logical shape — collide exactly: `logical_shape` is equal, `padded_shape` is equal (both pad to
the same extents for these tile heights), `layout()` is `TILE` for both, `dtype` and
`memory_config` are equal, and the computed output spec is equal too, because
`compute_output_specs` constructs the output layout with `PageConfig(input_tensor.layout())`
(`slice_device_operation.cpp:252`), which discards the input's tile and defaults to 32x32 in both
cases. So the second call inherits the first's compile-time args 3 and 4 and slices on the wrong
tile grid.

**The non-tile-aware half.** `SliceTileProgramFactory` computes everything from the architectural
constants:

```26:39:ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_program_factory_tile.cpp
    uint32_t num_unpadded_tiles = output.physical_volume() / TILE_HW;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        args.sub_core_grids.has_value()
            ? tt::tt_metal::split_work_to_cores(args.sub_core_grids.value(), num_unpadded_tiles)
            : tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_unpadded_tiles);

    tt::tt_metal::Buffer* src0_buffer = input.buffer();
    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);
```

`tt::tile_size(format)` returns the byte size of a 32x32 tile (the tile-aware call is
`tile.get_tile_size(format)`), and it sets the CB page size and total size at lines 52-57. The
tile-count conversions at lines 66-71 use bare `TILE_WIDTH`/`TILE_HEIGHT`. The device operation
itself does the same in `get_tiled_start_offset` and `get_upper_start_offset`
(`slice_device_operation.cpp:35`, `60-71`), and the validator's tile checks are likewise against
the constants (`slice_device_operation.cpp:156-167`) rather than against the tensor's tile. So for
a non-32x32 input this factory builds a program with the wrong CB page size and the wrong work
split — and because `page_config` is unhashed, it does not even build a fresh wrong program, it
reuses the 32x32 entry.

Nothing anywhere in the op validates the tile geometry. The two factories using different idioms
for the same concept is itself a hazard: a reader of one would reasonably assume the other behaves
the same way.

### 3. `tensor_args.preallocated_output` — its own spec is not hashed

**Verdict: BUG.**

What the hash contains is `compute_output_specs(...)`, the *computed* spec (lines 339-347), not
the spec of the tensor actually written to. When a preallocated output is supplied,
`create_output_tensors` returns it unchanged and the factories bind its buffer:

```255:265:ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_device_operation.cpp
Tensor SliceDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output.value();
    }

    const auto& input = tensor_args.input;
    const auto output_spec = compute_output_specs(args, tensor_args);

    return create_device_tensor(output_spec, input.device());
}
```

Validation checks only the shape:

```135:143:ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_device_operation.cpp
    if (tensor_args.preallocated_output.has_value()) {
        const auto output_shape_required = compute_output_specs(args, tensor_args).logical_shape();
        const auto& out_tensor = tensor_args.preallocated_output.value();
        TT_FATAL(
            out_tensor.padded_shape() == output_shape_required,
            "The preallocated output tensor needs a shape of {}, however it is {}",
            output_shape_required,
            out_tensor.padded_shape());
    }
```

That is the only check. Nothing compares the preallocated tensor's dtype, memory config or buffer
type against the computed spec, and the writer's compile-time args are derived from the real
buffer:

```147:150:ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_program_factory_tile.cpp
    // --- Writer Kernel Descriptor ---
    // CB index via named compile-time arg (essential for fusion CB remapping).
    std::vector<uint32_t> writer_compile_time_args = {};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);
```

`TensorAccessorArgs` encodes the buffer's memory space into `ArgConfig::IsDram`
(`tt_metal/impl/buffers/tensor_accessor_args.cpp:146-157`), a compile-time arg. Mode A re-derives
runtime args and CB addresses; it never touches compile-time args, which are baked into the cached
`Program`.

**Reachability.** `ttnn::prim::slice` is directly callable and takes `output_mem_config` and
`preallocated_output` as independent parameters — nothing requires the former to describe the
latter. The `TT_FATAL` above constrains only the shape.

**Reproduction.** Take a `[1, 1, 64, 64]` TILE `bfloat16` input and slice
`[0,0,0,0]`-`[1,1,32,32]` with unit step.

- **Call 1**: `ttnn::prim::slice(input, start, end, step, /*output_mem_config=*/dram_interleaved,
  /*preallocated_output=*/out_dram)` where `out_dram` is a DRAM tensor of the required shape.
- **Call 2**: the same call in every argument except that `preallocated_output` is `out_l1`, an L1
  tensor of the same shape and dtype. `output_mem_config` is still `dram_interleaved`.

The hash carries `compute_output_specs(...)`, which is built from the input and
`args.output_mem_config` — identical across the two calls — so call 2 hits. The shape `TT_FATAL`
passes, since only the shape is compared. But the cached writer kernel was compiled with
`IsDram = true`, and call 2's destination is in L1. The writer issues DRAM-addressed NOC
transactions against an L1 address: the slice output is written to the wrong memory space, and the
L1 tensor the caller passed is left untouched.

The dtype variant of the same defect is reachable the same way. The computed spec always takes
`input_tensor.dtype()` (`slice_device_operation.cpp:252`), so a preallocated output with a
different dtype hashes identically while changing the writer's element size.

The severity is bounded by the wrapper rather than by any check: `ttnn::slice` derives
`output_mem_config` *from* the preallocated tensor on both entry points (`slice.cpp:123-124` and
`slice.cpp:459-460`), so callers that go through it cannot construct the mismatch. That is not
enforcement, and the primitive is public, so it does not change the verdict.

The fix is a `TT_FATAL` comparing the preallocated tensor's full `tensor_spec()` against the
computed one rather than just its shape. That is strictly better than hashing the preallocated
spec, since a mismatch here is a caller error in every case, not a configuration worth building a
second program for.

### 4. `input.storage` variant kind, and `start_tensor.storage` kind

**Verdict: VALID — pinned by validation.**

```100:101:ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_device_operation.cpp
    TT_FATAL(tensor_args.input.storage_type() == StorageType::DEVICE, "Operands to unpad need to be on device!");
    TT_FATAL(tensor_args.input.buffer() != nullptr, "Operands to unpad need to be allocated in buffers on device!");
```

The start tensor is pinned at the wrapper (`slice.cpp:468-470`) and its buffer is null-checked in
the factory (`slice_program_factory_tile_tensor_args.cpp:43`). Both carry no information, and the
input's check is re-run on hits through the substitution branch quoted under "Which validator runs
on a cache hit".

### 5. `input.tensor_layout().get_alignment()` (and the start tensor's)

**Verdict: VALID — invariant.**

`Alignment` influences the program only through quantities that are themselves hashed. Its primary
effect is on `padded_shape`, which the hash carries explicitly alongside `logical_shape` (lines
321-322) — that pairing is what makes the alignment redundant. The buffer-level alignment the
factories actually read is not the tensor's `Alignment` at all but the buffer's, derived from its
memory space:

```89:95:ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_program_factory_rm.cpp
    auto src_buffer_alignment = input_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? ::hal::get_dram_alignment()
                                    : ::hal::get_l1_alignment();
    auto dst_buffer_alignment = output_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? ::hal::get_dram_alignment()
                                    : ::hal::get_l1_alignment();
    auto alignment = std::max(src_buffer_alignment, dst_buffer_alignment);
```

which is a function of `input.memory_config()` and `output_spec.memory_config()`, both hashed.
`slice_program_factory_rm_stride.cpp:59-60` reads `Buffer::alignment()` directly, which is
likewise determined by buffer type plus page size, and page size is fixed by the hashed shape,
dtype and memory config. No factory reads `tensor_layout().get_alignment()`.

### 6. Buffer addresses (omitted by the default hash as well)

**Verdict: VALID — patched.**

Four of five factories have every address refreshed by the descriptor rebuild, covering per-core
args (for example the writer binding at `slice_program_factory_rm.cpp:375`) and common args (the
source binding at `slice_program_factory_tile.cpp:141` and the source/start/end trio at
`slice_program_factory_tile_tensor_args.cpp:180-182`). The RM-sharded factory carries its
addresses on CB bindings (`slice_program_factory_rm_sharded.cpp:281` and `293`), which the
override's CB-address-only descriptor patches positionally in the order the factory pushed them —
an ordering the factory comment at line 271 explicitly flags as a contract to maintain.

## Keys the custom hash adds beyond the default

Three, and each is load-bearing:

- **`input.padded_shape()`** (line 322), hashed *alongside* `logical_shape` rather than instead of
  it. Every factory computes its work split from padded extents, so this closes the gap the
  default's derivation-free key would leave.
- **`factory.index()`** (line 314). `select_program_factory` branches on layout, step, and the
  input/output sharding combination (`slice_device_operation.cpp:267-296`); folding the resulting
  variant index into the key means two configurations that would map to different kernels can
  never share an entry, independently of whether the inputs that drove the choice are all hashed.
- **The full computed output spec** (lines 339-347). The default hashes only inputs, so an output
  spec that varies through `compute_output_specs`'s shard-spec synthesis
  (`slice_device_operation.cpp:224-248`) would otherwise be invisible to the key. Since the
  synthesis depends on `generate_transpose_shard_spec` and on tile-alignment adjustments, this is
  a real risk that the op has closed.

The header comment (lines 300-303) frames all of this as a fix for weak hash distribution on
small-integer shape sequences, citing issue #47602. Whatever the original motivation, the effect
is a key that is strictly stronger than the default on the tensor-spec axis.

## Framework side effect of having a custom hash

```1012:1014:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
        if constexpr (requires { DeviceOperation::compute_program_hash(attrs, tensor_args); }) {
            return key;  // custom hash -> opt out beyond the op-identity prefix
        } else {
```

`ProgramCacheKey::canonical` degrades to the op type name alone, so a genuine 64-bit collision
resolves to a wrong hit instead of a rebuild. This is a mild irony here: the hash exists partly to
*improve* collision behaviour over the default's `hash_combine` distribution, and defining it
simultaneously removes the framework's collision backstop. The net is still favourable given how
much more the custom key mixes in, but it is worth knowing that the safety net is gone.

## Summary

| Omitted vs. default | Used by program? | Patched on hit? | Verdict |
|---|---|---|---|
| `end_tensor` (spec and engaged-ness) | Yes — reader compile-time args via `TensorAccessorArgs` | Address yes, accessor config **no** | **BUG** |
| `input.page_config` (`Tile`) | Yes — compile-time args (tensor-args factory), CB page size and work split (tile factory) | **No** | **BUG** |
| `preallocated_output` spec | Yes — writer `TensorAccessorArgs` compile-time args, element size | **No** | **BUG** |
| `input.storage` kind, `start_tensor.storage` kind | n/a | n/a | VALID — pinned by validation |
| `input.tensor_layout.alignment` | Only via hashed derivatives | n/a | VALID — invariant |
| Buffer addresses | Yes | Yes (rebuild; CB patch on RM-sharded) | VALID — patched |

**Three program-cache correctness bugs were found.** None stems from the patching design, which is
sound; all three are gaps in hash coverage that mode A structurally cannot compensate for, because
compile-time args are frozen in the cached `Program` and every one of these findings lands on a
compile-time arg. The `end_tensor` omission is an asymmetry with `start_tensor` that reads as an
oversight. The `page_config` omission is the unguarded-tile pattern, aggravated here because one
tile factory genuinely varies its compile-time args with the tile shape while the other hardcodes
32x32. The `preallocated_output` omission is different in character: the op does check the
preallocated tensor, but only its shape, so a caller reaching `ttnn::prim::slice` directly can
supply an output in a different memory space or dtype than the hashed computed spec describes and
inherit a writer kernel compiled for the wrong one.

**On whether this op is a reference implementation:** the CSV's `specimen_done=Y` is justified for
the *patching* half and for the treatment of `slice_start`/`slice_end`/`step`, and the code should
be read that way by other ops. The override re-derives through the same
`select_program_factory` + `create_descriptor` path the miss used, so there is no hand-maintained
arg-index table to fall out of sync — contrast `interleaved_to_sharded_partial`, whose
hand-written two-slot patch depends on constants that must track its factory. The one place slice
does take a shortcut (RM-sharded, CB addresses only) is accompanied by an argument for why the
remaining args are hash-pinned, and that argument checks out. The hash is also unusually
disciplined in the ways that matter for structure: it keys the factory index and the computed
output spec, neither of which the default covers. What the op is *not* a specimen for is
completeness of tensor coverage — it hashes `start_tensor` in full and `end_tensor` not at all, it
omits `page_config` while one of its factories consumes the tile shape at compile time, and it
hashes the computed output spec while the tensor actually written to is checked on shape alone.
Worth citing as a model for the cache-hit contract, with those three gaps called out.

Two CSV corrections. `get_dynamic_runtime_args` should be **N**: the member does not exist, only
an uncalled helper named `slice_rm_reader_dynamic_args`, and mode A would bypass it regardless.
And `own_hit_validator = N` understates the situation, since the dispatcher substitutes
`validate_on_program_cache_miss` on hits (`ttnn/api/ttnn/device_operation.hpp:262-266`), which
verdict #4 relies on.

## Recommendations

1. Hash `end_tensor` symmetrically with `start_tensor`: add
   `tensor_args.end_tensor.has_value()` to the first `hash_operation` call and an
   `if (tensor_args.end_tensor.has_value())` block mirroring lines 327-337. This is the fix for
   finding #1 and it is mechanical.
2. Close finding #2. The minimal fix is a tile guard in `validate_on_program_cache_miss` rejecting
   non-32x32 tiles, following `interleaved_to_sharded_op.cpp:94-98`; that makes the omission
   correct by construction but discards the tile-awareness
   `SliceTileTensorArgsProgramFactory` already has. The fix that preserves it is to add
   `input.tensor_spec().tile()` to the hash *and* make `SliceTileProgramFactory` and the
   `get_tiled_start_offset` helpers tile-aware in the same change. Do not add the hash term
   without the factory work, or vice versa: either alone leaves a wrong-program path.
3. Delete `slice_rm_reader_dynamic_args` and its declaration, and correct the three comments in
   `slice_program_factory_rm.cpp` (lines 34-40, 103-104) that describe reader arg 0 as riding on
   `get_dynamic_runtime_args`. Dead code that documents a mechanism the op does not use is
   actively misleading for exactly this kind of audit.
4. Strengthen the preallocated-output check at `slice_device_operation.cpp:135-143` to compare the
   full `tensor_spec()` against the computed spec rather than only `padded_shape`. This is the fix
   for finding #3. Prefer it over hashing the preallocated spec: a mismatch is a caller error in
   every case, so rejecting it loudly is better than building a second program for it. Because the
   op has no hit validator, the strengthened check runs on hits as well as misses.
5. Run this op's tests once under `-DTT_DESCRIPTOR_PATCHING_PARITY_CHECK`. The parity oracle in
   the mode-A branch (`mesh_device_operation_adapter.hpp:679-693`) will not catch findings #1, #2
   or #3, since all three are compile-time-arg staleness rather than runtime-arg staleness, but it does
   validate the RM-sharded shortcut at #6, which is the one place the op patches rather than
   rebuilds.
