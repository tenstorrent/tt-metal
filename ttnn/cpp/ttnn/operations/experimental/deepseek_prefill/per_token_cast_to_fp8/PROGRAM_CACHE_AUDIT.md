# Program Cache Audit — `experimental/deepseek_prefill/per_token_cast_to_fp8`

Audit of `PerTokenCastToFp8DeviceOperation::compute_program_hash` against the framework default
("hash everything") key.

| | |
|---|---|
| Device operation | `ttnn::experimental::prim::per_token_cast_to_fp8::PerTokenCastToFp8DeviceOperation` (`device/per_token_cast_to_fp8_device_operation.hpp:16`) |
| Custom hash | `device/per_token_cast_to_fp8_device_operation.cpp:159-174` |
| `operation_attributes_t` | `PerTokenCastToFp8Params` — `output_memory_config`, `round_scale_to_power_of_two` |
| `tensor_args_t` | `PerTokenCastToFp8Inputs` — `input_tensor` |
| Return | `std::tuple<Tensor, Tensor>` — the FP8_E4M3 output and the FLOAT32 per-token scale tensor, both op-allocated |
| Program factories | one: `PerTokenCastToFp8ProgramFactory` (legacy `CachedProgram`), branching internally on `input.layout()` for the ROW_MAJOR vs TILE work split |
| `override_runtime_arguments` | **Yes** (`device/per_token_cast_to_fp8_program_factory.cpp:352-358`, body at `323-339`) |
| `get_dynamic_runtime_args` | **No** |
| Cache-hit patch mechanism | **Op-owned override** — the factory re-derives per-dispatch state itself; `resolve_bindings` is never involved |

## Cache-hit patch mechanism

This is a `CachedProgram`-style factory, so the mesh adapter calls the factory's
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

Because the op owns its cache-hit re-derivation, the descriptor-era binding machinery is bypassed
entirely: no `resolve_bindings`, no `ResolvedBindings`, no `allow_inplace_output_tensor_alias`
decision, and no chance of a bail-to-slow-path. Aliasing is not a concern here in any case — both
outputs are freshly allocated by the op, so no input buffer can appear in the output region:

```152:157:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/per_token_cast_to_fp8/device/per_token_cast_to_fp8_device_operation.cpp
PerTokenCastToFp8DeviceOperation::tensor_return_value_t PerTokenCastToFp8DeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    auto [output_e4m3_spec, scale_spec] = compute_output_specs(attrs, tensor_args);
    auto* device = tensor_args.input_tensor.device();
    return {create_device_tensor(output_e4m3_spec, device), create_device_tensor(scale_spec, device)};
}
```

**Obligation on the hash.** Having an override is not the same as having a *complete* one, and this
override is deliberately address-only:

```323:339:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/per_token_cast_to_fp8/device/per_token_cast_to_fp8_program_factory.cpp
void apply_io_overrides(
    ttnn::device_operation::CachedProgram<PerTokenCastToFp8SharedVariables>& cached_program,
    const PerTokenCastToFp8Inputs& tensor_args,
    std::tuple<Tensor, Tensor>& tensor_return_value) {
    const auto& [output_e4m3, output_scale] = tensor_return_value;
    const auto& shared = cached_program.shared_variables;
    const uint32_t src_addr = tensor_args.input_tensor.buffer()->address();
    const uint32_t dst_e4m3_addr = output_e4m3.buffer()->address();
    const uint32_t dst_scale_addr = output_scale.buffer()->address();
    for (const auto& core : shared.all_cores_vec) {
        auto& reader_args = tt::tt_metal::GetRuntimeArgs(cached_program.program, shared.reader_kernel_id, core);
        reader_args[0] = src_addr;
        auto& writer_args = tt::tt_metal::GetRuntimeArgs(cached_program.program, shared.writer_kernel_id, core);
        writer_args[0] = dst_e4m3_addr;
        writer_args[1] = dst_scale_addr;
    }
}
```

The reader takes up to five runtime args, the writer up to six, and the compute kernel one. Only
`reader[0]`, `writer[0]` and `writer[1]` — the three buffer addresses — are re-applied. Every other
slot (`num_blocks`, `offset`/`start_row`, `core_units`, `H`, `rows_per_plane`,
`row_tiles_per_plane`, and the compute kernel's `num_blocks`) is frozen at the first miss, as are
all compile-time args, all CB sizes and formats, the `MathFidelity`/`fp32_dest_acc_en` settings, and
the core grid. The rest of this audit checks that each of those is a pure function of the hashed
set.

**Which validator runs on a cache hit.** This op leans on validation rather than hashing for the
tile configuration, so it matters exactly which validator executes on the offending second call. The
dispatcher runs one, not both:

```262:266:ttnn/api/ttnn/device_operation.hpp
    if constexpr (HasValidateOnProgramCacheHit<mesh_device_operation_t>) {
        mesh_device_operation_t::validate_on_program_cache_hit(operation_attributes, tensor_args);
    } else {
        mesh_device_operation_t::validate_on_program_cache_miss(operation_attributes, tensor_args);
    }
```

This op **defines** `validate_on_program_cache_hit`, so it takes the first branch and the hit
validator *replaces* the miss validator on every hit. A hit validator that pins less than the miss
validator would silently lose the difference on exactly the calls that reuse a cached program; here
it is a total delegation (`device/per_token_cast_to_fp8_device_operation.cpp:125-128`, quoted in
item 1), so the hit path drops nothing and every `TT_FATAL` in
`validate_on_program_cache_miss` runs on hits as well as misses. That is what licenses the
"VALID — pinned by validation" verdicts in items 1 and 3.

## Baseline: what the default hash would cover

`hash_objects_with_default_seed(type_hash<PerTokenCastToFp8DeviceOperation>, attrs, tensor_args)`
would produce:

| Source | Fields |
|---|---|
| `operation_attributes` | `output_memory_config`, `round_scale_to_power_of_two` |
| `input_tensor` | storage kind; `logical_shape`; `dtype`; `page_config`; `memory_config`; `alignment` |

Both output tensors are op-allocated and therefore outside both keys; their specs derive from
`input.logical_shape()` and `attrs.output_memory_config`
(`device/per_token_cast_to_fp8_device_operation.cpp:130-150`).

## What the custom hash covers

```159:174:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/per_token_cast_to_fp8/device/per_token_cast_to_fp8_device_operation.cpp
ttsl::hash::hash_t PerTokenCastToFp8DeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input_tensor;
    const auto tile_shape = input.tensor_spec().tile().get_tile_shape();
    const auto face_shape = input.tensor_spec().tile().get_face_shape();
    return tt::tt_metal::operation::hash_operation<PerTokenCastToFp8DeviceOperation>(
        attrs,
        input.dtype(),
        input.layout(),  // ROW_MAJOR and TILE select different program factories
        input.memory_config(),
        input.logical_shape(),
        tile_shape[0],
        tile_shape[1],
        face_shape[0],
        face_shape[1]);
}
```

`attrs` is passed whole, so both attribute fields survive. The input tensor is decomposed
selectively: `page_config` is expanded into `layout()` plus the tile and face *shapes*, but not the
tile's transpose flags. `storage` and `alignment` are dropped.

## Omitted parameters

### 1. `input.tensor_spec().tile()`'s transpose configuration

**Verdict: VALID — pinned by validation.**

The hash records four numbers from the `Tile` (`tile_h`, `tile_w`, `face_h`, `face_w`) but not
`get_transpose_within_face()` / `get_transpose_of_faces()`. Those are pinned off, with the hash gap
called out explicitly at the guard site:

```70:76:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/per_token_cast_to_fp8/device/per_token_cast_to_fp8_device_operation.cpp
    const auto& tile = input.tensor_spec().tile();
    // The compute kernels build cb_in with the default, non-transposed tile descriptor and
    // compute_program_hash only records tile/face shapes. A transposed tile would be unpacked with the
    // wrong face ordering and silently produce incorrect scales and FP8 values, so reject it here.
    TT_FATAL(
        !tile.get_transpose_within_face() && !tile.get_transpose_of_faces(),
        "per_token_cast_to_fp8: transposed TILE inputs are not supported");
```

This is a good example of the pattern done right: the omission is deliberate, documented at the
guard rather than at the hash, and — critically — the guard runs on cache hits too:

```125:128:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/per_token_cast_to_fp8/device/per_token_cast_to_fp8_device_operation.cpp
void PerTokenCastToFp8DeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(attrs, tensor_args);
}
```

A miss-only validator would be worthless here — the whole risk is the *second* call inheriting the
first call's program, and a miss-only check never runs on that call. This delegation is the
load-bearing fact behind every "pinned by validation" verdict in this document.

Two refinements are worth recording. First, the guard does **not** stand alone: the two `TT_FATAL`s
immediately after it pin the tile's *height and width* as well, not just the transpose flags.
`tile_h * tile_w == 1024` and `BLOCK_W % tile_w == 0` are enforced unconditionally
(`device/per_token_cast_to_fp8_device_operation.cpp:82-94`), and only one entry in the hardware's
table of legal tile shapes has `tile_h * tile_w == 1024`:

```19:34:tt_metal/impl/data_format/tile.cpp
constexpr std::array<std::array<std::array<uint32_t, 2>, 2>, 12> TILE_FACE_HW_CHOICES = {
    {// TODO: add other tile shapes once llk supported it
     {{{32, 32}, {16, 16}}},
     {{{16, 32}, {16, 16}}},
     {{{32, 16}, {16, 16}}},
     {{{16, 16}, {16, 16}}},
     // these shapes are not supported yet on llk, just for host loopback
     {{{8, 32}, {8, 16}}},
     {{{4, 32}, {4, 16}}},
     {{{2, 32}, {2, 16}}},
     {{{1, 32}, {1, 16}}},
     // these shapes are not supported yet on llk, just for host loopback
     {{{8, 16}, {8, 16}}},
     {{{4, 16}, {4, 16}}},
     {{{2, 16}, {2, 16}}},
     {{{1, 16}, {1, 16}}}}};
```

A `Tile` whose shape is absent from that table cannot be constructed at all
(`tt_metal/impl/data_format/tile.cpp:45-47` throws), so the admissible tile space here is the single
value `{32, 32}` with faces `{16, 16}`. See omission 7 for the full tile-awareness analysis.

Second, the in-source comment describes the transpose flags as something "`compute_program_hash`
only records tile/face shapes" leaves out — implying a gap specific to the custom hash. It is
broader than that: `Tile`'s own reflected attribute set omits the transpose flags entirely, so the
*default* hash would not have distinguished a transposed tile either.

```46:47:tt_metal/api/tt-metalium/tile.hpp
    static constexpr auto attribute_names = std::forward_as_tuple("tile_shape", "face_shape", "num_faces");
    auto attribute_values() const { return std::forward_as_tuple(tile_shape, face_shape, num_faces); }
```

```49:58:tt_metal/api/tt-metalium/tile.hpp
private:
    std::array<uint32_t, 2> tile_shape = {constants::TILE_HEIGHT, constants::TILE_WIDTH};
    std::array<uint32_t, 2> face_shape = {constants::FACE_HEIGHT, constants::FACE_WIDTH};
    uint32_t tile_hw = constants::TILE_HW;
    uint32_t face_hw = constants::FACE_HW;
    uint32_t num_faces = constants::TILE_HW / constants::FACE_HW;
    uint32_t partial_face = 0;
    uint32_t narrow_tile = 0;
    bool transpose_within_face = false;  // transpose datums within each face
    bool transpose_of_faces = false;     // transpose the face order
```

`Tile::operator==` compares only `tile_shape` and `face_shape`
(`tt_metal/impl/data_format/tile.cpp:122-124`), so the canonical collision-resolution key would not
have caught it either. The `TT_FATAL` is therefore the only mechanism that can close this hole, for
any op, and rejecting the input is the right call.

The hashed tile/face shapes are genuinely load-bearing and correctly kept: `tile_h`/`tile_w` set the
work-split unit, the CB page sizes and `tiles_per_block`, and all four are passed as compile-time
args to both dataflow kernels
(`device/per_token_cast_to_fp8_program_factory.cpp:84-95, 159-160, 175-185`) where the reader uses
them to compute `face_elems`, `num_faces` and `tiles_per_block`
(`device/kernels/dataflow/reader_per_token_cast_to_fp8.cpp:41-49`).

### 2. `input.tensor_layout().get_alignment()`

**Verdict: CAVEAT.**

This is the only omission with a live path into a frozen compile-time arg. The input's accessor
args are compile-time:

```159:161:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/per_token_cast_to_fp8/device/per_token_cast_to_fp8_program_factory.cpp
    std::vector<uint32_t> reader_ct_args = {
        cb_in_idx, input_block_bytes, cb_scaler_idx, tile_h, tile_w, face_h, face_w};
    TensorAccessorArgs(src_buffer).append_to(reader_ct_args);
```

and for a non-sharded buffer that emits the config bitset plus `aligned_page_size`:

```194:198:tt_metal/impl/buffers/tensor_accessor_args.cpp
    if (args_config_.test(tensor_accessor::ArgConfig::Sharded)) {
        CMAKE_UNIQUE_NAMESPACE::append_sharded_args(*buffer_, args_config_, compile_time_args, /* is_runtime */ false);
    } else {
        compile_time_args.push_back(args_config_.raw());
        auto aligned_page_size = buffer_ ? buffer_->aligned_page_size() : 0;
```

```656:658:tt_metal/impl/buffers/buffer.cpp
uint32_t Buffer::alignment() const { return allocator_->get_alignment(this->buffer_type()); }

DeviceAddr Buffer::aligned_page_size() const { return align(page_size(), this->alignment()); }
```

The final rounding uses an allocator constant keyed on `buffer_type`, which lives inside the hashed
`memory_config` (and is pinned to DRAM interleaved anyway by `validate_device_tensor`,
`device/per_token_cast_to_fp8_device_operation.cpp:27-31`). But `page_size()` derives from the
*padded* shape, and the padded shape is a function of the logical shape **and** the
`TensorLayout::Alignment`. The hash keeps `logical_shape()` and not `Alignment`.

What would break it: an input built with a non-canonical `Alignment` that widens the padded last dim
while leaving `logical_shape` unchanged. `validate_on_program_cache_miss` constrains the *logical*
shape only (`H % BLOCK_W == 0`, `rank >= 2`, `M > 0` — lines 96-122), never the padded width. Two
such calls collide in the hash and the second inherits the first's `aligned_page_size` compile-time
arg, so the reader strides through DRAM pages at the wrong pitch and quantizes misaligned data. The
`ROW_MAJOR` path is the more exposed one; on the TILE path the padded shape is forced to tile
multiples, which the hashed tile shape already pins.

Nothing in the codebase constructs such a tensor today — the production caller feeds the output of
`ttnn.to_layout(latent, ttnn.ROW_MAJOR_LAYOUT)`
(`models/demos/deepseek_v3_d_p/utils/kv_cache_utils.py:170-173`), which uses canonical alignments.
The guard that closes it: add `input.padded_shape()` to the hash (as the sibling
`moe_padding_config` does), or hash `input.tensor_spec()` outright (as the sibling
`per_token_cast_back` does), or `TT_FATAL` that the padded and logical last dims agree.

### 3. `input.storage` variant kind (device vs host)

**Verdict: VALID — pinned by validation.**

```27:31:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/per_token_cast_to_fp8/device/per_token_cast_to_fp8_device_operation.cpp
void validate_device_tensor(const Tensor& tensor, const std::string& name) {
    TT_FATAL(tensor.storage_type() == ttnn::StorageType::DEVICE, "{} must be on device", name);
    TT_FATAL(tensor.buffer() != nullptr, "{} must have a buffer", name);
    TT_FATAL(is_dram_interleaved(tensor.memory_config()), "{} must be DRAM interleaved", name);
}
```

Constant across every admissible call, so it carries no information.

### 4. Buffer addresses (omitted by the default hash too)

**Verdict: VALID — patched, and required.** The input address and both output addresses are
re-applied on every hit by `apply_io_overrides`
(`device/per_token_cast_to_fp8_program_factory.cpp:332-338`, quoted above). Addresses must never be
hashed.

### 5. The FP8 scale values themselves

**Verdict: VALID — invariant.**

The per-token scale is computed on device — the compute kernel takes a block amax and forms
`scale = clamp(amax, 1e-4) / 448` — and is written to a tensor the op allocates. No scale value ever
crosses the host boundary, so none can reach a runtime or compile-time arg and none is hashable.

What *is* structural about the FP8 path is hashed or `constexpr`:

- `BLOCK_W = 128`, `E4M3_MAX_NORMAL = 448.0f`, `SCALE_CLAMP_MIN = 1.0e-4f` are `constexpr`
  (`per_token_cast_to_fp8.hpp:16-18`) and are bit-cast straight into the compute kernel's
  compile-time args (`device/per_token_cast_to_fp8_program_factory.cpp:200-215`). They cannot vary,
  so nothing can go stale.
- `round_scale_to_power_of_two` **does** vary per call and **is** a compile-time arg
  (`device/per_token_cast_to_fp8_program_factory.cpp:215`), which is exactly why hashing it matters.
  It is covered because `attrs` is hashed whole
  (`device/per_token_cast_to_fp8_device_operation.cpp:165`) and the field is a public member of
  `PerTokenCastToFp8Params` (`device/per_token_cast_to_fp8_device_operation_types.hpp:11-14`). Had
  the hash cherry-picked `attrs.output_memory_config` instead of passing `attrs`, this would have
  been a straightforward BUG: two calls differing only in `round_scale_to_power_of_two` would share
  a program compiled for the other rounding mode.

The block size and the scale value fall on opposite sides of the structure/data line, and this op
places both correctly: the block size is a compile-time constant, the scale is device data.

### 6. Non-address runtime args and the core grid (set only in `create`)

**Verdict: VALID — invariant** (every value below is a function of the hashed set).

Not omitted parameters as such, but they are the slots an incomplete override would strand, so they
need proof rather than assumption. All of them come from `M`, `H`, `tile_h`, `scale_blocks_per_row`
and the compute grid:

```240:245:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/per_token_cast_to_fp8/device/per_token_cast_to_fp8_program_factory.cpp
    const auto& input_shape = input.logical_shape();
    const auto [M, H] = fold_M_H(input_shape);  // M = rows, H = width (last dim)
    const auto tile_shape = input.tensor_spec().tile().get_tile_shape();
    const uint32_t tile_h = tile_shape[0];
    const uint32_t scale_blocks_per_row = H / fp8::BLOCK_W;  // 128-wide column-blocks per row
    const uint32_t scale_page_bytes = output_scale.buffer()->aligned_page_size();
```

- `M` and `H` fold `input.logical_shape()` (hashed) — `fold_M_H` is a pure function of it
  (`device/per_token_cast_to_fp8_program_factory.cpp:32-38`).
- `tile_h` is hashed directly.
- `rows_per_plane` / `row_tiles_per_plane` / `total_units` on the TILE branch read
  `input_shape[rank-2]` and divide by `tile_h` — both hashed
  (`device/per_token_cast_to_fp8_program_factory.cpp:261-265`).
- `scale_page_bytes` comes from `output_scale`, which the op itself allocated from
  `scale_output_shape(input.logical_shape())` and `attrs.output_memory_config` — both hashed
  (`device/per_token_cast_to_fp8_device_operation.cpp:33-43, 142-147`). No unhashed alignment can
  leak in through this one, unlike the *input* accessor in item 2.
- `compute_with_storage_grid_size()` is a device property, and the program cache is per-device.
  `split_work_to_cores` is deterministic in `(grid, total_units)`, so `num_cores`, `all_cores` and
  the per-core unit counts are fixed by the hashed key. That matters doubly because the core
  `CoreRangeSet` is baked into the cached `Program` and is refreshed by nothing.
- The layout branch itself (which changes `total_units`, `scale_scratch_bytes`, the
  `INPUT_TILE_LAYOUT` define, and the runtime-arg shape) is selected by `input.layout()`
  (`device/per_token_cast_to_fp8_program_factory.cpp:348`), which is hashed. The two layouts
  therefore cannot share a cache entry — which is necessary, since their reader/writer runtime args
  have entirely different meanings at the same indices
  (`device/kernels/dataflow/reader_per_token_cast_to_fp8.cpp:25-35`).

### 7. Tile and face geometry (`page_config`) — the tile-awareness check

**Verdict: VALID — invariant** (nothing is omitted: the tile geometry is in the key). `tile_shape[0..1]` and
`face_shape[0..1]` are hashed explicitly, and everything else `Tile` carries is either derived from
those two shapes or pinned by a validator that runs on the hit path.

This op is genuinely tile-aware — it reads the tensor's real tile spec rather than assuming 32x32 —
so the obligation runs in the opposite direction from the usual hardcoded-32x32 hazard: because the
program provably varies with the tile, the tile *must* be reachable from the key.

What the geometry feeds. It is read once per factory entry:

```82:87:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/per_token_cast_to_fp8/device/per_token_cast_to_fp8_program_factory.cpp
    const auto tile_shape = input.tensor_spec().tile().get_tile_shape();
    const auto face_shape = input.tensor_spec().tile().get_face_shape();
    const uint32_t tile_h = tile_shape[0];
    const uint32_t tile_w = tile_shape[1];
    const uint32_t face_h = face_shape[0];
    const uint32_t face_w = face_shape[1];
```

and reaches all four categories of frozen state:

- **CB page sizes and totals.** `TILE_BYTES_FP32 = tile_h * tile_w * 4` and
  `block_wt = block_w / tile_w` set `tiles_per_block`
  (`device/per_token_cast_to_fp8_program_factory.cpp:92-95`); `in_tile_bytes` and
  `output_e4m3_page_bytes` are `tile_h * tile_w` scaled by the element size
  (`101-102`). Those drive `cb_in` (`132-134`), the five fp32 CBs (`137-141`) and `cb_output_e4m3`
  (`145-148`).
- **Compile-time args.** All four scalars are passed to the reader
  (`device/per_token_cast_to_fp8_program_factory.cpp:159-160`) and the writer (`175-185`), and
  `tile_w` to the compute kernel (`203-215`). The reader turns them into `face_elems`, `num_faces`
  and `tiles_per_block` as `constexpr`
  (`device/kernels/dataflow/reader_per_token_cast_to_fp8.cpp:41-49`).
- **Per-core work split.** On the TILE branch `row_tiles_per_plane = div_up(rows_per_plane, tile_h)`
  feeds `total_units`, hence `split_work_to_cores`, hence the baked-in `CoreRangeSet`
  (`device/per_token_cast_to_fp8_program_factory.cpp:261-265, 278-279`). On the ROW_MAJOR branch it
  reappears in the per-core `num_blocks` (`310`).
- **Scratch sizing.** `scale_scratch_bytes = tile_h * scale_page_bytes` on the TILE branch (`265`).

Where it reaches the key. Directly, as four scalars:

```162:173:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/per_token_cast_to_fp8/device/per_token_cast_to_fp8_device_operation.cpp
    const auto tile_shape = input.tensor_spec().tile().get_tile_shape();
    const auto face_shape = input.tensor_spec().tile().get_face_shape();
    return tt::tt_metal::operation::hash_operation<PerTokenCastToFp8DeviceOperation>(
        attrs,
        input.dtype(),
        input.layout(),  // ROW_MAJOR and TILE select different program factories
        input.memory_config(),
        input.logical_shape(),
        tile_shape[0],
        tile_shape[1],
        face_shape[0],
        face_shape[1]);
```

Two calls with the same padded shape, dtype and memory config but different `Tile` therefore land on
different cache entries. The mirror-image bug does not apply here.

The four hashed scalars are also sufficient, not merely present. `Tile`'s remaining state is
computed from them in the constructor — `tile_hw`, `face_hw`, `num_faces`, `partial_face` and
`narrow_tile` are all pure functions of `tile_shape` and `face_shape`:

```63:67:tt_metal/impl/data_format/tile.cpp
    tile_hw = this->tile_shape[0] * this->tile_shape[1];
    face_hw = face_shape[0] * face_shape[1];
    num_faces = tile_hw / face_hw;
    partial_face = static_cast<uint32_t>(this->tile_shape[0] < constants::TILE_HEIGHT);
    narrow_tile = static_cast<uint32_t>(this->tile_shape[1] < constants::TILE_WIDTH);
```

and `face_shape` is itself a table lookup on `tile_shape`
(`tt_metal/impl/data_format/tile.cpp:36-47`), so hashing both is redundant but harmless. The only
`Tile` state left over is the transpose pair, handled in item 1 — where the guard, as shown there,
pins tile height and width too, leaving `{32, 32}` as the single admissible tile shape.

Note also that `PageConfig` only *carries* a `Tile` on the TILE alternative
(`tt_metal/api/tt-metalium/experimental/tensor/spec/layout/page_config.hpp:19-27`); for a ROW_MAJOR
input `tensor_spec().tile()` returns the default-constructed 32x32 `Tile{}`
(`tt_metal/impl/tensor/spec/layout/page_config.cpp:179-184`), so on that branch the four hashed
scalars are constants. They are load-bearing only on the TILE branch, and `input.layout()` already
keeps the two branches in separate cache entries.

## Keys the custom hash adds beyond the default

None. `input.layout()` plus the tile and face shapes are a partial decomposition of the default's
`page_config`, not an addition; every other hashed value is also in the default key. The
decomposition is lossy in exactly one place — the transpose flags — but as item 1 shows, `Tile` does
not reflect those, so the default hash would have lost them too.

## Framework side effect of having a custom hash

```1012:1014:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
        if constexpr (requires { DeviceOperation::compute_program_hash(attrs, tensor_args); }) {
            return key;  // custom hash -> opt out beyond the op-identity prefix
        } else {
```

Defining `compute_program_hash` degrades `ProgramCacheKey::canonical` to just the op type name, so a
64-bit collision between two different `per_token_cast_to_fp8` configurations resolves to a wrong hit
rather than a rebuild. Inherent to every custom-hash op, but it raises the cost of the alignment
CAVEAT above.

## Summary

| Omitted param | Used by program? | Patched on hit? | Verdict |
|---|---|---|---|
| `input.tile()` transpose flags | Would be (unpack face order) | n/a | VALID — pinned by validation |
| `input.tensor_layout.alignment` | Yes (`aligned_page_size` compile-time arg) | No | CAVEAT |
| `input.storage` kind | n/a | n/a | VALID — pinned by validation |
| Buffer addresses (3) | Yes | Yes (`apply_io_overrides`) | VALID — patched |
| FP8 scale values | Yes, as device data | n/a | VALID — invariant |
| Non-address rt args, core grid (create-only) | Yes | No | VALID — invariant (functions of the hashed set) |
| `input.page_config` tile/face geometry | Yes (CB pages, CT args, work split) | No | VALID — unused/covered (hashed as four scalars) |

**No program-cache correctness bug found.** Every frozen slot — both dataflow kernels' compile-time
args, the compute kernel's `constexpr` FP8 constants and rounding mode, every CB size and format,
the `INPUT_TILE_LAYOUT` define, the core grid, and all six non-address runtime args — is a pure
function of {`output_memory_config`, `round_scale_to_power_of_two`, `input.dtype`, `input.layout`,
`input.memory_config`, `input.logical_shape`, `tile_h`, `tile_w`, `face_h`, `face_w`} plus values
pinned by a validator that runs on hits as well as misses, plus device-fixed constants. The one live
assumption is that no caller supplies a non-canonically-aligned input.

The tile-geometry check (item 7) also comes back clean, in the way that matters for a tile-aware op:
this factory never uses `tt::tile_size(...)` or a bare `TILE_HW` / `TILE_WIDTH` / `TILE_HEIGHT`
tile-count conversion — it reads `tensor_spec().tile()` everywhere — and the geometry it reads is
hashed. It is worth stating that a tile-aware factory and a cherry-picked hash are a dangerous
combination in general: if these four scalars were ever dropped from the hash while the factory kept
reading the real tile, a second call with the same padded shape, dtype and memory config but a
different `Tile` would silently reuse the first call's CB page sizes and tile counts. Here they are
present, and the `tile_h * tile_w == 1024` guard pins the tile to `{32, 32}` on top of that.

Relative to the family: the alignment gap in item 2 is **not** specific to this op. The sibling
`pack_scaled_fp8_kv_cache` has the identical structure — hash `logical_shape`, consume an
`aligned_page_size` derived from the padded shape — while `per_token_cast_back` avoids it by hashing
each input's whole `TensorSpec`. Since two of the four ops share the gap and a third already
demonstrates the fix in the same directory, this deserves a family-level change rather than a
one-file patch.

## Recommendations

1. Close the alignment CAVEAT. The cheapest correct fix is to replace
   `input.dtype() / input.layout() / input.memory_config() / input.logical_shape()` with
   `input.tensor_spec()`, keeping the four tile/face scalars only if you want to keep them
   self-documenting (they are already inside the spec). This makes the op's key strictly a
   *superset* of the current one, and costs no extra cache entries for legal inputs, because
   `validate_on_program_cache_miss` already pins every additional field a `TensorSpec` would
   contribute. Apply the same change to `pack_scaled_fp8_kv_cache` in the same pass. Keep the
   transpose `TT_FATAL` regardless: `TensorSpec` does not close that hole either, because `Tile`
   reflects only `tile_shape` / `face_shape` / `num_faces`
   (`tt_metal/api/tt-metalium/tile.hpp:46-47`).
2. If keeping the current cherry-picked form, add `input.padded_shape()` to the hash — one line, and
   it is exactly what makes dropping `alignment` safe (this is the pattern the sibling
   `moe_padding_config` uses).
3. Add a program-cache regression test that calls the op twice on one device with different `M`
   (same `H`, same dtype, same layout) and asserts both results. `assert_fastpath_parity`
   (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:679-693`) is the natural oracle for an
   incomplete `override_runtime_arguments`, but it is wired into the descriptor adapter only and this
   op still uses the legacy `CachedProgram` factory, so it does not cover this op today.
