# Program Cache Audit — `experimental/deepseek_prefill/per_token_cast_back`

Audit of `PerTokenCastBackDeviceOperation::compute_program_hash` against the framework default
("hash everything") key.

| | |
|---|---|
| Device operation | `ttnn::experimental::prim::per_token_cast_back::PerTokenCastBackDeviceOperation` (`device/per_token_cast_back_device_operation.hpp:17`) |
| Custom hash | `device/per_token_cast_back_device_operation.cpp:229-244` |
| `operation_attributes_t` | `PerTokenCastBackParams` — `output_dtype`, `output_memory_config`, `narrow_scales_to_bf16`, `token_count_aware`, `experts_per_chip`, `scales_from_metadata` |
| `tensor_args_t` | `PerTokenCastBackInputs` — `input_e4m3`, `input_scale`, and the optionals `expert_region_offsets`, `expert_token_counts`, `global_expert_idx_table` |
| Program factories | one: `PerTokenCastBackProgramFactory` (legacy `CachedProgram`), branching internally on `attrs.token_count_aware` |
| `override_runtime_arguments` | **Yes** (`device/per_token_cast_back_program_factory.cpp:434-474`) |
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

Because the op supplies `override_runtime_arguments`, the descriptor-era binding machinery is
bypassed entirely: there is no `resolve_bindings` call, no `ResolvedBindings`, no
`allow_inplace_output_tensor_alias` decision, and no bail-to-slow-path. Aliasing is not a concern
anyway — the output is a fresh allocation
(`device/per_token_cast_back_device_operation.cpp:224-227`), so no input buffer appears in the
output region.

**Obligation on the hash.** The override is address-only, and on the token-count-aware path it has
to cover five addresses per kernel rather than one:

```434:473:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/per_token_cast_back/device/per_token_cast_back_program_factory.cpp
void PerTokenCastBackProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const PerTokenCastBackParams& operation_attributes,
    const PerTokenCastBackInputs& tensor_args,
    Tensor& tensor_return_value) {
    auto& program = cached_program.program;
    auto& shared = cached_program.shared_variables;

    uint32_t src_e4m3_addr = tensor_args.input_e4m3.buffer()->address();
    uint32_t src_scale_addr = tensor_args.input_scale.buffer()->address();
    uint32_t dst_addr = tensor_return_value.buffer()->address();
    const bool token_count_aware = operation_attributes.token_count_aware;

    if (token_count_aware) {
        const uint32_t region_addr = tensor_args.expert_region_offsets->buffer()->address();
        const uint32_t counts_addr = tensor_args.expert_token_counts->buffer()->address();
        const uint32_t table_addr = tensor_args.global_expert_idx_table->buffer()->address();
        for (const auto& core : shared.all_cores_vec) {
            auto& reader_args = tt::tt_metal::GetRuntimeArgs(program, shared.reader_kernel_id, core);
            reader_args[0] = src_e4m3_addr;
            reader_args[1] = src_scale_addr;
            reader_args[2] = region_addr;
            reader_args[3] = counts_addr;
            reader_args[4] = table_addr;

            auto& writer_args = tt::tt_metal::GetRuntimeArgs(program, shared.writer_kernel_id, core);
            writer_args[0] = dst_addr;
            writer_args[1] = region_addr;
            writer_args[2] = counts_addr;
            writer_args[3] = table_addr;
        }
    } else {
        for (const auto& core : shared.all_cores_vec) {
            auto& reader_args = tt::tt_metal::GetRuntimeArgs(program, shared.reader_kernel_id, core);
            reader_args[0] = src_e4m3_addr;
            reader_args[1] = src_scale_addr;
            auto& writer_args = tt::tt_metal::GetRuntimeArgs(program, shared.writer_kernel_id, core);
            writer_args[0] = dst_addr;
        }
    }
```

Checking this against the create path arg-for-arg: on the token-count-aware branch the reader is set
with `{e4m3, scale, region, counts, table, i, H}` and the writer with
`{dst, region, counts, table, i, H}` (`device/per_token_cast_back_program_factory.cpp:377-397`).
Every address slot in both is re-applied; `i` (the core index) and `H` are left frozen. On the plain
branch the reader is `{e4m3, scale, num_blocks, row_offset, rows_for_core, H}` and the writer is
`{dst, num_blocks, row_offset, rows_for_core, H}`
(`device/per_token_cast_back_program_factory.cpp:418-425`); again every address is re-applied and
`num_blocks` / `row_offset` / `rows_for_core` / `H` are frozen. The compute kernel's args
(`{num_blocks}` on the plain path, empty on the token-count-aware path) are never touched.

The branch selection itself is correct: `token_count_aware` is read from the *current*
`operation_attributes`, and it is hashed, so a cached program can never be visited by a call on the
other branch.

So the hash must cover: `H`, `num_blocks`, `row_offset`, `rows_for_core`, the core index enumeration,
every compile-time arg, every CB size and format, `MathFidelity`, `unpack_to_dest_mode`, the
`TOKEN_COUNT_AWARE` define, and the core grid.

**Which validator runs on a cache hit.** One verdict below (item 1) rests on a `TT_FATAL` rather
than on the hash, so it matters exactly which validator executes on the offending second call. The
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
it is a total delegation, so the hit path drops nothing:

```211:214:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/per_token_cast_back/device/per_token_cast_back_device_operation.cpp
void PerTokenCastBackDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(attrs, tensor_args);
}
```

Every `TT_FATAL` reached from `validate_on_program_cache_miss` therefore runs on hits too.

## Baseline: what the default hash would cover

`hash_objects_with_default_seed(type_hash<PerTokenCastBackDeviceOperation>, attrs, tensor_args)`
would produce:

| Source | Fields |
|---|---|
| `operation_attributes` | `output_dtype`, `output_memory_config`, `narrow_scales_to_bf16`, `token_count_aware`, `experts_per_chip`, `scales_from_metadata` |
| `input_e4m3` | storage kind; `logical_shape`; `dtype`; `page_config`; `memory_config`; `alignment` |
| `input_scale` | storage kind; `logical_shape`; `dtype`; `page_config`; `memory_config`; `alignment` |
| `expert_region_offsets` | engaged flag, then storage kind + the same `tensor_spec` fields |
| `expert_token_counts` | engaged flag, then storage kind + the same `tensor_spec` fields |
| `global_expert_idx_table` | engaged flag, then storage kind + the same `tensor_spec` fields |

The output tensor is op-allocated and outside both keys; its spec is `input_e4m3.logical_shape()`
plus `attrs.output_dtype` and `attrs.output_memory_config`
(`device/per_token_cast_back_device_operation.cpp:216-222`).

## What the custom hash covers

```229:244:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/per_token_cast_back/device/per_token_cast_back_device_operation.cpp
ttsl::hash::hash_t PerTokenCastBackDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    // Hash each tensor's full TensorSpec (shape + dtype + layout + tile + memory config) rather than
    // cherry-picking fields, so nothing that changes program structure is missed. The token-count-aware
    // metadata tensors are optional (absent on the plain path, which attrs.token_count_aware distinguishes).
    const auto opt_spec = [](const std::optional<Tensor>& t) {
        return t.has_value() ? std::optional<tt::tt_metal::TensorSpec>(t->tensor_spec()) : std::nullopt;
    };
    return tt::tt_metal::operation::hash_operation<PerTokenCastBackDeviceOperation>(
        attrs,
        tensor_args.input_e4m3.tensor_spec(),
        tensor_args.input_scale.tensor_spec(),
        opt_spec(tensor_args.expert_region_offsets),
        opt_spec(tensor_args.expert_token_counts),
        opt_spec(tensor_args.global_expert_idx_table));
}
```

This is the most complete hash of the four `deepseek_prefill` ops audited. `attrs` is passed whole,
so all six attribute fields survive. Each of the five tensors contributes its entire `TensorSpec` —
which is exactly what a `Tensor` contributes to the default key apart from its storage variant.
`opt_spec` maps `nullopt` to `nullopt`, so the engaged/disengaged distinction is preserved too.

**The CSV classification of this op as "SELECTIVE tensor hashing" is imprecise.** Nothing is being
selected: the hash is the default key minus the storage variant kind of five tensors, and nothing
else. The distinction matters for triage — an auditor looking for a cherry-picked field list will not
find one here, and the interesting question about this op is the opposite one (see the
recommendations).

## Omitted parameters

### 1. `storage` variant kind of all five tensors

**Verdict: VALID — pinned by validation.**

This is the only field the custom hash drops relative to the default. Both required inputs are
forced to device storage:

```26:31:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/per_token_cast_back/device/per_token_cast_back_device_operation.cpp
void validate_tensor_specs(const Tensor& tensor, const std::string& name) {
    TT_FATAL(tensor.storage_type() == ttnn::StorageType::DEVICE, "{} must be on device", name);
    TT_FATAL(tensor.buffer() != nullptr, "{} must have a buffer", name);
    TT_FATAL(tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR, "{} must be ROW_MAJOR", name);
    TT_FATAL(is_dram_interleaved(tensor.memory_config()), "{} must be DRAM interleaved", name);
}
```

and the three optional metadata tensors go through `validate_index_tensor`, which calls the same
helper before adding its own dtype and rank constraints
(`device/per_token_cast_back_device_operation.cpp:33-41`, invoked at lines 81-83). `DeviceStorage`
and `HostStorage` both have empty attribute tuples, so only the variant kind ever contributes to the
default hash, and validation pins it to a single value across every admissible call.

As with the rest of this family, the argument holds on the hit path because the hit validator is the
miss validator:

```211:214:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/per_token_cast_back/device/per_token_cast_back_device_operation.cpp
void PerTokenCastBackDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(attrs, tensor_args);
}
```

Without that, a validation-based verdict would be worthless here: the whole risk in this bug class is
the second call inheriting the first call's program, and a miss-only check never runs on the second
call.

### 2. Buffer addresses (omitted by the default hash too)

**Verdict: VALID — patched, and required.**

Two on the plain path, six on the token-count-aware path (`input_e4m3`, `input_scale`, output, and
the three metadata tensors, the latter three appearing in *both* the reader and the writer arg
vectors). All are re-applied on every hit
(`device/per_token_cast_back_program_factory.cpp:447-472`, quoted above). I checked each create-path
`SetRuntimeArgs` call against the override slot-by-slot; no address slot is missed on either branch.

### 3. The per-expert token counts and region offsets (the values inside the metadata tensors)

**Verdict: VALID — invariant.**

This is the moving index for this op — how many rows of the dispatch buffer are actually valid
changes with every routing decision. It is deliberately kept off the host dispatch path: the host
lights up the whole grid and commits no per-core work, and each core derives its own slice on device
from the counts tensor.

```144:148:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/per_token_cast_back/device/per_token_cast_back_program_factory.cpp
    if (token_count_aware) {
        // Light up the whole grid; the host commits no per-core work here. Each core decides at runtime,
        // while the kernels run, how much of the work it takes on (from the device-side token counts).
        num_cores = compute_grid.x * compute_grid.y;
        all_cores = CoreRangeSet{CoreRange{{0, 0}, {compute_grid.x - 1, compute_grid.y - 1}}};
```

The reader reads the counts vector into L1, derives the valid prefix length from it, splits that
across cores itself, and publishes the resulting `num_blocks` to the compute kernel through a
circular buffer rather than a runtime arg:

```141:147:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/per_token_cast_back/device/kernels/dataflow/reader_per_token_cast_back.cpp
    // Split the work across the cores: this core takes a contiguous slice of the flattened compute-block space.
    const uint32_t total_tile_rows = total_valid_rows / tile_h;
    const uint32_t total_compute_blocks = total_tile_rows * blocks_per_row;
    const uint32_t cb_start = (total_compute_blocks * core_id) / num_cores;
    const uint32_t cb_end = (total_compute_blocks * (core_id + 1)) / num_cores;
    const uint32_t num_blocks = cb_end - cb_start;
    // Flattened scale-block indices (tile_h scale-blocks per compute-block). One-time div/mod only.
```

```155:158:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/per_token_cast_back/device/kernels/dataflow/reader_per_token_cast_back.cpp
    // Publish num_blocks to the compute kernel (read via read_tile_value on the TRISCs).
    cb_loop_count_obj.reserve_back(1);
    reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_loop_count_obj.get_write_ptr())[0] = num_blocks;
    cb_loop_count_obj.push_back(1);
```

Because the value never becomes a host-side scalar, there is no stale slot to patch and nothing to
hash. The default hash would not have covered it either — the contents of a device tensor are not
reflected. One cached program serves every routing distribution, which is the point.

What *is* structural about this path is hashed: `experts_per_chip` reaches both the reader and the
writer as a compile-time arg (`device/per_token_cast_back_program_factory.cpp:283` and `316`), and
`scales_from_metadata` selects `scales_start_offset`, itself a reader compile-time arg
(`device/per_token_cast_back_program_factory.cpp:116-117` and `284`). Both are public members of
`PerTokenCastBackParams` and so are covered because `attrs` is hashed whole. So is
`num_routed_experts`, which is read from `expert_region_offsets->logical_shape()[-1]`
(`device/per_token_cast_back_program_factory.cpp:107-108`) — hashed via that tensor's `TensorSpec`.

### 4. Non-address runtime args, compile-time args, and the core grid (set only in `create`)

**Verdict: VALID — invariant** (every value below is a function of the hashed set).

Not omitted parameters, but the slots an incomplete override would strand, so they need proof:

- `H` (both branches) and `M` (a compile-time arg on the token-count-aware branch,
  `device/per_token_cast_back_program_factory.cpp:286, 318`) come from `fold_M_H` over
  `input_e4m3.logical_shape()` (`device/per_token_cast_back_program_factory.cpp:75-76`), which is
  inside the hashed `TensorSpec`.
- `num_blocks` / `row_offset` / `rows_for_core` on the plain branch come from
  `split_work_to_cores(compute_grid, M)` plus `scale_blocks_per_row = H / BLOCK_W` and `tile_h`
  (`device/per_token_cast_back_program_factory.cpp:155-156, 403-410`). `compute_with_storage_grid_size()`
  is a device property and the program cache is per-device; `split_work_to_cores` is deterministic
  in `(grid, M)`. `tile_h` / `tile_w` / `face_h` / `face_w` come from
  `input_e4m3.tensor_spec().tile()` (`device/per_token_cast_back_program_factory.cpp:79-84`), inside
  the hashed spec.
- `num_cores` and the core index `i` follow from the grid, which is device-fixed. This matters
  doubly because the core `CoreRangeSet` is baked into the cached `Program` and refreshed by
  nothing.
- `scales_start_offset` derives from `input_scale.logical_shape()[-1]`, `H` and
  `attrs.scales_from_metadata` (`device/per_token_cast_back_program_factory.cpp:114-117`) — all
  hashed.
- The CB formats and sizes: `TILE_BYTES` from `tile_h`/`tile_w`; `output_df` from
  `attrs.output_dtype`; `compute_df` / `compute_tile_bytes` / `MathFidelity` /
  `unpack_to_dest_mode` from `attrs.narrow_scales_to_bf16`
  (`device/per_token_cast_back_program_factory.cpp:120-123, 160-162, 352-360`) — all hashed.
- The alignment-derived page sizes — `scale_aligned_page_bytes` from `input_scale`
  (`device/per_token_cast_back_program_factory.cpp:99`) and the three metadata
  `aligned_page_size` values (lines 229-231) — are covered here where the two sibling ops leave a
  gap. `Buffer::aligned_page_size()` is `align(page_size(), allocator_alignment(buffer_type))`
  (`tt_metal/impl/buffers/buffer.cpp:656-658`), and `page_size()` derives from the padded shape,
  which is a function of `logical_shape` **and** `TensorLayout::Alignment` — both of which are inside
  the hashed `TensorSpec`. The same applies to every `TensorAccessorArgs(...)` compile-time pair
  (`device/per_token_cast_back_program_factory.cpp:288-294, 320-324`), which for a non-sharded
  buffer emits the `IsDram` config bitset and that aligned page size
  (`tt_metal/impl/buffers/tensor_accessor_args.cpp:194-205`).

That last point is the substantive difference between this op and its siblings, and it is a direct
consequence of hashing the whole `TensorSpec` instead of cherry-picking. The sibling
`pack_scaled_fp8_kv_cache` and `per_token_cast_to_fp8` both hash `logical_shape` while consuming an
`aligned_page_size` computed from the padded shape, which leaves a (narrow, unreached) alignment
gap. This op has no such gap.

### 5. Tile and face geometry (`page_config`) — the tile-awareness check

**Verdict: VALID — invariant.** Nothing is omitted here: `page_config`, and with it the `Tile`,
sits inside every hashed `TensorSpec`. This subsection states that explicitly rather than leaving it
implicit, because this op *is* genuinely tile-aware and a reader auditing it needs the reassurance.

This op reads the tensor's real tile spec instead of assuming a 32x32 tile, which is the correct
idiom but creates the mirror obligation: because the generated program provably varies with the tile
and face geometry, the geometry must be reachable from the cache key.

What the geometry feeds. It is read once, from `input_e4m3`:

```79:84:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/per_token_cast_back/device/per_token_cast_back_program_factory.cpp
    const auto tile_shape = input_e4m3.tensor_spec().tile().get_tile_shape();
    const auto face_shape = input_e4m3.tensor_spec().tile().get_face_shape();
    const uint32_t tile_h = tile_shape[0];
    const uint32_t tile_w = tile_shape[1];
    const uint32_t face_h = face_shape[0];
    const uint32_t face_w = face_shape[1];
```

and from there into every category of state that a cache hit freezes:

- **CB page sizes and totals.** `block_wt = block_w / tile_w` gives `tiles_per_block`
  (`device/per_token_cast_back_program_factory.cpp:90-92`); `input_e4m3_tile_bytes` and
  `out_tile_bytes` are `tile_h * tile_w` scaled by the element size (`97-98`); `TILE_BYTES` and
  `compute_tile_bytes` likewise (`120, 162`). Those set `cb_input_e4m3` (`195-198`), the compute and
  fp32 tile CBs (`201-208`), `cb_out` (`213-216`) and the reader scratch
  `scale_scratch_bytes = tile_h * scale_aligned_page_bytes` (`219-223`).
- **Compile-time args.** All four go to the reader
  (`device/per_token_cast_back_program_factory.cpp:264-274`), `tile_h` to the writer (`308`), and
  `tile_h`/`tile_w` to compute (`339-349`). The reader derives `tiles_per_block`, `face_elems`,
  `faces_per_row` and `FACE_ROWS` from them as `constexpr`
  (`device/kernels/dataflow/reader_per_token_cast_back.cpp:76-79`).
- **Per-core work.** On the plain path the per-core `num_blocks` is
  `div_up(rows_for_core * scale_blocks_per_row, tile_h)`
  (`device/per_token_cast_back_program_factory.cpp:403-413`).

Where it reaches the key. Through the hashed `TensorSpec` of each tensor, without any cherry-picking
(`device/per_token_cast_back_device_operation.cpp:229-244`, quoted in full above). The chain is
`TensorSpec` -> `tensor_layout` -> `page_config`:

```96:97:tt_metal/api/tt-metalium/experimental/tensor/spec/tensor_spec.hpp
    static constexpr auto attribute_names = std::forward_as_tuple("logical_shape", "tensor_layout");
    auto attribute_values() const { return std::forward_as_tuple(logical_shape_, tensor_layout_); }
```

```75:76:tt_metal/api/tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp
    static constexpr auto attribute_names = std::forward_as_tuple("dtype", "page_config", "memory_config", "alignment");
    std::tuple<const DataType&, const PageConfig&, const MemoryConfig&, const Alignment&> attribute_values() const;
```

```50:51:tt_metal/api/tt-metalium/experimental/tensor/spec/layout/page_config.hpp
    static constexpr auto attribute_names = std::forward_as_tuple("config");
    auto attribute_values() const { return std::forward_as_tuple(config_); }
```

So two calls with the same padded shape, dtype and memory config but a different `Tile` would land
on different cache entries. The mirror-image bug does not apply, and — as with the alignment
question in item 4 — this op gets the right answer purely as a consequence of hashing whole
`TensorSpec`s rather than a field list.

One precise limit on that coverage, which applies tree-wide and not only here: hashing `page_config`
buys the tile *shape*, not the whole `Tile`. `Tile::attribute_values()` exposes only `tile_shape`,
`face_shape` and `num_faces` (`tt_metal/api/tt-metalium/tile.hpp:46-47`) and `Tile::operator==`
compares only the first two (`tt_metal/impl/data_format/tile.cpp:122-124`), so `transpose_within_face`
and `transpose_of_faces` reach neither the reflection hash nor the canonical collision key. They are
moot for this op for the reason given immediately below — every input is pinned `ROW_MAJOR`, and a
`RowMajorPageConfig` carries no `Tile` at all — but the coverage claim above should not be read as
covering them.

A second, independent reason the geometry cannot vary here is worth recording, because it explains
why the factory's tile-awareness has never been exercised. Every input is required to be ROW_MAJOR:

```29:29:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/per_token_cast_back/device/per_token_cast_back_device_operation.cpp
    TT_FATAL(tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR, "{} must be ROW_MAJOR", name);
```

applied to `input_e4m3` and `input_scale` directly and to the three metadata tensors via
`validate_index_tensor` (`device/per_token_cast_back_device_operation.cpp:33-34, 57-58, 81-83`). A
`PageConfig` only carries a `Tile` on its TILE alternative:

```19:27:tt_metal/api/tt-metalium/experimental/tensor/spec/layout/page_config.hpp
struct RowMajorPageConfig {
    bool operator==(const RowMajorPageConfig&) const = default;
};

struct TilePageConfig {
    Tile tile;

    bool operator==(const TilePageConfig&) const = default;
};
```

and `get_tile()` on a row-major config returns a default-constructed 32x32 tile:

```179:184:tt_metal/impl/tensor/spec/layout/page_config.cpp
Tile PageConfig::get_tile() const {
    if (const auto* tile_config = std::get_if<TilePageConfig>(&config_)) {
        return tile_config->tile;
    }
    return Tile{};
}
```

`TensorSpec::tile()` is that call (`tt_metal/api/tt-metalium/experimental/tensor/spec/tensor_spec.hpp:39`
-> `tt_metal/impl/tensor/spec/layout/tensor_layout_impl.hpp:29`), so `tile_h`, `tile_w`, `face_h` and
`face_w` in this factory are today always 32, 32, 16, 16. The `tile_h * tile_w == 1024` `TT_FATAL`
at `device/per_token_cast_back_device_operation.cpp:136-145` is consequently trivially satisfied —
but it is worth keeping as a tripwire, since it is what would fire first if the op were ever relaxed
to accept TILE inputs whose tile the kernels cannot handle. The reassuring conclusion is that this op
is safe on both fronts at once: the geometry is constant *and* it is in the key, so relaxing the
ROW_MAJOR requirement later would not open a cache hole.

## Keys the custom hash adds beyond the default

None. The custom key is the default key with the five storage variant kinds removed. There is no
field here that the default would not have covered.

## Framework side effect of having a custom hash

```1012:1014:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
        if constexpr (requires { DeviceOperation::compute_program_hash(attrs, tensor_args); }) {
            return key;  // custom hash -> opt out beyond the op-identity prefix
        } else {
```

Defining `compute_program_hash` degrades `ProgramCacheKey::canonical` to just the op type name, so a
64-bit hash collision between two different `per_token_cast_back` configurations resolves to a wrong
hit instead of a rebuild.

This op is the clearest case in the family where that trade is a net loss. The custom hash removes
exactly one thing from the key — a storage variant that validation already pins to a single value —
and in exchange gives up attribute-level collision resolution for a six-field attribute struct and
five tensors. See recommendation 1.

## Summary

| Omitted param | Used by program? | Patched on hit? | Verdict |
|---|---|---|---|
| `storage` kind of all five tensors | n/a | n/a | VALID — pinned by validation |
| Buffer addresses (2 plain / 6 token-count-aware) | Yes | Yes (override, all slots verified) | VALID — patched |
| Per-expert token counts / region offset values | Yes, read on device | n/a — never enters an arg | VALID — invariant |
| Non-address rt args, compile-time args, core grid (create-only) | Yes | No | VALID — invariant (functions of the hashed set) |
| `page_config` tile/face geometry | Yes (CB pages, CT args, per-core work) | No | VALID — invariant (inside the hashed `TensorSpec`) |

**No program-cache correctness bug found, and no caveat either.** This op's hash is the strongest of
the four `deepseek_prefill` ops audited here: every field that reaches a frozen slot — including the
alignment-derived page sizes that two of its siblings leave uncovered, and the tile and face geometry
that this factory genuinely consumes — is inside the hashed `TensorSpec`s or the whole-`attrs` hash.
The override was checked slot-by-slot against both create branches and re-applies every address. The
one omitted field is pinned by a validator that runs on cache hits as well as misses.

The tile-geometry check (item 5) is clean in both of its forms. The factory never uses
`tt::tile_size(...)` or a bare `TILE_HW` / `TILE_WIDTH` / `TILE_HEIGHT` tile-count conversion — it
reads `tensor_spec().tile()` throughout — so there is no hardcoded-32x32 defect; and because the
hash carries whole `TensorSpec`s, the `Tile` the factory reads is in the key, so there is no
tile-aware-but-unhashed defect either.

The family-level observation worth carrying out of this document: all four `deepseek_prefill` ops
audited here define `validate_on_program_cache_hit`, and all four make it a total delegation back to
the miss validator (or, in `moe_padding_config`'s case, to the same shared checker the miss validator
calls). That delegation is what makes "VALID — pinned by validation" a sound verdict rather than
wishful thinking, and it is worth more than it looks: defining a hit validator *replaces* the miss
validator on hits, so a narrower one would silently disable the dropped checks on exactly the calls
that reuse a cached program. None of the four has that defect, but the idiom deserves a comment at
each site, because the safety depends on the delegation staying total as the validators grow. They
do **not** share a hashing idiom — this op hashes whole `TensorSpec`s, `pack_scaled_fp8_kv_cache`
and `per_token_cast_to_fp8` cherry-pick fields (and both consequently carry an alignment caveat),
and `moe_padding_config` hashes `padded_shape` explicitly to close the same hole a third way. This
op is the pattern the other two should converge on.

## Recommendations

1. **Consider deleting `compute_program_hash` entirely.** The hash's only effect is to drop five
   storage variant kinds that `validate_tensor_specs` already pins to `StorageType::DEVICE`, so it
   buys no cache reuse whatsoever. In exchange it opts the op out of `canonical`-key collision
   resolution (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:1006-1022`), turning a 64-bit
   collision between two configurations into a wrong hit rather than a rebuild. Falling back to the
   default reflection hash would be strictly safer at zero cost. If the custom hash is kept for
   documentation value, the comment at
   `device/per_token_cast_back_device_operation.cpp:231-233` should say so — it currently explains
   why the hash is complete without noting that a complete custom hash is exactly the case where the
   default is preferable.
2. Whether or not (1) is taken, keep the "hash the whole `TensorSpec`" idiom and propagate it to
   `pack_scaled_fp8_kv_cache` and `per_token_cast_to_fp8`, which cherry-pick fields and thereby
   leave `TensorLayout::Alignment` out of the key while consuming an `aligned_page_size` derived
   from it. That is a family-wide gap and deserves a family-wide fix.
3. Add a program-cache regression test on the token-count-aware path specifically: two calls with
   the same shapes and attributes but different per-expert token counts, asserting one cache entry
   and two correct results. That is the property item 3 rests on (the counts are device data, not
   program structure), and it is the property most likely to be broken by a future optimization that
   moves the counts host-side for a better work split.
