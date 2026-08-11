# Program Cache Audit — `data_movement/sharded/interleaved_to_sharded`

Audit of `ttnn::prim::InterleavedToShardedDeviceOperation::compute_program_hash` against the
framework default ("hash everything") key.

| | |
|---|---|
| Device operation | `ttnn::prim::InterleavedToShardedDeviceOperation` (`device/interleaved_to_sharded_op.hpp`) |
| Custom hash | `device/interleaved_to_sharded_op.cpp:141` |
| `operation_attributes_t` | `InterleavedToShardedParams` — `output_mem_config`, `output_dtype`, `keep_l1_aligned` |
| `tensor_args_t` | `InterleavedToShardedInputs` — `input_tensor`, `output_tensor` (optional preallocated) |
| Program factories | `InterleavedToShardedProgramFactory` (single-alternative variant, `ProgramDescriptor`-based) |
| `override_runtime_arguments` | **No** |
| `get_dynamic_runtime_args` | **No** |
| Own cache-hit validator | **No** — but see below; the framework substitutes the miss validator |
| Cache-hit patch mechanism | **Framework buffer-binding fast path** (mode B) |

## Cache-hit patch mechanism

With neither `override_runtime_arguments` nor `get_dynamic_runtime_args`, the adapter falls
through to the simple per-coord branch, where the choice between the buffer-binding fast path and
the slow-path rebuild hinges on whether the factory declared any `Buffer*` runtime-arg binding:

```726:731:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
                    if (!sv.resolved_bindings.rt_args.empty() ||
                        (!dynamic_args.empty() && !sv.resolved_bindings.empty())) {
                        auto collected =
                            collect_tensor_buffers(tensor_args, tensor_return_value, sv.workload_descriptor);
                        tt::tt_metal::apply_resolved_bindings(program, sv.resolved_bindings, collected.buffers);
                        tt::tt_metal::apply_dynamic_runtime_args(program, dynamic_args);
```

This factory does declare one: the source buffer is pushed as arg 0 of the reader on every core,
via `RTArgList::push_back(Buffer*)`, which auto-registers a `BufferBinding`
(`tt_metal/api/tt-metalium/program_descriptors.hpp:170-188`):

```288:298:ttnn/cpp/ttnn/operations/data_movement/sharded/interleaved_to_sharded/device/interleaved_to_sharded_program_factory.cpp
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

So `resolved_bindings.rt_args` is non-empty and **the fast path is always taken**. There is no
slow-path rebuild for this op.

**Consequence for this audit:** on a cache hit, *only* buffer addresses are re-patched — the
`src_buffer`/`dst_buffer` runtime-arg bindings and the output CB's `.buffer` binding. `dynamic_args`
is empty, so every other runtime arg (`shard_height`, `shard_width`, `padded_offset`,
`num_units_offset`, `curr_num_units_per_shard`, `curr_idx_h + curr_idx_w`, `starting_idx_h`,
`aligned*`, `start_id`, `output_width_in_pages`) is frozen at the value computed on the first
miss, as are all compile-time args, CB sizes/formats and core ranges. Every one of those must be
a pure function of the hashed set.

## Which validator runs on a cache hit

This op defines **no** `validate_on_program_cache_hit` — `interleaved_to_sharded_op.hpp:26`
declares only `validate_on_program_cache_miss`. That places it in the favourable branch of the
dispatcher, which substitutes the miss validator on every hit:

```262:266:ttnn/api/ttnn/device_operation.hpp
    if constexpr (HasValidateOnProgramCacheHit<mesh_device_operation_t>) {
        mesh_device_operation_t::validate_on_program_cache_hit(operation_attributes, tensor_args);
    } else {
        mesh_device_operation_t::validate_on_program_cache_miss(operation_attributes, tensor_args);
    }
```

The mesh adapter mirrors the same rule for the adapted type:

```228:234:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
    static void validate_on_program_cache_hit(const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
        if constexpr (HasValidateOnProgramCacheHit<DeviceOperation>) {
            DeviceOperation::validate_on_program_cache_hit(attrs, tensor_args);
        } else {
            DeviceOperation::validate_on_program_cache_miss(attrs, tensor_args);
        }
    }
```

`validate_on_program_cache_miss` forwards to `validate_inputs`
(`interleaved_to_sharded_op.cpp:109-113`), so every check in it — the 32x32 tile guard, the
INTERLEAVED-input requirement, and the full preallocated-output comparison — is a live constraint
on cache hits, not just on misses. Verdicts #2, #4 and #5 below are "VALID — pinned by validation"
precisely because of this substitution branch; had the op defined even a narrow hit validator, all
three would degrade to "pinned only on the miss path". The CSV's `own_hit_validator = N` reads like
a weakness and is in fact the opposite.

## Baseline: what the default hash would cover

| Source | Fields |
|---|---|
| `operation_attributes` | `output_mem_config`, `output_dtype`, `keep_l1_aligned` |
| `input_tensor.storage` | storage variant kind |
| `input_tensor.tensor_spec` | `logical_shape`, and `tensor_layout` = { `dtype`, `page_config`, `memory_config`, `alignment` } |
| `output_tensor` (optional) | engaged flag, plus the same six fields when engaged |

`padded_shape` is *not* in the default key — it is a cached derivation of `logical_shape` +
`page_config` + `alignment`.

## What the custom hash covers

```141:152:ttnn/cpp/ttnn/operations/data_movement/sharded/interleaved_to_sharded/device/interleaved_to_sharded_op.cpp
ttsl::hash::hash_t InterleavedToShardedDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    return tt::tt_metal::operation::hash_operation<InterleavedToShardedDeviceOperation>(
        operation_attributes.output_mem_config,
        operation_attributes.output_dtype,
        operation_attributes.keep_l1_aligned,
        input_tensor.dtype(),
        input_tensor.memory_config(),
        input_tensor.layout(),
        input_tensor.padded_shape());
}
```

All three operation attributes are kept. The input tensor is decomposed selectively —
`logical_shape` is swapped for `padded_shape`, `page_config` is narrowed to `layout()`, and
`alignment` and the storage kind are dropped. The optional `output_tensor` is dropped entirely.

## Omitted parameters

### 1. `input_tensor.logical_shape()` — replaced by `padded_shape()`

**Verdict: BUG** on the row-major path (a relaxation win on the tile path).

On the **tile** path this is exactly the trade the exemplar reshape audit describes, and it is
correct. The factory reads only padded/physical quantities:

```106:118:ttnn/cpp/ttnn/operations/data_movement/sharded/interleaved_to_sharded/device/interleaved_to_sharded_program_factory.cpp
        num_units_per_shard_height = shard_spec.shape[0] / TILE_HEIGHT;
        num_units_per_shard_width = shard_spec.shape[1] / TILE_WIDTH;
        num_units_per_shard = num_units_per_shard_height * num_units_per_shard_width;
        num_units_per_row = input.padded_shape()[-1] / TILE_WIDTH;
        num_units_offset = num_units_per_row;
        uint32_t num_units_height = (input.physical_volume() / input.padded_shape()[-1]) / TILE_HEIGHT;
        num_units_per_shard_height_last =
            num_units_per_shard_height -
            (tt::round_up(num_units_height, num_units_per_shard_height) - num_units_height);
        num_units_per_shard_width_last =
            num_units_per_shard_width -
            (tt::round_up(num_units_per_row, num_units_per_shard_width) - num_units_per_row);
        padded_offset_bytes = (num_units_per_shard_width - num_units_per_shard_width_last) * input_unit_size;
```

Sub-tile logical shapes that pad to the same tile grid legitimately share one program: input
logical `[1,1,33,64]` and `[1,1,64,64]` both pad to `[1,1,64,64]`, and their outputs — built by
`compute_output_specs` from the input logical shape and `output_mem_config` — also pad to the same
padded shape, so even `get_optimal_worker_cores_for_sharded_tensor(output)` (which walks the
output buffer's distribution spec, `ttnn/core/tensor/tensor_utils.cpp:82-83`) returns the same core
list. The default hash would have forced a needless recompile.

On the **row-major** path the factory reads the logical shape directly, twice:

```119:134:ttnn/cpp/ttnn/operations/data_movement/sharded/interleaved_to_sharded/device/interleaved_to_sharded_program_factory.cpp
    } else {
        input_unit_size = static_cast<uint32_t>(shard_spec.shape[1] * input.element_size());
        output_unit_size = static_cast<uint32_t>(shard_spec.shape[1] * output.element_size());
        num_units_per_shard_height = shard_spec.shape[0];
        num_units_per_shard_width = 1;
        num_units_per_shard = num_units_per_shard_height * num_units_per_shard_width;
        num_units_per_row = static_cast<uint32_t>(input.logical_shape()[-1] * input.element_size());
        num_units_offset = 1;
        uint32_t num_units_height = static_cast<uint32_t>(input.logical_volume() / input.logical_shape()[-1]);
        num_units_per_shard_height_last =
            num_units_per_shard_height -
            (tt::round_up(num_units_height, num_units_per_shard_height) - num_units_height);
        // TODO: Use a different variable name. Units refers to pages, but this is being used as size
        num_units_per_shard_width_last =
            input_unit_size - (tt::round_up(num_units_per_row, input_unit_size) - num_units_per_row);
        // Adjust accordingly to l1 alignment, do it for all archs
```

`num_units_per_row` becomes reader arg 1 and drives the `curr_idx_h`/`curr_idx_w` walk
(`interleaved_to_sharded_program_factory.cpp:388, 417-421`); `num_units_height` sets
`num_units_per_shard_height_last`, the last core's `shard_height`. None of those are patched on a
hit.

The op relies on a property of `TensorLayout` rather than on anything it checks. Validation forces
the input to be `INTERLEAVED` (`interleaved_to_sharded_op.cpp:71-73`), and the *default* alignment
for an interleaved row-major tensor is a single `1`:

```47:57:tt_metal/impl/tensor/spec/layout/page_config.cpp
Alignment create_default_alignment_rm(const RowMajorPageConfig&, DataType, const MemoryConfig& memory_config) {
    if (memory_config.shard_spec().has_value()) {
        const auto& shard_spec = memory_config.shard_spec().value();
        return Alignment({shard_spec.shape[1]});
    }
    if (memory_config.nd_shard_spec().has_value()) {
        const auto& nd_shard_spec = *memory_config.nd_shard_spec();
        return Alignment({nd_shard_spec.shard_shape[-1]});
    }
    return Alignment({1});
}
```

and with `Alignment{1}` the padding loop is the identity, so `padded_shape == logical_shape` and
hashing the former pins the latter.

That holds only for the *default* alignment, and a non-default one is constructible. The width
alignment is a public constructor parameter:

```34:38:tt_metal/api/tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp
    TensorLayout(
        DataType dtype,
        const PageConfig& page_config,
        const MemoryConfig& memory_config,
        const Alignment& alignment = {});
```

and for an interleaved row-major tensor nothing rejects a non-default value. `validate_alignment`
runs on every `TensorLayout` construction (`tensor_layout.cpp:162-163`) and dispatches to the
row-major arm, whose only unconditional requirement is non-emptiness — the width check is gated on
the tensor being sharded, which an interleaved input by definition is not:

```77:94:tt_metal/impl/tensor/spec/layout/page_config.cpp
void validate_alignment_rm(
    const RowMajorPageConfig&, const Alignment& alignment, DataType, const MemoryConfig& memory_config) {
    TT_FATAL(!alignment.empty(), "Alignment must contain at least one dimension for Row Major layout.");
    const uint32_t width_alignment = alignment[-1];

    // TODO: Do we need to validate sharded width here if we are guaranteed that physical_shard_width is set as
    // width_alignment
    if (memory_config.shard_spec().has_value() && memory_config.memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED) {
        const auto& physical_shard_shape = memory_config.shard_spec().value().shape;
        const auto physical_shard_width = physical_shard_shape[1];
        TT_FATAL(
            physical_shard_width % width_alignment == 0,
            "Alignment mismatch for sharded tensor: Expected physical shard shape {} to be aligned to {} along the "
            "width for Row Major layout.",
            physical_shard_width,
            width_alignment);
    }
}
```

The padded shape then rounds the last dimension up to that alignment:

```402:404:tt_metal/impl/tensor/spec/layout/tensor_layout.cpp
        // The last 2 dimensions of a shape are special
        if (rank_index >= static_cast<int>(shape.rank()) - 2) {
            padded_shape_value = CMAKE_UNIQUE_NAMESPACE::round_up(shape_value, alignment_value);
```

**Reproduction.** Both calls use `Layout::ROW_MAJOR`, `bfloat16`, a DRAM-interleaved input, and an
identical `output_mem_config` of HEIGHT_SHARDED L1 with shard shape `[32, 64]` — 128 bytes wide,
which satisfies the L1-alignment check at `interleaved_to_sharded_op.cpp:83-88`.

- **Call 1**: input logical `[1, 1, 32, 64]` with the default `Alignment{1}`, so
  `padded_shape == [1, 1, 32, 64]`. The factory computes
  `num_units_per_row = 64 * 2 = 128` bytes.
- **Call 2**: input logical `[1, 1, 32, 60]` constructed with an explicit `Alignment{64}`, so
  `padded_shape` is also `[1, 1, 32, 64]`. The correct
  `num_units_per_row` is `60 * 2 = 120` bytes.

Every hashed term matches: `output_mem_config`, `output_dtype`, `keep_l1_aligned`, input `dtype`,
input `memory_config` (DRAM interleaved, no shard spec, so identical), input `layout`, and input
`padded_shape`. The differing `logical_shape` and `Alignment` are both absent from the key, so call
2 hits call 1's program.

`num_units_per_row` is reader arg 1 — a raw `uint32_t`, not a buffer binding — so in mode B it is
frozen at 128:

```385:397:ttnn/cpp/ttnn/operations/data_movement/sharded/interleaved_to_sharded/device/interleaved_to_sharded_program_factory.cpp
            // Reader run-time args: arg 0 is the source-buffer base address (binding).
            KernelDescriptor::RTArgList reader_rt;
            reader_rt.push_back(src_buffer);
            reader_rt.push_back(num_units_per_row);
            reader_rt.push_back(shard_height);
            reader_rt.push_back(shard_width);
            reader_rt.push_back(padded_offset_bytes);
            reader_rt.push_back(static_cast<uint32_t>(aligned));
            reader_rt.push_back(aligned_width_offset);
            reader_rt.push_back(aligned_shard_width);
            reader_rt.push_back(aligned_offset);
            reader_rt.push_back(curr_idx_h);
            reader_desc.emplace_runtime_args(core, reader_rt);
```

Call 2's reader therefore treats each row as 128 bytes of payload when only the first 120 are real
data, pulling 8 bytes of inter-row padding into every shard row and shifting the tensor
progressively. `num_units_per_shard_width_last` (line 132-133 above) diverges for the same reason —
128 versus 120 — and lands in reader arg 3. The result is silently wrong data, with no cache miss
and no assertion.

The severity is bounded by how rarely tensors are built with explicit alignments: the ordinary
`ttnn` construction paths take the default. But that is a property of current call sites, not an
enforced constraint, so it does not change the verdict.

The fix: hash `input_tensor.logical_shape()` in addition to `padded_shape()`. It costs nothing on
the tile path, where the two are already jointly determined by the hashed `padded_shape` and the
validated 32x32 tile. A `TT_FATAL(input.logical_shape() == input.padded_shape())` on the row-major
branch would also close it, at the price of rejecting tensors the factory could otherwise handle
correctly if it read the padded shape consistently.

### 2. `input_tensor.tensor_spec().page_config()` — only `layout()` is hashed

**Verdict: VALID — pinned by validation.**

`layout()` discards the `Tile` shape, but unlike the reshape exemplar this op does not merely
*assume* 32x32 — it rejects anything else, on hits as well as misses:

```94:98:ttnn/cpp/ttnn/operations/data_movement/sharded/interleaved_to_sharded/device/interleaved_to_sharded_op.cpp
    if (input_tensor.layout() == Layout::TILE) {
        auto tile = input_tensor.tensor_spec().tile();
        if (tile.get_height() != tt::constants::TILE_HEIGHT || tile.get_width() != tt::constants::TILE_WIDTH) {
            return {false, fmt::format("interleaved_to_sharded requires standard 32x32 tiles, got {}x{}", tile.get_height(), tile.get_width())};
        }
```

With the tile pinned to 32x32 and the layout hashed, `PageConfig` carries no residual
information, and the factory's use of `TILE_HEIGHT`/`TILE_WIDTH` and `tt::tile_size(format)`
is consistent with it.

### 3. `input_tensor.tensor_layout().get_alignment()`

**Verdict: BUG on the row-major path — the same defect as #1, not a second one.**

For the tile path, `Alignment` is pinned by omission #2: the default is
`{tile.height, tile.width}` = `{32, 32}`, and an explicit alignment must be a multiple of the tile
dims (`validate_alignment_tile`, `tt_metal/impl/tensor/spec/layout/page_config.cpp:59-75`), so
anything non-default shows up in the hashed `padded_shape`.

For the row-major path it is the mechanism behind #1 rather than an independent gap. `Alignment` is
the *only* thing that can make `padded_shape != logical_shape` on an interleaved row-major tensor,
so dropping both collapses the pair `(logical_shape, alignment)` onto a single hashed
`padded_shape`. It is listed separately because the two omissions have to be reasoned about
together — hashing `logical_shape` alone is sufficient, since `(logical_shape, padded_shape)`
determines the width alignment for these purposes — but it is one bug, counted once, and closing #1
closes this.

The one alignment-derived quantity the factory reads directly is the buffer alignment:

```142:150:ttnn/cpp/ttnn/operations/data_movement/sharded/interleaved_to_sharded/device/interleaved_to_sharded_program_factory.cpp
    uint32_t input_cb_index = tt::CBIndex::c_0;
    uint32_t scratch_cb_index = tt::CBIndex::c_1;
    uint32_t out_cb_index = input_cb_index;
    uint32_t num_input_units = num_units_per_shard;
    uint32_t output_page_size = tt::align(output_unit_size, dst_buffer->alignment());

    ProgramDescriptor desc;

    if (convert_df) {
```

`Buffer::alignment()` is a function of the buffer type and page size, both determined by the
hashed `memory_config` + `dtype` + `padded_shape`, so the CB page sizes it feeds are safe.

### 4. `input_tensor.storage` variant kind (device vs host)

**Verdict: VALID — pinned by validation.**

```19:24:ttnn/cpp/ttnn/operations/data_movement/sharded/interleaved_to_sharded/device/interleaved_to_sharded_op.cpp
    if (input_tensor.storage_type() != StorageType::DEVICE) {
        return {false, "Operands to shard need to be on device!"};
    }
    if (input_tensor.buffer() == nullptr) {
        return {false, "Operands to shard need to be allocated in buffers on device!"};
    }
```

Constant across every admissible call, so it carries no information.

### 5. `tensor_args.output_tensor` — the entire optional preallocated output

**Verdict: VALID — pinned by validation** (with the residual of #1/#3).

This is the largest structural omission: when engaged, `compute_output_specs` returns the
preallocated tensor's spec verbatim
(`interleaved_to_sharded_op.cpp:117-119`), and the factory then derives its shard spec, core
ranges, CB sizes and every per-core index from that `output`. None of it is hashed. It is
nonetheless safe because `validate_inputs` — which, per the substitution branch quoted under
"Which validator runs on a cache hit", runs on hits too — pins the preallocated output to exactly
the hashed attributes:

```47:70:ttnn/cpp/ttnn/operations/data_movement/sharded/interleaved_to_sharded/device/interleaved_to_sharded_op.cpp
    if (tensor_args.output_tensor.has_value()) {
        const auto& output_tensor = tensor_args.output_tensor.value();
        if (output_tensor.logical_shape() != input_tensor.logical_shape()) {
            return {false, "Mismatched output shape"};
        }
        if (output_tensor.memory_config() != resolved_output_mem_config) {
            return {false, "Mismatched output memory config"};
        }
        if (output_tensor.dtype() != output_dtype) {
            return {false, "Mismatched output dtype"};
        }
        if (output_tensor.storage_type() != StorageType::DEVICE) {
            return {false, "Operands to shard need to be on device!"};
        }
        if (output_tensor.buffer() == nullptr) {
            return {false, "Operands to shard need to be allocated in buffers on device!"};
        }
        if (output_tensor.device() != input_tensor.device()) {
            return {false, "Operands to shard need to be on the same device!"};
        }
        if (output_tensor.layout() != input_tensor.layout()) {
            return {false, "Output tensor layout must match input tensor layout"};
        }
```

plus the tile check at `interleaved_to_sharded_op.cpp:99-104`. That fixes the output's
`logical_shape` (to the input's), `memory_config`, `dtype`, `layout` and `tile` — i.e. its whole
`TensorSpec` except `Alignment`, the same residual as #1/#3. The engaged/disengaged distinction
also carries no program difference, because when disengaged `compute_output_specs` constructs the
spec from precisely those same pinned values:

```121:127:ttnn/cpp/ttnn/operations/data_movement/sharded/interleaved_to_sharded/device/interleaved_to_sharded_op.cpp
    const auto& input_tensor = tensor_args.input_tensor;
    return tt::tt_metal::TensorSpec(
        input_tensor.logical_shape(),
        tt::tt_metal::TensorLayout(
            operation_attributes.output_dtype,
            tt::tt_metal::PageConfig(input_tensor.layout()),
            operation_attributes.output_mem_config));
```

Only the output *address* differs between the two forms, and that is patched (#6).

### 6. Buffer addresses (omitted by both the default hash and this one)

**Verdict: VALID — patched, and required.**

The mode-B fast path patches exactly two classes of slot, and this factory declares both:

- `src_buffer` at reader arg 0 on every core (tile path: line 290; row-major path: line 387),
  and `dst_buffer` at writer arg 0 when the destination is DRAM (lines 304, 405).
- The output CB's `.buffer` binding when the destination is sharded L1:

```164:173:ttnn/cpp/ttnn/operations/data_movement/sharded/interleaved_to_sharded/device/interleaved_to_sharded_program_factory.cpp
    // Output CB. When destination is sharded (non-DRAM) we bind it to the output buffer
    // for dynamic-CB rebinding on cache hits via cb.buffer. When dst is DRAM, no binding.
    push_i2s_cb_pair(
        desc,
        out_cb_index,
        output_cb_data_format,
        num_input_units * output_page_size,
        output_page_size,
        all_cores,
        /*bound_buffer=*/dst_is_dram ? nullptr : dst_buffer);
```

The two cases are complementary — DRAM output goes through the writer runtime arg, L1 output
through the CB — so there is no configuration in which the destination address goes unpatched.
The unbound CBs (the `convert_df` input CB and the alignment scratch CB) are L1-local and carry no
tensor address.

## Keys the custom hash adds beyond the default

- `input_tensor.padded_shape()` — not in the default key (derived there). Adding it is what makes
  dropping `logical_shape` viable on the tile path.

## Framework side effect of having a custom hash

```1012:1014:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
        if constexpr (requires { DeviceOperation::compute_program_hash(attrs, tensor_args); }) {
            return key;  // custom hash -> opt out beyond the op-identity prefix
        } else {
```

`ProgramCacheKey::canonical` degrades to just the op type name, so a 64-bit collision between two
different sharding configurations resolves to a wrong hit rather than a rebuild. Because this op
is mode B — no rebuild path at all — a wrong hit here silently reuses a whole core grid's worth of
frozen work-split arguments.

## Summary

| Omitted vs. default | Used by program? | Patched on hit? | Verdict |
|---|---|---|---|
| `input.logical_shape` | Tile path: no. Row-major path: **yes** (`num_units_per_row`, `num_units_height`) | **No** | **BUG** on the row-major path (relaxation win on the tile path) |
| `input.page_config` (`Tile`) | No (32x32 constants) | n/a | VALID — pinned by validation |
| `input.tensor_layout.alignment` | Only via hashed derivatives on the tile path; the enabler of #1 on the row-major path | No | **BUG** — same defect as #1, counted once |
| `input.storage` kind | n/a | n/a | VALID — pinned by validation |
| `tensor_args.output_tensor` | Yes — shard spec, core ranges, CB sizes, all indices | Address only | VALID — pinned by validation |
| Buffer addresses | Yes | Yes (`resolved_bindings` + CB `.buffer`) | VALID — patched |

**One program-cache correctness bug was found**, on the row-major path only, and it is narrow.
Every compile-time arg, CB size and core range is a function of the hashed set
{`output_mem_config`, `output_dtype`, input `dtype`, input `memory_config`, input `layout`,
input `padded_shape`} plus device-fixed constants (compute grid, arch, HAL alignments) that the
per-device cache already partitions on. That much is sound, and the tile path is sound outright —
its use of `padded_shape` over `logical_shape` is a genuine relaxation win, and its 32x32 tile
guard is the guard other ops in this class are missing.

The row-major branch is the exception: it reads `logical_shape` and `logical_volume` directly for
`num_units_per_row` and `num_units_height`, while the key carries only `padded_shape`. The two
coincide for the default `Alignment{1}`, but an interleaved row-major tensor may carry an explicit
non-default alignment — the `TensorLayout` constructor accepts one and `validate_alignment_rm`
declines to constrain it for unsharded tensors — and nothing in `validate_inputs` rejects such an
input. Two calls whose logical widths differ but pad to the same width therefore share a key, and
the frozen mode-B reader arg makes the second one read the first one's row stride.

The distinction worth drawing against the other findings in this batch is one of *reachability
cost*, not of enforcement: the colliding input is constructible through the public `TensorLayout`
API without tripping any `TT_FATAL`, but it takes a deliberately-alignment-annotated tensor to
produce, which no ordinary `ttnn` construction path emits. That bounds the severity well below the
shape omissions found in `roll` and `interleaved_to_sharded_partial`, where the colliding inputs
are the everyday case.

The CSV row for this op is accurate on `hash_kind`, `tensor_input` and the two hook columns.
`own_hit_validator = N` is literally true but misleading: the framework runs
`validate_on_program_cache_miss` on hits, so this op does have effective hit-time validation, and
verdicts #2, #4 and #5 depend on it.

## Recommendations

1. Add `input_tensor.logical_shape()` to `compute_program_hash`. This is the fix for the BUG and it
   closes omissions #1 and #3 together. It costs nothing in the common case, because for every
   tensor with a default alignment the logical shape is already determined by the hashed padded
   shape, so no call that hits today would start missing. The alternative —
   `TT_FATAL(input.logical_shape() == input.padded_shape())` on the row-major branch of
   `validate_inputs` — also closes it, but by rejecting inputs the factory would otherwise handle.
2. `keep_l1_aligned` is hashed but the factory ignores it — it is hardcoded:

```64:65:ttnn/cpp/ttnn/operations/data_movement/sharded/interleaved_to_sharded/device/interleaved_to_sharded_program_factory.cpp
    // Keep explicit bool init to match legacy behavior which forced it true
    bool keep_l1_aligned = true;  // operation_attributes.keep_l1_aligned;
```

   Hashing a value the program does not depend on is harmless for correctness but splits the cache
   for no reason. Either honour the attribute or drop it from both the params struct and the hash.
3. Run this op's tests once under `-DTT_DESCRIPTOR_PATCHING_PARITY_CHECK`. The
   `assert_fastpath_parity` oracle is already wired into the mode-B branch
   (`mesh_device_operation_adapter.hpp:732-747`) and is the cheapest way to keep the
   "every non-address runtime arg is hash-determined" invariant from regressing as this factory
   grows.
4. If this op ever gains a slow-path-eligible variant (or `get_dynamic_runtime_args`), re-audit:
   the row-major `logical_shape` reads would become harmless, and the hash could be relaxed
   further.
