# Program Cache Audit — `data_movement/sharded_partial/interleaved_to_sharded_partial`

Audit of `ttnn::prim::InterleavedToShardedPartialDeviceOperation::compute_program_hash` against
the framework default ("hash everything") key.

| | |
|---|---|
| Device operation | `ttnn::prim::InterleavedToShardedPartialDeviceOperation` (`device/interleaved_to_sharded_partial_op.hpp:14`) |
| Custom hash | `device/interleaved_to_sharded_partial_op.cpp:88-111` |
| `operation_attributes_t` | `InterleavedToShardedPartialParams` — `grid_size`, `shard_spec`, `num_slices`, `slice_index`, `output_mem_config`, `output_dtype` |
| `tensor_args_t` | `Tensor` (the input tensor itself, not a wrapper struct) |
| Program factories | one: `InterleavedToShardedPartialProgramFactory` (`ProgramDescriptor`-based) |
| `override_runtime_arguments` | **Yes**, on the **program factory** (`device/interleaved_to_sharded_partial_program_factory.cpp:435`, declared at `device/interleaved_to_sharded_partial_program_factory.hpp:20`). Hand-written targeted patch, not a rebuild. |
| `get_dynamic_runtime_args` | **No** — no such member exists (see CSV note below) |
| Own cache-hit validator | No — the framework substitutes `validate_on_program_cache_miss` |
| Cache-hit patch mechanism | **Op-owned cache-hit re-derivation** (mode A) |

## Post-fix status — commit fab067a

**Verdict: CLEAR — with justified relaxation(s).** All three program-cache bugs this document
originally recorded (#2 the missing shape term, #4 the missing input `memory_config`, #5 the
unguarded 32x32 tile) are closed. Two omissions remain and both are deliberate category-2
relaxations that buy real reuse: `slice_index`, and keying the *padded* geometry in place of
`(logical_shape, alignment)`. Everything else the key drops is pinned by a `TT_FATAL` that runs on
the hit path. There is no remaining configuration in which a frozen slot can go stale under an
identical key.

What `fab067a` changed in this op:

- Added `input_tensor.padded_shape()` and `input_tensor.memory_config()` to `compute_program_hash`
  (`device/interleaved_to_sharded_partial_op.cpp:109-110`), the two terms the non-partial sibling
  already keyed. This closes findings #2 and #4 and makes the override's stated premise true.
- Added a standard-tile `TT_FATAL` to `validate_on_program_cache_miss`
  (`device/interleaved_to_sharded_partial_op.cpp:27-37`). Because this op declares **no**
  `validate_on_program_cache_hit`, the framework substitutes the miss validator on every hit
  (`ttnn/api/ttnn/device_operation.hpp:265-269`), so the guard runs on the offending second call —
  which is what makes leaving `page_config` out of the key correct by construction (finding #5).
- Corrected the hash comment that named the non-existent `get_dynamic_runtime_args` as the
  re-application mechanism; it now names
  `InterleavedToShardedPartialProgramFactory::override_runtime_arguments`
  (`device/interleaved_to_sharded_partial_op.cpp:90-100`).
- Dropped the trailing "Replaces get_dynamic_runtime_args" from the override's own comment
  (`device/interleaved_to_sharded_partial_program_factory.cpp:441-445`). The rest of that comment —
  "the static work-split … is pinned by the hashed shape/shard-spec" — was false when written and is
  now accurate, because the shape it appeals to is in the key.
- Documented the tile requirement on the Python binding
  (`interleaved_to_sharded_partial_nanobind.cpp:24`).

What remains open:

- **Nothing that affects correctness.** Recommendation 5 below (the hardcoded reader/writer arg
  indices `0` and `7`, kernel indices `0`/`1`, and the positional CB ordering the override depends
  on) was not addressed. It is a maintenance hazard, not a defect: an arg insertion in
  `create_descriptor` would silently shift `starting_idx_h` off slot 7.
- Recommendation 6 (a run under `-DTT_DESCRIPTOR_PATCHING_PARITY_CHECK`, wired into this branch at
  `ttnn/api/ttnn/mesh_device_operation_adapter.hpp:688-702`) was not done. It is now a regression net
  rather than a way to find finding #2.

**Metal 2.0 port: clear.** The op carries a logical-vs-padded relaxation — it keys
`padded_shape()` and drops `logical_shape` — and that maps exactly onto
`TensorSpecRelaxations::match_padded_shape_only`
(`tt_metal/api/tt-metalium/experimental/metal2_host_api/tensor_spec_relaxations.hpp:41,49`), whose
`pertinent_fields` is `PertinentFields{.padded_shape = true}`
(`tt_metal/impl/metal2_host_api/tensor_spec_relaxations.cpp:77-79`). Declaring that flag on the
input `TensorParameter` reproduces this document's key exactly, and because
`hash_tensorspec_with_relaxation` (`:116`) and `tensorspecs_match_with_relaxation` (`:161-201`)
consume the same field set, the ported key and the run-time accept/reject predicate cannot disagree.
The `slice_index` relaxation needs no flag — it is an attribute, not a tensor spec, and stays
omitted-and-patched.

## Cache-hit patch mechanism

The factory defines `override_runtime_arguments`
(`device/interleaved_to_sharded_partial_program_factory.hpp:20`), so the adapter selects the
op-owned branch and bypasses `resolve_bindings` and `get_dynamic_runtime_args` entirely. The hook
lives on the **program factory**, not on the device operation — `interleaved_to_sharded_partial_op.hpp`
declares no such member — so `DescriptorFactory::override_runtime_arguments` is what the
`ProgramDescriptor` arm calls:

```676:687:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
                } else if constexpr (has_override_runtime_arguments()) {
                    // ProgramDescriptor variant, factory owns its cache-hit re-derivation (the
                    // descriptor-era override_runtime_arguments()): re-apply ALL per-dispatch state —
                    // every runtime arg AND every tensor-backed CB address — for the current tensors.
                    // No resolve_bindings (address inference) and no get_dynamic; correct by
                    // construction for in-place, mixed-aliasing, and work-set shifts.
                    DescriptorFactory::override_runtime_arguments(
                        program,
                        attrs,
                        tensor_args,
                        tensor_return_value,
                        std::optional<ttnn::MeshCoordinate>(coordinate_range.start_coord()));
```

Mode A's "re-apply ALL per-dispatch state" guarantee is a description of what a *correct* override
does, not something the framework enforces. This override is deliberately **partial** — it patches
two runtime-arg slots per core plus the output CB address, and nothing else:

```441:473:ttnn/cpp/ttnn/operations/data_movement/sharded_partial/interleaved_to_sharded_partial/device/interleaved_to_sharded_partial_program_factory.cpp
    // Cache-hit fast path. Only TILE layout is supported (validate_on_program_cache_miss); the static
    // work-split (shard extents, curr_idx, num_units) is pinned by the hashed shape/shard-spec. The only
    // per-dispatch values are the source/output buffer addresses and the slice_index-dependent
    // starting_idx_h. Patch just those slots in place instead of rebuilding the whole descriptor (O(1)
    // per core rather than O(num_cores) descriptor work).
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
curr_idx, num_units) is pinned by the hashed shape/shard-spec"*. **Pre-fix, there was no hashed
shape** — `compute_program_hash` contained no shape term of any kind, and that single false premise
was the root of findings #2 and #4. Post-`fab067a` the premise holds: `padded_shape` and the input
`memory_config` are both in the key (`interleaved_to_sharded_partial_op.cpp:109-110`), so every
work-split value the override leaves frozen is now determined by a hashed term.

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
`interleaved_to_sharded_partial_op.hpp:20-30` declares only `validate_on_program_cache_miss`,
`compute_output_specs`, `create_output_tensors` and `compute_program_hash` — the override is a
member of the *factory* (`interleaved_to_sharded_partial_program_factory.hpp:20`), not of the device
operation, so an earlier revision of this document describing it as op-owned in the C++ sense was
wrong. `fab067a` fixed the two comments that named the wrong mechanism: the hash comment now names
`InterleavedToShardedPartialProgramFactory::override_runtime_arguments`
(`interleaved_to_sharded_partial_op.cpp:90-93`) instead of `get_dynamic_runtime_args`, and the
override's own comment no longer ends with "Replaces get_dynamic_runtime_args"
(`interleaved_to_sharded_partial_program_factory.cpp:441-445`). The outcome for `slice_index` was
never affected; only the CSV column and the prose were.

The factory header states the contract accurately and makes no promise the implementation does not
keep:

```18:20:ttnn/cpp/ttnn/operations/data_movement/sharded_partial/interleaved_to_sharded_partial/device/interleaved_to_sharded_partial_program_factory.hpp
    // slice_index is excluded from the program hash, so a hit for a different slice must re-derive
    // starting_idx_h and the buffer addresses; see the .cpp for the patched slots.
    static void override_runtime_arguments(
```

It scopes the re-derivation to `starting_idx_h` and the addresses — exactly what the implementation
does — rather than claiming a full `create_descriptor` rebuild. That narrow contract is only sound
because the shape-derived args it leaves alone are keyed, which is what `fab067a` established.

## Which validator runs on a cache hit

This op defines **no** `validate_on_program_cache_hit` — `interleaved_to_sharded_partial_op.hpp:20`
declares only `validate_on_program_cache_miss`. It therefore takes the favourable branch of the
dispatcher, which substitutes the miss validator on every hit:

```265:269:ttnn/api/ttnn/device_operation.hpp
    if constexpr (HasValidateOnProgramCacheHit<mesh_device_operation_t>) {
        mesh_device_operation_t::validate_on_program_cache_hit(operation_attributes, tensor_args);
    } else {
        mesh_device_operation_t::validate_on_program_cache_miss(operation_attributes, tensor_args);
    }
```

Every `TT_FATAL` in the miss validator is consequently a live constraint on hits. Three verdicts
below depend directly on this branch: #1, where the TILE-only requirement is what makes the
override's unconditional `reader_rt[7]` write a valid index on the hit path; #7, where the
device-storage requirement is what makes the storage kind carry no information; and — since
`fab067a` — #5, where the new 32x32 `TT_FATAL`
(`interleaved_to_sharded_partial_op.cpp:27-37`) is what pins the tile geometry on the hit path and
so licenses leaving `page_config` out of the key. Pre-fix, #5 relied on this branch in the negative
direction: the miss validator ran on hits and contained no tile check, so nothing pinned the tile on
either path.

Had this op defined even a narrow hit validator, all three pinned verdicts would degrade to "pinned
only on the miss path", the `reader_rt[7]` write would become an unguarded out-of-bounds hazard on
the row-major path rather than a benign one, and the tile guard `fab067a` added would stop
protecting the omission it was added to protect. **Any future `validate_on_program_cache_hit` on
this op must call the miss validator or duplicate all three checks.**

## Baseline: what the default hash would cover

`tensor_args_t` is a bare `Tensor`, so reflection decomposes it directly:

| Source | Fields |
|---|---|
| `operation_attributes` | `grid_size`, `shard_spec`, `num_slices`, `slice_index`, `output_mem_config`, `output_dtype` |
| `input_tensor.storage` | storage variant kind |
| `input_tensor.tensor_spec` | `logical_shape`, and `tensor_layout` = { `dtype`, `page_config`, `memory_config`, `alignment` } |

## What the custom hash covers

```101:110:ttnn/cpp/ttnn/operations/data_movement/sharded_partial/interleaved_to_sharded_partial/device/interleaved_to_sharded_partial_op.cpp
    return tt::tt_metal::operation::hash_operation<InterleavedToShardedPartialDeviceOperation>(
        operation_attributes.grid_size,
        operation_attributes.shard_spec,
        operation_attributes.num_slices,
        operation_attributes.output_mem_config,
        operation_attributes.output_dtype,
        input_tensor.dtype(),
        input_tensor.layout(),
        input_tensor.padded_shape(),
        input_tensor.memory_config());
```

Five of six attributes are kept; `slice_index` is the documented omission. From the input tensor,
`dtype()`, `layout()`, `padded_shape()` and `memory_config()` survive — **no logical shape, no
alignment, no tile**. The last two terms are `fab067a`'s addition; pre-fix the tensor contributed
only `dtype()` and `layout()`, i.e. **no shape, no memory config, no tile**, which is what findings
#2 and #4 were about.

**Comparison against the sibling op.** The non-partial `interleaved_to_sharded` hashes
`input_tensor.memory_config()` *and* `input_tensor.padded_shape()` on top of dtype and layout
(`sharded/interleaved_to_sharded/device/interleaved_to_sharded_op.cpp:144-151`). Pre-fix this op ran
a near-identical factory with both of those terms missing, making the partial one the outlier;
`fab067a` brought the two keys into agreement on this axis. The remaining asymmetry runs the other
way and favours the partial op: the sibling's row-major branch reads `logical_shape()` /
`logical_volume()` while keying only `padded_shape`, which is a live bug in *that* op
(`sharded/interleaved_to_sharded/PROGRAM_CACHE_AUDIT.md`, finding #1). This op cannot reproduce it
because it rejects row-major outright (`interleaved_to_sharded_partial_op.cpp:26`).

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

**Verdict: RESOLVED by fab067a** (was BUG).

`fab067a` added `input_tensor.padded_shape()` to `compute_program_hash`
(`interleaved_to_sharded_partial_op.cpp:109`). That single term closes the finding outright: every
work-split value below is a function of `padded_shape()` and `physical_volume()`, and
`physical_volume()` *is* `padded_shape().volume()` (`ttnn/core/tensor/tensor.cpp:438`), so both
reproductions now produce a cache miss and a correctly rebuilt program. Hashing the padded rather
than the logical shape was the right choice — it preserves the relaxation described in #3.

Pre-fix: the TILE branch derives the entire per-core work split from the input shape:

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

Pre-fix, every hashed term was bit-identical across the two calls (`grid_size` `(8,8)`, the
`ShardSpec`, `num_slices = 1`, `output_mem_config` `{HEIGHT_SHARDED, L1}`, `output_dtype`, input
`dtype`, input `layout`), so call 2 hit call 1's program. The end core keeps `shard_height = 2` and
`curr_num_units_per_shard = 8`, and starts at `curr_idx_h + curr_idx_w = 8`. The input holds only
`3 * 4 = 12` tiles, so the reader issues NOC reads for tiles 8 through 15 — **four tiles past the
end of the input buffer** — and the second output shard is filled with whatever follows the
allocation in DRAM. There is no cache miss, no assertion, and `validate_on_program_cache_miss`
checks only `total_height % num_slices == 0` (lines 27-29), never that the shard grid matches the
input extent.

Pre-fix, the width dimension collided the same way: `num_units_per_row` (reader arg 4, and the driver
of the `curr_idx_w`/`curr_idx_h` walk at lines 315-319) is `padded_shape[-1] / TILE_WIDTH` and was
completely absent from the key, so a WIDTH_SHARDED call whose width changed without changing
`div_up(total_width, shard_shape[1])` produced the same collision on `num_units_offset` and
`padded_offset_bytes`.

The fix applied was exactly the one recommended: hash `input_tensor.padded_shape()`, matching the
non-partial sibling. Both dimensions are covered by it, and the relaxation described in #3 survives.

### 3. `input_tensor.logical_shape()` specifically, as distinct from `padded_shape()`

**Verdict: VALID — relaxation win** (the condition is now met: #2 was fixed by hashing
`padded_shape`, not the whole tensor).

Listing this separately because the right fix for #2 is not "hash the tensor". Nothing in the
factory or in `compute_output_specs` reads the logical shape — the TILE branch above uses only
`padded_shape()[-1]` and `physical_volume()`, and the output spec is built from the padded shape:

```60:80:ttnn/cpp/ttnn/operations/data_movement/sharded_partial/interleaved_to_sharded_partial/device/interleaved_to_sharded_partial_op.cpp
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

This is the op's one logical-vs-padded relaxation, and it is the shape a Metal 2.0 port expresses
directly. `TensorSpecRelaxations::match_padded_shape_only`
(`tt_metal/api/tt-metalium/experimental/metal2_host_api/tensor_spec_relaxations.hpp:41,49`) resolves
to `PertinentFields{.padded_shape = true}`
(`tt_metal/impl/metal2_host_api/tensor_spec_relaxations.cpp:77-79`), i.e. the input's
`tensor_layout` plus its padded shape — the same equivalence class this hash defines. Because
`hash_tensorspec_with_relaxation` (`:116`) and `tensorspecs_match_with_relaxation` (`:161-201`)
consume one shared field set, declaring the flag keeps the ported key and the run-time accept/reject
predicate in agreement by construction, rather than leaving a projection that hits and then throws.

### 4. `input_tensor.memory_config()` — the input's buffer type is not hashed

**Verdict: RESOLVED by fab067a** (was BUG).

`fab067a` added `input_tensor.memory_config()` to `compute_program_hash`
(`interleaved_to_sharded_partial_op.cpp:110`). The `MemoryConfig` carries `buffer_type()`, so the
reader's `TensorAccessorArgs` `IsDram` bit, the scratch CB's existence and the `convert_df` input CB
page size are all keyed now; a DRAM-then-L1 pair misses and rebuilds. Note the fix had to be in the
hash rather than in a validator: the op legitimately supports both placements, so there was nothing
to pin.

Pre-fix: validation constrains the input to be interleaved but says nothing about where it lives:

```45:47:ttnn/cpp/ttnn/operations/data_movement/sharded_partial/interleaved_to_sharded_partial/device/interleaved_to_sharded_partial_op.cpp
    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Input tensor must be Interleaved");
```

DRAM-interleaved and L1-interleaved inputs are both admissible and, pre-fix, hashed identically. The
source buffer's placement is baked into the reader's **compile-time** args:

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

**Pre-fix reproduction**: call 1 with a TILE `bfloat16` **DRAM-interleaved** input; call 2 with the
same-shaped, same-dtype tensor in **L1 interleaved**, all attributes identical. The hash was
unchanged, so call 2 hit. The override patches reader arg 0 to the L1 buffer's address, but the
compiled `TensorAccessor` still resolved pages through the DRAM bank table: the reader issued DRAM
NOC reads at an offset derived from an L1 address, the L1 input was never touched, and the output
shards contained unrelated memory.

The same omission had two further structural effects on the TILE path, either of which was
independently sufficient to make this a BUG. `src_is_dram` gates whether the scratch CB is created
at all:

```169:185:ttnn/cpp/ttnn/operations/data_movement/sharded_partial/interleaved_to_sharded_partial/device/interleaved_to_sharded_partial_program_factory.cpp
    uint32_t dram_alignment = hal::get_dram_alignment();
    uint32_t l1_alignment = hal::get_l1_alignment();
    uint32_t num_trids = 4;
    if ((src_is_dram && (input_unit_size % dram_alignment != 0)) || (is_blackhole || is_quasar) || keep_l1_aligned) {
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

```144:156:ttnn/cpp/ttnn/operations/data_movement/sharded_partial/interleaved_to_sharded_partial/device/interleaved_to_sharded_partial_program_factory.cpp
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

The fix applied was the one-liner recommended here: add `input_tensor.memory_config()` to the hash,
as the non-partial sibling already does.

### 5. `input_tensor.tensor_spec().page_config()` — only `layout()` is hashed, and the factory hardcodes 32x32

**Verdict: RESOLVED by fab067a** (was BUG).

`fab067a` closed this with a guard rather than a hash term, which is the cheaper of the two routes
the recommendation offered: `validate_on_program_cache_miss` now rejects any tile other than 32x32
(`interleaved_to_sharded_partial_op.cpp:27-37`). What makes that sufficient rather than miss-only is
that this op declares **no** `validate_on_program_cache_hit`, so the framework substitutes the miss
validator on hits (`ttnn/api/ttnn/device_operation.hpp:265-269`) and the `TT_FATAL` fires on the
offending second call. With the tile pinned and `layout()` hashed, `PageConfig` carries no residual
information and the factory's bare `TILE_HEIGHT`/`TILE_WIDTH` arithmetic is consistent with it.

Pre-fix: `layout()` collapses `PageConfig` to `ROW_MAJOR` vs `TILE`, discarding the `Tile` shape, and
all three conditions for the unguarded-tile bug class held.

The op accepts `Layout::TILE` — indeed it accepts nothing else
(`interleaved_to_sharded_partial_op.cpp:26`). The factory then computes every tile quantity from
the architectural constants rather than the tensor's own tile:

```89:104:ttnn/cpp/ttnn/operations/data_movement/sharded_partial/interleaved_to_sharded_partial/device/interleaved_to_sharded_partial_program_factory.cpp
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
`tile.get_tile_size(format)` — and lines 98-104 convert shapes to tile counts with bare
`TILE_HEIGHT`/`TILE_WIDTH`. `calculate_starting_idx_h` does the same
(`sharded_common.cpp:21-22`). The `TT_FATAL` at lines 92-97 checks the *shard shape* against the
constants; it says nothing about the tensor's `Tile`, and pre-fix
`validate_on_program_cache_miss` had no tile check either. Nothing in the factory reads
`tensor_spec().tile()`; that remains true — the guard lives in the validator, not the factory.

Pre-fix, because the factory hardcodes 32x32 *and* `page_config` is unhashed, the two defects
compounded: a `Tile{16, 32}` input did not even get a freshly-built wrong program. It silently
inherited the 32x32 entry built for a same-dtype, same-shard-spec 32x32 tensor, with
`input_unit_size`, the CB page sizes and every per-core tile count computed for the wrong tile
geometry.

The guard `fab067a` added is the `TT_FATAL` form of the one the non-partial sibling already carried:

```94:98:ttnn/cpp/ttnn/operations/data_movement/sharded/interleaved_to_sharded/device/interleaved_to_sharded_op.cpp
    if (input_tensor.layout() == Layout::TILE) {
        auto tile = input_tensor.tensor_spec().tile();
        if (tile.get_height() != tt::constants::TILE_HEIGHT || tile.get_width() != tt::constants::TILE_WIDTH) {
            return {false, fmt::format("interleaved_to_sharded requires standard 32x32 tiles, got {}x{}", tile.get_height(), tile.get_width())};
        }
```

```30:37:ttnn/cpp/ttnn/operations/data_movement/sharded_partial/interleaved_to_sharded_partial/device/interleaved_to_sharded_partial_op.cpp
    {
        const auto tile = input_tensor.tensor_spec().tile();
        TT_FATAL(
            tile.get_height() == tt::constants::TILE_HEIGHT && tile.get_width() == tt::constants::TILE_WIDTH,
            "interleaved_to_sharded_partial does not currently support tiles other than 32x32, got {}x{}",
            tile.get_height(),
            tile.get_width());
    }
```

That makes the `page_config` omission correct by construction. Making the factory genuinely
tile-aware instead would still require adding `page_config` to the hash in the same change — and
would also mean revisiting `calculate_starting_idx_h`, which is shared with the non-partial path.
Note the residual the guard cannot reach: `Tile::attribute_values()` and `Tile::operator==` both
exclude `transpose_within_face` / `transpose_of_faces`, so those flags are invisible to every hash in
the repo and to the new check. The factory never reads them.

### 6. `input_tensor.tensor_layout().get_alignment()`

**Verdict: VALID — relaxation win** (was CAVEAT; the condition it was waiting on is now met).

The factory never reads `Alignment` directly. For a tile-layout tensor it reaches the program only
through `padded_shape` and `physical_volume` — both of which were the subject of finding #2 — and
through the buffer page size, which depends on `memory_config` (finding #4). `fab067a` hashed both,
which is exactly the condition this entry named: with `padded_shape` and `memory_config` in the key
the omission of `Alignment` is genuinely safe, since a tile tensor's alignment is constrained to
multiples of the tile dimensions (`validate_alignment_tile`,
`tt_metal/impl/tensor/spec/layout/page_config.cpp:59-75`) and contributes nothing beyond those two
hashed derivations. `Buffer::alignment()` is a separate quantity —
`allocator_->get_alignment(buffer_type())` (`tt_metal/impl/buffers/buffer.cpp:768`) — and is fixed by
the hashed `memory_config`, not by `TensorLayout::Alignment`.

It was a CAVEAT rather than a BUG for reachability reasons, and those still hold in the resolved
state: there was no configuration in which the `Alignment` omission produced a wrong hit that was not
already the reproduction given for #2 or #4, because every route by which a different alignment
changes the program passes through the padded shape, the physical volume or the page size. Marking it
a fourth BUG would have counted one defect three times. Unlike the corresponding entry in the
non-partial `interleaved_to_sharded` audit, `Alignment` is not the *enabler* of anything here —
findings #2 and #4 reproduced with entirely default alignments, and the sibling's row-major bug (for
which alignment *is* the enabler) has no counterpart on a TILE-only op.

Dropping `(logical_shape, alignment)` in favour of the hashed `padded_shape` is one relaxation seen
from two sides; #3 records the other side and the `match_padded_shape_only` mapping that expresses it
in Metal 2.0.

### 7. `input_tensor.storage` variant kind (device vs host)

**Verdict: VALID — pinned by validation.**

```42:43:ttnn/cpp/ttnn/operations/data_movement/sharded_partial/interleaved_to_sharded_partial/device/interleaved_to_sharded_partial_op.cpp
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to shard need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to shard need to be allocated in buffers on device!");
```

Constant across every admissible call, and re-checked on hits through the substitution branch
quoted under "Which validator runs on a cache hit".

### 8. Buffer addresses (omitted by the default hash as well)

**Verdict: VALID — patched.**

Addresses reach the program by two routes and the override covers both. A DRAM output rides on
writer arg 0; an L1 sharded output rides on the output CB's `.buffer` binding (line 167), which
the override refreshes with a synthetic address-only descriptor:

```475:488:ttnn/cpp/ttnn/operations/data_movement/sharded_partial/interleaved_to_sharded_partial/device/interleaved_to_sharded_partial_program_factory.cpp
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
drift. Note that the scratch CB (line 177) is pushed *after* the output CB and carries no buffer,
so it does not disturb the indices the override relies on.

## Keys the custom hash adds beyond the default

- `input.padded_shape()` — not in the default key, which hashes `logical_shape` and lets the padded
  shape be a derived value. Added by `fab067a`
  (`interleaved_to_sharded_partial_op.cpp:109`). This is the term that makes dropping
  `logical_shape` and `alignment` safe simultaneously, and it is what turns findings #2, #3 and #6
  from "the op took a relaxation without adding the compensating key" into a properly compensated
  relaxation.

Nothing else. Every other term (`grid_size`, `shard_spec`, `num_slices`, `output_mem_config`,
`output_dtype`, input `dtype`, input `layout`, input `memory_config`) is either an attribute the
default also hashes or a projection of a tensor field the default hashes in full. Pre-fix the key was
a strict *subset* of the default, with no compensating term at all — which is precisely why #2, #4
and #5 were bugs rather than relaxations.

## Framework side effect of having a custom hash

```1012:1014:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
        if constexpr (requires { DeviceOperation::compute_program_hash(attrs, tensor_args); }) {
            return key;  // custom hash -> opt out beyond the op-identity prefix
        } else {
```

`ProgramCacheKey::canonical` degrades to the op type name alone, so even a genuine 64-bit
collision between two distinct configurations resolves to a wrong hit rather than a rebuild. Pre-fix
that compounded the findings above, which were exact-key collisions rather than hash collisions and
therefore occurred with probability 1. Post-`fab067a` it is the op's only remaining second-order
exposure: the key is now complete, so the residual is a true 64-bit collision between two
legitimately different configurations, with no attribute-level tiebreak behind it. That is inherent
to every custom-hash op and is not a reason to withhold the CLEAR verdict.

## Summary

| Omitted vs. default | Used by program? | Patched on hit? | Verdict |
|---|---|---|---|
| `slice_index` | Yes — `starting_idx_h` (reader arg 7, writer arg 7) | Yes (override) | VALID — patched |
| `input.padded_shape` / `physical_volume` | Yes — reader args 1-6, writer args 1-6, compute arg 0 | No — but now hashed | **RESOLVED by fab067a** (was BUG) |
| `input.logical_shape` (beyond the padded shape) | No | n/a | VALID — relaxation win |
| `input.memory_config()` | Yes — `TensorAccessorArgs` compile-time args, scratch-CB existence, input CB page size | No — but now hashed | **RESOLVED by fab067a** (was BUG) |
| `input.page_config` (`Tile`) | Yes — `tt::tile_size`, bare `TILE_HEIGHT`/`TILE_WIDTH` | No — now pinned to 32x32 by the miss validator, which runs on hits | **RESOLVED by fab067a** (was BUG) |
| `input.tensor_layout.alignment` | Only via `padded_shape` and page size, both now hashed | No | VALID — relaxation win (the other side of #3) |
| `input.storage` kind | n/a | n/a | VALID — pinned by validation |
| Buffer addresses | Yes | Yes (override: rt args + output CB) | VALID — patched |

**Post-`fab067a`: zero program-cache correctness bugs.** All three the original audit found (#2, #4,
#5) are closed, and they shared one root cause — the override justified its narrowness by appealing
to a hash term that did not exist. That term exists now.

The comment guarding the two-slot patch reads:

```441:445:ttnn/cpp/ttnn/operations/data_movement/sharded_partial/interleaved_to_sharded_partial/device/interleaved_to_sharded_partial_program_factory.cpp
    // Cache-hit fast path. Only TILE layout is supported (validate_on_program_cache_miss); the static
    // work-split (shard extents, curr_idx, num_units) is pinned by the hashed shape/shard-spec. The only
    // per-dispatch values are the source/output buffer addresses and the slice_index-dependent
    // starting_idx_h. Patch just those slots in place instead of rebuilding the whole descriptor (O(1)
    // per core rather than O(num_cores) descriptor work).
```

and the hash it appeals to now carries the shape:

```101:110:ttnn/cpp/ttnn/operations/data_movement/sharded_partial/interleaved_to_sharded_partial/device/interleaved_to_sharded_partial_op.cpp
    return tt::tt_metal::operation::hash_operation<InterleavedToShardedPartialDeviceOperation>(
        operation_attributes.grid_size,
        operation_attributes.shard_spec,
        operation_attributes.num_slices,
        operation_attributes.output_mem_config,
        operation_attributes.output_dtype,
        input_tensor.dtype(),
        input_tensor.layout(),
        input_tensor.padded_shape(),
        input_tensor.memory_config());
```

"Pinned by the hashed shape/shard-spec" was half right when written — the shard spec was hashed, the
shape was not, in either form, and neither was the input's `memory_config`. The work split the comment
calls static is computed from `padded_shape()` and `physical_volume()`
(`interleaved_to_sharded_partial_program_factory.cpp:101-111`), and those are now keyed, so the
sentence is true as written. The narrow patch is sound *because* of that, not in spite of it: the
override may keep patching four slots precisely because everything else it leaves frozen is a
function of a hashed term.

So the op no longer has the weakest hash of the pair. It matches the non-partial
`interleaved_to_sharded` on `padded_shape` and `memory_config`, and on the tile axis it is now
*stronger* than that sibling in one respect and equal in another: both reject non-32x32 tiles, but the
sibling's row-major branch reads `logical_shape()` / `logical_volume()` while keying only
`padded_shape`, a live bug this op cannot reproduce because it rejects row-major outright
(`interleaved_to_sharded_partial_op.cpp:26`).

The `slice_index` omission — the one the op deliberately made and documented — was correct all along,
and the set of slots the override patches (reader 0/7, writer 0/7, output CB) exactly matches the set
of values that depend on `slice_index` and on buffer addresses.

Two CSV corrections. `get_dynamic_runtime_args` should be **N**: no such member exists, and the
factory's `override_runtime_arguments` supersedes it. And `own_hit_validator = N` understates the
situation, since the dispatcher substitutes `validate_on_program_cache_miss` on hits
(`ttnn/api/ttnn/device_operation.hpp:265-269`), which verdicts #1, #5 and #7 all rely on.

## Recommendations

Items 1-4 were implemented by `fab067a` and are retained for provenance; 5 and 6 remain open.

1. ~~Add `input_tensor.padded_shape()` to `compute_program_hash`.~~ **Done**
   (`interleaved_to_sharded_partial_op.cpp:109`). The padded rather than the logical shape was
   hashed, so the relaxation in #3 is preserved.
2. ~~Add `input_tensor.memory_config()` to `compute_program_hash`.~~ **Done**
   (`interleaved_to_sharded_partial_op.cpp:110`). The key now matches the non-partial
   `interleaved_to_sharded` in coverage, which was the right target given the two factories are
   near-copies.
3. ~~Add the equivalent of the 32x32 tile check at `interleaved_to_sharded_op.cpp:94-98` to
   `validate_on_program_cache_miss`, as a `TT_FATAL` to match this op's validator style.~~ **Done**
   (`interleaved_to_sharded_partial_op.cpp:27-37`). It is hit-reachable only because the op has no
   `validate_on_program_cache_hit`; see the caveat under "Which validator runs on a cache hit".
4. ~~Fix the stale comments.~~ **Done** for the two that existed: the hash comment no longer names
   `get_dynamic_runtime_args` (`interleaved_to_sharded_partial_op.cpp:90-100`) and the override's
   comment no longer claims to replace it
   (`interleaved_to_sharded_partial_program_factory.cpp:441-445`). The third item on the original
   list — a device-operation header comment claiming the override re-runs `create_descriptor` — does
   not exist in the tree; the override is declared on the factory
   (`interleaved_to_sharded_partial_program_factory.hpp:18-20`) with an accurate comment.
5. **Open.** The override depends on hardcoded arg indices (0 and 7), kernel indices (0 and 1) and CB
   ordering that must stay in lockstep with `create_descriptor`. Extract the reader/writer arg
   layout into named constants shared by both functions so a future arg insertion cannot silently
   shift `starting_idx_h` off slot 7.
6. **Open.** Run this op's tests once under `-DTT_DESCRIPTOR_PATCHING_PARITY_CHECK`. The parity oracle
   is wired into the mode-A branch (`mesh_device_operation_adapter.hpp:688-702`) and rebuilds the
   descriptor as a reference. It would have flagged finding #2 directly as a mismatch on reader args
   1-6; with the key fixed it is now a regression net against a future arg being added to
   `create_descriptor` without a matching line in the override.
