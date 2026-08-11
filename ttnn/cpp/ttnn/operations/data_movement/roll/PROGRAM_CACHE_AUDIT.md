# Program Cache Audit — `data_movement/roll`

Audit of `ttnn::prim::RollDeviceOperation::compute_program_hash` against the framework
default ("hash everything") key.

| | |
|---|---|
| Device operation | `ttnn::prim::RollDeviceOperation` (`device/roll_device_operation.hpp`) |
| Custom hash | `device/roll_device_operation.cpp:44` |
| `operation_attributes_t` | `RollParams` — `shift`, `dim`, `output_mem_config` |
| `tensor_args_t` | `RollInputs` — `input` |
| Program factories | `RollShardedProgramFactory` (single-alternative variant, `ProgramDescriptor`-based) |
| `override_runtime_arguments` | **Yes** — on the device operation (`device/roll_program_factory.cpp:562`) |
| `get_dynamic_runtime_args` | **No** |
| Own cache-hit validator | **Yes** — `validate_on_program_cache_hit` (`device/roll_device_operation.cpp:39`), delegating to the same helper as the miss validator |
| Cache-hit patch mechanism | **Op-owned cache-hit re-derivation** (mode A) |

## Cache-hit patch mechanism

`RollDeviceOperation` declares `override_runtime_arguments`, so the descriptor adapter takes the
op-owned branch and never consults `resolve_bindings`:

```657:678:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
                } else if constexpr (has_override_runtime_arguments()) {
                    // ProgramDescriptor variant, op owns its cache-hit re-derivation (the descriptor-era
                    // override_runtime_arguments()): re-apply ALL per-dispatch state — every runtime arg
                    // AND every tensor-backed CB address — for the current tensors.  No resolve_bindings
                    // (address inference) and no get_dynamic; correct by construction for in-place,
                    // mixed-aliasing, and work-set shifts. Prefer the factory's hook; fall back to the
                    // DeviceOperation for direct ops that predate the factory-struct shape.
```

The op's implementation is the strongest possible form of mode A — it re-runs the factory itself
and replays the whole descriptor:

```562:572:ttnn/cpp/ttnn/operations/data_movement/roll/device/roll_program_factory.cpp
void RollDeviceOperation::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // Re-derive all per-dispatch state from the single source of truth (create_descriptor) for the
    // current tensors and re-apply to the cached program -- no rebuild, still a cache hit.
    auto desc = RollShardedProgramFactory::create_descriptor(operation_attributes, tensor_args, tensor_return_value);
    tt::tt_metal::apply_descriptor_runtime_args(program, desc);
}
```

**This is not a blanket guarantee.** `apply_descriptor_runtime_args` copies runtime-arg *values*
into the storage the cached `Program` already owns; it does not resize anything:

```187:192:tt_metal/impl/program/program_descriptors.cpp
        for (const auto& [core, args] : kernel.runtime_args) {
            auto& prog_args = GetRuntimeArgs(program, k, core);
            for (uint32_t i = 0; i < static_cast<uint32_t>(args.size()); ++i) {
                prog_args[i] = args[i];
            }
        }
```

and `RuntimeArgsData::operator[]` only bounds-checks under `TT_ASSERT`, i.e. not in release
builds:

```36:39:tt_metal/api/tt-metalium/runtime_args_data.hpp
    std::uint32_t& operator[](std::size_t index) noexcept {
        TT_ASSERT(in_bounds(index));
        return this->rt_args_data[index];
    }
```

So the obligation on this op's hash is:

1. Everything that changes a compile-time arg, a CB size/format, a core range or the kernel
   source must be hashed (mode A does not refresh those; they are baked into the cached `Program`).
2. Everything that changes the **number** of runtime args per core must also be hashed, because
   the cached program's per-core arg vector was sized on the first miss.

Runtime-arg *values* and CB base addresses are safe — `apply_descriptor_runtime_args` also calls
`UpdateDynamicCircularBufferAddress` for every `desc.cbs[i].buffer`
(`tt_metal/impl/program/program_descriptors.cpp:221-233`), which covers the L1-mode CB0/CB16
bindings.

## Which validator runs on a cache hit

Roll is the less common of the two cases, and it is the hazardous one in general: it **defines**
`validate_on_program_cache_hit`, so the dispatcher runs that and *not* the miss validator on hits.

```262:266:ttnn/api/ttnn/device_operation.hpp
    if constexpr (HasValidateOnProgramCacheHit<mesh_device_operation_t>) {
        mesh_device_operation_t::validate_on_program_cache_hit(operation_attributes, tensor_args);
    } else {
        mesh_device_operation_t::validate_on_program_cache_miss(operation_attributes, tensor_args);
    }
```

An op in this branch normally loses every check that lives only in its miss validator. Roll does
not, because both entry points delegate to the same helper and neither adds anything of its own:

```34:42:ttnn/cpp/ttnn/operations/data_movement/roll/device/roll_device_operation.cpp
void RollDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_roll(operation_attributes, tensor_args);
}

void RollDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_roll(operation_attributes, tensor_args);
}
```

Diffed against each other the two are identical — both bodies are the single call
`validate_roll(operation_attributes, tensor_args)` and neither has any other statement — so the hit
path pins exactly what the miss path pins: all five `TT_FATAL`s at `roll_device_operation.cpp:19-30`
(device storage, non-null buffer, input sharded, output sharded, single-rectangle input grid). Any
verdict below that says "pinned by validation" is therefore legitimate on hits, and unusually it does
not depend on the framework's substitution branch at all.

**Nothing is dropped on the hit path, so there is no reachability analysis to do here.** The
hit-path filter — asking of each dropped check whether the value it constrains is itself in the cache
key, and so whether the check can be evaded by a call that hits — applies only to ops whose hit
validator pins strictly less than their miss validator. Roll's does not; the dropped set is empty. No
verdict in this document rests on the hit validator being narrower than the miss validator, and none
is affected by the filter.

The identity is worth preserving deliberately: if a future change adds a check to `validate_roll`'s
miss path only, or narrows the hit validator, the hit path silently loses it.

Note separately that roll gets a *second* round of checking on hits that most ops do not, because
its `override_runtime_arguments` re-runs `create_descriptor`, and that function carries its own
`TT_FATAL`s (shard-shape equality at lines 146-148, divisibility at 149 and 158, the
single-rectangle output grid at 167-168). Verdicts below distinguish carefully between checks in
`validate_roll` and checks in `create_descriptor`, since only the latter see the *output* shard
spec.

## Baseline: what the default hash would cover

`hash_objects_with_default_seed(type_hash<RollDeviceOperation>, attrs, tensor_args)` walks
reflection, so the default key is:

| Source | Fields |
|---|---|
| `operation_attributes` | `shift`, `dim`, `output_mem_config` |
| `input.storage` | storage variant kind (`DeviceStorage` / `HostStorage`; both have empty attribute tuples) |
| `input.tensor_spec` | `logical_shape`, and `tensor_layout` = { `dtype`, `page_config`, `memory_config`, `alignment` } |

## What the custom hash covers

```44:48:ttnn/cpp/ttnn/operations/data_movement/roll/device/roll_device_operation.cpp
ttsl::hash::hash_t RollDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& args) {
    return tt::tt_metal::operation::hash_operation<RollDeviceOperation>(
        attrs.shift, attrs.dim, args.input.memory_config(), args.input.dtype(), args.input.layout());
}
```

Five values. Note what is *not* there: `output_mem_config` (an operation attribute the default
would have hashed) and **any** description of the input tensor's shape.

## Omitted parameters

### 1. `input.logical_shape()` / `input.padded_shape()` — the tensor shape is not hashed at all

**Verdict: BUG.**

The factory reads the input's padded shape and decomposes it into per-dimension extents that drive
the entire gather plan:

```116:120:ttnn/cpp/ttnn/operations/data_movement/roll/device/roll_program_factory.cpp
    const auto& shape = input.padded_shape();
    const uint32_t rank = shape.rank();
    const uint32_t shift = operation_attributes.shift;
    const int32_t dim = operation_attributes.dim;
    const bool is_last_dim = (static_cast<uint32_t>(dim) == rank - 1);
```

```151:157:ttnn/cpp/ttnn/operations/data_movement/roll/device/roll_program_factory.cpp
    // Cell-row dim sizes (dims 0..rank-2 collapsed): the height dim is measured in tile-rows.
    std::vector<uint32_t> rd(rank, 1);
    uint32_t H_cells = 1;
    for (uint32_t i = 0; i + 1 < rank; i++) {
        rd[i] = (i == rank - 2) ? shape[i] / cell_h : shape[i];
        H_cells *= rd[i];
    }
```

```221:230:ttnn/cpp/ttnn/operations/data_movement/roll/device/roll_program_factory.cpp
    // Coordinate shift for the rolled dim, in cell-row units (tile-rows for the height dim).
    const uint32_t dim_size_cells = (static_cast<uint32_t>(dim) == rank - 2) ? rd[dim] : shape[dim];
    const uint32_t shift_cells = (static_cast<uint32_t>(dim) == rank - 2) ? shift / cell_h : shift;

    auto rolled_src_row = [&](uint32_t r) -> uint32_t {
        // Decrement the dim-th coordinate by shift (mod dim_size); other coords unchanged.
        const uint32_t coord_d = (r / row_stride[dim]) % dim_size_cells;
        const uint32_t src_coord_d = (coord_d + dim_size_cells - (shift_cells % dim_size_cells)) % dim_size_cells;
        return r + (src_coord_d - coord_d) * row_stride[dim];
    };
```

Two aggregate quantities *are* pinned by the hashed `input.memory_config()`: the total cell-row
count `H_cells` (= number of shards × shard height) and the width `W_cells` (= shard width ×
number of shard columns), because a valid 2D shard spec fixes the 2D physical extent. What is
**not** pinned is how `H_cells` factorises into `rd[0..rank-2]`, i.e. the N-D decomposition — and
`dim_size_cells` / `row_stride[dim]` are read straight out of that decomposition.

Changing the decomposition changes the *permutation*, which changes the number of coalesced
transfer runs per core, which changes the runtime-arg count:

```343:348:ttnn/cpp/ttnn/operations/data_movement/roll/device/roll_program_factory.cpp
    const uint32_t args_per_transfer = is_dram_rm ? 7u : (is_dram ? 7u : 9u);
    const uint32_t args_overhead = is_dram_rm ? 8u : (is_dram ? 3u : 1u);
    TT_FATAL(
        args_overhead + max_num_transfers * args_per_transfer <= runtime_args_limit,
        "Native sharded roll: too many copy segments per core ({}). Reduce grid/shape.",
        max_num_transfers);
```

```381:396:ttnn/cpp/ttnn/operations/data_movement/roll/device/roll_program_factory.cpp
    auto build_runtime_args_l1 = [&](const std::vector<RollTransferDesc>& descs) {
        KernelDescriptor::CoreRuntimeArgs args;
        args.reserve(1 + descs.size() * 9);
        args.push_back(static_cast<uint32_t>(descs.size()));
        for (const auto& td : descs) {
            args.push_back(td.src_physical_core.x);
            args.push_back(td.src_physical_core.y);
            args.push_back(input_cb_id);
            args.push_back(td.src_l1_offset);
            args.push_back(td.dst_offset);
            args.push_back(td.copy_size);
            args.push_back(td.src_stride);
            args.push_back(td.dst_stride);
            args.push_back(td.num_rows);
        }
        return args;
    };
```

**Reproduction.** Take a `bfloat16`, `ROW_MAJOR`, HEIGHT_SHARDED-in-L1 tensor with
`shard_spec = {grid = CoreRange((0,0),(1,0)), shape = [32, 64], ROW_MAJOR}`. Both calls use
`ttnn.roll(t, shifts=[1], dim=[2])`, which reaches
`ttnn::prim::roll_sharded(result, /*shift=*/1, /*dim=*/2, input.memory_config())`
(`roll.cpp:90-91`) with the shift already normalised to 1 in both cases.

- **Call 1** — input logical/padded shape `[1, 2, 32, 64]`. `rd = {1, 2, 32}`,
  `dim_size_cells = 32`, `row_stride[2] = 1`. The row permutation is `src = r - 1` with one wrap
  per 32-row block, so each destination core coalesces into **2** transfers →
  `1 + 2*9 = 19` runtime args per core.
- **Call 2** — input logical/padded shape `[1, 4, 16, 64]`. Same `H_cells = 64`, same
  `W_cells = 64`, same `MemoryConfig`, same dtype, same layout, same `shift`, same `dim` →
  **identical program hash**. But `rd = {1, 4, 16}` so `dim_size_cells = 16`, giving a wrap every
  16 rows; each destination core now needs **4** transfers → `1 + 4*9 = 37` runtime args per core.

On the cache hit, `override_runtime_arguments` builds the 37-arg vector and
`apply_descriptor_runtime_args` writes `prog_args[0..36]` into a `RuntimeArgsData` whose
`rt_args_count` is 19. In a debug build this is a `TT_FATAL` from `in_bounds`; in a release build
it is a silent out-of-bounds write past the end of the kernel's runtime-arg region in the cached
program, corrupting whatever follows it, while the reader kernel still executes with the 19 slots
it can address and therefore performs the wrong gather.

The reverse order (call 2 first) is "merely" wrong rather than corrupting: 19 args are written
into 37 slots, arg 0 (the transfer count) is correct, and the stale tail is ignored — but that is
luck, not design.

Note that neither `validate_on_program_cache_hit` nor the `TT_FATAL`s inside `compute_roll_plan`
catch this. The validator only checks storage/sharding/grid-cardinality
(`roll_device_operation.cpp:21-29`), and the plan's own assertions
(`roll_program_factory.cpp:146-158`) check divisibility of `H_cells`/`W_cells` by the shard
extents, which both shapes satisfy.

### 2. `operation_attributes.output_mem_config`

**Verdict: BUG.**

This is an operation attribute the default hash would have covered. The factory reads the *output*
shard spec for the shard extents, the grid and the orientation:

```140:158:ttnn/cpp/ttnn/operations/data_movement/roll/device/roll_program_factory.cpp
    const uint32_t W_cells = shape[rank - 1] / cell_w;

    const auto& out_ss = output.shard_spec().value();
    const auto& in_ss = input.shard_spec().value();
    const uint32_t shard_cells_h = out_ss.shape[0] / cell_h;
    const uint32_t shard_cells_w = out_ss.shape[1] / cell_w;
    TT_FATAL(
        in_ss.shape[0] == out_ss.shape[0] && in_ss.shape[1] == out_ss.shape[1],
        "Native sharded roll expects identical input/output shard shapes");
    TT_FATAL(W_cells % shard_cells_w == 0, "Shard width must evenly divide the tensor");
```

`shard_cells_h`/`shard_cells_w` feed `row_pitch_bytes`, `shard_l1_size` and `scratch_half`, which
are CB sizes *and* compile-time args:

```353:362:ttnn/cpp/ttnn/operations/data_movement/roll/device/roll_program_factory.cpp
    std::vector<uint32_t> compile_time_args = {
        output_cb_id,
        scratch_cb_id,
        l1_alignment,
        scratch_half,
        mode,
        shard_l1_size,
        dram_rm_src0_cb_id,
        dram_rm_src1_cb_id,
        dram_rm_dst_cb_id};
```

Exactly one component of the output shard spec is enforced. The shard **shape** is pinned to the
input's by the `TT_FATAL` at lines 146-148, and because `override_runtime_arguments` re-runs
`create_descriptor` that assertion is re-evaluated on every cache hit, not just on the miss; the
input shard shape is hashed inside `input.memory_config()`. So the shape carries no information.

**Grid and orientation are not pinned by anything.** The only assertion touching the grid requires
that it be a single rectangle:

```166:174:ttnn/cpp/ttnn/operations/data_movement/roll/device/roll_program_factory.cpp
    const bool row_major_orient = out_ss.orientation == ShardOrientation::ROW_MAJOR;
    TT_FATAL(
        out_ss.grid.ranges().size() == 1, "Native sharded roll requires a single contiguous rectangular CoreRange");
    const auto& grid_range = *out_ss.grid.ranges().begin();
    const uint32_t grid_cols = grid_range.end_coord.x - grid_range.start_coord.x + 1;
    const uint32_t grid_rows = grid_range.end_coord.y - grid_range.start_coord.y + 1;

    // Number of shard positions in the tensor width direction (used in shard_linear).
    const uint32_t n_shard_cols = W_cells / shard_cells_w;
```

and `out_ss.orientation` then selects the shard-to-core mapping outright:

```185:195:ttnn/cpp/ttnn/operations/data_movement/roll/device/roll_program_factory.cpp
    // shard_linear maps (cell_row, cell_col) → core enumeration index c.
    // The core enumeration is y-outer, x-inner, so c = gy * grid_cols + gx.
    // ROW_MAJOR: shard (sr,sc) → (gy=sr, gx=sc) → c = sr*n_shard_cols + sc.
    // COL_MAJOR: shard (sr,sc) → (gx=sr, gy=sc) → c = sc*grid_cols + sr.
    auto shard_linear = [&](uint32_t row, uint32_t col) -> uint32_t {
        const uint32_t sr = row / shard_cells_h;
        const uint32_t sc = col / shard_cells_w;
        return row_major_orient ? sr * n_shard_cols + sc : sc * grid_cols + sr;
    };

    const uint32_t num_cores = grid_rows * grid_cols;
```

`out_ss.grid` also becomes the CB and kernel `core_ranges`
(`roll_program_factory.cpp:494-556`), which are structural and never refreshed on a hit.

**Reachability.** `ttnn::prim::roll_sharded` is a public C++ entry point declared in
`roll_device_operation.hpp:51` and takes `output_mem_config` as a caller-supplied parameter. The
only enforced constraint on it is `output_mem_config.is_sharded()`
(`roll_device_operation.cpp:24`) plus the shard-shape equality above. Neither the grid extent nor
the orientation is checked against the input's, so the bad configuration is reachable without
violating any enforced constraint.

**Reproduction.** Take a `[1, 1, 64, 64]` TILE `bfloat16` tensor block-sharded on a 2x2 grid with
shard shape `[32, 32]` and `ShardOrientation::ROW_MAJOR`.

- **Call 1**: `ttnn::prim::roll_sharded(input, /*shift=*/1, /*dim=*/2, mc_row_major)` where
  `mc_row_major` is the input's own memory config.
- **Call 2**: the same input and the same shift and dim, but `output_mem_config` is the same
  memory config with `ShardOrientation::COL_MAJOR` and an otherwise identical 2x2 grid and
  `[32, 32]` shard shape.

`attrs.shift`, `attrs.dim`, `input.memory_config()`, `input.dtype()` and `input.layout()` are all
identical, so the two calls produce the same key and call 2 hits. The shard-shape `TT_FATAL` passes
(the shapes are equal), and the single-rectangle check passes. But `row_major_orient` is now false,
so `shard_linear` computes `sc * grid_cols + sr` instead of `sr * n_shard_cols + sc` — cells are
routed to transposed cores. Because the cached program's `core_ranges` and compile-time args are
those of call 1 and only the runtime args are re-applied, the roll gathers from the wrong cores and
the output shards are permuted. For the two off-diagonal shards of a 2x2 grid this silently swaps
half the tensor.

The severity is currently limited by the call graph rather than by any check: the sole in-tree
caller passes the input's own memory config (`roll.cpp:90-91`, with
`native_mem_config = input_tensor.memory_config()` at `roll.cpp:60`), and
`ttnn::prim::roll_sharded` is not bound in `roll_nanobind.cpp`, which exposes only the three
`ttnn::roll` overloads. That bounds the blast radius today but is not enforcement, so it does not
change the verdict.

The fix is either adding `attrs.output_mem_config` to the hash — cheap, since it is constant per
call site today and so costs no extra misses — or extending the existing `TT_FATAL` to require
`in_ss.grid == out_ss.grid && in_ss.orientation == out_ss.orientation`.

### 3. `input.tensor_spec().page_config()` — only `layout()` is hashed, and the factory hardcodes 32x32

**Verdict: BUG.**

`layout()` collapses `PageConfig` to `ROW_MAJOR` vs `TILE`, discarding the `Tile` shape. The
factory never reads the tensor's tile; it uses the architectural constants and the
format-derived tile size:

```122:129:ttnn/cpp/ttnn/operations/data_movement/roll/device/roll_program_factory.cpp
    // The gather works in "cells": a cell is one element for ROW_MAJOR, one tile for TILE.
    // Tile cells are contiguous in L1 and naturally aligned, so a tile-aligned roll is just a
    // permutation/rotation of whole tiles — identical to the row-major element gather.
    const bool is_tile = input.layout() == Layout::TILE;
    const tt::DataFormat cb_data_format = datatype_to_dataformat_converter(output.dtype());
    const uint32_t cell_h = is_tile ? tt::constants::TILE_HEIGHT : 1;
    const uint32_t cell_w = is_tile ? tt::constants::TILE_WIDTH : 1;
    const uint32_t cell_size = is_tile ? tt::tile_size(cb_data_format) : input.element_size();
```

All three conditions of the unguarded-tile bug pattern hold. The op accepts `Layout::TILE` (the
`is_tile` branch above). `cell_h` and `cell_w` are the bare architectural constants rather than
`tensor_spec().tile().get_tile_shape()`, and `tt::tile_size(format)` returns the byte size of a
32x32 tile rather than `tile.get_tile_size(format)`. And nothing validates the tile geometry:
`validate_roll` (`roll_device_operation.cpp:19-30`) checks storage, buffer, sharding on both
sides, and grid rectangularity, but makes no assertion about the tile.

The two defects compound. On its own, a hardcoded 32x32 factory fed a `Tile{16, 32}` tensor would
at least build a fresh (wrong) program. Because `page_config` is also absent from the key, the
non-32x32 tensor instead inherits the cache entry built for a 32x32 tensor of the same
`memory_config`, `dtype` and `layout` — `cell_h`, `cell_size`, the derived cell counts and every
per-core transfer descriptor are those of the 32x32 program. The symptom is wrong data or a hang
with no cache miss to point at the cause.

Note that this omission is independent of finding #1: even after the input shape is added to the
key, two tensors with the same shape and different tiles still collide, because `Tile` reaches the
program only through `page_config`.

The minimal fix is a `TT_FATAL` in `validate_roll` rejecting non-32x32 tiles on the `TILE` path,
which makes omitting `page_config` correct by construction. Making the factory genuinely
tile-aware instead would require adding `page_config` to the hash in the same change.

### 4. `input.tensor_spec().tensor_layout().get_alignment()`

**Verdict: VALID — unused.**

The factory never reads the tensor's `Alignment`. It recomputes the row pitch itself from the HAL
alignment appropriate to the backing memory:

```197:204:ttnn/cpp/ttnn/operations/data_movement/roll/device/roll_program_factory.cpp
    const uint32_t l1_alignment = tt::tt_metal::hal::get_l1_alignment();
    // Sharded buffers store one shard cell-row per page, padded up to the backing memory's
    // alignment. DRAM pages use the (larger) DRAM alignment — 64B on Blackhole, 32B on Wormhole —
    // whereas L1 pages use the L1 alignment. The row pitch must match whichever memory actually
    // holds the data, otherwise the staged copy and the host-computed offsets disagree.
    const uint32_t page_alignment = is_dram ? tt::tt_metal::hal::get_dram_alignment() : l1_alignment;
    const uint32_t row_pitch_bytes =
        ((shard_cells_w * cell_size + page_alignment - 1) / page_alignment) * page_alignment;
```

`is_dram` derives from `input.memory_config().buffer_type()` (hashed), and the HAL alignments are
device constants that the per-device cache already partitions on. For a sharded tensor the default
`Alignment` is itself a function of the shard spec (`Alignment{shard_spec.shape[1]}` for row-major,
`{tile_h, tile_w}` for tile), all of which live inside the hashed `memory_config` / `layout`.

### 5. `input.storage` variant kind (device vs host)

**Verdict: VALID — pinned by validation.**

```21:23:ttnn/cpp/ttnn/operations/data_movement/roll/device/roll_device_operation.cpp
    TT_FATAL(input.storage_type() == ttnn::StorageType::DEVICE, "Operands to roll need to be on device!");
    TT_FATAL(input.buffer() != nullptr, "Operands need to be allocated in buffers on device!");
    TT_FATAL(input.is_sharded(), "Native sharded roll requires a sharded input");
```

`validate_on_program_cache_hit` runs the same `validate_roll`
(`roll_device_operation.cpp:39-42`), so the constraint holds on hits as well as misses. The
parameter is constant across every admissible call and carries no information.

### 6. Buffer addresses (omitted by both the default hash and this one)

**Verdict: VALID — patched.**

Addresses must not be hashed. In L1 mode they ride on the CB `.buffer` bindings
(`roll_program_factory.cpp:493-513`); in the two DRAM modes the base+offset values are baked
directly into the runtime args:

```399:410:ttnn/cpp/ttnn/operations/data_movement/roll/device/roll_program_factory.cpp
    auto build_runtime_args_dram = [&](uint32_t dst_core_idx, const std::vector<RollTransferDesc>& descs) {
        KernelDescriptor::CoreRuntimeArgs args;
        args.reserve(3 + descs.size() * 7);
        args.push_back(dram_bank_id(dst_core_idx));
        // dst bank base = output buffer address + shard offset, from the current buffer.
        args.push_back(dram_bank_base(plan.output_buffer, dst_core_idx));
        args.push_back(static_cast<uint32_t>(descs.size()));
        for (const auto& td : descs) {
            // src_bank_id, src_bank_addr (= bank_base + intra_shard_offset), dst_offset,
            // copy_size, src_stride, dst_stride, num_rows
            args.push_back(dram_bank_id(td.src_dram_shard_idx));
            args.push_back(dram_bank_base(plan.input_buffer, td.src_dram_shard_idx) + td.src_l1_offset);
```

Both are re-applied by `apply_descriptor_runtime_args` on every hit — the runtime args by the
value copy, the CBs by `UpdateDynamicCircularBufferAddress`. This is exactly the case that
motivates mode A: a plain `Buffer*` binding cannot express `base + shard_offset`.

## Keys the custom hash adds beyond the default

None. Every value in the custom hash is a projection of something the default already covers; the
custom hash is a strict weakening.

## Framework side effect of having a custom hash

Defining `compute_program_hash` opts this op out of attribute-level hash-collision resolution:

```1012:1014:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
        if constexpr (requires { DeviceOperation::compute_program_hash(attrs, tensor_args); }) {
            return key;  // custom hash -> opt out beyond the op-identity prefix
        } else {
```

`ProgramCacheKey::canonical` degrades to just the op type name, so a 64-bit collision between two
different roll configurations resolves to a wrong hit instead of a rebuild. This is inherent to
every custom-hash op, but it raises the cost of the gap in omission #1: there is no second line of
defence behind the 64-bit key.

## Summary

| Omitted vs. default | Used by program? | Patched on hit? | Verdict |
|---|---|---|---|
| `input.logical_shape` / `padded_shape` | Yes — sets `rd[]`, `dim_size_cells`, `row_stride`, hence the transfer count | Values yes, **arg count no** | **BUG** |
| `attrs.output_mem_config` | Yes — shard extents, grid (core ranges), orientation | Shape re-asserted each hit; grid/orientation **not** | **BUG** |
| `input.page_config` (`Tile`) | Yes — `cell_h`/`cell_w`/`cell_size` via 32x32 constants, unguarded | **No** | **BUG** |
| `input.tensor_layout.alignment` | No | n/a | VALID — unused |
| `input.storage` kind | n/a | n/a | VALID — pinned by validation |
| Buffer addresses | Yes | Yes (mode A re-derivation) | VALID — patched |

**Three program-cache correctness bugs were found.** Dropping the input shape from the key is only safe
for a factory whose descriptor *shape* (arg counts, CB count, core ranges) is invariant under the
shape. `RollShardedProgramFactory` is not such a factory: the number of coalesced transfer runs —
and therefore the number of runtime args per core — is a function of the N-D decomposition of the
input, which the hashed `MemoryConfig` only constrains in aggregate (`H_cells`, `W_cells`). Two
`ttnn.roll` calls that differ only in how the same 2D shard geometry is spelled as an N-D shape
collide, and the cache-hit re-derivation then writes past the end of the cached program's
runtime-arg storage.

The second (omission #2) is `output_mem_config`, an operation attribute the default key would have
covered. Only its shard *shape* is enforced against the input's; its grid and orientation are free,
and both are structural — the grid becomes the kernel and CB core ranges, and the orientation selects
the shard-to-core mapping. It is reachable through `ttnn::prim::roll_sharded`, which takes the memory
config as a parameter and checks only that it is sharded. That the sole in-tree caller happens to
pass the input's own config bounds the severity but is not a constraint the op enforces, so it does
not soften the verdict.

The third (omission #3) is the unguarded 32x32 tile assumption: the factory derives its cell geometry
from `TILE_HEIGHT`/`TILE_WIDTH`/`tt::tile_size` while the hash carries only `layout()`, so a
non-32x32 tiled input reuses a program built for 32x32 cells.

All three are independent: fixing any one leaves the other two open.

## Recommendations

1. Add the input shape to `compute_program_hash`. `args.input.padded_shape()` is the minimum
   (that is what the factory actually reads); it also subsumes the `H_cells`/`W_cells`
   reasoning so the hash no longer depends on the shard spec implying the 2D extent. This is the
   fix for the first BUG.
2. Add `attrs.output_mem_config` to the hash. This is the fix for omission #2, and it is
   effectively free: the attribute is constant per call site today (always
   `input.memory_config()`), so it costs zero extra cache misses. Alternatively, or in addition,
   extend the `TT_FATAL` at `roll_program_factory.cpp:146-148` to require
   `in_ss.grid == out_ss.grid && in_ss.orientation == out_ss.orientation`, which would make the
   whole output shard spec redundant with the already-hashed input one.
3. Add a `TT_FATAL` in `validate_roll` rejecting a non-32x32 `Tile` on the `TILE` path. This is the
   fix for omission #3, and because `validate_roll` is also the cache-hit validator
   (`roll_device_operation.cpp:39-42`) it takes effect on hits as well as misses. The alternative —
   making the factory read `tensor_spec().tile().get_tile_shape()` — requires adding `page_config`
   to the hash in the same change.
4. Even after (1), consider making `override_runtime_arguments` assert
   `desc_args.size() == GetRuntimeArgs(program, k, core).rt_args_count` before copying. Mode A
   silently depends on the arg count being hash-invariant, and that invariant is currently
   implicit everywhere it is relied on.
5. Build the roll unit tests once under `-DTT_DESCRIPTOR_PATCHING_PARITY_CHECK`; the
   `assert_fastpath_parity` oracle (`mesh_device_operation_adapter.hpp:679-693`) is wired into the
   mode-A branch and would have caught the stale-arg half of the first BUG automatically.
