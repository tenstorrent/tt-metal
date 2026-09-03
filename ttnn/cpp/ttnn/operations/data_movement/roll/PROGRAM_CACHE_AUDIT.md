# Program Cache Audit — `data_movement/roll`

Audit of `ttnn::prim::RollDeviceOperation::compute_program_hash` against the framework
default ("hash everything") key.

| | |
|---|---|
| Device operation | `ttnn::prim::RollDeviceOperation` (`device/roll_device_operation.hpp`) |
| Custom hash | `device/roll_device_operation.cpp:57` |
| `operation_attributes_t` | `RollParams` — `shift`, `dim`, `output_mem_config` |
| `tensor_args_t` | `RollInputs` — `input` |
| Program factories | `RollShardedProgramFactory` (single-alternative variant, `ProgramDescriptor`-based) |
| `override_runtime_arguments` | **Yes** — on the program factory (`device/roll_program_factory.cpp:572`) |
| `get_dynamic_runtime_args` | **No** |
| Own cache-hit validator | **Yes** — `validate_on_program_cache_hit` (`device/roll_device_operation.cpp:52`), delegating to the same helper as the miss validator |
| Cache-hit patch mechanism | **Factory-owned cache-hit re-derivation** (mode A), planner re-run with an explicit arg-count check |

## Post-fix status — commit fab067a

**CLEAR — with a justified relaxation.** All three program-cache bugs this audit found are fixed.
`fab067a` added the two missing key terms (`attrs.output_mem_config` and `args.input.padded_shape()`)
and the 32x32 tile guard, and the arg-count hazard the first finding turned on is additionally
back-stopped by an explicit equality check in `override_runtime_arguments`. Every remaining omission
is pinned by a `TT_FATAL` that runs on the hit path or derived from a hashed term, except
`input.logical_shape()`, which is a genuine relaxation: the planner reads only `padded_shape`. Zero
remaining program-cache bugs. There is, separately, one **non-cache correctness defect** — the
gather extent is taken from `padded_shape` while `ttnn::roll` normalises the shift against
`logical_shape` — recorded under its own heading below; `padded_shape` *is* hashed, so it is a
factory defect, not a cache defect.

What `fab067a` changed in this op:

- Added `attrs.output_mem_config` and `args.input.padded_shape()` to `compute_program_hash`
  (`device/roll_device_operation.cpp:63-70`), with a comment naming why each is load-bearing. These
  are the fixes for omissions #1 and #2 (recommendations 1 and 2).
- Added a 32x32 tile guard to `validate_roll` on the `TILE` path
  (`device/roll_device_operation.cpp:32-42`). Because `validate_roll` is the body of *both*
  `validate_on_program_cache_miss` and `validate_on_program_cache_hit`
  (`device/roll_device_operation.cpp:47-55`), it is live on hits without relying on the framework's
  substitution branch. This is the fix for omission #3 (recommendation 3).
- Documented the constraint on the public API: `roll_nanobind.cpp:22` now reads "A TILE layout tensor
  must use the standard 32x32 tile."

One of this document's other recommendations was already implemented before `fab067a`, by another
commit, and the body below has been corrected accordingly:

- Recommendation 4 (assert the arg count before copying) landed in `85238dbd08b`, which replaced the
  blanket `apply_descriptor_runtime_args` re-application with an in-place patch that re-runs the
  planner and refuses to write on a length mismatch
  (`device/roll_program_factory.cpp:584-591`). `5804b6a0049` moved the hook from the device operation
  onto `RollShardedProgramFactory` in the same era. This document's description of the cache-hit
  patch mechanism was written against the pre-`85238dbd08b` code and has been rewritten below.

One further pre-`fab067a` commit changed this op without bearing on any finding here: `86bc0a6d672`
("Fix roll DRAM_RM 3-shard OOB") added `dram_rm_roll_needs_extra_source_shards`
(`device/roll_program_factory.cpp:607`) and the DRAM-RM filter `roll.cpp:85` applies before
dispatch.

What remains open:

- The `Tile` transpose flags are still unguarded, as recommendation 3's alternative noted. Inert
  under the 32x32 guard: `Tile`'s constructor derives every other field from `tile_shape`
  (`tt_metal/impl/data_format/tile.cpp:36-68`), `get_tile_size` ignores the flags (`:70-118`), and the
  planner reads neither.
- Recommendation 5 (a run under `-DTT_DESCRIPTOR_PATCHING_PARITY_CHECK`) has not been done. The
  oracle is still wired into the mode-A branch (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:688-702`).
- The non-cache defect under "Non-cache correctness defects" below.

**Metal 2.0 port: clear**, with one flag to declare. The logical-vs-padded relaxation in omission #1
is exactly the shape of `TensorSpecRelaxations::match_padded_shape_only`
(`tt_metal/api/tt-metalium/experimental/metal2_host_api/tensor_spec_relaxations.hpp:41`), which
`pertinent_fields` maps to `PertinentFields{.padded_shape = true}`
(`tt_metal/impl/metal2_host_api/tensor_spec_relaxations.cpp:77-79`). The input `TensorParameter` must
declare it, or a call differing only in logical shape will hit the key and then be *rejected* by
`ValidateTensorArgs` (`tt_metal/impl/metal2_host_api/program_run_args.cpp:176-189`) rather than
reusing the program. `hash_tensorspec_with_relaxation`
(`tt_metal/impl/metal2_host_api/tensor_spec_relaxations.cpp:116`) and
`tensorspecs_match_with_relaxation` (`:161-201`) derive their field set from the same
`pertinent_fields` call, so key and validation cannot disagree. The flag is *tighter* than roll's
current key — it pins the whole `tensor_layout()`, including the `page_config` and `Alignment` the
custom hash drops — so the port strengthens the key, and omissions #3 and #4 become moot by
construction. Note that for a sharded tensor the flag does **not** relax the distribution geometry,
which is what roll's whole plan is built on; that is the correct behaviour here.

## Cache-hit patch mechanism

`RollShardedProgramFactory` declares `override_runtime_arguments`, so the descriptor adapter takes
the factory-owned branch and never consults `resolve_bindings`:

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

**Correction to the original audit.** It described the hook as living on `RollDeviceOperation` and
replaying the whole descriptor through `apply_descriptor_runtime_args`. That was true of the tree the
audit was written against, but two commits reachable from `fab067a^` changed it: `5804b6a0049` moved
the hook onto the program factory, and `85238dbd08b` replaced the blanket descriptor replay with an
in-place patch that re-runs the *planner* — `create_descriptor`'s own source of truth — and writes
its args directly:

```572:591:ttnn/cpp/ttnn/operations/data_movement/roll/device/roll_program_factory.cpp
void RollShardedProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const RollParams& operation_attributes,
    const RollInputs& tensor_args,
    Tensor& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // Buffer addresses and bank ids move per dispatch, so re-run the planner -- create_descriptor's own
    // source of truth -- and write its args into the cached program instead of rebuilding it.
    const RollPlan plan = compute_roll_plan(operation_attributes, tensor_args, tensor_return_value);
    for (const auto& [core, args] : plan.per_core_args) {
        auto& a = tt::tt_metal::GetRuntimeArgs(program, 0, core);
        // The arg count encodes the transfer count, which padded_shape fixes and the hash keys on, so a
        // mismatch means the key stopped matching the program -- never silently write a prefix.
        TT_FATAL(
            a.size() == args.size(),
            "roll cache hit on core ({}, {}) expected {} runtime args, cached program has {}",
            core.x,
            core.y,
            args.size(),
            a.size());
```

**This is still not a blanket guarantee, but the arg-count hazard is now checked.** The
`TT_FATAL` above is exactly what recommendation 4 of this document asked for, and it landed in
`85238dbd08b` rather than in `fab067a`. It matters because the underlying value copy never resizes
anything — the CB-address tail of the same function still goes through
`apply_descriptor_runtime_args`, which writes into the storage the cached `Program` already owns:

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
   the cached program's per-core arg vector was sized on the first miss. Post-`fab067a` this
   obligation is discharged — see omission #1 — and violating it now fails loudly at the `TT_FATAL`
   instead of writing out of bounds.

Runtime-arg *values* and CB base addresses are safe — the function re-applies every per-core arg
above, and for L1 mode hands a CB-address-only descriptor to `apply_descriptor_runtime_args`
(`device/roll_program_factory.cpp:599-604`), which calls `UpdateDynamicCircularBufferAddress` for
every `desc.cbs[i].buffer` (`tt_metal/impl/program/program_descriptors.cpp:220-232`), covering the
L1-mode CB0/CB16 bindings.

## Which validator runs on a cache hit

Roll is the less common of the two cases, and it is the hazardous one in general: it **defines**
`validate_on_program_cache_hit`, so the dispatcher runs that and *not* the miss validator on hits.

```265:269:ttnn/api/ttnn/device_operation.hpp
    if constexpr (HasValidateOnProgramCacheHit<mesh_device_operation_t>) {
        mesh_device_operation_t::validate_on_program_cache_hit(operation_attributes, tensor_args);
    } else {
        mesh_device_operation_t::validate_on_program_cache_miss(operation_attributes, tensor_args);
    }
```

An op in this branch normally loses every check that lives only in its miss validator. Roll does
not, because both entry points delegate to the same helper and neither adds anything of its own:

```47:55:ttnn/cpp/ttnn/operations/data_movement/roll/device/roll_device_operation.cpp
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
path pins exactly what the miss path pins: the five `TT_FATAL`s at `roll_device_operation.cpp:23-31`
(device storage, non-null buffer, input sharded, output sharded, single-rectangle input grid) and,
since `fab067a`, the 32x32 tile check on the `TILE` path (`:32-42`). Any
verdict below that says "pinned by validation" is therefore legitimate on hits, and unusually it does
not depend on the framework's substitution branch at all. That property is what lets omission #3's
new verdict rest on the guard rather than on the framework.

**Nothing is dropped on the hit path, so there is no reachability analysis to do here.** The
hit-path filter — asking of each dropped check whether the value it constrains is itself in the cache
key, and so whether the check can be evaded by a call that hits — applies only to ops whose hit
validator pins strictly less than their miss validator. Roll's does not; the dropped set is empty. No
verdict in this document rests on the hit validator being narrower than the miss validator, and none
is affected by the filter.

The identity is worth preserving deliberately: if a future change adds a check to `validate_roll`'s
miss path only, or narrows the hit validator, the hit path silently loses it.

Note separately that roll gets a *second* round of checking on hits that most ops do not, because
its `override_runtime_arguments` re-runs the planner `compute_roll_plan`, and that function carries
its own `TT_FATAL`s (shard-shape equality at lines 148-150, divisibility at 151 and 160, the
single-rectangle output grid at 169-170). Verdicts below distinguish carefully between checks in
`validate_roll` and checks in `compute_roll_plan`, since only the latter see the *output* shard
spec. (The audit originally attributed these to `create_descriptor`; since `85238dbd08b` the shared
body is the `compute_roll_plan` helper that both `create_descriptor` and
`override_runtime_arguments` call, so the checks still run on both paths.)

## Baseline: what the default hash would cover

`hash_objects_with_default_seed(type_hash<RollDeviceOperation>, attrs, tensor_args)` walks
reflection, so the default key is:

| Source | Fields |
|---|---|
| `operation_attributes` | `shift`, `dim`, `output_mem_config` |
| `input.storage` | storage variant kind (`DeviceStorage` / `HostStorage`; both have empty attribute tuples) |
| `input.tensor_spec` | `logical_shape`, and `tensor_layout` = { `dtype`, `page_config`, `memory_config`, `alignment` } |

## What the custom hash covers

Post-`fab067a`, seven values:

```63:70:ttnn/cpp/ttnn/operations/data_movement/roll/device/roll_device_operation.cpp
    return tt::tt_metal::operation::hash_operation<RollDeviceOperation>(
        attrs.shift,
        attrs.dim,
        attrs.output_mem_config,
        args.input.memory_config(),
        args.input.dtype(),
        args.input.layout(),
        args.input.padded_shape());
```

Pre-fix it was five, and the two absentees were the subjects of findings #1 and #2:

```
    return tt::tt_metal::operation::hash_operation<RollDeviceOperation>(
        attrs.shift, attrs.dim, args.input.memory_config(), args.input.dtype(), args.input.layout());
```

`output_mem_config` — an operation attribute the default would have hashed — and **any** description
of the input tensor's shape were both missing. Both are now present.

## Omitted parameters

### 1. `input.logical_shape()` / `input.padded_shape()` — the tensor shape is not hashed at all

**Verdict: RESOLVED by fab067a** (was BUG). **`logical_shape` remains omitted as a relaxation.**

`fab067a` added `args.input.padded_shape()` to `compute_program_hash`
(`device/roll_device_operation.cpp:70`) — the minimum this document's recommendation 1 asked for, and
exactly the term the planner reads. With it, the N-D decomposition `rd[0..rank-2]`, hence
`dim_size_cells`, `row_stride[dim]`, the transfer coalescing and therefore the per-core runtime-arg
count are all pinned. The release-build out-of-bounds write in the reproduction below can no longer
occur, and `85238dbd08b`'s length `TT_FATAL`
(`device/roll_program_factory.cpp:584-591`) is a second line of defence if it ever regresses.

`logical_shape` itself is still not hashed, and that is now a deliberate relaxation rather than an
oversight — see the note at the end of this finding.

Pre-fix: the planner read the input's padded shape and decomposed it into per-dimension extents that
drove the entire gather plan, with nothing in the key describing that shape:

```118:122:ttnn/cpp/ttnn/operations/data_movement/roll/device/roll_program_factory.cpp
    const auto& shape = input.padded_shape();
    const uint32_t rank = shape.rank();
    const uint32_t shift = operation_attributes.shift;
    const int32_t dim = operation_attributes.dim;
    const bool is_last_dim = (static_cast<uint32_t>(dim) == rank - 1);
```

```153:159:ttnn/cpp/ttnn/operations/data_movement/roll/device/roll_program_factory.cpp
    // Cell-row dim sizes (dims 0..rank-2 collapsed): the height dim is measured in tile-rows.
    std::vector<uint32_t> rd(rank, 1);
    uint32_t H_cells = 1;
    for (uint32_t i = 0; i + 1 < rank; i++) {
        rd[i] = (i == rank - 2) ? shape[i] / cell_h : shape[i];
        H_cells *= rd[i];
    }
```

```223:232:ttnn/cpp/ttnn/operations/data_movement/roll/device/roll_program_factory.cpp
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

```345:350:ttnn/cpp/ttnn/operations/data_movement/roll/device/roll_program_factory.cpp
    const uint32_t args_per_transfer = is_dram_rm ? 7u : (is_dram ? 7u : 9u);
    const uint32_t args_overhead = is_dram_rm ? 8u : (is_dram ? 3u : 1u);
    TT_FATAL(
        args_overhead + max_num_transfers * args_per_transfer <= runtime_args_limit,
        "Native sharded roll: too many copy segments per core ({}). Reduce grid/shape.",
        max_num_transfers);
```

```383:398:ttnn/cpp/ttnn/operations/data_movement/roll/device/roll_program_factory.cpp
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

**Reproduction (pre-fix).** Take a `bfloat16`, `ROW_MAJOR`, HEIGHT_SHARDED-in-L1 tensor with
`shard_spec = {grid = CoreRange((0,0),(1,0)), shape = [32, 64], ROW_MAJOR}`. Both calls use
`ttnn.roll(t, shifts=[1], dim=[2])`, which reaches
`ttnn::prim::roll_sharded(result, /*shift=*/1, /*dim=*/2, input.memory_config())`
(`roll.cpp:106-107`) with the shift already normalised to 1 in both cases.

- **Call 1** — input logical/padded shape `[1, 2, 32, 64]`. `rd = {1, 2, 32}`,
  `dim_size_cells = 32`, `row_stride[2] = 1`. The row permutation is `src = r - 1` with one wrap
  per 32-row block, so each destination core coalesces into **2** transfers →
  `1 + 2*9 = 19` runtime args per core.
- **Call 2** — input logical/padded shape `[1, 4, 16, 64]`. Same `H_cells = 64`, same
  `W_cells = 64`, same `MemoryConfig`, same dtype, same layout, same `shift`, same `dim` →
  **identical program hash**. But `rd = {1, 4, 16}` so `dim_size_cells = 16`, giving a wrap every
  16 rows; each destination core now needs **4** transfers → `1 + 4*9 = 37` runtime args per core.

On the cache hit, `override_runtime_arguments` built the 37-arg vector and wrote
`prog_args[0..36]` into a `RuntimeArgsData` whose
`rt_args_count` was 19. In a debug build this was a `TT_FATAL` from `in_bounds`; in a release build
it was a silent out-of-bounds write past the end of the kernel's runtime-arg region in the cached
program, corrupting whatever follows it, while the reader kernel still executed with the 19 slots
it could address and therefore performed the wrong gather.

The reverse order (call 2 first) was "merely" wrong rather than corrupting: 19 args written
into 37 slots, arg 0 (the transfer count) correct, the stale tail ignored — but that was
luck, not design.

Neither `validate_on_program_cache_hit` nor the `TT_FATAL`s inside `compute_roll_plan`
caught this. The validator only checks storage/sharding/grid-cardinality
(`roll_device_operation.cpp:23-31`), and the plan's own assertions
(`roll_program_factory.cpp:148-160`) check divisibility of `H_cells`/`W_cells` by the shard
extents, which both shapes satisfy.

**Post-fix.** The two calls now differ in `args.input.padded_shape()` (`[1,2,32,64]` vs
`[1,4,16,64]`), so call 2 misses and compiles its own program. The arg-count invariant mode A depends
on is no longer implicit either — `override_runtime_arguments` asserts it
(`roll_program_factory.cpp:584-591`).

**What is still omitted, and why it is a relaxation.** `input.logical_shape()` is absent from the
key. For a sharded tensor the default `Alignment` is `{shard_spec.shape[1]}` for ROW_MAJOR and
`{tile_h, tile_w}` for TILE (`tt_metal/impl/tensor/spec/layout/page_config.cpp:43-57`), so two calls
can share a padded shape while differing logically — logical `[1,1,64,60]` and `[1,1,64,64]` under a
shard width of 64 both pad to `[1,1,64,64]`. The planner reads only `padded_shape`, the shard
geometry, `shift` and `dim`, so the transfer plan, arg counts, CB sizes and core ranges are
bit-identical, and `compute_output_specs` rebuilds the output spec from `input.logical_shape()` on
every dispatch (`roll_device_operation.cpp:73-81`), so per-call output metadata stays correct. The
omission therefore buys a real hit the default key would miss. The op does not **need** it: no
in-tree caller varies the logical shape under a fixed padded shape, so hashing `logical_shape()` too
would cost zero real misses. Keeping it is what obliges the Metal 2.0 port to declare
`match_padded_shape_only` — see the post-fix status section. Note also that this same
padded-vs-logical split is the root of the non-cache defect recorded at the end of this document.

### 2. `operation_attributes.output_mem_config`

**Verdict: RESOLVED by fab067a** (was BUG).

`fab067a` added `attrs.output_mem_config` to `compute_program_hash`
(`device/roll_device_operation.cpp:66`), which is the first of the two routes this document's
recommendation 2 offered — the cheap one, since the attribute is constant per call site today and so
costs no extra misses. The output shard grid and orientation that drove the reproduction below are
now part of the key, so the two calls no longer collide. The alternative route (extending the
shard-shape `TT_FATAL` to require `in_ss.grid == out_ss.grid && in_ss.orientation ==
out_ss.orientation`) was not taken and is no longer needed.

Pre-fix: this was an operation attribute the default hash would have covered. The planner reads the
*output* shard spec for the shard extents, the grid and the orientation:

```142:160:ttnn/cpp/ttnn/operations/data_movement/roll/device/roll_program_factory.cpp
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

```355:364:ttnn/cpp/ttnn/operations/data_movement/roll/device/roll_program_factory.cpp
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

Exactly one component of the output shard spec was enforced. The shard **shape** is pinned to the
input's by the `TT_FATAL` at lines 148-150, and because `override_runtime_arguments` re-runs
`compute_roll_plan` that assertion is re-evaluated on every cache hit, not just on the miss; the
input shard shape is hashed inside `input.memory_config()`. So the shape carries no information.

**Grid and orientation are not pinned by anything.** The only assertion touching the grid requires
that it be a single rectangle:

```168:176:ttnn/cpp/ttnn/operations/data_movement/roll/device/roll_program_factory.cpp
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

```187:197:ttnn/cpp/ttnn/operations/data_movement/roll/device/roll_program_factory.cpp
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
(`roll_program_factory.cpp:503-568`), which are structural and never refreshed on a hit.

**Reachability.** `ttnn::prim::roll_sharded` is a public C++ entry point declared in
`roll_device_operation.hpp:40` and takes `output_mem_config` as a caller-supplied parameter. The
only enforced constraint on it is `output_mem_config.is_sharded()`
(`roll_device_operation.cpp:26`) plus the shard-shape equality above. Neither the grid extent nor
the orientation is checked against the input's, so the bad configuration is reachable without
violating any enforced constraint.

**Reproduction (pre-fix).** Take a `[1, 1, 64, 64]` TILE `bfloat16` tensor block-sharded on a 2x2 grid with
shard shape `[32, 32]` and `ShardOrientation::ROW_MAJOR`.

- **Call 1**: `ttnn::prim::roll_sharded(input, /*shift=*/1, /*dim=*/2, mc_row_major)` where
  `mc_row_major` is the input's own memory config.
- **Call 2**: the same input and the same shift and dim, but `output_mem_config` is the same
  memory config with `ShardOrientation::COL_MAJOR` and an otherwise identical 2x2 grid and
  `[32, 32]` shard shape.

Pre-fix, `attrs.shift`, `attrs.dim`, `input.memory_config()`, `input.dtype()` and `input.layout()`
were all the key had, so the two calls produced the same key and call 2 hit. The shard-shape `TT_FATAL` passes
(the shapes are equal), and the single-rectangle check passes. But `row_major_orient` is now false,
so `row_major_orient` became false and `shard_linear` computed `sc * grid_cols + sr` instead of
`sr * n_shard_cols + sc` — cells routed to transposed cores. Because the cached program's
`core_ranges` and compile-time args were those of call 1 and only the runtime args are re-applied,
the roll gathered from the wrong cores and the output shards came out permuted. For the two
off-diagonal shards of a 2x2 grid this silently swapped half the tensor.

The severity was limited by the call graph rather than by any check: the sole in-tree
caller passes the input's own memory config (`roll.cpp:106-107`, with
`native_mem_config = input_tensor.memory_config()` at `roll.cpp:60`), and
`ttnn::prim::roll_sharded` is not bound in `roll_nanobind.cpp`, which exposes only the three
`ttnn::roll` overloads. That bounded the blast radius but was not enforcement, so it did not
change the verdict. Post-`fab067a` the two calls differ in the key and call 2 misses.

### 3. `input.tensor_spec().page_config()` — only `layout()` is hashed, and the factory hardcodes 32x32

**Verdict: RESOLVED by fab067a** (was BUG).

`fab067a` added a `TT_FATAL` rejecting any tile other than 32x32 to `validate_roll`'s `TILE` branch
(`device/roll_device_operation.cpp:32-42`) — recommendation 3's primary route. Because `validate_roll`
*is* the body of `validate_on_program_cache_hit` as well as of the miss validator
(`device/roll_device_operation.cpp:52-55`), the guard executes on the offending dispatch, which is
the property this finding turned on. `Tile` can no longer vary, so omitting `page_config` from the
key is now correct by construction and the omission is a category-3 zero-value omission.

Pre-fix: `layout()` collapses `PageConfig` to `ROW_MAJOR` vs `TILE`, discarding the `Tile` shape. The
planner never reads the tensor's tile; it uses the architectural constants and the
format-derived tile size:

```124:131:ttnn/cpp/ttnn/operations/data_movement/roll/device/roll_program_factory.cpp
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
`validate_roll` (`roll_device_operation.cpp:23-31`) checks storage, buffer, sharding on both
sides, and grid rectangularity, but makes no assertion about the tile.

The two defects compounded. On its own, a hardcoded 32x32 factory fed a `Tile{16, 32}` tensor would
at least build a fresh (wrong) program. Because `page_config` was also absent from the key, the
non-32x32 tensor instead inherited the cache entry built for a 32x32 tensor of the same
`memory_config`, `dtype` and `layout` — `cell_h`, `cell_size`, the derived cell counts and every
per-core transfer descriptor were those of the 32x32 program. The symptom was wrong data or a hang
with no cache miss to point at the cause.

Note that this omission was independent of finding #1: adding the input shape to the key would not
have helped, because `Tile` reaches the program only through `page_config`. That is why `fab067a`
needed both changes.

The minimal fix was a `TT_FATAL` in `validate_roll` rejecting non-32x32 tiles on the `TILE` path,
which makes omitting `page_config` correct by construction, and that is what shipped. The guard is
total, not partial: `Tile`'s only public constructor derives `face_shape`, `num_faces`, `tile_hw`,
`face_hw`, `partial_face` and `narrow_tile` from `tile_shape`
(`tt_metal/impl/data_format/tile.cpp:36-68`), so pinning 32x32 pins everything `tt::tile_size` and
the cell geometry could read. It does not cover the tile's transpose flags, which reach neither the
hash nor the canonical key framework-wide — but `Tile::get_tile_size` ignores them
(`tt_metal/impl/data_format/tile.cpp:70-118`) and the planner never reads them, so under the guard
that gap is inert. Making the planner genuinely tile-aware instead would require adding
`page_config` to the hash in the same change.

### 4. `input.tensor_spec().tensor_layout().get_alignment()`

**Verdict: VALID — unused.**

The factory never reads the tensor's `Alignment`. It recomputes the row pitch itself from the HAL
alignment appropriate to the backing memory:

```199:206:ttnn/cpp/ttnn/operations/data_movement/roll/device/roll_program_factory.cpp
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

```23:25:ttnn/cpp/ttnn/operations/data_movement/roll/device/roll_device_operation.cpp
    TT_FATAL(input.storage_type() == ttnn::StorageType::DEVICE, "Operands to roll need to be on device!");
    TT_FATAL(input.buffer() != nullptr, "Operands need to be allocated in buffers on device!");
    TT_FATAL(input.is_sharded(), "Native sharded roll requires a sharded input");
```

`validate_on_program_cache_hit` runs the same `validate_roll`
(`roll_device_operation.cpp:52-55`), so the constraint holds on hits as well as misses. The
parameter is constant across every admissible call and carries no information.

### 6. Buffer addresses (omitted by both the default hash and this one)

**Verdict: VALID — patched.**

Addresses must not be hashed. In L1 mode they ride on the CB `.buffer` bindings
(`roll_program_factory.cpp:503-523`); in the two DRAM modes the base+offset values are baked
directly into the runtime args:

```401:412:ttnn/cpp/ttnn/operations/data_movement/roll/device/roll_program_factory.cpp
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

Both are re-applied on every hit — the runtime args by `override_runtime_arguments`' own value copy
(`roll_program_factory.cpp:592-594`), the CBs by the `UpdateDynamicCircularBufferAddress` tail of
`apply_descriptor_runtime_args`, which it feeds a CB-address-only descriptor
(`roll_program_factory.cpp:599-604`). This is exactly the case that motivates mode A: a plain
`Buffer*` binding cannot express `base + shard_offset`.

## Keys the custom hash adds beyond the default

None. Every value in the custom hash is a projection of something the default already covers; the
custom hash is a strict weakening.

## Framework side effect of having a custom hash

Defining `compute_program_hash` opts this op out of attribute-level hash-collision resolution:

```1035:1037:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
        if constexpr (requires { DeviceOperation::compute_program_hash(attrs, tensor_args); }) {
            return key;  // custom hash -> opt out beyond the op-identity prefix
        } else {
```

`ProgramCacheKey::canonical` degrades to just the op type name, so a 64-bit collision between two
different roll configurations resolves to a wrong hit instead of a rebuild. This is inherent to
every custom-hash op. Post-`fab067a` it is the main residual argument for narrowing the relaxation in
omission #1: with `logical_shape` hashed as well, roll's key would be a strict superset of the
default's on every term it keeps, and the op could drop `compute_program_hash` altogether and regain
the tiebreaker.

## Non-cache correctness defects

These are **not** program-cache bugs. The parameter involved is hashed, so the cache partitions
correctly; the factory simply computes the wrong thing with it. Recorded here because the audit
surfaced them and because the fix does not belong in `compute_program_hash`.

### N1. The gather extent comes from `padded_shape` while `ttnn::roll` normalises the shift against `logical_shape`

**Verdict: FACTORY DEFECT** (not a cache bug — `padded_shape` is hashed as of `fab067a`).

`ttnn::roll` reduces the user's shift modulo the **logical** dimension size before handing it to the
primitive:

```44:50:ttnn/cpp/ttnn/operations/data_movement/roll/roll.cpp
    for (size_t i = 0; i < adjusted_shifts.size(); ++i) {
        int shift = adjusted_shifts[i];
        int dim = input_dims[i];

        int shift_size = input_tensor.logical_shape()[dim];
        adjusted_shifts[i] = ((shift % shift_size) + shift_size) % shift_size;
    }
```

The planner then performs the rotation over the **padded** extent:

```223:232:ttnn/cpp/ttnn/operations/data_movement/roll/device/roll_program_factory.cpp
    // Coordinate shift for the rolled dim, in cell-row units (tile-rows for the height dim).
    const uint32_t dim_size_cells = (static_cast<uint32_t>(dim) == rank - 2) ? rd[dim] : shape[dim];
    const uint32_t shift_cells = (static_cast<uint32_t>(dim) == rank - 2) ? shift / cell_h : shift;
```

with `shape = input.padded_shape()` (`roll_program_factory.cpp:118`). When the rolled dimension is
padded, the two disagree in two ways at once. The modulus is wrong — a shift of `logical_size` should
be the identity but is reduced to 0 and then applied to a *larger* extent, so it rotates by 0 where a
padded-extent roll would rotate by `logical_size mod padded_size` — and, more importantly, the
rotation drags the padding through the logical region: elements that should come from the tail of the
logical data are read from padding instead.

**Reachability.** Requires a logical extent along `dim` that is not a multiple of the alignment in
that dimension. On the ROW_MAJOR sharded path the default `Alignment` touches only the last dimension
(`Alignment{shard_spec.shape[1]}`, `tt_metal/impl/tensor/spec/layout/page_config.cpp:43-57`), so
rolling any earlier dimension is exact. On the `TILE` path the default alignment is `{32, 32}`, so any
logical height or width that is not a multiple of 32 exposes it — e.g. a `[1, 1, 40, 64]` tiled tensor
padded to `[1, 1, 64, 64]` and rolled along `dim=2`. The tile path measures the shift in tile-rows
(`shift / cell_h`), which additionally truncates any shift that is not a multiple of 32, so a
sub-tile shift on that path is silently rounded down.

**Why this is not a cache defect.** `args.input.padded_shape()` is in the key
(`roll_device_operation.cpp:70`) and the logical shape reaches the primitive only through the
already-reduced `shift`, which is also hashed. Two calls that would want different plans therefore
get different keys; each call gets a program that faithfully implements what the planner was asked
for. The defect is in *what* it was asked for.

**Fix.** Either pass the logical extent into the planner and have it roll only the logical region
(the correct behaviour, and what the tile path would need to stop truncating sub-tile shifts), or add
a `TT_FATAL` to `validate_roll` requiring `logical_shape()[dim] == padded_shape()[dim]` for the
rolled dimension. The latter is the minimal safe change and, being in `validate_roll`, would run on
both cache paths. Note that constraining the shape this way would also make the omission-#1
relaxation vacuous along `dim`.

## Summary

| Omitted vs. default | Used by program? | Patched on hit? | Verdict |
|---|---|---|---|
| `input.padded_shape` | Yes — sets `rd[]`, `dim_size_cells`, `row_stride`, hence the transfer count | Values yes, arg count no | **RESOLVED by fab067a** — now hashed |
| `input.logical_shape` | No (padded shape used instead) | n/a | VALID — relaxation win |
| `attrs.output_mem_config` | Yes — shard extents, grid (core ranges), orientation | Shape re-asserted each hit; grid/orientation not | **RESOLVED by fab067a** — now hashed |
| `input.page_config` (`Tile`) | Yes — `cell_h`/`cell_w`/`cell_size` via 32x32 constants | No | **RESOLVED by fab067a** — pinned by the 32x32 guard, live on hits |
| `input.tensor_layout.alignment` | No | n/a | VALID — unused |
| `input.storage` kind | n/a | n/a | VALID — pinned by validation |
| Buffer addresses | Yes | Yes (mode A re-derivation) | VALID — patched |

**Zero program-cache correctness bugs remain.** All three this audit found are closed, and each
mattered for a different reason.

The first was dropping the input shape from the key. That is only safe
for a factory whose descriptor *shape* (arg counts, CB count, core ranges) is invariant under the
shape, and `RollShardedProgramFactory` is not such a factory: the number of coalesced transfer runs —
and therefore the number of runtime args per core — is a function of the N-D decomposition of the
input, which the hashed `MemoryConfig` only constrains in aggregate (`H_cells`, `W_cells`). Two
`ttnn.roll` calls that differed only in how the same 2D shard geometry is spelled as an N-D shape
collided, and the cache-hit re-derivation then wrote past the end of the cached program's
runtime-arg storage. `fab067a` added `args.input.padded_shape()`; `85238dbd08b` had already added a
length `TT_FATAL` that turns any regression into a loud failure rather than a silent one.

The second (omission #2) was `output_mem_config`, an operation attribute the default key would have
covered. Only its shard *shape* was enforced against the input's; its grid and orientation were free,
and both are structural — the grid becomes the kernel and CB core ranges, and the orientation selects
the shard-to-core mapping. `fab067a` added the attribute to the key.

The third (omission #3) was the unguarded 32x32 tile assumption: the planner derives its cell geometry
from `TILE_HEIGHT`/`TILE_WIDTH`/`tt::tile_size` while the hash carries only `layout()`, so a
non-32x32 tiled input reused a program built for 32x32 cells. `fab067a` pinned the tile in
`validate_roll`, which is the body of both validators.

One omission remains a genuine relaxation rather than a zero-value drop: `input.logical_shape`
(omission #1). It is sound — the planner reads only `padded_shape` — and it buys real hits, but the op
does not need it, and giving it up would let roll drop `compute_program_hash` entirely and regain the
canonical-key tiebreaker. Separately, one non-cache factory defect is open: the gather extent is the
padded extent while the shift is normalised against the logical one (finding N1).

## Recommendations

1. **Fixed by `fab067a`.** ~~Add~~ Added the input shape to `compute_program_hash`:
   `args.input.padded_shape()` (`roll_device_operation.cpp:70`), which is what the planner actually
   reads; it also subsumes the `H_cells`/`W_cells`
   reasoning so the hash no longer depends on the shard spec implying the 2D extent.
2. **Fixed by `fab067a`.** ~~Add~~ Added `attrs.output_mem_config` to the hash
   (`roll_device_operation.cpp:66`), the cheap route: the attribute is constant per call site today
   (always `input.memory_config()`), so it costs zero extra cache misses. The alternative —
   extending the `TT_FATAL` at `roll_program_factory.cpp:148-150` to require
   `in_ss.grid == out_ss.grid && in_ss.orientation == out_ss.orientation` — was not taken and is no
   longer needed.
3. **Fixed by `fab067a`.** ~~Add~~ Added a `TT_FATAL` in `validate_roll` rejecting a non-32x32 `Tile`
   on the `TILE` path (`roll_device_operation.cpp:32-42`), and because `validate_roll` is also the
   cache-hit validator (`roll_device_operation.cpp:52-55`) it takes effect on hits as well as misses.
   The alternative — making the planner read `tensor_spec().tile().get_tile_shape()` — would have
   required adding `page_config` to the hash in the same change.
4. **Fixed earlier, by `85238dbd08b`, not by `fab067a`.** `override_runtime_arguments` now asserts
   `a.size() == args.size()` before copying (`roll_program_factory.cpp:584-591`). Mode A silently
   depended on the arg count being hash-invariant; that invariant is now explicit at the one place
   roll relies on it.
5. **Still open.** Build the roll unit tests once under `-DTT_DESCRIPTOR_PATCHING_PARITY_CHECK`; the
   `assert_fastpath_parity` oracle (`mesh_device_operation_adapter.hpp:688-702`) is wired into the
   mode-A branch and would have caught the stale-arg half of the first BUG automatically.
6. **New.** Fix or guard the padded-vs-logical gather extent (finding N1) — either roll only the
   logical region, or `TT_FATAL` on `logical_shape()[dim] != padded_shape()[dim]` in `validate_roll`.
7. **New.** Before the Metal 2.0 port, either add `args.input.logical_shape()` to the key — which
   also lets the op drop its custom hash and recover the canonical-key tiebreaker — or declare
   `match_padded_shape_only` on the input `TensorParameter`. Doing neither leaves a key that hits
   where `ValidateTensorArgs` then rejects.
