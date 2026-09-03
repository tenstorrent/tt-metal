# Program Cache Audit — `data_movement/slice`

Audit of `ttnn::prim::SliceDeviceOperation::compute_program_hash` against the framework default
("hash everything") key.

| | |
|---|---|
| Device operation | `ttnn::prim::SliceDeviceOperation` (`device/slice_device_operation.hpp:31`) |
| Custom hash | `device/slice_device_operation.cpp:343-426` |
| `operation_attributes_t` | `SliceParams` — `slice_start`, `slice_end`, `step`, `output_mem_config`, `use_tensor_args`, `slice_dim`, `num_devices`, `sub_core_grids` |
| `tensor_args_t` | `SliceInputs` — `input`, `start_tensor`, `end_tensor`, `preallocated_output` |
| Program factories | five: `SliceRmProgramFactory`, `SliceRmShardedProgramFactory`, `SliceRmStrideProgramFactory`, `SliceTileProgramFactory`, `SliceTileTensorArgsProgramFactory` |
| `override_runtime_arguments` | **Yes** — one per factory, all five delegating to the shared `patch_slice_program_addresses` (`device/slice_program_factory_rm_sharded.cpp:354`) |
| `get_dynamic_runtime_args` | **No** — the member does not exist (see CSV note) |
| Own cache-hit validator | No — the framework substitutes `validate_on_program_cache_miss` |
| Cache-hit patch mechanism | **Factory-owned cache-hit re-derivation** (mode A), narrow address+scalar patch for four factories and a CB-address-only patch for the fifth |

## Post-fix status — commit fab067a

**NOT CLEAR.** Two of the three program-cache bugs this audit found are fully fixed, and the third
(`preallocated_output`) is fixed on every path the op is normally used on — but one residual hole
survives. `fab067a` hashes the preallocated output's `logical_shape`, `padded_shape`, `layout`,
`dtype` and `memory_config`, and pins its layout, dtype and tile against the input's by `TT_FATAL` —
but those `TT_FATAL`s landed *inside* the `if (input.layout() == Layout::TILE)` branch of the
validator, so on a **ROW_MAJOR input** the preallocated output's `page_config()` is neither hashed nor
guarded, and it reaches the writer's compile-time args as the destination buffer's aligned page size.
That is finding #7 below. Every other omission is pinned by a `TT_FATAL` that runs on the hit path or
is functionally determined by a hashed term; the op has no logical-vs-padded relaxation at all,
because it hashes both shapes for every tensor it hashes.

What `fab067a` changed in this op:

- Added `tensor_args.end_tensor.has_value()` to the first `hash_operation` call
  (`device/slice_device_operation.cpp:361`) and a full end-tensor spec block mirroring the
  start-tensor one (`:386-398`), with a comment naming the reader's `TensorAccessorArgs` as the
  reason. This is the fix for finding #1, exactly as recommendation 1 specified.
- Added `tensor_args.preallocated_output.has_value()` (`:362`) and a full preallocated-output spec
  block (`:410-424`). This is the fix for finding #3, taking the *hashing* route rather than the
  spec-comparison route recommendation 4 preferred.
- Added a 32x32 tile guard on the `TILE` input path (`:166-174`). Since the op has no hit validator,
  the dispatcher substitutes the miss validator on hits
  (`ttnn/api/ttnn/device_operation.hpp:265-269`), so the guard is live on the offending call. This is
  the minimal half of recommendation 2; the tile-aware alternative was not taken.
- Added, in the same `TILE` branch, three `TT_FATAL`s pinning a preallocated output's `layout`,
  `dtype` and `Tile` to the input's (`:175-198`), with a comment explaining why layout and dtype must
  be checked before the tile (`PageConfig::get_tile()` reports a default 32x32 tile for a ROW_MAJOR
  tensor).
- Added a `TT_FATAL` refusing `use_tensor_args` on a non-TILE input (`:111-114`), closing a path
  `ttnn::prim::slice` could reach directly even though the public `slice()` wrapper already refused it.
- Documented the constraints on the public API: `slice_nanobind.cpp:68` now states that TILE tensors
  must use the standard 32x32 tile and that a preallocated output must match the input's layout,
  dtype and tile.

One of this document's recommendations was implemented before `fab067a`, by another commit, and two
more pre-`fab067a` commits invalidated its description of the patch mechanism:

- Recommendation 3 (delete the dead `slice_rm_reader_dynamic_args`) landed in `1b0d9d1258a`
  ("Prepare Slice for Metal 2.0 Port"), which
  removed the function, its declaration and the three misleading comments. The body below has been
  corrected accordingly.
- `5804b6a0049` moved `override_runtime_arguments` from `SliceDeviceOperation` onto each of the five
  program factories, and `85238dbd08b` replaced the full descriptor rebuild with the shared
  address-and-scalar `patch_slice_program_addresses` helper. The "Cache-hit patch mechanism" section
  below describes the pre-`85238dbd08b` code in the original audit and has been rewritten.

What remains open:

- **Finding #7** — the ROW_MAJOR-path `preallocated_output.page_config()` hole described above.
- Recommendation 2's tile-aware alternative. `SliceTileTensorArgsProgramFactory` still reads
  `tensor_spec().tile().get_tile_shape()` (`device/slice_program_factory_tile_tensor_args.cpp:75-77`)
  while `SliceTileProgramFactory` hardcodes the constants; the guard makes both correct but leaves
  the two idioms inconsistent.
- The `Tile` transpose flags are unguarded, as recommendation 2 anticipated. Inert under the 32x32
  guard: `Tile`'s constructor derives everything else from `tile_shape`
  (`tt_metal/impl/data_format/tile.cpp:36-68`) and `get_tile_size` ignores the flags (`:70-118`).
- Recommendation 5 (a run under `-DTT_DESCRIPTOR_PATCHING_PARITY_CHECK`) has not been done.

**Metal 2.0 port: clear once finding #7 is closed, and it needs no relaxation flags.** Slice hashes
`logical_shape` *and* `padded_shape` for every tensor it hashes (`:367-372`, `:378-383`, `:392-397`,
`:403-408`, `:418-423`), so it carries no logical-vs-padded relaxation and does not need
`match_padded_shape_only` (`tt_metal/api/tt-metalium/experimental/metal2_host_api/tensor_spec_relaxations.hpp:41`).
The default `PertinentFields{.whole_spec = true}` path of `pertinent_fields`
(`tt_metal/impl/metal2_host_api/tensor_spec_relaxations.cpp:85-86`) is what it wants, and that path
is strictly *stronger* than the current key — it compares whole `TensorSpec`s, which includes the
`page_config` finding #7 turns on, so declaring no relaxation would close finding #7 as a
side effect. Because `hash_tensorspec_with_relaxation`
(`tt_metal/impl/metal2_host_api/tensor_spec_relaxations.cpp:116`) and
`tensorspecs_match_with_relaxation` (`:161-201`) both derive their field set from
`pertinent_fields`, and `ValidateTensorArgs` delegates to the latter
(`tt_metal/impl/metal2_host_api/program_run_args.cpp:176-189`), key and validation cannot disagree.
Fix finding #7 in the current tree regardless, rather than waiting for the port.

## Cache-hit patch mechanism

**Correction to the original audit.** This section originally described `override_runtime_arguments`
as living on `SliceDeviceOperation` and performing a *full descriptor rebuild* for four of the five
factories. That was true of the tree the audit was written against; `5804b6a0049` and `85238dbd08b`,
both reachable from `fab067a^`, changed each half respectively.
Each of the five factories now declares its own hook — for example
`SliceRmShardedProgramFactory::override_runtime_arguments`
(`device/slice_program_factory_rm_sharded.cpp:415-422`), and likewise
`slice_program_factory_rm.cpp:396`, `slice_program_factory_rm_stride.cpp:178`,
`slice_program_factory_tile.cpp:189`, `slice_program_factory_tile_tensor_args.cpp:195` — and all five
delegate to a single shared helper that patches *only addresses*, never rebuilding the descriptor.
The conclusions of this document are unaffected, because a narrower patch imposes a strictly
*larger* obligation on the hash, and the findings below are all about the hash. But the description
mattered enough to fix.

Since each factory declares the hook, the adapter takes the factory-owned branch and bypasses
`resolve_bindings` and `get_dynamic_runtime_args`:

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

The shared helper is an address-and-scalar patch, factory-dispatched:

```354:381:ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_program_factory_rm_sharded.cpp
void patch_slice_program_addresses(
    tt::tt_metal::Program& program,
    const SliceDeviceOperation::program_factory_t& factory,
    const SliceParams& operation_attributes,
    const SliceInputs& tensor_args,
    Tensor& output) {
    // Height-sharded RM is CB-bound: the reader args are all keyed, so only the two sharded CB
    // addresses move. CBs are matched positionally -- src0, then c_16.
    if (std::holds_alternative<SliceRmShardedProgramFactory>(factory)) {
        tt::tt_metal::ProgramDescriptor cb_addr_only;
        cb_addr_only.cbs.push_back(tt::tt_metal::CBDescriptor{.buffer = tensor_args.input.buffer()});
        cb_addr_only.cbs.push_back(tt::tt_metal::CBDescriptor{.buffer = output.buffer()});
        tt::tt_metal::apply_descriptor_runtime_args(program, cb_addr_only);
        return;
    }

    // A slot holding 0 belongs to a core create_descriptor left zero-filled; leave those alone.
    constexpr uint32_t kReaderKernelIdx = 0, kWriterKernelIdx = 1;
    const auto patch_slot0 = [&program](uint32_t kernel_idx, uint32_t addr) {
        for (auto& col : tt::tt_metal::GetRuntimeArgs(program, kernel_idx)) {
            for (auto& a : col) {
                if (a.size() > 0 && a[0] != 0) {
                    a[0] = addr;
                }
            }
        }
    };
    patch_slot0(kWriterKernelIdx, output.buffer()->address());
```

The two RM factories then patch reader slot 0 with the input address; the two tile factories emit
`DynamicRuntimeArg`s for the source (and, on the tensor-args path, the start and end tensor)
addresses plus a per-core scalar refresh through `slice_tile_dynamic_args`
(`slice_program_factory_rm_sharded.cpp:383-412`). The per-core scalar refresh exists because a
divergent-partition hit would otherwise leave the writer's `num_pages` at 0 and produce an all-zero
output (issue #52651). The scalar patch itself still routes through the descriptor machinery, whose
value copy writes into storage the cached `Program` already owns and refreshes **common** runtime
args as well as per-core ones:

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
logical-to-physical core mapping (line 159) — the first four are all hashed (input `padded_shape`
and `memory_config`, `slice_start`, and the output spec's `memory_config` respectively), and the
last is fixed for a given device. So the two CB addresses really are the only per-dispatch values.
The helper's own comment (`slice_program_factory_rm_sharded.cpp:350-352`) states this contract for
all five factories: "Every shape-derived arg is keyed … so addresses are all that move on a hit."

**The obligation this leaves on the hash** is therefore broad and absolute, and broader than the
original audit's version of this paragraph implied. Mode A never refreshes compile-time args, kernel
sources, CB sizes/formats, semaphores or core ranges, so everything feeding those must be hashed —
and because the patch is now address-and-scalar rather than a rebuild, every *other* shape-derived
runtime arg must be hashed too. That is precisely the contract the helper's comment asserts, and
every finding below is an instance of it.

**CSV correction.** The CSV records `get_dynamic_runtime_args = Y`. The op has no such member
(`slice_device_operation.hpp:43-56` declares `select_program_factory`,
`validate_on_program_cache_miss`, `compute_output_specs`, `create_output_tensors`,
`compute_program_hash` and `create_op_performance_model`; `override_runtime_arguments` now lives on
each factory instead). The original audit also flagged a leftover free function
`slice_rm_reader_dynamic_args` in `slice_program_factory_rm.cpp`, called from nowhere, together with
three comments describing reader arg 0 as riding on `get_dynamic_runtime_args`. **Already fixed** —
`1b0d9d1258a` deleted the function, its declaration in `slice_program_factory_rm.hpp` and the
misleading comments, which is recommendation 3 of this document. That commit predates `fab067a`.

## Which validator runs on a cache hit

This op defines **no** `validate_on_program_cache_hit` — `slice_device_operation.hpp:45` declares
only `validate_on_program_cache_miss`. It therefore takes the favourable branch of the dispatcher,
which substitutes the miss validator on every hit:

```265:269:ttnn/api/ttnn/device_operation.hpp
    if constexpr (HasValidateOnProgramCacheHit<mesh_device_operation_t>) {
        mesh_device_operation_t::validate_on_program_cache_hit(operation_attributes, tensor_args);
    } else {
        mesh_device_operation_t::validate_on_program_cache_miss(operation_attributes, tensor_args);
    }
```

Every `TT_FATAL` in the miss validator is therefore live on hits. Three verdicts below depend on this
branch directly: verdict #4 (the device-storage requirements on the input and the start tensor are
what make the storage kinds carry no information, and they live only in the miss validator), and
post-`fab067a` verdict #2 (the 32x32 tile guard at `slice_device_operation.cpp:166-174`) and the
preallocated-output half of verdict #3 (the layout/dtype/tile checks at `:175-198`).

The substitution cuts both ways for this op, and finding #7 is where it shows. Every check in the
miss validator runs on hits, but a check that is *in the wrong branch* is not rescued by running on
hits: on a ROW_MAJOR input the three preallocated-output checks are skipped on both paths, and the
shape-only `TT_FATAL` at `slice_device_operation.cpp:144-152` is all that is left. Had `slice` defined
a narrow hit validator instead, verdicts #2, #3 and #4 would all degrade to "pinned only on the miss
path".

## Baseline: what the default hash would cover

| Source | Fields |
|---|---|
| `operation_attributes` | `slice_start`, `slice_end`, `step`, `output_mem_config`, `use_tensor_args`, `slice_dim`, `num_devices`, `sub_core_grids` |
| `tensor_args.input` | storage kind, `logical_shape`, `dtype`, `page_config`, `memory_config`, `alignment` |
| `tensor_args.start_tensor` | engaged-ness, then the same six fields |
| `tensor_args.end_tensor` | engaged-ness, then the same six fields |
| `tensor_args.preallocated_output` | engaged-ness, then the same six fields |

## What the custom hash covers

Post-`fab067a`, the first `hash_operation` call carries eleven terms and there are five spec blocks:

```350:362:ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_device_operation.cpp
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
        tensor_args.start_tensor.has_value(),
        tensor_args.end_tensor.has_value(),
        tensor_args.preallocated_output.has_value());
```

followed by the input (`:364-372`), the start tensor when engaged (`:374-384`), the end tensor when
engaged (`:386-398`), the computed output spec (`:400-408`) and the preallocated output when engaged
(`:410-424`) — each contributing the same six terms:

```418:423:ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_device_operation.cpp
            po.logical_shape().rank(),
            po.logical_shape(),
            po.padded_shape(),
            po.layout(),
            po.dtype(),
            po.memory_config());
```

Pre-fix, the first call ended at `tensor_args.start_tensor.has_value()` and there were three spec
blocks: input, start tensor, computed output spec. The end tensor and the preallocated output
contributed nothing, which is findings #1 and #3.

This is a *widening* custom hash, not a narrowing one. Every operation attribute is kept, each
tensor contributes five of its six default fields plus `padded_shape`, and the hash additionally
folds in the selected factory index and the full computed output spec — neither of which the
default key contains. What every block still drops is `page_config`, which is finding #2 for the
input (now guarded) and finding #7 for the preallocated output (still open on the ROW_MAJOR path).

### `slice_start` / `slice_end` / `step`: all three hashed, none patched, all structural

All three are keyed at lines 351-353, and none is treated as a dynamic runtime arg. That is the
correct choice, because each one changes program structure and the address-and-scalar patch would
not repair any of them:

- **`slice_end` and `step`** determine the output shape via
  `(end - start + step - 1) / step` (`slice_device_operation.cpp:255`), which sets the total
  work unit count and hence the `split_work_to_cores` result — the **core ranges** and per-group
  work counts. Core ranges are baked into the cached `Program`
  (`slice_program_factory_tile.cpp:28-41`, `slice_program_factory_rm.cpp:304-307`).
- **`step`** additionally selects the factory: any non-unit step routes to
  `SliceRmStrideProgramFactory` (`slice_device_operation.cpp:334-336`), a different kernel source
  entirely, and it is emitted directly into per-core runtime args
  (`slice_program_factory_rm_stride.cpp:127-133`).
- **`slice_start`** feeds `get_tiled_start_offset` / `get_rm_start_offset` (runtime, and so
  patchable), but it also feeds the row-major **CB page size** through the misalignment
  computation:

```204:219:ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_program_factory_rm.cpp
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
  at lines 222-231, changing `cb_page_size` and the CB `total_size` at
  `slice_program_factory_rm.cpp:322-330`. CB sizing is structural. So `slice_start` must be hashed
  even though most of its influence is on runtime args.

So the answer to "which of start/end/step are hashed, which are patched, which must be hashed" is:
all three are hashed, none is patched as a dynamic arg, and all three must be hashed. The op gets
this right.

## Omitted parameters

### 1. `tensor_args.end_tensor` — omitted entirely, including its engaged-ness

**Verdict: RESOLVED by fab067a** (was BUG).

`fab067a` added `tensor_args.end_tensor.has_value()` to the first `hash_operation` call
(`device/slice_device_operation.cpp:361`) and a full end-tensor spec block — rank, `logical_shape`,
`padded_shape`, `layout`, `dtype`, `memory_config` — mirroring the start-tensor block
(`:386-398`), with a comment naming the reader's `TensorAccessorArgs` as the reason. That is exactly
recommendation 1, and it means the end tensor's `IsDram` bit is now part of the key.

Pre-fix: the hash folded in `tensor_args.start_tensor.has_value()` and, when engaged, the start
tensor's full spec. There was no corresponding term for `end_tensor` anywhere in
`compute_program_hash`. But `SliceTileTensorArgsProgramFactory` consumes it, and specifically
consumes its buffer at **compile time**:

```80:87:ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_program_factory_tile_tensor_args.cpp
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

The end tensor is caller-supplied and its memory config is unconstrained: `slice.cpp:490-492`
requires only that it be on device, and its rank be 1 (`slice.cpp:451-454`). Nothing pins its
buffer type, so nothing but the hash can distinguish the two calls below.

**Reproduction (pre-fix)** (through `ttnn.slice` with tensor-valued bounds, i.e. the
`use_tensor_args` path, TILE layout, unit step, with `slice_dim` and `num_devices` supplied):

- **Call 1**: `start_tensor` in DRAM, `end_tensor` in **DRAM**. The reader compiles with
  `ArgConfig::IsDram` set for the end accessor.
- **Call 2**: identical input, identical `start_tensor`, `slice_dim`, `num_devices` and memory
  config, but `end_tensor` in **L1**. Every hashed term was unchanged — the end tensor contributed
  nothing to the key — so this was a cache hit.

The patch refreshed the end-tensor address as a `DynamicRuntimeArg`
(`slice_program_factory_rm_sharded.cpp:396`), but the compiled `TensorAccessor` at
`reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp:45` still resolved through the DRAM
bank table. The kernel read the slice bounds from the wrong memory space and then sliced with
whatever integers it found, producing a wrong-shaped read pattern with no error.

The asymmetry with `start_tensor` — hashed in full three lines earlier — made this look
like a simple oversight rather than a deliberate relaxation, which is how `fab067a` treated it.

### 2. `input.tensor_spec().page_config()` — only `layout()` is hashed, and the tile factories disagree about tiles

**Verdict: RESOLVED by fab067a** (was BUG).

`fab067a` added a `TT_FATAL` rejecting any tile other than 32x32 to the `TILE` branch of
`validate_on_program_cache_miss` (`device/slice_device_operation.cpp:166-174`), which — because this
op declares no `validate_on_program_cache_hit` — the dispatcher substitutes onto the hit path as
well. That is the minimal half of recommendation 2. `Tile` can no longer vary, so omitting
`page_config` from the key is correct by construction and the omission is a category-3 zero-value
omission. Note the guard makes the *tile-aware* factory's compile-time args constant too, so the
"mixed idiom" hazard described below is now a readability problem rather than a correctness one.

Pre-fix: `layout()` collapses `PageConfig` to `ROW_MAJOR` vs `TILE`, discarding the `Tile` shape. `slice`
accepts `Layout::TILE` (`slice_device_operation.cpp:103-106`), and this op was the "mixed" case:
one tile factory is tile-aware and the other is not, so the omission was a bug under *both* halves
of the rule.

**The tile-aware half.** `SliceTileTensorArgsProgramFactory` reads the real tile shape and passes
it into the reader's compile-time args:

```75:84:ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_program_factory_tile_tensor_args.cpp
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
calls that differed only in tile geometry — say `Tile{32, 32}` versus `Tile{16, 32}` on the same
logical shape — collided exactly: `logical_shape` equal, `padded_shape` equal (both pad to
the same extents for these tile heights), `layout()` `TILE` for both, `dtype` and
`memory_config` equal, and the computed output spec equal too, because
`compute_output_specs` constructs the output layout with `PageConfig(input_tensor.layout())`,
which discards the input's tile and defaults to 32x32 in both
cases. So the second call inherited the first's compile-time args 3 and 4 and sliced on the wrong
tile grid. `fab067a`'s guard now rejects the `Tile{16, 32}` call outright, on the hit path as well
as the miss path.

**The non-tile-aware half.** `SliceTileProgramFactory` computes everything from the architectural
constants:

```28:41:ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_program_factory_tile.cpp
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
`tile.get_tile_size(format)`), and it sets the CB page size and total size at lines 54-59. The
tile-count conversions at lines 68-73 use bare `TILE_WIDTH`/`TILE_HEIGHT`. The device operation
itself does the same in `get_tiled_start_offset` and `get_upper_start_offset`
(`slice_device_operation.cpp:36`, `61-72`), and the validator's tile-alignment checks are likewise against
the constants (`slice_device_operation.cpp:204-209`) rather than against the tensor's tile. So for
a non-32x32 input this factory built a program with the wrong CB page size and the wrong work
split — and because `page_config` is unhashed, it did not even build a fresh wrong program, it
reused the 32x32 entry.

Pre-fix, nothing anywhere in the op validated the tile geometry; `fab067a` added exactly that check.
The two factories using different idioms
for the same concept remains a hazard for maintainers: a reader of one would reasonably assume the
other behaves the same way, and the guard is what currently reconciles them.

### 3. `tensor_args.preallocated_output` — its own spec is not hashed

**Verdict: RESOLVED by fab067a** (was BUG). **See finding #7 for a residual hole on the ROW_MAJOR
input path.**

`fab067a` closed this from both directions. It added `preallocated_output.has_value()`
(`device/slice_device_operation.cpp:362`) and a full spec block for the tensor actually written to —
rank, `logical_shape`, `padded_shape`, `layout`, `dtype`, `memory_config` (`:410-424`) — so the
DRAM-vs-L1 reproduction below now produces two distinct keys. It also added three `TT_FATAL`s pinning
the preallocated output's `layout`, `dtype` and `Tile` to the input's (`:175-198`), which is
recommendation 4's spirit if not its letter: the check compares against the input rather than against
the whole computed spec, and it lives inside the `TILE` branch. That last detail is what leaves
finding #7 open.

Pre-fix: what the hash contained was `compute_output_specs(...)`, the *computed* spec, not
the spec of the tensor actually written to. When a preallocated output is supplied,
`create_output_tensors` returns it unchanged and the factories bind its buffer:

```297:306:ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_device_operation.cpp
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

```144:152:ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_device_operation.cpp
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

Pre-fix that was the only check. Nothing compared the preallocated tensor's dtype, memory config or
buffer type against the computed spec, and the writer's compile-time args are derived from the real
buffer:

```149:152:ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_program_factory_tile.cpp
    // --- Writer Kernel Descriptor ---
    // CB index via named compile-time arg (essential for fusion CB remapping).
    std::vector<uint32_t> writer_compile_time_args = {};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);
```

`TensorAccessorArgs` encodes the buffer's memory space into `ArgConfig::IsDram`
(`tt_metal/impl/buffers/tensor_accessor_args.cpp:146-157`) and its aligned page size into a second
compile-time word (`:198-204`). Mode A patches addresses and per-core scalars; it never touches
compile-time args, which are baked into the cached `Program`.

**Reachability.** `ttnn::prim::slice` is directly callable and takes `output_mem_config` and
`preallocated_output` as independent parameters — nothing requires the former to describe the
latter. Pre-fix, the `TT_FATAL` above constrained only the shape.

**Reproduction (pre-fix).** Take a `[1, 1, 64, 64]` TILE `bfloat16` input and slice
`[0,0,0,0]`-`[1,1,32,32]` with unit step.

- **Call 1**: `ttnn::prim::slice(input, start, end, step, /*output_mem_config=*/dram_interleaved,
  /*preallocated_output=*/out_dram)` where `out_dram` is a DRAM tensor of the required shape.
- **Call 2**: the same call in every argument except that `preallocated_output` is `out_l1`, an L1
  tensor of the same shape and dtype. `output_mem_config` is still `dram_interleaved`.

Pre-fix the hash carried only `compute_output_specs(...)`, built from the input and
`args.output_mem_config` — identical across the two calls — so call 2 hit. The shape `TT_FATAL`
passed, since only the shape was compared. But the cached writer kernel had been compiled with
`IsDram = true`, and call 2's destination is in L1. The writer issued DRAM-addressed NOC
transactions against an L1 address: the slice output went to the wrong memory space, and the
L1 tensor the caller passed was left untouched. Post-`fab067a` the two calls differ in
`po.memory_config()` and call 2 misses.

The dtype variant of the same defect was reachable the same way. The computed spec always takes
`input_tensor.dtype()`, so a preallocated output with a
different dtype hashed identically while changing the writer's element size. `fab067a` closed this
twice over: `po.dtype()` is hashed (`:422`), and on a TILE input a mismatching dtype is now a
`TT_FATAL` (`:185-189`).

The severity was bounded by the wrapper rather than by any check: `ttnn::slice` derives
`output_mem_config` *from* the preallocated tensor on both entry points (`slice.cpp:141` and
`slice.cpp:478`), so callers going through it could not construct the mismatch. That was not
enforcement, and the primitive is public, so it did not change the verdict.

Recommendation 4 asked for a `TT_FATAL` comparing the preallocated tensor's full `tensor_spec()`
against the computed one rather than just its shape, in preference to hashing the preallocated spec,
on the grounds that a mismatch is a caller error in every case. `fab067a` did both — it hashes the
spec *and* checks layout, dtype and tile — but the checks compare against the *input* rather than
the computed spec, and they sit inside the `TILE` branch. Finding #7 is what falls through that gap.

### 4. `input.storage` variant kind, and `start_tensor.storage` kind

**Verdict: VALID — pinned by validation.**

```101:102:ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_device_operation.cpp
    TT_FATAL(tensor_args.input.storage_type() == StorageType::DEVICE, "Operands to unpad need to be on device!");
    TT_FATAL(tensor_args.input.buffer() != nullptr, "Operands to unpad need to be allocated in buffers on device!");
```

The start tensor is pinned at the wrapper (`slice.cpp:487-489`) and its buffer is null-checked in
the factory (`slice_program_factory_tile_tensor_args.cpp:44`). Both carry no information, and the
input's check is re-run on hits through the substitution branch quoted under "Which validator runs
on a cache hit".

### 5. `input.tensor_layout().get_alignment()` (and the start tensor's)

**Verdict: VALID — invariant.**

`Alignment` influences the program only through quantities that are themselves hashed. Its primary
effect is on `padded_shape`, which the hash carries explicitly alongside `logical_shape` (lines
368-369) — that pairing is what makes the alignment redundant, and it is also why slice needs no
logical-vs-padded relaxation. The buffer-level alignment the
factories actually read is not the tensor's `Alignment` at all but the buffer's, derived from its
memory space:

```75:81:ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_program_factory_rm.cpp
    auto src_buffer_alignment = input_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? ::hal::get_dram_alignment()
                                    : ::hal::get_l1_alignment();
    auto dst_buffer_alignment = output_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? ::hal::get_dram_alignment()
                                    : ::hal::get_l1_alignment();
    auto alignment = std::max(src_buffer_alignment, dst_buffer_alignment);
```

which is a function of `input.memory_config()` and `output_spec.memory_config()`, both hashed.
`slice_program_factory_rm_stride.cpp:61-62` reads `Buffer::alignment()` directly, which is
likewise determined by buffer type plus page size, and page size is fixed by the hashed shape,
dtype and memory config — with the one exception finding #7 covers, where the destination buffer's
page size is set by an unhashed `page_config`.

### 6. Buffer addresses (omitted by the default hash as well)

**Verdict: VALID — patched.**

Four of five factories have every address refreshed by the shared patch helper, covering per-core
slot 0 (for example the writer binding at `slice_program_factory_rm.cpp:387`) and the tile
factories' common args (the source binding at `slice_program_factory_tile.cpp:143` and the
source/start/end trio at `slice_program_factory_tile_tensor_args.cpp:182-184`). The RM-sharded
factory carries its
addresses on CB bindings (`slice_program_factory_rm_sharded.cpp:282` and `293`), which the
helper's CB-address-only descriptor patches positionally in the order the factory pushed them —
an ordering the factory comment at line 279 explicitly flags as a contract to maintain.

### 7. `preallocated_output.tensor_spec().page_config()` on the ROW_MAJOR input path

**Verdict: BUG** (new; not in the pre-fix audit).

`fab067a` added a full spec block for the preallocated output — but the block, like every other
block in this hash, omits `page_config` (`device/slice_device_operation.cpp:418-423`). It also added
`TT_FATAL`s pinning the preallocated output's `layout`, `dtype` and `Tile` to the input's, which
would make the omission harmless — except that all three landed *inside* the
`if (tensor_args.input.layout() == Layout::TILE)` branch that opens at
`device/slice_device_operation.cpp:165`:

```175:198:ttnn/cpp/ttnn/operations/data_movement/slice/device/slice_device_operation.cpp
        if (tensor_args.preallocated_output.has_value()) {
            const auto& out_tensor = tensor_args.preallocated_output.value();
            // Layout and dtype must be checked before the tile: PageConfig::get_tile() reports a default
            // 32x32 Tile for a ROW_MAJOR tensor, so the comparison below would accept one. Both properties
            // are pinned to the input by compute_output_specs, which is what the tile factories are built
            // against, so this rejects only combinations the op never produced for itself.
            TT_FATAL(
                out_tensor.layout() == Layout::TILE,
                "The preallocated output tensor must be TILE layout to match the input, but it has {} layout",
                out_tensor.layout());
```

So on a **ROW_MAJOR input** none of the three checks runs, and the preallocated output's
`page_config` is neither hashed nor validated. It reaches the compiled program: the writer's
`TensorAccessorArgs` emits the destination buffer's aligned page size as a compile-time word
(`tt_metal/impl/buffers/tensor_accessor_args.cpp:198-204`), and `Buffer::aligned_page_size()` is
`align(page_size(), alignment())` (`tt_metal/impl/buffers/buffer.cpp:770`) — for a TILE-paged buffer,
`page_size()` is `tile.get_tile_size(dtype)`, a function of the tile geometry.

**Reachability.** Narrow, and it requires a caller-supplied output whose layout disagrees with the
input's — a configuration the op never produces for itself, since `compute_output_specs` builds the
output layout with `PageConfig(input_tensor.layout())`. `ttnn::prim::slice` takes
`preallocated_output` as an independent parameter and, on a ROW_MAJOR input, the only surviving check
is the `padded_shape` comparison at `slice_device_operation.cpp:144-152`.

**Reproduction.** Take a `[1, 1, 64, 64]` ROW_MAJOR `bfloat16` input, slice
`[0,0,0,0]`-`[1,1,32,64]` with unit step, and pass a **TILE-layout** preallocated output of the
required padded shape.

- **Call 1**: `preallocated_output` is a TILE tensor with `Tile{32, 32}`.
- **Call 2**: identical in every argument, but the preallocated output has `Tile{16, 32}`. Every
  hashed term matches — `po.logical_shape()`, `po.padded_shape()`, `po.layout()` (`TILE` for both),
  `po.dtype()` and `po.memory_config()` are all equal, and `page_config` is not in the key — so
  call 2 hits.

The ROW_MAJOR input path takes the layout branch at `slice_device_operation.cpp:210`, so none of the
tile guards fires; the destination buffer's aligned page size is halved relative to what the cached
writer was compiled against, and the writer's page arithmetic addresses the wrong offsets.

**Fix.** Two clean options. Either move the three preallocated-output checks out of the `TILE`
branch — the layout check in particular belongs unconditionally, phrased as
`out_tensor.layout() == tensor_args.input.layout()`, which subsumes the tile question by making the
computed spec and the real destination agree — or take recommendation 4's original form and compare
the preallocated tensor's full `tensor_spec()` against `compute_output_specs(...)`, which pins
`page_config` along with everything else. The second is preferable: it is one comparison, it covers
every field the hash omits, and a mismatch is a caller error in every case. Adding `page_config` to
the hash instead would work but would build a second program for a configuration that should be
rejected.

## Keys the custom hash adds beyond the default

Five, and each is load-bearing:

- **`input.padded_shape()`** (line 369), hashed *alongside* `logical_shape` rather than instead of
  it. Every factory computes its work split from padded extents, so this closes the gap the
  default's derivation-free key would leave. The same pairing appears in all five spec blocks, which
  is why slice carries no logical-vs-padded relaxation.
- **`factory.index()`** (line 359). `select_program_factory` branches on layout, step, and the
  input/output sharding combination (`slice_device_operation.cpp:311-341`); folding the resulting
  variant index into the key means two configurations that would map to different kernels can
  never share an entry, independently of whether the inputs that drove the choice are all hashed.
- **The full computed output spec** (lines 400-408). The default hashes only inputs, so an output
  spec that varies through `compute_output_specs`'s shard-spec synthesis would otherwise be
  invisible to the key. Since the
  synthesis depends on `generate_transpose_shard_spec` and on tile-alignment adjustments, this is
  a real risk that the op has closed.
- **The full end-tensor spec** (lines 386-398), added by `fab067a`. The default would have covered
  this; the pre-fix custom hash dropped it, which was finding #1.
- **The full preallocated-output spec** (lines 410-424), added by `fab067a`. This one the default
  would *not* have covered in the form that matters — the default hashes the tensor arg, but the
  pre-fix custom hash substituted the *computed* spec for it, which was finding #3.

The header comment (lines 345-348) frames all of this as a fix for weak hash distribution on
small-integer shape sequences, citing issue #47602. Whatever the original motivation, the effect
is a key that is strictly stronger than the default on the tensor-spec axis, with `page_config` the
one field it still drops everywhere.

## Framework side effect of having a custom hash

```1035:1037:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
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
| `end_tensor` (spec and engaged-ness) | Yes — reader compile-time args via `TensorAccessorArgs` | Address yes, accessor config no | **RESOLVED by fab067a** — now hashed in full |
| `input.page_config` (`Tile`) | Yes — compile-time args (tensor-args factory), CB page size and work split (tile factory) | No | **RESOLVED by fab067a** — pinned by the 32x32 guard, live on hits |
| `preallocated_output` spec | Yes — writer `TensorAccessorArgs` compile-time args, element size | No | **RESOLVED by fab067a** — now hashed, plus layout/dtype/tile `TT_FATAL`s |
| `preallocated_output.page_config` on a ROW_MAJOR input | Yes — writer's aligned page size, a compile-time arg | No | **BUG** — neither hashed nor guarded (checks sit inside the TILE branch) |
| `input.storage` kind, `start_tensor.storage` kind | n/a | n/a | VALID — pinned by validation |
| `input.tensor_layout.alignment` | Only via hashed derivatives | n/a | VALID — invariant |
| Buffer addresses | Yes | Yes (address patch; CB patch on RM-sharded) | VALID — patched |

**One program-cache correctness bug remains**, down from three. None ever stemmed from the patching
design, which is sound; all of them are gaps in hash coverage that mode A structurally cannot
compensate for, because compile-time args are frozen in the cached `Program` and every one of these
findings lands on a compile-time arg.

`fab067a` closed the original three. The `end_tensor` omission was an asymmetry with `start_tensor`
that read as an oversight, and it is now hashed in full. The `page_config` omission was the
unguarded-tile pattern, aggravated because one
tile factory genuinely varies its compile-time args with the tile shape while the other hardcodes
32x32; a 32x32 guard in the validator now pins it. The `preallocated_output` omission was different
in character: the op checked the
preallocated tensor, but only its shape, so a caller reaching `ttnn::prim::slice` directly could
supply an output in a different memory space or dtype than the hashed computed spec described and
inherit a writer kernel compiled for the wrong one; the spec is now hashed, and layout, dtype and
tile are additionally asserted against the input's.

The residual bug (finding #7) is the tail of that third fix. `page_config` is the one field every
spec block in this hash still drops, and the `TT_FATAL`s that would otherwise pin it for the
preallocated output live inside the `TILE`-input branch of the validator. On a ROW_MAJOR input with a
caller-supplied TILE-layout output, the destination buffer's aligned page size — a writer
compile-time arg — is free to vary across a cache hit. It is a narrow path that requires an output
layout the op never produces for itself, but the fix is one comparison and is worth taking.

**On whether this op is a reference implementation:** the CSV's `specimen_done=Y` is justified for
the *patching* half and for the treatment of `slice_start`/`slice_end`/`step`, and the code should
be read that way by other ops. The patch routes every factory through one shared
`patch_slice_program_addresses` helper whose contract is stated in a comment above it, so there is no
per-factory arg-index table to fall out of sync — contrast `interleaved_to_sharded_partial`, whose
hand-written two-slot patch depends on constants that must track its factory. The one place slice
takes a further shortcut (RM-sharded, CB addresses only) is accompanied by an argument for why the
remaining args are hash-pinned, and that argument checks out. The hash is also unusually
disciplined in the ways that matter for structure: it keys the factory index and the computed
output spec, neither of which the default covers, and post-`fab067a` it keys every tensor arg it
touches. What the op is *not* a specimen for is
completeness of tensor coverage: it drops `page_config` from all five spec blocks and relies on
validator guards to make that safe, and one of those guards is in the wrong branch. Worth citing as
a model for the cache-hit contract and for widening a key, with finding #7 called out.

Two CSV corrections. `get_dynamic_runtime_args` should be **N**: the member does not exist, and mode
A would bypass it regardless. (The uncalled `slice_rm_reader_dynamic_args` helper the original audit
cited here has since been deleted by `1b0d9d1258a`.)
And `own_hit_validator = N` understates the situation, since the dispatcher substitutes
`validate_on_program_cache_miss` on hits (`ttnn/api/ttnn/device_operation.hpp:265-269`), which
verdicts #2, #4 and #7 all rely on.

## Recommendations

1. **Fixed by `fab067a`.** ~~Hash~~ Hashed `end_tensor` symmetrically with `start_tensor`:
   `tensor_args.end_tensor.has_value()` in the first `hash_operation` call
   (`slice_device_operation.cpp:361`) and a block mirroring the start-tensor one (`:386-398`).
2. **Fixed by `fab067a`, minimal route.** A tile guard in `validate_on_program_cache_miss` rejecting
   non-32x32 tiles (`slice_device_operation.cpp:166-174`), following
   `interleaved_to_sharded_op.cpp:95-97`; that makes the omission
   correct by construction but discards the tile-awareness
   `SliceTileTensorArgsProgramFactory` already has. The fix that would preserve it — add
   `input.tensor_spec().tile()` to the hash *and* make `SliceTileProgramFactory` and the
   `get_tiled_start_offset` helpers tile-aware in the same change — remains available if non-32x32
   tiles are ever needed here. Do not add the hash term
   without the factory work, or vice versa: either alone leaves a wrong-program path.
3. **Fixed earlier, by `1b0d9d1258a`, not by `fab067a`.** `slice_rm_reader_dynamic_args`, its
   declaration in `slice_program_factory_rm.hpp` and the three comments in
   `slice_program_factory_rm.cpp` describing reader arg 0 as riding on `get_dynamic_runtime_args`
   are all gone. Dead code that documents a mechanism the op does not use is
   actively misleading for exactly this kind of audit.
4. **Partly fixed by `fab067a`; finish it.** The recommendation was to compare the preallocated
   tensor's full `tensor_spec()` against the computed spec rather than only `padded_shape`.
   `fab067a` instead hashed the preallocated spec (`:410-424`) and added layout/dtype/tile `TT_FATAL`s
   against the *input* (`:175-198`) — and put them inside the `TILE` branch. Take the original form:
   one `tensor_spec()` comparison in `validate_on_program_cache_miss`, outside any layout branch.
   That closes finding #7 and covers `page_config` along with everything else. A mismatch is a caller
   error in every case, so rejecting it loudly beats building a second program for it, and because
   the op has no hit validator the check runs on hits as well as misses.
5. **Still open.** Run this op's tests once under `-DTT_DESCRIPTOR_PATCHING_PARITY_CHECK`. The parity
   oracle in the mode-A branch (`mesh_device_operation_adapter.hpp:688-702`) will not catch findings
   #1, #2, #3 or #7, since all four are compile-time-arg staleness rather than runtime-arg staleness,
   but it does validate the RM-sharded shortcut at #6, which is the one place the op patches rather
   than re-derives per core.
