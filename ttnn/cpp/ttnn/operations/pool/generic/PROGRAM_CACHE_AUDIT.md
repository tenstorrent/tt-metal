# Program Cache Audit — `pool/generic`

Audit of `Pool2D::compute_program_hash` against the framework default ("hash everything") key.
`Pool2D` is the shared device op behind `ttnn.max_pool2d`, `ttnn.avg_pool2d`, and the
`return_indices` max-pool-with-indices variant.

| | |
|---|---|
| Device operation | `Pool2D` (`device/pool_op.hpp:26-71`) |
| Custom hash | `device/pool_op.cpp:168-185` |
| `operation_attributes_t` | 11 fields (`device/pool_op.hpp:27-39`) |
| `tensor_args_t` | `input_tensor_` (single tensor) |
| Program factory | `Pool2D::MultiCore` (single), declarative `create_workload_descriptor` |
| `override_runtime_arguments` | No |
| `get_dynamic_runtime_args` | No |
| `validate_on_program_cache_hit` | **Yes**, and it runs the full miss-time validation (`device/pool_op.cpp:83-85`) |
| Cache-hit patch mechanism | **WorkloadDescriptor buffer-binding fast path — no rebuild ever** |

**Result: one BUG.** `SlidingWindowConfig::ceil_pad_hw` is program-shaping but is absent from the
string that `SlidingWindowConfig::get_hash()` hashes, so two pool configurations that compile to
different kernels, different circular-buffer geometry, and a different host-generated halo table
can collide on one cache entry. No `TT_FATAL` anywhere constrains the field and `ttnn::prim::pool2d`
accepts the config from the caller verbatim, so the bad configuration is reachable through the
public API without violating an enforced constraint. The defect is inherited from
`SlidingWindowConfig`'s own hash rather than introduced by `Pool2D::compute_program_hash`; the
default reflection hash would have the identical hole (see "Attribution"). Every other omission in
this op is sound, and the op is otherwise unusually well built for cache safety — its hit-time
validator is the same function as its miss-time validator, and its output spec is computed without
reading the input tensor at all. In particular the factory's bare 32x32 tile arithmetic is *not* the
unguarded-tile defect it resembles, because no caller-supplied tile can reach this op on either the
input or the output side; see omission 4.

## Cache-hit patch mechanism

`Pool2D::MultiCore` defines `create_workload_descriptor` (`device/pool_op.hpp:56-60`), which selects
the declarative WorkloadDescriptor branch of `DescriptorMeshWorkloadAdapter`
(`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:344-352`). The cache-hit path for that branch is:

```641:656:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
                if constexpr (has_workload_descriptor) {
                    // WorkloadDescriptor variant — declarative: there is no slow-path rebuild
                    // because re-running create_workload_descriptor would re-allocate
                    // workload-scoped resources (GlobalSemaphores, MeshBuffers).
                    // CB bindings are always populated by resolve_bindings, so the
                    // fast path covers cache hits even when the factory only sets
                    // `desc.cbs[i].buffer` and declares no rt-arg buffer bindings.
                    if (!sv.resolved_bindings.empty()) {
                        auto collected =
                            collect_tensor_buffers(tensor_args, tensor_return_value, sv.workload_descriptor);
                        tt::tt_metal::apply_resolved_bindings(program, sv.resolved_bindings, collected.buffers);
                    }
                    // The WorkloadDescriptor variant never rebuilds, so a value a custom hash
                    // excluded would stay frozen at first miss — re-apply declared dynamic args.
                    apply_dynamic_runtime_args_if_declared(
                        program, attrs, tensor_args, tensor_return_value, coordinate_range);
                }
```

This is the most restrictive of the three patch modes, and the "no override, no dynamic" combination
the brief flagged puts `Pool2D` squarely in it. On a hit, **only buffer addresses are refreshed**:

- **Patched:** the addresses behind the `.buffer`-bound CBs — the input shard CB
  (`pool_multi_core_program_factory.cpp:461`), the output CB (`:696-703`), the index output CB
  (`:712-717`), and, on the L1 path, the reader-indices CB (`:491-496`) and scalar-config CB
  (`:759`).
- **Frozen forever:** every compile-time arg (reader `:762-819`, compute `:862-904`), every raw
  `uint32_t` runtime arg (`:962-978`), every CB total size / page size / data format / face
  geometry, the `ComputeConfigDescriptor` (`:933-938`), the kernel source selection (`:828-831`,
  `:923-925`), the `#define`s from `get_defines(pool_type)` (`:920-921`), and all core ranges
  (`all_cores` = `input.shard_spec()->grid`, `:312`).
- **Frozen forever, and unique to this op:** the *contents* of two host-generated device tensors.
  `create_workload_descriptor` builds the sliding-window halo lookup table (`:1131-1145`) and, for
  avg-pool variants needing per-stick divisors, the scalar config tensor (`:1197-1214`), uploads
  them, and parks them on the descriptor so they outlive the cache entry (`:1155-1157`, `:1215-1217`).
  Because there is no rebuild, those bytes are computed exactly once. Any hash omission that would
  change the table's contents is therefore not merely a stale-arg problem; it silently feeds the
  kernels a lookup table for a different pooling geometry.

Two consequences worth stating plainly, because they set the bar for every verdict below:

1. The hash must cover **everything** the factory reads, since nothing except a buffer address is
   ever recomputed.
2. The buffer addresses that *are* baked in as compile-time args — `config_buffer->address()`
   (reader CT arg 33, `:796`) and `reader_indices_buffer->address()` (CT arg 35, `:798`) — are
   nonetheless safe, because those two buffers are owned by the cached workload
   (`WorkloadDescriptor::buffers`) and so keep the same address for the entry's whole life. They are
   deliberately *not* per-call tensors.

The framework also mixes the workload's mesh coordinates into the key independently of the custom
hash (`mesh_device_operation_adapter.hpp:989-992`), so the `tensor_coords` that
`create_workload_descriptor` iterates (`:1278-1285`) are covered without the op doing anything.

## Baseline: what the default hash would cover

`hash_objects_with_default_seed(type_hash<Pool2D>, attrs, tensor_args)`:

| Source | Fields |
|---|---|
| `operation_attributes` | `sliding_window_config_`, `pool_type_`, `output_dtype_`, `output_layout_`, `memory_config_`, `compute_kernel_config_`, `count_include_pad_`, `divisor_override_`, `return_indices_`, `memory_used`, `config_tensor_in_dram` |
| `input_tensor_.storage` | storage variant kind |
| `input_tensor_.tensor_spec` | `logical_shape`, and `tensor_layout` = { `dtype`, `page_config`, `memory_config`, `alignment` } |

`SlidingWindowConfig` is not hashed field-by-field even on the default path: it has a `std::hash`
specialization (`ttnn/cpp/ttnn/operations/sliding_window/sliding_window.hpp:202-207`), and
`hash_object` prefers `std::hash` over reflection (`tt_stl/tt_stl/reflection.hpp:1309-1313`). This
matters for attribution below.

## What the custom hash covers

```168:185:ttnn/cpp/ttnn/operations/pool/generic/device/pool_op.cpp
ttsl::hash::hash_t Pool2D::compute_program_hash(const operation_attributes_t& op_attr, const tensor_args_t& tensor) {
    auto input_mem_config = tensor.input_tensor_.memory_config();
    auto in_dtype = tensor.input_tensor_.dtype();
    auto out_dtype = op_attr.output_dtype_;
    return tt::tt_metal::operation::hash_operation<Pool2D>(
        op_attr.sliding_window_config_.get_hash(),
        op_attr.pool_type_,
        op_attr.output_layout_,
        op_attr.memory_config_,
        op_attr.compute_kernel_config_,
        op_attr.divisor_override_,
        op_attr.count_include_pad_,
        op_attr.return_indices_,
        op_attr.config_tensor_in_dram,
        input_mem_config,
        in_dtype,
        out_dtype);
}
```

Answering the brief's specific questions directly: `pool_type_` **is** hashed (`:174`), so max and
average pool cannot share a cache entry — the `get_defines(pool_type)` map (`:920`), the reader's
CT arg 22, and `bf16_scalar` / `bf16_init_value` (`:324-325`) are all safely keyed.
`divisor_override_` (`:178`), `count_include_pad_` (`:179`) and `return_indices_` (`:180`) are
hashed too, which is required: `return_indices_` alone switches the kernel source
(`:828-831`, `:923-925`) and adds eight CBs (`:553-624`).

The only `operation_attributes_t` field the custom hash drops is `memory_used`.

## Omitted parameters

### 1. `sliding_window_config_.ceil_pad_hw`

**Verdict: BUG.**

`get_hash()` is a hash of `to_string()`:

```17:17:ttnn/cpp/ttnn/operations/sliding_window/sliding_window.cpp
std::size_t SlidingWindowConfig::get_hash() const { return std::hash<std::string>{}(to_string()); }
```

and `to_string()` enumerates every field of the struct except one:

```1283:1298:ttnn/cpp/ttnn/operations/sliding_window/sliding_window.cpp
std::string SlidingWindowConfig::to_string() const {
    return "batch=" + std::to_string(batch_size) + "_ch=" + std::to_string(channels) +
           "_in_h=" + std::to_string(std::get<0>(input_hw)) + "_in_w=" + std::to_string(std::get<1>(input_hw)) +
           "_win_h=" + std::to_string(std::get<0>(window_hw)) + "_win_w=" + std::to_string(std::get<1>(window_hw)) +
           "_stride_h=" + std::to_string(std::get<0>(stride_hw)) +
           "_stride_w=" + std::to_string(std::get<1>(stride_hw)) + "_pad_t=" + std::to_string(padding[0]) +
           "_pad_b=" + std::to_string(padding[1]) + "_pad_l=" + std::to_string(padding[2]) +
           "_pad_r=" + std::to_string(padding[3]) + "_out_pad_h=" + std::to_string(std::get<0>(output_pad_hw)) +
           "_out_pad_w=" + std::to_string(std::get<1>(output_pad_hw)) +
           "_dil_h=" + std::to_string(std::get<0>(dilation_hw)) + "_dil_w=" + std::to_string(std::get<1>(dilation_hw)) +
           "_scale_h=" + std::to_string(scale_h) + "_scale_w=" + std::to_string(scale_w) +
           "_cores_nhw=" + std::to_string(num_cores_nhw) + "_cores_c=" + std::to_string(num_cores_c) +
           "_grid=" + core_range_set.str() + (snap_to_tile ? "_snap_to_tile" : "") + (is_bilinear ? "_bilinear" : "") +
           (is_transpose ? "_transpose" : "") + (ceil_mode ? "_ceil_mode" : "") +
           (padding_mode == PaddingMode::Replicate ? "_replicate_pad" : "");
}
```

`ceil_pad_hw` (`sliding_window.hpp:57`) is missing. `ceil_mode` is present, but it is only the
boolean; the explicit pad override it gates is not. The omission looks like an oversight rather than
a decision: the `fmt` formatter for the same struct *does* print the ceil pad
(`sliding_window.cpp:1335`, `:1351-1352`), and the enum immediately above the struct carries a
comment warning that these values are "part of the serialized program-cache key via
`SlidingWindowConfig::to_string()`" (`sliding_window.hpp:17-18`).

**Why it is program-shaping.** `ceil_pad_hw` is not a cosmetic field. When set it is returned
verbatim by the accessor that everything downstream uses:

```125:131:ttnn/cpp/ttnn/operations/sliding_window/sliding_window.cpp
uint32_pair_t SlidingWindowConfig::get_ceil_pad_hw() const {
    if (!ceil_mode) {
        return {0, 0};
    }
    if (ceil_pad_hw.has_value()) {
        return ceil_pad_hw.value();
    }
```

From there it reaches, in this op alone:

- **The output shape.** `get_output_shape()` adds `get_ceil_pad_h()` / `get_ceil_pad_w()` to the
  numerator (`sliding_window.cpp:88-93`), and `Pool2D::compute_output_specs` derives `out_nhw`,
  the padded output shape and therefore the output shard shape from it (`pool_op.cpp:93-138`).
- **Reader compile-time arg 14** (`ceil_pad_w`, `pool_multi_core_program_factory.cpp:777`).
- **Compute compile-time args 26 and 27** (`in_h_padded`, `in_w_padded`, `:890-891`), computed as
  `in_h + pad_h + ceil_pad_h` and `in_w + pad_w + ceil_pad_w` (`:342-343`).
- **The scalar-config decision** `one_scalar_per_core` (`:344-345`, and again at `:1164-1172`),
  which controls whether the avg-pool scalar config tensor and its CB exist at all (`:733-761`),
  reader CT args 23/24/33/34, and compute CT arg 12.
- **The contents of the scalar config tensor**, through `AvgPoolConfig::ceil_h` / `ceil_w`
  (`:1188-1189`) consumed by `get_bf16_avg_pool_config_scalars` (`:55-118`).
- **The contents of the halo lookup table**, through `generate_pad_metadata`
  (`sliding_window.cpp:254-257`), `generate_op_trace_metadata` (`:373-377`) and
  `generate_shard_boundaries` (`:393`), all called from `create_workload_descriptor`
  (`pool_multi_core_program_factory.cpp:1131-1136`).
- **Per-core runtime args** `start_row` / `start_col` for the indices variant, via `in_w_padded`
  and `in_h_padded` (`:967-969`).

None of these are refreshed on a cache hit — the reader-indices and scalar-config *tensors* are not
even regenerated. So a wrong hit here does not produce a slightly-off program; it runs the wrong
lookup table.

**Why the hash cannot separate them.** Two `SlidingWindowConfig` values that agree on all 18
stringified fields but differ in `ceil_pad_hw` produce byte-identical `to_string()` output and
therefore identical `get_hash()`, identical `compute_program_hash`, and — because a custom hash
opts the op out of canonical-key collision resolution
(`mesh_device_operation_adapter.hpp:1012-1013`) — a wrong hit rather than a rebuild. The hit-time
validator does not help: `validate_pool2d` never inspects the ceil pad
(`pool_op.cpp:19-64`).

**Reachability, and why this is a BUG and not a CAVEAT.** The test is whether the bad configuration
is reachable through the op's public API without violating an *enforced* constraint, where enforced
means a `TT_FATAL` on the relevant path. Both halves resolve against the bug.

*Nothing enforces `ceil_pad_hw`.* A search of the whole `ttnn` tree for the identifier returns the
declaration (`sliding_window.hpp:57`), the accessors (`:98-100`), the computations that consume it,
the formatter, and the pool call sites — and not a single `TT_FATAL`, in `validate_pool2d`
(`pool_op.cpp:19-64`) or anywhere else. The field can hold any value, including one inconsistent
with the `input_hw` / `padding` / `stride_hw` it sits next to.

*The field is caller-supplied through a public entry point.* `Pool2D` is reached through
`ttnn::prim::pool2d`, which takes the `SlidingWindowConfig` from the caller by reference and passes
it through unmodified:

```76:88:ttnn/cpp/ttnn/operations/pool/generic/device/pool_op.hpp
std::vector<ttnn::Tensor> pool2d(
    const Tensor& input_tensor,
    const ttnn::operations::sliding_window::SlidingWindowConfig& sliding_window_config,
    ttnn::operations::pool::Pool2DType pool_type,
    DataType output_dtype,
    Layout output_layout,
    MemoryConfig memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    bool count_include_pad,
    std::optional<int32_t> divisor_override,
    bool return_indices,
    uint32_t memory_used,
    bool config_tensor_in_dram);
```

So the verdict does not depend on the behaviour of any particular in-tree caller. That said, the
in-tree caller situation is worth recording as severity context, because it shows the field is
exercised rather than dormant. The pool DRAM slicer sets it on every slice of a `ceil_mode` pool:
`Pool2DSliceHandler::get_input_slice_and_padding` computes a per-slice `this_ceil_pad` that is zero
on interior slices and the whole-tensor ceil pad on the trailing slice (`generic_pools.cpp:536`,
`:545`, `:554`), and `run_L1_op` forwards it into `pool2d_L1` (`:706`), which plants it in the
config (`:182-184`). For most slice geometries the explicit value coincides with what
`get_ceil_pad_hw()` would have derived from the hashed fields, because trimming the input slice at
the tensor edge removes exactly the ceil pad — so those calls are accidentally safe. The
coincidence breaks where the slicer rewrites `pad_right` to round the output slice up to a tile
(`:559-569`): that recomputed padding makes the slice geometry ceil-exact, so the derived ceil pad
is zero while the explicit one is the whole-tensor value. At that point the config carries
information the hash cannot see.

An unenforced, publicly reachable, program-shaping field grades as a BUG whether or not a failing
end-to-end pair has been demonstrated, and "every current caller happens to comply" is not a
defence — it is the normal condition of a latent bug. I did not construct a concrete failing pair of
`ttnn.max_pool2d` calls; that affects how urgent the fix is, not what the finding is. See
`## Open questions`.

**Fix.** One line, in the shared type rather than in `Pool2D`: append the ceil pad to `to_string()`.
Using the accessor rather than the raw optional keeps the string stable for the overwhelmingly
common `nullopt` case, so it does not invalidate any currently-shared cache entry:

```cpp
+ "_ceil_pad_h=" + std::to_string(get_ceil_pad_h()) +
+ "_ceil_pad_w=" + std::to_string(get_ceil_pad_w()) +
```

The same change also fixes the other affected consumer of `SlidingWindowConfig::get_hash()` — both
halo variants, which key a memoization map on the same string. Note that halo's failure mode is a
wrong memoized stick count rather than a wrong program, and that conv2d is not affected at all;
recommendation 1 sets out which op gets what.

### Attribution

This omission is **not introduced by `Pool2D::compute_program_hash`.** The default reflection hash
would have exactly the same hole, because `SlidingWindowConfig` is `std::hash`-specialized and
`hash_object` takes the `std::hash` branch before it ever considers reflection attributes
(`tt_stl/tt_stl/reflection.hpp:1309-1313`), landing in the same `to_string()`. Switching `Pool2D` to
the default hash would not fix it. It is nonetheless a program-cache correctness bug for this op,
and this op is where its consequences are worst, because the WorkloadDescriptor path never rebuilds.

Conversely, the custom hash is strictly *better* than the default in one respect: it passes
`get_hash()` (a `std::size_t`) into `hash_operation` directly, where `hash_object` returns integers
unchanged (`reflection.hpp:1304-1308`). The default path would instead call
`std::hash<SlidingWindowConfig>`, which narrows through `int`:

```202:207:ttnn/cpp/ttnn/operations/sliding_window/sliding_window.hpp
template <>
struct std::hash<ttnn::operations::sliding_window::SlidingWindowConfig> {
    size_t operator()(const ttnn::operations::sliding_window::SlidingWindowConfig& config) const {
        return std::hash<int>()(config.get_hash());
    }
};
```

discarding the top 32 bits of the string hash. The custom hash avoids that narrowing.

### 2. `op_attr.memory_used`

**Verdict: VALID — unused.**

The only read of this field anywhere in the op is a debug consistency assertion:

```1001:1017:ttnn/cpp/ttnn/operations/pool/generic/device/pool_multi_core_program_factory.cpp
    uint32_t post_allocate_size =
        input.device()->allocator()->get_statistics(tt::tt_metal::BufferType::L1).total_allocated_bytes;
    uint32_t actual_global_cb_size = post_allocate_size == 0 ? 0 : post_allocate_size - memory_used;

    // For now assume that if post_op_l1_allocation_size == 0 op is being run
    // in graph capture NO_DISPATCH mode.
    bool is_graph_capture_no_dispatch_mode = post_allocate_size == 0;
    TT_FATAL(
        actual_local_cb_size == cb_sizes.local_cb_total() || is_graph_capture_no_dispatch_mode,
        "Local CB size mismatch: actual {} != expected {}",
        actual_local_cb_size,
        cb_sizes.local_cb_total());
    TT_FATAL(
        actual_global_cb_size == cb_sizes.global_cb_total() || is_graph_capture_no_dispatch_mode,
        "Global CB size mismatch: actual {} != expected {}",
        actual_global_cb_size,
        cb_sizes.global_cb_total());
```

It never reaches a compile-time arg, a runtime arg, a CB, a kernel, or a core range (verified by
grep: the field appears only at `pool_op.hpp:37`, `pool_op.hpp:87`, `pool_op.cpp:236`,
`pool_op.cpp:250`, `pool_multi_core_program_factory.cpp:298`, `:1003` and `:1275`). Omitting it is
correct and is a genuine relaxation: `memory_used` is a caller-supplied snapshot of the L1
high-water mark, so hashing it would fragment the cache on an allocator statistic that has nothing
to do with the compiled program. The cost is only that the assertion stops running on hits, which
is what one wants from a miss-time sanity check.

### 3. `input_tensor_.logical_shape()`

**Verdict: VALID — unused.**

The factory does not read the input's logical shape at all. It reads `padded_shape()`, and only
element `[3]` (the channel count), at four places (`pool_multi_core_program_factory.cpp:374`,
`:377`, `:383`, `:385-388`). Every other shape-like quantity comes from either the hashed
`SlidingWindowConfig` (`in_n`, `in_c`, `in_h`, `in_w`, `out_h`, `out_w`, `out_c` — all assigned in
`compute_pool_setup`, `:1091-1114`) or the hashed shard specs
(`max_in_nhw_per_core = input.shard_spec()->shape[0]`, `:317`;
`max_out_nhw_per_core = outputs[0].shard_spec()->shape[0]`, `:316`).

`padded_shape()[3]` itself is pinned by hashed values: the op requires a sharded input
(`pool_op.cpp:33`, checked on hits), so the padded channel count is recoverable from
`input.memory_config().shard_spec()->shape[1]` (hashed as part of `input_mem_config`) together with
`num_shards_c = sliding_window_config.num_cores_c` (hashed). The factory asserts precisely that
relationship (`:376-380`), and the op re-asserts it on every hit for the non-height-sharded cases
(`pool_op.cpp:56-63`).

`compute_output_specs` reinforces this: its `tensor_args` parameter is unnamed and unused
(`pool_op.cpp:87-88`), so the entire output spec — shape, padding, shard shape — is a function of
hashed attributes only.

### 4. `input_tensor_.tensor_spec().page_config()` and `.tensor_layout().get_alignment()`

**Verdict: VALID — pinned by validation, on a check that runs on hits.**

Neither is read by the factory, and the layout half is pinned outright:

```31:31:ttnn/cpp/ttnn/operations/pool/generic/device/pool_op.cpp
    TT_FATAL(input.layout() == Layout::ROW_MAJOR, "Only ROW_MAJOR supported for now. Tracked by issue #23338");
```

This is one of the few "pinned by validation" verdicts in this audit set that needs no qualification
about which validator runs when. `Pool2D` defines `validate_on_program_cache_hit`, and it is the
complete miss validator rather than a subset (`pool_op.cpp:66-85`, discussed under "The validation
split"), so the `TT_FATAL` above executes on every dispatch, hit or miss.

`Alignment` could in principle reach the program by way of `padded_shape()[3]`, but that value is
pinned by the hashed shard spec as argued above. Note that the *output* layout, which does shape the
program (`is_out_tiled`, `:627`; the pre-tilize CB pair, `:644-690`), is a separate hashed attribute
(`output_layout_`, `pool_op.cpp:175`), not the input's.

**Tile geometry specifically.** The `Tile` inside `page_config` deserves its own treatment, because
the surface pattern of the unguarded-32x32 defect is present: the factory does bare architectural
tile arithmetic and checks the tensor's real `tile().get_height()` / `get_width()` nowhere. The
instances are `tt::constants::TILE_WIDTH` at `pool_multi_core_program_factory.cpp:383-388` (the
per-shard channel rounding), `:521`, `:555`, `:636`, and `tt::tile_size(params.data_format)` at
`:656`, which is the fast-tilize CB page size and the one call that would have to become
`tile.get_tile_size(fmt)` in a tile-aware rewrite.

It is nonetheless not the defect, because no caller-supplied tile can reach this op on either side.

*Input side.* The `ROW_MAJOR` pin above means `page_config` always holds the row-major alternative,
and `PageConfig::get_tile()` returns a default-constructed `Tile{}` there:

```179:184:tt_metal/impl/tensor/spec/layout/page_config.cpp
Tile PageConfig::get_tile() const {
    if (const auto* tile_config = std::get_if<TilePageConfig>(&config_)) {
        return tile_config->tile;
    }
    return Tile{};
}
```

So the input carries no tile to disagree with, and — unlike an op that merely happens to have
compliant callers — this one would reject a non-32x32 tiled input loudly at the `TT_FATAL`, on every
dispatch.

*Output side.* The output can be `Layout::TILE`, and `is_out_tiled` genuinely reshapes the program.
But the output tensor is always constructed by the op; there is no preallocated-output parameter in
`tensor_args_t` (`pool_op.hpp:41-43`), and `create_output_tensors` builds both the data and index
tensors from `compute_output_specs` (`pool_op.cpp:147-166`). That spec names only a `Layout`:

```141:144:ttnn/cpp/ttnn/operations/pool/generic/device/pool_op.cpp
    return tt::tt_metal::TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout::fromPaddedShape(
            output_dtype, op_attr.output_layout_, mem_config, output_shape, padded_output_shape));
```

`fromPaddedShape` takes a `const PageConfig&`
(`tt_metal/api/tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp:48-53`), so
`op_attr.output_layout_` is implicitly converted through `PageConfig(Layout)`
(`page_config.hpp:34`), which supplies no tile and therefore the default 32x32. The output tile is
invariant by construction, and `output_layout_` — the only thing that varies — is hashed
(`pool_op.cpp:175`).

A repo-wide sweep files this op as an unguarded-32x32 case on the strength of the constants listed
above. The constants are real, and a future relaxation of the `ROW_MAJOR` input requirement or the
addition of a caller-supplied output tensor would turn them into exactly that defect. Today they are
not one, and the omission of `page_config` is correct. See recommendation 5.

### 5. `input_tensor_.storage` variant kind

**Verdict: VALID — pinned by validation.**

```28:29:ttnn/cpp/ttnn/operations/pool/generic/device/pool_op.cpp
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Pool2D input must be on device!");
    TT_FATAL(input.buffer() != nullptr, "Pool2D input must be allocated in buffers on device!");
```

### 6. Buffer addresses (omitted by the default hash too)

**Verdict: VALID — patched.**

The input shard, both outputs, and (on the L1 path) the reader-indices and scalar-config buffers are
all bound through `CBDescriptor::buffer` (`:429`), which `resolve_bindings` picks up at miss time
(`mesh_device_operation_adapter.hpp:556-561`) and `apply_resolved_bindings` refreshes on every hit
(`:651`). The WorkloadDescriptor branch passes `allow_inplace_output_tensor_alias=true`
unconditionally, with the reason spelled out in the source (`:549-555`): this branch has no
slow-path rebuild to fall back on, so it must never let `resolve_bindings` bail to an empty result.
Since `Pool2D` has exactly one input tensor there is no ambiguous-duplicate case to worry about.

The two compile-time-baked addresses (`:796`, `:798`) are addresses of workload-owned buffers whose
lifetime is the cache entry's (`:1147-1157`, `:1215-1217`), so they cannot go stale. The ownership
comment at `:1148-1154` explains why the *Tensor* rather than the `shared_ptr<MeshBuffer>` is
parked — `~Tensor` would force-free the device memory regardless of buffer refcount.

## The validation split

Worth recording as a positive finding, because it is what makes several verdicts above hold. The
dispatcher runs exactly one validator on a hit, and defining a hit hook *replaces* the miss
validator rather than supplementing it:

```262:266:ttnn/api/ttnn/device_operation.hpp
    if constexpr (HasValidateOnProgramCacheHit<mesh_device_operation_t>) {
        mesh_device_operation_t::validate_on_program_cache_hit(operation_attributes, tensor_args);
    } else {
        mesh_device_operation_t::validate_on_program_cache_miss(operation_attributes, tensor_args);
    }
```

For most ops that defines a hit hook, this means the hit path silently loses whatever the miss
validator checked and the hit validator does not. `Pool2D` is immune to that hazard by construction,
because both hooks call the same function:

```66:85:ttnn/cpp/ttnn/operations/pool/generic/device/pool_op.cpp
// Validation is the same for both cache hit and miss
static void validate_pool2d_operation(
    const Pool2D::operation_attributes_t& op_attr, const Pool2D::tensor_args_t& tensor) {
    validate_pool2d(
        tensor.input_tensor_,
        op_attr.pool_type_,
        op_attr.sliding_window_config_,
        op_attr.memory_config_,
        op_attr.divisor_override_,
        op_attr.return_indices_,
        op_attr.output_layout_);
}

void Pool2D::validate_on_program_cache_miss(const operation_attributes_t& op_attr, const tensor_args_t& tensor) {
    validate_pool2d_operation(op_attr, tensor);
}

void Pool2D::validate_on_program_cache_hit(const operation_attributes_t& op_attr, const tensor_args_t& tensor) {
    validate_pool2d_operation(op_attr, tensor);
}
```

For an op that can never rebuild, this is the right default: every input property the frozen program
assumes is re-checked on every dispatch. The `ceil_pad_hw` bug is not caught by it only because the
validator does not look at that field.

## Framework side effect of having a custom hash

```1012:1014:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
        if constexpr (requires { DeviceOperation::compute_program_hash(attrs, tensor_args); }) {
            return key;  // custom hash -> opt out beyond the op-identity prefix
        } else {
```

`ProgramCacheKey::canonical` collapses to the op type name, so a 64-bit collision becomes a wrong hit
rather than a rebuild. This matters more than usual for `Pool2D`, whose key includes a
`std::hash<std::string>` of a long config string and a full sharded `MemoryConfig`, and which has no
rebuild path to recover on. It is also the reason the `ceil_pad_hw` hole is unrecoverable: with the
default hash, a `ceil_pad_hw` difference would at least have a chance of showing up in the canonical
key comparison, since that key is built by reflection over the attributes rather than from
`to_string()`.

## Summary

| Omitted vs. default | Used by program? | Patched on hit? | Verdict |
|---|---|---|---|
| `sliding_window_config_.ceil_pad_hw` (via `to_string()`) | Yes — CT args, CB set, halo table and scalar table contents, output shape | No (nothing is) | **BUG** |
| `attrs.memory_used` | No — debug assertion only | n/a | VALID — unused |
| `input.logical_shape` | No (only `padded_shape()[3]`, pinned by hashed shard spec) | n/a | VALID — unused |
| `input.page_config` / `alignment` | No | n/a | VALID — pinned by validation (runs on hits) |
| `input.page_config`'s `Tile` specifically | The factory does bare 32x32 arithmetic, but no caller-supplied tile can reach it: the input is pinned `ROW_MAJOR` and the output tile is op-constructed | n/a | VALID — pinned by validation (input) / VALID — invariant (output) |
| `input.storage` kind | n/a | n/a | VALID — pinned by validation |
| Buffer addresses | Yes | Yes (`apply_resolved_bindings`) | VALID — patched |

**One program-cache correctness bug was found**, and it is the `ceil_pad_hw` omission in the shared
`SlidingWindowConfig` hash. The bug count is unchanged from the initial audit; the change is that
the verdict no longer rests on a judgement call. Nothing in the tree enforces the field with a
`TT_FATAL`, and `ttnn::prim::pool2d` accepts the config from the caller verbatim, so the bad
configuration is reachable without violating an enforced constraint. Every other omission is sound,
and this op is unusually well built for cache safety in two respects worth repeating: its hit-time
validator is the *same function* as its miss-time validator, so no check is lost on the hit path,
and its output spec is computed without reading the input tensor at all.

## Discrepancies with the CSV classification

None. The row (explicit hash, SELECTIVE tensor hashing, has its own cache-hit validator, no
`override_runtime_arguments`, no `get_dynamic_runtime_args`) matches the code exactly. Two
clarifications the row cannot express, both of which change how the classification should be read:

- "Has own cache-hit validator" understates it — the hit validator is the *complete* miss validator
  (`pool_op.cpp:83-85`), not a reduced subset, which is why several tensor-spec omissions are safe.
- "No `override_runtime_arguments`, no `get_dynamic_runtime_args`" is more severe here than for a
  `ProgramDescriptor` op with the same row, because the WorkloadDescriptor branch has no slow-path
  rebuild at all (`mesh_device_operation_adapter.hpp:642-644`). For this op the combination means
  *nothing but buffer addresses is ever refreshed*, including the contents of two host-generated
  device tensors.

## Recommendations

1. Add `ceil_pad_hw` to `SlidingWindowConfig::to_string()` as shown in omission 1. Make the fix in
   the shared type rather than locally in `Pool2D::compute_program_hash`, because `to_string()` is
   consumed by more than one caller — but be precise about which, since the affected ops and their
   failure modes differ:

   - **`pool/generic` (this op)** — `get_hash()` feeds `compute_program_hash`
     (`device/pool_op.cpp:173`), so a collision is a wrong program-cache hit. This is the bug in
     omission 1.
   - **Both halo variants** — `ttnn::prim::HaloDeviceOperation` calls `config.get_hash()` at
     `ttnn/cpp/ttnn/operations/sliding_window/halo/device/halo_device_operation.cpp:119`, and the
     Quasar copy does the same at
     `ttnn/cpp/ttnn/operations/experimental/quasar/halo/device/halo_device_operation.cpp:124`. Same
     root cause, different mechanism and different symptom: the string hash is the key into a
     thread-local memoization map for `max_out_nsticks_per_core`
     (`halo_device_operation.hpp:20`), not into the program cache. A collision returns a stick count
     memoized from `generate_shard_boundaries` on a *different* ceil pad
     (`halo_device_operation.cpp:120-127`), so the failure is a wrong shard-boundary derivation
     rather than a stale program. Worth fixing for the same one-line change, but anyone told "this
     also fixes halo" would otherwise look for it in the wrong place.
   - **conv2d is not affected.** The identifier `ceil_mode` does not appear anywhere under
     `ttnn/cpp/ttnn/operations/conv`; it exists only in the `sliding_window` and `pool` trees. Since
     `get_ceil_pad_hw()` short-circuits to `{0, 0}` whenever `ceil_mode` is false
     (`sliding_window.cpp:126-128`), the field is inert for conv2d whatever the string contains.
2. Add the missing case to the existing hash test, which is one flip short of catching this. It
   already asserts key injectivity for four fields and never touches the fifth:

```28:54:tests/ttnn/unit_tests/gtests/test_sliding_window_infra.cpp
    // flip snap_to_tile
    sliding_window_b.snap_to_tile = !sliding_window_a.snap_to_tile;
    log_info(tt::LogTest, "sliding_window_a:[{}] {}", sliding_window_a.get_hash(), sliding_window_a.to_string());
    log_info(tt::LogTest, "sliding_window_b:[{}] {}", sliding_window_b.get_hash(), sliding_window_b.to_string());
    EXPECT_NE(sliding_window_a.get_hash(), sliding_window_b.get_hash());
    sliding_window_b.snap_to_tile = !sliding_window_a.snap_to_tile;

    // ... the same block again for is_bilinear and is_transpose ...

    // flip ceil_mode
    sliding_window_b.ceil_mode = !sliding_window_a.ceil_mode;
    log_info(tt::LogTest, "sliding_window_a:[{}] {}", sliding_window_a.get_hash(), sliding_window_a.to_string());
    log_info(tt::LogTest, "sliding_window_b:[{}] {}", sliding_window_b.get_hash(), sliding_window_b.to_string());
    EXPECT_NE(sliding_window_a.get_hash(), sliding_window_b.get_hash());
    sliding_window_b.ceil_mode = !sliding_window_a.ceil_mode;
```

   A fifth block setting `ceil_pad_hw` and asserting `EXPECT_NE` fails today and passes after
   recommendation 1, which makes it both the reproduction and the regression net. It tests the
   defect directly — non-injectivity of the key over a program-shaping field — rather than through
   a full pool invocation, so it is a better answer to the reachability question than the end-to-end
   pair discussed under "Open questions".

   One detail matters when writing it. `get_ceil_pad_hw()` returns `{0, 0}` whenever `ceil_mode` is
   false (`sliding_window.cpp:126-128`), and the parameterised config the suite instantiates has
   `.ceil_mode = false` (`test_sliding_window_infra.cpp:74`). The new case must therefore set
   `ceil_mode = true` on both configs before varying `ceil_pad_hw`, or the accessor-based fix in
   recommendation 1 will produce identical strings and the assertion will fail even after the bug is
   gone.
3. Consider replacing the stringly-typed `get_hash()` with a real field-wise hash. Beyond the missing
   field, the current encoding is fragile in a second way: the trailing boolean and enum fields are
   encoded as presence/absence of a literal suffix (`"_ceil_mode"`, `"_replicate_pad"`, and the
   others at `sliding_window.cpp:1295-1297`), so a third `PaddingMode` encoded as the empty string
   would silently alias with `Zeros`. The comment at `sliding_window.hpp:17-18` already asks future
   authors to be careful here; a field-wise hash would remove the need to be careful.
4. Drop the narrowing in `std::hash<SlidingWindowConfig>` (`sliding_window.hpp:205`): `std::hash<int>()`
   on a `std::size_t` throws away half the bits for every consumer that goes through the default
   reflection hash rather than through `Pool2D`'s custom one.
5. Because this op can never rebuild, it would benefit from an assertion (debug builds) that the
   halo lookup table regenerated from the current attributes matches the parked one. That is the
   WorkloadDescriptor-path analog of `TT_DESCRIPTOR_PATCHING_PARITY_CHECK`
   (`mesh_device_operation_adapter.hpp:732-747`), which today only covers the `ProgramDescriptor`
   branches and so gives this op no regression net at all.
6. Add a tile guard, not because the op is broken today but because two of the three things keeping
   it correct are external to the factory. The factory's bare `TILE_WIDTH` / `TILE_HEIGHT`
   arithmetic and its `tt::tile_size(params.data_format)` call
   (`pool_multi_core_program_factory.cpp:383-388`, `:521`, `:555`, `:636`, `:656`) are safe only
   because the input is pinned to `ROW_MAJOR` (`pool_op.cpp:31`, a pin the comment itself marks as
   temporary — "Only ROW_MAJOR supported for now. Tracked by issue #23338") and because the output
   tensor is always op-constructed. Whichever of those changes first, the factory silently becomes a
   32x32-only program keyed without the tile. The canonical guard is three lines:

```95:97:ttnn/cpp/ttnn/operations/data_movement/sharded/interleaved_to_sharded/device/interleaved_to_sharded_op.cpp
        auto tile = input_tensor.tensor_spec().tile();
        if (tile.get_height() != tt::constants::TILE_HEIGHT || tile.get_width() != tt::constants::TILE_WIDTH) {
            return {false, fmt::format("interleaved_to_sharded requires standard 32x32 tiles, got {}x{}", tile.get_height(), tile.get_width())};
```

   Adding it to `validate_pool2d` costs nothing today (the condition cannot fire) and turns issue
   #23338 into a loud failure rather than a silent cache hole. The alternative — making the factory
   read `tensor_spec().tile()` — would require adding `page_config` to
   `Pool2D::compute_program_hash` in the same change.

## Open questions

- I could not construct a concrete pair of end-to-end `ttnn.max_pool2d` / `ttnn.avg_pool2d` calls
  that collide on `ceil_pad_hw` within a single process. This is a question about severity, not
  about the verdict: under the reachability rule an unenforced, publicly reachable, program-shaping
  field is a BUG without a demonstrated pair, and omission 1 confirms that is the situation here (no
  `TT_FATAL` anywhere constrains `ceil_pad_hw`, and `ttnn::prim::pool2d` takes the config from the
  caller verbatim). What remains open is only how easily the *high-level* wrappers can be driven
  into it.

  Two things narrow the question. First, the field is exercised in-tree rather than dormant:
  `ceil_pad` is a parameter threaded through the pool implementation — `pool2d_L1` takes it at
  `generic_pools.cpp:146` (file-local, so this is evidence of use rather than of public
  reachability; the public route is `ttnn::prim::pool2d`), the DRAM slicer computes a per-slice
  value (`:536`, `:545`, `:554`), `run_L1_op` forwards it (`:706`), and it lands in the config at
  `:182-184`. The analysis in omission 1 of when that explicit value
  diverges from what `get_ceil_pad_hw()` would derive — the `pad_right` tile-rounding rewrite at
  `:559-569` — identifies the specific geometry to aim at. The slice geometries I worked through by
  hand happened to keep `in_h` / `in_w` / `padding` in step with the ceil pad, which incidentally
  separates them in the hash.

  Second, an end-to-end pair is not the best evidence to chase. The defect is non-injectivity of
  `get_hash()` over a program-shaping field, and the existing unit test asserts exactly that
  property for four sibling fields already; adding a `ceil_pad_hw` case (recommendation 2) exercises
  the defect directly, fails today, passes after the fix, and does not depend on finding a slice
  geometry that happens to route around the incidental separation described above. An end-to-end
  pool test — run a `ceil_mode` pool through `pool2d_DRAM` with a tile-rounded output slice width,
  then a plain L1 pool with the resulting slice geometry and no explicit ceil pad, compared against
  golden — would additionally demonstrate the user-visible symptom, but it is the harder artifact
  and not the one that pins the bug.
