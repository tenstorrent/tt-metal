# Program Cache Audit — `eltwise/unary_backward/tanh_bw`

Audit of `ttnn::operations::unary_backward::tanh_bw::TanhBwDeviceOperation::compute_program_hash`
against the framework default ("hash everything") key.

| | |
|---|---|
| Device operation | `TanhBwDeviceOperation` (`device/tanh_bw_device_operation.hpp:20`) |
| Custom hash | `device/tanh_bw_device_operation.cpp:104-118` |
| `operation_attributes_t` | `TanhBwParams` — `output_dtype`, `output_memory_config` |
| `tensor_args_t` | `TanhBwInputs` — `grad_output`, `input`, `preallocated_input_grad` (`std::optional<Tensor>`) |
| Program factories | `TanhBwProgramFactory` (single, `ProgramDescriptor`-based) |
| `override_runtime_arguments` | **No** |
| `get_dynamic_runtime_args` | **No** |
| `validate_on_program_cache_hit` | **No** (so `validate_on_program_cache_miss` also runs on hits) |
| Cache-hit patch mechanism | Framework **buffer-binding fast path** |

**Result: three program-cache correctness BUGs.** The omitted preallocated output tensor
(omission 1), the omitted and unvalidated `grad_output.layout()` (omission 4b), and the unguarded
32x32 tile assumption (omission 5).

## Two scoping notes before the analysis

The task brief raised two concerns about this op that the code does not bear out; both are worth
stating explicitly because they change what needs auditing.

*This is not a shared `unary_backward` hash.* `tanh_bw` has its own dedicated device operation,
its own `operation_attributes_t`, and its own single-purpose program factory that hard-codes the
tanh-derivative compute kernel:

```122:126:ttnn/cpp/ttnn/operations/eltwise/unary_backward/tanh_bw/device/tanh_bw_program_factory.cpp
    KernelDescriptor compute_desc;
    compute_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary_backward/tanh_bw/device/"
        "kernels/compute/eltwise_bw_tanh_deriv.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
```

The hash is seeded with `type_hash<TanhBwDeviceOperation>` via
`operation::hash_operation<TanhBwDeviceOperation>`, and sibling backward ops that have their own
device operation (e.g. `gelu_bw`) are seeded with their own type hash. Two different
`unary_backward` ops therefore cannot collide on the identity prefix. The rest of
`unary_backward.cpp` is composite (built from `ttnn::multiply`, `ttnn::where`, …) and never reaches
this cache entry at all.

*There is no `approx_mode` / fast-and-approx flag on `tanh_bw`.* `TanhBwParams` has exactly two
fields (`device/tanh_bw_device_operation_types.hpp:12-15`), and the public entry point takes no
approximation argument:

```380:384:ttnn/cpp/ttnn/operations/eltwise/unary_backward/unary_backward.hpp
std::vector<std::optional<Tensor>> tanh_bw(
    const Tensor& grad_tensor_arg,
    const Tensor& input_tensor_arg,
    const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
    const std::optional<Tensor>& input_grad = std::nullopt);
```

The factory leaves `ComputeConfigDescriptor::math_approx_mode` at its default. The sibling that
*does* carry an approximation mode is `gelu_bw` (`unary_backward.cpp:1559`), a different device
operation with a different hash. Nothing to audit here.

## Cache-hit patch mechanism

The factory registers both input addresses and the output address as `Buffer*` entries through
`KernelDescriptor::emplace_runtime_args`, which auto-registers a `BufferBinding` at each position
(`tt_metal/api/tt-metalium/program_descriptors.hpp:110-118`, `190-194`):

```146:152:ttnn/cpp/ttnn/operations/eltwise/unary_backward/tanh_bw/device/tanh_bw_program_factory.cpp
        reader_desc.emplace_runtime_args(
            core, {src0_buffer, src1_buffer, num_tiles_per_core, num_tiles_written, 0u, 0u, num_cores_y});

        compute_desc.runtime_args.emplace_back(
            core, KernelDescriptor::CoreRuntimeArgs{num_tiles_per_core, 1});

        writer_desc.emplace_runtime_args(core, {dst_buffer, num_tiles_per_core, num_tiles_written});
```

`resolved_bindings.rt_args` is therefore non-empty, and since the op defines neither
`override_runtime_arguments` nor `get_dynamic_runtime_args`, the adapter takes the fast path:

```726:731:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
                    if (!sv.resolved_bindings.rt_args.empty() ||
                        (!dynamic_args.empty() && !sv.resolved_bindings.empty())) {
                        auto collected =
                            collect_tensor_buffers(tensor_args, tensor_return_value, sv.workload_descriptor);
                        tt::tt_metal::apply_resolved_bindings(program, sv.resolved_bindings, collected.buffers);
                        tt::tt_metal::apply_dynamic_runtime_args(program, dynamic_args);
```

**Obligation on the hash.** On a hit, exactly three things change: the three buffer addresses.
Every other runtime arg (`num_tiles_per_core`, `num_tiles_written`, `num_cores_y`) is frozen at the
first miss, and — critically for this op — **every compile-time arg is frozen too**, because
compile-time args are baked into the cached `Program` and no cache-hit mode in the framework
refreshes them. So every compile-time arg must be a pure function of the hashed set.

Two secondary points about this fast path:

- `resolve_bindings` is called with `allow_inplace_output_tensor_alias` at its default `false`
  (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:588-589`), so a call where `grad_output` and
  `input` are the same tensor produces a duplicate inside the input region, `resolve_bindings`
  returns an empty `ResolvedBindings`, and the op falls through to the slow-path rebuild. That is a
  *safer* mode, not a hazard.
- With `preallocated_input_grad` engaged, the same buffer appears once in the input region and once
  as the return value. That is the "output aliases an input" case, which is skipped rather than
  bailed (`tt_metal/impl/program/program_descriptor_patching.cpp:92-94`), so the fast path is kept.
  The recorded `tensor_buffer_idx` for `dst_buffer` happens to be 2 in both the preallocated and
  non-preallocated enumerations (the optional adds exactly one input-region entry, and the output
  entry it duplicates resolves back to it via `std::find`), so *address* patching stays correct in
  both shapes. The damage is entirely on the compile-time-arg side.

## Which validator runs on a cache hit

This decides several verdicts below, and it runs the opposite way to the intuitive reading, so it is
worth pinning down before the omissions. The dispatcher runs exactly one validator on a hit:

```262:266:ttnn/api/ttnn/device_operation.hpp
    if constexpr (HasValidateOnProgramCacheHit<mesh_device_operation_t>) {
        mesh_device_operation_t::validate_on_program_cache_hit(operation_attributes, tensor_args);
    } else {
        mesh_device_operation_t::validate_on_program_cache_miss(operation_attributes, tensor_args);
    }
```

`TanhBwDeviceOperation` declares no `validate_on_program_cache_hit`
(`device/tanh_bw_device_operation.hpp:20-33` declares only the miss variant, at `:27`), so the framework
substitutes `validate_on_program_cache_miss` on **every** hit. Every `TT_FATAL` in that function
therefore executes on the offending call, and a "pinned by validation" verdict resting on one of
them is legitimate rather than miss-only. The CSV's `own_hit_validator = N` is, on this framework,
the *safer* of the two rows.

The corollary matters just as much: anything the miss validator does **not** check is unpinned on
both paths. `grad_output` is checked nowhere in it — no storage check, no layout check, no shape
check against `input` — which is what turns omission 4b below into a bug rather than a caveat.

## Baseline: what the default hash would cover

`hash_objects_with_default_seed(type_hash<TanhBwDeviceOperation>, attrs, tensor_args)` walks
reflection over both aggregates, so the default key is:

| Source | Fields |
|---|---|
| `operation_attributes` | `output_dtype`, `output_memory_config` |
| `grad_output` | storage variant kind; `logical_shape`; `dtype`; `page_config`; `memory_config`; `alignment` |
| `input` | storage variant kind; `logical_shape`; `dtype`; `page_config`; `memory_config`; `alignment` |
| `preallocated_input_grad` | engaged/disengaged, and if engaged the same six fields again |

## What the custom hash covers

```104:118:ttnn/cpp/ttnn/operations/eltwise/unary_backward/tanh_bw/device/tanh_bw_device_operation.cpp
ttsl::hash::hash_t TanhBwDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& grad_output = tensor_args.grad_output;
    const auto& input_shape = input_tensor.padded_shape();
    operation::Hash hash = operation::hash_operation<TanhBwDeviceOperation>(
        args,
        input_tensor.dtype(),
        input_tensor.memory_config(),
        grad_output.dtype(),
        grad_output.memory_config(),
        input_shape.volume());

    return hash;
}
```

`args` is passed whole, so both `operation_attributes_t` fields survive. The two input tensors are
decomposed selectively. `preallocated_input_grad` does not appear anywhere.

## Omitted parameters

### 1. `tensor_args.preallocated_input_grad` — the entire optional output tensor

**Verdict: BUG.**

The preallocated output tensor determines the `dst_buffer` that the writer kernel's
`TensorAccessorArgs` are built from, and those are **compile-time** args:

```89:90:ttnn/cpp/ttnn/operations/eltwise/unary_backward/tanh_bw/device/tanh_bw_program_factory.cpp
    std::vector<uint32_t> writer_compile_time_args = {static_cast<uint32_t>(output_cb_index)};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);
```

`TensorAccessorArgs` derives its `ArgConfig` from the buffer, and `ArgConfig::IsDram` is set from
`buffer_->is_dram()`:

```146:158:tt_metal/impl/buffers/tensor_accessor_args.cpp
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

and both the raw config word and the buffer's aligned page size are emitted as compile-time args:

```196:205:tt_metal/impl/buffers/tensor_accessor_args.cpp
    } else {
        compile_time_args.push_back(args_config_.raw());
        auto aligned_page_size = buffer_ ? buffer_->aligned_page_size() : 0;
        TT_FATAL(
            aligned_page_size <= std::numeric_limits<uint32_t>::max(),
            "Aligned page size {} exceeds uint32_t max {}",
            aligned_page_size,
            std::numeric_limits<uint32_t>::max());
        compile_time_args.push_back(static_cast<uint32_t>(aligned_page_size));
    }
```

The writer kernel reads that word back and builds its `TensorAccessor` from it:

```20:36:ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr auto dst_args = TensorAccessorArgs<1>();

    // Get page size from CB interface (works for both TILE and ROW_MAJOR layouts)
    const uint32_t page_bytes = get_local_cb_interface(cb_id_out).fifo_page_size;

    Noc noc;
    DataflowBuffer dfb(cb_id_out);

#ifdef OUT_SHARDED
    dfb.wait_front(num_pages);
#else

    // single-page ublocks (works for both TILE and ROW_MAJOR layouts)
    constexpr uint32_t onepage = 1;

    const auto s = TensorAccessor(dst_args, dst_addr);
```

So the *buffer type of the output* is a compile-time property of the cached program. Now note that
the caller never folds the preallocated tensor's memory config into `output_memory_config`:

```292:305:ttnn/cpp/ttnn/operations/eltwise/unary_backward/unary_backward.cpp
std::vector<std::optional<Tensor>> tanh_bw(
    const Tensor& grad,
    const Tensor& input,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<Tensor>& input_grad) {
    std::vector<std::optional<Tensor>> grad_tensor;

    DataType output_dtype = input.dtype();
    auto output_memory_config = output_mem_config.value_or(input.memory_config());
    auto result_tensor = ttnn::operations::unary_backward::tanh_bw::launch_tanh_bw(
        grad, input, output_dtype, output_memory_config, input_grad);
    grad_tensor.emplace_back(result_tensor);
    return grad_tensor;
}
```

`input_grad` is not consulted when building `output_memory_config` (contrast `gelu_bw` at
`unary_backward.cpp:1567-1568`, which *does* use `input_grad->memory_config()`). And validation only
compares the memory *layout*, not the buffer type:

```44:48:ttnn/cpp/ttnn/operations/eltwise/unary_backward/tanh_bw/device/tanh_bw_device_operation.cpp
    TT_FATAL(
        input_tensor.memory_config().memory_layout() == out_memory_config.memory_layout(),
        "TANH_BW operation requires Input and Output memory layout to match. Input layout: {}, Output layout: {}",
        input_tensor.memory_config().memory_layout(),
        out_memory_config.memory_layout());
```

**Two-call reproduction.** Let `grad` and `input` both be DRAM-interleaved, `TILE`, `bfloat16`,
shape `[1, 1, 32, 32]`.

- **Call 1**: `ttnn::tanh_bw(grad, input)` — no `input_grad`. `args = {output_dtype = BFLOAT16,
  output_memory_config = DRAM interleaved}`. `create_output_tensors` allocates a DRAM output.
  The writer is compiled with `ArgConfig::IsDram` set.
- **Call 2**: `ttnn::tanh_bw(grad, input, /*output_mem_config=*/std::nullopt,
  /*input_grad=*/ttnn::empty_like(input, ttnn::L1_MEMORY_CONFIG))` — an L1-interleaved
  preallocated output of identical shape and dtype. `args` is byte-identical to call 1 (the caller
  still derives `output_memory_config` from `input`), and `preallocated_input_grad` is not in the
  hash, so **the hash is identical** and the cache hits.
- Validation does not stop it. The op has no `validate_on_program_cache_hit`, so
  `validate_on_program_cache_miss` runs on the hit
  (`ttnn/api/ttnn/device_operation.hpp:262-266`); it recomputes `out_memory_config` from the
  preallocated tensor, checks dtype equality (passes, both BFLOAT16), checks
  `memory_layout` equality (passes, `INTERLEAVED == INTERLEAVED`), and checks the logical shape
  (passes).
- **What goes stale**: writer compile-time arg 1 (`args_config_.raw()`, with `IsDram` set) and
  compile-time arg 2 (`aligned_page_size`). The `dst_buffer` *address* is patched correctly by
  `apply_resolved_bindings`, but the kernel resolves that address through the DRAM bank map.
- **Symptom**: the writer issues NOC writes to DRAM banks using an L1-relative address. The
  returned `input_grad` tensor contains uninitialised L1 contents, and unrelated DRAM is
  overwritten. Silent wrong results, not a crash.

The mirror case (call 1 with an L1 preallocated output, call 2 with none / a DRAM one) fails the
same way. A third variant is worse: nothing validates the preallocated tensor's **layout**, so a
`ROW_MAJOR` preallocated output changes `aligned_page_size` (row bytes instead of tile bytes) under
the same hash.

This is exactly the aliasing class the framework's parity oracle is designed to catch — but
`assert_fastpath_parity`
(`tt_metal/api/tt-metalium/experimental/program_descriptor_patching.hpp:191-192`) only compares
runtime args and CB addresses, not compile-time args, so it would *not* flag this one. The fix has
to be in the hash.

### 2. `tensor_args.input.logical_shape()` — replaced by `padded_shape().volume()`

**Verdict: VALID — relaxation win.**

`padded_shape().volume()` is exactly `physical_volume()`
(`ttnn/core/tensor/tensor.cpp:438`: `uint64_t Tensor::physical_volume() const { return
padded_shape().volume(); }`), and the physical volume is the only shape-derived quantity the
factory reads:

```30:37:ttnn/cpp/ttnn/operations/eltwise/unary_backward/tanh_bw/device/tanh_bw_program_factory.cpp
    uint32_t num_tiles = input.physical_volume() / tt::constants::TILE_HW;

    IDevice* device = input.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, num_tiles);
```

Everything downstream — the core set, the per-core tile counts, the `num_tiles_written` prefix
offsets — is a function of `num_tiles` and the (device-fixed, per-device-cache) compute grid. The
CB sizes depend only on dtype. Nothing reads a specific dimension.

Hashing the scalar volume rather than the shape is strictly better than the default: `[1,1,32,64]`
and `[1,1,64,32]` (and `[1,1,1,2048]`, and any other tile-aligned rearrangement of 2 tiles)
legitimately share one program, where the default key would force a recompile for each. The output
`TensorSpec` still carries the correct per-call logical shape because `compute_output_specs` runs on
every invocation.

### 3. `tensor_args.grad_output.logical_shape()` / `padded_shape()` — omitted entirely

**Verdict: VALID — unused.**

The factory reads nothing from `grad_output` except its dtype (for the src0 CB format and page
size) and its buffer (for the accessor args and the address binding). The work split comes solely
from `input.physical_volume()`. So two calls whose `grad_output` shapes differ but whose `input`
shapes agree genuinely produce the same descriptor, and sharing a cache entry is correct.

There is a separate defect in the same area, recorded here so it is not confused with a cache
finding: `validate_on_program_cache_miss` never checks that `grad_output` and `input` have the same
shape, yet the reader is told to fetch `num_tiles_per_core` pages from both. That is a plain
validation gap — a mismatched `grad_output` is broken with or without a program cache, and no hash
change would fix it — so it is not counted among the bugs below. It still wants a `TT_FATAL`; see
recommendation 3. The layout of `grad_output` is a different matter and *is* a cache bug, because
layout does reach a compile-time arg; that is omission 4b.

### 4a. `input.layout()` / `page_config` — the coarse layout

**Verdict: VALID — pinned by validation.**

Neither tensor's `page_config` (nor even the coarse `layout()`) appears in the hash. For `input`'s
layout that is safe, because validation pins it to `TILE` and, per the substitution branch above,
that `TT_FATAL` runs on hits as well as misses:

```52:56:ttnn/cpp/ttnn/operations/eltwise/unary_backward/tanh_bw/device/tanh_bw_device_operation.cpp
    TT_FATAL(
        input_tensor.layout() == Layout::TILE,
        "TANH_BW operation requires tensor to be in Tile layout when working with non-sharded input tensor. Input "
        "tensor layout: {}",
        input_tensor.layout());
```

This verdict covers only the `ROW_MAJOR` / `TILE` discriminator. The `Tile` *inside* the tile page
config is a separate omission and gets the opposite verdict — see omission 5.

### 4b. `grad_output.layout()` / `page_config`

**Verdict: BUG.**

There is no check on `grad_output` anywhere in `validate_on_program_cache_miss`
(`device/tanh_bw_device_operation.cpp:13-74`): the function reads `tensor_args.input` and
`tensor_args.preallocated_input_grad` and never touches `tensor_args.grad_output`. So a `ROW_MAJOR`
gradient is reachable through the public API — `ttnn::tanh_bw(grad, input, ...)` accepts any
`Tensor` for `grad` (`unary_backward.hpp:380-384`) — without violating any enforced constraint.
Under the reachability rule that makes it a bug, not a caveat. The fact that no in-tree caller
passes a row-major gradient today is severity context, not a defence.

`grad_output`'s layout reaches the program through the reader's compile-time accessor args:

```85:87:ttnn/cpp/ttnn/operations/eltwise/unary_backward/tanh_bw/device/tanh_bw_program_factory.cpp
    std::vector<uint32_t> reader_compile_time_args = {0};
    TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*src1_buffer).append_to(reader_compile_time_args);
```

`src0_buffer` is `grad_output.buffer()` (`tanh_bw_program_factory.cpp:45`), and `append_to` emits
`buffer_->aligned_page_size()` as the second compile-time word
(`tt_metal/impl/buffers/tensor_accessor_args.cpp:196-205`, quoted under omission 1). For a `TILE`
bfloat16 tensor that page size is the 2048-byte tile; for a `ROW_MAJOR` tensor of the same logical
shape it is the row stride.

**Two-call reproduction.** `input` DRAM-interleaved `TILE` bfloat16 `[1, 1, 32, 32]` in both calls.

- **Call 1**: `grad` also DRAM-interleaved `TILE` bfloat16 `[1, 1, 32, 32]`. Reader compile-time
  arg 2 is `2048`.
- **Call 2**: identical except `grad = ttnn::to_layout(grad, ttnn::ROW_MAJOR_LAYOUT)`. The hash
  inputs are `args` (unchanged), `input.dtype()`, `input.memory_config()`, `grad_output.dtype()`
  (unchanged — layout is not dtype), `grad_output.memory_config()` (unchanged — layout is not part
  of `MemoryConfig`) and `input.padded_shape().volume()` (unchanged, `input` was not touched). The
  hash is identical and the cache hits.
- **What goes stale**: reader compile-time arg 2, still `2048`, against a buffer whose real page is
  64 bytes.
- **Symptom**: the reader computes `page_id * 2048` offsets into a buffer laid out in 64-byte pages,
  reading 32x past the end of the allocation. Garbage gradients, or a NOC fault on the last page.

The mirror ordering (row-major first) fails the same way with the page size too small.

Note the compounding: because `grad_output.layout()` is neither hashed nor validated, call 2 does
not even get a freshly built (still-wrong) program — it silently inherits call 1's. Adding a
`TT_FATAL` on `grad_output.layout() == Layout::TILE` closes it completely and is the better fix,
since a row-major gradient is not something the compute kernel supports at any hash.

### 5. The `Tile` inside `page_config` — the unguarded 32x32 assumption

**Verdict: BUG.**

The factory never reads the tensor's actual tile. It sizes every circular buffer from
`tt::tile_size(...)`, which returns the byte size of a **32x32** tile, and it converts the physical
volume into a tile count with a bare `TILE_HW`:

```23:30:ttnn/cpp/ttnn/operations/eltwise/unary_backward/tanh_bw/device/tanh_bw_program_factory.cpp
    tt::DataFormat src0_cb_data_format = datatype_to_dataformat_converter(input.dtype());
    uint32_t src0_single_tile_size = tt::tile_size(src0_cb_data_format);
    tt::DataFormat src1_cb_data_format = datatype_to_dataformat_converter(grad_output.dtype());
    uint32_t src1_single_tile_size = tt::tile_size(src1_cb_data_format);
    tt::DataFormat dst_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);

    uint32_t num_tiles = input.physical_volume() / tt::constants::TILE_HW;
```

The tile-aware equivalents are `tile.get_tile_size(data_format)` and
`tensor_spec().tile().get_tile_shape()`; neither appears anywhere in this op. All three conditions
for the defect hold:

1. **The op accepts (indeed requires) `Layout::TILE`** — `device/tanh_bw_device_operation.cpp:52-56`,
   quoted above.
2. **Host-side code does bare 32x32 tile math** — `tanh_bw_program_factory.cpp:24`, `:26`, `:28`,
   `:30`, feeding `total_size` and `page_size` on all three CB descriptors (`:53-81`) and the work
   split (`:36-37`).
3. **Nothing validates the tile geometry** — `validate_on_program_cache_miss`
   (`device/tanh_bw_device_operation.cpp:13-74`) contains no reference to `tile()`,
   `get_height()`, or `get_width()`.

Non-32x32 tiles are a supported TTNN configuration and are constructible straight from Python:

```220:226:ttnn/cpp/ttnn-nanobind/tensor.cpp
    py_tile
        .def(nb::init<const std::array<uint32_t, 2>&, bool>(), nb::arg("tile_shape"), nb::arg("transpose_tile") = false)
        .def(
            "__init__",
            [](Tile* t, const std::array<uint32_t, 2>& tile_shape, bool transpose_tile = false) {
                new (t) Tile{tile_shape, transpose_tile};
            })
```

**Two-call reproduction.** Both calls DRAM-interleaved `TILE` bfloat16, logical shape
`[1, 1, 64, 64]`, no preallocated output. 64 is divisible by both 32 and 16, so the padded shape —
and therefore `padded_shape().volume()`, the only shape term in the hash — is `4096` in both.

- **Call 1**: default `Tile{32, 32}`. `num_tiles = 4096 / 1024 = 4`; each CB page is
  `tt::tile_size(Float16_b) = 2048` bytes.
- **Call 2**: identical tensors built with `Tile{16, 32}`. `dtype`, `memory_config` and the padded
  volume are unchanged, and `page_config` is not in the hash, so the hash is identical and the cache
  hits.
- **What goes stale**: everything derived from the tile. The correct tile count for call 2 is
  `(64/16) * (64/32) = 8`, but the cached program moves `4`; the correct page size is
  `16 * 32 * 2 = 1024` bytes, but the cached CBs are `2048`.
- **Symptom**: half the tensor is never read or written, and each CB page straddles two real tiles,
  so the compute kernel's face addressing is wrong for every tile it does process. Silent wrong
  results with no cache miss to hint at the cause.

This is the compounding case: the factory bug (hardcoded 32x32) and the hash bug (no `page_config`)
each make the other worse. Fixing only the hash would produce a freshly built program that is still
wrong; fixing only the factory without adding `page_config` to the hash would produce a correct
program for call 1 that call 2 then reuses. The cheap fix is the guard — see recommendation 5.

Two related points, both framework-wide rather than op-local, recorded so the verdict is not
over-read. First, `Tile::attribute_values()` exposes only `tile_shape`, `face_shape` and `num_faces`
(`tt_metal/api/tt-metalium/tile.hpp:46-47`), and `Tile::operator==` compares only `tile_shape` and
`face_shape` (`tt_metal/impl/data_format/tile.cpp:122-124`), so `transpose_within_face` and
`transpose_of_faces` are invisible to the hash *and* to canonical-key collision resolution for every
op in the repo. Hashing `page_config` would therefore not close the transpose variant of this hole;
only an explicit `TT_FATAL` on `get_transpose_within_face()` / `get_transpose_of_faces()` would.
Second, the guard recommended below closes the whole family at once, which is why it is preferable
to hashing `page_config`.

### 6. `input.tensor_layout().get_alignment()` and `grad_output`'s alignment

**Verdict: VALID — unused** (only reachable through hashed derivatives; low residual risk).

`Alignment` reaches the program through two paths, both already covered. It is one of the inputs to
`padded_shape` — and `padded_shape().volume()` is hashed explicitly. And it feeds
`Buffer::aligned_page_size()`, which is the second `TensorAccessorArgs` compile-time word; for the
interleaved tile-layout tensors this op accepts, the page size is the dtype-determined tile size
and the alignment applied is the HAL DRAM/L1 constant selected by `buffer_type()`, which lives
inside the hashed `memory_config`. The residual exposure is a `TensorLayout` constructed with an
explicit non-canonical `Alignment` that leaves the padded volume unchanged; nothing on this op's
call path produces one.

### 7. `input.storage` / `grad_output.storage` variant kind (device vs host)

**Verdict: VALID — pinned by validation.**

```35:42:ttnn/cpp/ttnn/operations/eltwise/unary_backward/tanh_bw/device/tanh_bw_device_operation.cpp
    TT_FATAL(
        input_tensor.storage_type() == StorageType::DEVICE,
        "TANH_BW operation requires input to be on Device. Input storage type: {}",
        input_tensor.storage_type());

    TT_FATAL(
        input_tensor.buffer() != nullptr,
        "Operands to TANH_BW need to be allocated in buffers on the device. Buffer is null.");
```

`grad_output` is not explicitly checked, but the factory dereferences `grad_output.buffer()`
unconditionally, so a host-storage gradient faults before it can reach the cache. Constant across
every admissible call, so it carries no information.

### 8. Buffer addresses (omitted by the default hash too)

**Verdict: VALID — patched, and required.**

Addresses must not be hashed. All three are registered as `Buffer*` bindings and re-applied on
every hit by `apply_resolved_bindings`
(`tt_metal/impl/program/program_descriptor_patching.cpp:262`). The set of active cores and the
per-core work split are functions of the hashed `padded_shape().volume()`, so an entry can never be
reused with a different core set — which is what makes the recorded
`(kernel_idx, core, arg_idx)` binding positions valid across hits.

## Keys the custom hash adds beyond the default

- `input.padded_shape().volume()` — the default key contains `logical_shape` plus the ingredients of
  padding (`page_config`, `alignment`) but not the padded volume itself. Hashing the volume directly
  is what makes dropping `logical_shape` safe *and* is what buys the shape-rearrangement relaxation
  in omission 2.

Nothing else. Note the hash contains no `program_factory.index()` because there is only one factory.

## Framework side effect of having a custom hash

Defining `compute_program_hash` opts this op out of attribute-level hash-collision resolution:

```1012:1014:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
        if constexpr (requires { DeviceOperation::compute_program_hash(attrs, tensor_args); }) {
            return key;  // custom hash -> opt out beyond the op-identity prefix
        } else {
```

`ProgramCacheKey::canonical` degrades to just the op type name, so a 64-bit hash collision between
two different `tanh_bw` configurations resolves to a wrong hit rather than a rebuild. This is
inherent to every custom-hash op, but it raises the cost of the gap in omission 1: there is no
second line of defence.

## Summary

| Omitted vs. default | Used by program? | Patched on hit? | Verdict |
|---|---|---|---|
| `preallocated_input_grad` (whole optional) | Yes — writer `TensorAccessorArgs` compile-time args | Address only; compile-time args never | **BUG** |
| `input.logical_shape` | No (padded volume used instead) | n/a | VALID — relaxation win |
| `grad_output.logical_shape` / `padded_shape` | No | n/a | VALID — unused (the missing shape `TT_FATAL` is a validation gap, not a cache one) |
| `input.layout()` (`ROW_MAJOR` vs `TILE`) | Yes — accessor page size | n/a | VALID — pinned by validation |
| `grad_output.layout()` / `page_config` | Yes — reader `aligned_page_size` compile-time arg | No | **BUG** |
| `page_config`'s `Tile` (both inputs and the output) | Yes — CB page sizes and the tile count, all hardcoded 32x32 | No | **BUG** |
| `input` / `grad_output` alignment | Only via hashed derivatives | n/a | VALID — unused (low residual risk) |
| `input` / `grad_output` storage kind | n/a | n/a | VALID — pinned by validation |
| Buffer addresses | Yes | Yes (`resolved_bindings`) | VALID — patched |

**Three program-cache correctness bugs were found**, all of the same shape: a tensor property that
lands in a compile-time arg, is absent from the hash, and is not pinned by a `TT_FATAL`.

1. Omitting `preallocated_input_grad` lets two calls that differ in the *buffer type* (or layout) of
   the caller-supplied output tensor share one cached program, while the output buffer type is baked
   into the writer kernel's compile-time `TensorAccessorArgs`.
2. Omitting `grad_output.layout()` — which is also validated nowhere — lets a row-major gradient
   inherit a program compiled with a tile-sized page in the reader's accessor args.
3. Omitting `page_config` while the factory hardcodes 32x32 tile arithmetic lets a `Tile{16, 32}`
   tensor inherit the 32x32 program wholesale.

Because the buffer-binding fast path patches addresses but never compile-time args — and neither
does the slow-path rebuild — nothing recovers from any of these at dispatch. What remains sound:
every non-address runtime arg and every other compile-time arg is a function of {`output_dtype`,
`output_memory_config`, `input.dtype`, `input.memory_config`, `grad_output.dtype`,
`grad_output.memory_config`, `input.padded_shape().volume()`} plus device-fixed constants, and the
op's lack of a hand-written hit validator means its miss-time `TT_FATAL`s do run on every hit.

## Recommendations

1. Fix omission 1 by hashing the preallocated output's program-relevant spec. The minimal change is to
   add `tensor_args.preallocated_input_grad.has_value()` plus, when engaged, its `memory_config()`,
   `dtype()` and `layout()` to `compute_program_hash`. Hashing
   `compute_output_specs(args, tensor_args)` would cover all of it in one term and would stay
   correct if `compute_output_specs` grows.
2. Alternatively (or additionally) fix the front end: make `ttnn::tanh_bw` derive
   `output_memory_config` from `input_grad->memory_config()` when a preallocated output is supplied,
   the way `gelu_bw` already does at `unary_backward.cpp:1567-1568`. That folds the buffer type back
   into the hashed `args` and makes the two paths consistent. It does not cover the layout case, so
   pair it with recommendation 3.
3. Fix omission 4b by adding `TT_FATAL(grad_output.layout() == Layout::TILE, ...)` to
   `validate_on_program_cache_miss`, alongside the `grad_output` checks that are missing entirely:
   `storage_type() == StorageType::DEVICE`, `buffer() != nullptr`, and
   `grad_output.padded_shape() == input.padded_shape()`. Because this op has no hit validator, every
   check added there runs on hits too (`ttnn/api/ttnn/device_operation.hpp:262-266`), so a `TT_FATAL`
   is a complete fix and not merely a miss-path one.
4. Fix omission 5 with the standard tile guard rather than by hashing `page_config`. The canonical
   form is:

```95:97:ttnn/cpp/ttnn/operations/data_movement/sharded/interleaved_to_sharded/device/interleaved_to_sharded_op.cpp
        auto tile = input_tensor.tensor_spec().tile();
        if (tile.get_height() != tt::constants::TILE_HEIGHT || tile.get_width() != tt::constants::TILE_WIDTH) {
            return {false, fmt::format("interleaved_to_sharded requires standard 32x32 tiles, got {}x{}", tile.get_height(), tile.get_width())};
```

   Apply it to `input`, `grad_output` and, when engaged, `preallocated_input_grad`. This makes
   omitting `page_config` correct by construction and is a few lines. The alternative — making the
   factory tile-aware by switching to `tile.get_tile_size(fmt)` and `tile().get_tile_shape()` —
   requires adding `page_config` to the hash *in the same change*, or the mirror-image bug appears
   (a genuinely tile-varying program keyed without the tile). Whichever route is taken, note that
   neither closes the transpose flags, which no hash can reach
   (`tt_metal/api/tt-metalium/tile.hpp:46-47`); only an explicit check on
   `get_transpose_within_face()` / `get_transpose_of_faces()` does.
5. Build this op under `-DTT_DESCRIPTOR_PATCHING_PARITY_CHECK` in CI, and consider extending
   `assert_fastpath_parity` to diff compile-time args and CB page sizes as well as runtime args and
   CB addresses. The check as written would already catch omission 5, because the tile count reaches
   the `num_tiles_per_core` runtime arg it does compare. It would miss omissions 1 and 4b entirely,
   since both go stale only in compile-time accessor args
   (`tt_metal/api/tt-metalium/experimental/program_descriptor_patching.hpp:191-192`) — and that is
   the dominant failure mode for descriptor factories that pass a `Buffer*` into
   `TensorAccessorArgs`, so the extension is worth more than this one op.
