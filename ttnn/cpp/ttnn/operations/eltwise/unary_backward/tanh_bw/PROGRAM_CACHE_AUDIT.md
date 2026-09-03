# Program Cache Audit — `eltwise/unary_backward/tanh_bw`

Audit of `ttnn::operations::unary_backward::tanh_bw::TanhBwDeviceOperation::compute_program_hash`
against the framework default ("hash everything") key.

| | |
|---|---|
| Device operation | `TanhBwDeviceOperation` (`device/tanh_bw_device_operation.hpp:20`) |
| Custom hash | `device/tanh_bw_device_operation.cpp:156-180` |
| `operation_attributes_t` | `TanhBwParams` — `output_dtype`, `output_memory_config` |
| `tensor_args_t` | `TanhBwInputs` — `grad_output`, `input`, `preallocated_input_grad` (`std::optional<Tensor>`) |
| Program factories | `TanhBwProgramFactory` (single, `ProgramDescriptor`-based) |
| `override_runtime_arguments` | **No** |
| `get_dynamic_runtime_args` | **No** |
| `validate_on_program_cache_hit` | **No** (so `validate_on_program_cache_miss` also runs on hits) |
| Cache-hit patch mechanism | Framework **buffer-binding fast path** |

## Post-fix status — commit fab067a

**Verdict: NOT CLEAR.** All three bugs this document originally recorded are closed, but one
category-1 bug remains, and it is a *narrower* version of the one the pre-fix audit cleared: when
`grad_output` is **sharded**, its page geometry (`tensor_shape_in_pages`, and the whole distribution
geometry) is baked into the reader's compile-time `TensorAccessorArgs`, while the hash keys only
`input.padded_shape().volume()` — a scalar. Two calls whose padded shapes are different
rearrangements of the same volume therefore share a program with stale sharded accessor args. See
new omission 3b. `grad_output.memory_config()` *is* hashed, so the shard spec agrees between the two
calls; it is the tensor shape in pages that does not, and that is not derivable from a volume.

What `fab067a` changed in this op:

- Added a `require_standard_tile` lambda to `validate_on_program_cache_miss` and applied it to
  `input`, `grad_output` and the preallocated output
  (`device/tanh_bw_device_operation.cpp:72-87`, `:107`). It enforces both `Layout::TILE` and a 32x32
  tile, which closes omissions 4b and 5. Because this op declares **no**
  `validate_on_program_cache_hit`, the framework substitutes the miss validator on hits
  (`ttnn/api/ttnn/device_operation.hpp:265-269`), so these `TT_FATAL`s fire on the offending second
  call rather than only on the first.
- Added the preallocated output's `dtype()`, `layout()` and `memory_config()` to
  `compute_program_hash` when it is engaged (`device/tanh_bw_device_operation.cpp:173-177`), closing
  omission 1.
- Added the `grad_output` checks the miss validator was missing entirely — `storage_type()`,
  `buffer() != nullptr`, and `padded_shape() == input.padded_shape()`
  (`device/tanh_bw_device_operation.cpp:92-103`) — which closes the validation gap recorded under
  omission 3 (now 3a).
- Replaced the preallocated-output shape check, which compared against
  `compute_output_specs(args, tensor_args)` — a tautology, since that function *returns* the
  preallocated tensor's own spec — with logical- and padded-shape comparisons against `input`
  (`device/tanh_bw_device_operation.cpp:108-124`).
- Documented the tile requirement on the Python binding (`unary_backward_nanobind.cpp:1145-1151`).
- Left the program factory untouched; it still does bare 32x32 arithmetic
  (`device/tanh_bw_program_factory.cpp:23-30`), which is now consistent with the validator.

What remains open:

- **Omission 3b (BUG).** Nothing rejects a sharded `grad_output`, and nothing keys its page
  geometry. `TT_FATAL(!input_tensor.is_sharded(), ...)`
  (`device/tanh_bw_device_operation.cpp:53`) covers `input` only, and the new
  `padded_shape() == input.padded_shape()` check pins `grad_output`'s shape to `input`'s — whose
  *volume*, not shape, is what the key carries.
- The volume-only shape key is now the single load-bearing relaxation in this op, and it is the one
  the remaining bug rides on.

**Metal 2.0 port: not clear until 3b is closed.** The relaxation here is not logical-vs-padded, so
`match_padded_shape_only` does not express it: that flag keeps `padded_shape` pertinent
(`tt_metal/impl/metal2_host_api/tensor_spec_relaxations.cpp:77-79`), whereas this op drops the shape
in favour of its volume, which no `TensorSpecRelaxations` field can express
(`tt_metal/api/tt-metalium/experimental/metal2_host_api/tensor_spec_relaxations.hpp:41`). The closest
flag is `dynamic_tensor_shape`, and it is instructive about the bug rather than a way to keep it: under
that flag a sharded argument's distribution geometry stays load-bearing and a mismatching argument is
**rejected** rather than silently mis-addressed (`tensor_spec_relaxations.hpp:68-74`), because the
squeeze that produces the geometry depends on the shape values. So the port must narrow the key —
either hash `input.padded_shape()` instead of the volume, or reject sharded gradients — and then
declare `match_padded_shape_only` to keep the logical-shape freedom it actually needs.

**Result: one program-cache correctness BUG.** The sharded-`grad_output` page geometry keyed only by
`input`'s padded volume (omission 3b). The three the pre-fix audit found — the omitted preallocated
output tensor (omission 1), the omitted and unvalidated `grad_output.layout()` (omission 4b), and the
unguarded 32x32 tile assumption (omission 5) — are all resolved by `fab067a`.

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

```265:269:ttnn/api/ttnn/device_operation.hpp
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
the *safer* of the two rows. This is what makes every guard `fab067a` added a complete fix rather
than a miss-path one, and it is why **any future `validate_on_program_cache_hit` on this op must call
the miss validator or duplicate its checks** — adding a narrow hit validator would silently reopen
omissions 4b and 5.

The corollary matters just as much: anything the miss validator does **not** check is unpinned on
both paths. Pre-fix, `grad_output` was checked nowhere in it — no storage check, no layout check, no
shape check against `input` — which is what turned omission 4b into a bug rather than a caveat.
`fab067a` added all three (`device/tanh_bw_device_operation.cpp:72-103`), and the one property it
still does not check is whether `grad_output` is **sharded**: the `is_sharded` `TT_FATAL` at `:53`
names `input_tensor` only. That is what leaves omission 3b open.

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

```156:180:ttnn/cpp/ttnn/operations/eltwise/unary_backward/tanh_bw/device/tanh_bw_device_operation.cpp
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

    // args only carries the requested output_dtype/output_memory_config; when the caller supplies its
    // own output tensor that is what the factory actually binds, sizing the destination CB from its
    // dtype and baking a TensorAccessorArgs for its buffer into the writer's compile-time args. Neither
    // can be refreshed on a cache hit, so key on the tensor that is really used.
    if (tensor_args.preallocated_input_grad.has_value()) {
        const auto& preallocated = tensor_args.preallocated_input_grad.value();
        hash =
            ttsl::hash::hash_objects(hash, preallocated.dtype(), preallocated.layout(), preallocated.memory_config());
    }

    return hash;
}
```

`args` is passed whole, so both `operation_attributes_t` fields survive. The two input tensors are
decomposed selectively. `preallocated_input_grad` contributes its engaged flag plus `dtype`, `layout`
and `memory_config` when engaged — that conditional block is `fab067a`'s addition; pre-fix it did not
appear anywhere. The one term that is a *projection* rather than a field is
`input_shape.volume()`, and it is the source of the remaining bug (omission 3b).

## Omitted parameters

### 1. `tensor_args.preallocated_input_grad` — the entire optional output tensor

**Verdict: RESOLVED by fab067a** (was BUG).

`fab067a` added the engaged case to the key: `preallocated.dtype()`, `preallocated.layout()` and
`preallocated.memory_config()` (`device/tanh_bw_device_operation.cpp:173-177`). `memory_config()`
carries `buffer_type()`, which is the `IsDram` bit and the `aligned_page_size` in the writer's
compile-time `TensorAccessorArgs`; `layout()` covers the row-major variant of the same hole; `dtype()`
covers the destination CB's page size. The `has_value()` discriminator is implicit — an engaged
optional folds three extra terms into the key, so an engaged and a disengaged call cannot collide.
`fab067a` also replaced the tautological shape check with real ones against `input`
(`:108-124`), and applied `require_standard_tile` to the preallocated tensor (`:107`), so its layout
and tile are pinned in the hit-reachable miss validator as well as keyed.

Pre-fix: the preallocated output tensor determines the `dst_buffer` that the writer kernel's
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

`input_grad` is still not consulted when building `output_memory_config` (contrast `gelu_bw` at
`unary_backward.cpp:1567-1568`, which *does* use `input_grad->memory_config()`) — `fab067a` fixed this
in the hash rather than in the front end, so recommendation 2 below remains a consistency
improvement rather than a fix. And validation only compares the memory *layout*, not the buffer type:

```47:51:ttnn/cpp/ttnn/operations/eltwise/unary_backward/tanh_bw/device/tanh_bw_device_operation.cpp
    TT_FATAL(
        input_tensor.memory_config().memory_layout() == out_memory_config.memory_layout(),
        "TANH_BW operation requires Input and Output memory layout to match. Input layout: {}, Output layout: {}",
        input_tensor.memory_config().memory_layout(),
        out_memory_config.memory_layout());
```

**Pre-fix two-call reproduction.** Let `grad` and `input` both be DRAM-interleaved, `TILE`,
`bfloat16`, shape `[1, 1, 32, 32]`.

- **Call 1**: `ttnn::tanh_bw(grad, input)` — no `input_grad`. `args = {output_dtype = BFLOAT16,
  output_memory_config = DRAM interleaved}`. `create_output_tensors` allocates a DRAM output.
  The writer is compiled with `ArgConfig::IsDram` set.
- **Call 2**: `ttnn::tanh_bw(grad, input, /*output_mem_config=*/std::nullopt,
  /*input_grad=*/ttnn::empty_like(input, ttnn::L1_MEMORY_CONFIG))` — an L1-interleaved
  preallocated output of identical shape and dtype. `args` is byte-identical to call 1 (the caller
  still derives `output_memory_config` from `input`), and pre-fix `preallocated_input_grad` was not in
  the hash, so **the hash was identical** and the cache hit. Post-`fab067a` the L1 `memory_config()`
  is in the key and call 2 misses.
- Validation did not stop it. The op has no `validate_on_program_cache_hit`, so
  `validate_on_program_cache_miss` runs on the hit
  (`ttnn/api/ttnn/device_operation.hpp:265-269`); it recomputes `out_memory_config` from the
  preallocated tensor, checks dtype equality (passes, both BFLOAT16), checks
  `memory_layout` equality (passes, `INTERLEAVED == INTERLEAVED`), and checked the logical shape
  against `compute_output_specs`, which returns the preallocated tensor's own spec — a tautology that
  could never fire, since fixed (`device/tanh_bw_device_operation.cpp:108-124`).
- **What goes stale**: writer compile-time arg 1 (`args_config_.raw()`, with `IsDram` set) and
  compile-time arg 2 (`aligned_page_size`). The `dst_buffer` *address* is patched correctly by
  `apply_resolved_bindings`, but the kernel resolves that address through the DRAM bank map.
- **Symptom**: the writer issues NOC writes to DRAM banks using an L1-relative address. The
  returned `input_grad` tensor contains uninitialised L1 contents, and unrelated DRAM is
  overwritten. Silent wrong results, not a crash.

The mirror case (call 1 with an L1 preallocated output, call 2 with none / a DRAM one) failed the
same way. A third variant was worse: nothing validated the preallocated tensor's **layout**, so a
`ROW_MAJOR` preallocated output changed `aligned_page_size` (row bytes instead of tile bytes) under
the same hash. `fab067a` closed that one twice over — `layout()` is now hashed and
`require_standard_tile` rejects it in the hit-reachable validator (`:107`).

This is exactly the aliasing class the framework's parity oracle is designed to catch — but
`assert_fastpath_parity`
(`tt_metal/api/tt-metalium/experimental/program_descriptor_patching.hpp:191-192`) only compares
runtime args and CB addresses, not compile-time args, so it would *not* have flagged this one. The fix
had to be in the hash, and that is where it landed.

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

The verdict holds for every *interleaved* configuration, which is all this op supports for `input`
(`device/tanh_bw_device_operation.cpp:53`, `:61-65`). It does **not** extend to a sharded
`grad_output`, whose page geometry depends on the shape and not merely its volume; that is the
distinction omission 3b turns on. Read this entry as "the volume is enough for everything the factory
computes from `input`", not as "the shape is unused".

### 3a. `tensor_args.grad_output.logical_shape()` / `padded_shape()` — omitted entirely (interleaved gradients)

**Verdict: VALID — unused** (holds for interleaved `grad_output` only; see 3b).

The factory reads nothing from `grad_output` except its dtype (for the src0 CB format and page
size) and its buffer (for the accessor args and the address binding). The work split comes solely
from `input.physical_volume()`. So two calls whose `grad_output` shapes differ but whose `input`
shapes agree genuinely produce the same descriptor, and sharing a cache entry is correct — **provided
`grad_output` is interleaved**, in which case the only shape-derived thing its `TensorAccessorArgs`
emits is `aligned_page_size`, which for a TILE tensor is fixed by dtype. For a sharded
`grad_output` the accessor bakes the tensor's shape in pages, and this verdict does not hold; see 3b.

The separate defect recorded here in the pre-fix audit — that `validate_on_program_cache_miss` never
checked that `grad_output` and `input` have the same shape, while the reader is told to fetch
`num_tiles_per_core` pages from both — was **fixed by `fab067a`**
(`device/tanh_bw_device_operation.cpp:99-103`), along with the missing `storage_type()` and
`buffer() != nullptr` checks (`:92-98`). It was a plain validation gap rather than a cache finding, so
it was never counted among the bugs; recommendation 3 is now implemented. Note the direction of the
new check: it pins `grad_output.padded_shape()` to `input.padded_shape()`, which is *not* the same as
pinning it to a hashed value, because the key carries only that shape's volume. The layout of
`grad_output` was a different matter and *was* a cache bug, because layout reaches a compile-time arg;
that is omission 4b, also now closed.

### 3b. `tensor_args.grad_output`'s page geometry when it is SHARDED — the padded volume is not enough

**Verdict: BUG** (new; missed by the pre-fix audit, which cleared `grad_output`'s shape without
distinguishing the sharded case).

The hash keys `input_shape.volume()` (`device/tanh_bw_device_operation.cpp:167`) — a scalar — and
`grad_output.memory_config()` (`:166`). Neither carries `grad_output`'s shape in pages. For an
interleaved gradient that is fine (3a). For a **sharded** one it is not, because `TensorAccessorArgs`
takes a different branch that bakes the whole distribution geometry into compile-time args:

```194:196:tt_metal/impl/buffers/tensor_accessor_args.cpp
    if (args_config_.test(tensor_accessor::ArgConfig::Sharded)) {
        CMAKE_UNIQUE_NAMESPACE::append_sharded_args(*buffer_, args_config_, compile_time_args, /* is_runtime */ false);
    } else {
```

```19:22:tt_metal/impl/buffers/tensor_accessor_args.cpp
    const auto& buffer_distribution_spec = buffer.buffer_distribution_spec().value();
    const auto& tensor_shape = buffer_distribution_spec.tensor_shape_in_pages();
    const auto& shard_shape = buffer_distribution_spec.shard_shape_in_pages();
    const auto& bank_coords = buffer_distribution_spec.cores();
```

```70:75:tt_metal/impl/buffers/tensor_accessor_args.cpp
    if (add_tensor_shape) {
        args.insert(args.end(), tensor_shape.cbegin(), tensor_shape.cend());
    }
    if (add_shard_shape) {
        args.insert(args.end(), shard_shape.cbegin(), shard_shape.cend());
    }
```

The factory constructs the args with the default `ArgsConfig`, so no `Runtime*` bit is set and
`add_tensor_shape` is true on the `is_runtime = false` pass — `tensor_shape_in_pages` is a
**compile-time** word:

```85:87:ttnn/cpp/ttnn/operations/eltwise/unary_backward/tanh_bw/device/tanh_bw_program_factory.cpp
    std::vector<uint32_t> reader_compile_time_args = {0};
    TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*src1_buffer).append_to(reader_compile_time_args);
```

`src0_buffer` is `grad_output.buffer()` (`tanh_bw_program_factory.cpp:45`). The single-vector
`append_to` overload even `TT_FATAL`s if any `Runtime*` bit is set
(`tt_metal/impl/buffers/tensor_accessor_args.cpp:190-193`), so on this path the geometry is
*necessarily* static. The device then divides and mods the page coordinate by that baked shape.

Reachability: `TT_FATAL(!input_tensor.is_sharded(), ...)`
(`device/tanh_bw_device_operation.cpp:53`) names `input_tensor` only, and `require_standard_tile`
checks layout and tile, not sharding. Nothing anywhere in `validate_on_program_cache_miss` rejects a
sharded `grad_output`, and `ttnn::tanh_bw` accepts any `Tensor` for `grad`
(`unary_backward.hpp:380-384`).

**Two-call reproduction.** `input` DRAM-interleaved `TILE` `bfloat16` in both calls, no preallocated
output. `grad` L1 `TILE` `bfloat16` ND-sharded with an identical `NdShardSpec` in both calls — shard
shape `[64, 32]` (2x1 tile pages), the same two-core grid, `ROW_MAJOR`, `ROUND_ROBIN_1D` — so
`grad_output.memory_config()` is bit-identical across the two calls. Reachable from Python via
`ttnn.MemoryConfig(ttnn.BufferType.L1, ttnn.NdShardSpec(ttnn.Shape([64, 32]), grid))`
(`ttnn/cpp/ttnn-nanobind/tensor.cpp:496-513`, `:650-664`). ND sharding is the vehicle because a 2D
`ShardSpec` largely pins the padded shape, whereas an `NdShardSpec` deliberately does not.

- **Call 1**: `input` and `grad` padded shape `[1, 1, 64, 64]` — 2x2 tile pages.
  `input_shape.volume() = 4096`. `grad`'s distribution squeezes to
  `tensor_shape_in_pages = [2, 2]`, `shard_shape_in_pages = [2, 1]`, 2 shards on 2 cores, and the
  reader's compile-time accessor args are baked with exactly that.
- **Call 2**: `input` and `grad` padded shape `[1, 1, 128, 32]` — 4x1 tile pages, the same 4096
  elements. `input_shape.volume()` is unchanged; `input.dtype()`, `input.memory_config()`,
  `grad_output.dtype()`, `grad_output.memory_config()` and `args` are all unchanged. **The hash is
  identical and the cache hits.** `grad`'s real distribution is now rank-1:
  `squeeze_shape_ranks` merges the dims because the inner shard extent divides the tensor
  (`tt_metal/impl/buffers/buffer_distribution_spec.cpp:346-371`), giving
  `tensor_shape_in_pages = [4]`, `shard_shape_in_pages = [2]` — a different rank, a different shape
  and a different shard shape from what the cached program was compiled with. The core list is
  unchanged, since `ROUND_ROBIN_1D` derives it from the grid rather than the shape
  (`buffer_distribution_spec.cpp:158-164`), which is exactly why nothing downstream notices.
- Validation does not stop it. `grad_output.padded_shape() == input_tensor.padded_shape()` passes in
  *both* calls — it compares the two operands of the same call, not the current call against the
  cached one. The tile and layout checks pass. `input` is not sharded, so `:53` passes. Nothing
  compares `grad_output`'s geometry against the cached program's.
- **What goes stale**: the reader's baked rank, `tensor_shape_in_pages` and `shard_shape_in_pages`
  words — `2, [2,2], [2,1]` for a tensor that is really `1, [4], [2]`. `num_tiles = 4` and the
  per-core work split are unchanged (they come from the volume), so the work *quantity* is right and
  nothing faults.
- **Symptom**: `TensorAccessor` divides and mods the linear page id by the stale shard shape, so
  pages 1 and 2 resolve to the wrong bank and the wrong shard-local offset. The gradient tiles the
  compute kernel consumes are whatever lives at those addresses — silently wrong results, no cache
  miss, no assertion. The collision is available for any pair of tile-aligned padded shapes of equal
  volume whose squeeze differs, which is precisely the reuse omission 2 exists to buy.

**Two closures, either sufficient.**

1. `TT_FATAL(!tensor_args.grad_output.is_sharded(), ...)` in `validate_on_program_cache_miss`,
   alongside the existing `input` check at `:53`. Because the op has no hit validator, this fires on
   the offending call. It is the honest guard: the reader kernel
   (`reader_binary_interleaved_start_id.cpp`) is the interleaved variant, and the op supports no
   sharded path for either operand, so a sharded gradient is out of contract rather than merely
   mis-keyed.
2. Key `input_tensor.padded_shape()` instead of `input_shape.volume()`. Since
   `grad_output.padded_shape()` is now pinned equal to `input.padded_shape()` (`:99-103`), keying the
   input's shape transitively keys the gradient's, and the geometry follows. The cost is the
   relaxation in omission 2: `[1,1,32,64]` and `[1,1,64,32]` would then compile twice.

(1) is preferable — it keeps the relaxation and matches what the kernels actually support. Doing both
is strictly safest.

**Metal 2.0 note.** The volume-only key has no `TensorSpecRelaxations` expression: the flags
relax fields, and `pertinent_fields` can drop `padded_shape` or keep it but cannot substitute a
projection of it (`tt_metal/impl/metal2_host_api/tensor_spec_relaxations.cpp:67-87`). So the key must
be narrowed before the port regardless of which closure is chosen. The framework's own reasoning
about `dynamic_tensor_shape` describes this exact failure and refuses to permit it: the shard shape in
pages, the bank count and the bank coordinates are fixed when the `ProgramSpec` is built, the squeeze
that produces them depends on the shape **values**, and "two shapes sharing a shard spec can still
resolve to different geometry. Such an argument is REJECTED rather than silently mis-addressed"
(`tt_metal/api/tt-metalium/experimental/metal2_host_api/tensor_spec_relaxations.hpp:68-74`; the
implementation keeps `shard_distribution` pertinent at
`tt_metal/impl/metal2_host_api/tensor_spec_relaxations.cpp:71-75`). Under Metal 2.0 this call pair
would be rejected at `SetProgramRunArgs`; under the current key it is accepted and mis-addressed.

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

**Verdict: RESOLVED by fab067a** (was BUG).

`fab067a` added `require_standard_tile(tensor_args.grad_output, "the grad_output tensor")`
(`device/tanh_bw_device_operation.cpp:87`), whose first `TT_FATAL` is
`tensor.layout() == Layout::TILE` (`:73-77`). It lives in `validate_on_program_cache_miss`, and since
this op declares no hit validator that function runs on hits too
(`ttnn/api/ttnn/device_operation.hpp:265-269`) — so the row-major gradient is rejected on the very
call that would have taken the stale program, not merely on a first miss. That is the closure the
recommendation preferred, and it is complete: the omission is now correct by construction rather than
compensated by a key term.

Pre-fix: there was no check on `grad_output` anywhere in `validate_on_program_cache_miss`
(`device/tanh_bw_device_operation.cpp:13-74` as it then stood): the function read `tensor_args.input`
and `tensor_args.preallocated_input_grad` and never touched `tensor_args.grad_output`. So a `ROW_MAJOR`
gradient was reachable through the public API — `ttnn::tanh_bw(grad, input, ...)` accepts any
`Tensor` for `grad` (`unary_backward.hpp:380-384`) — without violating any enforced constraint.
Under the reachability rule that made it a bug, not a caveat. The fact that no in-tree caller
passed a row-major gradient was severity context, not a defence.

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

**Pre-fix two-call reproduction.** `input` DRAM-interleaved `TILE` bfloat16 `[1, 1, 32, 32]` in both
calls.

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

The mirror ordering (row-major first) failed the same way with the page size too small.

Note the compounding: because `grad_output.layout()` was neither hashed nor validated, call 2 did
not even get a freshly built (still-wrong) program — it silently inherited call 1's. Adding a
`TT_FATAL` on `grad_output.layout() == Layout::TILE` closes it completely and was the better fix,
since a row-major gradient is not something the compute kernel supports at any hash. That is what
`fab067a` did (`:73-77`, applied at `:87`).

### 5. The `Tile` inside `page_config` — the unguarded 32x32 assumption

**Verdict: RESOLVED by fab067a** (was BUG).

`fab067a` added the standard-tile `TT_FATAL` to `validate_on_program_cache_miss`
(`device/tanh_bw_device_operation.cpp:78-84`) and applied it to all three tensors — `input`,
`grad_output` and, when engaged, the preallocated output (`:86-87`, `:107`). Because the op has no
`validate_on_program_cache_hit`, that guard runs on hits (`ttnn/api/ttnn/device_operation.hpp:265-269`),
so a `Tile{16, 32}` operand is rejected on the call that would have inherited the 32x32 program. The
recommendation's preferred route was taken: guard rather than hash, which also means the factory's bare
32x32 arithmetic stays consistent with what the op accepts.

Pre-fix: the factory never reads the tensor's actual tile. It sizes every circular buffer from
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
`tensor_spec().tile().get_tile_shape()`; neither appears anywhere in the factory. Pre-fix all three
conditions for the defect held:

1. **The op accepts (indeed requires) `Layout::TILE`** — `device/tanh_bw_device_operation.cpp:55-59`,
   quoted above.
2. **Host-side code does bare 32x32 tile math** — `tanh_bw_program_factory.cpp:24`, `:26`, `:28`,
   `:30`, feeding `total_size` and `page_size` on all three CB descriptors (`:53-81`) and the work
   split (`:36-37`). Still true; `fab067a` did not touch the factory.
3. **Nothing validates the tile geometry** — no longer true. `validate_on_program_cache_miss` now
   checks `tensor_spec().tile()` against `tt::constants::TILE_HEIGHT`/`TILE_WIDTH` for every operand
   (`device/tanh_bw_device_operation.cpp:78-87`, `:107`), which is what closes the finding: condition
   3 is the one that made (1) and (2) exploitable.

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

**Pre-fix two-call reproduction.** Both calls DRAM-interleaved `TILE` bfloat16, logical shape
`[1, 1, 64, 64]`, no preallocated output. 64 is divisible by both 32 and 16, so the padded shape —
and therefore `padded_shape().volume()`, the only shape term in the hash — is `4096` in both.

- **Call 1**: default `Tile{32, 32}`. `num_tiles = 4096 / 1024 = 4`; each CB page is
  `tt::tile_size(Float16_b) = 2048` bytes.
- **Call 2**: identical tensors built with `Tile{16, 32}`. `dtype`, `memory_config` and the padded
  volume are unchanged, and `page_config` is not in the hash, so the hash is identical and the cache
  hit. Post-`fab067a` the call is rejected before it gets that far.
- **What goes stale**: everything derived from the tile. The correct tile count for call 2 is
  `(64/16) * (64/32) = 8`, but the cached program moves `4`; the correct page size is
  `16 * 32 * 2 = 1024` bytes, but the cached CBs are `2048`.
- **Symptom**: half the tensor is never read or written, and each CB page straddles two real tiles,
  so the compute kernel's face addressing is wrong for every tile it does process. Silent wrong
  results with no cache miss to hint at the cause.

This was the compounding case: the factory bug (hardcoded 32x32) and the hash bug (no `page_config`)
each made the other worse. Fixing only the hash would have produced a freshly built program that is
still wrong; fixing only the factory without adding `page_config` to the hash would have produced a
correct program for call 1 that call 2 then reuses. The cheap fix was the guard, and that is what
landed — the factory is still 32x32-only, and now says so.

Two related points, both framework-wide rather than op-local, recorded so the verdict is not
over-read. First, `Tile::attribute_values()` exposes only `tile_shape`, `face_shape` and `num_faces`
(`tt_metal/api/tt-metalium/tile.hpp:46-47`), and `Tile::operator==` compares only `tile_shape` and
`face_shape` (`tt_metal/impl/data_format/tile.cpp:122-124`), so `transpose_within_face` and
`transpose_of_faces` are invisible to the hash *and* to canonical-key collision resolution for every
op in the repo. Hashing `page_config` would therefore not have closed the transpose variant of this
hole; only an explicit `TT_FATAL` on `get_transpose_within_face()` / `get_transpose_of_faces()` would,
and `fab067a`'s guard checks `get_height()`/`get_width()` only, so the transpose flags remain outside
both the key and the check. Nothing in this op reads them. Second, the guard closes the whole
tile-shape family at once, which is why it was preferable to hashing `page_config`.

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

Note the premise: "for the interleaved tile-layout tensors this op accepts". `input` is genuinely
pinned to interleaved (`device/tanh_bw_device_operation.cpp:53`, `:61-65`), but `grad_output` is not,
and for a sharded gradient the alignment feeds the page count rather than just a page size. That
exposure is recorded as omission 3b rather than here, since the shape term is the load-bearing gap.

### 7. `input.storage` / `grad_output.storage` variant kind (device vs host)

**Verdict: VALID — pinned by validation.**

```38:45:ttnn/cpp/ttnn/operations/eltwise/unary_backward/tanh_bw/device/tanh_bw_device_operation.cpp
    TT_FATAL(
        input_tensor.storage_type() == StorageType::DEVICE,
        "TANH_BW operation requires input to be on Device. Input storage type: {}",
        input_tensor.storage_type());

    TT_FATAL(
        input_tensor.buffer() != nullptr,
        "Operands to TANH_BW need to be allocated in buffers on the device. Buffer is null.");
```

`grad_output` is now explicitly checked too — `fab067a` added the same two `TT_FATAL`s for it
(`device/tanh_bw_device_operation.cpp:92-98`). Pre-fix it was unchecked, but the factory dereferences
`grad_output.buffer()` unconditionally, so a host-storage gradient faulted as a null deref before it
could reach the cache; the new checks turn that into a reported error. Either way the storage kind is
constant across every admissible call, so it carries no information.

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
  in omission 2. It is also the term omission 3b turns on: a volume is strictly coarser than a shape,
  and a sharded operand's page geometry is not a function of it.

Nothing else. The preallocated-output terms `fab067a` added (`:173-177`) are a *subset* of what the
default key would cover, not an addition. Note the hash contains no `program_factory.index()` because
there is only one factory.

## Framework side effect of having a custom hash

Defining `compute_program_hash` opts this op out of attribute-level hash-collision resolution:

```1012:1014:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
        if constexpr (requires { DeviceOperation::compute_program_hash(attrs, tensor_args); }) {
            return key;  // custom hash -> opt out beyond the op-identity prefix
        } else {
```

`ProgramCacheKey::canonical` degrades to just the op type name, so a 64-bit hash collision between
two different `tanh_bw` configurations resolves to a wrong hit rather than a rebuild. This is
inherent to every custom-hash op, but it raises the cost of the gap in omission 3b: there is no
second line of defence. Note also that this op's `ProgramDescriptor` factory means
`report_tensor_arg_mismatch` — the framework's strict accept/reject on `ProgramSpec` ops, which turns a
projection-keyed hit into a throw — never runs here, so the key and the hit-reachable validator are
the only two lines available. Both are silent on `grad_output`'s page geometry.

## Summary

| Omitted vs. default | Used by program? | Patched on hit? | Verdict |
|---|---|---|---|
| `preallocated_input_grad` (whole optional) | Yes — writer `TensorAccessorArgs` compile-time args | Address only; compile-time args never | **RESOLVED by fab067a** (was BUG) |
| `input.logical_shape` | No (padded volume used instead) | n/a | VALID — relaxation win |
| `grad_output.logical_shape` / `padded_shape`, interleaved | No | n/a | VALID — unused (3a; the missing shape `TT_FATAL` was a validation gap, now fixed) |
| `grad_output`'s page geometry, **sharded** | Yes — reader `tensor_shape_in_pages` / `shard_shape_in_pages` compile-time args | **No** | **BUG** (3b, new) |
| `input.layout()` (`ROW_MAJOR` vs `TILE`) | Yes — accessor page size | n/a | VALID — pinned by validation |
| `grad_output.layout()` / `page_config` | Yes — reader `aligned_page_size` compile-time arg | No — now pinned to TILE by the miss validator, which runs on hits | **RESOLVED by fab067a** (was BUG) |
| `page_config`'s `Tile` (both inputs and the output) | Yes — CB page sizes and the tile count, all hardcoded 32x32 | No — now pinned to 32x32 for all three tensors | **RESOLVED by fab067a** (was BUG) |
| `input` / `grad_output` alignment | Only via hashed derivatives | n/a | VALID — unused (low residual risk; sharded gradients covered by 3b) |
| `input` / `grad_output` storage kind | n/a | n/a | VALID — pinned by validation |
| Buffer addresses | Yes | Yes (`resolved_bindings`) | VALID — patched |

**Post-`fab067a`: one program-cache correctness bug.** The three the pre-fix audit found are closed,
all of them by the same insight — a tensor property that lands in a compile-time arg must be either
hashed or pinned by a `TT_FATAL` in the validator that actually runs on hits.

1. ~~Omitting `preallocated_input_grad`~~ — closed by hashing its `dtype`, `layout` and
   `memory_config` when engaged (`device/tanh_bw_device_operation.cpp:173-177`).
2. ~~Omitting `grad_output.layout()`~~ — closed by `require_standard_tile`'s layout `TT_FATAL`
   (`:73-77`, applied at `:87`).
3. ~~Omitting `page_config` while the factory hardcodes 32x32 tile arithmetic~~ — closed by the same
   lambda's tile `TT_FATAL` (`:78-84`), applied to all three tensors.

The one that remains is narrower and was missed by the pre-fix audit:

4. Keying `input.padded_shape().volume()` rather than the shape lets two calls whose padded shapes are
   equal-volume rearrangements share a program. That is harmless for interleaved operands and is the
   relaxation omission 2 exists to buy — but a **sharded** `grad_output`'s `tensor_shape_in_pages` and
   `shard_shape_in_pages` are baked into the reader's compile-time `TensorAccessorArgs`, and nothing
   rejects a sharded gradient (`:53` names `input_tensor` only). See omission 3b.

Because the buffer-binding fast path patches addresses but never compile-time args — and neither
does the slow-path rebuild — nothing recovers from that at dispatch. What remains sound: every
non-address runtime arg and every other compile-time arg is a function of {`output_dtype`,
`output_memory_config`, `input.dtype`, `input.memory_config`, `grad_output.dtype`,
`grad_output.memory_config`, `input.padded_shape().volume()`, and the preallocated output's `dtype` /
`layout` / `memory_config` when engaged} plus device-fixed constants, and the op's lack of a
hand-written hit validator means its miss-time `TT_FATAL`s — including all of `fab067a`'s — do run on
every hit.

## Recommendations

Items 1 and 3-5 were implemented by `fab067a`; 2 is optional, and 6 is new and open.

1. ~~Fix omission 1 by hashing the preallocated output's program-relevant spec.~~ **Done**
   (`device/tanh_bw_device_operation.cpp:173-177`) — the minimal form was taken: `dtype()`,
   `layout()` and `memory_config()` when engaged, rather than hashing
   `compute_output_specs(args, tensor_args)`.
2. **Optional, not done.** Fix the front end as well: make `ttnn::tanh_bw` derive
   `output_memory_config` from `input_grad->memory_config()` when a preallocated output is supplied,
   the way `gelu_bw` already does at `unary_backward.cpp:1567-1568`. This is now a consistency
   improvement rather than a fix, since the hash covers the buffer type directly.
3. ~~Fix omission 4b by adding `TT_FATAL(grad_output.layout() == Layout::TILE, ...)` to
   `validate_on_program_cache_miss`, alongside the `grad_output` checks that are missing entirely:
   `storage_type() == StorageType::DEVICE`, `buffer() != nullptr`, and
   `grad_output.padded_shape() == input.padded_shape()`.~~ **Done** — all four
   (`device/tanh_bw_device_operation.cpp:73-77`, `:87`, `:92-103`). Because this op has no hit
   validator, every check added there runs on hits too
   (`ttnn/api/ttnn/device_operation.hpp:265-269`), so the `TT_FATAL`s are complete fixes and not
   merely miss-path ones.
4. ~~Fix omission 5 with the standard tile guard rather than by hashing `page_config`.~~ **Done**
   (`device/tanh_bw_device_operation.cpp:78-84`, applied to `input`, `grad_output` and the
   preallocated output at `:86-87` and `:107`). The canonical form is:

```95:97:ttnn/cpp/ttnn/operations/data_movement/sharded/interleaved_to_sharded/device/interleaved_to_sharded_op.cpp
        auto tile = input_tensor.tensor_spec().tile();
        if (tile.get_height() != tt::constants::TILE_HEIGHT || tile.get_width() != tt::constants::TILE_WIDTH) {
            return {false, fmt::format("interleaved_to_sharded requires standard 32x32 tiles, got {}x{}", tile.get_height(), tile.get_width())};
```

   It was applied to `input`, `grad_output` and, when engaged, `preallocated_input_grad`, as a
   `TT_FATAL` rather than a returned string. This makes omitting `page_config` correct by
   construction. The alternative — making the factory tile-aware by switching to
   `tile.get_tile_size(fmt)` and `tile().get_tile_shape()` — would require adding `page_config` to the
   hash *in the same change*, or the mirror-image bug appears (a genuinely tile-varying program keyed
   without the tile). Note that neither route closes the transpose flags, which no hash can reach
   (`tt_metal/api/tt-metalium/tile.hpp:46-47`); only an explicit check on
   `get_transpose_within_face()` / `get_transpose_of_faces()` does, and the guard as landed does not
   include one.
5. Build this op under `-DTT_DESCRIPTOR_PATCHING_PARITY_CHECK` in CI, and consider extending
   `assert_fastpath_parity` to diff compile-time args and CB page sizes as well as runtime args and
   CB addresses. The check as written would already have caught omission 5, because the tile count
   reaches the `num_tiles_per_core` runtime arg it does compare. It would miss omissions 1, 4b and 3b
   entirely, since all three go stale only in compile-time accessor args
   (`tt_metal/api/tt-metalium/experimental/program_descriptor_patching.hpp:191-192`) — and that is
   the dominant failure mode for descriptor factories that pass a `Buffer*` into
   `TensorAccessorArgs`, so the extension is worth more than this one op. **Open.**
6. **Open — close omission 3b.** Add `TT_FATAL(!tensor_args.grad_output.is_sharded(), ...)` next to
   the existing `input` check (`device/tanh_bw_device_operation.cpp:53`); the reader kernel is the
   interleaved variant, so a sharded gradient is out of contract rather than merely mis-keyed. If a
   sharded gradient is wanted later, key `input_tensor.padded_shape()` in place of
   `input_shape.volume()` instead — which costs the equal-volume relaxation in omission 2. Either way
   the volume-only key must be narrowed before the Metal 2.0 port, because no
   `TensorSpecRelaxations` field expresses "shape replaced by its volume"
   (`tt_metal/impl/metal2_host_api/tensor_spec_relaxations.cpp:67-87`).
