# Program Cache Audit — `experimental/ccl/llama_reduce_scatter_matmul`

Audit of `ttnn::operations::experimental::ccl::Matmul_RS::compute_program_hash` against the framework
default ("hash everything") key.

| | |
|---|---|
| Device operation | `ttnn::operations::experimental::ccl::Matmul_RS` (`device/rs_matmul_op.hpp:30-81`) |
| Custom hash | `device/rs_matmul_op.cpp:86-98` |
| `operation_attributes_t` | `rs` (a default-constructed `LlamaReduceScatterDeviceOperation` tag), `rs_op` (the reduce-scatter attributes: `dim`, `cross_device_semaphore`, `subdevice_id`, `cluster_axis`, `output_mem_config`, `ring_devices`, `num_links`, `topology`, `use_noc1_only`), `matmul` (the full `MatmulParams`) |
| `tensor_args_t` | `rs` (= `{input_tensor, intermediate_packet_buffer}`), `matmul` (= `{input_tensor, weight_tensor}`), `matmul_output_tensors`, `second_weight_tensor` |
| Program factories | `Matmul_RS_PF` (single; builds the reduce-scatter and the 1D matmul into one `Program`) |
| `override_runtime_arguments` | **Yes** (`device/rs_matmul_program_factory.cpp:98-139`) |
| `get_dynamic_runtime_args` | No |
| `validate_on_program_cache_hit` | Present and non-empty (`device/rs_matmul_op.cpp:17-33`) — but see below |
| Cache-hit patch mechanism | **Op-owned re-derivation** (the factory's `override_runtime_arguments` runs on every hit) |

The CSV row (*explicit / SELECTIVE / has own hit validator / has `override_runtime_arguments` / no
`get_dynamic_runtime_args`*) is accurate on all five columns. The "own hit validator" column is
technically true and materially misleading, which is worth stating up front:

```17:33:ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter_matmul/device/rs_matmul_op.cpp
void Matmul_RS::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.second_weight_tensor.has_value()) {
        operation_attributes_t::matmul_device_t::validate_on_program_cache_miss(
            operation_attributes.matmul,
            {{tensor_args.matmul.input_tensor,
              tensor_args.matmul.weight_tensor,
              tensor_args.second_weight_tensor.value()},
             {std::nullopt},
             {}});
    } else {
        operation_attributes_t::matmul_device_t::validate_on_program_cache_miss(
            operation_attributes.matmul,
            {{tensor_args.matmul.input_tensor, tensor_args.matmul.weight_tensor}, {std::nullopt}, {}});
    }
    LlamaReduceScatterDeviceOperation::validate_on_program_cache_hit(operation_attributes.rs_op, tensor_args.rs);
}
```

This validator checks the *current call* for internal consistency. It has no access to the cached
program and makes no comparison against it, so it cannot detect that the current call's matmul
configuration differs from the one the cached program was compiled for. It offers zero protection
against a hash gap. (The reduce-scatter half it delegates to is an empty stub:
`ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter/device/llama_reduce_scatter_device_operation.cpp:57-58`.)

## Cache-hit patch mechanism

The factory exposes `override_runtime_arguments` and no `apply_descriptor`, so the framework calls it
on every hit:

```279:285:ttnn/api/ttnn/device_operation.hpp
        if constexpr (requires { &WorkloadFactory::apply_descriptor; }) {
            WorkloadFactory::apply_descriptor(
                cached_mesh_workload, operation_attributes, tensor_args, tensor_return_value);
        } else {
            WorkloadFactory::override_runtime_arguments(
                cached_mesh_workload, operation_attributes, tensor_args, tensor_return_value);
        }
```

The override delegates to the two halves' own patch routines and does nothing else:

```123:137:ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter_matmul/device/rs_matmul_program_factory.cpp
        for (auto& [range, program] : cached_workload.workload.get_programs()) {
            const auto& shared_variables = cached_workload.shared_variables.at(range);
            LlamaReduceScatterDeviceOperation::LlamaReduceScatterAdd::override_runtime_arguments_per_program(
                shared_variables.rs_shared_vars,
                program,
                operation_attributes.rs_op,
                tensor_args.rs,
                tensor_return_value.at(1));
            ttnn::prim::reuse_mcast_1d_optimized_helpers::override_program_parameters(
                shared_variables.matmul_shared_vars,
                operation_attributes.matmul.global_cb,
                program,
                {{tensor_args.matmul.input_tensor, tensor_args.matmul.weight_tensor}, {}},
                {tensor_return_value.at(0)});
        }
```

Both of those refresh only buffer addresses and the semaphore address — the reduce-scatter side
re-points its three globally-allocated CBs and rewrites slot 0 of the reader and writer args
(`.../llama_reduce_scatter/device/llama_reduce_scatter_program_factory.cpp:884-893`), and the matmul
side dispatches to one of the three per-variant address patchers
(`ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp:3062-3082`).
No compile-time arg, no kernel `define`, no CB size, and no core range is refreshed.

So the hash must cover everything structural on **both** halves of the fusion. It covers a strict
subset of one half.

## Baseline: what the default hash would cover

| Source | Fields |
|---|---|
| `operation_attributes.rs_op` | all 9 fields, including `cross_device_semaphore`, `subdevice_id`, `output_mem_config` |
| `operation_attributes.matmul` | all 14 `MatmulParams` fields, including `program_config`, `output_dtype`, `output_mem_config`, `compute_kernel_config`, `user_fused_activation`, `transpose_a/b`, `output_tile`, `global_cb`, `sub_device_id` |
| `tensor_args.rs.input_tensor` | storage kind, `logical_shape`, `dtype`, `page_config`, `memory_config`, `alignment` |
| `tensor_args.rs.intermediate_packet_buffer` | the same six |
| `tensor_args.matmul.input_tensor` | the same six |
| `tensor_args.matmul.weight_tensor` | the same six |
| `tensor_args.matmul_output_tensors` | the same six, per element |
| `tensor_args.second_weight_tensor` | the same six, plus the engaged/disengaged bit of the `optional` |
| appended by framework | the mesh coordinates of the tensors |

## What the custom hash covers

```88:97:ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter_matmul/device/rs_matmul_op.cpp
    return tt::tt_metal::operation::hash_operation<Matmul_RS>(
        operation_attributes.rs_op.dim,
        operation_attributes.rs_op.cluster_axis,
        operation_attributes.rs_op.ring_devices,
        operation_attributes.rs_op.num_links,
        operation_attributes.rs_op.topology,
        operation_attributes.rs_op.use_noc1_only,
        tensor_args.rs.input_tensor.dtype(),
        tensor_args.rs.input_tensor.memory_config(),
        tensor_args.rs.input_tensor.device()->id());
```

Nine terms, all from the reduce-scatter half. The hash is a verbatim copy of the standalone
`LlamaReduceScatterDeviceOperation` key with the op type swapped — the matmul half of the fusion was
never added. **Not one attribute of the matmul, and not one of the four matmul-side tensors, reaches
the cache key.**

## Omitted parameters

### 1. `tensor_args.matmul.input_tensor` and `tensor_args.matmul.weight_tensor`

**Verdict: BUG.** This is the most severe finding in this audit.

These are the matmul's A and B operands. They are distinct tensors from
`tensor_args.rs.input_tensor` — the launcher wires them separately:

```191:195:ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter_matmul/device/rs_matmul_op.cpp
    auto tensor_args = OperationType::tensor_args_t{
        .rs = std::move(rs_tensor_args),
        .matmul = {.input_tensor = input_tensor, .weight_tensor = weight_tensor},
        .matmul_output_tensors = std::move(matmul_output_tensors),
        .second_weight_tensor = second_weight_tensor_arg};
```

and `rs_tensor_args` is built from `new_rs_tensor`, which is either the caller's separate `rs_tensor`
or the matmul's *output*, never its input
(`device/rs_matmul_op.cpp:165-186`). So the hashed
`tensor_args.rs.input_tensor.{dtype, memory_config}` says nothing about A or B.

They are passed straight into the matmul builder, which derives M, K, N, the tile sizes, all three CB
data formats and the entire work split from them:

```80:94:ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter_matmul/device/rs_matmul_program_factory.cpp
    auto matmul_sv = ttnn::prim::matmul_multi_core_reuse_mcast_1d_optimized_helper(
        program,
        tensor_args.matmul.input_tensor,
        {tensor_args.matmul.weight_tensor},
        std::nullopt /*bias*/,
        {tensor_return_value.at(0)},
        operation_attributes.matmul.bcast_batch.value(),
        operation_attributes.matmul.compute_kernel_config.value(),
        operation_attributes.matmul.program_config.value(),
        operation_attributes.matmul.untilize_out,
        fused_op_signaler,
        operation_attributes.matmul.global_cb,
        sub_device_id /*sub_device_id*/,
        tt::CBIndex::c_6 /*start cb index*/,
        reduce_scatter_core_range);
```

**Reproduction.** Fix everything on the reduce-scatter side (`rs_tensor`, packet buffer, semaphore,
`dim`, `cluster_axis`, `num_links`, `topology`, `use_noc1_only`) and vary only the weight:

- Call 1: `weight_tensor` of shape `[1, 1, 2048, 3584]`, `bfloat8_b`.
- Call 2: `weight_tensor` of shape `[1, 1, 2048, 7168]`, `bfloat8_b`.

The keys are identical — the weight tensor is not a hash term. Call 2 hits call 1's entry. The matmul
compute kernel is still compiled for `Nt = 112` with call 1's `per_core_N` blocking and call 1's in1
CB size; the writer's per-core `N` loop bounds are compile-time args. Call 2's output tensor is
allocated at the correct (doubled) size and correctly re-pointed by `override_program_parameters`,
but only the first half of it is ever written and the in1 reader walks off the end of its CB. The
symptom is half-garbage output with no error.

The same reproduction works with a dtype change (`bfloat16` vs `bfloat8_b` weights change
`in1_data_format` and the in1 tile size) or a sharded-vs-interleaved change on either operand.

### 2. `operation_attributes.matmul` (the entire `MatmulParams`)

**Verdict: BUG.**

Every field the factory reads at `device/rs_matmul_program_factory.cpp:61-69` and
`device/rs_matmul_program_factory.cpp:86-94` — `bcast_batch`, `compute_kernel_config`,
`program_config`, `untilize_out`, `global_cb` — is structural, and none is hashed. `program_config`
alone carries `compute_with_storage_grid_size`, `in0_block_w`, `out_subblock_h/w`, `out_block_h/w`,
`per_core_M`, `per_core_N`, `fuse_batch`, `fused_activation`, `mcast_in0`, `gather_in0`, `hop_cores`,
`num_global_cb_receivers` and `stream_in1`; the helper unpacks all of them into the program
(`ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp:5604-5657`).

All of it is user-reachable. The ttnn-level entry point exposes `program_config`,
`compute_kernel_config`, `dtype`, `memory_config_mm`, `global_cb`, `core_grid`, `transpose_a`,
`transpose_b`, `activation` and `output_tile` as ordinary arguments
(`rs_matmul.cpp:24-36`), and they are packed verbatim into `MatmulParams`:

```142:159:ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter_matmul/device/rs_matmul_op.cpp
    auto matmul_struct = ttnn::prim::create_matmul_attributes(
        input_tensor,
        weight_tensor,
        /*parameters=*/
        {program_config,
         /*bcast_batch=*/std::nullopt,
         memory_config_mm.value_or(input_tensor.memory_config()),
         dtype.value_or(input_tensor.dtype()),
         compute_kernel_config,
         /*untilize_out=*/false,
         user_core_coord,
         ttnn::operations::matmul::utilities::get_fused_activation(activation),
         user_run_batched,
         transpose_a,
         transpose_b,
         output_tile,
         global_cb},
        {});
```

**Reproduction.** Identical inputs and weights; call 1 with `activation=None`, call 2 with
`activation="silu"`. `get_fused_activation` turns the string into `user_fused_activation`, which
`create_matmul_attributes` folds into the program config's `fused_activation`, which becomes a
compile-time define/arg on the matmul compute kernel. Same hash, so call 2 reuses the un-activated
kernel and silently returns un-activated results. A `dtype=` change is the same class and additionally
mismatches the output CB data format.

### 3. `tensor_args.second_weight_tensor`

**Verdict: BUG.**

Its presence selects between two structurally different programs — a two-output matmul with a
`MatmulFusedOpSignaler` versus a single-output matmul with no signaler:

```43:71:ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter_matmul/device/rs_matmul_program_factory.cpp
    if (tensor_args.second_weight_tensor.has_value()) {
        ttnn::experimental::ccl::MatmulFusedOpSignaler base_signaler = ttnn::experimental::ccl::MatmulFusedOpSignaler(
            ttnn::experimental::ccl::MatmulFusedOpSignalerType::LLAMA_REDUCE_SCATTER);
        base_signaler.init_llama_rs_cores_rs(rs_cores, program);
        std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler> fused_op_signaler = base_signaler;
        auto reduce_scatter_sv = LlamaReduceScatterDeviceOperation::LlamaReduceScatterAdd::create_at_program_processing(
            operation_attributes.rs_op,
            mesh_coordinate,
            tensor_args.rs,
            tensor_return_value.at(2),
            program,
            fused_op_signaler);
```

It also changes the *arity* of the return value: `create_output_tensors` returns three tensors in the
`second_weight_tensor` case and two otherwise
(`device/rs_matmul_op.cpp:73-84`), and `override_runtime_arguments` branches on the same condition and
indexes `tensor_return_value.at(2)` (`device/rs_matmul_program_factory.cpp:103-121`).

It is not hashed. There is a partial, accidental mitigation: in the `second_weight_tensor` case
`rs.input_tensor` is set to `matmul_output_tensors.at(0)` rather than the caller's `rs_tensor`
(`device/rs_matmul_op.cpp:165-172`), so the hashed `rs.input_tensor.memory_config()` will usually
differ between the two modes. "Usually" is not a guarantee — a caller passing an `rs_tensor` whose
memory config happens to match the matmul output's gets a collision. The `TT_FATAL` at
`device/rs_matmul_op.cpp:132-134` enforces that exactly one of the two is supplied, which means both
modes are genuinely reachable in one process.

**Reproduction.** Call 1 in single-weight mode with an `rs_tensor` whose dtype and memory config match
what the two-weight mode would produce; call 2 in two-weight mode with the same reduce-scatter
parameters. Same key. Call 2 hits a program built with one matmul output and no fused-op signaler,
then `override_runtime_arguments` takes the two-weight branch and calls
`override_program_parameters` with two output tensors against `matmul_shared_vars` recorded for one —
at best an out-of-range access, at worst a silently mis-patched CB.

### 4. `operation_attributes.rs_op.output_mem_config`

**Verdict: BUG.**

Unlike its sibling `llama_reduce_scatter_create_heads`, where the equivalent field is dead, here
`output_mem_config` determines the reduce-scatter output tensor's shard spec:

```86:91:ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter/device/llama_reduce_scatter_device_operation.cpp
    if (attributes.output_mem_config.has_value()) {
        return {tt::tt_metal::TensorSpec(
            Shape(output_shape),
            TensorLayout(
                input_tensor.dtype(), PageConfig(input_tensor.layout()), attributes.output_mem_config.value()))};
    }
```

and that shard grid becomes a compile-time kernel define plus a set of CB sizes on the
reduce-scatter side:

```671:671:ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter/device/llama_reduce_scatter_program_factory.cpp
    reader_defines["OUTPUT_CORE_XY"] = detail::cores_to_string(to_worker_cores(output_cores, ncores_output));
```

`output_cores` comes from `output_grid = output_shard_spec.grid`
(`.../llama_reduce_scatter_program_factory.cpp:434`, `.../llama_reduce_scatter_program_factory.cpp:614-615`),
and `output_tiles_per_core_width` (derived from the same shard spec at
`.../llama_reduce_scatter_program_factory.cpp:410-411`) sizes the output CB and the accumulator CB
(`.../llama_reduce_scatter_program_factory.cpp:498`, `.../llama_reduce_scatter_program_factory.cpp:598`)
and appears in reader, writer and compute compile-time args
(`.../llama_reduce_scatter_program_factory.cpp:653`, `:702`, `:733`).

The only validation is a shard-height check
(`.../llama_reduce_scatter/device/llama_reduce_scatter_device_operation.cpp:47-54`); the grid is free.

**Reproduction.** Call 1 with `memory_config_rs=None` (the op derives a grid from the compute grid at
`.../llama_reduce_scatter_device_operation.cpp:94-105`); call 2 with an explicit `memory_config_rs`
placing the output shards on a different core set of the same size. Same key. The cached reader still
writes back to the old cores via the `OUTPUT_CORE_XY` define, and the freshly allocated output tensor
is never written.

### 5. `operation_attributes.rs_op.subdevice_id`

**Verdict: BUG** (via the matmul half only).

`subdevice_id` is unwrapped at the top of the factory and handed to the matmul builder:

```39:41:ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter_matmul/device/rs_matmul_program_factory.cpp
    tt::tt_metal::SubDeviceId sub_device_id = operation_attributes.rs_op.subdevice_id.value();
    auto [part_cores, rs_cores] =
        LlamaReduceScatterDeviceOperation::get_rs_core_grids(operation_attributes.rs_op, tensor_args.rs);
```

where it selects the core pool the matmul is laid out on:

```2122:2128:ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp
    auto subdevice_cores = device->worker_cores(
        tt::tt_metal::HalProgrammableCoreType::TENSIX,
        sub_device_id.has_value() ? *sub_device_id : device->get_sub_device_ids().at(0));
    if (restricted_cores.has_value()) {
        subdevice_cores = subdevice_cores.subtract(restricted_cores.value());
    }
    for (const auto& cr : subdevice_cores.ranges()) {
```

and, in the mcast variants, the multicast origin:
`.../matmul_multicore_reuse_mcast_1d_program_factory.cpp:5262-5268`.

The reduce-scatter half happens to be immune: `get_rs_core_grids` takes its available cores from
`llama_specific::get_custom_cores(num_links)` rather than from the sub-device
(`.../llama_reduce_scatter_program_factory.cpp:324`), and the `sub_device_cores` local computed at
`.../llama_reduce_scatter_program_factory.cpp:436-438` is never subsequently read. So the exposure is
entirely on the matmul side, but it is real: two calls differing only in `subdevice_id` share a cache
entry and the second runs matmul kernels on the first's cores.

### 6. `tensor_args.rs.intermediate_packet_buffer`

**Verdict: BUG** for the spec; **VALID — patched** for the address.

The packet buffer's shard grid selects the packet-worker cores, and thence the whole core range the
reduce-scatter kernels and CBs are built on:

```305:328:ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter/device/llama_reduce_scatter_program_factory.cpp
    auto intermediate_packet_buffer_grid = tensor_args.intermediate_packet_buffer.shard_spec().value().grid;
    uint32_t ncores_input = (input_tensor_width + input_shard_width - 1) / input_shard_width;
    if (ncores_input % num_devices != 0) {
        ncores_input = ((ncores_input + num_devices - 1) / num_devices) * num_devices;
    }
    uint32_t input_shard_cores_per_device = ncores_input / num_devices;
    auto fabric_max_packet_size = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    size_t packet_size_bytes =
        input_tensor.dtype() == DataType::BFLOAT16 ? std::bit_floor(fabric_max_packet_size) : fabric_max_packet_size;
    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t input_page_size = tile_size(cb_data_format);
    uint32_t num_pages_per_packet = packet_size_bytes / input_page_size;
    uint32_t num_packets_total_per_device =
        (input_shard_cores_per_device * input_tiles_per_core_width + num_pages_per_packet - 1) / num_pages_per_packet;
    auto packet_worker_cores_grid = detail::get_worker_cores(
        intermediate_packet_buffer_grid,
        num_packets_total_per_device,
        input_shard_spec.orientation == ShardOrientation::ROW_MAJOR);

    auto available_cores = llama_specific::get_custom_cores(num_links);

    auto sender_core_grid = detail::get_worker_cores(
        available_cores, num_links, input_shard_spec.orientation == ShardOrientation::ROW_MAJOR);
    auto all_cores_grid = packet_worker_cores_grid.merge(sender_core_grid);
```

Note the compounding effect specific to *this* op: `rs_cores` (the second element of the returned
tuple, i.e. `all_cores_grid`) is passed to the matmul as `restricted_cores`
(`device/rs_matmul_program_factory.cpp:42`, `device/rs_matmul_program_factory.cpp:69` and
`device/rs_matmul_program_factory.cpp:94`), so the packet buffer's grid also shifts the *matmul's*
core placement. A single unhashed tensor spec moves both halves of the fusion.

Its buffer *address* is correctly refreshed —
`UpdateDynamicCircularBufferAddress(program, shared_variables.cb_handles[2], *packet_buffer)` at
`.../llama_reduce_scatter_program_factory.cpp:886`.

### 7. `operation_attributes.rs_op.cross_device_semaphore` (`GlobalSemaphore`)

**Verdict: VALID — patched.**

Slot 0 of both the reduce-scatter reader and writer args
(`.../llama_reduce_scatter_program_factory.cpp:747-748` and
`.../llama_reduce_scatter_program_factory.cpp:774-775`), rewritten on every core on every hit:

```884:893:ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter/device/llama_reduce_scatter_program_factory.cpp
    UpdateDynamicCircularBufferAddress(program, shared_variables.cb_handles[0], *input_tensor_buffer);
    UpdateDynamicCircularBufferAddress(program, shared_variables.cb_handles[1], *output_tensor_buffer);
    UpdateDynamicCircularBufferAddress(program, shared_variables.cb_handles[2], *packet_buffer);

    for (const auto& core : cores) {
        auto& writer_runtime_args = tt::tt_metal::GetRuntimeArgs(program, unary_writer_kernel_id, core);
        writer_runtime_args[0] = (uint32_t)operation_attributes.cross_device_semaphore->address();
        auto& reader_runtime_args = tt::tt_metal::GetRuntimeArgs(program, unary_reader_kernel_id, core);
        reader_runtime_args[0] = (uint32_t)operation_attributes.cross_device_semaphore->address();
    }
```

Never a compile-time arg. Correctly omitted, correctly patched.

### 8. `tensor_args.rs.input_tensor.logical_shape()`

**Verdict: BUG.**

The logical width is read directly and drives the work split:

```298:302:ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter/device/llama_reduce_scatter_program_factory.cpp
    const auto& input_tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    auto input_tensor_width = input_tensor.logical_shape()[-1];
    auto input_shard_spec = input_tensor.shard_spec().value();
    uint32_t input_shard_width = input_shard_spec.shape[1];
    uint32_t input_tiles_per_core_width = input_shard_width / input_tile_shape[1];
```

`ncores_input` (line 306, quoted in #6) feeds `input_shard_cores_per_device`, the packet count, the
work `schedule`, and thence the `SCHEDULE` define and several compile-time args.

Unlike `llama_reduce_scatter_create_heads`, there is **no hashed proxy** here: that op recovers the
padded width from the hashed `head_dim * (num_heads + 2*num_kv_heads)`, but this op has no such
attributes. And `ncores_input` is specifically *not* recoverable from the hashed `memory_config`,
because this op deliberately supports an over-provisioned shard grid — that is what the padding
branch of `compute_output_specs` is for:

```64:81:ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter/device/llama_reduce_scatter_device_operation.cpp
    // input is unpadded, output is padded. Ex, input: 3584, 112 tiles, padded to 5 tiles per core, total width is 120
    // tiles (3840). this should be changed to use unpadded output in the future.
    auto input_tensor = tensor_args.input_tensor;
    auto tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    const auto& input_spec = input_tensor.tensor_spec();
    auto input_shard_spec = input_tensor.shard_spec().value();
    auto input_grid = input_shard_spec.grid;
    auto input_shard_width = input_shard_spec.shape[1];
    auto input_num_cores = input_grid.num_cores();
    auto input_shape = input_spec.logical_shape();
    auto input_width = input_shape[attributes.dim];
    auto input_width_in_tiles = input_width / tile_shape[1];
    auto padded_input_width_in_tiles =
        input_num_cores * ((input_width_in_tiles + input_num_cores - 1) / input_num_cores);
    auto padded_input_width = padded_input_width_in_tiles * tile_shape[1];

    uint32_t final_width = input_width % input_shard_width != 0 ? padded_input_width / attributes.ring_devices
                                                                : input_width / attributes.ring_devices;
```

The comment is explicit: the logical width (3584) and the shard-grid capacity (3840) are allowed to
differ. So the logical width carries information the memory config does not.

**Reproduction.** Two `rs_tensor`s with the identical memory config — 24 cores, shard width 160 — one
with logical width 3584 and one with logical width 3520. Same dtype, same everything else. Same key.
`ncores_input` is `ceil(3584/160)=23` versus `ceil(3520/160)=22`, so the work `schedule` and the
`SCHEDULE` compile-time define differ, and the second call reuses the first's. One core's worth of
data is either read twice or never read.

### 9. `tensor_args.rs.input_tensor.page_config` (the `Tile`) and `layout()` — the unguarded 32x32 assumption

**Verdict: BUG.**

**`page_config()` is confirmed absent from this hash.** The nine terms are six `rs_op` scalars plus
exactly three tensor terms, and none of them is the page config, the tensor spec or even the layout:

```95:97:ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter_matmul/device/rs_matmul_op.cpp
        tensor_args.rs.input_tensor.dtype(),
        tensor_args.rs.input_tensor.memory_config(),
        tensor_args.rs.input_tensor.device()->id());
```

So the `Tile` is not in the key by any route, and the BUG grade below turns on that. Both failure
modes of the unguarded-32x32 pattern are then present at once, in the same factory.

**Tile-aware for tile counts.** The reduce-scatter half reads the real tile and divides the shard
width by it:

```296:302:ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter/device/llama_reduce_scatter_program_factory.cpp
    const auto& input_tensor = tensor_args.input_tensor;
    const uint32_t ring_size = operation_attributes.ring_devices;
    const auto& input_tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    auto input_tensor_width = input_tensor.logical_shape()[-1];
    auto input_shard_spec = input_tensor.shard_spec().value();
    uint32_t input_shard_width = input_shard_spec.shape[1];
    uint32_t input_tiles_per_core_width = input_shard_width / input_tile_shape[1];
```

and the same again in the program builder for both operands
(`.../llama_reduce_scatter_program_factory.cpp:398-411`), producing `input_tiles_per_core_width`,
`output_tiles_per_core_width` and `output_tensor_width_in_tiles`.

**Hardcoded 32x32 for the page size in bytes.** Two lines later it abandons the tensor's tile and
takes the byte size of a 32x32 tile from the free function:

```426:426:ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter/device/llama_reduce_scatter_program_factory.cpp
    uint32_t input_page_size = tile_size(cb_data_format);
```

(and identically at `.../llama_reduce_scatter_program_factory.cpp:315`). The tile-aware API is
`tile.get_tile_size(data_format)`; `tt::tile_size(fmt)` always returns the 32x32 size.

**Both values are structural.** `input_page_size` is the page size of every circular buffer in the
reduce-scatter half — the input CB (`.../llama_reduce_scatter_program_factory.cpp:492-493`), the
output CB (`:498-499`), the fabric sender CB (`:541-542`), the fabric receiver CB (`:583-584`) and
the accumulator CB (`:598-600`) — and it also derives `num_pages_per_packet`
(`.../llama_reduce_scatter_program_factory.cpp:444`), which governs the packet split. All four of
these land in the reader's compile-time args:

```644:657:ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter/device/llama_reduce_scatter_program_factory.cpp
    std::vector<uint32_t> reader_compile_time_args = {
        input_tensor_cb_id,
        fabric_sender_cb_index,
        packet_header_cb_index,
        fabric_receiver_cb_index,
        accumulator_cb_index,
        output_tensor_cb_id,
        (uint32_t)chip_id,
        input_tiles_per_core_width,
        output_tiles_per_core_width,
        num_pages_per_packet,
        input_shard_cores_per_device,
        num_devices,
        input_page_size,
```

and in the writer's (`.../llama_reduce_scatter_program_factory.cpp:703-706`). The matmul half is
tile-aware throughout, as every matmul factory in the tree is, so it too varies with `Tile`.

**No guard.** The miss validator pins the packet buffer's tile to *match the input's*, but never pins
the input's own tile to a value:

```42:46:ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter/device/llama_reduce_scatter_device_operation.cpp
    TT_FATAL(
        tensor_args.intermediate_packet_buffer.tensor_spec().tile().get_tile_shape() == tile_shape,
        "intermediate_packet_buffer must have the same tile shape ({}, {}) as input_tensor",
        tile_shape[0],
        tile_shape[1]);
```

That is a consistency check between two tensors, not a geometry check. All three adjudication
criteria hold — the op accepts `Layout::TILE`, host-side code calls `tt::tile_size(...)`, and nothing
validates the tile geometry — and the mirror-image criterion holds too, since the tile-count
arithmetic provably varies with `Tile` while `Tile` is absent from the key.

**Reproduction.** Two calls with the same `rs_tensor` dtype and memory config (say 24 cores, shard
width 160), the same `dim`, `cluster_axis`, `ring_devices`, `num_links`, `topology` and
`use_noc1_only`; the first with `Tile{32, 32}` and the second with `Tile{16, 32}`. The nine hash terms
are identical, so the second call hits the first's entry. The cached program has
`input_tiles_per_core_width = 160/32 = 5` and `input_page_size = 2048` bytes (bfloat16 32x32); the
second call's tensor genuinely has 5 tiles of 1024 bytes per core along the width but twice as many
tile rows. Every CB page size in the reduce-scatter half is double the real tile, `num_pages_per_packet`
is halved, and the packet schedule baked into the `SCHEDULE` define no longer describes the data.
Symptom is wrong reduction results, or a hang waiting on packets that were never sized to arrive.

**One note on the sweep.** A directory-scoped search of `llama_reduce_scatter_matmul/` finds no
tile math and would classify this op as not applicable. That is misleading: this op's program factory
is the shared one under `experimental/ccl/llama_reduce_scatter/`, reached through
`LlamaReduceScatterDeviceOperation::get_rs_core_grids` and
`LlamaReduceScatterDeviceOperation::LlamaReduceScatterAdd::create_at_program_processing`
(`device/rs_matmul_program_factory.cpp:41`, `:48`, `:73`), and that is where all the tile arithmetic
lives. Exactly two ops build through that shared factory — this one and the standalone
`experimental/ccl/llama_reduce_scatter` — so both inherit the defect and both would be fixed by a
single change there. The third sibling, `experimental/ccl/llama_reduce_scatter_create_heads`, has its
own factory and is genuinely free of host-side tile math, so it neither inherits the defect nor the
fix.

### 10. `tensor_args.rs.input_tensor` alignment and storage kind, and all buffer addresses

**Verdict: VALID — unused** (alignment); **VALID — patched** (buffer addresses); **CAVEAT — pinned
only on the miss path** (storage kind).

Alignment reaches the program only via the buffer page size, itself determined by the hashed
`{memory_config, dtype}` plus the tile discussed in #9. Storage kind is pinned by the miss validator's
requirement of a shard spec — but this op defines `validate_on_program_cache_hit`, so that
requirement does not run on a hit (rule 4a-ii), which is why the verdict is CAVEAT rather than VALID.
It is only a CAVEAT and not a BUG because the factory's `buffer()` dereferences fault on a
host-storage tensor rather than aliasing silently. Buffer addresses must not
be hashed and are all patched: the reduce-scatter input, output and packet CBs at
`.../llama_reduce_scatter_program_factory.cpp:884-886`, and the matmul operands and outputs through
`override_program_parameters`
(`ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp:3062-3082`).

### 11. `ring_index` and forward/backward fabric neighbours

**Verdict: VALID — invariant** (determined by the mesh coordinates the framework appends to every
key, plus hashed attributes).

The reduce-scatter half derives its chip index and neighbours from `mesh_coordinate` together with
`cluster_axis`, `ring_devices` and `topology`. The latter three are hashed; the first is folded in by
the framework for both hash paths:

```989:992:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
        // Combine with the mesh coordinates the workload is targeting.
        for (const auto& coord : mesh_device_operation_utils::extract_tensor_coordinates(tensor_args, mesh_device)) {
            hash = ttsl::hash::hash_objects(hash, coord);
        }
```

so per-device variation is keyed and is not an omission.

### 12. Fabric connection runtime args

**Verdict: CAVEAT.**

`append_fabric_connection_rt_args` puts EDM router coordinates, buffer bases and flow-control
semaphore addresses in the writer's tail runtime args on the sender cores; the override rewrites only
slot 0. Frozen at first-miss values, which is safe while the fabric configuration is fixed for the
lifetime of the mesh device. `num_links` is hashed, which closes the most likely variant. Same
assumption as every fabric CCL op in the tree.

## Keys the custom hash adds beyond the default

`tensor_args.rs.input_tensor.device()->id()`. The default reflection hash does not include the device
id (`DeviceStorage` has an empty attribute tuple, so neither buffer nor device reaches the key) and
the framework appends only mesh coordinates, which do not identify which mesh. Adding the id makes
an entry non-transferable between two `MeshDevice`s. It does not compensate for any of the omissions
above.

## Framework side effect of having a custom hash

```1012:1014:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
        if constexpr (requires { DeviceOperation::compute_program_hash(attrs, tensor_args); }) {
            return key;  // custom hash -> opt out beyond the op-identity prefix
        } else {
```

`ProgramCacheKey::canonical` degrades to the op type name, so a 64-bit collision resolves to a wrong
hit rather than a rebuild. Given how few terms this hash has, the effective key space here is also
unusually small.

## Summary

| Omitted vs. default | Used by program? | Patched on hit? | Verdict |
|---|---|---|---|
| `matmul.input_tensor`, `matmul.weight_tensor` | Yes — M/K/N, tile sizes, all CB formats, work split | Addresses only | **BUG** |
| `operation_attributes.matmul` (all `MatmulParams`) | Yes — compute kernel structure, blocking, activation, output dtype | No | **BUG** |
| `second_weight_tensor` (presence) | Yes — selects the program branch and the output arity | No | **BUG** |
| `rs_op.output_mem_config` | Yes — `OUTPUT_CORE_XY` define, output/accumulator CB sizes, compile args | No | **BUG** |
| `rs_op.subdevice_id` | Yes — matmul core pool and mcast origin | No | **BUG** |
| `rs.intermediate_packet_buffer` spec | Yes — packet-worker cores, and via `restricted_cores` the matmul placement too | Address only | **BUG** |
| `rs.input_tensor.logical_shape` | Yes — `ncores_input`, `SCHEDULE` define, compile args | No | **BUG** |
| `rs_op.cross_device_semaphore` | Yes — reader/writer slot 0 | Yes | VALID — patched |
| `rs.input_tensor.page_config` (`Tile`) / `layout` | Yes — tiles-per-core arithmetic, and every CB page size via a hardcoded 32x32 `tile_size()` | No | **BUG** |
| `rs.input_tensor.alignment` | Only via hashed derivatives | n/a | VALID — unused |
| `rs.input_tensor` storage kind | n/a | n/a | CAVEAT — pinned only on the miss path (faults in the factory rather than aliasing) |
| All buffer addresses | Yes | Yes | VALID — patched |
| `ring_index`, fabric neighbours | Yes — compile args | n/a | VALID — invariant (keyed via the mesh coordinates the framework appends) |
| Fabric connection rt args | Yes | No | CAVEAT — relies on fixed fabric config |

**Program-cache bugs were found — eight of them, and this is the weakest hash of the four CCL ops in
this audit.** The root cause is structural rather than incidental: `compute_program_hash` is a
verbatim copy of the standalone `LlamaReduceScatterDeviceOperation` key, and the matmul half of the
fusion was never added to it. Two calls that differ *only* in their matmul — different weights,
different program config, different activation, different output dtype — are indistinguishable to the
cache. The eighth is the tile omission (#9), which is independent of the fusion problem and would be
present in the standalone reduce-scatter too. The `validate_on_program_cache_hit` hook, which might
have been expected to catch this, only re-checks the current call's internal consistency and provides
no protection.

The op works in the Llama model because every one of these parameters is fixed there. It is not safe
against its own public API.

## Recommendations

1. Add the matmul-side tensors to the hash. At minimum
   `tensor_args.matmul.input_tensor.{padded_shape, dtype, layout, memory_config}` and the same four
   for `weight_tensor`; mirror whatever selectivity the standalone
   `ttnn::prim::MatmulDeviceOperation` hash applies so the two stay consistent.
2. Hash `operation_attributes.matmul`. It is fully reflectable, so a single extra argument to
   `hash_operation` covers `program_config`, `compute_kernel_config`, `output_dtype`,
   `output_mem_config`, `activation`, `transpose_a/b`, `output_tile` and `global_cb` at once.
3. Hash `tensor_args.second_weight_tensor.has_value()` at the very least — one bit that currently
   selects between two incompatible programs — and preferably the tensor's properties as in
   recommendation 1.
4. Hash `rs_op.output_mem_config` and `rs_op.subdevice_id`.
5. Hash `tensor_args.rs.intermediate_packet_buffer.memory_config()`; it moves both halves of the
   fusion through `restricted_cores`.
6. Hash `tensor_args.rs.input_tensor.logical_shape()`. The over-provisioned-shard-grid case this op
   explicitly supports makes the shape genuinely independent of the hashed memory config.
7. Close the tile gap (#9), **in the shared factory rather than here**. The hardcoded
   `tile_size(cb_data_format)` calls are at
   `ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter/device/llama_reduce_scatter_program_factory.cpp:315`
   and `:426`, and the tile-count arithmetic that already reads the real tile is in the same file at
   `:298`, `:302` and `:398-411`. Two ops build through that factory and both are affected:
   `experimental/ccl/llama_reduce_scatter_matmul` (this op) and the standalone
   `experimental/ccl/llama_reduce_scatter`. One change fixes both; a fix applied only in
   `llama_reduce_scatter_matmul/` would fix neither.

   The cheapest correct form is the standard guard from `interleaved_to_sharded_op.cpp:95-97` added to
   each op's `validate_on_program_cache_miss` — that pins `page_config` out of the key and makes the
   hardcoded `tile_size` correct by construction. If non-32x32 tiles are meant to be supported
   instead, then replacing the two `tile_size` calls with
   `input_tensor.tensor_spec().tile().get_tile_size(cb_data_format)` and hashing
   `tensor_spec().page_config()` must land in the same change. Fixing only the page size would leave
   a factory that is fully tile-aware and a key that still cannot tell two tiles apart, which reads as
   correct and aliases anyway. Note that hashing `page_config()` covers the tile *shape* only:
   `Tile::attribute_values()` exposes just `tile_shape`, `face_shape` and `num_faces`
   (`tt_metal/api/tt-metalium/tile.hpp:46-47`) and `Tile::operator==` compares only the first two
   (`tt_metal/impl/data_format/tile.cpp:122-124`), so `transpose_within_face` and `transpose_of_faces`
   stay invisible to both halves of the key. That hole is framework-wide rather than specific to this
   op, and only an explicit `TT_FATAL` on the two transpose accessors closes it.
8. Consider deriving this op's hash by *composing* the two component hashes —
   `hash_objects(LlamaReduceScatter-key, Matmul-key)` — rather than hand-copying one of them. That
   makes the fusion's key automatically track future changes to either component and prevents this
   class of copy-and-forget regression.
