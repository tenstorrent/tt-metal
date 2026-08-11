# Program Cache Audit — `experimental/ccl/llama_all_gather_matmul_async`

Audit of `ttnn::experimental::prim::LlamaAllGatherMatmulAsyncDeviceOperation::compute_program_hash`
against the framework default ("hash everything") key.

| | |
|---|---|
| Device operation | `ttnn::experimental::prim::LlamaAllGatherMatmulAsyncDeviceOperation` (`device/llama_all_gather_matmul_async_device_operation.hpp:22-33`) |
| Custom hash | `device/llama_all_gather_matmul_async_device_operation.cpp:91-130` |
| `operation_attributes_t` | `LlamaAllGatherMatmulAsyncParams` — `matmul_struct`, `devices`, `dim`, `num_links`, `ring_size`, `output_memory_config`, `topology`, `semaphore`, `sub_device_id`, `cluster_axis` |
| `tensor_args_t` | `LlamaAllGatherMatmulAsyncInputs` — `input0`, `input1`, `intermediate` |
| Program factories | `LlamaAllGatherMatmulAsyncProgramFactory` (single, mesh-workload style; internally also builds the fused matmul via `llama_1d_mm_fusion.cpp`) |
| `override_runtime_arguments` | **Yes** (`device/llama_all_gather_matmul_async_program_factory.cpp:495-544`) |
| `get_dynamic_runtime_args` | No |
| `validate_on_program_cache_miss` | Yes, but it inspects only `input0` and `num_links` |
| Cache-hit patch mechanism | **Op-owned re-derivation** (the factory's `override_runtime_arguments` runs on every hit) |

The CSV row for this op reads *explicit / SELECTIVE / no own hit validator / has `override_runtime_arguments` /
no `get_dynamic_runtime_args`*. All five columns are correct against the code: there is genuinely no
`validate_on_program_cache_hit` anywhere in this op directory, so nothing at all is checked on a
reuse. What the row does not convey is that the *miss*-time validator is also very thin
(`device/llama_all_gather_matmul_async_device_operation.cpp:12-33`) — it never looks at `input1`, at
`tensor_args.intermediate`, or at any field of `matmul_struct`, so it pins almost nothing that the
hash drops. This op is therefore the least-guarded of the four: no hit validation, and a miss
validator that covers only `input0` and `num_links`.

## Cache-hit patch mechanism

This factory is a mesh-workload factory (`create_mesh_workload` returning `AdaptedCachedMeshWorkload`)
and exposes no `apply_descriptor`, so the framework takes the `override_runtime_arguments` branch of
the cache-hit dispatcher:

```279:285:ttnn/api/ttnn/device_operation.hpp
        if constexpr (requires { &WorkloadFactory::apply_descriptor; }) {
            WorkloadFactory::apply_descriptor(
                cached_mesh_workload, operation_attributes, tensor_args, tensor_return_value);
        } else {
            WorkloadFactory::override_runtime_arguments(
                cached_mesh_workload, operation_attributes, tensor_args, tensor_return_value);
        }
```

This is the strongest of the three cache-hit modes *in principle*: the op re-derives per-dispatch
state itself, so nothing is inferred by the framework. But it is also the mode with the least safety
net — there is no `resolve_bindings`, no automatic circular-buffer address patching, and no
`get_dynamic_runtime_args`. Anything the op's own `override_runtime_arguments` does not explicitly
rewrite stays frozen at the values computed on the first miss, and everything structural (kernel
compile-time args, CB sizes and data formats, core ranges, program-scoped semaphores) is baked into
the cached `Program` and is never refreshed at all.

The resulting obligation on the hash is therefore:

1. every compile-time arg, CB geometry/format, and core range must be a pure function of the hashed
   set (plus the mesh coordinates the framework appends), and
2. every runtime arg and every globally-allocated CB address that varies per call must appear
   explicitly in `override_runtime_arguments`.

This op violates both.

## Baseline: what the default hash would cover

`hash_objects_with_default_seed(type_hash<Op>, attrs, tensor_args)` walks reflection over the whole
attribute struct and the whole tensor-args struct. Note that `LlamaAllGatherMatmulAsyncParams`
defines an `attributes()` method (`device/llama_all_gather_matmul_async_device_operation_types.hpp:58-76`)
that omits `matmul_struct` — but `attributes()` is *not* a hashing hook. `ttsl::hash::hash_object`
dispatches on `to_hash()`, then the `attribute_names`/`attribute_values()` pair, then containers, and
finally `reflect::for_each` over public members
(`tt_stl/tt_stl/reflection.hpp:1314-1334`, `tt_stl/tt_stl/reflection.hpp:1418-1424`); `attributes()`
is used for printing, not hashing. So the default key would be:

| Source | Fields |
|---|---|
| `operation_attributes` | `matmul_struct` (all 14 `MatmulParams` fields, including `program_config`, `output_dtype`, `output_mem_config`, `compute_kernel_config`, `global_cb`, `sub_device_id`), `devices`, `dim`, `num_links`, `ring_size`, `output_memory_config`, `topology`, `semaphore`, `sub_device_id`, `cluster_axis` |
| `input0` | storage kind, `logical_shape`, `dtype`, `page_config`, `memory_config`, `alignment` |
| `input1` | storage kind, `logical_shape`, `dtype`, `page_config`, `memory_config`, `alignment` |
| `intermediate` | storage kind, `logical_shape`, `dtype`, `page_config`, `memory_config`, `alignment` |
| appended by framework | the mesh coordinates of the tensors |

## What the custom hash covers

```111:129:ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_all_gather_matmul_async_device_operation.cpp
    return tt::tt_metal::operation::hash_operation<LlamaAllGatherMatmulAsyncDeviceOperation>(
        args.dim,
        args.num_links,
        args.ring_size,
        args.output_memory_config,
        args.topology,
        args.cluster_axis,
        input0_shape,
        input0_memory_layout,
        input0_dtype,
        input0_memory_config,
        input1_shape,
        input1_memory_layout,
        input1_dtype,
        input1_memory_config,
        intermediate_shape,
        intermediate_memory_layout,
        intermediate_dtype,
        intermediate_memory_config);
```

The `intermediate_*` locals do **not** come from the intermediate tensor. They are read off `input1`:

```106:109:ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_all_gather_matmul_async_device_operation.cpp
    auto intermediate_shape = input1.padded_shape();
    auto intermediate_memory_layout = input1.layout();
    auto intermediate_dtype = input1.dtype();
    auto intermediate_memory_config = input1.memory_config();
```

So the last four hash terms are exact duplicates of the four preceding ones, and
`tensor_args.intermediate` contributes nothing to the key. This is treated as omission #2 below.

Effective hashed set:
`{dim, num_links, ring_size, output_memory_config, topology, cluster_axis, input0.padded_shape,
input0.layout, input0.dtype, input0.memory_config, input1.padded_shape, input1.layout, input1.dtype,
input1.memory_config}` + mesh coordinates.

## Omitted parameters

### 1. `operation_attributes.matmul_struct` (the entire `MatmulParams`)

**Verdict: BUG.**

`MatmulParams` carries the matmul program config, the compute-kernel config, the output dtype, the
output memory config, the global CB and the sub-device id
(`ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation_types.hpp:15-30`). None of these
appear in the hash. All of them are user-settable per call from Python — `program_config`,
`compute_kernel_config`, `dtype`, `global_cb`, `mm_memory_config` are all keyword arguments of the
binding (`llama_all_gather_matmul_async_nanobind.cpp:83-86`) and are packed into `MatmulParams`
verbatim (`device/llama_all_gather_matmul_async_device_operation.cpp:174-192`).

The factory reads them directly and hands them to the fused-matmul builder:

```63:65:ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_all_gather_matmul_async_program_factory.cpp
    const auto& compute_kernel_config = args.matmul_struct.compute_kernel_config.value();
    const auto& program_config = args.matmul_struct.program_config.value();
    const auto& global_cb = args.matmul_struct.global_cb;
```

```468:482:ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_all_gather_matmul_async_program_factory.cpp
    auto matmul_shared_variables = ttnn::operations::llama_matmul::matmul_multi_core_agmm_fusion_helper(
        program,
        aggregated_tensor,         // in0
        {input1},                  // in1
        std::nullopt,              // bias
        {output_tensor},           // out0
        false,                     // broadcast_batch
        compute_kernel_config,     // compute_kernel_config
        program_config,            // program_config
        false,                     // untilize_out
        matmul_fused_op_signaler,  // fused_op_signaler
        global_cb,                 // global_cb
        args.sub_device_id,        // sub_device_id
        matmul_fused_op_signaler->start_cb_index,
        std::nullopt);
```

Every field of the program config becomes a structural parameter of the compute kernel:

```942:960:ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_1d_mm_fusion.cpp
        config.compute_with_storage_grid_size,
        compute_kernel_config,
        ttnn::get_throttle_level(compute_kernel_config),
        config.in0_block_w,
        config.out_subblock_h,
        config.out_subblock_w,
        config.out_block_h,
        config.out_block_w,
        config.per_core_M,
        config.per_core_N,
        config.fuse_batch,
        config.fused_activation,
        config.mcast_in0,
        config.gather_in0,
        config.hop_cores,
        untilize_out,
        fused_op_signaler,
        global_cb,
        config.num_global_cb_receivers,
```

and the compute-kernel config expands into compile-time math settings, while the *output dtype*
selects the pack data format:

```816:817:ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_1d_mm_fusion.cpp
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);
```

```761:763:ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_1d_mm_fusion.cpp
    tt::DataFormat in0_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());          // in0
    tt::DataFormat in1_data_format = tt_metal::datatype_to_dataformat_converter(b.dtype());          // in1
    tt::DataFormat output_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());  // output
```

None of that is refreshed on a hit. The matmul's cache-hit hook only rewrites CB base addresses and
one writer runtime arg:

```670:707:ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_1d_mm_fusion.cpp
    const auto& global_cb = operation.global_cb;

    auto* src_buffer_a = input_tensors[0].buffer();
    auto* src_buffer_b = input_tensors[1].buffer();

    bool src0_sharded = input_tensors[0].is_sharded();
    bool src1_sharded = input_tensors[1].is_sharded();
    bool out_sharded = output_tensors[0].is_sharded();

    // Manually unroll sender core
    if (src0_sharded) {
        UpdateDynamicCircularBufferAddress(program, override_variables.cbs[0], *src_buffer_a);
    }
    if (src1_sharded) {
        if (!global_cb.has_value() && !src_buffer_b->is_dram()) {
            UpdateDynamicCircularBufferAddress(program, override_variables.cbs[1], *src_buffer_b);
        }
    }
    if (out_sharded) {
        for (uint32_t i = 0; i < override_variables.cbs.size() - 2; ++i) {
            // cbs 0 and 1 contain cb_src0 and cb_src1
            // the rest contains the actual output cbs
            const auto& cb_output = override_variables.cbs[i + 2];
            const auto& out_buffer = output_tensors[i].buffer();
            UpdateDynamicCircularBufferAddress(program, cb_output, *out_buffer);
        }
    }

    if (not src1_sharded) {
        auto& writer_runtime_args_by_core = GetRuntimeArgs(program, override_variables.kernels.at(0));
        for (const auto& core : override_variables.cores) {
            auto& writer_runtime_args = writer_runtime_args_by_core[core.x][core.y];

            /* in1 */
            writer_runtime_args[1] = src_buffer_b->address();
        }
    }
```

**Reproduction.** With a fixed `input_tensor0`, `input_tensor1`, `intermediate_tensor`, `dim`,
`cluster_axis`, `topology` and `ag_memory_config`:

- Call 1: `ttnn.experimental.llama_all_gather_matmul_async(..., dtype=ttnn.bfloat16)`
- Call 2: identical, but `dtype=ttnn.bfloat8_b`

Both calls produce the same 64-bit key (the hash never touches `matmul_struct`). Call 2 hits call 1's
cache entry. `create_output_tensors` correctly allocates a `bfloat8_b` output, and
`override_agmm_fusion_program_parameters` correctly re-points the output CB at it — but the cached
compute kernel was compiled with `output_data_format = Float16_b` and the output CB was created with
a 2048-byte tile page size. The packer writes 32x32xbf16 tiles into a buffer laid out for
`bfloat8_b` tiles: the results are numerically garbage and the write runs off the end of the shard.

The same reproduction works for `program_config` (change `per_core_N` or `in0_block_w` and the
compute kernel's blocking compile-time args go stale, together with the CB depths and the core grid),
and for `compute_kernel_config` (change `math_fidelity` or `fp32_dest_acc_en` and the cached kernel
silently keeps the old precision — a quieter, harder-to-spot variant).

Note that in the *fused* path some of these do partly ride on hashed values: when the caller passes
`program_config=None`, `create_matmul_attributes` derives the config from the input tensors
(`ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp:2736-2796`), which are hashed.
That is why the op works in the Llama model, where the config is fixed. It does not make the hash
correct — the config is an explicit public argument and nothing pins it.

### 2. `tensor_args.intermediate` (all six tensor properties)

**Verdict: BUG.**

As shown above the "intermediate" hash terms are aliases of `input1`, so the intermediate tensor's
shape, dtype, layout and memory config are all absent from the key. `validate_on_program_cache_miss`
does not examine the intermediate tensor either
(`device/llama_all_gather_matmul_async_device_operation.cpp:12-33`), so nothing pins it.

The factory uses the intermediate tensor's memory config to place a kernel and to size a CB — both
structural:

```186:189:ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_all_gather_matmul_async_program_factory.cpp
    const auto intermediate_tensor_shard_shape = intermediate_tensor.memory_config().shard_spec()->shape;
    const auto intermediate_tensor_shard_num_pages =
        intermediate_tensor_shard_shape[0] * intermediate_tensor_shard_shape[1] / TILE_HW;
    const auto intermediate_tensor_page_size = intermediate_tensor.buffer()->page_size();
```

```213:219:ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_all_gather_matmul_async_program_factory.cpp
    uint32_t inter_cb_index = tt::CB::c_in2;
    tt::tt_metal::CircularBufferConfig cb_inter_config =
        tt::tt_metal::CircularBufferConfig(
            intermediate_tensor_shard_num_pages * intermediate_tensor_page_size, {{inter_cb_index, df}})
            .set_page_size(inter_cb_index, intermediate_tensor_page_size)
            .set_globally_allocated_address(*intermediate_tensor.buffer());
    CreateCircularBuffer(program, intermediate_tensor_cores, cb_inter_config);
```

`intermediate_tensor_cores` is the receiver kernel's core range
(`...program_factory.cpp:174`, used at `...program_factory.cpp:293`), and it is also subtracted from
the pool the CCL sender workers are chosen from (`...program_factory.cpp:175-179`). None of that is
touched on a cache hit.

**Reproduction.** Call 1 with `intermediate_tensor` width-sharded over a 3x1 core grid; call 2 with a
byte-identical intermediate tensor width-sharded over a 6x1 grid (same total elements, half the shard
width). Nothing in the hash changes. Call 2 reuses call 1's program: the receiver kernel still exists
only on the 3 original cores, `cb_inter` is still sized for the wide shard, and the
`intermediate_tensor_shard_num_pages` runtime arg (`...program_factory.cpp:338`) still describes the
old shard. Half the gathered data is never multicast to the matmul cores, and the CB overruns the new
(narrower) shard on the cores that do run.

Callers today derive the intermediate tensor from `ag_memory_config`, which *is* hashed, so the two
move together in practice — that is the only reason this has not fired.

### 3. Intermediate tensor's globally-allocated CB address

**Verdict: BUG.**

Separate from #2, and true even when the intermediate tensor's *spec* is unchanged. `cb_inter` is
bound to the intermediate tensor's buffer via `set_globally_allocated_address`
(`...program_factory.cpp:218`), and the receiver kernel reads through it:

```77:81:ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/kernels/worker_receiver.cpp
    size_t l1_read_addr = cb_inter.get_read_ptr();
    const uint64_t multicast_addr_noc = get_noc_multicast_addr(bbox_start_x, bbox_start_y, bbox_end_x, bbox_end_y, 0);
    uint64_t aggregated_tensor_addr_this_core =
        (uint64_t)aggregated_tensor_addr + mm_core_offset * intermediate_tensor_shard_num_pages * tensor0_page_size;
    const uint64_t multicast_addr = multicast_addr_noc | aggregated_tensor_addr_this_core;
```

`override_runtime_arguments` never calls `UpdateDynamicCircularBufferAddress` for `cb_inter`
(`...program_factory.cpp:495-544`); it patches the intermediate address only into the *reader* and
*writer* runtime args:

```516:534:ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_all_gather_matmul_async_program_factory.cpp
        for (const auto& core : shared_vars.sender_worker_cores) {
            // reader
            auto& worker_reader_sender_runtime_args = worker_reader_sender_runtime_args_by_core[core.x][core.y];
            worker_reader_sender_runtime_args[0] = input0.buffer()->address();
            worker_reader_sender_runtime_args[1] = intermediate_tensor.buffer()->address();
            worker_reader_sender_runtime_args[8] = args.semaphore.address();
            // writer
            auto& worker_writer_sender_runtime_args = worker_writer_sender_runtime_args_by_core[core.x][core.y];
            worker_writer_sender_runtime_args[0] = intermediate_tensor.buffer()->address();
            worker_writer_sender_runtime_args[1] = args.semaphore.address();
        }

        // update worker receiver
        auto& worker_receiver_runtime_args_by_core = GetRuntimeArgs(program, shared_vars.worker_receiver_kernel_id);
        for (const auto& core : shared_vars.intermediate_cores_vec) {
            auto& worker_receiver_runtime_args = worker_receiver_runtime_args_by_core[core.x][core.y];
            worker_receiver_runtime_args[0] = args.semaphore.address();
            worker_receiver_runtime_args[3] = aggregated_tensor.buffer()->address();
        }
```

Note the asymmetry: the matmul's in0/in1/out CBs *are* re-pointed (via
`override_agmm_fusion_program_parameters`, quoted in #1) and the sender-side intermediate addresses
*are* re-written, but the receiver's `cb_inter` is not. This is a straightforward omission.

**Reproduction.** Call 1 with intermediate tensor `I1`. Deallocate `I1`, allocate an unrelated L1
tensor, then allocate `I2` with the identical spec at a different L1 address, and issue call 2. The
hash is unchanged (addresses are never hashed, by design), the reader/writer write into `I2`, but
`cb_inter.get_read_ptr()` still resolves to `I1`'s old address, so the receiver multicasts whatever
now lives there.

### 4. `operation_attributes.semaphore` (`GlobalSemaphore`)

**Verdict: VALID — patched.**

The semaphore's L1 address is a per-call allocation and correctly absent from the key. It appears in
three runtime-arg slots at build time — the receiver's arg 0
(`...program_factory.cpp:299`, `...program_factory.cpp:329`), the reader's arg 8
(`...program_factory.cpp:414`) and the writer's arg 1 (`...program_factory.cpp:434`) — and
`override_runtime_arguments` rewrites exactly those three slots on every hit
(lines 521, 525 and 532, quoted in #3). It is never used as a compile-time arg and never baked into a
CB. This is the textbook correct handling.

### 5. `operation_attributes.sub_device_id`

**Verdict: BUG.**

`sub_device_id` selects the worker-core pool that the CCL sender cores are carved out of:

```162:179:ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_all_gather_matmul_async_program_factory.cpp
    auto sub_device_core_range_set = mesh_device->worker_cores(
        tt::tt_metal::HalProgrammableCoreType::TENSIX,
        args.sub_device_id.value_or(mesh_device->get_sub_device_ids().at(0)));
    // auto bbox = sub_device_core_range_set.bounding_box();
    // CoreRangeSet bbox_crs(bbox);

    auto aggregated_tensor_cores = aggregated_tensor.memory_config().shard_spec()->grid;
    auto bbox = aggregated_tensor_cores.bounding_box();
    auto bbox_physical_start_core = mesh_device->worker_core_from_logical_core(bbox.start_coord);
    auto bbox_physical_end_core = mesh_device->worker_core_from_logical_core(bbox.end_coord);

    auto output_tensor_cores = output_tensor.memory_config().shard_spec()->grid;
    auto intermediate_tensor_cores = intermediate_tensor.memory_config().shard_spec()->grid;
    auto available_cores = sub_device_core_range_set.subtract(intermediate_tensor_cores);
    available_cores = available_cores.subtract(output_tensor_cores);

    const auto [sender_worker_core_range, sender_worker_cores] =
        ar_choose_worker_cores(args.num_links, num_workers_per_link, available_cores);
```

`sender_worker_core_range` is where the reader and writer kernels are *created*
(`...program_factory.cpp:248`, `...program_factory.cpp:272`) — a structural property of the cached
`Program`. It is also forwarded into the matmul builder (`...program_factory.cpp:480`), where it
constrains the matmul core placement. `override_runtime_arguments` iterates
`shared_vars.sender_worker_cores`, i.e. the *cached* core list, so a hit cannot relocate kernels.

**Reproduction.** Call 1 with `subdevice_id=None` (falls back to sub-device 0, the full grid); call 2
with `subdevice_id=<a sub-device covering only the top half of the grid>`, everything else identical.
The hash is unchanged, so call 2 hits, and the CCL workers keep running on cores outside the
requested sub-device. On a mesh where the other half of the grid is concurrently owned by another
sub-device's program this is a hard correctness and dispatch-ordering violation, not just a
performance surprise.

This is the classic "structural, baked into the cached Program, not refreshed by
`override_runtime_arguments`" case called out for fabric/sub-device parameters.

### 6. `operation_attributes.devices`

**Verdict: VALID — unused (on every reachable call path).**

`devices` is only consulted when `cluster_axis` has no value:

```73:85:ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_all_gather_matmul_async_program_factory.cpp
    if (args.cluster_axis.has_value()) {
        devices_to_use = (args.cluster_axis.value() == 0) ? mesh_view.get_devices_on_column(mesh_coordinate[1])
                                                          : mesh_view.get_devices_on_row(mesh_coordinate[0]);
        fabric_node_ids = (args.cluster_axis.value() == 0) ? mesh_view.get_fabric_node_ids_on_column(mesh_coordinate[1])
                                                           : mesh_view.get_fabric_node_ids_on_row(mesh_coordinate[0]);
    } else {
        devices_to_use = args.devices;
        fabric_node_ids.reserve(devices_to_use.size());
        for (auto* device : devices_to_use) {
            auto coord = mesh_view.find_device(device->id());
            fabric_node_ids.push_back(mesh_device->get_fabric_node_id(coord));
        }
    }
```

`cluster_axis` is a non-optional `uint32_t` on the only entry point
(`device/llama_all_gather_matmul_async_device_operation.hpp:44` and
`llama_all_gather_matmul_async.cpp:18`) and is stored into the optional unconditionally
(`device/llama_all_gather_matmul_async_device_operation.cpp:204`), so `cluster_axis.has_value()` is
always true and the `devices` branch is dead. `devices` also holds raw `IDevice*` pointers, which are
exactly the kind of value one should *not* hash. Dropping it is correct — but see the recommendation
below about deleting the dead branch so the invariant is enforced rather than incidental.

### 7. `ring_index` / device index, forward and backward fabric neighbours

**Verdict: VALID — invariant** (determined by the mesh coordinates the framework appends to every
key, plus hashed attributes).

This deserves an explicit argument rather than an assumption. `ring_index` is a genuine compile-time
arg:

```235:239:ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_all_gather_matmul_async_program_factory.cpp
    reader_kernel_config.compile_args = {
        ring_index,                 // my_chip_id
        src0_cb_index,              // cb0_id
        op_config.get_page_size(),  // tensor0_page_size
    };
```

and it, plus `num_targets_forward` / `num_targets_backward` / `dynamic_alternate`
(`...program_factory.cpp:260-262`), plus the `forward_fabric_node_id` / `backward_fabric_node_id`
selection (`...program_factory.cpp:87-105`), all derive from exactly three things: `mesh_coordinate`,
`args.cluster_axis`, `args.ring_size` and `args.topology`. The last three are hashed explicitly; the
first is appended by the framework to both the default and the custom hash path:

```989:992:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
        // Combine with the mesh coordinates the workload is targeting.
        for (const auto& coord : mesh_device_operation_utils::extract_tensor_coordinates(tensor_args, mesh_device)) {
            hash = ttsl::hash::hash_objects(hash, coord);
        }
```

so per-device variation is already keyed. Note this only works because the *set* of coordinates is
folded in; one cache entry holds the whole mesh workload, and a call over a different set of
coordinates gets a different key.

### 8. Fabric connection runtime args

**Verdict: CAVEAT.**

`append_fabric_connection_rt_args` pushes router coordinates, the EDM buffer base address and the
flow-control semaphore addresses onto the writer's runtime args at build time:

```451:462:ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_all_gather_matmul_async_program_factory.cpp
        writer_rt_args.push_back(forward_fabric_node_id.has_value());
        if (forward_fabric_node_id.has_value()) {
            const auto sender_fabric_node_id = mesh_device->get_fabric_node_id(mesh_coordinate);
            tt::tt_fabric::append_fabric_connection_rt_args(
                sender_fabric_node_id, forward_fabric_node_id.value(), link, program, {core}, writer_rt_args);
        }
        writer_rt_args.push_back(backward_fabric_node_id.has_value());
        if (backward_fabric_node_id.has_value()) {
            const auto sender_fabric_node_id = mesh_device->get_fabric_node_id(mesh_coordinate);
            tt::tt_fabric::append_fabric_connection_rt_args(
                sender_fabric_node_id, backward_fabric_node_id.value(), link, program, {core}, writer_rt_args);
        }
```

`override_runtime_arguments` rewrites only slots 0 and 1 of the writer args, so these tail slots are
frozen at first-miss values. That is safe as long as the fabric configuration is fixed for the
lifetime of a mesh device (it is established at fabric init and does not change per op call), and it
is the same assumption every fabric-based CCL op in the tree makes. The assumption that would break
it is a fabric teardown/re-init, or a change of `num_links` routing that reassigns EDM channels,
between two calls that share a cache entry. `num_links` is hashed, which closes the most likely
variant. Worth stating explicitly rather than leaving implicit.

### 9. Tensor properties dropped from `input0` and `input1`

**Verdict: mixed — one label per property below. All VALID except `page_config`, which is a **BUG**
adjudicated separately in #11.**

- **`logical_shape` replaced by `padded_shape`** — VALID — relaxation win. The factory works entirely
  in pages and shard shapes (`...program_factory.cpp:182-188`), never in logical elements. Two calls
  whose logical shapes differ but pad to the same tiled shape legitimately share a program; the
  default hash would force a recompile. The per-call output `TensorSpec` is still recomputed by
  `compute_output_specs` on every invocation, so the returned tensor carries the right logical shape.
- **`page_config` reduced to `layout()`** — BUG. `layout()` collapses `PageConfig` to
  `ROW_MAJOR`/`TILE`, discarding the `Tile` shape, and this op's two halves disagree about whether
  that matters. Adjudicated in full in #11 below.
- **`alignment`** — VALID — unused (low residual risk). It reaches the program only through the buffer page size
  and `padded_shape`, both of which are determined by the hashed `{memory_config, dtype,
  padded_shape}` for the layouts this op accepts (`device/llama_all_gather_matmul_async_device_operation.cpp:26-32`).
- **storage kind** — VALID — pinned by validation (the op declares no
  `validate_on_program_cache_hit`, so this `TT_FATAL` is substituted onto the hit path):

```18:20:ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_all_gather_matmul_async_device_operation.cpp
    TT_FATAL(input0.storage_type() == StorageType::DEVICE, "Operands to llama_all_gather_matmul need to be on device!");
    TT_FATAL(
        input0.buffer() != nullptr, "Operands to llama_all_gather_matmul need to be allocated in buffers on device!");
```

  This only covers `input0`; `input1` and `intermediate` are unchecked, but a host-storage tensor
  would fail earlier at `buffer()` dereference in the factory, so no cache aliasing is reachable.

### 10. Buffer addresses of `input0`, `input1`, `aggregated`, `output`

**Verdict: VALID — patched.** Addresses must not be hashed. `input0` and the intermediate go through
the reader/writer patch (quoted in #3), the aggregated tensor through receiver arg 3
(`...program_factory.cpp:533`), and the matmul in0/in1/out through
`override_agmm_fusion_program_parameters` (quoted in #1). The one gap in this family is `cb_inter`,
covered as its own finding in #3.

### 11. Tile geometry — the unguarded 32x32 assumption

**Verdict: BUG.**

The hash keeps `layout()` but not `tensor_spec().page_config()`
(`device/llama_all_gather_matmul_async_device_operation.cpp:97`, `:102`, `:107`), so the `Tile` shape
is not in the key for any of `input0`, `input1` or the intermediate. This op is the *mixed* case: its
two halves use opposite idioms, and each half is independently broken by the omission.

**The all-gather half hardcodes 32x32.** Shard tile counts come from the architectural constant:

```184:189:ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_all_gather_matmul_async_program_factory.cpp
    const auto input_tensor_shard_shape = input0.memory_config().shard_spec()->shape;
    const auto input_tensor_shard_num_pages = input_tensor_shard_shape[0] * input_tensor_shard_shape[1] / TILE_HW;
    const auto intermediate_tensor_shard_shape = intermediate_tensor.memory_config().shard_spec()->shape;
    const auto intermediate_tensor_shard_num_pages =
        intermediate_tensor_shard_shape[0] * intermediate_tensor_shard_shape[1] / TILE_HW;
    const auto intermediate_tensor_page_size = intermediate_tensor.buffer()->page_size();
```

and the weight width from a bare literal:

```114:114:ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_all_gather_matmul_async_program_factory.cpp
    const uint32_t weight_tensor_width = input1.padded_shape()[3] / 32;
```

`intermediate_tensor_shard_num_pages` sizes the `cb_inter` circular buffer
(`...program_factory.cpp:216`) and is a compile-time arg of the receiver kernel
(`...program_factory.cpp:308` and `...program_factory.cpp:338`);
`input_tensor_shard_num_pages` determines which input cores each worker reads from
(`...program_factory.cpp:378-379`). Note also that `intermediate_tensor_page_size` on the very next
line is read from the buffer and *is* tile-aware, so this half mixes a correct non-32x32 page size
with tile counts computed for 32x32 — the two sides of the address arithmetic disagree.

**The matmul half is genuinely tile-aware.** It reads the real tile off both operands and constructs
the output tile from them:

```753:758:ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_1d_mm_fusion.cpp
    auto in0_tile = a.tensor_spec().tile();
    auto in1_tile = b.tensor_spec().tile();
    // cannot use the output tensor tile directly as that might be changed by user override
    auto in0_tile_shape = in0_tile.get_tile_shape();
    auto in1_tile_shape = in1_tile.get_tile_shape();
    auto output_tile = tt::tt_metal::Tile({in0_tile_shape[0], in1_tile_shape[1]});
```

and derives the entire work split from it:

```824:827:ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_1d_mm_fusion.cpp
    uint32_t B = get_batch_size(ashape);
    uint32_t Mt = ashape[-2] / in0_tile_shape[0];
    uint32_t Kt = ashape[-1] / in0_tile_shape[1];
    uint32_t Nt = bshape[-1] / in1_tile_shape[1];
```

The tile-derived values are pervasive in the generated program. The per-tile byte sizes set every CB
page size, and the `Tile` object itself is baked into the CB configuration:

```199:204:ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_1d_mm_fusion.cpp
    tt_metal::CircularBufferConfig src0_cb_config =
        tt_metal::CircularBufferConfig(in0_CB_size, {{src0_cb_index, in0_data_format}})
            .set_page_size(src0_cb_index, in0_single_tile_size)
            .set_tile_dims(src0_cb_index, in0_tile)
            .set_globally_allocated_address(*in0_buffer);
    auto cb_src0 = tt_metal::CreateCircularBuffer(program, all_cores, src0_cb_config);
```

and they reach compile-time args in all three kernels: `multicast_chunk_width_in_tiles` (derived from
`Kt_total = in0_buffer->shard_spec().shape()[1] / in0_tile.get_tile_shape()[1]` at
`llama_1d_mm_fusion.cpp:115`) is receiver arg 0 at `llama_1d_mm_fusion.cpp:328`;
`in1_tensor_width_in_tiles`, `in1_block_page_size`, `in1_block_page_size_last`,
`in1_block_width_num_pages` and `in1_shard_width_in_dram` are in1-writer args at
`llama_1d_mm_fusion.cpp:350-356`; and `in0_block_w`, `in0_block_num_tiles`, `in0_subblock_num_tiles`,
`in1_block_num_tiles`, `in1_block_size_bytes` and `in1_tensor_size_bytes` are compute-kernel args at
`llama_1d_mm_fusion.cpp:371-379`.

**No guard anywhere.** `validate_on_program_cache_miss`
(`device/llama_all_gather_matmul_async_device_operation.cpp:12-33`) checks page alignment, storage,
`num_links` and the memory layout, and never touches `tile()`. The matmul helper's only tile-related
assertions are divisibility checks against the tile shape it has already read
(`llama_1d_mm_fusion.cpp:795-814`), which every valid tile satisfies by construction.

So all three adjudication criteria hold for the all-gather half (accepts `Layout::TILE`, bare
`TILE_HW`/literal-32 tile-count conversion, no tile-geometry guard), and the mirror-image case holds
for the matmul half (provably varies with `Tile`, `Tile` not in the key). Both point at the same
missing hash term.

**Reproduction.** Two calls with identical padded shapes, dtypes, memory configs, `num_links`,
`ring_size`, `topology` and `cluster_axis`; the first with the default `Tile{32, 32}` on `input0` and
`input1`, the second with `Tile{16, 32}`. Because the hash carries only `layout()` (`TILE` in both
cases), the keys are identical and the second call hits the first's entry. The cached matmul was
built with `Mt = ashape[-2]/32` and `in0_single_tile_size` for a 32x32 tile; the second call's
operands have twice as many tile rows and half the bytes per tile, so `cb_src0`'s page size and the
compute kernel's `in0_block_num_tiles` are both wrong. Simultaneously the cached all-gather half's
`intermediate_tensor_shard_num_pages` under-counts by 2x while `cb_inter`'s page size, taken from the
buffer, is correct — so the receiver walks half the shard. Symptom is wrong data with no cache miss
to hint at the cause.

**The internal inconsistency is itself a hazard,** independent of the cache. Anyone fixing the
all-gather half by making it tile-aware would produce an op that is correct end-to-end for non-32x32
tiles *and still* silently aliases in the cache, because `page_config` would still be missing from
the key. The two changes have to land together.

## Keys the custom hash adds beyond the default

`input0.padded_shape()` and `input1.padded_shape()` are not in the default key (the default hashes
`logical_shape` and derives padding). Adding them is what makes dropping `logical_shape` safe. There
is nothing else here that the default would not already cover — this hash is strictly a narrowing.

## Framework side effect of having a custom hash

Defining `compute_program_hash` opts the op out of attribute-level collision resolution:

```1012:1014:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
        if constexpr (requires { DeviceOperation::compute_program_hash(attrs, tensor_args); }) {
            return key;  // custom hash -> opt out beyond the op-identity prefix
        } else {
```

`ProgramCacheKey::canonical` degrades to the op type name, so a 64-bit collision between two distinct
configurations becomes a wrong hit rather than a rebuild. Inherent to every custom-hash op, but it
compounds the gaps found above.

## Summary

| Omitted vs. default | Used by program? | Patched on hit? | Verdict |
|---|---|---|---|
| `matmul_struct` (program config, compute kernel config, output dtype, output mem config, global CB) | Yes — compute-kernel compile args, CB formats/depths, core grid | No | **BUG** |
| `tensor_args.intermediate` (all properties; hash aliases `input1`) | Yes — receiver core range, CB size/page size, shard page counts | No | **BUG** |
| Intermediate globally-allocated CB address (`cb_inter`) | Yes — receiver read pointer | **No** | **BUG** |
| `sub_device_id` | Yes — worker core pool, kernel placement | No (core list is cached) | **BUG** |
| `semaphore` (`GlobalSemaphore` address) | Yes — 3 runtime-arg slots | Yes | VALID — patched |
| `devices` | No (dead branch; `cluster_axis` always set) | n/a | VALID — unused |
| `ring_index`, fabric neighbours, `num_targets_*` | Yes — compile args | n/a | VALID — invariant (keyed via the mesh coordinates the framework appends) |
| Fabric connection rt args (EDM addresses) | Yes | No | CAVEAT — relies on fixed fabric config |
| `input*.logical_shape` (padded used instead) | No | n/a | VALID — relaxation win |
| `input*.page_config` (`Tile`) | Yes — hardcoded 32x32 in the all-gather half, genuinely tile-derived in the matmul half | No | **BUG** |
| `input*.alignment` | Only via hashed derivatives | n/a | VALID — unused |
| storage kind | n/a | n/a | VALID — pinned by validation |
| Buffer addresses (in0/in1/aggregated/output) | Yes | Yes | VALID — patched |

**Program-cache bugs were found.** Five of them, and they are independent. The most serious is #1:
`matmul_struct` is a first-class public argument set — `program_config`, `compute_kernel_config`,
`dtype`, `mm_memory_config`, `global_cb` are all reachable from Python — and none of it is in the
cache key, while all of it is compiled into the cached program. #2 and #3 concern the intermediate
tensor, which is a caller-supplied tensor that the hash simply never reads (the `intermediate_*`
locals are a copy-paste of the `input1_*` block) and whose globally-allocated CB is never re-pointed.
#5 lets a sub-device change silently reuse kernels placed on the wrong cores. #11 is the tile
omission, which is doubled here: the all-gather half hardcodes 32x32 with no guard while the matmul
half derives its entire blocking from the real tile, so the missing `page_config` term breaks both
halves in opposite ways. The op appears to work today only because the Llama model calls it with a
frozen configuration.

**One warning to carry out of #11, because it is easy to get wrong.** The obvious repair for the tile
problem is to make the all-gather half tile-aware, matching what the matmul half already does. That
alone is not enough and is arguably worse than doing nothing: it produces an op that computes the
right answer for a non-32x32 tensor on a cold cache and *still* silently aliases onto a 32x32 entry
on a warm one, because `page_config` would still be missing from the key. A correct factory with an
incorrect key is harder to diagnose than an incorrect factory, since the code now reads as if it
handles arbitrary tiles. Either add the guard (which makes the omission correct by construction) or
change the factory and the hash in the same commit.

## Recommendations

1. Hash `args.matmul_struct`. It is fully reflectable — `hash_operation<...>(..., args.matmul_struct,
   ...)` is a one-line change. If the full struct is judged too coarse (e.g. `global_cb` identity
   causing spurious misses), hash at minimum `program_config`, `output_dtype`, `output_mem_config` and
   `compute_kernel_config`.
2. Fix the copy-paste in `compute_program_hash`: lines 106-109 should read from
   `tensor_args.intermediate`, not `input1`. This is almost certainly the original intent given the
   variable names.
3. Add `UpdateDynamicCircularBufferAddress(program, <cb_inter handle>, *intermediate_tensor.buffer())`
   to `override_runtime_arguments`, storing the `CBHandle` in `LlamaAllGatherMatmulAsyncSharedVariables`
   alongside the kernel handles.
4. Hash `args.sub_device_id`, or assert in `validate_on_program_cache_miss` that it equals
   `mesh_device->get_sub_device_ids().at(0)`.
5. Close the tile gap (#11). The cheapest correct fix is the standard guard in
   `validate_on_program_cache_miss`, mirroring
   `interleaved_to_sharded_op.cpp:95-97`, applied to `input0`, `input1` and `tensor_args.intermediate`
   — that converts the `page_config` omission to "VALID — pinned by validation" and makes the
   all-gather half's `TILE_HW` arithmetic correct by construction. If instead the intent is to support
   non-32x32 tiles, then three things must land in the same change: replace `TILE_HW` at
   `...program_factory.cpp:185,188` and the literal `32` at `...program_factory.cpp:114` with
   `tensor_spec().tile().get_tile_shape()`, and hash `tensor_spec().page_config()` in place of
   `layout()`. Making the factory tile-aware without the hash change leaves the aliasing bug intact.
   Note that hashing `page_config()` covers the tile *shape* only: `Tile::attribute_values()` exposes
   just `tile_shape`, `face_shape` and `num_faces` (`tt_metal/api/tt-metalium/tile.hpp:46-47`) and
   `Tile::operator==` compares only the first two (`tt_metal/impl/data_format/tile.cpp:122-124`), so
   `transpose_within_face` and `transpose_of_faces` remain invisible to both the hash and the
   canonical key. That is a framework-wide hole affecting every op, and only an explicit `TT_FATAL` on
   the two transpose accessors closes it.
6. Add validation pinning the remaining relaxation: `TT_FATAL` that `tensor_args.intermediate`'s spec
   matches the `intermediate_tensor_spec` that `compute_output_specs` computes
   (`device/llama_all_gather_matmul_async_device_operation.cpp:44-46`). This is a good alternative to
   recommendation 2 — it converts the omission to "pinned by validation" instead.
7. Delete the dead `else` branch at `...program_factory.cpp:78-85` (or `TT_FATAL` on
   `!cluster_axis.has_value()`), so the "`devices` is unused" verdict is enforced by the code rather
   than by an argument about the call graph.
8. Run this op under `-DTT_DESCRIPTOR_PATCHING_PARITY_CHECK`-style parity checking if/when the
   equivalent oracle is wired up for mesh-workload factories; findings #3 and the tail of #8 are
   exactly what such a check catches.
