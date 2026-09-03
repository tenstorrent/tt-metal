# Program Cache Audit — `experimental/ccl/llama_all_gather_matmul_async`

Audit of `ttnn::experimental::prim::LlamaAllGatherMatmulAsyncDeviceOperation::compute_program_hash`
against the framework default ("hash everything") key.

| | |
|---|---|
| Device operation | `ttnn::experimental::prim::LlamaAllGatherMatmulAsyncDeviceOperation` (`device/llama_all_gather_matmul_async_device_operation.hpp:22-33`) |
| Custom hash | `device/llama_all_gather_matmul_async_device_operation.cpp:119-184` |
| `operation_attributes_t` | `LlamaAllGatherMatmulAsyncParams` — `matmul_struct`, `devices`, `dim`, `num_links`, `ring_size`, `output_memory_config`, `topology`, `semaphore`, `sub_device_id`, `cluster_axis` |
| `tensor_args_t` | `LlamaAllGatherMatmulAsyncInputs` — `input0`, `input1`, `intermediate` |
| Program factories | `LlamaAllGatherMatmulAsyncProgramFactory` (single, mesh-workload style; internally also builds the fused matmul via `llama_1d_mm_fusion.cpp`) |
| `override_runtime_arguments` | **Yes** (`device/llama_all_gather_matmul_async_program_factory.cpp:496-550`) |
| `get_dynamic_runtime_args` | No |
| `validate_on_program_cache_miss` | Yes (`device/llama_all_gather_matmul_async_device_operation.cpp:15-61`) — `input0`, `num_links`, and since fab067a the layout and tile geometry of all three tensors |
| Cache-hit patch mechanism | **Op-owned re-derivation** (the factory's `override_runtime_arguments` runs on every hit) |

## Post-fix status — commit fab067a

**Verdict: CLEAR — with justified relaxation(s).** All five program-cache bugs recorded below are closed.
Every value the compiled program depends on is now either in the key, patched on every hit, or pinned by a
`TT_FATAL` that runs on the hit path. Two omissions remain and both are deliberate relaxations rather than
gaps: `logical_shape` (see #9 below) and the `GlobalSemaphore`'s `(cores, buffer_type)` identity (see #4).

What fab067a changed here:

- **Hashed the whole `MatmulParams`** — `args.matmul_struct`
  (`device/llama_all_gather_matmul_async_device_operation.cpp:159`), plus the global CB's `buffer_address()`
  and `config_address()` (`:165-168`), which reflection cannot reach because
  `GlobalCircularBuffer::attribute_values` is only `(sender_receiver_core_mapping, size, buffer_type)`
  (`tt_metal/api/tt-metalium/global_circular_buffer.hpp:77-82`).
- **Fixed the `intermediate_*` copy-paste** — those four locals now read `tensor_args.intermediate` instead
  of `input1` (`device/llama_all_gather_matmul_async_device_operation.cpp:139-143`).
- **Hashed `sub_device_id`** (`:158`), widened to `uint32_t` with an `0xFFFFFFFF` sentinel so a disengaged
  optional cannot collide with sub-device 0.
- **Hashed `tensor_spec().page_config()` for all three tensors** (`:173`, `:178`, `:183`).
- **Added a tile/layout guard** — `require_standard_tile` pins `Layout::TILE` and a 32x32 tile on `input0`,
  `input1` and the intermediate (`device/llama_all_gather_matmul_async_device_operation.cpp:43-60`). It lives
  in `validate_on_program_cache_miss`, and because this op declares no `validate_on_program_cache_hit` the
  framework substitutes that validator onto the hit path (`ttnn/api/ttnn/device_operation.hpp:265-268`), so
  the guard runs on both.
- **Re-pointed the globally-allocated intermediate CB on every hit** — handle stored in the shared variables
  and rewritten per coordinate; see the dedicated subsection under finding #3.

What remains open:

- Nothing in the cache key. The two surviving omissions are relaxations, not gaps.
- Two non-cache correctness defects, tracked in their own section at the end of this document: the
  all-gather half is still not tile-aware while the fused matmul half is, and `compute_output_specs`
  disagrees with `create_output_tensors` about the `.mm` spec.
- Two residual caveats that are **not** omissions relative to the default key, because the default hash
  would not see them either: a `SubDeviceId` names a core set that depends on which sub-device manager is
  loaded on the mesh device, and the fabric-connection runtime args stay frozen at first-miss values
  (finding #8).

**Metal 2.0 port readiness: clear.** The op's one substantive relaxation maps directly onto a supported
flag. `TensorSpecRelaxations::match_padded_shape_only`
(`tt_metal/api/tt-metalium/experimental/metal2_host_api/tensor_spec_relaxations.hpp:41`) is exactly this
op's "pin `padded_shape`, let `logical_shape` float": `pertinent_fields` maps it to
`PertinentFields{.padded_shape = true}`
(`tt_metal/impl/metal2_host_api/tensor_spec_relaxations.cpp:67-87`), and `hash_tensorspec_with_relaxation`
(`:116`) and `tensorspecs_match_with_relaxation` (`:161-201`) are driven from that same field set, so the
key and the accept/reject predicate cannot disagree — `ValidateTensorArgs` delegates to the predicate
(`tt_metal/impl/metal2_host_api/program_run_args.cpp:176-189`). Declaring `match_padded_shape_only` on
`input0`, `input1` and the intermediate reproduces the current key's behaviour exactly; declaring nothing
is also correct and merely costs recompiles for shapes that pad identically.

The CSV row for this op reads *explicit / SELECTIVE / no own hit validator / has `override_runtime_arguments` /
no `get_dynamic_runtime_args`*. All five columns are correct against the code: there is genuinely no
`validate_on_program_cache_hit` anywhere in this op directory. That absence is a feature rather than a
hole, because the dispatcher then substitutes the *miss* validator onto the hit path
(`ttnn/api/ttnn/device_operation.hpp:265-268`), so every `TT_FATAL` in
`device/llama_all_gather_matmul_async_device_operation.cpp:15-61` runs on a reuse.

Pre-fix, that validator was very thin — it never looked at `input1`, at `tensor_args.intermediate`, or at
any field of `matmul_struct`, so it pinned almost nothing that the hash dropped. Post-fix it additionally
pins `Layout::TILE` and a 32x32 tile on all three tensors (`:43-60`), which is what makes the surviving
`TILE_HW` and literal-`32` arithmetic in the all-gather half correct by construction.

## Cache-hit patch mechanism

This factory is a mesh-workload factory (`create_mesh_workload` returning `AdaptedCachedMeshWorkload`)
and exposes no `apply_descriptor`, so the framework takes the `override_runtime_arguments` branch of
the cache-hit dispatcher:

```282:288:ttnn/api/ttnn/device_operation.hpp
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

Pre-fix this op violated both. Post-fix it satisfies both: obligation 1 by hashing `matmul_struct`,
`sub_device_id`, `page_config` and the real intermediate tensor, and by pinning the tile geometry the
factory hardcodes; obligation 2 by adding the missing `UpdateDynamicCircularBufferAddress` for `cb_inter`.

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

```148:183:ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_all_gather_matmul_async_device_operation.cpp
    return tt::tt_metal::operation::hash_operation<LlamaAllGatherMatmulAsyncDeviceOperation>(
        args.dim,
        args.num_links,
        args.ring_size,
        args.output_memory_config,
        args.topology,
        args.cluster_axis,
        // sub_device_id selects the sender worker core pool and is forwarded to the matmul half, so it
        // is structural to the compiled program rather than per-call data. Widened past the uint8 range
        // so the disengaged optional cannot collide with a real sub-device id.
        args.sub_device_id.has_value() ? static_cast<uint32_t>(args.sub_device_id->get()) : 0xFFFFFFFFu,
        args.matmul_struct,
        // MatmulParams reaches global_cb through reflection, which covers the GCB's structure (core
        // mapping, size, buffer type) but not which allocation it is. The matmul's remote CB is created
        // against the GCB and bakes both addresses at build time, and UpdateDynamicCircularBufferAddress
        // refuses to re-point a GCB-backed CB, so two same-shaped GCBs at different allocations must not
        // share a program. Same reasoning as dram_prefetcher_validator.
        static_cast<uint64_t>(
            args.matmul_struct.global_cb.has_value() ? args.matmul_struct.global_cb->buffer_address() : 0),
        static_cast<uint64_t>(
            args.matmul_struct.global_cb.has_value() ? args.matmul_struct.global_cb->config_address() : 0),
        input0_shape,
        input0_memory_layout,
        input0_dtype,
        input0_memory_config,
        input0_page_config,
        input1_shape,
        input1_memory_layout,
        input1_dtype,
        input1_memory_config,
        input1_page_config,
        intermediate_shape,
        intermediate_memory_layout,
        intermediate_dtype,
        intermediate_memory_config,
        intermediate_page_config);
```

Pre-fix, the `intermediate_*` locals did **not** come from the intermediate tensor — they were read off
`input1`, so the last four hash terms were exact duplicates of the four preceding ones and
`tensor_args.intermediate` contributed nothing to the key. That is omission #2 below. fab067a corrected
them, and they now read the tensor their names claim:

```139:143:ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_all_gather_matmul_async_device_operation.cpp
    auto intermediate_shape = intermediate.padded_shape();
    auto intermediate_memory_layout = intermediate.layout();
    auto intermediate_dtype = intermediate.dtype();
    auto intermediate_memory_config = intermediate.memory_config();
    auto intermediate_page_config = intermediate.tensor_spec().page_config();
```

Effective hashed set:
`{dim, num_links, ring_size, output_memory_config, topology, cluster_axis, sub_device_id, matmul_struct
(all 14 fields), global_cb.buffer_address, global_cb.config_address}` plus, for each of `input0`, `input1`
and `intermediate`, `{padded_shape, layout, dtype, memory_config, page_config}` — plus mesh coordinates.
`layout` is now redundant alongside `page_config`, which determines it; harmless duplication.

## Omitted parameters

### 1. `operation_attributes.matmul_struct` (the entire `MatmulParams`)

**Verdict: RESOLVED by fab067a** (was BUG).

Closed by hashing `args.matmul_struct` wholesale
(`device/llama_all_gather_matmul_async_device_operation.cpp:159`). The struct is a plain aggregate with no
`attribute_names`, so reflection walks all 14 members — `program_config`, `compute_kernel_config`,
`output_dtype`, `output_mem_config`, `output_tile`, `untilize_out`, `transpose_a/b`,
`user_fused_activation` and the rest. The single thing reflection cannot see is *which*
`GlobalCircularBuffer` allocation `global_cb` names, because `GlobalCircularBuffer::attribute_values` is
only `(sender_receiver_core_mapping, size, buffer_type)`
(`tt_metal/api/tt-metalium/global_circular_buffer.hpp:77-82`); that is closed separately by hashing
`buffer_address()` and `config_address()` (`:165-168`).

Pre-fix: `MatmulParams` carries the matmul program config, the compute-kernel config, the output dtype, the
output memory config, the global CB and the sub-device id
(`ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation_types.hpp:15-30`). None of these
appeared in the hash. All of them are user-settable per call from Python — `program_config`,
`compute_kernel_config`, `dtype`, `global_cb`, `mm_memory_config` are all keyword arguments of the
binding (`llama_all_gather_matmul_async_nanobind.cpp:83-86`) and are packed into `MatmulParams`
verbatim (`device/llama_all_gather_matmul_async_device_operation.cpp:228-246`).

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

**Reproduction (pre-fix).** With a fixed `input_tensor0`, `input_tensor1`, `intermediate_tensor`, `dim`,
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

Note that `create_matmul_attributes` normalizes several `MatmulParams` fields to concrete values before
they are stored — `bcast_batch`, `output_dtype`, `compute_kernel_config` and `output_tile`
(`ttnn/cpp/ttnn/operations/matmul/device/matmul_device_operation.cpp:2871-2885`) — so the struct the hash
now walks is fully resolved rather than carrying `nullopt` placeholders. It does **not** normalize
`program_config`, which is passed through verbatim; this op's factory dereferences it unconditionally
(`device/llama_all_gather_matmul_async_program_factory.cpp:64`), so a caller-supplied config is mandatory
here and is keyed exactly as supplied.

### 2. `tensor_args.intermediate` (all six tensor properties)

**Verdict: RESOLVED by fab067a** (was BUG).

Closed by pointing the four `intermediate_*` locals at `tensor_args.intermediate` instead of `input1`
(`device/llama_all_gather_matmul_async_device_operation.cpp:139-142`), which puts the intermediate's
`padded_shape`, `layout`, `dtype` and `memory_config` into the key for the first time; a fifth term,
`page_config`, was added alongside them (`:143`). Belt and braces, `require_standard_tile` now also runs
against the intermediate tensor (`:60`), on both the miss and the hit path.

Pre-fix: the "intermediate" hash terms were aliases of `input1`, so the intermediate tensor's
shape, dtype, layout and memory config were all absent from the key, and `validate_on_program_cache_miss`
did not examine the intermediate tensor either, so nothing pinned it.

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

**Reproduction (pre-fix).** Call 1 with `intermediate_tensor` width-sharded over a 3x1 core grid; call 2 with a
byte-identical intermediate tensor width-sharded over a 6x1 grid (same total elements, half the shard
width). Nothing in the hash changes. Call 2 reuses call 1's program: the receiver kernel still exists
only on the 3 original cores, `cb_inter` is still sized for the wide shard, and the
`intermediate_tensor_shard_num_pages` runtime arg (`...program_factory.cpp:338`) still describes the
old shard. Half the gathered data is never multicast to the matmul cores, and the CB overruns the new
(narrower) shard on the cores that do run.

Callers today derive the intermediate tensor from `ag_memory_config`, which *is* hashed, so the two
move together in practice — that is the only reason this has not fired.

### 3. Intermediate tensor's globally-allocated CB address

**Verdict: RESOLVED by fab067a** (was BUG).

Closed on the patch side rather than the hash side — a buffer address must never be hashed, so the only
correct fix was to re-point the CB. See "The fix, in detail" below for the verification.

Pre-fix this was separate from #2, and true even when the intermediate tensor's *spec* was unchanged. `cb_inter` is
bound to the intermediate tensor's buffer via `set_globally_allocated_address`
(`...program_factory.cpp:218`), and the receiver kernel reads through it:

```77:81:ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/kernels/worker_receiver.cpp
    size_t l1_read_addr = cb_inter.get_read_ptr();
    const uint64_t multicast_addr_noc = get_noc_multicast_addr(bbox_start_x, bbox_start_y, bbox_end_x, bbox_end_y, 0);
    uint64_t aggregated_tensor_addr_this_core =
        (uint64_t)aggregated_tensor_addr + mm_core_offset * intermediate_tensor_shard_num_pages * tensor0_page_size;
    const uint64_t multicast_addr = multicast_addr_noc | aggregated_tensor_addr_this_core;
```

Pre-fix, `override_runtime_arguments` never called `UpdateDynamicCircularBufferAddress` for `cb_inter`;
it patched the intermediate address only into the *reader* and *writer* runtime args:

```517:535:ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_all_gather_matmul_async_program_factory.cpp
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

Note the asymmetry pre-fix: the matmul's in0/in1/out CBs *were* re-pointed (via
`override_agmm_fusion_program_parameters`, quoted in #1) and the sender-side intermediate addresses
*were* re-written, but the receiver's `cb_inter` was not. This was a straightforward omission.

**Reproduction (pre-fix).** Call 1 with intermediate tensor `I1`. Deallocate `I1`, allocate an unrelated L1
tensor, then allocate `I2` with the identical spec at a different L1 address, and issue call 2. The
hash is unchanged (addresses are never hashed, by design), the reader/writer write into `I2`, but
`cb_inter.get_read_ptr()` still resolves to `I1`'s old address, so the receiver multicasts whatever
now lives there.

**The fix, in detail.** Three things had to be true, and all three are:

*The handle is stored in the shared variables.* `CreateCircularBuffer` now returns into a named local
(`...program_factory.cpp:219`), a `CBHandle` field was added to the shared-variable struct, and the local
is written into it when the cached program is constructed (`...program_factory.cpp:492`):

```20:23:ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_all_gather_matmul_async_program_factory.hpp
    // The intermediate CB is globally allocated over the intermediate tensor's buffer, so its address
    // has to be re-pointed on every cache hit; keep the handle to do that.
    tt::tt_metal::CBHandle cb_inter{};
    ttnn::prim::matmul_mcast_1d_common_override_variables_t matmul_shared_variables;
```

The handle is per *program*, not per workload: `create_at` returns it inside
`cached_program.shared_variables`, and `create_mesh_workload` keys those by `MeshCoordinateRange`
(`...program_factory.cpp:43-47`), which is the same map the override indexes at `...program_factory.cpp:510`.
So each device's CB gets its own handle rather than one shared across the mesh.

*It is re-pointed on every hit, unconditionally.* The call sits inside the per-coordinate loop opened at
`...program_factory.cpp:509`, with no `if` around it:

```537:540:ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_all_gather_matmul_async_program_factory.cpp
        // The intermediate CB is globally allocated over the intermediate tensor's buffer. Patching the
        // reader/writer address args above does not move the CB itself, so without this a cached program
        // keeps pointing every intermediate access at the first call's allocation.
        UpdateDynamicCircularBufferAddress(program, shared_vars.cb_inter, *intermediate_tensor.buffer());
```

*It is re-pointed at the right buffer.* `intermediate_tensor` is bound to `tensor_args.intermediate`
(`...program_factory.cpp:503`) — the same tensor whose buffer `cb_inter_config` was given at build time via
`.set_globally_allocated_address(*intermediate_tensor.buffer())` (`...program_factory.cpp:218`). So the CB
follows the caller-supplied intermediate tensor's current allocation on every dispatch.

**Why this works here but could not have worked for the global CB.** `UpdateDynamicCircularBufferAddress`
accepts a CB that was globally allocated over an ordinary `Buffer`, but refuses one backed by a
`GlobalCircularBuffer`:

```1686:1691:tt_metal/impl/host_api/tt_metal.cpp
void UpdateDynamicCircularBufferAddress(Program& program, CBHandle cb_handle, const Buffer& buffer) {
    auto circular_buffer = program.impl().get_circular_buffer(cb_handle);
    TT_FATAL(!circular_buffer->is_global_circular_buffer(), "CircularBuffer must not be a GlobalCircularBuffer!");
    circular_buffer->config().set_globally_allocated_address(buffer);
    circular_buffer->assign_global_address();
}
```

`cb_inter` is the first kind, so patching is legal. The matmul's remote CB is the second kind: it is created
against `matmul_struct.global_cb` and bakes both the GCB's buffer and config addresses at build time, and
that `TT_FATAL` makes re-pointing it on a hit impossible. That is precisely why the global CB's
`buffer_address()` and `config_address()` had to go into the *key* instead
(`device/llama_all_gather_matmul_async_device_operation.cpp:165-168`) — for a CB that cannot be patched,
hashing the allocation identity is the only remaining option.

### 4. `operation_attributes.semaphore` (`GlobalSemaphore`)

**Verdict: VALID — patched.**

The semaphore's L1 address is a per-call allocation and correctly absent from the key. It appears in
three runtime-arg slots at build time — the receiver's arg 0
(`...program_factory.cpp:299`, `...program_factory.cpp:329`), the reader's arg 8
(`...program_factory.cpp:414`) and the writer's arg 1 (`...program_factory.cpp:434`) — and
`override_runtime_arguments` rewrites exactly those three slots on every hit
(lines 522, 526 and 533, quoted in #3). It is never used as a compile-time arg and never baked into a
CB. This is the textbook correct handling.

Worth being precise about what dropping the whole `semaphore` attribute actually costs, since the address
was never the issue: `GlobalSemaphore::attribute_values` is `(cores, buffer_type)` and excludes the address
(`tt_metal/api/tt-metalium/global_semaphore.hpp:73-74`), so those two fields are all the default key would
have carried. Neither is read by the factory. Omitting them is therefore a genuine relaxation with a real
(if modest) payoff: one cached program serves semaphores allocated over different core sets or buffer
types, which the default hash would rebuild for.

### 5. `operation_attributes.sub_device_id`

**Verdict: RESOLVED by fab067a** (was BUG).

Closed by adding it to the key
(`device/llama_all_gather_matmul_async_device_operation.cpp:158`). It is hashed as a `uint32_t` rather than
the underlying `uint8_t`, with `0xFFFFFFFF` standing in for the disengaged optional, so "no sub-device
specified" cannot collide with the real sub-device 0 that `value_or` then selects.

Pre-fix: `sub_device_id` selects the worker-core pool that the CCL sender cores are carved out of:

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

**Reproduction (pre-fix).** Call 1 with `subdevice_id=None` (falls back to sub-device 0, the full grid); call 2
with `subdevice_id=<a sub-device covering only the top half of the grid>`, everything else identical.
The hash is unchanged, so call 2 hits, and the CCL workers keep running on cores outside the
requested sub-device. On a mesh where the other half of the grid is concurrently owned by another
sub-device's program this is a hard correctness and dispatch-ordering violation, not just a
performance surprise.

This was the classic "structural, baked into the cached Program, not refreshed by
`override_runtime_arguments`" case called out for fabric/sub-device parameters.

One residual caveat survives the fix, and it is not an omission relative to the default key, because the
default hash would not have caught it either: a `SubDeviceId` is just an id, and the core set it names
depends on which sub-device manager is currently loaded on the mesh device. Two calls with the same id
across a manager reload share a key and mean different grids. Every sub-device-aware op in the tree has
this exposure.

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

```1012:1015:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
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

### 9. Tensor properties dropped from `input0`, `input1` and `intermediate`

**Verdict: mixed — one label per property below. All VALID; `page_config`, formerly the one **BUG** here,
is now hashed and adjudicated in #11.**

- **`logical_shape` replaced by `padded_shape`** — VALID — relaxation win, and the only substantive
  relaxation this op still takes. The factory works entirely in pages and shard shapes
  (`...program_factory.cpp:182-188`) and in `input1.padded_shape()[3]` (`:114`), never in logical
  elements; the fused matmul takes `ashape` from the aggregated tensor's shard shape and `bshape` from
  `b.padded_shape()` (`device/llama_1d_mm_fusion.cpp:747-752`); and the CCL slicer is padded-only
  (`ttnn/cpp/ttnn/operations/ccl/ccl_common.hpp:576-609`). Two calls whose logical shapes differ but pad
  to the same tiled shape legitimately share a program; the default hash would force a recompile. The
  per-call output `TensorSpec` is still recomputed by `compute_output_specs` on every invocation, so the
  returned tensor carries the right logical shape. Note the direction: `padded_shape` is a function of
  `logical_shape` plus the hashed alignment, but not the reverse, so keying padded is a strict widening of
  the equivalence class rather than a swap.
- **`page_config`** — no longer dropped. fab067a hashes `tensor_spec().page_config()` for all three
  tensors (`device/llama_all_gather_matmul_async_device_operation.cpp:173`, `:178`, `:183`). `layout()` is
  still hashed beside it and is now redundant, since `PageConfig` determines the layout. See #11.
- **`alignment`** — VALID — unused (low residual risk). It reaches the program only through the buffer page size
  and `padded_shape`, both of which are determined by the hashed `{memory_config, dtype, page_config,
  padded_shape}` for the layouts this op accepts (`device/llama_all_gather_matmul_async_device_operation.cpp:29-35`).
  Post-fix this is stronger than it was: with `Layout::TILE` and a 32x32 tile pinned by
  `require_standard_tile`, the default alignment is exactly the tile dims
  (`tt_metal/impl/tensor/spec/layout/page_config.cpp:43-57`), so there is nothing left for it to vary.
- **storage kind** — VALID — pinned by validation (the op declares no
  `validate_on_program_cache_hit`, so this `TT_FATAL` is substituted onto the hit path):

```21:23:ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_all_gather_matmul_async_device_operation.cpp
    TT_FATAL(input0.storage_type() == StorageType::DEVICE, "Operands to llama_all_gather_matmul need to be on device!");
    TT_FATAL(
        input0.buffer() != nullptr, "Operands to llama_all_gather_matmul need to be allocated in buffers on device!");
```

  This only covers `input0`; `input1` and `intermediate` are unchecked, but a host-storage tensor
  would fail at the `buffer()` dereference in `override_runtime_arguments`
  (`...program_factory.cpp:521`) rather than aliasing silently, so no cache aliasing is reachable.

### 10. Buffer addresses of `input0`, `input1`, `aggregated`, `output`

**Verdict: VALID — patched.** Addresses must not be hashed. `input0` and the intermediate go through
the reader/writer patch (quoted in #3), the aggregated tensor through receiver arg 3
(`...program_factory.cpp:534`), and the matmul in0/in1/out through
`override_agmm_fusion_program_parameters` (quoted in #1). The one gap in this family was `cb_inter`,
covered as its own finding in #3 and closed by fab067a; the family is now complete.

### 11. Tile geometry — the 32x32 assumption

**Verdict: RESOLVED by fab067a** (was BUG).

Closed by **both** available mechanisms, which is the right answer for this op because its two halves fail
in opposite ways:

- `tensor_spec().page_config()` is now hashed for `input0`, `input1` and the intermediate
  (`device/llama_all_gather_matmul_async_device_operation.cpp:173`, `:178`, `:183`). `PageConfig`'s
  reflected attribute is its inner variant
  (`tt_metal/api/tt-metalium/tensor/spec/layout/page_config.hpp:50-51`), which carries the `Tile`, so tile
  geometry is finally in the key. This is what the genuinely tile-aware fused matmul half needed.
- `require_standard_tile` pins `Layout::TILE` and a 32x32 tile on all three tensors
  (`device/llama_all_gather_matmul_async_device_operation.cpp:43-60`). The guard lives in
  `validate_on_program_cache_miss`, and since this op declares no `validate_on_program_cache_hit` the
  framework substitutes that validator onto the hit path (`ttnn/api/ttnn/device_operation.hpp:265-268`), so
  it fires on both. This is what the all-gather half needed: hashing alone would only have given a
  mis-compiled non-32x32 program its own cache entry, not made it correct.

The pairing matters. Hashing without the guard would have left the all-gather half computing wrong
programs; guarding without hashing would have left the matmul half aliasing across tiles the guard does not
reject. fab067a landed both in one change, which is exactly what the pre-fix warning at the end of this
finding asked for.

Pre-fix: the hash kept `layout()` but not `tensor_spec().page_config()`, so the `Tile` shape
was not in the key for any of `input0`, `input1` or the intermediate. This op is the *mixed* case: its
two halves use opposite idioms, and each half was independently broken by the omission.

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

**No guard anywhere — pre-fix.** `validate_on_program_cache_miss` checked page alignment, storage,
`num_links` and the memory layout, and never touched `tile()`. The matmul helper's only tile-related
assertions are divisibility checks against the tile shape it has already read
(`llama_1d_mm_fusion.cpp:795-814`), which every valid tile satisfies by construction. fab067a added the
missing geometry check as `require_standard_tile`
(`device/llama_all_gather_matmul_async_device_operation.cpp:43-60`).

So all three adjudication criteria held for the all-gather half (accepts `Layout::TILE`, bare
`TILE_HW`/literal-32 tile-count conversion, no tile-geometry guard), and the mirror-image case held
for the matmul half (provably varies with `Tile`, `Tile` not in the key). Both pointed at the same
missing hash term.

**Reproduction (pre-fix).** Two calls with identical padded shapes, dtypes, memory configs, `num_links`,
`ring_size`, `topology` and `cluster_axis`; the first with the default `Tile{32, 32}` on `input0` and
`input1`, the second with `Tile{16, 32}`. Because the hash carries only `layout()` (`TILE` in both
cases), the keys are identical and the second call hits the first's entry. The cached matmul was
built with `Mt = ashape[-2]/32` and `in0_single_tile_size` for a 32x32 tile; the second call's
operands have twice as many tile rows and half the bytes per tile, so `cb_src0`'s page size and the
compute kernel's `in0_block_num_tiles` are both wrong. Simultaneously the cached all-gather half's
`intermediate_tensor_shard_num_pages` under-counts by 2x while `cb_inter`'s page size, taken from the
buffer, is correct — so the receiver walks half the shard. Symptom is wrong data with no cache miss
to hint at the cause.

**The internal inconsistency is itself a hazard,** independent of the cache — and it outlived the cache
fix. fab067a chose the guard, so the all-gather half's `TILE_HW` and literal-`32` arithmetic is still
there, now correct by construction rather than by accident. Anyone who later makes that half tile-aware
must delete the guard and keep `page_config` hashed in the same change; deleting the guard alone would
re-open this finding, and making the half tile-aware while leaving the guard in place would be dead code.
The surviving inconsistency is tracked as a non-cache defect at the end of this document.

## Keys the custom hash adds beyond the default

Two things.

`input0.padded_shape()`, `input1.padded_shape()` and `intermediate.padded_shape()` are not in the default
key (the default hashes `logical_shape` and derives padding). Adding them is what makes dropping
`logical_shape` safe.

Since fab067a, the global CB's `buffer_address()` and `config_address()` are also beyond the default:
`GlobalCircularBuffer::attribute_values` is `(sender_receiver_core_mapping, size, buffer_type)`
(`tt_metal/api/tt-metalium/global_circular_buffer.hpp:77-82`), so reflection keys the GCB's *structure*
but not which allocation it is — and a GCB-backed CB cannot be re-pointed on a hit
(`tt_metal/impl/host_api/tt_metal.cpp:1688`). Everything else this hash carries the default would already
cover; apart from those two additions it remains a narrowing.

## Framework side effect of having a custom hash

Defining `compute_program_hash` opts the op out of attribute-level collision resolution:

```1035:1037:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
        if constexpr (requires { DeviceOperation::compute_program_hash(attrs, tensor_args); }) {
            return key;  // custom hash -> opt out beyond the op-identity prefix
        } else {
```

`ProgramCacheKey::canonical` degrades to the op type name, so a 64-bit collision between two distinct
configurations becomes a wrong hit rather than a rebuild. Inherent to every custom-hash op, and unchanged
by fab067a — though it is much less consequential now that the key is thirty terms wide instead of
fourteen, since the gaps it used to compound are closed.

## Summary

| Omitted vs. default | Used by program? | Patched on hit? | Verdict |
|---|---|---|---|
| `matmul_struct` (program config, compute kernel config, output dtype, output mem config, global CB) | Yes — compute-kernel compile args, CB formats/depths, core grid | No | **RESOLVED by fab067a** — now hashed whole, plus the GCB's two addresses |
| `tensor_args.intermediate` (all properties; hash aliased `input1`) | Yes — receiver core range, CB size/page size, shard page counts | No | **RESOLVED by fab067a** — locals re-pointed at `tensor_args.intermediate` |
| Intermediate globally-allocated CB address (`cb_inter`) | Yes — receiver read pointer | **Yes** | **RESOLVED by fab067a** — handle kept in shared vars, re-pointed every hit |
| `sub_device_id` | Yes — worker core pool, kernel placement | No (core list is cached) | **RESOLVED by fab067a** — hashed with a sentinel for the disengaged optional |
| `semaphore` (`GlobalSemaphore`; the default keys only `cores`/`buffer_type`, never the address) | Address: yes — 3 runtime-arg slots. `cores`/`buffer_type`: no | Yes | VALID — patched; dropping the two structural fields is a deliberate relaxation |
| `devices` | No (dead branch; `cluster_axis` always set) | n/a | VALID — unused |
| `ring_index`, fabric neighbours, `num_targets_*` | Yes — compile args | n/a | VALID — invariant (keyed via the mesh coordinates the framework appends) |
| Fabric connection rt args (EDM addresses) | Yes | No | CAVEAT — relies on fixed fabric config |
| `input*.logical_shape` (padded used instead) | No | n/a | VALID — relaxation win (the op's one substantive relaxation) |
| `input*.page_config` (`Tile`) | Yes — hardcoded 32x32 in the all-gather half, genuinely tile-derived in the matmul half | No | **RESOLVED by fab067a** — hashed *and* pinned to 32x32 by `require_standard_tile` |
| `input*.alignment` | Only via hashed derivatives | n/a | VALID — unused |
| storage kind | n/a | n/a | VALID — pinned by validation |
| Buffer addresses (in0/in1/aggregated/output/`cb_inter`) | Yes | Yes | VALID — patched |

**Zero program-cache bugs remain.** All five recorded above are closed by fab067a, and they were closed at
the right layer in each case: #1, #2, #5 and the hash half of #11 by widening the key; #3 by adding the
missing `UpdateDynamicCircularBufferAddress`, since a buffer address must never be hashed; and the factory
half of #11 by a `TT_FATAL` rather than a hash term, because hashing alone would only have given a
mis-compiled non-32x32 program its own cache entry. What is left is two relaxations — `logical_shape` and
the `GlobalSemaphore`'s `(cores, buffer_type)` — both of which widen the equivalence class over values the
program is genuinely invariant to.

**One warning to carry out of #11, because it is still easy to get wrong.** fab067a took the guard route,
so the all-gather half's `TILE_HW` and literal-`32` arithmetic survives. The tempting follow-up is to make
that half tile-aware, matching what the matmul half already does. On its own that is not an improvement: it
would produce an op that computes the right answer for a non-32x32 tensor on a cold cache only if the guard
is also removed — and removing the guard without keeping `page_config` in the key re-opens the aliasing. A
correct factory with an incorrect key is harder to diagnose than an incorrect factory, since the code then
reads as if it handles arbitrary tiles. Change the factory, drop the guard, and keep the hash term, all in
the same commit — or leave it alone.

## Recommendations

Recommendations 1 through 5 are **done**, implemented by fab067a; they are kept here with their outcome so
the reasoning behind each change stays on record. 6 through 8 remain open.

1. **Done.** Hash `args.matmul_struct`. It is fully reflectable — `hash_operation<...>(..., args.matmul_struct,
   ...)` is a one-line change. If the full struct is judged too coarse (e.g. `global_cb` identity
   causing spurious misses), hash at minimum `program_config`, `output_dtype`, `output_mem_config` and
   `compute_kernel_config`. *Landed as the full struct (`device_operation.cpp:159`), plus the GCB's
   `buffer_address()`/`config_address()` (`:165-168`) that reflection cannot reach.*
2. **Done.** Fix the copy-paste in `compute_program_hash`: the `intermediate_*` locals should read from
   `tensor_args.intermediate`, not `input1`. This is almost certainly the original intent given the
   variable names. *Landed at `device_operation.cpp:139-143`, with `page_config` added alongside.*
3. **Done.** Add `UpdateDynamicCircularBufferAddress(program, <cb_inter handle>, *intermediate_tensor.buffer())`
   to `override_runtime_arguments`, storing the `CBHandle` in `LlamaAllGatherMatmulAsyncSharedVariables`
   alongside the kernel handles. *Landed exactly as described — `program_factory.hpp:22`,
   `program_factory.cpp:219`, `:492`, `:537-540`.*
4. **Done.** Hash `args.sub_device_id`, or assert in `validate_on_program_cache_miss` that it equals
   `mesh_device->get_sub_device_ids().at(0)`. *Landed as the hash term (`device_operation.cpp:158`),
   widened to `uint32_t` with an `0xFFFFFFFF` sentinel so the disengaged optional cannot collide with
   sub-device 0.*
5. **Done.** Close the tile gap (#11). The cheapest correct fix is the standard guard in
   `validate_on_program_cache_miss`, mirroring
   `interleaved_to_sharded_op.cpp:95-97`, applied to `input0`, `input1` and `tensor_args.intermediate`
   — that converts the `page_config` omission to "VALID — pinned by validation" and makes the
   all-gather half's `TILE_HW` arithmetic correct by construction. If instead the intent is to support
   non-32x32 tiles, then three things must land in the same change: replace `TILE_HW` at
   `...program_factory.cpp:185,188` and the literal `32` at `...program_factory.cpp:114` with
   `tensor_spec().tile().get_tile_shape()`, and hash `tensor_spec().page_config()` in place of
   `layout()`. Making the factory tile-aware without the hash change leaves the aliasing bug intact.
   *Landed as both: the guard (`device_operation.cpp:43-60`, which also pins `Layout::TILE`) and the
   `page_config` hash terms (`:173`, `:178`, `:183`). `layout()` was kept beside `page_config` rather than
   replaced, which is redundant but harmless.*
   Note that hashing `page_config()` covers the tile *shape* only: `Tile::attribute_values()` exposes
   just `tile_shape`, `face_shape` and `num_faces` (`tt_metal/api/tt-metalium/tile.hpp:46-47`) and
   `Tile::operator==` compares only the first two (`tt_metal/impl/data_format/tile.cpp:122-124`), so
   `transpose_within_face` and `transpose_of_faces` remain invisible to both the hash and the
   canonical key. That is a framework-wide hole affecting every op, and only an explicit `TT_FATAL` on
   the two transpose accessors closes it. Here the guard makes it moot for the three input tensors;
   `MatmulParams::output_tile` is still exposed to it in principle.
6. **Open.** Add validation pinning the remaining relaxation: `TT_FATAL` that `tensor_args.intermediate`'s spec
   matches the `intermediate_tensor_spec` that `compute_output_specs` computes
   (`device/llama_all_gather_matmul_async_device_operation.cpp:72-74`). This was offered as an alternative to
   recommendation 2; recommendation 2 was taken instead, so the intermediate is keyed rather than pinned.
   The check is still worth having as a diagnostic, but it is no longer load-bearing for the cache.
7. **Open.** Delete the dead `else` branch at `...program_factory.cpp:78-85` (or `TT_FATAL` on
   `!cluster_axis.has_value()`), so the "`devices` is unused" verdict is enforced by the code rather
   than by an argument about the call graph.
8. **Open.** Run this op under `-DTT_DESCRIPTOR_PATCHING_PARITY_CHECK`-style parity checking if/when the
   equivalent oracle is wired up for mesh-workload factories; finding #3 and the tail of #8 are
   exactly what such a check catches — #3 in particular went unnoticed until a manual read.

## Non-cache correctness defects

These were found while auditing the cache key but are **not** program-cache bugs and must not be counted as
such. Neither affects the key's soundness; both are factory-level issues that survive fab067a.

### N1. The all-gather half is not tile-aware while the fused matmul half is

`...program_factory.cpp:114` computes `weight_tensor_width = input1.padded_shape()[3] / 32` from a bare
literal, and `:185` / `:188` convert shard shapes to page counts by dividing by `TILE_HW`. One line later,
`:189` reads `intermediate_tensor.buffer()->page_size()`, which *is* tile-aware — so the two sides of the
same address arithmetic disagree about what a tile is. Meanwhile the fused matmul half reads the real tile
off both operands and derives its entire work split from it
(`device/llama_1d_mm_fusion.cpp:753-758`, `:824-827`).

`require_standard_tile` (`device/llama_all_gather_matmul_async_device_operation.cpp:43-60`) makes this
latent rather than live: a non-32x32 tile is now rejected on both the miss and the hit path, so the
hardcoded arithmetic is correct for every input the op accepts. The defect is that the op *reads* as though
it supports arbitrary tiles on one side and 32x32 on the other. See the warning under the Summary before
changing it.

### N2. `compute_output_specs` and `create_output_tensors` disagree about the `.mm` spec

`compute_output_specs` derives the matmul output spec from `{input0, input1}`:

```96:97:ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_all_gather_matmul_async_device_operation.cpp
    tt::tt_metal::TensorSpec matmul_output_specs =
        ttnn::prim::MatmulDeviceOperation::compute_output_specs(args.matmul_struct, {{input0, input1}, {}})[0];
```

but the tensor actually allocated is built from `{aggregated_tensor, input1}`:

```112:114:ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_all_gather_matmul_async_device_operation.cpp
    // Matmul output tensor
    ttnn::Tensor matmul_output_tensor = ttnn::prim::MatmulDeviceOperation::create_output_tensors(
        args.matmul_struct, {{aggregated_tensor, input1}, {}})[0];
```

`aggregated_tensor`'s shape is `input0.padded_shape()` with the last dim scaled by `ring_size` and then by
60 (`:69-78`), so the two calls see different M and K. The advertised spec is therefore not the spec of the
returned tensor. This is harmless for the cache — `create_output_tensors` is what the dispatcher allocates
from, and everything it depends on is hashed — but any caller or framework path that trusts
`compute_output_specs().mm` is being told the wrong thing.
