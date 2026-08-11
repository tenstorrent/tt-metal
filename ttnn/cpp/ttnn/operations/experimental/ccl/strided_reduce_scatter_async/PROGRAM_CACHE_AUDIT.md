# Program Cache Audit — `experimental/ccl/strided_reduce_scatter_async`

Audit of `StridedReduceScatterAsyncDeviceOperation::compute_program_hash` against the framework
default ("hash everything") key.

| | |
|---|---|
| Device operation | `ttnn::operations::experimental::ccl::strided_reduce_scatter_async::detail::StridedReduceScatterAsyncDeviceOperation` (`device/strided_reduce_scatter_async_op_device_operation.hpp`) |
| Custom hash | `device/strided_reduce_scatter_async_op_device_operation.cpp:159-191` |
| `operation_attributes_t` | `dim`, `num_links`, `ring_size`, `output_mem_config`, `optional_intermediate_mem_config`, `topology`, `semaphore` (vector), `barrier_semaphore`, `using_persistent_buffers`, `sub_device_id`, `cluster_axis`, `num_workers_per_link`, `num_buffers_per_channel`, `mm_cores_y`, `mm_block_ht`, `mm_block_wt`, `mm_N_full_block_wt`, `chunk_width_in_mm_blocks` (`device/strided_reduce_scatter_async_op_device_operation_types.hpp:34-78`) |
| `tensor_args_t` | `input_tensor`, `optional_intermediate_tensor`, `optional_output_tensor` |
| Program factories | `RingStridedReduceScatterMeshWorkloadFactory` (single; `select_program_factory` rejects any topology but Ring) |
| `override_runtime_arguments` | **Yes** (`device/strided_reduce_scatter_async_program.cpp:888-919`, delegating to `:753-806`) |
| `get_dynamic_runtime_args` | No |
| `validate_on_program_cache_hit` | Present and non-empty (storage and buffer liveness) |
| Cache-hit patch mechanism | **Op-owned re-derivation** (the factory's `override_runtime_arguments` runs on every hit) |

The CSV row (*explicit / SELECTIVE / has own hit validator / has `override_runtime_arguments` / no
`get_dynamic_runtime_args`*) matches the code on all five columns. Two refinements worth recording:
the hit validator is genuinely non-empty here (unlike the sibling `llama_reduce_scatter_*` ops) but
only checks storage and buffer liveness on the *input*, which by rule 4a-ii makes it a hazard rather
than a safeguard — it replaces the miss validator on hits, and one of this op's two cache bugs (#3)
is a direct consequence; and "SELECTIVE" understates the input-tensor handling — this hash keeps
*more* of the input tensor than the framework default does (see below).

This is by a wide margin the most carefully constructed of the four CCL hashes in this audit.

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

Nothing is inferred by the framework — no `resolve_bindings`, no automatic CB patching. The op's own
override rewrites the three buffer addresses and all four semaphore addresses on every worker core:

```783:802:ttnn/cpp/ttnn/operations/experimental/ccl/strided_reduce_scatter_async/device/strided_reduce_scatter_async_program.cpp
                // sender reader
                auto& worker_reader_sender_runtime_args = reader_runtime_args[core.x][core.y];
                worker_reader_sender_runtime_args[0] = input.buffer()->address();
                worker_reader_sender_runtime_args[1] = intermed.buffer()->address();
                worker_reader_sender_runtime_args[2] = semaphore.at(dir).address();
                if (reader_addcmul_rt_arg_offset > 0 && addcmul_a.has_value() && addcmul_b.has_value()) {
                    worker_reader_sender_runtime_args[reader_addcmul_rt_arg_offset] = addcmul_a->buffer()->address();
                    worker_reader_sender_runtime_args[reader_addcmul_rt_arg_offset + 1] =
                        addcmul_b->buffer()->address();
                }
                // sender writer
                auto& worker_writer_sender_runtime_args = writer_runtime_args[core.x][core.y];
                worker_writer_sender_runtime_args[0] = intermed.buffer()->address();
                worker_writer_sender_runtime_args[1] = output.buffer()->address();
                worker_writer_sender_runtime_args[4] = semaphore.at(dir).address();
                worker_writer_sender_runtime_args[5] = semaphore.at(num_directions_per_link).address();

                if (barrier_semaphore.has_value()) {
                    worker_writer_sender_runtime_args[7] = barrier_semaphore.value().address();
                }
```

The slot indices match the build-time layout exactly (reader args 0-2 at
`device/strided_reduce_scatter_async_program.cpp:659-666`, writer args 0, 1, 4, 5 and 7 at
`device/strided_reduce_scatter_async_program.cpp:693-707`). No CB in this program is
globally-allocated, so there is no CB address to patch.

The obligation on the hash is therefore the structural one: every compile-time arg, every kernel
`define`, every CB size, and every core range must be a pure function of the hashed set plus the mesh
coordinates the framework appends. The compile-time surface is large — 24 reader args
(`device/strided_reduce_scatter_async_program.cpp:468-493`), 23 writer args plus fabric-mux and
unicast/mcast args (`device/strided_reduce_scatter_async_program.cpp:526-566`), 16 compute args
(`device/strided_reduce_scatter_async_program.cpp:591-608`), plus `TensorAccessorArgs` /
sharding args appended per tensor.

## Baseline: what the default hash would cover

`operation_attributes_t` is an aggregate with no `to_hash` and no `attribute_names`, so the default
would reflect over its public members. (Its `attributes()` method at
`device/strided_reduce_scatter_async_op_device_operation_types.hpp:55-77` is a printing hook, not a
hashing hook — `ttsl::hash::hash_object` dispatches on `to_hash()`, then
`attribute_names`/`attribute_values()`, then containers, then `reflect::for_each`;
`tt_stl/tt_stl/reflection.hpp:1314-1334` and `tt_stl/tt_stl/reflection.hpp:1418-1424`.)

| Source | Fields |
|---|---|
| `operation_attributes` | all 18 fields, including the `semaphore` vector, `barrier_semaphore` and `sub_device_id` in full |
| `input_tensor` | storage kind, `logical_shape`, `dtype`, `page_config`, `memory_config`, `alignment` |
| `optional_intermediate_tensor` | the same six, plus the engaged bit |
| `optional_output_tensor` | the same six, plus the engaged bit |
| appended by framework | the mesh coordinates of the tensors |

## What the custom hash covers

```163:190:ttnn/cpp/ttnn/operations/experimental/ccl/strided_reduce_scatter_async/device/strided_reduce_scatter_async_op_device_operation.cpp
    return ttsl::hash::hash_objects(
        operation_attributes.dim,
        operation_attributes.num_links,
        operation_attributes.ring_size,
        operation_attributes.output_mem_config,
        operation_attributes.optional_intermediate_mem_config,
        operation_attributes.topology,
        operation_attributes.barrier_semaphore.has_value(),
        operation_attributes.using_persistent_buffers,
        operation_attributes.sub_device_id.has_value(),
        operation_attributes.sub_device_id.has_value()
            ? input_tensor.device()->worker_cores(
                  tt::tt_metal::HalProgrammableCoreType::TENSIX, operation_attributes.sub_device_id.value())
            : CoreRangeSet(CoreRange({0, 0}, {0, 0})),
        operation_attributes.cluster_axis,
        operation_attributes.num_workers_per_link,
        operation_attributes.num_buffers_per_channel,
        operation_attributes.mm_cores_y,
        operation_attributes.mm_block_ht,
        operation_attributes.mm_block_wt,
        operation_attributes.mm_N_full_block_wt,
        operation_attributes.chunk_width_in_mm_blocks,
        input_tensor.logical_shape(),
        input_tensor.padded_shape(),
        input_tensor.tensor_spec().page_config(),
        input_tensor.dtype(),
        input_tensor.layout(),
        input_tensor.memory_config());
```

Every scalar attribute that reaches a compile-time arg is present: `ring_size`, `num_links`,
`num_workers_per_link`, `num_buffers_per_channel`, `mm_cores_y`, `mm_block_ht`, `mm_block_wt`,
`mm_N_full_block_wt`, `chunk_width_in_mm_blocks`. So the CCL-specific parameters the audit brief
flags as commonly-dropped — ring size, topology, `cluster_axis`, worker and chunk counts — are all
keyed here.

Two structural notes on the call itself, both worth recording:

- **`dim` is the seed, not a hashed term.** `ttsl::hash::hash_objects(hash_t seed, const Types&...
  args)` (`tt_stl/tt_stl/reflection.hpp:1452-1454`) takes its first parameter as the running seed. So
  `dim` is folded in as the initial value rather than mixed through `hash_object`. It still
  contributes and still distinguishes different `dim` values, so this is a style wart rather than a
  correctness problem — and `validate_on_program_cache_miss` pins `dim == 3` anyway
  (`device/strided_reduce_scatter_async_op_device_operation.cpp:62-65`).
- **`type_hash<Op>` is absent.** Unlike the other three ops this one calls `hash_objects` directly
  rather than `operation::hash_operation<Op>`, so the op's type identity is not in the 64-bit value.
  That is safe because the cache key is a pair and `canonical` always carries the op-identity prefix:

```113:119:tt_metal/api/tt-metalium/program_cache.hpp
struct ProgramCacheKey {
    uint64_t hash = 0;
    std::string canonical;

    bool operator==(const ProgramCacheKey& other) const {
        return hash == other.hash && canonical == other.canonical;
    }
```

  and `compute_mesh_workload_canonical_key` returns `std::string key{op_type_name}` for custom-hash
  ops (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:1011-1013`). So two different ops cannot alias
  even on a 64-bit collision. Still worth switching to `hash_operation<Op>` for consistency.

## Omitted parameters

### 1. `operation_attributes.semaphore` (the `GlobalSemaphore` vector) and `barrier_semaphore` address

**Verdict: VALID — patched.**

Four distinct semaphore addresses reach the program, all as runtime args:

```659:666:ttnn/cpp/ttnn/operations/experimental/ccl/strided_reduce_scatter_async/device/strided_reduce_scatter_async_program.cpp
                std::vector<uint32_t> reader_rt_args = {
                    input_tensor.buffer()->address(),         // input_tensor_address
                    intermediate_tensor.buffer()->address(),  // intermediate_tensor_address
                    semaphore.at(dir).address(),              // out_ready_semaphore
                    dir,                                      // direction
                    worker_id,                                // worker_id
                    num_workers,                              // num_workers
                };
```

```693:707:ttnn/cpp/ttnn/operations/experimental/ccl/strided_reduce_scatter_async/device/strided_reduce_scatter_async_program.cpp
                std::vector<uint32_t> writer_rt_args = {
                    intermediate_tensor.buffer()->address(),                     // intermediate_tensor_address
                    output_tensor.buffer()->address(),                           // output_tensor_address
                    virtual_core.x,                                              // out_ready_sem_noc0_x
                    virtual_core.y,                                              // out_ready_sem_noc0_y
                    semaphore.at(dir).address(),                                 // out_ready_fwd_semaphore
                    semaphore.at(num_directions_per_link).address(),             // batch_ready_semaphore
                    barrier_semaphore.has_value() && !using_persistent_buffers,  // use_barrier_sem
                    barrier_semaphore.has_value()                                // barrier_sem
                        ? barrier_semaphore.value().address()
                        : 0,
                    dir,          // direction
                    worker_id,    // worker_id
                    num_workers,  // num_workers
                };
```

All four — `semaphore.at(dir)` in both kernels, the batch-ready semaphore
`semaphore.at(num_directions_per_link)`, and `barrier_semaphore` — are rewritten by the override
(lines 787, 797, 798, 801, quoted above). None is a compile-time arg. Correctly omitted, completely
patched.

The one *derived* value that is not patched is writer arg 6,
`barrier_semaphore.has_value() && !using_persistent_buffers` — a raw `uint32_t` boolean frozen at the
first miss. Both of its inputs are explicitly hashed
(`...device_operation.cpp:170-171`), so it cannot go stale. This is exactly the discipline the
brief asks for: the *address* is patched and unhashed, the *presence bit* that changes program
behaviour is hashed and unpatched. The vector's length is additionally pinned by validation:

```77:83:ttnn/cpp/ttnn/operations/experimental/ccl/strided_reduce_scatter_async/device/strided_reduce_scatter_async_op_device_operation.cpp
    // Validate semaphore count
    constexpr auto num_expected_semaphores = 3;
    TT_FATAL(
        operation_attributes.semaphore.size() == num_expected_semaphores,
        "Expected {} semaphores but got {}",
        num_expected_semaphores,
        operation_attributes.semaphore.size());
```

### 2. `operation_attributes.sub_device_id` (the id itself)

**Verdict: VALID — relaxation win.**

The id is not hashed; what is hashed is the *resolved core set* it denotes
(`...device_operation.cpp:172-176`). That is strictly better than hashing the id, because
`sub_device_id` reaches the program only through the cores it selects:

```252:253:ttnn/cpp/ttnn/operations/experimental/ccl/strided_reduce_scatter_async/device/strided_reduce_scatter_async_program.cpp
    const auto [all_core_range, all_cores] =
        choose_worker_cores(num_links, num_cores_per_link, mesh_device, sub_device_id, core_grid_offset);
```

and through the default worker count:

```229:238:ttnn/cpp/ttnn/operations/experimental/ccl/strided_reduce_scatter_async/device/strided_reduce_scatter_async_program.cpp
    uint32_t num_workers_per_direction =
        num_workers_per_direction_opt.value_or(ttnn::experimental::ccl::reduce_scatter_default_workers(
            *mesh_device,
            sub_device_id,
            topology,
            input_data_size_bytes,
            num_links,
            ring_size,
            num_directions_per_link,
            num_mux_cores_per_direction_per_link));
```

Two different `SubDeviceId`s that resolve to the same Tensix core set produce the same program and
correctly share one cache entry — the default hash would force a needless recompile after any
sub-device reconfiguration that preserved the core set. This is the pattern the other three ops in
this audit should copy; all three of them omit `sub_device_id` outright and are buggy for it.

The one thing to note is that the `else` arm of the ternary — the `CoreRangeSet(CoreRange({0, 0}, {0,
0}))` placeholder — is dead through the public path, because the launcher resolves the optional
before constructing the attributes:

```223:223:ttnn/cpp/ttnn/operations/experimental/ccl/strided_reduce_scatter_async/device/strided_reduce_scatter_async_op_device_operation.cpp
    const auto resolved_sub_device_id = sub_device_id.value_or(input_tensor.device()->get_sub_device_ids().at(0));
```

so `sub_device_id.has_value()` is always true here. Harmless, and the placeholder would still be
sound (the nullopt case resolves to sub-device 0 downstream, a device-fixed value).

### 3. `tensor_args.optional_output_tensor`

**Verdict: BUG.**

The output tensor's sharding is genuinely structural: it selects a kernel define and a block of
compile-time accessor args.

```415:417:ttnn/cpp/ttnn/operations/experimental/ccl/strided_reduce_scatter_async/device/strided_reduce_scatter_async_program.cpp
    if (output_is_sharded) {
        writer_compute_defines["OUTPUT_IS_SHARDED"] = "1";
    }
```

```573:577:ttnn/cpp/ttnn/operations/experimental/ccl/strided_reduce_scatter_async/device/strided_reduce_scatter_async_program.cpp
    if (output_is_sharded) {
        shard_builder::extend_sharding_compile_time_args(output_tensor, sender_writer_compile_args);
    } else {
        tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(sender_writer_compile_args);
    }
```

On a **miss**, every property of the caller-supplied tensor that feeds those is pinned to a hashed
value. `validate_on_program_cache_miss` forwards `operation_attributes.output_mem_config` and
`tensor_args.optional_output_tensor` into the shared CCL validator:

```35:42:ttnn/cpp/ttnn/operations/experimental/ccl/strided_reduce_scatter_async/device/strided_reduce_scatter_async_op_device_operation.cpp
    ttnn::experimental::ccl::reduce_scatter_common_validates(
        input_tensor,
        operation_attributes.topology,
        operation_attributes.dim,
        operation_attributes.num_links,
        operation_attributes.ring_size,
        operation_attributes.output_mem_config,
        tensor_args.optional_output_tensor);
```

and that function pins the passed buffer hard:

```64:84:ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_common/reduce_scatter_validate_utils.cpp
    if (optional_output_tensor.has_value()) {
        const auto& output_tensor = optional_output_tensor.value();

        TT_FATAL(
            output_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED ||
                output_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
                output_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
                output_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED ||
                output_tensor.memory_config().memory_layout() == TensorMemoryLayout::ND_SHARDED,
            "Unsupported output tensor memory layout");

        TT_FATAL(output_tensor.storage_type() == StorageType::DEVICE, "Output tensor must be on device");
        TT_FATAL(
            output_tensor.layout() == input_tensor.layout(), "Output tensor layout must match input tensor layout");
        TT_FATAL(output_tensor.dtype() == input_tensor.dtype(), "Output tensor dtype must match input tensor dtype");
        TT_FATAL(
            output_tensor.tensor_spec().page_config() == input_tensor.tensor_spec().page_config(),
            "Output tensor page config must match input tensor page config");
        TT_FATAL(
            output_tensor.memory_config() == memory_config,
            "Output tensor memory config must match provided memory_config");
```

with the per-dimension shape checks against `input_shape` and `ring_size` following at
`.../reduce_scatter_validate_utils.cpp:86-100`. `memory_config` here is
`operation_attributes.output_mem_config`
(`device/strided_reduce_scatter_async_op_device_operation.cpp:41`), which is hashed; the input's
layout, dtype, page config and padded shape are all hashed; `ring_size` is hashed. When the caller
does *not* supply the tensor, it is built from `compute_output_specs`
(`device/strided_reduce_scatter_async_op_device_operation.cpp:126-140`) out of the same hashed
values. So on a miss the output tensor's spec is a function of the key.

**None of that runs on a hit.** This op defines `validate_on_program_cache_hit` (quoted at the end of
#4), so under rule 4a-ii the hit validator *replaces* the miss validator rather than supplementing
it. What survives on the hit path is two checks on the *input* tensor — storage kind and buffer
non-null — and nothing at all about the output tensor.

**The 4a-iii reachability filter does not clear the drop.** Most miss-only pins are unreachable
because they constrain a hashed value, so the offending call misses and is rejected on its first
occurrence. That argument fails here: every check above constrains the caller-supplied output
tensor's *own* spec, which appears nowhere in `compute_program_hash`.

| Dropped check (`reduce_scatter_validate_utils.cpp`) | Value it constrains | In the key? | Reachable on a hit? |
|---|---|---|---|
| memory layout enumeration (67-73) | passed output's `memory_config` | No | Yes |
| `storage_type() == DEVICE` (75) | passed output's storage kind | No | Yes, but loud — a host tensor fails to resolve a buffer |
| `layout() == input.layout()` (76-77) | passed output's layout vs hashed input layout | No | Yes |
| `dtype() == input.dtype()` (78) | passed output's dtype vs hashed input dtype | No | Yes |
| `page_config() == input.page_config()` (79-81) | passed output's page config vs hashed input page config | No | Yes |
| **`memory_config() == memory_config` (82-84)** | passed output's memory config vs hashed `output_mem_config` | **No** | **Yes — and silent** |
| `padded_shape` rank and per-dim checks (86-100) | passed output's padded shape vs hashed input shape and `ring_size` | No | Yes |
| block-sharded implies L1 (102-106) | passed output's memory config | No | Yes |
| all checks on `input_tensor` (before line 64) | hashed input properties | Yes | No — self-enforcing, rejected on the first miss |

Line 82-84 is the decisive one. It relates an **unhashed** value (the passed buffer's own memory
config) to a **hashed** one (`output_mem_config`), which is exactly the shape of an evadable pin: two
calls carrying the same `output_mem_config` but different persistent output buffers compute the same
key. The second hits, meets only the narrow input-side hit validator, and runs a program whose output
addressing was built for the first buffer.

**Reachable from Python without violating any enforced constraint.** `persistent_output_buffers` is
an exposed optional list (`strided_reduce_scatter_async_nanobind.cpp:38,62`), and element 1 becomes
`optional_output_tensor` verbatim:

```45:64:ttnn/cpp/ttnn/operations/experimental/ccl/strided_reduce_scatter_async/strided_reduce_scatter_async.cpp
    const bool using_persistent_buffers = persistent_output_buffers.has_value();

    std::optional<ttnn::Tensor> optional_intermediate_tensor = std::nullopt;
    std::optional<ttnn::Tensor> optional_output_tensor = std::nullopt;

    if (using_persistent_buffers) {
        const auto& buffers = persistent_output_buffers.value();
        if (!buffers.empty()) {
            optional_intermediate_tensor = buffers[0];
        }
        if (buffers.size() >= 2) {
            optional_output_tensor = buffers[1];
        }
    }

    // Call the prim operation
    auto result = ttnn::prim::strided_reduce_scatter_async(
        input_tensor,
        optional_intermediate_tensor,
        optional_output_tensor,
```

`memory_config` and `persistent_output_buffers` are independent keyword arguments — the former
becomes `output_mem_config` via `memory_config.value_or(input_tensor.memory_config())`
(`strided_reduce_scatter_async.cpp:68`) — and nothing on the hit path ties one to the other.

**Reproduction.** Same input tensor, same semaphores, same `mm_*` parameters, same
`memory_config=M` in both calls:

- Call 1: `persistent_output_buffers=[I, O_dram_interleaved]`, where `O_dram_interleaved`'s memory
  config is `M` (as the miss validator requires). Miss; the writer is compiled without
  `OUTPUT_IS_SHARDED` and with interleaved `TensorAccessorArgs`.
- Call 2: `persistent_output_buffers=[I, O_l1_width_sharded]`, same shape, dtype, layout and page
  config, memory config `M' != M`.

Both calls hash `output_mem_config = M`, and the passed buffer contributes nothing to the key, so the
keys are identical and call 2 hits. `create_output_tensors` returns the caller's tensor verbatim
(`device/strided_reduce_scatter_async_op_device_operation.cpp:152-154`), and
`override_runtime_arguments` dutifully writes `O_l1_width_sharded`'s base address into writer arg 1 —
so the kernel applies DRAM interleaving math to an L1 shard base, and the sharding runtime-arg tail
(#9) is stale as well. Silent wrong data and out-of-bounds writes, not a throw: the miss validator
that would have caught the mismatch is the one the hit path replaced.

**This is structurally the same defect as #4**, and should be read alongside it: an unhashed
caller-supplied buffer selects a kernel define and a block of compile-time accessor args, and the
pin that would have constrained it is absent from the path the offending call takes. The two differ
only in *why* the pin is missing — for the intermediate the memory-config check is skipped on both
paths when `intermediate_memory_config` is `nullopt`, whereas for the output it exists but lives in
the validator the hit path does not run. Both grade BUG under rule 4a-i: the bad configuration is
reachable through the public API without violating any enforced constraint.

**There is an exact in-repo precedent for the fix.** The sibling `ttnn::prim::ReduceScatterDeviceOperation`
solves precisely this case in its hit validator:

```32:42:ttnn/cpp/ttnn/operations/ccl/reduce_scatter/device/reduce_scatter_device_operation.cpp
void ReduceScatterDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        auto output_specs = compute_output_specs(operation_attributes, tensor_args);
        TT_FATAL(
            tensor_args.optional_output_tensor.value().tensor_spec() == output_specs.at(1),
            "Output tensor spec {} does not match computed output spec {}",
            tensor_args.optional_output_tensor.value().tensor_spec(),
            output_specs.at(1));
    }
}
```

One `TensorSpec` comparison against the spec the op would have computed itself, guarded on the
optional being present. Because `compute_output_specs` derives that spec entirely from hashed values,
the comparison pins the passed buffer to the key. Note that the sibling's *miss* validator opens by
calling its own hit validator (`reduce_scatter_device_operation.cpp:18`), so the pin holds on both
paths from a single definition — worth copying, since this op's two validators are independent and
have already drifted. Extended to the intermediate tensor, the same check also closes #4; the port,
its per-dispatch cost and why the extension works are set out in recommendation 1.

### 4. `tensor_args.optional_intermediate_tensor`

**Verdict: BUG**, in one specific configuration: when the caller supplies persistent buffers *and*
leaves `intermediate_memory_config` unset.

The intermediate tensor's sharding is structural in the same way as the output's — it selects two
defines and two blocks of compile-time args:

```411:414:ttnn/cpp/ttnn/operations/experimental/ccl/strided_reduce_scatter_async/device/strided_reduce_scatter_async_program.cpp
    if (intermediate_is_sharded) {
        reader_compute_defines["INTERMEDIATE_IS_SHARDED"] = "1";
        writer_compute_defines["INTERMEDIATE_IS_SHARDED"] = "1";
    }
```

```500:504:ttnn/cpp/ttnn/operations/experimental/ccl/strided_reduce_scatter_async/device/strided_reduce_scatter_async_program.cpp
    if (intermediate_is_sharded) {
        shard_builder::extend_sharding_compile_time_args(intermediate_tensor, sender_reader_compile_args);
    } else {
        tt::tt_metal::TensorAccessorArgs(intermediate_tensor.buffer()).append_to(sender_reader_compile_args);
    }
```

(and the same at `device/strided_reduce_scatter_async_program.cpp:568-572` for the writer).

Its layout, dtype, page config and shape are all pinned to hashed values — layout/dtype/page config
by the shared validator, shape by this op's own check:

```110:128:ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_common/reduce_scatter_validate_utils.cpp
void validate_intermediate_tensor(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& intermediate_tensor,
    const std::optional<ttnn::MemoryConfig>& optional_intermediate_mem_config) {
    TT_FATAL(intermediate_tensor.storage_type() == StorageType::DEVICE, "Intermediate tensor must be on device");
    TT_FATAL(
        intermediate_tensor.layout() == input_tensor.layout(),
        "Intermediate tensor layout must match input tensor layout");
    TT_FATAL(
        intermediate_tensor.dtype() == input_tensor.dtype(), "Intermediate tensor dtype must match input tensor dtype");
    TT_FATAL(
        intermediate_tensor.tensor_spec().page_config() == input_tensor.tensor_spec().page_config(),
        "Intermediate tensor page config must match input tensor page config");

    if (optional_intermediate_mem_config.has_value()) {
        TT_FATAL(
            intermediate_tensor.memory_config() == optional_intermediate_mem_config.value(),
            "Intermediate tensor memory config must match provided intermediate_mem_config");
    }
```

Note the guard on the memory-config check: it only fires when
`optional_intermediate_mem_config.has_value()`. When it is `nullopt` — which is its default — the
supplied intermediate tensor's memory config is **neither validated nor hashed**. The hash contains
`optional_intermediate_mem_config` (`...device_operation.cpp:168`), which is `nullopt` in both calls
and therefore carries no information about the actual tensor.

The two arguments are independent at the public API: `persistent_output_buffers` and
`intermediate_memory_config` are separate parameters, and the intermediate tensor is taken straight
from the buffer list without reference to the memory config:

```45:58:ttnn/cpp/ttnn/operations/experimental/ccl/strided_reduce_scatter_async/strided_reduce_scatter_async.cpp
    const bool using_persistent_buffers = persistent_output_buffers.has_value();

    std::optional<ttnn::Tensor> optional_intermediate_tensor = std::nullopt;
    std::optional<ttnn::Tensor> optional_output_tensor = std::nullopt;

    if (using_persistent_buffers) {
        const auto& buffers = persistent_output_buffers.value();
        if (!buffers.empty()) {
            optional_intermediate_tensor = buffers[0];
        }
        if (buffers.size() >= 2) {
            optional_output_tensor = buffers[1];
        }
    }
```

and `create_output_tensors` uses it verbatim
(`device/strided_reduce_scatter_async_op_device_operation.cpp:148-150`).

**Reproduction.** Same input tensor, same semaphores, same `mm_*` parameters,
`intermediate_memory_config=None` in both calls:

- Call 1: `persistent_output_buffers=[I_dram_interleaved, O]`
- Call 2: `persistent_output_buffers=[I_l1_width_sharded, O]` (same shape, dtype, layout and page
  config; only the memory config differs)

The keys are identical. Call 2 hits call 1's entry. The cached reader and writer were compiled
without `INTERMEDIATE_IS_SHARDED` and with interleaved `TensorAccessorArgs` describing DRAM banking;
the override correctly writes `I_l1_width_sharded`'s base address into reader arg 1 and writer arg 0,
so the kernels compute NOC addresses by applying DRAM interleaving math to an L1 shard base. Every
intermediate read and write lands at the wrong address. Because the intermediate is the ring's
carry buffer, the symptom is wrong numerical results on every ring step past 0, plus writes into
whatever else occupies those L1/DRAM locations.

Note also that this validation is *miss*-only: `validate_on_program_cache_hit` checks only storage
and buffer liveness

```22:28:ttnn/cpp/ttnn/operations/experimental/ccl/strided_reduce_scatter_async/device/strided_reduce_scatter_async_op_device_operation.cpp
void StridedReduceScatterAsyncDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    // Lightweight validation for cache hits
    const auto& input_tensor = tensor_args.input_tensor;
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Input tensor must be on device");
    TT_FATAL(input_tensor.buffer() != nullptr, "Input tensor must have a buffer");
}
```

so the pins in #3 and the rest of #4 hold only for the call that *created* the entry, not for the
call that reuses it. Because this op defines a hit validator, rule 4a-ii forbids grading those
"VALID — pinned by validation"; they are at best **CAVEAT — pinned only on the miss path**. The
pinned values are the caller-supplied optional tensors' own specs, which are absent from the key, so
rule 4a-iii's reachability filter does not clear them either — which is what escalates the output
tensor in #3 from a caveat to a bug in its own right. The memory-config gap adjudicated here is a BUG
for a stronger reason still: in the `nullopt` configuration the pin does not exist on *either* path,
so even deleting this op's hit validator would not close it.

The intermediate tensor's remaining properties — layout, dtype, page config and shape — are recorded
as CAVEAT because they are pinned on the miss path only. They are not separately counted as bugs
because the memory-config hole above already grades this omission a BUG and is the strictly worse
instance of the same exposure; a strict reading of 4a-i would put them on the same footing as #3,
and the fix in recommendation 1 closes all of them in one change.

### 5. `input_tensor` alignment and storage kind

**Verdict: VALID — unused** (alignment); **VALID — pinned by validation** (storage kind).

These two are the *only* input-tensor properties the hash drops relative to the default. Storage kind
is pinned twice over — on misses by
`ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_common/reduce_scatter_validate_utils.cpp:26-27`
and on hits by the hit validator quoted above, which is exactly the guard that makes this
"pinned by validation" verdict hold for hits as well. Alignment reaches the program only through
`input_tensor.buffer()->page_size()` (`device/strided_reduce_scatter_async_program.cpp:246`) and
`->size()` / `->num_pages()` (lines 228 and 333), all determined by the hashed
`{padded_shape, dtype, page_config, memory_config}` — and the miss validator additionally requires
`page_size % alignment == 0`
(`.../reduce_scatter_validate_utils.cpp:29-30`).

### 6. Fused-op signalers and the `addcmul` fusion

**Verdict: VALID — unused.**

The program builder supports a reduce-scatter fused-op signaler, a matmul fused-op signaler, and a
fused `addcmul` epilogue, all of which would add defines
(`device/strided_reduce_scatter_async_program.cpp:418-428`,
`device/strided_reduce_scatter_async_program.cpp:436-439`), extra CBs
(`device/strided_reduce_scatter_async_program.cpp:380-398`), extra compile-time args
(`device/strided_reduce_scatter_async_program.cpp:506-511`,
`device/strided_reduce_scatter_async_program.cpp:610-614`) and extra runtime args. None of them is
reachable from this device operation — `create_at` passes `std::nullopt` for all five:

```851:853:ttnn/cpp/ttnn/operations/experimental/ccl/strided_reduce_scatter_async/device/strided_reduce_scatter_async_program.cpp
    std::optional<ttnn::experimental::ccl::ReduceScatterFusedOpSignaler> fused_op_signaler = std::nullopt;
    std::optional<ttnn::experimental::ccl::StridedReduceScatterFusedOpSignaler> mm_fused_op_signaler = std::nullopt;
    tt::tt_metal::Program program{};
```

```881:883:ttnn/cpp/ttnn/operations/experimental/ccl/strided_reduce_scatter_async/device/strided_reduce_scatter_async_program.cpp
        std::nullopt,   // fused_ternary_scalar
        std::nullopt,   // addcmul_input_tensor1
        std::nullopt);  // addcmul_input_tensor2
```

so `fuse_op`, `fuse_mm_op` and `fuse_rs_addcmul` are compile-time constants `false` on this path, and
`shared_vars.reader_addcmul_rt_arg_offset` is always 0, which is why the override's addcmul branch
(`device/strided_reduce_scatter_async_program.cpp:788-792`) never fires. Nothing to hash today. This
becomes a live omission the moment another caller of
`build_ring_strided_reduce_scatter_async_program_artifacts` enables them — see the recommendations.

### 7. `ring_index`, forward/backward neighbour coordinates, unicast/mcast fabric configuration

**Verdict: VALID — invariant** (determined by the mesh coordinates the framework appends to every
key, plus hashed attributes).

`ring_index` is compile-time arg 0 of both the reader and the writer, and arg 15 of the compute
kernel. It and the neighbour coordinates are derived from the mesh coordinate plus hashed attributes:

```842:849:ttnn/cpp/ttnn/operations/experimental/ccl/strided_reduce_scatter_async/device/strided_reduce_scatter_async_program.cpp
    const auto forward_coord = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
        input_tensor, mesh_coordinate, 1, operation_attributes.topology, operation_attributes.cluster_axis);
    const auto backward_coord = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
        input_tensor, mesh_coordinate, -1, operation_attributes.topology, operation_attributes.cluster_axis);
    TT_FATAL(forward_coord.has_value() || backward_coord.has_value(), "forward_coord or backward_coord is null");

    const uint32_t ring_index = ::ttnn::ccl::get_linearized_index_from_physical_coord(
        input_tensor, mesh_coordinate, operation_attributes.cluster_axis);
```

`topology` and `cluster_axis` are hashed; the mesh coordinate is appended by the framework to both the
default and the custom path:

```989:992:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
        // Combine with the mesh coordinates the workload is targeting.
        for (const auto& coord : mesh_device_operation_utils::extract_tensor_coordinates(tensor_args, mesh_device)) {
            hash = ttsl::hash::hash_objects(hash, coord);
        }
```

The `unicast_forward_args` / `mcast_forward_args` blocks appended to the writer's compile-time args
(`device/strided_reduce_scatter_async_program.cpp:247-250`,
`device/strided_reduce_scatter_async_program.cpp:559-566`) derive from the same coordinates plus
`ring_size`, so they are keyed too. This is the case the brief asks to work out explicitly rather
than assume, and here it genuinely does hold.

### 8. Fabric mux runtime args

**Verdict: CAVEAT.**

The mux kernel's runtime args and the workers' mux-connection runtime args are set at build time and
never re-patched:

```638:648:ttnn/cpp/ttnn/operations/experimental/ccl/strided_reduce_scatter_async/device/strided_reduce_scatter_async_program.cpp
                const auto src_node_id = mesh_device->get_fabric_node_id(sender_device_coord);
                if (dir) {  // forward
                    const auto dst_node_id = mesh_device->get_fabric_node_id(forward_coord.value());
                    mux_rt_args = mux_kernel_config.get_fabric_mux_run_time_args(
                        src_node_id, dst_node_id, link, program, {mux_logical_core});
                } else {
                    const auto dst_node_id = mesh_device->get_fabric_node_id(backward_coord.value());
                    mux_rt_args = mux_kernel_config.get_fabric_mux_run_time_args(
                        src_node_id, dst_node_id, link, program, {mux_logical_core});
                }
                tt::tt_metal::SetRuntimeArgs(program, mux_kernel_id, {mux_logical_core}, mux_rt_args);
```

The workers' own mux-connection runtime args, appended to the writer at index 11 onwards by
`append_fabric_mux_connection_rt_args`
(`device/strided_reduce_scatter_async_program.cpp:708-718`), are in the same position.
`override_runtime_arguments` touches only reader indices 0-2 and writer indices 0, 1, 4, 5 and 7,
never `mux_kernel_id` and never the mux-connection tail. This
is safe while the fabric configuration is fixed for the lifetime of the mesh device — the mux's own
geometry (`num_full_size_channels = num_workers_per_direction`, `num_buffers_full_size_channels`,
`mux_base_l1_address`) is derived from hashed attributes and the L1 base allocator address, and the
node ids come from hashed/appended coordinates. It would break on a fabric teardown and re-init, or
an allocator base-address change, between two calls sharing an entry. Same assumption as every
fabric CCL op in the tree.

### 9. Sharding runtime args for input, intermediate and output

**Verdict: CAVEAT.**

When a tensor is sharded, the factory appends `shard_builder::extend_sharding_run_time_args(...)`
after the fixed runtime args
(`device/strided_reduce_scatter_async_program.cpp:667-672`,
`device/strided_reduce_scatter_async_program.cpp:719-724`). These tail slots encode the shard core
list and are not re-written by the override, which only touches indices 0-2 (reader) and 0, 1, 4, 5,
7 (writer). They are safe for the input, whose memory config is hashed outright, and they would be
safe for the intermediate and the output if those tensors' memory configs were reliably pinned to a
hashed value — which is exactly what #3 and #4 show they are not. In both of those configurations
these tail args go stale along with the compile-time ones.

## Non-cache correctness defects

Everything above concerns the program cache. This section records a defect found during the audit
that is **not** a cache defect: the offending parameter is in the key, the cache does exactly the
right thing, and the wrong behaviour comes entirely from the factory. It is documented here rather
than dropped because it is, in practice, the most reachable of this op's three problems — the cache
bugs in #3 and #4 each need two calls in a particular configuration, whereas this one goes wrong on
the very first call.

### Tile geometry — the unguarded 32x32 assumption

**Not a program-cache bug. A factory bug, and the easier of the two to hit.**

`input_tensor.tensor_spec().page_config()` *is* in the hash
(`device/strided_reduce_scatter_async_op_device_operation.cpp:187`), which is more than the other
three CCL ops in this family manage. The tile *shape* is therefore part of the cache key, and none of
the aliasing consequences described for the sibling ops apply here. The defect is that the factory
hardcodes 32x32 tile geometry with no guard, so the freshly-built program it produces for a
non-32x32 tensor is wrong.

**One framework-wide caveat on that coverage: the transpose flags are not covered by anything.**
`Tile` carries `transpose_within_face` and `transpose_of_faces` as private members
(`tt_metal/api/tt-metalium/tile.hpp:57-58`), and neither reaches the cache key by any route. The
reflection hash sees only three of them:

```46:47:tt_metal/api/tt-metalium/tile.hpp
    static constexpr auto attribute_names = std::forward_as_tuple("tile_shape", "face_shape", "num_faces");
    auto attribute_values() const { return std::forward_as_tuple(tile_shape, face_shape, num_faces); }
```

and the canonical-key tiebreaker sees only two:

```122:124:tt_metal/impl/data_format/tile.cpp
bool Tile::operator==(const Tile& other) const {
    return tile_shape == other.tile_shape && face_shape == other.face_shape;
}
```

So hashing `page_config()` — or a whole `TensorSpec`, for that matter — buys tile *shape* coverage
and nothing more; a transposed tile is indistinguishable from an untransposed one in both halves of
the key. This is not a property of this op's custom hash: it applies identically to every op in the
tree, including default-hash ops, and only an explicit `TT_FATAL` on `get_transpose_within_face()` /
`get_transpose_of_faces()` closes it. It is noted here rather than counted as a finding because this
document is the only one of the four that relies on `page_config` being hashed, so it is the only
place where "the tile is covered" could otherwise be read as unconditional.

The factory converts the padded shape into tile counts using the architectural constants rather than
the tensor's actual `Tile`:

```307:313:ttnn/cpp/ttnn/operations/experimental/ccl/strided_reduce_scatter_async/device/strided_reduce_scatter_async_program.cpp
    const uint32_t input_tensor_Ht = input_tensor_shape[-2] / tt::constants::TILE_HEIGHT;
    const uint32_t input_tensor_Wt = input_tensor_shape[-1] / tt::constants::TILE_WIDTH;

    const uint32_t slice_B = input_tensor_B;
    const uint32_t slice_C = input_tensor_C;
    const uint32_t slice_Ht = input_tensor_Ht;
    const uint32_t slice_Wt = input_tensor_Wt / ring_size;
```

There is no tile-aware call anywhere in this op — no `tensor_spec().tile()`, no `get_tile_shape()`,
no `get_tile_size()`, in either the device operation or the program. The only tile-related code is
the divisibility check immediately above:

```288:297:ttnn/cpp/ttnn/operations/experimental/ccl/strided_reduce_scatter_async/device/strided_reduce_scatter_async_program.cpp
    TT_FATAL(
        !(input_tensor_shape[-2] % tt::constants::TILE_HEIGHT),
        "Input tensor height ({}) must be divisible by tile height ({}).",
        input_tensor_shape[-2],
        tt::constants::TILE_HEIGHT);
    TT_FATAL(
        !(input_tensor_shape[-1] % tt::constants::TILE_WIDTH),
        "Input tensor width ({}) must be divisible by tile width ({}).",
        input_tensor_shape[-1],
        tt::constants::TILE_WIDTH);
```

That is a *shape* check, not a tile-geometry check: it asserts the padded shape divides by 32, which
a `Tile{16, 32}` tensor of shape `[1, 1, 256, 1024]` satisfies perfectly. It never inspects
`tile().get_height()` or `tile().get_width()`. Nor does the shared validator —
`ttnn/cpp/ttnn/operations/experimental/ccl/reduce_scatter_common/reduce_scatter_validate_utils.cpp`
contains no reference to `tile()` at all. All three criteria of the unguarded-32x32 pattern hold: the
op accepts `Layout::TILE`, host-side code performs a bare `TILE_HEIGHT`/`TILE_WIDTH` tile-count
conversion, and nothing validates the tile geometry.

`input_tensor_Ht` and `input_tensor_Wt` are not incidental. They propagate into `slice_Ht` and
`slice_Wt`, and thence into `mm_cores_y_val`, `padded_slice_Ht`, `slice_Ht_per_core` and
`mm_M_unit_blocks_per_core`
(`device/strided_reduce_scatter_async_program.cpp:318-331`), all of which are compile-time args of
both the reader (indices 10, 12, 14, 21, 23 at
`device/strided_reduce_scatter_async_program.cpp:479-492`) and the compute kernel (indices 6, 11,
13, 14 at `device/strided_reduce_scatter_async_program.cpp:598-606`).

**What actually goes wrong.** With `Tile{16, 32}` and a padded shape of `[1, 1, 256, 1024]`, the true
tile-row count is 16 but `input_tensor_Ht` computes 8. Every downstream per-core block count is
halved, so the kernels traverse half the tensor. Compounding it, `page_size` on the very next lines
is read from the buffer and *is* tile-aware:

```246:246:ttnn/cpp/ttnn/operations/experimental/ccl/strided_reduce_scatter_async/device/strided_reduce_scatter_async_program.cpp
    uint32_t page_size = input_tensor.buffer()->page_size();
```

so the program mixes a correct 16x32 page size with tile counts computed for 32x32 — the two halves
of the address arithmetic disagree.

**The program cache behaves correctly here — do not go looking for an aliasing bug.** Because
`page_config` is hashed, a `Tile{16, 32}` tensor produces a different key from an otherwise identical
`Tile{32, 32}` tensor, misses the cache, and gets its own freshly-built program. There is no silent
inheritance of a 32x32 entry and no wrong hit. The failure is a *correctly-keyed but incorrectly-built*
program: the cache faithfully reproduces whatever the factory hands it, and what the factory hands it
is wrong. This is the same situation as `rotary_embedding_indexed`, where framework spec validation
rather than the key is what prevents the aliasing, and it is graded the same way — outside the cache
bug count, with the factory defect recorded in full.

None of that makes the defect less severe in practice. It needs a single call to trigger, where the
cache bugs need two: #4 a specific combination of persistent buffers and a `None`
`intermediate_memory_config`, #3 two calls whose persistent output buffers differ in memory config
under the same `memory_config` argument. If anything a user is more likely to hit this one; it is
simply not the cache's fault.

**The fix is cheaper here than in the sibling ops** precisely because `page_config` is already keyed:
adding the standard guard is sufficient on its own, with no hash change required.

```95:97:ttnn/cpp/ttnn/operations/data_movement/sharded/interleaved_to_sharded/device/interleaved_to_sharded_op.cpp
        auto tile = input_tensor.tensor_spec().tile();
        if (tile.get_height() != tt::constants::TILE_HEIGHT || tile.get_width() != tt::constants::TILE_WIDTH) {
            return {false, fmt::format("interleaved_to_sharded requires standard 32x32 tiles, got {}x{}", tile.get_height(), tile.get_width())};
```

The alternative — making the factory genuinely tile-aware by replacing `TILE_HEIGHT`/`TILE_WIDTH` at
`device/strided_reduce_scatter_async_program.cpp:307-308` with
`input_tensor.tensor_spec().tile().get_tile_shape()` — is also correct and, unusually, also needs no
hash change, again because `page_config` is already in the key. In the other three ops in this family
the equivalent change would have to be accompanied by a hash change or it would open an aliasing hole.

## Keys the custom hash adds beyond the default

Two, and both are deliberate improvements:

- `input_tensor.padded_shape()` in addition to `logical_shape()`. The default derives padding from
  `logical_shape` + `page_config` + `alignment`; hashing it explicitly is what lets `alignment` be
  dropped safely (#5). The factory works exclusively in padded terms
  (`device/strided_reduce_scatter_async_program.cpp:287-313`).
- The resolved sub-device `CoreRangeSet` in place of the `SubDeviceId` (#2) — a genuine relaxation.

## Framework side effect of having a custom hash

```1012:1014:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
        if constexpr (requires { DeviceOperation::compute_program_hash(attrs, tensor_args); }) {
            return key;  // custom hash -> opt out beyond the op-identity prefix
        } else {
```

`ProgramCacheKey::canonical` degrades to the op type name, so a 64-bit collision between two distinct
configurations of *this* op resolves to a wrong hit rather than a rebuild. Distinct ops still cannot
alias, because the op-identity prefix survives.

## Summary

| Omitted vs. default | Used by program? | Patched on hit? | Verdict |
|---|---|---|---|
| `semaphore` vector addresses | Yes — reader arg 2, writer args 4/5 | Yes | VALID — patched |
| `barrier_semaphore` address (presence bit is hashed) | Yes — writer arg 7 | Yes | VALID — patched |
| `sub_device_id` identity (resolved core set is hashed) | Yes — worker core selection, default worker count | n/a | VALID — relaxation win |
| `optional_output_tensor` spec (memory config above all) | Yes — `OUTPUT_IS_SHARDED` define, accessor CT args, sharding rt args | Address only | **BUG** |
| `optional_intermediate_tensor` memory config, when `intermediate_memory_config` is `nullopt` | Yes — `INTERMEDIATE_IS_SHARDED` define, accessor CT args, sharding rt args | Address only | **BUG** |
| `optional_intermediate_tensor` layout / dtype / page config / shape | Yes | n/a | CAVEAT — pinned only on the miss path |
| `input.alignment` | Only via hashed derivatives | n/a | VALID — unused |
| `input` storage kind | n/a | n/a | VALID — pinned by validation (miss *and* hit) |
| Fused-op signalers, `addcmul` fusion | No — hard-coded `nullopt` on this path | n/a | VALID — unused |
| `ring_index`, neighbour coords, unicast/mcast CT args | Yes — compile args | n/a | VALID — invariant (keyed via the mesh coordinates the framework appends) |
| Fabric mux rt args | Yes | No | CAVEAT — relies on fixed fabric config |
| Sharding rt-arg tails | Yes | No | CAVEAT — safe for the input, stale in both BUG rows |
| Buffer addresses (input, intermediate, output) | Yes | Yes | VALID — patched |

The tile geometry does not appear in this table: `page_config` is hashed, so the tile shape is not an
omission at all. (The `Tile` transpose flags are invisible to every hash in the tree, this one
included — see the note under `## Non-cache correctness defects` — but that is a framework property,
not something this op's custom hash drops.) The factory's hardcoded 32x32 assumption is a real defect
but not a cache one, and it is recorded under `## Non-cache correctness defects` above.

**Two program-cache bugs were found, and they are the same defect twice.** The hash itself is close
to a model of how this should be done: every scalar that reaches a compile-time arg is keyed; the
semaphore addresses are omitted *and* completely patched; the input tensor is hashed more completely
than the default would, `page_config` included; and `sub_device_id` is hashed by its resolved core
set, which is both correct and a real relaxation over the default. Both gaps are on the same axis —
a caller-supplied persistent buffer whose own spec is neither hashed nor pinned on the hit path,
while it selects a kernel define and a block of compile-time accessor args:

- **#3, the output tensor.** Pinned to `output_mem_config` by the shared validator, but only on the
  miss path; this op's narrow `validate_on_program_cache_hit` replaces that validator on hits, so a
  second call carrying a differently-configured output buffer under the same `memory_config`
  argument hits and runs with the first buffer's addressing.
- **#4, the intermediate tensor.** Worse: when `intermediate_memory_config` is `nullopt` — its
  default — the memory-config pin does not exist on *either* path, so deleting the hit validator
  would not close it. It needs a hash entry or a new `TT_FATAL`.

That is two bugs, not zero, so "close to a reference implementation" was too generous as an overall
characterisation of the op — and it remains too generous even after the tile finding is reclassified,
because both persistent-buffer gaps are genuine silent-wrong-hit bugs. Read alongside the non-cache
defect above, this op has three real correctness problems; it is the *hash* that is close to
exemplary, not the op. Recommendation 1 closes both cache bugs in a single change.

## Recommendations

1. Close both program-cache bugs at once by porting the sibling op's hit validator. Rewrite
   `StridedReduceScatterAsyncDeviceOperation::validate_on_program_cache_hit`
   (`device/strided_reduce_scatter_async_op_device_operation.cpp:22-28`) to keep its two input-side
   checks and add, on the pattern of
   `ReduceScatterDeviceOperation::validate_on_program_cache_hit`
   (`ttnn/cpp/ttnn/operations/ccl/reduce_scatter/device/reduce_scatter_device_operation.cpp:32-42`),
   a `TensorSpec` comparison for each supplied persistent buffer against the spec this op would
   compute itself:

   ```cpp
   if (tensor_args.optional_intermediate_tensor.has_value() || tensor_args.optional_output_tensor.has_value()) {
       const auto specs = compute_output_specs(operation_attributes, tensor_args);
       if (tensor_args.optional_intermediate_tensor.has_value()) {
           TT_FATAL(tensor_args.optional_intermediate_tensor->tensor_spec() == specs[0], "...");
       }
       if (tensor_args.optional_output_tensor.has_value()) {
           TT_FATAL(tensor_args.optional_output_tensor->tensor_spec() == specs[1], "...");
       }
   }
   ```

   **This closes #4 as well as #3**, which is worth stating because the sibling's version does not:
   it compares only `output_specs.at(1)`, the output. Extending it to `specs[0]` works here because
   `compute_output_specs` derives the intermediate spec from
   `optional_intermediate_mem_config.value_or(input_tensor.memory_config())` plus the Ring batch-1
   shard adjustment (`device/strided_reduce_scatter_async_op_device_operation.cpp:91-123`), and every
   one of those inputs is hashed — including `input_tensor.memory_config()`, which is precisely the
   value the `nullopt` configuration in #4 leaves unpinned. So the comparison pins the supplied
   intermediate to the key on the very path where the existing `TT_FATAL` is skipped.

   **Per-dispatch cost**, since this lands on the fast path: the whole block is guarded on at least
   one optional being present, and costs one `compute_output_specs` call — which constructs two
   `TensorSpec`s — plus one `TensorSpec` equality comparison per supplied buffer. Calls that pass no
   persistent buffers pay nothing. Hoisting the `compute_output_specs` call above both branches, as
   sketched, keeps it to one construction rather than the two the sibling's shape would produce.

   Two things to check when porting. First, mirror the sibling's structure of having
   `validate_on_program_cache_miss` call `validate_on_program_cache_hit` first
   (`reduce_scatter_device_operation.cpp:18`), so the pin is defined once and holds on both paths;
   this op's two validators are currently independent and have already drifted. Second, a full
   `TensorSpec` comparison is stricter than the piecewise checks in
   `reduce_scatter_common_validates` — it compares `logical_shape` and the whole `TensorLayout`,
   including alignment, where the shared validator compares `padded_shape` dimension by dimension and
   never looks at alignment — so an in-tree caller passing a compatible-but-not-identical persistent
   buffer would begin to fail. That is the correct outcome, but it is a behavioural change and should
   be landed expecting it.

   The narrower alternative for #4 alone is to hash
   `tensor_args.optional_intermediate_tensor.transform([](const auto& t) { return t.memory_config(); })`,
   or to extend `validate_intermediate_tensor` to compare against `input_tensor.memory_config()` when
   `optional_intermediate_mem_config` is `nullopt` — already what `compute_output_specs` assumes
   (`device/strided_reduce_scatter_async_op_device_operation.cpp:91-93`). Either is a smaller change,
   but neither touches #3, which lives on the hit path and can only be closed there.
2. Add the standard tile guard to `validate_on_program_cache_miss`, mirroring
   `interleaved_to_sharded_op.cpp:95-97`. This addresses the non-cache defect, not a cache one, but it
   is a two-line change and it is the cheapest correct fix available. Note that the divisibility
   checks at `device/strided_reduce_scatter_async_program.cpp:288-297` are not a substitute; they
   constrain the shape, not the tile.
3. Switch `ttsl::hash::hash_objects(...)` to `tt::tt_metal::operation::hash_operation<StridedReduceScatterAsyncDeviceOperation>(...)`.
   This puts `type_hash<Op>` in the 64-bit value and stops `dim` being silently used as the seed. No
   behavioural change today (the canonical key already carries op identity), purely defensive.
4. When the `addcmul` / fused-op-signaler paths are enabled from a device operation, the hash must
   grow accordingly: `fused_ternary_scalar` presence, `addcmul_input_tensor1/2` memory configs and
   `addcmul_input_tensor2.logical_shape()[-2] <= 1` (which selects the `ADDCMUL_B_BROADCAST` define at
   `device/strided_reduce_scatter_async_program.cpp:423-427`). Worth a comment in
   `compute_program_hash` pointing at this so the two do not drift.
5. The remaining miss-only pins on the intermediate tensor — layout, dtype, page config and shape —
   are subsumed by recommendation 1: a full `TensorSpec` comparison covers all of them, so no
   separate change is needed. The blunt alternative is to *delete*
   `validate_on_program_cache_hit` entirely, which under rule 4a-ii makes the framework substitute
   the miss validator on hits and restores every check at once
   (`ttnn/api/ttnn/device_operation.hpp:262-266`). It is the simplest and safest option, but it puts
   the entire miss validator — the full shared CCL validation, the semaphore-count check, the
   per-dimension shape loop — on every dispatch of a hot CCL op, which is presumably why the narrow
   hit validator was written in the first place. Prefer recommendation 1.
