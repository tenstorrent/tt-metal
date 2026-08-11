# Program Cache Audit — `experimental/ccl/llama_reduce_scatter_create_heads`

Audit of `LlamaReduceScatterCreateHeadsDeviceOperation::compute_program_hash` against the framework
default ("hash everything") key.

| | |
|---|---|
| Device operation | `ttnn::operations::experimental::ccl::LlamaReduceScatterCreateHeadsDeviceOperation` (`device/llama_reduce_scatter_create_heads_device_op.hpp:20-97`) |
| Custom hash | `device/llama_reduce_scatter_create_heads_device_op.cpp:126-143` |
| `operation_attributes_t` | `dim`, `cross_device_semaphore`, `subdevice_id`, `cluster_axis`, `output_mem_config`, `ring_devices`, `topology`, `num_links`, `num_heads`, `num_kv_heads`, `head_dim`, `slice_size`, `qkv_memory_config`, `use_noc1_only`, `use_optimal_ccl_for_llama` |
| `tensor_args_t` | `input_tensor`, `intermediate_packet_buffer` |
| Program factories | `LlamaReduceScatterCreateHeads` (single; `create_mesh_workload` / `create_at`) |
| `override_runtime_arguments` | **Yes** (`device/llama_reduce_scatter_create_heads_program_factory.cpp:858-904`) |
| `get_dynamic_runtime_args` | No |
| `validate_on_program_cache_hit` | Present but **empty** |
| Cache-hit patch mechanism | **Op-owned re-derivation** (the factory's `override_runtime_arguments` runs on every hit) |

The CSV row (*explicit / SELECTIVE / has own hit validator / has `override_runtime_arguments` / no
`get_dynamic_runtime_args`*) matches the code, with one important qualification: the "own hit
validator" is a no-op stub and therefore pins nothing.

```50:51:ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter_create_heads/device/llama_reduce_scatter_create_heads_device_op.cpp
void LlamaReduceScatterCreateHeadsDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*attributes*/, const tensor_args_t& /*tensor_args*/) {}
```

Anywhere below that reads "not pinned by validation", the empty hit validator is why. The *miss*
validator (`device/llama_reduce_scatter_create_heads_device_op.cpp:15-48`) is substantive and does
pin several things, and is cited where it applies. The next section establishes how much of that
pinning the hit path actually loses, which is much less than the empty override suggests.

## What the empty hit validator actually costs

Defining an empty hit validator nominally discards the entire miss validator on the hit path, but a
raw diff overstates the loss by roughly an order of magnitude. A miss-only pin on a value that is
*itself in the cache key* cannot be evaded: a call carrying a new value of a hashed parameter
computes a different key, misses, and the miss path unconditionally runs the miss validator before
building anything.

```301:301:ttnn/api/ttnn/device_operation.hpp
    mesh_device_operation_t::validate_on_program_cache_miss(operation_attributes, tensor_args);
```

So the offending configuration is rejected on its *first* occurrence and never reaches a hit. Only a
check constraining a value **absent** from the key is reachable on the hit path. Working the miss
validator through line by line:

| Dropped check | Value it constrains | In the key? | Reachable on a hit? |
|---|---|---|---|
| `dim == 3` (`device_op.cpp:19`) | `attributes.dim` | Yes — hash term 1 | No |
| `cluster_axis == 1` (`:20`) | `attributes.cluster_axis` | Yes — hash term 2 | No |
| `ring_devices == 4 or 2` (`:21-24`) | `attributes.ring_devices` | Yes — hash term 3 | No |
| `cross_device_semaphore.has_value()` (`:25`) | engagement of the semaphore optional | No | Yes in principle; not through the public API (see below) |
| `input_tensor.shard_spec().has_value()` (`:27`) | `input_tensor.memory_config()` | Yes — hash term 13 | No |
| `input_tensor.shard_spec()->shape[0] == 32` (`:28-31`) | `input_tensor.memory_config()` | Yes — hash term 13 | No |
| `intermediate_packet_buffer.shard_spec().has_value()` (`:33-35`) | packet buffer `memory_config` | **No** | **Yes** |
| `intermediate_packet_buffer.shard_spec()->shape[0] == 32` (`:36-39`) | packet buffer `memory_config` | **No** | **Yes** |
| `qkv_memory_config.has_value()` / `...shard_spec().has_value()` (`:40-42`) | engagement of two optionals | No | Pre-empted — both are dereferenced unconditionally earlier in the same dispatch (see below) |
| `qkv_memory_config->shard_spec()->shape[0] == 32` (`:43-46`) | qkv shard height | No | Yes, but the factory never reads that value |

Six of the ten checks cannot be reached on a hit. Five of those are self-enforcing: the two
input-tensor rows and the `dim` / `cluster_axis` / `ring_devices` rows all constrain values that are
hash terms, so a call violating any of them lands on its own key, misses, and is rejected on the miss
path. The sixth is pre-empted by an earlier unconditional dereference (see below). The empty hit
validator costs nothing for any of them, and the CSV's `own_hit_validator=Y` is not the hazard it
looks like here.

Two further pins are supplied by the framework rather than by this op, on both paths, before either
validator runs:

```453:457:ttnn/api/ttnn/device_operation.hpp
    for (const auto& input_tensor_ref : input_tensors) {
        const auto& input_tensor = input_tensor_ref.get();
        TT_FATAL(is_device_tensor(input_tensor), "Device Operations expect device tensors as inputs");
        TT_FATAL(input_tensor.is_allocated(), "Input Tensor is not allocated");
    }
```

That covers the storage kind and allocation of *both* tensors, including the packet buffer, on every
dispatch.

**Adjudicating the four reachable rows.** Per the reachability rule, only a *silent* failure justifies
paying for a check on the hit path, which is the fast path.

- **`cross_device_semaphore.has_value()`** — the only public entry point takes the semaphore by
  reference and cannot pass `nullopt`
  (`device/llama_reduce_scatter_create_heads_device_op.cpp:154`, `.cross_device_semaphore = semaphore`
  at `:172`). A caller constructing `operation_attributes_t` directly could disengage it, and then
  `override_runtime_arguments` dereferences the empty optional at
  `...program_factory.cpp:892` and `:898` — a fault, not a wrong result. Unreachable via the public
  API and loud if reached. **No fix recommended.**
- **The two `intermediate_packet_buffer` rows** — genuinely reachable and genuinely silent, but they
  are the narrow tip of omission #2: the packet buffer's *entire* spec is absent from the key, so its
  grid can move freely while these two checks pass. Hashing the packet buffer's `memory_config`
  (recommendation 2) puts the shard spec into the key and thereby makes both checks unreachable by
  construction, at zero per-dispatch cost. That is strictly better than replicating them into the hit
  validator. **No separate validator fix recommended.**
- **`qkv_memory_config` rows** — the two `has_value()` checks are dead as guards on either path,
  because `compute_output_specs` dereferences both optionals unconditionally at
  `device/llama_reduce_scatter_create_heads_device_op.cpp:66`
  (`attributes.qkv_memory_config.value().shard_spec()->grid`), and `create_output_tensors` runs at
  `ttnn/api/ttnn/device_operation.hpp:467` — before `launch_operation_with_adapter` and therefore
  before the dispatcher picks a validator at all. The shard-height check is reachable, but the value
  it constrains is never read: `compute_output_specs` takes only `->grid` from that shard spec and
  builds the Q/K/V shard shapes from the hashed `num_heads` and `head_dim` (`:87-89`). Restoring it
  would buy nothing. **No fix recommended.**

The consequence for this document is that the empty hit validator is **not** the reason any verdict
below is a BUG. All three bugs are pure hash omissions on values (`qkv_memory_config`'s grid, the
packet buffer's grid, `subdevice_id`) that are unchecked on the *miss* path too, so deleting the hit
validator would not catch a single one of the reproductions.

Because this op has a custom `compute_program_hash`, the canonical half of the cache key degrades to
the op-identity prefix (see "Framework side effect" below), so the five self-enforcing rows are
unreachable *absent a hash collision* rather than unconditionally unreachable.

## Cache-hit patch mechanism

The factory exposes `override_runtime_arguments` and no `apply_descriptor`, so the framework calls it
directly on every hit:

```279:285:ttnn/api/ttnn/device_operation.hpp
        if constexpr (requires { &WorkloadFactory::apply_descriptor; }) {
            WorkloadFactory::apply_descriptor(
                cached_mesh_workload, operation_attributes, tensor_args, tensor_return_value);
        } else {
            WorkloadFactory::override_runtime_arguments(
                cached_mesh_workload, operation_attributes, tensor_args, tensor_return_value);
        }
```

Nothing is inferred by the framework: no `resolve_bindings`, no automatic CB address patching, no
`get_dynamic_runtime_args`. Whatever `override_runtime_arguments` does not rewrite stays at its
first-miss value, and everything structural is baked into the cached `Program` and never refreshed.

This op's `override_runtime_arguments` is, for the address family, complete: it re-points both
globally-allocated circular buffers and rewrites all four per-call address/semaphore runtime-arg
slots on every core:

```887:901:ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter_create_heads/device/llama_reduce_scatter_create_heads_program_factory.cpp
        UpdateDynamicCircularBufferAddress(program, shared_variables.cb_handles[0], *input_tensor_buffer);
        UpdateDynamicCircularBufferAddress(program, shared_variables.cb_handles[1], *packet_buffer);

        for (const auto& core : cores) {
            auto& writer_runtime_args = tt::tt_metal::GetRuntimeArgs(program, unary_writer_kernel_id, core);
            writer_runtime_args[0] = (uint32_t)operation_attributes.cross_device_semaphore->address();
            writer_runtime_args[8] = q_base_addr;
            writer_runtime_args[9] = k_base_addr;
            writer_runtime_args[10] = v_base_addr;

            auto& reader_runtime_args = tt::tt_metal::GetRuntimeArgs(program, unary_reader_kernel_id, core);
            reader_runtime_args[0] = (uint32_t)operation_attributes.cross_device_semaphore->address();
            reader_runtime_args[10] = q_base_addr;
            reader_runtime_args[11] = k_base_addr;
            reader_runtime_args[12] = v_base_addr;
        }
```

The obligation that remains on the hash is therefore the *structural* one: every compile-time arg,
every kernel `define`, every CB size, and every core range must be a pure function of the hashed set
plus the mesh coordinates the framework appends. This op has an unusually heavy structural surface —
core lists are baked into the kernels as preprocessor defines
(`...program_factory.cpp:642-652`) — which makes the omissions below more consequential than the
usual runtime-arg case.

## Baseline: what the default hash would cover

| Source | Fields |
|---|---|
| `operation_attributes` | all 15 fields, including `cross_device_semaphore`, `subdevice_id`, `output_mem_config` and `qkv_memory_config` |
| `input_tensor` | storage kind, `logical_shape`, `dtype`, `page_config`, `memory_config`, `alignment` |
| `intermediate_packet_buffer` | storage kind, `logical_shape`, `dtype`, `page_config`, `memory_config`, `alignment` |
| appended by framework | the mesh coordinates of the tensors |

## What the custom hash covers

```128:142:ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter_create_heads/device/llama_reduce_scatter_create_heads_device_op.cpp
    return tt::tt_metal::operation::hash_operation<LlamaReduceScatterCreateHeadsDeviceOperation>(
        attributes.dim,
        attributes.cluster_axis,
        attributes.ring_devices,
        attributes.num_links,
        attributes.num_heads,
        attributes.num_kv_heads,
        attributes.head_dim,
        attributes.slice_size,
        attributes.topology,
        attributes.use_noc1_only,
        attributes.use_optimal_ccl_for_llama,
        tensor_args.input_tensor.dtype(),
        tensor_args.input_tensor.memory_config(),
        tensor_args.input_tensor.device()->id());
```

Good news up front, since the task asks specifically about these: the **head-count and head-dim
attributes are all hashed** — `num_heads`, `num_kv_heads`, `head_dim` and `slice_size` are terms 5-8.
So are `topology`, `ring_devices` (the ring size), `cluster_axis` and `num_links`. The two behaviour
switches that change kernel configuration, `use_noc1_only` (which selects the NOC and NOC mode at
`...program_factory.cpp:663-665` and `...program_factory.cpp:699-701`) and
`use_optimal_ccl_for_llama` (which selects the sender core placement strategy at
`...program_factory.cpp:441-446`), are hashed too.

## Omitted parameters

### 1. `operation_attributes.qkv_memory_config`

**Verdict: BUG.**

This is the omission that matters most. `qkv_memory_config` supplies the sub-core grid that the Q, K
and V output shards are laid out on:

```66:67:ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter_create_heads/device/llama_reduce_scatter_create_heads_device_op.cpp
    auto sub_core_grid = attributes.qkv_memory_config.value().shard_spec()->grid;
    auto start_core_coord = sub_core_grid.bounding_box().start_coord;
```

```87:96:ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter_create_heads/device/llama_reduce_scatter_create_heads_device_op.cpp
    tt::tt_metal::ShardSpec q_shard_spec{q_shard_grid, {attributes.num_heads, head_dim}};
    tt::tt_metal::ShardSpec k_shard_spec{k_shard_grid, {attributes.num_heads, head_dim}};
    tt::tt_metal::ShardSpec v_shard_spec{v_shard_grid, {attributes.num_heads, head_dim}};
    const auto& qkv_memory_config = attributes.qkv_memory_config.value();
    tt::tt_metal::MemoryConfig q_mem_config =
        tt::tt_metal::MemoryConfig(qkv_memory_config.memory_layout(), qkv_memory_config.buffer_type(), q_shard_spec);
```

Those grids come straight back into the factory and are compiled into the kernels as preprocessor
defines:

```352:354:ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter_create_heads/device/llama_reduce_scatter_create_heads_program_factory.cpp
    const auto q_output_grid = q_output_shard_spec.grid;
    const auto k_output_grid = k_output_shard_spec.grid;
    const auto v_output_grid = v_output_shard_spec.grid;
```

```644:649:ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter_create_heads/device/llama_reduce_scatter_create_heads_program_factory.cpp
    reader_defines["Q_OUTPUT_CORE_XY"] =
        detail::rs_heads_fusion::cores_to_string(to_worker_cores(q_output_cores, ncores_output));
    reader_defines["K_OUTPUT_CORE_XY"] =
        detail::rs_heads_fusion::cores_to_string(to_worker_cores(k_output_cores, ncores_output));
    reader_defines["V_OUTPUT_CORE_XY"] =
        detail::rs_heads_fusion::cores_to_string(to_worker_cores(v_output_cores, ncores_output));
```

`reader_defines` is passed as `.defines` for the reader kernel
(`...program_factory.cpp:667`) and copied into `writer_defines`
(`...program_factory.cpp:691`) for the writer. A define is the most rigidly baked-in thing a program
can have — it changes the compiled binary. Nothing about it is recoverable on a cache hit.

The only thing the miss validator checks about `qkv_memory_config` is the shard height:

```40:47:ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter_create_heads/device/llama_reduce_scatter_create_heads_device_op.cpp
    if (attributes.qkv_memory_config.has_value()) {
        TT_FATAL(
            attributes.qkv_memory_config.value().shard_spec().has_value(), "qkv_memory_config must have a shard spec");
        TT_FATAL(
            attributes.qkv_memory_config.value().shard_spec().value().shape[0] == 32,
            "qkv_memory_config shard height must be 32 but got {}",
            attributes.qkv_memory_config.value().shard_spec().value().shape[0]);
    }
```

which says nothing about the grid, the memory layout or the buffer type.

**Reproduction.** Fixed input tensor, packet buffer, semaphore, `num_heads=8`, `num_kv_heads=1`,
`head_dim=128`, `slice_size=8`:

- Call 1: `qkv_memory_config` with a sub-core grid of `{(1,0)-(2,9)}`.
- Call 2: identical arguments except the sub-core grid is `{(3,0)-(4,9)}` (same core count, different
  location).

The hash is byte-identical — `qkv_memory_config` is not a term. Call 2 hits call 1's entry.
`create_output_tensors` correctly allocates Q/K/V on the *new* cores, and
`override_runtime_arguments` correctly pushes the new base addresses into the reader/writer args. But
the cached reader and writer kernels still contain `Q_OUTPUT_CORE_XY = {{1,0},...}` as a compile-time
define, so every NOC write goes to the *old* cores. The new output tensors are never written, and
whatever tensors now occupy the old cores get corrupted. The symptom is silent garbage in Q/K/V plus
memory corruption of unrelated L1 residents.

### 2. `tensor_args.intermediate_packet_buffer` (all spec properties)

**Verdict: BUG** (for the spec; its *address* is correctly patched — see #6).

The packet buffer's shard grid selects the packet-worker cores, which is the single most structural
decision in this factory:

```430:447:ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter_create_heads/device/llama_reduce_scatter_create_heads_program_factory.cpp
    auto intermediate_packet_buffer_grid = tensor_args.intermediate_packet_buffer.shard_spec().value().grid;
    // UNCOMMENT this once we can allocate persistent buffers across all device lifetimes
    uint32_t num_packets_total_per_device =
        (input_blocks_per_stick + num_blocks_per_packet - 1) / num_blocks_per_packet;
    auto packet_worker_cores_grid = detail::rs_heads_fusion::get_worker_cores(
        intermediate_packet_buffer_grid,
        num_packets_total_per_device,
        input_shard_spec.orientation == ShardOrientation::ROW_MAJOR);

    auto available_cores = sub_device_cores.subtract(packet_worker_cores_grid);

    auto sender_core_grid = operation_attributes.use_optimal_ccl_for_llama
                                ? llama_specific::get_custom_cores(num_workers_per_link * num_links)
                                : detail::rs_heads_fusion::get_worker_cores(
                                      available_cores,
                                      num_workers_per_link * num_links,
                                      input_shard_spec.orientation == ShardOrientation::ROW_MAJOR);
    auto all_cores_grid = packet_worker_cores_grid.merge(sender_core_grid);
```

`packet_worker_cores_grid` is the core range the compute kernel is created on
(`...program_factory.cpp:716`), `all_cores_grid` is the core range for the reader, the writer and
all five circular buffers (`...program_factory.cpp:571-579`, `...program_factory.cpp:660`,
`...program_factory.cpp:696`), and the derived physical coordinates go into compile-time args and
another define:

```632:637:ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter_create_heads/device/llama_reduce_scatter_create_heads_program_factory.cpp
        packet_start_worker_core.at(0).x,
        packet_start_worker_core.at(0).y,
        packet_end_worker_core.at(0).x,
        packet_end_worker_core.at(0).y,
        sender_cores.size(),
        total_num_read_txns};
```

```639:641:ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter_create_heads/device/llama_reduce_scatter_create_heads_program_factory.cpp
    if (packet_worker_cores_grid.num_cores() == 1) {
        reader_defines["SKIP_MCAST"] = "1";
    }
```

The miss validator only checks that the packet buffer has a shard spec with height 32
(`device/llama_reduce_scatter_create_heads_device_op.cpp:33-39`); its grid, dtype and width are free.

**Reproduction.** Call 1 with a packet buffer sharded over cores `{(0,0)-(0,7)}`; call 2 with an
otherwise identical packet buffer sharded over `{(5,0)-(5,7)}`. Same hash. Call 2 hits, and the
cached program still creates the fabric-receiver CB and runs the reduction compute kernel on column
0, while `UpdateDynamicCircularBufferAddress` re-points that CB at an address in column 5's L1 — an
address that is not even mapped as the packet buffer on the cores actually executing. The multicast
range compile-time args (`packet_start_worker_core` / `packet_end_worker_core`) also still describe
column 0.

Callers in the Llama model allocate the packet buffer once and reuse it, so the grid does not move in
practice; that is why this has not fired.

### 3. `operation_attributes.subdevice_id`

**Verdict: BUG** when `use_optimal_ccl_for_llama` is false; **VALID — unused** when it is true.

```410:412:ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter_create_heads/device/llama_reduce_scatter_create_heads_program_factory.cpp
    auto sub_device_cores = mesh_device->worker_cores(
        tt::tt_metal::HalProgrammableCoreType::TENSIX,
        operation_attributes.subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0)));
```

`sub_device_cores` feeds `available_cores` and thence `sender_core_grid` in the non-optimal branch
(quoted in #2), which is where the reader and writer kernels are created and where the fabric
connections are attached. It is structural and is never re-derived on a hit —
`override_runtime_arguments` iterates `shared_variables.core_range`, i.e. the *cached* core set
(`...program_factory.cpp:883-885`).

The saving grace is that `use_optimal_ccl_for_llama` **is** hashed, and when it is true the sender
cores come from `llama_specific::get_custom_cores(num_workers_per_link * num_links)`, a pure function
of the hashed `num_links`. So the exposure is precisely the `use_optimal_ccl_for_llama == false`
configuration.

**Reproduction.** With `use_optimal_ccl_for_llama=False`: call 1 with `subdevice_id` = the default
sub-device (full Tensix grid), call 2 with a sub-device restricted to a subset of the grid. The hash
is identical, so call 2 reuses call 1's program with sender kernels on cores outside the requested
sub-device — a dispatch-domain violation, and on a mesh where another sub-device owns those cores, a
data race.

### 4. `operation_attributes.cross_device_semaphore` (`GlobalSemaphore`)

**Verdict: VALID — patched.**

The semaphore's address occupies slot 0 of both the reader and writer runtime args at build time:

```722:723:ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter_create_heads/device/llama_reduce_scatter_create_heads_program_factory.cpp
    std::vector<uint32_t> reader_runtime_args = {
        cross_device_semaphore->address(), local_semaphore, false, false, 0, 0, false, 0, 0, 0, 0, 0, 0};
```

```762:763:ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter_create_heads/device/llama_reduce_scatter_create_heads_program_factory.cpp
        std::vector<uint32_t> writer_runtime_args = {
            cross_device_semaphore->address(), local_semaphore, false, false, 0, 0, 0, 0, 0, 0, 0};
```

and both are rewritten for every core on every hit (`...program_factory.cpp:892` and
`...program_factory.cpp:898`, quoted above). It is never a compile-time arg. Correctly omitted.

Note also `local_semaphore` (slot 1), which is a *program-scoped* `CreateSemaphore` id
(`...program_factory.cpp:654`), not an address. Semaphore ids are assigned deterministically in
program-build order, so the id is stable for a given program and correctly not hashed and correctly
not patched.

### 5. `operation_attributes.output_mem_config`

**Verdict: VALID — unused.**

The field is populated at `device/llama_reduce_scatter_create_heads_device_op.cpp:175` and then never
read: a repository search for `output_mem_config` inside this op directory finds only the
declaration, that assignment, and a mention in a docstring. `compute_output_specs` builds the Q/K/V
memory configs from `qkv_memory_config` instead
(`device/llama_reduce_scatter_create_heads_device_op.cpp:90-96`). Dropping a dead field from the hash
is a small relaxation win — the default hash would force a rebuild for a `memory_config=` argument
that does nothing.

### 6. Buffer addresses (input, packet buffer, Q/K/V outputs)

**Verdict: VALID — patched.**

Addresses must not be hashed. Two of them are globally-allocated CB bindings — the input tensor
(`...program_factory.cpp:466-469`) and the packet buffer
(`...program_factory.cpp:550-554`) — and both handles are stored in `shared_variables.cb_handles`
(`...program_factory.cpp:854`) and re-pointed with `UpdateDynamicCircularBufferAddress` on every hit
(`...program_factory.cpp:887-888`). The Q/K/V base addresses appear in reader slots 10/11/12 and
writer slots 8/9/10 at build time (`...program_factory.cpp:821-830`) and are rewritten at
`...program_factory.cpp:893-895` and `...program_factory.cpp:899-901`. The slot indices match the
ones the factory assigned (`...program_factory.cpp:732-744`).

This is the complete-override case that the all-gather sibling op gets wrong; here it is right.

### 7. `input_tensor.logical_shape()`

**Verdict: CAVEAT.**

The logical shape is genuinely read and genuinely structural:

```347:364:ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter_create_heads/device/llama_reduce_scatter_create_heads_program_factory.cpp
    auto input_tensor_width = input_tensor.logical_shape()[-1];
    auto input_shard_spec = input_tensor.shard_spec().value();
    auto q_output_shard_spec = q_output_tensor.shard_spec().value();
    auto k_output_shard_spec = k_output_tensor.shard_spec().value();
    auto v_output_shard_spec = v_output_tensor.shard_spec().value();
    const auto q_output_grid = q_output_shard_spec.grid;
    const auto k_output_grid = k_output_shard_spec.grid;
    const auto v_output_grid = v_output_shard_spec.grid;
    const auto& cross_device_semaphore = operation_attributes.cross_device_semaphore;

    uint32_t input_shard_width = input_shard_spec.shape[1];

    uint32_t ncores_input = (input_tensor_width + input_shard_width - 1) / input_shard_width;

    // uint32_t input_shard_cores_per_device = ncores_input / num_devices;
    uint32_t input_sticks_per_device = input_shape[-2] / num_devices;  // should be 8
    uint32_t input_blocks_per_stick = ncores_input;                    // should be 20
```

`ncores_input` is reader compile-time arg 9 and writer compile-time arg 8
(`...program_factory.cpp:628`, `...program_factory.cpp:682`), it drives the work `schedule`
(`...program_factory.cpp:449-450`) which is itself baked in as the `SCHEDULE` define
(`...program_factory.cpp:652`), and `input_sticks_per_device` feeds `input_block_size` which sizes
four circular buffers.

It is nevertheless pinned in the normal case, and the argument is worth spelling out. The two
components that are used are recoverable from hashed attributes, because the ttnn-level wrapper
derives `head_dim` and `slice_size` from the input's padded shape:

```35:36:ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter_create_heads/llama_reduce_scatter_create_heads.cpp
    uint32_t head_dim = input_tensor.padded_shape()[-1] / (num_heads + 2 * num_kv_heads);
    uint32_t slice_size = input_tensor.padded_shape()[-2] / ring_devices;
```

`head_dim`, `num_heads`, `num_kv_heads`, `slice_size` and `ring_devices` are all hashed, so
`padded_shape[-1]` and `padded_shape[-2]` are recoverable from the key, and `input_shard_width` comes
from the hashed `memory_config`.

The caveat is the gap between `logical_shape` and `padded_shape`. The factory reads the *logical*
width; the hash pins the *padded* one. For a tile-layout tensor whose width is a multiple of 32 they
are equal, but the op does not check that. Two calls with logical widths 3550 and 3584, both padding
to 3584 and both carrying the same shard spec, produce the same key and different `ncores_input`,
hence a different `SCHEDULE` define and different compile-time args. `input_shape[0]` is also
unhashed, but it only reaches the output `TensorSpec`
(`device/llama_reduce_scatter_create_heads_device_op.cpp:64`), which is recomputed every call.

What would close it: hash `input_tensor.logical_shape()` (one extra term), or `TT_FATAL` that
`logical_shape() == padded_shape()`.

### 8. `input_tensor.page_config` / `layout()`

**Verdict on the tile geometry: VALID — unused. Verdict on `layout()`: CAVEAT.**

**`page_config()` is confirmed absent from this hash.** The input tensor contributes exactly three
terms, and neither the page config, the tensor spec nor the layout is among them:

```140:142:ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter_create_heads/device/llama_reduce_scatter_create_heads_device_op.cpp
        tensor_args.input_tensor.dtype(),
        tensor_args.input_tensor.memory_config(),
        tensor_args.input_tensor.device()->id());
```

So the `Tile` is not in the key. Unlike the sibling ops that share this property, that costs nothing
here, because nothing in the program depends on it. The two halves of the omission adjudicate
differently.

**Tile geometry — VALID — unused.** The unguarded-32x32 check was performed against this op and
found not to apply: there is no host-side tile-geometry arithmetic anywhere in the op directory. The
factory works purely in bytes, deriving its block size from the shard geometry and the element size:

```396:398:ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter_create_heads/device/llama_reduce_scatter_create_heads_program_factory.cpp
    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    uint32_t input_block_size = input_sticks_per_device * input_shard_width * input_tensor.element_size();
```

`input_block_size` is the page size of every circular buffer in this program
(`...program_factory.cpp:468`, `:511`, `:553`, `:569`) and a compile-time arg of both dataflow kernels
(`...program_factory.cpp:630`, `:684`), and none of it involves a tile count: `input_sticks_per_device`
is `input_shape[-2] / num_devices` (`...program_factory.cpp:362`), the work schedule is built with a
literal `1` in the tiles-per-core slot (`...program_factory.cpp:449-450`), and the kernels' two
tiles-per-core compile-time args are the literals `1, 1` (`...program_factory.cpp:625-626`).

A search of this op's own directory for `tt::tile_size`, `TILE_HW`, `TILE_WIDTH`, `TILE_HEIGHT`,
`get_tile_shape`, `get_tile_size` and `tensor_spec().tile()` returns nothing in the host-side
sources, so neither the hardcoded-32x32 idiom nor its tile-aware mirror image is present here — and
no value the program consumes carries any information about `page_config`. Note also that this op
does *not* use the shared `experimental/ccl/llama_reduce_scatter/` factory, which is where its
sibling `llama_reduce_scatter_matmul` inherits its tile problems from.

**Two shared helpers reached from this factory were checked separately, since a directory-scoped
search would miss them.** `llama_specific::get_custom_cores`
(`...program_factory.cpp:442`) is tile-free by construction — it takes no tensor argument and returns
two hardcoded core ranges:

```9:11:ttnn/cpp/ttnn/operations/experimental/ccl/llama_common.cpp
CoreRangeSet get_custom_cores(uint32_t num_workers, bool row_wise) {
    CoreRangeSet worker_cores;
    std::vector<CoreRange> desired_core_range = {CoreRange({5, 3}, {6, 3}), CoreRange({2, 8}, {3, 8})};
```

The other one does read the real tile, and is the reason this subsection needs the hazard note below.

**Latent hazard: a dead `CCLOpConfig` sits one dereference away from making this a BUG.** The factory
constructs a shared CCL config object and immediately discards it:

```404:405:ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter_create_heads/device/llama_reduce_scatter_create_heads_program_factory.cpp
    [[maybe_unused]] const auto& op_config =
        ttnn::ccl::CCLOpConfig(input_tensors, output_tensors, operation_attributes.topology);
```

That constructor is emphatically tile-dependent:

```66:72:ttnn/cpp/ttnn/operations/ccl/ccl_host_datastructures.cpp
    if (input_tensors.at(0).layout() == Layout::TILE) {
        this->tile = input_tensors.at(0).tensor_spec().tile();
        this->page_size = this->tile.get_tile_size(this->df);
        // this->page_size = input_tensors.at(0).buffer()->page_size();
    } else {
        this->page_size = input_tensors.at(0).buffer()->page_size();
    }
```

The **VALID — unused** verdict survives, and for the strongest possible reason: `op_config` appears
exactly once in the entire op tree — the declaration above — and nothing is ever read from it, so no
tile-derived value reaches a compile-time arg, a CB, or a core range. A search for `op_config` across
`llama_reduce_scatter_create_heads/` returns that single line.

But the margin is one line of code, and the failure mode if it is crossed is severe. The moment
somebody writes `op_config.get_page_size()` (`ccl_host_datastructures.cpp:75`) and uses it in place of
the hand-rolled `input_block_size`, this op acquires a genuine aliasing bug with no other change: the
hash contains no `page_config()`, no `tile()` and no shape at all — only the three terms quoted at the
top of this subsection — so two calls differing solely in `Tile` would share a cache entry whose CB
page sizes were computed for the first tile. There would be no compiler warning either, because
`[[maybe_unused]]` exists precisely to suppress the diagnostic that would otherwise flag the dead
object.

This is a latent hazard rather than a current defect, and it is deliberately **not** counted among
this op's bugs. It is recorded because the usual signal that an op is tile-sensitive — the factory
reading `tensor_spec().tile()` — is already present in the call graph, just not yet load-bearing.

**`layout()` — CAVEAT.** The reduction is a tile-domain compute kernel
(`device/kernels/compute/reduction.cpp`, created at `...program_factory.cpp:713-717`), so a
`ROW_MAJOR` input would be reduced incorrectly. Nothing validates the layout, and the layout is not
in the key, so a `ROW_MAJOR` call would silently inherit the `TILE` program rather than at least
building its own. The shard-height check

```28:31:ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter_create_heads/device/llama_reduce_scatter_create_heads_device_op.cpp
    TT_FATAL(
        input_tensor.shard_spec().value().shape[0] == 32,
        "input_tensor shard height must be 32 but got {}",
        input_tensor.shard_spec().value().shape[0]);
```

constrains the shard geometry but says nothing about layout. This is a validation gap rather than a
hash gap; a `TT_FATAL(input_tensor.layout() == Layout::TILE)` converts it to
"VALID — pinned by validation".

### 9. `input_tensor` alignment and storage kind

**Verdict: VALID — unused** (alignment); **VALID — pinned by validation** (storage kind).

Alignment reaches the program only through the buffer page size, which is determined by the hashed
`{memory_config, dtype}` plus the shape discussed in #7. Storage kind is pinned: the miss validator
requires a shard spec (`device/llama_reduce_scatter_create_heads_device_op.cpp:27`), which a host
tensor does not have, and the factory dereferences `input_tensor.buffer()`
(`...program_factory.cpp:370`).

### 10. `ring_index`, `device_order`, forward/backward fabric neighbours

**Verdict: VALID — invariant** (these are determined by the mesh coordinates the framework appends
to every key, plus hashed attributes).

`ring_index` is derived by locating the target device in the row or column selected by
`cluster_axis`:

```311:335:ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter_create_heads/device/llama_reduce_scatter_create_heads_program_factory.cpp
    std::vector<IDevice*> devices = (operation_attributes.cluster_axis == 0)
                                        ? mesh_view.get_devices_on_column(mesh_coordinate[1])
                                        : mesh_view.get_devices_on_row(mesh_coordinate[0]);
    const auto fabric_node_ids = (operation_attributes.cluster_axis == 0)
                                     ? mesh_view.get_fabric_node_ids_on_column(mesh_coordinate[1])
                                     : mesh_view.get_fabric_node_ids_on_row(mesh_coordinate[0]);

    std::optional<tt::tt_fabric::FabricNodeId> forward_fabric_node_id = std::nullopt;
    std::optional<tt::tt_fabric::FabricNodeId> backward_fabric_node_id = std::nullopt;
    for (uint32_t i = 0; i < ring_size; ++i) {
        if (devices.at(i) == target_device) {
            ring_index = i;
            if (i != 0) {
                backward_fabric_node_id = fabric_node_ids.at(i - 1);
            } else if (operation_attributes.topology == ttnn::ccl::Topology::Ring) {
                backward_fabric_node_id = fabric_node_ids.at(ring_size - 1);
            }

            if (i != ring_size - 1) {
                forward_fabric_node_id = fabric_node_ids.at(i + 1);
            } else if (operation_attributes.topology == ttnn::ccl::Topology::Ring) {
                forward_fabric_node_id = fabric_node_ids.at(0);
            }
        }
    }
```

The inputs are `mesh_coordinate`, `cluster_axis`, `ring_devices` and `topology`. The last three are
hashed explicitly. The first is not an omission at all — the framework folds the tensors' mesh
coordinates into the key for both the default and the custom path:

```989:992:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
        // Combine with the mesh coordinates the workload is targeting.
        for (const auto& coord : mesh_device_operation_utils::extract_tensor_coordinates(tensor_args, mesh_device)) {
            hash = ttsl::hash::hash_objects(hash, coord);
        }
```

So `chip_id` (reader compile arg 5, `...program_factory.cpp:624`), the `DEVICE_ORDER` define
(`...program_factory.cpp:338-341`) and the forward/backward connection flags
(`...program_factory.cpp:752-759`) are all keyed. This op additionally hashes
`input_tensor.device()->id()`, which pins the mesh device itself — see the next section.

### 11. Fabric connection runtime args

**Verdict: CAVEAT.**

```791:813:ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter_create_heads/device/llama_reduce_scatter_create_heads_program_factory.cpp
            writer_runtime_args.push_back(forward_fabric_connection);
            if (forward_fabric_connection) {
                const auto target_device_fabric_node_id = mesh_device->get_fabric_node_id(mesh_coordinate);
                tt::tt_fabric::append_fabric_connection_rt_args(
                    target_device_fabric_node_id,
                    forward_fabric_node_id.value(),
                    link_idx,
                    program,
                    core,
                    writer_runtime_args);
            }

            writer_runtime_args.push_back(backward_fabric_connection);
            if (backward_fabric_connection) {
                const auto target_device_fabric_node_id = mesh_device->get_fabric_node_id(mesh_coordinate);
                tt::tt_fabric::append_fabric_connection_rt_args(
                    target_device_fabric_node_id,
                    backward_fabric_node_id.value(),
                    link_idx,
                    program,
                    core,
                    writer_runtime_args);
            }
```

These tail slots hold EDM router coordinates, buffer bases and flow-control semaphore addresses.
`override_runtime_arguments` rewrites only slots 0 and 8-10 of the writer args, so the fabric tail is
frozen at first-miss values. Safe as long as the fabric configuration is fixed for the lifetime of
the mesh device, which it is today; it would break on a fabric teardown/re-init between two calls
that share a cache entry. Same assumption as every fabric CCL op in the tree, and `num_links` (which
selects the EDM channel via `link_idx`) is hashed, closing the most likely variant.

## Keys the custom hash adds beyond the default

`tensor_args.input_tensor.device()->id()`. The default reflection hash does not include the device id
(a `Tensor`'s `DeviceStorage` has an empty attribute tuple, so the buffer and its device never reach
the key); the framework appends only mesh *coordinates*, which are relative to a mesh and do not
identify which mesh. Adding the id is defensive and costs nothing — it makes the entry
non-transferable between two `MeshDevice`s with the same coordinate layout. Note it does *not*
substitute for hashing `subdevice_id` (#3), which partitions cores within one device.

## Framework side effect of having a custom hash

```1012:1014:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
        if constexpr (requires { DeviceOperation::compute_program_hash(attrs, tensor_args); }) {
            return key;  // custom hash -> opt out beyond the op-identity prefix
        } else {
```

`ProgramCacheKey::canonical` degrades to the op type name, so a 64-bit collision between two distinct
configurations resolves to a wrong hit instead of a rebuild. Inherent to every custom-hash op.

## Summary

| Omitted vs. default | Used by program? | Patched on hit? | Verdict |
|---|---|---|---|
| `qkv_memory_config` | Yes — Q/K/V core lists baked in as kernel defines | No | **BUG** |
| `intermediate_packet_buffer` spec (grid) | Yes — packet-worker core range, compute-kernel placement, CB ranges, mcast compile args | No | **BUG** |
| `subdevice_id` | Yes, when `use_optimal_ccl_for_llama == false` — sender core pool | No | **BUG** (that branch only) |
| `cross_device_semaphore` address | Yes — reader/writer slot 0 | Yes | VALID — patched |
| `output_mem_config` | No — dead field | n/a | VALID — unused |
| Buffer addresses (input CB, packet CB, Q/K/V) | Yes | Yes | VALID — patched |
| `input.logical_shape` | Yes — `ncores_input`, `SCHEDULE` define, CB sizes | No | CAVEAT — pinned only while logical == padded |
| `input.page_config` (tile geometry) | No — the factory is byte-based, no host-side tile math | n/a | VALID — unused |
| `input.layout` (`TILE` vs `ROW_MAJOR`) | Yes, implicitly (tile-domain compute kernel) | No | CAVEAT — add a layout guard |
| `input.alignment`, storage kind | Only via hashed derivatives | n/a | VALID — unused (alignment) / VALID — pinned by validation (storage) |
| `ring_index`, `DEVICE_ORDER`, fabric neighbours | Yes — compile args and defines | n/a | VALID — invariant (keyed via the mesh coordinates the framework appends) |
| Fabric connection rt args | Yes | No | CAVEAT — relies on fixed fabric config |

**Program-cache bugs were found.** Three, all of the same shape: a caller-controlled parameter that
determines *which cores* the program is built on, absent from the key and unrecoverable on a hit.
`qkv_memory_config` (#1) is the worst because the core list is a preprocessor define; the packet
buffer's grid (#2) and `subdevice_id` (#3) are the same class. Everything in the address family is
handled correctly — this op's `override_runtime_arguments` is complete for what it covers, including
both globally-allocated circular buffers, which is more than several of its siblings manage. The
op is safe in the Llama model because the model allocates the packet buffer and the QKV sub-core grid
once and reuses them; it is not safe against the public API surface.

The count stays at three after the unguarded-32x32 tile check: this op is one of the few in the CCL
family whose factory contains no host-side tile arithmetic at all, so dropping `page_config` costs it
nothing (#8). It stays at three despite the dead `CCLOpConfig` documented in #8 as well — that is a
latent hazard, not a defect, since nothing reads the object. It is worth fixing precisely because the
distance between "clean" and "aliasing bug" is one dereference.

It also stays at three after the empty hit validator is filtered for reachability. That override
suppresses ten `TT_FATAL`s on the hit path, but six of them constrain hash terms and are therefore
self-enforcing, two are supplied by the framework's own device-tensor checks on every dispatch, and
the remainder are either unreachable through the public API, pre-empted by an unconditional
dereference in `compute_output_specs`, or constrain a value the factory never reads. None of the
three bugs is caused or widened by it: each is a hash omission on a value the *miss* validator does
not check either.

## Recommendations

1. Hash `attributes.qkv_memory_config`. It is a plain `std::optional<MemoryConfig>` and adds one term.
2. Hash the packet buffer's shard grid, e.g.
   `tensor_args.intermediate_packet_buffer.memory_config()`. Hashing the full memory config is the
   cheapest correct option and also covers its buffer type.
3. Hash `attributes.subdevice_id`, or `TT_FATAL` in `validate_on_program_cache_miss` that it equals
   `mesh_device->get_sub_device_ids().at(0)` whenever `use_optimal_ccl_for_llama` is false.
4. Add `input_tensor.logical_shape()` to the hash, or `TT_FATAL` that logical == padded, closing #7.
5. Add `TT_FATAL(input_tensor.layout() == Layout::TILE)`, closing the `layout()` half of #8 and
   documenting an assumption the compute kernel already relies on. A tile-geometry guard is *not*
   needed here — unlike the sibling CCL ops, this factory never converts a shape into a tile count —
   though adding one alongside would be harmless and would keep the family consistent.
6. **Do not** move miss-validator checks into `validate_on_program_cache_hit`, and do not delete the
   empty override to restore the whole miss validator on hits. This is a deliberate
   non-recommendation, not an oversight. Six of the miss validator's ten checks constrain values that
   are hash terms, so they are self-enforcing and cost nothing by being absent; of the four that are
   reachable, one is unreachable through the public API and faults rather than corrupts, two are
   closed for free by recommendation 2 (hashing the packet buffer's `memory_config` puts its shard
   spec in the key), and the last constrains a shard height the factory never reads. Deleting the
   override would put nine scalar comparisons plus four `std::optional` and `shard_spec()` accesses
   on every dispatch of a hot CCL op to buy nothing. The empty body is worth a comment recording
   *why* it is empty — the analysis above — rather than a change.
7. Remove the dead `output_mem_config` attribute, or wire it up. Carrying an unread public argument
   invites a future caller to assume it works.
8. Delete the dead `[[maybe_unused]] op_config` construction at
   `...program_factory.cpp:404-405`, closing the latent hazard in #8. Nothing reads it, and its
   constructor is the only tile-dependent code reachable from this factory. If it is instead a
   deliberate placeholder for planned work, leave it but add `input_tensor.tensor_spec().page_config()`
   to `compute_program_hash` in the same commit — otherwise the first use of `op_config.get_page_size()`
   silently turns a correct op into an aliasing bug, with `[[maybe_unused]]` suppressing the one
   diagnostic that would have drawn attention to the object.
