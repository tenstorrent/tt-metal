# Program Cache Audit — `experimental/test/prefetcher_consumer`

This directory holds **two** device operations, not one, and they share the same hashing strategy
and the same failure mode:

- `DramPrefetcherConsumerDeviceOperation` (`dram_prefetcher_consumer.{hpp,cpp}`) — the op the CSV
  row names.
- `DramPrefetcherValidatorDeviceOperation` (`dram_prefetcher_validator.{hpp,cpp}`) — bound from the
  same nanobind translation unit (`dram_prefetcher_consumer_nanobind.cpp:40-78`).

Both are audited below. The consumer is the primary subject; the validator gets its own section
because its hash makes an even more aggressive substitution.

| | Consumer | Validator |
|---|---|---|
| Device operation | `DramPrefetcherConsumerDeviceOperation` (`dram_prefetcher_consumer.hpp:23`) | `DramPrefetcherValidatorDeviceOperation` (`dram_prefetcher_validator.hpp:25`) |
| Custom hash | `dram_prefetcher_consumer.cpp:46-55` | `dram_prefetcher_validator.cpp:57-73` |
| `operation_attributes_t` | `num_iters`, `page_size_bytes`, `global_cb`, `mesh_device` | `num_layers`, `print_stride`, `global_cb`, `streaming`, `rotation` |
| `tensor_args_t` | **empty struct** (`dram_prefetcher_consumer.hpp:33`) | `source_tensor` |
| Program factory | `ProgramFactory` (`create_at` → `CachedProgram`) | `ProgramFactory` (`create_at` → `CachedProgram`) |
| `override_runtime_arguments` | Present but an **empty no-op** (`dram_prefetcher_consumer.cpp:89-95`) | Present but an **empty no-op** (`dram_prefetcher_validator.cpp:250-256`) |
| `get_dynamic_runtime_args` | No | No |
| `validate_on_program_cache_hit` | Present but an **empty no-op** (`dram_prefetcher_consumer.cpp:33-34`) | Present but an **empty no-op** (`dram_prefetcher_validator.cpp:44-45`) |
| Cache-hit patch mechanism | **Op-owned re-derivation (mode A) with an empty body — nothing is refreshed** | Same |

**Result: one BUG in the consumer, three in the validator.** All four share a root cause — an
allocation address used as an identity token — and all four are made unrecoverable by the empty
`override_runtime_arguments`, which means nothing at all is refreshed on a cache hit. The *second*
empty hook, `validate_on_program_cache_hit`, matters much less than a raw diff of the two validators
suggests, and it matters asymmetrically between the two ops; the next section works that out check by
check rather than asserting it.

## Where the CSV classification does not match the code

- **`tensor_input = SELECTIVE` is wrong for the consumer.** `tensor_args_t` is an empty struct; the
  op takes no tensors at all. The correct classification is `TENSORS-ABSENT`. The label is accurate
  for the validator, which does hash a strict subset of one tensor.
- **`own_hit_validator = Y` is technically true but inverted in meaning for both.** See the next
  section — an empty hit validator does not add nothing, it actively suppresses the miss validator.
- **`override_runtime_arguments = Y` is technically true but misleading for both.** The hooks exist
  and are empty. See "Cache-hit patch mechanism" below.

## What the empty hit validators actually suppress

The dispatcher runs exactly one validator on a hit, and defining the hook *replaces* the miss
validator rather than supplementing it:

```262:266:ttnn/api/ttnn/device_operation.hpp
    if constexpr (HasValidateOnProgramCacheHit<mesh_device_operation_t>) {
        mesh_device_operation_t::validate_on_program_cache_hit(operation_attributes, tensor_args);
    } else {
        mesh_device_operation_t::validate_on_program_cache_miss(operation_attributes, tensor_args);
    }
```

Both ops define the hook with an empty body
(`dram_prefetcher_consumer.cpp:33-34`, `dram_prefetcher_validator.cpp:44-45`). Had they simply not
defined it, the framework would have re-run every miss-time `TT_FATAL` on every dispatch.

That does **not** mean every dropped check is a hazard. A miss-only pin on a value that is itself in
the cache key cannot be evaded: a call carrying a new value of a hashed parameter computes a
different key, misses, and the miss path unconditionally runs the miss validator before building
anything.

```301:301:ttnn/api/ttnn/device_operation.hpp
    mesh_device_operation_t::validate_on_program_cache_miss(operation_attributes, tensor_args);
```

Only a check constraining a value **absent** from the key can be reached on a hit. Both validators
are worked through line by line below on that basis, and the answer is very different for the two
ops.

### Consumer — one of five dropped checks is reachable

Hashed: `num_iters`, `page_size_bytes`, `global_cb->config_address()`
(`dram_prefetcher_consumer.cpp:46-55`).

| Dropped check | Value it constrains | In the key? | Reachable on a hit? |
|---|---|---|---|
| `attrs.mesh_device != nullptr` (`:26`) | `attrs.mesh_device` | No | No — a null pointer faults in `launch` at `device_operation.hpp:474` (`mesh_device->get_view()`), before either validator on either path |
| `attrs.num_iters > 0` (`:27`) | `attrs.num_iters` | Yes — hash term 1 | No |
| `attrs.page_size_bytes > 0` (`:28`) | `attrs.page_size_bytes` | Yes — hash term 2 | No |
| `attrs.global_cb.has_value()` (`:29`) | engagement of the GCB optional | No | No — `compute_program_hash` dereferences it at `:54` before any validator runs, on both paths |
| `attrs.global_cb->receiver_cores().num_cores() > 0` (`:30`) | non-emptiness of the GCB receiver core set | **No** — only `config_address()` is hashed | **Yes** |

Both scalar attributes are self-enforcing: a call with `num_iters = 0` or `page_size_bytes = 0` lands
on its own key, misses, and is rejected on the miss path. Two more rows are dead as guards on either
path, because the values they test are used — and would fault — earlier in the same dispatch than
any validator runs.

**One row is reachable, and it is weaker than it looks.** `receiver_cores().num_cores() > 0` tests
only *non-emptiness*, not identity. In omission 1's reproduction, `gcb_b` has four receiver cores, so
this check passes; restoring it on the hit path would not reject that call. The empty hit validator
is therefore **not** load-bearing for the consumer's bug — the bug is a pure hash omission, and it
would be equally reachable if the hook did not exist. What the missing check does cover is the
degenerate case of a hit whose GCB has *no* receivers at all, which fails silently (the cached
program simply keeps executing on call 1's cores).

### Validator — almost every dropped check is reachable

Hashed: `num_layers`, `print_stride`, `streaming`, `rotation`, `global_cb->config_address()`, the
source buffer's address, and its dataformat (`dram_prefetcher_validator.cpp:57-73`). Because that set
carries nothing structural about either the GCB or the tensor spec, the filter removes almost
nothing here. Checks inside `create_at` are listed alongside the validator's own, since `create_at`
runs only on a miss and the effect on the hit path is identical.

| Check absent on the hit path | Where it lives | Value it constrains | In the key? | Reachable on a hit? |
|---|---|---|---|---|
| `attrs.num_layers > 0` | miss validator, `:33` | `attrs.num_layers` | Yes — hash term 1 | No |
| `tensor_buffer != nullptr` | miss validator, `:34-35` | source tensor storage kind | Effectively yes — the hash substitutes `0` for a null buffer (`:71`), and no device allocation sits at address 0; the framework also rejects a host tensor at `device_operation.hpp:455` on both paths | No |
| `tensor_buffer->is_dram()` | miss validator, `:36` | the source buffer's `buffer_type` | **No** — only the numeric address is hashed | **Yes** |
| `attrs.global_cb.has_value()` | miss validator, `:37` | engagement of the GCB optional | No | No — dereferenced by `compute_program_hash` at `:70` before any validator |
| `receiver_cores().num_cores() > 0` | miss validator, `:38` | non-emptiness of the receiver core set | **No** | **Yes** |
| `!sr_mapping.empty()` | miss validator, `:40-41` | non-emptiness of the sender/receiver mapping | **No** | **Yes** |
| `num_blocks % num_dram_banks == 0` | `create_at`, `:97-101` | GCB mapping shape | **No** | **Yes** |
| `padded_shape.rank() >= 2` | `create_at`, `:112-115` | `source_tensor.padded_shape()` | **No** | **Yes** |
| `K_elems % tile_h == 0 && N_elems % tile_w == 0` | `create_at`, `:122-128` | `padded_shape` against `Tile` | **No** — neither is hashed | **Yes** |
| `k_tiles % num_blocks == 0` | `create_at`, `:131-132` | `padded_shape`, `Tile`, GCB mapping | **No** | **Yes** |
| `total_n_tiles % ring_size == 0` | `create_at`, `:134-138` | same | **No** | **Yes** |
| `bank_local_recv < receivers_per_bank` | `create_at`, `:205-211` | GCB mapping shape | **No** | **Yes** |
| `ring_pos < rotation.size()` | `create_at`, `:226-231` | `rotation` against the GCB mapping | `rotation` yes, mapping no | **Yes** |

Only three of thirteen are filtered out. That is the diagnostic signature of a hash that keys on
allocation addresses instead of structure: because neither the GCB's mapping nor the tensor's spec is
in the key, virtually nothing the validator checks is self-enforcing. Note that only the first six
rows are attributable to the empty hook — the seven `create_at` rows were never on the hit path under
any hook, since `create_at` runs only when a program is built.

The `is_dram()` row deserves its own reproduction, because it turns the address-as-identity choice
into a bug even when nothing is reallocated. DRAM and L1 are separate address spaces that both start
near zero, so an L1 buffer and a DRAM buffer can hold the *same numeric address* simultaneously.
Call 1 with a DRAM source tensor at address `A` compiles the accessor with `IsDram` set; call 2 with
an L1 source tensor that happens to sit at address `A` and has the same dtype produces an identical
hash, hits, and is not rejected because `is_dram()` no longer runs. The kernel then resolves an L1
offset through the DRAM bank map. This is the one dropped check whose restoration closes a silent
hole outright rather than merely improving a diagnostic.

### What the filter changes, and what it does not

No verdict below moves. All four bugs are hash omissions on values that the *miss* validator does not
constrain either — a different receiver core set, a different tensor shape, a different `Tile` all
pass every `TT_FATAL` in both ops — so restoring the miss validator on the hit path would not reject
a single one of the reproductions. The empty `validate_on_program_cache_hit` overrides are a real but
secondary defect: between them they suppress eleven checks on the hit path, of which four are
reachable — one degenerate-case guard on the consumer and three on the validator — and of those four
exactly one, `is_dram()`, closes a silent failure rather than a degenerate one.

Note finally that both ops define a custom `compute_program_hash`, so the canonical half of the cache
key degrades to the op-identity prefix (see "Framework side effect" below). The rows marked
unreachable above are therefore unreachable *absent a hash collision*, not unconditionally.

## Cache-hit patch mechanism

Both factories satisfy `MeshWorkloadFactoryConcept` via `HasCreateAt`
(`ttnn/api/ttnn/operation_concepts.hpp:46-54`), so the framework dispatches straight to the
factory's own `override_runtime_arguments` on every hit — no `resolve_bindings`, no
`get_dynamic_runtime_args`, no descriptor rebuild:

```279:285:ttnn/api/ttnn/device_operation.hpp
        if constexpr (requires { &WorkloadFactory::apply_descriptor; }) {
            WorkloadFactory::apply_descriptor(
                cached_mesh_workload, operation_attributes, tensor_args, tensor_return_value);
        } else {
            WorkloadFactory::override_runtime_arguments(
                cached_mesh_workload, operation_attributes, tensor_args, tensor_return_value);
        }
```

This is the strongest cache-hit mode the framework offers — the op is trusted to re-derive *all*
per-dispatch state. Both ops decline to:

```89:95:ttnn/cpp/ttnn/operations/experimental/test/prefetcher_consumer/dram_prefetcher_consumer.cpp
void DramPrefetcherConsumerDeviceOperation::ProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& /*cached_workload*/,
    const operation_attributes_t& /*attrs*/,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& /*tensor_return_value*/) {
    // Nothing to override — all args are compile-time.
}
```

**Obligation on the hash.** Because the override is empty, *nothing whatsoever* is refreshed on a
cache hit. The cached `Program` is enqueued exactly as it was built on the first miss: same kernel
binaries, same compile-time args, same core ranges, same circular-buffer configuration, same
globally-allocated CB base addresses, same runtime args. The hash must therefore be a complete key
over every input that influences any of those. For the consumer that is a stronger obligation than
for a normal op, because the *core placement itself* comes from a hashed-away attribute.

For the consumer the claim in the comment is at least self-consistent — `create_at` calls no
`SetRuntimeArgs` at all, so there genuinely are no runtime args to refresh. The problem is not the
runtime args; it is everything else the program is made of.

## Baseline: what the default hash would cover

### Consumer

`hash_objects_with_default_seed(type_hash<DramPrefetcherConsumerDeviceOperation>, attrs,
tensor_args)` would cover:

| Source | Fields |
|---|---|
| `attrs.num_iters` | the value |
| `attrs.page_size_bytes` | the value |
| `attrs.global_cb` | engaged/disengaged, and if engaged `sender_receiver_core_mapping`, `size`, `buffer_type` |
| `attrs.mesh_device` | the raw pointer value |
| `tensor_args` | nothing — the struct is empty |

The `global_cb` row deserves emphasis, because the code comment justifying the custom hash asserts
the opposite:

```46:55:ttnn/cpp/ttnn/operations/experimental/test/prefetcher_consumer/dram_prefetcher_consumer.cpp
ttsl::hash::hash_t DramPrefetcherConsumerDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& /*tensor_args*/) {
    // GlobalCircularBuffer isn't reflection-hashable; hash its identity via config_address
    // (unique per GCB instance on this device) along with the other attrs.
    return ttsl::hash::hash_objects_with_default_seed(
        ttsl::hash::type_hash<DramPrefetcherConsumerDeviceOperation>,
        attrs.num_iters,
        attrs.page_size_bytes,
        static_cast<uint64_t>(attrs.global_cb->config_address()));
}
```

**`GlobalCircularBuffer` *is* hashable.** It carries a reflection attribute pair *and* a
`std::hash` specialization:

```59:64:tt_metal/api/tt-metalium/global_circular_buffer.hpp
    static constexpr auto attribute_names =
        std::forward_as_tuple("sender_receiver_core_mapping", "size", "buffer_type");
    auto attribute_values() const {
        return std::make_tuple(
            this->sender_receiver_core_mapping_, this->size_, cb_buffer_.get_buffer()->buffer_type());
    }
```

```608:611:tt_metal/impl/buffers/global_circular_buffer.cpp
std::size_t hash<tt::tt_metal::experimental::GlobalCircularBuffer>::operator()(
    const tt::tt_metal::experimental::GlobalCircularBuffer& global_circular_buffer) const {
    return ttsl::hash::hash_objects_with_default_seed(global_circular_buffer.attribute_values());
}
```

and `ttsl::hash::hash_object` reaches the `std::hash` specialization before it would ever fall
through to a static assertion:

```1303:1314:tt_stl/tt_stl/reflection.hpp
inline hash_t hash_object(const T& object) noexcept {
    if constexpr (std::numeric_limits<T>::is_integer) {
        if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
            fmt::print("Hashing integer of type {}: {}\n", get_type_name<T>(), object);
        }
        return object;
    } else if constexpr (detail::is_std_hashable_v<T>) {
        if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
            fmt::print("Hashing {} using std::hash: {}\n", get_type_name<T>(), object);
        }
        return std::hash<T>{}(object);
    } else if constexpr (ttsl::reflection::detail::supports_to_hash_v<T>) {
```

So the premise of the custom hash is false: the default key would have covered
`sender_receiver_core_mapping`, `size` and `buffer_type` by value, which is precisely the
information the custom hash discards. The custom hash is strictly weaker than the default here.

## What the custom hash covers

Consumer: `num_iters`, `page_size_bytes`, and `global_cb->config_address()`.

## Omitted parameters — consumer

### 1. `attrs.global_cb` — everything except `config_address()`

**Verdict: BUG.**

`config_address()` is not a stable identity. It is the base address of an ordinary L1 sharded
buffer handed out by the device allocator:

```423:423:tt_metal/impl/buffers/global_circular_buffer.cpp
DeviceAddr GlobalCircularBuffer::config_address() const { return cb_config_buffer_.get_buffer()->address(); }
```

```326:334:tt_metal/impl/buffers/global_circular_buffer.cpp
    ShardedBufferConfig cb_config_buffer_shard_config = {
        .device = device_,
        .size = cb_config_size,
        .page_size = cb_config_page_size,
        .buffer_type = buffer_type,
        .buffer_layout = TensorMemoryLayout::HEIGHT_SHARDED,
        .shard_parameters = std::move(shard_parameters),
    };
    cb_config_buffer_ = distributed::AnyBuffer::create(cb_config_buffer_shard_config);
```

The address is unique only among *simultaneously live* GCBs. Once a GCB is destroyed the allocation
is returned, and a subsequent same-sized allocation from the same allocator state receives the same
address. Note that `cb_config_page_size` is a function of `max_num_receivers_per_sender` and
`num_cores` only, so two GCBs with the same *shape* (same core count, same receivers per sender) but
different core *positions* allocate config buffers of identical size — the case most likely to
recycle an address, and also the case where the program differs most.

Meanwhile, the GCB determines nearly the whole program:

```66:84:ttnn/cpp/ttnn/operations/experimental/test/prefetcher_consumer/dram_prefetcher_consumer.cpp
    const auto& global_cb = operation_attributes.global_cb.value();
    const CoreRangeSet receiver_cores = global_cb.receiver_cores();

    // Configure the receiver-side CB. set_page_size matches what the sender resizes the CB to
    // (in_block_w_tiles * n_tiles_per_recv * tile_bytes); receiver wait_front/pop_front operate
    // in units of this page size.
    CircularBufferConfig cb_config(operation_attributes.page_size_bytes);
    cb_config.remote_index(kRemoteCBId)
        .set_page_size(operation_attributes.page_size_bytes)
        .set_data_format(tt::DataFormat::Float16_b);
    tt::tt_metal::experimental::CreateCircularBuffer(program, receiver_cores, cb_config, global_cb);

    const std::vector<uint32_t> compile_args = {kRemoteCBId, operation_attributes.num_iters};
    CreateKernel(
        program,
        "tests/tt_metal/tt_metal/test_kernels/misc/gcb_bench_discard_receiver.cpp",
        receiver_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0, .compile_args = compile_args});
```

`receiver_cores` (derived from `sender_receiver_core_mapping`) picks the kernel's core range *and*
the CB's core range, and the `global_cb` overload of `CreateCircularBuffer` pegs the remote CB to
that GCB's `buffer_address()` and config layout. Core ranges and CB configuration are baked into
the `Program`; no cache-hit mode in the framework rebuilds them, and this op's override is empty.

**Two-call reproduction** (Python, via `ttnn.experimental.test_dram_prefetcher_consumer`):

- **Call 1**: `gcb_a = CreateGlobalCircularBuffer(dev, [(CoreCoord(0,0), CoreRangeSet(CoreRange((1,0),(1,3))))], size=S, L1)`;
  then `test_dram_prefetcher_consumer(dev, num_iters=100, page_size_bytes=4096, global_cb=gcb_a)`.
  The cached program places the receiver kernel on cores `(1,0)..(1,3)` and pegs remote CB 31 to
  `gcb_a`'s buffer.
- Drop the last reference to `gcb_a`, freeing both its data and config L1 buffers.
- **Call 2**: `gcb_b = CreateGlobalCircularBuffer(dev, [(CoreCoord(0,0), CoreRangeSet(CoreRange((2,0),(2,3))))], size=S, L1)`
  — same core count and same receivers-per-sender, so the same config-buffer size, so the allocator
  returns the address it just freed; then
  `test_dram_prefetcher_consumer(dev, num_iters=100, page_size_bytes=4096, global_cb=gcb_b)`.
  `num_iters`, `page_size_bytes` and `config_address()` all match call 1, so **the hash is
  identical** and the cache hits. The empty `validate_on_program_cache_hit` runs and checks nothing,
  though that is incidental here: the only reachable check it suppresses,
  `TT_FATAL(attrs.global_cb->receiver_cores().num_cores() > 0, ...)` at
  `dram_prefetcher_consumer.cpp:30`, tests non-emptiness and `gcb_b` has four receiver cores, so it
  would have passed. Nothing on either validator path distinguishes `gcb_b` from `gcb_a`; the hash
  is the only defence available and it is the one that fails.
- **What goes stale**: the kernel's `CoreRangeSet` and the remote CB's `CoreRangeSet` (still row 1,
  not row 2) and the remote CB's pegged base address and config address (still `gcb_a`'s, now
  freed / possibly reallocated to something unrelated). `override_runtime_arguments` does nothing.
- **Symptom**: the consumer runs on the wrong cores. The prefetcher pushes to `gcb_b`'s receivers on
  row 2 while the cached consumer waits on row 1, so `wait_front` never satisfies — the bench hangs
  (or, with the sender's own timeout, reports a bogus bandwidth number). The receiver cores on row 1
  meanwhile poll and pop against a freed L1 region, corrupting whatever now owns it.

Note the failure is bidirectional. Even when the address is *not* recycled, keying on
`config_address()` makes the hash depend on an allocation address, so two semantically identical
GCBs created at different times force a needless kernel recompile. The hash is simultaneously too
weak (wrong hits) and too strong (spurious misses) — both symptoms of using an address as an
identity.

The GCB's `size` and `buffer_address()` are also absent from the hash, for the same reason and with
the same consequence: `size` fixes the ring geometry the receiver's `remote_index(31)` CB is
configured against, and `buffer_address()` is the pegged CB base.

### 2. `attrs.mesh_device`

**Verdict: VALID — invariant.**

The program cache is owned by the mesh device, so every entry reached through a given cache already
agrees on `mesh_device`. The pointer carries no information *within* a cache, and hashing it
(as the default would) would only add noise. `validate_on_program_cache_miss` also rejects null
(`dram_prefetcher_consumer.cpp:26`), so the disengaged case is excluded on the first call for each
hash.

### 3. Tensor arguments

**Verdict: n/a — there are none.** `tensor_args_t` is `struct tensor_args_t {};`
(`dram_prefetcher_consumer.hpp:33`), and `test_dram_prefetcher_consumer` constructs it empty
(`dram_prefetcher_consumer.cpp:109`). There is no tensor decomposition to audit, and the framework's
per-coordinate suffix contributes nothing because `extract_tensor_coordinates` finds no tensors.
This is why the CSV's `SELECTIVE` label is wrong for this op.

## The validator op in the same directory

The validator makes the same architectural choice and takes it further: it hashes a **DRAM buffer
address** in place of the whole source tensor.

```57:73:ttnn/cpp/ttnn/operations/experimental/test/prefetcher_consumer/dram_prefetcher_validator.cpp
ttsl::hash::hash_t DramPrefetcherValidatorDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    // GlobalCircularBuffer / Tensor aren't reflection-hashable here; pick the bits that
    // determine Program shape: scalar attrs, GCB identity, the source tensor's DRAM
    // address (compile-time arg via TensorAccessorArgs), and its dataformat.
    const auto* tensor_buffer = tensor_args.source_tensor.buffer();
    const tt::DataFormat dataformat = tt::tt_metal::datatype_to_dataformat_converter(tensor_args.source_tensor.dtype());
    return ttsl::hash::hash_objects_with_default_seed(
        ttsl::hash::type_hash<DramPrefetcherValidatorDeviceOperation>,
        attrs.num_layers,
        attrs.print_stride,
        attrs.streaming,
        attrs.rotation,
        static_cast<uint64_t>(attrs.global_cb->config_address()),
        static_cast<uint64_t>(tensor_buffer != nullptr ? tensor_buffer->address() : 0),
        static_cast<uint32_t>(dataformat));
}
```

### V1. `attrs.global_cb` beyond `config_address()`

**Verdict: BUG — identical to consumer omission 1, with a wider blast radius.**

The validator derives its entire ring topology from `sender_receiver_core_mapping`:

```87:102:ttnn/cpp/ttnn/operations/experimental/test/prefetcher_consumer/dram_prefetcher_validator.cpp
    const auto& sr_mapping = global_cb.sender_receiver_core_mapping();
    const uint32_t num_senders = static_cast<uint32_t>(sr_mapping.size());
    uint32_t num_blocks = 0;
    uint32_t max_bank_id = 0;
    for (const auto& [sender_logical, receivers] : sr_mapping) {
        const uint32_t bank_id = static_cast<uint32_t>(sender_logical.x);
        max_bank_id = bank_id > max_bank_id ? bank_id : max_bank_id;
        num_blocks += receivers.num_cores();
    }
    const uint32_t num_dram_banks = max_bank_id + 1;
```

`num_blocks` and `num_senders` are **compile-time args** (`dram_prefetcher_validator.cpp:174-182`),
`num_blocks` further sets `ring_size`, which sets `n_per_recv_tiles` and hence the remote and
scratch CB page sizes (`:164-171`), and the whole per-receiver runtime-arg table
(`bank_id`, `bank_local_recv`, `n_col_start`, `lead_block`) is keyed on the mapping
(`:197-245`). None of it is hashed; none of it is re-applied.

### V2. `tensor_args.source_tensor` — everything except buffer address and dtype

**Verdict: BUG.**

The omitted spec fields all feed compile-time args, CB page sizes and runtime args:

- `padded_shape()` → `k_tiles`, `total_n_tiles`, `n_per_recv_tiles`, `k_block_w_tiles`,
  `page_bytes_per_recv` (`:111-143`), which set the remote CB and scratch CB page sizes
  (`:164-171`) and four of the eight per-receiver runtime args (`:233-243`).
- `memory_config()` and the buffer's distribution spec → `is_recv_contig` /
  `is_shard_contiguous_recv_contig` (`:151-157`), which select the `ring_pos` formula
  (`:217-219`) and therefore `n_col_start` and `lead_block` for every receiver.
- `tensor_spec().tile()` → `tile_h`, `tile_w`, `tile_bytes` (`:116-142`). This one is severe enough
  on its own to warrant a separate subsection — see V3.
- `layout()`, `alignment`, storage kind → the `TensorAccessorArgs(*tensor_buffer)` compile-time
  words (`:183`), which for a sharded source encode rank, num banks, tensor shape, shard shape and
  bank coordinates (`tt_metal/impl/buffers/tensor_accessor_args.cpp:37-80`), and whose `IsDram` bit
  is unpinned on the hit path (see the `is_dram()` reproduction above).

Substituting `tensor_buffer->address()` is an attempt to make the frozen program self-consistent —
the address is itself baked in as `bank_base_addr` (`:195`, `:236`) and as the `TensorAccessor`
base, so keying on it does prevent the *address* from going stale. But the address is not a proxy
for the spec. Two-call reproduction:

- **Call 1**: allocate a width-sharded DRAM tensor `T1` of padded shape `[K, N1]` at DRAM address
  `A`; run `test_dram_prefetcher_validator(dev, T1, num_layers=1, print_stride=0, global_cb=gcb)`.
- Deallocate `T1`. Allocate `T2` with padded shape `[K, N2]`, `N2 != N1`, same dtype and same
  sharding scheme. If the DRAM allocator returns address `A` (the common case when `T2` is the first
  allocation after `T1` is freed and the sizes bucket the same way), then every hashed term matches
  call 1 — `num_layers`, `print_stride`, `streaming`, `rotation`, `config_address()`, the buffer
  address, and the dataformat.
- **Call 2**: `test_dram_prefetcher_validator(dev, T2, ...)` — cache hit. The empty
  `validate_on_program_cache_hit` skips the `is_dram()` and rank checks
  (`dram_prefetcher_validator.cpp:34-42`) and the tile-alignment / divisibility `TT_FATAL`s
  (`:122-138`).
- **What goes stale**: `total_n_tiles`, `n_per_recv_tiles` and `n_col_start` runtime args (still
  computed from `N1`), plus the remote and scratch CB page sizes, plus the
  `TensorAccessorArgs` sharded compile-time block if the shard shape changed.
- **Symptom**: the validator memcmps the received bytes against the wrong tile range of `T2`,
  DPRINTs a spurious mismatch and hangs the core — i.e. the validator reports a prefetcher bug that
  does not exist. For an op whose entire purpose is to be an oracle, a silent false positive is the
  worst possible failure.

Even when the hash *does* protect correctness, keying on a buffer address means the validator
recompiles its kernels every time the source tensor is reallocated — a full cache miss per layer in
any realistic multi-layer bench.

### V3. `source_tensor.page_config`'s `Tile` — a tile-aware factory keyed without the tile

**Verdict: BUG.**

This is the mirror image of the more common defect. Most ops in this codebase hardcode 32x32 and get
away with omitting `page_config` only by accident; the validator does the opposite. It is genuinely
tile-aware — it reads the tensor's real tile shape and uses the tile-aware `get_tile_size` rather
than the architectural `tt::tile_size`:

```116:143:ttnn/cpp/ttnn/operations/experimental/test/prefetcher_consumer/dram_prefetcher_validator.cpp
    const auto& tile_spec = source_tensor.tensor_spec().tile();
    const auto tile_shape = tile_spec.get_tile_shape();
    const uint32_t tile_h = tile_shape[0];
    const uint32_t tile_w = tile_shape[1];
    const uint32_t K_elems = padded_shape[-2];
    const uint32_t N_elems = padded_shape[-1];
    TT_FATAL(
        K_elems % tile_h == 0 && N_elems % tile_w == 0,
        "Validator: tensor padded shape ({}, {}) must be tile-aligned (tile {}x{})",
        K_elems,
        N_elems,
        tile_h,
        tile_w);
    const uint32_t k_tiles = K_elems / tile_h;
    const uint32_t total_n_tiles = N_elems / tile_w;
    TT_FATAL(
        k_tiles % num_blocks == 0, "Validator: k_tiles ({}) must be divisible by num_blocks ({})", k_tiles, num_blocks);
    const uint32_t ring_size = num_blocks;
    TT_FATAL(
        total_n_tiles % ring_size == 0,
        "Validator: total_n_tiles ({}) must be divisible by ring_size ({})",
        total_n_tiles,
        ring_size);
    const uint32_t n_per_recv_tiles = total_n_tiles / ring_size;
    const uint32_t k_block_w_tiles = k_tiles / num_blocks;
    const tt::DataFormat tensor_dataformat = datatype_to_dataformat_converter(source_tensor.dtype());
    const uint32_t tile_bytes = tile_spec.get_tile_size(tensor_dataformat);
    const uint32_t page_bytes_per_recv = k_block_w_tiles * n_per_recv_tiles * tile_bytes;
```

Because the program provably varies with `Tile`, `page_config` **must** be in the key, and it is
not: the hash carries the buffer address and the dataformat, nothing else from the tensor. The
reproduction is more direct than the shape one in V2 because it needs no shape change at all.

**Two-call reproduction.** One GCB, `num_layers`, `print_stride`, `streaming` and `rotation` fixed;
source tensor bfloat16, DRAM, padded shape `[256, 256]` in both calls.

- **Call 1**: `T1` built with the default `Tile{32, 32}` at DRAM address `A`. Then
  `tile_h = tile_w = 32`, `k_tiles = 8`, `total_n_tiles = 8`, `tile_bytes = 2048`.
- Deallocate `T1`; allocate `T2`, identical in every respect except
  `Tile{16, 32}`, and it lands back on address `A`. Both tiles divide 256, so both pass the
  alignment `TT_FATAL` — but that `TT_FATAL` lives inside `create_at` and never runs on a hit
  anyway.
- **Call 2**: hashed terms are the four scalars, `config_address()`, the address `A` and
  `Float16_b` — every one identical to call 1. Cache hit.
- **What goes stale**: `k_tiles` should be `16` and `tile_bytes` `1024`, so `k_block_w_tiles` and
  `page_bytes_per_recv` are both wrong. `page_bytes_per_recv` is the page size of *both* the remote
  CB and the scratch CB (`:164-171`), and `k_block_w_tiles` is runtime arg 3 on every receiver
  (`:233-243`).
- **Symptom**: the receiver's `wait_front`/`pop_front` units no longer match the bytes the sender
  pushes, and the scratch CB is sized for the wrong block. The memcmp compares misaligned data and
  the core hangs waiting for a page that never completes.

Non-32x32 tiles are constructible directly from Python
(`ttnn/cpp/ttnn-nanobind/tensor.cpp:220-226`), so this is reachable, not hypothetical. Note also
that hashing `page_config` would still not distinguish a transposed tile from an untransposed one:
`Tile::attribute_values()` omits both transpose flags
(`tt_metal/api/tt-metalium/tile.hpp:46-47`) and `Tile::operator==` ignores them
(`tt_metal/impl/data_format/tile.cpp:122-124`). That gap is framework-wide and not introduced here.

**The consumer is not affected by this class.** Its factory contains no tile arithmetic at all — the
sole CB is sized from `attrs.page_size_bytes` (`dram_prefetcher_consumer.cpp:72-75`), which is
hashed, and the op has no tensors. A repo-wide sweep files this directory as tile-aware, which is
right for the validator and inapplicable to the consumer.

## Keys the custom hash adds beyond the default

- Consumer: `global_cb->config_address()`. This is not in the default key (the GCB's `std::hash`
  covers the core mapping, size and buffer type, not the config allocation address). It is an
  *addition*, but it does not compensate for the three fields it displaces — see omission 1.
- Validator: `source_tensor.buffer()->address()`, likewise absent from the default key
  (`DeviceStorage` has an empty attribute tuple, so addresses never enter the default hash).

## Framework side effect of having a custom hash

```1012:1014:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
        if constexpr (requires { DeviceOperation::compute_program_hash(attrs, tensor_args); }) {
            return key;  // custom hash -> opt out beyond the op-identity prefix
        } else {
```

`ProgramCacheKey::canonical` degrades to the op type name for both ops, so a 64-bit collision
resolves to a wrong hit instead of a rebuild. For these two ops that matters less than usual only
because the deliberate gaps are already much wider than a chance collision.

## Summary

### Consumer

| Omitted vs. default | Used by program? | Patched on hit? | Verdict |
|---|---|---|---|
| `global_cb.sender_receiver_core_mapping` | Yes — kernel + CB core ranges, pegged CB | No (empty override) | **BUG** |
| `global_cb.size` | Yes — ring geometry of the pegged remote CB | No | **BUG** (same root cause) |
| `global_cb.buffer_type` | Yes — L1 vs L1_SMALL placement of the pegged CB | No | **BUG** (same root cause) |
| `mesh_device` | n/a | n/a | VALID — invariant |
| Tensor arguments (incl. any `page_config` / `Tile`) | n/a — none exist, and the factory does no tile arithmetic | n/a | n/a |

### Validator

| Omitted vs. default | Used by program? | Patched on hit? | Verdict |
|---|---|---|---|
| `global_cb.sender_receiver_core_mapping` / `size` / `buffer_type` | Yes — `num_blocks`/`num_senders` compile-time args, CB page sizes, all runtime args | No (empty override) | **BUG** |
| `source_tensor.padded_shape` | Yes — CB page sizes + 4 runtime args | No | **BUG** (V2) |
| `source_tensor.memory_config` | Yes — ring-pairing formula, accessor args | No | **BUG** (V2) |
| `source_tensor.page_config` (`Tile`) | Yes — `tile_bytes`, `page_bytes_per_recv`, CB page sizes | No | **BUG** (V3, tile-aware factory keyed without the tile) |
| `source_tensor.alignment` / `layout` / storage kind | Yes, via `TensorAccessorArgs` including the `IsDram` bit | No, and `is_dram()` is not re-checked on hits | **BUG** (subsumed by V2) |

**Four program-cache correctness bugs were found: one in the consumer, three in the validator.**
Both ops replace a genuinely hashable composite (`GlobalCircularBuffer`, and for the validator a
`Tensor`) with a single *allocation address* used as an identity token. Allocation addresses are
unique only among live objects; they are recycled, and DRAM and L1 addresses can coincide
numerically. An empty `override_runtime_arguments` then means literally nothing is refreshed on a
cache hit, so a recycled or colliding address yields a wrong hit against a program whose kernel
placement, CB configuration, compile-time args and runtime args all belong to a different
configuration.

The count is unchanged after the hit-path reachability filter. The empty
`validate_on_program_cache_hit` overrides do suppress guards that the key does not make
self-enforcing — three of the validator's six and one of the consumer's five — but no bug is caused
by them, and only one is widened. Every headline reproduction above survives the miss validator
being restored, because those checks test existence, non-emptiness or a scalar bound; not one tests
that a GCB or a tensor spec still *matches* the one the cached program was built for, which is
precisely what a hash keyed on an allocation address cannot establish. The single exception is
`is_dram()`, which does reject the DRAM/L1 address-coincidence variant of V2 and is therefore the one
place where the empty hook leaves a silent hole of its own. The empty hit validators are a genuine
but secondary defect on that basis.

The validator's third bug is of a different kind and does not depend on address recycling for its
diagnosis: its factory is genuinely tile-aware, so the program varies with `Tile` by construction,
and `Tile` is nowhere in the key.

These are bench-only debug ops, which is severity context rather than a mitigation; the validator's
case is the more damaging of the two because it turns an oracle into a source of false alarms.

## Recommendations

1. **Consumer**: hash `attrs.global_cb` directly. It already has a working `std::hash`
   specialization covering `sender_receiver_core_mapping`, `size` and `buffer_type`
   (`tt_metal/impl/buffers/global_circular_buffer.cpp:608-611`), so
   `ttsl::hash::hash_objects_with_default_seed(type_hash<...>, attrs.num_iters,
   attrs.page_size_bytes, attrs.global_cb)` is a one-line fix that is strictly stronger than what is
   there now. Delete the "isn't reflection-hashable" comment — it is not true. Keeping
   `config_address()` as an *additional* term is harmless but no longer necessary, and dropping it
   also removes the spurious-recompile-on-reallocation behaviour.
2. **Validator**: same change for `global_cb`, plus hash the source tensor's `tensor_spec()`
   instead of its buffer address. `tensor_spec()` is one term and covers all three of V2 and V3's
   omissions at once — `padded_shape()` is derived from it, and it carries `page_config` (hence the
   `Tile` that V3 turns on), `memory_config` and the alignment behind the accessor args. Then
   implement `override_runtime_arguments` to re-apply `bank_base_addr` for each receiver — that is
   exactly the `Buffer*`-address slot the mode-A hook exists for, and it lets the address leave the
   hash so a reallocated source tensor stops forcing a recompile.
3. Delete the validator's empty `validate_on_program_cache_hit`, and add
   `TT_FATAL(tensor_buffer->is_dram(), ...)` to `DramPrefetcherValidatorDeviceOperation::
   validate_on_program_cache_miss` if it is not already reached that way. With the override gone the
   framework substitutes the miss validator on every hit
   (`ttnn/api/ttnn/device_operation.hpp:262-266`), which restores `is_dram()` — the one suppressed
   check that closes a *silent* failure, namely the DRAM/L1 numeric-address coincidence above. The
   per-dispatch cost is one integer comparison on `num_layers`, two null/`buffer_type` tests on the
   source buffer, and two non-emptiness queries on the GCB: negligible for a bench op that already
   runs a whole-tensor memcmp on device.

   This is subordinate to recommendations 1 and 2, which is a change of emphasis from how it was
   first written. Restoring the miss validator does **not** reject any of the four reproductions
   above: every check in it tests existence or non-emptiness, and in each reproduction the second
   call's GCB and tensor are perfectly well-formed. Fixing the hashes is what closes the bugs; this
   recommendation closes one additional silent hole that the hash fixes would also cover (hashing
   `tensor_spec()` puts `memory_config` and hence `buffer_type` in the key), so if only one change is
   made it should be 1 and 2, not this.

   For the consumer the same change is not worth making. Its miss validator drops five checks on the
   hit path, of which two are self-enforcing (`num_iters` and `page_size_bytes` are hash terms), two
   are dead on both paths (a null `mesh_device` faults in `launch` before any validator, and
   `compute_program_hash` dereferences `global_cb` at `:54` before either), and the one reachable
   check tests only that the GCB has *some* receiver cores. Recommend against restoring it: it would
   charge every dispatch for a guard that catches nothing the hash fix in recommendation 1 does not
   already catch.

   Note also that none of this restores the checks inside `create_at`
   (`dram_prefetcher_validator.cpp:97-101`, `:112-115`, `:122-138`): those run only when a program is
   built, under any hook. The rank, tile-alignment and divisibility invariants all test values absent
   from the key and so are reachable on a hit, but the right fix for them is hashing `tensor_spec()`
   (recommendation 2), which makes them unreachable by construction rather than paying for them per
   dispatch.
4. If the empty `override_runtime_arguments` bodies are meant to say "this op genuinely has no
   per-dispatch state", say so in a comment that names the invariant being relied on (the GCB is
   fully hashed, so the pegged CB and core ranges cannot change under a hit). As written, the
   comment "Nothing to override — all args are compile-time" states a true fact that does not imply
   the conclusion.
