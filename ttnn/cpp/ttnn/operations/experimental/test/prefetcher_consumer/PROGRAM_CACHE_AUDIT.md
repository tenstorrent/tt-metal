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
| Custom hash | `dram_prefetcher_consumer.cpp:46-55` — **unchanged by fab067a** | `dram_prefetcher_validator.cpp:57-87` (post-fab067a; `:57-73` pre-fix, which is what the V-sections cite) |
| `operation_attributes_t` | `num_iters`, `page_size_bytes`, `global_cb`, `mesh_device` | `num_layers`, `print_stride`, `global_cb`, `streaming`, `rotation` |
| `tensor_args_t` | **empty struct** (`dram_prefetcher_consumer.hpp:33`) | `source_tensor` |
| Program factory | `ProgramFactory` (`create_at` → `CachedProgram`) | `ProgramFactory` (`create_at` → `CachedProgram`) |
| `override_runtime_arguments` | Present but an **empty no-op** (`dram_prefetcher_consumer.cpp:89-95`) | Present but an **empty no-op** (`dram_prefetcher_validator.cpp:264-270`; `:250-256` pre-fix) |
| `get_dynamic_runtime_args` | No | No |
| `validate_on_program_cache_hit` | Present but an **empty no-op** (`dram_prefetcher_consumer.cpp:33-34`) | Present but an **empty no-op** (`dram_prefetcher_validator.cpp:44-45`) |
| Cache-hit patch mechanism | **Op-owned re-derivation (mode A) with an empty body — nothing is refreshed** | Same |

## Post-fix status — commit fab067a

**The verdicts for the two ops in this directory now diverge, and that is the headline.** fab067a
rewrote `DramPrefetcherValidatorDeviceOperation::compute_program_hash` and left
`DramPrefetcherConsumerDeviceOperation` entirely untouched.

- **Validator: CLEAR.** All three of its bugs (V1, V2, V3) are resolved. One relaxation-free key now
  covers the whole `GlobalCircularBuffer`, both of its allocation addresses, and the source tensor's
  `page_config`, `padded_shape` and `memory_config`.
- **Consumer: NOT CLEAR — its bug is unchanged.** The one-line fix landed on the validator and not on
  its sibling. `DramPrefetcherConsumerDeviceOperation::compute_program_hash` still keys only
  `num_iters`, `page_size_bytes` and `global_cb->config_address()`
  (`dram_prefetcher_consumer.cpp:46-55`), and the comment asserting that "GlobalCircularBuffer isn't
  reflection-hashable" (`:48-49`) is still there and still wrong — the validator's own fix disproves it
  three lines of code away, by calling
  `std::hash<tt::tt_metal::experimental::GlobalCircularBuffer>{}` directly
  (`dram_prefetcher_validator.cpp:74-75`). The reproduction under omission 1 below still executes as
  written.

**What fab067a changed in the validator**

- `compute_program_hash` now keys the **whole** GCB via its `std::hash` specialization
  (`dram_prefetcher_validator.cpp:74-75`), guarded on `has_value()` with a `std::size_t{0}` sentinel.
  That puts `sender_receiver_core_mapping`, `size` and `buffer_type` in the key by value, which is
  what V1 asked for.
- It additionally keys **both** GCB allocation addresses — `buffer_address()` and `config_address()`
  (`:80-81`) — with the rationale spelled out at `:76-79`: `std::hash` covers the GCB's structure but
  not which allocation it is, the remote CB bakes both addresses at build time, and
  `UpdateDynamicCircularBufferAddress` refuses to re-point a GCB-backed CB, so two same-shaped GCBs at
  different allocations must not share a program. This is an *addition* to the fix V1 requested, not a
  substitution for it.
- It keys `source_tensor.tensor_spec().page_config()`, `padded_shape()` and `memory_config()`
  (`:84-86`). `page_config` is exactly what V3 turns on; `padded_shape` and `memory_config` are what
  V2 turns on.
- The hash comment was rewritten (`:59-65`) to enumerate what `create_at` reads and why the address
  alone is not a sufficient identity, replacing the "aren't reflection-hashable" claim.

**What remains open — validator**

- `source_tensor.logical_shape()` is not keyed; `padded_shape()` is. That is a genuine
  logical-vs-padded relaxation and it is sound: `create_at` reads `padded_shape[-2]` and
  `padded_shape[-1]` only (`:112-115`), never a logical extent, so a weight tensor with `K = 4090`
  padded to 4096 correctly shares a program with one at `K = 4096`. On a Metal 2.0 port this maps onto
  `TensorSpecRelaxations::match_padded_shape_only`
  (`tt_metal/api/tt-metalium/experimental/metal2_host_api/tensor_spec_relaxations.hpp:41,49`), which
  `pertinent_fields` reduces to `PertinentFields{.padded_shape = true}`
  (`tt_metal/impl/metal2_host_api/tensor_spec_relaxations.cpp:67-87`) — the field the hash already
  keys. Both `hash_tensorspec_with_relaxation` (`:116`) and `tensorspecs_match_with_relaxation`
  (`:161-201`) consume that one set and `ValidateTensorArgs` delegates to the predicate
  (`tt_metal/impl/metal2_host_api/program_run_args.cpp:176-189`), so the key and validation cannot
  disagree. Alternatively, keying the whole `tensor_spec()` and leaving relaxations default would be
  simpler and strictly finer, since nothing in `create_at` benefits from the extra hits — that is what
  `rotary_embedding_indexed` now does.
- `alignment` and `layout` are still not keyed by name, but they are no longer reachable omissions:
  the accessor's aligned page size and every shard-geometry word reduce to the now-hashed
  `page_config`, `padded_shape`, `memory_config` and dtype.
- The empty `validate_on_program_cache_hit` (`:44-45`) was **not** deleted, and the `is_dram()` check
  still runs only on the miss path. That is now moot rather than fixed: `is_dram()` is a projection of
  `buffer_type`, which is carried by the hashed `memory_config`, so the DRAM/L1
  numeric-address-coincidence variant of V2 now misses instead of hitting. The silent hole is closed by
  hashing rather than by validation, so recommendation 3's premise no longer holds.
- The empty `override_runtime_arguments` (`:264-270`) is unchanged, so the three allocation addresses stay
  in the key and a reallocation still forces a recompile. That is a cost, not a correctness defect;
  see `## Non-cache correctness defects`.

**What remains open — consumer**

Everything. Omission 1 is unresolved: `sender_receiver_core_mapping`, `size` and `buffer_type` are all
absent from the key, they determine the kernel's core range, the CB's core range and the pegged remote
CB's geometry, and the empty `override_runtime_arguments` (`:89-95`) refreshes none of it.

**Metal 2.0 port**

The validator is clear to port, carrying the one `match_padded_shape_only` relaxation described above.
The consumer is **not** clear: porting it would move it onto a dispatch path with a `TensorSpec`
contract, but the consumer has no tensors at all — its stale state is a `CoreRangeSet` and a pegged
global CB, neither of which any framework predicate compares. Nothing about the port would catch its
bug. It needs the one-line hash fix first, and the fix is already written down twice: in
recommendation 1 below, and in its sibling's source.

**Result: one BUG in the consumer, three in the validator, at the time of the original audit.** All
four share a root cause — an
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
(`dram_prefetcher_consumer.cpp:46-55`). **This section stands unaltered post-fab067a** — the commit did
not touch `dram_prefetcher_consumer.cpp`, so the hashed set, the table and the verdict are all current
as written.

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

### Validator — almost every dropped check is reachable (pre-fix)

Pre-fix hashed set: `num_layers`, `print_stride`, `streaming`, `rotation`,
`global_cb->config_address()`, the source buffer's address, and its dataformat
(`dram_prefetcher_validator.cpp:57-73`). Because that set carried nothing structural about either the
GCB or the tensor spec, the filter removed almost nothing. Checks inside `create_at` are listed
alongside the validator's own, since `create_at` runs only on a miss and the effect on the hit path is
identical.

The **In the key?** column below has two entries per row: the pre-fix answer, then the post-fab067a
answer, because that is where the change shows up. fab067a keys the whole GCB (`:74-75`), both of its
allocation addresses (`:80-81`), and the source tensor's `page_config`, `padded_shape` and
`memory_config` (`:84-86`).

| Check absent on the hit path | Where it lives | Value it constrains | In the key? (pre-fix → post-fix) | Reachable on a hit? (post-fix) |
|---|---|---|---|---|
| `attrs.num_layers > 0` | miss validator, `:33` | `attrs.num_layers` | Yes → Yes | No |
| `tensor_buffer != nullptr` | miss validator, `:34-35` | source tensor storage kind | Effectively yes → Yes | No — and never was; pinned by the framework, see `#### Framework correction` below |
| `tensor_buffer->is_dram()` | miss validator, `:36` | the source buffer's `buffer_type` | **No** → **Yes**, inside the hashed `memory_config` | **No** |
| `attrs.global_cb.has_value()` | miss validator, `:37` | engagement of the GCB optional | No → **Yes**, explicitly, via the `has_value()` ternaries at `:75-82` | No |
| `receiver_cores().num_cores() > 0` | miss validator, `:38` | non-emptiness of the receiver core set | **No** → **Yes**, inside the hashed `sender_receiver_core_mapping` | **No** |
| `!sr_mapping.empty()` | miss validator, `:40-41` | non-emptiness of the sender/receiver mapping | **No** → **Yes**, hashed by value | **No** |
| `num_blocks % num_dram_banks == 0` | `create_at`, `:97-101` | GCB mapping shape | **No** → **Yes** | **No** |
| `padded_shape.rank() >= 2` | `create_at`, `:112-115` | `source_tensor.padded_shape()` | **No** → **Yes** | **No** |
| `K_elems % tile_h == 0 && N_elems % tile_w == 0` | `create_at`, `:122-128` | `padded_shape` against `Tile` | **No** — neither hashed → **Yes**, both | **No** |
| `k_tiles % num_blocks == 0` | `create_at`, `:131-132` | `padded_shape`, `Tile`, GCB mapping | **No** → **Yes**, all three | **No** |
| `total_n_tiles % ring_size == 0` | `create_at`, `:134-138` | same | **No** → **Yes** | **No** |
| `bank_local_recv < receivers_per_bank` | `create_at`, `:205-211` | GCB mapping shape | **No** → **Yes** | **No** |
| `ring_pos < rotation.size()` | `create_at`, `:226-231` | `rotation` against the GCB mapping | `rotation` yes, mapping no → **Yes**, both | **No** |

Pre-fix, only three of thirteen were filtered out. That was the diagnostic signature of a hash that
keys on allocation addresses instead of structure: because neither the GCB's mapping nor the tensor's
spec was in the key, virtually nothing the validator checked was self-enforcing. Note that only the
first six rows were ever attributable to the empty hook — the seven `create_at` rows were never on the
hit path under any hook, since `create_at` runs only when a program is built.

**Post-fix the reachable column is empty.** Every value in the table is now either hashed by value or
pinned by the framework, so each of these checks is self-enforcing: a call carrying a new value of any
of them computes a different key, misses, and meets the check on the miss path. That is the correct
resolution — it makes the miss-only placement of all thirteen harmless *by construction* rather than by
audit, and it is why the empty `validate_on_program_cache_hit`, though still present (`:44-45`), no
longer suppresses anything reachable.

#### Framework correction

The `tensor_buffer != nullptr` row was graded "effectively yes" pre-fix on the strength of the hash's
`0` sentinel, with a parenthetical about `device_operation.hpp:455`. The correct statement is stronger,
and it is a framework guarantee rather than a property of this op's key: `launch()` asserts device
storage *and* allocation on every tensor argument, on every dispatch, before the key is computed or the
cache is probed.

```491:502:ttnn/api/ttnn/device_operation.hpp
    std::vector<std::reference_wrapper<const Tensor>> input_tensors;
    ttsl::reflection::visit_object_of_type<Tensor>(
        [&input_tensors](const Tensor& t) { input_tensors.push_back(std::cref(t)); }, tensor_args);
    // ...
    for (const auto& input_tensor_ref : input_tensors) {
        const auto& input_tensor = input_tensor_ref.get();
        TT_FATAL(is_device_tensor(input_tensor), "Device Operations expect device tensors as inputs");
        TT_FATAL(input_tensor.is_allocated(), "Input Tensor is not allocated");
    }
```

It sits at the top of `launch()` (`:487`), well before `launch_operation_with_adapter` computes the key
and probes the cache (`:409-418`), so it runs on hits and misses alike; it traverses via
`visit_object_of_type<Tensor>` (`:492-493`), which descends into `std::optional<Tensor>`; and it is
unconditional. So the source tensor's storage variant kind was never a reachable omission on any path,
and `tensor_buffer` can never be null when `compute_program_hash` runs. The `0` sentinel at `:82` is
defensive rather than load-bearing.

The `is_dram()` row deserves its own reproduction, because pre-fix it turned the address-as-identity
choice into a bug even when nothing was reallocated. DRAM and L1 are separate address spaces that both
start near zero, so an L1 buffer and a DRAM buffer can hold the *same numeric address* simultaneously.
Call 1 with a DRAM source tensor at address `A` compiled the accessor with `IsDram` set; call 2 with an
L1 source tensor that happened to sit at address `A` and had the same dtype produced an identical hash,
hit, and was not rejected because `is_dram()` no longer ran. The kernel then resolved an L1 offset
through the DRAM bank map. This was the one dropped check whose restoration would have closed a silent
hole outright rather than merely improving a diagnostic.

**fab067a closes it by hashing instead of by validating**, which is the better of the two fixes.
`is_dram()` is a projection of `buffer_type`, which is carried by the now-hashed `memory_config`
(`:86`), so the L1 tensor at address `A` computes a *different* key and misses. The check itself is
still miss-only, and it is still worth keeping there for its diagnostic — but it is no longer the only
thing standing between the op and a silent wrong hit.

### What the filter changes, and what it does not

No verdict below moved on the strength of the filter alone. All four bugs were hash omissions on values
that the *miss* validator does not constrain either — a different receiver core set, a different tensor
shape, a different `Tile` all pass every `TT_FATAL` in both ops — so restoring the miss validator on the
hit path would not have rejected a single one of the reproductions. The empty
`validate_on_program_cache_hit` overrides were a real but secondary defect: between them they suppress
eleven checks on the hit path, of which four were reachable — one degenerate-case guard on the consumer
and three on the validator — and of those four exactly one, `is_dram()`, closed a silent failure rather
than a degenerate one.

**That analysis is what fab067a acted on: it fixed the hashes and left the hooks alone.** For the
validator that was the right call and it worked — with the GCB, `page_config`, `padded_shape` and
`memory_config` in the key, all thirteen suppressed checks become self-enforcing and the empty hook
suppresses nothing reachable. The empty hook is still there (`:44-45`), and it is now genuinely
harmless rather than merely secondary. For the consumer nothing was fixed at all, so its bug and its
one reachable suppressed guard both stand exactly as described below.

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

Both overrides are still empty post-fab067a. The validator therefore still carries the full obligation
above, and it now discharges it — its key covers the GCB by structure and by both allocation
addresses, and the source tensor by `page_config`, `padded_shape` and `memory_config`. The price is
paid in the key rather than in the override: three allocation addresses in the hash means a recompile
on every reallocation, which is what implementing the override would have avoided. That trade is
recorded under `## Non-cache correctness defects`. The consumer still does not discharge the
obligation at all.

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

So the premise of the consumer's custom hash is false: the default key would have covered
`sender_receiver_core_mapping`, `size` and `buffer_type` by value, which is precisely the
information the custom hash discards. The consumer's custom hash is strictly weaker than the default.

**fab067a settled this empirically, in the file next door.** Its rewrite of the validator's hash calls
`std::hash<tt::tt_metal::experimental::GlobalCircularBuffer>{}(*attrs.global_cb)` directly
(`dram_prefetcher_validator.cpp:74-75`) and compiles. The consumer's comment claiming otherwise
(`dram_prefetcher_consumer.cpp:48-49`) was not touched, so the directory now contains a claim and its
counterexample side by side.

## What the custom hash covers

Consumer: `num_iters`, `page_size_bytes`, and `global_cb->config_address()`. **Unchanged by fab067a.**

Validator, post-fab067a (`dram_prefetcher_validator.cpp:70-86`):

```70:86:ttnn/cpp/ttnn/operations/experimental/test/prefetcher_consumer/dram_prefetcher_validator.cpp
        attrs.num_layers,
        attrs.print_stride,
        attrs.streaming,
        attrs.rotation,
        attrs.global_cb.has_value() ? std::hash<tt::tt_metal::experimental::GlobalCircularBuffer>{}(*attrs.global_cb)
                                    : std::size_t{0},
        // std::hash covers the GCB's structure (core mapping, size, buffer type) but not which
        // allocation it is. The remote CB is created against the GCB and bakes both of these addresses
        // at build time, and UpdateDynamicCircularBufferAddress refuses to re-point a GCB-backed CB, so
        // two same-shaped GCBs at different allocations must not share a program.
        static_cast<uint64_t>(attrs.global_cb.has_value() ? attrs.global_cb->buffer_address() : 0),
        static_cast<uint64_t>(attrs.global_cb.has_value() ? attrs.global_cb->config_address() : 0),
        static_cast<uint64_t>(tensor_buffer != nullptr ? tensor_buffer->address() : 0),
        static_cast<uint32_t>(dataformat),
        tensor_args.source_tensor.tensor_spec().page_config(),
        tensor_args.source_tensor.padded_shape(),
        tensor_args.source_tensor.memory_config());
```

The three `has_value()` ternaries also fix a latent defect the pre-fix code had: it dereferenced
`attrs.global_cb` unconditionally at `:70`, so a disengaged optional was a null dereference *inside the
hash*, before the miss validator's own `has_value()` check could report it. The disengaged case now
hashes to a triple of zeros, which no live GCB can produce — a real GCB's structural hash is not zero
and neither of its allocation addresses is.

## Omitted parameters — consumer

### 1. `attrs.global_cb` — everything except `config_address()`

**Verdict: BUG — UNCHANGED by fab067a.** This is the one finding in this directory that the commit did
not touch. The consumer's `compute_program_hash` still reads exactly as quoted above
(`dram_prefetcher_consumer.cpp:46-55`), the "isn't reflection-hashable" comment is still there and
still false, and `override_runtime_arguments` is still an empty no-op. The reproduction below still
executes as written. The one-line fix in recommendation 1 remains outstanding, and its sibling
`dram_prefetcher_validator.cpp:74-75` now shows exactly what it should look like.

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

**Pre-fix**, the validator made the same architectural choice and took it further: it hashed a **DRAM
buffer address** in place of the whole source tensor. fab067a rewrote this function — see
`## Post-fix status` above and the verdicts in V1-V3 — but the pre-fix body is retained here because
the three reproductions are written against it.

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

**Verdict: RESOLVED by fab067a** (was BUG — identical to consumer omission 1, with a wider blast
radius). The key now carries the whole `GlobalCircularBuffer` by value through its `std::hash`
specialization (`dram_prefetcher_validator.cpp:74-75`), which covers
`sender_receiver_core_mapping`, `size` and `buffer_type` — so the mapping that determines the ring
topology is in the key rather than being proxied by an allocation address.

fab067a also went a step further than V1 asked, and the extra step is worth recording because the
reasoning is not obvious: it keys `buffer_address()` **and** `config_address()` alongside the
structural hash (`:80-81`). The structural hash establishes that two GCBs have the same shape; it does
not establish that they are the same *allocation*. `CreateCircularBuffer`'s GCB overload pegs the
remote CB to the GCB's buffer and config addresses at build time, and
`UpdateDynamicCircularBufferAddress` refuses to re-point a GCB-backed CB, so two same-shaped GCBs at
different allocations genuinely must not share a program. Keying an address is normally the defect —
this is the case where it is the fix, precisely because the empty `override_runtime_arguments` means
the pegged address can never be refreshed. The cost is recorded in
`## Non-cache correctness defects`.

**The consumer's identical omission is unresolved** — see omission 1. Everything below describes the
pre-fix validator.

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

**Verdict: RESOLVED by fab067a** (was BUG). The key now carries `tensor_spec().page_config()`,
`padded_shape()` and `memory_config()` (`dram_prefetcher_validator.cpp:84-86`) alongside the address
and dataformat. Those three cover every field the bullets below identify as load-bearing: shape
arithmetic, the sharding-derived contiguity flags, the tile (via `page_config`, which V3 addresses),
and the `TensorAccessorArgs` compile-time block — whose sharded contents reduce to `memory_config`'s
`NdShardSpec`, `padded_shape` and the page size, and whose `IsDram` bit is `memory_config`'s
`buffer_type`. Two source tensors that differ in any of them now compute different keys and miss.

Two residual fields are still not keyed, and both are now graded rather than left open:

- **`logical_shape()` — a relaxation with real value.** The key carries `padded_shape` only, so two
  tensors with the same padded shape and different logical shapes share a program. That is correct
  here: `create_at` reads `padded_shape[-2]` and `padded_shape[-1]` and never touches
  `logical_shape`, so the program is genuinely invariant to it, and the reuse is real — a weight
  tensor with `K = 4090` and one with `K = 4096` both pad to `4096` and should share one program.
  This is the op's one logical-vs-padded relaxation; see `## Post-fix status` for how it maps onto
  Metal 2.0's `match_padded_shape_only`.
- **`alignment` and `layout()` — determined by hashed terms.** `alignment` only reaches the program
  through the aligned page size, which for this op's tile-layout source is the tile size — a function
  of `page_config` and dtype, both hashed — and through `padded_shape`, also hashed. `layout()` is a
  projection of `page_config`. Neither can move independently of the key.

**Everything below describes the pre-fix code.** The omitted spec fields all feed compile-time args,
CB page sizes and runtime args:

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

Pre-fix, substituting `tensor_buffer->address()` was an attempt to make the frozen program self-consistent —
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

Post-fix, `N2 != N1` changes `padded_shape` and therefore the key, so call 2 misses, builds a correct
program, and meets all of the `TT_FATAL`s on the miss path where they live.

Even when the hash *does* protect correctness, keying on a buffer address means the validator
recompiles its kernels every time the source tensor is reallocated — a full cache miss per layer in
any realistic multi-layer bench. fab067a did not change this, and in fact added two more addresses to
the key; it is a cost rather than a defect, and it is recorded under
`## Non-cache correctness defects`.

### V3. `source_tensor.page_config`'s `Tile` — a tile-aware factory keyed without the tile

**Verdict: RESOLVED by fab067a** (was BUG). `tensor_spec().page_config()` is now hashed
(`dram_prefetcher_validator.cpp:84`), and `Tile` is part of `TilePageConfig`'s attribute values, so the
tile the factory reads is the tile the key carries. The reproduction below now misses on call 2.

Note the shape of the fix: this op needed no new guard, because the defect was never that a check sat
in the wrong validator — the tile-alignment and divisibility `TT_FATAL`s at `:122-138` live in
`create_at` and are exactly where they belong. The defect was purely that the key did not distinguish
the inputs those checks accept. Hashing the field is the whole fix. Contrast the KV-cache ops in this
audit series, where the values *were* keyed-adjacent but the guards sat in
`validate_on_program_cache_miss` and so never ran on the hitting call.

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

**Two-call reproduction (pre-fix).** One GCB, `num_layers`, `print_stride`, `streaming` and `rotation`
fixed; source tensor bfloat16, DRAM, padded shape `[256, 256]` in both calls.

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
(`ttnn/cpp/ttnn-nanobind/tensor.cpp:220-226`), so this was reachable, not hypothetical. Note also
that hashing `page_config` still does not distinguish a transposed tile from an untransposed one:
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
  **Unchanged by fab067a.**
- Validator: `source_tensor.buffer()->address()`, likewise absent from the default key
  (`DeviceStorage` has an empty attribute tuple, so addresses never enter the default hash).
  fab067a adds two more of the same kind — `global_cb->buffer_address()` and
  `global_cb->config_address()` (`:80-81`) — so the validator's key is now the default key's field set
  *plus* three allocation addresses. The additions are load-bearing, not incidental: the remote CB is
  pegged to the GCB's addresses at build time and `override_runtime_arguments` is empty, so a
  same-shaped GCB at a different allocation must not share the program. The recompile cost that
  follows is recorded under `## Non-cache correctness defects`.

## Framework side effect of having a custom hash

```1012:1014:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
        if constexpr (requires { DeviceOperation::compute_program_hash(attrs, tensor_args); }) {
            return key;  // custom hash -> opt out beyond the op-identity prefix
        } else {
```

`ProgramCacheKey::canonical` degrades to the op type name for both ops, so a 64-bit collision
resolves to a wrong hit instead of a rebuild. For the consumer that still matters less than usual only
because its deliberate gap is much wider than a chance collision. For the validator, post-fab067a it is
now the residual exposure: with the field set correct, forfeiting attribute-level collision resolution
is the only way a wrong hit can still occur.

## Non-cache correctness defects

Recorded separately so they are not counted as program-cache bugs. These concern the override and the
cost model, not the key.

| Defect | Status | Note |
|---|---|---|
| The validator's key carries three allocation addresses — the source buffer's (`:82`) and the GCB's `buffer_address()` and `config_address()` (`:80-81`) — so any reallocation of either forces a full recompile. In a multi-layer bench that is a cache miss per layer. | **OPEN** — widened by fab067a, deliberately | This is a cost, not a correctness defect: the addresses are in the key because they are baked into the program and nothing refreshes them. It is forced by the no-op `override_runtime_arguments` (`:264-270`). Implementing that hook to re-apply `bank_base_addr` per receiver (`:195`, `:236`) would let the source address leave the key; the GCB pair is harder, since `UpdateDynamicCircularBufferAddress` refuses to re-point a GCB-backed CB, so those two have to stay until that restriction is lifted. See recommendation 2. |
| The consumer's empty `override_runtime_arguments` with the comment "Nothing to override — all args are compile-time". | **OPEN** — unchanged by fab067a | True as stated, but it does not imply the conclusion; the invariant actually being relied on is that the GCB is fully hashed, which for the consumer is still not the case. See recommendation 4. |
| The consumer's comment asserting `GlobalCircularBuffer` "isn't reflection-hashable" (`dram_prefetcher_consumer.cpp:48-49`). | **OPEN** — and now demonstrably false in-tree | fab067a's validator hash calls `std::hash<GlobalCircularBuffer>{}` directly (`dram_prefetcher_validator.cpp:74-75`) in the same directory. The comment is the stated justification for omission 1, so it is load-bearing misinformation rather than a stale note. |

The validator has no factory-level defects of the kind found in the KV-cache ops: it reads the
tensor's real tile shape and uses the tile-aware `get_tile_size` rather than assuming 32x32
(`:116-143`), so there was never a wrong-program-on-a-miss hazard to fence off.

## Summary

### Consumer

| Omitted vs. default | Used by program? | Patched on hit? | Verdict |
|---|---|---|---|
| `global_cb.sender_receiver_core_mapping` | Yes — kernel + CB core ranges, pegged CB | No (empty override) | **BUG — unchanged by fab067a** |
| `global_cb.size` | Yes — ring geometry of the pegged remote CB | No | **BUG — unchanged** (same root cause) |
| `global_cb.buffer_type` | Yes — L1 vs L1_SMALL placement of the pegged CB | No | **BUG — unchanged** (same root cause) |
| `mesh_device` | n/a | n/a | VALID — invariant |
| Tensor arguments (incl. any `page_config` / `Tile`) | n/a — none exist, and the factory does no tile arithmetic | n/a | n/a |

### Validator

| Omitted vs. default | Used by program? | Patched on hit? | Verdict |
|---|---|---|---|
| `global_cb.sender_receiver_core_mapping` / `size` / `buffer_type` | Yes — `num_blocks`/`num_senders` compile-time args, CB page sizes, all runtime args | No (empty override) | **RESOLVED by fab067a** (V1) — whole GCB hashed, plus both allocation addresses |
| `source_tensor.padded_shape` | Yes — CB page sizes + 4 runtime args | No | **RESOLVED by fab067a** (V2) — hashed at `:85` |
| `source_tensor.memory_config` | Yes — ring-pairing formula, accessor args | No | **RESOLVED by fab067a** (V2) — hashed at `:86` |
| `source_tensor.page_config` (`Tile`) | Yes — `tile_bytes`, `page_bytes_per_recv`, CB page sizes | No | **RESOLVED by fab067a** (V3) — hashed at `:84` |
| `source_tensor.alignment` / `layout` | Yes, via `TensorAccessorArgs` | No | VALID — determined by the hashed `page_config`, dtype and `padded_shape` |
| `source_tensor.logical_shape` | No — `create_at` reads only `padded_shape` | n/a | VALID — relaxation win; maps to `match_padded_shape_only` |
| `source_tensor` storage kind (incl. the `IsDram` bit) | Yes, via `TensorAccessorArgs` | n/a | VALID — device storage pinned by the framework; `IsDram` now inside the hashed `memory_config` |

**Four program-cache correctness bugs were found; three are resolved and one stands. The current count
is one: consumer omission 1.** Both ops replaced a genuinely hashable composite
(`GlobalCircularBuffer`, and for the validator a `Tensor`) with a single *allocation address* used as
an identity token. Allocation addresses are unique only among live objects; they are recycled, and
DRAM and L1 addresses can coincide numerically. An empty `override_runtime_arguments` then means
literally nothing is refreshed on a cache hit, so a recycled or colliding address yields a wrong hit
against a program whose kernel placement, CB configuration, compile-time args and runtime args all
belong to a different configuration.

fab067a applied exactly the fix recommendation 2 called for, to the validator: it hashes the GCB by
value and adds the source tensor's `page_config`, `padded_shape` and `memory_config`, which closes V1,
V2 and V3 together and, as a side effect, makes all thirteen of the validator's hit-path-suppressed
checks self-enforcing. **It did not apply the one-line equivalent from recommendation 1 to the
consumer**, whose hash, comment and empty override are byte-for-byte unchanged. The two files sit in
the same directory and now disagree with each other about whether `GlobalCircularBuffer` can be
hashed.

The count was unchanged by the hit-path reachability filter, and that assessment held up: the empty
`validate_on_program_cache_hit` overrides did suppress guards the key did not make self-enforcing —
three of the validator's six and one of the consumer's five — but no bug was caused by them, and only
one was widened. Every headline reproduction above survives the miss validator being restored, because
those checks test existence, non-emptiness or a scalar bound; not one tests that a GCB or a tensor spec
still *matches* the one the cached program was built for, which is precisely what a hash keyed on an
allocation address cannot establish. The single exception was `is_dram()`, which does reject the
DRAM/L1 address-coincidence variant of V2 — and fab067a closed that by putting `memory_config` in the
key rather than by touching the hook, which is the stronger of the two fixes. Both empty hooks are
still present; the validator's is now harmless, the consumer's still suppresses one reachable
degenerate-case guard.

The validator's third bug was of a different kind and did not depend on address recycling for its
diagnosis: its factory is genuinely tile-aware, so the program provably varies with `Tile`, and `Tile`
was nowhere in the key. Hashing `page_config` was the entire fix — no new guard was needed, because no
guard was in the wrong place. This is the cleanest contrast in the audit series with the KV-cache ops,
where the guards existed but sat in `validate_on_program_cache_miss` and so never ran on the hitting
call.

These are bench-only debug ops, which is severity context rather than a mitigation; the validator's
case was the more damaging of the two because it turned an oracle into a source of false alarms. That
is now fixed. The consumer's remains open.

## Recommendations

1. **OUTSTANDING — Consumer**: hash `attrs.global_cb` directly. It already has a working `std::hash`
   specialization covering `sender_receiver_core_mapping`, `size` and `buffer_type`
   (`tt_metal/impl/buffers/global_circular_buffer.cpp:608-611`), so
   `ttsl::hash::hash_objects_with_default_seed(type_hash<...>, attrs.num_iters,
   attrs.page_size_bytes, attrs.global_cb)` is a one-line fix that is strictly stronger than what is
   there now. Delete the "isn't reflection-hashable" comment — it is not true. Keeping
   `config_address()` as an *additional* term is harmless but no longer necessary, and dropping it
   also removes the spurious-recompile-on-reallocation behaviour.

   **This is the one recommendation in this document that fab067a did not act on**, and it is now a
   two-line copy of code that exists in the same directory: `dram_prefetcher_validator.cpp:74-75`.
2. **DONE in part — Validator**: fab067a made the `global_cb` change (`:74-75`) and keyed the source
   tensor's `page_config`, `padded_shape` and `memory_config` (`:84-86`) rather than `tensor_spec()`
   wholesale. That field-by-field form is equivalent for correctness and deliberately preserves the
   logical-vs-padded relaxation the wholesale form would have given up — see V2's residual-fields
   note. Compare `rotary_embedding_indexed`, which took the wholesale `tensor_spec()` route because it
   had no such relaxation to keep.

   **The override half is outstanding.** `override_runtime_arguments` is still empty (`:264-270`), so
   the buffer address is still in the key and the commit added two more addresses beside it.
   Re-applying `bank_base_addr` for each receiver (`:195`, `:236`) — exactly the `Buffer*`-address slot
   the mode-A hook exists for — would let the source address leave the hash so a reallocated source
   tensor stops forcing a recompile. The GCB's two addresses cannot follow until
   `UpdateDynamicCircularBufferAddress` allows re-pointing a GCB-backed CB. Recorded as a cost under
   `## Non-cache correctness defects`.
3. **SUPERSEDED for the validator, still OUTSTANDING as written.** Delete the validator's empty
   `validate_on_program_cache_hit`, and add
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
   `memory_config` puts `buffer_type` in the key), so if only one change is made it should be 1 and 2,
   not this.

   fab067a took that advice and it was correct: it fixed the hash and left the hook alone, and the
   `is_dram()` hole closed anyway because `memory_config` is now hashed (`:86`). The empty hook still
   sits at `:44-45` and the deletion is still worth doing for tidiness, but it no longer suppresses
   anything reachable — see the post-fix note under "Validator — almost every dropped check is
   reachable". Its priority drops from secondary to cosmetic.

   For the consumer the same change is not worth making. Its miss validator drops five checks on the
   hit path, of which two are self-enforcing (`num_iters` and `page_size_bytes` are hash terms), two
   are dead on both paths (a null `mesh_device` faults in `launch` before any validator, and
   `compute_program_hash` dereferences `global_cb` at `:54` before either), and the one reachable
   check tests only that the GCB has *some* receiver cores. Recommend against restoring it: it would
   charge every dispatch for a guard that catches nothing the hash fix in recommendation 1 does not
   already catch.

   Note also that none of this restores the checks inside `create_at`
   (`dram_prefetcher_validator.cpp:97-101`, `:112-115`, `:122-138`): those run only when a program is
   built, under any hook. The rank, tile-alignment and divisibility invariants all tested values absent
   from the key and so were reachable on a hit, but the right fix for them was hashing the tensor's
   spec fields (recommendation 2), which makes them unreachable by construction rather than paying for
   them per dispatch. **That is what happened**: with `page_config`, `padded_shape` and `memory_config`
   in the key, a call that would violate any of the three necessarily misses and meets the check where
   it lives.
4. **OUTSTANDING for both ops.** If the empty `override_runtime_arguments` bodies are meant to say
   "this op genuinely has no per-dispatch state", say so in a comment that names the invariant being
   relied on (the GCB is fully hashed, so the pegged CB and core ranges cannot change under a hit). As
   written, the comment "Nothing to override — all args are compile-time" states a true fact that does
   not imply the conclusion.

   Post-fab067a the invariant is *true* for the validator — the GCB is hashed by structure and by both
   allocation addresses, so nothing pegged can change under a hit — but it is still unstated, and the
   op now pays a recompile per reallocation for it. For the consumer the invariant remains false, so
   the comment there is not merely under-argued but wrong.
5. **Metal 2.0.** The validator is clear to port. Declare
   `TensorSpecRelaxations{.match_padded_shape_only = true}` for `source_tensor` to preserve the V2
   relaxation; `pertinent_fields` maps that flag to `PertinentFields{.padded_shape = true}`
   (`tt_metal/impl/metal2_host_api/tensor_spec_relaxations.cpp:67-87`), and because
   `hash_tensorspec_with_relaxation` (`:116`) and `tensorspecs_match_with_relaxation` (`:161-201`)
   consume the same field set, the key and `ValidateTensorArgs`' accept/reject predicate
   (`tt_metal/impl/metal2_host_api/program_run_args.cpp:176-189`) cannot disagree. The GCB and the
   three addresses stay in the workload attributes as they are today.

   The consumer is **not** clear to port while omission 1 stands. Porting it unchanged would carry the
   omission into a framework whose `UpdateProgramRunArgs` validates specs strictly but has no view
   into a `GlobalCircularBuffer` attribute the op declines to hash. Fix recommendation 1 first.
