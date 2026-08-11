# Program Cache Audit — `experimental/topk_large_indices`

Audit of `TopkLargeIndicesDeviceOperation::compute_program_hash` against the framework default
("hash everything") key.

| | |
|---|---|
| Device operation | `TopkLargeIndicesDeviceOperation` (`device/topk_large_indices_device_operation.hpp:17`) |
| Custom hash | `device/topk_large_indices_device_operation.cpp:88-101` |
| `operation_attributes_t` | `k`, `valid_length` (`std::optional<uint32_t>`) — `device/topk_large_indices_device_operation_types.hpp:29-36` |
| `tensor_args_t` | `input_tensor` |
| Program factories | `program::TopkLargeIndicesProgramFactory` (single, classic `create` → `CachedProgram`) |
| `override_runtime_arguments` | **Yes**, and complete (`device/topk_large_indices_program_factory.cpp:238-250`) |
| `get_dynamic_runtime_args` | No |
| `validate_on_program_cache_hit` | **Yes**, and non-empty (`device/topk_large_indices_device_operation.cpp:77-80`) — but narrower than the miss validator, which it replaces rather than supplements; see omission 4 |
| Cache-hit patch mechanism | **Op-owned re-derivation (mode A)** |

**Result: no program-cache correctness bug found; one caveat.** This op is the cleanest of the four
audited; its hash omissions are deliberate, documented in the source, and each one is backed by a
matching mechanism (a complete `override_runtime_arguments`, a hit-time validator, and a factory
that builds over the full worker grid so the core set never depends on an unhashed value). The
caveat is the hit validator's narrowness (omission 4), which is safe today but only by an unstated
convention.

## Where the task brief's expectations do not match the code

Several parameters the brief asked about do not exist on this op. Stating that up front avoids
auditing phantoms:

- **No `dim`.** The reduction is always over the last dimension
  (`topk_large_indices_program_factory.cpp:49`: `const uint32_t n = shape[shape.rank() - 1];`).
- **No `largest` / `sorted` flags.** `operation_attributes_t` has exactly two fields, `k` and
  `valid_length`.
- **No sub-core-grid attribute.** The factory unconditionally uses the whole worker grid
  (`:151-153`).
- **Single output, not values + indices.** `tensor_return_value_t` is a single `Tensor`
  (`topk_large_indices_device_operation_types.hpp:42`) holding UINT32 indices; there is no values
  tensor. The name is literal.
- **`k` is hashed** (`topk_large_indices_device_operation.cpp:94`), as the brief expected it should
  be. It has to be: the factory uses the exact `k` (not just its LLK bucket) for
  `output_slices_per_row = k / FACE_WIDTH` and `indices_row_bytes = k * 4` (`:165-167`), both
  compile-time args.

The CSV row itself (explicit hash, SELECTIVE tensor hashing, own hit validator, has
`override_runtime_arguments`, no `get_dynamic_runtime_args`) is accurate.

## Cache-hit patch mechanism

`TopkLargeIndicesProgramFactory` exposes `cached_program_t` + `create` +
`override_runtime_arguments` (`device/topk_large_indices_program_factory.hpp:21-35`), which
satisfies `ProgramFactoryConcept` (`ttnn/api/ttnn/operation_concepts.hpp:25-34`). The framework
wraps it in `MeshWorkloadFactoryAdapter`, whose cache-hit path calls the factory's own override for
every program in the workload:

```285:304:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
        static void override_runtime_arguments(
            cached_mesh_workload_t& cached_workload,
            const operation_attributes_t& attrs,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value) {
            ProgramFactory program_factory;

            for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
                auto& shared_variables = cached_workload.shared_variables.at(coordinate_range);

                mesh_device_operation_utils::apply_override_runtime_arguments(
                    program_factory,
                    program,
                    shared_variables,
                    attrs,
                    *(coordinate_range.begin()),
                    tensor_args,
                    tensor_return_value);
            }
        }
```

**Obligation on the hash.** Every *runtime* arg is re-derived from the live attributes and tensors
on each dispatch, so runtime args impose no obligation at all. What the hash must still cover is
everything baked into the cached `Program`: compile-time args, kernel sources, CB sizes/page
sizes/formats/face geometry, compute-config flags, and core ranges.

The override is not a stub — it calls the same `set_runtime_args` helper the cache-miss path calls,
with the live tensors and attributes:

```238:250:ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/topk_large_indices_program_factory.cpp
void TopkLargeIndicesProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    set_runtime_args(
        cached_program.program,
        cached_program.shared_variables,
        tensor_args.input_tensor,
        tensor_return_value,
        snap_to_llk_target_k(operation_attributes.k),
        operation_attributes.valid_length);
}
```

`set_runtime_args` is the *only* place `create` sets runtime args (`:233`), so by construction the
override reproduces a full rebuild's runtime-arg state. It re-writes all three kernels for every
core in `shared.cores` — including cores that become idle, which receive `rows = 0` rather than
being skipped (`:102-127`). There is no partial-override gap here.

## Baseline: what the default hash would cover

`hash_objects_with_default_seed(type_hash<TopkLargeIndicesDeviceOperation>, attrs, tensor_args)`:

| Source | Fields |
|---|---|
| `operation_attributes` | `k`; `valid_length` (engaged/disengaged plus value) |
| `input_tensor.storage` | storage variant kind (`DeviceStorage`/`HostStorage`, both with empty attribute tuples) |
| `input_tensor.tensor_spec` | `logical_shape`, and `tensor_layout` = { `dtype`, `page_config`, `memory_config`, `alignment` } |

## What the custom hash covers

```88:101:ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/topk_large_indices_device_operation.cpp
ttsl::hash::hash_t TopkLargeIndicesDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input_tensor;
    const auto grid = input.device()->compute_with_storage_grid_size();

    return tt::tt_metal::operation::hash_operation<TopkLargeIndicesDeviceOperation>(
        attrs.k,
        input.dtype(),
        input.layout(),
        input.memory_config().memory_layout(),
        input.memory_config().buffer_type(),
        grid.x,
        grid.y);
}
```

## Two design choices that make the omissions safe

Both are worth naming before the per-parameter verdicts, because most of the verdicts depend on
them.

**(a) Kernels and CBs are created over the entire worker grid, not the active subset.** The factory
says so explicitly:

```151:156:ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/topk_large_indices_program_factory.cpp
    const auto grid = input.device()->compute_with_storage_grid_size();
    const CoreRangeSet all_cores(CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
    const auto cores = corerange_to_cores(all_cores, std::nullopt, true);
    // Runtime row counts are intentionally patched through runtime args instead of the program hash.
    // Create kernels/CBs across the full worker grid so cache hits can use a different active core subset.
```

Core ranges are the one thing `override_runtime_arguments` cannot fix, so making them independent of
the row count is what licenses dropping the shape from the hash.

**(b) The tensor accessors are built from a synthetic interleaved config, not from the buffer.**

```64:67:ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/topk_large_indices_program_factory.cpp
tt::tt_metal::TensorAccessorArgs interleaved_accessor_args(const Tensor& tensor) {
    return tensor.buffer()->is_dram() ? tt::tt_metal::TensorAccessorArgs::create_dram_interleaved()
                                      : tt::tt_metal::TensorAccessorArgs::create_l1_interleaved();
}
```

`create_dram_interleaved()` / `create_l1_interleaved()` leave `buffer_` null
(`tt_metal/impl/buffers/tensor_accessor_args.cpp:138-144`), so `append_to` emits the config word
plus an `aligned_page_size` of **zero** (`:196-205`, `auto aligned_page_size = buffer_ ?
buffer_->aligned_page_size() : 0`). The kernels supply the real page size at *runtime* instead:

```22:24:ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/kernels/reader.cpp
    constexpr auto input_args = TensorAccessorArgs<4>();

    const auto input = TensorAccessor(input_args, src_addr, input_page_bytes);
```

where `input_page_bytes` is runtime arg 5 (`reader.cpp:16`), re-applied on every hit by
`set_runtime_args` (`topk_large_indices_program_factory.cpp:115-120`). The writer does the same
with a `k`-derived compile-time constant (`kernels/writer.cpp:89`, `:95`). The only spec-derived
bit that reaches a compile-time word is the `IsDram` flag — and `buffer_type()` is hashed. This is
the deliberate move that keeps the input's shape, alignment and page size out of the compile-time
surface.

## Omitted parameters

### 1. `attrs.valid_length`

**Verdict: VALID — patched, and a relaxation win.**

The omission is intentional and documented at the declaration:

```31:35:ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/topk_large_indices_device_operation_types.hpp
    // Restrict the search to the first `valid_length` columns of each row instead of the full last
    // dimension. Lets top-k run over the real prefix of an over-allocated row (whose tail may be stale)
    // without physically slicing the input. nullopt = search the full width. Runtime-only (hash-excluded,
    // validated on cache hit) so a serving loop growing valid_length reuses one program.
    std::optional<uint32_t> valid_length{};
```

It reaches the program only through three runtime args:

```45:62:ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/topk_large_indices_program_factory.cpp
RuntimeShapeArgs get_runtime_shape_args(
    const Tensor& input, LlkTargetK llk_target_k, std::optional<uint32_t> valid_length) {
    const uint32_t llk_k = to_uint32(llk_target_k);
    const auto& shape = input.logical_shape();
    const uint32_t n = shape[shape.rank() - 1];
    // Number of columns to actually read and scan per row. Defaults to the full physical width n; a
    // valid_length bounds it to the real prefix so the stale tail is never read or ranked. The row STRIDE
    // (input_row_bytes) stays n so per-row addressing is unchanged — only how much we pull from each row shrinks.
    const uint32_t search_len = valid_length.value_or(n);
    const uint32_t num_chunks = tt::div_up(search_len, llk_k);
    const uint32_t tail_elements = search_len - ((num_chunks - 1) * llk_k);
    return RuntimeShapeArgs{
        .num_rows = flattened_rows_excluding_last_dim(shape),
        .num_chunks = num_chunks,
        .tail_elements = tail_elements,
        .input_tail_chunk_bytes = tail_elements * input.element_size(),
        .input_row_bytes = n * input.element_size()};
}
```

`num_chunks` and `input_tail_chunk_bytes` go to reader runtime args 3 and 4; `num_chunks` and
`tail_elements` go to compute runtime args 1 and 2 (`:111-122`). All are re-applied by
`override_runtime_arguments`. Nothing about `valid_length` touches a compile-time arg or a CB
size — `input_chunk_bytes` and `tiles_per_chunk` come from `llk_k`, which comes from the hashed `k`
(`:161-162`, `:149`).

The relaxation is the point of the feature: an autoregressive serving loop that calls
`topk_large_indices(logits, k, valid_length=seq_len)` with `seq_len` incrementing each step gets one
program instead of one per step. Under the default hash every step would be a recompile.

The op also re-checks it on every hit, so a `valid_length` that violates the kernel's assumptions
cannot ride in on a stale entry:

```64:72:ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/topk_large_indices_device_operation.cpp
    if (attrs.valid_length.has_value()) {
        const uint32_t valid_length = attrs.valid_length.value();
        TT_FATAL(valid_length >= attrs.k, "topk_large_indices valid_length {} must be >= k {}", valid_length, attrs.k);
        TT_FATAL(
            valid_length <= n,
            "topk_large_indices valid_length {} must be <= the input last dimension {}",
            valid_length,
            n);
    }
```

### 2. `input_tensor.logical_shape()`

**Verdict: VALID — patched, and a relaxation win.**

Also documented as intentional (`topk_large_indices_device_operation.cpp:40-41`: *"Shape is
intentionally omitted from the program hash and patched through runtime args, so keep these checks
on both cache miss and cache hit"*).

The shape reaches the program through exactly four quantities, all runtime args:
`num_rows` → `start_row`/`rows` per core, `input_row_bytes`, `input_tail_chunk_bytes`, `num_chunks`.

```111:124:ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/topk_large_indices_program_factory.cpp
        tt::tt_metal::SetRuntimeArgs(
            program,
            shared.reader_kernel_id,
            core,
            {input.buffer()->address(),
             start_row,
             rows,
             runtime_args.num_chunks,
             runtime_args.input_tail_chunk_bytes,
             runtime_args.input_row_bytes});
        tt::tt_metal::SetRuntimeArgs(
            program, shared.compute_kernel_id, core, {rows, runtime_args.num_chunks, runtime_args.tail_elements});
        tt::tt_metal::SetRuntimeArgs(
            program, shared.writer_kernel_id, core, {indices.buffer()->address(), start_row, rows});
```

The work split is recomputed inside `set_runtime_args` from the *current* `num_rows` (`:92-98`), so
a hit with a different row count redistributes correctly. Design choice (a) above is what makes that
safe: the active core subset changes, but the *program's* core ranges do not. And design choice (b)
keeps the row byte length out of the compile-time accessor args.

Two shape-derived checks are re-run on every hit rather than being hashed
(`topk_large_indices_device_operation.cpp:45-60`, reached from `validate_on_program_cache_hit`):
rank ≥ 1, `num_rows > 0`, `n >= k`, `n <= 2^30`, and the row-bytes `uint32_t` bound. Those are
exactly the assumptions the frozen compile-time args and the CB sizing depend on.

The benefit: a decode loop whose batch dimension varies, or a `[1, 1, B, N]` logits tensor whose `B`
changes between prefill and decode, shares one program. The default hash would recompile for each.

### 3. `input_tensor.memory_config()` beyond `memory_layout` and `buffer_type`

**Verdict: VALID — pinned by validation, via a hashed discriminator.**

`MemoryConfig` also carries `shard_spec`, `nd_shard_spec`, `created_with_nd_shard_spec` and
`per_core_allocation` (`tt_metal/api/tt-metalium/experimental/tensor/spec/memory_config/memory_config.hpp:49-55`);
the hash keeps only the first two attributes. The factory rejects sharded inputs:

```34:34:ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/topk_large_indices_device_operation.cpp
    TT_FATAL(!input.is_sharded(), "topk_large_indices input must use interleaved memory");
```

That check lives in `validate_static_args`, which runs on cache *miss* only (`:82-86`, and see the
next subsection for why "miss only" is the exact truth here) — but it does not need to run on hits,
because the discriminator it tests is itself hashed. `is_sharded()` is a property of
`memory_layout`, and `memory_layout` is in the key, so a sharded input cannot reach an interleaved
entry: it produces a different hash, misses, and hits the `TT_FATAL`. With
`memory_layout == INTERLEAVED` pinned, the shard-spec optionals are all `nullopt` and carry no
information.

### 4. What the hand-written hit validator costs

**Verdict: CAVEAT — sound as written, but structurally fragile.**

This op defines `validate_on_program_cache_hit`, and doing so does not *add* a validator on the hit
path, it *replaces* the miss one:

```262:266:ttnn/api/ttnn/device_operation.hpp
    if constexpr (HasValidateOnProgramCacheHit<mesh_device_operation_t>) {
        mesh_device_operation_t::validate_on_program_cache_hit(operation_attributes, tensor_args);
    } else {
        mesh_device_operation_t::validate_on_program_cache_miss(operation_attributes, tensor_args);
    }
```

Since the hit hook calls only `validate_runtime_args` (`:77-80`) while the miss hook calls both
(`:82-86`), the hit path loses every check in `validate_static_args`
(`:19-35`): the `k` bounds and 16-divisibility, `arch == BLACKHOLE`, `layout() == ROW_MAJOR`,
`dtype() == BFLOAT16`, and `!is_sharded()`. Had the op defined no hit validator at all, all five
would run on every dispatch.

Losing them is harmless here, and for a precise reason: **each of the five tests a value that cannot
vary within one cache entry.** Four of them — `k`, `layout()`, `dtype()` and `memory_layout()`
(behind `is_sharded()`) — appear literally in the hash at `:94-97`, so a configuration that would
fail the check cannot produce a matching key: it misses, and on a miss `validate_static_args` runs
and rejects it. The fifth, `arch`, is read from the process-global HAL singleton
(`hal::get_arch()`, `:29`) rather than from the arguments, and the program cache is per mesh device,
so it cannot differ between the miss that built an entry and any later hit on it. The source states
this partition explicitly (`:22-23` and `:40-41`), and it is the right partition.

The caveat is that the property holding it up is unstated and easy to violate. Any check added to
`validate_static_args` in future that tests an *unhashed* value would be silently dead on the hit
path — no compiler error, no test failure, just a guard that stops running after the first call on
each key. The safe convention, if the split is kept, is that `validate_static_args` may only
reference values that `compute_program_hash` reads. See recommendation 4.

### 5. `input_tensor.tensor_spec().page_config()` — the tile geometry

**Verdict: VALID — invariant.**

`layout()` collapses `PageConfig` to `ROW_MAJOR` vs `TILE`, discarding tile shape and face geometry.
This op does contain the surface pattern of the unguarded-32x32 defect — bare architectural
constants used as tile arithmetic, with no check on the tensor's actual `tile().get_height()` /
`get_width()` anywhere:

```146:149:ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/topk_large_indices_program_factory.cpp
    const uint32_t k = operation_attributes.k;
    const auto llk_target_k = snap_to_llk_target_k(k);
    const uint32_t llk_k = to_uint32(llk_target_k);
    const uint32_t tiles_per_sequence = (llk_k + tt::constants::TILE_HW - 1) / tt::constants::TILE_HW;
```

```161:166:ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/topk_large_indices_program_factory.cpp
    const uint32_t input_chunk_bytes = llk_k * input.element_size();
    const uint32_t input_tile_bytes = tt::constants::TILE_HW * input.element_size();
    constexpr uint32_t row_slice_elements = tt::constants::FACE_WIDTH;
    const uint32_t source_slices_per_row = llk_k / row_slice_elements;
    const uint32_t output_slices_per_row = k / row_slice_elements;
    const uint32_t indices_slice_bytes = row_slice_elements * indices.element_size();
```

A repo-wide sweep files this op as an unguarded-32x32 case on the strength of exactly these lines.
Checked against the actual criteria, it is not one, and the reason is structural rather than
incidental.

**The first criterion fails: the op does not accept `Layout::TILE`.**

```32:32:ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/topk_large_indices_device_operation.cpp
    TT_FATAL(input.layout() == Layout::ROW_MAJOR, "topk_large_indices input must be ROW_MAJOR");
```

That pin is enforced on the only path where it can matter. It lives in `validate_static_args`, so by
the substitution branch quoted in omission 4 it runs on misses only — but `input.layout()` is hashed
(`:96`), so a `TILE` input cannot land on an existing entry, and the miss it forces is precisely
where the `TT_FATAL` fires. The output is likewise pinned: `compute_output_specs` constructs it with
`PageConfig(Layout::ROW_MAJOR)` (`:116`), so it is never caller-supplied.

**And a row-major tensor has no tile to lose.** `PageConfig::get_tile()` returns a
default-constructed `Tile{}` on the row-major alternative, so even a rewrite that made this factory
read `tensor_spec().tile()` would see 32x32 with 16x16 faces on every admissible input:

```179:184:tt_metal/impl/tensor/spec/layout/page_config.cpp
Tile PageConfig::get_tile() const {
    if (const auto* tile_config = std::get_if<TilePageConfig>(&config_)) {
        return tile_config->tile;
    }
    return Tile{};
}
```

So the constants above are not a truncated view of a tensor property. `TILE_HW` and `FACE_WIDTH`
here describe the LLK compute engine's DEST register and face geometry — the units the top-k LLK and
`pack_untilize` operate in — and they are applied to `llk_k`, which derives from the hashed `k`, not
to any shape or page size of the input. The input has no tiles at all; the reader streams row bytes
into a CB that the compute engine happens to consume in tile-shaped units.

Two consequences worth stating. The omission of `page_config` is correct on two independent grounds
at once (nothing reads the tensor's tile, and there is no tensor tile to read), so a later relaxation
of the `ROW_MAJOR` requirement would not open a cache hole through this path — it would fail loudly
at the `TT_FATAL` instead, and closing it properly would then mean making the factory tile-aware
*and* adding `page_config` to the hash in the same change. And the only route by which a `TILE`
input could still reach a cached row-major program is a raw 64-bit hash collision, which the custom
hash makes unrecoverable; see "Framework side effect" below.

### 6. `input_tensor.tensor_layout().get_alignment()`

**Verdict: VALID — unused.**

For a `ROW_MAJOR` interleaved tensor, `Alignment` would normally influence the program through the
buffer's page size. Design choice (b) removes that path entirely: the accessor's compile-time page
size is a literal `0` and the real page length is a runtime arg computed as
`n * input.element_size()` from the logical shape and dtype. `element_size()` follows from the
hashed `dtype`. No other quantity in the factory reads alignment.

This one is worth a guard — see recommendation 1.

### 7. `input_tensor.storage` variant kind (device vs host)

**Verdict: VALID — pinned by validation.**

```42:43:ttnn/cpp/ttnn/operations/experimental/topk_large_indices/device/topk_large_indices_device_operation.cpp
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "topk_large_indices input must be on device");
    TT_FATAL(input.buffer() != nullptr, "topk_large_indices input must have an allocated buffer");
```

These are in `validate_runtime_args`, so unlike most ops' storage checks they run on **both** miss
and hit. Constant across every admissible call.

### 8. Buffer addresses (omitted by the default hash too)

**Verdict: VALID — patched, and required.**

Both the input and the output address are re-applied on every hit as runtime args 0 of the reader
and writer respectively (`topk_large_indices_program_factory.cpp:115`, `:124`). No `Buffer*`
bindings and no `resolve_bindings` are involved — mode A bypasses that machinery entirely, which
also means the input-aliasing bail-out that trips up fast-path ops is not a concern here.

### 9. The output tensor

**Not an omission** — `tensor_return_value_t` is not part of the default key either (the framework
hashes only `attrs` and `tensor_args`). Recording it here because the brief asked: the output's spec
is a pure function of hashed values. `compute_output_specs` builds it from the input's logical shape
with the last dimension replaced by `k`, `DataType::UINT32`, `ROW_MAJOR`, and the input's
`memory_config` (`topk_large_indices_device_operation.cpp:103-117`). So `indices.element_size()` is
the constant 4, `indices_row_bytes = k * 4` follows from hashed `k`, and
`interleaved_accessor_args(indices)`'s `IsDram` bit follows from the hashed input `buffer_type`. Its
address is patched (writer runtime arg 0).

## Keys the custom hash adds beyond the default

- `grid.x`, `grid.y` from `input.device()->compute_with_storage_grid_size()`. Not in the default
  key. Because kernels and CBs span the whole grid (`:152`), the grid dimensions are genuinely a
  structural property of the program, so hashing them is correct. It is also largely redundant —
  the program cache is per mesh device — but it is defensive rather than harmful, and it documents
  the dependency.

## Framework side effect of having a custom hash

```1012:1014:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
        if constexpr (requires { DeviceOperation::compute_program_hash(attrs, tensor_args); }) {
            return key;  // custom hash -> opt out beyond the op-identity prefix
        } else {
```

`ProgramCacheKey::canonical` degrades to just the op type name, so a 64-bit collision between two
different `topk_large_indices` configurations becomes a wrong hit rather than a rebuild. The hashed
set here is small (`k`, dtype, layout, two memory-config enums, two grid dimensions), so the number
of distinct keys in practice is tiny and the collision risk is negligible — but it is worth noting
that the hit-time `validate_runtime_args` would not catch a collision either, since a colliding `k`
would still satisfy `n >= k`.

## Summary

| Omitted vs. default | Used by program? | Patched on hit? | Verdict |
|---|---|---|---|
| `attrs.valid_length` | Yes — 3 runtime args | Yes (`override_runtime_arguments`) | VALID — patched (also a relaxation win) |
| `input.logical_shape` | Yes — work split + 4 runtime args | Yes (`override_runtime_arguments`) | VALID — patched (also a relaxation win) |
| `input.memory_config` (shard specs) | No (interleaved only) | n/a | VALID — pinned by validation |
| Checks dropped by the hand-written hit validator | n/a | n/a | CAVEAT — sound only because all five test hashed values |
| `input.page_config` (`Tile`, faces) | No — the constants are LLK geometry, and a row-major tensor has no tile | n/a | VALID — invariant |
| `input.tensor_layout.alignment` | No (page size is a runtime arg) | n/a | VALID — unused |
| `input.storage` kind | n/a | n/a | VALID — pinned by validation (checked on hits too) |
| Buffer addresses | Yes | Yes (runtime args) | VALID — patched |

**No program-cache correctness bug was found; the bug count stands at zero, with one caveat.** Every
compile-time arg, CB page size, CB face geometry, compute-config flag, kernel source and core range
is a function of the hashed set {`k`, input `dtype`, input `layout`, input `memory_layout`, input
`buffer_type`, grid dimensions}. Everything that depends on an unhashed value — the shape and
`valid_length` — appears only in runtime args, and `override_runtime_arguments` re-derives all of
them by calling the same `set_runtime_args` helper the miss path uses.

Two verdicts deserve emphasis because they run against first impressions. The op's bare
`TILE_HW` / `FACE_WIDTH` arithmetic is not the unguarded-32x32 defect: those constants describe the
compute engine, the input is pinned to `ROW_MAJOR` by a check whose discriminator is hashed, and a
row-major tensor carries no tile in the first place (omission 5). And the op's hand-written hit
validator, which looks like extra safety, in fact removes five checks from the hit path; it is sound
only because each of the five tests a hashed value, which is a property no mechanism enforces
(omission 4).

## Recommendations

1. Add a comment (or a static assertion in a test) at `interleaved_accessor_args`
   (`device/topk_large_indices_program_factory.cpp:64-67`) recording *why* it uses
   `create_dram_interleaved()` / `create_l1_interleaved()` rather than
   `TensorAccessorArgs(*tensor.buffer())`. Switching to the buffer overload would look like a
   harmless simplification but would move `aligned_page_size` into the compile-time args and
   silently invalidate omissions 2 and 5 — the shape and alignment would then have to be hashed.
   This is the single most fragile assumption the op rests on.
2. Consider building this op with `-DTT_DESCRIPTOR_PATCHING_PARITY_CHECK` in CI. Mode-A ops go
   through the parity oracle at `ttnn/api/ttnn/mesh_device_operation_adapter.hpp:679-693` only for
   descriptor factories, not for `ProgramFactoryConcept` ones, so this op currently has no automatic
   regression net proving its override matches a rebuild. A targeted test that runs the op twice
   with different shapes and different `valid_length` values against a golden implementation would
   serve the same purpose.
3. `grid.x` / `grid.y` in the hash are redundant with the per-device cache. Keep them (they document
   the full-grid dependency introduced at `:152`), but if the op ever gains a sub-grid attribute,
   that attribute must be hashed too — core ranges are the one thing `override_runtime_arguments`
   cannot repair.
4. Record the invariant that makes the miss/hit validation split safe, at
   `validate_static_args` (`device/topk_large_indices_device_operation.cpp:19-23`): *every check in
   this function must test a value that `compute_program_hash` reads*. The existing comment says the
   checks "do not need to be rechecked on every cache hit", which states the conclusion but not the
   premise. Because defining `validate_on_program_cache_hit` replaces the miss validator rather than
   supplementing it (`ttnn/api/ttnn/device_operation.hpp:262-266`), a future check added here that
   tests an unhashed value would never run on a hit, with nothing to signal it.
5. If the `ROW_MAJOR` restriction is ever lifted, the `page_config` omission has to be revisited in
   the same change: the factory's `TILE_HW` / `FACE_WIDTH` arithmetic would then need to come from
   `tensor_spec().tile()`, and `page_config` would need to enter the hash, since the program would
   genuinely vary with the tile. Adding a `tile().get_height()` / `get_width()` guard is the cheaper
   alternative if only tiled *storage* is wanted rather than tiled geometry.
