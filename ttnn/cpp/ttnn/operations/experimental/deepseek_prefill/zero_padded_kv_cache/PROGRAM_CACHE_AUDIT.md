# Program Cache Audit — `experimental/deepseek_prefill/zero_padded_kv_cache`

Audit of
`ttnn::operations::experimental::deepseek_prefill::zero_padded_kv_cache::ZeroPaddedKvCacheDeviceOperation::compute_program_hash`
against the framework default ("hash everything") key.

| | |
|---|---|
| Device operation | `ZeroPaddedKvCacheDeviceOperation` (`device/zero_padded_kv_cache_device_operation.hpp:21`) |
| Custom hash | `device/zero_padded_kv_cache_device_operation.cpp:249-272` (post-fab067a; `:212-231` pre-fix, which is what the omission analysis below cites) |
| `operation_attributes_t` | `slot_idx`, `valid_global`, `chunk_size_global`, `pad_align`, `layer_idx`, `num_layers`, `cluster_axis` |
| `tensor_args_t` | `cache`, `std::optional<Tensor> slot_idx`, `std::optional<Tensor> valid_global` |
| Program factories | one: `ProgramFactory::create_descriptor` (`ProgramDescriptor`-based), with two internal layout branches — a dataflow-only ROW_MAJOR writer, and a TILE reader/compute/writer trio |
| `override_runtime_arguments` | **Yes**, on `MeshWorkloadFactory` (`device/zero_padded_kv_cache_device_operation.cpp:409-446`) |
| `get_dynamic_runtime_args` | **No** |
| `validate_on_program_cache_hit` | **Yes** (`device/zero_padded_kv_cache_device_operation.cpp:197-200`) — so it *replaces* the miss validator on hits rather than supplementing it |
| Validator actually run on a hit | `validate_runtime_args` only (`:66-145`); everything in `validate_on_program_cache_miss` before its delegation at `:194` is skipped |
| Cache-hit patch mechanism | **Op-owned override** at the device-operation level, implemented internally as the framework **buffer-binding fast path** plus a hand-written common-runtime-arg patch |
| In-place | Yes — `create_output_tensors` returns the `cache` tensor itself; there is no separate input |

## Post-fix status — commit fab067a

**Verdict: CLEAR — with one justified relaxation** (`cache.logical_shape`, plus the two per-request
scalars, which were always correct). Both program-cache bugs this audit found are closed, and both
caveats are regraded down. No program-cache correctness bug remains.

**Placement is the whole story for this op.** Both reproductions below are cache **hits** — call 1 is a
miss that compiles and caches, call 2 is the identical-keyed call that zeroes the wrong region of the
cache. A guard placed in `validate_on_program_cache_miss` would have run on call 1, passed it, and then
**not run at all** on call 2, because this op defines `validate_on_program_cache_hit` and the dispatcher
substitutes rather than supplements (see `### Which validator runs on a cache hit`). This op already
demonstrated the trap: its one explicit `buffer_type() == DRAM` pin for the cache sits in the miss
validator and consequently stops running after the first call. Every guard fab067a added went into the
shared `validate_runtime_args` — the only placement that works.

**What fab067a changed in this op** (line citations in this section are against the post-fix file; the
analysis further down keeps its original pre-fix citations)

- Added a 32x32 tile assertion to `validate_runtime_args`
  (`device/zero_padded_kv_cache_device_operation.cpp:71-86`), guarded by `layout() == Layout::TILE` so
  it only constrains the branch that reads a tile. This closes omission 5.
- Added the metadata tensor's memory config to the key
  (`compute_program_hash:268-271`, as `slot_idx.has_value() ? slot_idx->memory_config() : MemoryConfig{}`).
  This closes the hash half of omission 4.
- Added the `!meta.is_sharded()` check this op was missing relative to its sibling, inside
  `validate_meta` (`:102-108`), and a mutual-config assertion that `valid_global`'s memory config equals
  `slot_idx`'s (`:118-125`). Together these close omission 4's second half — the reader and writer each
  bake **one** `TensorAccessorArgs` built from `slot_idx` and use it for both metadata reads, and
  nothing previously required the two tensors to be interchangeable.
- Moved `cache.storage_type() == StorageType::DEVICE` into `validate_runtime_args` (`:136-141`),
  specifically because `cache.device()` is dereferenced on the next line (`:142`) — on a hit the fault
  used to land there as a null dereference rather than as a message. The miss validator now carries a
  pointer comment at the site it came from (`:194`).
- Documented the 32x32 TILE requirement in the nanobind docstring
  (`zero_padded_kv_cache_nanobind.cpp:49-52`).

**What remains open**

- The two per-request scalars (`valid_global`, `slot_idx`) stay out of the key by design, re-applied to
  every kernel on every hit and re-validated on every hit. Unchanged and correct.
- `cache.alignment` (omission 6) is still not hashed. **Regraded to
  `VALID — determined by hashed terms`**: with the tile now pinned to 32x32, the TILE branch's page size
  is `tt::tile_size` of the hashed dtype, and the ROW_MAJOR `row_page_size` is the last padded
  dimension's byte extent — both functions of the hashed {`dtype`, `layout`, `memory_config`,
  `padded_shape`}. No public Python path exposes a non-canonical `Alignment` independently of those.
- `cache.buffer()->buffer_type() == BufferType::DRAM` (`:195`) is still miss-only. That stays benign:
  the buffer type is carried by the hashed `cache.memory_config()`, so a DRAM-to-L1 change misses and
  meets the pin on the miss path.
- The op still defines `validate_on_program_cache_hit` (`:234`), so the miss validator is still
  replaced rather than supplemented, and everything above the delegation at `:231` remains miss-only.
  Post-fix that block is entirely unreachable — see the regraded reachability table below.
- The override's raw kernel-handle selection (`{0u, 2u}` at `:466`, `0..num_kernels` at `:476`) is
  unchanged. It is a factory-coupling defect, not a cache defect; see
  `## Non-cache correctness defects`.
- Defining a custom hash still forfeits attribute-level collision resolution
  (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:1012-1014`), and this op has no `TensorSpec`
  backstop on the hit path, so a 64-bit collision is still silent destruction of a shared cache.
  Unchanged.

**Metal 2.0 port**

Clear to port. The op does carry a genuine logical-vs-padded relaxation — omission 3, kept because the
kernels address the cache in pages and the page grid is the padded shape. The value is thinner here than
in `update_padded_kv_cache`, since a KV cache is allocated once and its logical extent rarely varies
across calls, but it is real: two caches whose logical sequence lengths are 4090 and 4096 both pad to
4096, share the page grid and every compile-time arg, and should share one program. That relaxation maps
exactly onto `TensorSpecRelaxations::match_padded_shape_only`
(`tt_metal/api/tt-metalium/experimental/metal2_host_api/tensor_spec_relaxations.hpp:41,49`), which
`pertinent_fields` resolves to `PertinentFields{.padded_shape = true}`
(`tt_metal/impl/metal2_host_api/tensor_spec_relaxations.cpp:67-87`). Declaring it on the `cache`
`TensorParameter` is the correct port: both `hash_tensorspec_with_relaxation` (`:116`) and
`tensorspecs_match_with_relaxation` (`:161-201`) consume that one field set, and `ValidateTensorArgs`
delegates its accept/reject to the predicate
(`tt_metal/impl/metal2_host_api/program_run_args.cpp:176-189`), so the key and validation cannot
disagree. What must **not** happen is the pre-fix `rotary_embedding_indexed` combination — hashing
`padded_shape` while leaving the relaxations default — which relaxes the key and then validates
strictly, turning an intended rebuild into a throw from `report_tensor_arg_mismatch`. See that op's
audit, `### The report_tensor_arg_mismatch interaction`, for the model pattern.

The metadata tensors should get default (exact) relaxations rather than
`match_padded_shape_only`: nothing about them is padded, and the accessor the kernels bake depends on
their whole layout. Under a `TensorParameter`-based port, the exact predicate would make omissions 4, 5
and 6 unreachable by construction, since `tensor_layout` is compared under every relaxation. The
`TT_FATAL`s fab067a added stay worth keeping regardless: they name the offending operand, and they do
not depend on nobody ever setting an extra relaxation flag.

## Cache-hit patch mechanism

The op sits at the junction of two mechanisms and both have to be read to classify it.

**Outer layer.** `select_program_factory` always returns `MeshWorkloadFactory`
(`device/zero_padded_kv_cache_device_operation.cpp:149-152`). That factory defines
`override_runtime_arguments` and not `apply_descriptor`, so the framework's cache-hit dispatcher hands
control to the op on every hit:

```279:285:ttnn/api/ttnn/device_operation.hpp
        if constexpr (requires { &WorkloadFactory::apply_descriptor; }) {
            WorkloadFactory::apply_descriptor(
                cached_mesh_workload, operation_attributes, tensor_args, tensor_return_value);
        } else {
            WorkloadFactory::override_runtime_arguments(
                cached_mesh_workload, operation_attributes, tensor_args, tensor_return_value);
        }
```

**Inner layer.** The override delegates address patching to the descriptor adapter and then writes the
per-call scalars itself:

```409:445:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/zero_padded_kv_cache/device/zero_padded_kv_cache_device_operation.cpp
void ZeroPaddedKvCacheDeviceOperation::MeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& args,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    descriptor_adapter_t::apply_descriptor(cached_workload, args, tensor_args, output);
    ...
    if (tensor_args.slot_idx.has_value()) {
        const uint32_t slot_idx_addr = static_cast<uint32_t>(tensor_args.slot_idx->buffer()->address());
        const uint32_t valid_global_addr = static_cast<uint32_t>(tensor_args.valid_global->buffer()->address());
        for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
            for (uint32_t kernel_handle : {0u, 2u}) {  // reader, writer (metadata path is TILE-only)
                auto& common = GetCommonRuntimeArgs(program, kernel_handle);
                TT_FATAL(
                    kValidGlobalAddrCommonArgIdx < common.size(),
                    "zero_padded_kv_cache kernel missing the metadata-tensor addr common args");
                common[kSlotIdxAddrCommonArgIdx] = slot_idx_addr;
                common[kValidGlobalAddrCommonArgIdx] = valid_global_addr;
            }
        }
    } else {
        const uint32_t num_kernels = tensor_args.cache.layout() == Layout::ROW_MAJOR ? 1u : 3u;
        for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
            for (uint32_t kernel_handle = 0; kernel_handle < num_kernels; ++kernel_handle) {
                auto& common = GetCommonRuntimeArgs(program, kernel_handle);
                TT_FATAL(
                    kSlotIdxCommonArgIdx < common.size(), "zero_padded_kv_cache kernel missing per-call common args");
                common[kValidGlobalCommonArgIdx] = args.valid_global;
                common[kSlotIdxCommonArgIdx] = args.slot_idx;
            }
        }
    }
}
```

`descriptor_adapter_t` is `DescriptorMeshWorkloadAdapter<ProgramFactory>` over
`DescriptorAdapterOperation`, a four-typedef helper (`device/zero_padded_kv_cache_device_operation.hpp:70-82`).
Neither the helper nor `ProgramFactory` declares `override_runtime_arguments` or
`get_dynamic_runtime_args`, so the inner adapter's `has_override_runtime_arguments()` is false and its
`apply_descriptor` takes the buffer-binding branch:

```726:731:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
                    if (!sv.resolved_bindings.rt_args.empty() ||
                        (!dynamic_args.empty() && !sv.resolved_bindings.empty())) {
                        auto collected =
                            collect_tensor_buffers(tensor_args, tensor_return_value, sv.workload_descriptor);
                        tt::tt_metal::apply_resolved_bindings(program, sv.resolved_bindings, collected.buffers);
                        tt::tt_metal::apply_dynamic_runtime_args(program, dynamic_args);
```

`rt_args` is non-empty because every kernel registers the cache as a `Buffer*` binding
(`create_descriptor:310,360,393`), so the slow-path rebuild is never taken.

**Obligation on the hash.** A cache hit refreshes exactly: the cache buffer address on all kernels; and
common args 3/9 (scalar path) or 10/11 (metadata path) on the kernels the override enumerates.
Everything else — common args 0, 1, 2, 4, 5, 6, 7, 8, every compile-time arg, all four CB page sizes
and totals, the single-core `CoreRangeSet`, and which of the two layout branches was compiled — is
frozen at the first miss and must be a function of the hashed set.

Two consequences of that worth stating up front, because they are correct decisions rather than
omissions: `chunk_size_global` and `pad_align` are per-request-looking values that end up in common
args 2 and 4 (`create_descriptor:273-286`) and are **not** patched by the override, so hashing them is
mandatory — and the op does hash both. Similarly `layer_idx` and `num_layers` land in common args 5/6
unpatched, and both are hashed.

### Which validator runs on a cache hit

The dispatcher runs exactly one validator on a hit, and which one is chosen has the opposite effect
from the intuitive reading:

```262:266:ttnn/api/ttnn/device_operation.hpp
    if constexpr (HasValidateOnProgramCacheHit<mesh_device_operation_t>) {
        mesh_device_operation_t::validate_on_program_cache_hit(operation_attributes, tensor_args);
    } else {
        mesh_device_operation_t::validate_on_program_cache_miss(operation_attributes, tensor_args);
    }
```

An op that defines no hit validator gets the miss validator substituted on hits, so all of its pins
hold. **This op defines one**, so the miss validator does not run on a hit at all — and this op's hit
validator is a bare delegation, with not even a comment:

```197:200:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/zero_padded_kv_cache/device/zero_padded_kv_cache_device_operation.cpp
void ZeroPaddedKvCacheDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_runtime_args(args, tensor_args);
}
```

The miss validator also ends by delegating to `validate_runtime_args` (`:194`), so the two paths differ
by exactly the checks the miss validator performs *before* that delegation — lines 157 through 193. The
hit path therefore loses all of the following:

- `cache.storage_type() == DEVICE` (`:157`). **fab067a moved this into `validate_runtime_args`**, and
  it turns out never to have been a real loss anyway — see the framework correction under omission 7.
- `cache.buffer()->buffer_type() == BufferType::DRAM` (`:158`). This one is not load-bearing for the
  cache key, since `cache.memory_config()` is hashed and carries the buffer type — but it is worth
  noting that the op's one explicit buffer-type pin is also its one pin that never runs twice. Still
  miss-only post-fix, and still benign for that reason.
- The TILE-or-ROW_MAJOR gate and the ROW_MAJOR dtype and FP8_E4M3 layout gates (`:159-170`).
- The metadata-path-is-TILE-only guard (`:174-179`). This is the sharpest loss in the list: it is the
  check that keeps the metadata path off the ROW_MAJOR program, and it is absent on every hit.
- The rank-4 check, the num-heads-is-1 check, `num_layers > 0`, the batch-divisibility check, and the
  `layer_idx < num_layers` range check (`:181-193`).

What *does* run on both paths is `validate_runtime_args` (`:66-145`): the `cluster_axis` check, the
paired-optional check, the `validate_meta` lambda for both metadata tensors (`:83-93`), the scalar-path
`slot_idx` range check, the 2D-mesh check, and the whole `chunk_size_global` / `pad_align` /
`chunk_local` / capacity block. Note that pre-fix this op's `validate_meta` was *thinner* than
`update_padded_kv_cache`'s — it omitted the `!meta.is_sharded()` check that its sibling had — which was
the gap called out in omission 4. **fab067a added it** (`:102-108`), along with the mutual-config
assertion, so the two ops' metadata validators are now equivalent.

Pre-fix, this was the reason omission 7 below was graded `CAVEAT — pinned only on the miss path` rather
than `VALID — pinned by validation`. It is also the reason every guard recommended at the end of this
document is specified to go into `validate_runtime_args` — a guard added to
`validate_on_program_cache_miss` would never run on the offending second call, which is the only call
that matters for a cache bug.

**fab067a acted on exactly that.** The new tile assertion, the new metadata checks and the relocated
`cache` storage pin all went into `validate_runtime_args`, not the miss validator. The structural
hazard — a narrow hit validator that disables everything above it — is unchanged; what changed is that
nothing load-bearing is left above the delegation.

**Which of the dropped checks are actually reachable — for this op, almost none.** The list above is
the mechanical diff, but a miss-only pin on a value that is itself in the cache key cannot be evaded:
any call carrying a new value of that parameter misses, and the miss validator runs and rejects it
there. This op's hash is unusually broad — it covers `cache.dtype()`, `cache.layout()`,
`cache.memory_config()`, `cache.padded_shape()`, `layer_idx`, `num_layers`, `chunk_size_global`,
`pad_align` and `slot_idx.has_value()` (`compute_program_hash:220-230`) — so filtering the dropped list
against it leaves exactly one line:

| Dropped check | Constrains | In the key? | Reachable on a hit? |
|---|---|---|---|
| `cache.storage_type() == DEVICE` (`:157`) | storage variant kind | No | No — pinned by the framework in `launch()`; see omission 7. Also **moved into `validate_runtime_args` by fab067a** |
| `cache.buffer()->buffer_type() == DRAM` (`:158`) | buffer type | Yes, inside `cache.memory_config()` | No |
| Layout gate, ROW_MAJOR dtype gate, FP8 layout gate (`:159-170`) | `cache.layout()`, `cache.dtype()` | Yes, both | No |
| Metadata-path-is-TILE-only (`:174-179`) | `cache.layout()` and `slot_idx.has_value()` | Yes, both | No |
| Rank-4, num-heads-is-1 (`:181-182`) | `cache.padded_shape()` | Yes | No |
| `num_layers > 0`, batch divisibility, `layer_idx < num_layers` (`:183-193`) | `num_layers`, `layer_idx`, `cache.padded_shape()` | Yes, all three | No |

The metadata-path-is-TILE-only guard deserves a specific note, because it looks like the most dangerous
loss and is not one. Both values it constrains — `cache.layout()` and `slot_idx.has_value()` — are in
the key, so the combination (ROW_MAJOR, metadata) has its own cache slot. Its first occurrence is
necessarily a miss, the full miss validator runs, and the call is rejected there. It is impossible to
reach a hit on that key without having already been rejected on it. The same argument disposes of the
kernel-handle coupling in the override: the override's `{0u, 2u}` metadata-path selection rests on this
guard, and the guard is effectively enforced despite living in the miss validator.

So the only genuinely reachable dropped check in this op was the single `storage_type()` pin at `:157`,
and its failure mode was a crash rather than silent corruption. That conclusion drove the
recommendations at the end of this document. **Post-fix the table is empty of reachable losses**: that
one row is both relocated into `validate_runtime_args` by fab067a (`:136-141`) and, independently,
pinned by the framework in `launch()` on every dispatch — see `#### Framework correction` under
omission 7.

## Baseline: what the default hash would cover

`hash_objects_with_default_seed(type_hash<ZeroPaddedKvCacheDeviceOperation>, attrs, tensor_args)`
walks reflection, giving:

| Source | Fields |
|---|---|
| `operation_attributes` | `slot_idx`, `valid_global`, `chunk_size_global`, `pad_align`, `layer_idx`, `num_layers`, `cluster_axis` |
| `cache` | storage variant kind, `logical_shape`, `dtype`, `page_config`, `memory_config`, `alignment` |
| `slot_idx` (optional tensor) | engaged/disengaged, plus the same six fields when engaged |
| `valid_global` (optional tensor) | engaged/disengaged, plus the same six fields when engaged |

`padded_shape` is not directly in the default key — it is derived from `logical_shape`, `page_config`
and `alignment`. Mesh coordinates are folded in by the framework on both paths
(`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:989-992`).

## What the custom hash covers

```257:271:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/zero_padded_kv_cache/device/zero_padded_kv_cache_device_operation.cpp
    return tt::tt_metal::operation::hash_operation<ZeroPaddedKvCacheDeviceOperation>(
        tensor_args.slot_idx.has_value(),
        args.layer_idx,
        args.num_layers,
        args.cluster_axis,
        args.chunk_size_global,
        args.pad_align,
        cache.dtype(),
        cache.layout(),
        cache.memory_config(),
        cache.padded_shape(),
        // On the metadata path the reader and writer bake a TensorAccessorArgs built from slot_idx into
        // their compile-time args, so the bank table it selects cannot be refreshed on a cache hit. Only
        // the config is keyed, never the value; valid_global is pinned to match by validate_runtime_args.
        tensor_args.slot_idx.has_value() ? tensor_args.slot_idx->memory_config() : MemoryConfig{});
```

Five of the seven attributes are kept; the two per-request scalars are dropped. The cache is
decomposed selectively into four of its six default components.

**Pre-fix the two optional metadata tensors collapsed to a single `has_value()` bit.** fab067a added the
final term: `slot_idx`'s `memory_config` when the metadata path is taken, a neutral `MemoryConfig{}`
otherwise. That is what closes omission 4 — the buffer type, shardedness and aligned page size baked
into the reader's and writer's metadata `TensorAccessorArgs` are now keyed. The ternary is deliberate:
it must not dereference a disengaged optional, and a real metadata tensor's memory config can never
equal a default-constructed one, so the two paths cannot collide.

## Omitted parameters

### 1. `operation_attributes.valid_global`

**Verdict: VALID — patched.**

This is the moving index the task brief warns about: the number of real tokens written so far, which
advances on every prefill chunk and defines where the pad window `[valid_global, ceil_pad_align(valid_global))`
starts. It is a host scalar on the scalar path, placed in common arg 3 at build time:

```273:286:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/zero_padded_kv_cache/device/zero_padded_kv_cache_device_operation.cpp
    const std::vector<uint32_t> common_runtime_args = {
        my_sp_coord,
        sp_factor,
        chunk_local,
        args.valid_global,
        args.pad_align,
        args.layer_idx,
        args.num_layers,
        Wt,
        cache_CH_pages,
        args.slot_idx,
        slot_idx_addr,
        valid_global_addr,
    };
```

and re-applied on every hit at `override_runtime_arguments:441`, across all three TILE kernels or the
single ROW_MAJOR writer. All of the window arithmetic is done on-device from that arg — nothing
derived from it is baked:

```23:44:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/zero_padded_kv_cache/device/kernels/zero_padded_kv_cache_common.hpp
inline ZeroPadTokenRange zero_pad_compute_token_range(uint32_t valid_global) {
    const uint32_t my = get_common_arg_val<uint32_t>(0);
    const uint32_t sp = get_common_arg_val<uint32_t>(1);
    const uint32_t chunk_local = get_common_arg_val<uint32_t>(2);
    const uint32_t pad_align = get_common_arg_val<uint32_t>(4);

    const uint32_t pad_end = ((valid_global + pad_align - 1) / pad_align) * pad_align;
    if (pad_end == valid_global) {
        return {0, 0};
    }

    const uint32_t chunk_global = sp * chunk_local;
    const uint32_t slab = valid_global / chunk_global;
    const uint32_t chip_begin = slab * chunk_global + my * chunk_local;
    const uint32_t chip_end = chip_begin + chunk_local;
    const uint32_t begin = valid_global > chip_begin ? valid_global : chip_begin;
    const uint32_t end = pad_end < chip_end ? pad_end : chip_end;
    if (begin >= end) {
        return {0, 0};
    }

    return {end - begin, slab * chunk_local + begin - chip_begin};
}
```

Crucially, the amount of *work* also varies with `valid_global` (`w.count`, `w.first_partial`,
`w.row_start`), and the design absorbs that on-device rather than in the work split: the program always
runs on exactly one core (`create_descriptor:261`), the reader unconditionally pushes `Wt` source tiles
and a mask (`reader_zero_padded_kv_cache.cpp:71-95`), the compute unconditionally processes `Wt` tiles
(`zero_padded_kv_cache.cpp:33-46`), and the writer discards them when there is nothing to write back
(`writer_zero_padded_kv_cache.cpp:62-73`). That unconditional CB protocol is what makes the omission
safe — with a `valid_global`-dependent core split or a conditional CB push, the frozen per-core args
would go stale.

On the metadata path `valid_global` arrives as device data instead: the reader and writer NoC-read
element [0] of a 1-element uint32 tensor (`reader_zero_padded_kv_cache.cpp:56-59`,
`writer_zero_padded_kv_cache.cpp:49-52`), and only the tensor's address needs patching, which the
override does at line 431.

Because the value is not hashed, the capacity check re-runs on every hit rather than only on a miss —
`validate_on_program_cache_hit` calls `validate_runtime_args`
(`device/zero_padded_kv_cache_device_operation.cpp:197-200`), which contains:

```138:144:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/zero_padded_kv_cache/device/zero_padded_kv_cache_device_operation.cpp
    if (!tensor_args.valid_global.has_value()) {
        TT_FATAL(
            args.valid_global <= global_capacity,
            "valid_global ({}) exceeds cache capacity ({})",
            args.valid_global,
            global_capacity);
    }
```

### 2. `operation_attributes.slot_idx`

**Verdict: VALID — patched.**

Same mechanism, common arg 9: written at `create_descriptor:283`, re-applied at
`override_runtime_arguments:442`, consumed on-device only as the batch-slot linearisation
`(slot * num_layers + layer) * cache_CH_pages`
(`device/kernels/zero_padded_kv_cache_common.hpp:66`, and its ROW_MAJOR counterpart at `:103`). Both
`num_layers` and `layer_idx` are hashed, so only the free variable is omitted, and it is range-checked
on every hit:

```100:103:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/zero_padded_kv_cache/device/zero_padded_kv_cache_device_operation.cpp
    if (!tensor_args.slot_idx.has_value()) {
        const uint32_t num_slots = cache.padded_shape()[0] / args.num_layers;
        TT_FATAL(args.slot_idx < num_slots, "slot_idx ({}) out of range for num_slots ({})", args.slot_idx, num_slots);
    }
```

### 3. `cache.logical_shape()` — replaced by `padded_shape()`

**Verdict: VALID — relaxation win.** Unchanged by fab067a, and the one relaxation this op should carry
forward into its Metal 2.0 port as `match_padded_shape_only`.

The factory reads only the padded shape:

```250:253:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/zero_padded_kv_cache/device/zero_padded_kv_cache_device_operation.cpp
    const bool is_row_major = cache.layout() == Layout::ROW_MAJOR;
    const uint32_t Wt = is_row_major ? 1 : cache_shape[-1] / TILE_WIDTH;
    const uint32_t cache_H_pages = is_row_major ? cache_shape[-2] : cache_shape[-2] * Wt / TILE_HEIGHT;
    const uint32_t cache_CH_pages = cache_shape[1] * cache_H_pages;
```

and the kernels address the cache in pages, which is exactly the padded-shape grid. Since the op is
in-place (`compute_output_specs` returns `cache.tensor_spec()` at line 202-205) there is no derived
output spec that could inherit a stale logical shape. Two callers whose KV caches differ only in an
unpadded logical sequence length correctly share one program; the default hash would have forced a
recompile for no reason.

The value is thinner than in `update_padded_kv_cache`, and it is worth being honest about that: a KV
cache is allocated once per model and its logical extent rarely varies call to call, so the extra hits
only materialise when two *distinct* cache tensors share a padded shape and differ in logical shape —
say logical sequence lengths of 4090 and 4096, both padding to 4096. That is marginal but real, and it
meets the bar the audit rubric sets (the value can vary; omitting it produces genuine extra hits;
the program is invariant to it). On the Metal 2.0 port this is what
`TensorSpecRelaxations::match_padded_shape_only`
(`tt_metal/api/tt-metalium/experimental/metal2_host_api/tensor_spec_relaxations.hpp:41,49`) expresses
directly; `pertinent_fields` reduces it to `PertinentFields{.padded_shape = true}`
(`tt_metal/impl/metal2_host_api/tensor_spec_relaxations.cpp:67-87`), which is exactly the field this
hash already keys. If the relaxation is judged not worth declaring, the alternative is to key the whole
`tensor_spec()` and leave the relaxations default, as `rotary_embedding_indexed` now does — strictly
finer, and it costs only the recompiles the relaxation was buying.

### 4. The metadata tensors' specs — only `slot_idx.has_value()` is hashed

**Verdict: RESOLVED by fab067a** (was BUG). Closed on all three of its parts. The key now carries
`slot_idx.has_value() ? slot_idx->memory_config() : MemoryConfig{}` (`compute_program_hash:268-271`), so
the buffer type and aligned page size baked into the reader's and writer's metadata `TensorAccessorArgs`
are keyed and the reproduction's call 2 now **misses** and compiles a second program. `validate_meta`
gained the `!meta.is_sharded()` check this op was missing (`:102-108`), which closes the shardedness
gap on the miss path too. And a new mutual-config assertion (`:118-125`) requires `valid_global`'s
memory config to equal `slot_idx`'s, which closes the "one accessor for two tensors" problem recorded
at the end of this section. All three checks sit inside `validate_runtime_args`, so they hold on hits.

**Pre-fix:** the metadata tensors' memory space, shardedness and aligned page size became
reader-and-writer compile-time args and were neither hashed nor patchable.

On the metadata path the op appends a second tensor accessor to two kernels:

```346:357:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/zero_padded_kv_cache/device/zero_padded_kv_cache_device_operation.cpp
    reader.compile_time_args = {
        kSrcCbIndex,
        kMaskCbIndex,
        cache_tile_size,
        static_cast<uint32_t>(has_metadata),
        has_metadata ? kMetaCbIndex : 0u};
    TensorAccessorArgs(cache.buffer()).append_to(reader.compile_time_args);
    if (has_metadata) {
        // One metadata accessor, reused for both 1-element tensors (identical layout); the kernel reads
        // each from its own DRAM address (common args 10/11).
        TensorAccessorArgs(tensor_args.slot_idx->buffer()).append_to(reader.compile_time_args);
    }
```

(and the identical block for the writer at `:386-390`). For a non-sharded buffer,
`TensorAccessorArgs::append_to` emits the args-config word — which carries the `IsDram` and `Sharded`
bits (`tt_metal/impl/buffers/tensor_accessor_args.cpp:153-157`) — and `aligned_page_size`
(`:196-205`). For a sharded buffer it emits an entirely different, longer block. All of that is
compile-time state baked into the cached `Program`; no cache-hit path can refresh it. The pre-fix hash
contained only `tensor_args.slot_idx.has_value()`.

The metadata validator does run on every hit (`validate_on_program_cache_hit` →
`validate_runtime_args`), but pre-fix it pinned less than it needed to:

```82:96:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/zero_padded_kv_cache/device/zero_padded_kv_cache_device_operation.cpp
    if (tensor_args.slot_idx.has_value()) {
        auto validate_meta = [&cache](const Tensor& meta, const char* name) {
            TT_FATAL(meta.storage_type() == StorageType::DEVICE, "metadata tensor {} must be on device", name);
            TT_FATAL(meta.dtype() == DataType::UINT32, "metadata tensor {} must be UINT32", name);
            TT_FATAL(meta.layout() == Layout::ROW_MAJOR, "metadata tensor {} must be ROW_MAJOR", name);
            TT_FATAL(
                meta.logical_volume() == 1,
                "metadata tensor {} must be a single element (got {})",
                name,
                meta.logical_volume());
            TT_FATAL(meta.device() == cache.device(), "metadata tensor {} must be on the same device as cache", name);
        };
        validate_meta(tensor_args.slot_idx.value(), "slot_idx");
        validate_meta(tensor_args.valid_global.value(), "valid_global");
    }
```

Dtype, layout, element count and device were pinned. **Buffer type was not, and neither was
shardedness.** An interleaved-L1 or a height-sharded single-element uint32 tensor passed all five
checks.

Two-call reproduction (pre-fix). Note again that call 2 is a **hit**: strengthening
`validate_on_program_cache_miss` would have been useless, because the miss validator does not run on
the call that destroys the cache.

- **Call 1:** `zero_padded_kv_cache(cache, slot_idx_t, valid_global_t, 0, layer_idx=0, num_layers=61,
  0, chunk_size_global=1024, cluster_axis=1, pad_align=128)` with both metadata tensors allocated
  `MemoryConfig{INTERLEAVED, DRAM}`. Miss; the reader and writer compile with the metadata accessor's
  `IsDram` bit set and the DRAM-aligned page size.
- **Call 2:** identical, except the metadata tensors are allocated `MemoryConfig{INTERLEAVED, L1}` (a
  natural thing to do for a 4-byte value the host rewrites every chunk). The hash is unchanged — only
  `has_value()` participates — so this is a cache hit. The override patches common args 10/11 to the
  new addresses, which is all it is able to do.
- **Stale slot:** the metadata `TensorAccessorArgs<kMetaArgsOffset>` compile-time block referenced at
  `reader_zero_padded_kv_cache.cpp:46-47` and `writer_zero_padded_kv_cache.cpp:39-40`. The kernels
  resolve an L1 address through DRAM banking.
- **Symptom:** `slot` and `valid_global` are read as garbage on-device. `batch_page_base` points at an
  arbitrary batch slot and `zero_pad_compute_token_range` computes an arbitrary window, so the writer
  zeroes pages belonging to some other user or layer. The KV cache is silently destroyed in a region
  the op was never asked to touch, and the op's return value (the cache handle) looks fine.

The shardedness gap was worse still: a sharded metadata tensor changes the *number* of compile-time args
the accessor emits, so on a hit the kernel's fixed `TensorAccessorArgs<5>` / `kMetaArgsOffset` decoding
would misparse the cache accessor's own arguments too. **fab067a closes that separately from the hash**,
because hashing the config keeps a sharded tensor off a *cached* program but would still let it compile
on a fresh miss:

```102:108:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/zero_padded_kv_cache/device/zero_padded_kv_cache_device_operation.cpp
            // A sharded metadata tensor changes the shape and length of the accessor block the kernels
            // append, and both sibling ops reject it; the hashed memory config keeps it off a cached
            // program but would still let it compile on a fresh miss.
            TT_FATAL(
                !meta.is_sharded(),
                "zero_padded_kv_cache does not currently support sharded metadata tensors, but {} is sharded",
                name);
```

**This omission was family-wide.** `update_padded_kv_cache` had the same construction — one
`TensorAccessorArgs` built from `slot_idx->buffer()` appended to the writer's compile args, with only
`has_value()` hashed. Its validator was a strict superset of this one: it also asserted
`!meta.is_sharded()`. The third member of the family, `rotary_embedding_indexed`, always hashed
something of its metadata tensor's spec in addition to `has_value()` and so never had the gap. Because
two of the three ops shared it and the third showed the intended shape, the fix belonged at the family
level rather than in this file alone — and fab067a applied it to both.

A second, miss-path-only problem fell out of the same code: the comment at line 354-355 asserts the
two metadata tensors have "identical layout" and reuses one accessor for both reads, but nothing
validated that `valid_global`'s buffer matched `slot_idx`'s. A DRAM `slot_idx` paired with an L1
`valid_global` was wrong even on a fresh compile. **fab067a closes this half too:**

```118:125:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/zero_padded_kv_cache/device/zero_padded_kv_cache_device_operation.cpp
        // The reader and writer each emit a single TensorAccessorArgs built from slot_idx and use it for
        // both metadata reads, so a differing memory config on valid_global would resolve its address
        // through the wrong bank table.
        TT_FATAL(
            tensor_args.slot_idx->memory_config() == tensor_args.valid_global->memory_config(),
            "zero_padded_kv_cache does not currently support slot_idx and valid_global having different "
            "memory configs, because one TensorAccessor serves both reads (got buffer types {} and {})",
            tensor_args.slot_idx->memory_config().buffer_type(),
            tensor_args.valid_global->memory_config().buffer_type());
```

That pairing is also what makes hashing `slot_idx`'s config *alone* sufficient: `valid_global` is now
required to agree with it, so one keyed config covers the accessor that serves both reads.

### 5. `cache.page_config()` (the `Tile`) — the unguarded 32x32 assumption

**Verdict: RESOLVED by fab067a** (was BUG). A 32x32 tile assertion in the shared
`validate_runtime_args` (`:71-86`) rejects a non-32x32 cache before the key is consulted, on every
dispatch. The factory and the kernels are still 32x32-only and `page_config` is still not hashed — but
that combination is now correct by construction, because the only tile the op admits is the one they
assume.

**Placement, again, is the fix.** The reproduction below is a cache **hit**. A tile guard in
`validate_on_program_cache_miss` would have run on the `Tile{32, 32}` call, passed it, and then not run
on the `Tile{16, 32}` call that destroys live KV data — the guard would have been in the codebase and
the bug would have been unchanged. This op is the clearest illustration of that trap in the family,
because its one existing buffer-type pin is already stranded in the miss validator for exactly this
reason. fab067a put the tile check in `validate_runtime_args`, which both validators delegate to:

```71:86:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/zero_padded_kv_cache/device/zero_padded_kv_cache_device_operation.cpp
    // The TILE branch of the factory derives Wt/cache_H_pages and the cache_tile_size compile arg from
    // the architectural 32x32 constants, and the reader's face loop hardcodes the 32x32 four-face
    // layout; the tile is absent from compute_program_hash. Checked here rather than in the miss
    // validator so it also runs on the cache-hit path, where a non-standard tile would otherwise alias
    // onto a cached 32x32 program and zero the wrong region of the cache.
    if (cache.layout() == Layout::TILE) {
        const auto tile = cache.tensor_spec().tile();
        TT_FATAL(
            tile.get_height() == TILE_HEIGHT && tile.get_width() == TILE_WIDTH,
            "zero_padded_kv_cache does not currently support tiles other than 32x32, but cache has a {}x{} "
            "tile",
            tile.get_height(),
            tile.get_width());
    }
```

The `Layout::TILE` guard matters: `PageConfig::get_tile()` reports a default 32x32 tile for ROW_MAJOR,
so an unconditional check would have been vacuous on that branch rather than wrong, but the guard is
honest about only constraining the branch that reads a tile.

**Pre-fix:** `cache.layout()` is hashed, which collapses `PageConfig` to `ROW_MAJOR` vs `TILE`
and discards the tile shape. The op accepts `Layout::TILE`, computes every page-unit quantity from the
architectural 32x32 constants rather than the cache's actual `Tile`, validated nothing about the tile
geometry, and does not hash `page_config`. A non-32x32 cache therefore did not even get a freshly-built
wrong program — it silently inherited the cache entry built for a 32x32 cache of the same padded shape.

Non-32x32 tiles are a supported TTNN configuration, so this was reachable rather than hypothetical.

**The factory is entirely 32x32-hardcoded.** The page-count derivation uses bare `TILE_WIDTH` and
`TILE_HEIGHT` rather than `cache.tensor_spec().tile().get_tile_shape()`:

```249:253:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/zero_padded_kv_cache/device/zero_padded_kv_cache_device_operation.cpp
    const tt::DataFormat cache_format = datatype_to_dataformat_converter(cache.dtype());
    const bool is_row_major = cache.layout() == Layout::ROW_MAJOR;
    const uint32_t Wt = is_row_major ? 1 : cache_shape[-1] / TILE_WIDTH;
    const uint32_t cache_H_pages = is_row_major ? cache_shape[-2] : cache_shape[-2] * Wt / TILE_HEIGHT;
    const uint32_t cache_CH_pages = cache_shape[1] * cache_H_pages;
```

and the byte sizes come from `tt::tile_size`, which returns the size of a 32x32 tile, not
`tile.get_tile_size(format)`:

```315:330:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/zero_padded_kv_cache/device/zero_padded_kv_cache_device_operation.cpp
    const uint32_t cache_tile_size = tt::tile_size(cache_format);
    const tt::DataFormat mask_format = tt::DataFormat::Float16_b;
    const uint32_t mask_tile_size = tt::tile_size(mask_format);

    // CBs: src (partial tile read), mask (bf16 row-mask), out (masked partial), zero (write scratch).
    auto add_cb = [&](uint32_t index, tt::DataFormat fmt, uint32_t page, uint32_t npages) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = npages * page,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{.buffer_index = index, .data_format = fmt, .page_size = page}}},
        });
    };
    add_cb(kSrcCbIndex, cache_format, cache_tile_size, Wt);
    add_cb(kMaskCbIndex, mask_format, mask_tile_size, 1);
    add_cb(kOutCbIndex, cache_format, cache_tile_size, Wt);
    add_cb(kZeroCbIndex, cache_format, cache_tile_size, 1);
```

`cache_tile_size` is not only a CB page size — it is compile-time arg index 2 of both the reader
(`create_descriptor:346-351`) and the writer (`:379-384`), and `TensorAccessorArgs(cache.buffer())`
(lines 307, 352, 385) separately emits the *buffer's* real `aligned_page_size`. Under a non-32x32 tile
those two disagree, in a program nothing can rebuild.

And the reader hard-codes the 32x32 four-face layout when building the row mask:

```86:94:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/zero_padded_kv_cache/device/kernels/dataflow/reader_zero_padded_kv_cache.cpp
    for (uint32_t face = 0; face < 4; ++face) {
        const uint32_t row_base = (face >= 2) ? 16u : 0u;  // faces 0,1 -> rows 0-15; 2,3 -> rows 16-31
        for (uint32_t fr = 0; fr < 16; ++fr) {
            const uint16_t val = ((row_base + fr) < rs) ? kBf16One : 0u;
            for (uint32_t fc = 0; fc < 16; ++fc) {
                m[face * 256 + fr * 16 + fc] = val;
            }
        }
    }
```

as does the shared header (`device/kernels/zero_padded_kv_cache_common.hpp:72`,
`constexpr uint32_t tile_height = 32;`).

**Pre-fix, nothing validated the tile geometry.** There was no `tensor_spec().tile()` read and no
tile-geometry `TT_FATAL` anywhere in the op directory. The one alignment check that existed is a check
on `chunk_local` against the same architectural constant, not on the tile:

```123:127:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/zero_padded_kv_cache/device/zero_padded_kv_cache_device_operation.cpp
    TT_FATAL(
        chunk_local % TILE_HEIGHT == 0,
        "chunk_local ({}) must be tile-aligned (multiple of {})",
        chunk_local,
        TILE_HEIGHT);
```

A `chunk_local` of 32 satisfies it regardless of whether that is one 32-row tile or two 16-row tiles.

**Two-call reproduction (pre-fix).**

- **Call 1:** `cache` `BFLOAT16`, `Layout::TILE`, `Tile{32, 32}`, interleaved DRAM, padded
  `[1, 1, 128, 128]`; `layer_idx=0, num_layers=61, cluster_axis=1, chunk_size_global=128, pad_align=32`,
  scalar path. Miss; the program compiles with `Wt = 4`, `cache_H_pages = 16`, `cache_CH_pages = 16`,
  `cache_tile_size = 2048`, four CBs paged at 2048 bytes, and reader/writer compile-time arg 2 = 2048.
- **Call 2:** identical in every hashed respect — same dtype, same `Layout::TILE`, same memory config,
  same padded shape, same attributes — but the cache carries `Tile{16, 32}`. The `Tile` lives inside
  `page_config`, and   `compute_program_hash` hashes only `cache.dtype/layout/memory_config/padded_shape`
  (`compute_program_hash:227-230`), so the key is byte-identical and the cache hits.
  `validate_on_program_cache_hit` runs `validate_runtime_args`, which contains no tile check.
- **Stale slots:** reader compile-time arg 2 and writer compile-time arg 2 (`cache_tile_size`) stay
  2048 where the real page is 1024 bytes; the `src`, `out` and `zero` CBs stay paged and sized at 2048
  bytes; common args 7 and 8 (`Wt`, `cache_CH_pages`) stay at the values produced by dividing by 32,
  which is half the true page count; and the cache `TensorAccessorArgs`' compile-time
  `aligned_page_size` stays 2048, disagreeing with the buffer it is now pointed at.
- **Symptom:** the writer indexes pages by a stale `cache_CH_pages`, so the pad window lands at the
  wrong sequence position within the slab, and every page write moves 2048 bytes into a 1024-byte page,
  running over into the following page. Because this op's whole purpose is to *zero* a region, the
  observable result is that live KV data outside the pad window is destroyed — the most damaging
  possible failure for an in-place cache op, and it happens with no cache miss to hint at the cause.

**This defect was family-wide.** `update_padded_kv_cache` had the identical shape — `tt::tile_size` at
its line 276, `Wt`/`input_Ht`/`cache_HtWt` via bare constants at 277-279, `writer_tile_height =
TILE_HEIGHT` at 280, and no tile guard. `rotary_embedding_indexed` hardcodes 32x32 just as thoroughly
(five `tt::tile_size` calls and four bare-constant tile-count conversions) and also had no guard, yet
its verdict was only CAVEAT — purely because it dispatches through Metal 2.0 `UpdateProgramRunArgs`,
whose exact `TensorSpec` equality check covers `page_config` and therefore threw on the mismatched
second call instead of executing it. This op goes through the descriptor buffer-binding fast path
(`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:726-731`), which performs no spec comparison of any
kind, so the same source-level mistake degraded from a loud rejection to silent data loss. The
difference in outcome was entirely a property of the cache-hit mechanism the op was built on, not of the
op's own code quality. fab067a fixed all three, with the same guard, in the same shared validator — the
point being that the op should not depend on which dispatch layer it happens to sit on.

**The guard.** The fix is small and makes omitting `page_config` correct by construction — the same
shape as the check the non-partial `interleaved_to_sharded` already carries:

```94:98:ttnn/cpp/ttnn/operations/data_movement/sharded/interleaved_to_sharded/device/interleaved_to_sharded_op.cpp
    if (input_tensor.layout() == Layout::TILE) {
        auto tile = input_tensor.tensor_spec().tile();
        if (tile.get_height() != tt::constants::TILE_HEIGHT || tile.get_width() != tt::constants::TILE_WIDTH) {
            return {false, fmt::format("interleaved_to_sharded requires standard 32x32 tiles, got {}x{}", tile.get_height(), tile.get_width())};
        }
```

fab067a adopted exactly that shape, in `validate_runtime_args` (quoted at the top of this section).
Making the factory tile-aware instead is a much larger change here than in the two siblings, because
the 32x32 assumption also lives in the kernel sources (the four-face mask loop above and
`tile_height = 32` in the shared header) — and it would additionally require adding `page_config` to
`compute_program_hash`, since the program would then provably vary with `Tile`. Fencing rather than
fixing was clearly the right order of operations for this op in particular.

### 6. `cache.tensor_layout().get_alignment()`

**Verdict: VALID — determined by hashed terms** (was CAVEAT). This is a regrade on closer tracing
rather than a change fab067a made, though the new tile guard is what makes the argument airtight.
`alignment` never moves `aligned_page_size` independently of terms already in the key: on the TILE
branch the page size is `tt::tile_size` of the hashed `dtype` with the tile now pinned to 32x32, and on
the ROW_MAJOR branch `row_page_size` is the last padded dimension's byte extent — a function of the
hashed `padded_shape` and `dtype`. Combined with the hashed `memory_config`, which fixes the buffer type
and therefore the canonical alignment, the hashed four determine every baked page quantity.

The residual is narrow and stays worth recording: `alignment` is not read as a tensor property, but it
does move `aligned_page_size`, which is a compile-time arg.

Alignment enters through the `TensorAccessorArgs(cache.buffer())` door at lines 307, 352 and 385, plus
the ROW_MAJOR `row_page_size` at line 291, which sizes the zero CB and is a compile-time arg to the
ROW_MAJOR writer (line 306). Compile-time args and CB sizes are baked into the cached `Program` and are
refreshed by no cache-hit path.

This is safe because every caller builds the cache with the canonical alignment for its buffer type,
which the hashed {`dtype`, `layout`, `memory_config`, `padded_shape`} then fully determines. What would
break it is a `TensorLayout` constructed with an explicit non-canonical `Alignment` that leaves those
four unchanged. No Python path can produce that: every `TensorSpec` constructor exposed through nanobind
routes to the three-argument `TensorLayout` constructor, which *computes* the canonical alignment from
dtype, page config and memory config rather than accepting one. Only a C++ caller reaching
`ttnn::prim::zero_padded_kv_cache` directly with a hand-built `TensorLayout` could reach it, which is
outside the public API this audit grades against — hence the regrade from `CAVEAT` to
`VALID — determined by hashed terms`. Unlike the tile in omission 5, no supported TTNN configuration
reaches it.

### 7. `cache.storage` variant kind

**Verdict: VALID — pinned by the framework** (was CAVEAT — pinned only on the miss path). This is a
correction to the pre-fix grade rather than something fab067a changed: device storage was never a
reachable omission on any path. See `#### Framework correction` below. fab067a did move the pin into
`validate_runtime_args` (`:136-141`), but for a different and local reason — it guards a `cache.device()`
dereference on the next line.

```157:158:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/zero_padded_kv_cache/device/zero_padded_kv_cache_device_operation.cpp
    TT_FATAL(cache.storage_type() == StorageType::DEVICE, "cache must be on device");
    TT_FATAL(cache.buffer()->buffer_type() == BufferType::DRAM, "zero_padded_kv_cache requires a DRAM-backed cache");
```

The cache is constrained to a single storage kind *and* a single buffer type, so neither carries
information on the first call. But pre-fix both `TT_FATAL`s sat above the `validate_runtime_args`
delegation at `:194`, so under the dispatcher branch quoted in `## Cache-hit patch mechanism` they ran
once and never again.

The pre-fix reasoning ran: severity is low, because a host-storage cache has no `buffer()`, so the hit
path faults immediately in `collect_tensor_buffers` when it tries to collect an address rather than
executing a stale program; and the buffer type is independently carried by the hashed
`cache.memory_config()`, so a DRAM-to-L1 change produces a genuine cache miss rather than a wrong hit.
The grade was a caveat on structural grounds rather than a live hazard. That was arguing about the
consequence of something that cannot happen.

#### Framework correction

`launch()` asserts device storage and allocation on **every** tensor argument, on **every** dispatch,
before the key is computed or the cache is probed:

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

Three properties make this decisive. It sits at the top of `launch()` (`:487`), well before
`launch_operation_with_adapter` computes the key and probes the cache (`:409-418`), so it runs on hits
and misses alike. It traverses via `visit_object_of_type<Tensor>` (`:492-493`), which descends into
`std::optional<Tensor>`, so the two metadata tensors are covered as well as `cache`. And it is
unconditional — no op can opt out.

So the storage variant kind cannot vary along any reachable path. It is a zero-value omission, not a
caveat: there is no admissible call in which it differs, on either path. The same correction applies to
the `buffer()` null-safety reasoning throughout this document — including the miss validator's
`cache.buffer()->buffer_type()` dereference at `:158`, which is safe because `is_allocated()` was
asserted on the same framework lines.

This also strengthens the recommendation not to buy the pin back on the hot path: it would cost a
`storage_type()` query on every dispatch to re-check something the framework has already made
impossible. fab067a's move is not a counter-example — `cache.device()` is dereferenced immediately
afterwards, and a message beats a null dereference.

The observation this subsection previously rested on still holds and is worth keeping: the contrast
with omission 4 is that the op knows how to pin a buffer type when it wants to — it just never applied
the same discipline to the metadata tensors, where the buffer type genuinely does reach a baked
compile-time arg. fab067a closed that asymmetry from the other side, by hashing the metadata config and
pinning the two metadata tensors to agree, rather than by adding a DRAM pin they have no reason to need.

### 8. Buffer address of `cache`

**Verdict: VALID — patched, and required.** Registered as a `Buffer*` binding on every kernel —
`create_descriptor:310` (ROW_MAJOR writer), `:360` (reader), `:393` (writer) — which is what puts the
inner adapter on the fast path. The compute kernel takes a literal `0u` placeholder
(`:370`) because it reads no addresses.

On in-place aliasing specifically: the cache is simultaneously `tensor_args.cache` (input region) and
the value `create_output_tensors` returns (output region), so it appears twice in
`collect_tensor_buffers`. The resolver treats that as the safe in-place case rather than the ambiguous
`matmul(X, X)` duplicate, and does not bail:

```90:94:tt_metal/impl/program/program_descriptor_patching.cpp
            const bool is_input = i < num_input_buffers;
            // An output/workload buffer that aliases an input is the safe in-place case — skip it.
            if (!is_input && input_buffers.contains(buf)) {
                continue;
            }
```

Worth being precise about, since the op has `override_runtime_arguments`: that hook is on the *outer*
`MeshWorkloadFactory`, and it delegates addresses to the inner adapter rather than re-deriving them.
So this op does **not** bypass `resolve_bindings` the way a hand-rolled mode-A op does; its aliasing
safety comes from the resolver's output-region skip quoted above. The metadata tensors deliberately do
not use `Buffer*` bindings — their addresses ride in common args 10/11
(`create_descriptor:245-247,284-285`), which is exactly why the op must patch them by hand.

### 9. `my_sp_coord`, `sp_factor`, `chunk_local` (derived, not attributes)

**Verdict: VALID — invariant.** `sp_factor` is the mesh extent along the hashed `cluster_axis`,
`chunk_local = chunk_size_global / sp_factor` is derived from a hashed attribute, and `my_sp_coord`
comes from the dispatch coordinate (`create_descriptor:255-258`). Coordinates are folded into the key
by the framework for both the default and custom paths
(`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:989-992`), and the program cache is per-device, so a
program can never be reused at a different mesh position. These three sit in unpatched common args
0/1/2, so this argument is load-bearing rather than incidental.

## Keys the custom hash adds beyond the default

- `cache.padded_shape()` — a derivation in the default key, promoted to first-class. This is what makes
  dropping `cache.logical_shape()` safe.
- `cache.layout()` — a lossy projection of `page_config`; see omission 5. Safe now that the tile guard
  pins the other half of `page_config` on both paths.
- `tensor_args.slot_idx.has_value()` — pre-fix a lossy projection of the two optional tensors rather
  than an addition (see omission 4). Post-fix it is joined by `slot_idx->memory_config()`, so the pair
  keys the program variant *and* the accessor configuration the variant bakes.
- `slot_idx->memory_config()` on the metadata path, `MemoryConfig{}` otherwise
  (`compute_program_hash:268-271`) — added by fab067a. The default-constructed sentinel is safe because
  a real device tensor's memory config can never equal it, so the scalar and metadata paths cannot
  collide on it.

## Framework side effect of having a custom hash

Defining `compute_program_hash` opts this op out of attribute-level collision resolution:

```1012:1014:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
        if constexpr (requires { DeviceOperation::compute_program_hash(attrs, tensor_args); }) {
            return key;  // custom hash -> opt out beyond the op-identity prefix
        } else {
```

`ProgramCacheKey::canonical` degrades to the op type name, so a 64-bit collision between two different
configurations resolves to a wrong hit instead of a rebuild. This op's whole purpose is to destroy data
in a shared KV cache at a computed offset, so a wrong hit means zeroing the wrong region — the least
observable and most damaging failure mode available. That raises the cost of every gap above.

## Non-cache correctness defects

Recorded separately so they are not counted as program-cache bugs. These concern the factory and the
override, not the key.

| Defect | Status | Note |
|---|---|---|
| The override selects kernels by raw handle — `{0u, 2u}` on the metadata path (`override_runtime_arguments:466`) and `0..num_kernels` on the scalar path with `num_kernels` re-derived from `cache.layout()` (`:476`). | **OPEN** — unchanged by fab067a | A three-way coupling between the override, `create_descriptor`'s kernel push order (which `ProgramImpl` preserves, `tt_metal/impl/program/program.cpp:402-466`), and the metadata-path-is-TILE-only guard in a third function. It is sound today — see the argument in omission 4's reachability note and recommendation 6 — but it is sound by a three-step argument rather than by construction, and the override asserts arg *counts* rather than kernel *identity*, so a reordered descriptor would silently patch the wrong kernel's common args. `update_padded_kv_cache` carries the same class of defect with its hard-coded `kWriterKernelHandle = 1`. |
| The miss-path 32x32 factory assumption: `cache_tile_size` from `tt::tile_size` and `Wt`/`cache_H_pages` from bare `TILE_WIDTH`/`TILE_HEIGHT` (`create_descriptor:251-252`, `:315-317`), plus the reader's hard-coded four-face mask loop (`reader_zero_padded_kv_cache.cpp:86-94`) and `tile_height = 32` in the shared kernel header (`:72`). Even on a cache **miss**, a `Tile{16, 32}` cache would have compiled a wrong program. | **RESOLVED by fab067a** | The tile assertion in `validate_runtime_args` (`:71-86`) rejects the call before `create_descriptor` runs. The factory and kernels are still 32x32-only — fenced rather than made tile-aware, which is especially the right order here because the assumption reaches into the kernel sources. |
| The 32x32 TILE constraint was undocumented in the public API. | **RESOLVED by fab067a** | Stated in the nanobind docstring (`zero_padded_kv_cache_nanobind.cpp:49-52`). |

## Summary

| Omitted vs. default | Used by program? | Patched on hit? | Verdict |
|---|---|---|---|
| `attrs.valid_global` | Yes (common arg 3) | Yes (override) | VALID — patched |
| `attrs.slot_idx` | Yes (common arg 9) | Yes (override) | VALID — patched |
| `cache.logical_shape` | No (padded shape used) | n/a | VALID — relaxation win (→ `match_padded_shape_only` on the Metal 2.0 port) |
| metadata tensors' `memory_config` / shardedness / `alignment` | Yes (accessor compile-time args) | No (compile-time) | RESOLVED by fab067a — `slot_idx->memory_config()` hashed, `!is_sharded()` added, and `valid_global` pinned to match `slot_idx` (was BUG) |
| `cache.page_config` (`Tile`) | Yes (`cache_tile_size` compile-time arg, CB page sizes, `Wt` / `cache_CH_pages`) | No (compile-time) | RESOLVED by fab067a — 32x32 asserted in `validate_runtime_args` (was BUG) |
| `cache.alignment` | Only via `aligned_page_size` | No | VALID — determined by hashed {`dtype`, `layout`, `memory_config`, `padded_shape`} (was CAVEAT) |
| `cache.storage` kind | n/a | n/a | VALID — pinned by the framework at `device_operation.hpp:500-501` (was CAVEAT — pinned only on the miss path; that grade was wrong) |
| `cache` buffer address | Yes | Yes (`resolved_bindings`) | VALID — patched, required |
| `my_sp_coord`, `sp_factor`, `chunk_local` | Yes (common args 0-2) | n/a (coordinate / hashed attrs) | VALID — invariant |

**Two program-cache bugs were found; both are RESOLVED by fab067a. Zero remain.** The two moving
per-request indices behaved correctly from the start: `valid_global` and `slot_idx` are omitted from the
hash, re-applied to every kernel on every hit, and re-validated on every hit, and the design
deliberately keeps all `valid_global`-dependent work off the host by fixing the program to one core with
an unconditional CB protocol. Likewise the genuinely structural padding parameters —
`chunk_size_global` and `pad_align`, which land in unpatched common args and would be the obvious place
for this class of bug — were correctly hashed. Both defects were compile-time-arg defects, the one
category no cache-hit path can repair.

The first was the optional metadata tensors: they contributed only a `has_value()` bit to the key, yet
their buffer type, shardedness and aligned page size are compiled into the reader's and writer's
`TensorAccessorArgs`. Passing L1-allocated (or sharded) metadata tensors after a DRAM-allocated first
call produced a silent wrong hit in which the kernels read garbage indices and zeroed an arbitrary
region of another user's KV cache. fab067a keys `slot_idx->memory_config()`
(`compute_program_hash:268-271`), so that second call now misses; adds the `!is_sharded()` check this op
was missing (`:102-108`), which also blocks a sharded metadata tensor on a fresh compile; and pins
`valid_global`'s config equal to `slot_idx`'s (`:118-125`), which is what makes one keyed config
sufficient for the accessor that serves both reads.

The second was the unguarded 32x32 tile assumption (omission 5). The op accepts `Layout::TILE`, derives
`cache_tile_size` from `tt::tile_size` and `Wt`/`cache_H_pages` from bare `TILE_WIDTH`/`TILE_HEIGHT`
(`create_descriptor:251-252`, `:315-317`), never reads the cache's actual `Tile`, and did not hash
`page_config`. `cache_tile_size` is compile-time arg 2 of both the reader and the writer as well as the
page size of three CBs, and the reader's row-mask builder assumes the 32x32 four-face layout outright. A
`Tile{16, 32}` call following an otherwise-identical `Tile{32, 32}` call hit the cache, zeroed the wrong
page range, and overran each 1024-byte page with a 2048-byte write — destroying live KV data in an op
whose only job is to zero a bounded window. fab067a took the rejecting route: a 32x32 assertion in
`validate_runtime_args` (`:71-86`). The same defect was present in `update_padded_kv_cache` and
`rotary_embedding_indexed` and was fixed the same way in both; pre-fix it was only in
`rotary_embedding_indexed` that the Metal 2.0 dispatch path turned it into a loud throw rather than
silent corruption, and the fix removes the dependence on that accident.

**Both fixes turn on validator placement, and that is the transferable lesson.** Each reproduction above
is a cache **hit**: call 1 misses, compiles, caches, and passes every check the miss validator makes;
call 2 computes the same key, hits, and destroys the cache. Because this op defines
`validate_on_program_cache_hit`, the dispatcher *substitutes* rather than supplements
(`ttnn/api/ttnn/device_operation.hpp:262-266`), so the miss validator does not execute on call 2 at all.
A guard added to `validate_on_program_cache_miss` — the intuitive place, and where this op's existing
cache buffer-type pin already sits — would have been present in the source and completely inert on the
only call that matters. Every guard fab067a added went into the shared `validate_runtime_args`, which
both validators delegate to (`:194`, `:199`). That is the only placement that works, and this op is the
family's cleanest demonstration of why: it already had a pin stranded in the miss validator, doing
nothing after the first call, before anyone went looking for cache bugs.

A third, lower-severity finding is structural and remains: because the op defines
`validate_on_program_cache_hit`, that validator *replaces* the miss validator on hits rather than
supplementing it, and this op's hit validator is a bare delegation to `validate_runtime_args`.
Everything the miss validator checks before its own delegation at `:194` is therefore absent on the hit
path — post-fix that is the DRAM buffer-type pin behind omission 7, the layout and dtype gates, the
metadata-path-is-TILE-only guard, and the rank, num-heads and `num_layers` checks. A narrow hit
validator is a hazard rather than a safeguard in general: by existing, it disables everything above it.

In this op's case, however, the practical damage was close to nil even pre-fix, and that is worth
stating as plainly as the hazard itself. This op hashes an unusually broad set — `cache.dtype()`,
`cache.layout()`, `cache.memory_config()`, `cache.padded_shape()`, `layer_idx`, `num_layers`,
`chunk_size_global`, `pad_align`, `slot_idx.has_value()` and now `slot_idx->memory_config()` — and a
miss-only pin on a hashed value cannot be evaded, because any call carrying a new value of it misses and
meets the pin there. Filtering the drop set against the key left exactly one reachable check, the
`storage_type()` pin at `:157`, and its failure mode was a crash rather than silent corruption. The
reachability table in `## Cache-hit patch mechanism` works through this line by line. Post-fix even that
row is gone twice over: fab067a relocated it into `validate_runtime_args`, and the framework pins device
storage on every dispatch anyway (`ttnn/api/ttnn/device_operation.hpp:500-501`). Nothing in the drop set
is now both reachable and unguarded.

## Recommendations

**Status: recommendations 1, 2 and 3 landed in fab067a; 4 stands as written; 5 and 6 are outstanding.**
Each item carries its own status line. The section is left in place because its reasoning about
placement and pricing is what transfers to the Metal 2.0 port.

**Every guard below names the function it must go into, and for this op that function is always
`validate_runtime_args`.** Because the op defines `validate_on_program_cache_hit`, the miss validator is
skipped entirely on a hit; a guard placed in `validate_on_program_cache_miss` would not run on the
offending second call, which is the only call a cache bug reaches. `validate_runtime_args` is the right
home for all of them because both validators delegate to it (`:194`, `:199`), so one placement covers
both paths.

**And every guard below is priced.** The cache-hit path is the fast path — it is what the program cache
exists to make cheap — so a `TT_FATAL` added to `validate_runtime_args` is paid on every dispatch for
the life of the process. That is why only one new check is recommended here, and why the op's single
reachable miss-only pin is deliberately left as a documented caveat rather than fixed.

There are two distinct ways to close a miss-only pin in this op, and they are not interchangeable:

- **Targeted (recommended):** move the specific `TT_FATAL` into `validate_runtime_args`. Adds only that
  check per dispatch.
- **Wholesale (alternative):** delete `validate_on_program_cache_hit` entirely, putting the op on the
  dispatcher's substitution branch so the full miss validator runs on every hit. Simplest and safest,
  and immune to someone later adding a check to the wrong function — but it puts all of `:156-193` on
  the hot path: a `storage_type()` query, a buffer-type dereference, five layout/dtype gates, two shape
  queries and two divisibility checks, on every prefill step. For this op that is especially poor value,
  because the reachability table in `## Cache-hit patch mechanism` shows that exactly one of those lines
  can be reached on a hit at all, and its failure mode is already a crash. Prefer the targeted approach,
  which here means adding the tile guard and nothing else.

1. Hash the metadata tensors' specs. Mirror what `rotary_embedding_indexed` already does: add
   `tensor_args.slot_idx->memory_config()` and `tensor_args.valid_global->memory_config()` (with
   neutral defaults on the scalar path) to `compute_program_hash`. This is a family-wide gap — apply the
   same change to `update_padded_kv_cache`.
   **Status: DONE in fab067a**, with one refinement worth noting: only `slot_idx`'s config is keyed
   (`:268-271`), and `valid_global` is instead *pinned equal to it* by the new mutual-config assertion
   (`:118-125`). That is equivalent and tighter — both accessors are built from `slot_idx` alone, so a
   second keyed config would have been redundant with an invariant the op needs to assert anyway. The
   same change landed in `update_padded_kv_cache`.
2. Bring this op's `validate_meta` up to `update_padded_kv_cache`'s: add the missing
   `TT_FATAL(!meta.is_sharded(), ...)`. Then go further in both ops and add
   `TT_FATAL(meta.buffer()->buffer_type() == BufferType::DRAM, ...)`, plus an assertion that the
   `slot_idx` and `valid_global` tensors share a buffer type and aligned page size — the kernels reuse
   one accessor for both reads (`create_descriptor:353-357`, `:386-390`) and nothing enforces that they
   are interchangeable, which is wrong even on a cache miss.
   **Target function:** the `validate_meta` lambda already lives inside `validate_runtime_args`
   (`:83-93`), so it is on the hit path and is the correct home as-is. This is worth stating because the
   equivalent guard must *not* go into `validate_on_program_cache_miss` — that is where this op's
   existing `buffer_type() == DRAM` pin for the *cache* lives (`:158`), and that pin consequently stops
   running after the first call. Repeating that placement for the metadata tensors would produce a guard
   that passes the DRAM-allocated first call and is then absent on the L1-allocated second call, which
   is precisely the wrong hit described in omission 4.
   **Per-dispatch cost:** two or three extra checks per metadata tensor, and only on the metadata path
   — the lambda is already called there, so this adds to an existing cost rather than creating one.
   Worth it: unlike omission 7 below, the defect these close is the silent wrong hit that is this
   document's headline BUG.
   **Status: DONE in fab067a**, in a better shape than asked. The missing `!meta.is_sharded()` landed
   (`:102-108`), and the interchangeability assertion landed as a whole-`memory_config` equality between
   `slot_idx` and `valid_global` (`:118-125`) — which pins buffer type and aligned page size together.
   The suggested `buffer_type() == BufferType::DRAM` pin was correctly *not* added: it would hard-code a
   buffer type the op has no reason to require, and the property the shared accessor actually needs is
   that the two tensors agree, which the equality expresses directly.
3. Reject a non-32x32 `Tile` on the TILE path, closing omission 5. Assert
   `cache.tensor_spec().tile().get_height() == TILE_HEIGHT` and the same for `get_width()`, in the same
   shape as the `interleaved_to_sharded` guard quoted in omission 5.
   **Target function:** `validate_runtime_args`, not `validate_on_program_cache_miss`. The reproduction
   in omission 5 is a *hit*, so a guard in the miss validator would let the first `Tile{32,32}` call
   through and then not run at all on the `Tile{16,32}` call that destroys live KV data. Placing it in
   `validate_runtime_args` covers the miss path too, via the delegation at `:194`.
   **Per-dispatch cost:** two `uint32_t` comparisons against constants. This is the only new hit-path
   check this document recommends, and it is the one clearly worth its price — it closes a BUG whose
   symptom is destroying live KV data in an op whose only job is to zero a bounded window.
   The mask builder in `reader_zero_padded_kv_cache.cpp:86-94` and `tile_height = 32` in
   `device/kernels/zero_padded_kv_cache_common.hpp:72` already assume 32x32 unconditionally, so the
   guard only makes an existing assumption explicit — and it makes omitting `page_config` correct by
   construction. Making the op genuinely tile-aware instead is a much larger change (it reaches into
   the kernel sources) and would require adding `page_config` to the hash in the same commit. This is a
   family-wide gap: apply the same guard to `update_padded_kv_cache` and `rotary_embedding_indexed`.
   **Status: DONE in fab067a** — landed at `:71-86` in `validate_runtime_args`, guarded by
   `layout() == Layout::TILE` so it does not fire vacuously on the ROW_MAJOR branch. The same guard
   landed in both siblings.
4. **Do not move `:157` onto the hit path to close omission 7.** This is a deliberate
   non-recommendation, recorded so it is not mistaken for an oversight. The original reasoning was that
   the failure it prevents is not silent — a host-storage cache has no device buffer, so the hit path
   faults in `collect_tensor_buffers` on the same call — so the moved check would cost a
   `storage_type()` query on every dispatch and buy only a clearer message.
   **Status: STANDS, and the reasoning is now stronger than when written.** The pin is not merely
   low-value, it is provably zero-value: `launch()` asserts `is_device_tensor` and `is_allocated` on
   every tensor argument on every dispatch (`ttnn/api/ttnn/device_operation.hpp:500-501`), so the
   failure cannot occur at all — see `#### Framework correction` under omission 7, which also regrades
   that row from CAVEAT to `VALID — pinned by the framework`. fab067a did relocate the pin (`:136-141`),
   but on unrelated grounds: `cache.device()` is dereferenced on the next line, and a message beats a
   null dereference.

   Two related checks explicitly do **not** need moving, contrary to how the mechanical drop set reads:
   the DRAM buffer-type pin at `:158` is subsumed by the hashed `cache.memory_config()`, and the
   metadata-path-is-TILE-only guard at `:174-179` constrains `cache.layout()` and
   `slot_idx.has_value()`, both of which are hashed — so a (ROW_MAJOR, metadata) call has its own cache
   slot, misses on first occurrence, and is rejected by the miss validator there. Neither is reachable
   on a hit, so neither is worth its per-dispatch cost. fab067a left both where they are, consistent
   with this.
5. Run this op's tests under `-DTT_DESCRIPTOR_PATCHING_PARITY_CHECK`. The oracle covers runtime args and
   CB addresses only (`tt_metal/api/tt-metalium/experimental/program_descriptor_patching.hpp:176-186`),
   so it would not have caught the compile-time-arg defect in omission 4 — but it will catch any
   regression in the hand-written common-arg patch, which currently has three separate index constants
   (`kValidGlobalCommonArgIdx`, `kSlotIdxCommonArgIdx`, and the pair of address indices) that must stay
   in sync with `create_descriptor`'s vector literal and with three kernels' `get_common_arg_val` calls.
   **Status: OUTSTANDING.** fab067a adds no tests, by design: its own note says the guarded situations
   have no real use case, so most of the change is `TT_FATAL`s on inputs nothing exercises. That
   reasoning does not extend to the parity oracle, which guards the patch path that *is* exercised on
   every dispatch — and for this op that patch path is the most index-fragile of the three.
6. The override selects kernels by raw handle — `{0u, 2u}` on the metadata path and `0..num_kernels` on
   the scalar path, with `num_kernels` re-derived from `cache.layout()`
   (`override_runtime_arguments:425,435`). That works because kernel handles follow descriptor push
   order (`tt_metal/impl/program/program.cpp:402-466`) and because the metadata path is validated
   TILE-only (`validate_on_program_cache_miss:174-179`), but it is a three-way coupling between the
   override, `create_descriptor`'s push order, and a layout guard in a third function. The third leg
   survives the cache-hit analysis, which is worth recording because it initially looks as though it
   does not: the TILE-only guard lives in the miss validator and so does not run on the hit at which the
   override selects kernels, but both values it constrains are hashed, so a (ROW_MAJOR, metadata) call
   cannot reach a hit without having first missed and been rejected. The coupling is sound; it is just
   fragile to read. Deriving the kernel set from the cached descriptor, or naming the handles in one
   place, would make the arrangement robust without depending on that argument.
   **Status: OUTSTANDING.** Not a cache defect — recorded in `## Non-cache correctness defects` so it is
   not counted as one — and out of scope for fab067a, which confined itself to keys and validators.
   `update_padded_kv_cache` carries the same class of coupling in its hard-coded
   `kWriterKernelHandle = 1`.
