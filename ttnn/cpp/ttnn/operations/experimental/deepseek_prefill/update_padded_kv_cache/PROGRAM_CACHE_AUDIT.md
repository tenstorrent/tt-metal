# Program Cache Audit — `experimental/deepseek_prefill/update_padded_kv_cache`

Audit of
`ttnn::operations::experimental::deepseek_prefill::update_padded_kv_cache::UpdatePaddedKvCacheDeviceOperation::compute_program_hash`
against the framework default ("hash everything") key.

| | |
|---|---|
| Device operation | `UpdatePaddedKvCacheDeviceOperation` (`device/update_padded_kv_cache_device_operation.hpp:21`) |
| Custom hash | `device/update_padded_kv_cache_device_operation.cpp:332-364` (post-fab067a; `:208-234` pre-fix, which is what the omission analysis below cites) |
| `operation_attributes_t` | `slot_idx`, `kv_actual_global`, `layer_idx`, `num_layers`, `cluster_axis` |
| `tensor_args_t` | `cache`, `input`, `std::optional<Tensor> slot_idx`, `std::optional<Tensor> kv_actual_global` |
| Program factories | one: `ProgramFactory::create_descriptor` (`ProgramDescriptor`-based), wrapped by `MeshWorkloadFactory` |
| `override_runtime_arguments` | **Yes**, on `MeshWorkloadFactory` (`device/update_padded_kv_cache_device_operation.cpp:439-464`) |
| `get_dynamic_runtime_args` | **No** |
| `validate_on_program_cache_hit` | **Yes** (`device/update_padded_kv_cache_device_operation.cpp:189-194`) — so it *replaces* the miss validator on hits rather than supplementing it |
| Validator actually run on a hit | `validate_runtime_args` only (`:57-126`); everything in `validate_on_program_cache_miss` before its delegation at `:186` is skipped |
| Cache-hit patch mechanism | **Op-owned override** at the device-operation level, implemented internally as the framework **buffer-binding fast path** plus a hand-written common-runtime-arg patch |
| In-place | Yes — `create_output_tensors` returns the `cache` tensor itself |

## Post-fix status — commit fab067a

**Verdict: CLEAR — with two justified relaxations** (the per-request scalars, and logical-vs-padded
shape). Both of the program-cache bugs this audit found are closed, and so are both caveats: one by a
new hash term, three by `TT_FATAL`s placed in the shared `validate_runtime_args`, and one by a framework
correction this audit had wrong. No program-cache correctness bug remains.

**Placement is the whole story for this op.** Both reproductions below are cache **hits** — call 1 is
a miss that compiles and caches, call 2 is the identical-keyed call that corrupts the KV cache. A guard
placed in `validate_on_program_cache_miss` would have run on call 1, passed it, and then **not run at
all** on call 2, because this op defines `validate_on_program_cache_hit` and the dispatcher substitutes
rather than supplements (see `### Which validator runs on a cache hit`). Every guard fab067a added went
into `validate_runtime_args`, which both validators delegate to — the only placement that works.

**What fab067a changed in this op** (line citations in this section are against the post-fix file; the
analysis further down keeps its original pre-fix citations)

- Added `require_standard_tile` to `validate_runtime_args`
  (`device/update_padded_kv_cache_device_operation.cpp:112-130`), applied to `cache` and `input`
  (`:129-130`). It early-returns for ROW_MAJOR and otherwise asserts 32x32 tile geometry. This closes
  omission 6, the `writer_tile_height` bug.
- Moved `cache.dtype() == input.dtype()` and `cache.layout() == input.layout()` out of the miss
  validator and into `validate_runtime_args` (`:132-144`), leaving pointer comments at the sites they
  came from (`:256`, `:279`). This closes omission 4.
- Added the metadata tensor's memory config to the key
  (`compute_program_hash:360-363`, as `slot_idx.has_value() ? slot_idx->memory_config() : MemoryConfig{}`).
  This closes the hash half of omission 5.
- Added `meta.memory_config() == slot_idx->memory_config()` to the `validate_meta` lambda
  (`:158`, `:174-181`), which closes omission 5's second, miss-path half: `create_descriptor` builds
  **one** `TensorAccessorArgs` from `slot_idx` and the writer uses it for every metadata read, so the
  two metadata tensors must be interchangeable. Nothing asserted that before.
- Moved `cache.storage_type() == StorageType::DEVICE` into `validate_runtime_args` (`:75-80`),
  specifically because `cache.device()` is dereferenced on the next line (`:81`) — on a hit the fault
  used to land there as a null dereference rather than as a message.
- Documented the 32x32 TILE requirement in the nanobind docstring
  (`update_padded_kv_cache_nanobind.cpp:53-54`).

**What remains open**

- The two per-request scalars (`slot_idx`, `kv_actual_global`) stay out of the key by design, patched
  into writer common args on every hit and re-validated on every hit. Unchanged and correct.
- `alignment` of `cache` and `input` (omission 7) is still not hashed. **Regraded to
  `VALID — determined by hashed terms`**: `aligned_page_size` for the TILE branch is `tt::tile_size` of
  the hashed dtype, and for ROW_MAJOR it is the last dimension's byte extent, so it is a function of
  {`dtype`, `layout`, `memory_config`, `padded_shape`} — all hashed — and no public Python path exposes
  a non-canonical `Alignment` independently of those.
- The op still defines `validate_on_program_cache_hit` (`:304`), so the miss validator is still
  replaced rather than supplemented, and everything above the delegation at `:301` remains miss-only.
  That is now benign for every row in the summary table, but the argument must be re-derived whenever
  a check is added or the key is loosened.
- The hard-coded `kWriterKernelHandle = 1` (`:600`) is unchanged. It is a factory-coupling defect, not
  a cache defect; see `## Non-cache correctness defects`.
- Defining a custom hash still forfeits attribute-level collision resolution
  (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:1012-1014`), and this op has no `TensorSpec`
  backstop on the hit path, so a 64-bit collision is still silent in-place corruption. Unchanged.

**Metal 2.0 port**

Clear to port. The op does carry a genuine logical-vs-padded relaxation — omission 3, kept because the
kernels address pages and the page grid is the padded shape, and because prefill chunking legitimately
produces a ragged final chunk whose logical sequence length differs while its padded shape does not.
That relaxation maps exactly onto `TensorSpecRelaxations::match_padded_shape_only`
(`tt_metal/api/tt-metalium/experimental/metal2_host_api/tensor_spec_relaxations.hpp:41,49`), which
`pertinent_fields` resolves to `PertinentFields{.padded_shape = true}`
(`tt_metal/impl/metal2_host_api/tensor_spec_relaxations.cpp:67-87`). Declaring it on the `cache` and
`input` `TensorParameter`s is the correct port: both `hash_tensorspec_with_relaxation` (`:116`) and
`tensorspecs_match_with_relaxation` (`:161-201`) consume that one field set, and `ValidateTensorArgs`
delegates its accept/reject to the predicate (`tt_metal/impl/metal2_host_api/program_run_args.cpp:176-189`),
so the key and validation cannot disagree. What must **not** happen is the pre-fix
`rotary_embedding_indexed` combination — hashing `padded_shape` while leaving the relaxations default —
which relaxes the key and then validates strictly, turning an intended rebuild into a throw from
`report_tensor_arg_mismatch`. See that op's audit, `### The report_tensor_arg_mismatch interaction`, for
the model pattern.

Once ported, the exact-spec predicate would also make omissions 4, 6 and 7 unreachable by construction
rather than by validator, since `tensor_layout` is compared under every relaxation. The `TT_FATAL`s
fab067a added stay worth keeping regardless: they name the offending operand, and they do not depend on
nobody ever setting an extra relaxation flag.

## Cache-hit patch mechanism

Two layers have to be read together here, and the op's classification depends on both.

**Outer layer.** `select_program_factory` always returns `MeshWorkloadFactory`
(`device/update_padded_kv_cache_device_operation.cpp:130-133`), and that factory defines
`override_runtime_arguments` but not `apply_descriptor`. The framework's cache-hit dispatcher
therefore calls the op's own hook on every hit:

```279:285:ttnn/api/ttnn/device_operation.hpp
        if constexpr (requires { &WorkloadFactory::apply_descriptor; }) {
            WorkloadFactory::apply_descriptor(
                cached_mesh_workload, operation_attributes, tensor_args, tensor_return_value);
        } else {
            WorkloadFactory::override_runtime_arguments(
                cached_mesh_workload, operation_attributes, tensor_args, tensor_return_value);
        }
```

**Inner layer.** The op's override immediately delegates to the descriptor adapter and then patches
by hand:

```439:463:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/update_padded_kv_cache_device_operation.cpp
void UpdatePaddedKvCacheDeviceOperation::MeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& args,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    // Default adapter behaviour: patch operand buffer-binding addresses on cache hits.
    descriptor_adapter_t::apply_descriptor(cached_workload, args, tensor_args, output);
    ...
    const bool has_metadata = tensor_args.slot_idx.has_value();
    const uint32_t arg8 = has_metadata ? tensor_args.slot_idx->buffer()->address() : args.slot_idx;
    const uint32_t arg9 = has_metadata ? tensor_args.kv_actual_global->buffer()->address() : args.kv_actual_global;
    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& writer_common = GetCommonRuntimeArgs(program, kWriterKernelHandle);
        TT_FATAL(
            kArg9 < writer_common.size(), "update_padded_kv_cache writer is missing its per-call common runtime args");
        writer_common[kArg8] = arg8;
        writer_common[kArg9] = arg9;
    }
}
```

`descriptor_adapter_t` is `DescriptorMeshWorkloadAdapter<ProgramFactory>` parameterised on
`DescriptorAdapterOperation`, a minimal four-typedef helper
(`device/update_padded_kv_cache_device_operation.hpp:65-78`). Neither that helper nor `ProgramFactory`
declares `override_runtime_arguments` or `get_dynamic_runtime_args`, so
`DescriptorMeshWorkloadAdapter::has_override_runtime_arguments()` is false for the *inner* adapter and
its `apply_descriptor` lands in the buffer-binding branch:

```726:731:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
                    if (!sv.resolved_bindings.rt_args.empty() ||
                        (!dynamic_args.empty() && !sv.resolved_bindings.empty())) {
                        auto collected =
                            collect_tensor_buffers(tensor_args, tensor_return_value, sv.workload_descriptor);
                        tt::tt_metal::apply_resolved_bindings(program, sv.resolved_bindings, collected.buffers);
                        tt::tt_metal::apply_dynamic_runtime_args(program, dynamic_args);
```

`sv.resolved_bindings.rt_args` is non-empty because both kernels register their address slot as a
`Buffer*` (`create_descriptor:416,420`), so the inner path is the fast path, never the slow-path
rebuild. `resolve_bindings` does **not** bail on this op's in-place aliasing: `cache.buffer()` appears
once in the input region and once in the output region, and an output-region entry that aliases an
input is explicitly skipped rather than treated as an ambiguous duplicate
(`tt_metal/impl/program/program_descriptor_patching.cpp:90-94`).

**Obligation on the hash.** On a hit, exactly three things get refreshed: the `input` and `cache`
buffer addresses (via `resolved_bindings`), and writer common runtime args 8 and 9. Everything else —
common args 0-7, both kernels' per-core runtime args, every compile-time arg, the CB page sizes and
`total_size`, and the core ranges — is frozen at the first miss. So every one of those must be a pure
function of the hashed set.

Note also that having `override_runtime_arguments` on the *outer* factory does **not** make
`resolve_bindings` unnecessary here, because the op delegates to the inner adapter rather than
re-deriving addresses itself. The aliasing safety comes from `resolve_bindings`'s output-region skip,
not from the mode-A bypass described for ops that hand-roll their whole cache-hit path.

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
validator is a single delegation:

```189:194:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/update_padded_kv_cache_device_operation.cpp
void UpdatePaddedKvCacheDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // Re-run the non-hashed structural checks on every hit. slot_idx and kv_actual_global are no longer
    // host attributes -- they live in the metadata tensor and are validated host-side by the caller.
    validate_runtime_args(args, tensor_args);
}
```

The miss validator also ends by delegating to `validate_runtime_args` (`:186`), so the two paths differ
by exactly the checks the miss validator performs *before* that delegation — lines 140 through 182. The
hit path therefore loses all of the following:

- `cache.storage_type() == DEVICE` and `input.storage_type() == DEVICE` (`:140-141`). **fab067a moved
  the `cache` half into `validate_runtime_args`**, and the `input` half turns out never to have been a
  real loss — see the framework correction under omission 8.
- `cache.dtype() == input.dtype()` (`:142`). **fab067a moved this into `validate_runtime_args`**, which
  is what closes omission 4.
- `cache.layout() == input.layout()` (`:149`) — **likewise moved** — and the TILE-or-ROW_MAJOR gate and
  the block-float and FP8_E4M3 layout gates (`:150-156`), which stay miss-only and are unreachable
  because they constrain the hashed `input.layout()` / `input.dtype()`.
- The rank-4 checks, the head-dim and num-heads equalities (`:163-166`).
- The seq tile-alignment and `cache_seq % input_seq` checks (`:173-175`).
- `num_layers > 0` and `cache_shape[0] % num_layers == 0` (`:177-182`).

What *does* run on both paths is `validate_runtime_args` (`:57-126`): the `cluster_axis` and
`layer_idx` range checks, the 2D-mesh check, the paired-optional check, the whole `validate_meta`
lambda for both metadata tensors (`:85-100`, including `!meta.is_sharded()` at `:94`), and the
scalar-path `slot_idx` / `kv_actual_global` value checks.

Pre-fix, this was the reason omissions 4 and 8 below were graded `CAVEAT — pinned only on the miss path`
rather than `VALID — pinned by validation`: their `TT_FATAL`s sat in the block the hit path skips. It is
also the reason every guard recommended at the end of this document is specified to go into
`validate_runtime_args` — a guard added to `validate_on_program_cache_miss` would never run on the
offending second call, which is the only call that matters for a cache bug.

**fab067a acted on exactly that.** It moved the `cache`-vs-`input` dtype and layout checks and the
`cache` storage check down into `validate_runtime_args`, and put the new tile guard there too, rather
than in the miss validator. The structural hazard — a narrow hit validator that disables everything
above it — is unchanged; what changed is that nothing load-bearing is left above the delegation.

**Which of the dropped checks are actually reachable.** The list above is the mechanical diff, but most
of those checks constrain values that are themselves in the cache key, and a miss-only pin on a *hashed*
value cannot be evaded: any call carrying a new value of that parameter misses, and the miss validator
runs and rejects it there. Filtering the list against
`compute_program_hash:223-233` leaves only three lines that a hit can actually reach:

| Dropped check | Constrains | In the key? | Reachable on a hit? |
|---|---|---|---|
| `storage_type() == DEVICE` ×2 (`:140-141`) | storage variant kind | No | No — pinned by the framework in `launch()`; see omission 8 |
| `cache.dtype() == input.dtype()` (`:142`) | `cache.dtype()` | No (only `input.dtype()` is) | No longer — **moved into `validate_runtime_args` by fab067a** |
| `cache.layout() == input.layout()` (`:149`) | `cache.layout()` | No (only `input.layout()` is) | No longer — **moved into `validate_runtime_args` by fab067a** |
| TILE-or-ROW_MAJOR, block-float and FP8 gates (`:150-156`) | `input.layout()`, `input.dtype()` | Yes, both | No |
| Rank-4, head-dim, num-heads (`:163-166`) | both padded shapes | Yes, both | No |
| Seq alignment, `cache_seq % input_seq` (`:173-175`) | both padded shapes | Yes, both | No |
| `num_layers > 0`, batch divisibility (`:177-182`) | `num_layers`, `cache.padded_shape()` | Yes, both | No |

Pre-fix the practical loss was `:140-141`, `:142` and `:149` — four `TT_FATAL`s, not forty lines — and
of those, only `:142` and `:149` failed *silently*; the storage pair merely relocated a crash. That is
the split the recommendations at the end of this document acted on, and it is what fab067a implemented:
the two silent rows were moved onto the hit path, the storage rows were left alone (and have since been
shown to be unreachable anyway). Post-fix, nothing in this table is both reachable and unguarded.

## Baseline: what the default hash would cover

`hash_objects_with_default_seed(type_hash<UpdatePaddedKvCacheDeviceOperation>, attrs, tensor_args)`
walks reflection, giving:

| Source | Fields |
|---|---|
| `operation_attributes` | `slot_idx`, `kv_actual_global`, `layer_idx`, `num_layers`, `cluster_axis` |
| `cache` | storage variant kind, `logical_shape`, `dtype`, `page_config`, `memory_config`, `alignment` |
| `input` | storage variant kind, `logical_shape`, `dtype`, `page_config`, `memory_config`, `alignment` |
| `slot_idx` (optional tensor) | engaged/disengaged, plus the same six fields when engaged |
| `kv_actual_global` (optional tensor) | engaged/disengaged, plus the same six fields when engaged |

`padded_shape` is not in the default key directly — it is a derivation of `logical_shape`,
`page_config` and `alignment`. The mesh coordinates are appended by the framework
(`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:989-992`) on both the default and custom paths, so
`my_sp_coord` is never an omission.

## What the custom hash covers

```348:363:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/update_padded_kv_cache_device_operation.cpp
    return tt::tt_metal::operation::hash_operation<UpdatePaddedKvCacheDeviceOperation>(
        tensor_args.slot_idx.has_value(),
        tensor_args.valid_global.has_value() || args.valid_global.has_value(),
        args.layer_idx,
        args.num_layers,
        args.cluster_axis,
        input.dtype(),
        input.layout(),  // TILE vs ROW_MAJOR drives the page-unit math; must not collide
        input.memory_config(),
        input.padded_shape(),
        cache.memory_config(),
        cache.padded_shape(),
        // On the metadata path the writer bakes a TensorAccessorArgs built from slot_idx into its
        // compile-time args, so the bank table it selects cannot be refreshed on a cache hit. Only the
        // config is keyed, never the value; kv_actual_global is pinned to match by validate_runtime_args.
        tensor_args.slot_idx.has_value() ? tensor_args.slot_idx->memory_config() : MemoryConfig{});
```

The two per-request scalars are dropped, and both real operands are decomposed selectively — with the
notable asymmetry that `input` contributes `dtype` and `layout` while `cache` contributes neither.
`cache`'s dtype and layout are instead pinned equal to `input`'s inside `validate_runtime_args`
(`:132-144`), which runs on both paths.

**Pre-fix the metadata tensors collapsed to a single `has_value()` bit.** fab067a added the final term:
`slot_idx`'s `memory_config` when the metadata path is taken, a neutral `MemoryConfig{}` otherwise. That
is what closes omission 5 — the buffer type and aligned page size baked into the writer's metadata
`TensorAccessorArgs` are now keyed. The ternary is deliberate: it must not dereference a disengaged
optional, and a real metadata tensor's config never equals a default-constructed one.

## Omitted parameters

### 1. `operation_attributes.kv_actual_global`

**Verdict: VALID — patched.**

This is the per-step index the task brief flags as the classic hazard: the prior valid global KV length
in tokens, which advances on every prefill chunk. It reaches the writer as common runtime arg 9 on the
scalar path:

```384:395:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/update_padded_kv_cache_device_operation.cpp
        writer_kernel.emplace_common_runtime_args({
            my_sp_coord,
            sp_factor,
            input_Ht,
            args.layer_idx,
            args.num_layers,
            Wt,
            cache_HtWt,
            cache_CHtWt,
            args.slot_idx,
            args.kv_actual_global,
        });
```

and is re-applied on every hit at `override_runtime_arguments:456,462`. The kernel derives the entire
write offset from it on-device — no host-computed offset is baked anywhere:

```108:122:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/kernels/dataflow/writer_update_padded_kv_cache.cpp
    const uint32_t chunk_global_t = sp_factor * chunk_local_t;
    const uint32_t boundary_slab_idx = kv_actual_global_t / chunk_global_t;
    const uint32_t boundary_chip = (kv_actual_global_t / chunk_local_t) % sp_factor;
    const uint32_t boundary_offset_t = kv_actual_global_t % chunk_local_t;

    // From the current slab base, chips before the boundary advance a full slab, the boundary chip
    // advances by its pad offset, and chips after it stay at the base.
    const uint32_t update_idxt =
        boundary_slab_idx * chunk_local_t +
        (my_sp_coord < boundary_chip ? chunk_local_t : (my_sp_coord == boundary_chip ? boundary_offset_t : 0));

    const uint32_t input_Ht = chunk_local_t;
    const uint32_t start_idx = batch_idx * cache_CHtWt + update_idxt * Wt;
```

This is the omitted-and-patched pattern done correctly. The work split (`num_blocks_of_work`, per-core
`num_blocks_per_core`, `num_blocks_written`) is a function of `input_shape[1] * input_Ht` and the
compute grid only (`create_descriptor:293-297`), so a changing `kv_actual_global` never shifts core
membership — the frozen per-core args stay correct.

On the metadata path the value is not a host scalar at all: the writer NoC-reads element [0] of a
1-element uint32 tensor (`writer_update_padded_kv_cache.cpp:74-92`). There the omission is trivially
correct — it is device data, and only the tensor's address (common arg 8/9) needs patching, which the
override does.

Because the value is not hashed, `validate_on_program_cache_hit` re-runs the range and alignment
checks on every dispatch rather than only on a miss:

```103:124:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/update_padded_kv_cache_device_operation.cpp
    if (!tensor_args.slot_idx.has_value()) {
        // The writer divides kv_actual_global by TILE_HEIGHT to get its tile offset, so it must be aligned.
        TT_FATAL(
            args.kv_actual_global % TILE_HEIGHT == 0,
            "kv_actual_global ({}) must be tile-aligned (a multiple of {})",
            args.kv_actual_global,
            TILE_HEIGHT);
        const uint32_t num_slots = cache.padded_shape()[0] / args.num_layers;
        TT_FATAL(args.slot_idx < num_slots, "slot_idx ({}) out of range for num_slots ({})", args.slot_idx, num_slots);

        // This chunk is written at a per-chip offset derived from kv_actual_global; the prior valid KV
        // plus this chunk must fit the global cache capacity (sp_factor slabs of cache_seq tokens each),
        // else the write spills past the cache. sp_factor = mesh extent along cluster_axis.
        const uint32_t sp_factor = (args.cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();
        const uint32_t chunk_global_tokens = sp_factor * tensor_args.input.padded_shape()[-2];
        const uint32_t global_cache_capacity = sp_factor * cache.padded_shape()[-2];
        TT_FATAL(
            args.kv_actual_global + chunk_global_tokens <= global_cache_capacity,
            "kv_actual_global ({}) + chunk_global ({}) would overflow global cache capacity ({})",
            args.kv_actual_global,
            chunk_global_tokens,
            global_cache_capacity);
    }
```

That is the right structure: the hash relaxation is paid for with a per-hit validator.

### 2. `operation_attributes.slot_idx`

**Verdict: VALID — patched.**

Same mechanism as omission 1, one arg over: created at `create_descriptor:393`, re-applied at
`override_runtime_arguments:455,461`, consumed on-device only as
`batch_idx = slot_idx * num_layers + layer_idx` (`writer_update_padded_kv_cache.cpp:101`), which feeds
the page index and nothing structural. `num_layers` and `layer_idx` are both hashed, so the
linearisation itself is pinned; only the free variable is omitted. Range-checked on every hit at
`device/update_padded_kv_cache_device_operation.cpp:110-111`.

### 3. `cache.logical_shape()` and `input.logical_shape()` — replaced by `padded_shape()`

**Verdict: VALID — relaxation win.** Unchanged by fab067a, and the one relaxation this op should carry
forward into its Metal 2.0 port as `match_padded_shape_only`.

`create_descriptor` reads padded shapes exclusively:

```251:282:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/update_padded_kv_cache_device_operation.cpp
    const auto& cache_shape = cache.padded_shape();
    const auto& input_shape = input.padded_shape();

    const tt::DataFormat data_format = datatype_to_dataformat_converter(input.dtype());
    ...
    if (is_row_major) {
        // ROW_MAJOR: page = one token row; use the buffer's aligned page size (handles row padding).
        single_page_size = cache.buffer()->aligned_page_size();
        Wt = 1;
        input_Ht = input_shape[-2];
        cache_HtWt = cache_shape[-2];
        writer_tile_height = 1;
    } else {
        single_page_size = tt::tile_size(data_format);
        Wt = cache_shape[-1] / TILE_WIDTH;
        input_Ht = input_shape[-2] / TILE_HEIGHT;
        cache_HtWt = cache_shape[-2] * Wt / TILE_HEIGHT;
        writer_tile_height = TILE_HEIGHT;
    }
    const uint32_t cache_CHtWt = cache_shape[1] * cache_HtWt;
```

The op is a page copy; the kernels address pages, and the page grid is the padded shape. Two prefill
chunks whose logical sequence lengths differ but pad to the same tile-aligned padded shape correctly
share one program, which the default hash would have forced apart. Since the output is the cache
tensor itself (`create_output_tensors:202-206`), there is no freshly-derived output spec that could go
stale from this relaxation.

The value is concrete rather than theoretical, which is why it survives as a relaxation where
`rotary_embedding_indexed`'s equivalent did not. DeepSeek prefill chunks a sequence into fixed-size
slabs, and the final chunk is ragged: a logical sequence length of 4090 and one of 4096 both pad to a
padded sequence of 4096, so they share the page grid, the work split and every compile-time arg. Under
the default hash they would compile two identical programs. On the Metal 2.0 port this is what
`TensorSpecRelaxations::match_padded_shape_only`
(`tt_metal/api/tt-metalium/experimental/metal2_host_api/tensor_spec_relaxations.hpp:41,49`) expresses
directly; `pertinent_fields` reduces it to `PertinentFields{.padded_shape = true}`
(`tt_metal/impl/metal2_host_api/tensor_spec_relaxations.cpp:67-87`), which is exactly the field this
hash already keys, so the port is a declaration rather than a redesign.

### 4. `cache.dtype()` and `cache.layout()`

**Verdict: RESOLVED by fab067a** (was CAVEAT). The two `TT_FATAL`s that pin them to the hashed `input`
values were moved out of `validate_on_program_cache_miss` and into the shared `validate_runtime_args`
(`:132-144`), so they now run on the cache-hit path as well as the miss path. That placement is the
whole fix: the reproduction below is a **hit**, so the checks were passing call 1 and never executing on
call 2. With them in the shared function, `cache.dtype()` and `cache.layout()` are provably redundant
with hashed values on every dispatch, which is what `VALID — pinned by validation` requires.

**Pre-fix:** both are genuinely consumed. `cache.layout()` selects the ROW_MAJOR branch above via the
*input's* layout, and `cache.dtype()` + `cache.layout()` together determine `cache.buffer()->aligned_page_size()`,
which becomes a writer **compile-time** arg through the tensor accessor:

```345:351:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/update_padded_kv_cache_device_operation.cpp
    KernelDescriptor::CompileTimeArgs writer_compile_args = {
        kSrcCbIndex, static_cast<uint32_t>(has_metadata), has_metadata ? kMetaCbIndex : 0u, writer_tile_height};
    TensorAccessorArgs(cache.buffer()).append_to(writer_compile_args);
    if (has_metadata) {
        // One accessor reused for both 1-element tensors (identical layout).
        TensorAccessorArgs(tensor_args.slot_idx->buffer()).append_to(writer_compile_args);
    }
```

`TensorAccessorArgs::append_to` emits the args-config word and the aligned page size as compile-time
args for a non-sharded buffer (`tt_metal/impl/buffers/tensor_accessor_args.cpp:196-205`), and
compile-time args are baked into the cached `Program` — nothing on the hit path refreshes them.

What made this safe on the miss path was the pair of consistency checks:

```142:150:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/update_padded_kv_cache_device_operation.cpp
    TT_FATAL(cache.dtype() == input.dtype(), "cache and input dtype must match");
    ...
    TT_FATAL(cache.layout() == input.layout(), "cache and input layout must match");
    TT_FATAL(input.layout() == Layout::TILE || input.layout() == Layout::ROW_MAJOR, "layout must be TILE or ROW_MAJOR");
```

Given those, `cache.dtype()` and `cache.layout()` carry no information beyond the hashed
`input.dtype()` / `input.layout()`. The unchecked assumption was that they hold *on a cache hit* — and
pre-fix they were not re-asserted there. `validate_on_program_cache_hit` calls only
`validate_runtime_args` (`device/update_padded_kv_cache_device_operation.cpp:189-194`), which did not
contain either check.

What broke it — and note that this is a cache **hit**, which is why the placement of the fix is the
whole point: call 1 with `cache` and `input` both `BFLOAT16`, TILE, `cache` padded `[1,1,128,128]`,
`input` padded `[1,1,32,128]`, same memory configs — compiles and caches, and the miss validator's line
142 runs and passes. Call 2 with an identical `input` but a `cache` of dtype `BFLOAT8_B` at the same
padded shape and memory config. The hash is byte-identical (`cache.dtype()` is not in it), so the cache
hits, **the miss validator does not run at all**, the hit validator does not catch the mismatch, and the
writer runs with a compile-time page size of 2048 bytes against a buffer whose pages are 1088 bytes —
the KV cache is overwritten at the wrong stride. Strengthening line 142 would have changed nothing:
call 2 never reaches it.

**The guard fab067a landed** is the one this section named — move the two `TT_FATAL`s out of
`validate_on_program_cache_miss` and into `validate_runtime_args` so they run on both paths:

```132:144:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/update_padded_kv_cache_device_operation.cpp
    // cache's dtype and layout are absent from the key (only input's are hashed), and both set the
    // writer's page size, so they have to be re-checked on hits too: a second call that changes only
    // the cache would otherwise reuse a program built for the first one's page geometry.
    TT_FATAL(
        cache.dtype() == tensor_args.input.dtype(),
        "cache and input dtype must match (got {} and {})",
        cache.dtype(),
        tensor_args.input.dtype());
    TT_FATAL(
        cache.layout() == tensor_args.input.layout(),
        "cache and input layout must match (got {} and {})",
        cache.layout(),
        tensor_args.input.layout());
```

The sites they came from now carry pointer comments (`:256`, `:279`) so the next reader does not restore
them to the miss validator. Adding `cache.dtype()` / `cache.layout()` to the hash would also have closed
it, but is strictly worse — it grows the key with values that are constrained to be redundant.

### 5. The metadata tensors' specs — only `slot_idx.has_value()` is hashed

**Verdict: RESOLVED by fab067a** (was BUG). Closed on both halves. The key now carries
`slot_idx.has_value() ? slot_idx->memory_config() : MemoryConfig{}` (`compute_program_hash:360-363`), so
the buffer type and aligned page size baked into the writer's metadata `TensorAccessorArgs` are keyed
and the reproduction's call 2 now **misses** and compiles a second program. Separately,
`validate_meta` now asserts that both metadata tensors carry the same `memory_config` as `slot_idx`
(`:174-181`), which closes the miss-path half where one accessor served two potentially
non-interchangeable tensors. That check sits inside `validate_runtime_args`, so it holds on hits too.

**Pre-fix:** the metadata tensor's memory space and page size became writer compile-time args and were
neither hashed nor patchable.

On the metadata path the op appends a second `TensorAccessorArgs` built from
`tensor_args.slot_idx->buffer()` (line 350, quoted above). The emitted compile-time args are
`args_config.raw()` — which carries the `IsDram` and `Sharded` bits
(`tt_metal/impl/buffers/tensor_accessor_args.cpp:153-157`) — followed by
`buffer_->aligned_page_size()` (`:197-205`). Both depend on the metadata tensor's `memory_config`
and `alignment`. The hash contains only `tensor_args.slot_idx.has_value()`.

Validation pins several properties of these tensors, and unlike omission 4 these checks *do* run on
every hit (`validate_on_program_cache_hit` → `validate_runtime_args`):

```84:101:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/update_padded_kv_cache_device_operation.cpp
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
            TT_FATAL(!meta.is_sharded(), "metadata tensor {} must not be sharded", name);
            // The writer resolves meta.buffer()->address() against cache.device(); a tensor on a
            // different mesh device would bake the wrong address and fail obscurely downstream.
            TT_FATAL(meta.device() == cache.device(), "metadata tensor {} must be on the same device as cache", name);
        };
        validate_meta(tensor_args.slot_idx.value(), "slot_idx");
        validate_meta(tensor_args.kv_actual_global.value(), "kv_actual_global");
    }
```

Dtype, layout, element count and shardedness are all pinned. **Buffer type was not.** An interleaved
L1 metadata tensor passed every one of these checks.

Two-call reproduction (pre-fix). Note again that call 2 is a **hit**: strengthening
`validate_on_program_cache_miss` would have been useless, because the miss validator does not run on
the call that corrupts the cache.

- **Call 1:** `update_padded_kv_cache(cache, input, slot_idx_t, kv_t, 0, 0, layer_idx=0, num_layers=61,
  cluster_axis=1)` with `slot_idx_t` and `kv_t` allocated `MemoryConfig{INTERLEAVED, DRAM}`. Miss.
  The writer compiles with the metadata accessor's `IsDram` bit set and the DRAM-aligned page size.
- **Call 2:** identical, except `slot_idx_t` and `kv_t` are allocated
  `MemoryConfig{INTERLEAVED, L1}`. The hash is unchanged — only `has_value()` participates — so this
  is a cache hit. `override_runtime_arguments` patches common args 8/9 to the new addresses, which is
  all it can do; the compile-time accessor config is baked.
- **Stale slot:** the metadata `TensorAccessorArgs<kMetaArgsOffset>` compile-time block in
  `writer_update_padded_kv_cache.cpp:72-73`. The kernel resolves the L1 address through DRAM banking.
- **Symptom:** `slot_idx` and `kv_actual_global` are read as garbage, so
  `batch_idx = slot_idx * num_layers + layer_idx` and `update_idxt` point somewhere arbitrary in the
  cache. The chunk is written over another user's or another layer's KV, silently. No PCC check on
  this op's own output would catch it — the op returns the cache handle unchanged.

The same omission also made the miss path fragile in a way worth recording: line 350 builds **one**
accessor from `slot_idx` and the writer uses it for both reads
(`writer_update_padded_kv_cache.cpp:79,88`), but nothing asserted that
`tensor_args.kv_actual_global` had the same buffer type and page size. A DRAM `slot_idx` paired with an
L1 `kv_actual_global` was wrong even on a fresh compile. **fab067a closes this half too**, with a
mutual-config assertion inside `validate_meta`:

```174:181:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/update_padded_kv_cache_device_operation.cpp
            TT_FATAL(
                meta.memory_config() == meta_memory_config,
                "update_padded_kv_cache does not currently support metadata tensors with different memory "
                "configs: one TensorAccessor serves every metadata read, so {} must match slot_idx (got "
                "buffer types {} and {})",
                name,
                meta.memory_config().buffer_type(),
                meta_memory_config.buffer_type());
```

That pairing is what makes hashing `slot_idx`'s config alone sufficient: every metadata tensor is now
required to agree with it, so one keyed config covers the accessor that serves them all.

**This omission was family-wide.** `zero_padded_kv_cache` had the identical construction (its
`create_descriptor` appends `TensorAccessorArgs(tensor_args.slot_idx->buffer())` to both the reader and
writer compile args, hashing only `has_value()`), and its metadata validator was a strict subset of this
one — it omitted even the `!is_sharded()` check. By contrast the third member of this family,
`rotary_embedding_indexed`, always hashed something of its metadata tensor's spec alongside
`metadata.has_value()`, and now hashes the whole spec. That asymmetry was strong evidence the omission
here was an oversight rather than a deliberate relaxation, and fab067a applied the fix across the family
— including the missing `!is_sharded()` check in `zero_padded_kv_cache`.

### 6. `page_config` (the `Tile`) of `cache` and `input` — the unguarded 32x32 assumption

**Verdict: RESOLVED by fab067a** (was BUG). `require_standard_tile` in the shared
`validate_runtime_args` (`:112-130`) rejects a non-32x32 `Tile` on `cache` and `input` before the key is
even consulted, on every dispatch. The factory is still 32x32-only and `page_config` is still not
hashed — but that combination is now correct by construction, because the only tile the op admits is the
one the factory assumes.

**Placement, again, is the fix.** The reproduction below is a cache **hit**. A tile guard in
`validate_on_program_cache_miss` would have run on the `Tile{32, 32}` call, passed it, and then not run
on the `Tile{16, 32}` call that corrupts the cache — the guard would have been in the codebase and the
bug would have been unchanged. fab067a put it in `validate_runtime_args`, which both validators delegate
to, so it covers the hit path and the miss path with one placement:

```112:130:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/update_padded_kv_cache_device_operation.cpp
    // create_descriptor's TILE branch derives its page size, Wt/input_Ht/cache_HtWt and the
    // writer_tile_height compile arg from the architectural 32x32 constants, and the tile is absent
    // from compute_program_hash. Checked here rather than in the miss validator so it also runs on the
    // cache-hit path, where a non-standard tile would otherwise alias onto a cached 32x32 program.
    const auto require_standard_tile = [](const Tensor& tensor, const char* name) {
        if (tensor.layout() != Layout::TILE) {
            return;
        }
        const auto tile = tensor.tensor_spec().tile();
        TT_FATAL(
            tile.get_height() == TILE_HEIGHT && tile.get_width() == TILE_WIDTH,
            "update_padded_kv_cache does not currently support tiles other than 32x32, but {} has a {}x{} "
            "tile",
            name,
            tile.get_height(),
            tile.get_width());
    };
    require_standard_tile(cache, "cache");
    require_standard_tile(tensor_args.input, "input");
```

The ROW_MAJOR early-return matters: `PageConfig::get_tile()` reports a default 32x32 tile for ROW_MAJOR,
so an unconditional check would have been vacuous there rather than wrong, but the guard is honest about
only constraining the branch that reads a tile.

**Pre-fix:** the op accepted `Layout::TILE`, computed all of its tile geometry from the architectural
32x32 constants rather than the tensor's actual `Tile`, validated nothing about the tile, and did not
hash `page_config`. A non-32x32 tensor therefore did not even get a freshly-built wrong program — it
silently inherited the cache entry built for a 32x32 tensor of the same padded shape.

Non-32x32 tiles are a supported TTNN configuration, so this was reachable rather than hypothetical.

**The factory is entirely 32x32-hardcoded.** The TILE branch derives every page-unit quantity from
`tt::tile_size` (which returns the byte size of a 32x32 tile, not `tile.get_tile_size(format)`) and from
bare `TILE_WIDTH`/`TILE_HEIGHT`:

```275:281:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/update_padded_kv_cache_device_operation.cpp
    } else {
        single_page_size = tt::tile_size(data_format);
        Wt = cache_shape[-1] / TILE_WIDTH;
        input_Ht = input_shape[-2] / TILE_HEIGHT;
        cache_HtWt = cache_shape[-2] * Wt / TILE_HEIGHT;
        writer_tile_height = TILE_HEIGHT;
    }
```

Pre-fix there was no `tensor_spec().tile()` read and no tile-geometry `TT_FATAL` anywhere in the op
directory. The seq-alignment checks that do exist are shape checks against the same architectural
constant, not tile checks, so they do not incidentally pin the geometry:

```170:174:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/update_padded_kv_cache_device_operation.cpp
    // Seq / offset arithmetic stays tile-granular (multiples of 32) in BOTH layouts: the writer's
    // update_idxt boundary math counts tile-rows even when ROW_MAJOR makes each page a single token
    // row, so input/cache seq must be 32-aligned regardless of layout.
    TT_FATAL(input_seq % TILE_HEIGHT == 0, "input seq dim ({}) must be tile-aligned", input_seq);
    TT_FATAL(cache_seq % TILE_HEIGHT == 0, "cache seq dim ({}) must be tile-aligned", cache_seq);
```

A padded sequence dimension of 32 satisfies both regardless of whether that is one 32-row tile or two
16-row tiles.

**This op is structurally worse than its two siblings, because `writer_tile_height` is a
compile-time arg.** Line 280 above assigns `writer_tile_height = TILE_HEIGHT`, and line 346 pushes it
into the writer's compile-time argument list at index 3:

```345:347:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/update_padded_kv_cache_device_operation.cpp
    KernelDescriptor::CompileTimeArgs writer_compile_args = {
        kSrcCbIndex, static_cast<uint32_t>(has_metadata), has_metadata ? kMetaCbIndex : 0u, writer_tile_height};
    TensorAccessorArgs(cache.buffer()).append_to(writer_compile_args);
```

where the kernel consumes it as a `constexpr` and uses it as the divisor that converts the per-request
token count into the page-row unit:

```49:51:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/kernels/dataflow/writer_update_padded_kv_cache.cpp
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr uint32_t tile_height = get_compile_time_arg_val(3);
    constexpr auto cache_args = TensorAccessorArgs<4>();
```

```96:97:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/kernels/dataflow/writer_update_padded_kv_cache.cpp
        slot_idx = get_common_arg_val<uint32_t>(8);
        kv_actual_global_t = get_common_arg_val<uint32_t>(9) / tile_height;
```

That matters because compile-time args are baked into the cached `Program` and are refreshed by no
cache-hit path at all — not the buffer-binding fast path, not the op's own
`override_runtime_arguments`, and not even the mode-C slow-path rebuild
(`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:748-753` re-applies runtime args only). A stale
runtime arg could at least in principle be patched by extending the override; a stale compile-time arg
can only be fixed by hashing the value or by rejecting the input.

**Two-call reproduction (pre-fix).**

- **Call 1:** `cache` and `input` both `BFLOAT16`, `Layout::TILE`, `Tile{32, 32}`, interleaved DRAM,
  `cache` padded `[1, 1, 128, 128]` and `input` padded `[1, 1, 32, 128]`;
  `layer_idx=0, num_layers=61, cluster_axis=1`, scalar path. Miss; the program compiles with
  `writer_tile_height = 32`, `Wt = 4`, `input_Ht = 1`, `cache_HtWt = 16`, a cache accessor whose
  compile-time `aligned_page_size` is 2048 bytes, and a source CB of `2 * 2048` bytes.
- **Call 2:** identical in every hashed respect — same dtypes, same `Layout::TILE`, same memory
  configs, same padded shapes — but both tensors carry `Tile{16, 32}`. The `Tile` lives inside
  `page_config`, which this hash does not include
  (`compute_program_hash:223-233` keeps `input.dtype/layout/memory_config/padded_shape` and
  `cache.memory_config/padded_shape` only), so the key is byte-identical and the cache hits.
  `validate_on_program_cache_hit` runs `validate_runtime_args`, which contains no tile check.
- **Stale slots:** writer compile-time arg 3 (`writer_tile_height`) stays 32 where the tensor's rows
  per page is 16; the writer's and reader's `TensorAccessorArgs` compile-time `aligned_page_size`
  stays 2048 where the real page is 1024 bytes; the source CB's `page_size` and `total_size`
  (`create_descriptor:302-310`) stay sized for 2048-byte pages; and common args 2, 5, 6, 7
  (`input_Ht`, `Wt`, `cache_HtWt`, `cache_CHtWt`) all remain the values computed by dividing by 32.
- **Symptom:** two independent corruptions compound. The writer computes
  `kv_actual_global_t = kv_actual_global / 32` instead of `/ 16`, so `update_idxt` is half the correct
  page-row offset and the chunk is written at the wrong sequence position. Each page copy then moves
  2048 bytes into a 1024-byte page (`noc.async_write(cb, s, page_bytes, {}, {.page_id = i})` at
  `writer_update_padded_kv_cache.cpp:134`, with `page_bytes` read from the stale CB), overrunning into
  the following page. The result is silent KV-cache corruption plus an out-of-bounds DRAM write past
  the last page, with no cache miss anywhere to hint at the cause.

**This defect was family-wide.** `zero_padded_kv_cache` had the identical shape — `tt::tile_size` at its
lines 315 and 317, `Wt`/`cache_H_pages` via bare `TILE_WIDTH`/`TILE_HEIGHT` at 251-252, and no tile
guard. `rotary_embedding_indexed` hardcodes 32x32 just as thoroughly (five `tt::tile_size` calls and
four bare-constant tile-count conversions) and also had no guard, yet its verdict was only CAVEAT —
purely because it dispatches through Metal 2.0 `UpdateProgramRunArgs`, whose exact `TensorSpec`
equality check covers `page_config` and therefore threw on the mismatched second call instead of
executing it. This op goes through the descriptor buffer-binding fast path
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
Making the factory genuinely tile-aware instead (reading `tile().get_tile_shape()` and
`tile.get_tile_size(format)`) remains valid, but then the program provably varies with `Tile` and
`page_config` must be added to `compute_program_hash` in the same change.

### 7. `alignment` of `cache` and `input`

**Verdict: VALID — determined by hashed terms** (was CAVEAT). This is a regrade on closer tracing
rather than a change fab067a made. `alignment` never moves `aligned_page_size` independently of terms
already in the key, so it carries no information: in the TILE branch the page size is `tt::tile_size` of
the hashed `dtype` with the tile pinned to 32x32 by `require_standard_tile`, and in the ROW_MAJOR branch
it is the last padded dimension's byte extent — a function of the hashed `padded_shape` and `dtype`.
Combined with the hashed `memory_config`, which fixes the buffer type and therefore the canonical
alignment, the hashed four determine every baked page quantity.

The residual is narrow and stays worth recording: `alignment` is not read as a tensor property, but it
does move `aligned_page_size`, which is a compile-time arg.

`Buffer::aligned_page_size()` is a function of the page size and the buffer alignment, and it appears
in three baked places — the input accessor (line 328), the cache accessor (line 347), and the CB
total size:

```302:310:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/update_padded_kv_cache_device_operation.cpp
    desc.cbs.push_back(CBDescriptor{
        .total_size = kNumInputPagesDoubleBuffered * single_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = kSrcCbIndex,
            .data_format = data_format,
            .page_size = single_page_size,
        }}},
    });
```

CB sizes are baked into the cached `Program` and are not refreshed on a hit. The hashed
{`dtype`, `layout`, `memory_config`, `padded_shape`} pins all of this for any tensor built the ordinary
way, because the alignment is then the canonical one for the buffer type. The residual is a
`TensorLayout` constructed with an explicit non-canonical `Alignment` that leaves the hashed four
unchanged. No Python path can produce that: every `TensorSpec` constructor exposed through nanobind
routes to the three-argument `TensorLayout` constructor, which *computes* the canonical alignment from
dtype, page config and memory config rather than accepting one. Only a C++ caller reaching
`ttnn::prim::update_padded_kv_cache` directly with a hand-built `TensorLayout` could reach it, which is
outside the public API this audit grades against — hence the regrade from `CAVEAT` to
`VALID — determined by hashed terms`. Unlike the tile in omission 6, no supported TTNN configuration
reaches it.

### 8. `cache.storage` and `input.storage` variant kind

**Verdict: VALID — pinned by the framework** (was CAVEAT — pinned only on the miss path). This is a
correction to the pre-fix grade rather than something fab067a changed: device storage was never a
reachable omission on any path. See `#### Framework correction` below. fab067a did move the `cache` half
into `validate_runtime_args` (`:75-80`), but for a different and local reason — it guards a
`cache.device()` dereference on the next line.

```140:141:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/update_padded_kv_cache_device_operation.cpp
    TT_FATAL(cache.storage_type() == StorageType::DEVICE, "cache must be on device");
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "input must be on device");
```

Both `TT_FATAL`s sat above the `validate_runtime_args` delegation at `:186`, so under the dispatcher
branch quoted in `## Cache-hit patch mechanism` they ran on the first call and never again. That is the
same structural shape as omission 4 — and it is why the pre-fix document graded the two consistently.
The grades should not in fact have matched, because the two omissions differ in whether the *value* can
vary at all, and this one cannot.

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
`std::optional<Tensor>`, so the two metadata tensors are covered as well as `cache` and `input`. And it
is unconditional — no op can opt out.

So the storage variant kind cannot vary along any reachable path. It is a zero-value omission, not a
caveat: there is no admissible call in which it differs, on either path. The pre-fix reasoning — that a
host tensor would fault in `collect_tensor_buffers` rather than corrupt, so the row was a low-severity
caveat — was arguing about the *consequence* of something that cannot happen. The same correction
applies to the `buffer() != nullptr` reasoning throughout this document, since `is_allocated()` is
asserted on the same lines.

This also strengthens rather than weakens the recommendation not to move these pins onto the hit path:
they would cost two `storage_type()` queries on every decode step of every layer to re-check something
the framework has already made impossible. fab067a's move of the `cache` half is not a counter-example —
it is there because `cache.device()` is dereferenced immediately afterwards and a null dereference is a
worse diagnostic than a message, not because the storage kind could reach the key.

### 9. Buffer addresses of `cache` and `input`

**Verdict: VALID — patched, and required.** Addresses must never be hashed. Both are declared as
`Buffer*` bindings, which is what puts the inner adapter on the fast path:

```411:423:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/update_padded_kv_cache/device/update_padded_kv_cache_device_operation.cpp
    for (uint32_t i = 0; i < num_cores; ++i) {
        const CoreCoord& core = cores.at(i);
        const uint32_t num_blocks_per_core = (i < g1_numcores) ? num_blocks_per_core_g1 : num_blocks_per_core_g2;

        // Reader: (src_addr, num_tiles, src_start_tile_id)
        reader_kernel.emplace_runtime_args(core, {src_buffer, num_blocks_per_core * Wt, num_blocks_written * Wt});

        // Writer: (dst_addr, num_pages, core_blocks_written) — kernel derives update_idxt + head
        // offset from the slot_idx/kv_actual_global it reads from the metadata tensors.
        writer_kernel.emplace_runtime_args(core, {dst_buffer, num_blocks_per_core * Wt, num_blocks_written});
```

Regarding in-place aliasing specifically: because the cache is both `tensor_args.cache` (input region)
and the value returned by `create_output_tensors` (output region), it appears twice in
`collect_tensor_buffers`. That is the safe in-place case, explicitly skipped by the resolver rather than
treated as the ambiguous `matmul(X, X)` duplicate:

```90:94:tt_metal/impl/program/program_descriptor_patching.cpp
            const bool is_input = i < num_input_buffers;
            // An output/workload buffer that aliases an input is the safe in-place case — skip it.
            if (!is_input && input_buffers.contains(buf)) {
                continue;
            }
```

So the op never bails to an empty `ResolvedBindings`, and the whole class of in-place aliasing concern
is resolved by the framework here. The metadata tensors deliberately do *not* use `Buffer*` bindings —
their addresses ride in common args 8/9 (`create_descriptor:368-382`), which is why the op has to patch
them by hand.

### 10. `my_sp_coord` / `sp_factor` (derived, not attributes)

**Verdict: VALID — invariant.** `sp_factor` is the mesh extent along `cluster_axis` (hashed) and
`my_sp_coord` is derived from the dispatch coordinate (`create_descriptor:286-288`). Coordinates are
folded into the key by the framework for both the default and custom paths
(`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:989-992`), and the program cache is per-device, so a
program can never be reused at a different coordinate.

## Keys the custom hash adds beyond the default

- `input.padded_shape()` and `cache.padded_shape()` — derivations in the default key, promoted to
  first-class here. This is precisely what makes dropping both `logical_shape`s safe.
- `input.layout()` — a collapse of `page_config` to `ROW_MAJOR`/`TILE`, which is the only distinction
  the page-unit branch cares about. Safe now that `require_standard_tile` pins the other half of
  `page_config` on both paths.
- `tensor_args.slot_idx.has_value()` — pre-fix a lossy projection of the optional tensors rather than an
  addition (see omission 5). Post-fix it is joined by `slot_idx->memory_config()`, so the pair keys the
  program variant *and* the accessor configuration the variant bakes.
- `slot_idx->memory_config()` on the metadata path, `MemoryConfig{}` otherwise
  (`compute_program_hash:360-363`) — added by fab067a. The default-constructed sentinel is safe because
  a real device tensor's memory config can never equal it.
- `valid_global.has_value() || args.valid_global.has_value()` — the clamp variant is a writer compile
  arg, so the engagement bit must be keyed while the value stays a patched runtime arg.

## Framework side effect of having a custom hash

Defining `compute_program_hash` opts this op out of attribute-level collision resolution:

```1012:1014:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
        if constexpr (requires { DeviceOperation::compute_program_hash(attrs, tensor_args); }) {
            return key;  // custom hash -> opt out beyond the op-identity prefix
        } else {
```

`ProgramCacheKey::canonical` degrades to the op type name, so a 64-bit collision between two different
KV-cache configurations resolves to a wrong hit rather than a rebuild. For an op that mutates a shared
KV cache in place, a wrong hit is silent multi-user data corruption rather than a wrong tensor, which
raises the cost of every gap above.

## Non-cache correctness defects

Recorded separately so they are not counted as program-cache bugs. These concern the factory and the
override, not the key.

| Defect | Status | Note |
|---|---|---|
| The override hard-codes `kWriterKernelHandle = 1` (`override_runtime_arguments:600`) on the assumption that kernel handles follow descriptor push order. | **OPEN** — unchanged by fab067a | It holds today because `ProgramImpl` creates kernels in `descriptor.kernels` order (`tt_metal/impl/program/program.cpp:402-466`), but it is an implicit coupling between two functions several hundred lines apart, and nothing fails loudly if the descriptor's push order changes. The override already asserts the *arg count* (`kArg9 < writer_common.size()`) but not the *identity* of the kernel it is patching, so a reordered descriptor would silently patch the reader's common args instead. Deriving the handle from the descriptor, or at minimum asserting the kernel count and name, would close it. `zero_padded_kv_cache` carries the same defect. |
| The miss-path 32x32 factory assumption: `single_page_size`, `Wt`, `input_Ht`, `cache_HtWt` and `writer_tile_height` all derive from `tt::tile_size` and bare `TILE_HEIGHT`/`TILE_WIDTH` (`create_descriptor:275-281`) with no read of the tensor's actual `Tile`. Even on a cache **miss**, a `Tile{16, 32}` call would have compiled a wrong program. | **RESOLVED by fab067a** | `require_standard_tile` (`:112-130`) rejects the call before `create_descriptor` runs. The factory is still 32x32-only — fenced rather than made tile-aware, which is the right order of operations. |
| The 32x32 TILE constraint was undocumented in the public API. | **RESOLVED by fab067a** | Stated in the nanobind docstring (`update_padded_kv_cache_nanobind.cpp:53-54`). |

## Summary

| Omitted vs. default | Used by program? | Patched on hit? | Verdict |
|---|---|---|---|
| `attrs.kv_actual_global` | Yes (writer common arg 9) | Yes (override) | VALID — patched |
| `attrs.slot_idx` | Yes (writer common arg 8) | Yes (override) | VALID — patched |
| `cache.logical_shape`, `input.logical_shape` | No (padded shapes used) | n/a | VALID — relaxation win (→ `match_padded_shape_only` on the Metal 2.0 port) |
| `cache.dtype`, `cache.layout` | Yes (accessor page size, CB) | No | RESOLVED by fab067a — pins moved into `validate_runtime_args`, so they run on hits (was CAVEAT) |
| metadata tensors' `memory_config` / `alignment` | Yes (accessor compile-time args) | No (compile-time) | RESOLVED by fab067a — `slot_idx->memory_config()` hashed, and both metadata tensors pinned to agree with it (was BUG) |
| `page_config` (`Tile`) of cache/input | Yes (`writer_tile_height` compile-time arg, CB page size, tile counts) | No (compile-time) | RESOLVED by fab067a — 32x32 asserted in `validate_runtime_args` (was BUG) |
| `alignment` of cache/input | Only via `aligned_page_size` | No | VALID — determined by hashed {`dtype`, `layout`, `memory_config`, `padded_shape`} (was CAVEAT) |
| `cache.storage`, `input.storage` kind | n/a | n/a | VALID — pinned by the framework at `device_operation.hpp:500-501` (was CAVEAT — pinned only on the miss path; that grade was wrong) |
| `cache` / `input` buffer addresses | Yes | Yes (`resolved_bindings`) | VALID — patched, required |
| `my_sp_coord` / `sp_factor` | Yes (common args 0-1) | n/a (coordinate hashed) | VALID — invariant |

**Two program-cache bugs were found; both are RESOLVED by fab067a. Zero remain.** The two per-request
scalars — the values one would most expect to be mishandled in an in-place KV-cache op — were handled
correctly from the start: omitted from the hash, re-applied on every hit, and re-validated on every hit.
The defects were elsewhere, and both were compile-time-arg defects, which is the one category no
cache-hit path can repair.

The first was the optional metadata tensors, reduced to a single `has_value()` bit while their buffer
type and aligned page size were baked into the writer's compile-time `TensorAccessorArgs`. A caller who
allocated the metadata tensors in L1 after a DRAM-allocated first call got a silent wrong hit and
corrupted the KV cache at an arbitrary offset. fab067a keys `slot_idx->memory_config()`
(`compute_program_hash:360-363`), so that second call now misses, and pins both metadata tensors to
share that config (`:174-181`), which also closes the miss-path variant where one accessor served two
non-interchangeable tensors.

The second was the unguarded 32x32 tile assumption (omission 6). The op accepts `Layout::TILE`, derives
`single_page_size`, `Wt`, `input_Ht`, `cache_HtWt` and `writer_tile_height` from `tt::tile_size` and
bare `TILE_WIDTH`/`TILE_HEIGHT` (`create_descriptor:275-281`), never reads the tensor's actual `Tile`,
and did not hash `page_config`. `writer_tile_height` is a *compile-time* arg
(`create_descriptor:345-346`), which made this structurally worse here than a runtime-arg-only
exposure: a stale runtime arg could at least be patched by extending the override, whereas a stale
compile-time arg can only be fixed by hashing the value or rejecting the input. A `Tile{16, 32}` call
following an otherwise-identical `Tile{32, 32}` call hit the cache, wrote at half the correct sequence
offset, and copied 2048-byte pages into 1024-byte pages. fab067a took the rejecting route:
`require_standard_tile` in `validate_runtime_args` (`:112-130`). The same defect was present in
`zero_padded_kv_cache` and `rotary_embedding_indexed` and was fixed the same way in both; pre-fix it was
only in `rotary_embedding_indexed` that the Metal 2.0 dispatch path turned it into a loud throw rather
than silent corruption, and the fix removes the dependence on that accident.

**Both fixes turn on validator placement, and that is the transferable lesson.** Each reproduction above
is a cache **hit**: call 1 misses, compiles, caches, and passes every check the miss validator makes;
call 2 computes the same key, hits, and corrupts the cache. Because this op defines
`validate_on_program_cache_hit`, the dispatcher *substitutes* rather than supplements
(`ttnn/api/ttnn/device_operation.hpp:262-266`), so the miss validator does not execute on call 2 at all.
A guard added to `validate_on_program_cache_miss` — the intuitive place, and where the pre-existing
`cache`-vs-`input` checks already lived — would have been present in the source and completely inert on
the only call that matters. Every guard fab067a added went into the shared `validate_runtime_args`,
which both validators delegate to (`:186`, `:193`). That is the only placement that works.

A third, lower-severity finding is structural and remains: because the op defines
`validate_on_program_cache_hit`, that validator *replaces* the miss validator on hits rather than
supplementing it, and this op's hit validator delegates to `validate_runtime_args` and does nothing
else. Everything the miss validator checks before its own delegation at `:186` is therefore absent on
the hit path — post-fix that is the `input` storage pin, the `buffer()` and layout gates, and the rank,
shape-equality, seq-alignment and `num_layers` checks. A narrow hit validator is a hazard rather than a
safeguard here: by existing, it disables all of them. What fab067a changed is not the hazard but the
inventory: nothing load-bearing is left above the delegation.

Filtered for reachability, the practical loss is now nil: those checks constrain values that are in the
cache key, and a miss-only pin on a hashed value cannot be evaded, because any call carrying a new value
of it misses and meets the pin there. The storage rows are pinned by `launch()` on every dispatch
(`ttnn/api/ttnn/device_operation.hpp:500-501`), and the two rows that failed silently — `:142` and
`:149` — now live in `validate_runtime_args`.

## Recommendations

**Status: recommendations 1, 2 and 4 landed in fab067a; 3's storage half stands as written; 5 and 6 are
outstanding.** Each item carries its own status line. The section is left in place because its reasoning
about placement and pricing is what transfers to the Metal 2.0 port.

**Every guard below names the function it must go into, and for this op that function is almost always
`validate_runtime_args`.** Because the op defines `validate_on_program_cache_hit`, the miss validator is
skipped entirely on a hit; a guard placed in `validate_on_program_cache_miss` would not run on the
offending second call, which is the only call a cache bug reaches. `validate_runtime_args` is the right
home for all of them because both validators delegate to it (`:186`, `:193`), so one placement covers
both paths.

**And every guard below is priced.** The cache-hit path is the fast path — it is what the program cache
exists to make cheap — so a `TT_FATAL` added to `validate_runtime_args` is paid on every dispatch for
the life of the process. That is the cost side of every recommendation here, and it is why this document
recommends moving *specific* pins rather than the whole miss-time block, and why one of the regraded
rows is deliberately left as a documented caveat rather than fixed.

There are two distinct ways to close a miss-only pin in this op, and they are not interchangeable:

- **Targeted (recommended):** move the specific `TT_FATAL`s into `validate_runtime_args`. Adds only
  those comparisons per dispatch. This is what recommendations 3 and 4 mean.
- **Wholesale (alternative):** delete `validate_on_program_cache_hit` entirely, putting the op on the
  dispatcher's substitution branch so the full miss validator runs on every hit. Simplest and safest,
  and it can never be silently regressed by someone adding a check to the wrong function — but it puts
  all of `:137-182` on the hot path: two `storage_type()` calls, four layout/dtype gates, two rank
  queries, five shape comparisons and two arithmetic divisibility checks, on every single decode step.
  For this op that is poor value, because the reachability table in `## Cache-hit patch mechanism`
  shows only four of those lines can be reached on a hit at all. Prefer the targeted move.

1. Hash the metadata tensors' specs. The minimal fix mirrors what `rotary_embedding_indexed` already
   does: add `tensor_args.slot_idx->memory_config()` and `tensor_args.kv_actual_global->memory_config()`
   (with neutral defaults on the scalar path) to `compute_program_hash`. This is a family-wide gap —
   apply the same change to `zero_padded_kv_cache`, whose metadata validator is even thinner.
   **Status: DONE in fab067a**, with one refinement worth noting: only `slot_idx`'s config is keyed
   (`:360-363`), and `kv_actual_global` is instead *pinned equal to it* by the new `validate_meta` check
   (`:174-181`). That is equivalent and tighter — the accessor is built from `slot_idx` alone, so a
   second keyed config would have been redundant with an invariant the op needs to assert anyway. The
   same change landed in `zero_padded_kv_cache`.
2. Independently, add `TT_FATAL(meta.buffer()->buffer_type() == BufferType::DRAM, ...)` to the
   `validate_meta` lambda, and assert that the `slot_idx` and `kv_actual_global` tensors share a buffer
   type and aligned page size — the writer reuses one accessor for both
   (`create_descriptor:348-351`) and nothing currently enforces that they are interchangeable, which is
   a correctness gap even on a cache miss.
   **Target function:** `validate_meta` already lives inside `validate_runtime_args` (`:85-98`), so it
   is on the hit path and is the correct home as-is. This is worth stating because the equivalent guard
   must *not* go into `validate_on_program_cache_miss`: the defect it closes is a wrong *hit*, and a
   miss-only guard would pass the first call and never see the second.
   **Per-dispatch cost:** two extra checks per metadata tensor, and only on the metadata path — the
   lambda is already called there, so this adds to an existing cost rather than creating one. Worth it:
   unlike the storage rows below, the defect these close is the silent wrong hit that is this
   document's headline BUG.
   **Status: DONE in fab067a**, in the shape the second sentence asks for — `validate_meta` now asserts
   that every metadata tensor's `memory_config` equals `slot_idx`'s (`:174-181`), which pins buffer type
   and aligned page size together rather than pinning DRAM specifically. That is the better form: it
   makes the two tensors interchangeable (which is the property the shared accessor actually needs)
   without hard-coding a buffer type the op has no reason to require.
3. **Move exactly two `TT_FATAL`s into `validate_runtime_args`:** `cache.dtype() == input.dtype()`
   (`:142`) and `cache.layout() == input.layout()` (`:149`). These are the pins behind omission 4, and
   they are the only dropped checks in this op whose absence produces *silent* misbehaviour: both
   constrain a `cache` property that is not in the cache key against an `input` property that is, so a
   mismatched second call hits and executes a program built for the wrong cache page size. Two scalar
   enum comparisons per dispatch is the right price for closing a silent-corruption path, and it is
   cheaper and more targeted than adding the two values to the hash. This upgrades omission 4 to
   `VALID — pinned by validation`.
   **Status: DONE in fab067a** — exactly these two, exactly there (`:132-144`), with pointer comments
   left at the sites they came from (`:256`, `:279`).

   **Do not also move the two `storage_type() == StorageType::DEVICE` checks (`:140-141`).** The
   original reasoning was that the failure they prevent is not silent, so the move buys only a better
   error message and is not worth two `storage_type()` queries on every decode step of every layer.
   **Status: STANDS, and the reasoning is now stronger than when written.** The pins are not merely
   low-value, they are provably zero-value: `launch()` asserts `is_device_tensor` and `is_allocated` on
   every tensor argument on every dispatch (`ttnn/api/ttnn/device_operation.hpp:500-501`), so the
   failure cannot occur at all — see `#### Framework correction` under omission 8, which also regrades
   that row from CAVEAT to `VALID — pinned by the framework`. fab067a did move the `cache` half
   (`:75-80`), but on unrelated grounds: `cache.device()` is dereferenced on the next line, and a
   message beats a null dereference.
4. Reject a non-32x32 `Tile` on the TILE path, closing omission 6. Assert
   `cache.tensor_spec().tile().get_height() == TILE_HEIGHT` and the same for `get_width()`, on `cache`
   and `input`, in the same shape as the `interleaved_to_sharded` guard quoted in omission 6.
   **Target function:** `validate_runtime_args`, not `validate_on_program_cache_miss`. The reproduction
   in omission 6 is a *hit*, so a guard in the miss validator would let the first `Tile{32,32}` call
   through and then not run at all on the `Tile{16,32}` call that corrupts the cache. Placing it in
   `validate_runtime_args` covers the miss path too, via the delegation at `:186`.
   **Per-dispatch cost:** two `uint32_t` comparisons against constants, on two tensors. This is the
   clearest case in the document of a check worth its price — it closes a BUG with silent
   KV-cache corruption and an out-of-bounds DRAM write as its symptom.
   This is minimal and makes omitting `page_config` correct by construction. The alternative — making
   the factory tile-aware via `tile.get_tile_size(data_format)` and `tile().get_tile_shape()`, and
   setting `writer_tile_height = tile.get_height()` — requires adding `page_config` to
   `compute_program_hash` in the same change, because the program would then provably vary with `Tile`.
   This is a family-wide gap: apply the same guard to `zero_padded_kv_cache` and
   `rotary_embedding_indexed`.
   **Status: DONE in fab067a** — landed as `require_standard_tile` (`:112-130`) in
   `validate_runtime_args`, on both `cache` and `input`, with an early return for ROW_MAJOR so the guard
   only constrains the branch that reads a tile. The same guard landed in both siblings.
5. Run this op's tests under `-DTT_DESCRIPTOR_PATCHING_PARITY_CHECK`. Note that the parity oracle only
   covers runtime args and CB addresses
   (`tt_metal/api/tt-metalium/experimental/program_descriptor_patching.hpp:176-186`), so it would *not*
   have caught the compile-time-arg defect in omission 5 — that one needed the hash fix or the
   validation guard. It will, however, catch any future regression in the common-arg patch.
   **Status: OUTSTANDING.** fab067a adds no tests, by design: its own note says the guarded situations
   have no real use case, so most of the change is `TT_FATAL`s on inputs nothing exercises. That
   reasoning does not extend to the parity oracle, which guards the patch path that *is* exercised on
   every dispatch.
6. The override hard-codes `kWriterKernelHandle = 1` on the assumption that kernel handles follow
   descriptor push order (`override_runtime_arguments:600`). That holds today because
   `ProgramImpl` creates kernels in `descriptor.kernels` order
   (`tt_metal/impl/program/program.cpp:402-466`), but it is an implicit coupling between two functions
   several hundred lines apart. Deriving the handle from the descriptor, or at least asserting the
   kernel count, would make it robust.
   **Status: OUTSTANDING.** Not a cache defect — recorded in
   `## Non-cache correctness defects` so it is not counted as one — and out of scope for fab067a, which
   confined itself to keys and validators. `zero_padded_kv_cache` carries the same coupling.
