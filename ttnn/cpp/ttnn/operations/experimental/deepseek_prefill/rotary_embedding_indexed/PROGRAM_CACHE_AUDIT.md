# Program Cache Audit — `experimental/deepseek_prefill/rotary_embedding_indexed`

Audit of
`ttnn::operations::experimental::deepseek_prefill::rotary_embedding_indexed::RotaryEmbeddingIndexedDeviceOperation::compute_program_hash`
against the framework default ("hash everything") key.

| | |
|---|---|
| Device operation | `RotaryEmbeddingIndexedDeviceOperation` (`device/rotary_embedding_indexed_device_operation.hpp:22`) |
| Custom hash | `device/rotary_embedding_indexed_device_operation.cpp:208-251` |
| `operation_attributes_t` | `cluster_axis`, `kv_actual_global`, `output_mem_config`, `compute_kernel_config` |
| `tensor_args_t` | `input`, `cos`, `sin`, `trans_mat`, `std::optional<Tensor> metadata` |
| Program factories | one: `MeshWorkloadFactory`, a hand-rolled Metal 2.0 `ProgramSpec` factory that builds a per-coordinate program via `create_at` + `MakeProgramFromSpec` |
| `override_runtime_arguments` | **Yes** (`device/rotary_embedding_indexed_device_operation.cpp:612-641`) |
| `get_dynamic_runtime_args` | **No** |
| `validate_on_program_cache_hit` | **Yes** (`device/rotary_embedding_indexed_device_operation.cpp:187-192`) — so it *replaces* the miss validator on hits rather than supplementing it |
| Validator actually run on a hit | `validate_runtime_args` only (`:60-126`); everything in `validate_on_program_cache_miss` before its delegation at `:184` is skipped. The framework's `TensorSpec` equality check partly compensates |
| Cache-hit patch mechanism | **Op-owned override**, applied through Metal 2.0 `UpdateProgramRunArgs`. `resolve_bindings` and the descriptor buffer-binding fast path are not involved at all |
| In-place | No — the op allocates a fresh output tensor |

## Cache-hit patch mechanism

`select_program_factory` always returns `MeshWorkloadFactory`
(`device/rotary_embedding_indexed_device_operation.cpp:130-133`). That factory defines
`override_runtime_arguments` and not `apply_descriptor`, so the framework's cache-hit dispatcher hands
control straight to the op:

```279:285:ttnn/api/ttnn/device_operation.hpp
        if constexpr (requires { &WorkloadFactory::apply_descriptor; }) {
            WorkloadFactory::apply_descriptor(
                cached_mesh_workload, operation_attributes, tensor_args, tensor_return_value);
        } else {
            WorkloadFactory::override_runtime_arguments(
                cached_mesh_workload, operation_attributes, tensor_args, tensor_return_value);
        }
```

Unlike its two `kv_cache` siblings in this family, this op does not wrap the descriptor adapter — it
implements `create_mesh_workload` itself (`:596-610`), calling `MakeProgramFromSpec` +
`SetProgramRunArgs` per mesh coordinate (`:591-592`). There is no `ProgramDescriptor`, therefore no
`collect_tensor_buffers`, no `resolve_bindings`, and no descriptor fast path. **The entire
address-inference and aliasing-bail machinery in `program_descriptor_patching.hpp` is bypassed for this
op**, which removes that whole class of concern: there is no possibility of the resolver mapping two
logically distinct operands onto one `Buffer*` slot, and no possibility of it bailing to an empty
`ResolvedBindings` and silently skipping address patching. Tensor addresses are re-bound by name
through the Metal 2.0 tensor-parameter table instead. (The op is also not in-place, so the in-place
alias case does not arise in the first place.)

The override:

```612:641:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/rotary_embedding_indexed/device/rotary_embedding_indexed_device_operation.cpp
void RotaryEmbeddingIndexedDeviceOperation::MeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& args,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    ...
    ProgramRunArgs run_args;
    run_args.tensor_args = {
        {INPUT_PARAM, TensorArgument{tensor_args.input.mesh_tensor()}},
        {COS_PARAM, TensorArgument{tensor_args.cos.mesh_tensor()}},
        {SIN_PARAM, TensorArgument{tensor_args.sin.mesh_tensor()}},
        {TRANS_MAT_PARAM, TensorArgument{tensor_args.trans_mat.mesh_tensor()}},
        {OUTPUT_PARAM, TensorArgument{output.mesh_tensor()}}};
    if (tensor_args.metadata.has_value()) {
        run_args.tensor_args.emplace(METADATA_PARAM, TensorArgument{tensor_args.metadata->mesh_tensor()});
    } else {
        KernelRunArgs reader_run{.kernel = READER};
        reader_run.common_runtime_arg_values = {{"kv_actual_global", args.kv_actual_global}};
        run_args.kernel_run_args = {reader_run};
    }

    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        UpdateProgramRunArgs(program, run_args);
    }
}
```

`UpdateProgramRunArgs` is documented as a *partial* update — anything omitted keeps its prior value
(`tt_metal/api/tt-metalium/experimental/metal2_host_api/program_run_args.hpp:62-64,78-79`) — so leaving
the per-core `batch_start`/`batch_end`/`seq_t_start`/`seq_t_end` args untouched is deliberate and
correct, not an oversight, provided they are functions of the hashed set (they are; see omission 6).

**Obligation on the hash.** A hit refreshes: the six tensor bindings, and on the scalar path the
reader's `kv_actual_global` common arg. Frozen at the first miss: every kernel's compile-time args
(`create_at:441-451,467-468,511`), the `RELOAD_IMPL` / `HAS_METADATA` defines (`:400-404`), all nine or
ten dataflow-buffer sizes (`:329-382`), the compute hardware config (`:397-398`), the core range
(`:304`), and every per-core runtime arg (`:545-569`).

There is one additional, unusually strong guarantee here that shapes most of the verdicts below.
`UpdateProgramRunArgs` validates every supplied `TensorArgument` against the `TensorParameter` spec
baked at creation:

```116:127:tt_metal/impl/metal2_host_api/program_run_args.cpp
    for (const auto& [param_name, tensor_arg] : tensor_args) {
        tensor_parameters_with_params.insert(param_name.get());
        const TensorSpec* expected_spec = program_impl.get_tensor_parameter_layout(param_name.get());
        TT_FATAL(expected_spec != nullptr, "TensorArgument references unknown TensorParameter '{}'.", param_name);
        const TensorSpec& runtime_spec = mesh_tensor_of(tensor_arg).tensor_spec();
        const TensorSpecRelaxations relaxation = program_impl.get_tensor_parameter_relaxations(param_name.get());
        // Authoritative accept/reject via the same predicate the program-cache hash keys on, so
        // run-time validation and cache-equivalence cannot disagree. On rejection,
        // report_tensor_arg_mismatch emits a specific diagnostic (and always throws).
        if (!tensorspecs_match_with_relaxation(runtime_spec, *expected_spec, relaxation)) {
            report_tensor_arg_mismatch(param_name, runtime_spec, *expected_spec, relaxation);
        }
    }
```

The op declares its tensor parameters with `.unique_id` and `.spec` only (`create_at:385-395`), leaving
`TensorParameter::relaxations` default-constructed, and "a default-constructed `TensorSpecRelaxations`
requires an exact match"
(`tt_metal/api/tt-metalium/experimental/metal2_host_api/tensor_spec_relaxations.hpp:28`). So any drift
in any operand's `{logical_shape, dtype, page_config, memory_config, alignment}` on a cache hit is
**rejected with a diagnostic**, never silently mis-executed. Every spec-component omission below
therefore fails safe. The residual exposure is availability, not correctness: an omission that lets two
legitimately different specs collide produces a hard throw where a recompile was wanted.

### Which validator runs on a cache hit

Separately from the framework spec check, the dispatcher runs exactly one of the op's own validators on
a hit, and which one is chosen has the opposite effect from the intuitive reading:

```262:266:ttnn/api/ttnn/device_operation.hpp
    if constexpr (HasValidateOnProgramCacheHit<mesh_device_operation_t>) {
        mesh_device_operation_t::validate_on_program_cache_hit(operation_attributes, tensor_args);
    } else {
        mesh_device_operation_t::validate_on_program_cache_miss(operation_attributes, tensor_args);
    }
```

An op that defines no hit validator gets the miss validator substituted on hits, so all of its pins
hold. **This op defines one** (`device/rotary_embedding_indexed_device_operation.hpp:90`), so the miss
validator does not run on a hit at all:

```187:192:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/rotary_embedding_indexed/device/rotary_embedding_indexed_device_operation.cpp
void RotaryEmbeddingIndexedDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // kv_actual_global is not hashed and can differ from the compiled program's call; re-validate
    // every hit. Structural constraints are hashed and so guaranteed unchanged here.
    validate_runtime_args(args, tensor_args);
}
```

The miss validator also ends by delegating to `validate_runtime_args` (`:184`), so the two paths differ
by exactly the checks the miss validator performs *before* that delegation — lines 142 through 182. The
hit path therefore loses all of the following:

- `storage_type() == DEVICE` on `input`, `cos`, `sin` and `trans_mat` (`:142-145`).
- The `buffer() != nullptr` checks and the `device() == input.device()` checks on all four operands
  (`:149-155`).
- `Layout::TILE` on all four operands (`:157-160`).
- The rank-4 checks and the `trans_mat` single-tile check (`:166-176`).
- `cos.dtype() == sin.dtype()`, `cos_shape == sin_shape`, and the input-vs-cos head-dim equality
  (`:177-179`).
- The input seq tile-alignment check (`:182`).

What *does* run on both paths is `validate_runtime_args` (`:60-126`): the `cluster_axis` check, the
2D-mesh check, the `chunk_local_t > 0` check, the entire metadata-tensor block (`:82-94`, including the
`metadata.dtype() == UINT32` pin that omission 2 relies on), and the scalar-path `kv_actual_global`
tile-alignment and shard-bound checks.

The comment on the hit validator asserts that "structural constraints are hashed and so guaranteed
unchanged here." That is true of the shapes and dtypes it has in mind, but it is not true of the four
`Layout::TILE` pins or the four `storage_type()` pins, which are neither hashed nor re-run — hence the
regrade of omissions 4 and 7 below from `VALID — pinned by validation` to
`CAVEAT — pinned only on the miss path`. Omission 2 is unaffected, because its pin genuinely does live
in `validate_runtime_args`.

This op is better off than its two siblings in the same situation, for the same reason it is better off
on the tile: the framework's exact `TensorSpec` comparison independently rejects a `layout` divergence
on a hit, since `layout` is a projection of `page_config`. That is a backstop, not a pin, and it does
not cover the storage-kind or cross-tensor-equality checks — so the caveats are real, just low-severity.

**Which of the dropped checks are actually reachable.** The list above is the mechanical diff, but most
of those checks constrain values that are themselves in the cache key, and a miss-only pin on a *hashed*
value cannot be evaded: any call carrying a new value of that parameter misses, and the miss validator
runs and rejects it there. Filtering the list against `compute_program_hash:230-250`:

| Dropped check | Constrains | In the key? | Reachable on a hit? |
|---|---|---|---|
| `storage_type() == DEVICE` ×4 (`:142-145`) | storage variant kind | No | **Yes** |
| `buffer() != nullptr` ×4, `device() == input.device()` ×3 (`:149-155`) | allocation and device identity | No | **Yes** |
| `Layout::TILE` ×4 (`:157-160`) | `cos`/`sin`/`trans_mat` layout | No (only `input.layout()` is) | **Yes**, but the framework spec check rejects it |
| Rank-4 on `input` and `cos` (`:166-167`) | both padded shapes | Yes, both | No |
| `trans_mat` single-tile check (`:170-176`) | `trans_mat.padded_shape()` | Yes | No |
| `cos.dtype() == sin.dtype()` (`:177`) | both dtypes | Yes, both | No |
| `cos_shape == sin_shape`, input-vs-cos head dim (`:178-179`) | both padded shapes | Yes, both | No |
| Input seq tile-alignment (`:182`) | `input.padded_shape()` | Yes | No |

So the reachable losses are the allocation-and-device block (`:142-155`) and the three layout pins
(`:157-160`) — and none of them fails silently. The allocation block fails as a fault when
`UpdateProgramRunArgs` tries to resolve a buffer; the layout pins are caught by the framework spec
comparison and throw. Everything in the lower half of the block is unreachable because the shapes and
dtypes it constrains are all in the key. This is what drives the recommendations at the end of this
document: there is no silent-corruption path here to buy back, so no new per-dispatch check is
justified on this account.

## How the position index actually arrives

The brief asks which form the "indexed" position takes, because it changes the verdict. Reading the
code: **there is no per-token index tensor.** Despite the name, the op does not take position indices
per token. It takes a single scalar — `kv_actual_global`, the prior valid global KV length in tokens —
and derives the cos/sin shard offset arithmetically from it plus the device's own coordinate along the
sequence-parallel axis:

```77:89:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/rotary_embedding_indexed/device/kernels/dataflow/reader_rotary_embedding_indexed_interleaved_start_id.cpp
    const uint32_t kv_actual_global_t = kv_actual_global / tile_height;
    // Derive this chip's tile-row offset into its (block-cyclic) cos/sin shard from the global
    // valid KV length. Ht == chunk_local_t (per-device new chunk in tiles); chunk_global == sp*Ht.
    // Identical math to the per-chip kv-cache writer's update_idxt -- see writer_update_padded_kv_cache.
    const uint32_t chunk_global_t = sp_factor * Ht;
    const uint32_t boundary_slab_idx = chunk_global_t == 0 ? 0 : kv_actual_global_t / chunk_global_t;
    const uint32_t boundary_chip = Ht == 0 ? 0 : (kv_actual_global_t / Ht) % sp_factor;
    const uint32_t boundary_offset_t = Ht == 0 ? 0 : kv_actual_global_t % Ht;
    // From the current slab base, chips before the boundary advance a full slab, the boundary chip
    // advances by its pad offset, and chips after it stay at the base.
    const uint32_t update_idxt =
        boundary_slab_idx * Ht +
        (my_sp_coord < boundary_chip ? Ht : (my_sp_coord == boundary_chip ? boundary_offset_t : 0));
```

That scalar reaches the kernel one of two ways, selected by whether the caller supplies the optional
`metadata` tensor:

- **Scalar path** (`metadata` empty): a host `uint32_t` in `operation_attributes_t`, carried as the
  reader's `kv_actual_global` common runtime argument. This is a host scalar, so it must be either
  hashed or explicitly patched — it is patched (omission 1).
- **Metadata path** (`metadata` set): a 1-element uint32 DRAM tensor whose element [0] the reader
  NoC-reads on-device (`reader_...cpp:51-71`). Here the value is data, correctly not hashed at all;
  only the tensor's *spec* needs hashing and its *address* needs rebinding.

So both of the shapes the brief contemplates are present in one op, and the audit has to treat them as
two distinct program variants — which the hash does, by keying on `metadata.has_value()`.

## Baseline: what the default hash would cover

`hash_objects_with_default_seed(type_hash<RotaryEmbeddingIndexedDeviceOperation>, attrs, tensor_args)`
walks reflection, giving:

| Source | Fields |
|---|---|
| `operation_attributes` | `cluster_axis`, `kv_actual_global`, `output_mem_config`, `compute_kernel_config` |
| `input` | storage variant kind, `logical_shape`, `dtype`, `page_config`, `memory_config`, `alignment` |
| `cos` | the same six |
| `sin` | the same six |
| `trans_mat` | the same six |
| `metadata` (optional) | engaged/disengaged, plus the same six when engaged |

`padded_shape` is not directly in the default key — it is a derivation of `logical_shape`,
`page_config` and `alignment`. Mesh coordinates are appended by the framework on both the default and
custom paths (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:989-992`), so the per-device
`my_sp_coord` is never an omission.

## What the custom hash covers

```230:250:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/rotary_embedding_indexed/device/rotary_embedding_indexed_device_operation.cpp
    return tt::tt_metal::operation::hash_operation<RotaryEmbeddingIndexedDeviceOperation>(
        tensor_args.metadata.has_value(),
        metadata_mem_config,
        metadata_padded_shape,
        args.cluster_axis,
        args.compute_kernel_config,
        args.output_mem_config,
        input.dtype(),
        input.memory_config(),
        input.logical_shape(),
        input.padded_shape(),
        input.layout(),
        cos.dtype(),
        cos.memory_config(),
        cos.padded_shape(),
        sin.dtype(),
        sin.memory_config(),
        sin.padded_shape(),
        trans_mat.dtype(),
        trans_mat.memory_config(),
        trans_mat.padded_shape());
```

Three of the four attributes are kept; only `kv_actual_global` is dropped. All five tensors
participate, decomposed selectively — `input` most fully (five components), `cos`/`sin`/`trans_mat`
with three each, `metadata` with two plus its engagement bit.

Note that the explanatory comment above this body claims the hash covers "the full input, cos, sin and
trans_mat specs" (`:210-214`). It does not; it covers projections of them. The claim matters because it
is the stated safety argument for the whole function, and omissions 3-5 below are exactly the gap
between the claim and the code.

## Omitted parameters

### 1. `operation_attributes.kv_actual_global`

**Verdict: VALID — patched** on the scalar path; **VALID — unused** on the metadata path.

This is the only attribute the hash drops, and the drop is the entire point: it advances on every
prefill chunk, so hashing it would force a recompile per chunk. On the scalar path it is declared as a
common runtime arg in the reader's schema and set at build time:

```429:433:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/rotary_embedding_indexed/device/rotary_embedding_indexed_device_operation.cpp
    KernelSpec::RuntimeArgSchema reader_schema{
        .runtime_arg_names = {"batch_start", "batch_end", "seq_t_start", "seq_t_end"}};
    if (!has_metadata) {
        reader_schema.common_runtime_arg_names = {"kv_actual_global"};
    }
```

```542:544:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/rotary_embedding_indexed/device/rotary_embedding_indexed_device_operation.cpp
    if (!has_metadata) {
        reader_run.common_runtime_arg_values = {{"kv_actual_global", args.kv_actual_global}};
    }
```

and re-applied on every hit at `override_runtime_arguments:633-635`. Nothing host-side is derived from
it — no compile-time arg, no CB size, no core assignment. The reader consumes it directly and does all
the offset arithmetic on-device (quoted above). That is what makes the omission safe rather than merely
convenient: `update_idxt` never leaks into a frozen slot.

On the metadata path `kv_actual_global` is unused (the caller passes 0) and the reader instead reads
element [0] of the metadata tensor. Because `metadata.has_value()` is hashed, the two variants can never
share a cache entry, so the reader's `#ifdef HAS_METADATA` branch always matches the program that was
compiled.

Since the value is not hashed, the op re-runs its bounds checks on every hit rather than only on a
miss:

```187:192:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/rotary_embedding_indexed/device/rotary_embedding_indexed_device_operation.cpp
void RotaryEmbeddingIndexedDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // kv_actual_global is not hashed and can differ from the compiled program's call; re-validate
    // every hit. Structural constraints are hashed and so guaranteed unchanged here.
    validate_runtime_args(args, tensor_args);
}
```

and `validate_runtime_args` reproduces the kernel's per-chip `update_idxt` derivation exactly to bound
the largest shard row any chip will touch:

```105:125:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/rotary_embedding_indexed/device/rotary_embedding_indexed_device_operation.cpp
    // Bound the largest update_idxt any chip reads from by the per-device cos/sin shard height.
    // Mirror the reader kernel's per-chip update_idxt exactly: each chip reads chunk_local_t tiles
    // starting at update_idxt, where chips before the boundary chip jump to the next slab
    // ((boundary_slab+1)*chunk_local_t), the boundary chip starts at boundary_slab*chunk_local_t +
    // offset, and chips after it stay on this slab. The max is the pre-boundary value WHEN a
    // pre-boundary chip exists (boundary_chip > 0); when kv_actual_global is exactly slab-aligned
    // (boundary_chip == 0) no chip jumps ahead, so a flat (+1 slab) bound would be off by a slab.
    const uint32_t sp_factor = (args.cluster_axis == 0) ? mesh_view.num_rows() : mesh_view.num_cols();
    const uint32_t kv_actual_global_t = args.kv_actual_global / TILE_HEIGHT;
    const uint32_t cos_shard_Ht = cos.padded_shape()[-2] / TILE_HEIGHT;
    const uint32_t chunk_global_t = sp_factor * chunk_local_t;
    const uint32_t boundary_slab_t = (kv_actual_global_t / chunk_global_t) * chunk_local_t;
    const uint32_t boundary_chip = (kv_actual_global_t / chunk_local_t) % sp_factor;
    const uint32_t boundary_offset_t = kv_actual_global_t % chunk_local_t;
    const uint32_t max_update_idxt =
        (boundary_chip > 0) ? boundary_slab_t + chunk_local_t : boundary_slab_t + boundary_offset_t;
    TT_FATAL(
        max_update_idxt + chunk_local_t <= cos_shard_Ht,
        "kv_actual_global ({} tok) + chunk would index past the per-device cos/sin shard ({} tiles)",
        args.kv_actual_global,
        cos_shard_Ht);
```

This is the pattern the whole audit is looking for, done correctly: relax the hash, then pay for the
relaxation with a per-hit validator that mirrors the kernel's arithmetic.

### 2. `metadata.dtype()`

**Verdict: VALID — pinned by validation.** The hash keeps `metadata`'s `memory_config` and
`padded_shape` but drops its dtype, and the omission is explicitly compensated:

```79:94:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/rotary_embedding_indexed/device/rotary_embedding_indexed_device_operation.cpp
        // on-device as uint32, so validate the tensor itself here (runs on both cache miss and hit,
        // since the metadata tensor can differ per call). dtype is NOT part of the program hash, so
        // without this guard a uint32-then-bf16 sequence would silently reuse the cached program.
        const auto& metadata = tensor_args.metadata.value();
        TT_FATAL(metadata.storage_type() == StorageType::DEVICE, "metadata must be on device");
        TT_FATAL(metadata.buffer() != nullptr, "metadata must be allocated in a buffer on device");
        TT_FATAL(metadata.device() == input.device(), "metadata must be on the same device as input");
        TT_FATAL(
            metadata.dtype() == DataType::UINT32,
            "metadata must be uint32 (holds kv_actual_global, read on-device as uint32), got {}",
            metadata.dtype());
        TT_FATAL(
            metadata.logical_shape().volume() == 1,
            "metadata must be a single-element tensor (kv_actual_global at element [0]), got {} elements",
            metadata.logical_shape().volume());
        return;
```

`validate_runtime_args` runs on both the miss and hit paths (`:184` and `:191`), so the constraint is
enforced where it matters. This is what separates this verdict from omissions 4 and 7, whose pins sit in
the part of the miss validator that the hit validator replaces: this pin is inside the shared function,
so it survives on the hit path and `VALID — pinned by validation` is legitimate. The op's own comment at
`:79-81` states the intent explicitly ("runs on both cache miss and hit"), and the placement matches the
intent. Pinned to one value, the dtype carries no information. Belt and braces: the
metadata tensor is also a `TensorParameter` (`create_at:392-395`), so a dtype change would additionally
be rejected by the `UpdateProgramRunArgs` spec check.

Worth calling out because it is the family contrast: this op hashes `metadata->memory_config()` and
`metadata->padded_shape()` (`compute_program_hash:226-233`), so its metadata tensor's buffer type and
page geometry are part of the cache key. Its two siblings in this family, `update_padded_kv_cache` and
`zero_padded_kv_cache`, hash only the `has_value()` bit while compiling their metadata tensor's
`TensorAccessorArgs` — buffer type and aligned page size included — into kernel compile-time args. This
op is the correct model for that pattern.

### 3. `cos.logical_shape()`, `sin.logical_shape()`, `trans_mat.logical_shape()` — replaced by `padded_shape()`

**Verdict: CAVEAT.** Correct as a relaxation of what the *program* depends on, but the framework's
exact-spec check turns any exercised difference into a hard failure rather than the recompile the
relaxation implies.

The factory reads padded shapes only:

```279:288:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/rotary_embedding_indexed/device/rotary_embedding_indexed_device_operation.cpp
    const uint32_t batch = input.padded_shape()[0];
    const uint32_t n_heads = input.padded_shape()[1];
    const uint32_t seq_len_t = input.padded_shape()[2] / TILE_HEIGHT;
    const uint32_t head_dim_t = input.padded_shape()[3] / TILE_WIDTH;
    const uint32_t cos_seq_len_t = cos.padded_shape()[2] / TILE_HEIGHT;
    const uint32_t sin_seq_len_t = sin.padded_shape()[2] / TILE_HEIGHT;
    // cos/sin are the (much taller) per-device shards, so rotary coverage is bounded by the input.
    const uint32_t rotary_seq_len_t = seq_len_t;
    // Flag for whether or not sin/cos vary per head. If false, they will be broadcasted across heads.
    const bool freq_per_head = cos.padded_shape()[1] == n_heads;
```

These feed the `cos_Ht` / `sin_Ht` / `freq_per_head` compile-time args (`:441-451`), which are frozen on
a hit — so hashing the padded shapes is both necessary and sufficient for the program itself. Nothing
reads a logical shape of `cos`, `sin` or `trans_mat`.

The catch is that `cos.tensor_spec()` is what gets baked as the `COS_PARAM` `TensorParameter`
(`create_at:387`), and the spec check on every hit is *exact*. Two calls whose `cos` tensors share a
padded shape, dtype and memory config but differ in logical shape produce the same hash, hit the cache,
and then throw from `report_tensor_arg_mismatch`. That is fail-safe — no corruption, and the diagnostic
names the binding — but the outcome is a crash on a call that should simply have compiled a second
program.

What would break it: a caller who trims the logical extent of a cos/sin cache without changing its
padded extent. Nothing in DeepSeek prefill does that today, since the cos/sin shards are allocated once
per model at a fixed shape. The guard that closes it is to hash `cos.logical_shape()`,
`sin.logical_shape()` and `trans_mat.logical_shape()` alongside the padded shapes — cheap, since these
are per-model constants that will never actually diverge, and it converts a potential throw into a
correct rebuild. Alternatively, declaring `TensorSpecRelaxations::match_padded_shape_only` on those
three tensor parameters would make the relaxation explicit at the framework level; that flag exists for
exactly this and loosens validation only along `logical_shape`
(`tt_metal/api/tt-metalium/experimental/metal2_host_api/tensor_spec_relaxations.hpp:42-49`).

### 4. `cos.layout()`, `sin.layout()`, `trans_mat.layout()`

**Verdict: CAVEAT — pinned only on the miss path.** The pin exists but does not re-run on a hit; the
framework spec check is what actually holds the line there.

`input.layout()` is hashed but the other three operands' layouts are not. They are constrained on the
miss path:

```157:160:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/rotary_embedding_indexed/device/rotary_embedding_indexed_device_operation.cpp
    TT_FATAL(input.layout() == Layout::TILE, "input must be TILE layout");
    TT_FATAL(cos.layout() == Layout::TILE, "cos must be TILE layout");
    TT_FATAL(sin.layout() == Layout::TILE, "sin must be TILE layout");
    TT_FATAL(trans_mat.layout() == Layout::TILE, "trans_mat must be TILE layout");
```

These live in `validate_on_program_cache_miss` and are *not* repeated in `validate_runtime_args`. Since
this op defines a hit validator, the miss validator is replaced rather than supplemented on a hit (see
`### Which validator runs on a cache hit`), so these four `TT_FATAL`s run on the first call and never
again. Under the audit rule that a pin living only in the miss validator is at most a caveat, this
cannot be graded `VALID — pinned by validation` even though the pin is real and the value is
single-valued in every admissible call.

What keeps it safe in practice is not the pin but the framework: the `UpdateProgramRunArgs` spec check
catches a layout divergence independently, because `layout` is a projection of `page_config` and
`page_config` is part of the exact match. A ROW_MAJOR `cos` on a hit throws from the framework rather
than executing. That is a backstop rather than a pin, which is exactly the shape of a caveat — safe
today, resting on a mechanism outside the op, and producing a throw where a rebuild was intended.

(This is a meaningful difference from the two `kv_cache` siblings, where the analogous cross-tensor
consistency checks are also miss-path-only but there is no framework backstop, so the same structure
degrades to silent corruption.)

The guard that would close it is to repeat the three unhashed `Layout::TILE` checks in
`validate_runtime_args`, which both paths call. This document does **not** recommend doing so: that
function runs on the cache-hit path, so the checks would be paid on every dispatch, and what they buy
over the existing framework rejection is a clearer error message rather than a correctness improvement.
See recommendation 4.

### 5. `page_config` (the `Tile`) and `alignment` of all five tensors — the unguarded 32x32 assumption

**Verdict: CAVEAT, not BUG.** This op meets two of the three criteria for the tile bug — it requires
`Layout::TILE` and it derives all of its tile geometry from the architectural 32x32 constants with no
tile-geometry guard anywhere in the directory — but the framework rescues it. Its Metal 2.0 dispatch
path performs an exact `TensorSpec` comparison on every hit, and that comparison provably covers
`page_config`, so a differing `Tile` throws a diagnostic rather than silently executing the wrong
program. It fails loudly where a rebuild was intended, which is a caveat, not corruption.

**The factory is entirely 32x32-hardcoded, exactly like its siblings.** Five `tt::tile_size` calls
(which return the byte size of a 32x32 tile, not `tile.get_tile_size(format)`) size every dataflow
buffer, and bare `TILE_HEIGHT`/`TILE_WIDTH` do all the tile-count arithmetic:

```268:284:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/rotary_embedding_indexed/device/rotary_embedding_indexed_device_operation.cpp
    const tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(input.dtype());
    const uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    const tt::DataFormat cos_cb_data_format = datatype_to_dataformat_converter(cos.dtype());
    const uint32_t cos_single_tile_size = tt::tile_size(cos_cb_data_format);
    const tt::DataFormat sin_cb_data_format = datatype_to_dataformat_converter(sin.dtype());
    const uint32_t sin_single_tile_size = tt::tile_size(sin_cb_data_format);
    const tt::DataFormat trans_mat_cb_data_format = datatype_to_dataformat_converter(trans_mat.dtype());
    const uint32_t trans_mat_single_tile_size = tt::tile_size(trans_mat_cb_data_format);
    const tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(out.dtype());
    const uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    const uint32_t batch = input.padded_shape()[0];
    const uint32_t n_heads = input.padded_shape()[1];
    const uint32_t seq_len_t = input.padded_shape()[2] / TILE_HEIGHT;
    const uint32_t head_dim_t = input.padded_shape()[3] / TILE_WIDTH;
    const uint32_t cos_seq_len_t = cos.padded_shape()[2] / TILE_HEIGHT;
    const uint32_t sin_seq_len_t = sin.padded_shape()[2] / TILE_HEIGHT;
```

`TILE_HEIGHT` also lands directly in a reader compile-time arg, the same structural exposure that
`writer_tile_height` creates in `update_padded_kv_cache`:

```448:449:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/rotary_embedding_indexed/device/rotary_embedding_indexed_device_operation.cpp
             {"rotary_Ht", rotary_seq_len_t},
             {"tile_height", TILE_HEIGHT},  // reader divides kv_actual_global (tokens) into tiles
```

Nothing validates the geometry. There is no `tensor_spec().tile()` read and no tile-geometry `TT_FATAL`
anywhere in the op directory. The validator does require `Layout::TILE` on four tensors (`:157-160`) and
does pin `trans_mat` to a single tile — but that check is a *shape* check against the architectural
constant, not a tile check, and under `Tile{16, 32}` a `[1, 1, 32, 32]` trans_mat is two tiles, not one:

```168:176:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/rotary_embedding_indexed/device/rotary_embedding_indexed_device_operation.cpp
    // The reader pushes trans_mat as a single page (page 0) into a one-tile CB, so it must be exactly
    // one tile -- a larger tensor would be silently truncated to its first tile.
    TT_FATAL(
        trans_mat_shape.rank() == 4 && trans_mat_shape[0] == 1 && trans_mat_shape[1] == 1 &&
            trans_mat_shape[-2] == TILE_HEIGHT && trans_mat_shape[-1] == TILE_WIDTH,
        "trans_mat must be a single tile [1, 1, {}, {}] (got {})",
        TILE_HEIGHT,
        TILE_WIDTH,
        trans_mat_shape);
```

So on a cache *miss* a non-32x32 call would compile a program with mis-sized dataflow buffers and a
truncated trans_mat. That is a factory bug, and it is real — but it is not the program-cache bug class,
because a miss means the program was at least built for the tensor in front of it.

**Why the cache-hit exposure is a throw and not corruption.** The spec check quoted in
`## Cache-hit patch mechanism` above delegates to `tensorspecs_match_with_relaxation`, and the
`page_config` coverage was traced through the whole chain rather than assumed:

```61:68:tt_metal/impl/metal2_host_api/tensor_spec_relaxations.cpp
    switch (relaxation_mode(relaxation)) {
        case RelaxationMode::DynamicRank:
            return a.tensor_layout() == b.tensor_layout() && a.logical_shape().rank() == b.logical_shape().rank();
        case RelaxationMode::PaddedShapeOnly:
            return a.tensor_layout() == b.tensor_layout() && a.padded_shape() == b.padded_shape();
        case RelaxationMode::Strict: break;
    }
    return a == b;
```

This op leaves `TensorParameter::relaxations` default-constructed, so it takes the `Strict` branch and
compares whole `TensorSpec`s. From there:

- `TensorSpec::operator==` is `= default`
  (`tt_metal/api/tt-metalium/experimental/tensor/spec/tensor_spec.hpp:26`), so it compares
  `logical_shape_` and `tensor_layout_` memberwise.
- `TensorLayout::operator==` forwards to its impl
  (`tt_metal/impl/tensor/spec/layout/tensor_layout.cpp:492`), whose `operator==` is `= default`
  (`tt_metal/impl/tensor/spec/layout/tensor_layout_impl.hpp:25`) over members that include
  `page_config_` (`:30`).
- `PageConfig::operator==` is `= default` over its `config_` variant, and `TilePageConfig::operator==`
  is `= default` over its `Tile tile`
  (`tt_metal/api/tt-metalium/experimental/tensor/spec/layout/page_config.hpp:26,47`).
- `Tile::operator==` compares the tile and face shapes:

```122:124:tt_metal/impl/data_format/tile.cpp
bool Tile::operator==(const Tile& other) const {
    return tile_shape == other.tile_shape && face_shape == other.face_shape;
}
```

So `page_config` is genuinely covered, and it is covered in *every* relaxation mode — all three branches
above compare `tensor_layout()` unconditionally, so no `TensorSpecRelaxations` setting can ever relax the
tile away. That is a stronger guarantee than this op needs and it is worth relying on.

One precise limit: `Tile::operator==` does not compare the `transpose_within_face` / `transpose_of_faces`
flags. Those escape the check — but they equally escape the framework's default hash, whose
`Tile::attribute_values()` is `(tile_shape, face_shape, num_faces)`
(`tt_metal/api/tt-metalium/tile.hpp:46-47`). They are therefore not an omission relative to the default,
and this factory never reads them.

**Two-call sequence, and how it differs from the siblings.** Call 1: `input`, `cos`, `sin`, `trans_mat`
all `BFLOAT16`, `Layout::TILE`, `Tile{32, 32}`, interleaved DRAM. Call 2: identical padded shapes,
dtypes and memory configs, but `Tile{16, 32}`. The hash omits `page_config`
(`compute_program_hash:230-250` hashes `dtype`, `memory_config`, `padded_shape` and — for `input` only —
`logical_shape` and `layout`, never the tile), so the key is identical and the cache hits — exactly as it does in
the two `kv_cache` ops. The divergence is in what happens next. Here, `override_runtime_arguments`
calls `UpdateProgramRunArgs`, which validates before it patches
(`tt_metal/impl/metal2_host_api/program_run_args.cpp:1105-1107`, delegating to
`ValidateUpdateProgramRunArgs` and thence to `ValidateTensorArgs` at `:1087`), the `Tile{16, 32}` spec
fails the comparison above, and `report_tensor_arg_mismatch` throws a named diagnostic before any
kernel runs. In `update_padded_kv_cache` and `zero_padded_kv_cache` the same source-level mistake reaches
the descriptor buffer-binding fast path
(`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:726-731`), which patches addresses and dynamic scalars
and compares nothing, so the stale 32x32 compile-time args and CB page sizes are simply executed against
a 16-row-tile buffer and the KV cache is silently corrupted.

**This is the single most useful observation across the three documents.** Three sibling ops, written by
the same team against the same hardcoded 32x32 assumption, with the same `page_config` omission from the
hash. Two of them silently corrupt data; this one fails loudly. The entire difference is which cache-hit
mechanism the op happens to be built on — Metal 2.0 `ProgramSpec` with named tensor parameters and an
enforced spec contract, versus a `ProgramDescriptor` with raw buffer bindings and no contract at all. The
safety here is not a property of this op's code; it is a property of the dispatch layer, and it would
evaporate the moment someone set a relaxation or passed `skip_validation`. It should not be mistaken for
the op being correct.

`alignment` deserves a separate note: it does not appear in this factory at all. The op reads and writes
through Metal 2.0 `TensorAccessor`s bound by name (`reader_...cpp:97-100`) rather than through
host-emitted `TensorAccessorArgs`, so no aligned page size is baked into a compile-time arg the way it is
in this family's two `kv_cache` ops. Its omission is `CAVEAT` purely for the throw-not-rebuild reason,
with no underlying factory defect behind it.

**The guard.** The recommended fix is the same for all three siblings, and for this one it converts the
verdict to `VALID — pinned by validation` while also fixing the miss-path factory bug:

```94:98:ttnn/cpp/ttnn/operations/data_movement/sharded/interleaved_to_sharded/device/interleaved_to_sharded_op.cpp
    if (input_tensor.layout() == Layout::TILE) {
        auto tile = input_tensor.tensor_spec().tile();
        if (tile.get_height() != tt::constants::TILE_HEIGHT || tile.get_width() != tt::constants::TILE_WIDTH) {
            return {false, fmt::format("interleaved_to_sharded requires standard 32x32 tiles, got {}x{}", tile.get_height(), tile.get_width())};
        }
```

Making the factory genuinely tile-aware instead would require adding `page_config` to
`compute_program_hash` in the same change, since the program would then provably vary with `Tile`.

### 6. Per-core runtime args and the work split (not re-applied by the override)

**Verdict: VALID — invariant.**

The override deliberately does not re-set `batch_start`/`batch_end`/`seq_t_start`/`seq_t_end`, relying
on `UpdateProgramRunArgs`'s partial-update semantics. That is sound because the split is a pure
function of hashed values plus device constants:

```309:326:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/rotary_embedding_indexed/device/rotary_embedding_indexed_device_operation.cpp
    const bool row_major = true;
    const uint32_t num_cores = num_cores_x * num_cores_y;
    const uint32_t batch_parallel_factor = std::min(batch, num_cores);
    const uint32_t seq_parallel_factor = std::min(num_cores / batch_parallel_factor, seq_len_t);
    const uint32_t batch_per_core = (batch + batch_parallel_factor - 1) / batch_parallel_factor;
    const uint32_t seq_per_core = (seq_len_t + seq_parallel_factor - 1) / seq_parallel_factor;

    const uint32_t num_sin_cos_rows_per_core = (seq_len_t + seq_parallel_factor - 1) / seq_parallel_factor;
    const uint32_t num_rows_per_core = num_sin_cos_rows_per_core * n_heads;

    uint32_t num_cos_sin_tiles = 2 * head_dim_t * num_sin_cos_rows_per_core;
    uint32_t input_cb_num_tiles = num_sin_cos_rows_per_core * num_input_tiles;

    const bool use_reload_impl = num_rows_per_core > 8 || freq_per_head;
    if (use_reload_impl) {
        input_cb_num_tiles = num_input_tiles;
        num_cos_sin_tiles = num_input_tiles;
    }
```

`batch`, `n_heads`, `seq_len_t` and `head_dim_t` all come from `input.padded_shape()` (hashed);
`freq_per_head` from `cos.padded_shape()` (hashed); `num_cores_x`/`num_cores_y` from
`compute_with_storage_grid_size()`, a device constant that the per-device program cache already
partitions on. `use_reload_impl` additionally drives the `RELOAD_IMPL` define on all three kernels
(`:400,438,459,475`), a compile-time value — it too is fully determined by the hashed set. The padded
sequence and head dimensions, which the brief flags as the most likely place for a structural leak,
are hashed in full via `input.padded_shape()` rather than a volume, so no two differently-shaped inputs
can collide onto one work split.

### 7. `input.storage`, `cos.storage`, `sin.storage`, `trans_mat.storage` variant kind

**Verdict: CAVEAT — pinned only on the miss path.**

```142:145:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/rotary_embedding_indexed/device/rotary_embedding_indexed_device_operation.cpp
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "input must be on device");
    TT_FATAL(cos.storage_type() == StorageType::DEVICE, "cos must be on device");
    TT_FATAL(sin.storage_type() == StorageType::DEVICE, "sin must be on device");
    TT_FATAL(trans_mat.storage_type() == StorageType::DEVICE, "trans_mat must be on device");
```

Constant across every admissible call, so on the miss path it carries no information. But all four sit
above the `validate_runtime_args` delegation at `:184`, and this op's hit validator replaces the miss
validator rather than supplementing it, so none of them re-run on a hit. Unlike omission 4, the
framework spec check is *no* backstop here: `TensorSpec` covers `logical_shape` and `tensor_layout` and
says nothing about the storage variant, so a host-storage tensor is not rejected by the comparison.

The severity is nevertheless low. A host tensor has no device allocation to bind, so the hit path faults
when `UpdateProgramRunArgs` tries to resolve its buffer, rather than executing a stale program against a
valid-looking one. The outcome is a crash on the offending call, not silent corruption on a later one.

That difference in kind is why this document deliberately does **not** recommend closing it. Repeating
the four checks in `validate_runtime_args` would upgrade the verdict to `VALID — pinned by validation`,
but `validate_runtime_args` runs on the cache-hit path, which is the fast path, so four
`storage_type()` queries would be paid on every dispatch for the life of the process. The only thing
they would buy is a clearer error message in front of a fault that already happens on the same call.
The right disposition is to leave this as a recorded caveat; see recommendation 4.

### 8. Buffer addresses of all operands and the output

**Verdict: VALID — patched, and required.** Addresses must never be hashed; they are re-bound by name
on every hit through `run_args.tensor_args` (`override_runtime_arguments:624-631`), covering all five
mandatory parameters plus `metadata`. Because `has_metadata` is hashed, the conditional emplace at
`:630-631` can never disagree with the cached program's parameter set.

The output tensor deserves a specific note. It is freshly allocated on every call
(`create_output_tensors:202-206`), so its address changes call to call — and its spec must therefore be
reproducible from the hashed set, or the exact-match check would reject it. It is:

```194:200:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/rotary_embedding_indexed/device/rotary_embedding_indexed_device_operation.cpp
RotaryEmbeddingIndexedDeviceOperation::spec_return_value_t RotaryEmbeddingIndexedDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    return tt::tt_metal::TensorSpec(
        input.logical_shape(),
        tt::tt_metal::TensorLayout(input.dtype(), tt::tt_metal::PageConfig(input.layout()), args.output_mem_config));
}
```

Every input to that construction — `input.logical_shape()`, `input.dtype()`, `input.layout()`,
`args.output_mem_config` — is hashed. Note it builds the output `PageConfig` from `input.layout()`
rather than `input.page_config()`, so the output tile is always canonical regardless of the input's;
that is what keeps the output spec deterministic despite omission 5.

### 9. `my_sp_coord` and `sp_factor` (derived, not attributes)

**Verdict: VALID — invariant.** Both are baked as per-coordinate compile-time args
(`create_at:290-296`, consumed at `:450-451`), and both are determined by the dispatch coordinate and
the hashed `cluster_axis`. Coordinates are folded into the key by the framework for the custom-hash
path as well as the default one:

```989:993:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
        // Combine with the mesh coordinates the workload is targeting.
        for (const auto& coord : mesh_device_operation_utils::extract_tensor_coordinates(tensor_args, mesh_device)) {
            hash = ttsl::hash::hash_objects(hash, coord);
        }
        return hash;
```

so a workload compiled for one coordinate set can never be reused for another. This matters more here
than for a coordinate-blind op, because `create_mesh_workload` stamps a *different* program per
coordinate (`:596-610`) rather than one program across the range.

## Keys the custom hash adds beyond the default

- `input.padded_shape()`, `cos.padded_shape()`, `sin.padded_shape()`, `trans_mat.padded_shape()` —
  derivations in the default key, promoted to first-class. These are what the factory actually reads,
  so promoting them is what allows the corresponding `logical_shape`s to be dropped.
- `metadata->padded_shape()` — likewise.
- `input.layout()` — a projection of `page_config`, kept because it feeds the output spec.
- `tensor_args.metadata.has_value()` — separates the two program variants; the reader's
  `HAS_METADATA` define, its `META_DFB` dataflow buffer, its `METADATA_PARAM` binding and its common-arg
  schema all switch on it (`create_at:376-382,392-395,402-404,420-433`), and all of those are frozen on
  a hit, so hashing it is mandatory rather than optional.

## Framework side effect of having a custom hash

Defining `compute_program_hash` opts this op out of attribute-level collision resolution:

```1012:1014:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
        if constexpr (requires { DeviceOperation::compute_program_hash(attrs, tensor_args); }) {
            return key;  // custom hash -> opt out beyond the op-identity prefix
        } else {
```

`ProgramCacheKey::canonical` degrades to the op type name, so a 64-bit collision between two different
configurations resolves to a wrong hit instead of a rebuild. This op is better insulated than most
custom-hash ops: on a colliding hit, any operand spec difference is caught by the exact
`TensorParameter` match, so the collision would have to be between two configurations whose five tensor
specs are all identical and which differ only in `cluster_axis`, `compute_kernel_config` or
`output_mem_config`. Still worth noting, since a `compute_kernel_config` collision would silently run
at the wrong math fidelity.

## Summary

| Omitted vs. default | Used by program? | Patched on hit? | Verdict |
|---|---|---|---|
| `attrs.kv_actual_global` | Yes (reader common arg) on the scalar path; no on the metadata path | Yes (override) / n/a | VALID — patched / VALID — unused |
| `metadata.dtype` | Only via the on-device read width | No | VALID — pinned by validation |
| `cos`/`sin`/`trans_mat` `logical_shape` | No (padded shapes used) | n/a | CAVEAT — exact spec match turns a divergence into a throw |
| `cos`/`sin`/`trans_mat` `layout` | No | n/a | CAVEAT — pinned only on the miss path (framework spec check is the real backstop) |
| `page_config` (`Tile`) of all five tensors | Yes (`tile_height` compile-time arg, DFB sizes, tile counts — all hardcoded 32x32) | No — but a mismatch is rejected before dispatch | CAVEAT — fail-safe: exact spec equality covers `page_config`, so a differing `Tile` throws where its two siblings silently corrupt |
| `alignment` of all five tensors | No (no host-emitted `TensorAccessorArgs`) | n/a | CAVEAT — same throw-not-rebuild exposure |
| Per-core work-split args | Yes (per-core RTAs, `RELOAD_IMPL`) | No (deliberately) | VALID — invariant (function of hashed shapes) |
| Operand `storage` kinds | n/a | n/a | CAVEAT — pinned only on the miss path (no spec-check backstop; fails as a crash) |
| All buffer addresses (incl. the fresh output) | Yes | Yes (`UpdateProgramRunArgs` tensor bindings) | VALID — patched, required |
| `my_sp_coord`, `sp_factor` | Yes (compile-time args) | n/a (coordinate hashed) | VALID — invariant |

**No program-cache correctness bug was found.** The single omitted attribute is the per-chunk position
scalar, and it is handled the way this class of value should be: kept out of the key, re-applied to the
reader's common runtime argument on every hit, consumed only on-device so nothing derived from it can
be baked, and re-validated on every hit by a host-side check that reproduces the kernel's own
`update_idxt` arithmetic. Every compile-time argument, dataflow-buffer size, kernel define and per-core
runtime argument in `create_at` is a function of the hashed set plus device constants the per-device
cache already partitions on. The tensor-spec omissions do not threaten correctness because
`UpdateProgramRunArgs` enforces exact `TensorSpec` equality against the baked `TensorParameter`s on
every hit; their residual cost is a hard rejection instead of a recompile in call patterns nothing
currently exercises.

That verdict includes the tile omission, and it is worth being explicit about how narrowly it was
earned. This op carries the same unguarded 32x32 assumption as its two siblings — five `tt::tile_size`
calls, four bare-constant tile-count conversions, a `tile_height` compile-time arg fixed to
`TILE_HEIGHT`, and no tile-geometry check anywhere in the directory — and its hash omits `page_config`
just as theirs do. In `update_padded_kv_cache` and `zero_padded_kv_cache` that combination is a BUG: a
`Tile{16, 32}` call hits the cache entry built for `Tile{32, 32}` and executes stale compile-time args
and CB page sizes against it, silently corrupting the KV cache. Here the same call is rejected before
dispatch, because the exact `TensorSpec` comparison performed by `UpdateProgramRunArgs` reaches all the
way down to `Tile::operator==` and therefore covers `page_config` in every relaxation mode. The op is
not safer than its siblings; the dispatch layer it was built on is. It still has a genuine miss-path
factory bug — a non-32x32 call would compile mis-sized dataflow buffers and a truncated trans_mat — and
the same one-line guard fixes both halves.

A separate, lower-severity finding runs through omissions 4 and 7: because the op defines
`validate_on_program_cache_hit`, that validator *replaces* the miss validator on hits rather than
supplementing it, and this op's hit validator delegates to `validate_runtime_args` and does nothing
else. Everything the miss validator checks before its own delegation at `:184` is therefore absent on
the hit path — the four `Layout::TILE` pins, the four `storage_type() == DEVICE` pins, the
`buffer() != nullptr` and same-device checks, the `trans_mat` single-tile check, and the `cos`/`sin`
dtype and shape equalities. A narrow hit validator is a hazard rather than a safeguard in general: by
existing, it disables everything above it.

Filtered for reachability, the practical loss is smaller than that list suggests and, importantly, it
is entirely non-silent. Most of the dropped checks constrain shapes and dtypes that are in the cache
key, and a miss-only pin on a hashed value cannot be evaded — any call carrying a new value of it
misses and meets the pin there. What survives the filter is the allocation-and-device block
(`:142-155`), which faults when `UpdateProgramRunArgs` cannot resolve a buffer, and the three unhashed
`Layout::TILE` pins (`:158-160`), which the framework's exact `TensorSpec` comparison rejects with a
diagnostic. Both fail loudly on the offending call.

That is why the recommendations leave both regraded rows as documented caveats rather than closing
them. `validate_runtime_args` runs on the cache-hit path, so every check moved into it is paid on every
dispatch for the life of the process, and here the purchase would be a better error message in front of
a failure that already occurs — not a correctness improvement. The one hit-path check this document
does recommend adding is the tile guard, and only because that one closes a defect nothing else
catches.

Two family-level observations. First, all three `deepseek_prefill` ops audited here (this one,
`update_padded_kv_cache`, `zero_padded_kv_cache`) share the same correct core idiom: the moving
per-request index is omitted from the hash, patched on every hit, and re-checked by a validator that
runs on both the miss and hit paths — with a dual "scalar or 1-element device tensor" path selected by
a hashed `has_value()` bit. They also share the same 32x32 defect, and the divergence in its
consequences is purely a dispatch-mechanism artifact. All three also define a narrow hit validator that
delegates to a shared `validate_runtime_args` and thereby drops the rest of their miss-time pins on the
hit path; this op's shared function is the most complete of the three, which is why its regrades are
confined to two low-severity rows. In all three, the only miss-only pin worth buying back on the hot
path is the one whose absence is *silent* — which in this op is none of them, and in
`update_padded_kv_cache` is the `cache`-vs-`input` dtype and layout pair. Second, this op is the only
one of the three that
hashes its metadata tensor's `memory_config` and `padded_shape`; the other two hash only the engagement
bit while baking that tensor's `TensorAccessorArgs` into kernel compile-time args, which is a real
defect in those two. The pattern implemented here is the one they should adopt.

## Recommendations

**Every guard below names the function it must go into, and for this op that function is
`validate_runtime_args`.** Because the op defines `validate_on_program_cache_hit`, the miss validator is
skipped entirely on a hit; a guard placed in `validate_on_program_cache_miss` would not run on the
offending second call, which is the only call a cache bug reaches. `validate_runtime_args` is the right
home because both validators delegate to it (`:184`, `:191`), so one placement covers both paths.

**And every guard below is priced.** The cache-hit path is the fast path — it is what the program cache
exists to make cheap — so a `TT_FATAL` added to `validate_runtime_args` is paid on every dispatch for
the life of the process. That is why only one new check is recommended here (the tile guard, which
closes a real factory bug) and why both of the regraded rows are deliberately left as documented
caveats rather than fixed.

There are two distinct ways to close a miss-only pin in this op, and they are not interchangeable:

- **Targeted (recommended):** move the specific `TT_FATAL`s into `validate_runtime_args`. Adds only
  those checks per dispatch. This is what recommendation 2 means.
- **Wholesale (alternative, and the more expensive one):** delete `validate_on_program_cache_hit` so
  the op falls onto the dispatcher's substitution branch and the full miss validator runs on every hit.
  Recommendation 5 states what that costs concretely.

1. Hash `cos.logical_shape()`, `sin.logical_shape()` and `trans_mat.logical_shape()` alongside the
   padded shapes, or declare `TensorSpecRelaxations::match_padded_shape_only` on those three
   `TensorParameter`s. Either makes omission 3 explicit; today the hash relaxes what the framework then
   requires exactly, so the "relaxation" can only ever surface as a crash.
2. Reject a non-32x32 `Tile` on every operand, closing omission 5. Assert
   `tensor_spec().tile().get_height() == TILE_HEIGHT` and the same for `get_width()`, on `input`, `cos`,
   `sin` and `trans_mat`, in the same shape as the `interleaved_to_sharded` guard quoted in omission 5.
   This is the highest-value change in this list even though the tile omission is only a caveat here,
   because it fixes two things at once: it converts the omission's verdict to
   `VALID — pinned by validation`, and it closes the genuine miss-path factory bug (a non-32x32 call
   currently compiles mis-sized dataflow buffers and silently truncates `trans_mat` to its first tile).
   **Target function:** `validate_runtime_args`, not `validate_on_program_cache_miss`. The miss-path
   factory bug is caught either way, since the miss validator delegates at `:184` — but only the shared
   function also runs on the hit, which is where the caller currently gets an opaque
   `TensorParameter`-named spec-mismatch diagnostic instead of a message naming the tile. Placing it in
   the miss validator alone would leave that second-call diagnostic exactly as unhelpful as it is today.
   **Per-dispatch cost:** two `uint32_t` comparisons against constants, on four tensors — eight
   comparisons per dispatch. This is the one new hit-path check this document recommends. It is worth
   the price because it is the only recommendation here that fixes a genuine defect rather than
   improving a diagnostic: a non-32x32 call currently compiles a mis-sized program on the miss path,
   and no framework mechanism catches that. If the eight comparisons are judged too expensive for this
   op's dispatch rate, the acceptable reduction is to check `input` only in `validate_runtime_args` and
   leave the other three in `validate_on_program_cache_miss` — the four tiles are required to be
   mutually consistent by the shape checks, and `input` is the one whose geometry drives the work split.
   This is a family-wide gap: apply the same guard to `update_padded_kv_cache` and
   `zero_padded_kv_cache`, where it is not a caveat but a fix for silent data corruption, and where the
   `validate_runtime_args` placement is load-bearing rather than merely preferable.
3. Correct two stale comments, each of which is the stated safety argument for the function it sits on,
   and each of which currently claims more than the code delivers. An inaccurate safety comment is worse
   than none, because it talks the next reader out of checking.
   - `compute_program_hash` (`:210-214`) states the hash covers "the full input, cos, sin and trans_mat
     specs". It covers projections of them, and that gap is exactly omissions 3-5.
   - `validate_on_program_cache_hit` (`:189-190`) states that "structural constraints are hashed and so
     guaranteed unchanged here". True of the shapes and dtypes it has in mind; false of the four
     `Layout::TILE` pins and the four `storage_type()` pins, which are neither hashed nor re-checked —
     omissions 4 and 7.
4. **Leave the regraded omissions 4 and 7 as documented CAVEATs — do not move their pins onto the hit
   path.** This is a deliberate non-recommendation, recorded so it is not mistaken for an oversight.

   The available fix is to move the three unhashed `Layout::TILE` checks (`:158-160`) and the four
   `storage_type() == StorageType::DEVICE` checks (`:142-145`) into `validate_runtime_args`, upgrading
   both rows to `VALID — pinned by validation`. Neither is worth its price, for the same underlying
   reason: **neither failure is silent.** The layout divergence is already rejected on the hit by the
   framework's exact `TensorSpec` comparison, which throws a named diagnostic before any kernel runs;
   the storage divergence already faults when `UpdateProgramRunArgs` tries to resolve a buffer that a
   host tensor does not have. In both cases the caller gets a hard failure on the offending call today.
   Moving the checks buys a better error message, and charges seven extra `TT_FATAL`s per dispatch for
   it, forever. For a rotary embedding on the prefill hot path that is the wrong trade.

   The judgement would flip if either failure were silent — that is exactly the distinction that makes
   the tile guard in recommendation 2 worth paying for and these two not. It would also flip for the
   layout row specifically if anyone ever set `TensorSpecRelaxations` on these `TensorParameter`s or
   passed `skip_validation`, since the framework backstop is the only thing holding that row up; a
   comment at the `TensorParameter` declarations (`create_at:385-395`) noting that the audit relies on
   default-constructed relaxations would be cheaper insurance than the runtime checks.

5. **Do not delete `validate_on_program_cache_hit` to fix this.** Deleting it would put the op on the
   dispatcher's substitution branch, so the full miss validator would run on every hit and every pin
   above would hold by construction — genuinely the simplest and safest fix, and immune to a future
   check being added to the wrong function. It is recorded here as the alternative rather than the
   recommendation because the cost is substantial and easy to miss:

   the whole of `:137-182` would move onto the hot path. Concretely, per dispatch: four
   `storage_type()` queries, four `buffer()` null checks, three `device()` comparisons, four `layout()`
   queries, four `padded_shape()` accesses plus two rank queries, a five-term compound predicate on
   `trans_mat`'s shape, a dtype equality, two shape equalities and a modulo — roughly twenty-five
   `TT_FATAL` conditions, on every call, for the life of the process. The narrow hit validator was
   almost certainly written the way it is precisely to avoid that, and the reachability table in
   `## Cache-hit patch mechanism` vindicates the choice: fewer than half of those lines can be reached
   on a hit at all, and none of the reachable ones fails silently.

   The one thing the current arrangement genuinely costs is fragility — the hit validator's comment
   already claims more than the code delivers (see recommendation 3), and nothing stops the next person
   adding a load-bearing check above the delegation and not noticing it never runs. The cheap mitigation
   is a comment at the top of `validate_on_program_cache_miss` stating that everything above the
   `validate_runtime_args` call at `:184` is miss-only by design, and that any check which must hold on
   a hit belongs inside `validate_runtime_args`.
