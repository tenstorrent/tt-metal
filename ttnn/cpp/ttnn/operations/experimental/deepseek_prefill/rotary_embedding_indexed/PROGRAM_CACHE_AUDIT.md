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

## Post-fix status — commit fab067a

**Verdict: CLEAR — with one justified relaxation** (`kv_actual_global`). Every other omission relative
to the framework default is now zero-value: pinned by a `TT_FATAL` that genuinely runs on the
cache-hit path, or functionally determined by a term that is in the key. No program-cache correctness
bug remains, and the miss-path factory defect recorded in omission 5 is closed as well.

**What fab067a changed in this op**

- `compute_program_hash` now hashes each operand's whole `tensor_spec()` rather than a projection of
  it (`device/rotary_embedding_indexed_device_operation.cpp:262-275`), with the optional `metadata`
  tensor's spec folded in when engaged (`:271-274`). That one change subsumes pre-fix omissions 2, 3,
  4 and 5: `logical_shape`, `dtype`, `page_config` (hence the `Tile`), `memory_config` and `alignment`
  are now in the key for all five tensors.
- Added a `require_standard_tile` lambda to the **shared** `validate_runtime_args` (`:87-109`),
  applied to all four operands (`:106-109`). It asserts both `layout() == Layout::TILE` and 32x32 tile
  geometry, so the three unhashed `Layout::TILE` pins of omission 4 run on hits for the first time,
  and a non-32x32 call is now reported by operand name instead of compiling a mis-sized program.
- Added `cos.storage_type() == StorageType::DEVICE` to `validate_runtime_args` (`:72-75`) — the one
  operand whose `device()` is dereferenced on the following line.
- Rewrote the two comments that overclaimed safety: the hash rationale (`:245-261`) and the hit
  validator's (`:224-227`), which now states explicitly that anything which must hold on a hit belongs
  inside `validate_runtime_args`.
- Documented the 32x32 tile requirement in the nanobind docstring
  (`rotary_embedding_indexed_nanobind.cpp:45-47`).

**What remains open**

- `attrs.kv_actual_global` stays out of the key by design — a reader common runtime arg re-applied on
  every hit (`:658-661`) and consumed only on-device. This is the relaxation the op exists to exploit,
  and it is paid for by the per-hit shard-bound check at `:140-160`.
- The op still defines `validate_on_program_cache_hit`
  (`device/rotary_embedding_indexed_device_operation.hpp:90`), so the miss validator is still
  *replaced* rather than supplemented on hits, and everything above the delegation at `:219` remains
  miss-only. That is now benign — see the regraded reachability table below — but the argument has to
  be re-derived whenever the key is loosened.
- Defining a custom hash still forfeits attribute-level collision resolution
  (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:1035-1036`); unchanged by this commit. On a
  colliding hit every operand spec is now compared exactly, so a collision would have to be between
  two configurations whose five specs all agree and which differ only in `cluster_axis`,
  `compute_kernel_config` or `output_mem_config`.

**Metal 2.0 port**

Clear to port, and this op is the model the other three should follow. It carries **no**
logical-vs-padded relaxation at all: pre-fix it hashed `padded_shape` and dropped the operands'
`logical_shape`, and fab067a moved it to hashing `tensor_spec()` wholesale, which is strictly finer —
`padded_shape` is a derivation of `logical_shape`, `page_config` and `alignment`, all now in the key.
So there is nothing here to express as a `TensorSpecRelaxations` flag, and the op's `TensorParameter`s
are correctly left with default-constructed relaxations
(`tt_metal/api/tt-metalium/experimental/metal2_host_api/tensor_spec_relaxations.hpp:41`), which require
an exact match. The next subsection is why that agreement is the property to preserve in the ports of
its two `kv_cache` siblings.

### The `report_tensor_arg_mismatch` interaction

This is the specific defect the hash rewrite closes, and it is worth stating precisely, because it is
the failure mode the siblings' Metal 2.0 ports will inherit if they key on a projection.

The framework derives the run-time accept/reject predicate and the relaxation-aware hash from **one**
field set. `pertinent_fields` maps a `TensorSpecRelaxations` to the fields that are load-bearing
(`tt_metal/impl/metal2_host_api/tensor_spec_relaxations.cpp:67-87` — `match_padded_shape_only` maps to
`PertinentFields{.padded_shape = true}`, and a default-constructed relaxation maps to
`PertinentFields{.whole_spec = true}`), and both `hash_tensorspec_with_relaxation` (`:116`) and
`tensorspecs_match_with_relaxation` (`:161-201`) consume that same set. `ValidateTensorArgs` delegates
its accept/reject to the predicate on every dispatch
(`tt_metal/impl/metal2_host_api/program_run_args.cpp:176-189`), throwing through
`report_tensor_arg_mismatch` on rejection.

**Pre-fix:** the key was a projection of each spec, while the `TensorParameter`s were declared with
default relaxations and therefore compared `whole_spec`. The two disagreed. Two `cos` tensors differing
only in a field the projection dropped — `logical_shape`, or `alignment` — computed the *same* key, hit
the cache, and were then rejected by `report_tensor_arg_mismatch` when `UpdateProgramRunArgs` compared
the whole spec. Fail-safe, but the outcome was a hard throw on a call that should simply have compiled
a second program: the hash relaxed exactly what the framework then required exactly.

**Post-fix:** the key hashes `tensor_spec()` wholesale, which is the same field set `whole_spec`
compares, so key and predicate agree **by construction**. The two `cos` tensors now compute different
keys, miss, and rebuild. No spec difference can produce a hit-then-throw any more.

The generalisable rule for the ports: the hash must key on the same fields the declared
`TensorSpecRelaxations` makes load-bearing. Either hash the whole spec and leave the relaxations
default, as this op now does, or declare the relaxation you actually want — `match_padded_shape_only`
(`tensor_spec_relaxations.hpp:49`) for a genuine padded-shape-only dependence — and key on
`padded_shape`. Hashing a projection while validating strictly is the one combination that is always
wrong.

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

```222:229:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/rotary_embedding_indexed/device/rotary_embedding_indexed_device_operation.cpp
void RotaryEmbeddingIndexedDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // kv_actual_global is not hashed and can differ from the compiled program's call; re-validate
    // every hit. Structural specs are hashed wholesale, so a structural change misses rather than
    // arriving here -- but note the checks above the delegation in validate_on_program_cache_miss are
    // miss-only, so anything that must hold on a hit belongs in validate_runtime_args.
    validate_runtime_args(args, tensor_args);
}
```

The miss validator also ends by delegating to `validate_runtime_args` (`:184`), so the two paths differ
by exactly the checks the miss validator performs *before* that delegation — lines 142 through 182. The
hit path therefore loses all of the following:

- `storage_type() == DEVICE` on `input`, `cos`, `sin` and `trans_mat` (`:142-145`).
- The `buffer() != nullptr` checks and the `device() == input.device()` checks on all four operands
  (`:149-155`).
- `Layout::TILE` on all four operands (`:157-160`). **Post-fix this loss is nominal**: fab067a's
  `require_standard_tile` re-asserts `layout() == Layout::TILE` on all four inside
  `validate_runtime_args` (`:92-96`), so the pin holds on both paths through a different line.
- The rank-4 checks and the `trans_mat` single-tile check (`:166-176`).
- `cos.dtype() == sin.dtype()`, `cos_shape == sin_shape`, and the input-vs-cos head-dim equality
  (`:177-179`).
- The input seq tile-alignment check (`:182`).

What *does* run on both paths is `validate_runtime_args` (`:60-126`): the `cluster_axis` check, the
2D-mesh check, the `chunk_local_t > 0` check, the entire metadata-tensor block (`:82-94`, including the
`metadata.dtype() == UINT32` pin that omission 2 relies on), and the scalar-path `kv_actual_global`
tile-alignment and shard-bound checks.

Pre-fix, the comment on the hit validator asserted that "structural constraints are hashed and so
guaranteed unchanged here." That was true of the shapes and dtypes it had in mind, but not of the four
`Layout::TILE` pins or the four `storage_type()` pins, which were then neither hashed nor re-run —
which is what drove the pre-fix regrade of omissions 4 and 7 to
`CAVEAT — pinned only on the miss path`. fab067a closes both: the layout pins are re-asserted inside
`validate_runtime_args` by `require_standard_tile` (`:92-96`) *and* the layout is now hashed as part of
`page_config`, and the storage pins turn out never to have been reachable at all — see the framework
correction under omission 7. The comment itself was rewritten (`:224-227`) and now describes the actual
contract. Omission 2 was never affected, because its pin genuinely does live in
`validate_runtime_args`.

This op is better off than its two siblings in the same situation, for the same reason it is better off
on the tile: the framework's exact `TensorSpec` comparison independently rejects a `layout` divergence
on a hit, since `layout` is a projection of `page_config`. That is a backstop, not a pin, and it does
not cover the storage-kind or cross-tensor-equality checks — so the caveats are real, just low-severity.

**Which of the dropped checks are actually reachable.** The list above is the mechanical diff, but most
of those checks constrain values that are themselves in the cache key, and a miss-only pin on a *hashed*
value cannot be evaded: any call carrying a new value of that parameter misses, and the miss validator
runs and rejects it there. Filtering the list against the post-fix
`compute_program_hash:262-275`, which keys every operand's whole `tensor_spec()`:

| Dropped check | Constrains | In the key? | Reachable on a hit? |
|---|---|---|---|
| `storage_type() == DEVICE` ×4 (`:142-145`) | storage variant kind | No | No — pinned by the framework in `launch()`, see omission 7 |
| `buffer() != nullptr` ×4 (`:149-152`) | allocation | No | No — pinned by the framework in `launch()`, see omission 7 |
| `device() == input.device()` ×3 (`:153-155`) | device identity | No | **Yes** |
| `Layout::TILE` ×4 (`:157-160`) | all four operands' layout | Yes, via `page_config` | No — and re-asserted on both paths by `require_standard_tile` (`:92-96`) |
| Rank-4 on `input` and `cos` (`:166-167`) | both specs | Yes, both | No |
| `trans_mat` single-tile check (`:170-176`) | `trans_mat`'s spec | Yes | No |
| `cos.dtype() == sin.dtype()` (`:177`) | both dtypes | Yes, both | No |
| `cos_shape == sin_shape`, input-vs-cos head dim (`:178-179`) | both specs | Yes, both | No |
| Input seq tile-alignment (`:182`) | `input`'s spec | Yes | No |

Post-fix the only reachable loss is the three `device() == input.device()` comparisons (`:153-155`),
and it does not fail silently: a cross-device operand faults when `UpdateProgramRunArgs` tries to
resolve its buffer against the wrong mesh device. Everything else is now unreachable — the shapes,
dtypes, layouts and tiles are all in the key via `tensor_spec()`, and the storage-kind and allocation
rows were never reachable on any path (omission 7). This is what drives the recommendations at the end
of this document: there is no silent-corruption path here to buy back, so no new per-dispatch check is
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

```262:275:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/rotary_embedding_indexed/device/rotary_embedding_indexed_device_operation.cpp
    auto hash = tt::tt_metal::operation::hash_operation<RotaryEmbeddingIndexedDeviceOperation>(
        tensor_args.metadata.has_value(),
        args.cluster_axis,
        args.compute_kernel_config,
        args.output_mem_config,
        tensor_args.input.tensor_spec(),
        tensor_args.cos.tensor_spec(),
        tensor_args.sin.tensor_spec(),
        tensor_args.trans_mat.tensor_spec());
    if (tensor_args.metadata.has_value()) {
        // metadata is an optional TensorParameter, compared just as strictly when it is present.
        hash = ttsl::hash::hash_objects(hash, tensor_args.metadata->tensor_spec());
    }
    return hash;
```

Three of the four attributes are kept; only `kv_actual_global` is dropped. All five tensors
participate, each by its **whole** `TensorSpec` — so the key covers `logical_shape`, `dtype`,
`page_config`, `memory_config` and `alignment` for every one of them, and `padded_shape` as a
derivation of those.

**Pre-fix, this body hashed a projection instead** (`dtype`, `memory_config` and `padded_shape` per
tensor, plus `logical_shape` and `layout` for `input` only, and `memory_config` + `padded_shape` for
`metadata`), and its explanatory comment claimed the hash covered "the full input, cos, sin and
trans_mat specs" when it covered projections of them. That gap between the claim and the code was
exactly pre-fix omissions 3-5, and fab067a closed both the gap and the comment (`:245-261`). The
projection form is retained in each omission below so the reproductions stay readable.

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

```222:229:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/rotary_embedding_indexed/device/rotary_embedding_indexed_device_operation.cpp
void RotaryEmbeddingIndexedDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // kv_actual_global is not hashed and can differ from the compiled program's call; re-validate
    // every hit. Structural specs are hashed wholesale, so a structural change misses rather than
    // arriving here -- but note the checks above the delegation in validate_on_program_cache_miss are
    // miss-only, so anything that must hold on a hit belongs in validate_runtime_args.
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

**Verdict: NO LONGER OMITTED after fab067a** (was VALID — pinned by validation). The hash now folds in
`metadata->tensor_spec()` when the tensor is engaged (`compute_program_hash:271-274`), so the dtype is
in the key outright and the pin below is belt-and-braces rather than the sole guarantee.

**Pre-fix:** the hash kept `metadata`'s `memory_config` and `padded_shape` but dropped its dtype, and
the omission was explicitly compensated:

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

Worth calling out because it is the family contrast: this op now hashes `metadata->tensor_spec()`
outright (`compute_program_hash:271-274`), so its metadata tensor's dtype, buffer type and page geometry
are all part of the cache key. Its two siblings in this family, `update_padded_kv_cache` and
`zero_padded_kv_cache`, still hash only the `has_value()` bit while compiling their metadata tensor's
`TensorAccessorArgs` — buffer type and aligned page size included — into kernel compile-time args, and
close the gap with `TT_FATAL`s in `validate_runtime_args` instead. This op is the correct model for that
pattern.

### 3. `cos.logical_shape()`, `sin.logical_shape()`, `trans_mat.logical_shape()` — replaced by `padded_shape()`

**Verdict: RESOLVED by fab067a** (was CAVEAT). The hash now keys each operand's whole `tensor_spec()`
(`compute_program_hash:262-275`), which includes `logical_shape`, so there is no longer any
logical-vs-padded relaxation in this op to disagree with the framework's exact-spec check. The two
calls in the reproduction below now compute *different* keys, miss, and compile a second program —
which is what the relaxation was implying should happen but could not deliver.

**Pre-fix:** correct as a relaxation of what the *program* depends on, but the framework's exact-spec
check turned any exercised difference into a hard failure rather than the recompile the relaxation
implied. The analysis below describes that pre-fix code.

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

The catch was that `cos.tensor_spec()` is what gets baked as the `COS_PARAM` `TensorParameter`
(`create_at:387`), and the spec check on every hit is *exact*. Two calls whose `cos` tensors shared a
padded shape, dtype and memory config but differed in logical shape produced the same hash, hit the
cache, and then threw from `report_tensor_arg_mismatch`. That was fail-safe — no corruption, and the
diagnostic names the binding — but the outcome was a crash on a call that should simply have compiled a
second program. See `### The report_tensor_arg_mismatch interaction` above for the full mechanism.

What would have broken it: a caller who trims the logical extent of a cos/sin cache without changing its
padded extent. Nothing in DeepSeek prefill does that today, since the cos/sin shards are allocated once
per model at a fixed shape.

Two fixes were available, and fab067a took the first. **(a)** Hash the logical shapes alongside the
padded ones so the key matches the strict predicate — cheap, since these are per-model constants that
will never actually diverge, and it converts a potential throw into a correct rebuild. The commit went
further and hashed the whole spec, which subsumes it. **(b)** Declare
`TensorSpecRelaxations::match_padded_shape_only` on those three tensor parameters to make the relaxation
explicit at the framework level
(`tt_metal/api/tt-metalium/experimental/metal2_host_api/tensor_spec_relaxations.hpp:41,49`); that flag
maps to `PertinentFields{.padded_shape = true}`
(`tt_metal/impl/metal2_host_api/tensor_spec_relaxations.cpp:67-87`), which both
`hash_tensorspec_with_relaxation` (`:116`) and `tensorspecs_match_with_relaxation` (`:161-201`) consume,
so key and validation would have agreed that way too. Option (a) was the right call for this op: nothing
in the factory benefits from the extra hits, so paying nothing for a strictly finer key is preferable to
opting into an unsafe-by-default flag. Its two `kv_cache` siblings, which *do* benefit, are the ones that
should carry the flag.

### 4. `cos.layout()`, `sin.layout()`, `trans_mat.layout()`

**Verdict: RESOLVED by fab067a** (was CAVEAT — pinned only on the miss path). Closed twice over: the
layouts are now *in the key* as part of each operand's hashed `page_config`
(`compute_program_hash:262-275`), and fab067a's `require_standard_tile` lambda re-asserts
`layout() == Layout::TILE` on all four operands from inside the shared `validate_runtime_args`
(`:92-96`, applied at `:106-109`) — the placement that runs on the hit path. A ROW_MAJOR `cos` on a hit
is now rejected by the op, by name, instead of relying on the framework spec comparison to notice.

**Pre-fix:** `input.layout()` was hashed but the other three operands' layouts were not. They were
constrained on the miss path only:

```157:160:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/rotary_embedding_indexed/device/rotary_embedding_indexed_device_operation.cpp
    TT_FATAL(input.layout() == Layout::TILE, "input must be TILE layout");
    TT_FATAL(cos.layout() == Layout::TILE, "cos must be TILE layout");
    TT_FATAL(sin.layout() == Layout::TILE, "sin must be TILE layout");
    TT_FATAL(trans_mat.layout() == Layout::TILE, "trans_mat must be TILE layout");
```

These live in `validate_on_program_cache_miss` and were *not* repeated in `validate_runtime_args`. Since
this op defines a hit validator, the miss validator is replaced rather than supplemented on a hit (see
`### Which validator runs on a cache hit`), so these four `TT_FATAL`s ran on the first call and never
again. Under the audit rule that a pin living only in the miss validator is at most a caveat, this
could not be graded `VALID — pinned by validation` even though the pin was real and the value is
single-valued in every admissible call.

What kept it safe pre-fix was not the pin but the framework: the `UpdateProgramRunArgs` spec check
catches a layout divergence independently, because `layout` is a projection of `page_config` and
`page_config` is part of the exact match. A ROW_MAJOR `cos` on a hit threw from the framework rather
than executing. That was a backstop rather than a pin, which is exactly the shape of a caveat — safe,
resting on a mechanism outside the op, and producing a throw where a rebuild was intended.

(This was a meaningful difference from the two `kv_cache` siblings, where the analogous cross-tensor
consistency checks are also miss-path-only but there is no framework backstop, so the same structure
degraded to silent corruption.)

**The guard fab067a landed.** The fix is the one this section named — repeat the unhashed
`Layout::TILE` checks in `validate_runtime_args`, which both paths call — and the commit did exactly
that, folding them into the same `require_standard_tile` lambda that carries the tile-geometry check of
omission 5:

```87:109:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/rotary_embedding_indexed/device/rotary_embedding_indexed_device_operation.cpp
    // Runs on cache hit as well as miss, which is the point: the factory bakes 32x32 tile arithmetic
    // into compile-time args, and page_config is hashed, so a differing tile misses -- but layout and
    // tile geometry must still be rejected by name rather than by a downstream spec mismatch.
    const auto require_standard_tile = [](const Tensor& tensor, std::string_view name) {
        TT_FATAL(tensor.layout() == Layout::TILE, "{} must be TILE layout", name);
        const auto tile = tensor.tensor_spec().tile();
        TT_FATAL(
            tile.get_height() == TILE_HEIGHT && tile.get_width() == TILE_WIDTH,
            "{} must use standard {}x{} tiles, got {}x{}",
            name,
            TILE_HEIGHT,
            TILE_WIDTH,
            tile.get_height(),
            tile.get_width());
    };
    require_standard_tile(input, "input");
    require_standard_tile(cos, "cos");
    require_standard_tile(sin, "sin");
    require_standard_tile(trans_mat, "trans_mat");
```

This document previously argued against that change on the grounds that it bought a clearer error
message rather than a correctness improvement. That reasoning was right about the *layout* half and
wrong about the *tile* half — the same lambda is what closes the miss-path factory defect in omission 5,
which was a genuine correctness gap — so the combined check earns its per-dispatch cost. See
recommendation 4.

### 5. `page_config` (the `Tile`) and `alignment` of all five tensors — the unguarded 32x32 assumption

**Verdict: RESOLVED by fab067a** (was CAVEAT, not BUG). Both halves are closed. `page_config` — hence
the `Tile` — is now in the key as part of each operand's hashed `tensor_spec()`
(`compute_program_hash:262-275`), and the miss-path factory defect described below is blocked outright
by `require_standard_tile` in the shared `validate_runtime_args` (`:87-109`), which asserts 32x32 tile
geometry on all four operands. A non-32x32 call is now rejected by operand name on the first call,
before any mis-sized dataflow buffer is ever built, and on every subsequent call too. `alignment` is
likewise now hashed as part of the spec.

**Pre-fix:** this op met two of the three criteria for the tile bug — it requires `Layout::TILE` and it
derives all of its tile geometry from the architectural 32x32 constants with no tile-geometry guard
anywhere in the directory — but the framework rescued it. Its Metal 2.0 dispatch path performs an exact
`TensorSpec` comparison on every hit, and that comparison provably covers `page_config`, so a differing
`Tile` threw a diagnostic rather than silently executing the wrong program. It failed loudly where a
rebuild was intended, which was a caveat, not corruption. The rest of this section describes that
pre-fix code; the factory itself is unchanged, and is now fenced by the validator instead.

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

Pre-fix, nothing validated the geometry: there was no `tensor_spec().tile()` read and no tile-geometry
`TT_FATAL` anywhere in the op directory. The validator did require `Layout::TILE` on four tensors
(`:157-160`) and did pin `trans_mat` to a single tile — but that check is a *shape* check against the
architectural constant, not a tile check, and under `Tile{16, 32}` a `[1, 1, 32, 32]` trans_mat is two
tiles, not one:

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
truncated trans_mat. That was a factory bug, and it was real — but it was not the program-cache bug
class, because a miss means the program was at least built for the tensor in front of it. This is the
half fab067a's `require_standard_tile` blocks outright; it is recorded under
`## Non-cache correctness defects` below rather than counted as a cache bug.

**Why the cache-hit exposure was a throw and not corruption.** The spec check quoted in
`## Cache-hit patch mechanism` above delegates to `tensorspecs_match_with_relaxation`, and the
`page_config` coverage was traced through the whole chain rather than assumed:

```178:184:tt_metal/impl/metal2_host_api/tensor_spec_relaxations.cpp
    const relaxation_fields::PertinentFields fields = relaxation_fields::pertinent_fields(relaxation);
    if (fields.whole_spec) {
        return a == b;
    }
    if (!(a.tensor_layout() == b.tensor_layout())) {
        return false;
    }
```

This op leaves `TensorParameter::relaxations` default-constructed, which `pertinent_fields` maps to
`PertinentFields{.whole_spec = true}` (`:67-87`), so it takes the `a == b` branch and compares whole
`TensorSpec`s. From there:

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

So `page_config` is genuinely covered, and it is covered under *every* relaxation — the `whole_spec`
branch compares the specs outright and every other path compares `tensor_layout()` unconditionally
(`:173-175`), so no `TensorSpecRelaxations` setting can relax the tile away. That is a stronger
guarantee than this op needs, and post-fix it is a second line rather than the only one.

One precise limit: `Tile::operator==` does not compare the `transpose_within_face` / `transpose_of_faces`
flags. Those escape the check — but they equally escape the framework's default hash, whose
`Tile::attribute_values()` is `(tile_shape, face_shape, num_faces)`
(`tt_metal/api/tt-metalium/tile.hpp:46-47`). They are therefore not an omission relative to the default,
and this factory never reads them.

**Two-call sequence (pre-fix), and how it differed from the siblings.** Call 1: `input`, `cos`, `sin`,
`trans_mat` all `BFLOAT16`, `Layout::TILE`, `Tile{32, 32}`, interleaved DRAM. Call 2: identical padded
shapes, dtypes and memory configs, but `Tile{16, 32}`. The pre-fix hash omitted `page_config` (it hashed
`dtype`, `memory_config`, `padded_shape` and — for `input` only — `logical_shape` and `layout`, never
the tile), so the key was identical and the cache hit — exactly as it does in the two `kv_cache` ops.
Post-fix, call 2 does not even reach the cache: `require_standard_tile` rejects it in
`validate_runtime_args`, and had the guard not been there the hashed `page_config` would have made it
miss. The divergence pre-fix was in what happened next. Here, `override_runtime_arguments`
calls `UpdateProgramRunArgs`, which validates before it patches
(`tt_metal/impl/metal2_host_api/program_run_args.cpp:1105-1107`, delegating to
`ValidateUpdateProgramRunArgs` and thence to `ValidateTensorArgs` at `:1087`), the `Tile{16, 32}` spec
fails the comparison above, and `report_tensor_arg_mismatch` throws a named diagnostic before any
kernel runs. In `update_padded_kv_cache` and `zero_padded_kv_cache` the same source-level mistake reaches
the descriptor buffer-binding fast path
(`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:726-731`), which patches addresses and dynamic scalars
and compares nothing, so the stale 32x32 compile-time args and CB page sizes are simply executed against
a 16-row-tile buffer and the KV cache is silently corrupted.

**This was the single most useful observation across the three documents.** Three sibling ops, written by
the same team against the same hardcoded 32x32 assumption, with the same `page_config` omission from the
hash. Two of them silently corrupted data; this one failed loudly. The entire difference was which
cache-hit mechanism the op happens to be built on — Metal 2.0 `ProgramSpec` with named tensor parameters
and an enforced spec contract, versus a `ProgramDescriptor` with raw buffer bindings and no contract at
all. The safety was not a property of this op's code; it was a property of the dispatch layer, and it
would have evaporated the moment someone set a relaxation or passed `skip_validation`. That is precisely
why fab067a did not leave it there: the guard now lives in the op, and the framework check is a backstop.

`alignment` deserves a separate note: it does not appear in this factory at all. The op reads and writes
through Metal 2.0 `TensorAccessor`s bound by name (`reader_...cpp:97-100`) rather than through
host-emitted `TensorAccessorArgs`, so no aligned page size is baked into a compile-time arg the way it is
in this family's two `kv_cache` ops. Its omission was `CAVEAT` purely for the throw-not-rebuild reason,
with no underlying factory defect behind it, and it is now hashed as part of the spec regardless.

**The guard, as landed.** The fix this section recommended is the pattern already established elsewhere
in the repo:

```94:98:ttnn/cpp/ttnn/operations/data_movement/sharded/interleaved_to_sharded/device/interleaved_to_sharded_op.cpp
    if (input_tensor.layout() == Layout::TILE) {
        auto tile = input_tensor.tensor_spec().tile();
        if (tile.get_height() != tt::constants::TILE_HEIGHT || tile.get_width() != tt::constants::TILE_WIDTH) {
            return {false, fmt::format("interleaved_to_sharded requires standard 32x32 tiles, got {}x{}", tile.get_height(), tile.get_width())};
        }
```

fab067a adopted it as `require_standard_tile` (`:87-109`, quoted under omission 4) and — critically —
placed it in the **shared** `validate_runtime_args` rather than in `validate_on_program_cache_miss`.
Placing it in the miss validator would have blocked the factory defect but left the hit path uncovered;
placing it in the shared function covers both. Making the factory genuinely tile-aware instead remains
the alternative, and would now be safe to attempt, since `page_config` is in the key and the program may
provably vary with `Tile`.

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

**Verdict: VALID — pinned by the framework** (was CAVEAT — pinned only on the miss path). This is a
correction to the pre-fix grade, not a change fab067a made: device storage was never a reachable
omission on any path. See `#### Framework correction` below. fab067a additionally added a `cos` storage
pin to `validate_runtime_args` (`:72-75`) for the one operand whose `device()` is dereferenced on the
next line, so the op no longer relies solely on the framework for that operand.

```142:145:ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/rotary_embedding_indexed/device/rotary_embedding_indexed_device_operation.cpp
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "input must be on device");
    TT_FATAL(cos.storage_type() == StorageType::DEVICE, "cos must be on device");
    TT_FATAL(sin.storage_type() == StorageType::DEVICE, "sin must be on device");
    TT_FATAL(trans_mat.storage_type() == StorageType::DEVICE, "trans_mat must be on device");
```

Constant across every admissible call, so on the miss path it carries no information. All four sit
above the `validate_runtime_args` delegation at `:184`, and this op's hit validator replaces the miss
validator rather than supplementing it, so none of them re-run on a hit. Unlike omission 4, the
framework spec check is *no* backstop: `TensorSpec` covers `logical_shape` and `tensor_layout` and says
nothing about the storage variant, so a host-storage tensor is not rejected by the comparison. That is
what drove the pre-fix `CAVEAT` grade, and it was wrong — the backstop is elsewhere.

#### Framework correction

`launch()` asserts device storage on **every** tensor argument, on **every** dispatch, before any
op-specific validator or hash runs:

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
`std::optional<Tensor>`, so the engaged `metadata` tensor is covered too. And it is unconditional — no
op can opt out.

So the storage variant kind cannot vary along any reachable path, which makes it a zero-value omission
rather than a caveat: there is no admissible call in which it differs, on either path, and the op's own
four `TT_FATAL`s are redundant with a framework guarantee rather than the only thing enforcing it. The
same correction applies to the `buffer() != nullptr` rows in the reachability table above, since
`is_device_tensor` implies a device allocation.

This is why the document no longer recommends repeating the four checks in `validate_runtime_args`:
they would cost four `storage_type()` queries on every dispatch and buy nothing at all, not even a
better message than the framework's. fab067a's one addition here (`cos`, at `:72-75`) is justified on
different grounds — it guards a `device()` dereference on the immediately following line, so it is
local defensive coding rather than a cache guard.

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

- `tensor_args.metadata.has_value()` — separates the two program variants; the reader's
  `HAS_METADATA` define, its `META_DFB` dataflow buffer, its `METADATA_PARAM` binding and its common-arg
  schema all switch on it (`create_at:376-382,392-395,402-404,420-433`), and all of those are frozen on
  a hit, so hashing it is mandatory rather than optional.

Post-fix that is the whole list. Pre-fix the hash also promoted the four operands' `padded_shape()` and
`metadata->padded_shape()` to first-class terms, and kept `input.layout()` as a projection of
`page_config`, in order to justify dropping the corresponding `logical_shape`s. Hashing each
`tensor_spec()` wholesale makes all of those redundant: `padded_shape` and `layout` are derivations of
the spec, so they are covered without being named, and there is nothing left to drop.

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

## Non-cache correctness defects

Recorded separately so they are not counted as program-cache bugs. Both entries concern the factory,
not the key.

| Defect | Status | Note |
|---|---|---|
| Miss-path 32x32 factory defect: five `tt::tile_size` calls (`create_at:268-276`) size every dataflow buffer for a 32x32 tile, bare `TILE_HEIGHT`/`TILE_WIDTH` do all tile-count arithmetic (`:277-283`), `tile_height` is baked as a reader compile-time arg fixed to `TILE_HEIGHT` (`:448-449`), and the `trans_mat` single-tile check (`:168-176`) is a shape check against those constants rather than a tile check. On a cache **miss** a `Tile{16, 32}` call would have compiled mis-sized dataflow buffers and silently truncated `trans_mat` to its first tile. | **RESOLVED by fab067a** | `require_standard_tile` in `validate_runtime_args` (`:87-109`) rejects the call before `create_at` runs. The factory is still 32x32-only; it is now fenced rather than fixed, which is the correct order of operations — the fence is one lambda, making the factory tile-aware is a rewrite. |
| The 32x32 assumption is undocumented in the public API, so a caller had no way to know the constraint existed before hitting it. | **RESOLVED by fab067a** | Stated in the nanobind docstring (`rotary_embedding_indexed_nanobind.cpp:45-47`). |

No hard-coded kernel-handle indices here: this op's override rebinds tensors by parameter name through
`run_args.tensor_args` (`:624-631`) rather than by position, so it has none of the handle-coupling
defect recorded in its two `kv_cache` siblings' documents.

## Summary

| Omitted vs. default | Used by program? | Patched on hit? | Verdict |
|---|---|---|---|
| `attrs.kv_actual_global` | Yes (reader common arg) on the scalar path; no on the metadata path | Yes (override) / n/a | VALID — patched / VALID — unused |
| `metadata.dtype` | Only via the on-device read width | No | NO LONGER OMITTED — hashed via `metadata->tensor_spec()` (was VALID — pinned by validation) |
| `cos`/`sin`/`trans_mat` `logical_shape` | No (padded shapes used) | n/a | RESOLVED by fab067a — whole `tensor_spec()` hashed, so the relaxation is gone (was CAVEAT) |
| `cos`/`sin`/`trans_mat` `layout` | No | n/a | RESOLVED by fab067a — hashed via `page_config` **and** pinned in `validate_runtime_args` (was CAVEAT — pinned only on the miss path) |
| `page_config` (`Tile`) of all five tensors | Yes (`tile_height` compile-time arg, DFB sizes, tile counts — all hardcoded 32x32) | No — call now rejected by the op | RESOLVED by fab067a — hashed, and 32x32 asserted in `validate_runtime_args` (was CAVEAT) |
| `alignment` of all five tensors | No (no host-emitted `TensorAccessorArgs`) | n/a | RESOLVED by fab067a — hashed as part of the spec (was CAVEAT) |
| Per-core work-split args | Yes (per-core RTAs, `RELOAD_IMPL`) | No (deliberately) | VALID — invariant (function of hashed specs) |
| Operand `storage` kinds | n/a | n/a | VALID — pinned by the framework at `device_operation.hpp:500-501` (was CAVEAT — pinned only on the miss path; that grade was wrong) |
| All buffer addresses (incl. the fresh output) | Yes | Yes (`UpdateProgramRunArgs` tensor bindings) | VALID — patched, required |
| `my_sp_coord`, `sp_factor` | Yes (compile-time args) | n/a (coordinate hashed) | VALID — invariant |

**No program-cache correctness bug was found, pre-fix or post-fix; post-fix, no caveat remains either.**
The single omitted attribute is now the per-chunk position scalar, and it is handled the way this class
of value should be: kept out of the key, re-applied to the reader's common runtime argument on every
hit, consumed only on-device so nothing derived from it can be baked, and re-validated on every hit by a
host-side check that reproduces the kernel's own `update_idxt` arithmetic. Every compile-time argument,
dataflow-buffer size, kernel define and per-core runtime argument in `create_at` is a function of the
hashed set plus device constants the per-device cache already partitions on. The five pre-fix
tensor-spec caveats (omissions 2-5) are closed at the source: the hash keys each operand's whole
`tensor_spec()`, so it no longer relaxes anything the framework then requires exactly.

That last point is the substance of the change and is worth restating as a rule. Pre-fix the op's
safety rested entirely on `UpdateProgramRunArgs` enforcing exact `TensorSpec` equality against the baked
`TensorParameter`s — which it does, reaching all the way down to `Tile::operator==` — so a spec omission
surfaced as a hard rejection rather than corruption. That is fail-safe but it is not correct: the
residual cost was a throw instead of a recompile, and the safety was a property of the dispatch layer
rather than of the op. In `update_padded_kv_cache` and `zero_padded_kv_cache`, built on a
`ProgramDescriptor` with raw buffer bindings and no spec contract, the same source-level omissions were
BUGs that silently corrupted the KV cache. fab067a removes the dependence on that accident: the key and
the framework predicate now agree by construction (see
`### The report_tensor_arg_mismatch interaction`), and the 32x32 assumption is fenced by the op's own
validator rather than by a framework comparison that a future `TensorSpecRelaxations` or
`skip_validation` could switch off.

A separate, lower-severity structural finding remains, and it is the one thing here that is not fixed:
because the op defines `validate_on_program_cache_hit`, that validator *replaces* the miss validator on
hits rather than supplementing it, and this op's hit validator delegates to `validate_runtime_args` and
does nothing else. Everything the miss validator checks before its own delegation at `:184` is absent
on the hit path — the four `storage_type() == DEVICE` pins, the `buffer() != nullptr` and same-device
checks, the four `Layout::TILE` pins, the `trans_mat` single-tile check, and the `cos`/`sin` dtype and
shape equalities. A narrow hit validator is a hazard rather than a safeguard in general: by existing, it
disables everything above it.

Filtered for reachability, the practical loss is now near zero. Most of the dropped checks constrain
specs that are in the cache key, and a miss-only pin on a hashed value cannot be evaded — any call
carrying a new value of it misses and meets the pin there. The storage and allocation rows were never
reachable, because `launch()` asserts both on every dispatch
(`ttnn/api/ttnn/device_operation.hpp:500-501`). The layout pins are now both hashed and re-asserted in
the shared function. What survives the filter is the three `device() == input.device()` comparisons
(`:153-155`), which fault when `UpdateProgramRunArgs` cannot resolve a buffer against the wrong mesh
device. That fails loudly on the offending call.

So the recommendations' original disposition — leave the regraded rows as caveats, add only the tile
guard — was the right call, and fab067a implemented exactly that plus the hash widening. The general
principle holds for the ports: `validate_runtime_args` runs on the cache-hit path, so every check moved
into it is paid on every dispatch for the life of the process, and it is worth paying only where the
failure it catches would otherwise be *silent*.

Two family-level observations, both now historical for this op and live for its siblings. First, all
three `deepseek_prefill` ops audited here (this one, `update_padded_kv_cache`, `zero_padded_kv_cache`)
share the same correct core idiom: the moving per-request index is omitted from the hash, patched on
every hit, and re-checked by a validator that runs on both the miss and hit paths — with a dual "scalar
or 1-element device tensor" path selected by a hashed `has_value()` bit. They also shared the same 32x32
defect, and the divergence in its consequences was purely a dispatch-mechanism artifact; fab067a fenced
all three, in each case by putting the guard in the shared `validate_runtime_args` rather than in
`validate_on_program_cache_miss`, which for the two `kv_cache` ops is the difference between fixing the
bug and not, since their reproductions are cache hits. All three also define a narrow hit validator that
delegates to a shared `validate_runtime_args` and thereby drops the rest of their miss-time pins on the
hit path; this op's shared function is the most complete of the three. Second, this op is the only one
of the three that hashes its metadata tensor's spec at all; the other two hash only the engagement bit
while baking that tensor's `TensorAccessorArgs` into kernel compile-time args, and close the resulting
gap with `TT_FATAL`s instead. That works, but the pattern implemented here — hash the spec, leave the
relaxations default, let key and validation agree by construction — is the one they should adopt when
they are ported to Metal 2.0.

## Recommendations

**Status: recommendations 1, 2 and 3 landed in fab067a; 4 and 5 stand as non-recommendations.** Each
item below carries its own status line. The section is left in place because the reasoning is what
transfers to the two siblings' ports, not because anything here is outstanding.

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
   `TensorParameter`s. Either makes omission 3 explicit; pre-fix the hash relaxed what the framework
   then required exactly, so the "relaxation" could only ever surface as a crash.
   **Status: DONE in fab067a**, and more than asked — the hash now keys the whole `tensor_spec()`
   (`:262-275`), which subsumes the logical shapes and leaves the relaxations correctly default.
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
   **Status: DONE in fab067a** — landed as `require_standard_tile` (`:87-109`), in
   `validate_runtime_args` as specified, on all four operands, with the `Layout::TILE` checks of
   omission 4 folded into the same lambda. The same guard landed in both siblings.
3. Correct two stale comments, each of which is the stated safety argument for the function it sits on,
   and each of which claimed more than the code delivered. An inaccurate safety comment is worse
   than none, because it talks the next reader out of checking.
   - `compute_program_hash` (`:210-214`) stated the hash covered "the full input, cos, sin and trans_mat
     specs". It covered projections of them, and that gap was exactly omissions 3-5.
   - `validate_on_program_cache_hit` (`:189-190`) stated that "structural constraints are hashed and so
     guaranteed unchanged here". True of the shapes and dtypes it had in mind; false of the four
     `Layout::TILE` pins and the four `storage_type()` pins, which were neither hashed nor re-checked —
     omissions 4 and 7.

   **Status: DONE in fab067a.** Both were rewritten (`:245-261` and `:224-227`), and the second now
   states the contract positively: anything that must hold on a hit belongs inside
   `validate_runtime_args`. That is also the mitigation recommendation 5 asks for, so it is covered.
4. **Do not move the four `storage_type()` pins onto the hit path.** This is a deliberate
   non-recommendation, recorded so it is not mistaken for an oversight.

   **Status: STANDS, and strengthened.** As originally written this item also covered the three
   unhashed `Layout::TILE` checks, on the grounds that the framework spec comparison already rejected a
   layout divergence and moving the pins would buy only a better message. fab067a moved them anyway —
   correctly, because they ride along in the tile lambda of recommendation 2 at no extra dispatch cost
   beyond three `layout()` queries, and because relying on a framework comparison a relaxation could
   switch off was the weak part of the argument. The storage half of this item is now on firmer ground
   than when it was written: the pins are not merely low-value, they are **provably** zero-value, since
   `launch()` asserts `is_device_tensor` on every tensor argument on every dispatch
   (`ttnn/api/ttnn/device_operation.hpp:500-501`) — see `#### Framework correction` under omission 7.
   Moving them into `validate_runtime_args` would charge four `storage_type()` queries per dispatch,
   forever, to re-check something the framework has already made impossible. fab067a's one addition
   (`cos`, `:72-75`) is not this: it guards a `device()` dereference on the next line.

   The general judgement stands: buy back a miss-only pin on the hot path only when the failure it
   catches would otherwise be *silent*. That is exactly the distinction that made the tile guard worth
   paying for and the storage pins not.

5. **Do not delete `validate_on_program_cache_hit` to fix this. Status: STANDS.** Deleting it would put the op on the
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

   The one thing the current arrangement genuinely costs is fragility — nothing stops the next person
   adding a load-bearing check above the delegation and not noticing it never runs. The cheap mitigation
   is a comment stating that everything above the `validate_runtime_args` call at `:184` is miss-only by
   design, and that any check which must hold on a hit belongs inside `validate_runtime_args`.
   **fab067a landed that mitigation** on the hit validator (`:224-227`), which is where a reader looking
   for the contract will actually be standing.
