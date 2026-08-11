# Program Cache Audit — `generic`

Audit of `ttnn::operations::generic::GenericOpDeviceOperation::compute_program_hash` against the
framework default ("hash everything") key.

| | |
|---|---|
| Device operation | `ttnn::operations::generic::GenericOpDeviceOperation` (`device/generic_op_device_operation.hpp`) |
| Custom hash | `device/generic_op_device_operation.cpp:115` (delegates to `compute_program_descriptor_hash`, same file line 48) |
| `operation_attributes_t` | `tt::tt_metal::experimental::MeshProgramDescriptor` — one field, `mesh_programs` (`vector<pair<MeshCoordinateRange, ProgramDescriptor>>`) |
| `tensor_args_t` | `io_tensors` (`const std::vector<Tensor>&`), `output_tensor` (`const Tensor&`, aliases `io_tensors.back()`) |
| Program factories | `program::GenericMeshDescriptorFactory` (`ProgramDescriptor`-based, per-coordinate) |
| `override_runtime_arguments` | **No** |
| `get_dynamic_runtime_args` | **No** |
| `validate_on_program_cache_hit` | **Yes**, but narrow — see below |
| Cache-hit patch mechanism | Framework **slow-path descriptor rebuild** |

## Validation on the cache-hit path

This op does define a hit validator, and it runs the same check as the miss validator:

```26:34:ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation.cpp
void GenericOpDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& /*tensor_args*/) {
    verify_no_duplicate_mesh_coord_ranges(attributes.mesh_programs);
}

void GenericOpDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& /*tensor_args*/) {
    verify_no_duplicate_mesh_coord_ranges(attributes.mesh_programs);
}
```

Because this op *defines* a hit validator, the dispatcher takes the first branch and the miss
validator does not run on hits at all:

```262:266:ttnn/api/ttnn/device_operation.hpp
    if constexpr (HasValidateOnProgramCacheHit<mesh_device_operation_t>) {
        mesh_device_operation_t::validate_on_program_cache_hit(operation_attributes, tensor_args);
    } else {
        mesh_device_operation_t::validate_on_program_cache_miss(operation_attributes, tensor_args);
    }
```

Here that is harmless only because the two functions are byte-identical — each is the single call
`verify_no_duplicate_mesh_coord_ranges(attributes.mesh_programs)`, each ignores `tensor_args`, and
neither has any other statement — so **the hit path drops nothing**.

Because the dropped set is empty, the hit-path reachability filter has nothing to filter. That
analysis — asking of each dropped check whether the value it constrains is itself in the cache key,
and so whether a call violating it could ever reach a hit rather than being rejected on its first,
missing occurrence — applies only where the hit validator pins strictly less than the miss validator.
No verdict below is affected by it, and none rests on the two validators differing.

Note that the identity is load-bearing rather than incidental: the hit validator *replaces* the miss
validator, so if anyone later adds a check to `validate_on_program_cache_miss` without mirroring it
into `validate_on_program_cache_hit`, that check will silently not run on hits. A hand-written hit
validator is a hazard in that direction, not a safeguard.

The problem here is scope, not placement: the only thing checked is that no two
`MeshCoordinateRange`s in `mesh_programs` collide, and `tensor_args` is explicitly ignored on both
paths. So none of the omissions catalogued below is constrained by validation, and no verdict in this
document rests on it. That is why this op has no `VALID — pinned by validation` rows, in contrast
with most other ops in this audit.

## Cache-hit patch mechanism

This op is a `ProgramDescriptor`-variant factory with neither `override_runtime_arguments` nor
`get_dynamic_runtime_args`, so `DescriptorMeshWorkloadAdapter::apply_descriptor` reaches the final
`else` branch and chooses between the buffer-binding fast path and a full descriptor rebuild:

```726:731:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
                    if (!sv.resolved_bindings.rt_args.empty() ||
                        (!dynamic_args.empty() && !sv.resolved_bindings.empty())) {
                        auto collected =
                            collect_tensor_buffers(tensor_args, tensor_return_value, sv.workload_descriptor);
                        tt::tt_metal::apply_resolved_bindings(program, sv.resolved_bindings, collected.buffers);
                        tt::tt_metal::apply_dynamic_runtime_args(program, dynamic_args);
```

`dynamic_args` is always empty here, so the branch turns entirely on `sv.resolved_bindings.rt_args`.
**For this op that vector is always empty**, and the reason is structural rather than accidental in
effect but accidental in cause. `tensor_args_t` carries the output tensor twice — once as the last
element of `io_tensors` and once as `output_tensor`:

```16:20:ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation_types.hpp
// NOTE: output tensor is the last element in the vector io_tensors
struct tensor_args_t {
    const std::vector<Tensor>& io_tensors;
    const Tensor& output_tensor;
};
```

```137:137:ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation.cpp
    auto tensor_args = OperationType::tensor_args_t{.io_tensors = io_tensors, .output_tensor = io_tensors.back()};
```

`collect_tensor_buffers` walks all of `tensor_args` before the boundary, so both copies land in the
*input* region (`collected.num_input_buffers = buffers.size()` after the `tensor_args` walk):

```408:421:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
        static CollectedTensorBuffers collect_tensor_buffers(
            const tensor_args_t& tensor_args,
            const tensor_return_value_t& tensor_return_value,
            const tt::tt_metal::WorkloadDescriptor& workload_descriptor) {
            CollectedTensorBuffers collected;
            auto& buffers = collected.buffers;
            extract_tensor_buffers_into(tensor_args, buffers);
            collected.num_input_buffers = buffers.size();
            extract_tensor_buffers_into(tensor_return_value, buffers);
            for (const auto& wb : workload_descriptor.buffers) {
                buffers.push_back(wb.buffer);
            }
            return collected;
        }
```

The adapter resolves bindings with `allow_inplace_output_tensor_alias` left at its `false` default
(`resolve_bindings(program, desc, collected.buffers, collected.num_input_buffers)`):

```587:592:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
                            auto collected = collect_tensor_buffers(tensor_args, tensor_return_value, empty_descriptor);
                            auto bindings = tt::tt_metal::resolve_bindings(
                                program, desc, collected.buffers, collected.num_input_buffers);
                            mesh_workload.add_program(device_range, std::move(program));
                            shared_variables[device_range] =
                                shared_variables_t{.resolved_bindings = std::move(bindings)};
```

so the duplicate output buffer inside the input region trips the ambiguity bail before any binding is
examined:

```103:108:tt_metal/impl/program/program_descriptor_patching.cpp
            // Otherwise a repeat is ambiguous (matmul(X, X), or a repeated output) — bail to slow path.
            auto& seen = is_input ? input_buffers : output_buffers;
            if (!seen.insert(buf).second) {
                return ResolvedBindings{};
            }
        }
    }
```

Every `generic_op` call therefore lands on the slow path:

```748:753:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
                    } else {
                        const ttnn::MeshCoordinate mesh_coord = coordinate_range.start_coord();
                        const std::optional<ttnn::MeshCoordinate> mesh_dispatch_coordinate(mesh_coord);
                        auto desc = invoke_per_coord(attrs, tensor_args, tensor_return_value, mesh_dispatch_coordinate);
                        tt::tt_metal::apply_descriptor_runtime_args(program, desc);
                    }
```

`invoke_per_coord` calls `GenericMeshDescriptorFactory::create_descriptor`, which returns the
caller's *current* `ProgramDescriptor` verbatim:

```12:29:ttnn/cpp/ttnn/operations/generic/device/generic_op_program_factory.cpp
tt::tt_metal::ProgramDescriptor GenericMeshDescriptorFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& /*tensor_return_value*/,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    const auto& mesh_programs = operation_attributes.mesh_programs;
    TT_FATAL(!mesh_programs.empty(), "generic_op: MeshProgramDescriptor.mesh_programs must not be empty");

    if (mesh_dispatch_coordinate.has_value()) {
        const auto& coord = mesh_dispatch_coordinate.value();
        for (const auto& [range, desc] : mesh_programs) {
            if (range.contains(coord)) {
                return desc;
            }
        }
        // No mesh_program covers this coordinate. Return an empty descriptor
        return tt::tt_metal::ProgramDescriptor{};
    }
```

and `apply_descriptor_runtime_args` copies **every** per-core runtime arg, every common runtime arg,
the Blaze named-arg values, and every tensor/buffer-backed CB address from that fresh descriptor into
the cached `Program`:

```184:192:tt_metal/impl/program/program_descriptors.cpp
void apply_descriptor_runtime_args(Program& program, const ProgramDescriptor& desc) {
    for (uint32_t k = 0; k < desc.kernels.size(); ++k) {
        const auto& kernel = desc.kernels[k];
        for (const auto& [core, args] : kernel.runtime_args) {
            auto& prog_args = GetRuntimeArgs(program, k, core);
            for (uint32_t i = 0; i < static_cast<uint32_t>(args.size()); ++i) {
                prog_args[i] = args[i];
            }
        }
```

**Obligation on the hash.** Because runtime-arg values and CB addresses are re-applied from the
caller's live descriptor on every hit, the hash only has to cover what is *baked into the `Program`
at construction*: kernel sources and their JIT build inputs, compile-time args, defines, core ranges,
CB structure (sizes, formats, page sizes, tile/face geometry), and semaphores. Anything in that set
that the hash misses is a genuine wrong-program bug; anything outside it that the hash misses is
correct and intentional.

## Baseline: what the default hash would cover

`hash_objects_with_default_seed(type_hash<GenericOpDeviceOperation>, attrs, tensor_args)` would give:

| Source | Fields |
|---|---|
| `operation_attributes` | **only `mesh_programs.size()`** — `MeshProgramDescriptor` hand-writes an attribute tuple that reduces the whole descriptor set to a count |
| `io_tensors[i]` | storage variant kind, `logical_shape`, and `tensor_layout` = { `dtype`, `page_config`, `memory_config`, `alignment` } for every tensor |
| `output_tensor` | the same set again (it aliases `io_tensors.back()`) |

The attributes side of the default is degenerate:

```15:22:tt_metal/api/tt-metalium/experimental/mesh_program_descriptor.hpp
struct MeshProgramDescriptor {
    using MeshPrograms = std::vector<std::pair<distributed::MeshCoordinateRange, ProgramDescriptor>>;
    MeshPrograms mesh_programs;

    // ProgramDescriptor too large for reflection inline storage.
    static constexpr auto attribute_names = std::forward_as_tuple("num_mesh_programs");
    auto attribute_values() const { return std::make_tuple(mesh_programs.size()); }
};
```

So the default key would distinguish two `generic_op` calls only by tensor specs and the *number* of
mesh programs — every kernel source, compile-time arg and CB in the descriptor would be invisible.
The custom hash is not a relaxation of the default here; on the attributes axis it is a large
strengthening, and on the tensor axis it is a total removal.

## What the custom hash covers

```115:123:ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation.cpp
ttsl::hash::hash_t GenericOpDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
    size_t hash = 0;
    for (const auto& [mesh_coord_range, program_descriptor] : operation_attributes.mesh_programs) {
        ttsl::hash::hash_combine(hash, mesh_coord_range);
        ttsl::hash::hash_combine(hash, compute_program_descriptor_hash(program_descriptor));
    }
    return hash;
}
```

with the per-descriptor walk:

```48:105:ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation.cpp
ttsl::hash::hash_t compute_program_descriptor_hash(const tt::tt_metal::ProgramDescriptor& program_descriptor) {
    if (program_descriptor.custom_program_hash) {
        return *program_descriptor.custom_program_hash;
    }

    auto hash_kernel = [&](const KernelDescriptor& kernel) -> size_t {
        return ttsl::hash::hash_objects_with_default_seed(
            kernel.kernel_source,
            kernel.source_type,
            kernel.core_ranges,
            kernel.compile_time_args,
            kernel.named_compile_time_args,
            kernel.defines,
            kernel.common_runtime_args.size(),
            // ... blaze named-arg schema ...
            kernel.runtime_args.size(),
            kernel.config.index(),
            kernel.config);
    };
    // ... hash_cb_format_descriptor / hash_circular_buffer / hash_semaphore ...
    size_t hash = 0;
    for (const auto& kernel : program_descriptor.kernels) {
        ttsl::hash::hash_combine(hash, hash_kernel(kernel));
    }
```

## Omitted parameters

### 1. All of `tensor_args_t` — `io_tensors` and `output_tensor`

**Verdict: VALID — invariant.**

The signature discards `tensor_args` entirely (`const tensor_args_t& /*tensor_args*/`, line 116).
That is correct for this op because *no tensor property reaches the program through the framework*.
`create_descriptor` ignores `tensor_args` and `tensor_return_value` (both commented out in its
parameter list, `generic_op_program_factory.cpp:14-15`) and returns the caller's descriptor. Anything
a tensor contributes — an address, a page size, an accessor compile-time arg — is something the
*caller* already baked into the `ProgramDescriptor` it handed in, and every such value is either
covered by the descriptor hash (compile-time args, CB page sizes, core ranges) or re-applied on every
dispatch from the caller's fresh descriptor (runtime args, CB addresses).

Concretely: two calls with the same descriptor but different input tensors share a cache entry, and
that is right — the descriptor *is* the program. If the caller wants a different program for a
different tensor shape, it must build a different descriptor, and then the descriptor hash separates
them. Note also that `compute_output_specs` and `create_output_tensors` return
`tensor_args.output_tensor` unchanged on every call (lines 36-46), so no output metadata is frozen at
first miss.

The class-level comment states this contract explicitly:

```19:22:ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation.hpp
struct GenericOpDeviceOperation {
    // This op never derives an address from a tensor: create_descriptor returns the caller's
    // ProgramDescriptor verbatim, so resolving per-core addresses is the caller's job
    static constexpr bool supports_per_core_allocation = true;
```

**`supports_per_core_allocation = true` does matter, and it is consistent.** That flag makes
`launch()` accept a per-core-allocated tensor, which no other op may take, because a per-core buffer
has a different L1 address on every core and `Buffer::address()` only reports the first core's
(`ttnn/api/ttnn/operation_concepts.hpp:232-256`). The opt-in is safe here *only* because this op
never resolves an address itself: the caller writes each core's address into the descriptor's
per-core runtime args, and the slow path re-copies those exact per-core values on every hit. Had this
op landed on the buffer-binding fast path instead, `apply_resolved_bindings` would write the same
`current_buffers[idx]->address()` to every bound core
(`tt_metal/impl/program/program_descriptor_patching.cpp:262`) and the per-core opt-in would be
actively wrong. The flag and the slow-path mode are load-bearing for each other.

### 2. Runtime-arg VALUES — only `runtime_args.size()` and `common_runtime_args.size()` are hashed

**Verdict: VALID — patched.** (See omission 6 for the part of this that is *not* safe.)

This is the central design decision and it is correct under mode C. The values are re-applied from
the caller's live descriptor by `apply_descriptor_runtime_args` (quoted above,
`tt_metal/impl/program/program_descriptors.cpp:184-219`), including the Blaze named-arg values, which
that function re-applies for exactly this reason (lines 206-217). Hashing the values would defeat the
cache entirely for the intended use — a Python model driver dispatching the same fused kernel every
step with new addresses and new counts.

### 3. CB buffer addresses — only `cb.buffer != nullptr` is hashed

**Verdict: VALID — patched.**

Addresses must not be in a cache key. The nullness *is* hashed because it changes the CB's kind
(globally allocated vs. locally allocated), which is baked into the `Program`. The address itself is
re-applied on every hit:

```221:233:tt_metal/impl/program/program_descriptors.cpp
    auto program_cbs = program.circular_buffers();
    for (uint32_t ci = 0; ci < static_cast<uint32_t>(desc.cbs.size()); ++ci) {
        const auto& cb_desc = desc.cbs[ci];
        TT_FATAL(
            !(cb_desc.buffer && cb_desc.tensor),
            "CBDescriptor cannot specify both buffer and tensor as the globally-allocated backing storage");
        if (cb_desc.tensor) {
            Buffer* buf = cb_desc.tensor->mesh_buffer().get_reference_buffer();
            UpdateDynamicCircularBufferAddress(program, program_cbs[ci]->id(), *buf, cb_desc.address_offset);
        } else if (cb_desc.buffer) {
            UpdateDynamicCircularBufferAddress(program, program_cbs[ci]->id(), *cb_desc.buffer, cb_desc.address_offset);
        }
    }
```

`cb.address_offset` rides along on the same call, so it is likewise patched.

### 4. `kernel.opt_level` and `kernel.compiler_include_paths`

**Verdict: BUG.**

Both feed the JIT build and are therefore baked into the compiled kernel binary held by the cached
`Program`. Nothing on the cache-hit path recompiles:

```410:421:tt_metal/impl/program/program.cpp
        std::vector<std::filesystem::path> compiler_include_paths(
            kernel_descriptor.compiler_include_paths.begin(), kernel_descriptor.compiler_include_paths.end());

        auto config = std::visit(
            ttsl::overloaded{
                [&](const ReaderConfigDescriptor&) -> std::variant<DataMovementConfig, ComputeConfig> {
                    return ReaderDataMovementConfig{
                        std::move(compile_args),
                        std::move(defines),
                        std::move(named_compile_args),
                        kernel_descriptor.opt_level.value_or(KernelBuildOptLevel::O2),
                        std::move(compiler_include_paths)};
                },
```

Neither field appears in `hash_kernel` above. This is a drift from tt-metal's own descriptor hasher,
which does cover them:

```132:144:tt_metal/impl/program/program_descriptors.cpp
static inline ttsl::hash::hash_t hash_kernel_descriptor(const KernelDescriptor& kernel) {
    return ttsl::hash::hash_objects_with_default_seed(
        kernel.kernel_source,
        kernel.source_type,
        kernel.core_ranges,
        kernel.compile_time_args,
        kernel.named_compile_time_args,
        kernel.defines,
        kernel.opt_level.has_value(),
        kernel.opt_level.value_or(KernelBuildOptLevel{}),
        kernel.compiler_include_paths,
        kernel.common_runtime_args.size(),
        kernel.runtime_args.size(),
```

Both fields are settable from Python (`ttnn-nanobind/program_descriptors.cpp:758` and `:964-967`),
so this is reachable from the public API.

**Reproduction.**
- Call 1: `ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=[ttnn.KernelDescriptor(kernel_source=K, core_ranges=R, compile_time_args=A, config=cfg, opt_level=ttnn.KernelBuildOptLevel.O2)], ...))`
- Call 2: identical descriptor except `opt_level=ttnn.KernelBuildOptLevel.Os` (or an added
  `compiler_include_paths=["my/headers"]` that resolves a differently-defined header).
- Both hash identically, call 2 is a cache hit.
- Stale artifact: the compiled RISC-V binary from call 1. Call 2 silently executes the `O2` build,
  or executes code compiled against the *first* call's include path.
- Symptom: for `opt_level`, a silent performance regression or, for `Os`/`O0`-sensitive kernels, a
  timing-dependent hang. For `compiler_include_paths`, wrong constants compiled into the kernel and
  therefore wrong numerical results with no error.

### 5. `CBFormatDescriptor::face_geometry`

**Verdict: BUG.**

`hash_cb_format_descriptor` hashes `buffer_index`, `data_format`, `page_size` and `tile`, but not
`face_geometry` (`generic_op_device_operation.cpp:71-77`). Face geometry is not cosmetic — it is
consumed at CB construction:

```115:119:tt_metal/impl/buffers/circular_buffer_config.cpp
        if (format_descriptor.face_geometry) {
            const auto& [face_r_dim, num_faces] = *format_descriptor.face_geometry;
            validate_unpack_face_geometry(face_r_dim, num_faces);
            this->unpack_face_geometry_[format_descriptor.buffer_index] = format_descriptor.face_geometry;
        }
```

and then flows into the *compile-time* JIT build options for every kernel on those cores:

```1977:1992:tt_metal/impl/program/program.cpp
void detail::ProgramImpl::set_cb_data_fmt_and_tile(
    const std::vector<CoreRange>& crs, JitBuildOptions& build_options) const {
    TTZoneScopedD(PROGRAM);
    for (const auto& logical_cr : crs) {
        const auto& cbs_on_core = this->circular_buffers_on_corerange(logical_cr);
        for (const auto& circular_buffer : cbs_on_core) {
            for (auto buffer_index : circular_buffer->buffer_indices()) {
                const CBIndex cb_index = static_cast<CBIndex>(buffer_index);
                const DataFormat data_format = circular_buffer->data_format(buffer_index);
                const auto& tile_opt = circular_buffer->tile(buffer_index);
                const auto& unpack_geom = circular_buffer->unpack_face_geometry(buffer_index);
                build_options.set_cb_data_fmt_tile_and_face_geometry(cb_index, data_format, tile_opt, unpack_geom);
            }
        }
    }
}
```

The tt-metal hasher covers it (`program_descriptors.cpp:152-159`); this fork does not. It is exposed
to Python (`ttnn-nanobind/program_descriptors.cpp:393-396`).

**Reproduction.**
- Call 1: descriptor with `CBFormatDescriptor(buffer_index=0, data_format=..., page_size=P, tile=T)`
  and `face_geometry = FaceGeometry{16, 4}`.
- Call 2: identical except `face_geometry = FaceGeometry{1, 4}` (a row-vector unpack layout).
- Same hash, cache hit.
- Stale artifact: the unpacker configuration compiled into the compute kernel for CB 0.
- Symptom: the compute kernel unpacks 16 rows per face when the caller asked for 1 — silent garbage
  in the destination registers, no error.

This is the tile-geometry aliasing hazard in a descriptor-driven guise, and the split is what makes
it insidious: `hash_cb_format_descriptor` *does* hash `tile`, so a caller who supplies a non-standard
tile gets a correct rebuild for the tile itself while the `face_geometry` that accompanies it is
silently dropped from the key. `set_cb_data_fmt_tile_and_face_geometry` (quoted above) consumes the
two together as one unpacker configuration, so hashing half of a coupled pair is worse than hashing
neither — it creates the appearance of tile-awareness while leaving the aliasing window open on the
half that a non-32x32 tile is most likely to change. Adding `face_geometry` to the hasher closes it
and restores parity with `tt_metal`'s own `hash_cb_format_descriptor`.

One qualification on "does hash `tile`", framework-wide rather than a defect of this op's hasher:
hashing a `Tile` covers `tile_shape`, `face_shape` and `num_faces` but not `transpose_within_face` or
`transpose_of_faces`, because `Tile::attribute_values()`
(`tt_metal/api/tt-metalium/tile.hpp:46-47`) omits them and `Tile::operator==`
(`tt_metal/impl/data_format/tile.cpp:122-124`) ignores them, so they are absent from the canonical
key too. `tt_metal`'s hasher has the same gap, so recommendation 1 does not close it; only an
explicit guard on `get_transpose_within_face()` / `get_transpose_of_faces()` would.

### 6. The *shape* of `kernel.runtime_args` beyond its element count

**Verdict: BUG.**

`kernel.runtime_args` is `std::vector<std::pair<CoreCoord, CoreRuntimeArgs>>`. Hashing
`runtime_args.size()` captures only *how many cores have args* — not which cores, and not how many
args each core has. `kernel.core_ranges` is hashed, so the cores must come from the same range set,
but nothing pins the subset or the per-core arg length.

The `Program` is built by calling `SetRuntimeArgs` once per entry, so only the cores actually present
get a non-empty `RuntimeArgsData`:

```476:479:tt_metal/impl/program/program.cpp
            for (const auto& [core_coord, core_runtime_args] : kernel_descriptor.runtime_args) {
                SetRuntimeArgs(*this, kernel_handle, core_coord, core_runtime_args);
            }
            SetCommonRuntimeArgs(*this, kernel_handle, kernel_descriptor.common_runtime_args);
```

and `apply_descriptor_runtime_args` writes `prog_args[i] = args[i]` with no bounds check against the
cached program's actual storage. `RuntimeArgsData::operator[]` guards only under `TT_ASSERT`, i.e.
debug builds:

```36:39:tt_metal/api/tt-metalium/runtime_args_data.hpp
    std::uint32_t& operator[](std::size_t index) noexcept {
        TT_ASSERT(in_bounds(index));
        return this->rt_args_data[index];
    }
```

**Reproduction (different core subset).**
- Call 1: one kernel over `core_ranges = {(0,0)-(0,3)}`, `runtime_args = [((0,0), [a,b]), ((0,1), [c,d])]`.
- Call 2: identical descriptor except `runtime_args = [((0,2), [a,b]), ((0,3), [c,d])]`.
- `runtime_args.size()` is 2 in both, `core_ranges` is identical, so the hashes are equal → cache hit.
- Cores (0,2) and (0,3) in the cached program have zero-length `RuntimeArgsData`, so
  `apply_descriptor_runtime_args` performs an out-of-bounds write in a release build; cores (0,0) and
  (0,1) keep call 1's frozen args and still execute.
- Symptom: heap corruption on the host, plus the wrong cores running with the wrong data.

**Reproduction (different per-core arg count).** Same core set, but call 1 gives each core 4 args
and call 2 gives 6. Hashes are equal; the cached program's per-core storage is 4 words wide; the
final two writes run past it.

### 7. `CBDescriptor::tensor` nullness

**Verdict: CAVEAT.**

`hash_circular_buffer` hashes `cb.buffer != nullptr` and `cb.global_circular_buffer != nullptr` but
never `cb.tensor != nullptr` (`generic_op_device_operation.cpp:92-93`). A `tensor`-backed CB is
globally allocated exactly like a `buffer`-backed one:

```70:75:tt_metal/impl/buffers/circular_buffer_config.cpp
    const Buffer* backing_buffer = descriptor.buffer;
    if (!backing_buffer && descriptor.tensor) {
        backing_buffer = descriptor.tensor->mesh_buffer().get_reference_buffer();
    }
    if (backing_buffer) {
        this->set_globally_allocated_address(*backing_buffer);
```

so a locally-allocated CB and a tensor-backed CB of identical size, core ranges and formats produce
the same hash and would share a cache entry, even though one occupies L1 scratch and the other
aliases a tensor. What keeps this from being an active bug today is reachability: `CBDescriptor.tensor`
is not exposed by the nanobind layer (`ttnn-nanobind/program_descriptors.cpp:398-499` binds
`total_size`, `core_ranges`, `format_descriptors`, `remote_format_descriptors`, `address_offset`,
`has_buffer`, and the global-CB setters, but nothing that assigns `tensor`), so only a C++ caller can
reach it. tt-metal's own hasher has the identical gap
(`tt_metal/impl/program/program_descriptors.cpp:174`). **What would break it:** exposing `tensor` to
Python, or any C++ `generic_op` caller that conditionally pins a CB to a tensor. **Guard:** add
`cb.tensor != nullptr` alongside the existing `cb.buffer != nullptr` in both hashers.

### 8. `SemaphoreDescriptor::id`

**Verdict: CAVEAT.**

`hash_semaphore` covers `core_ranges`, `core_type` and `initial_value` but not `id`
(`generic_op_device_operation.cpp:97-100`), while `Program{desc}` passes the id straight through to
`add_semaphore` (`tt_metal/impl/program/program.cpp:394-400`), where it selects the semaphore's slot
and therefore its L1 address. Two descriptors identical except for the semaphore ids would share a
cache entry and run against the first call's slot assignment. In practice callers allocate ids via
`ProgramDescriptor::find_available_semaphore_id`, which derives them deterministically from the
descriptor's existing semaphores, so equal-hash descriptors get equal ids; and again tt-metal's
hasher has the same omission (`program_descriptors.cpp:179-182`). **Guard:** hash `semaphore.id`.

### 9. `type_hash<GenericOpDeviceOperation>` — the hash seeds from raw `0`

**Verdict: VALID — invariant, now covered by the framework.**

`compute_program_hash` starts from `size_t hash = 0` rather than
`operation::hash_operation<GenericOpDeviceOperation>(...)`. `FusionDispatchOpDeviceOperation` reuses
the very same `compute_program_descriptor_hash` over the very same `mesh_programs` shape and
explicitly compensates:

```35:44:ttnn/cpp/ttnn/operations/experimental/fusion/device/fusion_dispatch_op_device_operation.cpp
ttsl::hash::hash_t FusionDispatchOpDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t&) {
    // Must differ from GenericOpDeviceOperation::compute_program_hash — same descriptor would
    // otherwise hit the wrong cached_mesh_workload_t layout (segfault in override).
    size_t hash = ttsl::hash::type_hash<FusionDispatchOpDeviceOperation>;
    for (const auto& [mesh_coord_range, program_descriptor] : operation_attributes.mesh_programs) {
        ttsl::hash::hash_combine(hash, mesh_coord_range);
        ttsl::hash::hash_combine(hash, ttnn::operations::generic::compute_program_descriptor_hash(program_descriptor));
    }
    return hash;
}
```

That hazard is now closed at the framework level and no longer depends on either op's seed. The cache
key is a pair, and the canonical half always carries the op-identity prefix even for custom-hash ops:

```1006:1014:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
    static std::string compute_mesh_workload_canonical_key(
        [[maybe_unused]] tt::tt_metal::distributed::MeshDevice* mesh_device,
        std::string_view op_type_name,
        const operation_attributes_t& attrs,
        const tensor_args_t& tensor_args) {
        std::string key{op_type_name};
        if constexpr (requires { DeviceOperation::compute_program_hash(attrs, tensor_args); }) {
            return key;  // custom hash -> opt out beyond the op-identity prefix
        } else {
```

and `ProgramCacheKey::operator==` compares both halves:

```113:119:tt_metal/api/tt-metalium/program_cache.hpp
struct ProgramCacheKey {
    uint64_t hash = 0;
    std::string canonical;

    bool operator==(const ProgramCacheKey& other) const {
        return hash == other.hash && canonical == other.canonical;
    }
```

`ttsl::get_type_name<GenericOpDeviceOperation>()` and
`ttsl::get_type_name<FusionDispatchOpDeviceOperation>()` differ, so a colliding descriptor between
the two ops now resolves to a miss, not a cross-op wrong hit into a differently-shaped
`cached_mesh_workload_t`. The fusion op's explicit type hash is now redundant defence in depth rather
than the only thing standing between the two ops and a segfault. Adding the type hash to
`GenericOpDeviceOperation` for symmetry would still be worthwhile documentation.

### 10. Everything, when the caller sets `ProgramDescriptor::custom_program_hash`

**Verdict: CAVEAT — a complete, unvalidated caller escape hatch.**

```48:51:ttnn/cpp/ttnn/operations/generic/device/generic_op_device_operation.cpp
ttsl::hash::hash_t compute_program_descriptor_hash(const tt::tt_metal::ProgramDescriptor& program_descriptor) {
    if (program_descriptor.custom_program_hash) {
        return *program_descriptor.custom_program_hash;
    }
```

When set, the structural walk is skipped entirely and the caller's 64-bit value becomes the whole
per-descriptor key. Combined with the canonical-key opt-out (section below), a Python caller that
reuses one memoized hash across two structurally different descriptors gets a silent wrong hit with
no diagnostic. The field is `def_rw` on `ttnn.ProgramDescriptor`
(`ttnn-nanobind/program_descriptors.cpp:1111-1114`) and the intended use is memoization — the fusion
build cache sets it to `compute_program_descriptor_hash(desc)` itself
(`models/experimental/ops/descriptors/fusion/fusion.py:284-285`), which is sound. The framework does
protect the one case it can: `merge_program_descriptors` invalidates it
(`tt_metal/impl/program/program_descriptors.cpp:126-127`). **What would break it:** any caller that
computes the hash from a subset of the descriptor, or caches the value across a descriptor mutation
that is not a merge. **Guard:** under `-DTT_DESCRIPTOR_PATCHING_PARITY_CHECK` (or a debug build),
assert `*custom_program_hash == <structural walk>` instead of trusting it.

### Note: the 32x32 tile-assumption check

Checked and not applicable as a standalone finding. This op has no host-side tile math of its own —
a search of the whole `generic` directory for `tt::tile_size`, bare `TILE_HW` / `TILE_WIDTH` /
`TILE_HEIGHT`, and `tensor_spec().tile()` returns nothing, which is expected given that
`create_descriptor` returns the caller's `ProgramDescriptor` verbatim (omission 1). Any tile
arithmetic lives in the caller that built the descriptor, and its results arrive as CB page sizes,
compile-time args and `CBFormatDescriptor::tile` — all hashed. The one place the tile-geometry class
does bite this op is through `CBFormatDescriptor::face_geometry`, which is recorded as a BUG under
omission 5 rather than duplicated here.

## Keys the custom hash adds beyond the default

Effectively the entire descriptor: `mesh_coord_range` per entry, and per descriptor the kernel
sources, source types, core ranges, compile-time args, named compile-time args, defines, Blaze
named-arg schema, kernel config variant, CB total sizes / core ranges / format descriptors
(index, data format, page size, tile) and remote format descriptors, CB backing-kind flags, and
semaphore core ranges / core types / initial values. None of this is in the default key, because
`MeshProgramDescriptor::attribute_values()` collapses to a count.

## Framework side effect of having a custom hash

Defining `compute_program_hash` opts this op out of attribute-level collision resolution: the
canonical key degrades to `"GenericOpDeviceOperation"` (quoted at omission 9). Cross-*op* aliasing is
still impossible, but two *different descriptors* whose 64-bit hashes collide resolve to a wrong hit
rather than a rebuild. For an op whose key is a hand-rolled fold over an unbounded amount of
structure, that raises the cost of every gap listed above.

## Summary

| Omitted vs. default / vs. the program | Used by program? | Patched on hit? | Verdict |
|---|---|---|---|
| `io_tensors`, `output_tensor` (all specs, storage) | No — factory ignores `tensor_args` | n/a | VALID — invariant |
| Runtime-arg values (per-core, common, Blaze named) | Yes | Yes (`apply_descriptor_runtime_args`) | VALID — patched |
| CB buffer address, `cb.address_offset` | Yes | Yes (`UpdateDynamicCircularBufferAddress`) | VALID — patched |
| `kernel.opt_level` | Yes (JIT build) | **No** | **BUG** |
| `kernel.compiler_include_paths` | Yes (JIT build) | **No** | **BUG** |
| `CBFormatDescriptor::face_geometry` | Yes (unpacker build option) | **No** | **BUG** |
| `runtime_args` core subset / per-core lengths | Yes (arg storage layout) | **No** | **BUG** |
| `CBDescriptor::tensor` nullness | Yes (CB allocation kind) | Partially | CAVEAT |
| `SemaphoreDescriptor::id` | Yes (L1 slot) | No | CAVEAT |
| `type_hash<GenericOpDeviceOperation>` | n/a | n/a | VALID — invariant (op identity is carried by the canonical prefix) |
| Whole descriptor, when `custom_program_hash` is set | Yes | No | CAVEAT |

**Program-cache bugs were found.** The op's core design — key on the descriptor's structure, ignore
tensors and runtime-arg values, and let the slow-path rebuild refresh everything per-dispatch — is
sound, and the mode-C classification makes the two large omissions (all tensors, all runtime-arg
values) correct. The defects are all in the *structural* half of the hash, where the slow path
provides no cover: `compute_program_descriptor_hash` is a fork of tt-metal's
`std::hash<ProgramDescriptor>` that has drifted behind it on `opt_level`, `compiler_include_paths`
and `face_geometry`, and both versions under-specify `runtime_args` by hashing only its element
count.

The one structural safety property that is *not* written down anywhere is the mode selection itself:
this op is only safe because `resolve_bindings` always bails, and it only bails because
`tensor_args_t` happens to name the output tensor twice. If `tensor_args_t` were tidied to hold
`io_tensors` alone, or if the adapter were ever changed to pass
`allow_inplace_output_tensor_alias=true` on this branch, `generic_op` would silently switch to the
buffer-binding fast path — at which point every runtime-arg value would freeze at the first miss and
the "hash only `runtime_args.size()`" design would become catastrophically wrong.

## Recommendations

1. Delete `compute_program_descriptor_hash` and call tt-metal's `std::hash<tt::tt_metal::ProgramDescriptor>{}(desc)`
   (`tt_metal/impl/program/program_descriptors.cpp:350-354`) instead. It already covers `opt_level`,
   `compiler_include_paths` and `face_geometry`, and a single implementation cannot drift again. Keep
   the ttnn symbol as a thin forwarding wrapper so the nanobind export and
   `FusionDispatchOpDeviceOperation` are unaffected. This closes bugs 4 and 5 for both ops at once.
2. Replace `kernel.runtime_args.size()` with a hash of the per-entry `(CoreCoord, args.size())`
   pairs, and `common_runtime_args.size()` likewise. Values stay excluded; only the *layout* becomes
   part of the key. This closes bug 6 without costing any cache reuse, since two calls that differ
   only in values keep the same layout.
3. Add `cb.tensor != nullptr` and `semaphore.id` to the descriptor hash (caveats 7 and 8).
4. Add a `static_assert` or a comment on `tensor_args_t` recording that the duplicated
   `output_tensor` is what routes this op to the slow path, and that removing the duplication would
   silently move it to the fast path. Better still, make the intent explicit rather than emergent —
   e.g. give the op a trivial `override_runtime_arguments` that calls `apply_descriptor_runtime_args`
   on a freshly built descriptor, which selects mode A by construction and no longer depends on the
   aliasing accident.
5. Under `-DTT_DESCRIPTOR_PATCHING_PARITY_CHECK`, verify a supplied `custom_program_hash` against the
   structural walk rather than trusting it (caveat 10).
6. `validate_inputs` is declared on the struct (`generic_op_device_operation.hpp:32`) but never
   defined or called, and both the miss and hit validators check only for duplicate mesh coordinate
   ranges while ignoring `tensor_args` entirely. Either implement it (at minimum, that every
   `io_tensor` is on device with a non-null buffer) or drop the declaration. Any such check must go
   into `validate_on_program_cache_hit` — or into a shared helper that both validators call — because
   the hit validator replaces the miss validator on hits, so a check added only to
   `validate_on_program_cache_miss` would not run on the call that inherits a stale program and could
   not support a "pinned by validation" verdict. Factoring the two into one helper, the way
   `verify_no_duplicate_mesh_coord_ranges` is used today, is the change that makes the current
   identity structural instead of a coincidence that the next edit can break.
