# Program Cache Audit — `ccl/reduce_to_root`

Audit of `ttnn::operations::ccl::ReduceToRootOp`'s hand-written
`attribute_names` / `attribute_values()` against the framework default ("hash everything") key.

| | |
|---|---|
| Device operation | `ttnn::operations::ccl::ReduceToRootOp` (`device/reduce_to_root_op.hpp`) |
| Custom hash | **None.** No `compute_program_hash` exists. The omission is done by hand-writing the reflection tuple on `operation_attributes_t` (`device/reduce_to_root_op.hpp:27-28`) |
| `operation_attributes_t` | `root_coord`, `scale_fp32`, `topology`, `input_mux_cores`, `_input_tensor_spec` |
| `attribute_values()` returns | `root_coord`, `scale_fp32`, `topology` — **`input_mux_cores` and `_input_tensor_spec` are excluded** |
| `tensor_args_t` | `input_tensor_l`, `input_tensor_s`, `input_tensor_m`, `optional_output_tensor_l/_s/_m`, `optional_intermediate_tensor` — no attribute tuple, fully reflected |
| Program factories | `ReduceToRootOp::ReduceToRoot` (native `MeshWorkloadFactory`: `create_mesh_workload` + `create_at`) |
| `override_runtime_arguments` | **Yes** (`device/reduce_to_root_program.cpp:843`) |
| `get_dynamic_runtime_args` | **No** |
| `validate_on_program_cache_hit` | **No** — so `validate_on_program_cache_miss` is substituted on every hit |
| Cache-hit patch mechanism | **Op-owned `override_runtime_arguments`** |

The CSV classification (`hash_kind=backdoor`, omits `_input_tensor_spec` and `input_mux_cores`,
`override_runtime_arguments=Y`) matches the code. Its `tensor_input=SPEC-OMITTED` label is
misleading, though: see omission 2 — the input tensor specs are *not* missing from the key.

## Validation on the cache-hit path

`ReduceToRootOp` declares only a miss validator, which delegates to a private `validate`:

```87:97:ttnn/cpp/ttnn/operations/ccl/reduce_to_root/device/reduce_to_root_op.hpp
    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
        validate(operation_attributes, tensor_args);
    };

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

private:
    static void validate(const operation_attributes_t&, const tensor_args_t&);
```

There is no `validate_on_program_cache_hit` anywhere in the op directory, so the dispatcher takes the
`else` branch and substitutes the miss validator on every hit:

```262:266:ttnn/api/ttnn/device_operation.hpp
    if constexpr (HasValidateOnProgramCacheHit<mesh_device_operation_t>) {
        mesh_device_operation_t::validate_on_program_cache_hit(operation_attributes, tensor_args);
    } else {
        mesh_device_operation_t::validate_on_program_cache_miss(operation_attributes, tensor_args);
    }
```

This is the favourable case, and it is the reverse of the intuitive reading: defining no hit
validator means the op is *fully* validated on hits, whereas defining a narrow one would have
silently disabled every miss-time check. Every `TT_FATAL` reached from `ReduceToRootOp::validate`
therefore executes on the offending call, not merely on the first build. Two consequences for this
document:

- A "pinned by validation" verdict would be legitimate here rather than miss-only. In fact no such
  verdict appears below, because none of the omissions is constrained by `validate` — the two
  `TT_FATAL`s that touch an omitted parameter constrain only the *size* of `input_mux_cores`
  (omission 1) and the page-size alignment, neither of which pins a value.
- Any guard added to close omission 1 belongs in `ReduceToRootOp::validate`, and because that
  function is what runs on hits under the substitution branch, such a guard would genuinely fire on
  the call that would otherwise inherit a stale program. See recommendation 3.

A CSV `own_hit_validator=N` row therefore does not mean the hit path is unvalidated; under this
branch it usually means the opposite.

## Cache-hit patch mechanism

`ReduceToRoot` defines `cached_mesh_workload_t` and `create_mesh_workload`, so it satisfies
`MeshWorkloadFactoryConcept` directly (`ttnn/api/ttnn/operation_concepts.hpp:54`) and is dispatched
without any adapter wrapping. Because it has no `apply_descriptor`, the cache-hit dispatcher calls
its `override_runtime_arguments`:

```279:285:ttnn/api/ttnn/device_operation.hpp
        if constexpr (requires { &WorkloadFactory::apply_descriptor; }) {
            WorkloadFactory::apply_descriptor(
                cached_mesh_workload, operation_attributes, tensor_args, tensor_return_value);
        } else {
            WorkloadFactory::override_runtime_arguments(
                cached_mesh_workload, operation_attributes, tensor_args, tensor_return_value);
        }
```

This is the strongest of the three cache-hit modes — the op re-derives per-dispatch state itself —
but it is only as strong as what the callback actually rewrites. Reading it
(`device/reduce_to_root_program.cpp:843-934`), it rewrites a fixed, small prefix of each kernel's
per-core runtime args and nothing else:

| Device role | Kernel | Indices re-applied |
|---|---|---|
| sender | reader | `[0][1][2]` — the three input tensor addresses |
| sender | writer | `[0][1]` — intermediate tensor address, semaphore 0 |
| root | reader | `[1]…[6]` — three inputs, intermediate, semaphores 0 and 1 |
| root | writer | `[0][1][2]` — three output addresses |
| root2 | reader | `[0]…[4]` — three inputs, intermediate, semaphore 0 |
| root2 | writer | `[0][1]` — intermediate address, semaphore 1 |

Note what it does **not** touch: the fabric-mux runtime args that `fabric_mux_rt_args` appends after
those prefixes (`fabric_mux_x`, `fabric_mux_y`, the mux channel/handshake/flow-control addresses, the
termination-master NoC coordinates), the mux kernel's own runtime args, and — structurally
unreachable by any callback — the core ranges the kernels were created on.

**Obligation on the hash.** Everything except the address/semaphore prefixes above must be a pure
function of the hashed set: all compile-time args, all CB sizes and formats, every kernel's
`CoreRangeSet`, and every runtime arg the callback skips.

## Baseline: what the default hash would cover

There is no `compute_program_hash`, so the framework takes the reflection branch:

```982:987:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
        if constexpr (requires { DeviceOperation::compute_program_hash(attrs, tensor_args); }) {
            hash = DeviceOperation::compute_program_hash(attrs, tensor_args);
        } else {
            hash =
                ttsl::hash::hash_objects_with_default_seed(ttsl::hash::type_hash<DeviceOperation>, attrs, tensor_args);
        }
```

A "hash everything" key with no hand-written tuple would be:

| Source | Fields |
|---|---|
| `operation_attributes` | `root_coord`, `scale_fp32`, `topology`, `input_mux_cores`, `_input_tensor_spec` |
| `input_tensor_l/_s/_m` | storage variant kind, `logical_shape`, and `tensor_layout` = { `dtype`, `page_config`, `memory_config`, `alignment` } — three times |
| `optional_output_tensor_l/_s/_m`, `optional_intermediate_tensor` | engaged flag plus, when engaged, the same six-field decomposition |

`tensor_args_t` has no attribute tuple, so all seven tensors are reflected in full and none of the
tensor-side information is omitted.

## What the custom key covers

```19:29:ttnn/cpp/ttnn/operations/ccl/reduce_to_root/device/reduce_to_root_op.hpp
    struct operation_attributes_t {
        const MeshCoordinate& root_coord;
        const float scale_fp32;
        const tt::tt_fabric::Topology topology;
        const std::optional<std::vector<ttnn::CoreCoord>> input_mux_cores;

        const std::vector<tt::tt_metal::TensorSpec> _input_tensor_spec;

        static constexpr auto attribute_names = std::forward_as_tuple("root_coord", "scale_fp32", "topology");
        auto attribute_values() const { return std::forward_as_tuple(root_coord, scale_fp32, topology); };
    };
```

`attribute_values()` names three of the five members. The two unnamed members are the omissions
audited below. Reflection honours the tuple over member enumeration
(`tt_stl/tt_stl/reflection.hpp:1319-1334`), so the excluded members are invisible to the hash.

Critically, they are invisible to the *canonical key* as well — `append_canonical` takes the same
branch:

```1499:1500:tt_stl/tt_stl/reflection.hpp
    } else if constexpr (ttsl::reflection::detail::supports_compile_time_attributes_v<T>) {
        std::apply([&out](const auto&... a) { (append_canonical(out, a), ...); }, object.attribute_values());
```

So unlike a hash-only gap, there is no exact-comparison fallback: two calls differing only in an
excluded member produce a byte-identical `ProgramCacheKey` and *always* hit, never rebuild.

## Omitted parameters

### 1. `input_mux_cores`

**Verdict: BUG.**

`input_mux_cores` selects the logical cores on which the fabric-mux kernel is placed:

```464:488:ttnn/cpp/ttnn/operations/ccl/reduce_to_root/device/reduce_to_root_program.cpp
    std::vector<CoreCoord> mux_cores = {
        CoreCoord(2, 0), CoreCoord(2, 1), CoreCoord(2, 2), CoreCoord(2, 3)};  // to be modified based on device type

    if (operation_attributes.input_mux_cores.has_value()) {
        mux_cores = operation_attributes.input_mux_cores.value();
    }
    auto all_mux_cores = mux_cores;
    if (is_sender_device) {
        mux_cores = {mux_cores[0], mux_cores[2]};
    }

    CoreRangeSet mux_core_range_set = CoreRangeSet(mux_cores);

    tt::tt_fabric::FabricMuxConfig mux_kernel_config = tt::tt_fabric::FabricMuxConfig(
        num_workers_per_direction, 0, 2, 0, buffer_size_bytes_full_size_channel, mux_base_l1_address);

    auto mux_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp",
        mux_core_range_set,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = mux_kernel_config.get_fabric_mux_compile_time_args(),
            .opt_level = tt::tt_metal::KernelBuildOptLevel::O3});
```

This is a *structural* dependency in two independent ways.

**(a) Kernel placement.** `mux_core_range_set` is the `CoreRangeSet` handed to `CreateKernel`. Core
ranges are fixed when the `Program` is built and no cache-hit callback can move a kernel to a
different core — `override_runtime_arguments` writes into existing `RuntimeArgsData`, it cannot
create or relocate kernels.

**(b) Worker-side mux coordinates.** The mux's *virtual* core coordinates are baked into the worker
reader/writer runtime args via `fabric_mux_rt_args`:

```105:107:ttnn/cpp/ttnn/operations/ccl/reduce_to_root/device/reduce_to_root_program.cpp
    worker_rt_args.push_back(is_termination_master);  // is_termination_master
    worker_rt_args.push_back(mux_virtual_core.x);     // fabric_mux_x
    worker_rt_args.push_back(mux_virtual_core.y);     // fabric_mux_y
```

with `mux_virtual_core` derived from `all_mux_cores`, e.g. on the sender path:

```723:723:ttnn/cpp/ttnn/operations/ccl/reduce_to_root/device/reduce_to_root_program.cpp
                CoreCoord mux_virtual_core = mesh_device->worker_core_from_logical_core(all_mux_cores[start_idx]);
```

For the sender writer these land at indices 5 and 6 (after `{intermediate_addr, semaphore, noc_x,
noc_y}` and the `is_termination_master` flag), and `override_runtime_arguments` rewrites only indices
0 and 1. The mux kernel's own runtime args
(`SetRuntimeArgs(program, mux_kernel_id, {mux_logical_core}, mux_rt_args)`, line 695) are never
touched by the callback at all — `shared_variables_t` does not even record `mux_kernel_id`
(`device/reduce_to_root_op.hpp:45-59`).

`input_mux_cores` appears in neither `attribute_values()` nor `tensor_args_t`, so it contributes
nothing to the key. The only guard on it is a size check, which does not constrain *which* cores:

```73:78:ttnn/cpp/ttnn/operations/ccl/reduce_to_root/device/reduce_to_root_op.cpp
    if (operation_attributes.input_mux_cores.has_value()) {
        TT_FATAL(
            operation_attributes.input_mux_cores.value().size() == 4,
            "Input mux cores size must be 4, got {}",
            operation_attributes.input_mux_cores.value().size());
    }
```

**Reproduction.**
- Call 1: `ttnn.reduce_to_root(l, s, m, root_coord=(0,1), scale_fp32=1.0, topology=Linear, input_mux_cores=None)`.
  The mux kernel is created on `{(2,0),(2,1),(2,2),(2,3)}`; every worker's `fabric_mux_x/y` points at
  the virtual coordinates of those cores.
- Call 2: same three tensors, same `root_coord`, same `scale_fp32`, same `topology`, but
  `input_mux_cores=[(3,0),(3,1),(3,2),(3,3)]`.
- The hashed set is `{root_coord, scale_fp32, topology}` plus the seven tensors, all identical, and
  the canonical key is byte-identical. Call 2 is a cache hit.
- Stale artifacts: (i) the `tt_fabric_mux.cpp` kernel is still placed on `(2,0)…(2,3)`, with no mux
  running on `(3,0)…(3,3)`; (ii) each worker's `fabric_mux_x`/`fabric_mux_y` runtime args still name
  the row-2 cores, and `override_runtime_arguments` does not rewrite them.
- Symptom: the requested placement is silently ignored. The dispatched program is internally
  consistent, so it does not fault — which is what makes this dangerous. `input_mux_cores` exists so
  a caller can steer mux traffic away from cores it needs for something else, and the mux writes into
  L1 at the allocator base address (`mux_base_l1_address = l1_unreserved_base_address`, lines
  458-460). A second call that moves the mux to free up row 2 for a co-resident kernel keeps
  hammering row 2's L1, producing corruption or a fabric hang whose cause is nowhere near the call
  site.

The fix is one line: add `input_mux_cores` to `attribute_names` / `attribute_values()`. Because it is
an `std::optional<std::vector<CoreCoord>>`, reflection handles it natively
(`tt_stl/tt_stl/reflection.hpp:1409-1416`, `:1356-1364`). The cost is a rebuild whenever placement
changes, which is exactly the correct behaviour — the program genuinely differs.

### 2. `_input_tensor_spec`

**Verdict: VALID — invariant (fully redundant with `tensor_args`).**

Despite the CSV's `SPEC-OMITTED` label, no tensor-spec information is missing from the key. The
member is constructed as an exact copy of the three input tensors' specs:

```224:229:ttnn/cpp/ttnn/operations/ccl/reduce_to_root/device/reduce_to_root_op.cpp
        OperationType::operation_attributes_t{
            root_coord,
            scale_fp32,
            topology,
            input_mux_cores,
            {input_tensor_l.tensor_spec(), input_tensor_s.tensor_spec(), input_tensor_m.tensor_spec()}},
```

and those three tensors are themselves reflected members of `tensor_args_t`
(`device/reduce_to_root_op.hpp:31-39`), which has no attribute tuple. A `Tensor` hashes as
`(storage, tensor_spec())`, so `_input_tensor_spec` is a strict subset of what `tensor_args`
already contributes. Excluding it from `attribute_values()` removes exactly zero information and
saves a duplicated traversal.

Confirming that the program's structural inputs really do ride on those specs: the CB sizes and the
kernel core ranges all derive from `input_tensor_l`'s spec and shard grid, which live inside the
hashed `memory_config`:

```181:202:ttnn/cpp/ttnn/operations/ccl/reduce_to_root/device/reduce_to_root_program.cpp
    TT_FATAL(input_tensor_l.is_sharded(), "Input tensor must be sharded");
    const auto& shard_spec = input_tensor_l.shard_spec().value();
    const auto& shard_grid = shard_spec.grid;

    // Get all cores from the shard grid
    std::vector<CoreCoord> all_coord_cores;
    all_coord_cores.reserve(shard_grid.num_cores());
    for (const auto& core_range : shard_grid.ranges()) {
        auto cores = corerange_to_cores(core_range, std::nullopt);
        all_coord_cores.insert(all_coord_cores.end(), cores.begin(), cores.end());
    }
    const CoreRangeSet all_cores = shard_grid;
    const uint32_t num_shard_cores = all_coord_cores.size();

    uint32_t input_l_total_num_pages = data_movement::get_num_pages(input_tensor_l);
    const uint32_t input_l_num_pages = input_l_total_num_pages / num_shard_cores;
    const uint32_t input_num_tiles = input_l_num_pages;

    const uint32_t input_page_size_bytes = input_tensor_l.tensor_spec().compute_page_size_bytes();
    const uint32_t l1_alignment = tt::tt_metal::hal::get_l1_alignment();

    uint32_t packet_size_bytes = input_num_tiles * input_page_size_bytes;
```

`input_num_tiles`, `input_page_size_bytes` and `packet_size_bytes` feed every kernel's compile-time
args (lines 491-505, 522-535, 550, 557-570, 585-595) and every CB's `total_size` / `page_size` (lines
227-396). `shard_grid` is the `all_cores` range set every kernel is created on. All of it is a pure
function of the hashed input specs.

The member is dead weight rather than a hazard, but it is confusingly named — the leading underscore
and the parallel comment in `point_to_point` ("put this in here to hash on tensor spec") suggest an
author who believed the field was what made the spec reachable. It is not; `tensor_args` is.

### 3. Tile geometry (`page_config` / `Tile`)

**Verdict: VALID — unused (read by the factory, but only via a value that is hashed in full).**

This op is *genuinely tile-aware* — it reads the input tensor's real tile dimensions rather than
assuming 32x32:

```206:221:ttnn/cpp/ttnn/operations/ccl/reduce_to_root/device/reduce_to_root_program.cpp
    // sdpa compute values
    const auto tile_width = input_tensor_l.tensor_spec().tile().get_width();
    const auto tile_height = input_tensor_l.tensor_spec().tile().get_height();

    bool use_mla = true;
    uint32_t q_heads_parallel_factor = 1;
    uint32_t head_dim_v = input_num_tiles * tile_width;
    // auto q_shape = {1, 1, 8, 512} ; //{1, B, PNH, DH};
    // auto k_shape = {1, 8, 256, 512}; //{B, NKV, S, DH};
    uint32_t PNH = 8;  // q_shape[2],
    uint32_t DH = input_num_tiles * tile_width;  // k_shape[3];
    uint32_t DHt = DH / tile_width;
    uint32_t vDHt = use_mla ? head_dim_v / tile_width : DHt;
    uint32_t PNHt = PNH / q_heads_parallel_factor / tile_height;

    const uint32_t Sq_chunk_t = PNHt;
```

That makes this the *mirror-image* case: the program provably varies with `Tile`, so `Tile` must be
reachable from the key or a non-32x32 tensor silently inherits a 32x32 tensor's program.

**Where the two values actually go.** `tile_width` and `tile_height` reach exactly one place —
compile-time args 18 and 19 of the compute kernel:

```628:642:ttnn/cpp/ttnn/operations/ccl/reduce_to_root/device/reduce_to_root_program.cpp
        compute_ct_args = {compute_out_cb_l, compute_cb_l,      compute_cb_2_l,    compute_cb_s,     compute_cb_2_m,
                           compute_cb_m,     compute_out_cb_m,  cb_exp_max_diff_2, compute_cb_2_s,   cb_exp_max_diff,
                           compute_out_cb_s, cb_m_temp,         cb_s_temp,         cb_s1_temp,       cb_s2_temp,
                           cb_l1_temp,       cb_l2_temp,        scale_val,         Sq_chunk_t,       vDHt,
                           loop_size,        intermediate_cb_l, intermediate_cb_s, intermediate_cb_m};
        tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/ccl/reduce_to_root/device/kernels/compute_kernel.cpp",
            all_cores,
            tt::tt_metal::ComputeConfig{
                .math_fidelity = math_fidelity,
                .fp32_dest_acc_en = true,
                .math_approx_mode = math_approx_mode,
                .compile_args = compute_ct_args,
            });
```

Compile-time args are baked into the JIT-compiled binary and are the category
`override_runtime_arguments` structurally cannot repair, so this is the worst place for a
tile-dependent value to land if the tile were unhashed.

Of the two, only one genuinely varies. `vDHt` is `head_dim_v / tile_width` where
`head_dim_v = input_num_tiles * tile_width`, so `tile_width` cancels algebraically and `vDHt` is
just `input_num_tiles` for any tile width (`DHt` is the same expression and is dead, since `use_mla`
is hardcoded `true`). `Sq_chunk_t = PNHt = 8 / 1 / tile_height` does vary: it is `0` for any tile
16 rows or taller and `1` for `Tile{8, 32}`. So there is a real, observable dependence of a
compile-time arg on the input tile height. (That `0` is not a hypothetical — it is what the standard
32x32 tile produces, and it is the subject of `## Non-cache correctness defects` below. The cache
handles the dependence correctly; the factory does not.)

**The tile is in the key.** Verified end to end rather than assumed. `tensor_args_t` carries no
attribute tuple:

```31:39:ttnn/cpp/ttnn/operations/ccl/reduce_to_root/device/reduce_to_root_op.hpp
    struct tensor_args_t {
        const Tensor input_tensor_l;
        const Tensor input_tensor_s;
        const Tensor input_tensor_m;
        const std::optional<Tensor> optional_output_tensor_l;
        const std::optional<Tensor> optional_output_tensor_s;
        const std::optional<Tensor> optional_output_tensor_m;
        const std::optional<Tensor> optional_intermediate_tensor;
    };
```

so `input_tensor_l` is reflected in full, and a `Tensor` hashes as `(storage, tensor_spec())`. From
there the tile is reached in four hops. First, `TensorLayout` names `page_config` among its hashed
attributes:

```75:76:tt_metal/api/tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp
    static constexpr auto attribute_names = std::forward_as_tuple("dtype", "page_config", "memory_config", "alignment");
    std::tuple<const DataType&, const PageConfig&, const MemoryConfig&, const Alignment&> attribute_values() const;
```

Second, `PageConfig` forwards to its variant:

```50:51:tt_metal/api/tt-metalium/experimental/tensor/spec/layout/page_config.hpp
    static constexpr auto attribute_names = std::forward_as_tuple("config");
    auto attribute_values() const { return std::forward_as_tuple(config_); }
```

Third, `hash_object` on a variant hashes the active index and visits the active alternative:

```1286:1292:tt_stl/tt_stl/reflection.hpp
inline hash_t hash_object(const std::variant<Ts...>& variant) noexcept {
    if constexpr (DEBUG_HASH_OBJECT_FUNCTION) {
        fmt::print("Hashing std::variant: {}\n", variant);
    }
    auto active_variant = variant.index();
    return std::visit([&](const auto& value) { return hash_objects(active_variant, value); }, variant);
}
```

Fourth, the active alternative for a tiled tensor is `TilePageConfig`, whose sole member is the
`Tile` (`page_config.hpp:23-27`), and `Tile` hashes its dimensions explicitly:

```46:47:tt_metal/api/tt-metalium/tile.hpp
    static constexpr auto attribute_names = std::forward_as_tuple("tile_shape", "face_shape", "num_faces");
    auto attribute_values() const { return std::forward_as_tuple(tile_shape, face_shape, num_faces); }
```

One framework-wide qualification on that chain, which is not a defect of this op: `Tile` hashes
`tile_shape`, `face_shape` and `num_faces` but not `transpose_within_face` or `transpose_of_faces`,
and `Tile::operator==` (`tt_metal/impl/data_format/tile.cpp:122-124`) ignores them too, so the
transpose flags are absent from both halves of the key for every op. It costs nothing here — the
factory reads only `get_width()` and `get_height()`, so the program does not vary with them — but
"the tile is hashed" should be read as the shapes and face count, not transpose.

Nothing removes the tile geometry again. The hand-written tuple that creates this op's backdoor is on
`operation_attributes_t`, not on `tensor_args_t`, so it cannot reach the tensors; and the one
attributes member that carries spec data, `_input_tensor_spec`, is excluded but redundant
(omission 2). Two calls differing only in `Tile{8,32}` versus `Tile{32,32}` therefore produce
different hashes *and* different canonical keys, and the second call rebuilds. `Sq_chunk_t` is
recomputed from the new tile on that rebuild.

That is the cache question settled, and it is worth stating explicitly: `reduce_to_root` is one of
the ops where the tile reaches the key in full, so there is no aliasing hazard here of the kind the
Class-A ops have. It is emphatically *not* a statement that the op handles a non-8x32 tile correctly
— the value `Sq_chunk_t` is recomputed *to* is zero for the standard 32x32 tile, which is a factory
defect written up under `## Non-cache correctness defects`. The cache does its job; what it caches is
wrong.

The remaining tile-derived quantities are likewise safe. `input_page_size_bytes` is
`input_tensor_l.tensor_spec().compute_page_size_bytes()` (line 199), a pure function of the hashed
spec, and it is the multiplier behind every CB `total_size` and `page_size` in the factory (lines
227-396) and behind `packet_size_bytes` (line 202). `input_num_tiles` derives from
`data_movement::get_num_pages(input_tensor_l)` and the shard grid's core count (lines 195-197), both
functions of the hashed `logical_shape` and `memory_config`.

**One inconsistency worth flagging, and it turns out to be a real defect.** Every CB in the factory
sets its tile dimensions from a hardcoded 8x32 tile rather than from the input tensor's tile:

```223:224:ttnn/cpp/ttnn/operations/ccl/reduce_to_root/device/reduce_to_root_program.cpp
    const auto tiny_tile = tt::tt_metal::Tile({8, 32});
    auto stats_tile = tiny_tile;
```

`stats_tile` is passed to `set_tile_dims` on all 24 CBs (lines 235, 241, 247, 254, 260, 266, 277,
284, 292, 299, 306, 313, 319, 325, 333, 340, 347, 355, 361, 368, 375, 382, 389, 396), including the
packet-header CBs whose format is `RawUInt32`. That is the mixed-idiom pattern: the same factory
reads the real tile for the compute kernel's compile-time args while pinning every CB's tile dims to
a literal.

It is **not** a cache-correctness defect — a compile-time constant cannot go stale across a cache
hit, since it is the same value on every dispatch — so it earns no verdict in this section and does
not enter the cache bug count. But the two halves of the factory do not merely use different idioms,
they disagree about the tile the op is built for, and nothing checks which one the caller supplied.
That is a genuine correctness defect on the very first call, and it is written up in full under
`## Non-cache correctness defects` below.

## Non-cache correctness defects

Everything above concerns the program cache. This section records a defect found during the audit
that is **not** a cache defect: the offending parameter is in the key, the cache does exactly the
right thing, and the wrong behaviour comes entirely from the factory. It is documented here rather
than dropped because it is the more reachable of this op's two problems — the cache bug in omission 1
needs two calls that differ in mux placement, whereas this one goes wrong on the very first call.

### The 8x32 tile literal, and the guard that is missing

**Not a program-cache bug. A factory bug, and the easier of the two to hit.**

The factory computes its tile geometry from two sources that disagree with each other.

**Source one: the input tensor's real tile.**

```206:221:ttnn/cpp/ttnn/operations/ccl/reduce_to_root/device/reduce_to_root_program.cpp
    // sdpa compute values
    const auto tile_width = input_tensor_l.tensor_spec().tile().get_width();
    const auto tile_height = input_tensor_l.tensor_spec().tile().get_height();

    bool use_mla = true;
    uint32_t q_heads_parallel_factor = 1;
    uint32_t head_dim_v = input_num_tiles * tile_width;
    // auto q_shape = {1, 1, 8, 512} ; //{1, B, PNH, DH};
    // auto k_shape = {1, 8, 256, 512}; //{B, NKV, S, DH};
    uint32_t PNH = 8;  // q_shape[2],
    uint32_t DH = input_num_tiles * tile_width;  // k_shape[3];
    uint32_t DHt = DH / tile_width;
    uint32_t vDHt = use_mla ? head_dim_v / tile_width : DHt;
    uint32_t PNHt = PNH / q_heads_parallel_factor / tile_height;

    const uint32_t Sq_chunk_t = PNHt;
```

`PNH` is a hardcoded 8 (line 215), `q_heads_parallel_factor` a hardcoded 1 (line 211), so line 219
reduces to `PNHt = 8 / tile_height` in integer arithmetic, and line 221 makes that the chunk count
`Sq_chunk_t` handed to the compute kernel as compile-time arg 18:

```628:632:ttnn/cpp/ttnn/operations/ccl/reduce_to_root/device/reduce_to_root_program.cpp
        compute_ct_args = {compute_out_cb_l, compute_cb_l,      compute_cb_2_l,    compute_cb_s,     compute_cb_2_m,
                           compute_cb_m,     compute_out_cb_m,  cb_exp_max_diff_2, compute_cb_2_s,   cb_exp_max_diff,
                           compute_out_cb_s, cb_m_temp,         cb_s_temp,         cb_s1_temp,       cb_s2_temp,
                           cb_l1_temp,       cb_l2_temp,        scale_val,         Sq_chunk_t,       vDHt,
                           loop_size,        intermediate_cb_l, intermediate_cb_s, intermediate_cb_m};
```

**Source two: a literal 8-row tile**, pinned onto every circular buffer (lines 223-224, applied by
`set_tile_dims` on all 24 CBs between lines 235 and 396, quoted under omission 3).

The two are coherent for exactly one input tile. With `Tile{8, 32}`, `PNHt = 8 / 1 / 8 = 1` and the
CB literal matches the tensor. With the **standard 32x32 tile**, `PNHt = 8 / 1 / 32 = 0` by integer
division, and a zero chunk count is baked into the compute kernel's binary while the CBs are still
declared as 8-row tiles.

**Nothing guards it.** There is no `TT_FATAL` on tile height or width anywhere in the op.
`ReduceToRootOp::validate` (`device/reduce_to_root_op.cpp:21-78`) checks the optional output tensors'
specs, the page-size alignment, and the size of `input_mux_cores`; it never inspects
`tensor_spec().tile()`. A search of the whole op directory for tile-related identifiers finds
`get_width()`/`get_height()` reads at lines 207-208 and the `Tile({8, 32})` literal at line 223, and
no comparison against either.

**What goes wrong on a single call.** Pass three tiled inputs with the default 32x32 tile, everything
else valid. `Sq_chunk_t` is compiled in as `0`, so the compute kernel's work quantum
`out_chunk_tiles = Sq_chunk_t * vDHt` (`device/kernels/compute_kernel.cpp:130`, and `out_tiles` at
line 49) is zero: every accumulate, move and write loop runs zero iterations and the kernel pushes
nothing. Meanwhile the readers and writers size their transfers from `input_num_tiles`, an unrelated
compile-time arg derived from the page count, and the root writer waits on a nonzero count
(`cb_wait_front(cb_int_cb_l, input_num_tiles)`, `device/kernels/root_receive_writer_kernel.cpp:20`).
The expected symptom is therefore a hang on a CB that is never filled, rather than silently wrong
data — but it is a hang with no diagnostic pointing at the tile.

Independently of that, the CB declarations are internally inconsistent for any non-8x32 tile: every
CB's byte size and page size come from `input_page_size_bytes`, which is
`input_tensor_l.tensor_spec().compute_page_size_bytes()` (line 199) and therefore reflects the real
tile, while the tile dims attached to the same CB are the 8x32 literal. The two halves of each CB
description disagree.

**Read the literal as deliberate, and the missing guard as the defect.** The 8x32 tile is the
flash-decode "tiny tile" idiom, not a typo: it sits alongside `PNH = 8`, `use_mla`,
`q_heads_parallel_factor` and `Sq_chunk_t`, the variable naming is lifted from SDPA, and the only
in-tree caller constructs its tensors with `ttnn.Tile((8, 32))`
(`tests/ttnn/unit_tests/operations/ccl/blackhole_CI/box/nightly/test_reduce_to_root_trace.py:122`).
The problem is not that the op assumes an 8x32 tile; it is that it assumes it *silently*. The op is
correct for exactly one input tile geometry and says so nowhere — not in a `TT_FATAL`, not in
`validate`, not in the nanobind docstring, which advertises only `TILE` layout
(`reduce_to_root_nanobind.cpp:53`).

**The program cache behaves correctly here — do not go looking for an aliasing bug.** This op has no
`compute_program_hash`, and its hand-written attribute tuple is on `operation_attributes_t`, which
cannot reach the tensors. `page_config`, and hence `Tile`, is in the key in full via
`tensor_args.input_tensor_l` — the four-hop chain is traced in omission 3. A 32x32 tensor therefore
computes a different hash *and* a different canonical key from an 8x32 tensor, misses, and gets its
own freshly built program. It is simply built wrong. Under rule 4c this is a correctly-keyed but
incorrectly-built program, so it is excluded from the cache bug count, which remains **one**
(`input_mux_cores`).

**The fix**, in two variants, and choosing between them needs the op author's intent:

- **Pin the assumption.** Add to `ReduceToRootOp::validate` a `TT_FATAL` requiring
  `input_tensor_l.tensor_spec().tile() == tt::tt_metal::Tile({8, 32})`. This is by a wide margin the
  smaller change — a few lines in one function, no effect on the single in-tree caller, which already
  passes an 8x32 tile — and because this op declares no `validate_on_program_cache_hit`, the
  substitution branch (`ttnn/api/ttnn/device_operation.hpp:262-266`) runs `validate` on hits as well
  as misses, so the guard fires on every offending call.
- **Generalise the op.** Derive `stats_tile` from `input_tensor_l.tensor_spec().tile()` instead of
  the literal. Note that this is *not* sufficient on its own: `PNHt = 8 / tile_height` still
  collapses to zero for a 32-row tile, so generalising also requires deciding what `PNH = 8` denotes
  — eight rows, or eight tiles — and reworking line 219 accordingly. That is a semantic question
  about the flash-decode formulation, not a mechanical substitution, which is why it needs the
  author. It requires no hash change either way, since the tile is already in the key.

Pinning is the right default: it is cheap, it is honest about what the op supports today, and it
converts a hang into an error message. Generalising is the right answer only if this op is intended
to serve inputs other than the MLA stats tensors it was written for.

## Keys the custom key adds beyond the default

None. The hand-written tuple is strictly a subset of the members.

Worth noting in the other direction: `topology` is hashed but is *unused* by the program factory,
which hardcodes the topology it actually builds for:

```163:163:ttnn/cpp/ttnn/operations/ccl/reduce_to_root/device/reduce_to_root_op.cpp
    auto topology = tt::tt_fabric::Topology::Linear;
```

`operation_attributes.topology` is never read in `reduce_to_root_program.cpp`. Hashing it is
harmless over-keying (an extra rebuild if a caller passes `Ring`, which would then build a Linear
program anyway) — a correctness question for the factory, not for the cache.

## Framework side effect of having a custom hash

Not applicable in the usual sense: this op defines no `compute_program_hash`, so it keeps the exact
canonical key (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:1014-1020`) and a 64-bit hash
collision between two distinguishable configurations resolves to a rebuild rather than a wrong hit.

The backdoor is worse than a custom hash on one specific axis, though, and it is worth being explicit
about it: because `canonical_key` walks `attribute_values()` too (quoted above), an excluded member is
excluded from *both* halves of the key. A custom `compute_program_hash` that skipped
`input_mux_cores` would at least leave the door open to a future canonical-key improvement; the
attribute-tuple backdoor forecloses it.

## Summary

| Omitted vs. default | Used by program? | Patched on hit? | Verdict |
|---|---|---|---|
| `input_mux_cores` | Yes — mux kernel `CoreRangeSet`, and `fabric_mux_x/y` in worker runtime args | **No** — kernel placement is structural; `override_runtime_arguments` rewrites only the address/semaphore prefixes | **BUG** |
| `_input_tensor_spec` | Yes, but only via `tensor_args`, which is fully hashed | n/a | VALID — invariant |
| Tile geometry (`page_config` / `Tile`) | Yes — `tile_height` feeds compute-kernel compile-time arg 18 (`Sq_chunk_t`) | n/a — reaches the key in full via `tensor_args.input_tensor_l` | VALID — invariant (not an omission: the tile is in the key) |

**A program-cache correctness bug was found: `input_mux_cores`.** Two `reduce_to_root` calls that
differ only in the requested mux placement produce an identical `ProgramCacheKey` — identical hash
*and* identical canonical string — and the second call silently reuses the first call's mux
placement. Because the reused program is self-consistent it runs to completion, so the failure
surfaces as L1 contention or a fabric hang on whatever other work the caller put on the cores it
thought it had freed, rather than as an error from this op.

The other backdoor omission, `_input_tensor_spec`, is safe and in fact redundant: `tensor_args_t`
carries no attribute tuple, so all three input tensors (and all four optional tensors) are hashed in
full. Every compile-time arg, CB size and kernel core range in the factory is a function of that set.

The tile-assumption check passes cleanly **on the cache half**. This op belongs to the genuinely
tile-aware class — it reads `tensor_spec().tile()` rather than assuming 32x32 — and the tile geometry
does reach the cache key, through `tensor_args.input_tensor_l` and the `tensor_spec` to `page_config`
to `TilePageConfig` to `Tile` chain, none of which the attribute-tuple backdoor touches. A non-32x32
input therefore gets its own cache entry rather than inheriting a 32x32 program.

It does not pass on the factory half. The program the op builds is coherent only for an 8x32 input
tile: `Sq_chunk_t` collapses to zero for the standard 32x32 tile, every CB's tile dims are pinned to
an 8x32 literal while its byte sizes follow the real tile, and no `TT_FATAL` anywhere states the
requirement. That is a real defect, and on a single call rather than two — but it is a factory
defect, not a cache defect, because the tile is in the key and the cache correctly builds a new
program for each tile. Per rule 4c it is recorded under `## Non-cache correctness defects` above and
excluded from the count, which stays at **one program-cache bug**.

## Recommendations

1. Add `input_mux_cores` to `attribute_names` / `attribute_values()`:

   ```cpp
   static constexpr auto attribute_names =
       std::forward_as_tuple("root_coord", "scale_fp32", "topology", "input_mux_cores");
   auto attribute_values() const {
       return std::forward_as_tuple(root_coord, scale_fp32, topology, input_mux_cores);
   }
   ```

   This is the whole fix; `std::optional<std::vector<CoreCoord>>` needs no special handling in either
   `hash_object` or `append_canonical`.
2. Delete `_input_tensor_spec`. It is never read (the factory reads the tensors directly) and never
   hashed, and its presence invites the reader to believe the specs would otherwise be missing from
   the key. If it is retained for lifetime reasons, rename it and add a comment saying it is
   deliberately excluded because `tensor_args` already covers it.
3. Extend the `input_mux_cores` validation beyond the size check — at minimum, assert the four cores
   are distinct and disjoint from `input_tensor_l.shard_spec()->grid`, since the mux writes at the L1
   allocator base on its cores. Put it in `ReduceToRootOp::validate`
   (`device/reduce_to_root_op.cpp:21`), which is where the existing size check lives and which is the
   function that actually runs on the hit path: the op declares no `validate_on_program_cache_hit`, so
   the dispatcher substitutes `validate_on_program_cache_miss` — and hence `validate` — on every hit
   (`ttnn/api/ttnn/device_operation.hpp:262-266`). A guard added there fires on the offending call
   rather than only on the first build. If a hit validator is ever added to this op, mirror these
   checks into it or route both through one helper, since defining one would otherwise disable all of
   them on hits. Note this is a defence-in-depth measure, not a substitute for recommendation 1: a
   distinctness or disjointness assertion does not pin `input_mux_cores` to a single value, so it
   would not downgrade the BUG.
4. Either honour `operation_attributes.topology` in the program factory or drop the parameter; today
   it is hashed, accepted from the caller, and then overwritten with `Topology::Linear`.
5. Guard the 8x32 tile assumption — see `## Non-cache correctness defects` for the full analysis.
   This is no longer an open question about a possibly-deliberate idiom: the 8x32 literal on the CBs
   is deliberate, but the op is correct only for an 8x32 input tile and enforces that nowhere, so a
   standard 32x32 input compiles `Sq_chunk_t = 0` into the compute kernel and is expected to hang.
   The small fix is a `TT_FATAL` in `ReduceToRootOp::validate` requiring
   `input_tensor_l.tensor_spec().tile() == tt::tt_metal::Tile({8, 32})`, which costs a few lines,
   breaks no in-tree caller, and runs on hits as well as misses under the substitution branch. The
   large fix is to make the op genuinely general — derive `stats_tile` from the input tile *and*
   rework `PNHt = PNH / q_heads_parallel_factor / tile_height` at
   `device/reduce_to_root_program.cpp:219`, which requires settling what `PNH = 8` denotes. Neither
   needs a hash change; the tile is already in the key. Pick the guard unless the op is meant to
   serve inputs beyond the MLA stats tensors it was written for.
6. Once `input_mux_cores` is hashed, consider dropping the hand-written attribute tuple entirely and
   letting reflection enumerate the members. With `_input_tensor_spec` removed there is nothing left
   to hide, and the default walk cannot drift out of sync with the member list the way a hand-written
   tuple did here.
