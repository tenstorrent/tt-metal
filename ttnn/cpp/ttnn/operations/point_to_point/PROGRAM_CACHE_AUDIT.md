# Program Cache Audit — `point_to_point`

Audit of `ttnn::operations::point_to_point::PointToPointOp`'s hand-written
`attribute_names` / `attribute_values()` against the framework default ("hash everything") key.

| | |
|---|---|
| Device operation | `ttnn::operations::point_to_point::PointToPointOp` (`device/host/point_to_point_device_op.hpp`) |
| Custom hash | **None.** No `compute_program_hash` exists. The omission is done by hand-writing the reflection tuple on `operation_attributes_t` (`device/host/point_to_point_device_op.hpp:29-30`) |
| `operation_attributes_t` | `receive_coord`, `send_coord`, `topology`, `_input_tensor_spec` |
| `attribute_values()` returns | `send_coord`, `receive_coord`, `topology` — **`_input_tensor_spec` is excluded** |
| `tensor_args_t` | `input_tensor`, `optional_output_tensor`, `optional_intermediate_tensor` — no attribute tuple, fully reflected |
| Program factories | `PointToPointOp::SendReceive` (`create_workload_descriptor`, i.e. the declarative `WorkloadDescriptor` variant), building `send_program_factory` + `receive_program_factory`, or `local_copy_program_factory` for the same-device case |
| `override_runtime_arguments` | **No** |
| `get_dynamic_runtime_args` | **No** |
| `validate_on_program_cache_hit` | **No** — so `validate_on_program_cache_miss` is substituted on every hit |
| Cache-hit patch mechanism | Framework **buffer-binding fast path** (`WorkloadDescriptor` variant — no slow-path rebuild) |

The CSV classification is correct on `hash_kind=backdoor` and on the omitted member
(`_input_tensor_spec`), but **wrong on two counts**: it records
`override_runtime_arguments=Y`, and neither `PointToPointOp` nor `SendReceive` defines one — the only
occurrence of the string in this directory is a comment in `send_program_factory.cpp:141` explaining
its *absence*. And its `tensor_input=SPEC-OMITTED` label overstates the omission: see omission 1.

## Validation on the cache-hit path

`PointToPointOp` declares only a miss validator, which delegates to a private `validate`:

```66:79:ttnn/cpp/ttnn/operations/point_to_point/device/host/point_to_point_device_op.hpp
    // Validate the operation when it creates a program.
    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
        validate(operation_attributes, tensor_args);
    };

    // Compute the output shapes based on the operation attributes and tensor args
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    // Create the output tensors based on the operation attributes and tensor args
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

This is the favourable case, and it is the reverse of the intuitive reading: declaring no hit
validator means the op is *fully* validated on hits, whereas declaring a narrow one would have
silently replaced — and so disabled — every miss-time check. Every `TT_FATAL` reached from
`PointToPointOp::validate` (`device/host/point_to_point_device_op.cpp:101`) executes on the offending
call, not merely on the first build.

That matters more for this op than for most, because it is the one mode where the framework refreshes
almost nothing on a hit (see the next section). It does not change any verdict below — no omission
here is constrained by `validate`, so there is no "pinned by validation" row to regrade — but it does
strengthen two arguments that are made later. The sharded-input rejection at
`point_to_point_device_op.cpp:103` and the spec equality checks against `compute_output_specs` are
enforced on every dispatch, so the frozen compile-time args and CB sizes cannot be reached by a
second call that changes shardedness or supplies a mismatched optional output. Had this op defined a
hit validator that omitted those checks, that reasoning would have collapsed to "pinned only on the
miss path".

A CSV `own_hit_validator=N` row therefore does not mean the hit path is unvalidated; under this
branch it usually means the opposite.

## Cache-hit patch mechanism

`SendReceive` provides `create_workload_descriptor`, which makes it a `ProgramDescriptorFactoryConcept`
(`ttnn/api/ttnn/operation_concepts.hpp:69-74`) wrapped by `DescriptorMeshWorkloadAdapter` with
`has_workload_descriptor == true`. That selects the first branch of `apply_descriptor`, which has no
rebuild fallback at all:

```641:656:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
                if constexpr (has_workload_descriptor) {
                    // WorkloadDescriptor variant — declarative: there is no slow-path rebuild
                    // because re-running create_workload_descriptor would re-allocate
                    // workload-scoped resources (GlobalSemaphores, MeshBuffers).
                    // CB bindings are always populated by resolve_bindings, so the
                    // fast path covers cache hits even when the factory only sets
                    // `desc.cbs[i].buffer` and declares no rt-arg buffer bindings.
                    if (!sv.resolved_bindings.empty()) {
                        auto collected =
                            collect_tensor_buffers(tensor_args, tensor_return_value, sv.workload_descriptor);
                        tt::tt_metal::apply_resolved_bindings(program, sv.resolved_bindings, collected.buffers);
                    }
                    // The WorkloadDescriptor variant never rebuilds, so a value a custom hash
                    // excluded would stay frozen at first miss — re-apply declared dynamic args.
                    apply_dynamic_runtime_args_if_declared(
                        program, attrs, tensor_args, tensor_return_value, coordinate_range);
```

`get_dynamic_runtime_args` is not declared, so `apply_dynamic_runtime_args_if_declared` compiles
away. **On a cache hit, the only thing that changes in the cached programs is the set of buffer
addresses registered as `Buffer*` bindings.** Every raw `uint32_t` runtime arg, every compile-time
arg, every CB size and page size, every `CoreRangeSet`, and the fabric routing arguments are frozen
at the first miss.

The three factories do register their addresses correctly: `send_program_factory` pushes
`input_tensor.buffer()` at reader arg 0 and `output_tensors.at(0).buffer()` (the intermediate) at
writer arg 0 (`send_program_factory.cpp:143` and `:179`); `receive_program_factory` pushes the
intermediate at reader arg 3 and the final output at writer arg 0 (`receive_program_factory.cpp:167`
and `:177`); `local_copy_program_factory` pushes input and output at arg 0 of each kernel
(`local_copy_program_factory.cpp:96` and `:103`). The op also ships a custom buffer-enumeration
specialisation so the intermediate/output aliasing of the optionals does not trip the ambiguity bail
in `resolve_bindings` and silently disable the fast path:

```149:156:ttnn/cpp/ttnn/operations/point_to_point/device/host/point_to_point_device_op.hpp
template <>
struct extract_tensor_buffers_t<::ttnn::operations::point_to_point::PointToPointOp::tensor_args_t, void> {
    template <typename Out>
    static void call(
        const ::ttnn::operations::point_to_point::PointToPointOp::tensor_args_t& args, Out& out) {
        out.push_back(args.input_tensor.buffer());
    }
};
```

**Obligation on the hash.** Everything that is not a `Buffer*`-bound address must be a pure function
of the hashed set. That is a demanding contract, and it is the lens for the omission below: the
question is not merely whether the omitted `_input_tensor_spec` is used, but whether it reaches
compile-time args, CB page sizes, core ranges, or fabric topology — none of which a fast-path patch
can repair.

## Baseline: what the default hash would cover

No `compute_program_hash` exists, so the framework takes the reflection branch
(`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:982-987`). A "hash everything" key with no
hand-written tuple would be:

| Source | Fields |
|---|---|
| `operation_attributes` | `receive_coord`, `send_coord`, `topology`, `_input_tensor_spec` |
| `input_tensor` | storage variant kind, `logical_shape`, and `tensor_layout` = { `dtype`, `page_config`, `memory_config`, `alignment` } |
| `optional_output_tensor` | engaged flag plus, when engaged, the same six-field decomposition |
| `optional_intermediate_tensor` | engaged flag plus, when engaged, the same six-field decomposition |

`tensor_args_t` carries no attribute tuple (`device/host/point_to_point_device_op.hpp:33-37`), so all
three tensors are reflected in full.

## What the custom key covers

```21:31:ttnn/cpp/ttnn/operations/point_to_point/device/host/point_to_point_device_op.hpp
    struct operation_attributes_t {
        const MeshCoordinate& receive_coord;
        const MeshCoordinate& send_coord;
        const ::ttnn::ccl::Topology topology;

        // put this in here to hash on tensor spec
        const tt::tt_metal::TensorSpec _input_tensor_spec;

        static constexpr auto attribute_names = std::forward_as_tuple("send_coord", "receive_coord", "topology");
        auto attribute_values() const { return std::forward_as_tuple(send_coord, receive_coord, topology); };
    };
```

Three of the four members are named. Reflection prefers the tuple over member enumeration
(`tt_stl/tt_stl/reflection.hpp:1319-1334`), and so does the canonical-key encoder:

```1499:1500:tt_stl/tt_stl/reflection.hpp
    } else if constexpr (ttsl::reflection::detail::supports_compile_time_attributes_v<T>) {
        std::apply([&out](const auto&... a) { (append_canonical(out, a), ...); }, object.attribute_values());
```

so `_input_tensor_spec` is absent from both halves of the `ProgramCacheKey`.

## Omitted parameters

### 1. `_input_tensor_spec`

**Verdict: VALID — invariant (fully redundant with `tensor_args`).**

The comment above the member ("put this in here to hash on tensor spec") states an intent that
`attribute_values()` then does not carry out — the field is written but never named. The net effect
is nil, because the field is constructed as a copy of a value that `tensor_args` already contributes:

```271:275:ttnn/cpp/ttnn/operations/point_to_point/device/host/point_to_point_device_op.cpp
    using OperationType = ttnn::operations::point_to_point::PointToPointOp;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{receiver_coord, sender_coord, topology, input_tensor.tensor_spec()},
        OperationType::tensor_args_t{input_tensor, optional_output_tensor, optional_intermediate_tensor});
```

`input_tensor` is a reflected member of `tensor_args_t`, and a `Tensor` hashes as
`(storage, tensor_spec())`. So `_input_tensor_spec` is a strict subset of the key, and dropping it
from the attribute tuple removes exactly zero information. Note this is a *de-duplication*, not a
relaxation: unlike an op that swaps `logical_shape` for `padded_shape`, nothing here is coarsened.

Because this is the demanding mode-B case, it is worth verifying the four categories the prompt flags
— compile-time args, CB page sizes, core ranges, and fabric topology — against the hashed set rather
than resting on the subset argument alone.

**Compile-time args.** All three factories derive their compile-time args from the input tensor's
buffer/dtype and from hashed attributes:

```89:106:ttnn/cpp/ttnn/operations/point_to_point/device/host/send_program_factory.cpp
    std::vector<uint32_t> reader_ct_args;
    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(reader_ct_args);

    tt::tt_metal::KernelDescriptor reader_kernel_desc;
    reader_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/point_to_point/device/kernels/dataflow/reader_unary_interleaved_start_id_gen.cpp";
    reader_kernel_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel_desc.core_ranges = all_cores;
    reader_kernel_desc.compile_time_args = std::move(reader_ct_args);
    reader_kernel_desc.config = tt::tt_metal::ReaderConfigDescriptor{};

    const auto this_fabric_id = mesh_device->get_fabric_node_id(send_coord);

    const auto [num_hops, dst_is_forward, next_fabric_id] =
        detail::fabric_1d_routing(mesh_device, send_coord, receive_coord, topology);

    std::vector<uint32_t> writer_ct_args = {sender_cb_id, packet_header_cb_id, packet_cb_id, l1_alignment};
    tt::tt_metal::TensorAccessorArgs(output_tensors.at(0).buffer()).append_to(writer_ct_args);
```

`TensorAccessorArgs` encodes the buffer's layout — buffer type, page/shard structure — all of which
lives inside the hashed `memory_config` and `page_config`. `l1_alignment` is a HAL constant, and the
program cache is already per-device. The intermediate tensor (`output_tensors.at(0)`) is either the
caller's `optional_intermediate_tensor`, which is hashed, or is computed from the input spec plus
`send_coord`/`receive_coord` — see `compute_output_specs`
(`device/host/point_to_point_device_op.cpp:143-178`), whose only inputs are `input_tensor.tensor_spec()`,
`input_tensor.dtype()`, `hal::get_l1_alignment()`, and the two coords.

**CB page sizes and totals.** In the two fabric factories every CB is sized from the outputs of
`compute_aligned_packet_dims(dtype, page_size_bytes, num_pages, alignment)` plus the aligned page
size (`send_program_factory.cpp:28-34`, `:49-86`; `receive_program_factory.cpp:27-34`, `:47-85`); in
the local-copy factory the single CB is `2 * round_up(page_size_bytes, l1_alignment)`
(`local_copy_program_factory.cpp:23-27`, `:42-52`). Every input to those expressions is the tensor's
dtype, page size or page count — all hashed — or the HAL alignment constant.

**Core ranges.** `split_work_to_cores(use_cores, total_packets)` with `use_cores = {1,1}` fixed
(`send_program_factory.cpp:37-40`, `receive_program_factory.cpp:36-40`), and for the local-copy path
`split_work_to_cores(compute_grid, num_pages)` with `compute_grid` a device constant
(`local_copy_program_factory.cpp:33-36`). `total_packets` and `num_pages` derive from the hashed
input spec — including, via `get_num_pages`, its tile shape; see omission 2, which examines that
dependence in detail because it is the one place a structural property of the program varies with the
tile.

**Fabric topology and routing.** `fabric_1d_routing(mesh_device, send_coord, receive_coord, topology)`
(`device/host/point_to_point_device_op.cpp:66-98`) reads only the two coords, the topology and the
mesh shape, all hashed or device-fixed. Its outputs `num_hops`, `dst_is_forward` and `next_fabric_id`
land in frozen raw runtime args and in `append_fabric_connection_rt_args`
(`send_program_factory.cpp:154-175`), which is precisely why the coords and topology must stay in the
key — and they do.

**Which factory runs** is chosen by `send_coord == receive_coord`
(`device/host/point_to_point_device_op.cpp:204`), both hashed, so the local-copy and fabric variants
can never share an entry.

The remaining frozen raw runtime args are the per-core page ranges and sizes — `page_idx_start`,
`page_idx_end`, `increment`, the page and packet sizes, `num_pages_per_packet` and
`num_page_segments` in the fabric factories (`send_program_factory.cpp:126-165`,
`receive_program_factory.cpp:124-151`, `:176-181`), and `core_pages` / `start_id` /
`page_size_bytes` in the local-copy factory (`local_copy_program_factory.cpp:84-107`) — every one a
function of the same hashed spec. The one non-derived value is `semaphore.address()`
(`send_program_factory.cpp:163`, `receive_program_factory.cpp:149`), and that is safe because the
`GlobalSemaphore` is allocated once per cache miss and parked in the workload descriptor so it
outlives the entry:

```242:243:ttnn/cpp/ttnn/operations/point_to_point/device/host/point_to_point_device_op.cpp
    tt::tt_metal::WorkloadDescriptor workload_descriptor;
    workload_descriptor.semaphores.push_back(semaphore);
```

Its address is therefore identical on every dispatch that hits this entry, so freezing it is correct.

### 2. Tile geometry (`page_config` / `Tile`)

**Verdict: VALID — unused (the tile is read, but only through values that are hashed).**

This op *does* perform host-side tile math. A search confined to the `point_to_point` directory finds
none, which is misleading: the arithmetic is inherited from the shared
`data_movement/common` helper the three factories are built on.

```714:720:ttnn/cpp/ttnn/operations/data_movement/common/common.cpp
uint32_t get_num_pages(const ttnn::Tensor& tensor) {
    if (tensor.layout() == ttnn::ROW_MAJOR_LAYOUT) {
        return tt::div_up(tensor.padded_shape().volume(), tensor.padded_shape()[-1]);
    }
    const auto& tile_shape = tensor.tensor_spec().tile().get_tile_shape();
    return tt::div_up(tensor.padded_shape().volume(), tile_shape[0] * tile_shape[1]);
}
```

`get_num_pages` is called from all three factories and from the device operation itself —
`send_program_factory.cpp:28`, `receive_program_factory.cpp:27`,
`local_copy_program_factory.cpp:23` and `point_to_point_device_op.cpp:161` — and the resulting page
count is frozen into the cached program in three places worth naming, since mode B refreshes none of
them:

- **Send factory, packet CB `c_2`.** `packet_size_bytes` becomes both the CB's `total_size` and its
  `page_size` (`send_program_factory.cpp:78-86`), and it comes from `compute_aligned_packet_dims`,
  which takes `num_pages` directly (`point_to_point_device_op.cpp:20-43`).
- **Receive factory, CBs `c_1` and `c_2`.** `c_1` is sized at `packet_size_bytes` and `c_2` at
  `3 * num_pages_per_packet` pages (`receive_program_factory.cpp:64-78`), both derived from the same
  packet dimensions.
- **Local-copy factory, the core ranges.** This is the strongest case, because the dependence is
  structural rather than a size:

```34:36:ttnn/cpp/ttnn/operations/point_to_point/device/host/local_copy_program_factory.cpp
    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_pages_per_core_group_1, num_pages_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(compute_grid, num_pages);
```

  `all_cores` becomes the CB's `core_ranges` (`:46`) and both kernels' `core_ranges` (`:61`, `:72`).
  Halving the tile height doubles `num_pages` and stamps the program onto a different number of
  cores — a property no cache-hit patch of any mode can repair.

So the verdict does not rest on absence of tile math. It rests on two independent facts, both of
which hold.

**First, the tile is genuinely in the cache key.** As established in omission 1, this op defines no
`compute_program_hash`, so the framework hashes `operation_attributes` and `tensor_args` wholesale
through the reflection branch, and `tensor_args_t` carries no attribute tuple. The reflection walk
reaches the tile geometry: `Tensor` hashes as `(storage, tensor_spec())`; `TensorSpec` exposes
`tensor_layout_`; `TensorLayout` names `page_config` among its hashed attributes
(`tt_metal/api/tt-metalium/experimental/tensor/spec/layout/tensor_layout.hpp:75-76`); `PageConfig`
forwards to `config_`
(`tt_metal/api/tt-metalium/experimental/tensor/spec/layout/page_config.hpp:50-51`); the tiled
alternative `TilePageConfig` has the `Tile` as its sole member (`page_config.hpp:23-27`); and `Tile`
hashes its dimensions explicitly:

```46:47:tt_metal/api/tt-metalium/tile.hpp
    static constexpr auto attribute_names = std::forward_as_tuple("tile_shape", "face_shape", "num_faces");
    auto attribute_values() const { return std::forward_as_tuple(tile_shape, face_shape, num_faces); }
```

A `Tile{16, 32}` input therefore produces a different hash and a different canonical key from a
`Tile{32, 32}` input of the same padded shape and dtype, and gets its own cache entry rather than
inheriting the other's core ranges and CB sizes.

**Second, `get_num_pages` is tile-correct.** It divides by the tensor's *actual* `get_tile_shape()`
rather than by `tt::constants::TILE_HW`, so a non-32x32 input yields the right page count and the
freshly built program is correctly sized. There is no factory defect here either, and so nothing to
record under a non-cache heading.

The op is thus correct for a `Tile{16, 32}` input on both counts at once — correctly keyed and
correctly built — which is why it needs no tile guard even though `validate` accepts `Layout::TILE`
without saying anything about tiles (`device/host/point_to_point_device_op.cpp:101-141`).

**Why this still matters to a future reader.** The correctness is contingent on this op continuing
to have no custom hash. The three frozen sites above are real: if anyone later adds a
`compute_program_hash` to `PointToPointOp` — the obvious motivation being to skip the optional output
and intermediate tensors, or to key on `padded_shape` instead of `logical_shape` — and omits
`page_config`, all three become an aliasing bug the same day. A second call with the same padded
shape and dtype but a shorter tile would hit the first call's entry and run a program whose packet
CBs are sized for the wrong page count and, on the local-copy path, whose kernels are placed on the
wrong number of cores. Any such change must keep `page_config` in the key, and the mode-B
classification is what makes that non-negotiable: there is no slow-path rebuild to absorb the error.

One framework-wide qualification on the "the tile is in the key" claim, which applies to every op and
is not a defect of this one. `Tile::attribute_values()` (quoted above) covers `tile_shape`,
`face_shape` and `num_faces` but not `transpose_within_face` or `transpose_of_faces`, and
`Tile::operator==` excludes them too:

```122:124:tt_metal/impl/data_format/tile.cpp
bool Tile::operator==(const Tile& other) const {
    return tile_shape == other.tile_shape && face_shape == other.face_shape;
}
```

Since `attribute_values()` drives the hash and `operator==` drives canonical-key collision
resolution, the transpose flags are invisible to both halves of the key framework-wide. This costs
nothing here — no code in this op or in `get_num_pages` reads either flag, so the program does not
vary with them — but "hashing `page_config` covers the tile" should be read as covering the shapes
and face count, not transpose. Closing that hole would require an explicit `TT_FATAL` on
`get_transpose_within_face()` / `get_transpose_of_faces()`, which is a framework-level decision
rather than something to add here.

Two include-driven dead ends, recorded so they are not re-investigated: `moe_utils.hpp` is included
at `send_program_factory.cpp:6` but no symbol from it is used anywhere in that file (the only live
use of the name in this op is the unrelated device-side `kernels/moe_utils.hpp` in
`device/kernels/dataflow/writer_send.cpp:9`), and the `ccl_common.hpp` include
(`device/host/point_to_point_device_op.hpp:7`, `.cpp:8`) resolves only to the `ccl::Topology` enum —
no `CclOpTensorConfig` is constructed anywhere in the op.

## Keys the custom key adds beyond the default

None. The hand-written tuple is strictly a subset of the members.

## Framework side effect of having a custom hash

Not applicable: this op defines no `compute_program_hash`, so it stays on the full canonical-key path
(`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:1014-1020`) and a 64-bit hash collision between two
distinguishable configurations resolves to a rebuild.

One property of the attribute-tuple backdoor is worth stating even though it costs nothing here: an
excluded member is excluded from the canonical key as well as the hash (quoted above), so unlike a
hash-only gap there is no exact-comparison fallback. That is benign for `_input_tensor_spec` because
the information is duplicated elsewhere in the key, but it means the tuple is not a place to hide
anything that actually matters.

## Summary

| Omitted vs. default | Used by program? | Patched on hit? | Verdict |
|---|---|---|---|
| `_input_tensor_spec` | Yes, but only via `tensor_args.input_tensor`, which is hashed in full | n/a | VALID — invariant |
| Tile geometry (`page_config` / `Tile`) | **Yes** — via the shared `data_movement::get_num_pages`, which freezes into packet CB sizes and the local-copy core ranges | n/a — reaches the key in full via `tensor_args.input_tensor` | VALID — invariant (not an omission: the tile is in the key) |

**No program-cache correctness bug was found.** The single backdoor omission is a duplicate of
information that `tensor_args` already carries, so the effective key is identical to the default
"hash everything" key. That matters more here than for most ops, because the `WorkloadDescriptor`
variant has no slow-path rebuild: only `Buffer*`-bound addresses are refreshed on a hit, and every
compile-time arg, CB size, core range, fabric routing value and per-core page range in the three
factories was checked above and is a function of {`send_coord`, `receive_coord`, `topology`, input
`dtype`/`logical_shape`/`page_config`/`memory_config`/`alignment`, and the optional output and
intermediate specs when supplied} plus device-fixed constants.

## Recommendations

1. Delete `_input_tensor_spec` and the hand-written `attribute_names` / `attribute_values()` pair
   entirely, letting reflection enumerate `receive_coord`, `send_coord` and `topology` directly. All
   three are already hashable, the resulting key is unchanged, and a hand-written tuple that silently
   drops a newly added member is exactly the failure mode that put `input_mux_cores` outside the key
   in the sibling `ccl/reduce_to_root` op. If the member must stay for lifetime reasons, at least
   replace its comment — "put this in here to hash on tensor spec" is the opposite of what the code
   does.
2. Correct the source audit CSV row on two columns: `point_to_point` has no
   `override_runtime_arguments` — its cache-hit safety rests entirely on `Buffer*` bindings, a
   materially weaker guarantee — and it is not a "no host-side tile math" op. Its tile arithmetic is
   inherited from the shared `data_movement::get_num_pages`, which a directory-scoped sweep cannot
   see. The classification is benign here, but any sweep entry derived the same way should be
   re-confirmed against where the op's factory actually reads its page counts.
3. If a `compute_program_hash` is ever added to `PointToPointOp`, keep `page_config` in it. Omission 2
   names the three sites that would alias immediately without it: the send factory's packet CB
   sizing, the receive factory's `c_1`/`c_2` sizing, and the local-copy factory's core ranges. A short
   comment on the op stating that `page_config` is load-bearing would be cheaper than rediscovering
   this.
4. Enable `-DTT_DESCRIPTOR_PATCHING_PARITY_CHECK` for this op in CI. The mode-B argument above is a
   manual audit of three factories; `assert_fastpath_parity`
   (`tt_metal/impl/program/program_descriptor_patching.cpp:270-349`) turns it into an enforced
   invariant and would catch any future factory change that introduces a frozen arg not derivable
   from the key.
5. If sharded input support is ever added (today `PointToPointOp::validate` rejects it —
   `device/host/point_to_point_device_op.cpp:103`), re-audit: `TensorAccessorArgs` would then encode
   shard structure into compile-time args, and any CB pinned to a shard would need its `.buffer`
   binding checked against the fast path. Any replacement guard belongs in
   `PointToPointOp::validate`, the function reached from `validate_on_program_cache_miss`, which is
   what the dispatcher substitutes on hits while this op declares no
   `validate_on_program_cache_hit` (`ttnn/api/ttnn/device_operation.hpp:262-266`).
6. Do not add a `validate_on_program_cache_hit` to this op unless it delegates to
   `validate_on_program_cache_miss`. Defining one takes the first dispatcher branch and *replaces*
   the miss validator on hits, which would silently drop the sharded-input rejection, the mesh
   containment checks and the optional-output spec equality from every cached dispatch — precisely
   the checks that keep this op's frozen compile-time args and CB sizes unreachable by a mismatched
   second call.
