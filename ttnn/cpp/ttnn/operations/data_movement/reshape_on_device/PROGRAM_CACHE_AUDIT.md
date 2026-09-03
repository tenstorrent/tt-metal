# Program Cache Audit — `data_movement/reshape_on_device`

Audit of `ttnn::prim::ReshapeDeviceOperation::compute_program_hash` against the framework
default ("hash everything") key.

| | |
|---|---|
| Device operation | `ttnn::prim::ReshapeDeviceOperation` (`device/reshape_op.hpp`) |
| Custom hash | `device/reshape_op.cpp:109` |
| `operation_attributes_t` | `ReshapeOnDeviceParams` — `logical_output_shape`, `padded_output_shape`, `output_mem_config` |
| `tensor_args_t` | `ReshapeOnDeviceInputs` — `input_tensor` |
| Program factories | `ReshapeTileProgramFactory`, `ReshapeRMProgramFactory` (both `ProgramDescriptor`-based) |
| `override_runtime_arguments` | **No** |
| `get_dynamic_runtime_args` | **No** |
| Cache-hit patch mechanism | Framework **buffer-binding fast path** |

## Post-fix status — commit fab067a

**CLEAR — with a justified relaxation.** The one program-cache bug this audit found (omission #2, the
unguarded 32x32 tile assumption) is fixed. Every remaining omission is now either pinned by a
`TT_FATAL` that executes on the hit path or functionally determined by a hashed term, with the single
exception of `input.logical_shape()`, which is a genuine relaxation: the key carries `padded_shape`
instead, which is exactly the projection of the input shape both factories observe. Zero remaining
bugs.

What `fab067a` changed in this op:

- Added a 32x32 tile guard to the TILE branch of `validate_on_program_cache_miss`
  (`device/reshape_op.cpp:48-58`). Because the op declares no `validate_on_program_cache_hit`, the
  dispatcher substitutes the miss validator on hits (`ttnn/api/ttnn/device_operation.hpp:265-269`), so
  the guard is live on the offending call, not just on the build call. This is the minimal fix this
  audit recommended (recommendation 1) rather than the tile-aware-factory alternative.
- Documented the constraint on the public API: `reshape_nanobind.cpp:52` now reads "A TILE layout
  tensor must use the standard 32x32 tile."
- `compute_program_hash` was **not** touched. The eight terms are unchanged.

What remains open:

- The `Tile` transpose flags (`get_transpose_within_face()` / `get_transpose_of_faces()`) are still
  unguarded, as recommendation 1 anticipated. This is now harmless rather than latent: with
  `tile_shape` pinned to 32x32, `Tile`'s constructor derives every other field from it
  (`tt_metal/impl/data_format/tile.cpp:36-68`) and `Tile::get_tile_size` ignores the transpose flags
  entirely (`tt_metal/impl/data_format/tile.cpp:70-118`), so nothing either factory reads can move.
  It is a framework-wide gap, not a reshape defect — the flags reach neither the default hash nor the
  canonical key.
- Recommendation 2 (drop the redundant `program_factory.index()`) was not taken. No correctness
  impact either way.
- Recommendation 3 stands: this op still has no sharded path, so re-audit if one is added.

**Metal 2.0 port: clear**, with one flag to declare. The logical-vs-padded relaxation below is not
ad-hoc — it is exactly the shape of `TensorSpecRelaxations::match_padded_shape_only`
(`tt_metal/api/tt-metalium/experimental/metal2_host_api/tensor_spec_relaxations.hpp:41`), which
`pertinent_fields` maps to `PertinentFields{.padded_shape = true}`
(`tt_metal/impl/metal2_host_api/tensor_spec_relaxations.cpp:77-79`). The input `TensorParameter` must
declare it, or a second call differing only in logical shape will hit the key and then be *rejected*
by `ValidateTensorArgs` (`tt_metal/impl/metal2_host_api/program_run_args.cpp:176-189`) instead of
reusing the program. Declaring it is safe: `hash_tensorspec_with_relaxation`
(`tt_metal/impl/metal2_host_api/tensor_spec_relaxations.cpp:116`) and
`tensorspecs_match_with_relaxation` (`:161-201`) derive their field set from the same
`pertinent_fields` call, so the key and the run-time predicate cannot disagree. Note the flag is
*tighter* than this op's current key: it pins the whole `tensor_layout()` — including `page_config`
and `Alignment`, which the custom hash drops — so porting strengthens the key rather than weakening
it, and omissions #2 and #3 below become moot by construction.

## Cache-hit patch mechanism (what actually gets refreshed)

Both factories declare their address slots through `emplace_runtime_args({buffer, ...})`, which
produces `resolved_bindings.rt_args`. In `MeshDeviceOperationAdapter`'s descriptor cache-hit path
this selects the fast path:

```735:740:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
                    if (!sv.resolved_bindings.rt_args.empty() ||
                        (!dynamic_args.empty() && !sv.resolved_bindings.empty())) {
                        auto collected =
                            collect_tensor_buffers(tensor_args, tensor_return_value, sv.workload_descriptor);
                        tt::tt_metal::apply_resolved_bindings(program, sv.resolved_bindings, collected.buffers);
                        tt::tt_metal::apply_dynamic_runtime_args(program, dynamic_args);
```

**Consequence for this audit:** on a cache hit, *only* the input/output buffer addresses are
re-patched. Every other runtime arg — stick counts, tile counts, `curr_sticks_read/write`
prefix offsets, output shape dims — is frozen at the value computed on the first miss. So every
quantity that feeds a non-address runtime arg or a compile-time arg **must** be captured by the
hash, either directly or via a value it is derived from.

## Which validator runs on a cache hit

This decides omission #4 below, and it runs the opposite way to the intuitive reading, so it is
worth settling before the omissions. The dispatcher runs exactly *one* validator on a hit:

```265:269:ttnn/api/ttnn/device_operation.hpp
    if constexpr (HasValidateOnProgramCacheHit<mesh_device_operation_t>) {
        mesh_device_operation_t::validate_on_program_cache_hit(operation_attributes, tensor_args);
    } else {
        mesh_device_operation_t::validate_on_program_cache_miss(operation_attributes, tensor_args);
    }
```

The mesh adapter mirrors the same rule for the adapted type:

```229:235:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
    static void validate_on_program_cache_hit(const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
        if constexpr (HasValidateOnProgramCacheHit<DeviceOperation>) {
            DeviceOperation::validate_on_program_cache_hit(attrs, tensor_args);
        } else {
            DeviceOperation::validate_on_program_cache_miss(attrs, tensor_args);
        }
    }
```

`ReshapeDeviceOperation` declares **no** `validate_on_program_cache_hit` — `device/reshape_op.hpp:22-35`
declares only `validate_on_program_cache_miss`, at `:25-26`, defined at `device/reshape_op.cpp:27`. That
places the op in the favourable branch: the framework substitutes the *miss* validator on every hit, so
every `TT_FATAL` in it executes on the offending call rather than only on the build call. The pins that
therefore hold on the hit path are the storage-type and non-null-buffer checks
(`device/reshape_op.cpp:30-31`), the BFLOAT16-or-FLOAT32 dtype check (`:32-35`), the TILE-or-ROW_MAJOR
layout check (`:37-39`), the INTERLEAVED memory-layout checks on both the input and the output
memory config (`:41-46`), and — since `fab067a` — the 32x32 tile geometry check on the TILE path
(`:53-58`). A "VALID — pinned by validation" verdict resting on any of these is legitimate,
not miss-only; that is what licenses omission #4, and post-`fab067a` it is also what licenses
omission #2.

The corollary bounds the other direction: whatever the miss validator does *not* check is unpinned on
both paths. Pre-`fab067a` the validator never inspected the tile geometry, which is what made
omission #2 a bug rather than a caveat; `fab067a` closed exactly that gap.

## Baseline: what the default hash would cover

`ttsl::hash::hash_objects_with_default_seed(type_hash<ReshapeDeviceOperation>, attrs, tensor_args)`
walks reflection, so the default key is:

| Source | Fields |
|---|---|
| `operation_attributes` | `logical_output_shape`, `padded_output_shape`, `output_mem_config` |
| `input_tensor.storage` | storage variant kind (`DeviceStorage` / `HostStorage`; both have empty attribute tuples) |
| `input_tensor.tensor_spec` | `logical_shape`, and `tensor_layout` = { `dtype`, `page_config`, `memory_config`, `alignment` } |

Note the default key does **not** contain `padded_shape` directly — it is derived from
`logical_shape` + `page_config` + `alignment`.

## What the custom hash covers

```114:122:ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/device/reshape_op.cpp
    return operation::hash_operation<ReshapeDeviceOperation>(
        operation_attributes.logical_output_shape,
        operation_attributes.padded_output_shape,
        operation_attributes.output_mem_config,
        program_factory.index(),
        input_tensor.dtype(),
        input_tensor.memory_config(),
        input_tensor.layout(),
        input_tensor.padded_shape());
```

All three `operation_attributes` fields are kept. The input tensor is decomposed selectively.

## Omitted parameters

### 1. `input_tensor.logical_shape()` — replaced by `padded_shape()`

**Verdict: VALID — relaxation win.**

Neither factory reads the logical shape. The RM factory derives its whole work split from the
padded shape and element size:

```40:45:ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/device/reshape_rm_program_factory.cpp
    uint32_t num_old_sticks =
        input_tensor.padded_shape()[0] * input_tensor.padded_shape()[1] * input_tensor.padded_shape()[2];
    uint32_t num_new_sticks = output_shape[0] * output_shape[1] * output_shape[2];

    uint32_t old_stick_size = input_tensor.padded_shape()[3] * input_tensor.element_size();
    uint32_t new_stick_size = output_shape[3] * output_tensor.element_size();
```

The tile factory likewise uses `physical_volume()` and `padded_shape()[3]`. The output side is
driven entirely by `operation_attributes.padded_output_shape` (hashed) and the input dtype
(hashed), because `compute_output_specs` builds the output spec from the op attributes rather than
from the input logical shape.

Swapping `logical_shape` for `padded_shape` is the *right* trade here: `padded_shape` is exactly
the projection of `logical_shape` that the kernels observe. Two calls with logical `[1,1,3,32]`
and `[1,1,5,32]` (both padding to `[1,1,32,32]` in tile layout) legitimately share one program —
the default hash would have forced a needless recompile. The freshly-computed output `TensorSpec`
still carries the correct per-call logical shape, since `compute_output_specs` /
`create_output_tensors` run on every invocation, hit or miss.

Post-`fab067a` this is the only omission in this document that is a relaxation rather than a
zero-value drop, and two qualifications belong with it. First, it is **TILE-path only**: for
ROW_MAJOR interleaved the default `Alignment` is `{1}`
(`tt_metal/impl/tensor/spec/layout/page_config.cpp:47-57`), so `padded_shape == logical_shape` and
the relaxation is vacuous there. Second, reshape preserves logical volume, so the extra hits it buys
are narrower than the example above suggests: `logical_output_shape` is hashed, so the two calls must
also agree on the reshape target — logical `[1,1,2,16]` and `[1,1,16,2]` (both padding to
`[1,1,32,32]`, both of volume 32) reshaped to the same `[1,1,4,8]`. The op does not **need** the
relaxation; see the summary and recommendation 4 for what removing it would cost.

### 2. `input_tensor.tensor_spec().page_config()` — only `layout()` is hashed

**Verdict: RESOLVED by fab067a** (was BUG).

`fab067a` added a `TT_FATAL` rejecting any tile other than 32x32 to the TILE branch of
`validate_on_program_cache_miss` (`device/reshape_op.cpp:53-58`), which — because this op declares no
`validate_on_program_cache_hit` — the dispatcher substitutes onto the hit path as well. The hash was
deliberately left alone: with `Tile` pinned, omitting `page_config` is correct by construction, which
is the minimal route recommendation 1 below sets out. The `Tile` value can no longer vary, so the
omission is now a category-3 zero-value omission rather than a defect.

Pre-fix: the tile factory was only correct for 32x32 tiles, nothing validated that, and
the tile geometry was not hashed — so a non-32x32 input silently inherited a wrong cached program.

The third clause is the one that made this a *cache* defect rather than only a factory defect, so it
is worth demonstrating rather than asserting. `compute_program_hash` (`device/reshape_op.cpp:109-123`)
hashes exactly eight terms, unchanged by `fab067a`:

```114:122:ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/device/reshape_op.cpp
    return operation::hash_operation<ReshapeDeviceOperation>(
        operation_attributes.logical_output_shape,
        operation_attributes.padded_output_shape,
        operation_attributes.output_mem_config,
        program_factory.index(),
        input_tensor.dtype(),
        input_tensor.memory_config(),
        input_tensor.layout(),
        input_tensor.padded_shape());
```

`page_config()` — the only carrier of `Tile` — appears nowhere in that list, and `layout()` is just the
`Layout` enum (`TILE` / `ROW_MAJOR`), not the geometry. `program_factory.index()` is selected from that
same enum (`device/reshape_op.cpp:19-25`), so it adds nothing either. The tile is therefore genuinely
absent from the key, and two tensors differing only in `Tile` land on the same cache entry.

`layout()` collapses `PageConfig` down to `ROW_MAJOR` vs `TILE`, discarding the `Tile` shape and
face/transpose configuration. The tile factory sizes its CB from the *architectural* tile and
indexes with the global tile constants, never `input_tensor.tensor_spec().tile()`:

```30:39:ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/device/reshape_tile_program_factory.cpp
    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    Buffer* src0_buffer = input_tensor.buffer();

    auto output_shape = output_tensor.padded_shape();

    // Tile count must come from the output tensor to match what the reader produces; deriving it
    // from the input deadlocks when the input and output have different tile padding.
    uint32_t num_tiles = output_tensor.physical_volume() / tt::constants::TILE_HW;
```

(The `num_tiles` line reads `output_tensor` rather than `input_tensor` since `c4bc9b32ae5`, the
tile-reshape deadlock fix, which also added the equal-padded-volume `TT_FATAL` at
`device/reshape_op.cpp:73-79`. That commit predates `fab067a`; it changed which padded volume the
factory reads, not whether the tile is hardcoded.)

`tt::tile_size(format)` returns the byte size of a 32x32 tile of that format; the tile-aware API is
`tile.get_tile_size(format)`. Likewise `TILE_HW`, `TILE_WIDTH` and `TILE_HEIGHT` are the
architectural constants, not this tensor's geometry. Non-32x32 tiles are a supported TTNN
configuration — sibling data-movement factories such as `untilize`, `transpose` and `slice` read
`tensor_spec().tile().get_tile_shape()` precisely because of it.

Pre-fix, validation did not close the gap. It constrained the layout enum but never the tile
geometry:

```37:39:ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/device/reshape_op.cpp
    TT_FATAL(
        input_tensor_a.layout() == Layout::TILE || input_tensor_a.layout() == Layout::ROW_MAJOR,
        "Only tile and row major reshape supported!");
```

The two defects compounded, and that is what made this a cache bug rather than only a factory bug.
Taken alone, the hardcoding would at least produce a fresh (wrong) program per call. Because the
tile geometry was also absent from the hash, a `Tile{16, 32}` input reused the cache entry built for
a `Tile{32, 32}` input of the same padded shape and dtype, so the CB page size, `num_tiles`, and the
reader's `padded_shape[3] / TILE_WIDTH` argument were all silently those of the 32x32 program. The
symptom was wrong data or a hang, with no cache miss to hint at the cause.

The right fix is the guard, not a hash change, and that is the route `fab067a` took. Once `Tile` is
constrained to 32x32, omitting `page_config` from the key becomes correct by construction — and the
constraint is total, not partial: `Tile`'s only public constructor takes `(tile_shape,
transpose_tile)` and derives `face_shape`, `num_faces`, `tile_hw`, `face_hw`, `partial_face` and
`narrow_tile` from `tile_shape` (`tt_metal/impl/data_format/tile.cpp:36-68`), so pinning 32x32 pins
every field that reaches a page size or a CB. The shipped check matches the canonical form the
codebase already had:

```95:97:ttnn/cpp/ttnn/operations/data_movement/sharded/interleaved_to_sharded/device/interleaved_to_sharded_op.cpp
        auto tile = input_tensor.tensor_spec().tile();
        if (tile.get_height() != tt::constants::TILE_HEIGHT || tile.get_width() != tt::constants::TILE_WIDTH) {
            return {false, fmt::format("interleaved_to_sharded requires standard 32x32 tiles, got {}x{}", tile.get_height(), tile.get_width())};
```

If the factory is ever made genuinely tile-aware instead, `page_config` must be added to the hash
at the same time and the guard relaxed in the same change. Both routes have one gap in common —
neither a height/width guard nor a hashed `page_config` covers the tile's transpose flags, which are
invisible to the cache framework-wide, so `fab067a`'s guard does not cover them either. Under the
guard that gap is inert rather than latent: `Tile::get_tile_size` never reads the flags
(`tt_metal/impl/data_format/tile.cpp:70-118`) and neither factory reads them at all. Recommendation 1
sets out why, and what to add if the op ever becomes tile-aware.

### 3. `input_tensor.tensor_spec().tensor_layout().get_alignment()`

**Verdict: VALID — unused.**

Nothing pins `Alignment`: the miss validator never mentions it, so a `TensorLayout` carrying an
explicit non-canonical `Alignment` is reachable through the public API with no enforced constraint
standing in the way. That reachability does not create a defect here, because neither factory reads
any quantity that `Alignment` can move independently of a hashed value.

`Alignment` influences the hashed values rather than the program directly. It is one of the inputs to
`padded_shape`, which is hashed explicitly, and every shape-derived quantity the factories read is a
projection of that same padded shape rather than of the alignment. `physical_volume()`, which drives
the tile factory's `num_tiles`, is literally the padded volume:

```437:438:ttnn/core/tensor/tensor.cpp
uint64_t Tensor::logical_volume() const { return logical_shape().volume(); }
uint64_t Tensor::physical_volume() const { return padded_shape().volume(); }
```

and the RM factory's stick counts and stick sizes come from `padded_shape()[0..3]` and
`element_size()` (dtype, hashed). The buffer page size that reaches the kernels as a
`TensorAccessorArgs` compile-time arg is fixed by `page_config` + `dtype` for TILE and by
`padded_shape[-1]` * `element_size` for ROW_MAJOR — all hashed, except `page_config`, whose absence is
omission #2's finding rather than this one's.

The one alignment-*named* quantity the tile factory reads is a different thing entirely, the DRAM/L1
alignment:

```60:61:ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/device/reshape_tile_program_factory.cpp
    bool src0_is_dram = src0_buffer->buffer_type() == BufferType::DRAM;
    uint32_t alignment = src0_is_dram ? hal::get_dram_alignment() : hal::get_l1_alignment();
```

which is a HAL/device constant selected by `buffer_type()` — and `buffer_type` lives inside the hashed
`memory_config`. It is not `TensorLayout`'s `Alignment` at all. It also gates the second CB and the
reader's first compile-time arg, both of which are therefore pinned by `memory_config` plus the
per-device cache partition.

As severity context only, and not as part of the verdict: no in-tree call path constructs a reshape
input with a hand-supplied `Alignment` today, so even if a residual dependence were later introduced
it would start out latent.

### 4. `input_tensor.storage` variant kind (device vs host)

**Verdict: VALID — pinned by validation.**

```30:31:ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/device/reshape_op.cpp
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to reshape need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr, "Operands to reshape need to be allocated in buffers on device!");
```

The parameter is constant across every admissible call, so it carries no information.

This verdict is stronger than the bare `TT_FATAL` makes it look, and the reason is the substitution
branch established above. Those two checks live in `validate_on_program_cache_miss`, and a pin that only
ran on misses would be worth no more than "pinned only on the miss path" — the offending call is by
definition the one that *hits*. Because `ReshapeDeviceOperation` declares no
`validate_on_program_cache_hit`, the dispatcher runs the miss validator on hits too:

```265:269:ttnn/api/ttnn/device_operation.hpp
    if constexpr (HasValidateOnProgramCacheHit<mesh_device_operation_t>) {
        mesh_device_operation_t::validate_on_program_cache_hit(operation_attributes, tensor_args);
    } else {
        mesh_device_operation_t::validate_on_program_cache_miss(operation_attributes, tensor_args);
    }
```

so a host-storage tensor is rejected on the call that would have inherited the cached program, not
merely on the call that built it. The same substitution carries the dtype, TILE-or-ROW_MAJOR layout and
INTERLEAVED memory-layout pins onto the hit path, which is why `dtype`, `layout` and `memory_config`
never need to defend themselves against out-of-range values in the omissions above.

The property is worth protecting: adding even a narrow `validate_on_program_cache_hit` to this op would
*replace* the miss validator on hits and silently drop all four pins, downgrading this verdict and
weakening three others.

### 5. Buffer addresses (omitted by both the default hash and this one)

**Verdict: VALID — patched, and required.** Addresses must not be hashed — that is the whole point of the
cache. They are re-patched on hit via the `resolved_bindings` fast path shown above, which is
sound here because the factories register every address slot as a `Buffer*`
(`{src0_buffer, ...}` / `{dst_buffer, ...}`), including the deliberate `0u` placeholder for idle
cores that intentionally registers no binding:

```176:184:ttnn/cpp/ttnn/operations/data_movement/reshape_on_device/device/reshape_rm_program_factory.cpp
        const bool reader_idle = num_old_sticks_per_core == 0;
        if (reader_idle) {
            reader_desc.emplace_runtime_args(
                core,
                {0u,
                 num_old_sticks_per_core_read,
                 num_old_sticks_read_per_barrier,
                 num_old_sticks_per_cb_push,
                 curr_sticks_read});
```

Idle/active core membership is a function of the hashed shapes, so an entry can never be reused
with a different idle set.

## Keys the custom hash adds beyond the default

- `program_factory.index()` — redundant with `input_tensor.layout()`, since `select_program_factory`
  branches on exactly that. Harmless, and self-documenting.
- `input_tensor.padded_shape()` — not in the default key (derived there). Adding it is what makes
  dropping `logical_shape` safe.

## Framework side effect of having *any* custom hash

Defining `compute_program_hash` opts this op out of attribute-level hash-collision resolution:

```1035:1037:ttnn/api/ttnn/mesh_device_operation_adapter.hpp
        if constexpr (requires { DeviceOperation::compute_program_hash(attrs, tensor_args); }) {
            return key;  // custom hash -> opt out beyond the op-identity prefix
        } else {
```

`ProgramCacheKey::canonical` degrades to just the op type name, so a 64-bit hash collision between
two different reshape configurations resolves to a (wrong) hit instead of a rebuild. This is
inherent to every custom-hash op, not specific to reshape, but it raises the cost of a hash bug
here relative to a default-hashed op.

## Summary

| Omitted vs. default | Used by program? | Patched on hit? | Verdict |
|---|---|---|---|
| `input.logical_shape` | No (padded shape used instead) | n/a | VALID — relaxation win (TILE path only) |
| `input.page_config` (`Tile`) | Yes, but only as a hardcoded 32x32 | No | **RESOLVED by fab067a** — pinned by the 32x32 guard, live on hits |
| `input.tensor_layout.alignment` | Only via hashed derivatives | n/a | VALID — unused |
| `input.storage` kind | n/a | n/a | VALID — pinned by validation (on hits too) |
| Buffer addresses | Yes | Yes (`resolved_bindings`) | VALID — patched |

**Zero program-cache bugs remain.** The one this audit found — the unguarded 32x32 tile assumption
(omission #2), under which a `Tile{16, 32}` input reused the program built for a `Tile{32, 32}` input
of the same padded shape and dtype — was fixed by `fab067a`'s guard at `device/reshape_op.cpp:53-58`.

The rest of the key is sound. Every non-address runtime arg and every
compile-time arg in both factories is a function of the hashed set
{`logical_output_shape`, `padded_output_shape`, `output_mem_config`, input `dtype`,
input `memory_config`, input `layout`, input `padded_shape`} plus the now-pinned 32x32 tile and
device-fixed constants (compute grid, HAL alignments) that the per-device cache already partitions
on.

The one omission that remains a genuine *relaxation* rather than a zero-value drop is
`input.logical_shape` (omission #1), and it applies to the TILE path alone: on ROW_MAJOR interleaved
the default `Alignment` is `{1}` (`tt_metal/impl/tensor/spec/layout/page_config.cpp:47-57`), so
`padded_shape == logical_shape` and hashing one is hashing the other. It is not *needed* — no
in-tree caller varies the input logical shape under a fixed padded shape and a fixed reshape target,
so adding `logical_shape()` back would cost zero real misses and would restore the collision
backstop discussed above. Worth doing as part of the Metal 2.0 port, where the alternative is
declaring `match_padded_shape_only` on the input binding.

## Recommendations

1. **Fixed by `fab067a`.** ~~Add~~ Added a `TT_FATAL` in `validate_on_program_cache_miss` rejecting
   any `Tile` other than 32x32, matching the wording used by `interleaved_to_sharded`
   (`device/reshape_op.cpp:53-58`). This was the minimal fix and
   it makes omission #2 correct by construction. It went in the miss validator specifically: the op has
   no hit validator, so the substitution branch runs it on hits as well, at a cost of two integer
   comparisons per dispatch. The hole was deliberately *not* closed by adding a hit validator — that
   would replace the miss validator on hits and drop the four pins omission #4 rests on.

   The alternative — making both factories genuinely tile-aware via `tile.get_tile_size(format)` and
   `tile.get_tile_shape()` — requires adding `page_config` to the hash in the same change. If you take
   that route, be aware that hashing `page_config` does **not** close the tile hole completely.
   `Tile::attribute_values()` exposes only the tile shape, face shape and face count:

```46:47:tt_metal/api/tt-metalium/tile.hpp
    static constexpr auto attribute_names = std::forward_as_tuple("tile_shape", "face_shape", "num_faces");
    auto attribute_values() const { return std::forward_as_tuple(tile_shape, face_shape, num_faces); }
```

   and `Tile::operator==`, which drives canonical-key collision resolution, compares even less:

```122:124:tt_metal/impl/data_format/tile.cpp
bool Tile::operator==(const Tile& other) const {
    return tile_shape == other.tile_shape && face_shape == other.face_shape;
}
```

   `transpose_within_face` and `transpose_of_faces` reach neither the hash nor the tiebreaker, so two
   tiles differing only in a transpose flag are indistinguishable to the cache no matter what is
   hashed. This is framework-wide and not something reshape's custom hash introduced, but it means a
   tile-aware reshape would still need an explicit `TT_FATAL` on
   `get_transpose_within_face()` / `get_transpose_of_faces()` alongside the hash change. The 32x32
   guard `fab067a` shipped has the same gap — it checks height and width only — so if the tile-aware
   route is ever taken, guard the transpose flags too. Under the guard as shipped the gap is inert,
   since nothing either factory reads depends on the flags.
2. Consider dropping `program_factory.index()` (already implied by `layout()`), or keep it as
   documentation — no correctness impact either way. Not taken by `fab067a`.
3. If this op ever gains a sharded path, `memory_config` alone will no longer pin the
   `TensorAccessorArgs` compile-time layout; re-audit at that point. Still open — the op remains
   INTERLEAVED-only on both sides (`device/reshape_op.cpp:41-46`).
4. Before the Metal 2.0 port, either add `input_tensor.logical_shape()` to the key or declare
   `match_padded_shape_only` on the input `TensorParameter` — see the post-fix status section. Doing
   neither leaves a key that hits where `ValidateTensorArgs` then rejects.
