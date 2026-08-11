# Program-cache audit — cross-cutting summary

Covers 24 ops. Each has a `PROGRAM_CACHE_AUDIT.md` in its own directory under this tree with the
full per-parameter analysis; this document records only what is visible across ops and not from any
one of them. Paths below are relative to `ttnn/cpp/ttnn/operations/` unless stated otherwise.

Each audit compares what the op actually hashes against the framework default "hash everything" key,
and adjudicates every difference. 198 omissions were adjudicated in total: 121 valid, 27 caveats,
48 distinct cache defects.

**A BUG here means a program-cache defect specifically**: the parameter changes the compiled program,
is not refreshed on a cache hit, and is not in the cache key. A factory that builds a wrong program
on a *correctly keyed* miss is a real defect but a different one, recorded separately in the affected
documents so the counts stay comparable.

---

## Counts

48 distinct cache defects across 18 of the 24 ops. Three ops also carry a factory defect recorded
outside these counts (`experimental/ccl/strided_reduce_scatter_async`,
`experimental/deepseek_prefill/rotary_embedding_indexed`, `ccl/reduce_to_root`).

These are *distinct defects*, which is not the same as the number of BUG rows in the per-op tables.
Some defects are reachable through more than one parameter and get a row each; one subsection covers
two independent defects. Where the two differ, the per-op document is authoritative and the
divergence is footnoted below.

| Op | Bugs | Op | Bugs |
| --- | --- | --- | --- |
| `experimental/ccl/llama_reduce_scatter_matmul` | 8 | `data_movement/sharded/interleaved_to_sharded` | 1* |
| `experimental/ccl/llama_all_gather_matmul_async` | 5 | `experimental/ccl/strided_reduce_scatter_async` | 2 |
| `experimental/test/prefetcher_consumer` | 4† | `experimental/deepseek_prefill/moe_padding_config` | 1 |
| `data_movement/roll` | 3 | `ccl/reduce_to_root` | 1 |
| `data_movement/sharded_partial/interleaved_to_sharded_partial` | 3 | `pool/generic` | 1 |
| `data_movement/slice` | 3 | `data_movement/reshape_on_device` | 1 |
| `eltwise/unary_backward/tanh_bw` | 3 | `experimental/fusion` | 1 |
| `experimental/ccl/llama_reduce_scatter_create_heads` | 3 | `point_to_point` | 0 |
| `generic` | 4‡ | `experimental/topk_large_indices` | 0 |
| `experimental/deepseek_prefill/update_padded_kv_cache` | 2 | `.../pack_scaled_fp8_kv_cache` | 0 |
| `experimental/deepseek_prefill/zero_padded_kv_cache` | 2 | `.../per_token_cast_back`, `.../per_token_cast_to_fp8` | 0 |
| | | `.../rotary_embedding_indexed` | 0 |

\* `interleaved_to_sharded` carries two BUG verdicts for one defect, reached through two parameters.
† `prefetcher_consumer` is two ops (1 defect in the consumer, 3 in the validator) and its tables show
eight rows, the extras marked "same root cause" or "subsumed".
‡ `generic`'s four defects live in three verdict subsections; one covers both `kernel.opt_level` and
`kernel.compiler_include_paths`.

The distribution matters more than the total. The two fused CCL ops hold 13 of the 48 between them.
The clean end is the more useful half: five of the six clean ops *do* define a custom
`compute_program_hash` — only `point_to_point` relies on reflection — so writing one is not itself
the hazard. What those five share is that the parameters they drop are held down by something the
audit could point at: a `TT_FATAL` on the miss validator (which, per class 4, still runs on hits), a
`ROW_MAJOR` pin that fixes tile geometry, or a genuinely unread field. The bugs below are the cases
where nothing held.

---

## What the valid omissions have in common

121 of the 198 adjudicated omissions are legitimate, and they collapse into a single rule with four
branches. An omission is safe when the value is **re-supplied** on every dispatch, **derived** from
something already in the key, **pinned** to one value by a validator, or simply **dead**. Every valid
omission across all 24 ops is one of those four.

| Justification | Rows | What gets skipped | Why it holds | How it fails |
|---|---|---|---|---|
| Re-supplied (patched) | 37 | Buffer addresses above all; also semaphore addresses, `slice_index`, `slot_idx`, `valid_length` | The value is rewritten into the program on every dispatch, so the cached program never carries a stale copy | The patch covers fewer slots than the factory writes — the failure behind `interleaved_to_sharded_partial` |
| Derived (invariant) | 28 | Per-core work splits, `ring_index` and fabric neighbours, `my_sp_coord` / `sp_factor`, create-only runtime args | A pure function of values already in the key — including mesh coordinates, which the framework appends itself | The derivation stops being a function of the hashed set, usually when a new input joins the factory |
| Pinned by validation | 27 | Storage kind almost everywhere; layout pins; interleaved-only and `ROW_MAJOR` memory-config pins | A `TT_FATAL` rejects every value but one, so the parameter cannot vary across calls that reach the program | The op defines a narrow hit validator, which replaces the miss validator and drops the pin |
| Dead (unused) | 21 | `alignment` in most ops; genuinely dead fields like `output_mem_config` in create_heads, `memory_used` in pool | Nothing in the factory reads it, or it only reaches the program through a value already hashed | The parameter reaches a compile-time arg after all |
| Relaxation win | 8 | `logical_shape`, where the factory works only in padded terms | Two tensors with the same padded shape compile identically, so keying on the logical shape forces needless rebuilds | A path where the logical shape *does* reach the work split — the `interleaved_to_sharded` bug |

The fifth is not a separate justification: it is the "derived" argument applied deliberately for
performance. It is listed separately because it is the only omission class motivated by speed rather
than by safety, and therefore the only one where the trade is the point.

### The five parameters nearly every op skips

Five arguments account for most of the valid omissions. The useful signal is not how many ops skip
each one but how well they **agree on why** — agreement predicts the defect rate almost exactly.

| Parameter | Ops | Dominant justification | Agreement | Defects traced to it |
|---|---|---|---|---|
| Buffer addresses | 21 | Patched, in every op | Unanimous | 0 |
| Storage kind | 16 | Pinned by validation (16 of 18 rows) | Near-unanimous | 0 |
| `alignment` | 11 | Unused (9 of 12 rows) | Strong, with 5 caveats | 0, but 5 caveats |
| `logical_shape` | 10 | Relaxation win (7 of 11 rows) | Strong | 1 |
| `page_config` / `Tile` | 10 | Split three ways: invariant (5), pinned (4), unused (3) | None | 7 ops — the largest class |

Counts are distinct ops with at least one VALID row naming the parameter; row counts differ where an
op tables the same parameter for several tensors.

Where the codebase has settled on one reason for dropping a parameter, that reason is load-bearing
and gets checked. Where each op invents its own justification, some of those justifications are
wrong — which is exactly the shape of the tile bug class.

Three parameter-level notes worth keeping:

- **Buffer addresses are the one true universal.** Every op with a buffer omits and patches the
  address, and not one gets it wrong; this is the entire point of the cache. The single exception
  inverts the convention: `experimental/test/prefetcher_consumer` *hashes* `global_cb->config_address()`
  and the source buffer's address, using an allocation address as an identity token for structure.
  That inversion is where its four defects come from.
- **`alignment` is the parameter that splits the set.** Note first that two distinct things carry
  the name. `TensorLayout`'s `Alignment` is a per-dimension padding granularity inside `TensorSpec`,
  defaulting to `{tile.height, tile.width}` for tile layout and the shard width for row-major
  (`tt_metal/impl/tensor/spec/layout/page_config.cpp:43-56`) — a default, not a pin, since a caller
  may supply any multiple of it. `Buffer::alignment()` is an unrelated allocator byte constant keyed
  on `buffer_type` (`tt_metal/impl/buffers/buffer.cpp:656-658`). They meet at `aligned_page_size`,
  where the byte constant is safely pinned by the hashed `memory_config` but `page_size()` still
  derives from the tensor's own layout. The verdict then turns on one question: does it reach a
  compile-time arg? Where it only feeds already-hashed derivatives it is genuinely dead
  (`reshape_on_device`, `roll`, `slice`, the `llama_*` ops). Where it becomes an `aligned_page_size`
  compile-time arg it cannot be refreshed on a hit, and the same omission becomes a caveat
  (`pack_scaled_fp8_kv_cache`, `per_token_cast_to_fp8`, both KV-cache ops,
  `interleaved_to_sharded_partial`). Same parameter, same reasoning, opposite conclusion.
- **Mesh coordinates are free.** `ring_index`, fabric neighbours, `my_sp_coord` and `sp_factor` are
  omitted across every CCL and deepseek op and valid in all of them, because the framework appends
  the mesh coordinates to the key itself.

---

## The recurring classes

Six ops are clean. The defects in the rest concentrate into five patterns, and four of the five
appear in unrelated op families — which is what makes them worth fixing structurally rather than
op by op.

### 1. `TensorAccessorArgs` emitted as compile-time args

**The single most common defect. Six ops, four unrelated families.**

`TensorAccessorArgs::append_to` emits the `ArgsConfig` bitset — carrying the `IsDram` bit — and the
aligned page size. When these go into *compile-time* args they are baked into the cached program and
**no cache-hit path can refresh them**, not the buffer-binding fast path, not `override_runtime_arguments`,
not a descriptor rebuild. So if the tensor's `buffer_type` is not in the key, a second call with an
otherwise identical tensor in the other memory space hits the cached program, has its address patched
correctly, and then resolves that address through the wrong bank table.

Affected: `moe_padding_config`, `update_padded_kv_cache`, `zero_padded_kv_cache`, `slice` (via the
unhashed `end_tensor`), `tanh_bw` (via the unhashed preallocated output), `dram_prefetcher_validator`.

Every one of the 24 ops emits `TensorAccessorArgs` as compile-time args and none as dynamic args.
That single habit accounts for the largest bug class, across four op families sharing no code.

### 2. Hashes with no shape term

**Three ops, two unrelated families.** `llama_reduce_scatter_matmul` (nine terms, no shape),
`llama_reduce_scatter_create_heads` (same nine-term shape) and `interleaved_to_sharded_partial`
(no shape term in either form) all rely on the shard spec inside a hashed `MemoryConfig` to stand in
for a shape. That is incidental protection, not a design: it holds only while the tensor is sharded
and while logical equals padded.

`interleaved_to_sharded_partial` is the sharpest instance because its own `override_runtime_arguments`
comment asserts the work split is "pinned by the hashed shape/shard-spec" while the hash contains no
shape at all.

### 3. Unguarded 32x32 tile assumptions

**Seven ops.** A factory computes page sizes, tile counts or core ranges from `tt::tile_size` and the
bare `TILE_HEIGHT`/`TILE_WIDTH` constants, accepts `Layout::TILE`, never checks the tensor's actual
`Tile`, and does not hash `page_config`. Non-32x32 tiles are constructible from Python, so this is
reachable rather than theoretical.

Affected: `reshape_on_device`, `roll`, `interleaved_to_sharded_partial`, `slice`, `tanh_bw`,
`update_padded_kv_cache`, `zero_padded_kv_cache`. The mirror image — a genuinely *tile-aware* factory
whose key omits `page_config` — appears in `llama_all_gather_matmul_async`,
`llama_reduce_scatter_matmul` and `dram_prefetcher_validator`.

Two important discriminators, both of which changed verdicts during this audit:

- **A `Layout::TILE` check is not a tile guard**, and neither is a shape-divisibility check against
  `TILE_HEIGHT`/`TILE_WIDTH`. Only a check on `tile().get_height()`/`get_width()` counts.
- **A `ROW_MAJOR` pin *is* effectively a guard**, because `PageConfig::get_tile()` returns a default
  32x32 `Tile{}` on that branch, so no caller-supplied tile can reach the factory. This is what
  cleared `topk_large_indices` and `pool/generic`. `per_token_cast_back` carries the same pin but
  does not depend on it: it hashes each input's whole `TensorSpec`, so the tile is in its key and the
  omission would be clean even if the pin were relaxed — its own document grades the pin as the
  second of two independent reasons, not the clearing one.

### 4. Hit validators that suppress the miss validator

**Sixteen of the 24 ops define a hit validator; eight of them pin strictly less than their miss
validator, and after the reachability filter the residue is tiny.** The dispatcher runs exactly one
validator on a hit (`ttnn/api/ttnn/device_operation.hpp:262-266`): the op's own if it defines one,
otherwise the miss validator, substituted. The consequence is inverted from the intuitive reading —
**defining a narrow hit validator disables every check in the miss validator on the hit path**, while
an op with *no* hit validator is fully validated on hits.

Pinning strictly less than the miss validator:

- Empty stubs suppressing the whole miss validator: `llama_reduce_scatter_create_heads` and both ops
  in `prefetcher_consumer`.
- Partial delegation dropping large blocks: `update_padded_kv_cache` (lines 140-182),
  `zero_padded_kv_cache` (157-193), `rotary_embedding_indexed` (142-182).
- `topk_large_indices` calls only `validate_runtime_args`, dropping the whole of
  `validate_static_args` (`k` bounds, `arch`, `layout()`, `dtype()`, `!is_sharded()`) — all five
  filtered out as unreachable, four because they test hashed values and `arch` because it is a
  process-global read that cannot vary within one device's cache.
- `strided_reduce_scatter_async` checks only storage kind and buffer liveness on hits, dropping the
  rest of its miss validator.

Pinning the same as the miss validator, so dropping nothing: `roll`, `generic`, `pool/generic`,
`moe_padding_config`, `per_token_cast_back`, `per_token_cast_to_fp8` and `pack_scaled_fp8_kv_cache`
route both hooks through the same helper or duplicate it verbatim. `llama_reduce_scatter_matmul`
also drops nothing, but for a weaker reason: it re-runs the matmul's *miss* validator on hits and
delegates the reduce-scatter half to an empty stub. Correct today in all of these, but by
construction only where the shared helper is used — a check added to one hook and not the other is
silently dead on the hit path.

`fusion` is a case of a different shape: it defines a hit validator, but *both* of its validators are
empty, so nothing is suppressed because nothing is checked. The cost is that no verdict in that op
can ever be "pinned by validation", which is part of why its address-slot aliasing grades as a BUG.

**The line counts overstate the hazard by roughly an order of magnitude, and the reason generalises.**
A miss-only pin on a value that is *itself in the key* cannot be evaded: a call carrying a new value
of a hashed parameter computes a different key, misses, and meets the pin on the miss path, which
always runs (`device_operation.hpp:301`). It is rejected on first occurrence and never reaches a hit.
Only checks constraining values *absent* from the key are reachable. Filtering line by line,
`update_padded_kv_cache`'s forty dropped lines reduce to four reachable checks and
`zero_padded_kv_cache`'s to one; the layout, rank, shape-equality and alignment gates are all
self-enforcing because they constrain hashed values.

Reachable is also not the same as worth fixing. The hit path is the fast path, so a check added there
is paid on every dispatch, and several reachable drops fail *loudly* anyway — a host-storage tensor
faults when its buffer cannot be resolved, a layout divergence is caught by the framework's own
`TensorSpec` comparison. Those buy a better diagnostic at a permanent cost. Across the three deepseek
ops only three checks were worth closing: two dtype/layout comparisons in `update_padded_kv_cache`
that silently select a wrong cache page size, and a tile guard in each of the other two.

Applying the filter to all eight suppressing ops changed no bug count, which is itself the result:
every bug in this set is a hash omission that the miss validator would not have caught either, so
deleting the narrow hit validators would fix none of them. It changed verdicts in exactly one op, and
that op is where the filter earned its keep in the other direction — by escalating rather than
clearing. `strided_reduce_scatter_async` had graded its caller-supplied `optional_output_tensor`, and
the layout/dtype/page-config/shape of its `optional_intermediate_tensor`, as "VALID — pinned by
validation" on the strength of pins living only in the miss validator. That op defines a narrow hit
validator checking just input storage and buffer non-null, and the pinned values are the optional
tensors' own specs, which are not in the key — so the filter does not clear them.

The output-tensor case is now a **BUG**. The shared validator pins the caller's buffer hard
(`experimental/ccl/reduce_scatter_common/reduce_scatter_validate_utils.cpp:82-84` requires
`output_tensor.memory_config() == memory_config`), the hash carries `output_mem_config` but nothing
about the passed tensor, and `persistent_output_buffers` is exposed straight to Python. Two calls
with the same `output_mem_config` and different buffers share a key; the second writes through
addressing built for the first. That is structurally identical to the intermediate-tensor omission
the same document already grades a BUG. The fix has an exact in-repo precedent:
`ReduceScatterDeviceOperation::validate_on_program_cache_hit`
(`ccl/reduce_scatter/device/reduce_scatter_device_operation.cpp:32-42`) compares the passed tensor's
`tensor_spec()` against `compute_output_specs`, only when the tensor is present. The cost is a little
more than the comparison itself, since `compute_output_specs` constructs two `TensorSpec`s; hoisting
that one call covers the output and the intermediate together, which closes both of the op's bugs at
once. Worth landing deliberately, as a whole-`TensorSpec` comparison also comes out stricter than the
piecewise miss-path checks it replaces — it brings `logical_shape` and alignment into scope.

This also names a pattern that cuts across the classes rather than forming one: a **caller-supplied
optional tensor whose spec is absent from the key**. It accounts for both `strided` bugs, the
preallocated-output bugs in `tanh_bw` and `slice` (which reach the program through class 1's
compile-time accessor args), and it is what the `ccl/reduce_scatter` hit validator exists to prevent.
An op accepting a preallocated output should either hash its spec or check it on the hit path.

The clearest correction from the filter is `prefetcher_consumer`, where an earlier pass claimed the
suppressed guards were the very fields its bugs turn on. The fields are indeed unhashed, but the
guards test existence, non-emptiness and scalar bounds — never that a global CB or tensor spec still
*matches* what the cached program was built for. In the consumer's reproduction the suppressed
`receiver_cores().num_cores() > 0` would have passed. Exactly one suppressed check anywhere in the
set closes a silent hole on its own: `is_dram()` in the prefetcher validator, guarding a DRAM/L1
numeric-address coincidence.

The one hole in the filter: for a custom-hash op the canonical key degrades to the op-identity prefix
(`mesh_device_operation_adapter.hpp:1012-1013`), so a 64-bit collision is a wrong hit that skipped the
miss validator entirely. Low probability, but it means "unreachable" should read "unreachable absent
a hash collision" for those ops.

### 5. Fused ops hashing only one half

`llama_reduce_scatter_matmul` is the extreme case: its `compute_program_hash` is a verbatim copy of
the standalone reduce-scatter key, and the entire matmul half — weights, `MatmulParams`, the optional
second weight tensor, output memory config, subdevice — never made it in.
`llama_all_gather_matmul_async` has the same shape, omitting `matmul_struct` entirely and reading its
`intermediate_*` hash terms from `input1` through a copy-paste error.

Deriving a fused op's key by *composing* the two component keys, rather than hand-copying one, would
prevent both and keep the fusion tracking future changes to either component.

### Not a class, but the one defect unique to descriptor-based dispatch

`experimental/fusion` patches a cached descriptor through `AddressSlots`, a map from descriptor
positions to the IO addresses they reference, computed once at build time by *matching on address
value* and index-based thereafter. `compute_address_slots` takes the first address match without
checking for a second, and `patch_stale_descriptor` silently skips a slot whose buffer is missing.
So if a build-time descriptor happens to alias an input and an output in place, the frozen slot map
sends the input's address to the output's slots on every later call — and the hot path allocates a
fresh output. The aliasing pattern is in neither the device hash nor the Python build cache key, both
validators are empty, and it reproduces in two calls.

Worth stating that this does **not** generalise: `compute_address_slots`, `patch_stale_descriptor`
and `AddressSlots` appear only under `experimental/fusion` and its Python descriptor and tests.
`generic`, the other descriptor-driven op here, does not use this machinery and its four bugs are
unrelated.

---

## Triage — what is fine, what has caveats, what is a time bomb

The distinction that matters is not how many defects an op has but what has to change for one to
fire. Sorted that way, most defects are already reachable through ordinary configuration changes.

### Armed — the defect fires on an ordinary configuration change

| Op | Defects | What has to change |
|---|---|---|
| `experimental/ccl/llama_reduce_scatter_matmul` | 8 | Any matmul parameter — weights, blocking, activation, output dtype, second weight tensor presence |
| `experimental/ccl/llama_all_gather_matmul_async` | 5 | The matmul config or the intermediate tensor; the intermediate's globally-allocated CB address is never repatched |
| `experimental/test/prefetcher_consumer` | 4 | Swap the global CB — core mapping, size or buffer type |
| `generic` | 4 | A user descriptor varying `opt_level`, `compiler_include_paths`, `face_geometry`, or per-core runtime-arg lengths |
| `experimental/ccl/llama_reduce_scatter_create_heads` | 3 | QKV memory config, packet-buffer grid, or subdevice |
| `data_movement/roll` | 3 | Output memory config grid or orientation; a shape that changes the runtime-arg count |
| `data_movement/slice` | 3 | Vary the `end_tensor` or pass a preallocated output |
| `data_movement/sharded_partial/interleaved_to_sharded_partial` | 3 | Input shape or memory config — the key has no shape term at all |
| `eltwise/unary_backward/tanh_bw` | 3 | Pass a preallocated gradient, or change `grad_output`'s layout |
| `experimental/ccl/strided_reduce_scatter_async` | 2 | `persistent_output_buffers` whose memory config differs from the built program's |
| `experimental/deepseek_prefill/update_padded_kv_cache` | 2 | Move a metadata tensor between DRAM and L1 |
| `experimental/deepseek_prefill/zero_padded_kv_cache` | 2 | Same metadata buffer-type move |
| `ccl/reduce_to_root` | 1 | Change the mux cores — kernel placement is structural |
| `experimental/fusion` | 1 | A descriptor aliasing an input and output in place at build time |
| `experimental/deepseek_prefill/moe_padding_config` | 1 | Move `actual_start` / `actual_end` metadata between DRAM and L1 |
| `pool/generic` | 1 | Two pools differing only in `ceil_pad_hw` with `ceil_mode` on |
| `data_movement/sharded/interleaved_to_sharded` | 1 | The row-major path with a non-canonical `Alignment` |

### Tile-gated — needs a tile geometry nobody currently passes

| Op | Defects | What has to change |
|---|---|---|
| `data_movement/reshape_on_device` | 1 | A non-32x32 tile — supported and constructible from Python, but nothing in tree does it |

The tile components of `roll`, `slice`, `interleaved_to_sharded_partial`, `tanh_bw` and both KV-cache
ops sit behind the same gate, but each of those ops has at least one other defect that is already
armed.

### Caveat only — no defect, but held by something worth knowing

| Op | Exposure |
|---|---|
| `experimental/deepseek_prefill/rotary_embedding_indexed` | Five caveats plus a factory defect: a non-32x32 tile compiles a mis-sized program on the miss path. Saved from aliasing by exact `TensorSpec` equality |
| `experimental/deepseek_prefill/pack_scaled_fp8_kv_cache` | A non-canonical `alignment` reaching the `aligned_page_size` compile-time arg |
| `experimental/deepseek_prefill/per_token_cast_to_fp8` | Same alignment exposure |
| `experimental/topk_large_indices` | Its hit validator is sound only because all five dropped checks test hashed values |

### Clean

`experimental/deepseek_prefill/per_token_cast_back` — no defects and no caveats; hashes the whole
`TensorSpec`.

---

## Framework-level findings

These are not op defects; they affect every op and were established once, in-source.

**Tile transpose flags are invisible to the whole cache key.** `Tile::attribute_values()` exposes
only `tile_shape`, `face_shape` and `num_faces` (`tt_metal/api/tt-metalium/tile.hpp:46-47`), and
`Tile::operator==` compares only the first two (`tt_metal/impl/data_format/tile.cpp:122-124`). Since
those drive the reflection hash and the canonical collision key respectively, `transpose_within_face`
and `transpose_of_faces` are covered by neither, for default-hash and custom-hash ops alike. Hashing
`page_config` or a whole `TensorSpec` buys shape coverage only; only an explicit `TT_FATAL` closes
transpose.

**`SlidingWindowConfig::to_string()` omits `ceil_pad_hw`** (`sliding_window/sliding_window.cpp:1283-1297`),
and `get_hash()` is a hash of that string. The `operator<<` overload immediately below *does* print
the field, which reads as an oversight. Because the type is `std::hash`-specialised, the default
reflection hash inherits the same gap, so no per-op custom hash can fix it. It reaches `pool/generic`
as a program-cache key and both halo variants as the key into a thread-local memoization map for
`max_out_nsticks_per_core` — same root cause, different failure. **conv2d is not affected**: it never
sets `ceil_mode`, and `get_ceil_pad_hw()` short-circuits to `{0,0}` in that state.

A one-case addition to `tests/ttnn/unit_tests/gtests/test_sliding_window_infra.cpp` would have caught
this: the suite already flips `snap_to_tile`, `is_bilinear`, `is_transpose` and `ceil_mode` and
asserts the hashes differ. It never flips `ceil_pad_hw`. (The new case must set `ceil_mode = true`
first, since the fixture defaults it false and the accessor short-circuits.)

**Defining any custom hash costs collision resolution.** `ProgramCacheKey`'s canonical string
degrades to the op type name when `compute_program_hash` exists, so a 64-bit collision becomes a
wrong hit rather than a rebuild. This compounds every omission above and is worth weighing before
adding a custom hash for performance.

**A default-hash op gets tile geometry for free**, because reflection walks `Tensor` → `TensorSpec`
→ `TensorLayout` → `PageConfig` → `Tile`. So inherited tile math in such an op can only ever be a
factory defect, never an aliasing one.

---

## Two hazards that are not yet bugs

Both ops are correct today for reasons nobody wrote down, and an innocent-looking edit would break
them. An op is safe when a parameter is absent from the key *and* absent from the factory; nothing
enforces that pairing, and nothing warns when the second half changes.

- **`llama_reduce_scatter_create_heads`** constructs a `CCLOpConfig` whose constructor reads the real
  tile and derives a page size, marked `[[maybe_unused]]` and never read. The op's hash has no
  `page_config`, no `tile()` and no shape. One dereference converts a clean op into an aliasing bug,
  with no compiler warning, because the attribute exists to suppress exactly that diagnostic.
- **`point_to_point`** inherits genuine tile math from `data_movement/common`'s `get_num_pages`, which
  reaches two CB sizings and — in the local-copy factory — the core ranges via `split_work_to_cores`.
  It is correct today only because the op has no custom hash and so hashes `tensor_args` wholesale.
  Adding a `compute_program_hash` that omits `page_config` — the standard performance move — would
  make all three sites aliasing bugs the same day.

---

## Method note

Two scoping lessons, both of which produced wrong answers before being caught:

1. **Searching an op's own directory is not enough.** `llama_reduce_scatter_matmul` and
   `point_to_point` were both initially filed as having no host-side tile math; both build through
   shared factories or helpers in sibling directories. Fixes for shared code should be targeted there
   and should name the other ops that inherit them.
2. **"No in-tree caller does this" is not a defence.** Nearly every finding here is latent for that
   reason — most of these ops are exercised by one model with a frozen configuration. Reachability
   through the public API without an *enforced* (`TT_FATAL`) constraint is the test that was applied;
   current call sites inform severity, not the verdict.
