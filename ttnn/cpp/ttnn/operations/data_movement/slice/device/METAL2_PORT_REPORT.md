# Metal 2.0 port — slice (SliceTileProgramFactory → create_program_spec)

STATUS: PORTED

## Factory chosen
The simplest single-program factory: the non-strided TILE path (`SliceTileProgramFactory`,
`slice_program_factory_tile.{hpp,cpp}`). Two kernels (reader + writer), one CB, one work-split
into up to two core groups. No semaphores, no op-owned tensors, no per-mesh-coord variation.

## The two-concept conflict and how it was resolved
`ccl/mesh_partition` builds its own `MeshWorkload` by calling
`SliceTileProgramFactory::create_descriptor(...)` directly, per mesh coordinate, inside its
own `create_at` / `override_runtime_arguments` (mesh_partition_program_factory.cpp lines ~127
and ~157). It needs a `ProgramDescriptor` to stamp into a `Program` and to feed
`apply_descriptor_runtime_args`.

A single factory struct cannot satisfy BOTH `ProgramDescriptorFactoryConcept` (has
`create_descriptor`) and `ProgramSpecFactoryConcept` (has `create_program_spec`) — the
`AllFactoriesValid` check (operation_concepts.hpp) requires each `program_factory_t` variant
alternative to satisfy EXACTLY one concept.

Resolution (the recommended split):
- **Kept** `SliceTileProgramFactory::create_descriptor` unchanged → mesh_partition keeps building.
- **Added** a separate struct `SliceTileSpecProgramFactory::create_program_spec` (Metal 2.0)
  in the same .hpp/.cpp. slice's own `select_program_factory` now routes the TILE path to it,
  and the `program_factory_t` variant lists `SliceTileSpecProgramFactory` in place of the old
  descriptor factory.
- mesh_partition still `std::visit`s slice's variant; its two lambdas were given a one-line
  `if constexpr (std::is_same_v<Factory, SliceTileSpecProgramFactory>)` branch that calls
  `SliceTileProgramFactory::create_descriptor` (the descriptor variant of the identical TILE
  work) for that alternative, and `Factory::create_descriptor` for the rest. This is the only
  edit outside the slice op dir, and it is minimal + documented.

## Kernels
Both slice-local kernels are used ONLY by the TILE factory, but the legacy descriptor path
(kept for mesh_partition) still consumes the originals via `get_named_compile_time_arg_val` /
positional `get_arg_val` / `get_common_arg_addr`. The Metal 2.0 path needs the
`experimental/kernel_args.h` named mechanism. So both were **forked** (not edited in place):

- `kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id_m2.cpp` (fork of the legacy reader)
- `kernels/dataflow/writer_unary_interleaved_start_id_m2.cpp` (fork of the legacy writer)

Port is access-mechanism-only (logic, #ifdefs, loop bounds, numeric paths UNCHANGED):
- src/dst addresses: raw `src_addr`/`dst_addr` runtime args → `TensorAccessor(ta::src)` /
  `TensorAccessor(ta::dst)` (address flows through TensorBinding; the appended
  `TensorAccessorArgs` CTAs are gone).
- CB ids: `get_named_compile_time_arg_val("cb_in"/"cb_out")` → `dfb::cb_in` / `dfb::cb_out`
  (one DataflowBufferSpec `cb_in_out`, reader=PRODUCER, writer=CONSUMER, both on `all_cores`,
  same WorkUnitSpec — local-DFB rule satisfied).
- `num_dims`: positional CTA → named CTA `args::num_dims`.
- reader `start_id`/`num_tiles`, writer `num_pages`/`start_id`: positional RTAs → named RTAs.
- reader per-dim `id_per_dim[]` array → runtime **varargs** (`get_vararg`); the common
  `[num_unpadded.., num_padded..]` arrays → common **varargs** (`get_common_vararg`). The
  m2 vararg API exposes value getters only (no writable pointer into the vararg region), so
  the per-dim running counters are copied into a local stack array (`uint32_t
  id_per_dim[num_dims]`, sized by the `num_dims` CTA) and mutated there — identical arithmetic.

## Files changed
- `device/slice_program_factory_tile.hpp` — added `SliceTileSpecProgramFactory` struct (kept
  `SliceTileProgramFactory`).
- `device/slice_program_factory_tile.cpp` — added `SliceTileSpecProgramFactory::create_program_spec`
  (legacy `create_descriptor` untouched).
- `device/slice_device_operation.hpp` — `program_factory_t` variant: `SliceTileProgramFactory`
  → `SliceTileSpecProgramFactory`.
- `device/slice_device_operation.cpp` — `select_program_factory` TILE path returns
  `SliceTileSpecProgramFactory{}`.
- `device/kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id_m2.cpp` — NEW (forked).
- `device/kernels/dataflow/writer_unary_interleaved_start_id_m2.cpp` — NEW (forked).
- `ccl/mesh_partition/device/mesh_partition_program_factory.cpp` — two `if constexpr` branches
  mapping the spec factory alternative back to `SliceTileProgramFactory::create_descriptor`.
  (Minimal, documented; the only change outside the slice op dir.)

The `SliceTileProgramFactory::create_descriptor` pybind hook (slice_nanobind.cpp) was LEFT in
place — it is not dangling (the descriptor factory still exists). No new pybind hook was added
for the spec factory (matches rand; the spec path is dispatched by the framework adapter).

## Blockers
None. Both slice and ccl/mesh_partition retain everything they need to build.

## Not built / not measured / not committed
Per instructions. (This worktree has no build dir; cannot compile or measure here.)

## Note on this worktree's base
This worktree is checked out on a base commit that has the Metal 2.0 host API
(`tt-metalium/experimental/metal2_host_api/*`) and adapter (`ProgramSpecMeshWorkloadFactoryAdapter`,
`ttnn/metal2_artifacts.hpp`) merged, but does NOT contain the worked-example op ports (rand /
matmul on this base are still legacy `create_descriptor`). The port was written against the
in-tree framework API, using the `dgomez/rand-metal2` rand factory as the structural template
for `create_program_spec` and the forked-kernel named-arg conventions.

---

# Metal 2.0 port — slice TILE tensor-args path (SliceTileTensorArgsSpecProgramFactory)

STATUS: PORTED

## Per-factory STATUS summary (all five slice factories)
- `slice_program_factory_tile.cpp` (TILE, non-strided) — **PORTED** (prior pass; see top of file).
- `slice_program_factory_tile_tensor_args.cpp` (TILE, tensor-args / `use_tensor_args`) — **PORTED** (prior pass).
- `slice_program_factory_rm.cpp` (ROW_MAJOR interleaved/W&B-sharded) — **BLOCKED** (per-shard
  page-size override on a `ta::`-bound accessor + per-dispatch host-composed base offset; see RM
  Case-2 re-examination below).
- `slice_program_factory_rm_sharded.cpp` (ROW_MAJOR HEIGHT-sharded in/out) — **PORTED** (this pass,
  Case-2 bridge: borrowed DFBs + raw NoC-map walk).
- `slice_program_factory_rm_stride.cpp` (ROW_MAJOR strided 4D/ND) — **PORTED** (this pass, Case-1:
  clean Buffer* address bindings, default page size).

> NOTE: the prior pass marked the three RM factories METAL-BLOCKED. The Case-2 bridge re-examination
> (this pass) flips two of them: rm_sharded does a raw NoC walk (no `TensorAccessor` fed to
> `noc_async_*_sharded` at all), and rm_stride builds its accessor with the *default* page size (no
> override) on a *clean* base address. Only rm.cpp genuinely hits the per-shard page-size-override
> gap. The blocker sections further down (written by the prior pass) are superseded for rm_sharded /
> rm_stride by the RM Case-2 re-examination section at the very bottom of this file.

## Factory chosen
The TILE tensor-args path (`SliceTileTensorArgsProgramFactory`,
`slice_program_factory_tile_tensor_args.{hpp,cpp}`), selected by `select_program_factory` when
`args.use_tensor_args` is set. It is the same structural shape as the already-ported non-strided
TILE factory (reader + writer, work-split into up to two core groups) but reads the slice
`start`/`end` bounds from two extra device tensors at runtime, via a second single-tile staging CB.

## Two-concept conflict — resolved exactly as the non-strided TILE path
`ccl/mesh_partition`'s `std::visit` over slice's `program_factory_t` instantiates
`Factory::create_descriptor(...)` for EVERY variant alternative (mesh_partition_program_factory.cpp,
the generic `else` branches). So removing `create_descriptor` from the tensor-args factory would
break mesh_partition's build, regardless of whether mesh_partition ever selects it at runtime.
Same constraint, same resolution as the non-strided TILE path:
- **Kept** `SliceTileTensorArgsProgramFactory::create_descriptor` unchanged.
- **Added** a separate struct `SliceTileTensorArgsSpecProgramFactory::create_program_spec` (Metal 2.0)
  in the same .hpp/.cpp. `select_program_factory`'s `use_tensor_args` path now routes to it, and the
  `program_factory_t` variant lists `SliceTileTensorArgsSpecProgramFactory` in place of the descriptor factory.
- mesh_partition's two `std::visit` lambdas were given a parallel
  `else if constexpr (std::is_same_v<Factory, SliceTileTensorArgsSpecProgramFactory>)` branch that
  calls `SliceTileTensorArgsProgramFactory::create_descriptor` (the descriptor variant of the
  identical work). Minimal, documented; the only edit outside the slice op dir, and identical in
  shape to the branch the prior TILE port already added.

## Kernels (one forked, one reused)
- **Reader — forked**: `kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id_tensor_args_m2.cpp`
  (NEW; fork of `reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp`, which the kept legacy
  descriptor path still consumes verbatim). Access-mechanism-only port:
  - src/start/end addresses: `get_common_arg_val<uint32_t>(0..2)` → `TensorAccessor(ta::src/ta::start/ta::end)`.
  - CB ids: `get_compile_time_arg_val(0)`/`(1)` → `dfb::cb_in` / `dfb::cb_tensor`.
  - `num_dims`/`tile_width`/`tile_height`: positional CTAs → named CTAs (`args::...`).
  - `start_id`/`num_tiles`: positional per-core RTAs → named RTAs.
  - per-dim `id_per_dim[]`: per-core RTA tail → runtime **varargs** (copied into a local stack array
    `uint32_t id_per_dim[num_dims]`, mutated locally — the m2 vararg API has value getters only;
    identical arithmetic).
  - common `[num_unpadded.., num_padded.., input_shape..]`: `get_common_arg_addr(3..)` → common
    **varargs**. Layout preserved exactly: `get_common_vararg(j)` = num_unpadded[j],
    `get_common_vararg(num_dims+j)` = num_padded[j], `get_common_vararg(2*num_dims+i)` = input_shape[i]
    (the legacy `get_common_arg_addr(3 + 2*num_dims)` base shifts to common-vararg index `2*num_dims`
    once the 3 buffer addresses leave the common-arg region).
- **Writer — reused**: `kernels/dataflow/writer_unary_interleaved_start_id_m2.cpp` (the existing fork
  added by the non-strided TILE port). The legacy tensor-args factory used the **shared eltwise**
  `eltwise/unary/.../writer_unary_interleaved_start_id.cpp` (read CB id positionally, `TensorAccessorArgs<1>`),
  whose Metal-2.0 shape — `dfb::cb_out` + `ta::dst`, named `num_pages`/`start_id` RTAs — is byte-for-byte
  what that existing m2 fork already provides, so no second writer fork was created (avoids editing the
  shared eltwise kernel in place, per the shared-kernel caution).

## Spec shape
- DFBs: `cb_in` (idx 0, double-buffered single-tile FIFO; reader=PRODUCER `cb_in`, writer=CONSUMER `cb_out`)
  and `cb_tensor` (idx 1, single-tile staging for the start/end reads; reader **self-loop** —
  PRODUCER + CONSUMER on the reader, since the reader is its only user: it writes start/end into it then
  reads them back). The `cb_in` producer/consumer share the one WorkUnitSpec on `all_cores` (local-DFB rule).
- TensorParameters: `src`, `start`, `end` (reader), `dst` (writer), each from `<tensor>.tensor_spec()`.
- Per-core / common runtime-arg computation copied verbatim from `create_descriptor`, including the
  zero-filled no-op-core path and the `constexpr start_offset = 0` (the tensor-args reader computes the
  real per-region offset kernel-side from the start/end tensors).

## Files changed (this pass)
- `device/slice_program_factory_tile_tensor_args.hpp` — added `SliceTileTensorArgsSpecProgramFactory`
  (kept `SliceTileTensorArgsProgramFactory`).
- `device/slice_program_factory_tile_tensor_args.cpp` — added
  `SliceTileTensorArgsSpecProgramFactory::create_program_spec` (legacy `create_descriptor` untouched).
- `device/slice_device_operation.hpp` — `program_factory_t` variant: `SliceTileTensorArgsProgramFactory`
  → `SliceTileTensorArgsSpecProgramFactory`.
- `device/slice_device_operation.cpp` — `select_program_factory` `use_tensor_args` path returns
  `SliceTileTensorArgsSpecProgramFactory{}`.
- `device/kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id_tensor_args_m2.cpp` — NEW (forked).
- `ccl/mesh_partition/device/mesh_partition_program_factory.cpp` — parallel `else if constexpr` branch in
  both lambdas. (Only change outside the slice op dir.)

No pybind hook references the tensor-args `create_descriptor` (the only slice pybind hook,
slice_nanobind.cpp:167, points at `SliceTileProgramFactory::create_descriptor` — the non-strided
factory — and is untouched). CMakeLists lists factory .hpp files only and already includes this one;
no CMake change (the new kernel is a runtime-loaded source, not a compiled TU).

## Blockers — the three ROW_MAJOR factories (re-confirmed METAL-BLOCKED, not ported)

All three route per-row NOC traffic through the shared kernel-lib helpers
`tt::data_movement::common::noc_async_read_sharded` / `noc_async_write_sharded`
(`ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp:230` / `:200`), and/or smuggle a
host-computed base offset / per-shard NOC layout through runtime args that a `ta::` TensorBinding cannot
carry. Precise sites:

- **`slice_program_factory_rm.cpp`** — BLOCKED.
  - `slice_program_factory_rm.cpp:88`: the reader's CRTA slot 0 is
    `start_addr + begins_bytes - misalignment` — the input buffer base address **pre-composed with a
    host-computed misalignment-adjusted byte offset**, threaded through a runtime arg. `start_addr =
    input.buffer()->address()` (`:72`) varies per dispatch, so this is a per-dispatch RTA, not a CTA.
    The sanctioned "Host-computed base-pointer offset → CTA offset + kernel-side addition" pattern is
    explicitly CTA-ONLY (its Constraint: "If your offset would need to be per-dispatch-varying (RTA),
    this pattern doesn't apply; capitulate"). No `TensorBinding` mechanism reproduces this.
  - `slice_program_factory_rm.cpp:85-89` and `:143-152`: `reader_page_size` / `writer_page_size` =
    host-computed per-shard page size (`shard_spec.shape[1] * element_size`) threaded as a runtime arg
    into the sharded NOC path. Not carried by a tensor identity binding.

- **`slice_program_factory_rm_sharded.cpp`** — BLOCKED.
  - `slice_program_factory_rm_sharded.cpp:153-178`: the reader runtime args carry, per core, a
    host-computed **physical NOC core-coordinate map** (`worker_core_from_logical_core` walk at `:142-147`,
    emitted as noc-x/noc-y pairs at `:158-162`) plus per-shard in-shard stick ids and contiguous-chunk
    descriptors. The kernel reads peer shards directly from these RTA-supplied NOC coordinates. A `ta::`
    binding carries a tensor identity, not a host-computed shard-grid NOC layout; there is no
    `TensorAccessor` path that reproduces the host-side shard-grid walk.
  - `slice_program_factory_rm_sharded.cpp:265` and `:277`: both CBs set `.buffer = input.buffer()` /
    `.buffer = output.buffer()` (borrowed-memory CBs onto the sharded tensors). Borrowed DFBs alone are
    expressible, but the NOC-coordinate-map blocker above is dispositive.

- **`slice_program_factory_rm_stride.cpp`** — BLOCKED.
  - The base address itself is a clean `Buffer*` RTA (`:126`, `:145` reader; `:134`, `:158` writer), which
    in isolation would be a Case-1 `TensorBinding`. The blocker is the **sharded read/write path**: the
    reader (`kernels/dataflow/reader_multicore_slice_4d.cpp:157-158`, and the ND variant) and writer
    (`writer_multicore_slice_4d.cpp:89`) pass the constructed `TensorAccessor` into the out-of-op
    kernel-lib helper `noc_async_read_sharded` / `noc_async_write_sharded`, which drive sharded address
    generation through the accessor's `dspec()` / `get_aligned_page_size()` / `get_noc_addr()`
    (`common.hpp:235-249`). This is the same per-shard sharded family flagged as not reproducible through
    a host-injected-metadata `ta::` accessor; consistent with the prior METAL-BLOCKED finding, it was not
    forced. (If the framework later confirms the m2 `TensorAccessor` exposes the same sharded `dspec`
    interface to out-of-op helpers, rm_stride is the most likely of the three to become portable — its
    address is not pre-offset. Flagging for the framework/kernel-lib owners.)

## Not built / not measured / not committed
Per instructions. (This worktree has no build dir; cannot compile or measure here.)

---

# RM Case-2 re-examination — rm_sharded + rm_stride PORTED, rm.cpp BLOCKED (this pass)

This pass re-examined the three ROW_MAJOR factories with the Case-2 bridge in mind (a data-movement
kernel that consumes a tensor's raw base address is PORTABLE: bind the tensor as a normal
`TensorParameter`/`TensorBinding`, pull the base kernel-side, keep the raw NoC arithmetic). The prior
pass had marked all three METAL-BLOCKED; two of them are in fact portable.

## `slice_program_factory_rm_sharded.cpp` — STATUS: PORTED (Case-2 bridge)

The legacy factory has a SINGLE reader kernel (`slice_reader_unary_unpad_dims_rm_sharded.cpp`) that:
- borrows two CBs onto the input/output shard buffers (`CBDescriptor::buffer = in/out.buffer()`) and
  reads their L1 base via `get_write_ptr()`, and
- reads peer shards by a hand-rolled NoC walk over a host-computed physical core-coordinate map
  (`device->worker_core_from_logical_core(...)` → noc-x/noc-y RTAs) plus per-shard stick-chunk
  descriptors (`noc.async_read(UnicastEndpoint{}, dst, ..., {.noc_x, .noc_y, .addr=...}, ...)`).

It does NOT hand a `TensorAccessor` to `noc_async_*_sharded` and has NO host-composed base offset and
NO per-shard page-size override — it is a pure raw-NoC-walk kernel. That is exactly the Case-2 shape.

Port:
- The two borrowed CBs become two `DataflowBufferSpec`s with `borrowed_from = src / dst` (sharded
  tensors are L1-backed; the borrowed DFB resolves its backing L1 base from the matching
  `TensorArgument`, refreshing on cache hits — replacing the legacy `CBDescriptor::buffer`). The
  kernel reads `get_write_ptr()` exactly as before. The src/dst base addresses thus flow through the
  typed tensor channel.
- Both DFBs are bound on the single reader as **self-loops** (PRODUCER + CONSUMER): `cb_in` is a pure
  address source (no real FIFO), `cb_out` is produced into but has no consumer kernel in this
  single-reader factory. Self-loop satisfies the DFB ≥1-producer-and-≥1-consumer invariant. (Fake-CB
  → self-loop pattern.)
- `stick_size_padded` / `stick_size_unpadded` / `num_sticks_unpadded`: positional CTAs → named CTAs.
- The per-core reader arg vector (`num_cores_read`, NoC x/y map, num-chunks, chunk descriptors) is
  variable-length, so it travels as **per-core runtime varargs** with identical contents and layout.
  Per-node vararg counts differ; the API's per-node-count override (`num_runtime_varargs_per_node`)
  is **deprecated** ("truly bizarre, will be removed"), so instead a single uniform
  `num_runtime_varargs` (= the longest per-core vector) is declared and each node's vararg vector is
  zero-padded up to it. The kernel derives every index from `get_vararg(0)` (num_cores_read) and
  never touches the padding tail, so padding is inert. Friction note: the deprecated per-node-count
  field is the "natural" fit here and its removal will force exactly this zero-pad workaround on any
  jagged-vararg op — flagging for the framework owners.

Kernel forked: `kernels/dataflow/slice_reader_unary_unpad_dims_rm_sharded_m2.cpp` (the legacy file
stays for the descriptor path that ccl/mesh_partition reuses).

## `slice_program_factory_rm_stride.cpp` — STATUS: PORTED (Case-1, no bridge needed)

The legacy strided factory selects between 4D and ND reader/writer kernels at runtime by rank. All
four kernels build their accessor as `TensorAccessor(args, addr)` with **no third (page-size) ctor
argument** — i.e. the default `AlignedPageSize`, which is exactly what the `ta::`-binding-token ctor
supplies. The base address is a **clean `Buffer*` RTA** (no host-composed offset). So this is an
ordinary **Case-1** binding, not even a Case-2 bridge: src/dst re-express as plain `TensorBinding`s
(`TensorAccessor(ta::src)` / `(ta::dst)`), and `noc_async_*_sharded` receives an accessor with the
same DSpec and same page size as the legacy path → identical behavior.

Port (one factory, runtime-selected kernels convert together so all four are forked):
- Single in CB (index 0) → one `DataflowBufferSpec` (`cb_in_out`), reader=PRODUCER / writer=CONSUMER,
  one `WorkUnitSpec` on `all_cores`.
- `element_size` leading CTA → named CTA `compile_time_element_size`.
- 4D kernels: the fixed positional RTAs → named RTAs. ND kernels: leading scalar RTAs → named RTAs,
  the five per-dim arrays (input/output dims, slice start/end/step — reader) and the output-dims
  array (writer) → **runtime varargs**, same concatenated layout. The ND kernels copy them once into
  local fixed-size stack arrays (the legacy kernel already caps rank at 16 via `coords[16]`) since
  the vararg API exposes value getters only; the indexing arithmetic is unchanged.
- The 4D vs ND branch is selected inside `create_program_spec` (multi-variant-factory pattern).

Kernels forked: `reader_multicore_slice_4d_m2.cpp`, `writer_multicore_slice_4d_m2.cpp`,
`reader_multicore_slice_nd_m2.cpp`, `writer_multicore_slice_nd_m2.cpp`.

## `slice_program_factory_rm.cpp` — STATUS: BLOCKED (the remaining genuine metal gap)

This one is genuinely blocked, for the precise reason the instructions named: the reader hands a
`TensorAccessor` constructed with a **per-shard page-size override** to `noc_async_read_sharded`, and
that override is not expressible on a `ta::`-bound accessor. Two distinct issues:

1. **Per-shard page-size override on a `ta::`-bound accessor (dispositive).**
   `slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp:34`:
   `TensorAccessor(src_args, src_addr, padded_stick_size)` — the THIRD ctor argument is the per-shard
   page size (`shard_W * element_size` for B/W-sharded; full row otherwise; host-computed at
   `slice_program_factory_rm.cpp:76-85`). It feeds `noc_async_read_sharded`'s multi-shard row split
   via `get_aligned_page_size()` (`common.hpp:235`). The writer is identical
   (`slice_writer_unary_stick_layout_interleaved_start_id.cpp:22,27`, host
   `slice_program_factory_rm.cpp:143-152`). The Metal 2.0 binding-token ctor
   (`tt_metal/hw/inc/api/tensor/tensor_accessor.h:99`) delegates with NO page-size argument — it
   hardcodes `page_size_in = TensorAccessorArgs<...>::AlignedPageSize`, the accessor's *natural*
   aligned page size from the `TensorSpec`. There is no field on `TensorParameter` / `TensorBinding`
   / the binding token to override `aligned_page_size`. So a `ta::`-bound accessor handed to
   `noc_async_*_sharded` cannot carry the host-computed per-shard page size. **This is precisely the
   "per-shard page-size override on a `ta::`-bound accessor passed to `noc_async_*_sharded`" gap.**

2. **Per-dispatch host-composed base offset (also blocking, independent of #1).**
   `slice_program_factory_rm.cpp:88`: the reader's common arg 0 is
   `start_addr + begins_bytes - misalignment` — the input buffer base (`input.buffer()->address()`,
   varies per dispatch) pre-composed with a host-computed misalignment-adjusted byte offset. The
   sanctioned "host-computed base-pointer offset → CTA offset + kernel-side addition" pattern is
   **CTA-only**; here the base is per-dispatch (an RTA), so that pattern explicitly does not apply.
   Even if #1 were solved, the offset would still need a non-CTA path. (The kernel then re-aligns the
   read with `tt_memmove` by `misalignment` — the offset and the page size are coupled.)

Recommended framework fix to unblock rm.cpp: expose an `aligned_page_size` override on the
`TensorParameter` / binding token (a per-binding constant — it does not vary per dispatch for a given
cache entry), plus a sanctioned way to carry a per-dispatch base offset (e.g. an offset CRTA that the
kernel adds after `get_bank_base_address()`, the RTA analogue of the existing CTA-offset pattern).
Flagging for the framework / TensorAccessor owners.

## Files changed (this pass)
- `device/slice_program_factory_rm_sharded.hpp` — added `SliceRmShardedSpecProgramFactory` (kept
  `SliceRmShardedProgramFactory`).
- `device/slice_program_factory_rm_sharded.cpp` — added `create_program_spec` (legacy
  `create_descriptor` untouched).
- `device/slice_program_factory_rm_stride.hpp` — added `SliceRmStrideSpecProgramFactory` (kept
  `SliceRmStrideProgramFactory`).
- `device/slice_program_factory_rm_stride.cpp` — added `create_program_spec` (legacy untouched).
- `device/slice_device_operation.hpp` — `program_factory_t`: `SliceRmShardedProgramFactory` →
  `SliceRmShardedSpecProgramFactory`, `SliceRmStrideProgramFactory` → `SliceRmStrideSpecProgramFactory`.
- `device/slice_device_operation.cpp` — `select_program_factory` HEIGHT-sharded path returns
  `SliceRmShardedSpecProgramFactory{}`; strided path returns `SliceRmStrideSpecProgramFactory{}`.
- `device/kernels/dataflow/slice_reader_unary_unpad_dims_rm_sharded_m2.cpp` — NEW (forked).
- `device/kernels/dataflow/reader_multicore_slice_4d_m2.cpp` — NEW (forked).
- `device/kernels/dataflow/writer_multicore_slice_4d_m2.cpp` — NEW (forked).
- `device/kernels/dataflow/reader_multicore_slice_nd_m2.cpp` — NEW (forked).
- `device/kernels/dataflow/writer_multicore_slice_nd_m2.cpp` — NEW (forked).
- `ccl/mesh_partition/device/mesh_partition_program_factory.cpp` — two parallel `else if constexpr`
  branches (one per new spec factory) in both `std::visit` lambdas, mapping each spec-factory
  alternative back to the corresponding legacy `create_descriptor` for mesh_partition's MeshWorkload
  build / per-coord rebuild. Only change outside the slice op dir; identical in shape to the TILE
  branches the prior passes added.

No custom `compute_program_hash` exists on `SliceDeviceOperation` (the report comments referencing
"compute_program_hash() folds padded_shape" describe the DEFAULT reflection hash, which includes
`padded_shape` — there is nothing to delete). No slice pybind hook references the rm_sharded or
rm_stride `create_descriptor` (the only slice pybind hook points at the non-strided TILE factory).
CMakeLists globs `slice/device/kernels/*`; the new *_m2 kernels are runtime-loaded sources, not
compiled TUs, and the factory `.cpp` edits are in already-compiled files — no CMake change.
