# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/data_movement/pad`

> Audit cleared all gates. This is your actionable input; the full record is in
> `METAL2_PREPORT_AUDIT.md`.

**In scope — all seven factories:**

- `PadRmReaderWriterProgramFactory` — `pad_rm_reader_writer_program_factory.cpp`
- `PadRmReaderWriterMultiCoreProgramFactory` — `pad_rm_reader_writer_multi_core_program_factory.cpp` *(unreachable from `select_program_factory` — raise before porting, see Watch-for)*
- `PadRmReaderWriterMultiCoreDefaultProgramFactory` — `pad_rm_reader_writer_multi_core_default_program_factory.cpp`
- `PadRmShardedHeightOnlyProgramFactory` — `pad_rm_sharded_height_only_program_factory.cpp` *(the only `CustomProgramSpecFactoryConcept` target — see below)*
- `PadRmShardedWidthOnlyProgramFactory` — `pad_rm_sharded_width_only_program_factory.cpp`
- `PadTileCoreProgramFactory` — `pad_tile_program_factory.cpp`
- `PadTileMulticoreProgramFactory` — `pad_tile_multicore_program_factory.cpp`

> **Note on `PadRmShardedHeightOnlyProgramFactory`.** Its readiness-sheet row still reads
> `Is able to port? = no`, derived from a `get_dynamic_runtime_args` cell that PR #52556
> (`90ec10f4bf4`, on `main` 2026-08-19) made obsolete — the hook is gone and an
> `override_runtime_arguments` replaced it. **The sheet is outdated; the audit overrode it on code
> evidence** and the factory is in scope. Don't be alarmed if you look the row up yourself and see a
> `no`. Every gate the audit checks directly clears for it.

`ttnn/cpp/ttnn/operations/experimental/quasar/pad/` is out of bounds. It holds a shortcut copy of this
op; its `_metal2` kernels are **not** forks to reuse and its idioms are not precedent. Don't read it.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ ·
TensorAccessor 3rd arg ✓

**Recipe docs:** `64668f470e4 2026-08-24 docs(metal_2.0): a run in flight freezes the kernel sources`
*(carry this line into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`). Carry them forward:

- **Current concept:** `descriptor` for `PadRmReaderWriterMultiCoreDefaultProgramFactory`,
  `PadRmShardedHeightOnlyProgramFactory`, `PadRmShardedWidthOnlyProgramFactory`,
  `PadTileCoreProgramFactory`, `PadTileMulticoreProgramFactory` · `WorkloadDescriptor` (secretly
  SPMD — collapses to single-program) for `PadRmReaderWriterProgramFactory` and
  `PadRmReaderWriterMultiCoreProgramFactory`.
- **Op-owned tensors:** yes, on the two `WorkloadDescriptor` factories only — the pad-value const
  tensor pushed onto `WorkloadDescriptor::buffers` at
  `pad_rm_reader_writer_program_factory.cpp:200` and
  `pad_rm_reader_writer_multi_core_program_factory.cpp:419`. `ProgramSpecFactoryConcept` carries
  these natively, so the `WorkloadDescriptor` wrapper goes away — it only ever existed to unlock the
  feature. **Keep holding the source `Tensor`, not just the `shared_ptr<MeshBuffer>`**: `~Tensor`
  force-deallocates the device memory through `DeviceStorage::deallocate` regardless of external
  `MeshBuffer` owners (issue #44565, and both factory headers say so).
- **Target concept:** `ProgramSpecFactoryConcept` for six factories (two of them carrying op-owned
  tensors) · **`CustomProgramSpecFactoryConcept` for `PadRmShardedHeightOnlyProgramFactory`** — the
  only factory defining `override_runtime_arguments`
  (`pad_rm_sharded_height_only_program_factory.cpp:412`, declared `.hpp:22`). Translate that method
  into one returning a `ProgramRunArgs` rather than deleting it; details and an open question about
  it in *Construct*.
- **Gate-cleared, confirmed absent** (each would have blocked the brief): a non-`none`
  `TensorParameter relaxation` (all seven sheet rows read `none`) · `get_dynamic_runtime_args` (the
  deprecated hook — absent from the whole op). Also absent, though none of them gate: a custom
  `compute_program_hash`, a backdoor `attribute_values` / `to_hash`, and a pybound
  `create_descriptor` (`pad_nanobind.cpp` binds only the public `ttnn::pad` overloads, so the port
  removes no user-visible API).
- **No compute kernels.** Every `KernelDescriptor` in all seven factories carries a
  `ReaderConfigDescriptor` or `WriterConfigDescriptor`. The compute `opt_level` question does not
  arise for this op.

## Construct — to do

**Tensor bindings** (per binding, per factory) — twelve Case 1, four clean, **no Case 2**:

- `PadRmReaderWriterProgramFactory` and `PadRmReaderWriterMultiCoreProgramFactory` (identical shape,
  same two kernels):
  - `input` — **Case 1** → `TensorParameter` / `TensorBinding`; kernel builds
    `TensorAccessor(tensor::…)`. Legacy: `Buffer*` at reader/writer RTA slot 0
    (`..._program_factory.cpp:143`, `..._multi_core_...:348`) → `TensorAccessor(src_args, src_addr)`
    at `reader_pad_dims_rm_interleaved.cpp:75`.
  - `output` — **Case 1**. Legacy: `Buffer*` at slot 1 (`:144` / `:349`) →
    `TensorAccessor(dst_args, dst_addr)` at `writer_pad_dims_rm_interleaved.cpp:32`.
  - `pad_value_const` *(op-owned)* — **Case 1**. Legacy: `Buffer*` at slot 13 (`:156` / `:361`) →
    `TensorAccessor(pad_tensor_args, pad_value_const_buffer_addr)` at
    `reader_pad_dims_rm_interleaved.cpp:77`.
- `PadRmReaderWriterMultiCoreDefaultProgramFactory`:
  - `input` — **Case 1**. `Buffer*` reader slot 0 (`:221`) →
    `TensorAccessor(src_args, src_addr, accessor_page_size)` at
    `reader_pad_dims_rm_interleaved_v2.cpp:95`.
  - `output` — **Case 1**. `Buffer*` writer slot 0 (`:222`) →
    `TensorAccessor(dst_args, dst_addr, accessor_page_size)` at
    `writer_pad_dims_rm_interleaved_v2.cpp:25`.
- `PadRmShardedHeightOnlyProgramFactory`:
  - `input` — **clean** (borrowed-memory DFB): CB `c_0` is globally allocated to the input buffer
    (`cb_src0.buffer = src_buffer`, `..._height_only_program_factory.cpp:294`) → express as
    `DataflowBufferSpec::borrowed_from` the input tensor parameter.
  - `output` — **clean** (borrowed-memory DFB): CB `c_16` ← output buffer (`:309`) →
    `borrowed_from` the output.
  - This factory passes **no buffer address to any kernel at all** — the comment at
    `..._height_only_program_factory.cpp:380-382` says so explicitly, and its reader/writer contain no
    `TensorAccessor`. Both bindings exist purely as `borrowed_from` sources.
- `PadRmShardedWidthOnlyProgramFactory`:
  - `input` — **clean** (borrowed-memory DFB): CB `c_0` is globally allocated to the input buffer
    (`cb_input.buffer = input_buffer`, `..._width_only_program_factory.cpp:75`) → express as
    `DataflowBufferSpec::borrowed_from` the input tensor parameter. No accessor, no address arg.
  - `output` — **clean** (borrowed-memory DFB): CB `c_16` ← output buffer (`:91`) →
    `borrowed_from` the output.
- `PadTileCoreProgramFactory`:
  - `input` — **Case 1**. `Buffer*` reader slot 0 (`:122`) → donor's
    `TensorAccessor(src_args, src_addr)` at `reader_unary_interleaved_start_id.cpp:30`. *(The donor's
    `_metal2` fork already expresses this as `tensor::src` — see Watch-for.)*
  - `output` — **Case 1**. `Buffer*` writer slot 0 (`:126`) →
    `TensorAccessor(dst_args, dst_addr)` at `writer_unary_pad_dims_interleaved.cpp:30`.
- `PadTileMulticoreProgramFactory`:
  - `input` — **Case 1**. `Buffer*` reader slot 0 (`:222`) →
    `TensorAccessor(dst_args, input_addr)` at `reader_pad_tiled.cpp:29`.
  - `output` — **Case 1**. `Buffer*` writer slot 0 (`:223`) →
    `TensorAccessor(dst_args, output_addr)` at `writer_pad_tiled.cpp:42`.

Every Case-1 binding today arrives as a `Buffer*` pushed into `KernelDescriptor::RTArgList`, i.e. the
framework's `BufferBinding` form — **not** a raw `->address()` RTA. So none of these is the
silently-wrong-on-cache-hit hazard; they are correct today and the port simply replaces the mechanism
with the typed binding.

**TensorParameter relaxation:** none — the only value that reaches a brief.

**TensorAccessor 3rd arg:** drop the redundant page-size argument at **two sites**, both in
`PadRmReaderWriterMultiCoreDefaultProgramFactory`:

- `reader_pad_dims_rm_interleaved_v2.cpp:95` — drop `accessor_page_size`, leaving
  `TensorAccessor(tensor::…)`.
- `writer_pad_dims_rm_interleaved_v2.cpp:25` — same.

Also drop what feeds them on the host: CTA slot 21 (`input_accessor_page_size`,
`..._default_program_factory.cpp:163`), CTA slot 4 (`output_accessor_page_size`, `:171`), and the
`:57-73` computation of both. Both sites are **Class 2 (redundant)** — the sharded branch passes
`buffer->aligned_page_size()` verbatim (exactly what Metal 2.0 supplies) and the interleaved branch
passes the true logical page, which the interleaved accessor realigns to the same value. **Do not set
`dynamic_tensor_shape`** — despite being interleaved row-major, these are *compile-time* args, so the
page size cannot vary across shapes sharing one compiled program.

Note that dropping the reader's CTA 21 shifts `TensorAccessorArgs<22>` at
`reader_pad_dims_rm_interleaved_v2.cpp:81` — but that whole args block disappears with the binding
anyway. Same for `TensorAccessorArgs<5>` at `writer_pad_dims_rm_interleaved_v2.cpp:23`.

**CB endpoints:**

- **Self-loop** (one toucher: single-ended / sync-free — bind the same kernel PRODUCER *and*
  CONSUMER):
  - `c_1` (`cb_pad`) in `PadRmReaderWriterMultiCoreDefaultProgramFactory` — reader-only
    (`reader_pad_dims_rm_interleaved_v2.cpp:19,98`).
  - `c_0` and `c_1` in `PadRmShardedHeightOnlyProgramFactory` — `c_0` is a reader-only raw peek
    (`reader_pad_dims_rm_sharded.cpp:32`, also `borrowed_from` the input); `c_1` (`cb_pad`) is
    writer-only (`writer_pad_dims_rm_sharded.cpp:16,22,43,82`).
  - `c_0` and `c_1` in `PadRmShardedWidthOnlyProgramFactory` — `c_0` is a reader-only raw peek
    (`reader_pad_dims_rm_sharded_stickwise.cpp:29`, also `borrowed_from` the input); `c_1`
    (`padding_value_cb`) is writer-only (`writer_pad_dims_rm_sharded_stickwise.cpp:24,60`).
  - `c_1` in `PadTileCoreProgramFactory` — writer-only; it `reserve_back(1)`s and takes
    `get_write_ptr()` but never pushes (`writer_unary_pad_dims_interleaved.cpp:35,37`).
  - `c_2` (`pad_val`) in `PadTileMulticoreProgramFactory` — writer-only
    (`writer_pad_tiled.cpp:48,49,57`).
- **1P+1C assignment — `c_16` in `PadRmShardedHeightOnlyProgramFactory`.** Two touchers on every
  node: the reader is a locked producer (`reserve_back` / `push_back` at
  `reader_pad_dims_rm_sharded.cpp:31,69`, plus `get_write_ptr` at `:33`), and the **writer raw-peeks
  the same buffer** via `dfb_out0_exp.get_write_ptr()` at `writer_pad_dims_rm_sharded.cpp:90` with no
  FIFO ops. The peek is role-free, so bind **reader PRODUCER, writer CONSUMER** — cosmetic on Gen1,
  no runtime effect. **This is not multi-binding; do not set the advanced option.** The writer *must*
  still be bound: in Metal 2.0 it cannot touch a DFB it hasn't bound, and this raw peek is easy to
  miss because a FIFO-sync trace doesn't show it. `c_16` is also `borrowed_from` the output.
- **Conditional DFB — `c_2` in `PadRmReaderWriterMultiCoreDefaultProgramFactory`. Do NOT drop it.**
  Dead under `stick_size_padded_front == 0 && !unaligned`, live otherwise: the host allocates it only
  under that condition (`..._default_program_factory.cpp:127-139`). Make the `DataflowBufferSpec`
  conditional on the same predicate — **and gate the kernel too.** The kernel constructs the buffer
  unconditionally (`reader_pad_dims_rm_interleaved_v2.cpp:90-93`) and calls
  `dfb_pad_align_exp.get_read_ptr()` unconditionally at `:99`; only the *uses* at `:134/:139/:142` sit
  behind `if constexpr`. Under Metal 2.0 a kernel may not reference a DFB it hasn't bound, so `:99`
  must move behind a real preprocessor `#ifdef` fed by a host-side define — **`if constexpr` will not
  work**, because it does not suppress `dfb::` name lookup, and the resulting failure is a build-time
  unbound-DFB reference rather than anything the tests would catch.
- **Dead-CB drop — `c_1` in `PadTileMulticoreProgramFactory`.** Confirmed dead in every config: no
  kernel touches it. Drop the `CBDescriptor` at `pad_tile_multicore_program_factory.cpp:70-78`, and
  drop the dead CTA carrying its index — `output_cb_index` is pushed as writer CTA 1 at
  `pad_tile_multicore_program_factory.cpp:125` and unpacked-then-never-used at
  `writer_pad_tiled.cpp:23`. Removing both changes L1 footprint and nothing else. **Record both
  `file:line` prominently in the port report.** (Dropping the CTA renumbers the writer's remaining
  compile-time args and shifts `TensorAccessorArgs<7>` at `writer_pad_tiled.cpp:40` — though that
  block disappears with the output binding regardless.)
- **All other CBs are legal 1:1** and need no action: `c_0` in both v1 RM factories, `c_0` in the
  default RM factory, `c_16` in the width-only sharded factory (reader consumes, writer produces),
  `c_0` in both tile factories. **No CB anywhere in this op needs the multi-binding advanced
  option** — no node reaches three touchers and no two kernels lock the same FIFO role. The
  hidden-second-writer hunt found exactly one cross-kernel raw touch (`c_16` above), and the op
  declares no semaphores at all, so the semaphore-gated co-fill shape cannot be present.
- **Four borrowed-memory CBs**, all binding at base with no `address_offset`: `c_0` / `c_16` in
  `PadRmShardedHeightOnlyProgramFactory` (`cb_src0.buffer` / `cb_output.buffer` at
  `..._height_only_program_factory.cpp:294,309`) and `c_0` / `c_16` in
  `PadRmShardedWidthOnlyProgramFactory` (`..._width_only_program_factory.cpp:75,91`) →
  `DataflowBufferSpec::borrowed_from`.

**Translating `override_runtime_arguments`** (`PadRmShardedHeightOnlyProgramFactory` only — the one
`CustomProgramSpecFactoryConcept` target). The method at
`pad_rm_sharded_height_only_program_factory.cpp:412-424` owns the whole cache-hit refresh. Its body
builds a two-entry CB-address-only `ProgramDescriptor` and calls `apply_descriptor_runtime_args`,
mirroring `create_descriptor`'s CB push order **positionally** — input CB, output CB, then the
pad-value CB it deliberately omits. That positional contract is fragile; carry it into the
translation as an explicit comment so a later CB reorder in `create_descriptor` can't silently
mis-target.

> **Open question — raise it, don't decide it silently.** The method exists *solely* to re-point the
> two borrowed CB base addresses. In Metal 2.0 that is exactly what
> `DataflowBufferSpec::borrowed_from` plus the `TensorBinding` refresh does natively, so the body may
> be wholly subsumed — in which case the factory drops to the plain `ProgramSpecFactoryConcept` and
> the override disappears entirely. The evidence that it may already be unnecessary:
> `PadRmShardedWidthOnlyProgramFactory` has the same two borrowed CBs, defines **no** override, and
> relies on the framework's `cb.buffer` patching (see its comment at
> `..._width_only_program_factory.cpp:170-173`). One of the two factories is doing redundant work.
> Dropping the override is a **concept change**, so surface it rather than taking it on your own
> judgement.

## Watch for

- **CB endpoints (multi-binding):** none. Nothing in this op needs the flag. But **do** bind the
  writer on `c_16` in `PadRmShardedHeightOnlyProgramFactory` — its raw peek at
  `writer_pad_dims_rm_sharded.cpp:90` is invisible to a FIFO-sync trace and is the one cross-kernel
  toucher in the whole op (see *Construct* → 1P+1C).
- **Cross-op / shared kernels:** `PadTileCoreProgramFactory` file-path-instantiates
  `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`
  (`pad_tile_program_factory.cpp:104-105`).
  - **A `_metal2` fork already exists beside it** — `reader_unary_interleaved_start_id_metal2.cpp`,
    same directory. **Bind it; do not re-fork and do not convert the legacy file in place.**
  - **The fork owns the vocabulary — conform the factory to it, never rename the kernel.** Its
    bindings are `dfb::in`, `tensor::src`, and named args `args::num_pages`, `args::start_id`. Its
    header states these are its interface and are not renamed once a consumer exists. Note it is
    `tensor::src` — *not* `tensor::input`.
  - The fork has already replaced `get_local_cb_interface(cb_id_in0).fifo_page_size` with
    `dfb.get_entry_size()`, so you inherit that move rather than making it.
  - **Trap:** a second, differently-named fork of the same kernel lives at
    `ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/dataflow/reader_unary_interleaved_start_id_metal2.cpp`
    (vocabulary `tensor::input`). It is an op-local fork, not the one beside the original. Do not bind
    it.
  - Other ops still binding the legacy file — a **sunset list, not authorization to convert the
    kernel in place**: `data_movement/untilize_with_unpadding` (2 factories), `examples/example` (2),
    `examples/example_multiple_return`, `experimental/transformer/nlp_create_qkv_heads_falcon7b`,
    `reduction/topk` (`topk_route_prep`), plus `tests/.../test_generic_op.cpp` and
    `tests/.../test_generic_op.py`. The legacy copy can be deleted only when the last of them
    migrates.
- **RTA varargs — two genuine sites.**
  **(1) `reader_pad_dims_rm_sharded.cpp:17-22`** (`PadRmShardedHeightOnlyProgramFactory`). The arg
  block is `num_cores`, then `2·num_cores` NoC x/y values, then `num_cores` chunk counts, then
  `2·Σchunks` `(start_id, length)` pairs — read through `get_arg_addr()` at **runtime-computed**
  offsets (`get_arg_addr(1 + num_cores_read * 2)`, `… * 3`), with per-core counts driving the loops
  at `:38,44`. Host side at `pad_rm_sharded_height_only_program_factory.cpp:158-186`. No
  per-argument names exist to infer — use the RTA vararg mechanism.
  **(2) `reader_pad_tiled.cpp:22-25` and `writer_pad_tiled.cpp:35-38`** (`PadTileMulticoreProgramFactory`).
  Four consecutive `num_dims`-long blocks (`input_page_shape`, `output_page_shape`,
  `input_id_per_dim`, `output_id_per_dim`) are reached via `get_arg_addr(rt_ind)` plus `+ num_dims`
  pointer strides and consumed in `for (d < num_dims)` loops (`:46`, `:66`, and
  `device/kernels/dataflow/common.hpp:12`); `num_dims` is CTA 2 = `output_padded_shape.rank()`. Host
  side at `pad_tile_multicore_program_factory.cpp:236-253`. Reach for the RTA vararg mechanism rather
  than naming each element. *(Rank is in fact pinned to 4 upstream — `pad.cpp:193` and
  `pad_device_operation.cpp:121` — so naming the 16 values would also be defensible if you'd rather
  keep the kernel's rank-generic loops intact. Either is fine; the vararg route is the safer default.)*
  - **These two look like varargs and are not — name them.**
    `reader_pad_dims_rm_interleaved_v2.cpp:59` (`start_dim_offset = get_arg_addr(7)`, read at `:112`)
    and `writer_pad_dims_rm_sharded.cpp:53` (`get_arg_addr(5)`, read at `:93`) both index the array at
    **fixed** positions `[1]`, `[2]`, `[3]` only. Each is three distinct nameable fields (`start_h`,
    `start_c`, `start_n`) reached through legacy pointer arithmetic, not a data-directed pick. Don't
    let them ride a vararg block — note the second one sits in the *same kernel pair* as genuine
    vararg site (1), so keep them apart.
  - **No CTA varargs anywhere.** The `kernel_compile_time_args[13]` read at
    `reader_pad_dims_rm_interleaved_v2.cpp:85` and `[10..12]` at
    `writer_pad_dims_rm_sharded.cpp:70-72` use *constant* indices inside
    `if constexpr (not_pad_by_zero)`, and the host emits those slots unconditionally
    (`..._default_program_factory.cpp:141-163`, `..._height_only_program_factory.cpp:337-350`) — fixed
    named CTAs on both branches, not variable-count blocks. Don't reach for `compile_time_varargs`.
    (Because the host emits them unconditionally, no `#ifdef` gymnastics are needed here either —
    unlike the `c_2` DFB above.)
- **`PadRmReaderWriterMultiCoreProgramFactory` is unreachable — raise it before you port it.** It is
  declared in the `program_factory_t` variant (`pad_device_operation.hpp:33`) but
  `select_program_factory` never returns it: the RM multicore path returns
  `PadRmReaderWriterMultiCoreDefaultProgramFactory` instead (`pad_device_operation.cpp:99-101`). No
  test can exercise it, so a port of its 433 lines — including an op-owned-tensor allocation and a
  hardcoded resnet core split — is unverifiable by construction. The audit asked the user whether it
  should be deleted rather than ported; check the answer before spending effort here. If it stays,
  note it must remain in the `program_factory_t` variant: a `create_program_artifacts` outside the
  variant builds green and fails on device.
- **Idle-core `0u` sentinel disappears by itself.** Two factories push a literal `0u` instead of the
  `Buffer*` on cores with no work (`..._default_program_factory.cpp:224-225`,
  `pad_tile_multicore_program_factory.cpp:225-226`) to skip `BufferBinding` registration. Under
  Metal 2.0 the tensor base rides a broadcast CRTA, so there is nothing to skip and the branch has no
  translation — drop it. Both kernels already short-circuit on `num_sticks_per_core == 0` /
  `num_pages_per_core == 0`, and in practice `split_work_to_cores` returns
  `all_cores == group_1 ∪ group_2`, so no idle core exists today anyway.
- **The v1 RM kernels are handed identical 27-slot arg lists.** `writer_rt_args = reader_rt_args`
  (`pad_rm_reader_writer_program_factory.cpp:172`,
  `pad_rm_reader_writer_multi_core_program_factory.cpp:385`), so each kernel receives roughly half
  the slots it never reads, and slot indices are shared across the two. When you convert to named
  args, give each kernel only the args it actually reads — don't preserve the shared layout. Slots
  each kernel unpacks but never uses (safe to drop entirely): reader `num_total_W`(3),
  `num_total_Y`(7), `start_src_stick_wi`(18), `full_unpadded_X_nbytes`(23); writer `num_total_W`(3),
  `num_total_Y`(7), `num_total_X`(9), `num_local_unpadded_Y`(22), `full_padded_X_nbytes`(24),
  `start_dst_stick_wi`(19). The writer's compile-time args likewise copy in the pad-value tensor's
  `TensorAccessorArgs` (`:85`) that it never instantiates.
- **`pad_value_const_buffer_nbytes` (slot 14) is a live arg the kernel ignores.**
  `reader_pad_dims_rm_interleaved.cpp:52` hardcodes `64` with a *"fails on BH when > 64"* comment
  (issue #21978) while the host still computes and passes the real value
  (`pad_rm_reader_writer_program_factory.cpp:157`). Port the kernel's behavior as-is — the hardcode
  stays. Just don't create a named arg for a value nothing reads.
- **The two `WorkloadDescriptor` factories replicate one descriptor across ranges.** Each builds a
  single `ProgramDescriptor` above the loop and pushes it once per range in `tensor_coords`
  (`pad_rm_reader_writer_program_factory.cpp:202-211`,
  `..._multi_core_program_factory.cpp:421-429`). Nothing is per-mesh-coordinate, so this collapses
  cleanly onto the single-program `ProgramSpecFactoryConcept` — you are removing a wrapper, not
  losing per-coordinate behavior.
