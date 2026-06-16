# Metal 2.0 Port Report — `pad` (PadTileCoreProgramFactory)

## Status: PORTED

One factory ported; the op's other six factories remain on their legacy concepts
(supported — the `program_factory_t` variant dispatches per-factory by concept, and
`AllFactoriesValid` in `operation_concepts.hpp` validates each variant alternative
independently).

## Factory chosen

`PadTileCoreProgramFactory` — the simplest single-program factory in the op:

- Single core `{0,0}` (no work-split, no per-core loop).
- Two kernels: a tilized reader + the pad writer.
- Two CBs; no semaphores; no op-owned/scratch device tensors; no `GlobalSemaphore`.

Selected at runtime for TILE-layout, non-multicore pad (`select_program_factory`,
`pad_device_operation.cpp:104-108`). Unchanged — still returns `PadTileCoreProgramFactory{}`.

## Legacy → Metal 2.0 mapping

| Legacy (ProgramDescriptor)                  | Metal 2.0                                              |
|---------------------------------------------|-------------------------------------------------------|
| `create_descriptor` → `ProgramDescriptor`   | `create_program_spec` → `ProgramArtifacts{spec,run}`  |
| CB c_0 (src0, 2 tiles)                       | DFB `in0` (reader PRODUCER, writer CONSUMER)          |
| CB c_1 (src1, 1 tile, writer-only scratch)   | DFB `pad` (writer self-loop: PRODUCER + CONSUMER)     |
| `TensorAccessorArgs(src0)` CTA + addr RTA    | `TensorParameter{src}` + `TensorBinding` on reader    |
| `TensorAccessorArgs(dst)` CTA + addr RTA     | `TensorParameter{out}` + `TensorBinding` on writer    |
| writer CTAs `{src0_cb_index, src1_cb_index}` | dropped (DFB bindings carry CB identity)              |
| positional reader RTAs `{addr, n, 0}`        | named `{num_pages, start_id}`                          |
| positional writer RTAs (10 slots)            | named (`pad_value` etc.); buffer-addr slot dropped    |

DFBs declared with `entry_size = single_tile_size`, `num_entries = {2, 1}`,
`data_format_metadata = cb_data_format` — copied 1:1 from the legacy CBs. No
`tile_format_metadata` (legacy CBs set no `.tile`).

## Local-DFB / WorkUnit structure

Both DFBs are local. `in0` has producer (reader) and consumer (writer); `pad` is a
writer self-loop. Both kernels run on the single core, so one `WorkUnitSpec`
(`single_core`) hosts `{reader, writer}` — satisfying the rule that every node hosting
a DFB hosts both endpoints.

## Kernels

- **Reader — FORKED.** `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`
  is shared by ~12 ops (`grep -rl`), so it was forked to
  `pad/device/kernels/dataflow/reader_unary_interleaved_start_id_m2.cpp` and the fork
  ported. Only the access mechanism changed: `cb_id_in0 (=0)` → `dfb::in0`,
  `TensorAccessor(src_args, src_addr)` → `TensorAccessor(ta::src)`, positional RTAs →
  `get_arg(args::num_pages/start_id)`. The `#ifdef BACKWARDS` branch (unused on the pad
  path) is preserved verbatim. (Note: matmul's port forked a *different* shared reader;
  this pad path uses the eltwise/unary reader, hence a separate fork.)

- **Writer — PORTED IN PLACE.** `pad/device/kernels/dataflow/writer_unary_pad_dims_interleaved.cpp`
  is used only by this factory (`grep -rl` → single hit), so it was ported in place.
  `cb_id_out0 (=0)` → `dfb::in0` (consumer), `cb_id_out1 (=1)` → `dfb::pad` (self-loop
  scratch), `TensorAccessor(dst_args, dst_addr)` → `TensorAccessor(ta::out)`, positional
  RTAs → named. `get_tile_size(dfb::in0)` relies on the `DFBAccessor → uint32_t` implicit
  conversion. Pad-fill loop / NoC writes / `CoreLocalMem` logic unchanged.

## Device-op-class edits

- **Custom `compute_program_hash`**: NONE on `PadDeviceOperation` — nothing to delete.
- **Pybind hook**: NONE — pad dispatches through the device-op adapter; there is no
  direct `create_descriptor` nanobind hook for this factory (verified: no
  `create_descriptor`/`create_program` references in `pad_nanobind.cpp`).
- Factory `.hpp` updated: declares `create_program_spec` returning `ProgramArtifacts`;
  dropped the now-unused `<tt-metalium/program_descriptors.hpp>` include, added
  `ttnn/metal2_artifacts.hpp`. The `program_factory_t` variant and `select_program_factory`
  are unchanged.

## Blockers

None. The port is a clean mechanical translation.

## Friction / notes

- **Worktree base mismatch (environmental, not a port blocker).** This worktree is NOT
  on `dgomez/rand-metal2`; it has the Metal 2.0 API headers but none of the worked
  example ports. The worked examples (matmul `create_program_spec`, rand
  `create_program_spec`, the `*_m2` kernels) live on `dgomez/rand-metal2` at commit
  `01c3e051d27`; I read them via `git show`. The port itself is unaffected.
- The legacy writer carries an unused local `src_tile_id` variable; preserved verbatim
  (scope discipline — not the port's concern).
- The reader's `num_pages` arg semantically equals the legacy `num_unpadded_tiles`, and
  `start_id` is hard `0`; both are per-core RTAs on the single core, matching the legacy
  `emplace_runtime_args(CoreCoord{0,0}, {src0_buffer, num_unpadded_tiles, 0})`.

## Verification

NOT built / not run (per instructions — this worktree has no build dir). Self-audited
against the kernel-side whitelist and the local-DFB rule. A grep for `CBDescriptor` /
`CircularBuffer` over the three ported files returns zero hits in code.

---

# Metal 2.0 Port Report — `pad` (PadTileMulticoreProgramFactory)

## Status: PORTED

Second factory of the op ported. The remaining five (RM single/multi-core, RM sharded
height/width, the legacy RM default) stay on their legacy concepts — supported, the
`program_factory_t` variant dispatches per-factory by concept.

## Factory chosen

`PadTileMulticoreProgramFactory` — the default-selected factory for normal TILE-layout pad
(`select_program_factory`, `pad_device_operation.cpp:104-106`: TILE layout + `use_multicore`).
Interleaved, multi-core, work-split over `split_work_to_cores`, two DM kernels (reader +
writer), no compute kernel, no semaphores, no op-owned tensors. `select_program_factory`
and the `program_factory_t` variant are UNCHANGED — still return/list
`PadTileMulticoreProgramFactory{}`; the framework now dispatches it via the
`MetalV2FactoryConcept` adapter because the factory exposes `create_program_spec`.

## Legacy → Metal 2.0 mapping

| Legacy (ProgramDescriptor)                       | Metal 2.0                                                       |
|--------------------------------------------------|----------------------------------------------------------------|
| `create_descriptor` → `ProgramDescriptor`        | `create_program_spec` → `ProgramArtifacts{spec,run}`           |
| CB c_0 input (2 pages)                           | DFB `in0` (reader PRODUCER, writer CONSUMER)                   |
| CB c_1 output (2 pages, allocated but unused)    | DFB `out` (writer self-loop PRODUCER+CONSUMER — see note)     |
| CB c_2 pad-val (1 page, writer scratch)          | DFB `pad` (writer self-loop PRODUCER+CONSUMER)                |
| `TensorAccessorArgs(input)` CTA + addr RTA slot0 | `TensorParameter{src}` + `TensorBinding` on reader            |
| `TensorAccessorArgs(output)` CTA + addr RTA slot0| `TensorParameter{dst}` + `TensorBinding` on writer            |
| reader CTAs `{input_cb, page_size, rank}`        | DFB binding + named CTAs `{page_size, num_dims}`              |
| writer CTAs `{in_cb,out_cb,pad_cb,page_size,rank,pad_value,elem_size}` | DFB bindings + named CTAs `{page_size,num_dims,pad_value,element_size}` |
| reader/writer per-core RTAs `{addr, n_pages, offset}` | named `{num_pages_to_write, start_offset}` (addr dropped) |
| per-core per-dim arrays (4×rank) via `get_arg_addr` | runtime **varargs** (`num_runtime_varargs = num_dims*4`)   |

DFB `entry_size = page_size`, `num_entries = {2, 2, 1}`, `data_format_metadata = cb_data_format`
— copied 1:1 from the legacy CBs (no `.tile` set legacy-side, so no `tile_format_metadata`).

## Local-DFB / WorkUnit structure

All three DFBs are local. `in0` is reader-PRODUCER → writer-CONSUMER; `out` and `pad` are
writer self-loops. Both kernels run on `all_cores`, so one `WorkUnitSpec` (`multi_core`) hosts
`{reader, writer}` — every node hosting a DFB hosts both endpoints.

## Kernels

- **Reader — `reader_pad_tiled.cpp`, PORTED IN PLACE.** Single consumer (`grep -rl` → this
  factory only). `input_cb_id` → `dfb::in0`; `TensorAccessor(dst_args, input_addr)` →
  `TensorAccessor(ta::src)` (addr via TensorBinding; the appended `TensorAccessorArgs<3>` CTA
  is gone); `page_size`/`num_dims` positional CTAs → named; `num_pages_to_write`/`start_offset`
  positional RTAs → named; the 4 per-dim arrays → runtime varargs read into local stack arrays.
- **Writer — `writer_pad_tiled.cpp`, PORTED IN PLACE.** Single consumer. `input_cb_id` →
  `dfb::in0` (CONSUMER); `pad_val_cb_id` → `dfb::pad` (self-loop); `TensorAccessor(dst_args,
  output_addr)` → `TensorAccessor(ta::dst)`; named CTAs/RTAs as above; per-dim arrays → varargs.
  The legacy `output_cb_id` CTA is dropped — the kernel never used it; its CB is preserved on
  the host as the `out` self-loop DFB (see Open items).
- **`common.hpp` (op-local, used only by these two kernels).** `advance_tensor_index` parameter
  type changed from `volatile tt_l1_ptr uint32_t*` to plain `uint32_t*` to match the new
  local-stack-array access mechanism (varargs have no writable L1 pointer). Wrap/carry
  arithmetic unchanged.

## Device-op-class edits

None. No custom `compute_program_hash`, no pybind `create_descriptor` hook. Only the factory
`.hpp` changed (now declares `create_program_spec`; dropped `<program_descriptors.hpp>`, added
`ttnn/metal2_artifacts.hpp`).

## Blockers

None.

## Open items for downstream

- **Fake-CB self-loop (`out` DFB) — interim hack, flag prominently.**
  `pad_tile_multicore_program_factory.cpp` binds the legacy `output` CB (c_1) as a writer
  self-loop (PRODUCER + CONSUMER, accessor `out`) purely to satisfy the validator's
  producer+consumer rule. It is **not** a real FIFO and the kernel never references `dfb::out`;
  the binding exists only to preserve the legacy L1 allocation (page_size×2). This CB appears
  unused in the legacy kernel as well — a candidate for outright removal in a follow-up
  (would change L1 footprint, so left in place here for behavior preservation).
- **Runtime varargs (`num_runtime_varargs = num_dims*4`)** in both kernels carry the per-dim
  shape/index arrays. Justified-vararg case (N-dim arrays gated on the `num_dims` CTA), same
  pattern as the slice tile port. To be revisited when typed `std::array` kernel args land.

## Friction / notes

- Followed the slice tile port (`slice_program_factory_tile.cpp`, `*_m2.cpp`) as the precedent
  for the per-dim-array → varargs translation and the local-stack-array mutation workaround.
- The `out` self-loop is the one judgment call: dropping the CB entirely was tempting but would
  change L1 footprint (not behavior-preserving), so it was preserved as a self-loop and flagged.

## Verification

NOT built / not run (per instructions). Self-audited against the kernel-side whitelist and the
local-DFB rule. Watch at build time: `common.hpp`'s `advance_tensor_index` now takes plain
`uint32_t*`; the local stack arrays bind directly (no address-space conversion). A grep for
`CBDescriptor` / `CircularBuffer` / `TensorAccessorArgs` over the ported files returns zero hits.

---

# Metal 2.0 Port Report — `pad` (remaining RM + sharded factories)

This pass ports the five remaining `pad` factories (the two tile factories were already done).
After this pass ALL seven factories of `PadDeviceOperation` are on `create_program_spec`. The
device-op `program_factory_t` variant and `select_program_factory` are UNCHANGED — the framework
dispatches each factory via the `MetalV2FactoryConcept` adapter now that each exposes
`create_program_spec`. No custom `compute_program_hash` exists on `PadDeviceOperation` (nothing to
keep/delete); no pybind `create_descriptor`/`create_workload_descriptor` hooks exist in
`pad_nanobind.cpp` (verified).

## Per-factory status

### 1. PadRmReaderWriterProgramFactory (single-core RM) — PORTED
- `create_workload_descriptor` (WorkloadDescriptor variant) → `create_program_spec`.
- DFB: `in0` (c_0, 16 pages, local) — reader PRODUCER, writer CONSUMER.
- TensorParameters: `src`, `dst`, and `pad` (the pad-value const tensor). The legacy const tensor
  was a `WorkloadDescriptor::buffers`-parked Tensor; it is now an **op-owned tensor** in
  `ProgramArtifacts::op_owned_tensors` (allocated once on a miss, parked at a stable address by the
  adapter, bound via a TensorParameter + TensorBinding, read kernel-side through `ta::pad`).
- Kernels FORKED to `*_m2.cpp` (shared with factory #2): `reader_pad_dims_rm_interleaved_m2.cpp`,
  `writer_pad_dims_rm_interleaved_m2.cpp`. The legacy reader/writer read a single shared 27-slot
  RTA vector by positional index with gaps; only the slots each kernel actually consumed survive as
  named RTAs. src/dst/pad addresses (slots 0/1/13) → TensorBindings. Logic unchanged.

### 2. PadRmReaderWriterMultiCoreProgramFactory — PORTED
- `create_workload_descriptor` → `create_program_spec`; multi-core work loop preserved verbatim
  (`split_across_cores`, the resnet-shape switch).
- Same DFB (`in0`), same three TensorParameters incl. the op-owned `pad` const tensor, same forked
  `*_m2` kernels as #1. Per-core scalar RTAs are now named; the three buffer-address slots are
  carried by the src/dst/pad TensorBindings.
- Dropped the `#if 1` debug-log block and the `log_rt_args` helper (their only inputs — the legacy
  CTA list / buffer `->address()` calls — no longer exist).

### 3. PadRmReaderWriterMultiCoreDefaultProgramFactory (RM `_v2`, default RM path) — PORTED
- `create_descriptor` (ProgramDescriptor variant) → `create_program_spec`.
- DFBs: `in0` (c_0, local; reader PRODUCER → writer CONSUMER), `pad` (c_1 pad-fill scratch — fake CB,
  reader self-loop), `pad_align` (c_2 — fake CB, reader self-loop, **conditionally** bound only when
  `stick_size_padded_front != 0 || unaligned`, exactly mirroring the legacy conditional c_2 alloc).
- Kernels PORTED IN PLACE (1:1, used only by this factory):
  `reader_pad_dims_rm_interleaved_v2.cpp`, `writer_pad_dims_rm_interleaved_v2.cpp`.
- Conditional-DFB pattern applied: a `FRONT_PAD_OR_UNALIGNED` define gates the kernel's
  `dfb::pad_align` alias and every reference to it (the `if constexpr(front_padding)` /
  `else if constexpr(unaligned)` chain plus the file-scope `get_read_ptr()`); the plain-read `else`
  runs when the define is absent. Promoting the in-kernel `get_read_ptr` of an unallocated CB into
  the `#ifdef` is a behavior preserve (legacy only *used* it under the same constexpr guards).
- `start_dim_offset[4]` per-core array → runtime varargs (fixed at 4, matching the kernel's [1]/[2]/[3]
  indexing and the post-first-iteration 4-element host vector).

### 4. PadRmShardedHeightOnlyProgramFactory — PORTED
- `create_descriptor` → `create_program_spec`.
- **Borrowed-memory DFBs**: `in0` `borrowed_from = src` (input shard L1), `out0` `borrowed_from = dst`
  (output shard L1); plus `pad` (scratch, writer self-loop). `out0` is reader PRODUCER (reserve/push)
  → writer CONSUMER (base-pointer fill); `in0` is reader base-pointer-only → reader self-loop.
- Kernels PORTED IN PLACE: `reader_pad_dims_rm_sharded.cpp`, `writer_pad_dims_rm_sharded.cpp`.
  In-kernel CB ids (c_0/c_16/c_1) → `dfb::in0`/`dfb::out0`/`dfb::pad`.
- The reader's packed **variable-length** per-core RTA tail (per-core NoC x/y + stick-chunk lists,
  read via `get_arg_addr` pointer arithmetic in legacy) → runtime varargs. `num_cores_read` is a
  named RTA; `num_runtime_varargs` is fixed at the **max tail length** across cores and shorter cores
  are zero-padded (the kernel only indexes within its own `num_cores_read` range, so trailing pad is
  never read — this avoids the deprecated `num_runtime_varargs_per_node`). Writer `start_dim_offset[4]`
  → varargs.

### 5. PadRmShardedWidthOnlyProgramFactory (stickwise) — PORTED
- `create_descriptor` → `create_program_spec`.
- **Borrowed-memory DFBs**: `in0` `borrowed_from = src`, `out0` `borrowed_from = dst`; plus `pad`
  (scratch, writer self-loop). `out0` is writer PRODUCER (reserve/push) → reader CONSUMER (in-place
  fill after wait_front); `in0` is reader base-pointer-only → reader self-loop.
- Kernels PORTED IN PLACE: `reader_pad_dims_rm_sharded_stickwise.cpp`,
  `writer_pad_dims_rm_sharded_stickwise.cpp`. In-kernel CB-id CTAs → `dfb::` handles. No RTAs (CTA-only
  kernels); the borrowed DFBs pull their L1 base from the src/dst tensor args at runtime.

## Blockers

None. No factory hit the slice-RM "runtime base-offset / per-shard page-size that `ta::` can't
express" gap: the interleaved RM readers/writers use plain page-id/offset `TensorAccessor` access
(Case-1), and the sharded factories never use `ta::` at all — they use **borrowed-memory DFBs** for
the input/output shards and base-pointer reads, which the API supports directly.

## Open items for downstream

- **Fake-CB self-loops (interim).** `_v2` reader: `pad` (c_1) and `pad_align` (c_2); height-only
  writer: `pad` (c_1); width-only writer: `pad`. All are address-source scratch CBs bound as
  self-loops purely to satisfy the DFB producer+consumer invariant — not real FIFOs. Replace with the
  forthcoming Metal 2.0 kernel-scratchpad / local-TensorAccessor resource.
- **Borrowed input shards as self-loops.** `in0` in both sharded factories is read by base pointer
  only (no FIFO); bound as a reader self-loop on top of the borrowed (`borrowed_from`) input shard.
- **Retained varargs.** `_v2` (start_dim_offset[4]); height-only (reader packed tail max-padded +
  writer start_dim_offset[4]). The height-only reader's tail is genuinely variable-length per core —
  flagged because the max-pad-then-zero workaround is the alternative to the deprecated
  `num_runtime_varargs_per_node`. Revisit when typed `std::array` kernel args land.
- **Forked shared kernels** `reader/writer_pad_dims_rm_interleaved_m2.cpp` shadow the legacy
  `*_interleaved.cpp` (still used by no other op — pad-internal share between factories #1 and #2).
  Sunset the legacy copies once nothing else references them.

## Verification

NOT built / not run (per instructions — no build dir). Self-audited against the kernel-side whitelist
and the local-DFB rule. A grep for `CBDescriptor` / `CircularBuffer` / `TensorAccessorArgs` /
`ProgramDescriptor` / `WorkloadDescriptor` over the five ported factories returns zero hits in code;
over the eight ported/forked kernels, the only `CircularBuffer`/`get_arg_addr` hits are in comments
documenting the legacy layout.
