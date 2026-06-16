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
