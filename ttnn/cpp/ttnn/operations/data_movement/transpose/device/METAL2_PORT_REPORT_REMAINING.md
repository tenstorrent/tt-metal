# Metal 2.0 Port Report — transpose (remaining 6 factories)

This pass ports the remaining six `transpose` factories to `create_program_spec`
(`ttnn::device_operation::ProgramArtifacts`), completing the device-op (CN and WH interleaved were
ported earlier — see `METAL2_PORT_REPORT.md` / `METAL2_PORT_REPORT_WH.md`). All ports are faithful,
behavior-preserving translations: legacy logic, `#ifdef`s, loop bounds, and comments are unchanged;
only the host API and the kernel-side access mechanism move to Metal 2.0 named bindings.

`select_program_factory` already routes to every factory by value (it returns the factory structs
directly), and the `program_factory_t` variant already lists all eight; no device-op wiring change
was needed. No legacy `create_descriptor` pybind hooks existed for any of these factories
(`transpose_nanobind.cpp` exposes only the user-facing op). The custom hash question does not apply —
`TransposeDeviceOperation` has no custom `compute_program_hash`.

## Per-factory STATUS

### 1. transpose_hc_rm — PORTED (interleaved RM)
- One DFB `src0` (legacy c_0); reader produces sticks, writer consumes.
- Kernels (both op-local, ported **in place**): `reader_unary_transpose_hc_interleaved_partitioned_rm.cpp`,
  `writer_unary_transpose_hc_interleaved_start_id_rm.cpp`.
- Buffer-address RTA (legacy slot 0) → `ta::input` / `ta::output` TensorBindings. The legacy
  `aligned_page_size` CTA was consumed only by the `TensorAccessorArgs` plumbing (never read by the
  kernel) and disappears with it. Per-core RTAs named.

### 2. transpose_hc_tiled — PORTED (interleaved tiled; misaligned scratch fake-CB self-loop)
- DFB `src0` (c_0, reader→writer). When `misaligned`, an extra DFB `scratch` (legacy c_1) is a **fake
  CB** — used purely as an address source (`get_write_ptr` + manual copy), bound as a **self-loop**
  (PRODUCER+CONSUMER) on the reader to satisfy the validator. The legacy `MISALIGNED` constexpr is
  promoted to the `MISALIGNED` define so `dfb::scratch` is name-looked-up only on the misaligned build.
- Reader ported **in place** (`reader_unary_transpose_hc_interleaved_partitioned.cpp`). Writer
  **reuses** the existing transpose-local fork `writer_unary_interleaved_start_id_m2.cpp` (the legacy
  writer was the shared `eltwise/unary/.../writer_unary_interleaved_start_id.cpp`); its magic c_0 CTA →
  `dfb::out` consuming the shared `src0` DFB.

### 3. transpose_hc_tiled_interleaved — PORTED (padding-aware; conditional padding FIFO DFB via #ifdef)
- DFB `src0` (c_0, reader→writer). When `needs_padding`, an extra real FIFO DFB `padding` (legacy c_1)
  is produced by the reader and consumed by the writer; conditionally bound on both kernels and gated
  kernel-side by the **promoted `NEEDS_PADDING` define** (legacy gated via `if constexpr (needs_padding)`).
- Reader is shared with the unmigrated `permute_tiled` factory → **forked** to
  `reader_unary_transpose_hc_interleaved_tiled_padding_aware_m2.cpp`. Writer ported **in place**
  (`writer_unary_transpose_hc_interleaved_tiled_padding_aware.cpp`); its magic c_0 CTA → `dfb::out`.
- `swap_hw`/`H`/`W`/… reader CTAs preserved verbatim (transpose always passes `swap_hw = 0` and the
  `1u` placeholders), since the forked kernel still reads them.

### 4. transpose_hc_sharded — PORTED (borrowed-memory DFBs; array-tail RTAs → runtime varargs)
- Two **borrowed** DFBs: `src0`←input, `out`←output (legacy c_0 / c_16 with `.buffer` set). Both are
  accessed by base pointer only (`get_read_ptr`/`get_write_ptr` + NOC), with no real FIFO, so each is
  bound as a **self-loop** on every kernel that touches it.
- The legacy variable-length per-core RTA tails (read via `get_arg_addr` pointer arithmetic — per-core
  stick offsets + NOC x/y in the special case; shard-grid x/y maps in the generic case) → **runtime
  varargs** (`advanced_options.num_runtime_varargs`, zero-padded to the per-launch max), with the
  leading scalars kept as named RTAs. The packed layout is preserved exactly.
- Multi-variant within one factory: generic vs `USE_SPECIAL_CASE`. The writer kernel exists **only** in
  the special case (the generic path runs everything through the reader, matching the legacy empty
  writer). Both kernels ported **in place** (`reader_/writer_unary_transpose_hc_sharded_rm.cpp`); the
  two legacy `get_runtime_args_*` host helpers are reproduced verbatim.

### 5. transpose_wh_sharded — PORTED (borrowed DFBs; 3 kernel forks)
- Two borrowed DFBs `src0`/`out` (c_0/c_16). reader pushes `src0`, writer waits on `out`, compute
  consumes `src0` and produces `out`.
- Three shared kernels **forked** into transpose's dir: `reader_unary_sharded_m2.cpp` (from
  eltwise/unary), `writer_unary_sharded_m2.cpp` (from data_movement/sharded), `transpose_wh_sharded_m2.cpp`
  (compute, from transpose; legacy source also used by the unmigrated create_qkv_heads* / split_qkv ops).
  Magic CB-index CTAs → `dfb::src0`/`dfb::out`. fp32 `unpack_to_dest_mode` preserved on `src0`.
- Targets the active shard cores (`all_cores`), like the tilize sharded exemplar. The legacy factory
  additionally launched the kernels on the full grid with zeroed args on the trailing no-op cores;
  those cores did no work, so this is behavior-preserving (noted inline).

### 6. transpose_wh_sharded_rm — PORTED (borrowed + scratch DFBs; ht>8 conditional; reuses transpose_wh_rm_m2 SHARDED branch)
- DFBs: `src0`←input (borrowed, reader self-loop base-ptr), `out`←output (borrowed),
  `in_scratch` (c_24, reader→compute), `tilize` (c_25, compute self-loop). When `ht > 8`, an extra
  `out_stage` (c_27) DFB: compute produces, writer consumes; `out` is then written by the writer via
  base pointer (writer self-loop). When `ht <= 8`, compute pack-untilizes directly into `out`
  (compute self-loop) and there is **no writer kernel**.
- The legacy c_26 ("im2") DFB is dead (no kernel reads/writes it) → omitted (a Metal 2.0 DFB requires
  ≥1 producer and ≥1 consumer), mirroring the WH-interleaved port's handling of its dead c_25.
- Compute **reuses the existing `transpose_wh_rm_m2.cpp` SHARDED branch**, which this pass finished
  porting: its raw `tt::CBIndex::c_24/c_25/c_27/c_16` constants → `dfb::in_scratch`/`dfb::tilize`/
  `dfb::out_stage`/`dfb::out`, its positional SHARDED CTAs (slots 3–8) → named CTAs, and its
  `(Ht > 8) ? c_27 : c_16` compile-time ternary → the **promoted `OUT_STAGE` define** so the unbound
  output token never enters name lookup. (The non-SHARDED interleaved path of that kernel is unchanged.)
  Reader/writer ported **in place** (`reader_/writer_unary_transpose_wh_sharded_rm.cpp`).

## Kernels: ported vs forked

- **Ported in place** (op-local): `reader_unary_transpose_hc_interleaved_partitioned_rm.cpp`,
  `writer_unary_transpose_hc_interleaved_start_id_rm.cpp`,
  `reader_unary_transpose_hc_interleaved_partitioned.cpp`,
  `writer_unary_transpose_hc_interleaved_tiled_padding_aware.cpp`,
  `reader_unary_transpose_hc_sharded_rm.cpp`, `writer_unary_transpose_hc_sharded_rm.cpp`,
  `reader_unary_transpose_wh_sharded_rm.cpp`, `writer_unary_transpose_wh_sharded_rm.cpp`.
- **Forked** (legacy shared by unmigrated consumers):
  `reader_unary_transpose_hc_interleaved_tiled_padding_aware_m2.cpp` (← permute_tiled),
  `reader_unary_sharded_m2.cpp` (← eltwise/unary, ~dozen ops),
  `writer_unary_sharded_m2.cpp` (← data_movement/sharded, ~dozen ops),
  `transpose_wh_sharded_m2.cpp` (← create_qkv_heads* / split_qkv).
- **Reused** (existing forks on this branch): `writer_unary_interleaved_start_id_m2.cpp` (hc_tiled
  writer), `transpose_wh_rm_m2.cpp` (wh_sharded_rm compute — SHARDED branch finished this pass).

## Open items / Friction (for downstream)

- **Fake-CB self-loops** (interim workarounds, not real FIFOs): hc_tiled `scratch` (misaligned address
  source); hc_sharded `src0`/`out` (borrowed base-pointer access); wh_sharded_rm `src0` (reader) and
  `out` (writer, ht>8). Each is bound PRODUCER+CONSUMER purely to satisfy the validator. These map to
  the forthcoming Metal 2.0 kernel-scratchpad / local-TensorAccessor resources.
- **hc_sharded special-case double self-loop (verify at build/validation):** in the special case both
  the reader and the writer touch the borrowed `src0` and `out` DFBs purely by base pointer, so each is
  self-looped on **both** kernels (yielding 2 producers + 2 consumers per DFB). The single-kernel
  borrowed self-loop is precedented (pad sharded `in0`), but the two-kernel form here is not yet
  exercised elsewhere; if the spec validator enforces a single-producer rule for these DFBs, this is
  the spot to revisit (e.g. designate one kernel PRODUCER and the other CONSUMER). Could not be
  build-verified in this worktree (no build dir).
- **Runtime varargs** retained in hc_sharded reader (+ writer in the special case): the per-core packed
  tails are genuinely variable-length and read via runtime indexing, so varargs are the right fit today
  (slated to migrate to typed `std::array` args when available).
- **`RuntimeTensorShape` accessors dropped**: like CN/WH interleaved, the legacy
  `TensorAccessorArgs(ArgConfig::RuntimeTensorShape)` (shape passed as a runtime arg so one cached
  program serves varying shapes) is replaced by the default `TensorParameter` binding, which bakes the
  shape into the accessor's CTAs. See the CN report's `dynamic_tensor_shape` note.
- **wh_sharded no-op cores dropped**: the legacy full-grid launch with zeroed no-op rows is replaced by
  targeting only the active shard cores; behavior-preserving (the no-op cores did nothing).
- **wh_sharded_rm dead c_26 ("im2") DFB omitted** (no kernel touches it).
- `transpose_wh_sharded_m2.cpp` (compute) and `transpose_wh_rm_m2.cpp` (SHARDED branch now live) carry
  the standard fork drift-discipline note: keep in sync with the legacy copies until their last
  unmigrated consumer ports.
