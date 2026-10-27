# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/data_movement/transpose`

> Audit cleared all gates for the two factories still on the `ProgramDescriptor` API. This is your actionable
> input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `akertesz/op-porting-recipe` @ `4bd4bf42bfe` *(carry this line into the port report's Provenance
section; the audit ran on a verbatim local export of the recipe branch's `docs/…/metal_2.0/` tree, which this
branch does not carry)*

**Code audited:** `origin/main` @ `0df0dde3316` (2026-10-27); all `file:line` references below are against it.

**Scope of this port: two factories.** `TransposeHCShardedProgramFactory` (`device/transpose_hc_sharded_program_factory.{hpp,cpp}`)
and `TransposeWHShardedRMProgramFactory` (`device/transpose_wh_sharded_rm_program_factory.{hpp,cpp}`). The other six
factories in the `program_factory_t` variant (`device/transpose_device_operation.hpp:29-37`) are **already on
Metal 2.0** (PR #54333); leave them alone. Five kernels, all owned by this op; two borrowed-memory tensor bindings
per factory; no semaphores; no `TensorAccessor`. Abbreviations below: `hc_sharded:` =
`transpose_hc_sharded_program_factory.cpp`, `wh_sharded_rm:` = `transpose_wh_sharded_rm_program_factory.cpp`,
`reader:` / `writer:` = the factory's own DM kernels, `compute:` = `device/kernels/compute/transpose_wh_rm.cpp`.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); both factories port to
**`CustomProgramSpecFactoryConcept`**. Carry them forward:

- **Current concept:** `descriptor` (PR #45071) on both. Both factories already live in the `program_factory_t`
  variant — no direct-descriptor conversion; the port is a method swap inside each struct
  (`transpose_hc_sharded_program_factory.hpp:15-25`, `transpose_wh_sharded_rm_program_factory.hpp:15-25`).
- **Op-owned tensors:** none.
- **Target concept:** **`CustomProgramSpecFactoryConcept`** — each ported-from factory has an
  `override_runtime_arguments` (`hc_sharded:462-474`, `wh_sharded_rm:253-265`). Both are **cb-addr-only**: they
  re-point exactly the two borrowed CBs (`CBDescriptor{.buffer = input}`, `{.buffer = output}` through
  `apply_descriptor_runtime_args`) and refresh nothing else. Translate each into a `ProgramRunArgs`-returning
  override whose body is `tensor_args = {{INPUT, TensorArgument{input}}, {OUTPUT, TensorArgument{output}}}` and
  **no** `kernel_run_args` — mirror that set exactly; the HC per-core args are pinned by the cache key and the
  legacy override does not touch them. Keep the return type `ProgramRunArgs` (a `void` override silently drops the
  factory to the base concept).
- **Custom `compute_program_hash`:** none (default reflection hash).
- **Gate-cleared, confirmed absent** (each would have blocked the brief): a `TensorParameter relaxation` that is
  neither `none` nor an analysis pointer (both cells read `none`) · `get_dynamic_runtime_args` (removed by
  PR #52566). A pybound `create_descriptor` is also absent (`transpose_nanobind.cpp:40-61` binds only the public
  op; nothing to delete).
- **Guards to carry:** the two `TT_ASSERT`s at the top of each `create_descriptor` (`hc_sharded:328-329`,
  `wh_sharded_rm:23-24`) move into `create_program_artifacts`; the overrides carry no guards.
- **Factory selection is untouched:** `select_program_factory` (`transpose_device_operation.cpp:300-364`) routes to
  HC sharded on `use_sharded_hc` (`:336`, `:352`) and to WH sharded RM on `is_row_major && use_sharded_wh_rm`
  (`:335`, `:343`). Device-op code is off-limits; nothing there needs to change.

## Construct — to do

**Tensor bindings** (per binding; identical in both factories):

- `input` — **clean** (borrowed-memory DFB): `c_0` is backed by the input shard (`hc_sharded:357-368`,
  `wh_sharded_rm:90-100`) → declare a `TensorParameter` from `input.tensor_spec()` and a `DataflowBufferSpec` with
  `borrowed_from` it (`num_entries = shard_height`, `entry_size = stick_size_bytes`). Kernels keep reading through
  `DataflowBuffer::get_read_ptr()`.
- `output` — **clean** (borrowed-memory DFB): `c_16` is backed by the output shard (`hc_sharded:370-380`,
  `wh_sharded_rm:103-113`) → same. **Mind the page size:** HC uses `stick_size_bytes` (`:377`); WH RM uses the
  computed `output_page_size` (`:38-66`, `:110`) with `shard_height * stick_size_bytes` total (`:105`) — carry the
  values verbatim.
- No Case 1, no Case 2. No `TensorAccessor` anywhere, so no `TensorAccessorArgs` plumbing to drop.

**TensorParameter relaxation:** `none`. Keep strict.

**TensorAccessor 3rd arg:** none.

**CB endpoints** — `TransposeHCShardedProgramFactory` (two kernel sets, selected by
`hc_rm_sharded_is_special_case(shard_height, H, C)`, `:317-320`, `:344`):

- `in0` (`c_0`): **special case** → **1P+1C**: reader PRODUCER, writer CONSUMER (both only peek
  `dfb_in.get_read_ptr()` and read *remote* cores' shards through the NoC — `reader:38`, `:66`; `writer:38`, `:60`;
  roles are cosmetic). **generic** → **self-loop** on the reader (sole toucher, `reader:149`; DM self-loop, legal on
  Gen1).
- `out` (`c_16`): **special case** → **1P+1C**: reader PRODUCER, writer CONSUMER (both raw-write via
  `dfb_out.get_write_ptr()` + NoC reads into it — `reader:39`, `:61`; `writer:39`, `:55`). **generic** →
  **self-loop** on the reader (`reserve_back` `:123` / `push_back` `:177`, nothing drains the resident output).
- No other CBs. No semaphores.

**CB endpoints** — `TransposeWHShardedRMProgramFactory` (one kernel set; CB set depends on `ht > 8`, `:142`):

- `in0` (`c_0`, `:90-100`) — **self-loop** on the reader (peek `reader:31`, loopback reads `:33-36`, `:45-52`).
- `in` (`c_24`, `2*wt` tiles, `:116-126`) — legal 1:1: reader PRODUCER (`reserve_back(Wt)` `:40`, `push_back(Wt)`
  `:57`), compute CONSUMER (the `tilize` helper's input side, `compute:151-158`).
- `tilize` (`c_25`, `ht*wt` tiles, `:129-139`) — **self-loop** on the compute kernel: the `tilize` helper packs into
  it and the kernel drains it (`wait_front(HtWt)` `compute:161`, `pop_front(HtWt)` `:181`).
- `im2` (`c_26`, `:143-153`, only when `ht > 8`) — **dead-CB drop**: no kernel references index 26 in any config
  (compute `:131-134`, reader `:22-23`, writer `:21-22`; no CTA carries it). A dead CB has no behavior, so removing
  its allocation changes none; record the drop with `file:line` in the port report.
- `out_staging` (`c_27`, `2*ht` tiles, `:156-166`, only when `ht > 8`) — legal 1:1: compute PRODUCER
  (`reserve_back(Ht)` / `push_back(Ht)`, `compute:83-96` or `:43-53`), writer CONSUMER (`wait_front(Ht)` `:41`,
  `pop_front(Ht)` `:58`). **Conditional DFB** — see *Watch for*.
- `out` (`c_16`, `:103-113`) — `ht > 8`: **self-loop** on the writer (peek `writer:30`, loopback writes `:34-37`,
  `:46-53`; the compute kernel packs into `c_27` instead). `ht <= 8`: **1P+1C** — compute PRODUCER
  (`cb_out_idx = c_16`, `compute:134`), writer CONSUMER (it still constructs `dfb_dst(c_16)` and peeks
  `get_write_ptr()` at `writer:22`, `:28-30` even though its transfer loop is compiled out).

No multi-binding flag anywhere.

## Watch for

- **CB endpoints (multi-binding):** none. The two-toucher CBs (HC special-case `in0` / `out`, WH RM `out` under
  `ht <= 8`) all fit 1P+1C; do not reach for the flag.
- **Cross-op / shared kernels — resolve Question 1 of the audit with the invoker before touching
  `transpose_wh_rm.cpp`.** The legacy compute kernel is bound by this factory with `SHARDED=1`
  (`wh_sharded_rm:221-222`, `:235-236`). Its `_metal2` fork `device/kernels/compute/transpose_wh_rm_metal2.cpp` is
  bound by the already-ported `TransposeWHProgramFactory` (`transpose_wh_program_factory.cpp:197`) and **carries only
  the non-sharded path** (fork `:11-16`): it lacks CTAs 3-8 (`compute:107-113`), the `use_narrow_row` / `row_size`
  selection with its yolov4 PCC note (`:114-124`), the `transpose_with_pack_untilize_narrow_row` helper (`:18-58`)
  and the `c_24` / `c_25` / `c_27`-or-`c_16` CB set (`:131-134`). Rung 1 fails the fit check; the fork has a
  consumer, so the caution says it is read-only and forbids a fork-of-fork. The audit recommends **extending the
  fork behind the same `#ifdef SHARDED` the legacy file already uses** (it cannot affect `TransposeWHProgramFactory`,
  which never defines `SHARDED`; the non-sharded names `dfb::in0` / `dfb::tilize` / `dfb::out` and args `Ht` /
  `Wt` / `HtWt` / `num_hw_blocks_per_core` stay untouched) — but only with the invoker's explicit go-ahead; the
  alternatives are a second fork (`transpose_wh_rm_sharded_metal2.cpp`) or in-place conversion of the legacy copy
  (its binder set on `main` is now exactly this factory; the quasar copy binds its own duplicate). Whatever is
  chosen, keep the sharded path's `constexpr` CTA form of `num_hw_blocks_per_core` (`compute:108`; it is an RTA on
  the non-sharded path, `:127` / fork `:76`) so the loop bound at `:147` stays compile-time, keep the legacy init
  pair `compute_kernel_hw_startup` + `copy_init` (`:144-145`; the fork drifted to `unary_op_init_common`, `:85`),
  and keep every comment (`:114-122` especially). The fork's role vocabulary fits the sharded roles exactly:
  `dfb::in0` ↔ `c_24`, `dfb::tilize` ↔ `c_25`, `dfb::out` ↔ `c_27` or `c_16` — **sunset list, not authorization to
  convert anything in place**: after this port the legacy `transpose_wh_rm.cpp` has no binder on `main`.
  No borrowed kernel file; the four DM kernels are bound only by this op (plus the quasar copy's own duplicates,
  which are not binders of these files).
- **RTA varargs (HC sharded only — the WH RM kernels take no runtime args):**
  - reader, special case (`reader:21-23`): three `num_cores_read`-long arrays (`read_stick_offset`, `noc_coord_x`,
    `noc_coord_y`) at offsets `5`, `5 + n`, `5 + 2n`, indexed `[core]` in loops (`:38`, `:44`, `:52`, `:66`, `:71`,
    `:79`) → **varargs**, `num_runtime_varargs = 3 * num_cores_read`, read as `get_vararg(core)`,
    `get_vararg(num_cores_read + core)`, `get_vararg(2 * num_cores_read + core)`. Name the five scalars
    (`read_single_h_block_per_core`, `num_C_blocks_per_core`, `num_sticks_per_shard_core`, `num_cores_read`,
    `read_stick_stride`; `:16-20`, host `:273-281`).
  - reader, generic (`:110-113`): `shard_grid_x_map` (`num_cores_x` entries) + `shard_grid_y_map` (`num_cores_y`
    entries), CTA-bounded, indexed by a data-derived shard id (`:142-147`) → **varargs**
    (`num_cores_x + num_cores_y`; host `:61-62`). Name the five scalars (`num_sticks_per_core`, `start_id`, `curr_c`,
    `curr_h`, `curr_n`; `:104-108`, host `:60`). `curr_n` and CTA `N` are dead plumbing — name them anyway (the
    kernel reads them) and report.
  - writer, special case (`writer:21-23`): the same three arrays at offsets `7`, `7 + n`, `7 + 2n` → **varargs**;
    name the seven scalars (`:14-20`, host `:292-302`).
  - `num_runtime_varargs` is one count per `KernelSpec` (`advanced_options.hpp:70-79`; the per-node override is
    deprecated, `:93-101`). The generic count is uniform; the special-case count `3 * num_non_repeat_cores`
    (`hc_sharded:202-233`) is uniform on a rectangular shard grid (every shard id passes the bbox filter `:183`).
    If a config ever varies it, zero-pad to the maximum — the kernels never index past `3 * num_cores_read`.
    Report the retained varargs in the port report.
- **Conditional DFB `out_staging` (`c_27`) and the dead `im2` — WH sharded RM.** Promote the host condition `ht > 8`
  (`wh_sharded_rm:142`) to a define (say `OUT_STAGING`) emitted through `compiler_options.defines` on **both** the
  compute and the writer specs, bind `out_staging` only when it is set, and `#ifdef`-gate every kernel-side
  reference (*Pattern: Conditional / optional resource bindings*): the compute kernel's `constexpr auto cb_out_idx
  = (Ht > 8) ? c_27 : c_16` (`compute:133-134`) becomes an `#ifdef`-selected alias (`dfb::out_staging` vs
  `dfb::out`), and the writer's `DataflowBuffer dfb_src(c_27)` (`writer:21`, `:27`) plus its `if constexpr (Ht >
  8)` body (`:33-62`) go under the same guard. The writer's `dfb_dst(c_16)` construction and peek (`:22`, `:28-30`)
  are unconditional today and stay so: bind `out` on the writer in every config. Drop `im2` (`c_26`, `:143-153`)
  outright — do not make it conditional.
- **Two kernel sets in HC sharded (*Pattern: Multi-variant factories*).** Special case: reader with
  `USE_SPECIAL_CASE` (`:398-401`) and CTAs `{in0, out, stick_size_bytes}` (`:383-384`) + writer with the same CTAs
  (`:444`). Generic: reader only (`:413-415`) with CTAs `{in0, out, N, H, C, stick_size_bytes, row_major,
  num_cores_x, num_cores_y}` (`:385-395`). Branch inside `create_program_artifacts`; the writer `KernelSpec` exists
  only on the special-case branch; the define moves to `compiler_options.defines`; the two per-core helpers
  (`:23-87`, `:89-314`) stay and feed named args + varargs per core. Keep the shard-orientation core enumeration
  (`:425-433`, `:447-455`) and note that `num_cores_x` / `num_cores_y` are bounding-box extents (`:352-355`).
  CB-index CTAs become `dfb::` tokens (`reader:25-26`, `:93-94`; `writer:25-26`) and the `DataflowBuffer`s are built
  from them (`reader:30-31`, `:120-121`; `writer:30-31`).
- **Remote-shard reads assume one L1 address for the input DFB on every core** (`reader:38`, `:66`, `:149`;
  `writer:38`, `:60`). A `borrowed_from` DFB inherits the sharded buffer's single address, so this holds; say so in
  the port report rather than re-deriving it.
- **Loopback endpoints in the WH RM reader / writer — keep as is** (`reader:33-36`, `:45-52`; `writer:34-37`,
  `:46-53`): `UnicastEndpoint` built from `my_x[noc.get_noc_id()]` / `my_y[…]`, the migration guide's *"Avoid"*
  form. The preferred local-handle form is a transfer rewrite, out of port scope. Precedent:
  `data_movement/sharded/device/kernels/dataflow/reader_unary_stick_layout_sharded_blocks_interleaved_start_id_metal2.cpp:133-137`.
- **Compute `hw_config` (WH RM) — Style B, direct `ComputeConfigDescriptor` (`:241-244`).** Carry
  `enable_32_bit_dest = (src0 == Float32)` (`:81`) and, under the same Float32 condition, `unpack_modes = {{IN,
  UnpackMode::UnpackToDest}, {TILIZE, UnpackMode::UnpackToDest}}` (legacy `unpack_to_dest_mode[c_24]` / `[c_25]` =
  `UnpackToDestFp32`, `:224-232`). Those two are exactly the Float32 DFBs the compute kernel consumes, so the
  validator's required-entry rule is met by them; `out_staging` / `out` are produced, not consumed — no entry. All
  other fields stay at their Metal defaults (HiFi4, `Precise`, `double_buffer_dest = true`, default
  `bfp_pack_precision_mode`), matching the legacy descriptor defaults. The live header is a flat
  `ComputeHardwareConfig` with nested `Compute1XXConfig` (`compute_hardware_config.hpp:43-127`), not the recipe's
  `ComputeGen1Config`. Defines: `SHARDED=1` (`:221-222`) → `compiler_options.defines`.
- **`opt_level`:** nothing set in either factory (`grep -n opt_level` → no hits) → the WH RM compute spec must say
  `KernelBuildOptLevel::O3` explicitly (`kernel_spec.hpp:122` defaults to `O2`); the four DM specs stay at O2. DM
  configs are `ReaderConfigDescriptor{}` / `WriterConfigDescriptor{}` (`hc_sharded:411`, `:445`; `wh_sharded_rm:188`,
  `:207`) → `create_reader_datamovement_config` / `create_writer_datamovement_config`.
- **Magic CB indices in the WH RM kernels** (`reader:22-23`, `writer:21-22`, `compute:131-134`) → `dfb::` tokens
  (whitelist rule 2); construct the `DataflowBuffer`s from them (rule 3).
- **Kernel-lib call keeps its id-only form:** `compute_kernel_lib::tilize<Wt, cb_in, cb_tilize, InitAndUninit,
  WaitBlock, NoReconfigure, Lossless>(Ht)` (`compute:151-158`) takes CB ids as `uint32_t` NTTPs
  (`tilize_helpers.hpp:176-190`); pass `dfb::` tokens there (*Pattern: Pass DFB handles directly to LLKs and
  kernel-lib helpers*). The by-reference form is deprecated only under `ARCH_QUASAR` (`id_only_helpers.hpp:14-21`)
  — not port work. The LLK free functions (`compute_kernel_hw_startup`, `copy_init`, `transpose_init`,
  `transpose_tile`, `pack_untilize_*`) take `uint32_t` cb ids likewise.
- **Dropped plumbing (WH RM):** reader CTAs 7-8 (`wh_sharded_rm:178-179`) duplicate CTAs 2-3 and are never read
  (`reader:14-20` reads 0-6) — drop them and record under *Dropped Plumbing*. Compute CTAs 5 (`pack_num_pages`)
  and 7 (`pack_num_pages_last_row`) are read (`compute:110`, `:112`) but unused — name them, note them. `num_cores`
  at `wh_sharded_rm:84` is only logged.
- **`constexpr` metadata:** none from CB metadata in any of the five kernels (no `get_tile_size(cb)` /
  `get_dataformat(cb)` reads); every size the kernels use arrives as a CTA.
- **Tests (no-regression baseline, confirm with the invoker):**
  `tests/tt_eager/python_api_testing/unit_testing/misc/test_transpose.py` — WH sharded RM:
  `test_transpose_hw_sharded_rm` (`:473`; `skip_for_blackhole`, #12349), `test_transpose_hw_sharded_rm_with_program_cache`
  (`:544`), `test_transpose_wh_sharded_rm_override_addr_change` (`:816`); HC sharded:
  `test_transpose_hc_sharded_with_program_cache` (`:665`), `test_transpose_hc_sharded_cache_hit_new_input` (`:721`;
  ids `generic` / `special_case_single_h_block` / `special_case_multi_h_block` cover both kernel sets and both
  special-case branches), `test_transpose_hc_sharded_override_addr_change` (`:807`); both: the row-major
  height/height cases of `test_transpose_sharded` (`:1561`) and `test_transpose_sharded_rm_width` (`:1629`).
  The listed WH RM cases run `ht ∈ {1, 4, 8}`; **no listed test obviously lands on `ht > 8`** (the `out_staging`
  / writer-transfer path) — add a local case with `H >= 288` (e.g. `(2, 8, 512, 16)` height-sharded, COL_MAJOR, as
  in `test_transpose_wh_sharded_rm_override_addr_change`) and run it against the legacy build first. Peer ops that
  bind the shared compute kernel: `TransposeWHProgramFactory` via the fork — rerun the WH row-major interleaved
  cases (`test_transpose_hw_rm_*`, `:382-452`) after any edit to `transpose_wh_rm_metal2.cpp`.
