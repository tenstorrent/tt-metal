# Port Plan — `data_movement/transpose` (HC sharded + WH sharded RM factories)

Port plan for `ttnn/cpp/ttnn/operations/data_movement/transpose`, ported from the `ProgramDescriptor` API
(`ProgramDescriptorFactoryConcept`) to Metal 2.0 (`CustomProgramSpecFactoryConcept`).
Written during the inventory and planning steps; committed alongside the port for review.

**Scope:** the two factories still on `ProgramDescriptor` — `TransposeHCShardedProgramFactory` and
`TransposeWHShardedRMProgramFactory`. The other six factories in `program_factory_t` are already on Metal 2.0
(PR #54333) and are not touched. Recipe: `akertesz/op-porting-recipe @ 4bd4bf42bfe` (read from a local export).
Code: `origin/main @ 0df0dde3316` + audit commit `2c413adb27e`; `file:line` below are against that tree.

Invoker decisions carried in (not re-litigated here): extend the existing compute fork
`device/kernels/compute/transpose_wh_rm_metal2.cpp` behind `#ifdef SHARDED`; drop dead `c_26`; make `c_27` a
conditional DFB via a define; explicit `O3` on the compute spec; drop reader CTAs 7-8.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` on both (`create_descriptor` + `void override_runtime_arguments`),
  living inside `program_factory_t` (`device/transpose_device_operation.hpp:29-37`) — a method swap inside each
  struct, no direct-descriptor exception.
- Variants: two factories; the HC factory has two kernel sets selected at cache-miss time by
  `hc_rm_sharded_is_special_case(shard_height, H, C)` (`transpose_hc_sharded_program_factory.cpp:317-320`, `:344`).
- Custom `compute_program_hash`: none — default reflection-based hash (no `compute_program_hash` /
  `attribute_values` / `to_hash` anywhere in the op).

### Variant: `TransposeHCShardedProgramFactory` (`device/transpose_hc_sharded_program_factory.cpp`)

#### Kernels
| unique_id | source | core_ranges | CTAs (positional, host emission order) | CTAs (named) | RTAs (per core) | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader (special case, `:383-384`, `:398-411`) | `device/kernels/dataflow/reader_unary_transpose_hc_sharded_rm.cpp` | `shard_spec.grid` (`:346`, `:408`) | `{c_0, c_16, stick_size_bytes}` | none | `[read_single_h_block_per_core, num_C_blocks_per_core_reader, num_sticks_per_shard_core_reader, num_non_repeat_cores, read_stick_stride, stick_offset[n], noc_x[n], noc_y[n]]` = 5 + 3n (`:271-287`) | none | `USE_SPECIAL_CASE=1` | O2 (unset, DM) | `ReaderConfigDescriptor{}` (RISCV_1 / NOC_0 / dedicated) |
| reader (generic, `:385-395`, `:403-411`) | same source | same | `{c_0, c_16, N, H, C, stick_size_bytes, row_major, num_cores_x, num_cores_y}` | none | `[num_sticks_per_core, curr_sticks_read, curr_c, curr_h, curr_n, shard_grid_x_map[num_cores_x], shard_grid_y_map[num_cores_y]]` = 5 + num_cores_x + num_cores_y (`:60-62`) | none | none | O2 | `ReaderConfigDescriptor{}` |
| writer (special case only, `:437-457`) | `device/kernels/dataflow/writer_unary_transpose_hc_sharded_rm.cpp` | same | `{c_0, c_16, stick_size_bytes}` | none | `[read_single_h_block_per_core, num_C_blocks_per_core_writer, num_sticks_per_shard_core_writer, num_non_repeat_cores, read_stick_stride, writer_read_stick_offset, writer_write_stick_offset, stick_offset[n], noc_x[n], noc_y[n]]` = 7 + 3n (`:289-308`) | none | none | O2 | `WriterConfigDescriptor{}` (RISCV_0 / NOC_1 / dedicated) |

Per-core RTAs are produced by `get_runtime_args_hc_rm_sharded` (`:23-87`, generic) and
`get_runtime_args_hc_rm_sharded_special_case` (`:89-314`, special case) and emplaced per core in shard-orientation
order (`i → {i % num_cores_x, i / num_cores_x}` for ROW_MAJOR, `{i / num_cores_y, i % num_cores_y}` otherwise;
`:425-433`, `:447-455`). `num_cores_x` / `num_cores_y` are the shard grid's bounding-box extents (`:352-355`).
`n = num_non_repeat_cores` is computed per core (`:202-233`).

#### CBs
| index | total_size | core_ranges | data_format | page_size | tile | buffer |
|---|---|---|---|---|---|---|
| `c_0` (`:357-368`) | `shard_height * stick_size_bytes` | `all_cores` | `src0` (input dtype) | `stick_size_bytes` | unset | `input_tensor.buffer()` (borrowed) |
| `c_16` (`:370-380`) | `shard_height * stick_size_bytes` | `all_cores` | `dst` (output dtype) | `stick_size_bytes` | unset | `output_tensor.buffer()` (borrowed) |

No GlobalCircularBuffer.

#### Semaphores
none

#### Tensor accessors
none — both tensors enter only through the borrowed CBs; no address RTA/CTA anywhere.

#### Work split
n/a — one shard per core on `shard_spec.grid`; `num_cores = shard_spec.num_cores()` (`:347`).

### Variant: `TransposeWHShardedRMProgramFactory` (`device/transpose_wh_sharded_rm_program_factory.cpp`)

#### Kernels
| unique_id | source | core_ranges | CTAs (positional, host emission order) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader (`:169-188`) | `device/kernels/dataflow/reader_unary_transpose_wh_sharded_rm.cpp` | `shard_spec.grid` (`:83`) | `{num_hw_blocks_per_core, ht, H_per_tile, H_per_tile_last, wt, stick_size_bytes, wt*elem*TILE_WIDTH, H_per_tile (dup), H_per_tile_last (dup)}` (9; the kernel reads 0-6) | none | none | none | none | O2 | `ReaderConfigDescriptor{}` |
| writer (`:190-207`) | `device/kernels/dataflow/writer_unary_transpose_wh_sharded_rm.cpp` | same | `{num_hw_blocks_per_core, ht, wt, W_per_tile, W_per_tile_last, H*out_elem, ht*out_elem*TILE_HEIGHT}` | none | none | none | none | O2 | `WriterConfigDescriptor{}` |
| compute (`:209-244`) | `device/kernels/compute/transpose_wh_rm.cpp` | same | `{ht, wt, ht*wt, num_hw_blocks_per_core, last_output_row_num_datums, pack_num_pages, pack_num_pages_last_col, pack_num_pages_last_row, pack_num_pages_last_row_col}` | none | none | none | `SHARDED=1` | **O3** (unset on a `ComputeConfigDescriptor`) | `ComputeConfigDescriptor{.fp32_dest_acc_en = (src0 == Float32), .unpack_to_dest_mode = [c_24]=[c_25]=UnpackToDestFp32 when src0 == Float32, else Default}`; math_fidelity HiFi4, math_approx_mode false, dst_full_sync_en false, bfp8_pack_precise false (all defaults) |

#### CBs
| index | total_size | core_ranges | data_format | page_size | tile | buffer |
|---|---|---|---|---|---|---|
| `c_0` (`:90-100`) | `shard_height * stick_size_bytes` | `all_cores` | `src0` | `stick_size_bytes` | unset | input (borrowed) |
| `c_16` (`:103-113`) | `stick_size_bytes * shard_height` | `all_cores` | `dst` | `output_page_size` (`:38-66`) | unset | output (borrowed) |
| `c_24` (`:116-126`) | `2 * wt * src0_tile` | `all_cores` | `src0` | `src0_tile` | unset | — |
| `c_25` (`:129-139`) | `ht * wt * src0_tile` | `all_cores` | `src0` | `src0_tile` | unset | — |
| `c_26` (`:143-153`, only `ht > 8`) | `ht * dst_tile` | `all_cores` | `dst` | `dst_tile` | unset | — (**dead**: no kernel references index 26) |
| `c_27` (`:156-166`, only `ht > 8`) | `2 * ht * dst_tile` | `all_cores` | `dst` | `dst_tile` | unset | — |

#### Semaphores
none

#### Tensor accessors
none

#### Work split
n/a — one shard per core on `shard_spec.grid`; `num_hw_blocks_per_core = shard_height / H` (`:76`).

### Shared kernels
| kernel | kind | binders on `main` (disambiguated census) | `_metal2` sibling? | rung |
|---|---|---|---|---|
| `device/kernels/compute/transpose_wh_rm.cpp` | intra-op | `TransposeWHShardedRMProgramFactory` (legacy copy, with `SHARDED=1`) · `TransposeWHProgramFactory` binds the **fork** `transpose_wh_rm_metal2.cpp` (`transpose_wh_program_factory.cpp:197`) · `experimental/quasar/transpose/` binds its own duplicate (not a binder of this file; not read) | yes — `transpose_wh_rm_metal2.cpp`, non-sharded path only (fork header `:11-16`) | **extend the fork under `#ifdef SHARDED`** (invoker decision; the recipe's rung ladder has no step for a fork that carries one variant of an `#ifdef`-split kernel — recorded as a deviation in the report) |
| `device/kernels/dataflow/reader_unary_transpose_hc_sharded_rm.cpp` | owned | HC sharded factory only | no | convert in place |
| `device/kernels/dataflow/writer_unary_transpose_hc_sharded_rm.cpp` | owned | HC sharded factory only | no | convert in place |
| `device/kernels/dataflow/reader_unary_transpose_wh_sharded_rm.cpp` | owned | WH sharded RM factory only | no | convert in place |
| `device/kernels/dataflow/writer_unary_transpose_wh_sharded_rm.cpp` | owned | WH sharded RM factory only | no | convert in place |

Fork binding vocabulary (the constraint the sharded path is built against): `dfb::in0` (tilize input),
`dfb::tilize` (tilize output / transpose input, self-loop), `dfb::out` (pack destination); named args `Ht`, `Wt`,
`HtWt` (CTA), `num_hw_blocks_per_core` (RTA on the non-sharded path). No `#ifdef` today. The sharded path reuses
those role names for `c_24` / `c_25` / `c_27`-or-`c_16`, adds the sharded CTAs as named args (with
`num_hw_blocks_per_core` a CTA there), and gates the `c_27` alias on a new `OUT_STAGING` define.

### Flags
- Unreferenced kernel files in the directory (not audited, not touched): `compute/transpose_wh.cpp`,
  `compute/transpose_wh_sharded.cpp`, `dataflow/reader_unary_transpose_hc_interleaved_tiled_padding_aware.cpp`,
  `dataflow/reader_unary_transpose_wh_interleaved.cpp` (legacy originals left behind by PR #54333).
- After this port `compute/transpose_wh_rm.cpp` has **no binder on `main`** — left in place (sunset is not the
  porter's job); its pointer comment is updated.
- `reader_unary_transpose_hc_sharded_rm.cpp:95` CTA `N` and `:108` / `:166` RTA `curr_n` are dead plumbing
  (read, incremented, never used) — named anyway because the kernel reads them.
- `transpose_wh_rm.cpp:110`, `:112` — `pack_num_pages` / `pack_num_pages_last_row` are read but unused — named anyway.
- `transpose_wh_sharded_rm_program_factory.cpp:178-179` — reader CTAs 7-8 duplicate CTAs 2-3 and are never read — dropped.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `CustomProgramSpecFactoryConcept` for both factories (each ported-from
  factory has an `override_runtime_arguments`: `hc_sharded:462-474`, `wh_sharded_rm:253-265`; both cb-addr-only).
- **Custom `compute_program_hash`**: none.
- **Implementation notes**: each override is translated to a `ProgramRunArgs`-returning method whose body is
  `tensor_args = {{INPUT, input}, {OUTPUT, output}}` and no `kernel_run_args` — exactly the two borrowed-CB
  re-points the legacy override performed, nothing more (the HC per-core args are pinned by the cache key). The
  signature follows the adapter's call (`mesh_device_operation_adapter.hpp:1034-1038`):
  `(const TransposeParams&, const TransposeInputs&, Tensor& output, const std::optional<ttnn::MeshCoordinate>& = std::nullopt)`.
  The two `TT_ASSERT`s at the top of each `create_descriptor` move into `create_program_artifacts`.
  Spec / DFB / kernel / tensor names are declared function-locally (this op's factories share one unity-build TU),
  as the six ported siblings do.

## Planned Spec Shape

### Variant: `TransposeHCShardedProgramFactory`
- **KernelSpecs**: `READER` (one `KernelSpec`, source `reader_unary_transpose_hc_sharded_rm.cpp`; CTAs / defines /
  bindings / schema differ per branch) and, on the special-case branch only, `WRITER`
  (`writer_unary_transpose_hc_sharded_rm.cpp`). Both `hw_config` via `create_reader_datamovement_config()` /
  `create_writer_datamovement_config()` (resolved legacy triples are the defaults). No `opt_level` (DM, O2 = legacy).
  - special-case reader: `compiler_options.defines = {{"USE_SPECIAL_CASE", "1"}}`;
    `compile_time_args = {{"stick_size_bytes", …}}`; `runtime_arg_schema = {read_single_h_block_per_core,
    num_C_blocks_per_core, num_sticks_per_shard_core, num_cores_read, read_stick_stride}`;
    `advanced_options.num_runtime_varargs = max over cores of 3 * num_non_repeat_cores` (zero-padded per core).
  - generic reader: `compile_time_args = {N, H, C, W_size_bytes, row_major, num_cores_x, num_cores_y}`;
    `runtime_arg_schema = {num_sticks_per_core, start_id, curr_c, curr_h, curr_n}`;
    `num_runtime_varargs = num_cores_x + num_cores_y`.
  - writer (special case): `compile_time_args = {{"stick_size_bytes", …}}`; `runtime_arg_schema =
    {read_single_h_block_per_core, num_C_blocks_per_core, num_sticks_per_shard_core, num_cores_read,
    read_stick_stride, src_read_stick_offset, dst_write_stick_offset}`; `num_runtime_varargs` as the reader's.
- **DataflowBufferSpecs**: `IN0` (`entry_size = stick_size_bytes`, `num_entries = shard_height`, `data_format =
  src0`, `borrowed_from = INPUT`); `OUT` (same with `dst` format, `borrowed_from = OUTPUT`).
- **SemaphoreSpecs**: none.
- **TensorParameters**: `INPUT` (`input_tensor.tensor_spec()`), `OUTPUT` (`output_tensor.tensor_spec()`); no
  relaxations (audit: `none`). No `TensorBinding` on any kernel — the parameters are used only through `borrowed_from`.
- **WorkUnitSpecs**: one, `"main"`, kernels `{READER}` (generic) or `{READER, WRITER}` (special case),
  `target_nodes = shard_spec.grid`.
- **DFB endpoints** (re-derived from the kernel-touch census; matches the brief):
  - special case: `IN0` → reader PRODUCER + writer CONSUMER; `OUT` → reader PRODUCER + writer CONSUMER (both
    kernels only peek `get_read_ptr` / `get_write_ptr` and move data through the NoC — role-free, 1P+1C cosmetic).
  - generic: `IN0` → reader PRODUCER + CONSUMER (self-loop, sole toucher, peek only); `OUT` → reader PRODUCER +
    CONSUMER (self-loop: `reserve_back` / `push_back`, nothing drains the resident output).
- **Run args**: per core, in the legacy shard-orientation enumeration, `AddRuntimeArgsForNode` for the named
  scalars and `advanced_options.runtime_varargs[core]` for the array tail; `tensor_args = {INPUT, OUTPUT}`.
- **Op-owned tensors**: none.

### Variant: `TransposeWHShardedRMProgramFactory`
- **KernelSpecs**: `READER` (`reader_unary_transpose_wh_sharded_rm.cpp`), `WRITER`
  (`writer_unary_transpose_wh_sharded_rm.cpp`), `COMPUTE` (`compute/transpose_wh_rm_metal2.cpp`, the extended fork).
  - reader: `compile_time_args = {num_hw_blocks_per_core, Ht, H_per_tile, H_per_tile_last, Wt, W_size_bytes,
    l1_write_offset_bytes}`; no RTAs; `create_reader_datamovement_config()`.
  - writer: `compile_time_args = {num_hw_blocks_per_core, Ht, Wt, W_per_tile, W_per_tile_last, H_size_bytes,
    l1_read_offset_bytes}`; no RTAs; `create_writer_datamovement_config()`; `defines = {OUT_STAGING=1}` iff `ht > 8`.
  - compute: `compile_time_args = {Ht, Wt, HtWt, num_hw_blocks_per_core, last_output_row_num_datums,
    pack_num_pages, pack_num_pages_last_col, pack_num_pages_last_row, pack_num_pages_last_row_col}`; no RTAs;
    `defines = {SHARDED=1}` + `{OUT_STAGING=1}` iff `ht > 8`; `compiler_options.opt_level = O3` (explicit);
    `hw_config = ComputeHardwareConfig{.enable_32_bit_dest = (src0 == Float32)}` with `unpack_modes = {{IN,
    UnpackToDest}, {TILIZE, UnpackToDest}}` under the same Float32 condition (Style B; every other field at its
    Metal default, which equals the legacy descriptor default).
- **DataflowBufferSpecs**: `IN0` (`c_0`: `entry_size = stick_size_bytes`, `num_entries = shard_height`, `src0`,
  `borrowed_from = INPUT`); `OUT` (`c_16`: `entry_size = output_page_size`, `num_entries = (stick_size_bytes *
  shard_height) / output_page_size`, `dst`, `borrowed_from = OUTPUT` — the legacy CB config required
  `total_size % page_size == 0` (`circular_buffer_config.cpp:164`), so the quotient is exact for every shape that
  ran before; the factory restates that check as a `TT_FATAL` because the DFB lowering no longer runs it);
  `IN` (`c_24`: `src0_tile`, `2 * wt`, `src0`); `TILIZE` (`c_25`: `src0_tile`, `ht * wt`, `src0`);
  `OUT_STAGING` (`c_27`, only `ht > 8`: `dst_tile`, `2 * ht`, `dst`). `c_26` (`im2`) → **no spec** (dead-CB drop).
- **SemaphoreSpecs**: none.
- **TensorParameters**: `INPUT`, `OUTPUT` (strict). No `TensorBinding`s (borrow-only).
- **WorkUnitSpecs**: one, `"main"`, `{READER, WRITER, COMPUTE}`, `target_nodes = shard_spec.grid`.
- **DFB endpoints** (re-derived): `IN0` → reader PRODUCER + CONSUMER (self-loop; peek only). `IN` → reader
  PRODUCER (accessor `in`), compute CONSUMER (accessor `in0`, the fork's role name for the tilize input).
  `TILIZE` → compute PRODUCER + CONSUMER (self-loop; accessor `tilize`). `OUT_STAGING` (`ht > 8`) → compute
  PRODUCER (accessor `out_staging`), writer CONSUMER (accessor `out_staging`). `OUT` → `ht > 8`: writer PRODUCER +
  CONSUMER (self-loop; accessor `out`, peek only); `ht <= 8`: compute PRODUCER (accessor `out`), writer CONSUMER
  (accessor `out`; it still constructs the handle and peeks `get_write_ptr`).
- **Run args**: no `kernel_run_args` (no kernel takes RTAs); `tensor_args = {INPUT, OUTPUT}`.
- **Op-owned tensors**: none.

## Preserved Multiplicity

none — no work-split multiplicity in legacy (one `KernelDescriptor` per role per configuration; the HC
special-case reader + writer are different sources over one grid, assigned 1P+1C on the two borrowed DFBs).

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| HC reader CTA slot 0 (`hc_sharded:384`, `:387`; kernel `:25`, `:93`) | `c_0` magic index | `DFBBinding{IN0, "in0", …}`; kernel `DataflowBuffer dfb_in(dfb::in0)` |
| HC reader CTA slot 1 (`:384`, `:388`; kernel `:26`, `:94`) | `c_16` magic index | `DFBBinding{OUT, "out", …}`; kernel `DataflowBuffer dfb_out(dfb::out)` |
| HC reader CTA slot 2 (special, `:384`; kernel `:27`) | positional `stick_size_bytes` | named CTA `stick_size_bytes` |
| HC reader CTA slots 2-8 (generic, `:389-395`; kernel `:95-101`) | positional `N, H, C, stick_size_bytes, row_major, num_cores_x, num_cores_y` | named CTAs `N`, `H`, `C`, `W_size_bytes`, `row_major`, `num_cores_x`, `num_cores_y` |
| HC reader RTAs 0-4 (special, `:273-281`; kernel `:16-20`) | positional scalars | named RTAs `read_single_h_block_per_core`, `num_C_blocks_per_core`, `num_sticks_per_shard_core`, `num_cores_read`, `read_stick_stride` |
| HC reader RTAs 5.. (special, `:282-287`; kernel `:21-23`, `get_arg_addr` pointer arrays) | three `n`-long arrays at offsets `5`, `5+n`, `5+2n` | runtime varargs `[offset(n), noc_x(n), noc_y(n)]`, read as `get_vararg(core)`, `get_vararg(num_cores_read + core)`, `get_vararg(2 * num_cores_read + core)` |
| HC reader RTAs 0-4 (generic, `:60`; kernel `:103-108`, `arg_idx++`) | positional scalars | named RTAs `num_sticks_per_core`, `start_id`, `curr_c`, `curr_h`, `curr_n` |
| HC reader RTAs 5.. (generic, `:61-62`; kernel `:110-113`) | `shard_grid_x_map[num_cores_x]`, `shard_grid_y_map[num_cores_y]` pointer arrays | runtime varargs `[x_map(num_cores_x), y_map(num_cores_y)]`, read as `get_vararg(i)` / `get_vararg(num_cores_x + i)` |
| HC writer CTA slots 0-2 (`:444`; kernel `:25-27`) | `c_0`, `c_16`, `stick_size_bytes` | `DFBBinding{IN0, "in0"}`, `DFBBinding{OUT, "out"}`, named CTA `stick_size_bytes` |
| HC writer RTAs 0-6 (`:292-302`; kernel `:14-20`) | positional scalars | named RTAs `read_single_h_block_per_core`, `num_C_blocks_per_core`, `num_sticks_per_shard_core`, `num_cores_read`, `read_stick_stride`, `src_read_stick_offset`, `dst_write_stick_offset` |
| HC writer RTAs 7.. (`:303-308`; kernel `:21-23`) | three `n`-long arrays at offsets `7`, `7+n`, `7+2n` | runtime varargs, same layout and reads as the reader's |
| WH RM reader CTAs 0-6 (`wh_sharded_rm:169-177`; kernel `:14-20`) | positional | named CTAs `num_hw_blocks_per_core`, `Ht`, `H_per_tile`, `H_per_tile_last`, `Wt`, `W_size_bytes`, `l1_write_offset_bytes` |
| WH RM reader CTAs 7-8 (`:178-179`) | duplicates of CTAs 2-3, never read | **dropped** |
| WH RM reader kernel `:22-23` | `tt::CBIndex::c_0`, `c_24` magic indices | `dfb::in0` (self-loop binding of `IN0`), `dfb::in` (PRODUCER binding of `IN`) |
| WH RM writer CTAs 0-6 (`:190-198`; kernel `:13-19`) | positional | named CTAs `num_hw_blocks_per_core`, `Ht`, `Wt`, `W_per_tile`, `W_per_tile_last`, `H_size_bytes`, `l1_read_offset_bytes` |
| WH RM writer kernel `:21-22` | `c_27`, `c_16` magic indices | `dfb::out_staging` (CONSUMER of `OUT_STAGING`, under `#ifdef OUT_STAGING`), `dfb::out` (binding of `OUT`) |
| WH RM compute CTAs 0-8 (`:209-219`; kernel `:104-113`) | positional | named CTAs `Ht`, `Wt`, `HtWt`, `num_hw_blocks_per_core`, `last_output_row_num_datums`, `pack_num_pages`, `pack_num_pages_last_col`, `pack_num_pages_last_row`, `pack_num_pages_last_row_col` |
| WH RM compute kernel `:131-134` (`SHARDED`) | `c_24`, `c_25`, `(Ht > 8) ? c_27 : c_16` | `dfb::in0`, `dfb::tilize`, `#ifdef OUT_STAGING ? dfb::out_staging : dfb::out` |
| WH RM `c_26` (`:143-153`) | dead CB allocation | **dropped**, no spec |
| `unpack_to_dest_mode` vector indexed by CB id (`:224-232`) | `[c_24]`, `[c_25]` = `UnpackToDestFp32` | `unpack_modes = {{IN, UnpackToDest}, {TILIZE, UnpackToDest}}` |

No buffer-address RTAs, no `TensorAccessorArgs`, no page-size 3rd-argument plumbing, no semaphore-ID RTAs.

## Applied Patterns

- *Self-loop DFB binding* / *Sync-free and single-ended CBs → self-loop DFB*: WH RM `IN0` on the reader (peek-only
  address source), `TILIZE` on the compute kernel (tilize fills, transpose drains), `OUT` on the writer under
  `ht > 8` (peek-only loopback target); HC generic `IN0` (peek) and `OUT` (single-ended packer-style fill, nothing
  drains) on the reader. DM self-loops are Gen1-legal; Quasar-uplift debt, recorded in the report.
- *Two-toucher DFB → assign 1P+1C*: HC special-case `IN0` / `OUT` (reader PRODUCER, writer CONSUMER — both
  role-free peeks); WH RM `OUT` under `ht <= 8` (compute locked PRODUCER, writer role-free CONSUMER).
- *Conditional / optional resource bindings*: `OUT_STAGING` (`c_27`) bound only when `ht > 8`; the host condition is
  promoted to the `OUT_STAGING` define on **both** the compute and writer specs; the compute kernel's file-scope
  ternary becomes an `#ifdef`-selected alias; the writer's `dfb_src` construction and its `if constexpr (Ht > 8)`
  transfer body go under `#ifdef OUT_STAGING`.
- *Multi-variant factories*: HC special-case vs generic kernel set, branched inside `create_program_artifacts`.
- *Pass DFB handles directly to LLKs and kernel-lib helpers*: `compute_kernel_lib::tilize<Wt, dfb_in, dfb_tilize,
  …>(Ht)`, `compute_kernel_hw_startup`, `copy_init`, `transpose_init`, `transpose_tile`, `pack_untilize_*` all take
  the `dfb::` tokens (already the fork's practice).
- *Caution: Avoid varargs*: retained only for the HC reader / writer's per-core NoC-coordinate and stick-offset
  arrays (indexed by `core` in loops; count is the runtime `num_cores_read`) and the generic reader's shard-grid
  maps (indexed by a data-derived shard id). All scalars are named.
- *Caution: Porting a shared kernel*: the compute kernel's fork is **extended** rather than reused as-is or
  re-forked (invoker decision; see Shared kernels). Its non-sharded vocabulary and args are untouched.
- *Unity-build hygiene*: all `*Name` constants declared function-locally, as the sibling factories do; the HC
  helper functions stay in the anonymous namespace with their existing names.

## Deferred / Flagged

- `OUT` (`c_16`) in WH RM is sized `num_entries = (stick_size_bytes * shard_height) / output_page_size`. Legacy
  validated divisibility in `CircularBufferConfig::set_page_size`; the DFB path does not, so the factory carries a
  `TT_FATAL` restating that invariant rather than silently truncating the buffer (the compute kernel's FIFO wrap
  depends on the total size under `ht <= 8`). This is the only guard added; recorded in the report's TT_FATAL census.
- Vararg count is one number per `KernelSpec`; the HC special-case count `3 * num_non_repeat_cores` is computed per
  core. The factory takes the maximum over cores and zero-pads shorter tails (the kernels index only up to their
  own named `num_cores_read`). A log line records when padding was needed.
- No structural issue found that the audit missed.
