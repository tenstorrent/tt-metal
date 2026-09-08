# Port Plan — transpose (data_movement)

Port plan for `ttnn/cpp/ttnn/operations/data_movement/transpose`, ported from the
`ProgramDescriptorFactoryConcept` (`create_descriptor`) to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

**Scope: the clean six-factory subset** the brief issues. The audit is RED at op level; the two
sharded-RM factories are gated and stay on the legacy descriptor path.

## Legacy Inventory

### Legacy factory shape

- Concept: `ProgramDescriptorFactoryConcept` — every one of the eight factories defines
  `create_descriptor(...) -> ProgramDescriptor`. Methods live in per-factory structs already
  (`program_factory_t` variant on `TransposeDeviceOperation`), so **not** the direct-descriptor
  shape — no exception-3 restructuring needed.
- Variants: 8 factories on one `TransposeDeviceOperation`. **In scope (6):** `TransposeCNProgramFactory`,
  `TransposeHCRMProgramFactory`, `TransposeHCTiledInterleavedProgramFactory`,
  `TransposeHCTiledProgramFactory`, `TransposeWHProgramFactory` (tiled **and** row-major, runtime-selected),
  `TransposeWHShardedProgramFactory`.
  **Out of scope (2, gated):** `TransposeHCShardedProgramFactory`, `TransposeWHShardedRMProgramFactory`.
- Custom `compute_program_hash`: **none** — default reflection-based hash. Verified: no
  `compute_program_hash`, no `attribute_values` / `to_hash` backdoor anywhere in the op.

### Audit-vs-main divergence (recorded during inventory)

The audit was written 2026-08-04; this port runs against `origin/main` @ `f6b36f3b1be` (2026-08-25).
One material change landed in between:

- **`get_dynamic_runtime_args` is gone from transpose.** Commit `383674438e5` ("Port transpose off
  `get_dynamic_runtime_args` onto `override_runtime_arguments`", #52566) moved the two gated
  factories onto `override_runtime_arguments`. The audit's gate rationale ("`Runtime-args update = yes`
  via the device-op `get_dynamic_runtime_args` hook at `transpose_hc_sharded_program_factory.cpp:432`")
  no longer describes the code.
- **Consequence for scope: none.** `override_runtime_arguments` is declared *only* on the two gated
  factories (`transpose_hc_sharded_program_factory.hpp:19`, `transpose_wh_sharded_rm_program_factory.hpp:19`).
  The six in-scope factories have none, so they stay on the base `ProgramSpecFactoryConcept`; the
  audit's target-concept decision still holds for the subset.
- **Consequence for the gated two:** their second gate conjunct (`Is safe to port? = no`, the
  readiness-sheet owner's correctness call) is untouched and is not mine to clear. If it ever clears,
  they now route to `CustomProgramSpecFactoryConcept` (they carry an `override_runtime_arguments`),
  **not** the base concept the audit recorded. Routed to the report.

### Kernels

| factory | unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| CN | reader | `…/transpose/…/dataflow/reader_unary_transpose_cn_interleaved_start_id.cpp` | `all_cores` | 0=`src0_cb_index`(0), 1=`src0->aligned_page_size()`, 2=`stick_size`, 3+=`TensorAccessorArgs(src0)` | — | `{src0_buffer, N, C, HtWt, batch_step, channel_step, num_pages_per_core, start_tile, hw, n}` | `CN_RM=1` iff row-major | absent → **O2** | `ReaderConfigDescriptor{}` |
| CN | writer | `…/transpose/…/dataflow/writer_unary_transpose_cn_interleaved_start_id.cpp` | `all_cores` | 0=`src0_cb_index`(0), 1=`dst->aligned_page_size()`, 2=`stick_size`, 3+=`TensorAccessorArgs(dst)` | — | `{dst_buffer, num_pages_per_core, num_pages_read}` | `CN_RM=1` iff row-major | absent → **O2** | `WriterConfigDescriptor{}` |
| HC-RM | reader | `…/dataflow/reader_unary_transpose_hc_interleaved_partitioned_rm.cpp` | `all_cores` | 0=`N`, 1=`H`, 2=`C`, 3=`stick_size`, 4=`src0->aligned_page_size()`, 5+=`TensorAccessorArgs(src0)` | — | `{input_buffer, num_sticks_per_core_read, num_read_per_barrier, curr_sticks_read, curr_c, curr_h, curr_n}` | — | absent → **O2** | `ReaderConfigDescriptor{}` |
| HC-RM | writer | `…/dataflow/writer_unary_transpose_hc_interleaved_start_id_rm.cpp` | `all_cores` | 0=`src0_cb_index`(0), 1=`stick_size`, 2=`dst->aligned_page_size()`, 3+=`TensorAccessorArgs(dst)` | — | `{output_buffer, num_sticks_per_core_read, num_read_per_barrier, curr_sticks_write}` | — | absent → **O2** | `WriterConfigDescriptor{}` |
| HC-Tiled-Intlv | reader | `…/dataflow/reader_unary_transpose_hc_interleaved_tiled_padding_aware.cpp` | `active_cores` | 0+=`TensorAccessorArgs(src)` only | `num_writes`, `padding_val_packed`, `needs_padding`, `swap_hw`=0, `H`=1, `W`=1, `accumulated_outer_dims`=1, `tile_height`=1, `tile_width`=1 | `{input_buffer, num_tiles_per_core, start_idx}` | — | absent → **O2** | `ReaderConfigDescriptor{}` |
| HC-Tiled-Intlv | writer | `…/dataflow/writer_unary_transpose_hc_interleaved_tiled_padding_aware.cpp` | `active_cores` | 0=`element_size`, 1=`CBIndex::c_0`, 2=`C`, 3=`H`, 4=`W`, 5=`tile_shape[0]`, 6=`tile_shape[1]`, 7=`face_shape[0]`, 8=`face_shape[1]`, 9=`needs_padding`, 10+=`TensorAccessorArgs(dst)` | — | `{output_buffer, start_idx, end_idx, padded_start_idx, padded_end_idx}` | — | absent → **O2** | `WriterConfigDescriptor{}` |
| HC-Tiled | reader | `…/dataflow/reader_unary_transpose_hc_interleaved_partitioned.cpp` | `all_cores` | 0=`sub_tile_line_bytes`, 1=`is_float32`, 2=`alignment`, 3+=`TensorAccessorArgs(src0)` | — | `{input_buffer, Wt, H, Ct, HW_bytes, CHW_bytes, num_tiles_read, num_tiles_per_core, …, h, …, ct, …, …}` (14 values) | — | absent → **O2** | `ReaderConfigDescriptor{}` |
| HC-Tiled | writer | **borrowed** `eltwise/unary/…/dataflow/writer_unary_interleaved_start_id.cpp` | `all_cores` | 0=`src0_cb_index`(0), 1+=`TensorAccessorArgs(dst)` | — | `{output_buffer, num_tiles_per_core, num_tiles_read}` | — | absent → **O2** | `WriterConfigDescriptor{}` |
| WH (tiled) | reader | `…/dataflow/reader_unary_transpose_wh_interleaved_start_id.cpp` | `all_cores` | 0+=`TensorAccessorArgs(src0)` only | — | `{input_buffer, num_tiles_per_core, start_tile, h, w, Ht, Wt, HtWt}` | — | absent → **O2** | `ReaderConfigDescriptor{}` |
| WH (tiled) | writer | **borrowed** `eltwise/unary/…/dataflow/writer_unary_interleaved_start_id.cpp` | `all_cores` | 0=`output_cb_index`(c_16), 1+=`TensorAccessorArgs(dst)` | — | `{output_buffer, num_tiles_per_core, num_tiles_read}` | — | absent → **O2** | `WriterConfigDescriptor{}` |
| WH (tiled) | compute | `…/compute/transpose_wh.cpp` | `all_cores` | *(empty)* | — | `{num_tiles_per_core}` | — | absent → **O3** | `ComputeConfigDescriptor{fp32_dest_acc_en, unpack_to_dest_mode}` |
| WH (RM) | reader | `…/dataflow/reader_unary_transpose_wh_interleaved_start_id_rm.cpp` | `all_cores` | 0=`ht`,1..8 (see factory), 9+=`TensorAccessorArgs(src0)` | — | `{input_buffer, num_sticks_read, num_hw_blocks_per_core}` | — | absent → **O2** | `ReaderConfigDescriptor{}` |
| WH (RM) | writer | `…/dataflow/writer_unary_transpose_wh_interleaved_start_id_rm.cpp` | `all_cores` | 0=`output_cb_index`, 1..9 (see factory), 10+=`TensorAccessorArgs(dst)` | — | `{output_buffer, num_sticks_write, num_hw_blocks_per_core}` | — | absent → **O2** | `WriterConfigDescriptor{}` |
| WH (RM) | compute | `…/compute/transpose_wh_rm.cpp` | `all_cores` | 0=`ht`, 1=`wt`, 2=`ht*wt` | — | `{num_hw_blocks_per_core}` | `DST_ACCUM_MODE=1` iff RM ∧ dtype∈{UINT32,INT32} | absent → **O3** | `ComputeConfigDescriptor{fp32_dest_acc_en, unpack_to_dest_mode}` |
| WH-Sharded | reader | **borrowed** `eltwise/unary/…/dataflow/reader_unary_sharded.cpp` | `total_cores` | 0=`src0_cb_index`(c_0) | — | `{num_blocks}` (noop cores: 1 zero) | — | absent → **O2** | `ReaderConfigDescriptor{}` |
| WH-Sharded | writer | **borrowed** `data_movement/sharded/…/dataflow/writer_unary_sharded.cpp` | `total_cores` | 0=`output_cb_index`(c_16) | — | `{num_blocks}` (noop cores: 1 zero) | — | absent → **O2** | `WriterConfigDescriptor{}` |
| WH-Sharded | compute | `…/compute/transpose_wh_sharded.cpp` | `total_cores` | 0=`src0_cb_index`, 1=`output_cb_index` | — | `{num_blocks, HtWt_tile_size, num_hw_blocks_per_shard, Ht_per_shard, Wts}` (noop cores: 5 zeros) | — | absent → **O3** | `ComputeConfigDescriptor{fp32_dest_acc_en, unpack_to_dest_mode}` |

**`opt_level` — resolved, not literal.** No factory in the op sets `opt_level` at all (`grep -c opt_level`
over all six factory `.cpp` files = 0). Under the descriptor API an absent field resolves as legacy does:
**O2** for reader/writer descriptors, **O3** for every `ComputeConfigDescriptor`. The three compute
kernels above therefore resolve to **O3** and each needs an explicit
`compiler_options.opt_level = KernelBuildOptLevel::O3` on its `KernelSpec`, since Metal 2.0's
`CompilerOptions` defaults to `O2` (`kernel_spec.hpp:116`).

### CBs

| factory | index | total_size | core_ranges | data_format | page_size | notes |
|---|---|---|---|---|---|---|
| CN | 0 | `2 * stick_size` | `all_cores` | `cb_data_format` | `stick_size` | |
| HC-RM | 0 | `num_sticks * stick_size` | `all_cores` | `cb_data_format` | `stick_size` | |
| HC-Tiled-Intlv | 0 | `2 * single_tile_size` | `active_cores` | `cb_data_format` | `single_tile_size` | |
| HC-Tiled-Intlv | 1 | `max_padding_write * element_size` | `active_cores` | `cb_data_format` | same | **conditional** on `needs_padding` (`C % tile_h != 0`) |
| HC-Tiled | 0 | `2 * single_tile_size` | `all_cores` | `cb_data_format` | `single_tile_size` | |
| HC-Tiled | 1 | `alignment` | `all_cores` | `cb_data_format` | `alignment` | **conditional** on `misaligned` (`dst alignment > sub_tile_line_bytes`) |
| WH | 0 | `num_input_tiles * src0_single_tile_size` | `all_cores` | `src0_cb_data_format` | `src0_single_tile_size` | `num_input_tiles = row_major ? wt*2 : 2` |
| WH | 16 | `num_output_tiles * dst_single_tile_size` | `all_cores` | `dst_cb_data_format` | `dst_single_tile_size` | `num_output_tiles = row_major ? ht*2 : 2` |
| WH | 24 | `ht*wt * src0_single_tile_size` | `all_cores` | `src0_cb_data_format` | `src0_single_tile_size` | **RM branch only** (tilize intermediate) |
| WH | 25 | `ht * dst_single_tile_size` | `all_cores` | `dst_cb_data_format` | `dst_single_tile_size` | **RM branch only — DEAD, dropped** (see below) |
| WH-Sharded | 0 | `num_tiles_per_shard * src0_single_tile_size` | `all_cores` (shard grid) | `src0_cb_data_format` | `src0_single_tile_size` | **borrowed** `.buffer = input_tensor.buffer()` |
| WH-Sharded | 16 | `num_tiles_per_shard * dst_single_tile_size` | `total_cores` | `dst_cb_data_format` | `dst_single_tile_size` | **borrowed** `.buffer = output_tensor.buffer()` |

No `GlobalCircularBuffer` anywhere. No aliased CBs (every `format_descriptors` list is single-element).

### Semaphores

none — the op declares no semaphores in any in-scope factory.

### Tensor accessors

| factory | host site | originating Tensor | RTA slot (host) |
|---|---|---|---|
| CN | reader / writer CTA tail | input / output | reader RTA 0, writer RTA 0 (`Buffer*` form) |
| HC-RM | reader / writer CTA tail | input / output | reader RTA 0, writer RTA 0 |
| HC-Tiled-Intlv | reader / writer CTA tail | input / output | reader RTA 0, writer RTA 0 |
| HC-Tiled | reader / writer CTA tail | input / output | reader RTA 0, writer RTA 0 |
| WH (both paths) | reader / writer CTA tail | input / output | reader RTA 0, writer RTA 0 |
| WH-Sharded | — | input / output | **none** — addresses reach the kernels through the two borrowed CBs, not accessors |

Every accessor is **2-arg** (`TensorAccessor(args, addr)`); no page-size third argument anywhere — matches
the audit's TensorAccessor-3rd-arg GREEN.

### Work split

- CN: `split_work_to_cores(compute_with_storage_grid_size, num_tensor_pages)`
- HC-RM: `split_work_to_cores(grid, NCH)`
- HC-Tiled-Intlv: **two** `split_work_to_cores` calls — unpadded (`num_tensor_tiles`) and padded
  (`padded_num_tensor_tiles`); `active_cores = num_cores > padded_num_cores ? all_cores : padded_all_cores`
- HC-Tiled: `split_work_to_cores(grid, num_tensor_tiles)`
- WH: `split_work_to_cores(grid, row_major ? NC : num_tensor_tiles)`
- WH-Sharded: n/a — shard-spec driven; `grid_to_cores_with_noop(...)` over the full grid, with
  zero-filled RTAs on the inactive (noop) cores

None of the six emits **per-group CTAs**; the group distinction only ever reaches RTA values. So there is
no multi-`KernelDescriptor` work-split multiplicity to preserve.

### Shared kernels

Established by `grep -rl <filename> ttnn/cpp/ttnn/operations/` and disambiguating the hits
(the `experimental/quasar/` tree is excluded from consideration entirely and was not read).

**Borrowed (live outside the transpose directory) — all three already have a `_metal2` fork on `main` → rung 1, reuse:**

| kernel | bound by | fork on main | fork vocabulary (the constraint on my `KernelSpec`) |
|---|---|---|---|
| `eltwise/unary/…/writer_unary_interleaved_start_id.cpp` | HC-Tiled writer, WH-tiled writer (+~26 host files) | `…_metal2.cpp` ✓ | `dfb::out`, `tensor::dst`, `tensor::output`, `args::num_pages`, `args::start_id`; `#ifdef BACKWARDS`, `#ifdef OUT_SHARDED` |
| `eltwise/unary/…/reader_unary_sharded.cpp` | WH-Sharded reader (+~11 host files) | `…_metal2.cpp` ✓ | `dfb::in`, `args::num_tiles_per_core` |
| `data_movement/sharded/…/writer_unary_sharded.cpp` | WH-Sharded writer (+~11 host files) | `…_metal2.cpp` ✓ | `dfb::out`, `args::num_units` |

**Lent (live in the transpose directory, bound by peer ops) — must NOT be converted in place:**

| kernel | peer consumers | fork on main | rung |
|---|---|---|---|
| `compute/transpose_wh.cpp` | `nlp_create_qkv_heads`, `nlp_create_qkv_heads_boltz`, `nlp_create_qkv_heads_vit`, `split_query_key_value_and_split_heads` | `transpose_wh_metal2.cpp` ✓ (owned by **permute**) | **1 — reuse** if its vocabulary fits; else **2 — create a second fork** |
| `compute/transpose_wh_sharded.cpp` | `create_qkv_heads`, `create_qkv_heads_from_separate_tensors`, `split_query_key_value_and_split_heads_sharded` | **none** | **2 — create the fork** beside the original + pointer comment |
| `dataflow/reader_unary_transpose_hc_interleaved_tiled_padding_aware.cpp` | *(none today — permute binds the `_metal2` fork, not this)* | `…_metal2.cpp` ✓ (owned by **permute**) | **1 — reuse** if vocabulary fits |

All other transpose kernels are **transpose-only** and convert in place.

### Flags — additions found during construction

- **`transpose_wh_rm.cpp` is an INTRA-OP shared kernel — the brief missed it.** It is bound both by the
  in-scope `TransposeWHProgramFactory` (compiled without `SHARDED`) and by the **gated**
  `TransposeWHShardedRMProgramFactory` (compiled with `SHARDED=1`, and staying on the legacy path).
  Converting it in place would break the gated factory at JIT time. Handled by forking to
  `transpose_wh_rm_metal2.cpp` (non-sharded path only) with a pointer comment in the original.
  A factory×kernel overlap check across all eight factories confirmed this is the **only** such
  collision. Full write-up in the port report.

### Flags

- **Dead RTA (CN reader).** `reader_unary_transpose_cn_interleaved_start_id.cpp:16` reads
  `C = get_arg_val<uint32_t>(2)` and never uses it. Preserved as a named RTA (a syntax swap does not
  drop live plumbing); reported.
- **Dead CTA slots (HC-RM).** The reader's slot 0 (`N`) and slot 4 (`aligned_page_size`) and the
  writer's slot 2 (`aligned_page_size`) are emitted by the host and never read by the kernel
  (`TensorAccessorArgs<5>` / `<3>` skip straight past them). Preserved as named CTAs; reported.
- **Unreferenced kernel file.** `device/kernels/dataflow/reader_unary_transpose_wh_interleaved.cpp` is
  bound by no factory (dead code in the directory). Not audited, not touched.
- **WH-Sharded DFB placement asymmetry.** Legacy allocates `c_0` on `all_cores` (the shard grid) but
  `c_16` on `total_cores` (the full grid), while all three kernels run on `total_cores`. Metal 2.0 has no
  `core_ranges` on `DataflowBufferSpec` — placement is derived from the kernel bindings — so the ported
  `c_0` will be placed across `total_cores`. For a **borrowed** DFB whose backing tensor is sharded only
  over `all_cores`, the noop cores have no shard to borrow. Flagged as the single most likely failure
  point of this port; resolved at construction/verification, not assumed away.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept` (the brief's `MetalV2FactoryConcept`)
  — re-confirmed against current `main`: none of the six in-scope factories declares an
  `override_runtime_arguments`, so the base concept applies and
  [Translating `override_runtime_arguments`] does not.
- **Custom `compute_program_hash`**: none — leave the default reflection hash alone.
- **Implementation notes**: each factory's `create_descriptor` becomes `create_program_artifacts`
  returning `ttnn::device_operation::ProgramArtifacts`. The `program_factory_t` variant on
  `TransposeDeviceOperation` will hold a **mix** of concepts (6 Metal 2.0, 2 legacy descriptor) — valid,
  dispatched per-factory by `std::visit`. No pybind `create_descriptor` to remove (nanobind exposes only
  the `transpose` free function), so device-op-class exceptions 1–3 all **do not apply**.

## Planned Spec Shape

Naming: DFB **spec** names are host-side identities (free choice); DFB **accessor** names are the
kernel-facing `dfb::` tokens and are *fixed by the shared forks* where one is reused. No name contains
`cb` in any casing.

### CN
- KernelSpecs: `READER`, `WRITER` (both DM)
- DataflowBufferSpecs: `IN0` (entry_size=`stick_size`, num_entries=2)
- TensorParameters: `INPUT` (accessor `src`), `OUTPUT` (accessor `dst`)
- WorkUnitSpecs: one — {READER, WRITER} over `all_cores`
- Semaphores / op-owned tensors: none

### HC-RM
- KernelSpecs: `READER`, `WRITER`
- DataflowBufferSpecs: `IN0` (entry_size=`stick_size`, num_entries=`num_sticks`)
- TensorParameters: `INPUT`, `OUTPUT`
- WorkUnitSpecs: one over `all_cores`

### HC-Tiled-Interleaved
- KernelSpecs: `READER`, `WRITER`
- DataflowBufferSpecs: `IN0`; `PAD` **conditional** on `needs_padding`
- TensorParameters: `INPUT`, `OUTPUT`
- WorkUnitSpecs: one over `active_cores`

### HC-Tiled
- KernelSpecs: `READER`, `WRITER` (writer = reused `writer_unary_interleaved_start_id_metal2.cpp`)
- DataflowBufferSpecs: `IN0` (accessor **must be `out`** on the writer — the fork's vocabulary);
  `SCRATCH` **conditional** on `misaligned`, **self-looped** (reader is its only toucher)
- TensorParameters: `INPUT`, `OUTPUT` (writer's accessor name fixed by the fork: `dst`/`output`)
- WorkUnitSpecs: one over `all_cores`

### WH (one factory, two runtime-selected source sets — both convert together)
- KernelSpecs: `READER`, `WRITER`, `COMPUTE` (sources chosen by `row_major`, as legacy does)
- DataflowBufferSpecs: `IN0`, `OUT`; `IM` (c_24) **RM only**, **self-looped** (produced and consumed
  inside the compute kernel). **`c_25` (im2): no spec — dead-CB drop.**
- TensorParameters: `INPUT`, `OUTPUT`
- WorkUnitSpecs: one over `all_cores`
- `COMPUTE` gets `compiler_options.opt_level = O3`

### WH-Sharded
- KernelSpecs: `READER` (reused fork), `WRITER` (reused fork), `COMPUTE`
- DataflowBufferSpecs: `IN0` `borrowed_from = INPUT`; `OUT` `borrowed_from = OUTPUT`
- TensorParameters: `INPUT`, `OUTPUT`
- WorkUnitSpecs: one over `total_cores`
- `COMPUTE` gets `compiler_options.opt_level = O3`

## Preserved Multiplicity

**none — no work-split multiplicity in legacy.** Every factory emits exactly one `KernelDescriptor` per
role; the two core groups differ only in RTA *values*, never in CTAs, so there is no per-group CTA to
preserve and no reason to split a `KernelSpec`. (HC-Tiled-Interleaved runs two `split_work_to_cores`
calls, but both feed RTA values on a single pair of kernels.)

## Dropped Plumbing

| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| CN reader CTA 0 / writer CTA 0 | `src0_cb_index` (0) | `DFBBinding{IN0, …}` |
| CN reader CTA 3+ / writer CTA 3+ | `TensorAccessorArgs(*buf).append_to(...)` | `TensorBinding` + `TensorAccessor(tensor::…)` |
| CN reader RTA 0 / writer RTA 0 | `Buffer*` in `emplace_runtime_args` | `TensorBinding` (auto base address) |
| CN reader kernel `TensorAccessorArgs<3>()` | accessor-args chain | `TensorAccessor(tensor::src)` |
| HC-RM reader kernel `constexpr auto dfb_in0 = tt::CBIndex::c_0` | **hardcoded magic CB index** | `dfb::in0` |
| HC-RM writer CTA 0 | `src0_cb_index` | `DFBBinding{IN0, …}` |
| HC-RM reader CTA 5+ / writer CTA 3+ | `TensorAccessorArgs` | `TensorBinding` |
| HC-RM reader/writer RTA 0 | `Buffer*` | `TensorBinding` |
| HC-Tiled-Intlv writer CTA 1 | `tt::CBIndex::c_0` (magic index) | `DFBBinding{IN0, …}` |
| HC-Tiled-Intlv reader CTA 0+ / writer CTA 10+ | `TensorAccessorArgs` | `TensorBinding` |
| HC-Tiled-Intlv reader/writer RTA 0 | `Buffer*` | `TensorBinding` |
| HC-Tiled writer CTA 0 | `src0_cb_index` | `DFBBinding{IN0, accessor "out", CONSUMER}` |
| HC-Tiled reader CTA 3+ / writer CTA 1+ | `TensorAccessorArgs` | `TensorBinding` |
| HC-Tiled reader/writer RTA 0 | `Buffer*` | `TensorBinding` |
| WH writer CTA 0 | `output_cb_index` (c_16) | `DFBBinding{OUT, …}` |
| WH reader CTA tail / writer CTA tail | `TensorAccessorArgs` | `TensorBinding` |
| WH reader/writer RTA 0 | `Buffer*` | `TensorBinding` |
| WH RM branch, `transpose_wh_program_factory.cpp` c_25 block | dead `CBDescriptor` (im2) | **no spec — allocation dropped** |
| WH-Sharded reader CTA 0 / writer CTA 0 / compute CTA 0,1 | `src0_cb_index`, `output_cb_index` | `DFBBinding`s |
| WH-Sharded `.buffer = …buffer()` on both CBs | borrowed-memory CB | `DataflowBufferSpec::borrowed_from` |
| **all six**, every remaining positional CTA | `get_compile_time_arg_val(N)` | named CTAs, `get_arg(args::…)` |
| **all six**, every remaining positional RTA | `get_arg_val<uint32_t>(N)` | named RTAs, `get_arg(args::…)` |

**Dead unread scalars — dropped, and reported as findings.** The HC-RM reader's `N` and both HC-RM
kernels' `aligned_page_size` CTA slots (and the same on the WH row-major pair) are computed and shipped
by the legacy host to slots each kernel's `TensorAccessorArgs<N>` boundary sits past — no kernel ever
reads them. They are not re-emitted: an unread value carries no behavior (the same reasoning the
dead-CB disposition uses), and keeping them would leave a named arg the kernel never references. The CN
reader's unused `C` RTA is the one exception — that kernel *does* read it into a local, exactly as the
legacy kernel did, so it is preserved verbatim.

## Applied Patterns

- **Sync-free / single-ended CB → self-loop DFB**: HC-Tiled `SCRATCH` (c_1) — only the reader touches it,
  via `get_write_ptr()`; bound PRODUCER **and** CONSUMER on the reader.
- **Self-loop DFB binding**: WH-RM `IM` (c_24) — produced and consumed inside the compute kernel.
- **Conditional / optional DFB bindings**: HC-Tiled-Interleaved `PAD` (gated on `needs_padding`) and
  HC-Tiled `SCRATCH` (gated on `misaligned`) — host binds conditionally, `KernelSpec::compiler_options.defines`
  carries the flag, kernel `#ifdef`-gates both the alias and every use.
- **Dead CB drop**: WH-RM `c_25` (im2) — no spec built, allocation removed.
- **Borrowed-memory DFBs**: WH-Sharded `IN0` / `OUT` via `borrowed_from`.
- **Porting a shared kernel**: rung 1 (reuse) for the three peer `_metal2` forks and permute's two
  transpose forks; rung 2 (create) for `transpose_wh_sharded.cpp`.
- **Multi-variant / runtime source selection**: WH factory selects tiled vs row-major sources at runtime;
  both source sets convert together (atomic unit).

## Deferred / Flagged

- **Audit-vs-main divergence** on `get_dynamic_runtime_args` → `override_runtime_arguments` (detailed
  above). Does not change the subset's scope or concept; routed to the report for the sheet owner.
- **WH-Sharded borrowed `c_0` placement** across `total_cores` vs a shard grid of `all_cores` — the
  open structural risk of this port (see Flags). **Resolved during construction:** accepted by the
  validator with no spec change, and the sharded tests pass at baseline parity. See the port report.
- **No multi-binding DFB anywhere.** Re-derived from the kernel-touch census rather than taken from the
  brief: every DFB in the subset has either exactly one producer and one consumer on distinct kernels,
  or a single toucher (self-loop). `allow_instance_multi_binding` is not set anywhere.
