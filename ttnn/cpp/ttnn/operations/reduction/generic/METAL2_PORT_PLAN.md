# Port Plan — `reduction/generic` (`ReduceDeviceOperation` + `WelfordReduceDeviceOperation`)

Port plan for `ttnn/cpp/ttnn/operations/reduction/generic/`, ported from `ProgramDescriptor` to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

**Porting unit:** all **four** factories in one change (see [Why all four together](#why-all-four-factories-convert-together)).

---

## Legacy Inventory

### Legacy factory shape

- Concept: `ProgramDescriptorFactoryConcept` — every factory declares
  `static tt::tt_metal::ProgramDescriptor create_descriptor(...)`
  (`reduce_op_device_operation.hpp:25,32,39`; `welford_reduce_device_operation.hpp:24`).
- Variants (two device-ops sharing this directory):
  - `ReduceDeviceOperation` → `ReduceSingleCoreHwProgramFactory`, `ReduceMultiCoreHProgramFactory`,
    `ReduceMultiCoreWProgramFactory`
  - `WelfordReduceDeviceOperation` → `WelfordReduceProgramFactory`
- Custom `compute_program_hash`: **none** — already the default reflection-based hash
  (`grep -rn 'compute_program_hash' ttnn/cpp/ttnn/operations/reduction/generic/` → no hits).
- `get_dynamic_runtime_args` / `override_runtime_arguments`: absent from both device-ops and all four factories.
- Pybind: `std_var_reductions_nanobind.cpp` binds only the user-facing `ttnn.std` / `ttnn.var`; no
  `create_descriptor` exposure, so **no pybind deletion is forced**.

### Why all four factories convert together

The recipe's atomic unit is one factory, and it defaults to porting them one at a time. That default is
wrong for this op: the four factories share **kernel sources densely**, so a one-at-a-time port would
have to create an intra-op `_metal2` fork of nearly every kernel in the directory.

| Kernel source (this op's) | Bound by |
|---|---|
| `dataflow/reader_unary_reduce_universal_start_id.cpp` | MultiCoreW (tiled), SingleCoreHw, Welford (W) |
| `dataflow/reader_unary_transpose_wh_universal_input_cols_partitioned.cpp` | MultiCoreH (interleaved-tiled), Welford (H, HW) |
| `dataflow/reader_unary_reduce_rm.cpp` | MultiCoreH (dense-RM), MultiCoreW (dense-RM) |
| `dataflow/writer_reduce_rm_scalar.cpp` | MultiCoreH (dense-RM), MultiCoreW (dense-RM) |
| `compute/reduce.cpp` | MultiCoreH (tiled), MultiCoreW (tiled), SingleCoreHw |
| `compute/reduce_rm.cpp` | MultiCoreH (dense-RM), MultiCoreW (dense-RM) |

Converting all four factories at once means **zero intra-op forks**: every own kernel converts in place and
only the two *borrowed* writers get `_metal2` forks. That is both the smaller diff and the smaller risk, and
it matches the brief's framing ("they were audited — and should be ported — together").

### Variant: `ReduceMultiCoreHProgramFactory` (`reduce_op_multi_core_h_program_factory.cpp`)

Four config branches inside one `create_descriptor`, selected by
`rm_path = operation_attributes.row_major_h_dense_path`, `use_width_sharding` (input **and** output
WIDTH_SHARDED), and `use_fpu_negate = negate && !is_sfpu_reduce`:

1. **interleaved-tiled** (default) — optionally + fused negate
2. **width-sharded** — optionally + fused negate
3. **dense-RM** (`rm_path`)

`use_fpu_negate` is orthogonal to (1)/(2); it adds `c_4`/`c_5` and swaps `reduce.cpp` → `reduce_h_neg.cpp`.
`rm_path` and `use_width_sharding` are mutually exclusive in practice (RM validates INTERLEAVED I/O).

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader (tiled) | `dataflow/reader_unary_transpose_wh_universal_input_cols_partitioned.cpp` | `all_cores` | `{Ht, Wt, HtWt, scaler_bits, use_welford=0, fp32_sfpu_reduce}` + `TensorAccessorArgs(a)` | — | `{a, start_tile_id, curr_col_in_batch, num_cols_per_core}` | — | `reduce_defines` + `ENABLE_FP32_DEST_ACC`, `DST_SYNC_FULL` | O2 (unset, DM) | `ReaderConfigDescriptor{}` |
| reader (width-sharded) | `dataflow/reader_unary_transpose_wh_interleaved_input_cols_partitioned_sharded.cpp` | `all_cores` | `{src0_cb, src1_cb, scaler_cb, scaler_bits, fp32_sfpu_reduce}` | — | `{num_tiles, shard_Wt, Ht, NC, shard_row_size, shard_batch_size}` | — | `reduce_defines` + `REDUCE_SCALER=1`, `ENABLE_FP32_DEST_ACC`, `DST_SYNC_FULL` | O2 | `ReaderConfigDescriptor{}` |
| reader (dense-RM) | `dataflow/reader_unary_reduce_rm.cpp` | `all_cores` | `build_rm_reader_ct_args(...)` = `{scaler_bits, W_logical, elem_bytes, padding_identity_bits, Wt, wt_tiles_per_chunk, rm_rows_per_tile, ht_tiles_per_chunk, H_logical}` + `TensorAccessorArgs(a)` | — | `{a, num_output_tiles_local, output_tiles_seen}` | — | `reduce_defines` | O2 | `ReaderConfigDescriptor{}` |
| writer (tiled) | **borrowed** `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | `all_cores` | `{output_cb}` + `TensorAccessorArgs(output)` | — | `{output, num_cols_per_core, num_cols_read}` | — | — | O2 | `WriterConfigDescriptor{}` |
| writer (width-sharded) | **borrowed** `data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp` | `all_cores` | `{output_cb}` | — | `{num_cols_per_core_group_1}` | — | — | O2 | `WriterConfigDescriptor{}` |
| writer (dense-RM) | `dataflow/writer_reduce_rm_scalar.cpp` | `all_cores` | `build_rm_writer_ct_args(...)` = `{dst_datum_size, Wt, W_logical, wt_tiles_per_chunk}` + `TensorAccessorArgs(output)` | — | `{output, num_output_tiles_local, output_tiles_seen}` | — | `reduce_defines` | O2 | `WriterConfigDescriptor{}` |
| compute_g1 / compute_g2 (tiled) | `compute/reduce.cpp` or `compute/reduce_h_neg.cpp` (`negate`) | `core_group_1` / `core_group_2` | `{Ht, compute_Wt, compute_NC, post_mul_scaler_bits, fp32_sfpu_reduce}` | — | — | — | `reduce_defines` (+`REDUCE_POST_MUL`) | **O3** (unset compute) | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en, dst_full_sync_en, unpack_to_dest_mode}` |
| compute_g1 / compute_g2 (dense-RM) | `compute/reduce_rm.cpp` | `core_group_1` / `core_group_2` | `build_rm_compute_ct_args(...)` = `{Ht_rm, Wt, 1, post_mul_scaler_bits, wt_tiles_per_chunk, ht_tiles_per_chunk}` | — | `{num_output_tiles_local, output_tiles_seen}` (**arg 1 dead**, see Flags) | — | `reduce_defines` | **O3** | as above |

#### CBs

| index | total_size | core_ranges | data_format | page_size | tile | notes |
|---|---|---|---|---|---|---|
| `c_24` cb_rm (RM only) | `2*rm_rows_per_tile * rm_staging_page_size` | all_cores | src0 | `rm_staging_page_size` | unset | RM staging, one page = one chunk-wide RM row |
| `c_4` clear_value (RM only) | `src0_single_tile_size` | all_cores | src0 | `src0_single_tile_size` | unset | reader-private identity template |
| `c_5` cb_acc (RM only) | `wt_tiles_per_chunk * dst_tile` | all_cores | dst | `dst_single_tile_size` | unset | compute accumulator |
| `c_0` src0 | RM: `max(2, wt*ht_per_chunk)*src0_tile`; sharded: `2*src0_tile`; else `(use_fpu_negate?chunk_size:2)*src0_tile` | all_cores | src0 | `src0_single_tile_size` | unset | |
| `c_1` src1 (sharded only) | `num_shard_tiles * src0_tile` | all_cores | src0 | `src0_single_tile_size` | unset | **`.tensor = &a`** → borrowed |
| `c_2` scaler | `1 * scaler_tile` | all_cores | `Float32` if src0 fp32 else `Float16_b` | `scaler_single_tile_size` | unset | |
| `c_3` out | RM: `max(2,wt_per_chunk)*dst_tile`; sharded: `out_shard_tiles*dst_tile`; else `(use_fpu_negate?chunk_size:2)*dst_tile` | all_cores | dst | `dst_single_tile_size` | unset | sharded: **`.tensor = &output`** → borrowed |
| `c_4` acc (negate only) | `Ht*lcm(Wt_g1,Wt_g2) * dst_tile` | all_cores | dst | `dst_single_tile_size` | unset | L1-fit-checked |
| `c_5` ineg (negate only) | same as `c_4` | all_cores | dst | `dst_single_tile_size` | unset | |

No `CBDescriptor` sets `address_offset`, `global_circular_buffer`, or a multi-element `format_descriptors`
(no aliasing anywhere in this op).

#### Semaphores

none — `grep -rn 'emaphore'` over the op directory returns nothing.

#### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `…multi_core_h…:396` `TensorAccessorArgs(a).append_to(reader_cta)` | input `a` | reader slot 0 (`{a, …}` @ `:602`) |
| `…multi_core_h…:433` `TensorAccessorArgs(output).append_to(writer_cta)` | output | writer slot 0 (`{output, …}` @ `:605`) |
| `common.cpp:113` (dense-RM reader, via `build_rm_reader_ct_args`) | input `a` | reader slot 0 (`:540`) |
| `common.cpp:132` (dense-RM writer, via `build_rm_writer_ct_args`) | output | writer slot 0 (`:547`) |
| width-sharded config | — | **no address RTA at all** (borrowed DFBs) |

#### Work split

- Driver: `split_work_to_cores(grid | *sub_core_grids, num_cols)` where `num_cols = NC * Wt`
  (`…multi_core_h…:96-104`), **overridden** for width-sharding to the shard grid (`:108-115`):
  `all_cores = a.shard_spec()->grid`, `core_group_1 = all_cores`, `core_group_2 = {}`,
  `num_cols_per_core_group_1 = NC * (shard_shape[1] / tile_width)`.
- Core iteration order: `corerange_to_cores(all_cores, {}, row_wise=true)` on the RM path; explicit
  x/y walk over `all_cores.ranges()` when `sub_core_grids` is set; else
  `grid_to_cores(num_cores, grid.x, grid.y, false)`.

### Variant: `ReduceMultiCoreWProgramFactory` (`reduce_op_multi_core_w_program_factory.cpp`)

Two config branches: **interleaved-tiled** (± fused negate) and **dense-RM** (`row_major_w_dense_path`).

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | RTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|
| reader (tiled) | `dataflow/reader_unary_reduce_universal_start_id.cpp` | all_cores | `{scaler_bits}` + `TensorAccessorArgs(a)` | `{a, num_tensor_tiles_per_core, num_tiles_read}` | `reduce_defines` | O2 | `ReaderConfigDescriptor{}` |
| reader (dense-RM) | `dataflow/reader_unary_reduce_rm.cpp` | all_cores | `build_rm_reader_ct_args(..., W)` (no `H_logical` slot) + `TensorAccessorArgs(a)` | `{a, num_rows_per_core, num_rows_read}` | `reduce_defines` | O2 | `ReaderConfigDescriptor{}` |
| writer (tiled) | **borrowed** `writer_unary_interleaved_start_id.cpp` | all_cores | `{output_cb}` + `TensorAccessorArgs(output)` | `{output, tiles/Wt, start/Wt}` | `reduce_defines` | O2 | `WriterConfigDescriptor{}` |
| writer (dense-RM) | `dataflow/writer_reduce_rm_scalar.cpp` | all_cores | `build_rm_writer_ct_args(..., W)` = `{dst_datum_size}` + `TensorAccessorArgs(output)` | `{output, num_rows_per_core, num_rows_read}` | `reduce_defines` | O2 | `WriterConfigDescriptor{}` |
| compute_g1/g2 (tiled) | `compute/reduce.cpp` / `compute/reduce_w_neg.cpp` | core_group_1 / core_group_2 | `{ht_per_core_group_N, Wt, 1, post_mul_scaler_bits, fp32_sfpu_reduce}` | — | `reduce_defines` | **O3** | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en, unpack_to_dest_mode}` — **`dst_full_sync_en` NOT forwarded** |
| compute_g1/g2 (dense-RM) | `compute/reduce_rm.cpp` | core_group_1 / core_group_2 | `build_rm_compute_ct_args(plan, ht_per_core_group_N, post_mul_bits)` | — | `reduce_defines` | **O3** | as above |

#### CBs

`c_24` / `c_4` (clear_value) / `c_5` (cb_acc, sized `ht_tiles_per_chunk * dst_tile`) on the RM path;
`c_0` (`max(2, wt_tiles_per_chunk)` on RM else 2), `c_2` scaler (1 tile), `c_3` out (2 tiles) always;
`c_4` acc + `c_5` inv (1 tile each) when `use_fpu_negate`. No borrowed CBs, no aliasing.

#### Semaphores / Tensor accessors / Work split

- Semaphores: none.
- Tensor accessors: input @ `reader_compile_time_args` (`:190` tiled / `common.cpp:113` RM), output @
  `writer_compile_time_args` (`:198` tiled / `common.cpp:132` RM); addresses at RTA slot 0 of each
  (`:366-379` RM, `:383-397` tiled).
- Work split: `split_work_to_cores(grid | *sub_core_grids, num_rows, split_row_wise=rm_path)` with
  `num_rows = rm_path ? NC*H_logical : NC*Ht`. RM iterates
  `corerange_to_cores(all_cores, {}, row_wise=true)`; per-group counts are logical rows on the RM path
  and converted to tile-rows for the compute CTA via `ceil_div(rows, rm_rows_per_tile)`.

### Variant: `ReduceSingleCoreHwProgramFactory` (`reduce_op_single_core_hw_program_factory.cpp`)

Single config (± `negate`). One node: `selected_core_coord` = `{0,0}` or the first coord of
`sub_core_grids`.

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | RTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|
| reader | `dataflow/reader_unary_reduce_universal_start_id.cpp` | `core_set` | `{bit_cast(sqrt(scaler))}` + `TensorAccessorArgs(a)` | `{a, num_tensor_tiles, 0}` | `reduce_defines` (HW) | O2 | `ReaderConfigDescriptor{}` |
| writer | **borrowed** `writer_unary_interleaved_start_id.cpp` | `core_set` | `{output_cb}` + `TensorAccessorArgs(output)` | `{output, num_tensor_tiles/(Ht*Wt), 0}` | — (none) | O2 | `WriterConfigDescriptor{}` |
| compute | `compute/reduce.cpp` / `compute/reduce_hw_neg.cpp` | `core_set` | `{Ht, Wt, NC, post_mul_scaler_bits, 0}` | — | `reduce_defines` | **O3** | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en}` — **no `dst_full_sync_en`, no `unpack_to_dest_mode`** |

#### CBs

`c_0` src0 (2 tiles), `c_2` scaler (1 tile), `c_3` out (2 tiles); `c_4` acc + `c_5` ineg (1 tile each)
when `negate`. No borrowed CBs, no aliasing.

#### Semaphores / Work split

- Semaphores: none.
- Work split: n/a — single node.

### Variant: `WelfordReduceProgramFactory` (`welford_reduce_program_factory.cpp`)

Three config branches on `operation_attributes.reduce_dim`: **W**, **H**, **HW**.

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|
| reader (W) | `dataflow/reader_unary_reduce_universal_start_id.cpp` | all_cores | `{scaler_bits}` + `TensorAccessorArgs(input)` | — | `{input, num_input_tiles_per_core, input_tiles_offset}` | `reduce_defines` + `ENABLE_FP32_DEST_ACC`, `DST_SYNC_FULL` (+`WELFORD_POST_MUL`) | O2 | `ReaderConfigDescriptor{}` |
| reader (H, HW) | `dataflow/reader_unary_transpose_wh_universal_input_cols_partitioned.cpp` | all_cores | `{Ht, Wt, HtWt, scaler_bits, use_welford=1, 0}` + `TensorAccessorArgs(input)` | — | `{input, col_start_tile_id, curr_col_in_batch, num_cols}` | as above | O2 | `ReaderConfigDescriptor{}` |
| writer (W, H) | **borrowed** `writer_unary_interleaved_start_id.cpp` | all_cores | `{output_cb}` + `TensorAccessorArgs(output)` | — | `{output, num_output_tiles_per_core, output_tiles_offset}` | `reduce_defines` | O2 | `WriterConfigDescriptor{}` |
| writer (HW) | `dataflow/writer_welford_hw.cpp` | all_cores | `{Wt, W, tile_width, H, correction, reduce_batch_size, narrow_scratch_to_bf16}` + `TensorAccessorArgs(output)` | — | `{output, nc_slices_per_core, output_offset}` | **none** (deliberate) | O2 | `WriterConfigDescriptor{}` |
| compute_g1/g2 (W) | `compute/welford_reduce_w.cpp` | core_group_1 / core_group_2 | `{Wt, W, tile_width, post_mul_scaler_bits, correction, is_std}` | `{"welford_fp32_input", 0/1}` | `{num_work_units_per_core}` | `reduce_defines` | **O3** | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en, unpack_to_dest_mode}` — **no `dst_full_sync_en`** |
| compute_g1/g2 (H) | `compute/welford_reduce_h.cpp` | core_group_1 / core_group_2 | `{Ht, H, tile_height, post_mul_scaler_bits, correction, is_std}` | — | `{num_cols_per_core}` | as above | **O3** | as above |
| compute_g1/g2 (HW) | `compute/welford_reduce_hw.cpp` | core_group_1 / core_group_2 | `{Ht, H, tile_height, Wt, post_mul_scaler_bits, reduce_batch_size, is_std}` | — | `{nc_slices_per_core}` | as above | **O3** | as above |

#### CBs

| index | total_size | data_format | page_size | config |
|---|---|---|---|---|
| `c_0` in | `2 * input_tile` | input dtype | `input_single_tile_size` | all |
| `c_2` scalar | `1 * scalar_tile` | `Float32` if input fp32 else `Float16_b` | `scalar_single_tile_size` | all |
| `c_16` out | `2 * dst_tile` | output dtype | `dst_single_tile_size` | all |
| `c_19` cb_var | `1 * scratch_tile` | `Float32` if `fp32_dest_acc_en && !narrow_scratch_to_bf16` else `Float16_b` | `scratch_single_tile_size` | **W only** |
| `c_21` partial | `4 * fp32_tile` | `Float32` | `partial_single_tile_size` | **HW only** |
| `c_22` combined | `1 * tile` | `Float16_b` if `narrow_scratch_to_bf16` else `Float32` | `combined_single_tile_size` | **HW only** |

`unpack_to_dest_mode`: `c_0` = `UnpackToDestFp32` when input is Float32; `c_19` = `UnpackToDestFp32` when
`reduce_w && fp32_dest_acc_en && !narrow_scratch_to_bf16`; `c_22` = `UnpackToDestFp32` when
`reduce_hw && fp32_dest_acc_en && !narrow_scratch_to_bf16`.

#### Semaphores / Tensor accessors / Work split

- Semaphores: none.
- Tensor accessors: input @ reader CTA (`:328` H/HW, `:336` W), output @ writer CTA (`:367` HW, `:376` W/H);
  addresses at RTA slot 0 (`:513,516` W · `:547-559` HW · `:579-584` H).
- Work split: `split_work_to_cores(grid | *sub_core_grids, num_work_units)` with
  `num_work_units = reduce_w ? NC*Ht : reduce_hw ? NC/reduce_batch_size : NC*Wt`. Core order:
  explicit x/y walk over `all_cores.ranges()` when `sub_core_grids` is set, else
  `grid_to_cores(num_cores, grid.x, grid.y, false)`.

### Shared kernels

**Borrowed** (outside this op's directory) — no `_metal2` sibling exists beside either, so both land on
**rung 2 (create the fork)**:

| kernel | consumers of the legacy copy (tree-wide) | reduce configs that bind it |
|---|---|---|
| `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | ~34 factories | MultiCoreH (interleaved-tiled), MultiCoreW (tiled), SingleCoreHw, Welford (W, H) |
| `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp` | ~11 factories | MultiCoreH (width-sharded) |

Locational rung-1 check (`ls` of each original's directory) confirms no `*_metal2.cpp` sibling.

**Lent**: none — every kernel in this op's `device/kernels/` is bound only by this op's own four factories.
(The audit already disambiguated the two false positives on `compute/reduce.cpp`.)

**Intra-op**: six sources are shared across factories (table under
[Why all four factories convert together](#why-all-four-factories-convert-together)), but because all four
factories convert in this change there are **no intra-op forks** — each converts in place.

Fork naming discipline: `writer_unary_interleaved_start_id_metal2.cpp` is one of the most widely bound
kernels in the tree, so its binding names come from the *kernel's own* vocabulary — `dfb::out`
(kernel local `cb_id_out`), `tensor::dst` (kernel local `dst_addr`), RTAs `num_pages` / `start_id` — not
from reduce's locals. `writer_unary_sharded_metal2.cpp` likewise: `dfb::out`, RTA `num_units`.

### Flags

Noticed during inventory, **not acted on** (routed to `METAL2_PORT_REPORT.md`):

1. No unreferenced kernel files — every file in `device/kernels/` is bound by a factory.
2. `reduce_op_multi_core_h_program_factory.cpp:555,557` passes `{num_output_tiles_local, output_tiles_seen}`
   to `reduce_rm.cpp`, which reads only arg 0 and documents arg 1 as unused
   (`compute/reduce_rm.cpp:121-124`). The port carries both forward as named RTAs
   (`num_output_tiles_local`, `output_tiles_seen`) so behaviour is unchanged; dropping the dead arg is a
   separate cleanup. **Correction to the audit's framing:** the RM compute kernel *is* declared with two
   RTAs on the H path only, so the dead slot exists only there.
3. Stale comment `reduce_op_multi_core_w_program_factory.cpp:365` ("Use raw addresses (not Buffer\*) …")
   describes a shape the code no longer has. Deleted as part of the RTA rewrite it annotates (the line it
   describes is itself being rewritten) — noted in the report.
4. `dst_full_sync_en` is dropped by three of the four factories (audit anomaly 3). The port **preserves the
   drop** — see [Hardware configuration](#hardware-configuration-decisions).
5. `math_approx_mode` and `packer_l1_acc` are destructured and unused in all four factories (audit
   anomaly 6). `math_approx_mode` matters to the port — see
   [Hardware configuration](#hardware-configuration-decisions).
6. Welford's `c_2` scalar CB is filled by the reader and read by no Welford compute kernel (audit anomaly
   2). Carried forward faithfully as a **self-loop** on the reader.
7. `MULTI_CORE_HW` never selects a distinct factory (audit anomaly 1). Untouched.

---

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept` — all four factories.
- **Custom `compute_program_hash`**: none.
- **Implementation notes**:
  - Each factory's `create_descriptor` becomes
    `static ttnn::device_operation::ProgramArtifacts create_program_artifacts(const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&)`
    in its device-op header. Both `program_factory_t` variants keep their existing membership; all
    factories in each variant flip together, so no mixed-concept variant arises.
  - `<tt-metalium/program_descriptors.hpp>` drops from both device-op headers; `"ttnn/metal_v2_artifacts.hpp"`
    is added.
  - Op-owned tensors: none.
  - `tensor_args_t` is a bare `Tensor` for both device-ops, so `tensor_args.mesh_tensor()` /
    `tensor_return_value.mesh_tensor()` are extracted once at the top of each factory (the Welford factory
    additionally reads `tensor_arg` fields directly today — those reads move onto the extracted
    `MeshTensor`).
  - **Unity-build hygiene**: the four factory `.cpp`s live in the same CMake target and are unity-built, so
    every spec-name constant is declared **function-local** (`const DFBSpecName IN_DFB{"in"};` inside
    `create_program_artifacts`), not in an anonymous namespace — per
    [Pattern: Unity-build hygiene](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-unity-build-hygiene-for-anonymous-namespace-symbols).

---

## Planned Spec Shape

### Variant: `ReduceMultiCoreH`

- **KernelSpecs** (per config): `READER`, `WRITER`, `COMPUTE_G1`, and `COMPUTE_G2` when
  `core_group_2` is non-empty. Sources exactly as inventoried per branch.
- **DataflowBufferSpecs**: one per legacy `CBDescriptor` of the selected branch. `SRC1_DFB` (width-sharded
  `c_1`) gets `borrowed_from = INPUT_TENSOR`; `OUT_DFB` on the width-sharded branch gets
  `borrowed_from = OUTPUT_TENSOR`.
- **SemaphoreSpecs**: none.
- **TensorParameters**: `INPUT_TENSOR` (input `a`) and `OUTPUT_TENSOR` (output) on **every** branch —
  on the width-sharded branch they are declared not for a `TensorAccessor` but to back the two borrowed
  DFBs, which satisfies the "every TensorParameter needs ≥1 binding" validator rule via `borrowed_from`
  *plus* an explicit `TensorBinding`… **see [Deferred / Flagged](#deferred--flagged) item 1** — this is the
  one open question the plan carries into construction.
- **WorkUnitSpecs**: `wu_g1 = {READER, WRITER, COMPUTE_G1} @ core_group_1`; `wu_g2 = {READER, WRITER,
  COMPUTE_G2} @ core_group_2` when present. Reader/writer therefore cover `all_cores` (the union), exactly
  matching the legacy `core_ranges = all_cores`.
- **Op-owned tensors**: none.

### Variant: `ReduceMultiCoreW`

Same shape: `READER`, `WRITER`, `COMPUTE_G1` (+`COMPUTE_G2`); DFBs per branch; no borrowed DFBs; two
`TensorParameter`s; `wu_g1`/`wu_g2`.

### Variant: `ReduceSingleCoreHw`

`READER`, `WRITER`, `COMPUTE`; DFBs `IN`/`SCALER`/`OUT` (+`ACC`/`INEG` when `negate`); two
`TensorParameter`s; one `WorkUnitSpec` `wu_main = {READER, WRITER, COMPUTE} @ selected_core_coord`.
RTA tables built with `MakeRuntimeArgsForSingleNode`.

### Variant: `WelfordReduce`

Per `reduce_dim` branch inside `create_program_artifacts`
([Pattern: Multi-variant factories](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-multi-variant-factories)):
`READER`, `WRITER`, `COMPUTE_G1` (+`COMPUTE_G2`); DFBs `IN`/`SCALAR`/`OUT` always, `VAR` on W,
`PARTIAL`+`COMBINED` on HW; two `TensorParameter`s; `wu_g1`/`wu_g2`.

---

## Preserved Multiplicity

Every factory's compute kernel is instantiated once per **core group** — the ordinary disjoint-node
work split, not the same-grid dual-instance shape.

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| `compute_desc_g1`, `compute_desc_g2` of `reduce.cpp` / `reduce_h_neg.cpp` (MultiCoreH tiled) | `COMPUTE_G1`, `COMPUTE_G2` | `wu_g1`, `wu_g2` | `IN` CONSUMER · `SCALER` CONSUMER · `OUT` PRODUCER (+ `ACC`, `INEG` self-loops when negate) |
| `compute_desc_g1`, `compute_desc_g2` of `reduce_rm.cpp` (MultiCoreH / MultiCoreW dense-RM) | `COMPUTE_G1`, `COMPUTE_G2` | `wu_g1`, `wu_g2` | `RM` CONSUMER · `SCALER` CONSUMER · `OUT` PRODUCER (+ `TILE_IN`, `ACC` self-loops) |
| `compute_desc_g1`, `compute_desc_g2` of `reduce.cpp` / `reduce_w_neg.cpp` (MultiCoreW tiled) | `COMPUTE_G1`, `COMPUTE_G2` | `wu_g1`, `wu_g2` | `IN` CONSUMER · `SCALER` CONSUMER · `OUT` PRODUCER (+ `ACC`, `INV` self-loops when negate) |
| `compute_desc_g1`, `compute_desc_g2` of `welford_reduce_{w,h,hw}.cpp` | `COMPUTE_G1`, `COMPUTE_G2` | `wu_g1`, `wu_g2` | `IN` CONSUMER · `OUT` PRODUCER (+ per-dim `VAR` self-loop / `PARTIAL` PRODUCER + `COMBINED` CONSUMER) |

Both instances of a pair bind the same DFB with the **same** endpoint role. That is legal without any
advanced option because their node sets are disjoint and they are the same kernel kind
([`dataflow_buffer_spec.hpp:41-50`](../../../../../../tt_metal/api/tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp)).
The per-group CTAs (`compute_Wt` / `ht_per_core_group_N`) stay **compile-time args** — no demotion to RTA.

`ReduceSingleCoreHw` has no multiplicity (single node, single compute spec).

---

## DFB endpoint dispositions (re-derived from the kernel-touch census)

Re-derived per `(DFB, config)` from the kernel bodies rather than transcribed. Agrees with the brief
everywhere; the two clarifications are flagged.

| Factory (config) | DFB | Touchers | Census | Disposition |
|---|---|---|---|---|
| H/W/HW tiled | `IN` (`c_0`) | reader `push_back` / compute `wait_front`+`pop_front` | 1P+1C | 1:1 |
| H/W/HW tiled | `SCALER` (`c_2`) | reader `prepare_reduce_scaler` / compute `wait_front`+`pop_front` | 1P+1C | 1:1 |
| H/W/HW tiled | `OUT` (`c_3`) | compute `push_back` / writer `wait_front`+`pop_front` | 1P+1C | 1:1 |
| H/W/HW + negate | `ACC` (`c_4`), `INEG`/`INV` (`c_5`) | compute only (packs in, unpacks back out) | 1 toucher | **self-loop** on compute |
| H width-sharded | `SRC1` (`c_1`, borrowed from input) | reader only — `reserve_back(num_tiles)` + `get_write_ptr()` peek, no push/pop | 1 toucher | **self-loop** on reader |
| H width-sharded | `OUT` (`c_3`, borrowed from output) | compute `push_back` / writer `wait_front`+`pop_front` | 1P+1C | 1:1 |
| H/W dense-RM | `RM` (`c_24`) | reader `push_back` / compute `tilize<…, cb_rm, …>` drains | 1P+1C | 1:1 |
| H/W dense-RM | `CLEAR_VALUE` (`c_4`) | reader only — `get_write_ptr()` fill, `push_back`, then `get_read_ptr()` re-read as a NoC source | 1 toucher | **self-loop** on reader |
| H/W dense-RM | `TILE_IN` (`c_0`) | compute only — `tilize` writes, `reduce` drains | 1 toucher | **self-loop** on compute |
| H/W dense-RM | `ACC` (`c_5`) | compute only — `Accumulate::at(cb_acc, …)` round-trip | 1 toucher | **self-loop** on compute |
| H/W dense-RM | `SCALER` (`c_2`) | reader / compute | 1P+1C | 1:1 |
| H/W dense-RM | `OUT` (`c_3`) | compute `push_back` / writer `wait_front`+`pop_front` | 1P+1C | 1:1 |
| Welford all dims | `IN` (`c_0`) | reader / compute | 1P+1C | 1:1 |
| Welford all dims | `SCALAR` (`c_2`) | reader only (`prepare_reduce_scaler`); **no** Welford compute kernel reads it | 1 toucher | **self-loop** on reader |
| Welford W, H | `OUT` (`c_16`) | compute / borrowed writer | 1P+1C | 1:1 |
| Welford W | `VAR` (`c_19`) | compute only — packs, then `wait_front`/`transpose_tile`/`pop_front` | 1 toucher | **self-loop** on compute |
| Welford HW | `PARTIAL` (`c_21`) | compute `push_back(2)` / writer `wait_front(2)`+`pop_front(2)` | 1P+1C | 1:1 |
| Welford HW | `COMBINED` (`c_22`) | writer `reserve_back`/`push_back` / compute `wait_front`+`pop_front` | 1P+1C (writer is the **producer**) | 1:1 |
| Welford HW | `OUT` (`c_16`) | compute / writer | 1P+1C | 1:1 |

**No multi-binding flag anywhere** — no `(DFB, config)` has ≥3 distinct touchers or two kernels locked to
the same FIFO role. **No dead DFB** — every DFB has at least one toucher, so nothing is dropped.

Two things worth stating explicitly because they invert the naive reading:

- `COMBINED` (Welford HW) is produced by the **writer** and consumed by **compute** — the reverse of the
  usual direction. The endpoint roles follow the FIFO calls, not the kernel names.
- `SRC1` (width-sharded) is *both* a borrowed-memory DFB and a self-loop. Those are orthogonal: the
  `borrowed_from` field decides where the bytes live; the endpoint pair decides who may touch it.

---

## Hardware configuration decisions

**Data movement.** Every DM kernel resolves to a plain `ReaderConfigDescriptor{}` /
`WriterConfigDescriptor{}` — i.e. the reader and writer defaults, byte-for-byte. So every reader takes
`ttnn::create_reader_datamovement_config(device->arch())` and every writer
`ttnn::create_writer_datamovement_config(device->arch())`. No custom triple, no `DM_DYNAMIC_NOC`
anywhere in this op.

**Compute — the important decision.** All four factories resolve a TTNN `ComputeKernelConfig` (via
`get_compute_kernel_config_args`) but then forward only a **subset** of it onto
`ComputeConfigDescriptor`, deliberately leaving the rest at the *Metal* defaults. That makes this op a
hybrid of the recipe's Style A and Style B, and a straight `to_compute_hardware_config(...)` would
silently change two settings:

| field | legacy resolved value | what `to_compute_hardware_config` would produce | action |
|---|---|---|---|
| `fpu_math_fidelity` | `math_fidelity` from the user config | same | helper is correct |
| `enable_32_bit_dest` | `fp32_dest_acc_en` from the user config | same | helper is correct |
| `sfpu_precision_mode` | **always `Precise`** — no factory sets `ComputeConfigDescriptor::math_approx_mode`, whose default is `false` | `Approximate` whenever the caller's `math_approx_mode` is true (and the `ComputeKernelConfig` struct default *is* true) | **override to `Precise`** on all four factories |
| `double_buffer_dest` | `!dst_full_sync_en` on **MultiCoreH only**; `true` on MultiCoreW / SingleCoreHw / Welford (they never forward the field, so the descriptor default `dst_full_sync_en=false` applies) | `!dst_full_sync_en` everywhere | **override to `true`** on MultiCoreW, SingleCoreHw, Welford |
| `bfp_pack_precision_mode` | `Approximate` (descriptor default `bfp8_pack_precise=false`) | `Approximate` (default) | no action |
| `unpack_modes` | per-factory `unpack_to_dest_mode` vector | left default | **set explicitly**, reindexed by DFB name |

So: call `ttnn::to_compute_hardware_config(device->arch(), config)`, then reach into the
`ComputeGen1Config` alternative and set `sfpu_precision_mode`, `double_buffer_dest` (where legacy dropped
it) and `unpack_modes`. Each override carries an inline comment stating the legacy value it reproduces.
Preserving the `dst_full_sync_en` drop is deliberate: fixing it would be a behaviour change, which belongs
in a separate PR (routed to the report).

**`unpack_modes` reindexing** (legacy `std::vector<UnpackToDestMode>` indexed by CB id → `Table<DFBSpecName,
UnpackMode>`), value mapping `Default → UnpackToSrc` (expressed by omitting the entry),
`UnpackToDestFp32 → UnpackToDest`:

| factory | legacy entry | Metal 2.0 entry |
|---|---|---|
| MultiCoreH, MultiCoreW | `unpack_to_dest_mode[c_0] = UnpackToDestFp32` when `fp32_sfpu_reduce` | `{IN_DFB, UnpackMode::UnpackToDest}` under the same condition |
| SingleCoreHw | none (never populated) | none |
| Welford | `[c_0]` when input is Float32; `[c_19]` when `reduce_w && fp32_dest_acc_en && !narrow_scratch_to_bf16`; `[c_22]` when `reduce_hw && …` | `{IN_DFB, UnpackToDest}`, `{VAR_DFB, UnpackToDest}`, `{COMBINED_DFB, UnpackToDest}` under the same conditions |

The Float32-consumer required-entry rule is already satisfied by these: every Float32 DFB a compute kernel
consumes under `enable_32_bit_dest` is exactly one of the above. Two to watch during verification —
Welford HW's `PARTIAL` (`c_21`, always Float32) is *produced* by compute, not consumed, so no entry is
required; and on the dense-RM path `RM`/`TILE_IN`/`ACC` carry Float32 when the input is fp32, which the
legacy code left at `Default`. If the validator demands entries there, they get `UnpackToSrc` (the legacy
`Default`), not `UnpackToDest`.

**`opt_level`.** No factory sets `KernelDescriptor::opt_level`, so every DM kernel resolves to `O2`
(= Metal 2.0's default; nothing to write) and every **compute** kernel resolves to `O3`. Every compute
`KernelSpec` therefore sets `.compiler_options.opt_level = KernelBuildOptLevel::O3` explicitly — that is
9 compute specs across the four factories (2 per multi-core factory × per-branch source, 1 for
SingleCoreHw), each needing its own statement.

**`tile_format_metadata`.** No legacy `CBFormatDescriptor` sets `.tile`, so the field stays `nullopt` on
every DFB.

---

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `…multi_core_h…:396` | `TensorAccessorArgs(a).append_to(reader_compile_time_args)` | `TensorBinding{INPUT_TENSOR, "src"}` |
| `…multi_core_h…:433` | `TensorAccessorArgs(output).append_to(writer_compile_time_args)` | `TensorBinding{OUTPUT_TENSOR, "dst"}` |
| `…multi_core_h…:602-603` | reader RTA slot 0 = `a` (MeshTensor binding overload) | `TensorBinding` + auto-injected base address |
| `…multi_core_h…:605-611` | writer RTA slot 0 = `output` | ditto |
| `…multi_core_h…:540-553` | dense-RM reader/writer RTA slot 0 = `a` / `output` | ditto |
| `…multi_core_w…:190,198` | `TensorAccessorArgs(a/output).append_to(...)` | `TensorBinding`s |
| `…multi_core_w…:366-379, 383-397` | reader/writer RTA slot 0 = `a` / `output` | ditto |
| `…single_core_hw…:125,154` | `TensorAccessorArgs(a/output).append_to(...)` | `TensorBinding`s |
| `…single_core_hw…:203,213` | reader/writer RTA slot 0 = `a` / `output` | ditto |
| `…welford…:328,336,367,376` | `TensorAccessorArgs(input/output).append_to(...)` | `TensorBinding`s |
| `…welford…:513,516,547-559,579-584` | reader/writer RTA slot 0 = `input` / `output` | ditto |
| `common.cpp:113,132` | `TensorAccessorArgs(src/dst).append_to(args)` inside the RM CT-arg builders | the builders return **named CTA tables** with no accessor args |
| `…multi_core_h…:381` reader CTA slots 0,1,2 (width-sharded) | `{src0_cb_index, src1_cb_index, scaler_cb_index}` — magic CB indices | `DFBBinding`s (`IN`, `SRC1`, `SCALER`) |
| `…multi_core_h…:426,432` / `…multi_core_w…:197` / `…single_core_hw…:153` / `…welford…:375` writer CTA slot 0 | `{output_cb_index}` — magic CB index | `DFBBinding` (`OUT`) |
| `…multi_core_h…` etc. reader/compute CTA lists | positional `{Ht, Wt, HtWt, scaler_bits, use_welford, fp32_sfpu_reduce}` … | named CTAs: `{"Ht",…}, {"Wt",…}, {"HtWt",…}, {"scaler_bits",…}, {"use_welford",…}, {"enable_fp32_sfpu",…}` |
| all remaining positional CTAs (RM reader/writer/compute, Welford W/H/HW compute, HW writer) | positional | named, per the CTA-name tables below |
| `reduce.cpp:11` | `#include "api/dataflow/circular_buffer.h"` + `CircularBuffer cb_scaler(c_2)` | `dataflow_buffer.h` + `DataflowBuffer` |
| `reduce_rm_dataflow_common.hpp:108,10` | `experimental::CB&` parameter + `pool/.../experimental_device_api.hpp` `CB` alias | `DataflowBuffer&` |

Nothing in this op passes a semaphore id (there are no semaphores) and there is no page-size third
`TensorAccessor` argument to drop.

### Named CTA / RTA vocabulary (per kernel)

Names are chosen to match the kernel's existing local variable names, so the kernel bodies change only at
the retrieval line.

| kernel | named CTAs | named RTAs |
|---|---|---|
| `reader_unary_reduce_universal_start_id.cpp` | `scaler_bits` | `num_tiles`, `start_id` |
| `reader_unary_transpose_wh_universal_input_cols_partitioned.cpp` | `Ht`, `Wt`, `HtWt`, `scaler_bits`, `use_welford`, `enable_fp32_sfpu` | `col_start_tile_id`, `curr_col_in_batch`, `num_cols` |
| `reader_unary_transpose_wh_interleaved_input_cols_partitioned_sharded.cpp` | `scaler_bits`, `enable_fp32_sfpu` (the three CB-index slots become DFB bindings) | `num_tiles`, `Wt`, `Ht`, `batch`, `row_size_bytes`, `batch_size_bytes` |
| `reader_unary_reduce_rm.cpp` | `scaler_bits`, `W_logical`, `elem_bytes`, `padding_identity_bits`, `Wt`, `wt_tiles_per_chunk`, `rm_rows_per_tile`, `ht_tiles_per_chunk`, `H_logical` (H path only) | `rt_count`, `rt_start` |
| `writer_reduce_rm_scalar.cpp` | `datum_bytes`, `Wt`, `W_logical`, `wt_tiles_per_chunk` (last three H path only) | `rt_count`, `rt_start` |
| `writer_welford_hw.cpp` | `Wt`, `W`, `tile_width`, `H`, `correction`, `reduce_batch_size`, `combined_is_bf16` | `NC_per_core`, `output_tile_start_id` |
| `compute/reduce.cpp`, `reduce_h_neg.cpp`, `reduce_w_neg.cpp`, `reduce_hw_neg.cpp` | `Ht`, `Wt`, `NC`, `post_mul_scaler_bits`, `enable_fp32_sfpu` | — |
| `compute/reduce_rm.cpp` | `Ht`, `Wt`, `NC`, `post_mul_scaler_bits`, `wt_tiles_per_chunk`, `ht_tiles_per_chunk` | `num_output_tiles_local`, `output_tiles_seen` (H path; slot 1 dead, preserved) |
| `compute/welford_reduce_w.cpp` | `Wt`, `W`, `tile_width`, `post_mul_scaler_bits`, `correction`, `is_std`, `welford_fp32_input` | `NCHt` |
| `compute/welford_reduce_h.cpp` | `Ht`, `H`, `tile_height`, `post_mul_scaler_bits`, `correction`, `is_std` | `NCWt` |
| `compute/welford_reduce_hw.cpp` | `Ht`, `H`, `tile_height`, `Wt`, `post_mul_scaler_bits`, `reduce_batch_size`, `is_std` | `NC_per_core` |
| `writer_unary_interleaved_start_id_metal2.cpp` (fork) | — (the `cb_id_out` slot becomes `dfb::out`) | `num_pages`, `start_id` |
| `writer_unary_sharded_metal2.cpp` (fork) | — | `num_units` |

`welford_fp32_input` is already a *named* legacy CTA (`named_compile_time_args`), so it maps 1:1 — and it is
emitted only on the W branch today. Because a name absent from `compile_time_args` produces no `args::`
declaration, the W-only emission must stay W-only; the other two Welford compute kernels never reference it.

**RTA varargs: none.** Every kernel reads a fixed set of runtime args at constant indices, so every RTA is
named.

**Conditionally-bound resources.** The `post_mul_scaler_bits` CTA is read only under `REDUCE_POST_MUL` /
`WELFORD_POST_MUL`, but the legacy factories emit it **unconditionally** — an unconditional named CTA is
harmless (the `args::` declaration exists whether or not the kernel reads it), so it stays unconditional and
no `#ifdef` gate is needed on the *host* side. The kernel-side `#ifdef` blocks stay exactly as they are.

The genuine conditional bindings are DFBs, and they are already `#ifdef`-free because the legacy code selects
a *different kernel source* per configuration rather than gating a binding inside one source:

- `ACC` / `INEG` / `INV` (`c_4`/`c_5`) exist only in the `_neg` compute sources, which are only bound when
  `negate` is set — so the binding and the source appear together. **No `#ifdef` needed.**
- `reduce_h_neg.cpp` / `reduce_w_neg.cpp` have an `if constexpr (is_sfpu_reduce_path<…>())` early-return
  branch that does **not** touch `c_4`/`c_5`, while the FPU branch below does. Both branches are in one
  source and the host binds `ACC`/`INEG` unconditionally for that source, so `dfb::acc` / `dfb::ineg`
  always resolve. **No `#ifdef` needed** — this is the case where the legacy CTA gate is *not* a binding
  gate.
- `REDUCE_SCALER` in the width-sharded reader gates `dfb::scaler` behind `#ifdef`. The reduce factory
  always defines `REDUCE_SCALER=1` for that kernel, so `SCALER` is bound unconditionally — but the
  `#ifdef` stays in the source (it is the kernel's own contract with its other potential callers, and the
  op is its only caller today, so nothing changes).
- The two writer forks' `#ifdef OUT_SHARDED` / `#ifdef BACKWARDS` blocks are preserved verbatim in the
  fork. Reduce defines neither; later consumers may.

---

## Applied Patterns

- [Sync-free and single-ended CBs → self-loop DFB](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb):
  `ACC`/`INEG`/`INV` on the fused-negate compute specs; `TILE_IN`/`ACC` on the dense-RM compute specs;
  `VAR` on Welford-W compute; `CLEAR_VALUE` on the dense-RM reader (a **DM** self-loop, Gen1-legal);
  `SRC1` on the width-sharded reader (DM self-loop *and* borrowed memory); `SCALAR` on the Welford reader
  (DM self-loop — the CB nothing reads).
- [Self-loop DFB binding](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-self-loop-dfb-binding):
  the mechanism the above use — one `accessor_name`, two `DFBBinding`s differing only in `endpoint_type`.
- [Demoting per-group CTA to RTA (avoided)](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta):
  two compute `KernelSpec`s per factory over disjoint core groups keep their per-group CTAs.
- [Multi-variant factories](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-multi-variant-factories):
  `reduce_dim` branching inside `WelfordReduceProgramFactory::create_program_artifacts`; the same shape for
  the Reduce factories' config branches.
- [Pass DFB handles directly to LLKs and kernel-lib helpers](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers):
  `dfb::name` flows into `reduce_init`, `pack_tile`, `copy_tile`, `compute_kernel_lib::reduce<>`,
  `compute_kernel_lib::tilize<>`, `Accumulate::at`, `dataflow_kernel_lib::prepare_reduce_scaler<>` — in
  both call-argument and non-type-template-parameter position.
- [Porting a shared kernel](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-porting-a-shared-kernel):
  **rung 2** for both borrowed writers.
- [Unity-build hygiene](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-unity-build-hygiene-for-anonymous-namespace-symbols):
  function-local spec-name constants in all four factory `.cpp`s.

Not applied: aliased DFBs (no legacy aliased CB), same-FIFO aliasing (no `uint32_t` CB-index alias in any
kernel), multi-binding advanced option (census never reaches ≥3 touchers), varargs, op-owned tensors.

---

## Deferred / Flagged

New findings from the planning step:

1. **Width-sharded `TensorParameter` binding shape — RESOLVED (a).** On the width-sharded H config the
   input and output tensors reach the kernels *only* through borrowed-memory DFBs — no kernel builds a
   `TensorAccessor` on them. The validator requires every `TensorParameter` to have ≥1 `TensorBinding`
   across the program's kernels, while `borrowed_from` is a *DFB*-side reference, not a `TensorBinding`.
   Two possible readings: (a) `borrowed_from` counts, and no `TensorBinding` is needed; (b) it does not,
   and an otherwise-unused `TensorBinding` must be added.
   **Resolved by reading the validator, not by trial:** `borrowed_from` explicitly registers the
   parameter as used — `tt_metal/impl/metal2_host_api/program_spec.cpp:533-552`
   ("register as used (no kernel user)"). So (a) holds and the width-sharded reader / writer declare no
   tensor bindings at all. Still a real doc gap (the two rules are documented independently); routed to
   the report.

2. **Constexpr DFB metadata has no object-based spelling.** Whitelist rule 7 says compile-time tile/format
   metadata moves onto the `DataflowBuffer` object, and the brief prescribes binding these sites "through a
   `constexpr` `DataflowBuffer`". That does not work: `DataflowBuffer`'s constructors are **not**
   `constexpr` (`dataflow_buffer.h:72,75`; the tt-1xx definition binds a reference to the runtime
   `get_local_cb_interface(id)` — `internal/tt-1xx/dataflow_buffer.inl:31`), so no `DataflowBuffer` object
   is usable in a constant expression, and its `constexpr` getters cannot be reached in one. The five
   affected sites all feed **template arguments**:
   - `constexpr DataFormat reduce_format = get_dataformat(dfb_id_in0);` →
     `reader_unary_transpose_wh_universal_input_cols_partitioned.cpp:35`,
     `reader_unary_transpose_wh_interleaved_input_cols_partitioned_sharded.cpp:38`
   - `constexpr DataFormat reduce_format = static_cast<DataFormat>(unpack_src_format[…]);` →
     `compute/reduce.cpp:41`, `compute/reduce_h_neg.cpp:38`, `compute/reduce_w_neg.cpp:40`

   Plan: keep the free-function / JIT-array spelling and index it with the **DFB handle** —
   `get_dataformat(dfb::in0)` in the DM kernels and `unpack_src_format[dfb::input]` in the compute kernels.
   The magic CB index still disappears (which is rule 7's substance); only the *retrieval* form stays
   legacy. The free-function form has in-tree Metal 2.0 precedent
   (`data_movement/scatter/device/kernels/dataflow/reader_bf16_reduction_scatter.cpp:137`). Routed to the
   report as a doc gap.

   Runtime metadata sites *do* move onto the object: `get_tile_size(id)` → `dfb.get_tile_size()` (×7) and
   `get_local_cb_interface(id).fifo_page_size` → `dfb.get_entry_size()` (×2, both on DM where
   `cb_addr_shift == 0`, so the value is byte-identical).

3. **`TensorParameter` relaxation field name drift.** The recipe calls it
   `TensorParameter::advanced_options`; the header field is `relaxations`
   (`tensor_parameter.hpp:45`). No relaxation is used here, so nothing turns on it — reported as a doc nit.

4. **The recipe's compute-config Style A / Style B dichotomy does not cover this op's shape** (resolves a
   TTNN `ComputeKernelConfig` but forwards only a subset onto the Metal descriptor). Following Style A
   literally would flip `sfpu_precision_mode` to `Approximate` for any caller-supplied config with
   `math_approx_mode = true` — a silent precision/perf change with no test signal. Handled per
   [Hardware configuration](#hardware-configuration-decisions); routed to the report as a doc gap.

5. **Two conditional-name hazards the planning pass missed — found during construction, both resolved
   with the whitelist-rule-6 mechanism (host define + kernel `#ifdef`).** The plan's
   [Dropped Plumbing](#dropped-plumbing) section concluded "no `#ifdef` needed" for the `_neg` kernels;
   that was wrong on one count, and a second case turned up in the RM compute kernel:

   - **`ACC` / `INEG` really are conditional bindings on `reduce_h_neg.cpp` / `reduce_w_neg.cpp`.** The
     plan reasoned that binding and source appear together because the source is selected on `negate`.
     But the *buffers* are allocated on `use_fpu_negate` (`= negate && !is_sfpu_reduce`), so on the
     Int32 negate path the `_neg` source is compiled with no `acc` / `ineg` buffers bound — and the
     kernel's FPU section is not in an `if constexpr` discarded branch at all (the SFPU branch
     `return`s and the FPU code follows at function scope), so `dfb::acc` / `dfb::ineg` must resolve
     regardless. Resolution: `REDUCE_FPU_NEGATE` emitted when `use_fpu_negate`, gating the FPU section.
   - **`reduce_rm.cpp`'s H-only compute RTA.** The H factory declares
     `num_output_tiles_local` / `output_tiles_seen`; the W factory declares no compute RTAs. Name
     lookup on `args::num_output_tiles_local` happens regardless of the `if constexpr`, so the W build
     would not compile. Resolution: `REDUCE_RM_H_PATH` gating the single read.

   The general rule this settles, applied throughout the port: a conditionally-needed **CTA** is
   emitted on *both* paths (free, and the name always resolves — this is why
   `build_rm_reader_ct_args` / `build_rm_writer_ct_args` now emit the full set on both dims), whereas a
   conditionally-needed **RTA** or **binding** gets the `#ifdef` treatment.

No structural blocker found. Nothing in this op requires a legacy workaround, a vararg, a multi-binding
flag, an address-through-RTA, or a hand-rolled primitive.
