# Port Plan — Reduction Op Family (W / H / HW + Welford)

Port plan for `ttnn::prim::ReduceDeviceOperation` and `ttnn::prim::WelfordReduceDeviceOperation`,
porting both from the `ProgramDescriptor` API to Metal 2.0 (`ProgramSpecFactoryConcept`).

Phase 0 audit: see `metal2_audit_reduction.md` at the repo root. Cleared YELLOW with user override on the borrowed-memory CB (PR #44662) and on the Device 2.0 DM holdovers (fold into port).

Written during Phases A–B; committed alongside the port for review.

## Legacy Inventory

*Phase A. Observation only. No decisions.*

### Factory shape

- Concept: **`ProgramDescriptorFactoryConcept`** (all four factories implement `create_descriptor` returning a `ProgramDescriptor`)
- Variants:
  - `ReduceDeviceOperation` has three program-factory variants, selected by `select_program_factory` (defined in `reduce_op_device_operation.cpp:13`) on the parallelization strategy:
    - `ReduceSingleCoreHwProgramFactory` (HW dim, single core; also receives multi-core HW)
    - `ReduceMultiCoreWProgramFactory` (W dim, multi-core)
    - `ReduceMultiCoreHProgramFactory` (H dim, multi-core; interleaved + width-sharded sub-paths)
  - `WelfordReduceDeviceOperation` has one program-factory variant with **three internal sub-variants** by `reduce_dim`:
    - W reduce
    - H reduce
    - HW reduce
- The Welford factory's three sub-variants share host code paths heavily; the W/H/HW selection happens inside `create_descriptor` via `if (reduce_w) ... else if (reduce_hw) ... else (reduce_h)` branches.

### Kernels

#### `ReduceSingleCoreHwProgramFactory`

| unique_id | source | core_ranges | CTAs (positional) | RTAs | defines | config |
|---|---|---|---|---|---|---|
| reader | `reduction/generic/device/kernels/dataflow/reader_unary_reduce_universal_start_id.cpp` | `core_set` (one core) | `{scaler_bits}` + `TensorAccessorArgs(src0_buffer)` (host-emitted CT block) | `{src_buffer*, num_tensor_tiles, 0u}` | `reduce_defines` (REDUCE_OP/REDUCE_DIM/REDUCE_POST_MUL?) | `ReaderConfigDescriptor` |
| writer | `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | `core_set` | `{output_cb_index}` + `TensorAccessorArgs(dst_buffer)` | `{dst_buffer*, num_tensor_tiles/out_dim_divider, 0u}` | (none) | `WriterConfigDescriptor` |
| compute | `reduction/generic/device/kernels/compute/reduce.cpp` or `reduce_hw_neg.cpp` (if `negate`) | `core_set` | `{Ht, Wt, NC, post_mul_scaler_bits}` | (none) | `reduce_defines` (+ REDUCE_POST_MUL?) | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en}` |

#### `ReduceMultiCoreWProgramFactory`

| unique_id | source | core_ranges | CTAs (positional) | RTAs (per core) | defines | config |
|---|---|---|---|---|---|---|
| reader | `…/reader_unary_reduce_universal_start_id.cpp` | `all_cores` | `{scaler}` + `TensorAccessorArgs(src)` | `{src_buffer*, num_tiles_per_core, num_tiles_read}` | `reduce_defines` | reader |
| writer | `eltwise/.../writer_unary_interleaved_start_id.cpp` | `all_cores` | `{output_cb_index}` + `TensorAccessorArgs(dst)` | `{dst_buffer*, num_tiles_per_core/Wt, num_tiles_read/Wt}` | `reduce_defines` | writer |
| compute (g1) | `reduce.cpp` or `reduce_w_neg.cpp` | `core_group_1` | `{num_rows_per_core_group_1 (=Ht), Wt, 1 (=NC), post_mul_scaler_bits}` | (none) | `reduce_defines` | compute |
| compute (g2) (optional) | (same source) | `core_group_2` | `{num_rows_per_core_group_2, Wt, 1, post_mul_scaler_bits}` | (none) | `reduce_defines` | compute |

**Work-split multiplicity:** two `KernelDescriptor`s of the same compute source, differing only on `Ht` (the per-group row count).

#### `ReduceMultiCoreHProgramFactory`

Two sub-paths gated by `use_width_sharding`:

**Interleaved path (`!use_width_sharding`):**

| unique_id | source | core_ranges | CTAs | RTAs (per core) | defines | config |
|---|---|---|---|---|---|---|
| reader | `…/reader_unary_transpose_wh_universal_input_cols_partitioned.cpp` | `all_cores` | `{Ht, Wt, HtWt, scaler_bits, 0 (=use_welford)}` + `TensorAccessorArgs(src)` | `{src_buffer*, col_start_tile_id, curr_col_in_batch, num_cols_per_core}` | `reduce_defines` + `ENABLE_FP32_DEST_ACC` + `DST_SYNC_FULL` | reader |
| writer | `eltwise/.../writer_unary_interleaved_start_id.cpp` | `all_cores` | `{output_cb_index}` + `TensorAccessorArgs(dst)` | `{dst_buffer*, num_cols_per_core, num_cols_read}` | (none) | writer |
| compute (g1) | `reduce.cpp` or `reduce_h_neg.cpp` | `core_group_1` | `{Ht, num_cols_per_core_group_1 (=Wt), 1 (=NC), post_mul_scaler_bits}` | (none) | `reduce_defines` | compute (+ `dst_full_sync_en`) |
| compute (g2) (optional) | (same source) | `core_group_2` | `{Ht, num_cols_per_core_group_2, 1, post_mul_scaler_bits}` | (none) | `reduce_defines` | compute |

**Width-sharded path (`use_width_sharding`):**

| unique_id | source | core_ranges | CTAs | RTAs (per core) | defines | config |
|---|---|---|---|---|---|---|
| reader | `…/reader_unary_transpose_wh_interleaved_input_cols_partitioned_sharded.cpp` | `all_cores` (=shard grid) | `{src0_cb_index, src1_cb_index, scaler_cb_index, scaler_bits}` | `{num_cols*Ht, shard_Wt, Ht, NC, shard_row_size, shard_batch_size}` | `reduce_defines` + `REDUCE_SCALER=1` + DEST flags | reader |
| writer | `data_movement/sharded/.../writer_unary_sharded.cpp` | `all_cores` | `{output_cb_index}` | `{num_cols_per_core_group_1}` | (none) | writer |
| compute (g1) | `reduce.cpp` or `reduce_h_neg.cpp` | `core_group_1` (=`all_cores`) | `{Ht, shard_Wt, NC, post_mul_scaler_bits}` | (none) | `reduce_defines` | compute |
| (no g2 for sharded) | | | | | | |

#### `WelfordReduceProgramFactory` (multi-variant by `reduce_dim`)

Shared across variants: input CB `c_0` (2 entries), scalar CB `c_2` (1 entry, bfloat16), output CB `c_16` (2 entries). Variant-specific CBs called out per-variant below.

**W variant (`reduce_dim == W`):**

| unique_id | source | core_ranges | CTAs | RTAs (per core) | defines | config |
|---|---|---|---|---|---|---|
| reader | `…/reader_unary_reduce_universal_start_id.cpp` | `all_cores` | `{scaler_bits}` + `TensorAccessorArgs(in)` | `{in_buffer*, num_input_tiles_per_core, input_tiles_offset}` | `reduce_defines` (+ FP32/DST_SYNC flags) | reader |
| writer | `eltwise/.../writer_unary_interleaved_start_id.cpp` | `all_cores` | `{output_cb_index (=16)}` + `TensorAccessorArgs(out)` | `{out_buffer*, num_output_tiles_per_core, output_tiles_offset}` | `reduce_defines` | writer |
| compute (g1) | `…/compute/welford_reduce_w.cpp` | `core_group_1` | `{Wt, W, tile_width, do_scale, correction, is_std}` | `{num_work_units_per_core}` | `reduce_defines` | compute |
| compute (g2) (optional) | (same) | `core_group_2` | (same CTAs) | `{num_work_units_per_core_g2}` | `reduce_defines` | compute |

Variant-specific extras: scratch CB `c_19` (W only); scaled CB `c_20` (W only, gated by `do_scale`).

**H variant (`reduce_dim == H`):**

| unique_id | source | core_ranges | CTAs | RTAs (per core) | defines | config |
|---|---|---|---|---|---|---|
| reader | `…/reader_unary_transpose_wh_universal_input_cols_partitioned.cpp` | `all_cores` | `{Ht, Wt, HtWt, scaler_bits, 1 (=use_welford)}` + `TensorAccessorArgs(in)` | `{in_buffer*, (num_cols_read/Wt*HtWt)+(num_cols_read%Wt), num_cols_read%Wt, num_cols_per_core}` | `reduce_defines` (+FP32/DST_SYNC) | reader |
| writer | `eltwise/.../writer_unary_interleaved_start_id.cpp` | `all_cores` | `{output_cb_index}` + `TensorAccessorArgs(out)` | `{out_buffer*, num_cols_per_core, num_cols_read}` | `reduce_defines` | writer |
| compute (g1) | `…/compute/welford_reduce_h.cpp` | `core_group_1` | `{Ht, H, tile_height, do_scale, correction, is_std}` | `{num_cols_per_core}` | `reduce_defines` | compute |
| compute (g2) (optional) | (same) | `core_group_2` | (same CTAs) | `{num_cols_per_core_g2}` | `reduce_defines` | compute |

**HW variant (`reduce_dim == HW`):**

| unique_id | source | core_ranges | CTAs | RTAs (per core) | defines | config |
|---|---|---|---|---|---|---|
| reader | `…/reader_unary_transpose_wh_universal_input_cols_partitioned.cpp` | `all_cores` | (same as H variant) | `{in_buffer*, col_start_tile_id, 0, num_cols=Wt*nc_slices_per_core}` | `reduce_defines` (+FP32/DST_SYNC) | reader |
| writer | `…/dataflow/writer_welford_hw.cpp` (custom HW writer) | `all_cores` | `{Wt, W, tile_width, H, correction, reduce_batch_size}` + `TensorAccessorArgs(out)` | `{out_buffer*, nc_slices_per_core, output_offset}` | (no reduce_defines — matches original) | writer |
| compute (g1) | `…/compute/welford_reduce_hw.cpp` | `core_group_1` | `{Ht, H, tile_height, Wt, do_scale, reduce_batch_size, is_std}` | `{nc_slices_per_core}` | `reduce_defines` | compute |
| compute (g2) (optional) | (same) | `core_group_2` | (same CTAs) | `{nc_slices_per_core_g2}` | `reduce_defines` | compute |

Variant-specific extras (HW): partial CB `c_21` (4 entries, Float32) and combined CB `c_22` (1 entry, Float32).

### CBs

| index symbol | factory | total_size | core_ranges | data_format | page_size | notes |
|---|---|---|---|---|---|---|
| `c_0` (src0) | all | `2 * src0_single_tile_size` | all_cores | input dtype | `src0_single_tile_size` | input |
| `c_1` (src1) | H factory, width-sharded only | `num_shard_tiles * src0_single_tile_size` | all_cores | input dtype | `src0_single_tile_size` | **borrowed memory** (`.buffer = a.buffer()`) — dynamic CB |
| `c_2` (scaler) | all | `1 * scaler_single_tile_size` | all_cores | Float32 or Float16_b (auto) | `scaler_single_tile_size` | scaler tile |
| `c_3` (out) | Reduce* (non-Welford) | `2 * dst_single_tile_size`, or `num_shard_tiles * dst_single_tile_size` (H sharded) | all_cores | output dtype | `dst_single_tile_size` | output. In H sharded path, **borrowed memory** (`.buffer = output.buffer()`) |
| `c_4` (acc) | Reduce*, `negate` only | varies by factory (H neg: `Ht * per_nc_advance * dst_size`; W/HW neg: `1 * dst_size`) | all_cores | output dtype | `dst_single_tile_size` | negate accumulator |
| `c_5` (ineg) | Reduce*, `negate` only | same as c_4 | all_cores | output dtype | `dst_single_tile_size` | negate inverse |
| `c_16` (out) | Welford | `2 * dst_single_tile_size` | all_cores | output dtype | `dst_single_tile_size` | Welford output |
| `c_19` (scratch) | Welford W only | `1 * scratch_single_tile_size` | all_cores | Float32 (if fp32_dest_acc_en) else Float16_b | matching | W transpose scratch |
| `c_20` (scaled) | Welford W, `do_scale` only | `1 * input_single_tile_size` | all_cores | input dtype | `input_single_tile_size` | scaled tile |
| `c_21` (partial) | Welford HW only | `4 * partial_single_tile_size` (Float32) | all_cores | Float32 | matching | per-column partials |
| `c_22` (combined) | Welford HW only | `1 * combined_single_tile_size` (Float32) | all_cores | Float32 | matching | combined scalar |

### Semaphores

None — no `SemaphoreDescriptor` in any of the four factories.

### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) | CTA offset (kernel) |
|---|---|---|---|
| `reduce_op_single_core_hw_program_factory.cpp:131` | `a` (input) | reader RTA slot 0 (`src_buffer*` → address) | reader CTA offset 1 (`TensorAccessorArgs<1>()`) |
| `reduce_op_single_core_hw_program_factory.cpp:160` | `output` | writer RTA slot 0 | writer CTA offset 1 |
| `reduce_op_multi_core_w_program_factory.cpp:107` | `a` (input) | reader RTA slot 0 | reader CTA offset 1 |
| `reduce_op_multi_core_w_program_factory.cpp:110` | `output` | writer RTA slot 0 | writer CTA offset 1 |
| `reduce_op_multi_core_h_program_factory.cpp:283` | `a` (interleaved input) | reader RTA slot 0 | reader CTA offset 5 |
| `reduce_op_multi_core_h_program_factory.cpp:314` | `output` (interleaved) | writer RTA slot 0 | writer CTA offset 1 |
| H sharded path | (no `TensorAccessorArgs` — uses CB on borrowed memory + UnicastEndpoint walk) | — | — |
| `welford_reduce_program_factory.cpp:277` | `input` (H/HW variants) | reader RTA slot 0 | reader CTA offset 5 |
| `welford_reduce_program_factory.cpp:285` | `input` (W variant) | reader RTA slot 0 | reader CTA offset 1 |
| `welford_reduce_program_factory.cpp:311` | `output` (HW variant) | writer RTA slot 0 | writer CTA offset 6 |
| `welford_reduce_program_factory.cpp:320` | `output` (W/H variants) | writer RTA slot 0 | writer CTA offset 1 |

### Work split

- **Single-core HW factory:** `n/a — single core`. Optional `sub_core_grids` selects core position via `selected_core_coord`.
- **Multi-core W factory:** `split_work_to_cores(grid_or_sub_grid, NC * Ht)`. `num_rows_per_core_group_{1,2}` partitions rows of the tile grid.
- **Multi-core H factory (interleaved):** `split_work_to_cores(grid_or_sub_grid, NC * Wt)`. `num_cols_per_core_group_{1,2}` partitions columns.
- **Multi-core H factory (width-sharded):** Driver: shard spec from `a.shard_spec()`. `num_cores = all_cores.num_cores()`; single group (g2 empty).
- **Welford factory:** `split_work_to_cores(grid_or_sub_grid, num_work_units)` where `num_work_units = reduce_w ? NC*Ht : reduce_hw ? NC/reduce_batch_size : NC*Wt`.

### Cross-op kernels

- `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` — used by all `Reduce*` and Welford W/H variants. **Caution case** ([catalog](../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#caution-modifying-a-shared-dataflow-kernel)).
- `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp` — used by H factory's width-sharded path. Caution case.

Cross-op kernels are *referenced*, not modified, by the legacy factories. If the port can use them as-is via `TensorBinding`, no caution-case action is needed; if a port-time modification is required (CTA/RTA reshape), this becomes a fork-or-modify decision.

### Flags

- `reduce_op_device_operation.hpp` declares the program-factory variant alternatives as `std::variant<...>`. All three must be migrated together because they share a `program_factory_t` type alias.
- The compute kernels themselves are clean (no Device 2.0 holdovers), but several dataflow kernels have CB-index-keyed helpers (`get_tile_size(cb_id)`, `get_local_cb_interface(cb_id).fifo_page_size`). Per audit YELLOW + user preference: fold into port-time cleanup where it doesn't expand blast radius (i.e., op-local kernels only; defer the cross-op `writer_unary_interleaved_start_id.cpp` site).
- The H factory's width-sharded reader doesn't use `TensorAccessor` — it's downstream of the borrowed-memory CB. Re-evaluation happens after Phase C constructs the borrowed-memory DFB binding.

## Planned Spec Shape

*Filled in during Phase B.*

This section is deferred to Phase B and will be filled per factory:

- **`ReduceSingleCoreHwProgramFactory`** — 3 KernelSpecs (reader, writer, compute), 3–5 DFBs (gated by negate), 0 Semaphores, 2 TensorParameters (input, output), 1 WorkUnitSpec.
- **`ReduceMultiCoreWProgramFactory`** — 4 KernelSpecs (reader, writer, compute_g1, compute_g2?), 3–5 DFBs (gated by negate), 0 Semaphores, 2 TensorParameters, 1–2 WorkUnitSpecs (one per compute group).
- **`ReduceMultiCoreHProgramFactory`** — interleaved path: 4 KernelSpecs, 3–5 DFBs, 2 TensorParameters, 1–2 WorkUnitSpecs. Sharded path: 3 KernelSpecs (no g2), 4 DFBs (2 borrowed-memory), 2 TensorParameters, 1 WorkUnitSpec.
- **`WelfordReduceProgramFactory`** — variant-dispatching factory ([Multi-variant factory pattern](../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#pattern-multi-variant-factories)). W: 4 KernelSpecs, 4–5 DFBs (gated by `do_scale`), 2 TensorParameters, 1–2 WorkUnitSpecs. H: same shape as W minus scratch CBs. HW: 4 KernelSpecs, 5 DFBs, 2 TensorParameters, 1–2 WorkUnitSpecs.

## Preserved Multiplicity

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (multi-binding) |
|---|---|---|---|
| Reduce W: 2× `reduce[_w_neg]` compute (g1 + g2 with different `Ht` CTAs) | 2× same-source KernelSpec, differing only on the `Ht` CTA binding | 2× WorkUnitSpec (`wu_g1`, `wu_g2`) | INPUT_DFB, SCALER_DFB, OUTPUT_DFB shared (multi-CONSUMER/multi-PRODUCER), plus ACC/INEG when `negate` |
| Reduce H (interleaved): 2× `reduce[_h_neg]` compute (different `Wt` CTAs) | 2× same-source KernelSpec | 2× WorkUnitSpec | similar |
| Welford W: 2× `welford_reduce_w` compute (g1 + g2; CTAs identical, RTA differs) | **1× KernelSpec** acceptable (identical CTAs) — but to preserve the per-group `WorkUnitSpec` partition we may still want 2× | 2× WorkUnitSpec | INPUT, SCALER, OUTPUT |
| Welford H: 2× `welford_reduce_h` compute (identical CTAs) | same as W | 2× WorkUnitSpec | same |
| Welford HW: 2× `welford_reduce_hw` compute (identical CTAs) | same | 2× WorkUnitSpec | INPUT, SCALER, OUTPUT, PARTIAL, COMBINED |

**Note:** In the Welford variants, CTAs are identical across the two core groups (only the per-group RTA differs). Per the Welford-port fixup recipe (`0842e862431`) referenced in the orientation, even when CTAs are identical, the per-group KernelSpec/WorkUnitSpec split is preserved if any DFB binding is conditional per group. Whether we need 2× KernelSpec or 1× KernelSpec + per-node RTA via NodeRangeSet target is a Phase B decision — defaulting to 2× to mirror the legacy multiplicity.

For Reduce W/H, the compute CTAs differ per group, so 2× KernelSpec is non-negotiable (per [Anti-pattern: Demoting per-group CTA to RTA](../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta)).

## Dropped Plumbing

For each legacy RTA / CTA that disappears in the port:

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `reduce_op_single_core_hw_program_factory.cpp:208` reader RTA slot 0 | `a.buffer()` → buffer address | `TensorBinding(INPUT)` |
| `reduce_op_single_core_hw_program_factory.cpp:218` writer RTA slot 0 | `output.buffer()` → buffer address | `TensorBinding(OUTPUT)` |
| `reduce_op_single_core_hw_program_factory.cpp:130` reader CTA slot 1+ | `TensorAccessorArgs(src0_buffer)` | (subsumed by `TensorBinding`) |
| `reduce_op_single_core_hw_program_factory.cpp:160` writer CTA slot 1+ | `TensorAccessorArgs(dst_buffer)` | (subsumed by `TensorBinding`) |
| `reduce_op_single_core_hw_program_factory.cpp:159` writer CTA slot 0 | `output_cb_index` (=3) | `DFBBinding(OUTPUT_DFB, "out_dfb", CONSUMER)` |
| W, H, Welford analogous | analogous | analogous |
| All compute kernels' positional CTAs `{Ht, Wt, NC, ...}` | positional CTAs | named CTAs (`compile_time_arg_bindings = {{"Ht", Ht}, {"Wt", Wt}, ...}`) |
| Welford H/HW reader CTA `use_welford=1` (positional slot 4) | positional CTA | named CTA `{"use_welford", 1}` |
| Welford W reader (no Welford flag) | (single positional CTA `scaler_bits`) | named CTA `{"scaler_bits", scaler_bits}` |
| H sharded path: reader CTAs `{src0_cb_index, src1_cb_index, scaler_cb_index, scaler_bits}` | positional CTAs | DFB bindings (for the three CB indices) + named CTA `{"scaler_bits", scaler_bits}` |
| Welford HW writer CTAs `{Wt, W, tile_width, H, correction, reduce_batch_size}` | positional CTAs | named CTAs |
| Welford-HW reader: positional CTA slot 0 of reader (`num_pages`/`num_tiles`) | per-core RTA | named RTA `{"num_cols", ...}` |

## Applied Patterns

- [Pattern: Multi-variant factories](../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#pattern-multi-variant-factories) — Welford W/H/HW dispatched inside `create_program_spec`.
- [Anti-pattern: Demoting per-group CTA to RTA](../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta) — avoid by keeping 2× compute KernelSpecs in Reduce W / Reduce H (interleaved) where CTAs differ per group.
- [Pattern: Conditional / optional DFB bindings](../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#pattern-conditional--optional-dfb-bindings) — ACC_DFB / INEG_DFB on Reduce* when `negate`; SCALED_DFB on Welford W when `do_scale`.
- [Pattern: Pass DFB handles directly to LLKs and kernel-lib helpers](../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers) — Use `dfb::input` etc. in `compute_kernel_lib::reduce<>` and `dataflow_kernel_lib::prepare_reduce_scaler<>` calls.
- [Pattern: Unity-build hygiene for anonymous-namespace symbols](../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#pattern-unity-build-hygiene-for-anonymous-namespace-symbols) — the four reduction factories likely live in one Unity build TU. Helper-hoisting to a shared header (per the Borys reference) is appropriate.
- [Caution: Modifying a shared dataflow kernel](../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/metal2_port_patterns.md#caution-modifying-a-shared-dataflow-kernel) — both `writer_unary_interleaved_start_id.cpp` (eltwise/unary) and `writer_unary_sharded.cpp` (data_movement/sharded) are cross-op consumers. **Decision:** fork with a `_metal2`-suffixed copy in the reduction op directory, since other consumers (eltwise/unary, data_movement) remain on legacy.

## Deferred / Flagged

- **YELLOW (resolved with user override): borrowed-memory DFB in H sharded path.** Use `DataflowBufferSpec::borrowed_from = INPUT` / `borrowed_from = OUTPUT` for the two sharded DFBs. See `dataflow_buffer_spec.hpp:95` for the API.
- **YELLOW (resolved with user override): Device 2.0 DM holdovers.** Fold the five op-local kernel sites into port-time cleanup; defer the shared `writer_unary_interleaved_start_id.cpp` site by forking the kernel.
- **TensorAccessor for sharded reader.** The H sharded reader will need redesign as part of the borrowed-memory-DFB binding — the kernel side will receive the DFB handle (`dfb::input` for the borrowed DFB) and need NOT thread the buffer address through an RTA. The UnicastEndpoint loop walks the DFB's borrowed L1 region, not tensor memory.
- **Compute kernels' `compute_kernel_hw_startup(c_0, c_2, c_3)` call.** Once CB indices become DFB handles, this call site must take `dfb::input`, `dfb::scaler`, `dfb::output` — the implicit conversion handles it (per the catalog pattern, PR #44646).
- **`compute_kernel_lib::reduce<...>` template parameters.** The template's CT params are `REDUCE_OP`, `REDUCE_DIM`, an `InputPolicy`, and a `ReconfigMode` — they remain as is. The CB-index runtime args (passed as `tt::CBIndex::c_0`, etc.) become `dfb::input` etc. by implicit conversion.
- **New finding (Phase B may revisit): per-group KernelSpec for identical-CTA Welford variants.** The Welford W/H/HW compute kernels have identical CTAs across core groups (only RTA `num_work_units_per_core` differs). Two options:
  1. 2× same-source KernelSpec, each with one RTA per core (mirrors legacy multiplicity).
  2. 1× KernelSpec with `target_nodes = all_cores`, with per-node RTAs supplying the per-core count.
  Option (1) is the safe default (matches the [Pattern: Self-loop DFB binding](#) preserved-multiplicity guidance). Option (2) is more compact but requires that all per-node RTAs land in a single `KernelRunParams` entry. Default to Option (1) for now.
