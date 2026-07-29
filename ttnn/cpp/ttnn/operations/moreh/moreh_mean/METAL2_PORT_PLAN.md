# Port Plan — `moreh_mean`

Port plan for `ttnn/cpp/ttnn/operations/moreh/moreh_mean`, ported from the legacy
`ProgramDescriptor` API to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

**Scope of this pass:** all three factories — `MorehMeanHFactory`, `MorehMeanWFactory`,
`MorehMeanNCFactory` — plus all 8 kernel entry points they bind. The invoker assigned the bundled
three-factory port; that is what makes the intra-op shared writer (below) convertible in place.

---

## Legacy Inventory

### Legacy factory shape

- Concept: `ProgramDescriptorFactoryConcept` — each factory is a
  `static tt::tt_metal::ProgramDescriptor create_descriptor(...)` on `MorehMeanOperation`
  (`device/moreh_mean_device_operation.hpp:34-53`).
- Variants: three, in `program_factory_t = std::variant<MorehMeanHFactory, MorehMeanNCFactory,
  MorehMeanWFactory>` (`moreh_mean_device_operation.hpp:55`), selected by reduced-dim position in
  `select_program_factory` (`moreh_mean_device_operation.cpp:34-47`). Exactly one runs per invocation.
- Custom `compute_program_hash`: **none** — already the default reflection-based hash. The device-op
  declares only `validate_tensors` / `select_program_factory` / `validate_on_program_cache_miss` /
  `compute_output_specs` / `create_output_tensors` (`moreh_mean_device_operation.hpp:57-61`).

*(The Metal 2.0 factory concept the port targets was chosen during the audit — see the brief's
TTNN factory analysis section. Carried forward in the [TTNN ProgramFactory](#ttnn-programfactory)
section below.)*

### Variant: H (`MorehMeanHFactory`, `moreh_mean_h_program_factory.cpp`)

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | config |
|---|---|---|---|---|---|---|---|---|
| reader | `kernels/reader_moreh_mean_h.cpp` | `all_cores` | `{Ht, Wt, HtWt}` + `TensorAccessorArgs(*input.buffer())` + `{origin_H}` (`:118-120`) | none | per node: `{input_buf(Buffer*), (tile_offset/Wt*HtWt)+(tile_offset%Wt), tile_offset%Wt, units_per_core, mask_h}` (`:212-214`) | none | `REDUCE_SCALER=1`; `DO_MASK_H=1` iff `origin_H % 32 != 0` (`:122-125`) | `ReaderConfigDescriptor{}` |
| writer | `kernels/writer_moreh_mean_unary_interleaved_start_id.cpp` | `all_cores` | `{CBIndex::c_16}` + `TensorAccessorArgs(*output.buffer())` (`:136-137`) | none | per node: `{output_buf(Buffer*), units_per_core, tile_offset}` (`:216`) | none | none | `WriterConfigDescriptor{}` |
| compute_1 | `kernels/moreh_mean_h.cpp` | `core_group_1` | `{Ht, units_per_core_group_1, 1, origin_H}` (`:162-167`) | none | none | none | `reduce_op_utils::get_defines(AVG, H)` → `REDUCE_OP`,`REDUCE_DIM`; `FP32_DEST_ACC_EN=1` iff fp32 (`:150-156`) | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en, dst_full_sync_en, unpack_to_dest_mode, math_approx_mode}` (`:169-175`) |
| compute_2 | `kernels/moreh_mean_h.cpp` | `core_group_2` (only if non-empty) | `{Ht, units_per_core_group_2, 1, origin_H}` (`:183-188`) | none | none | none | same as compute_1 | same as compute_1 |

`unpack_to_dest_mode` is `vector<UnpackToDestMode>(NUM_CIRCULAR_BUFFERS, Default)` with
`[CBIndex::c_24] = UnpackToDestFp32` **iff** `fp32_dest_acc_en` (`:151-155`).

#### CBs (`moreh_mean_h_program_factory.cpp:62-115`)

| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| `c_0` | `2 * tile_size(data_format)` | `all_cores` | `data_format` | `tile_size(data_format)` | unset |
| `c_2` | `tile_size(data_format)` | `all_cores` | `data_format` | `tile_size(data_format)` | unset |
| `c_3` | `tile_size(data_format)` | `all_cores` | `data_format` | `tile_size(data_format)` | unset |
| `c_24` | `tile_size(fp32_dest_acc_en_data_format)` | `all_cores` | `fp32_dest_acc_en_data_format` | `tile_size(fp32_dest_acc_en_data_format)` | unset |
| `c_25` | `tile_size(data_format)` | `all_cores` | `data_format` | `tile_size(data_format)` | unset |
| `c_16` | `tile_size(data_format)` | `all_cores` | `data_format` | `tile_size(data_format)` | unset |

`data_format = datatype_to_dataformat_converter(input.dtype())` (BFLOAT16-only op);
`fp32_dest_acc_en_data_format = fp32_dest_acc_en ? Float32 : data_format`.
No `.global_circular_buffer`, no `.address_offset`, no `.buffer` (borrowed memory), single-element
`format_descriptors` throughout — i.e. no GlobalCB, no aliasing, no borrowed-memory DFB.

#### Semaphores

none — the op declares no semaphores of any kind.

#### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `moreh_mean_h_program_factory.cpp:119` (`TensorAccessorArgs(*input.buffer()).append_to(...)`) | `tensor_args.input` | reader RTA 0 (`:214`, `Buffer*`) |
| `moreh_mean_h_program_factory.cpp:137` | `output` (`tensor_return_value`) | writer RTA 0 (`:216`, `Buffer*`) |

Kernel-side construction: `reader_moreh_mean_h.cpp:28,46`; `writer_moreh_mean_unary_interleaved_start_id.cpp:19,20`.

#### Work split

- Driver: `split_work_to_cores_wt_core_range(core_range, units_to_divide)` (`:49-50`), with
  `units_to_divide = input.physical_volume() / W / H * Wt` and
  `core_range = ({0,0},{grid.x-1, grid.y-1})`.
- `num_cores`, `all_cores`, `core_group_1`, `core_group_2`, `units_per_core_group_1`,
  `units_per_core_group_2`. `all_cores == core_group_1 ∪ core_group_2`
  (`tt_metal/common/work_split.cpp:339-...`).
- Per-node RTA loop walks `CoreCoord{i / core_h, i % core_h}` for `i < num_cores` (`:202-219`).

### Variant: W (`MorehMeanWFactory`, `moreh_mean_w_program_factory.cpp`)

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | config |
|---|---|---|---|---|---|---|---|---|
| reader | `kernels/reader_moreh_mean_w.cpp` | `all_cores` | `TensorAccessorArgs(*input.buffer())` + `{packed_scaler_value}` (`:122-124`) | none | per node: `{input_buf(Buffer*), num_tensor_tiles_per_core, tile_offset, mask_w}` (`:219`) | none | `DO_MASK_W=1` iff `origin_W % 32 != 0` (`:126-129`) | `ReaderConfigDescriptor{}` |
| writer | `kernels/writer_moreh_mean_unary_interleaved_start_id.cpp` | `all_cores` | `{CBIndex::c_16}` + `TensorAccessorArgs(*output.buffer())` (`:140-141`) | none | per node: `{output_buf(Buffer*), num_tensor_tiles_per_core / Wt, tile_offset / Wt}` (`:221-222`) | none | none | `WriterConfigDescriptor{}` |
| compute_1 | `kernels/moreh_mean_w.cpp` | `core_group_1` | `{units_per_core_group_1, Wt, 1, origin_W}` (`:166-171`) | none | none | none | `reduce_op_utils::get_defines(AVG, W)`; `FP32_DEST_ACC_EN=1` iff fp32 (`:154-158`) | `ComputeConfigDescriptor{...}` (`:173-179`) |
| compute_2 | `kernels/moreh_mean_w.cpp` | `core_group_2` (only if non-empty) | `{units_per_core_group_2, Wt, 1, origin_W}` (`:187-192`) | none | none | none | same | same |

`unpack_to_dest_mode` is left **entirely `Default`** here (`:160`) — unlike H. Preserved verbatim
(see Applied Patterns / the `unpack_modes` note).

#### CBs (`moreh_mean_w_program_factory.cpp:62-115`)

Identical shape and sizes to the H factory's six CBs (`c_0` ×2 entries, `c_2`/`c_3`/`c_24`/`c_25`/`c_16`
×1 entry), with `c_24` at `fp32_dest_acc_en_data_format`.

#### Semaphores

none.

#### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `moreh_mean_w_program_factory.cpp:123` | `tensor_args.input` | reader RTA 0 (`:219`) |
| `moreh_mean_w_program_factory.cpp:141` | `output` | writer RTA 0 (`:221-222`) |

Kernel-side: `reader_moreh_mean_w.cpp:16,34`; `writer_moreh_mean_unary_interleaved_start_id.cpp:19,20`.

#### Work split

Same driver as H, with `units_to_divide = input.physical_volume() / W / H * Ht` (`:48-50`);
`num_tensor_tiles_per_core = units_per_core * Wt`, `out_dim_divider = Wt` (`:204,217`).

### Variant: NC (`MorehMeanNCFactory`, `moreh_mean_nc_program_factory.cpp`)

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | config |
|---|---|---|---|---|---|---|---|---|
| reader | `kernels/reader_moreh_mean_nc.cpp` | `all_cores` | `TensorAccessorArgs(*input.buffer())` only (`:115-116`) | none | per node: `{input_buf(Buffer*), num_reduce_input_tile, units_per_core, input_tile_stride, tile_offset, HtWt, inner_size}` (`:191-199`) | none | none | `ReaderConfigDescriptor{}` |
| writer | `kernels/writer_moreh_mean_nc.cpp` | `all_cores` | `TensorAccessorArgs(*output.buffer())` only (`:126-127`) | none | per node: `{output_buf(Buffer*), units_per_core, tile_offset}` (`:201`) | none | none | `WriterConfigDescriptor{}` |
| compute_1 | `kernels/moreh_mean_nc.cpp` | `core_group_1` | `{units_per_core_group_1}` (`:147`) | none | per node (cg1 only): `{num_reduce_input_tile, units_per_core}` (`:183`) | none | `FP32_DEST_ACC_EN=1` iff fp32 (`:137-140`) | `ComputeConfigDescriptor{...}` (`:149-155`) |
| compute_2 | `kernels/moreh_mean_nc.cpp` | `core_group_2` (only if non-empty) | `{units_per_core_group_2}` (`:163`) | none | per node (cg2 only): `{num_reduce_input_tile, units_per_core}` (`:186`) | none | same | same |

`unpack_to_dest_mode` all `Default` (`:141`).

#### CBs (`moreh_mean_nc_program_factory.cpp:68-112`)

| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| `c_0` | `2 * tile_size(cb_data_format)` | `all_cores` | `cb_data_format` | `tile_size(cb_data_format)` | unset |
| `c_1` | `tile_size(cb_data_format)` | `all_cores` | `cb_data_format` | `tile_size(cb_data_format)` | unset |
| `c_2` | `tile_size(cb_data_format)` | `all_cores` | `cb_data_format` | `tile_size(cb_data_format)` | unset |
| `c_24` | `tile_size(cb_data_format)` | `all_cores` | `cb_data_format` | `tile_size(cb_data_format)` | unset |
| `c_16` | `2 * tile_size(cb_data_format)` | `all_cores` | `cb_data_format` | `tile_size(cb_data_format)` | unset |

`cb_data_format = datatype_to_dataformat_converter(output.dtype())` — **not** widened for fp32
(`c_24` stays at `cb_data_format`; this is audit anomaly 5, preserved verbatim).

#### Semaphores

none.

#### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `moreh_mean_nc_program_factory.cpp:116` | `tensor_args.input` | reader RTA 0 (`:191-199`) |
| `moreh_mean_nc_program_factory.cpp:127` | `output` | writer RTA 0 (`:201`) |

Kernel-side: `reader_moreh_mean_nc.cpp:38,39`; `writer_moreh_mean_nc.cpp:20,21`.

#### Work split

Same driver, `units_to_divide = output.physical_volume() / TILE_HW` (`:55-60`).

### Shared kernels

| kernel source | kind | consumers | `_metal2` fork present? | rung |
|---|---|---|---|---|
| `kernels/writer_moreh_mean_unary_interleaved_start_id.cpp` | **intra-op** | `MorehMeanHFactory`, `MorehMeanWFactory` | no | **rung 3 — convert in place** |

Census run per the [shared-kernel Caution](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-porting-a-shared-kernel):
`grep -rl <kernel-filename> ttnn/` over all 8 kernel files returns only (a) this op's own factories,
(b) the two `METAL2_*.md` audit artifacts, and (c) `ttnn/ttnn.egg-info/SOURCES.txt` (a packaging
manifest, not a consumer). **No cross-op sharing (borrowed or lent) anywhere.**

Rung 3 applies because the invoker explicitly assigned the bundled port of **all three** factories,
so both consumers of the shared writer convert in the same change. Post-port both bind the same
Metal 2.0 writer with an identical schema (`dfb::out`, `tensor::dst`, RTAs `num_tiles` / `start_id`),
so no fork and no sunset list. The other 7 kernels have a single consumer each.

### Flags

- **No unreferenced kernel files** — all 8 are instantiated.
- **Dead CTA, H reader:** `HtWt` (CTA slot 2, `reader_moreh_mean_h.cpp:21`) is read into a local the
  kernel body never uses. Preserved verbatim as a named CTA (`args::HtWt`); reported, not fixed.
- **Dead CTA, NC compute:** `units_per_core_group_N` (`moreh_mean_nc_program_factory.cpp:147,163`)
  is emitted but `moreh_mean_nc.cpp` reads **no** compile-time arg at all. Preserved verbatim as a
  named CTA (it is the per-group CTA that distinguishes the two `KernelDescriptor`s — dropping it
  would collapse the preserved multiplicity); reported, not fixed.
- **Misnamed compute CTAs** (audit anomaly 2 / brief *Watch for*): H compute CTA(1) and W compute
  CTA(0) are read into kernel locals named `Wt` / `Ht` but actually carry `units_per_core_group_N`.
  The **binding** is named `units_per_core` in both; the kernel **locals** keep their legacy names
  (renaming them is out of scope).
- **`fp32_dest_acc_en` asymmetry across the three factories** (audit anomaly 5) — preserved
  verbatim; see Deferred / Flagged.
- **Dead include:** `reader_moreh_mean_nc.cpp:5` includes `api/debug/dprint.h` with no `DPRINT` in
  the file (audit anomaly 6). Left in place — out of scope.

---

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept` — all three factories.
- **Custom `compute_program_hash`**: none (already the default reflection-based hash). No deletion.
- **Implementation notes**:
  - The three factories stay nested inside `MorehMeanOperation`; only the entry-point signature
    changes (`create_descriptor` → `create_program_artifacts`, returning
    `ttnn::device_operation::ProgramArtifacts`). `program_factory_t` is unchanged, and all three
    alternatives satisfy exactly one concept, so `AllFactoriesValid` holds.
  - `op_owned_tensors` is left default-empty (audit: no op-owned tensors).
  - No pybind entry point to remove: `moreh_mean_nanobind.cpp:19-31` binds only
    `&ttnn::moreh_mean`; `create_descriptor` was never pybound.
  - **Unity-build hygiene** ([catalog](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-unity-build-hygiene-for-anonymous-namespace-symbols)):
    `ttnncpp` is a unity-build target and the three factory `.cpp`s share names. All typed name
    constants (`KernelSpecName` / `DFBSpecName` / `TensorParamName`) are declared **function-local**
    inside each `create_program_artifacts`, so no anonymous-namespace symbols are introduced and no
    per-factory prefixing is needed.
  - Tensor extraction: the factory bodies keep their existing `ttnn::Tensor` accessors for geometry /
    dtype / device (unchanged lines) and use `tensor.tensor_spec()` for the `TensorParameter` and
    `tensor.mesh_tensor()` for the `TensorArgument`. Rewriting the whole body onto `MeshTensor`
    would restructure code the port does not otherwise touch (`MeshTensor::device()` returns a
    reference, not the `IDevice*` the existing lines use), which scope discipline rules out.

---

## Planned Spec Shape

Default: 1:1 with legacy, per variant.

### Variant: H

- **KernelSpecs** (4, or 3 when `core_group_2` is empty):
  `READER{"reader"}`, `WRITER{"writer"}`, `COMPUTE_G1{"compute_g1"}`, `COMPUTE_G2{"compute_g2"}`.
- **DataflowBufferSpecs** (6, one per legacy `CBDescriptor`; `entry_size` = legacy `page_size`,
  `num_entries` = `total_size / page_size`, `data_format_metadata` = legacy `data_format`,
  `tile_format_metadata` left `nullopt` because the legacy `.tile` was unset):

  | DFB name | legacy CB | entry_size | num_entries | data_format_metadata |
  |---|---|---|---|---|
  | `INPUT{"input"}` | `c_0` | `tile_size(data_format)` | 2 | `data_format` |
  | `SCALER{"scaler"}` | `c_2` | `tile_size(data_format)` | 1 | `data_format` |
  | `MASK_H{"mask_h"}` | `c_3` | `tile_size(data_format)` | 1 | `data_format` |
  | `ACCUM_DST{"accum_dst"}` | `c_24` | `tile_size(fp32_fmt)` | 1 | `fp32_fmt` |
  | `MASKED_INPUT{"masked_input"}` | `c_25` | `tile_size(data_format)` | 1 | `data_format` |
  | `OUT{"out"}` | `c_16` | `tile_size(data_format)` | 1 | `data_format` |

- **SemaphoreSpecs**: none — no legacy `SemaphoreDescriptor`.
- **TensorParameters** (2): `INPUT_T{"input"}` ← `tensor_args.input`, `OUTPUT_T{"output"}` ← `output`.
  Both strict (no `advanced_options` relaxation — the op has no `ArgConfig::Runtime*` use and no
  custom hash; the readiness sheet's `TensorParameter relaxation` cell resolves to `none`).
- **WorkUnitSpecs** (2, or 1 when `core_group_2` is empty):
  `wu_g1{READER, WRITER, COMPUTE_G1} → core_group_1`,
  `wu_g2{READER, WRITER, COMPUTE_G2} → core_group_2`.
  Reader/writer belong to both, so their derived node set is `core_group_1 ∪ core_group_2 == all_cores`
  — the legacy `core_ranges`.
- **Op-owned tensors**: none.

#### DFB endpoint assignment (re-derived from the kernel-touch census, per node)

| DFB | reader | writer | compute | census | disposition |
|---|---|---|---|---|---|
| `INPUT` | P (`reader_moreh_mean_h.cpp:57,60`) | — | C (`reduce<input,…>`; `:61,78`) | 1P+1C | plain 1:1 |
| `SCALER` | P (`:33-37` → `reduce_helpers_dataflow.inl`) | — | C (`moreh_mean_h.cpp:35,100`) | 1P+1C | plain 1:1 |
| `MASK_H` | P **iff `DO_MASK_H`** (`:41-44`) | — | C always; **P iff `!do_mask_h`** | masked: 1P+1C · unmasked: 1 toucher | 1:1 · **self-loop** |
| `ACCUM_DST` | — | — | P (`:54`) + C (`:84,92`) | 1 toucher | **self-loop** |
| `MASKED_INPUT` | — | — | P (`:72,76`) + C (`:81`) | 1 toucher | **self-loop** |
| `OUT` | — | C (`writer…:29,33`) | P (`:81,89`) | 1P+1C | plain 1:1 |

Census **agrees with the brief** on all six. `MASK_H` is *not* a dead CB in the unmasked config: the
compute kernel still binds/constructs it, and the reader genuinely fills it whenever
`origin_H % 32 != 0`. The two same-source compute `KernelSpec`s cover **disjoint** node sets, so each
node hosts exactly one compute instance — no multi-binding flag anywhere in this port.

#### Kernel bindings and argument schema (H)

| KernelSpec | dfb_bindings | tensor_bindings | named CTAs | named RTAs | hw_config |
|---|---|---|---|---|---|
| `READER` | `INPUT`→`"input"` P; `SCALER`→`"scaler"` P; *(iff `do_mask_h`)* `MASK_H`→`"mask_h"` P | `INPUT_T`→`"src"` | `Ht`, `Wt`, `HtWt`, `reduce_factor` | `col_start_tile_id`, `curr_col_in_batch`, `num_cols`, `mask_h` | `ttnn::create_reader_datamovement_config(arch)` |
| `WRITER` | `OUT`→`"out"` C | `OUTPUT_T`→`"dst"` | none | `num_tiles`, `start_id` | `ttnn::create_writer_datamovement_config(arch)` |
| `COMPUTE_G1/2` | `INPUT`→`"input"` C; `SCALER`→`"scaler"` C; `MASK_H`→`"mask_h"` C **+ P iff `!do_mask_h`**; `ACCUM_DST`→`"accum_dst"` P+C; `MASKED_INPUT`→`"masked_input"` P+C; `OUT`→`"out"` P | none | `Ht`, `units_per_core`, `NC`, `origin_H` | none | `ttnn::to_compute_hardware_config(arch, compute_kernel_config)` + `unpack_modes` (below) |

### Variant: W

- **KernelSpecs**: same four names/roles as H, with `moreh_mean_w.cpp` compute and
  `reader_moreh_mean_w.cpp` reader; the writer is the **same source** as H's.
- **DataflowBufferSpecs** (6): identical to H except `MASK_H` → `MASK_W{"mask_w"}` (`c_3`).
- **SemaphoreSpecs**: none. **TensorParameters** (2): as H. **Op-owned tensors**: none.
- **WorkUnitSpecs**: as H (`wu_g1` / `wu_g2`).

#### DFB endpoint assignment (W)

| DFB | reader | writer | compute | census | disposition |
|---|---|---|---|---|---|
| `INPUT` | P (`reader_moreh_mean_w.cpp:41,44`) | — | C (`moreh_mean_w.cpp:57,63,76,94,99,120`) | 1P+1C | plain 1:1 |
| `SCALER` | P (`:21` → `generate_mm_scaler`) | — | C (`:36,130`) | 1P+1C | plain 1:1 |
| `MASK_W` | P **iff `DO_MASK_W`** (`:25-27`) | — | C always; **P iff `!do_mask_w`** | masked: 1P+1C · unmasked: 1 toucher | 1:1 · **self-loop** |
| `ACCUM_DST` | — | — | P (`:67,71`) + C (`:101,122`) | 1 toucher | **self-loop** |
| `MASKED_INPUT` | — | — | P (`:88,92`) + C (`:99,120`, via the `cb_input` reassign at `:95`) | 1 toucher | **self-loop** |
| `OUT` | — | C (`writer…:29,33`) | P (`:114,118`) | 1P+1C | plain 1:1 |

#### Kernel bindings and argument schema (W)

| KernelSpec | dfb_bindings | tensor_bindings | named CTAs | named RTAs | hw_config |
|---|---|---|---|---|---|
| `READER` | `INPUT`→`"input"` P; `SCALER`→`"scaler"` P; *(iff `do_mask_w`)* `MASK_W`→`"mask_w"` P | `INPUT_T`→`"src"` | `scaler` | `num_tiles`, `start_id`, `mask_w` | reader default |
| `WRITER` | `OUT`→`"out"` C | `OUTPUT_T`→`"dst"` | none | `num_tiles`, `start_id` | writer default |
| `COMPUTE_G1/2` | `INPUT`→`"input"` C; `SCALER`→`"scaler"` C; `MASK_W`→`"mask_w"` C **+ P iff `!do_mask_w`**; `ACCUM_DST`→`"accum_dst"` P+C; `MASKED_INPUT`→`"masked_input"` P+C; `OUT`→`"out"` P | none | `units_per_core`, `Wt`, `NC`, `origin_W` | none | compute (below) |

### Variant: NC

- **KernelSpecs**: `READER`, `WRITER`, `COMPUTE_G1`, `COMPUTE_G2` (own reader/writer/compute sources).
- **DataflowBufferSpecs** (5):

  | DFB name | legacy CB | entry_size | num_entries | data_format_metadata |
  |---|---|---|---|---|
  | `INPUT{"input"}` | `c_0` | `tile_size(cb_data_format)` | 2 | `cb_data_format` |
  | `IN1{"in1"}` | `c_1` | `tile_size(cb_data_format)` | 1 | `cb_data_format` |
  | `SCALAR{"scalar"}` | `c_2` | `tile_size(cb_data_format)` | 1 | `cb_data_format` |
  | `INTERMED0{"intermed0"}` | `c_24` | `tile_size(cb_data_format)` | 1 | `cb_data_format` |
  | `OUT{"out"}` | `c_16` | `tile_size(cb_data_format)` | 2 | `cb_data_format` |

- **SemaphoreSpecs**: none. **TensorParameters** (2): as H. **Op-owned tensors**: none.
- **WorkUnitSpecs**: as H (`wu_g1` / `wu_g2`).

#### DFB endpoint assignment (NC)

| DFB | reader | writer | compute | census | disposition |
|---|---|---|---|---|---|
| `INPUT` | P (`reader_moreh_mean_nc.cpp:52,55`) | — | C (`moreh_mean_nc.cpp:43,52`) | 1P+1C | plain 1:1 |
| `IN1` | P (`:31-32` → `fill_cb_with_value`) | — | C (`moreh_mean_nc.cpp:34` `wait_front`, never popped) | 1P+1C | plain 1:1 |
| `SCALAR` | P (`:35-36`) | — | C (`:35` `wait_front`, never popped) | 1P+1C | plain 1:1 |
| `INTERMED0` | — | — | P (`:58,62`) + C (`:45,54,69,79`) | 1 toucher | **self-loop** |
| `OUT` | — | C (`writer_moreh_mean_nc.cpp:29,33`) | P (`:74,78`) | 1P+1C | plain 1:1 |

`IN1` / `SCALAR` are `wait_front`-ed but never `pop_front`-ed (audit anomaly 8) — deliberate
fill-once/read-many reuse. Per the recipe's stop-signal list this is **not** an unbalanced FIFO to
"fix" by adding a `pop`; the kernel is untouched.

#### Kernel bindings and argument schema (NC)

| KernelSpec | dfb_bindings | tensor_bindings | named CTAs | named RTAs | hw_config |
|---|---|---|---|---|---|
| `READER` | `INPUT`→`"input"` P; `IN1`→`"in1"` P; `SCALAR`→`"scalar"` P | `INPUT_T`→`"input"` | none | `num_input_tiles`, `num_output_tiles`, `input_tile_stride`, `start_id`, `HtWt`, `inner_size` | reader default |
| `WRITER` | `OUT`→`"out"` C | `OUTPUT_T`→`"output"` | none | `num_tiles`, `start_id` | writer default |
| `COMPUTE_G1/2` | `INPUT`→`"input"` C; `IN1`→`"in1"` C; `SCALAR`→`"scalar"` C; `INTERMED0`→`"intermed0"` P+C; `OUT`→`"out"` P | none | `units_per_core` | `num_input_tiles`, `num_output_tiles` | compute (below) |

### Hardware configuration (all three variants)

- **DM kernels.** Legacy uses `ReaderConfigDescriptor{}` / `WriterConfigDescriptor{}` — i.e. the
  resolved reader/writer defaults `(RISCV_1, NOC_0, DM_DEDICATED_NOC)` and
  `(RISCV_0, NOC_1, DM_DEDICATED_NOC)`. Port to the arch-agnostic TTNN helpers
  `ttnn::create_reader_datamovement_config(arch)` / `create_writer_datamovement_config(arch)`.
  No custom triple anywhere, no `DM_DYNAMIC_NOC`.
- **Compute kernels — Style A.** All three factories resolve a TTNN `ComputeKernelConfig` via
  `init_device_compute_kernel_config` and read it back with `get_compute_kernel_config_args`
  (a pure passthrough, `compute_kernel_config.cpp:97-106`). Port with
  `ttnn::to_compute_hardware_config(device->arch(), compute_kernel_config)`, which carries
  `math_fidelity → fpu_math_fidelity`, `math_approx_mode → sfpu_precision_mode`,
  `fp32_dest_acc_en → enable_32_bit_dest`, and `dst_full_sync_en → !double_buffer_dest`.
  `packer_l1_acc` is destructured but never read by any of the three factories (audit anomaly 4) —
  it is not part of the Metal 2.0 compute config either, so nothing is lost.
- **`bfp_pack_precision_mode`**: legacy never sets `bfp8_pack_precise` → leave at its default
  (`Precision::Approximate`), which matches.
- **`unpack_modes`** (the reindexed + value-translated legacy `unpack_to_dest_mode`):

  | factory | legacy vector | ported `unpack_modes` | why |
  |---|---|---|---|
  | H | `[c_24] = UnpackToDestFp32` iff fp32, else all `Default` | `{{ACCUM_DST, UnpackMode::UnpackToDest}}` iff `fp32_dest_acc_en`, else empty | 1:1 value translation of the one non-default entry |
  | W | all `Default` | `{{ACCUM_DST, UnpackMode::UnpackToSrc}}` iff `fp32_dest_acc_en`, else empty | **newly-required explicit entry**: the compute kernel *consumes* `ACCUM_DST`, which is `Float32` when fp32, with `enable_32_bit_dest = true`. Value derived from the legacy `Default` → `UnpackToSrc` (`program_spec.cpp:1051-1070`) |
  | NC | all `Default` | empty | `c_24` is never widened to `Float32` here, so no DFB the compute kernel consumes is FP32 — the required-entry rule does not fire and legacy `Default` == omitted entry |

  No other DFB in any factory is `Float32` (the op is BFLOAT16-only, `moreh_mean_device_operation.cpp:25-26`).
- **Gen2**: not populated; no `arch == QUASAR` branch added. The two TTNN helpers supply the Gen2
  alternative for free in the default cases.

---

## Preserved Multiplicity

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| H: `compute_desc_1` (`core_group_1`) + `compute_desc_2` (`core_group_2`) of `moreh_mean_h.cpp`, differing only in the `units_per_core_group_N` CTA | `COMPUTE_G1`, `COMPUTE_G2` — same source, same bindings, CTA `units_per_core` differs | `wu_g1` (`core_group_1`), `wu_g2` (`core_group_2`) | `INPUT` C, `SCALER` C, `MASK_H` C (+P iff unmasked), `ACCUM_DST` P+C, `MASKED_INPUT` P+C, `OUT` P — each bound once per role by each KernelSpec |
| W: same shape, `moreh_mean_w.cpp` | `COMPUTE_G1`, `COMPUTE_G2` | `wu_g1`, `wu_g2` | as H, with `MASK_W` in place of `MASK_H` |
| NC: same shape, `moreh_mean_nc.cpp` | `COMPUTE_G1`, `COMPUTE_G2` | `wu_g1`, `wu_g2` | `INPUT` C, `IN1` C, `SCALAR` C, `INTERMED0` P+C, `OUT` P |

The two compute `KernelSpec`s cover **disjoint** node sets, so each node sees exactly one instance —
each shared DFB is an ordinary per-node 1:1 (or self-loop) and the `allow_instance_multi_binding`
flag is **not** involved. This is the
[disjoint-node work-split](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta),
not the same-grid two-toucher case. `COMPUTE_G2` / `wu_g2` are emitted only when `core_group_2` is
non-empty, mirroring the legacy `has_core_group_2` guard.

No per-group CTA is demoted to an RTA.

---

## Dropped Plumbing

### H factory

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `..._h_program_factory.cpp:119` | `TensorAccessorArgs(*input.buffer()).append_to(reader_compile_time_args)` | `TensorParameter INPUT_T` + reader `TensorBinding{INPUT_T, "src"}` |
| `..._h_program_factory.cpp:137` | `TensorAccessorArgs(*output.buffer()).append_to(writer_compile_time_args)` | `TensorParameter OUTPUT_T` + writer `TensorBinding{OUTPUT_T, "dst"}` |
| `..._h_program_factory.cpp:214` (reader RTA slot 0) | `input_buf` (`Buffer*`) | `TensorBinding` (auto-injected base address) |
| `..._h_program_factory.cpp:216` (writer RTA slot 0) | `output_buf` (`Buffer*`) | `TensorBinding` |
| `..._h_program_factory.cpp:136` (writer CTA slot 0) | `static_cast<uint32_t>(CBIndex::c_16)` | `DFBBinding{OUT, "out", CONSUMER}` |
| `..._h_program_factory.cpp:118,162,183` | positional `compile_time_args` vectors | named CTAs (`Ht`,`Wt`,`HtWt`,`reduce_factor`; `Ht`,`units_per_core`,`NC`,`origin_H`) |
| `..._h_program_factory.cpp:151-155,173,194` | `std::vector<UnpackToDestMode>` indexed by CB id | `ComputeGen1Config::unpack_modes` keyed by `DFBSpecName` |
| `reader_moreh_mean_h.cpp:12` | `src_addr = get_arg_val<uint32_t>(0)` | dropped; `TensorAccessor(tensor::src)` |
| `reader_moreh_mean_h.cpp:23,31,40` | `constexpr uint32_t cb_id_in0/in2/mask_h = tt::CBIndex::c_*` | `dfb::input` / `dfb::scaler` / `dfb::mask_h` |
| `reader_moreh_mean_h.cpp:28,32,46` | `TensorAccessorArgs<3>()`, `next_compile_time_args_offset()`, `TensorAccessor(src_args, src_addr)` | `TensorAccessor(tensor::src)`; `reduce_factor` becomes a plain named CTA |
| `reader_moreh_mean_h.cpp:50` | `get_tile_size(cb_id_in0)` | `dfb_in0.get_tile_size()` |
| `writer_..._start_id.cpp:11,15,19,20` | `dst_addr` RTA, `cb_id_out` CTA, `TensorAccessorArgs<1>()`, `TensorAccessor(dst_args, dst_addr)` | `dfb::out`, `TensorAccessor(tensor::dst)` |
| `writer_..._start_id.cpp:24` | `get_tile_size(cb_id_out)` | `dfb_out.get_tile_size()` |
| `moreh_mean_h.cpp:21,23,25,27,28,30` | `constexpr auto cb_* = tt::CBIndex::c_*` | `dfb::input`/`scaler`/`mask_h`/`accum_dst`/`masked_input`/`out` |

### W factory

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `..._w_program_factory.cpp:123` | `TensorAccessorArgs(*input.buffer()).append_to(...)` | `TensorBinding{INPUT_T, "src"}` |
| `..._w_program_factory.cpp:141` | `TensorAccessorArgs(*output.buffer()).append_to(...)` | `TensorBinding{OUTPUT_T, "dst"}` |
| `..._w_program_factory.cpp:219` (reader RTA slot 0) | `input_buf` (`Buffer*`) | `TensorBinding` |
| `..._w_program_factory.cpp:221` (writer RTA slot 0) | `output_buf` (`Buffer*`) | `TensorBinding` |
| `..._w_program_factory.cpp:140` (writer CTA slot 0) | `static_cast<uint32_t>(CBIndex::c_16)` | `DFBBinding{OUT, "out", CONSUMER}` |
| `..._w_program_factory.cpp:122-124,166,187` | positional `compile_time_args` | named CTAs (`scaler`; `units_per_core`,`Wt`,`NC`,`origin_W`) |
| `..._w_program_factory.cpp:160,177,198` | `unpack_to_dest_mode` vector | `unpack_modes` table (see Hardware configuration) |
| `reader_moreh_mean_w.cpp:12,16,17,34` | `src_addr` RTA, `TensorAccessorArgs<0>()`, `next_compile_time_args_offset()`, `TensorAccessor(src_args, src_addr)` | `TensorAccessor(tensor::src)`; `scaler` becomes a plain named CTA |
| `reader_moreh_mean_w.cpp:19,23,29` | `constexpr uint32_t cb_id_in2/mask_w/in0 = tt::CBIndex::c_*` | `dfb::scaler` / `dfb::mask_w` / `dfb::input` |
| `reader_moreh_mean_w.cpp:38` | `get_tile_size(cb_id_in0)` | `dfb_in0.get_tile_size()` |
| `moreh_mean_w.cpp:21,22,24,26,28,30` | `cb_input` / `constexpr auto cb_* = tt::CBIndex::c_*` | `dfb::input` (via a mutable `uint32_t`, see Applied Patterns) / `dfb::scaler`/`mask_w`/`accum_dst`/`masked_input`/`out` |

### NC factory

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `..._nc_program_factory.cpp:116` | `TensorAccessorArgs(*input.buffer()).append_to(...)` — the reader's **only** CTA | `TensorBinding{INPUT_T, "input"}`; reader has no CTAs at all |
| `..._nc_program_factory.cpp:127` | `TensorAccessorArgs(*output.buffer()).append_to(...)` — the writer's **only** CTA | `TensorBinding{OUTPUT_T, "output"}`; writer has no CTAs |
| `..._nc_program_factory.cpp:193` (reader RTA slot 0) | `input_buf` (`Buffer*`) | `TensorBinding` |
| `..._nc_program_factory.cpp:201` (writer RTA slot 0) | `output_buf` (`Buffer*`) | `TensorBinding` |
| `..._nc_program_factory.cpp:147,163` | positional compute `compile_time_args` | named CTA `units_per_core` |
| `..._nc_program_factory.cpp:141,153,169` | `unpack_to_dest_mode` vector | omitted (all-`Default` ⇒ no entries) |
| `reader_moreh_mean_nc.cpp:12-19` | `uint32_t i = 0; get_arg_val<uint32_t>(i++)` ×7 | 6 named RTAs (the `input_addr` read is dropped); the `i` counter disappears — **not** varargs (fixed run of distinct fields) |
| `reader_moreh_mean_nc.cpp:22,23,24` | `constexpr uint32_t cb_id_in0/in1/in2 = tt::CBIndex::c_*` | `dfb::input` / `dfb::in1` / `dfb::scalar` |
| `reader_moreh_mean_nc.cpp:38,39` | `TensorAccessorArgs<0>()`, `TensorAccessor(input_args, input_addr)` | `TensorAccessor(tensor::input)` |
| `reader_moreh_mean_nc.cpp:43` | `get_tile_size(cb_id_in0)` | `dfb_in0.get_tile_size()` |
| `writer_moreh_mean_nc.cpp:13,17,20,21` | `output_addr` RTA, `constexpr uint32_t cb_id_out = 16` (**hardcoded magic index**), `TensorAccessorArgs<0>()`, `TensorAccessor(output_args, output_addr)` | `dfb::out`, `TensorAccessor(tensor::output)` |
| `writer_moreh_mean_nc.cpp:25` | `get_tile_size(cb_id_out)` | `dfb_out.get_tile_size()` |
| `moreh_mean_nc.cpp:14,15` | `get_arg_val<uint32_t>(0/1)` | `get_arg(args::num_input_tiles)` / `get_arg(args::num_output_tiles)` |
| `moreh_mean_nc.cpp:17,19,21,23,25,32,48,50,71` | `constexpr auto cb_* = tt::CBIndex::c_*` and the literal `tt::CBIndex::c_0/c_1/c_16` in `binary_op_init_common` | `dfb::input`/`in1`/`scalar`/`out`/`intermed0` |

**Semaphore-ID RTAs**: none — the op has no semaphores.
**Page-size 3rd-argument CTAs/RTAs**: none — all five accessor sites are 2-arg (audit: the subject
does not fire).
**Case 2 (raw-pointer) bindings**: none — all six bindings are Case 1 (via `TensorAccessor`), so no
`get_bank_base_address` bridge is used and no compute kernel needs a tensor binding.

---

## Applied Patterns

- **[Multi-variant factory](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-multi-variant-factories)**
  — realized as three sibling `program_factory_t` alternatives rather than a branch inside one
  `create_program_artifacts` (the legacy shape, preserved: `select_program_factory` already picks
  the variant).
- **[Self-loop DFB binding](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-self-loop-dfb-binding)**
  — `ACCUM_DST` and `MASKED_INPUT` on the H and W compute kernels; `INTERMED0` on the NC compute
  kernel. Genuine accumulator self-loops (real `reserve_back`/`push_back` **and**
  `wait_front`/`pop_front` on one kernel). Both endpoints share one `accessor_name`, which the spec
  validator explicitly permits (`program_spec.cpp:304-364`).
- **[Sync-free / single-ended CB → self-loop DFB](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb)**
  — `MASK_H` / `MASK_W` in the **unmasked** config only: the compute kernel is the single toucher
  (it binds and constructs the DFB while its FIFO calls are compile-time/runtime dead), so it is
  bound PRODUCER **and** CONSUMER there. Classified **per config**, exactly as that entry requires.
- **[Conditional / optional DFB bindings](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-conditional--optional-dfb-bindings)**
  — the **reader**'s `MASK_H` / `MASK_W` PRODUCER binding is emitted only when `do_mask_h` /
  `do_mask_w`, matching the existing `DO_MASK_H` / `DO_MASK_W` define the host already emits. The
  kernel-side consequence: `reader_moreh_mean_h.cpp:40` and `reader_moreh_mean_w.cpp:23` currently
  declare the mask CB constant **outside** the `#ifdef`; those declarations move inside it, so
  `dfb::mask_h` / `dfb::mask_w` never enter name lookup in the unmasked build. The **compute** side
  needs no `#ifdef`: it binds the mask DFB in every config (CONSUMER always, plus PRODUCER when
  unmasked), so the token always exists.
- **[Pass DFB handles directly to LLKs and kernel-lib helpers](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers)**
  — `dfb::name` is passed straight through to all five donor headers and to the LLKs, in both
  call-argument and **non-type-template-parameter** position
  (`dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<dfb::scaler, …>`,
  `compute_kernel_lib::reduce<REDUCE_OP, REDUCE_DIM, dfb::input, dfb::scaler, dfb::accum_dst>`).
  No `.id` extraction, no temporary `DataflowBuffer` wrappers, no donor edits.
- **[Same-FIFO aliasing](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-same-fifo-aliasing-one-dfb-multiple-kernel-side-names)**
  (path-dependent variant, runtime flavour) — `moreh_mean_w.cpp` keeps `cb_input` as a **mutable**
  variable that switches from `INPUT` to `MASKED_INPUT` mid-loop (`:21,51,95`) and constructs a
  throwaway `DataflowBuffer(cb_input)` at each use. Both DFBs are bound to the compute kernel and
  the variable stays `uint32_t`-valued, relying on `DFBAccessor`'s constexpr `operator uint32_t()`
  (`dataflow_buffer.h:55`). **Not** a token-for-token substitution — flagged by the brief.
- **[Unity-build hygiene](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-unity-build-hygiene-for-anonymous-namespace-symbols)**
  — all typed name constants are function-local (no anonymous namespace), so the three factory
  `.cpp`s can be concatenated by the `ttnncpp` unity build without symbol collisions.
- **[Porting a shared kernel](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-porting-a-shared-kernel)**
  — **rung 3 (convert in place)** for the intra-op shared writer, authorized by the invoker's
  bundled three-factory assignment. See Shared kernels above.

Patterns explicitly **not** used: `allow_instance_multi_binding` (no CB has ≥3 touchers or two
kernels locked to one role); `alias_with` (no multi-`buffer_index` `CBDescriptor`); `borrowed_from`
(no `CBDescriptor::buffer`); varargs (every RTA is a distinct field read once); `TensorParameter`
relaxations (strict matching kept); dead-CB drops (none — `MASK_H`/`MASK_W` are live under masking).

---

## Deferred / Flagged

- **Readiness-sheet cells (resolved by the invoker, not re-derived).** The audit left
  `Is safe to port?` and `TensorParameter relaxation` unread (the Drive connector cannot be
  authorized non-interactively). The invoker supplied all three factory rows: `Is safe to port? = yes`,
  `Is able to port? = yes`, `Concept = descriptor`, and confirmed the sheet revision has no
  `TensorParameter relaxation` column ⇒ that item resolves to **none**, consistent with the op
  having no custom hash. Nothing outstanding.
- **New findings during planning:**
  - **Two dead CTAs** surfaced that the audit did not call out: `HtWt` in the H reader
    (`reader_moreh_mean_h.cpp:21`, never used in the body) and the compute CTA in the NC factory
    (`moreh_mean_nc.cpp` reads no compile-time arg at all, yet
    `..._nc_program_factory.cpp:147,163` emit `units_per_core_group_N`). Both are preserved verbatim
    as named CTAs — dropping either is a separate cleanup, and dropping the NC one would additionally
    collapse the preserved per-group multiplicity. Routed to the report.
  - **The W factory needs an `unpack_modes` entry legacy did not have.** Not a behavior change — the
    value (`UnpackToSrc`) is exactly the legacy `Default` — but it is the one place the Metal 2.0
    validator is stricter than legacy in this op. Noted so a reviewer does not read it as a config
    change.
  - **`fp32_dest_acc_en` asymmetry (audit anomaly 5)** is preserved verbatim across the three
    factories, including NC's un-widened `c_24`. The port cements the existing behavior; if it is a
    latent bug it should be fixed separately. Routed to the report as a finding for the op owner.
- **No structural issue found that the audit missed.** No GlobalCircularBuffer, no
  `get_cb_tiles_acked_ptr` / `get_cb_tiles_received_ptr`, no cursor surgery, no host-folded base+offset,
  no compute-kernel `TensorAccessor` need, no out-of-op kernel edit required.
