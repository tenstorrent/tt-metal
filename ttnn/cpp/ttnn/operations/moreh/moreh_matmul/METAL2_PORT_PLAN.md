# Port Plan — moreh_matmul

Port plan for `moreh/moreh_matmul`, ported from the `ProgramDescriptor` (`create_descriptor`) API to Metal 2.0 (`MetalV2FactoryConcept` / `create_program_artifacts`).
Written during the inventory and planning steps; committed alongside the port for review.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` — `MultiCoreProgramFactory::create_descriptor(...)` returns `tt::tt_metal::ProgramDescriptor` (`device/moreh_matmul_device_operation.hpp:37-40`).
- Variants: single (`program_factory_t = std::variant<MultiCoreProgramFactory>`). No `select_program_factory` (framework default-selects the single variant).
- Custom `compute_program_hash`: none — already default reflection-based hash (confirmed by audit + code scan).

### Kernels
All op-owned, under `device/kernels/`. Grid: `split_work_to_cores(grid, num_output_tiles)` → `core_group_1` (count `num_output_tiles_per_core_group_1`) + optional `core_group_2` (count `_2`); reader/writer on `all_cores`, compute split per group.

| unique_id | source | core_ranges | CTAs (positional, legacy) | RTAs (legacy) | defines | config |
|---|---|---|---|---|---|---|
| reader | `device/kernels/reader_moreh_matmul.cpp` | all_cores | `Kt, transpose_input, transpose_other, input_mask_h, input_mask_w, other_mask_h, other_mask_w`, `TensorAccessorArgs(input)`, `TensorAccessorArgs(other)`, `[is_scalar_bias, TensorAccessorArgs(bias)]`(FUSE_BIAS) | `input_buf*, other_buf*, num_tiles_written, num_output_tiles_per_core, input_stride[8], other_stride[8], output_stride[8], input_not_bcast[8], other_not_bcast[8], [bias_buf*]`(FUSE_BIAS) | `FUSE_BIAS`(if bias) | ReaderConfigDescriptor{} |
| writer | `device/kernels/writer_moreh_matmul.cpp` | all_cores | `TensorAccessorArgs(output)` | `output_buf*, num_tiles_written, num_output_tiles_per_core` | — | WriterConfigDescriptor{} |
| compute (x1 or x2, per group) | `device/kernels/moreh_matmul.cpp` | core_group_1 / core_group_2 | `num_output_tiles(per-group), Mt, Nt, Kt, transpose_input, transpose_other, input_mask_h, input_mask_w, other_mask_h, other_mask_w, [is_scalar_bias]`(FUSE_BIAS) | `num_tiles_written, output_stride[8]` | `FUSE_BIAS`(if bias), `FP32_DEST_ACC_EN`(if fp32_dest_acc_en) | ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en, dst_full_sync_en, unpack_to_dest_mode, math_approx_mode} |

`*_buf` are bare `Buffer*` pushed via the `Buffer*`-binding RTA form (`program_factory.cpp:481-499`) — clean bases, no offset arithmetic (audit offset-base gate GREEN).

### CBs
`add_cb(idx, num_tiles, fmt)` → `total_size = num_tiles*tile_size(fmt)`, single `CBFormatDescriptor{buffer_index=idx, data_format=fmt, page_size=tile_size(fmt)}`, `core_ranges=all_cores`, `.tile` unset (default 32x32). `cb_data_format = datatype_to_dataformat_converter(output.dtype())` (BFLOAT16 → Float16_b). `im0/im3_data_format = fp32_dest_acc_en ? Float32 : cb_data_format`.

| index | num_tiles | data_format | tile |
|---|---|---|---|
| c_0 (in0) | in0_t=2 | cb_data_format | default |
| c_1 (in1) | in1_t=2 | cb_data_format | default |
| c_2 (in2, input mask) | in2_t=3 | cb_data_format | default |
| c_3 (in3, other mask) | in3_t=3 | cb_data_format | default |
| c_4 (in4, bias) | in4_t=2 | cb_data_format | default |
| c_24 (im0, matmul reload) | im0_t=1 | im0_data_format | default |
| c_25 (im1, input transpose) | im1_t=2 | cb_data_format | default |
| c_26 (im2, other transpose) | im2_t=2 | cb_data_format | default |
| c_27 (im3, bias-add temp) | im3_t=1 | im3_data_format | default |
| c_16 (out0) | out0_t=2 | cb_data_format | default |

All 10 CBs allocated **unconditionally** in legacy (even mask/transpose/bias CBs on inactive paths — small L1, no correctness impact; audit "Misc anomalies").

### Semaphores
none — op uses no semaphores.

### Tensor accessors
| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `TensorAccessorArgs(input.buffer())` `:325` | input | reader RTA slot 0 (`input_buf`) |
| `TensorAccessorArgs(other.buffer())` `:326` | other | reader RTA slot 1 (`other_buf`) |
| `TensorAccessorArgs(bias->buffer())` `:331` (FUSE_BIAS) | bias | reader RTA tail (`bias_buf`) |
| `TensorAccessorArgs(output.buffer())` `:336` | output | writer RTA slot 0 (`output_buf`) |

Kernel side: reader `TensorAccessor(input_args, input_addr)` / `(other_args, other_addr)` / `(bias_args, bias_addr)` (`:90-94`); writer `TensorAccessor(output_args, output_addr)` (`:21`). All 2-arg (no 3rd page-size arg). All Case 1 (base fed straight into `TensorAccessor`).

### Work split
- Driver: `split_work_to_cores(grid, num_output_tiles)` (`:260`)
- num_cores, all_cores, core_group_1, core_group_2, num_output_tiles_per_core_group_1, num_output_tiles_per_core_group_2
- Compute has 2 `KernelDescriptor`s (group_1, group_2) differing only in the `num_output_tiles` CTA + core_ranges; reader/writer single descriptor over all_cores.

### Cross-op kernels
none — all three kernel `.cpp` are op-owned. They `#include` shared moreh pool headers `ttnn/cpp/ttnn/kernel/dataflow/moreh_common.hpp` (reader/writer: `ArgFetcher`, `generate_mask_tiles`) and `ttnn/cpp/ttnn/kernel/compute/moreh_common.hpp` (compute: `ArgFetcher`). Both are Device 2.0 native (take `DataflowBuffer` / `cb_id`); not modified by this port.

### Flags
- `program_factory.cpp:388` allocates `unpack_to_dest_mode` of size `NUM_CIRCULAR_BUFFERS`, sets only `c_24` (under fp32). Benign default-fill.
- Host wrapper `moreh_matmul.cpp` may route 1-D inputs to `moreh_dot` (separate op) — above the device operation, unaffected.

## TTNN ProgramFactory
- **Concept (inherited from audit)**: `MetalV2FactoryConcept` (plain — no op-owned tensors).
- **Custom `compute_program_hash`**: none.
- **Implementation notes**: reader/writer span both work units (target `core_group_1` + `core_group_2`); compute is one `KernelSpec` per group. Extract `MeshTensor` refs for tensor bindings; keep the legacy `ttnn::Tensor` shape-math unchanged.

## Planned Spec Shape
- **KernelSpecs**: `reader` (DM), `writer` (DM), `compute_g1` (compute); `compute_g2` when `core_group_2` non-empty. (compute multiplicity preserves the per-group `num_output_tiles` CTA — no CTA→RTA demotion.)
- **DataflowBufferSpecs**: 10, 1:1 with legacy CBs — `IN0,IN1,IN2,IN3,IN4,IM0,IM1,IM2,IM3,OUT0`.
- **SemaphoreSpecs**: none.
- **TensorParameters**: `INPUT`, `OTHER`, `OUTPUT` always; `BIAS` when `bias.has_value()`.
- **WorkUnitSpecs**: `moreh_mm_g1` = {reader, writer, compute_g1} @ core_group_1; `moreh_mm_g2` = {reader, writer, compute_g2} @ core_group_2 (conditional).

### DFB endpoint dispositions (re-derived from kernel-touch census)
- `IN0,IN1,IN2,IN3,IN4`: reader PRODUCER; compute_g1/g2 CONSUMER (disjoint node sets → legal multi-consumer, **no** multi-binding flag). Legal 1:1 per node.
- `OUT0`: compute_g1/g2 PRODUCER (disjoint); writer CONSUMER. Legal 1:1 per node.
- `IM0,IM1,IM2,IM3`: single toucher = compute → **self-loop** (compute bound both PRODUCER and CONSUMER, shared accessor name). compute_g1/g2 self-loop on their disjoint node sets.
- No dead CB, no `allow_instance_multi_binding`. Matches audit CB-endpoint census.

## Preserved Multiplicity

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| compute_desc_1 (core_group_1), compute_desc_2 (core_group_2) of `moreh_matmul.cpp` | COMPUTE_G1, COMPUTE_G2 | moreh_mm_g1, moreh_mm_g2 (disjoint nodes) | IN0-IN4 (CONSUMER), OUT0 (PRODUCER), IM0-IM3 (self-loop PRODUCER+CONSUMER) — each a single-role binding per node (disjoint node sets), not the multi-binding flag |

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader RTA `input_buf` (`:481`) + CTA `TensorAccessorArgs(input)` (`:325`) | `Buffer*` RTA + accessor-args CTA | `TensorParameter INPUT` + `TensorBinding{INPUT,"input"}`; kernel `TensorAccessor(tensor::input)` |
| reader RTA `other_buf` (`:482`) + CTA (`:326`) | same | `OTHER` / `TensorAccessor(tensor::other)` |
| reader RTA `bias_buf` (`:494`) + CTA (`:331`) (FUSE_BIAS) | same | `BIAS` (conditional) / `TensorAccessor(tensor::bias)` |
| writer RTA `output_buf` (`:499`) + CTA (`:336`) | same | `OUTPUT` / `TensorAccessor(tensor::output)` |
| reader/compute positional CTAs | `get_compile_time_arg_val(N)` | named CTAs (`get_arg(args::name)`) |
| compute magic CB indices `tt::CBIndex::c_*` (`moreh_matmul.cpp:21-30`) | magic index constants | `dfb::in0..out0/im0..im3` bindings |
| reader magic CB ids `cb_id_in0..in4 = 0..4` (`reader:83-87`) | literals | `dfb::in0..in4` |
| writer magic CB id `cb_id_out = 16` (`writer:19`) | literal | `dfb::out0` |
| `get_tile_size(cb_id)` (reader `:120-124`, writer `:25`) | free fn by cb-id | `dfb.get_tile_size()` member (whitelist §B) |
| reader `input_stride/other_stride/output_stride/input_not_bcast/other_not_bcast` RTAs | 5×8 arrays via `ArgFetcher` | runtime **varargs** (40) — see Applied Patterns note |
| compute `output_stride` RTA | 8 array via `ArgFetcher` | runtime **varargs** (8) |
| reader `num_tiles_written`, `num_output_tiles_per_core` RTAs | positional | named RTAs `output_tile_start_idx`, `num_output_tiles` |
| writer `num_tiles_written`, `num_output_tiles_per_core` RTAs | positional | named RTAs `start_id`, `num_output_tiles` |
| compute `num_tiles_written` RTA | positional | named RTA `output_tile_start_idx` |

## Applied Patterns
- [Self-loop DFB binding](../shared/port_patterns.md#pattern-self-loop-dfb-binding): IM0/IM1/IM2/IM3 on each compute KernelSpec (PRODUCER + CONSUMER, shared accessor name).
- [Demoting per-group CTA to RTA (avoided)](../shared/port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta): compute per-group `num_output_tiles` kept as a CTA via two KernelSpecs over disjoint node sets (reader/writer bind the shared DFBs across both, single-role per node — not the multi-binding flag).
- Hardware config: Style A compute (`get_compute_kernel_config_args` → `to_compute_hardware_config`); `unpack_modes` set by hand under fp32 (IM0→UnpackToDest [legacy UnpackToDestFp32], IM3→UnpackToSrc [legacy Default, required-entry because it is a Float32 consumed DFB under enable_32_bit_dest]).
- **RTA varargs (deviates from brief — see report Friction)**: the 5 reader dimensional arrays + compute `output_stride` are treated as runtime **varargs**, matching [patterns catalog Caution "homogeneous literal-count array → vararg"](../shared/port_patterns.md#caution-avoid-varargs-unless-absolutely-necessary). The brief classified them as nameable; the audit's own "Recipe notes" flagged this exact tension. Chosen: varargs (cleaner minimal kernel diff, keeps the loops; matches the most-specific authoritative catalog text). `output_tile_start_idx`/`num_output_tiles`/`start_id` remain **named** (distinct scalar fields).

## Deferred / Flagged
- CRTA opportunity (not applied — out of scope): the reader stride/bcast varargs and compute `output_stride` are identical on every node → could be common runtime args. Left per-node (RTA/vararg) to avoid changing dispatch semantics (recipe §Construct). Noted for a later pass.
- All 10 DFBs allocated unconditionally (mirrors legacy). Mask/transpose/bias DFBs on inactive paths carry live bindings but are untouched by the kernel — harmless (per-execution DFB state), zero functional change.
