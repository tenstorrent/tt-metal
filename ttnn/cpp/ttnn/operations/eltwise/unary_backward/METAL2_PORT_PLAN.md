# Port Plan — `eltwise/unary_backward`

Port plan for `eltwise/unary_backward` (`UnaryBackwardDeviceOperation` → `UnaryBackwardProgramFactory`), ported from
the `ProgramDescriptor` API to Metal 2.0. Written during the inventory and planning steps; committed alongside the
port for review.

**Base.** `anasuya/metal2_port_eltwise_unary_bw` @ `f126189dbb6`. Its code equals `main` `a1ce48e3fe8`, which contains
`be782c33d2e` (#63334). Checked with `git merge-base --is-ancestor be782c33d2e HEAD` on 2026-11-03.

**Op types in scope.** `UnaryBackwardOpType` has exactly four values on this base: `SIGMOID_BW`, `TANH_BW`,
`GELU_BW`, `GELU_TANH_BW` (`device/unary_backward_op_types.hpp:17-22`). They select five compute kernels. Per the
invoker (2026-11-03), the port covers these four types only. It does not wait for, or anticipate, the ~43 open PRs
that add op types.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept`. `create_descriptor` is declared on the factory struct
  (`device/unary_backward_program_factory.hpp:18-21`) and defined at `device/unary_backward_program_factory.cpp:26-481`.
- Where the factory lives: in `program_factory_t = std::variant<UnaryBackwardProgramFactory>`
  (`device/unary_backward_device_operation.hpp:28`). This is not the direct-descriptor shape. Exception 3 does not apply.
- Variants: one factory. It selects one of three program shapes at runtime (below) and one of five compute kernels.
- Custom `compute_program_hash`: present at `device/unary_backward_device_operation.cpp:238-260`. It keys each bound
  tensor's whole `TensorSpec`. Left intact.
- No `override_runtime_arguments`, no `get_dynamic_runtime_args`, no pybound `create_descriptor`
  (0 hits in `unary_backward_nanobind.cpp`).

**Three program shapes** (all in the one factory):

| Mode | Condition (factory `.cpp`) | Reader source | Writer source |
|---|---|---|---|
| Addressed | TILE layout and `!can_alias_shards` (`:82-129`, `:137-138`) | `eltwise/binary/.../reader_binary_interleaved_start_id.cpp` (borrowed) | `eltwise/unary/.../writer_unary_interleaved_start_id.cpp` (borrowed) |
| Aliased | `can_alias_shards` (TILE or ROW_MAJOR) | the same reader, with `IN0_SHARDED` + `IN1_SHARDED` (`:326-328`) | the same writer, with `OUT_SHARDED` (`:345-347`) |
| RM blocked | ROW_MAJOR and `!can_alias_shards` (`rm_blocked`, `:137-138`) | `unary_backward/.../dataflow/reader_unary_backward_row_major.cpp` (own) | `unary_backward/.../dataflow/writer_unary_backward_row_major.cpp` (own) |

**Compute source** (`:386-390`, table in `device/unary_backward_op_utils.cpp:13-64`): `eltwise_bw_sigmoid.cpp`
(`SIGMOID_BW`), `eltwise_bw_tanh.cpp` (`TANH_BW`), `eltwise_bw_gelu_poly.cpp` (`GELU_BW`), `eltwise_bw_gelu_tanh.cpp`
(`GELU_TANH_BW`, 16-bit DEST) or `eltwise_bw_gelu_tanh_fp32.cpp` (`GELU_TANH_BW` with fp32 DEST). All five include
`kernels/compute/operand_reconfig.hpp`. The RM reader and writer include `kernels/dataflow/row_major_pages.hpp`.

Selectable entry points: 2 readers × 2 writers × 5 compute kernels = 9 sources. They all convert together.

### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader (addressed / aliased) | `reader_binary_interleaved_start_id.cpp` | `all_cores` | `{0}` (`block_or_width_sharded`), then `TensorAccessorArgs(grad_output)`, `TensorAccessorArgs(input)` when not aliased (`:324-335`) | none | 7: `grad_output` `Buffer*`, `input` `Buffer*`, `units`, `units_done`, `0`, `0`, `num_cores_y` (`:467-468`) | none | `IN0_SHARDED=1`, `IN1_SHARDED=1` when aliased | unset → O2 | `ReaderConfigDescriptor{}` → RISCV_1 / NOC_0 / dedicated |
| reader (RM blocked) | `reader_unary_backward_row_major.cpp` | `all_cores` | `TensorAccessorArgs(grad_output)`, `TensorAccessorArgs(input)` (`:306-312`) | none | 17 (`:431-449`): `units`, `units_done`, `row_elements`, `TILE_HW`, `chunks_per_row`, `rows_per_tile`, `total_rows`, `grad_output` `Buffer*`, 4 grad paging fields, `input` `Buffer*`, 4 input paging fields | none | none | unset → O2 | `ReaderConfigDescriptor{}` |
| writer (addressed / aliased) | `writer_unary_interleaved_start_id.cpp` | `all_cores` | `{c_2}`, then `TensorAccessorArgs(output)` always (`:340-344`) | none | 3: output `Buffer*`, `units`, `units_done` (`:470`) | none | `OUT_SHARDED=1` when aliased | unset → O2 | `WriterConfigDescriptor{}` → RISCV_0 / NOC_1 / dedicated |
| writer (RM blocked) | `writer_unary_backward_row_major.cpp` | `all_cores` | `TensorAccessorArgs(output)` (`:314-319`) | none | 12 (`:450-463`): `units`, `units_done`, `row_elements`, `TILE_HW`, `chunks_per_row`, `rows_per_tile`, `total_rows`, output `Buffer*`, 4 output paging fields | none | none | unset → O2 | `WriterConfigDescriptor{}` |
| compute | one of the five above | `all_cores` | none | none | 1: `units`, or `units * chunks_per_row` when RM blocked (`:465`, `:469`) | none | `MIXED_OPERAND_DATA_FORMATS=1` when the operand formats differ; `COPY_DEST_DATA_FORMAT`; `BF16_ROUNDING_MODE` (`:371-384`) | unset → **O3** | `ComputeConfigDescriptor{.math_fidelity = spec.math_fidelity, .fp32_dest_acc_en, .unpack_to_dest_mode}` (`:393-397`); every other field default |

Resolved DM configs (checked in `tt_metal/impl/kernels/kernel_types.cpp:13-41` and `kernel_types.hpp:141-153`):
reader `(RISCV_1, NOC_0, DM_DEDICATED_NOC)`, writer `(RISCV_0, NOC_1, DM_DEDICATED_NOC)` on WH and BH. Resolved
opt levels (`tt_metal/impl/program/program.cpp:526`, `:534`, `:563`): O2 for both DM kernels, O3 for compute.
`grep -n opt_level` on the factory: 0 hits.

Compute config fields the factory sets: `math_fidelity`, `fp32_dest_acc_en`, `unpack_to_dest_mode`. Left at the
`ComputeConfigDescriptor` defaults (`tt_metal/api/tt-metalium/program_descriptors.hpp:103-114`):
`dst_full_sync_en = false`, `bfp8_pack_precise = false`, `math_approx_mode = false`, `enable_trisc2_rvv = false`.
`unpack_to_dest_mode` is a 32-entry vector: `UnpackToDestFp32` at `c_0` and `c_1` exactly when
`fp32_dest_acc_en`, `Default` elsewhere (`:362-366`). `fp32_dest_acc_en` = `spec.force_fp32_dest_acc` or any of the
three formats is Float32 (`:355-357`).

### CBs

| index | total_size | core_ranges | data_format | page_size | tile | buffer |
|---|---|---|---|---|---|---|
| c_0 (`grad_output`) | `cb_pages * tile_size(grad fmt)` | `all_cores` | grad_output dtype | `tile_size(grad fmt)` | unset | `grad_output.buffer()` when aliased (`:256-260`) |
| c_1 (`input`) | `cb_pages * tile_size(input fmt)` | `all_cores` | input dtype | `tile_size(input fmt)` | unset | `input.buffer()` when aliased (`:261`) |
| c_2 (output) | `cb_pages * tile_size(output fmt)` | `all_cores` | output dtype | `tile_size(output fmt)` | unset | `output.buffer()` when aliased (`:262-266`) |
| c_3 (reader scratch) | `scratch_bytes` | `all_cores` | output dtype | `scratch_bytes` | unset | none. Created only when `rm_blocked` (`:272-289`) |
| c_4 (writer scratch) | `scratch_bytes` | `all_cores` | output dtype | `scratch_bytes` | unset | none. Created only when `rm_blocked` |

`cb_pages` = `shard_tiles` when aliased, otherwise 2 (`:229-230`). `scratch_bytes` = largest of the three tile
sizes + 3 × the largest buffer alignment (`:273-277`). No GlobalCircularBuffer, no `address_offset`, no aliased
(multi-index) CB.

### Semaphores
none

### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) | kernel site |
|---|---|---|---|
| `TensorAccessorArgs(*grad_output_buffer)` `:330` | `grad_output` | reader RTA 0 (`:468`) | `reader_binary_interleaved_start_id.cpp:34-35`, `:51` |
| `TensorAccessorArgs(*input_buffer)` `:331` | `input` | reader RTA 1 (`:468`) | same file `:35`, `:58` |
| `TensorAccessorArgs(*grad_input_buffer)` `:341` | output | writer RTA 0 (`:470`) | `writer_unary_interleaved_start_id.cpp:24`, `:42` (dead under `OUT_SHARDED`) |
| `TensorAccessorArgs(*grad_output_buffer)` `:307` | `grad_output` | RM reader RTA 7 (`:440`) | `reader_unary_backward_row_major.cpp:33`, `:52`, `:55` |
| `TensorAccessorArgs(*input_buffer)` `:308` | `input` | RM reader RTA 12 (`:445`) | same file `:40`, `:53-54`, `:56` |
| `TensorAccessorArgs(*grad_input_buffer)` `:315` | output | RM writer RTA 7 (`:459`) | `writer_unary_backward_row_major.cpp:26`, `:37-38` |

Every accessor takes two constructor arguments (no third, page-size argument). Each address is used only to build
its accessor (Case 1).

### Work split
- Addressed / RM blocked: `split_work_to_cores(compute_with_storage_grid_size, num_units)` (`:214-220`). `num_units`
  = tiles, or blocks when RM blocked (`:190`). Cores are walked column-major, `{i / num_cores_y, i % num_cores_y}`
  (`:413-416`). Two core groups with `num_units_per_core_group_{1,2}`.
- Aliased: the shard grid (`:207-212`). One group, `shard_tiles` units per core. Cores follow the shard orientation
  (`corerange_to_cores`, `:409-411`).
- `units_done` accumulates across the walk (`:419`, `:473`).

### Shared kernels

| Kernel | Kind | Other binders | `_metal2` fork beside it | Rung |
|---|---|---|---|---|
| `eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp` | borrowed | `tests/ttnn/unit_tests/gtests/test_generic_op.cpp:437` | yes, `reader_binary_interleaved_start_id_metal2.cpp` (blob `a834b1cd6a6`), no current binder | 1: reuse |
| `eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp` | borrowed | 3 factories, 5 gtest sites, 6 Python generic-op tests (sunset list in the brief) | yes, `writer_unary_interleaved_start_id_metal2.cpp` (blob `bfc0678d0f3`), 26 factory binders | 1: reuse |

Reused forks' binding vocabulary (the constraint the KernelSpecs are built against):
- Reader fork: `dfb::in0` (c_0, FIFO producer), `dfb::in1` (c_1, FIFO producer), `tensor::src0`, `tensor::src1`.
  Named RTAs `num_tiles`, `start_id`, `block_height`, `block_width`, `num_cores_y`. Named CTA
  `block_or_width_sharded`. Gates `tensor::src0` on `!IN0_SHARDED` and `tensor::src1` on `!IN1_SHARDED`.
- Writer fork: `dfb::out` (c_2, FIFO consumer), `tensor::dst`. Named RTAs `num_pages`, `start_id`. Gates
  `tensor::dst` on `!OUT_SHARDED`. Optional `ENTRIES_PER_PAGE` / `PAGE_BYTES` defines: not set (default is one page
  per DFB entry, `dfb.get_entry_size()`, which equals the legacy `fifo_page_size`).

Both originals already carry the pointer comment. The op's own kernels (RM reader, RM writer, `row_major_pages.hpp`,
`operand_reconfig.hpp`, the five compute kernels) are bound or included only by this op. They convert in place.

### Flags
- None unreferenced: every file under `device/kernels/` is bound in some configuration.
- The reader's `block_height`, `block_width`, `num_cores_y` are dead while `block_or_width_sharded` is 0. In aliased
  mode the legacy `Buffer*` RTAs are dead too. Both are carried as they are (the `Buffer*` RTAs become bindings).

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept`.
- **Custom `compute_program_hash`**: present at `device/unary_backward_device_operation.cpp:238-260`. Leave intact.
- **Implementation notes**: `create_descriptor` becomes `create_program_artifacts` inside the existing
  `UnaryBackwardProgramFactory`. No device-operation-class edit is forced. The legacy body keeps computing geometry
  on `ttnn::Tensor` (shapes, shard specs, `buffer()->page_size()` / `alignment()`), as before. The three
  `MeshTensor`s are extracted at the top for the `TensorParameter` specs and `TensorArgument`s.

## Planned Spec Shape

Names are function-local constants (unity-build hygiene).

- **KernelSpecs** (3, one per legacy `KernelDescriptor`):
  - `reader`: the reader fork, or the RM reader when `rm_blocked`.
    `create_reader_datamovement_config()`. No `opt_level` (O2 = legacy).
  - `writer`: the writer fork, or the RM writer when `rm_blocked`.
    `create_writer_datamovement_config()`. No `opt_level` (O2 = legacy).
  - `compute`: the selected compute source. `ComputeHardwareConfig` built directly (the legacy factory sets a
    `ComputeConfigDescriptor` itself, recipe "Style B"):
    `fpu_math_fidelity = spec.math_fidelity`, `enable_32_bit_dest = fp32_dest_acc_en`, and every other field at its
    default (`sfpu_precision_mode = Precise` = legacy `math_approx_mode = false`; `double_buffer_dest = true` = legacy
    `dst_full_sync_en = false`; `config_1xx` unset → `bfp8_pack_precise = false`, checked in
    `kernel_lowering.cpp:160-176`). `compiler_options.opt_level = O3`. Defines carried as they are.
- **DataflowBufferSpecs** (created in legacy index order, so the device slots stay 0-4):

  | spec | legacy | entry_size × num_entries | data_format | borrowed_from |
  |---|---|---|---|---|
  | `grad_output` | c_0 | `tile_size(grad fmt)` × `dfb_entries` | grad_output | `grad_output` when aliased |
  | `input` | c_1 | `tile_size(input fmt)` × `dfb_entries` | input | `input` when aliased |
  | `grad_input` | c_2 | `tile_size(output fmt)` × `dfb_entries` | output | `output` when aliased |
  | `reader_scratch` | c_3 | `scratch_bytes` × 1 | output | none; `rm_blocked` only |
  | `writer_scratch` | c_4 | `scratch_bytes` × 1 | output | none; `rm_blocked` only |

  `dfb_entries` is the legacy `cb_pages`. No `tile_format_metadata` (legacy `tile` unset).
- **SemaphoreSpecs**: none.
- **TensorParameters** (3): `grad_output`, `input`, `output`, each `.spec = <mesh tensor>.tensor_spec()` and
  `.relaxations = {}`, in every mode (relaxation doc §2, confirmed by the audit; invoker instruction 2026-11-03).
- **WorkUnitSpecs** (1): `main` = {reader, writer, compute} on `all_cores`.
- **Op-owned tensors**: none.

**Bindings per mode** (kernel-side names in `dfb::` / `tensor::`):

| Kernel | Addressed | Aliased | RM blocked |
|---|---|---|---|
| reader | `in0` P (grad_output), `in1` P (input); `src0` → grad_output, `src1` → input | `in0` P, `in1` P; no tensor binding | `grad` P (grad_output), `input` P (input), `scratch` P+C (reader_scratch); `grad` → grad_output, `input` → input |
| writer | `out` C (grad_input); `dst` → output | `out` C; no tensor binding | `dst` C (grad_input), `scratch` P+C (writer_scratch); `dst` → output |
| compute | `grad_out` C (grad_output), `input` C (input), `grad_in` P (grad_input) | same | same |

**Run args.** Per-node RTAs built in the legacy core loop with `AddRuntimeArgsForNode` (same cores, same values,
same order). `tensor_args` = all three `TensorParameter`s in every mode.

## Preserved Multiplicity

none — no work-split multiplicity in legacy. Each kernel is one `KernelDescriptor` over `all_cores`; the two core
groups differ only in RTA values.

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| factory `:468`, reader RTA 0 | `grad_output_buffer` (`Buffer*`) | `TensorBinding{grad_output, "src0"}` (addressed only) |
| factory `:468`, reader RTA 1 | `input_buffer` (`Buffer*`) | `TensorBinding{input, "src1"}` (addressed only) |
| factory `:470`, writer RTA 0 | `grad_input_buffer` (`Buffer*`) | `TensorBinding{output, "dst"}` (addressed only) |
| factory `:440`, RM reader RTA 7 | `grad_output_buffer` | `TensorBinding{grad_output, "grad"}` |
| factory `:445`, RM reader RTA 12 | `input_buffer` | `TensorBinding{input, "input"}` |
| factory `:459`, RM writer RTA 7 | `grad_input_buffer` | `TensorBinding{output, "dst"}` |
| factory `:252` (`.buffer = shard_buffer`) | aliased CB on a tensor buffer | `DataflowBufferSpec::borrowed_from` |
| factory `:330-331`, `:307-308` | `TensorAccessorArgs(...).append_to(reader CTAs)` | binding; kernel `TensorAccessor(tensor::…)` |
| factory `:341`, `:315` | `TensorAccessorArgs(...).append_to(writer CTAs)` | binding; kernel `TensorAccessor(tensor::…)` |
| reader fork original `:34-39` / RM reader `:52-54` / RM writer `:37` | `TensorAccessorArgs<N>()` chains | `TensorAccessor(tensor::name)` |
| factory `:340`, writer CTA 0 | `grad_input_cb_index` (c_2) | `DFBBinding{grad_input, "out", CONSUMER}` |
| reader original `:30-31` (hardcoded c_0 / c_1) | magic CB indices | `DFBBinding{grad_output, "in0", PRODUCER}`, `DFBBinding{input, "in1", PRODUCER}` |
| RM reader `:47-49` (c_0, c_1, c_3) | magic CB indices | `dfb::grad`, `dfb::input`, `dfb::scratch` |
| RM writer `:33-34` (c_2, c_4) | magic CB indices | `dfb::dst`, `dfb::scratch` |
| five compute kernels (`tt::CBIndex::c_0/c_1/c_2`) | magic CB indices | `dfb::grad_out`, `dfb::input`, `dfb::grad_in` |
| factory `:324`, reader CTA 0 | positional `0` | named CTA `block_or_width_sharded = 0` |
| factory `:467-468`, reader RTAs 2-6 | positional | named `num_tiles`, `start_id`, `block_height`, `block_width`, `num_cores_y` |
| factory `:470`, writer RTAs 1-2 | positional | named `num_pages`, `start_id` |
| factory `:431-449`, RM reader RTAs (15 non-address) | positional `arg++` run | named `num_blocks`, `start_block`, `row_elements`, `chunk_elements`, `chunks_per_row`, `rows_per_tile`, `total_rows`, `grad_page_elements`, `grad_pages_per_row`, `grad_element_bytes`, `grad_alignment`, `input_page_elements`, `input_pages_per_row`, `input_element_bytes`, `input_alignment` |
| factory `:450-463`, RM writer RTAs (11 non-address) | positional `arg++` run | named `num_blocks`, `start_block`, `row_elements`, `chunk_elements`, `chunks_per_row`, `rows_per_tile`, `total_rows`, `dst_page_elements`, `dst_pages_per_row`, `dst_element_bytes`, `dst_alignment` |
| factory `:465`, `:469`, compute RTA 0 | positional | named `per_core_tile_cnt` (the name in the kernel-spec contract, `unary_backward_op_utils.hpp:19-20`; the gelu-tanh kernels keep their local `num_tiles`) |
| factory `:362-366` | `unpack_to_dest_mode[c_0/c_1]` vector | `unpack_modes = {{grad_output, UnpackToDest}, {input, UnpackToDest}}` when fp32 DEST |

No semaphore IDs and no page-size third arguments to drop. No varargs: every RTA is a distinct field read once.

## Applied Patterns

- [Sync-free and single-ended CBs → self-loop DFB](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb):
  `reader_scratch` on the RM reader and `writer_scratch` on the RM writer. Census re-derived: each is touched by one
  kernel, through `get_write_ptr()` only (`row_major_pages.hpp:140-143`), no FIFO op. One accessor name `scratch`
  for both roles. DM self-loop: legal on Gen1, Quasar debt.
- Borrowed-memory DFBs (`borrowed_from`) for c_0, c_1, c_2 in aliased mode, one shard deep. 1P+1C in every mode
  (re-derived: reader FIFO-produces c_0/c_1, compute consumes; compute produces c_2, writer FIFO-consumes).
- [Conditional / optional resource bindings](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-conditional--optional-resource-bindings):
  the reader's `src0` / `src1` and the writer's `dst` tensor bindings are omitted exactly when `IN0_SHARDED` /
  `IN1_SHARDED` / `OUT_SHARDED` are defined. The forks already gate those tokens with `#ifdef`; the host condition
  and the defines share `can_alias_shards`.
- [Caution: Porting a shared kernel](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-porting-a-shared-kernel),
  rung 1 for both borrowed kernels.
- [Pass DFB handles directly to LLKs and kernel-lib helpers](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers):
  `compute_kernel_hw_startup(dfb::grad_out, dfb::grad_in)` and `ckl::input(dfb::…)` / `ckl::output(dfb::…)` in the
  five compute kernels. One `DataflowBuffer` object per DFB is kept for `CopyTile` / `PackTile`.

## Deferred / Flagged

- **Comments that name CBs.** The rename sweep covers the factory, the five compute kernels, `operand_reconfig.hpp`,
  the RM kernels and `row_major_pages.hpp` (locals `grad_cb`, `input_cb`, `cb_base`, `cb`, `cb_addr` → `dfb_*`;
  "CB page" → "DFB entry"). The kernel-contract comment in `unary_backward_op_utils.hpp:19-20` names `c_0` / `c_1`
  / `c_2` and "runtime arg 0"; it is updated to the DFB names because the port changes that contract. One hit stays:
  `unary_backward_device_operation.cpp:252` ("destination CB") is in the device-operation class, which the port may
  not edit. It goes to the report.
- **Recipe API names differ from the headers** (`ComputeGen1Config`, `DataMovementGen1Config`,
  `create_reader_datamovement_config(device->arch())`). The headers use `ComputeHardwareConfig` (common fields plus
  `config_1xx`) and `DataMovementHardwareConfig{.config_1xx}`; the TTNN DM helpers take no arch argument. The plan
  follows the headers. Recorded as friction.
- **Validation forcing.** Per the workstation note, `tt_metal/impl` scaffolding is not applied. The miss path
  validates by default (`mesh_device_operation_adapter.hpp:993-996`); the hit path is forced with
  `TTNN_CONFIG_OVERRIDES='{"validate_program_args": true}'` (`:830`). Recorded in the report.
- No new structural findings. No stop signal.
