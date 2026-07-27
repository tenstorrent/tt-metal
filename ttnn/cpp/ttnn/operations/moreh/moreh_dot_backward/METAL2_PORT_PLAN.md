# Port Plan — moreh_dot_backward

Port plan for `ttnn/cpp/ttnn/operations/moreh/moreh_dot_backward`, ported from the
legacy `ProgramDescriptor` / `create_descriptor` (single-descriptor) concept to Metal 2.0
(`MetalV2FactoryConcept` / `create_program_artifacts`).
Written during the inventory and planning steps; committed alongside the port for review.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` — realized as the "single-descriptor" shape
  (`create_descriptor` is a static method directly on `MorehDotBackwardOperation`, no separate
  ProgramFactory struct, no `program_factory_t`). Detected by the framework's `HasDirectDescriptor`.
- Variants: single.
- Custom `compute_program_hash`: none — already default reflection-based hash (audit confirmed).

*(Target concept `MetalV2FactoryConcept` was chosen during the audit; carried forward below.)*

### Kernels
| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | config |
|---|---|---|---|---|---|---|---|---|
| reader | device/kernels/reader_moreh_dot_backward.cpp | `{0,0}` | `TensorAccessorArgs(src0)`,`(src1)`,`(src2)` | none | has_input_grad, has_other_grad, src0_addr, src1_addr, src2_addr, num_tiles, start_id | none | none | ReaderConfigDescriptor{} |
| writer | device/kernels/writer_moreh_dot_backward.cpp | `{0,0}` | `c_16`, `c_17`, `TensorAccessorArgs(dst0)`,`(dst1)` | none | has_input_grad, has_other_grad, dst0_addr, dst1_addr, num_tiles, start_id | none | none | WriterConfigDescriptor{} |
| compute | device/kernels/moreh_dot_backward.cpp | `{0,0}` | none | none | has_input_grad, has_other_grad, per_core_block_cnt | none | none | ComputeConfigDescriptor{} (all defaults) |

All kernel sources are op-owned (inside the op directory). All on Device 2.0 (`Noc`, `DataflowBuffer`,
`TensorAccessor`). Kernel-source selection is fixed (one source per KernelDescriptor); no runtime selection.

### CBs
| CB | buffer_index | total_size | page_size | data_format | tile | producer | consumer |
|---|---|---|---|---|---|---|---|
| c_0 (in0 / output_grad scalar) | c_0 | 2·cb_tile | cb_tile | df(output_grad) | default | reader | compute |
| c_1 (in1 / input) | c_1 | 2·cb_tile | cb_tile | df | default | reader | compute |
| c_2 (in2 / other) | c_2 | 2·cb_tile | cb_tile | df | default | reader | compute |
| c_16 (out0 / input_grad) | c_16 | 2·cb_tile | cb_tile | df | default | compute | writer |
| c_17 (out1 / other_grad) | c_17 | 2·cb_tile | cb_tile | df | default | compute | writer |

`cb_data_format = datatype_to_dataformat_converter(output_grad.dtype())`, `cb_tile = tile_size(cb_data_format)`.
No GlobalCircularBuffer, no `address_offset`, no aliasing. Each CB is 1P+1C.

### Semaphores
None (op uses no semaphores).

### Tensor accessors
| Originating tensor | kernel | accessor | host RTA (dropped) |
|---|---|---|---|
| output_grad (input) | reader | s0 | `Buffer* src0_buffer` |
| input (input) | reader | s1 | `Buffer* src1_buffer` |
| other (input) | reader | s2 | `Buffer* src2_buffer` |
| input_grad (optional output) | writer | s0 | `Buffer* dst0_buffer` / `0u` when absent |
| other_grad (optional output) | writer | s1 | `Buffer* dst1_buffer` / `0u` when absent |

All Case 1 (accessed only through `TensorAccessor`). Both writer accessors are 2-arg (no page-size 3rd arg).

### Work split
None — single core `{0,0}`, `num_tiles = input.physical_volume() / TILE_HW`. No `split_work_to_cores`.

### Cross-op kernels
None — all three kernels op-owned; every `#include` resolves to `tt_metal/*` HAL/LLK.

### Flags
No unreferenced kernel files. No descriptor types outside the audit's scan.

## TTNN ProgramFactory
- Concept (inherited from audit): `MetalV2FactoryConcept`.
- Custom `compute_program_hash`: none.
- Implementation notes: The legacy op is on the `HasDirectDescriptor` shape (create_descriptor on the
  device-op, no `program_factory_t`). `MetalV2FactoryConcept` requires `create_program_artifacts` on a
  factory referenced by a `program_factory_t` variant (a device-op with `create_program_artifacts` but
  no `program_factory_t` does NOT satisfy `DeviceOperationConcept`). So the port introduces a nested
  `ProgramFactory` struct carrying `create_program_artifacts`, a single-type
  `program_factory_t = std::variant<ProgramFactory>`, and a `select_program_factory`. This is the forced
  concept-migration wiring, not a freelance device-op edit.

## Planned Spec Shape
- **KernelSpecs**: 3 — READER, WRITER, COMPUTE (1:1 with legacy). No work-split multiplicity.
- **DataflowBufferSpecs**: 5 — IN0, IN1, IN2 (entry_size=cb_tile, num_entries=2), OUT0, OUT1 (same).
  All carry `data_format_metadata = cb_data_format` (all are compute-bound). No borrowed_from, no aliasing.
- **SemaphoreSpecs**: none.
- **TensorParameters**: 5 — OUTPUT_GRAD, INPUT, OTHER (always); INPUT_GRAD, OTHER_GRAD (conditional).
- **WorkUnitSpecs**: 1 — {READER, WRITER, COMPUTE} on `{0,0}`.
- **Op-owned tensors**: none.

DFB bindings:
- reader: IN0/IN1/IN2 PRODUCER (accessor `in0`/`in1`/`in2`).
- compute: IN0/IN1/IN2 CONSUMER; OUT0/OUT1 PRODUCER (`in0`..`in2`,`out0`,`out1`).
- writer: OUT0/OUT1 CONSUMER (`out0`/`out1`).

Tensor bindings:
- reader: OUTPUT_GRAD→`s0`, INPUT→`s1`, OTHER→`s2` (always).
- writer: INPUT_GRAD→`s0` (iff has_input_grad), OTHER_GRAD→`s1` (iff has_other_grad).

## Preserved Multiplicity
None — no multi-`KernelDescriptor` work split in legacy.

## Dropped Plumbing
- **Buffer-address RTAs** → `TensorBinding`:
  - reader RTA slots 2,3,4 (`src0_buffer`,`src1_buffer`,`src2_buffer`) → OUTPUT_GRAD/INPUT/OTHER bindings.
  - writer RTA slots 2,3 (`dst0_buffer`/`0u`, `dst1_buffer`/`0u`) → INPUT_GRAD/OTHER_GRAD bindings (conditional).
- **Magic CB indices in CTAs** → `DFBBinding`:
  - writer CTA slots 0,1 (`c_16`, `c_17`) → OUT0/OUT1 DFB bindings.
- **`TensorAccessorArgs` plumbing** → binding mechanism:
  - reader: `TensorAccessorArgs(src0/1/2).append_to(reader_ct_args)` (all reader CTAs) + kernel-side
    `TensorAccessorArgs<0>()` / `<next…>()` chain.
  - writer: `TensorAccessorArgs(dst0/dst1).append_to(writer_ct_args)` (after the 2 CB indices) + kernel-side
    `TensorAccessorArgs<2>()` / `<next…>()` chain.
- **Page-size 3rd-argument CTAs/RTAs**: none.
- **Semaphore-ID RTAs**: none.
- **Positional CTAs**: after the above, reader/writer/compute have NO surviving CTAs (all become bindings);
  no named CTAs are introduced.

## Applied Patterns
- [Conditional / optional DFB (here: TENSOR) bindings]: INPUT_GRAD (writer `s0`) and OTHER_GRAD (writer `s1`)
  are `std::optional`; bound only when present. The selecting condition moves from the writer's runtime
  `has_input_grad`/`has_other_grad` RTAs to writer `compiler_options.defines` (`HAS_INPUT_GRAD` /
  `HAS_OTHER_GRAD`), and the writer kernel `#ifdef`-gates the `tensor::s0`/`tensor::s1` construction and
  each output's write block. (Rule 6.)
- Unconditional endpoint declaration for the guarded pipeline: the OUT0/OUT1 (and IN1/IN2) DFBs are bound
  1P+1C unconditionally; the runtime `has_*_grad` guards in reader/compute gate whether they are exercised.
  This is the sanctioned "declare the conditional-side endpoint unconditionally" shape — not a topology
  imbalance to fix in the kernel.

## Deferred / Flagged
- New findings: none. Planning surfaced nothing the audit missed.
- Note (mirrors audit misc anomalies, not porter-actionable): `start_id` RTA is always `0`; ported
  faithfully. `tensor_args_t::output_tensors` vestigial comment left untouched (op-level, off-limits).
