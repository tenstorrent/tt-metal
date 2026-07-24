# Port Plan — moreh_dot

Port plan for `ttnn/cpp/ttnn/operations/moreh/moreh_dot`, ported from the legacy
`ProgramDescriptor` (`ProgramDescriptorFactoryConcept`, single-descriptor / `HasDirectDescriptor`)
API to Metal 2.0 (`MetalV2FactoryConcept`).
Written during the inventory and planning steps; committed alongside the port for review.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` — realized as `HasDirectDescriptor`
  (`MorehDotOperation::create_descriptor` is a static method **directly on the device-op
  struct**; there is no `program_factory_t` variant). File: `device/moreh_dot_program_factory.cpp:22`.
- Variants: single (one factory, single core `{0,0}`).
- Custom `compute_program_hash`: none — already default reflection-based hash (confirmed in
  `device/moreh_dot_device_operation.hpp` / `.cpp`; audit `Custom hash = no`).

*(Target Metal 2.0 concept chosen by the audit: `MetalV2FactoryConcept`. Carried forward in
[TTNN ProgramFactory](#ttnn-programfactory) below.)*

### Kernels
| unique_id | source | core_ranges | CTAs (positional) | RTAs | defines | config |
|---|---|---|---|---|---|---|
| reader | `device/kernels/reader_moreh_dot.cpp` | `{0,0}` | `TensorAccessorArgs(src0)` + `TensorAccessorArgs(src1)` | `src0_addr(0), src1_addr(1), num_tiles(2), start_id=0(3), mask_h(4), mask_w(5)` | none | `ReaderConfigDescriptor{}` |
| writer | `device/kernels/writer_moreh_dot.cpp` | `{0,0}` | `c_16 (=CBIndex)` + `TensorAccessorArgs(dst)` | `dst_addr(0), num_tiles=1(1), start_id=0(2)` | none | `WriterConfigDescriptor{}` |
| compute | `device/kernels/moreh_dot.cpp` | `{0,0}` | none | `per_core_block_cnt=num_tiles(0), 1u(1) [DEAD — unread]` | `REDUCE_OP=PoolType::SUM`, `REDUCE_DIM=ReduceDim::REDUCE_ROW` | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en, dst_full_sync_en, math_approx_mode}` |

Compute config knobs come from `get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config)` (Style A — a TTNN `ComputeKernelConfig`).

### CBs (all single-core `{0,0}`, all `data_format = datatype_to_dataformat_converter(input_a.dtype())`, `page_size = tile_size(fmt)`)
| CB (buffer_index) | total_size (tiles) | tile field set? | producer / consumer |
|---|---|---|---|
| `c_0` (in0) | 2 | no | reader produces → compute consumes |
| `c_1` (in1) | 2 | no | reader produces → compute consumes |
| `c_2` (scaler) | 1 | no | reader produces (`calculate_and_prepare_reduce_scaler`) → compute consumes |
| `c_16` (out) | 2 | no | compute produces (reduce output, last block) → writer consumes |
| `c_24` (im0) | 1 | no | compute only (pack `mul` result, read back as reduce input) — **self-loop** |
| `c_25` (im1) | 1 | no | compute only (reduce accumulation CB) — **self-loop** |

No GlobalCircularBuffer, no borrowed-memory CB, no aliased CB, no `tile` field set anywhere.

### Semaphores
None.

### Tensor accessors (all Case 1, via `TensorAccessor`)
| tensor | origin | host RTA slot | kernel |
|---|---|---|---|
| `input_a` (src0) | input | reader RTA idx 0 | `TensorAccessor(src0_args, src0_addr)` (`reader_moreh_dot.cpp:80`) |
| `input_b` (src1) | input | reader RTA idx 1 | `TensorAccessor(src1_args, src1_addr)` (`reader_moreh_dot.cpp:82`) |
| `output` (dst) | output | writer RTA idx 0 | `TensorAccessor(dst_args, dst_addr)` (`writer_moreh_dot.cpp:21`) |

### Work split
None — single core `{0,0}`, `num_tiles = input_a.physical_volume() / TILE_HW`. No `split_work_to_cores`.

### Cross-op kernels
None — all three kernel sources are op-owned. (`moreh_dot_backward` is a separate op with its own kernels.)

### Runtime kernel-source selection
None — one fixed source per kernel.

### Flags
- Dead compute RTA index 1 (`1u`) at `moreh_dot_program_factory.cpp:153`, never read by `moreh_dot.cpp` — dropped in the port (not carried, not named). Routed to ops team in the report.
- No unreferenced kernel files.

## TTNN ProgramFactory
- Concept (inherited from audit): `MetalV2FactoryConcept`.
- Custom `compute_program_hash`: none.
- Implementation notes: the op currently satisfies `HasDirectDescriptor` (bare `create_descriptor`
  on the struct). Metal 2.0 requires the factory to live in a `program_factory_t` variant, so the
  port introduces a nested `struct ProgramFactory` with a static `create_program_artifacts`, plus
  `using program_factory_t = std::variant<ProgramFactory>;`. No custom `select_program_factory`
  needed (single-alternative variant → framework default). The `create_descriptor` method is removed.

## Planned Spec Shape
- **KernelSpecs** (3, 1:1 with legacy): `reader`, `writer`, `compute`.
- **DataflowBufferSpecs** (6, 1:1 with legacy CBs): `IN0`, `IN1`, `SCALER`, `OUT`, `IM0`, `IM1`.
  `entry_size = tile_size(fmt)`, `num_entries` = legacy total tiles (2/2/1/2/1/1),
  `data_format_metadata = fmt` (all are compute-bound). `tile_format_metadata` = nullopt (legacy `.tile` unset).
- **SemaphoreSpecs**: none.
- **TensorParameters** (3): `input_a`, `input_b`, `output`.
- **WorkUnitSpecs** (1): `{reader, writer, compute}` on `NodeCoord{0,0}`.
- **Op-owned tensors**: none.

### DFB bindings
| DFB | reader | compute | writer |
|---|---|---|---|
| IN0 | PRODUCER `in0` | CONSUMER `in0` | — |
| IN1 | PRODUCER `in1` | CONSUMER `in1` | — |
| SCALER | PRODUCER `scaler` | CONSUMER `scaler` | — |
| OUT | — | PRODUCER `out` | CONSUMER `out` |
| IM0 | — | PRODUCER+CONSUMER `im0` (self-loop) | — |
| IM1 | — | PRODUCER+CONSUMER `im1` (self-loop) | — |

### Tensor bindings
- reader: `input_a`→`src0`, `input_b`→`src1`.
- writer: `output`→`dst`.
- compute: none (compute kernels cannot bind TensorAccessors; none needed here).

## Preserved Multiplicity
None — no multi-`KernelDescriptor` work split in legacy.

## Dropped Plumbing
- **Buffer-address RTAs** → `TensorBinding`:
  - reader RTA idx 0 (`src0_buffer` / `src0_addr`) → `tensor::src0`.
  - reader RTA idx 1 (`src1_buffer` / `src1_addr`) → `tensor::src1`.
  - writer RTA idx 0 (`dst_buffer` / `dst_addr`) → `tensor::dst`.
- **`TensorAccessorArgs` plumbing** (host `append_to` + kernel `TensorAccessorArgs<N>()`):
  - reader host `moreh_dot_program_factory.cpp:118,119`; kernel `reader_moreh_dot.cpp:70,71`.
  - writer host `:131`; kernel `writer_moreh_dot.cpp:16`.
- **Magic CB index in CTA** → `DFBBinding`:
  - writer CTA slot 0 (`CBIndex::c_16`) → `dfb::out` binding; kernel `writer_moreh_dot.cpp:15` (`cb_id_out`) dropped.
  - reader/compute hard-coded `cb_id_in0/in1/in2` and `tt::CBIndex::c_*` constants → `dfb::` handles.
- **Positional CTAs** → named CTAs: none remain after the above (no scalar CTAs survive on any kernel).
- **Page-size 3rd-arg CTAs/RTAs**: none.
- **Semaphore-ID RTAs**: none.
- **Dead RTA**: compute RTA idx 1 (`1u`) dropped.
- Surviving RTAs become **named**:
  - reader: `num_tiles`, `start_id`, `mask_h`, `mask_w`.
  - writer: `num_tiles`, `start_id`.
  - compute: `per_core_block_cnt`.

## Applied Patterns
- [Self-loop DFB binding](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-self-loop-dfb-binding):
  `IM0`, `IM1` on the compute KernelSpec (both PRODUCER and CONSUMER, one-toucher). Shared accessor name form.
- Metadata-via-object (whitelist rule 7): `get_tile_size(cb_id)` → `dfb.get_tile_size()` in reader/writer.
- `dfb::name`→`uint32_t` implicit conversion at kernel-lib template-NTTP call sites
  (`calculate_and_prepare_reduce_scaler<dfb::scaler,...>`, `compute_kernel_lib::reduce<..., dfb::im0, dfb::scaler, dfb::out/im1, ...>`).

## Deferred / Flagged
None — no structural surprises beyond the pre-recorded dead compute RTA.
