# Port Plan — topk

Port plan for `reduction/topk`, ported from `ProgramDescriptor` to Metal 2.0.
Single-core factory ported; multi-core factory deferred on legacy (grounded stop — see audit/report).

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` (both factories return `tt::tt_metal::ProgramDescriptor`)
- Variants: two — `TopKSingleCoreProgramFactory`, `TopKMultiCoreProgramFactory` (selected by `select_program_factory` on dim size / K / cost)
- Custom `compute_program_hash`: none — default reflection hash
- `tensor_return_value_t = std::tuple<Tensor, Tensor>` (value, index); `tensor_args` = input + optional `indices` + optional preallocated outputs

### Single-core kernels
| unique_id | source | core_ranges | CTAs (named) | RTAs | config |
|---|---|---|---|---|---|
| reader | `dataflow/reader_create_index_tensor.cpp` | core_range | Ht, Wt, total_number_of_cores, uint16_output | id, work_per_core | DM READER, `GENERATE_INDICES=1` |
| writer | `dataflow/writer_binary_interleaved.cpp` | core_range | Ht, Kt, total_number_of_cores | id, work_per_core | DM WRITER |
| compute | `compute/topk.cpp` | core_range | Ht, Wt, Ktiles, largest | work_per_core | Compute, fp32_dest_acc_en=!uint16_output |

### Single-core CBs (all normal / non-borrowed)
| index | DFB name | num_entries | data_format |
|---|---|---|---|
| c_0 | cb_in0 | 2 | input |
| c_1 | cb_index | 2 | index |
| c_2 | transposed_val | 4 | compute (bf16 for bfp8/bfp4) |
| c_3 | transposed_ind | 4 | index |
| c_4 | result_prep_val | 2·Ktiles | compute |
| c_5 | result_prep_ind | 2·Ktiles | index |
| c_6 | output_val | Ktiles | value |
| c_7 | output_ind | Ktiles | index |

### Tensor accessors (single-core)
| binding | originating Tensor | legacy form | Metal 2.0 |
|---|---|---|---|
| input | tensor_args.input | RTA slot 0 (Tensor) → src_addr; `TensorAccessorArgs<6>` | TensorParameter `input` + `ta::input` (Case 1) |
| value | output[0] | RTA slot 0 → dst_addr0 | TensorParameter `value` + `ta::value` (Case 1) |
| index | output[1] | RTA slot 1 → dst_addr1 | TensorParameter `index` + `ta::index` (Case 1) |
| indices (opt) | tensor_args.indices | RTA slot 3 (dead, `#if not GENERATE_INDICES`) | `ta::indices` only inside dead branch; not bound on the always-on path |

### Semaphores (single-core)
none

### Work split (single-core)
`split_work_to_cores(sub_core_grids, Ht, true)` → two core groups; per-core RTAs `id` (core offset) + `work_per_core`.

## TTNN ProgramFactory
- **Concept (single-core)**: `ProgramSpecFactoryConcept` (`create_program_spec`)
- **Concept (multi-core)**: kept on legacy `create_descriptor`
- **Custom `compute_program_hash`**: none
- **Device-op `.hpp`**: `TopKSingleCoreProgramFactory::create_descriptor` → `create_program_spec` returning `ttnn::device_operation::ProgramArtifacts`; added `#include "ttnn/device_operation.hpp"` + `#include "ttnn/metal2_artifacts.hpp"`; kept `<tt-metalium/program_descriptors.hpp>` for the still-legacy multi-core factory.

## Planned Spec Shape (single-core)
- **KernelSpecs**: `reader` (READER), `writer` (WRITER), `compute` (Compute).
- **DataflowBufferSpecs**: 8 normal DFBs (table above).
- **SemaphoreSpecs**: none.
- **TensorParameters**: `input`, `value`, `index`.
- **WorkUnitSpec**: one — `{reader, writer, compute}` on `core_range`.

### DFB endpoint bindings
- `cb_in0`, `cb_index`: reader PRODUCER, compute CONSUMER.
- `transposed_val`, `transposed_ind`, `result_prep_val`, `result_prep_ind`: compute PRODUCER **and** CONSUMER (real staging self-loops).
- `output_val`, `output_ind`: compute PRODUCER, writer CONSUMER.

### Tensor bindings
- `input` (`ta::input`) on reader; `value`/`index` (`ta::value`/`ta::index`) on writer.

## Dropped Plumbing (single-core)
| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader/writer/compute CTA CB-index slots | magic CB indices | `DFBBinding`s + `dfb::name` in kernel |
| reader RTA slot 0 | input Tensor (`src_addr`) | `TensorBinding(input)` + `ta::input` |
| writer RTA slots 0,1 | value/index Tensors | `TensorBinding(value/index)` + `ta::value`/`ta::index` |
| reader RTA slot 3 | optional indices addr (dead) | dropped; `#ifdef GENERATE_INDICES`-gated `ta::indices` |
| all CTAs | `get_compile_time_arg_val(N)` | named CTAs `get_arg(args::…)` |
| all RTAs | `get_arg_val<uint32_t>(N)` | named RTAs `get_arg(args::…)` |
| compute `uint32_t cb0..cb3` reassigned per case | magic CB-index arithmetic | `(uint32_t)dfb::name` constexpr aliases + `DataflowBuffer obj(cbX)` (uint16_t ctor) + `dfb::name` to LLKs |

## Applied Patterns
- Runtime-dynamic CB selection: `(uint32_t)dfb::name` aliases → `DataflowBuffer obj(cbX)` + LLK pass-through.
- Pass DFB handles directly to LLKs / `generate_index_tile` (implicit `DFBAccessor::operator uint32_t`).
- Real self-loop DFBs for the `c_2..c_5` staging buffers (PRODUCER+CONSUMER on compute).
- Conditional/optional binding: `GENERATE_INDICES` define + `#ifdef`-gated `ta::indices` (binding omitted on the live path).

## Deferred / Flagged
- **Multi-core factory**: grounded stop (cross-core remote-CB write via `get_write_ptr()` + NoC, custom semaphore multicast, allocation-order-pinned L1 layout). Left on legacy `create_descriptor`. See audit + report.
