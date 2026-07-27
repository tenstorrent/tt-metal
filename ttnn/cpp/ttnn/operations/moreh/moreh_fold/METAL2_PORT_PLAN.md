# Port Plan — moreh/moreh_fold

Port plan for `moreh/moreh_fold` (`MorehFoldOperation`), ported from the legacy
`ProgramDescriptor` (`descriptor`) concept to Metal 2.0 (`MetalV2FactoryConcept`).
Written during the inventory and planning steps; committed alongside the port for review.

Single device operation, single factory. Audit cleared **GREEN**.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` — `create_descriptor()` returns `tt::tt_metal::ProgramDescriptor`,
  placed **directly** on `MorehFoldOperation` (the `HasDirectDescriptor` shape; no `program_factory_t` variant).
- Variants: single (`MorehFoldOperation (single-descriptor)`, `device/fold_program_factory_rm.cpp`).
- Custom `compute_program_hash`: none — already default reflection-based hash (audit confirmed, grep clean).

*(Target concept chosen in audit: `MetalV2FactoryConcept`. Carried forward in [TTNN ProgramFactory](#ttnn-programfactory).)*

### Kernels
| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | config |
|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/reader_fold_rm.cpp` | `all_cores` | `{input_cb(0), output_cb(1), scratch_cb(2)}` + `TensorAccessorArgs<3>(input.buffer())` | none | `[0]input.buffer()` (Buffer* base), `[1..14]` N,C,H,W,kh,kw,sh,sw,ph,pw,dh,dw,LH,LW, `[15]input_cb_page_size`, `[16]dram_aligned_input_cb_page_size`, `[17]aligned_output_cb_page_size`, `[18]start_id`, `[19]num_units_per_core`, `[20]aligned` | none | `DTYPE_BFLOAT16`/`DTYPE_FLOAT32` (per input dtype) | `ReaderConfigDescriptor{}` |
| writer | `device/kernels/writer_fold_rm.cpp` | `all_cores` | `{output_cb(0)}` + `TensorAccessorArgs<1>(output.buffer())` | none | `[0]output.buffer()` (Buffer* base), `[1]aligned_output_cb_page_size`, `[2]start_id`, `[3]num_units_per_core` | none | none | `WriterConfigDescriptor{}` |

### CBs
| index | total_size | core_ranges | data_format | page_size | tile |
|---|---|---|---|---|---|
| c_0 (input) | `aligned_input_cb_page_size * 2` | all_cores | input dtype | `aligned_input_cb_page_size` | (unset) |
| c_1 (scratch) | `4 * dram_aligned_input_cb_page_size` | all_cores | input dtype | `dram_aligned_input_cb_page_size` | (unset) — **conditional**: only allocated when `(src_is_dram && input_cb_page_size % dram_alignment != 0) \|\| is_blackhole` |
| c_16 (output) | `aligned_output_cb_page_size * 2` | all_cores | input dtype | `aligned_output_cb_page_size` | (unset) |

### Semaphores
none — op uses no semaphores.

### Tensor accessors
| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `reader_fold_rm.cpp:49` `TensorAccessor(input_args, input_addr, input_cb_page_size)` | input (io) | reader RTA[0] = `input.buffer()` |
| `writer_fold_rm.cpp:24` `TensorAccessor(output_args, output_addr, output_cb_page_size)` | output (io) | writer RTA[0] = `output.buffer()` |

Both **Case 1** (accessed via `TensorAccessor` page ops). Both carry a redundant **3rd (page-size) arg → dropped**.

### Work split
- Driver: `split_work_to_cores(grid, num_units)` where `num_units = output.logical_volume() / output.logical_shape()[-1]`.
- `num_cores, all_cores, core_group_1, core_group_2, num_units_per_core_group_1, num_units_per_core_group_2`.
- Per-core RTAs vary only `start_id` and `num_units_per_core`; all other reader/writer RTAs are the same value on every node.
- **No same-source work-split multiplicity** — reader and writer are one `KernelDescriptor` each, over one `all_cores` grid.

### Cross-op kernels
none — both kernels are op-owned; `#include` only `tt_metal` HAL/firmware headers (audit-confirmed).

### Flags
- `reader_fold_rm.cpp:15` `int i{0};` — pre-existing dead local (audit misc anomaly). Left as-is.
- `reader_fold_rm.cpp:91` `uint32_t l1_write_addr = input_dfb.get_write_ptr();` — dead local. Left as-is.
- reader RTA[17] `output_cb_page_size` — **dead read in the reader** (declared, never used). Left as-is (faithful 1:1).

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `MetalV2FactoryConcept`.
- **Custom `compute_program_hash`**: none.
- **Implementation notes**: legacy puts `create_descriptor` **directly** on `MorehFoldOperation` (no `program_factory_t`).
  The port introduces a single-alternative `program_factory_t = std::variant<MultiCore>` with the factory struct
  `MorehFoldOperation::MultiCore::create_program_artifacts`, plus a `select_program_factory` returning `MultiCore{}`.
  This is the standard wiring; `create_program_artifacts` can only be reached by the framework as a variant
  alternative (the concept is not detected directly on the op struct).

## Planned Spec Shape

- **KernelSpecs**: `reader` (1:1 with legacy reader), `writer` (1:1 with legacy writer).
- **DataflowBufferSpecs**: `INPUT_CB` (c_0), `OUTPUT_CB` (c_16), and `SCRATCH_CB` (c_1) — the last **conditional** on `use_scratch`.
- **SemaphoreSpecs**: none.
- **TensorParameters**: `INPUT` (input io), `OUTPUT` (output io).
- **WorkUnitSpecs**: one — `main`, kernels `{reader, writer}`, `target_nodes = all_cores`.

## Preserved Multiplicity

none — no work-split multiplicity in legacy (one reader + one writer `KernelDescriptor`, single grid).

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `fold_program_factory_rm.cpp:134` reader CTA[0] | `input_cb_index` | `DFBBinding(INPUT_CB, "input", PRODUCER+CONSUMER self-loop)` |
| `fold_program_factory_rm.cpp:134` reader CTA[1] | `output_cb_index` | `DFBBinding(OUTPUT_CB, "output", PRODUCER)` |
| `fold_program_factory_rm.cpp:134` reader CTA[2] | `scratch_cb_index` | `DFBBinding(SCRATCH_CB, "scratch", PRODUCER+CONSUMER self-loop)` (conditional) |
| `fold_program_factory_rm.cpp:135` | `TensorAccessorArgs(input.buffer()).append_to(reader_ct_args)` | `TensorBinding(INPUT, "input")` |
| `fold_program_factory_rm.cpp:174` reader RTA[0] | `input.buffer()` (Buffer*-binding base) | `TensorBinding(INPUT)` (base auto-injected) |
| `reader_fold_rm.cpp:41` | `TensorAccessorArgs<3>()` | dropped (binding supplies layout) |
| `reader_fold_rm.cpp:49` | `TensorAccessor(input_args, input_addr, input_cb_page_size)` 3rd arg | `TensorAccessor(tensor::input)` (aligned page size implicit) |
| `fold_program_factory_rm.cpp:146` writer CTA[0] | `output_cb_index` | `DFBBinding(OUTPUT_CB, "output", CONSUMER)` |
| `fold_program_factory_rm.cpp:147` | `TensorAccessorArgs(output.buffer()).append_to(writer_ct_args)` | `TensorBinding(OUTPUT, "output")` |
| `fold_program_factory_rm.cpp:197` writer RTA[0] | `output.buffer()` (Buffer*-binding base) | `TensorBinding(OUTPUT)` (base auto-injected) |
| `writer_fold_rm.cpp:18` | `TensorAccessorArgs<1>()` | dropped |
| `writer_fold_rm.cpp:24` | `TensorAccessor(output_args, output_addr, output_cb_page_size)` 3rd arg | `TensorAccessor(tensor::output)` |
| all reader/writer positional RTAs | positional `get_arg_val<uint32_t>(N)` | named `get_arg(args::name)` |

**NOT dropped (audit/brief disagreement, following the census):** `input_cb_page_size` (reader RTA[15]) and
`output_cb_page_size` (writer RTA[1]) are **not** dead — they are the NOC transfer sizes at
`reader_fold_rm.cpp:96,113` and `writer_fold_rm.cpp:31`, distinct from their (dropped) 3rd-arg feed. They are kept
as named RTAs. See METAL2_PORT_REPORT.md → Friction.

## Applied Patterns

- [Self-loop DFB binding](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md): `INPUT_CB` on the reader (one toucher — full FIFO on the reader alone; PRODUCER+CONSUMER, shared accessor "input").
- [Sync-free CB → self-loop DFB](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md): `SCRATCH_CB` on the reader (one toucher — raw `get_write_ptr` peek only; PRODUCER+CONSUMER).
- Legal 1:1: `OUTPUT_CB` — reader PRODUCER (`reserve_back`/`push_back`), writer CONSUMER (`wait_front`/`pop_front`).
- [Conditional / optional DFB bindings](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md): `SCRATCH_CB` conditionally bound; matching `HAS_SCRATCH_CB` define on the reader; kernel `#ifdef`-gates the scratch DFB construction and its two-step-read use.

## Deferred / Flagged

- New findings during planning: the brief's "drop the `input_cb_page_size` / `output_cb_page_size` RTAs — they are
  dead once the base moves to a binding" is incorrect; those RTAs are the NOC transfer sizes and must be kept. Only
  the *3rd constructor argument* and the *address RTA* are dropped. Following the kernel census per the recipe.
