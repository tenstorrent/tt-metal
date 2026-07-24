# Port Plan — scatter (`data_movement/scatter`)

Port plan for `ttnn/cpp/ttnn/operations/data_movement/scatter`, ported from the
`ProgramDescriptor` API (`create_descriptor`) to Metal 2.0 (`create_program_artifacts`).
Written during the inventory and planning steps; committed alongside the port for review.

Both program factories are ported together in this pass: they share the device operation,
the kernel-side common header (`device/kernels/common.hpp`), and the addressing model, so a
change to the shared vararg helper forces a joint port (see [Deferred / Flagged](#deferred--flagged)).

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` — both factories expose
  `create_descriptor(...)` returning a `tt::tt_metal::ProgramDescriptor`.
- Variants: the device operation `ScatterDeviceOperation` holds a two-alternative
  `program_factory_t = std::variant<ScatterProgramFactory, ScatterReduceBfloat16ProgramFactory>`.
  `select_program_factory` picks the reduce factory when a reduction is requested **and** the
  input dtype is `BFLOAT16` (`scatter_device_operation.cpp:15-22`), otherwise the general factory.
- Custom `compute_program_hash`: none — already the default reflection-based hash (grep clean).

*(Target concept `MetalV2FactoryConcept` was chosen during the audit; carried forward in
[TTNN ProgramFactory](#ttnn-programfactory) below.)*

The kernels are **already on Device 2.0 object wrappers** (`Noc`, `DataflowBuffer`,
`TensorAccessor`, `CoreLocalMem`, `UnicastEndpoint`). There is therefore **no
`CircularBuffer` → `DataflowBuffer` swap to do** on the kernel side; the port is a host-side
spec/binding rewrite plus replacing the numeric CB/DFB indices and RTA-delivered base
addresses with named `dfb::*` / `tensor::*` bindings, plus a named-argument conversion.

### Variant: `ScatterProgramFactory` (general path — all supported dtypes)

#### Kernels
| unique_id | source | core_ranges | CTAs (positional) | RTAs | config |
|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_scatter.cpp` | `all_cores` | 0-3: input/index/src/output **buffer addresses (DEAD, never read)**; 4-7: `INPUT`/`INDEX`/`SRC`/`DST` CB indices; 8-11: input/index/source/output_stick_size; 12-15: input/index/source/output_stick_size_bytes; 16: input_rank; then 4× `TensorAccessorArgs` blocks (input/index/src/output) | 0-2: input/index/src `Buffer*`; 3: stick_offset; 4: sticks_per_core; 5: input_and_output_chunk_size; 6: index_chunk_size; 7: source_chunk_size; 8: reduction; 9..9+N-1: input shape dims; 9+N..9+2N-1: index shape dims (N = input_rank-1) | `ReaderConfigDescriptor{}` |
| writer | `device/kernels/dataflow/writer_scatter.cpp` | `all_cores` | same shared CTA list as reader (both kernels get the same vector) | 0: output `Buffer*`; 1: stick_offset; 2: sticks_per_core; 3: input_and_output_chunk_size | `WriterConfigDescriptor{}` |

Reader CTA fields **actually read** by kernels: `input_dfb`/`index_dfb`/`source_dfb`/`output_dfb`
(CB indices), `input_stick_size`, `index_stick_size`, `source_stick_size`, `input_rank`, plus the
four `TensorAccessorArgs`. Writer reads `output_dfb`, `output_stick_size_bytes`, `output_args`.
Dead CTAs (never read by any kernel): the four buffer addresses (0-3), `output_stick_size`, and
`input`/`index`/`source_stick_size_bytes`.

#### CBs
| enum (index) | total_size | num_entries | core_ranges | data_format | page_size |
|---|---|---|---|---|---|
| `INPUT` (c_0) | `input_page_size_bytes` | 1 | all_cores | `input.dtype()` | `input_page_size_bytes` |
| `SRC` (c_1) | `source_page_size_bytes` | 1 | all_cores | `src.dtype()` | `source_page_size_bytes` |
| `INDEX` (c_2) | `index_page_size_bytes` | 1 | all_cores | `index.dtype()` | `index_page_size_bytes` |
| `DST` (c_3) | `output_page_size_bytes` | 1 | all_cores | `output.dtype()` | `output_page_size_bytes` |

(`total_size == page_size` for every CB, so each holds exactly one entry — `num_entries = 1`.
No `.tile` set on any format descriptor. No `GlobalCircularBuffer`, no `address_offset`.)

#### Semaphores
none.

#### Tensor accessors
| host site (file:line) | originating Tensor | delivered as |
|---|---|---|
| `reader_scatter.cpp:116` `TensorAccessor(ctas.input_args, input_buffer_address)` | input (tensor_args) | reader RTA slot 0 (`Buffer*`) + CTA `TensorAccessorArgs` |
| `reader_scatter.cpp:117` `TensorAccessor(ctas.index_args, index_buffer_address)` | index (tensor_args) | reader RTA slot 1 |
| `reader_scatter.cpp:118` `TensorAccessor(ctas.source_args, source_buffer_address)` | src (tensor_args) | reader RTA slot 2 |
| `writer_scatter.cpp:17` `TensorAccessor(ctas.output_args, output_buffer_address)` | output (tensor_return_value) | writer RTA slot 0 |

All four are **Case 1** (fed straight into a `TensorAccessor`), so each becomes a
`TensorParameter` + `TensorBinding`; kernel builds `TensorAccessor(tensor::name)`.

#### Work split
- Driver: `split_work_to_cores(grid_or_sub_core_grid, work_units)` where
  `work_units = input.logical_volume() / input_stick_size`.
- Yields `(num_cores, all_cores, core_group_1, core_group_2, num_sticks_per_core_group_1, num_sticks_per_core_group_2)`.
- **No per-group CTA variation.** One reader `KernelDescriptor` and one writer `KernelDescriptor`,
  each over `all_cores`; the per-group stick count differs only in the per-core **RTA** values
  (`stick_offset`, `sticks_per_core`) set in the bounding-box loop. → single `WorkUnitSpec`.

### Variant: `ScatterReduceBfloat16ProgramFactory` (bf16 + reduction path)

Identical in structure to the general factory, with these deltas:
- Kernel sources: `reader_bf16_reduction_scatter.cpp`, `writer_bf16_reduction_scatter.cpp`.
- One extra CB: `FP32_TEMP` (c_4), `data_format = FLOAT32`, `total_size = page_size = fp32_temp_page_size_bytes`, `num_entries = 1`. fp32 scratch for accumulating the reduction before bf16 down-conversion.
- Reader CTA list has the five CB indices (0-3 dead addrs, 4-8 CB indices incl. FP32_TEMP), so the
  `TensorAccessorArgs` blocks begin at CTA offset **18** (`scatter_bf16_reduction_common.hpp:43`),
  vs **17** in the general factory (`scatter_common.hpp:42`).
- Kernel-read CTA fields add `fp32_temp_dfb`.

### Cross-op kernels
none — the op owns all four kernels and all three kernel-side common headers. The only
cross-directory `#include`s are `tt_metal` LLK/HAL (`api/dataflow/*`, `api/core_local_mem.h`,
`api/numeric/bfloat16.h`).

### Flags
- **Two `scatter_common.hpp` files** (different directories, same basename): the host-side
  `device/scatter_common.hpp` (`ScatterCB` enum + `ceil32` + `calculate_optimal_chunk_size`,
  included by the factories) and the kernel-side `device/kernels/scatter_common.hpp`
  (`ScatterCTAs` + `get_ctas()`, included by the general kernels). They are unrelated.
- No unreferenced kernel files. No descriptor type outside the audit's scan.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `MetalV2FactoryConcept` (both factories).
- **Custom `compute_program_hash`**: none — nothing to delete.
- **Implementation notes**: The device-operation class (`validate_on_program_cache_miss`,
  `compute_output_specs`, `create_output_tensors`, `select_program_factory`) is unchanged — no
  custom hash, no pybind `create_descriptor`, no pybind-hook-only factory parameter. Only the two
  factory `.hpp`/`.cpp` files change signature (`create_descriptor` → `create_program_artifacts`).

## Planned Spec Shape

> Both factories share this shape; the reduce factory adds the FP32_TEMP DFB.

### Variant: `ScatterProgramFactory`
- **KernelSpecs**: `reader` (`reader_scatter.cpp`), `writer` (`writer_scatter.cpp`). One each; no multiplicity.
- **DataflowBufferSpecs**: `INPUT`, `INDEX`, `SRC`, `DST` (`entry_size = <page_size_bytes>`,
  `num_entries = 1`, `data_format_metadata = <tensor dtype>`). `data_format_metadata` is
  **required** even though these are DM-only DFBs, because the kernels select their C++ POD type at
  compile time via `get_dataformat(dfb::name)` (reads `unpack_src_format[]`, fed by this field —
  `program_spec.cpp:2617`).
- **SemaphoreSpecs**: none.
- **TensorParameters**: `input`, `index`, `source`, `output` (from each tensor's `tensor_spec()`).
- **WorkUnitSpecs**: one — `{reader, writer}` over `all_cores`.

### Variant: `ScatterReduceBfloat16ProgramFactory`
As above, plus **`FP32_TEMP`** DFB (`entry_size = fp32_temp_page_size_bytes`, `num_entries = 1`,
`data_format_metadata = Float32`). Kernel sources are the `*_bf16_reduction_*` pair.

### DFB endpoint dispositions (re-derived from the kernel-touch census)
| DFB | touchers | disposition |
|---|---|---|
| `INPUT` | reader only — fills (`load_to_dfb`: `reserve_back`/`push_back`) **and** drains (`wait_front`/`get_read_ptr`/`pop_front`) | **self-loop** (reader PRODUCER + CONSUMER) |
| `INDEX` | reader only (fills + drains) | **self-loop** |
| `SRC` | reader only (fills + drains) | **self-loop** |
| `DST` | reader produces (`reserve_back`/`get_write_ptr`/`push_back`); writer consumes (`write_to_output`: `wait_front`/`get_read_ptr`/`pop_front`) | **1:1** (reader PRODUCER, writer CONSUMER) |
| `FP32_TEMP` (reduce only) | reduce-reader only (`reserve_back`/`get_write_ptr`/`push_back`, then `wait_front`/`get_read_ptr`/`pop_front`) | **self-loop** |

Census matches the brief exactly. No dead CB, no hidden second toucher, no ≥3-toucher or
same-role-locked case — **the multi-binding flag is never used.**

## Preserved Multiplicity

none — no work-split CTA multiplicity in legacy. Each kernel is a single `KernelSpec` over
`all_cores`; the per-group stick count lives in per-node RTAs (`start_stick_id`,
`sticks_for_core`), not CTAs, so a single `WorkUnitSpec` covers both core groups.

## Dropped Plumbing

| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader RTA slots 0-2 (`scatter_program_factory.cpp:163-165`) | `Buffer*` input/index/src | `TensorBinding`s (`tensor::input`/`index`/`source`) |
| writer RTA slot 0 (`:181`) | `Buffer*` output | `TensorBinding` (`tensor::output`) |
| reader CTA slots 4-7 (`:82-85`; reduce 4-8 `:84-88`) | `ScatterCB::*` magic CB indices | `DFBBinding`s (`dfb::input`/`index`/`source`/`output`[/`fp32_temp`]) |
| CTA `TensorAccessorArgs(*buf).append_to(...)` (`:95-98`; reduce `:98-101`) | 4× accessor-arg blocks | folded into the `TensorBinding`s |
| reader CTA slots 0-3 (`:78-81`; reduce `:80-83`) | four `buffer->address()` — **DEAD** (never read; stale-pointer trap on cache hit) | removed (brief-directed); addresses now flow via `TensorBinding` |
| reader CTA `output_stick_size` + `input`/`index`/`source_stick_size_bytes` | dead CTAs (never read) | fall away naturally — the positional `get_ctas()` struct that carried them is removed; only kernel-referenced CTAs are re-emitted as named args (see report) |
| reader positional CTAs 8-11,16 (`input`/`index`/`source_stick_size`, `input_rank`) | positional `get_compile_time_arg_val(N)` | named CTAs (`args::input_stick_size`, …, `args::input_rank`) |
| writer positional CTA (`output_stick_size_bytes`) | positional | named CTA (`args::output_stick_size_bytes`) |
| reader/writer positional RTAs (`stick_offset`, `sticks_per_core`, chunk sizes, reduction) | positional `get_arg_val<uint32_t>(N)` | named RTAs |
| reader RTA shape-dim blocks (`make_shape_array_from_runtime_args<N>(9)` / `(9+N)`) | positional RTA loop | **runtime varargs** (`num_runtime_varargs = (input_rank-1)+(index_rank-1)`; helper reads `get_vararg`) |

## Applied Patterns

- [Self-loop DFB binding](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-self-loop-dfb-binding): `INPUT`/`INDEX`/`SRC` on the reader (PRODUCER + CONSUMER, shared accessor name), plus `FP32_TEMP` on the reduce reader — all DM self-loops, legal on Gen1.
- [Pass DFB handles directly to LLKs / kernel-lib helpers](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers): `dfb::name` passed to `load_to_dfb`/`write_to_output` (take a `uint32_t` cb id) and to the free `get_dataformat(operand)`, via the constexpr `DFBAccessor → uint32_t` conversion.
- Runtime varargs for the per-dimension shape blocks (the CTA-bounded-count vararg case; `N = rank-1`).

## Deferred / Flagged

- **Both factories ported together (not one-at-a-time).** They share `device/kernels/common.hpp`,
  whose `make_shape_array_from_runtime_args<N>` helper must switch from `get_arg_val` to
  `get_vararg` for the shape-dim blocks. That single shared change breaks any factory left on the
  legacy RTA layout, so the two factories cannot be split across passes — porting both is the
  correct atomic unit here.
- **`get_dataformat` stays a free function, not the DFB object getter (rule 7 exception).** The
  kernels use `std_type_t<get_dataformat(...)>` in a non-type-template-argument position, which
  requires a constant expression. `DataflowBuffer`'s constructor is **not** `constexpr`, so
  `dfb.get_dataformat()` cannot be evaluated there; the free `get_dataformat(dfb::name)` (constexpr,
  via the `DFBAccessor → uint32_t` conversion) is the only compile-time-correct form. Recorded in
  the port report as friction.
- **RTA→CRTA / common-vararg opportunity (not taken).** `input_and_output_chunk_size`,
  `index_chunk_size`, `source_chunk_size`, `scatter_reduction_type`, and both shape-dim vararg
  blocks hold the **same value on every node**, so they are really common runtime args / common
  varargs. Legacy emits them per-core; converting RTA→CRTA changes dispatch semantics, so it is
  **out of scope** for this port. Noted for a later cleanup pass.
- **`ScatterCB` enum removed.** Once the CB indices become named DFBs, the enum
  (`device/scatter_common.hpp`) is unused; it is the magic-CB-index vocabulary the port replaces.
