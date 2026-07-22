# Port Plan — `experimental/transformer/nlp_concat_heads_decode`

Port plan for `nlp_concat_heads_decode`, ported from the `ProgramDescriptor`
(`create_descriptor`) concept to Metal 2.0 (`MetalV2FactoryConcept`,
`create_program_artifacts`).
Written during the inventory and planning steps; committed alongside the port for review.

Porting unit: one DeviceOperation (`NLPConcatHeadsDecodeDeviceOperation`), **two factories** —
`NLPConcatHeadsDecodeProgramFactory` (full-grid, `on_subcoregrids == false`) and
`NLPConcatHeadsDecodeSubcoregridsProgramFactory` (`on_subcoregrids == true`). They share the
DeviceOperation, the single output CB, and the reader/writer kernel *structure*; each has its own
kernel file. Both factories are ported together in this pass (findings identical except where noted).

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` (`create_descriptor()` returning `ProgramDescriptor`),
  both factories (`nlp_concat_heads_decode_program_factory.cpp:17`,
  `nlp_concat_heads_decode_subcoregrids_program_factory.cpp:18`).
- Variants: two factories under one `program_factory_t` variant; `select_program_factory` dispatches
  on `operation_attributes.on_subcoregrids`.
- Custom `compute_program_hash`: **none** — already default reflection-based hash (confirmed by audit
  and by grep of the device-op class).

*(Target concept chosen by the audit: `MetalV2FactoryConcept` — carried forward below.)*

### Kernels

**Full-grid factory** (`nlp_concat_heads_decode_program_factory.cpp`)

| unique_id | source | core_ranges | CTAs (positional) | RTAs | config |
|---|---|---|---|---|---|
| reader | `.../reader_tm_tile_layout_nlp_concat_heads_decode.cpp` | `q_cores` | `[0]element_size, [1]sub_tile_line_bytes, [2]q_output_cb_index(=c_16), [3]head_size, [4]batch, [5]head_tiles, [6]PHASES_TO_READ=1, [7]in_num_cores_x, [8]in_num_cores_y` | `[0]in_tile_offset_by_batch (per-core), [1]in_buffer (Buffer*), [2..]noc_x_coords (count in_num_cores_x), [..]noc_y_coords (count in_num_cores_y)` | `ReaderConfigDescriptor{}` |
| writer | same source | `q_cores` | same as reader except `[6]PHASES_TO_READ=2` | same as reader | `WriterConfigDescriptor{}` |

**Subcoregrids factory** (`nlp_concat_heads_decode_subcoregrids_program_factory.cpp`)

| unique_id | source | core_ranges | CTAs (positional) | RTAs | config |
|---|---|---|---|---|---|
| reader | `.../reader_tm_tile_layout_nlp_concat_heads_decode_subcoregrid.cpp` | `q_cores` | `[0]element_size, [1]sub_tile_line_bytes, [2]q_output_cb_index(=c_16), [3]head_size, [4]batch, [5]head_tiles, [6]PHASES_TO_READ=1, [7]in_num_cores, [8]face_h, [9]face_hw` | `[0]in_tile_offset_by_batch (per-core), [1]in_buffer (Buffer*), [2..]noc_x_coords (count in_num_cores), [..]noc_y_coords (count in_num_cores)` | `ReaderConfigDescriptor{}` |
| writer | same source | `q_cores` | same except `[6]PHASES_TO_READ=2` | same | `WriterConfigDescriptor{}` |

`defines`: none. `named_compile_time_args`: none (all positional). `common_runtime_args`: none.

### CBs

| index | total_size | core_ranges | data_format | page_size | tile | buffer |
|---|---|---|---|---|---|---|
| `c_16` (q_output_cb_index) | `q_num_tiles * single_tile_size` | `q_cores` | `datatype_to_dataformat_converter(input.dtype())` | `single_tile_size` | unset (default 32×32) | `output.buffer()` (borrowed) |

One CB in each factory; **borrowed-memory** (`.buffer = output.buffer()`). Not a GlobalCircularBuffer.
`c_16` main: `..._program_factory.cpp:45-55`. `c_16` subcoregrids: `..._subcoregrids_program_factory.cpp:55-65`.

### Semaphores
none

### Tensor accessors
No `TensorAccessor` in either kernel today. The input tensor's base surfaces as a **`Buffer*` RTA**
(`in_buffer`, arg slot 1) — the Case-2 raw-pointer form. Host sites:
`..._program_factory.cpp:130`, `..._subcoregrids_program_factory.cpp:137`.

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `..._program_factory.cpp:130` | `tensor_args.input` | 1 (`Buffer*`) |
| `..._subcoregrids_program_factory.cpp:137` | `tensor_args.input` | 1 (`Buffer*`) |

Output tensor backs the borrowed CB `c_16` (`.buffer = output.buffer()`).

### Work split
- Driver: **not** `split_work_to_cores`. The output-side cores are `q_cores` (the output shard grid);
  one output core per head. Both reader- and writer-config kernels run over the **full** `q_cores`
  grid (no group split). Per-core work differs only by the head offset RTA.
- num_cores = `q_cores.num_cores()`; all_cores = `q_cores`; no core_group_2.
- Input NoC-coordinate blocks are gathered from `in_cores` (input shard grid), CTA-bounded by
  `in_num_cores_x`/`in_num_cores_y` (main) or `in_num_cores` (subcoregrids).

### Cross-op kernels
none — both kernel files live in this op's own directory; no out-of-directory `#include`s (only `api/*`).

### Flags
- Dead/shadowed local `q_write_addr` in both kernels (`reader...:47` then re-declared `:53`;
  `..._subcoregrid:46`/`:52`). Harmless, pre-existing; **not touched** by the port (kernel behavior
  must not change). Flagged in the audit's Misc anomalies.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `MetalV2FactoryConcept` (no op-owned tensors).
- **Custom `compute_program_hash`**: none — already default.
- **Implementation notes**: two factories, ported together; each grows a `create_program_artifacts`
  and loses `create_descriptor`. `select_program_factory` and the device-op class are unchanged.
  No pybind `create_descriptor` to remove (the nanobind binds the high-level function only).

## Planned Spec Shape

Identical shape for both factories (only kernel source path, the CTA set `[7..]`, and the vararg
count differ).

- **KernelSpecs**: 2 — `reader` (PHASES_TO_READ=1) and `writer` (PHASES_TO_READ=2), same source.
  Differ only in the `phases_to_read` CTA value and the DFB endpoint role.
- **DataflowBufferSpecs**: 1 — `q_out`, borrowed from the `output` TensorParameter
  (`entry_size = single_tile_size`, `num_entries = q_num_tiles`, `data_format_metadata = cb_data_format`).
- **SemaphoreSpecs**: none.
- **TensorParameters**: 2 — `input` (Case-2 raw-pointer binding) and `output` (backs the borrowed DFB).
- **WorkUnitSpec**: 1 — `{reader, writer}` over `q_cores`.
- **Op-owned tensors**: none.

## Preserved Multiplicity

The two same-source kernels run over the **same** grid (`q_cores`), both raw-writing disjoint tile
phases into the one borrowed output DFB → this is the **dual-instance work-split / two-toucher**
case, resolved **1P+1C**, *not* a work-split-by-CTA multiplicity across disjoint node sets.

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| reader (Reader-config) + writer (Writer-config) of one source over `q_cores` | `reader`, `writer` of same source | one WU `{reader, writer}` over `q_cores` | `q_out`: reader → PRODUCER, writer → CONSUMER (cosmetic on Gen1; both write via `get_write_ptr()`) |

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader/writer RTA slot 1 (`..._program_factory.cpp:130`; `..._subcoregrids...:137`) | `in_buffer` (`Buffer*`) pushed into RTA list; kernel `q_start_addr = get_arg_val<uint32_t>(1)` | `TensorParameter INPUT` + `TensorBinding`; kernel `TensorAccessor(tensor::input).get_bank_base_address()` (Case-2 bridge) |
| reader/writer CTA slot 2 (`..._program_factory.cpp:85`, kernel `:22`; subcoregrids `:91`, kernel `:20`) | `q_output_cb_index` (= `CBIndex::c_16`) magic CB index | `DFBBinding` (`dfb::q_out`); `DataflowBuffer(dfb::q_out)` |
| reader/writer CTAs slots 0–1,3–8/9 (positional) | `get_compile_time_arg_val(N)` | named CTAs `get_arg(args::<name>)` |
| reader/writer RTA slot 0 (per-core) | `get_arg_val<uint32_t>(0)` | named RTA `get_arg(args::in_tile_offset_by_head)` (stays a per-node RTA; already-split-out offset, not a fold) |
| reader/writer RTA slots 2.. (noc_x_coords / noc_y_coords) | `get_arg_addr(2)` / `get_arg_addr(2 + num_x)` cast to `tt_l1_ptr uint32_t*`, indexed | **RTA varargs** (`num_runtime_varargs`, per-node `runtime_varargs`); kernel `get_vararg(i)` |

The `output` CB's `q_output_cb_index` CTA drops (→ DFB); the `Buffer* in_buffer` RTA and the
`get_arg_val<uint32_t>(1)` base read drop (→ tensor binding). No page-size 3rd-arg to drop (no
`TensorAccessor` was constructed with one). No semaphore-ID RTAs.

## Applied Patterns

- **Two-toucher DFB → assign 1P+1C** (dual-instance work-split): `q_out` bound PRODUCER on `reader`,
  CONSUMER on `writer`; both over `q_cores`. No multi-binding flag (census = 2 role-free touchers).
- **Borrowed-memory DFB**: `q_out.borrowed_from = OUTPUT`; backing L1 address resolves from the
  `output` TensorArgument at runtime.
- **Case-2 (raw-pointer) tensor binding**: `input` bound as `TensorParameter`; base pulled kernel-side
  via `TensorAccessor::get_bank_base_address()`; the hand-rolled cross-core NoC gather is left unchanged.
- **RTA varargs** (kernel-side vararg mechanism): the two variable-count NoC-coordinate blocks.
- **Named CTAs / named RTAs throughout**.

## Deferred / Flagged

- New findings during planning: none that change the audit's disposition. Recorded for the report:
  the Case-2 kernel here previously used **no** `TensorAccessor`, so introducing one adds an
  `#include "api/tensor/tensor_accessor.h"` beyond the recipe's "exactly two headers" (kernel-side
  whitelist rule 5 assumes `TensorAccessor` was already in use). Mirrors the proven reshard-generic
  port. → Friction/Confusion note in the report.
- RTA-varargs recognition: the audit already flagged the `get_arg_addr`-into-L1-array shape as a
  vararg sub-shape not spelled out in the recipe's `get_arg_val` loop framing. Carried to the report.
