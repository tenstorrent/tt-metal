# Port Plan — experimental/transformer/concatenate_heads

Port plan for `ttnn.experimental.concatenate_heads` (`ConcatenateHeadsDeviceOperation`), ported from the
`ProgramDescriptor` API (direct `create_descriptor`) to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept`, **direct-descriptor shape**. `create_descriptor` is a static member
  of `ConcatenateHeadsDeviceOperation` itself (`device/concatenate_heads_device_operation.hpp:27-28`), body at
  `device/concatenate_heads_program_factory.cpp:20-143`. There is no `program_factory_t` (re-checked at port time:
  still none). This forces `ttnn_factory.md` exception 3.
- Variants: single. One code path, no runtime kernel-source selection.
- Custom `compute_program_hash`: none. Default reflection-based hash over `ConcatenateHeadsParams` +
  `ConcatenateHeadsInputs`. No `attribute_values` / `to_hash` backdoor.

*(The target concept was chosen in the audit; it is carried forward in [TTNN ProgramFactory](#ttnn-programfactory).)*

### Kernels
| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_tm_tile_layout_concat_heads.cpp` | `all_cores` = `(0,0)-(num_cores_c-1, num_cores_r-1)` (12 × B) | `[0] in0_w_tiles`, `[1] in0_c`, `[2] in0_HtWt`, `[3..] TensorAccessorArgs(in0_buffer)` (`factory.cpp:74-80`) | none | per core: `[0] in0_buffer` (`Buffer*`), `[1] in0_tensor_tile_id` (`:124-129`) | none | none | unset → **O2** | `ReaderConfigDescriptor{}` (RISCV_1 / NOC_0 / dedicated) |
| writer | `device/kernels/dataflow/writer_tm_tile_layout_concat_heads.cpp` | `all_cores` | `[0] in0_w_tiles`, `[1] in0_c`, `[2..] TensorAccessorArgs(out_buffer)` (`:81-86`) | none | per core: `[0] out_buffer` (`Buffer*`), `[1] out_tensor_tile_id` (`:130-135`) | none | none | unset → **O2** | `WriterConfigDescriptor{}` (RISCV_0 / NOC_1 / dedicated) |

`grep -n opt_level device/concatenate_heads_program_factory.cpp` → no hits. Both kernels are DM, so both resolve to O2.

### CBs
| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| 0 (`src0_cb_index`) | `per_core_tiles * 2 * single_tile_size` (64 tiles) | `all_cores` | `datatype_to_dataformat_converter(a.dtype())` | `single_tile_size` = `tile_size(fmt)` | not set |

Not a GlobalCircularBuffer, not buffer-backed, no `address_offset`.

### Semaphores
none

### Tensor accessors
| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `factory.cpp:80` `TensorAccessorArgs(in0_buffer)` → reader CTA 3.. ; kernel `TensorAccessorArgs<3>()` (reader `:24`), `TensorAccessor(in0_args, in0_tensor_addr)` (reader `:28`) | `tensor_args.input` | reader RTA 0 (`Buffer*`, `:127`) |
| `factory.cpp:86` `TensorAccessorArgs(out_buffer)` → writer CTA 2.. ; kernel `TensorAccessorArgs<2>()` (writer `:25`), `TensorAccessor(out_args, out_tensor_addr)` (writer `:29`) | output (`tensor_return_value`) | writer RTA 0 (`Buffer*`, `:133`) |

### Work split
n/a — no `split_work_to_cores`. Fixed grid: `num_cores_c = H/32` (12) columns × `num_cores_r = B` (7–9) rows; every
core gets the same 32-tile workload, offset by the per-core tile ids.

### Shared kernels
none. `grep -rl reader_tm_tile_layout_concat_heads ttnn/cpp/ttnn/operations/` and the writer equivalent hit only this
factory (outside `experimental/quasar/`, which is not consulted). No `_metal2` fork exists or is needed; both
kernels convert in place.

### Flags
none. No unreferenced kernel files in the op directory.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept` (no `override_runtime_arguments`).
- **Custom `compute_program_hash`**: none.
- **Implementation notes**: exception 3 — nest `struct ConcatenateHeadsProgramFactory` with
  `create_program_artifacts` in `ConcatenateHeadsDeviceOperation`, add
  `using program_factory_t = std::variant<ConcatenateHeadsProgramFactory>;`, remove the device-op-level
  `create_descriptor`. Its explanatory comment (the cache-hit refresh rationale) moves onto the new factory method,
  reworded from "runtime-arg bindings" to "tensor bindings". Factory body stays in
  `device/concatenate_heads_program_factory.cpp`. No pybind `create_descriptor` to delete.

## Planned Spec Shape

- KernelSpecs: `reader`, `writer` (1:1 with the legacy `KernelDescriptor`s).
- DataflowBufferSpecs: one, `in0` — `entry_size = single_tile_size`, `num_entries = per_core_tiles * 2` (64),
  `data_format_metadata = cb_data_format`. No tile metadata (legacy didn't set `tile`). Reader = PRODUCER,
  writer = CONSUMER (plain 1:1, re-derived from the kernel census: reader `reserve_back`/`push_back`, writer
  `wait_front`/`pop_front`; each `get_*_ptr` is a peek on its own binding).
- SemaphoreSpecs: none.
- TensorParameters: `input` (`tensor_args.input`), `output` (`tensor_return_value`).
- WorkUnitSpecs: one, `concatenate_heads` = {reader, writer} on `all_cores`.

## Preserved Multiplicity

none — no work-split multiplicity in legacy (one descriptor per kernel source).

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| factory `:127` / reader `:16` RTA 0 | `in0_buffer` (`Buffer*`) → `in0_tensor_addr` | `TensorBinding{input, "input"}` |
| factory `:133` / writer `:18` RTA 0 | `out_buffer` (`Buffer*`) → `out_tensor_addr` | `TensorBinding{output, "output"}` |
| factory `:80` / reader `:24` | `TensorAccessorArgs(in0_buffer).append_to(...)` / `TensorAccessorArgs<3>()` | `TensorAccessor(tensor::input)` |
| factory `:86` / writer `:25` | `TensorAccessorArgs(out_buffer).append_to(...)` / `TensorAccessorArgs<2>()` | `TensorAccessor(tensor::output)` |
| reader `:26`, writer `:27` | hardcoded `cb_id_in0 = 0` / `cb_id_out0 = 0` (host `src0_cb_index = 0`, `:107`) | reader `DFBBinding{in0, "in0", PRODUCER}`, writer `DFBBinding{in0, "out0", CONSUMER}`; kernels construct `DataflowBuffer(dfb::in0)` / `DataflowBuffer(dfb::out0)` |
| reader `:27`, writer `:28` | `get_tile_size(cb_id)` (non-`constexpr`) | `dfb_in0.get_tile_size()` / `dfb_out0.get_tile_size()` (whitelist rule 7) |
| reader CTAs 0..2 | positional `in0_w_tiles`, `in0_c`, `in0_HtWt` | named CTAs, same names |
| writer CTAs 0..1 | positional `in0_w_tiles`, `in0_c` | named CTAs, same names |
| reader RTA 1 | positional `in0_tensor_tile_id` | named RTA `in0_tensor_tile_id` |
| writer RTA 1 | positional `out_tensor_tile_id` | named RTA `out_tensor_tile_id` |

No page-size third-argument sites (both accessors are 2-arg). No semaphore RTAs. No varargs.

## Applied Patterns

- Plain 1:1 DFB (reader PRODUCER, writer CONSUMER) — no self-loop, no multi-binding flag.
- `ttnn_factory.md` §3 "Give a direct-descriptor op a conventional program factory".
- DM hw_config: both legacy configs are the defaults → `ttnn::create_reader_datamovement_config()` /
  `ttnn::create_writer_datamovement_config()`.
- `AddRuntimeArgsForNode` inside the legacy per-core loop (loop shape kept as is).

## Deferred / Flagged

- New findings during planning: none beyond the audit's Misc anomalies (writer `pop_front(34)` vs 32 pushed,
  64-tile CB of which 32 are used, debug-only grid asserts). All carried verbatim.
- Accessor naming: the DFB spec is `in0`. The reader binds it as accessor `in0` (legacy `cb_in0`); the writer binds
  the same spec as accessor `out0` (legacy `cb_out0`, commented "same as cb_id_in0"), so each kernel keeps its
  legacy name with only the `cb_` → `dfb_` rename.
