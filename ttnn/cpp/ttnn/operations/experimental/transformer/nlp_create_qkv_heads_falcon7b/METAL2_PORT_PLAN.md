# Port Plan — nlp_create_qkv_heads_falcon7b

Port plan for `experimental/transformer/nlp_create_qkv_heads_falcon7b`, ported from the ProgramDescriptor API (direct-descriptor shape) to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept`, **direct-descriptor shape**: `create_descriptor` is a static member of `NlpCreateHeadsFalcon7BDeviceOperation` itself (`device/nlp_create_qkv_heads_falcon7b_device_operation.hpp:27-30`, body `device/nlp_create_qkv_heads_falcon7b_program_factory.cpp:19-155`). There is no `program_factory_t`, so the port is forced through `ttnn_factory.md` exception 3.
- Variants: single (one code path; output is always interleaved).
- Custom `compute_program_hash`: none (default reflection-based hash). No `attribute_values` / `to_hash` backdoor.

*(Target concept chosen by the audit: `ProgramSpecFactoryConcept`. See [TTNN ProgramFactory](#ttnn-programfactory).)*

### Kernels
| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` (**borrowed**) | `all_cores` | `TensorAccessorArgs(*in0_buffer)` (factory `:75`) | none | per core: `[in0_buffer (Buffer*), num_blocks_per_core * per_tensor_tiles, num_blocks_written * per_tensor_tiles]` (`:129-130`) | none | none | O2 (unset, DM) | `ReaderConfigDescriptor{}` → (RISCV_1, NOC_0, DEDICATED) |
| writer | `device/kernels/dataflow/writer_tm_tile_layout_nlp_create_qkv_heads_falcon7b.cpp` (op-owned) | `all_cores` | `[q_num_tiles_per_tensor=142, kv_num_tiles_per_tensor=2, q_out_h_tiles, q_out_w_tiles=2, q_out_c=71, q_out_HtWt]` + `TensorAccessorArgs` ×3 for q/k/v (`:76-86`) | none | per core: `[q_buffer, k_buffer, v_buffer, num_blocks_per_core, q_out_h_dim, q_out_tensor_tile_id, num_blocks_written * 2]` (`:136-146`) | none | none | O2 (unset, DM) | `WriterConfigDescriptor{}` → (RISCV_0, NOC_1, DEDICATED) |

`grep -n opt_level` on the factory: no hits. Both kernels are DM, so both resolve to O2. There is no compute kernel.

### CBs
| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| 0 | `per_tensor_tiles * 2 * single_tile_size` (292 tiles for Falcon-7B) | `all_cores` | `datatype_to_dataformat_converter(a.dtype())` | `single_tile_size` | not set |

Not a GlobalCircularBuffer, no `address_offset`, not borrowed, not aliased.

### Semaphores
none

### Tensor accessors
| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `program_factory.cpp:75` / `:130` | input `a` | reader RTA 0 |
| `program_factory.cpp:84` / `:139` | output `q` | writer RTA 0 |
| `program_factory.cpp:85` / `:140` | output `k` | writer RTA 1 |
| `program_factory.cpp:86` / `:141` | output `v` | writer RTA 2 |

### Work split
- Driver: `split_work_to_cores(compute_with_storage_grid_size, num_blocks)`, with `num_blocks = B * 1 * S / TILE_HEIGHT` (`:51-53`)
- num_cores / all_cores: from the split
- core_group_1: `num_blocks_per_core_group_1` blocks per core
- core_group_2: `num_blocks_per_core_group_2` blocks per core (may be empty)
- Node walk: `core = {i / num_cores_y, i % num_cores_y}` for `i < num_cores`, with a running `num_blocks_written`. The per-core counts travel as **RTAs** (not CTAs), so there is a single reader and a single writer descriptor over `all_cores`.

### Shared kernels
- **reader**: borrowed `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp`. A `_metal2` fork already exists beside it: `reader_unary_interleaved_start_id_metal2.cpp`. That puts this port on **rung 1 (reuse)**. Fork binding vocabulary: `dfb::in` (PRODUCER), `tensor::src`, named RTAs `num_pages` and `start_id`, optional `BACKWARDS` define (leave it undefined, as the legacy factory does). Legacy-original consumers that remain unmigrated: `examples/example`, `reduction/topk` (topk_route_prep). `examples/example_multiple_return` has since been ported; see the report.
- **writer**: op-owned and bound by no other op (`grep -rl writer_tm_tile_layout_nlp_create_qkv_heads_falcon7b ttnn/cpp/ttnn/operations/` gives this factory only). Convert it in place.

### Flags
none. No unreferenced kernel files; `device/kernels/dataflow/` holds only the writer.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept`
- **Custom `compute_program_hash`**: none
- **Implementation notes**: forced device-op-class edit (exception 3). Nest `struct NlpCreateQkvHeadsFalcon7BProgramFactory` with `create_program_artifacts` in the device-op, add `using program_factory_t = std::variant<NlpCreateQkvHeadsFalcon7BProgramFactory>;`, and remove the device-op-level `create_descriptor`. The rationale comment above it (per-dispatch state = four addresses) moves onto the factory and is retargeted from "runtime-arg bindings" to "tensor bindings". There is no pybound `create_descriptor` (nanobind binds only the user function), so no pybind line is deleted. The body stays in `device/nlp_create_qkv_heads_falcon7b_program_factory.cpp`, which is already in `sources.cmake`.

## Planned Spec Shape

- KernelSpecs:
  - `reader`: `eltwise/unary/.../reader_unary_interleaved_start_id_metal2.cpp`; `hw_config = ttnn::create_reader_datamovement_config()`; no defines; default `compiler_options` (O2); runtime_arg_schema `{num_pages, start_id}`.
  - `writer`: the op's writer (converted in place); `hw_config = ttnn::create_writer_datamovement_config()`; default O2. Named CTAs: `q_num_tiles`, `kv_num_tiles`, `q_out_h_tiles`, `q_out_w_tiles`, `q_out_c`, `q_out_HtWt`. Named RTAs: `num_blocks`, `q_out_h_dim`, `q_out_tensor_tile_id`, `kv_out_tensor_tile_id`.
- DataflowBufferSpecs: one, `qkv` (legacy CB 0): `entry_size = single_tile_size`, `num_entries = per_tensor_tiles * 2`, `data_format_metadata = data_format`. Reader binds it as PRODUCER (`in`), writer as CONSUMER (`out`).
- SemaphoreSpecs: none.
- TensorParameters: `input`, `q`, `k`, `v` (each from `<tensor>.mesh_tensor().tensor_spec()`), with matching `TensorArgument`s.
- WorkUnitSpecs: `core_group_1` and `core_group_2` (the latter only if non-empty), each running {reader, writer}. Together they cover `all_cores`, matching the legacy `core_ranges`. This is the same shape as the sibling vit port.

## Preserved Multiplicity

none. Legacy has one reader and one writer `KernelDescriptor` over `all_cores`; per-group counts are RTAs, not CTAs.

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| factory `:75`, reader CTA 0.. | `TensorAccessorArgs(*in0_buffer).append_to(...)` | `TensorBinding{input → "src"}` |
| factory `:130`, reader RTA 0 | `in0_buffer` (`Buffer*`) | `TensorBinding{input → "src"}` |
| factory `:130`, reader RTAs 1, 2 | positional `num_pages`, `start_id` | named RTAs `num_pages`, `start_id` (fork interface) |
| factory `:84-86`, writer CTAs 6.. | `TensorAccessorArgs(*q/k/v_buffer)` | `TensorBinding`s `q`, `k`, `v` |
| factory `:139-141`, writer RTAs 0-2 | `q_buffer`, `k_buffer`, `v_buffer` (`Buffer*`) | `TensorBinding`s `q`, `k`, `v` |
| factory `:76-83`, writer CTAs 0-5 | positional | named CTAs (see above) |
| factory `:142-145`, writer RTAs 3-6 | positional | named RTAs (see above) |
| writer `:36` | `constexpr uint32_t cb_id_out0 = 0;` (magic CB index) | `DataflowBuffer dfb_out0(dfb::out)` |
| factory `:106` | `src0_cb_index = 0` | `DFBSpecName QKV{"qkv"}` |

No page-size 3rd-arg sites and no semaphore-ID RTAs.

## Applied Patterns

- Shared kernel, rung 1: reuse the existing `reader_unary_interleaved_start_id_metal2.cpp` fork.
- `AddRuntimeArgsForNode`: keep the legacy node-first per-core loop verbatim.
- Writer `get_tile_size(cb_id_out0)` (non-`constexpr`) → `dfb_out0.get_tile_size()` (whitelist rule 7).
- Writer RTAs that the loop mutates (`q_out_h_dim`, `q_out_tensor_tile_id`, `kv_out_tensor_tile_id`) are read into non-`const` locals, as today.

## Deferred / Flagged

- New findings during planning: none. The brief matches the code line-for-line.
