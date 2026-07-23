# Port Plan — fill_rm (`data_movement/fill_rm`)

Port plan for `fill_rm`, ported from `ProgramDescriptor` (`ProgramDescriptorFactoryConcept`) to Metal 2.0 (`MetalV2FactoryConcept`).
Written during the inventory and planning steps; committed alongside the port for review.

`fill_rm` and `fill_ones_rm` are two host entry points into the **same** device op (`fill_ones_rm` hardwires `val_hi=1, val_lo=0`); one device op, one program factory, one kernel — one porting unit.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` — `FillRMProgramFactory::create_descriptor()` returns `tt::tt_metal::ProgramDescriptor` (`device/fill_rm_program_factory.hpp:14`).
- Variants: single (`program_factory_t = std::variant<FillRMProgramFactory>`).
- Custom `compute_program_hash`: none — already default reflection-based hash (audit-confirmed, grep clean).

### Kernels
| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | config |
|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/fill_rm_interleaved.cpp` (op-owned) | `CoreRange{{0,0},{0,0}}` | `TensorAccessorArgs(*dst_buffer)` block only (`.cpp:69-70`) | none | slot0=`dst_buffer` (`Buffer*`), 1=`N*C`, 2=`H`, 3=`W`, 4=`hFill`, 5=`wFill`, 6=`val_hi` (bf16 bits), 7=`val_lo` (bf16 bits) (`.cpp:82-91`) | none | none | `ReaderConfigDescriptor{}` |

### CBs
| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| 0 | `16 * single_tile_size` | `{{0,0},{0,0}}` | `datatype_to_dataformat_converter(input.dtype())` | `single_tile_size` | not set |
| 1 | `16 * single_tile_size` | `{{0,0},{0,0}}` | same | `single_tile_size` | not set |

`single_tile_size = tt::tile_size(cb_data_format)`; `num_cb_tiles = 16`. No `.global_circular_buffer`, no `address_offset`, no aliasing (`.cpp:49-67`).

### Semaphores
none — op uses no semaphores.

### Tensor accessors
| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `fill_rm_program_factory.cpp:70` (`TensorAccessorArgs(*dst_buffer).append_to(...)`) → kernel `fill_rm_interleaved.cpp:28,31` | **output** (`tensor_return_value`) | slot 0 (`dst_buffer`, `Buffer*`) |

Input tensor (`tensor_args.input`) is host-only metadata (`input.dtype()`); the kernel never touches it → **not** a `TensorAccessor` / `TensorParameter`.

### Work split
n/a — single core (`CoreRange{{0,0},{0,0}}`, interleaved-only).

### Cross-op kernels
none — the op owns its only kernel; all kernel `#include`s are `tt_metal/hw/inc/api/*` (LLK/HAL).

### Flags
- Dead kernel locals in `fill_rm_interleaved.cpp` (`num_bytes_per_tile` `:39`, `num_bytes_per_tile_row` `:40`, `Wt` `:41`, `replicate_dest_addr` `:44`, `start_dram_addr_offset_for_tensor_row` `:45`) — audit "misc anomalies", **out of port scope** (kernel-body cleanup). The port does not remove them. One consequence: `num_bytes_per_tile = get_tile_size(cb_id_in0)` (`:39`) references a magic CB id the port removes, so its RHS is mechanically rewritten to the DFB object getter `dfb_in0.get_tile_size()` (whitelist rule 7) even though the value stays unused. Noted in report.

## TTNN ProgramFactory
- **Concept (inherited from audit)**: `MetalV2FactoryConcept`.
- **Custom `compute_program_hash`**: none — already default reflection-based hash.
- **Implementation notes**: `create_descriptor` → `create_program_artifacts`, same 3-arg signature `(FillRmParams, FillRmInputs, Tensor&)`. No device-op-class edits forced (no custom hash; pybind binds `&ttnn::fill_rm`/`&ttnn::fill_ones_rm` plain functions, not the factory — no `create_descriptor` pybind to remove).

## Planned Spec Shape
- KernelSpecs: **1** — `reader`, DM reader config (`create_reader_datamovement_config(arch)`).
- DataflowBufferSpecs: **2** — `in0`, `in1`; `entry_size = single_tile_size`, `num_entries = 16`, `data_format_metadata = cb_data_format`, `tile_format_metadata` unset (legacy `.tile` unset). Each **self-loop** (single toucher).
- SemaphoreSpecs: none.
- TensorParameters: **1** — `out` (`.spec = output.mesh_tensor().tensor_spec()`).
- WorkUnitSpecs: **1** — `main`, kernels `{reader}`, `target_nodes = NodeCoord{0,0}`.

## Preserved Multiplicity
none — no work-split multiplicity in legacy (single core, single `KernelDescriptor`).

## Dropped Plumbing
| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `fill_rm_program_factory.cpp:82-84` RTA slot 0 | `dst_buffer` (`Buffer*`) address RTA | `TensorParameter{out}` + `TensorBinding{out}` on reader; `TensorArgument{out}` |
| `fill_rm_program_factory.cpp:69-70` CTA block | `TensorAccessorArgs(*dst_buffer).append_to(cta)` | binding mechanism (host packs layout metadata) |
| `fill_rm_interleaved.cpp:28` | `constexpr auto dst_args = TensorAccessorArgs<0>();` | dropped — `TensorAccessor(tensor::out)` |
| `fill_rm_interleaved.cpp:31` | `TensorAccessor(dst_args, dst_addr, W << 1)` (3rd arg) | `TensorAccessor(tensor::out)` — Class 2 drop; also delete `:29-30` stale comment |
| `fill_rm_interleaved.cpp:19` | `dst_addr = get_arg_val<uint32_t>(0)` | dropped (address rides the binding) |
| `fill_rm_interleaved.cpp:34-35` | `cb_id_in0 = 0` / `dfb_id_in1 = 1` magic indices | `dfb::in0` / `dfb::in1` |
| `fill_rm_interleaved.cpp:20-26` | RTA slots 1-7 `get_arg_val<uint32_t>(N)` | named RTAs `get_arg(args::NC/H/W/fillH/fillW/val_hi/val_lo)` |
| `fill_rm_interleaved.cpp:39` | `get_tile_size(cb_id_in0)` (cb-id free fn) | `dfb_in0.get_tile_size()` (rule 7 object getter; value still dead) |

## Applied Patterns
- [Sync-free and single-ended CBs → self-loop DFB](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md): `in0` and `in1` are each touched by exactly one kernel (the reader FIFO-produces via `reserve_back`/`push_back`, fills via `get_write_ptr()`, and uses the DFB as the `noc.async_write` source — no FIFO consume). One toucher → bind the reader **both** PRODUCER and CONSUMER (shared accessor name), legal DM self-loop on Gen1.
- [Pass DFB handles directly to LLKs / DFB metadata via object](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md) (whitelist rule 7): `get_tile_size(cb_id)` → `dfb.get_tile_size()`.

## Deferred / Flagged
- New findings during planning: none. Census re-derived from the kernel touch agrees with the brief (both CBs single-toucher self-loops; no multi-binding). Dead-local cleanup remains an ops-team follow-up (audit misc anomalies), untouched by the port except the forced rule-7 rewrite noted under Flags.
