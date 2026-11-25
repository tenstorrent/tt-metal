# Port Plan — experimental/isin

Port plan for `ttnn/cpp/ttnn/operations/experimental/isin` (`IsInProgramFactory`), ported from `ProgramDescriptor` to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept`. `IsInProgramFactory::create_descriptor(const IsinParams&, const IsinInputs&, Tensor&)` (`device/isin_program_factory.hpp:13`, body `device/isin_program_factory.cpp:42-155`) lives in a `program_factory_t` variant (`device/isin_device_operation.hpp:17`), not on the device-op, so this is not the direct-descriptor shape.
- Variants: single.
- Custom `compute_program_hash`: none (default reflection-based hash over `IsinParams` + `IsinInputs`).

*(The Metal 2.0 factory concept the port targets was chosen during the audit — see the brief's TTNN factory analysis section. Carried forward in the [TTNN ProgramFactory](#ttnn-programfactory) section below.)*

### Kernels
| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/isin_reader.cpp` | `all_cores` | `[0, 0, 0, c_0, c_1, c_2, elements volume, test_elements volume, single_fetch_subchunk_size, invert, elements element_size, output element_size]` + `TensorAccessorArgs(elements)`, `(test_elements)`, `(output)` appended (`isin_program_factory.cpp:77-92`) | none | per core: `{elements_buffer, test_elements_buffer, subchunks_per_core, subchunks_offset}` (`:145-146`) | none | none | O2 (unset → DM default) | `ReaderConfigDescriptor{}` → RISCV_1 / NOC_0 / DM_DEDICATED_NOC |
| writer | `device/kernels/dataflow/isin_writer.cpp` | `all_cores` | identical vector to the reader (`:129`) | none | per core: `{output_buffer, subchunks_per_core, subchunks_offset}` (`:147`) | none | none | O2 (unset → DM default) | `WriterConfigDescriptor{}` → RISCV_0 / NOC_1 / DM_DEDICATED_NOC |

`grep -n opt_level` over the factory finds nothing, so both kernels resolve to the DM default, O2.

### CBs
| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| `c_0` ELEMENTS | `single_fetch_subchunk_size × elements element_size` | `all_cores` | `datatype_to_dataformat_converter(elements dtype)` | = total_size (1 page) | unset |
| `c_1` TEST_ELEMENTS | `single_fetch_subchunk_size × test_elements element_size` | `all_cores` | `datatype_to_dataformat_converter(test_elements dtype)` | = total_size (1 page) | unset |
| `c_2` OUTPUT | `single_fetch_subchunk_size × output element_size` (4 B) | `all_cores` | `datatype_to_dataformat_converter(UINT32)` | = total_size (1 page) | unset |

No GlobalCircularBuffer, no `address_offset`, and no buffer-backed (borrowed) CB.

### Semaphores
none

### Tensor accessors
| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `isin_program_factory.cpp:90` / reader `isin_reader.cpp:97-98` | `tensor_args.elements_tensor` | reader RTA0 (`Buffer*`) |
| `isin_program_factory.cpp:91` / reader `isin_reader.cpp:99-102` | `tensor_args.test_elements_tensor` | reader RTA1 (`Buffer*`) |
| `isin_program_factory.cpp:92` / writer `isin_writer.cpp:19` | `tensor_return_value` (output, or `*optional_out`) | writer RTA0 (`Buffer*`) |

All three accessor args are appended to *both* kernels' CTA lists, but each kernel constructs only its own accessors. The shared decoder `get_ctas()` (`isin_common.hpp:76-93`) reconstructs all three in both kernels.

### Work split
- Driver: `split_work_to_cores(compute_with_storage_grid_size, subchunks_num)`, where `subchunks_num = ceil(elements volume / single_fetch_subchunk_size)` (`isin_program_factory.cpp:98-107`)
- num_cores / all_cores: from the split
- core_group_1: `num_subchunks_per_core_group_1` subchunks per core
- core_group_2: `num_subchunks_per_core_group_2` subchunks per core
- One `KernelDescriptor` per kernel covers `all_cores`. The group split shows up only as per-core RTA values (`subchunks_per_core`, running `subchunks_offset`), walked in `grid_to_cores` order (`:132-150`).

### Shared kernels
none. `grep -rl isin_reader.cpp` and `grep -rl isin_writer.cpp` over `ttnn/cpp/ttnn/operations/` hit only `isin_program_factory.cpp:114/:116`. The kernels are op-owned and bound by this one factory, and no `_metal2` sibling exists. `isin_common.hpp` is an op-private helper included by both kernels.

### Flags
none (no unreferenced kernel files)

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept`
- **Custom `compute_program_hash`**: none
- **Implementation notes**: one `create_program_artifacts` replaces `create_descriptor`. There is no override to translate and no pybound `create_descriptor` to delete. `isin_device_operation.hpp` is untouched, because the variant already names `IsInProgramFactory`.

## Planned Spec Shape

- KernelSpecs: `reader` (`isin_reader.cpp`, `ttnn::create_reader_datamovement_config()` = RISCV_1/NOC_0/dedicated) and `writer` (`isin_writer.cpp`, `ttnn::create_writer_datamovement_config()` = RISCV_0/NOC_1/dedicated). Both have the default `compiler_options` (O2). They are 1:1 with the legacy `KernelDescriptor`s.
- DataflowBufferSpecs (all `num_entries = 1`, `entry_size` = the legacy page size, `data_format_metadata` = the legacy CB data format, `tile_format_metadata` unset as in legacy):
  - `elements`: reader PRODUCER + CONSUMER (self-loop)
  - `test_elements`: reader PRODUCER + CONSUMER (self-loop)
  - `output`: reader PRODUCER, writer CONSUMER
- SemaphoreSpecs: none.
- TensorParameters: `elements` (`elements_tensor.tensor_spec()`, bound by the reader), `test_elements` (bound by the reader), `output` (`output_tensor.tensor_spec()`, bound by the writer). Strict matching, no relaxations.
- WorkUnitSpecs: one, `{reader, writer}` over `all_cores`.

## Preserved Multiplicity

none — no work-split multiplicity in legacy (one `KernelDescriptor` per source over `all_cores`; the group split is RTA-valued only).

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader RTA0 (`isin_program_factory.cpp:146`; kernel `isin_reader.cpp:86`) | `elements_buffer` (`Buffer*`) | `TensorBinding{elements → "elements"}`; kernel `TensorAccessor(tensor::elements)` |
| reader RTA1 (`:146`; kernel `isin_reader.cpp:87`) | `test_elements_buffer` (`Buffer*`) | `TensorBinding{test_elements → "test_elements"}` |
| writer RTA0 (`:147`; kernel `isin_writer.cpp:12`) | `output_buffer` (`Buffer*`) | `TensorBinding{output → "output"}` |
| reader RTA2/3, writer RTA1/2 | positional `subchunks_per_core`, `subchunks_offset` | named RTAs `subchunks_per_core`, `subchunks_offset` (same values, same nodes) |
| both kernels' CTA 0-2 (`:78-80`) | dead `0u` placeholders | dropped (never read) |
| both kernels' CTA 3/4/5 (`:81-83`; kernel `isin_common.hpp:81-83`) | `c_0` / `c_1` / `c_2` indices | `DFBBinding`s → `dfb::elements` / `dfb::test_elements` / `dfb::output` |
| both kernels' CTA 6-11 (`:84-89`) | positional scalars | named CTAs `elements_size`, `test_elements_size`, `single_fetch_subchunk_size`, `invert`, `elements_tensor_datum_size`, `output_tensor_datum_size`. Both kernels receive all six, as legacy gave both the full list, because the shared decoder reads all six. |
| both kernels' CTA 12+ (`:90-92`; kernel `isin_common.hpp:77-79`) | `TensorAccessorArgs(*buffer).append_to` ×3 / `TensorAccessorArgs<12>()` chain | dropped; the bindings carry the layout |
| accessor 3rd arg `isin_reader.cpp:97-98`, `:99-102`, `isin_writer.cpp:19` | page-size override from CTAs | dropped (Class 2, per the brief) |

## Applied Patterns

- Sync-free and single-ended CBs → self-loop DFB (`port_patterns.md`, DM case): `elements` and `test_elements` on the reader. Census re-derived: the reader is the sole toucher of each (`load_to_cb` reserve/push, then `wait_front` / `get_read_ptr` / `pop_front` in `kernel_main`). The writer never touches either. Agrees with the brief.
- 1P+1C on `output`: reader is the locked producer (`reserve_back` / `push_back`, raw fills at `get_write_ptr`), writer the locked consumer (`wait_front` / `pop_front` in `write_to_dram`). Agrees with the brief.
- Token-form `get_dataformat(dfb::elements)` for the template-argument site `isin_reader.cpp:91` (`cb_dfb_api_whitelist.md` §A constant-expression case).

## Deferred / Flagged

- **CTA decoder reshape.** `IsInCTAs` keeps only the six scalar CTAs. The CB-index fields and accessor-args fields leave the struct, and each kernel constructs its `DataflowBuffer` / `TensorAccessor` from its own binding tokens. That way the shared header names no `dfb::` / `tensor::` token, so a kernel never refers to a resource it doesn't bind.
- **Helpers take the token.** `load_to_cb` / `write_to_dram` take a `DFBBindingToken` instead of a `uint32_t` index and build their `DataflowBuffer` from it (rule 3). The per-call wrapper stays, as legacy had it; on Gen1 it is a view over the shared CB interface. `load_to_cb` becomes `load_to_dfb` (the `cb` → `dfb` API rename).
- The three `TT_FATAL(*_buffer != nullptr)` guards (`isin_program_factory.cpp:58-60`) and the `TT_THROW` (`:142`) are kept verbatim.
