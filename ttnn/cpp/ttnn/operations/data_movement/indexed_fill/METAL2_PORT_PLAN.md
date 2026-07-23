# Port Plan — `data_movement/indexed_fill`

Port plan for `indexed_fill`, ported from the legacy `ProgramDescriptor` factory (`ProgramDescriptorFactoryConcept`) to Metal 2.0 (`MetalV2FactoryConcept`).
Written during the inventory and planning steps; committed alongside the port for review.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` — `IndexedFillProgramFactory::create_descriptor(...)` returns a `tt::tt_metal::ProgramDescriptor` ([`indexed_fill_program_factory.cpp:123`](device/indexed_fill_program_factory.cpp#L123)).
- Variants: single (`program_factory_t = std::variant<IndexedFillProgramFactory>` at [`indexed_fill_device_operation.hpp:19`](device/indexed_fill_device_operation.hpp#L19)). One factory selects one of **four internal code paths** at create time by tensor geometry (not separate variants), via the `kernel_mode` compile-time arg:
  - `MODE_GENERIC` (0) — interleaved input_a, or any `dim != 0` geometry. Reader + `indexed_fill_writer_strided.cpp`.
  - `MODE_NATIVE` (1) — `dim==0`, HEIGHT_SHARDED L1, one batch per core. Reader + `indexed_fill_writer.cpp` (wait/pop stub); data CB aliased to output.
  - `MODE_SHARD_LOCAL_INTERLEAVED_B` (2) — `dim==0`, WIDTH/BLOCK_SHARDED L1, interleaved input_b. Reader + writer stub; data CB aliased to output.
  - `MODE_SHARD_LOCAL_SHARDED_B` (3) — as (2) but input_b same WIDTH_SHARDED grid. Reader + writer stub; data CB aliased to output.
- Custom `compute_program_hash`: **none** — already default reflection-based hash (audit confirmed).

### Kernels
| unique_id | source | core_ranges | CTAs (positional) | RTAs (per-core, by path) | defines | config |
|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/indexed_fill_reader.cpp` | `all_cores` (= `operation_attributes.worker_grid`) | `{cb_index=0, batch_cb_index=1, kernel_page_size, kernel_mode}` then `TensorAccessorArgs(input_a)`, `TensorAccessorArgs(input_b)`, `TensorAccessorArgs(batch_ids)` appended | native (8): `{batch_ids*, b, input_a*, input_b*, batch_size, my_batch_id, 0, 0}`; shard-local (13): `{batch_ids*, b, input_a*, input_b*, shard_ppb, 0, batch_offset_a, total_batches, b_full_ppb, shard_tile_w, full_tile_w, col_page_off, col_byte_off}`; generic (10): `{batch_ids*, b, input_a*, input_b*, inner_count, slice_start, outer_count, outer_stride_a, outer_stride_b, num_slices}` | none | `ReaderConfigDescriptor{}` (reader default: RISCV_1 / NOC_0 / DEDICATED) |
| writer (native/shard-local) | `device/kernels/dataflow/indexed_fill_writer.cpp` | `all_cores` | `{cb_index=0}` | `{batch_size_in_pages}` (native = `local_batch_size`; shard-local = `total_pages_in_shard`) | none | `WriterConfigDescriptor{}` (writer default: RISCV_0 / NOC_1 / DEDICATED) |
| writer (generic) | `device/kernels/dataflow/indexed_fill_writer_strided.cpp` | `all_cores` | `{cb_index=0}` then `TensorAccessorArgs(output)` | `{output*, kernel_page_size, outer_count, inner_count, outer_stride_a, slice_start, num_slices}` | none | `WriterConfigDescriptor{}` |

`*` = a `Buffer*` pushed into `emplace_runtime_args` (auto-registered `BufferBinding`); these are the address plumbing the port replaces with typed bindings.

The reader kernel reads its args at fixed constexpr indices: `0` batch_ids addr, `1` `batch_id_size`, `2` input_a addr, `3` input_b addr, `4` `batch_size_in_pages`, `5` `my_batch_id`; then shard-local reads `6..12`, generic reads `6..9` — all inside `if constexpr (mode)` branches. No `arg_index++`, no varargs.

### CBs
| index | total_size | core_ranges | data_format | page_size | tile | notes |
|---|---|---|---|---|---|---|
| 0 (data) | native: `kernel_rounded_page_size * batch_size_in_pages`; shard-local: `* total_pages_in_shard`; generic: `2 * kernel_rounded_page_size` | `all_cores` | `cb_data_format` = `datatype_to_dataformat_converter(input_a.dtype())` | `kernel_rounded_page_size` | not set | native/shard-local: `.buffer = output.buffer()` (borrowed-memory / dynamic CB); generic: local double-buffered |
| 1 (batch) | `2 * batch_page_size`, `batch_page_size = round_up_to_mul32(b * sizeof(uint32_t))` | `all_cores` | `cb_data_format` (inert — CB holds raw `uint32` indices, filled/read by byte access; audit "Misc anomalies") | `batch_page_size` | not set | touched only by reader |

Neither CB sets `.tile` and neither has multi-element `format_descriptors` (no aliasing). Neither is a GlobalCircularBuffer.

### Semaphores
none — the op uses no semaphores.

### Tensor accessors
| host site | originating Tensor | kernel accessor | RTA slot (host) |
|---|---|---|---|
| reader `TensorAccessorArgs(input_a)` | input_tensor_a | `s0` @ [`indexed_fill_reader.cpp:44`](device/kernels/dataflow/indexed_fill_reader.cpp#L44) | reader arg 2 |
| reader `TensorAccessorArgs(input_b)` | input_tensor_b | `s1` @ [`indexed_fill_reader.cpp:45`](device/kernels/dataflow/indexed_fill_reader.cpp#L45) | reader arg 3 |
| reader `TensorAccessorArgs(batch_ids)` | batch_id | `batchAddr` @ [`indexed_fill_reader.cpp:49`](device/kernels/dataflow/indexed_fill_reader.cpp#L49) (3rd arg `batch_id_size << 2`) | reader arg 0 |
| writer-strided `TensorAccessorArgs(output)` | output | `dst` @ [`indexed_fill_writer_strided.cpp:34`](device/kernels/dataflow/indexed_fill_writer_strided.cpp#L34) | writer arg 0 |

### Work split
Custom (not `split_work_to_cores`):
- **Generic**: distribute `S_dim` slices across `num_cores_total` cores by ceiling division — `slices_per_core = S_dim / num_cores_total`, `extra_slices = S_dim % num_cores_total`. Cores `[0, extra)` get `slices_per_core + 1`; the rest get `slices_per_core`. Cores with index `>= S_dim` idle (`num_slices = 0`).
- **Native**: one batch per core; core `i` active iff `i < B`.
- **Shard-local**: per-core `(shard_row, cx)` derived from the shard grid; column offsets precomputed once per unique `cx`.

### Cross-op kernels
none — all three kernels are owned by this op and file-path-instantiated from its own `device/kernels/dataflow/` directory; every `#include` is `api/*` (Device 2.0 surface). No donor kernels, no port-together coupling.

### Flags
- The reader is a single source shared across all four modes, selected by the `mode` CTA and gated with `if constexpr`. Consequence for the port: see [Deferred / Flagged](#deferred--flagged) (union RTA schema).
- Every kernel is already structurally Device 2.0 (`Noc`, `DataflowBuffer`, `TensorAccessor`, `UnicastEndpoint`) — no Device 2.0 prep needed.

## TTNN ProgramFactory
- **Concept (inherited from audit)**: `MetalV2FactoryConcept`.
- **Custom `compute_program_hash`**: none — already default reflection-based hash; nothing to delete.
- **Implementation notes**: the four internal paths are realized by branching inside `create_program_artifacts` ([Pattern: Multi-variant factories](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-multi-variant-factories)); the spec differs per path (writer source, data-DFB `borrowed_from`, per-path RTA values). Device-op class needs no edit: `program_factory_t` is a single-variant, so the framework selects it without a `select_program_factory`, and the pybind binds only the free function `ttnn::indexed_fill` (no `create_descriptor` exposure to remove).

## Planned Spec Shape

- **KernelSpecs**: 2 per create-call — one reader (`IF_READER`, source `indexed_fill_reader.cpp`) and one writer (`IF_WRITER`). The writer's source, bindings, and RTA schema are path-dependent: generic → `indexed_fill_writer_strided.cpp` (+ `IF_OUTPUT` tensor binding); native/shard-local → `indexed_fill_writer.cpp` (wait/pop stub, no tensor binding). The reader is structurally identical across paths (same bindings, same union RTA schema); only the `page_size`/`mode` CTA values and RTA values differ.
- **DataflowBufferSpecs**: 2.
  - `IF_DATA_DFB` — `entry_size = kernel_rounded_page_size`; `num_entries` = native: `batch_size_in_pages`, shard-local: `total_pages_in_shard`, generic: `2`; `data_format_metadata = cb_data_format`; `borrowed_from = IF_OUTPUT` in native/shard-local, `nullopt` in generic.
  - `IF_BATCH_DFB` — `entry_size = batch_page_size`; `num_entries = 2`; `data_format_metadata = cb_data_format` (preserved verbatim from legacy; inert per audit).
- **SemaphoreSpecs**: none.
- **TensorParameters**: 4, all declared in every path — `IF_BATCH_IDS`, `IF_INPUT_A`, `IF_INPUT_B`, `IF_OUTPUT` (each `.spec = <tensor>.tensor_spec()`). Every one has ≥1 user in every path (reader tensor bindings for batch_ids/input_a/input_b; `IF_OUTPUT` via the strided-writer tensor binding in generic, via `IF_DATA_DFB.borrowed_from` in native/shard-local — the validator counts a `borrowed_from` reference as a user, `program_spec.cpp:540`).
- **WorkUnitSpecs**: 1 — `{IF_READER, IF_WRITER}` over `all_cores`.
- **Op-owned tensors**: none.

### Bindings summary (per kernel)
- **Reader** (all paths): DFB `IF_DATA_DFB` PRODUCER (`in0`); DFB `IF_BATCH_DFB` self-loop — PRODUCER + CONSUMER, shared accessor `batch`; tensors `IF_BATCH_IDS`→`batch_ids`, `IF_INPUT_A`→`input_a`, `IF_INPUT_B`→`input_b`.
- **Writer, generic**: DFB `IF_DATA_DFB` CONSUMER (`in0`); tensor `IF_OUTPUT`→`output`.
- **Writer, native/shard-local**: DFB `IF_DATA_DFB` CONSUMER (`in0`); no tensor binding.

## Preserved Multiplicity
none — no work-split multiplicity in legacy. Each path emits exactly one reader `KernelDescriptor` and one writer `KernelDescriptor` over the single `all_cores` range; per-core variation is by RTA only. (This is the standard single-instance-per-node shape, not a dual-instance work-split.)

## Dropped Plumbing
| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader CTA slot 0 | `cb_index = 0` (magic CB index) | `DFBBinding{IF_DATA_DFB, "in0", PRODUCER}` |
| reader CTA slot 1 | `batch_cb_index = 1` (magic CB index) | `DFBBinding{IF_BATCH_DFB, "batch", PRODUCER}` + `{…, CONSUMER}` (self-loop) |
| reader CTA slot 2 | `kernel_page_size` (positional) | named CTA `page_size` |
| reader CTA slot 3 | `kernel_mode` (positional) | named CTA `mode` |
| reader CTAs (appended) | `TensorAccessorArgs(input_a/input_b/batch_ids)` | `TensorParameter` + `TensorBinding` (`IF_INPUT_A`/`IF_INPUT_B`/`IF_BATCH_IDS`); kernel builds `TensorAccessor(tensor::name)` |
| reader RTA slot 0 | `batch_ids.buffer()` (`Buffer*`) | `TensorBinding IF_BATCH_IDS` |
| reader RTA slot 2 | `input_a.buffer()` (`Buffer*`) | `TensorBinding IF_INPUT_A` (Case 1 generic/native; Case 2 shard-local via `get_bank_base_address`) |
| reader RTA slot 3 | `input_b.buffer()` (`Buffer*`) | `TensorBinding IF_INPUT_B` (Case 1 generic/native/interleaved-B; Case 2 SAME_SHARDED_B) |
| reader RTA slots 1,4,5,6–12 | positional RTAs | named RTAs (union schema — see below) |
| reader `batch_ids` accessor 3rd arg @ [`:49`](device/kernels/dataflow/indexed_fill_reader.cpp#L49) | `batch_id_size << 2` page-size override | dropped — Class 2 redundant (accessor only read at `page_id = 0`) |
| writer-strided CTA slot 0 | `cb_index = 0` | `DFBBinding{IF_DATA_DFB, "in0", CONSUMER}` |
| writer-strided CTA (appended) | `TensorAccessorArgs(output)` | `TensorBinding IF_OUTPUT`→`output` |
| writer-strided RTA slot 0 | `output.buffer()` (`Buffer*`) | `TensorBinding IF_OUTPUT` |
| writer-strided RTA slots 1–6 | positional RTAs | named RTAs `page_size, outer_count, inner_count, outer_stride, slice_start, num_slices` |
| writer-stub CTA slot 0 | `cb_index = 0` | `DFBBinding{IF_DATA_DFB, "in0", CONSUMER}` |
| writer-stub RTA slot 0 | positional `batch_size_in_pages` | named RTA `batch_size_in_pages` |
| data CB `.buffer = output.buffer()` (native/shard-local) | dynamic/borrowed CB (`UpdateDynamicCircularBufferAddress` on cache hit) | `DataflowBufferSpec::borrowed_from = IF_OUTPUT` |

**Reader union RTA schema** (14 names — every `args::` name any `if constexpr` branch references must be declared for every compiled path; see [Deferred / Flagged](#deferred--flagged)):
`batch_id_size, batch_size_in_pages, my_batch_id` (all paths) · `outer_count, outer_stride_a, outer_stride_b, num_slices` (generic) · `batch_offset_a, total_local_batches, b_full_ppb, shard_tile_w, full_tile_w, col_page_offset, col_byte_offset` (shard-local). The host sets all 14 per core; a path leaves the names it does not use at `0`.

## Applied Patterns
- [Multi-variant factories](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-multi-variant-factories): four internal paths selected inside `create_program_artifacts` by tensor geometry.
- [Sync-free and single-ended CBs → self-loop DFB](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb) / [Self-loop DFB binding](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-self-loop-dfb-binding): `IF_BATCH_DFB` — the reader FIFO-produces (`reserve_back`/`push_back`) and reads the staged indices back through a raw `get_write_ptr` pointer (never FIFO-consumes). One toucher → bind the reader PRODUCER + CONSUMER (shared accessor `batch`). Legal DM self-loop on Gen1.
- Borrowed-memory DFB ([migration guide — DataflowBufferSpec](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/migration_guide.md#dataflowbufferspec)): `IF_DATA_DFB.borrowed_from = IF_OUTPUT` in native/shard-local — the reader writes directly into the output's resident L1 shard; the writer is a wait/pop stub.
- Case 2 raw-pointer binding via `get_bank_base_address` ([kernel-side whitelist rule 5](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#kernel-side-whitelist)): `input_a` (shard-local) and `input_b` (SAME_SHARDED_B) pull the raw L1 base from their `TensorAccessor`; the existing offset arithmetic is unchanged.
- TensorAccessor 3rd-arg drop: `batch_ids` accessor (Class 2 redundant).

## Deferred / Flagged
- **Union RTA schema (structural consequence, not a new finding).** The brief mandates keeping the reader's `if constexpr (mode)` structure. Because `args::<name>` tokens are emitted per-kernel from the schema and C++ name lookup still runs on discarded `if constexpr` branches in the non-template `kernel_main`, every `args::` name referenced in *any* branch must be declared in the schema for *every* compiled path. So the reader carries the 14-name union schema and the host sets `0` for the names a given path does not use. This is the faithful way to honor "keep the `if constexpr` structure" with named args; it is recorded as friction in the port report, not worked around (no `#ifdef` promotion, no varargs).
- **Test coverage gap (heads-up for the report).** The confirmed test set exercises MODE_GENERIC, MODE_NATIVE, and MODE_SHARD_LOCAL_INTERLEAVED_B (block-sharded + interleaved input_b). It does **not** appear to exercise MODE_SHARD_LOCAL_SHARDED_B (input_b same WIDTH_SHARDED grid → the Case 2 `input_b` raw-base path). The port preserves that path's kernel arithmetic byte-for-byte, but it is not covered by a no-regression test. Noted for the port report / downstream.
- **Batch CB data-format cosmetic mismatch.** Preserved verbatim (`data_format_metadata = cb_data_format` though the buffer holds `uint32` indices). Audit classified it inert and not porter work; the port does not change it.
