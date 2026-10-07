# Port Plan — nlp_kv_cache_load_slice

Port plan for `ttnn/cpp/ttnn/operations/experimental/transformer/nlp_kv_cache_load_slice`, ported from the
`ProgramDescriptor` API (direct-descriptor shape) to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept`, reached through the **direct-descriptor** shim. `create_descriptor`
  is a static member of `NlpKVCacheLoadSliceDeviceOperation` itself (`device/nlp_kv_cache_load_slice_device_operation.hpp:27-28`,
  body `device/nlp_kv_cache_load_slice_program_factory.cpp:19-108`). There is no `program_factory_t` and no
  factory struct, so [exception 3](ttnn_factory.md#3-give-a-direct-descriptor-op-a-conventional-program-factory) applies.
- Variants: single.
- Custom `compute_program_hash`: none (default reflection-based hash). No `attribute_values` / `to_hash` backdoor.
- No `override_runtime_arguments`, no `get_dynamic_runtime_args`, no pybound `create_descriptor`.

### Kernels
| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `nlp_kv_cache_load_slice/device/kernels/dataflow/reader_unary_unpad_dims_interleaved_start_id_shard_optimized.cpp` (own) | `all_cores` = output shard grid | 0 `num_tiles_per_core`, 1 `num_unpadded_tiles_head_dim`, 2 `num_unpadded_tiles_seqlen_dim`, 3 `num_padded_tiles_seqlen_dim`, 4 `num_cores_total` (kernel: `num_readers`), 5.. `TensorAccessorArgs(src0_buffer)` | none | per core: 0 `src0_buffer` (`Buffer*`), 1 `start_id` | none | none | unset → O2 (DM) | `ReaderConfigDescriptor{}` (RISCV_1 / NOC_0 / dedicated) |
| writer | `data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp` (borrowed) | `all_cores` | 0 `src0_cb_index` (= c_0) | none | per core: 0 `num_tiles_per_core` | none | none | unset → O2 (DM) | `WriterConfigDescriptor{}` (RISCV_0 / NOC_1 / dedicated) |

`grep -n opt_level` on the factory: no hits.

### CBs
| index | total_size | core_ranges | data_format | page_size | tile (if set) | buffer |
|---|---|---|---|---|---|---|
| 0 | `num_tiles_per_core * single_tile_size` | `all_cores` | `datatype_to_dataformat_converter(a.dtype())` | `single_tile_size` | not set | `dst_buffer` (output shard, borrowed memory) |

No GlobalCircularBuffer, no `address_offset`.

Kernel-touch census for c_0: reader = PRODUCER (`reserve_back(num_tiles)` → raw fill via `get_write_ptr()` → `push_back(num_tiles)`);
writer = CONSUMER (`wait_front` / `pop_front`). Plain 1P+1C. This matches the brief.

### Semaphores
none

### Tensor accessors
| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `program_factory.cpp:73` `TensorAccessorArgs(src0_buffer).append_to(...)` (CTA offset 5); reader `:29,34` | `tensor_args.input` | reader RTA 0 (`src0_buffer`, `program_factory.cpp:98`) |

The output has no accessor; it is reached only as the borrowed backing of CB c_0 (`program_factory.cpp:57`).

### Work split
- n/a. No `split_work_to_cores`. One core per fused batch-head: `all_cores` = output shard grid
  (`num_cores_to_corerangeset(fused_batch_heads, grid, row_wise=true)` in `compute_output_specs`). Every core gets
  `num_tiles_per_core` tiles. Per-core `start_id` advances by `num_tiles_shifted_per_core` (`program_factory.cpp:92-101`).
- Core coord for RTAs: `{i % num_cores_x, i / num_cores_x}`, with `num_cores_x` from the first range of the grid (`:33-34,96`).

### Shared kernels
- `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp`: **borrowed**.
  A `_metal2` fork already exists beside it (`writer_unary_sharded_metal2.cpp`) → **rung 1: reuse the fork.**
  Fork vocabulary: `dfb::out` (bound CONSUMER), `args::num_units` (RTA). No `#ifdef`s, no CTAs.
  Remaining legacy binders of the original (from the brief, re-checked with `grep -rl`): `data_movement/sharded_partial/interleaved_to_sharded_partial`,
  `data_movement/untilize` (nd-shard identical-spec factory), `experimental/padded_slice` (`padded_slice_rm`).
- The reader is op-owned and bound by no other factory (`grep -rl reader_unary_unpad_dims_interleaved_start_id_shard_optimized` hits only this factory). Convert it in place.

### Flags
- None. No unreferenced kernel files in the op directory.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept`.
- **Custom `compute_program_hash`**: none.
- **Implementation notes**: direct-descriptor op → exception 3. Add a nested `NlpKVCacheLoadSliceProgramFactory` with
  `create_program_artifacts` and declare `using program_factory_t = std::variant<NlpKVCacheLoadSliceProgramFactory>;`.
  The rest of the device-op class is untouched. The header comment about cache-hit refresh moves with the method,
  and its "CB c_0" wording follows the DFB rename. No pybind surface to remove.

## Planned Spec Shape

- KernelSpecs:
  - `reader`: reader `.cpp` (converted in place), `create_reader_datamovement_config()`.
  - `writer`: `writer_unary_sharded_metal2.cpp`, `create_writer_datamovement_config()`.
- DataflowBufferSpecs:
  - `out`: `entry_size = single_tile_size`, `num_entries = num_tiles_per_core`, `data_format_metadata = data_format`,
    no tile metadata (legacy `tile` unset), `borrowed_from = output`.
  - Bound by the reader as PRODUCER (accessor `in0`, from its `cb_id_in0` vocabulary) and by the writer as CONSUMER (accessor `out`, the fork's vocabulary).
- SemaphoreSpecs: none.
- TensorParameters:
  - `input`, from `input.tensor_spec()`. Bound by the reader as `tensor::src`.
  - `output`, from `output.tensor_spec()`. Used only as the `borrowed_from` backing; no kernel binding (the validator exempts borrow-only parameters).
- WorkUnitSpecs: one, `all_cores`, kernels {reader, writer}.
- Op-owned tensors: none.
- `ProgramRunArgs`: reader RTA `start_id` per core; writer RTA `num_units` per core; `tensor_args` = {input, output}.

## Preserved Multiplicity

none — no work-split multiplicity in legacy (one descriptor per kernel, over `all_cores`).

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `program_factory.cpp:98` reader RTA 0; reader `:21` | `src0_buffer` (`Buffer*`) → `get_arg_val<uint32_t>(0)` `src_addr` | `TensorBinding{input, "src"}`; kernel `TensorAccessor(tensor::src)` |
| `program_factory.cpp:73`; reader `:29` | `TensorAccessorArgs(src0_buffer).append_to(cta)` / `TensorAccessorArgs<5>()` | binding mechanism (same `TensorBinding`) |
| reader `:31` | hardcoded `constexpr uint32_t cb_id_in0 = 0` | `DFBBinding{out, "in0", PRODUCER}`; kernel `dfb::in0` |
| `program_factory.cpp:89` writer CTA 0 | `src0_cb_index` (= c_0) | `DFBBinding{out, "out", CONSUMER}` (fork reads `dfb::out`) |
| `program_factory.cpp:67-72` reader CTAs 0–4 | positional | named: `num_tiles`, `num_unpadded_tiles_head_dim`, `num_unpadded_tiles_seqlen_dim`, `num_padded_tiles_seqlen_dim`, `num_readers` |
| `program_factory.cpp:98` reader RTA 1 | positional `start_id` | named RTA `start_id` |
| `program_factory.cpp:99` writer RTA 0 | positional `num_tiles_per_core` | named RTA `num_units` (fork vocabulary) |
| `program_factory.cpp:49-58` | `CBDescriptor{.buffer = dst_buffer}` | `DataflowBufferSpec{.borrowed_from = output}` + `TensorArgument{output}` |

No page-size 3rd-argument sites. No semaphore IDs.

## Applied Patterns

- Borrowed-memory DFB (migration guide, "Borrowed-memory DFBs"): `out` borrowed from the `output` TensorParameter.
- Reuse an existing `_metal2` fork, rung 1 ([Caution: Porting a shared kernel](port_patterns.md#caution-porting-a-shared-kernel)): writer → `writer_unary_sharded_metal2.cpp`.
- `constexpr` DFB metadata keeps the token form (whitelist §A): reader `constexpr uint32_t tile_size = get_tile_size(cb_id_in0)`
  feeds the `get_barrier_read_threshold<tile_size, num_readers>()` template → `get_tile_size(dfb::in0)`.
- Node-first RTA loop kept, transposed with `AddRuntimeArgsForNode`.

## Deferred / Flagged

- New findings during planning: none. The audit's "Misc anomalies" (ignored `memory_config`, unvalidated preallocated
  output, `num_cores_x` taken from the first grid range) are preserved as-is and carried to the report.
