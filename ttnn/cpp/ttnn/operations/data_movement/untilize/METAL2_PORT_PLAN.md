# Port Plan — untilize (`UntilizeMultiCoreInputAndOutputNDShardTypeAndShardSpecIdenticalProgramFactory`)

Port plan for `data_movement/untilize`, factory
`UntilizeMultiCoreInputAndOutputNDShardTypeAndShardSpecIdenticalProgramFactory`, ported from the
`ProgramDescriptor` API (`create_descriptor`) to Metal 2.0 (`create_program_artifacts`).
Written during the inventory and planning steps; committed alongside the port for review.

Scope: this one factory. The other seven factories of `UntilizeDeviceOperation` are already Metal 2.0.
`codegen/` (`UntilizeCodegenDeviceOperation`) is a separate device-op and out of scope.

## Legacy Inventory

Source: `device/factories/untilize_multi_core_input_and_output_nd_shard_type_and_shard_spec_identical_program_factory.{hpp,cpp}` (line numbers below are pre-port).

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` (`create_descriptor` → `ProgramDescriptor`, `.hpp:14`)
- Where the factory methods live: a struct in `UntilizeDeviceOperation::program_factory_t` (`device/untilize_device_operation.hpp:46`) — not the direct-descriptor shape, so exception 3 does not apply.
- Variants: single
- Custom `compute_program_hash`: none — default reflection-based hash (no `compute_program_hash` / `attribute_values` / `to_hash` under `untilize/device/`).
- Selection: `device/untilize_device_operation.cpp:332-339` — input and output both L1-sharded, same memory layout, no matching legacy `shard_spec`, identical `nd_shard_spec`, and arch != Quasar. Left untouched.

### Kernels
| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp` (borrowed) | `grid` (= `nd_shard_spec.grid`) | `[0] src0_cb_index = c_0` | none | per core: `[0] num_tiles_to_process` | none | none | O2 (unset, DM) | `ReaderConfigDescriptor{}` (RISCV_1 / NOC_0 / dedicated) |
| writer | `data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp` (borrowed) | `grid` | `[0] output_cb_index = c_16` | none | per core: `[0] num_tiles_to_process` | none | none | O2 (unset, DM) | `WriterConfigDescriptor{}` (RISCV_0 / NOC_1 / dedicated) |
| compute | `data_movement/untilize/device/kernels/compute/untilize_variable_num_blocks.cpp` | `grid` | `[0] num_tiles_per_block`, `[1] src0_cb_index = c_0`, `[2] output_cb_index = c_16` | none | per core: `[0] num_blocks_to_process` | none | `compute_kernel_defines` — always empty (`.cpp:138`) | O3 (unset, compute) | `ComputeConfigDescriptor{.fp32_dest_acc_en, .unpack_to_dest_mode}` — `unpack_to_dest_mode[c_0] = UnpackToDestFp32` iff `fp32_dest_acc_en`, else all `Default` (`.cpp:139-142`, `.cpp:151-154`); all other fields default |

Per-core RTA values (`.cpp:157-176`): walk `corerange_to_cores(grid, nullopt, orientation == ROW_MAJOR)`; a core found in
`distribution_spec.compute_page_mapping().all_cores` at index `i` gets
`num_blocks_to_process = num_blocks_per_shard * distribution_spec.num_shards_per_core(i)` and
`num_tiles_to_process = num_tiles_per_block * num_blocks_to_process`; an unmapped core gets 0 for both.

### CBs
| index | total_size | core_ranges | data_format | page_size | tile (if set) | buffer |
|---|---|---|---|---|---|---|
| `c_0` | `num_tiles_per_shard * num_shards_per_core * input_single_tile_size` | `grid` | `datatype_to_dataformat_converter(a.dtype())` | `input_single_tile_size` | not set | `src0_buffer` (borrowed, `.cpp:100`) |
| `c_16` | `num_tiles_per_shard * num_shards_per_core * output_single_tile_size` | `grid` | `datatype_to_dataformat_converter(output.dtype())` | `output_single_tile_size` | not set | `dst_buffer` (borrowed, `.cpp:114`) |

`num_shards_per_core = distribution_spec.core_groups().num_shards_per_core_in_group_1` (`.cpp:76`).
No GlobalCircularBuffer; no `address_offset`.

### Semaphores
none

### Tensor accessors
none — no kernel constructs a `TensorAccessor`; both tensors reach the kernels only through borrowed CBs.

### Work split
- Driver: no `split_work_to_cores`. Work follows the ND shard distribution: `distribution_spec.compute_page_mapping()` + `num_shards_per_core(i)`.
- All three kernels on every core of `nd_shard_spec.grid`; cores with no shard get zero counts.

### Shared kernels
All three bound sources are shared; each already has a `_metal2` fork beside it → **rung 1 (reuse)**, no fork created, no edit to any fork or original.

| original | fork (bound by the port) | fork vocabulary | `#ifdef`s | legacy consumers left |
|---|---|---|---|---|
| `eltwise/unary/.../reader_unary_sharded.cpp` | `reader_unary_sharded_metal2.cpp` | `dfb::in`; RTA `args::num_tiles_per_core` | none | `experimental/slice_write`, `data_movement/sharded_partial/sharded_to_interleaved_partial` |
| `data_movement/sharded/.../writer_unary_sharded.cpp` | `writer_unary_sharded_metal2.cpp` (#52228) | `dfb::out`; RTA `args::num_units` | none | `data_movement/sharded_partial/interleaved_to_sharded_partial` |
| `data_movement/untilize/.../untilize_variable_num_blocks.cpp` | `untilize_variable_num_blocks_metal2.cpp` | `dfb::src`, `dfb::out`; RTA `args::per_core_block_cnt`; CTA `args::per_core_block_tile_cnt` | none | `experimental/deepseek_prefill/update_padded_kv_cache` |

### Flags
- `.cpp:73-82`: `total_shards` / `num_cores` feed only a `log_debug`. Kept as-is (also used during verification to confirm the factory is reached).
- `.cpp:157`: `orientation` only orders the per-core walk; each core gets its own RTAs, so it has no effect. Kept.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept` (brief: code-derived; readiness-sheet row missing, gate user-waived).
- **Custom `compute_program_hash`**: none.
- **Implementation notes**: swap `create_descriptor` for `create_program_artifacts` returning `ttnn::device_operation::ProgramArtifacts`, header shaped like the 7 sibling factories. No device-op / nanobind edit.

## Planned Spec Shape

- KernelSpecs: `reader`, `writer`, `compute` (1:1 with legacy descriptors), bound to the three forks.
- DataflowBufferSpecs:
  - `src0` (legacy `c_0`): `entry_size = input_single_tile_size`, `num_entries = num_tiles_per_shard * num_shards_per_core`, `data_format_metadata = input format`, `borrowed_from = input`.
  - `out` (legacy `c_16`): same with output tile size / format, `borrowed_from = output`.
- SemaphoreSpecs: none.
- TensorParameters: `input` (`a.tensor_spec()`), `output` (`output.tensor_spec()`) — needed for `borrowed_from`; supplied in `run_args.tensor_args` from `mesh_tensor()`.
- WorkUnitSpecs: one, `main`, kernels `{reader, writer, compute}`, `target_nodes = nd_shard_spec.grid`.
- Op-owned tensors: none.

## Preserved Multiplicity

none — no work-split multiplicity in legacy (one descriptor per kernel).

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader CTA slot 0 (`.cpp:118`) | `src0_cb_index = c_0` | `DFBBinding{src0, "in", PRODUCER}` |
| writer CTA slot 0 (`.cpp:127`) | `output_cb_index = c_16` | `DFBBinding{out, "out", CONSUMER}` |
| compute CTA slot 1 (`.cpp:136`) | `src0_cb_index = c_0` | `DFBBinding{src0, "src", CONSUMER}` |
| compute CTA slot 2 (`.cpp:136`) | `output_cb_index = c_16` | `DFBBinding{out, "out", PRODUCER}` |
| compute CTA slot 0 (`.cpp:136`) | positional `num_tiles_per_block` | named CTA `per_core_block_tile_cnt` |
| reader RTA slot 0 (`.cpp:173`) | positional `num_tiles_to_process` | named RTA `num_tiles_per_core` |
| writer RTA slot 0 (`.cpp:175`) | positional `num_tiles_to_process` | named RTA `num_units` |
| compute RTA slot 0 (`.cpp:174`) | positional `num_blocks_to_process` | named RTA `per_core_block_cnt` |
| CB `buffer` (`.cpp:100`, `.cpp:114`) | `cb.buffer = src0_buffer / dst_buffer` | `DataflowBufferSpec::borrowed_from` + `TensorArgument` |
| `unpack_to_dest_mode` vector (`.cpp:139-142`) | indexed by CB id, `[c_0] = UnpackToDestFp32` iff fp32 | `unpack_modes = {{src0, UnpackToDest}}` iff fp32 |

No buffer-address RTAs, no `TensorAccessorArgs`, no page-size 3rd args, no semaphore IDs.

## Applied Patterns

- Borrowed-memory DFB (`borrowed_from`) for both `src0` and `out`.
- Pass DFB handles directly to kernel-lib helpers — already done in the reused compute fork.
- Porting a shared kernel, rung 1 (reuse existing `_metal2` fork) — all three kernels.
- `AddRuntimeArgsForNode` to keep the legacy node-first RTA loop unchanged.

## Deferred / Flagged

- Borrowed-DFB sizing over an ND-sharded tensor is new for this op (the brief's watch-for). Must be exercised on an uneven, multi-shard-per-core ND case in verification.
