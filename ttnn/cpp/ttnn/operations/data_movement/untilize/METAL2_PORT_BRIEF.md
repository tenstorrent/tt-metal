# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/data_movement/untilize`

> Audit cleared all gates, one of them on a user waiver (below). This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Port scope: one factory only.** Port `UntilizeMultiCoreInputAndOutputNDShardTypeAndShardSpecIdenticalProgramFactory` (`device/factories/untilize_multi_core_input_and_output_nd_shard_type_and_shard_spec_identical_program_factory.{hpp,cpp}`). The other 7 factories of `UntilizeDeviceOperation` are already `MetalV2`. `codegen/` (`UntilizeCodegenDeviceOperation`) is a separate device-op and out of scope.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ (*user-waived*: the readiness sheet has no row for this factory; code cross-check clean) · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `ed3eb002115 2026-11-03 docs(metal_2.0): DFBAccessor -> DFBBindingToken; drop the dead `.id` form` *(carry this line into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`). Carry them forward:

- **Current concept:** `descriptor` (`create_descriptor` → `ProgramDescriptor`, `.hpp:14`)
- **Op-owned tensors:** none
- **Target concept:** `ProgramSpecFactoryConcept`
  - Swap `create_descriptor` for `create_program_artifacts` returning `ttnn::device_operation::ProgramArtifacts`, matching the 7 sibling factories' headers.
  - No `override_runtime_arguments` to translate.
  - *Source note:* taken from code evidence and the 7 sibling rows, not from the sheet's `Porting Target` cell (no row exists; user waived). Record this in the port report's Provenance.
- **Gate-cleared, confirmed absent:** a non-clearing `TensorParameter relaxation` (code-derived `none`, user-waived sheet read) · `get_dynamic_runtime_args`. Also absent, though not gating: custom hash, `override_runtime_arguments`, pybound `create_descriptor`. No device-op or nanobind edit is needed.

## Construct — to do

**Template:** `untilize_multi_core_input_and_output_shard_type_and_shard_spec_identical_program_factory.cpp` (the already-ported 2D sibling) has the same shape: borrowed DFB pair, same reader/writer forks, one work unit over the shard grid. Port this factory by following it, with the three differences listed under **Kernels** below.

**Tensor bindings** (per binding):

- `input` — **clean** (borrowed-memory DFB; legacy `cb_src0.buffer = src0_buffer`, `.cpp:81`)
  - `TensorParameter{.unique_id = INPUT, .spec = a.tensor_spec()}`
  - `DataflowBufferSpec SRC0{… .borrowed_from = INPUT}`
  - `run_args.tensor_args` entry `{INPUT, TensorArgument{a.mesh_tensor()}}`
- `output` — **clean** (borrowed-memory DFB; legacy `cb_output.buffer = dst_buffer`, `.cpp:95`)
  - same pattern with `OUTPUT` / `OUT`

No Case 1 / Case 2 work: no buffer address reaches any kernel argument.

**DFBs** (replacing the two `CBDescriptor`s at `.cpp:73-97`):

- `SRC0` (legacy `c_0`): `entry_size = input_single_tile_size`, `num_entries = num_tiles_per_shard * num_shards_per_core`, `data_format_metadata = input format`, `borrowed_from = INPUT`. Here `num_shards_per_core = groups.num_shards_per_core_in_group_1`, `.cpp:57`.
- `OUT` (legacy `c_16`): same, with the output tile size and format, `borrowed_from = OUTPUT`.

**Kernels:** bind the existing `_metal2` forks (see Watch for):

| Kernel | Source | DFB bindings | Args |
|---|---|---|---|
| reader | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded_metal2.cpp` | `SRC0` as accessor `in`, PRODUCER | RTA `num_tiles_per_core` = legacy `num_tiles_to_process` |
| writer | `ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_metal2.cpp` | `OUT` as accessor `out`, CONSUMER | RTA `num_units` = legacy `num_tiles_to_process` |
| compute | `ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_variable_num_blocks_metal2.cpp` | `SRC0` as `src`, CONSUMER; `OUT` as `out`, PRODUCER | CTA `per_core_block_tile_cnt` = `num_tiles_per_block`; **RTA** `per_core_block_cnt` = legacy `num_blocks_to_process` |

Differences from the 2D template:

1. **Compute source and arg kind.** Compute binds `untilize_variable_num_blocks_metal2.cpp`, not `untilize_metal2.cpp`. `per_core_block_cnt` is a **per-node RTA**: `runtime_arg_schema = {"per_core_block_cnt"}` plus a `KernelRunArgs` for compute. It is not a CTA. The ported `untilize_multi_core_nd_shard_input_program_factory.cpp:173-183` already binds this fork exactly this way; copy its compute `KernelSpec`.
2. **DFB sizing** uses ND geometry (above), not `shard_height / tile_height`.
3. **Per-node RTAs** come from the legacy loop at `.cpp:138-157`, kept unchanged:
   - walk `corerange_to_cores(grid, …, orientation == ROW_MAJOR)`;
   - look up each core in `distribution_spec.compute_page_mapping().all_cores`;
   - a mapped core gets `num_blocks_per_shard * num_shards_per_core(core_idx)`;
   - an **unmapped core gets 0** for all three RTAs.

   Emit `AddRuntimeArgsForNode` for reader, writer **and compute**. The 2D template has no compute run-args, so you add that third `KernelRunArgs`.

**Compute config** (as both ported siblings do):

- `ComputeHardwareConfig{.enable_32_bit_dest = fp32_dest_acc_en}`.
- If `fp32_dest_acc_en`, `unpack_modes.insert({SRC0, UnpackMode::UnpackToDest})`. This replaces the legacy `UnpackToDestFp32` on `c_0`, `.cpp:120-123`.
- `compiler_options.opt_level = KernelBuildOptLevel::O3`. Legacy compute was built at -O3; Metal 2.0 defaults compute to -O2.
- The legacy `compute_kernel_defines` is always empty (`.cpp:119`), so pass an empty `Defines`.

**Work unit:** a single `WorkUnitSpec` with all three kernels, `target_nodes = nd_shard_spec.grid`. This matches legacy, where all three kernels run on the whole grid, including shard-less cores.

**TensorParameter relaxation:** `none` (code-derived; see TTNN factory analysis).

**TensorAccessor 3rd arg:** none. No kernel constructs a `TensorAccessor`.

**CB endpoints:** all legal. `SRC0` is reader PRODUCER → compute CONSUMER; `OUT` is compute PRODUCER → writer CONSUMER. Both are 1:1 in every config, including shard-less cores, where compute early-returns and the DM kernels push/wait 0. No self-loop, multi-binding, dead-CB drop or conditional DFB.

## Watch for

- **CB endpoints (multi-binding):** none.
- **Cross-op / shared kernels — rung 1 (reuse) for all three.** Each `_metal2` fork already exists beside its original. Fit was checked: binding names, named args, and no `#ifdef`s in any fork.
  - Bind each fork as-is. **Do not edit any fork** (each already has consumers) and do not create new ones.
  - Do not add pointer comments to the originals: all three already carry one.
  - Paths: `eltwise/unary/device/kernels/dataflow/reader_unary_sharded_metal2.cpp` · `data_movement/sharded/device/kernels/dataflow/writer_unary_sharded_metal2.cpp` (issue #52228) · `data_movement/untilize/device/kernels/compute/untilize_variable_num_blocks_metal2.cpp`.
  - Legacy consumers left after this port — **sunset list, not authorization to convert any original in place:**
    - `reader_unary_sharded.cpp`: `experimental/slice_write` (tiled + RM sharded-input factories), `data_movement/sharded_partial/sharded_to_interleaved_partial`.
    - `writer_unary_sharded.cpp`: `data_movement/sharded_partial/interleaved_to_sharded_partial`.
    - `untilize_variable_num_blocks.cpp`: `experimental/deepseek_prefill/update_padded_kv_cache`.

    Record these in the port report.
  - Do **not** use anything under `ttnn/cpp/ttnn/operations/experimental/quasar/untilize/` as a reference. It carries a copy of this same factory with identical class names; it is out of bounds.
- **RTA varargs:** none. Every RTA and CTA is a fixed, nameable field.
- **Borrowed-DFB sizing validator.** `ValidateBorrowedMemory` (`tt_metal/impl/metal2_host_api/program_spec/validation/resource/dfb.cpp:63-88`) `TT_FATAL`s if `entry_size * num_entries` exceeds the tensor's per-bank allocation. Group-1 shard count × shard tiles × tile size should equal that allocation.
  - This is the first borrowed DFB over an ND-sharded tensor in this op: the ND-input sibling allocates its own DFBs.
  - Make sure the test run includes an **uneven, multi-shard-per-core** ND case. A failure here is loud, not silent.
- **Leave the Quasar exclusion alone** at `device/untilize_device_operation.cpp:335-339`. The factory stays Gen1-only after the port; lifting the guard is a behaviour change outside port scope.
- **Kernel side is already modernized.** All three forks are complete Metal 2.0 kernels, so this port is **host-side only**. If you find yourself editing a kernel, re-check the binding names against the fork instead.
- **Tests:** the factory is reached only with ND-sharded input and output having an identical `nd_shard_spec`, on WH/BH. Use `python .cursor/commands/ttnn/find_operation_tests.py untilize` and pick the ND-sharded identical-spec cases, including uneven sharding.
