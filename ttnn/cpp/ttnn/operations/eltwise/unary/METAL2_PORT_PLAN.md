# Port Plan — `eltwise/unary`

Port plan for `ttnn/cpp/ttnn/operations/eltwise/unary` (`UnaryDeviceOperation::ProgramFactory`), ported from the
`ProgramDescriptor` API to Metal 2.0 (`CustomProgramSpecFactoryConcept`).
Written during the inventory and planning steps; committed alongside the port for review.

Line numbers below are pre-port (`edf75ffab9c`).

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` (`create_descriptor` returns a `ProgramDescriptor`, `device/unary_device_operation.hpp:44`), plus a `void` `override_runtime_arguments` (`:51-56`, body `device/unary_program_factory.cpp:570-666`).
- Where the methods live: in a nested `ProgramFactory` struct, in `program_factory_t = std::variant<ProgramFactory>` (`unary_device_operation.hpp:43-59`). This is **not** the direct-descriptor shape, so `ttnn_factory.md` exception 3 does not apply.
- Variants: single. There is no `select_program_factory`.
- Custom `compute_program_hash`: present at `device/unary_device_operation.cpp:284-350`, plus the backdoor `operation_attributes_t::to_hash()` (`:113-123`). Left intact **except** for the user-sanctioned `distribution_key` source swap. See [TTNN ProgramFactory](#ttnn-programfactory).

The factory has two data paths, selected by `has_sharding = get_shard_specs(...).has_value()` (`common/unary_utils.cpp:61-112`):
- **(A) accessor path** (`has_sharding == false`): interleaved, plus every mixed / ND / DRAM / uneven / non-tile-aligned-RM sharded case. There is a ROW_MAJOR sub-variant (`rm_interleaved`).
- **(B) native-L1-sharded path** (`has_sharding == true`). This implies both tensors are 2D-sharded in L1 on one grid, so `src_sharded == dst_sharded == has_sharding`.

### Kernels
| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_unary.cpp` | `worker_grid` | `TensorAccessorArgs(*src_buffer, RuntimeTensorShape)` CTA block (`:460-463`) | — | (B): `[src_addr (Buffer*), num_pages, start_id]`; (A): those plus `[chunks_per_row, chunk_size, last_chunk_size, rows_per_tile, total_rows]` (zeros unless `rm_interleaved`). Noop cores: all zero. | accessor CRTA block (shape) | `SRC_SHARDED`, `RM_INTERLEAVED` ("0"/"1") | O2 (unset, DM) | `ReaderConfigDescriptor{}` = RISCV_1 / NOC_0 / dedicated |
| writer | `device/kernels/dataflow/writer_unary.cpp` | `worker_grid` | `TensorAccessorArgs(*dst_buffer, RuntimeTensorShape)` CTA block (`:479-482`) | — | same shape as reader with `dst_addr` (`Buffer*`) | accessor CRTA block | `DST_SHARDED`, `RM_INTERLEAVED` | O2 (unset, DM) | `WriterConfigDescriptor{}` = RISCV_0 / NOC_1 / dedicated |
| compute | runtime-selected, 1 of 9 under `device/kernels/compute/` (`get_compute_kernel_path`, `common/unary_op_utils.cpp:1194-1212`) | `worker_grid` | HARDSWISH: `[INP_FLOAT32?, INP_INT32\|INP_UINT32?, cb_data_format]`; LOGIT: `[logit_clamp_enabled, cb_data_format]`; else `[cb_data_format]` (`:498-506`) | — | `[num_tiles, packed_scalar1, packed_scalar2]` (noop: zeros) | none | `get_block_defines(op_chain)` + `add_input_dtype_defines` | O3 (unset, compute) | `ComputeConfigDescriptor{HiFi4, fp32_dest_acc_en, unpack_to_dest_mode, bfp8_pack_precise, math_approx_mode=false}`; `dst_full_sync_en` left at its default `false` |

The runtime-selected compute sources are `eltwise_sfpu.cpp` (the default), `eltwise_identity_kernel.cpp`, `where_tss_kernel.cpp`, `mac_tss_kernel.cpp`, `logit_kernel.cpp`, `hardswish_kernel.cpp`, `logsigmoid_kernel.cpp`, `lgamma_fast_kernel.cpp` (LGAMMA with bf16), and `lgamma_kernel.cpp` (LGAMMA with other dtypes).

Only the HARDSWISH and LOGIT kernels read CTAs (`hardswish_kernel.cpp:14-15`, `logit_kernel.cpp:16`). The trailing `cb_data_format` CTA is read by no kernel. Compute RTA 0 is read by all 9 kernels. RTAs 1 and 2 are read only by `where_tss`, `mac_tss`, and `logit`.

### CBs
| index | total_size | core_ranges | data_format | page_size | tile | buffer |
|---|---|---|---|---|---|---|
| `c_0` (src0) | `tile_size(in) × (src shard pages \| 2)` | `worker_grid` | `cb_data_format_for_input` (the output format under BITCAST, otherwise the input format) | `tile_size(input_df)` | unset | `src_buffer` iff `src_sharded` |
| `c_1` (tmp0) | `tile_size(in) × 2` | `worker_grid` | `cb_data_format` (input) | `tile_size(input_df)` | unset | — **only when `op_chain[0] == LOGIT`** |
| `c_2` (output) | `tile_size(out) × (dst shard pages \| 2)` | `worker_grid` | output format | `tile_size(output_df)` | unset | `dst_buffer` iff `dst_sharded` |

`unpack_to_dest_mode[c_0] = unpack_to_dest_mode[c_1] = UnpackToDestFp32` under `preserve_fp32_precision`, and `Default` otherwise (`:397-404`). The only caller of `prim::unary` (`unary.cpp:56-60`) sets `preserve_fp32_precision` exactly when the input is FLOAT32, and that also forces `fp32_dest_acc_en`.

There is no GlobalCircularBuffer, no `address_offset`, and no aliasing.

### Semaphores
none

### Tensor accessors
| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `unary_program_factory.cpp:462` (`TensorAccessorArgs(*src_buffer, RuntimeTensorShape)`) → `reader_unary.cpp:25-26` | `input` | reader RTA0 (`input.buffer()`, `:531/536/555`; `->address()` on a hit, `:585,613`) |
| `unary_program_factory.cpp:481` → `writer_unary.cpp:27-28` | `output` | writer RTA0 (`output.buffer()`, `:532/546/557`; `:586,616`) |

On path (B) both accessors are compiled out (`#if SRC_SHARDED` / `#if DST_SHARDED`). There the accessor payload and RTA0 are dead.

### Work split
- Driver: `enumerate_core_rt_args` (`:127-333`), shared by the miss path and the override.
- (A): `split_work_to_cores(compute_with_storage_grid | worker_grid, out_num_tiles, row_major)` gives `(num_cores, all_cores, core_group_1, core_group_2, npc_1, npc_2)`. Cores outside both groups inside `worker_grid` are noop and zero-filled.
- (B): `core_group_1 = shard grid`. Per-core tiles come from the shard spec (last-row / last-column tails). Cores outside the shard grid inside `worker_grid` are noop.
- Every core of `worker_grid` gets RTAs on both paths, so Metal 2.0's "every named RTA on every node" rule is met by construction.

### Shared kernels
Census: grep each bound filename, then disambiguate each hit. The `tests/tt_metal/**` and `tests/tt_eager/**` hits are same-named private copies. The only split-literal site is `unary_program_factory.cpp:412-414`, and `get_compute_kernel_path` has this factory as its only caller.
- `compute/eltwise_sfpu.cpp` — **lent.** Other binders: `examples/example` (`single_core_program_factory.cpp:91`, `multi_core_program_factory.cpp:89`), `examples/example_multiple_return` (`single_core_program_factory.cpp:80`), `tests/ttnn/unit_tests/gtests/test_generic_op.cpp:246`, and `tests/ttnn/unit_tests/operations/fused/parallel_sequential/test_parallel_sequential.py:1436`. There is **no** `_metal2` sibling (`ls device/kernels/compute/`). **Rung 2:** create `compute/eltwise_sfpu_metal2.cpp` beside it and add the pointer comment to the original. The `eltwise/unary/CMakeLists.txt` `GLOB_RECURSE kernels device/kernels/*.cpp` installs it.
- The other 10 bound kernels have no other binder and convert in place.

### Flags
- Ten unbound kernels in `device/kernels/dataflow/`, including four `_metal2` forks made by other ops' ports. None is a fork of `reader_unary.cpp` or `writer_unary.cpp`. All are out of scope and untouched.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `CustomProgramSpecFactoryConcept`. The ported-from factory has `override_runtime_arguments` (`unary_program_factory.cpp:570-666`).
- **Custom `compute_program_hash`**: present at `unary_device_operation.cpp:284-350` (+ `to_hash()` `:113-123`). **One sanctioned edit**, per the user's decision 1 and the relaxation doc §2: `distribution_key` (`:323-336`) swaps its source from the Buffer's `buffer_distribution_spec()` to `spec.compute_buffer_sharding_args()` on both slots. The swap lands in the same change as the relaxation declaration, and the `TODO(port)` (`:320-322`) is retired. Nothing else in the hash changes.
- **Implementation notes**:
  - The override becomes `static ProgramRunArgs override_runtime_arguments(attrs, tensor_args, output, coord)`. It drops the `Program&` parameter.
  - Miss and hit build their `kernel_run_args` + `tensor_args` through one shared helper that wraps `enumerate_core_rt_args`. The legacy override writes exactly the slot set `create_descriptor` writes: every RTA on every core, noop zero-fill included. So the override set is the same as the miss set, and one helper mirrors it without drift.

## Planned Spec Shape

- **KernelSpecs**: `reader`, `writer`, `compute`, one per legacy `KernelDescriptor`.
  - `compute.source` is the runtime-selected path, and `get_compute_kernel_path` now returns `eltwise_sfpu_metal2.cpp` as its default.
  - compute: `opt_level = O3`, set explicitly. DM: default `O2`.
- **DataflowBufferSpecs**:
  - `in` (c_0): `entry_size = tile_size(input_df)`, `num_entries = src shard pages | 2`, `data_format_metadata = cb_data_format_for_input`, `borrowed_from = INPUT` iff `src_sharded`.
  - `tmp0` (c_1): exists **only** when `op_chain[0] == LOGIT`. `entry_size = tile_size(input_df)`, `num_entries = 2`, input format.
  - `out` (c_2): `entry_size = tile_size(output_df)`, `num_entries = dst shard pages | 2`, output format, `borrowed_from = OUTPUT` iff `dst_sharded`.
  - `tile_format_metadata` stays unset, as in legacy.
- **SemaphoreSpecs**: none.
- **TensorParameters**: `input` and `output`, each from its tensor's `tensor_spec()`. Per the relaxation doc §2, **both** get `.relaxations = {.dynamic_tensor_shape = true, .relax_logical_rank = true}`, unconditionally. `match_page_size` and `match_padded_shape_only` are not set.
- **WorkUnitSpecs**: one, `{reader, writer, compute}` on `worker_grid`.

## Preserved Multiplicity

none — no work-split multiplicity in legacy (one `KernelDescriptor` per role over `worker_grid`; per-core counts are RTAs).

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader RTA0 (`unary_program_factory.cpp:531,536,555`; hit `:613`) | `input.buffer()` / `src_addr` | `TensorBinding{INPUT, "src"}` on reader → `TensorAccessor(tensor::src)` |
| writer RTA0 (`:532,546,557`; hit `:616`) | `output.buffer()` / `dst_addr` | `TensorBinding{OUTPUT, "dst"}` on writer → `TensorAccessor(tensor::dst)` |
| reader CTA block + CRTA block (`:460-463`; hit `:640-645`) | `TensorAccessorArgs(*src_buffer, RuntimeTensorShape).append_to(...)`; kernel `TensorAccessorArgs<0, 0>()` (`reader_unary.cpp:25`) | the binding (+ `dynamic_tensor_shape` relaxation, which re-emits shape as implicit CRTAs) |
| writer CTA block + CRTA block (`:479-482`; hit `:646-653`) | same for `dst` (`writer_unary.cpp:27`) | the binding |
| sharded CB addresses (hit `:655-665`, `apply_descriptor_runtime_args`) | `CBDescriptor{.buffer = ...}` re-applied | `borrowed_from` + the `tensor_args` returned by the override |
| reader `cb_id_src = tt::CBIndex::c_0` (`reader_unary.cpp:15`) | magic CB index | `DFBBinding{in, "src", PRODUCER}` → `dfb::src` |
| writer `cb_id_dst = tt::CBIndex::c_2` (`writer_unary.cpp:15`) | magic CB index | `DFBBinding{out, "dst", CONSUMER}` → `dfb::dst` |
| every compute kernel's `c_0` / `c_1` / `c_2` locals | magic CB index | `dfb::in` (CONSUMER), `dfb::tmp0` (PRODUCER+CONSUMER, LOGIT only), `dfb::out` (PRODUCER) |
| reader/writer RTAs 1-7 | positional `get_arg_val<uint32_t>(1..7)` | named: `num_pages`, `start_id`, `chunks_per_row`, `chunk_size`, `last_chunk_size`, `rows_per_tile`, `total_rows`. Path (B) declares only `num_pages`, `start_id`, matching legacy's 3-slot shape minus the address. |
| compute RTAs 0-2 | positional | named: `num_tiles`, `packed_scalar1`, `packed_scalar2` (declared on every compute source, as legacy always sent 3) |
| hardswish CTAs 0-1 | positional | named: `is_float32`, `is_int` |
| logit CTA 0 | positional | named: `do_clamp` |
| trailing compute CTA (`:506`) | positional `cb_data_format`, unread | named `input_data_format`, carried as-is, not dropped |
| `get_local_cb_interface(cb_id_*).fifo_page_size` (`reader_unary.cpp:57`, `writer_unary.cpp:60`) | `LocalCBInterface` field read (non-`constexpr` local) | `dfb_*.get_entry_size()` (whitelist §B) |

No semaphore-ID RTAs or page-size 3rd-argument sites exist.

## Applied Patterns

- [Self-loop DFB binding](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-self-loop-dfb-binding): `tmp0` on the LOGIT compute kernel (PRODUCER + CONSUMER, one accessor name). I re-derived it from the census: `logit_kernel.cpp` is the only toucher.
- [Conditional / optional resource bindings](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-conditional--optional-resource-bindings): the `tmp0` DFB spec and binding exist only for LOGIT, as in legacy. No kernel `#ifdef` is needed: the only kernel that references `dfb::tmp0` is `logit_kernel.cpp`, and it is selected exactly when the binding exists. The `unpack_modes` `tmp0` entry is gated on the same condition.
- Borrowed-memory DFBs: `in` / `out` on path (B), `borrowed_from` INPUT / OUTPUT.
- [Pass DFB handles directly to LLKs and kernel-lib helpers](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers): `compute_kernel_hw_startup`, `copy_init`, `copy_tile`, `pack_tile`, and `ckl::input` / `ckl::output` (NTTP position).
- [Porting a shared kernel](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-porting-a-shared-kernel), rung 2: `eltwise_sfpu_metal2.cpp`.

## Hardware configuration (planned)

- reader: `create_reader_datamovement_config(arch)`, the reader default triple.
- writer: `create_writer_datamovement_config(arch)`, the writer default triple.
- compute: Style B, a `ComputeGen1Config` built directly because legacy set a `ComputeConfigDescriptor` literally.
  - `fpu_math_fidelity = HiFi4`
  - `sfpu_precision_mode = Precise` (`math_approx_mode = false`)
  - `enable_32_bit_dest = fp32_dest_acc_en`
  - `bfp_pack_precision_mode = bfp8_pack_precise ? Precise : Approximate`
  - `double_buffer_dest = true` (legacy `dst_full_sync_en` default `false`)
  - `unpack_modes`, for each DFB compute consumes (`in`, and `tmp0` when present):
    - `preserve_fp32_precision` → `UnpackToDest` (legacy `UnpackToDestFp32`).
    - Otherwise, if `fp32_dest_acc_en` and the DFB format is Float32 → an explicit `UnpackToSrc` (legacy `Default`). Metal 2.0 requires this entry. It is reachable through BITCAST into FLOAT32 from a non-FLOAT32 input.
    - Otherwise no entry.

## Deferred / Flagged

- **Hash edit (sanctioned exception).** See above. It is recorded as a device-op-class edit and a recipe-deviation handoff in the report.
- **Header includes.** `unary_device_operation.hpp` swaps `program_descriptors.hpp` / `program_descriptor_patching.hpp` for `ttnn/metal_v2_artifacts.hpp`. This is forced by the factory-struct signature change.
- **`get_compute_kernel_path`** (`common/unary_op_utils.cpp:1210`) is a helper only this factory calls. Its default return changes to the fork's filename. This is a factory-helper edit, not a device-op-class edit.
