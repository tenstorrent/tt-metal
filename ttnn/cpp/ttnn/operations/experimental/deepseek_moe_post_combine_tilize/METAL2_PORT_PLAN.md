# Port Plan — deepseek_moe_post_combine_tilize

Port plan for `experimental/deepseek_moe_post_combine_tilize`, ported from the `ProgramDescriptor` API (direct-descriptor shape) to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept`, reached through the **direct-descriptor** shim (`HasDirectDescriptor`). `create_descriptor` is a static member of `DeepseekMoEPostCombineTilizeDeviceOperation` itself (`device/deepseek_moe_post_combine_tilize_device_operation.hpp:27`, defined at `device/deepseek_moe_post_combine_tilize_program_factory.cpp:23`). There is no factory struct and no `program_factory_t`, so [exception 3] (recipe `shared/ttnn_factory.md#3-give-a-direct-descriptor-op-a-conventional-program-factory`) applies.
- Variants: single.
- Custom `compute_program_hash`: none. There is no `compute_program_hash`, `attribute_values` or `to_hash` on the device-op, so it uses the default reflection-based hash.
- `override_runtime_arguments`: none. It was removed by #57409, and the comment at `device_operation.hpp:24-26` records that.

### Kernels
| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/deepseek_moe_post_combine_tilize_reader.cpp` | `op_cores` (= output `nd_shard_spec.grid`) | `TensorAccessorArgs(input_tensor.buffer())` appended at offset 0 (`program_factory.cpp:93-94`) | `tilize_input_cb_id`=`c_0`, `input_row_page_size`=`input.buffer()->page_size()`, `bytes_to_read_per_row`=`output_shard_width_bytes` (`:103-107`) | per core: [0] `intra_row_byte_offset`, [1] `row_page_offset`, [2] `input_tensor.buffer()` (a `Buffer*` binding) (`:147-164`) | none | none | `O2` (explicit, `:108`) | `DataMovementConfigDescriptor{RISCV_1, NOC_0, DM_DEDICATED_NOC}` (`:109-113`) |
| compute | `device/kernels/deepseek_moe_post_combine_tilize_compute.cpp` | `op_cores` | none | `tilize_input_cb_id`=`c_0`, `tilize_output_cb_id`=`c_1`, `num_tiles`=`output_shard_width_tiles` (`:122-126`) | none | none | none | unset, so it resolves to **`O3`** (a compute descriptor with `opt_level == nullopt` → O3) | `ComputeConfigDescriptor{}`, all defaults (`:127`) |
| writer | `device/kernels/deepseek_moe_post_combine_tilize_writer.cpp` | `op_cores` | none | `tilize_output_cb_id`=`c_1`, `num_tiles`=`output_shard_width_tiles` (`:136-139`) | none | none | none | `O2` (explicit, `:140`) | `DataMovementConfigDescriptor{RISCV_0, NOC_1, DM_DEDICATED_NOC}` (`:141-145`) |

### CBs
| index | total_size | core_ranges | data_format | page_size | tile (if set) | backing |
|---|---|---|---|---|---|---|
| `c_0` (`tilize_input_cb_id`) | `TILE_HEIGHT * output_shard_width_bytes` (→ 32 pages) | `op_cores` | `datatype_to_dataformat_converter(input.dtype())` (bf16) | `output_shard_width_bytes` | not set | own L1 (`:62-71`) |
| `c_1` (`tilize_output_cb_id`) | `output_shard_width_tiles * output_tile_page_size` (→ `output_shard_width_tiles` pages) | `op_cores` | same | `output_tile_page_size` | not set | **borrowed** from `output_tensor.buffer()` (`:76-86`) |

### Semaphores
none

### Tensor accessors
| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `program_factory.cpp:94` (`TensorAccessorArgs(input_tensor.buffer()).append_to(reader_ct_args)`), kernel `reader.cpp:25-26` | `tensor_args.input_tensor` (interleaved, ROW_MAJOR, bf16) | reader RTA[2], `Buffer*` via `emplace_runtime_args` (`:163`) |

The output tensor has no accessor. It is reached only as the backing memory of CB `c_1`.

### Work split
- No `split_work_to_cores`. Each core of `op_cores` owns one output shard (validated: `num_shards_wide * num_shards_high == grid.num_cores()`).
- Per-core iteration: `corerange_to_cores(op_cores, std::nullopt, is_row_major_shard_orientation)` (`:148-149`). The per-core `intra_row_byte_offset` / `row_page_offset` come from the index `i` in that order, with separate ROW_MAJOR and COL_MAJOR formulas (`:156-162`).

### Shared kernels
none. Each of the three kernel files is referenced only by this op's factory (`grep -rl` over `ttnn/cpp/ttnn/operations/` hits only `program_factory.cpp`, plus the out-of-bounds `experimental/quasar/` tree, which is excluded). They convert in place, and no `_metal2` fork is involved.

### Flags
- **Unread named CTA `input_row_page_size`** (`program_factory.cpp:105`). No kernel reads it. It is not a dead-CB CTA, so the dead-CB drop rule does not cover it.
- No unreferenced kernel files in the op directory.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept`. There is no `override_runtime_arguments`.
- **Custom `compute_program_hash`**: none.
- **Implementation notes**: the op arrives in the direct-descriptor shape, so [exception 3] (recipe `shared/ttnn_factory.md#3-give-a-direct-descriptor-op-a-conventional-program-factory`) applies. I'll add a nested `DeepseekMoEPostCombineTilizeProgramFactory` holding `create_program_artifacts`, and `using program_factory_t = std::variant<DeepseekMoEPostCombineTilizeProgramFactory>;`. With a single alternative the framework selects it automatically (`HasSelectProgramFactory` isn't required). The factory body stays in `device/deepseek_moe_post_combine_tilize_program_factory.cpp`, and nothing else in the device-op class changes. There's no pybind `create_descriptor`, so no binding is removed.

## Planned Spec Shape

- **KernelSpecs** (3, 1:1 with legacy):
  - `READER{"reader"}`: same source. CTAs `{bytes_to_read_per_row, input_row_page_size}`. RTA schema `{intra_row_byte_offset, row_page_offset}`. `dfb_bindings`: `tilize_input` PRODUCER. `tensor_bindings`: `input` → accessor `input`. `hw_config = ttnn::create_reader_datamovement_config()`, because the resolved triple RISCV_1 / NOC_0 / DM_DEDICATED_NOC equals the reader default exactly. `opt_level = O2`, carried verbatim.
  - `COMPUTE{"compute"}`: same source. CTAs `{num_tiles}`. `dfb_bindings`: `tilize_input` CONSUMER, `tilize_output` PRODUCER. `hw_config = ComputeHardwareConfig{}` (Style B). Legacy `ComputeConfigDescriptor{}` sets nothing, and every Metal 2.0 default matches: `HiFi4`, `Precise` (≡ `math_approx_mode=false`), `enable_32_bit_dest=false`, `double_buffer_dest=true` (≡ `!dst_full_sync_en`). No `unpack_modes` (no FP32, no `fp32_dest_acc_en`). `bfp_pack_precision_mode` is left at its default `Approximate` (≡ `bfp8_pack_precise=false`). `opt_level = O3`, set explicitly because legacy compute resolves to O3.
  - `WRITER{"writer"}`: same source. CTAs `{num_tiles}`. `dfb_bindings`: `tilize_output` CONSUMER. `hw_config = ttnn::create_writer_datamovement_config()`, because RISCV_0 / NOC_1 / DM_DEDICATED_NOC equals the writer default exactly. `opt_level = O2`, carried verbatim.
- **DataflowBufferSpecs** (2, 1:1 with the legacy CBs):
  - `TILIZE_INPUT{"tilize_input"}`: `entry_size = output_shard_width_bytes`, `num_entries = TILE_HEIGHT`, `data_format_metadata = data_format`, `tile_format_metadata` unset (the legacy `.tile` was unset). Reader PRODUCER, compute CONSUMER.
  - `TILIZE_OUTPUT{"tilize_output"}`: `entry_size = output_tile_page_size`, `num_entries = output_shard_width_tiles`, `data_format_metadata = data_format`, `borrowed_from = OUTPUT`. Compute PRODUCER, writer CONSUMER.
- **SemaphoreSpecs**: none.
- **TensorParameters**: `INPUT{"input"}` (`input.tensor_spec()`) and `OUTPUT{"output"}` (`output.tensor_spec()`). `OUTPUT` is borrow-only: no kernel binds it, which the validator allows for a `borrowed_from` target. Relaxation: none (strict).
- **WorkUnitSpecs**: one, `{READER, COMPUTE, WRITER}` on `op_cores`.
- **Op-owned tensors**: none.
- **ProgramRunArgs**: `tensor_args = {{INPUT, input}, {OUTPUT, output}}`. `kernel_run_args` = READER only, with per-node RTAs filled by `AddRuntimeArgsForNode` inside the unchanged legacy `cores` loop, which keeps the orientation-dependent ordering and both offset formulas.

## Preserved Multiplicity

none. There's no work-split multiplicity in legacy: one `KernelDescriptor` per source over a single core set.

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `program_factory.cpp:163` / `reader.cpp:23` | reader RTA[2] `input_tensor.buffer()` (`Buffer*` via `emplace_runtime_args`) → `input_tensor_address` | `TensorBinding{INPUT, "input"}` + `TensorArgument{INPUT, input}` |
| `program_factory.cpp:93-94` / `reader.cpp:25` | `TensorAccessorArgs(input_tensor.buffer()).append_to(reader_ct_args)` → `TensorAccessorArgs<0>()` | binding mechanism; kernel `TensorAccessor(tensor::input)` |
| `program_factory.cpp:104` / `reader.cpp:17` | named CTA `tilize_input_cb_id` = `c_0` | `DFBBinding{TILIZE_INPUT, "tilize_input", PRODUCER}`; kernel `dfb::tilize_input` |
| `program_factory.cpp:123-124` / `compute.cpp:13-14` | named CTAs `tilize_input_cb_id` = `c_0`, `tilize_output_cb_id` = `c_1` | `DFBBinding`s `tilize_input` CONSUMER / `tilize_output` PRODUCER; kernel `dfb::tilize_input` / `dfb::tilize_output` |
| `program_factory.cpp:137` / `writer.cpp:10` | named CTA `tilize_output_cb_id` = `c_1` | `DFBBinding{TILIZE_OUTPUT, "tilize_output", CONSUMER}`; kernel `dfb::tilize_output` |
| `program_factory.cpp:85` | `CBDescriptor::buffer = output_tensor.buffer()` | `DataflowBufferSpec::borrowed_from = OUTPUT` + `TensorArgument{OUTPUT, output}` |
| `reader.cpp:20-22` | positional RTAs [0] `intra_row_byte_offset`, [1] `row_page_offset` via `rt_args_idx++` | named RTAs `get_arg(args::intra_row_byte_offset)` / `get_arg(args::row_page_offset)`. Not varargs: two distinct fields, each read once. |
| `reader.cpp:18`, `compute.cpp:15`, `writer.cpp:11` | `get_named_compile_time_arg_val("bytes_to_read_per_row" / "num_tiles")` | `get_arg(args::bytes_to_read_per_row)` / `get_arg(args::num_tiles)` (named CTAs, same values) |

There are no positional CTAs beyond the TensorAccessorArgs block, no semaphore-ID RTAs, and no page-size third-argument sites.

**Kept, not dropped:** named CTA `input_row_page_size` (`program_factory.cpp:105`). No kernel reads it, and the recipe has no rule for unread named non-CB args. Dropping it would be behavior-neutral for numerics, but it changes the reader's compiled-kernel identity: today the value is baked into the binary hash, so readers for different input row widths compile separately. The syntax-swap invariant favors carrying it, so it becomes Metal 2.0 named CTA `input_row_page_size` with the same value. It's recorded in the report.

## Applied Patterns

- [Pass DFB handles directly to LLKs] (recipe `shared/port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers`): compute `compute_kernel_hw_startup` / `fast_tilize_init` / `fast_tilize_block` / `fast_tilize_uninit` take `dfb::tilize_input` / `dfb::tilize_output` directly. Those ids were `constexpr` in legacy, and the token's `uint32_t` conversion is constexpr, so they stay constant expressions.
- Borrowed-memory DFB (migration guide, *DataflowBufferSpec → Borrowed-memory DFBs*): `TILIZE_OUTPUT` `borrowed_from = OUTPUT`.
- Direct-descriptor → conventional program factory ([ttnn_factory exception 3] (recipe `shared/ttnn_factory.md#3-give-a-direct-descriptor-op-a-conventional-program-factory`)).

## Deferred / Flagged

- New findings during planning:
  - The brief says the DM NoC/RISC assignment is "swapped from the usual convention". It isn't: reader RISCV_1/NOC_0 and writer RISCV_0/NOC_1 are exactly `CreateReader/WriterDataMovementConfig()`, so the default helpers reproduce them byte-for-byte.
  - The recipe's hardware-config section names `ComputeGen1Config` / `DataMovementGen1Config` and `create_reader_datamovement_config(device->arch())`. The headers in this tree have `ComputeHardwareConfig` / `DataMovementHardwareConfig` with optional `config_1xx` / `config_2xx`, and an arch-less helper. I'm following the headers.
