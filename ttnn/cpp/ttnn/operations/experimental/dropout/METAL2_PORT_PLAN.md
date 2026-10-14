# Port Plan — `experimental/dropout`

Port plan for `experimental/dropout`, ported from the `ProgramDescriptor` API to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.
Line numbers cite the pre-port revision (`origin/main` @ `3c202f9944d`).

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept`, on both factories of the `program_factory_t` variant (`device/dropout_device_operation.hpp:25`). Neither is a direct-descriptor shape.
- Variants:
  - `DropoutProgramFactory`: selected when `use_per_device_seed == false` (`device/dropout_device_operation.cpp:14-20`). `create_descriptor(args, tensor_args, output)` (`device/dropout_program_factory.cpp:241-340`). Void-returning PD `override_runtime_arguments` (`:353-405`).
  - `DropoutMeshWorkloadFactory`: selected when `use_per_device_seed == true`, the default at both entry points. Per-coordinate `create_descriptor(…, mesh_dispatch_coordinate)` (`:342-351`) that delegates to `DropoutProgramFactory::create_descriptor` with `seed = per_device_seed(…)` (`:33-40`). Its override delegates to `DropoutProgramFactory`'s (`:407-415`). The descriptor adapter builds one Program per coordinate for it (`ttnn/api/ttnn/mesh_device_operation_adapter.hpp:608-612`).
  - Both variants build the identical program shape; only the compute `seed` runtime-arg value differs.
- Custom `compute_program_hash`: backdoor `DropoutParams::attribute_values()` at `device/dropout_device_operation_types.hpp:28-32` (excludes `seed`). Left intact.

### Kernels
| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_dropout_interleaved_start_id.cpp` | `all_cores` | `[0]` = `kSrc0CbIndex` (`c_0`); `[1..]` = `TensorAccessorArgs(src_buffer)` (`:278-279`) | none | `[0]` `src_buffer` (`Buffer*`), `[1]` `num_tiles`, `[2]` `tile_offset` (`:226`) | none | none | O2 (unset; DM default) | `ReaderConfigDescriptor{}` |
| writer | `device/kernels/dataflow/writer_dropout_interleaved_start_id.cpp` | `all_cores` | `[0]` = `kOutputCbIndex` (`c_2`); `[1..]` = `TensorAccessorArgs(dst_buffer)` (`:282-283`) | none | `[0]` `dst_buffer` (`Buffer*`), `[1]` `num_tiles`, `[2]` `tile_offset` (`:229`) | none | none | O2 (unset; DM default) | `WriterConfigDescriptor{}` |
| compute_group_1 | `device/kernels/compute/dropout_kernel.cpp` | `core_group_1` | `[num_tiles_per_core_group_1, 1, prob_int, uscale]` (`:299-304`) | none | `[0]` `seed` (`:217`) | none | none | O3 (unset; compute default) | `ComputeConfigDescriptor{HiFi4, fp32_dest_acc_en=false, dst_full_sync_en=false, math_approx_mode=false}` (`:131-136`) |
| compute_group_2 | same | `core_group_2`, only when non-empty (`:312`) | `[num_tiles_per_core_group_2, 1, prob_int, uscale]` (`:313-318`) | none | `[0]` `seed` (`:220`) | none | none | O3 (unset) | same |

`grep -n opt_level device/dropout_program_factory.cpp` prints nothing: no descriptor sets one, so DM resolves to O2 and compute to O3.

Kernel-side reads: reader `get_arg_val` 0/1/2 → `src_addr`, `num_tiles`, `start_id`; CTA 0 → `cb_id_in0`; `TensorAccessorArgs<1>()`. Writer: same with `dst_addr` / `cb_id_out`. Compute: CTAs 0-3 → `per_core_block_cnt`, `per_core_block_dim`, `int_probability`, `int_scale_factor`; RTA 0 → `seed`; CB ids hardcoded `tt::CBIndex::c_0` / `c_2` (`dropout_kernel.cpp:20-21`).

### CBs
| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| 0 (`kSrc0CbIndex`) | `2 * tile_size(fmt(input.dtype))` | `all_cores` | `fmt(input.dtype)` | `tile_size(fmt(input.dtype))` | not set |
| 2 (`kOutputCbIndex`) | `2 * tile_size(fmt(output.dtype))` | `all_cores` | `fmt(output.dtype)` | `tile_size(fmt(output.dtype))` | not set |

Both built by `create_circular_buffer` (`:60-78`): one `format_descriptor` each (no aliasing), no `.buffer` (program-local), no `address_offset`, no `global_circular_buffer`.

### Semaphores
none

### Tensor accessors
| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `dropout_program_factory.cpp:278-279` (CTAs), `:226` (RTA) | `tensor_args.input` | reader RTA 0 (`Buffer*`) |
| `dropout_program_factory.cpp:282-283` (CTAs), `:229` (RTA) | `output` (`tensor_return_value`) | writer RTA 0 (`Buffer*`) |

Cache hit: the override writes `input.buffer()->address()` / `tensor_return_value.buffer()->address()` into those slots (`:375-376,:392,:397`).

### Work split
- Driver: `split_work_to_cores(compute_with_storage_grid_size, physical_volume / TILE_HW)` (`dropout_core_split`, `:151-164`); default `row_major = false`.
- Core order: `core = {i / num_cores_y, i % num_cores_y}` for `i < num_cores` (`for_each_dropout_core`, `:177-193`), shared by the cache-miss build and the override.
- `core_group_1` gets `num_tiles_per_core_group_1` tiles per core, `core_group_2` gets `num_tiles_per_core_group_2`; `tile_offset` is the running sum.

### Shared kernels
none. `grep -rl` for each of the three kernel filenames under `ttnn/cpp/ttnn/operations/` returns only `dropout_program_factory.cpp`. Both of this op's factories bind the same three sources, and both convert in this change, so the kernels convert in place with no `_metal2` fork (the case paged_cache settled in its report, Friction #4).

### Flags
- `#ifdef BACKWARDS` (reader `:29`, writer `:31`) and `#ifdef OUT_SHARDED` (writer `:23`) are never compiled: the factory sets no defines. Converted mechanically and kept.
- No unreferenced kernel files in the op directory.

## TTNN ProgramFactory

- **Concept (inherited from the brief):**
  - `DropoutProgramFactory` → `CustomProgramSpecFactoryConcept`.
  - `DropoutMeshWorkloadFactory` → `MeshWorkloadSpecFactoryConcept`. Outside `metal2_port.md` §What this procedure covers; proceeding on the invoker's explicit authorisation (2026-10-14), following the `experimental/paged_cache` port (#56022, `PagedFillCacheMeshWorkloadFactory`).
- **Custom `compute_program_hash`**: backdoor `attribute_values()` at `device/dropout_device_operation_types.hpp:28-32`. Leave intact.
- **Implementation notes:**
  - One builder, `build_dropout_artifacts(args, tensor_args, output, seed)`, replaces today's `DropoutProgramFactory::create_descriptor` body. `DropoutProgramFactory::create_program_artifacts` calls it with `args.seed`.
  - `DropoutMeshWorkloadFactory::create_mesh_workload_artifacts` keeps the existing `TT_ASSERT(args.use_per_device_seed, …)`. It loops `tensor_coords.coords()` and emits one `PerCoordProgram{.range = MeshCoordinateRange(coord), …}` per coordinate, built with `per_device_seed(args, coord, input)`. This reproduces the ported-from program set exactly: one program per coordinate, each with its own seed.
  - `DropoutProgramFactory::override_runtime_arguments(attrs, tensor_args, output, std::optional<MeshCoordinate>)` returns a `ProgramRunArgs` mirroring the legacy override.
  - `DropoutMeshWorkloadFactory::override_runtime_arguments(attrs, tensor_args, output, const MeshCoordinateRange& range)` delegates to it with `range.start_coord()`. Every range is one coordinate, so the start coordinate is the whole range. This keeps today's delegation (`:413-414`).

## Planned Spec Shape

The spec is identical for both factories; only `seed` differs.

- KernelSpecs: `reader`, `writer`, `compute_g1`, and `compute_g2` (only when `core_group_2` is non-empty). Same sources as legacy.
  - reader / writer: `hw_config = create_reader_datamovement_config()` / `create_writer_datamovement_config()`. Their Gen1 triples equal the legacy `ReaderDataMovementConfig` / `WriterDataMovementConfig` (RISCV_1/NOC_0 and RISCV_0/NOC_1, dedicated NOC; `tt_metal/impl/kernels/kernel_types.cpp:20-38`). `opt_level` left at O2.
  - compute: `ComputeHardwareConfig{.fpu_math_fidelity = HiFi4, .sfpu_precision_mode = Precise, .enable_32_bit_dest = false, .double_buffer_dest = true}`, built directly (Style B) from the four fields the legacy descriptor set. `double_buffer_dest = !dst_full_sync_en`; `sfpu_precision_mode = math_approx_mode ? Approximate : Precise`. No `unpack_modes`: there is no FP32 Dest, and the legacy `unpack_to_dest_mode` is unset. `bfp8_pack_precise` is unset in legacy, so `config_1xx` stays unset (its default `Approximate` matches). `compiler_options.opt_level = O3` on both.
- DataflowBufferSpecs: `in0` (`entry_size = tile_size(fmt_in)`, `num_entries = 2`, `data_format_metadata = fmt_in`) and `out` (`tile_size(fmt_out)`, `2`, `fmt_out`). No borrowing, aliasing or advanced options.
- SemaphoreSpecs: none.
- TensorParameters: `input` (`tensor_args.input.tensor_spec()`), `output` (`output.tensor_spec()`). Strict; no relaxations.
- WorkUnitSpecs: `wu_g1 = {reader, writer, compute_g1}` on `core_group_1`; `wu_g2 = {reader, writer, compute_g2}` on `core_group_2` when non-empty. The reader and writer cover `core_group_1 ∪ core_group_2 = all_cores`, as in legacy.

## Preserved Multiplicity

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| `compute_group_1`, `compute_group_2` of `dropout_kernel.cpp` (`:308-322`) | `compute_g1` (CTA `per_core_block_cnt = num_tiles_per_core_group_1`), `compute_g2` (`= num_tiles_per_core_group_2`) | `wu_g1` (`core_group_1`), `wu_g2` (`core_group_2`) | `in0`: CONSUMER on both · `out`: PRODUCER on both. Disjoint node sets, so each node has one compute instance; no multi-binding flag |

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| factory `:226`, reader `:15` | reader RTA 0 `src_buffer` (`Buffer*`) → `src_addr` | `TensorBinding{input, "input"}`; kernel `TensorAccessor(tensor::input)` |
| factory `:229`, writer `:14` | writer RTA 0 `dst_buffer` (`Buffer*`) → `dst_addr` | `TensorBinding{output, "output"}`; kernel `TensorAccessor(tensor::output)` |
| factory `:278-279`, reader `:20` | `TensorAccessorArgs(src_buffer).append_to(cta)` / `TensorAccessorArgs<1>()` | binding mechanism |
| factory `:282-283`, writer `:19` | `TensorAccessorArgs(dst_buffer).append_to(cta)` / `TensorAccessorArgs<1>()` | binding mechanism |
| factory `:278`, reader `:19` | reader CTA 0 `cb_id_in0 = c_0` | `DFBBinding{in0, "in0", PRODUCER}` |
| factory `:282`, writer `:18` | writer CTA 0 `cb_id_out = c_2` | `DFBBinding{out, "out", CONSUMER}` |
| compute `:20-21` | hardcoded `tt::CBIndex::c_0` / `c_2` | `DFBBinding{in0, "in0", CONSUMER}`, `DFBBinding{out, "out", PRODUCER}`; kernel passes `dfb::in0` / `dfb::out` |
| factory `:299-304,:313-318`, compute `:13-16` | positional CTAs `[cnt, 1, prob, scale]` | named CTAs `per_core_block_cnt`, `per_core_block_dim`, `int_probability`, `int_scale_factor` |
| reader `:16-17`, writer `:15-16` | positional RTAs 1, 2 | named RTAs `num_tiles`, `start_id` |
| compute `:18` | positional RTA 0 | named RTA `seed` |
| factory `:233-239,:384-388` | `kReaderIdx`… enum + `GetRuntimeArgs(program, idx)` grids in the override | name-keyed `KernelRunArgs` |
| factory `:375-376,:392,:397` | override writes raw `buffer()->address()` into RTA 0 | `ProgramRunArgs::tensor_args` entries for `input` and `output` |
| factory `:60-78`, `:24-25` | `create_circular_buffer` / `CBDescriptor`, `kSrc0CbIndex`, `kOutputCbIndex` | `DataflowBufferSpec` `in0`, `out` |
| reader `:38`, writer `:40` | `get_tile_size(cb_id)` as an inline call argument | `dfb_in0.get_tile_size()` / `dfb_out.get_tile_size()` (whitelist rule 7; not a `constexpr` declaration) |

## Applied Patterns

- Demoting per-group CTA to RTA (port_patterns.md), applied in its correct form: two compute `KernelSpec`s, two `WorkUnitSpec`s.
- Pass DFB handles directly to LLKs and kernel-lib helpers (port_patterns.md): `compute_kernel_hw_startup(dfb::in0, dfb::out)`, `ckl::input(dfb::in0, …)`, `ckl::output(dfb::out, …)` in template-argument position.
- Precedent (invoker-supplied reference port): `PagedFillCacheMeshWorkloadFactory`, for the `MeshWorkloadSpecFactoryConcept` shape.

## Deferred / Flagged

- The mesh-workload target is outside the recipe's coverage (see TTNN ProgramFactory above). Proceeding on invoker authorisation; the four decisions paged_cache had to make are settled above: one program per coordinate, no omitted coordinates, ranges bounded by `tensor_coords.coords()`, and `range.start_coord()` in the override.
- In-place aliasing (`output_tensor = input`, `test_dropout.py:49`) binds both `TensorParameter`s to one tensor. Watch for a Metal 2.0 validation reaction on first run.
- API drift between the recipe and `main`, recorded in the report: the recipe names `ComputeGen1Config` / `DataMovementGen1Config` and `create_reader_datamovement_config(device->arch())`. `main` has `ComputeHardwareConfig` / `DataMovementHardwareConfig` with generation-specific `config_1xx` / `config_2xx` members, and the DM helpers take no arch argument.
