# Port Plan — `experimental/bcast_to`

Port plan for `ttnn/cpp/ttnn/operations/experimental/bcast_to`, ported from the direct-`ProgramDescriptor` API to Metal 2.0 (`ProgramSpecFactoryConcept`).
Written during the inventory and planning steps; committed alongside the port for review.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept`, **direct-descriptor shape**. `BcastToOperation::create_descriptor` is a static member of the device operation itself (`device/bcast_to_device_operation.hpp:40-43`, defined at `device/bcast_to_program_factory.cpp:135-224`). There is no `program_factory_t` and no factory struct. The op satisfies `DeviceOperationConcept` only through the `HasDirectDescriptor` shim. → `ttnn_factory.md` exception 3 applies.
- Variants: single factory. Kernel *sources* are runtime-selected on `SubtileBroadcastType` (`NONE` / `ROW` / `COL` / `SCALAR`) through `BcastToKernelConfig` (`device/bcast_to_utils.cpp:278-304`). See *Runtime kernel-source selection* below.
- Custom `compute_program_hash`: none. The default reflection hash covers `operation_attributes_t{output_shape, memory_config, subtile_broadcast_type}` + `tensor_args_t{input, output}`. No `attribute_values` / `to_hash` either.
- `override_runtime_arguments`: none (removed in #57409).

*(The Metal 2.0 factory concept the port targets came from the audit; see the brief's TTNN factory analysis. Carried forward in [TTNN ProgramFactory](#ttnn-programfactory) below.)*

### Kernels

All three descriptors run on `all_device_cores` = `CoreRange({0,0}, {grid.x-1, grid.y-1})` (the full compute-with-storage grid). Sources are picked per config (see below).

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `reader_interleaved_{no,row,col,scalar}_bcast_to.cpp` | `all_device_cores` | `[0] = c_0`, `[1..] = TensorAccessorArgs(input.buffer())` (`:172-173`) | none | 13: `src_addr` (`Buffer*` input, `:84`), `start_n, start_c, start_t, start_th, start_tw, num_tiles, n_stride, c_stride, N, C, Ht, Wt` | none | none | O2 (unset, DM default) | `ReaderConfigDescriptor{}` → RISCV_1 / NOC_0 / dedicated |
| writer | `writer_interleaved_{no,row,col,scalar}_bcast_to.cpp` | `all_device_cores` | `[0] = writer_cb_id` (`c_0` under NONE, else `c_1`; `:183-184`), `[1..] = TensorAccessorArgs(output.buffer())` (`:186`) | none | 14: `dst_addr` (`Buffer*` output, `:100`), the same 12 as the reader, then `start_tile_id` | none | none | O2 (unset, DM default) | `WriterConfigDescriptor{}` → RISCV_0 / NOC_1 / dedicated |
| compute | `compute_interleaved_{no,row,col,scalar}_bcast_to.cpp` | `all_device_cores` | `[0] = c_0`, `[1] = c_1` (`:209`) | none | 12: the same 12 as the reader (no address) | none | none | **O3** (unset, compute default) | `ComputeConfigDescriptor{.fp32_dest_acc_en = is_32bit_format, .unpack_to_dest_mode = {c_0: UnpackToDestFp32 if is_32bit_format}, .math_approx_mode = false}` (`:196-214`). Every other field is at its default: `math_fidelity = HiFi4`, `dst_full_sync_en = false`, `bfp8_pack_precise = false`, `enable_trisc2_rvv = false`. |

`is_32bit_format` = input format is `Float32` / `Int32` / `UInt32`.

Idle cores (outside `core_group_1` ∪ `core_group_2`) get all-zero RTA vectors of length 13 / 14 / 12 (`:65-70`). Every kernel exits on `num_tiles == 0`.

### Runtime kernel-source selection

| config | reader | writer | compute | `c_0` (reader→) consumer | `c_1` |
|---|---|---|---|---|---|
| `NONE` | `ReaderNoBcast` | `WriterNoBcast` | `ComputeNoBcast` (empty `kernel_main`) | writer | dead (0 touchers) |
| `ROW` | `ReaderRowBcast` | `WriterRowBcast` | `ComputeRowBcast` | compute | compute PRODUCER → writer CONSUMER |
| `COL` | `ReaderColBcast` | `WriterColBcast` | `ComputeColBcast` | compute | compute PRODUCER → writer CONSUMER |
| `SCALAR` | `ReaderScalarBcast` | `WriterScalarBcast` | `ComputeScalarBcast` | compute | compute PRODUCER → writer CONSUMER |

All 12 sources flip together (one atomic unit). `compute_interleaved_no_bcast_to.cpp` reads no CTA and no RTA, so it needs no edit.

### CBs
| index | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|
| `c_0` | `2 * tile_size(input_fmt)` | `all_device_cores` | input format | `tile_size(input_fmt)` | not set |
| `c_1` | `2 * tile_size(input_fmt)` | `all_device_cores` | input format | `tile_size(input_fmt)` | not set |

Both are allocated unconditionally in one loop (`:156-167`). No `GlobalCircularBuffer`, no `address_offset`, no borrowed `buffer`, no aliasing.

**Kernel-touch census** (re-derived from the kernels; it agrees with the brief):
- `c_0`: reader `reserve_back`/`push_back` (PRODUCER) in every config. Consumer is the writer `wait_front`/`pop_front` under NONE, and compute `ckl::input(..., WaitPolicy::PerTile, PopPolicy::PerTile)` under ROW/COL/SCALAR. Two touchers per config → 1P+1C.
- `c_1`: under ROW/COL/SCALAR, compute `ckl::output(..., ReservePolicy::PerTile, PushPolicy::PerTile)` (PRODUCER) → writer `wait_front`/`pop_front` (CONSUMER) → 1P+1C. Under NONE, zero touchers: the writer binds `c_0`, and the compute kernel is empty.

### Semaphores
none

### Tensor accessors
| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| `bcast_to_program_factory.cpp:84` (`emplace_runtime_args`, `input.buffer()`); CTA `:173` | `tensor_args.input` | reader RTA 0 |
| `bcast_to_program_factory.cpp:100` (`emplace_runtime_args`, `output.buffer()`); CTA `:186` | `tensor_return_value` (the output; may be the caller's preallocated `tensor_args.output`) | writer RTA 0 |

Every accessor uses the 2-arg form (no 3rd page-size argument). No `ArgConfig::Runtime*`.

### Work split
- Driver: `split_work_to_cores(compute_with_storage_grid_size, num_output_tiles, /*row_major=*/true)` (`:52-53`). Cores are walked with `grid_to_cores(num_cores_total, x, y, row_major)`.
- num_cores: `min(num_output_tiles, grid.x*grid.y)`
- core_group_1: `num_tiles_per_core_group_1` tiles per core
- core_group_2: `num_tiles_per_core_group_2` tiles per core
- Per-core counts reach the kernels as **RTAs** (`num_tiles`), not CTAs. There is one `KernelDescriptor` per role over the full grid, so there is no per-group CTA multiplicity to preserve.

### Shared kernels
none. All 12 kernel sources live in `device/kernels/` and are bound only by `bcast_to_utils.cpp:306-329` (`get_kernel_file_path`). The filename census (`grep -rl <file> ttnn/cpp/ttnn/operations/`) finds no other binder. No `_metal2` forks exist and none are needed. The kernels convert in place.

### Flags
- `compute_interleaved_no_bcast_to.cpp` has an empty `kernel_main`, but it is still dispatched with 12 RTAs and a compute config. This is preserved as-is.
- No unreferenced kernel files.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept`.
- **Custom `compute_program_hash`**: none (default reflection hash).
- **Implementation notes**: direct-descriptor op → `ttnn_factory.md` exception 3. Add a nested `BcastToOperation::BcastToProgramFactory { static ProgramArtifacts create_program_artifacts(...); }` and `using program_factory_t = std::variant<BcastToProgramFactory>;`. Remove the device-op's own `create_descriptor`. The factory body stays in `device/bcast_to_program_factory.cpp`. The existing comment above `create_descriptor` (every non-address RTA comes from hashed state) moves with the method. No pybind references `create_descriptor` (`bcast_to_nanobind.cpp` binds only the function).

## Planned Spec Shape

- **KernelSpecs** (one per legacy `KernelDescriptor`): `reader`, `writer`, `compute`. The source is chosen per config exactly as today (`get_kernel_file_path(kernel_config.*)`).
- **DataflowBufferSpecs**:
  - `src` (was `c_0`): `entry_size = tile_size(input_fmt)`, `num_entries = 2`, `data_format_metadata = input_fmt`. `tile_format_metadata` is left unset (the legacy `.tile` was unset). Always declared.
  - `dst` (was `c_1`): same shape. **Declared only when `subtile_broadcast_type != NONE`.** Under NONE it has zero touchers, and the validator rejects a bindingless DFB. This conditional is new host structure (the brief flags it).
- **SemaphoreSpecs**: none.
- **TensorParameters**: `input` (`input.tensor_spec()`) and `output` (`output.tensor_spec()`), strict (no relaxations).
- **WorkUnitSpecs**: one, `bcast_to`, with `{reader, writer, compute}` over `all_device_cores`.
- **Op-owned tensors**: none.

Bindings per config:

| KernelSpec | NONE | ROW / COL / SCALAR |
|---|---|---|
| reader | `src` PRODUCER as `"src"`; tensor `input` as `"input"` | same |
| writer | `src` CONSUMER as `"dst"`; tensor `output` as `"output"` | `dst` CONSUMER as `"dst"`; tensor `output` as `"output"` |
| compute | **no DFB bindings** | `src` CONSUMER as `"src"`, `dst` PRODUCER as `"dst"` |

The writer's accessor name is `"dst"` in every config. Only the DFB it binds changes, which mirrors the legacy `writer_cb_id` selection.

Hardware config:
- reader: `ttnn::create_reader_datamovement_config()` (RISCV_1 / NOC_0 / dedicated = legacy `ReaderConfigDescriptor`)
- writer: `ttnn::create_writer_datamovement_config()` (RISCV_0 / NOC_1 / dedicated = legacy `WriterConfigDescriptor`)
- compute: **Style B**, a `ComputeHardwareConfig` built directly: `.enable_32_bit_dest = is_32bit_format`. Every other field stays at its default, which equals the legacy default: `HiFi4`, `Precise` (= `math_approx_mode=false`), `double_buffer_dest = true` (= `dst_full_sync_en=false`), `bfp_pack_precision_mode = Approximate` (= `bfp8_pack_precise=false`). `unpack_modes = {{src, UnpackToDest}}` **only when `is_32bit_format` and the compute kernel binds `src`** (ROW/COL/SCALAR). Under NONE the compute kernel binds no DFB, so a `src` key would be rejected. The legacy `unpack_to_dest_mode[c_0]` there was inert, because the empty kernel unpacks nothing.
- compute `compiler_options.opt_level = O3` (the legacy compute default). DM kernels stay at O2.

RTA schemas (named, legacy order preserved):
- reader: `start_n, start_c, start_t, start_th, start_tw, num_tiles, n_stride, c_stride, N, C, Ht, Wt`
- writer: the same 12 plus `start_tile_id`
- compute: the same 12 as the reader
- Idle cores get 0 for every named arg.

## Preserved Multiplicity

none. There is no work-split multiplicity in the legacy op: one `KernelDescriptor` per role over the full grid, with per-core counts carried as RTAs.

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| factory `:84` / reader RTA slot 0 | `input.buffer()` (`Buffer*` via `emplace_runtime_args`) | `TensorBinding{input → "input"}` on reader |
| factory `:173` / reader CTA `[1..]` | `TensorAccessorArgs(input.buffer()).append_to(...)` | binding token (`TensorAccessor(tensor::input)`) |
| readers `:13,28,31` | `src_addr = get_arg_val(0)`, `TensorAccessorArgs<1>()`, `TensorAccessor(src_args, src_addr)` | `TensorAccessor(tensor::input)` |
| factory `:100` / writer RTA slot 0 | `output.buffer()` | `TensorBinding{output → "output"}` on writer |
| factory `:186` / writer CTA `[1..]` | `TensorAccessorArgs(output.buffer()).append_to(...)` | binding token |
| writers `:13,31-33` | `dst_addr`, `TensorAccessorArgs<1>()`, `TensorAccessor(dst_args, dst_addr)` | `TensorAccessor(tensor::output)` |
| factory `:172` / reader CTA 0 | `c_0` | `DFBBinding{src, "src", PRODUCER}` |
| factory `:183-185` / writer CTA 0 | `writer_cb_id` (`c_0` / `c_1`) | `DFBBinding{src or dst, "dst", CONSUMER}` |
| factory `:209` / compute CTAs 0, 1 | `c_0`, `c_1` | ROW/COL/SCALAR: `DFBBinding{src, "src", CONSUMER}`, `DFBBinding{dst, "dst", PRODUCER}`. NONE: dropped (dead; the empty kernel never reads them) |
| factory `:199-203` | `unpack_to_dest_mode[c_0] = UnpackToDestFp32` (vector by CB id) | `unpack_modes = {{src, UnpackToDest}}` (non-NONE only) |
| factory `:157-167` | `CBDescriptor` loop over `c_0`, `c_1` | `DataflowBufferSpec` `src` (always) + `dst` (non-NONE) |
| all kernels | positional `get_arg_val<uint32_t>(arg_index++)` | `get_arg(args::<name>)`, names as in the RTA schema above |
| compute kernels `:30-31` | `dfb_id_src_id = get_compile_time_arg_val(0)` / `dfb_id_dst_id = get_compile_time_arg_val(1)` | `dfb::src` / `dfb::dst` passed directly to `compute_kernel_hw_startup`, `unary_bcast_init`, `ckl::input`, `ckl::output` |

No page-size 3rd-argument sites. No semaphores. No positional CTAs survive (no scalar CTAs exist).

## Applied Patterns

- [Conditional / optional resource bindings](port_patterns.md#pattern-conditional--optional-resource-bindings), host side only. The `dst` DFB is declared and bound only for ROW/COL/SCALAR. No kernel-side `#ifdef` is needed, because the condition picks distinct kernel *sources*: the NONE sources never name `dfb::dst` (the NONE writer names its own binding `"dst"`, which there resolves to `src`), and the NONE compute kernel names nothing.
- [Pass DFB handles directly to LLKs and kernel-lib helpers](port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers): `dfb::src` / `dfb::dst` go into `compute_kernel_hw_startup`, `unary_bcast_init<…>`, and `ckl::input(...)` / `ckl::output(...)` in NTTP position.
- `ttnn_factory.md` exception 3: direct-descriptor op gets a conventional `BcastToProgramFactory`.
- `AddRuntimeArgsForNode` keeps the legacy node-first per-core loop intact.

## Deferred / Flagged

- **Brief correction, NONE coverage.** The brief says the `NONE` config has no active test coverage. The C++ gtest `tests/ttnn/unit_tests/gtests/test_broadcast_to.cpp` does exercise it (`ChannelAndBatch`: `{1,1,64,64} → {1,3,64,64}` / `{3,1,64,64}`; `LargeTensor`: `→ {1,32,64,64}` / `{32,1,64,64}`; `CombinedDimensions`: `{1,1,32,32} → {7,17,32,32}`; `NonAlignedDimensions`: `{1,1,7,13} → {1,1,7,13}` etc.). That gtest is in the confirmed baseline, so no ad-hoc run is needed.
- Nothing else new. The census matches the brief.
