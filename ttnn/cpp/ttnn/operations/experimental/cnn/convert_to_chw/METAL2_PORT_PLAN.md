# Port Plan — convert_to_chw

Port plan for `ttnn/cpp/ttnn/operations/experimental/cnn/convert_to_chw`, ported from the `ProgramDescriptor` API
(`ProgramDescriptorFactoryConcept` + a descriptor-style `override_runtime_arguments`) to Metal 2.0
(`CustomProgramSpecFactoryConcept`).
Written during the inventory and planning steps; committed alongside the port for review.

All `file:line` references below are to the pre-port tree (`1308c6a3fc8`). Shorthand: `.cpp` / `.hpp` =
`device/convert_to_chw_program_factory.{cpp,hpp}`; `reader` / `writer` / `compute` = the three sources under
`device/kernels/` (`reader_convert_to_chw.cpp`, `writer_convert_to_chw.cpp`, `convert_to_chw.cpp`).

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` — `ConvertToCHWProgramFactory::create_descriptor` returns a
  `ProgramDescriptor` (`.hpp:18-19`, `.cpp:42-148`), plus a descriptor-style
  `void override_runtime_arguments(Program&, …, const std::optional<MeshCoordinate>&)` (`.hpp:21-26`,
  `.cpp:150-167`).
- Where the factory methods live: in a conventional factory struct, `ConvertToCHWProgramFactory`, already the
  single alternative of `program_factory_t = std::variant<ConvertToCHWProgramFactory>`
  (`device/convert_to_chw_device_operation.hpp:22`). Not the direct-descriptor shape, so exception 3 does not
  apply; the port is a method swap inside the existing struct.
- Variants: single (one config: no defines, no branches, no runtime kernel-source selection).
- Custom `compute_program_hash`: none — default reflection-based hash over `ConvertToCHWParams{memory_config,
  dtype}` and the input tensor. No backdoor (`attribute_values` / `to_hash`) on `ConvertToCHWParams`
  (`device/convert_to_chw_device_operation_types.hpp:11-14`).

*(The Metal 2.0 factory concept was chosen during the audit — see the brief's TTNN factory analysis. Carried
forward in [TTNN ProgramFactory](#ttnn-programfactory) below.)*

### Kernels
All three over `core_ranges = input_core_grid` = the input shard grid (`a.shard_spec()->grid`, `.cpp:50`).

| unique_id | source | CTAs (positional, host emission order) | CTAs (named) | RTAs (per core) | CRTAs | defines | opt_level (resolved) | config |
|---|---|---|---|---|---|---|---|---|
| reader | `reader` (`.cpp:113-114`) | `[0] cb_in_id = c_0` (`.cpp:108`) | none | `[0] total_tiles_per_core` on every core of `input_cores` (`.cpp:27-39, 142`) | none | none | `O2` (unset; DM default) | `ReaderConfigDescriptor{}` → (`RISCV_1`, `NOC_0`, `DM_DEDICATED_NOC`) |
| writer | `writer` (`.cpp:121-122`) | `[0] cb_in_transpose_id = c_2`, `[1] cb_out_id = c_1`, `[2] C` (`.cpp:109`) | none | `[0] total_tiles_per_core` (same loop) | none | none | `O2` (unset; DM default) | `WriterConfigDescriptor{}` → (`RISCV_0`, `NOC_1`, `DM_DEDICATED_NOC`) |
| compute | `compute` (`.cpp:129-130`) | `[0] cb_in_id = c_0`, `[1] cb_in_transpose_id = c_2` (`.cpp:110`) | none | `[0] total_tiles_per_core` (same loop) | none | none | `O3` (unset; `ComputeConfigDescriptor` resolves `O3`, `tt_metal/impl/program/program.cpp:565`) | `ComputeConfigDescriptor{.math_fidelity = HiFi4, .fp32_dest_acc_en = false, .dst_full_sync_en = false, .math_approx_mode = false}` (`.cpp:134-139`); unset: `unpack_to_dest_mode = {}`, `bfp8_pack_precise = false`, `enable_trisc2_rvv = false` |

`grep -n opt_level` on `.cpp`: no hits — every level above is the resolved default
(`tt_metal/impl/program/program.cpp:528, 536, 565`). Reader/writer resolution checked in
`tt_metal/impl/kernels/kernel_types.cpp:13-43` (`preferred_noc_for_dram_read/write` → `NOC_0` / `NOC_1` on WH and BH).

Kernel-side reads: `reader:9` RTA0, `reader:10` CTA0; `writer:13` RTA0, `writer:18, 20, 21` CTA0-2;
`compute:33` RTA0, `compute:36-37` CTA0-1.

### CBs
All three over `input_core_grid`, built by the family helper `make_cnn_circular_buffer(…, log_cb = true)`
(`ttnn/cpp/ttnn/operations/experimental/cnn/cnn_cb_descriptor.hpp:15-41`; one format descriptor each, no `tile`,
no `address_offset`, no `global_circular_buffer`).

| index | total_size | page_size | data_format | tile | backing | site |
|---|---|---|---|---|---|---|
| `c_0` `cb_in` | `total_tiles_per_core * input_tile_size` | `input_tile_size` | input dtype (`Float16_b` / `Bfp8_b` in tests) | unset | **borrowed: input buffer** | `.cpp:82-86` |
| `c_1` `cb_out` | `output_shard_shape[0] * output_shard_shape[1] * element_size` | `output_shard_shape[1] * element_size` | output dtype | unset | **borrowed: output buffer** | `.cpp:88-94` |
| `c_2` `cb_in_transpose` | `16 * intermediary_tile_size` | `intermediary_tile_size` | `Float16_b` | unset | own L1 | `.cpp:96-106` |

Every `total_size` is an exact multiple of its `page_size`, so each maps to `entry_size = page_size`,
`num_entries = total_size / page_size` with no remainder. No GlobalCircularBuffer anywhere (no
`.global_circular_buffer`, no `global_cb` parameter, no `remote_cb_config`).

### Semaphores
none

### Tensor accessors
none — no kernel constructs a `TensorAccessor` and no buffer address reaches any kernel argument. Tensor memory
reaches the kernels only through the two borrowed CBs:

| host site (file:line) | originating Tensor | how it reaches the kernel |
|---|---|---|
| `.cpp:71, 85-86` | input (`tensor_args`) | `CBDescriptor::buffer` of `c_0` (not an RTA) |
| `.cpp:72, 93-94` | output (`tensor_return_value`) | `CBDescriptor::buffer` of `c_1` (not an RTA) |

### Work split
- n/a — no `split_work_to_cores`. Every core of the input shard grid runs the same three kernels with the same
  `total_tiles_per_core = div_up(HW / 32, input_cores.size())` (`.cpp:66-67`); `input_cores =
  corerange_to_cores(input_core_grid, nullopt, row_major)` (`.cpp:51-52`).

### Override (`override_runtime_arguments`, `.cpp:150-167`) — what it refreshes on a cache hit
| write | bucket | Metal 2.0 home |
|---|---|---|
| `c_0` address ← `tensor_args.buffer()` (CB-only descriptor, matched by position) | tensor address | `TensorArgument` for `INPUT` |
| `c_1` address ← `output.buffer()` | tensor address | `TensorArgument` for `OUTPUT` |
| (no runtime args) | — | no `kernel_run_args` |

Guards inside it: `TT_FATAL(input_buffer != nullptr, …)`, `TT_FATAL(output_buffer != nullptr, …)` (`.cpp:159-160`) —
carried across. It ignores `mesh_dispatch_coordinate`.

### Shared kernels
none. `grep -rl <filename> ttnn/cpp/ttnn/operations/` for each of the three kernel filenames hits only this
factory (`.cpp:113-130`), the two audit `.md` files, and — for `convert_to_chw.cpp` — the same-named *host* source
in `cnn/sources.cmake:10` (a build-list entry, discarded). No `_metal2` sibling exists beside any of them. Convert
all three in place.

Out-of-directory code the kernels reach (not kernels, no fork question): the pool donor header
`ttnn/cpp/ttnn/operations/pool/device/kernels/experimental_device_api.hpp` (reader + writer; untouched), and the
family host helper `cnn_cb_descriptor.hpp` (the factory stops calling it; untouched — `convert_to_hwc` still uses
it).

### Flags
- No unreferenced kernel files in `device/kernels/`.
- `reader:12` pushes `total_tiles` with no `reserve_back` (data already resident in the borrowed input shard);
  `writer:29` peeks `get_read_ptr()` before its first `wait_front` to program the NoC read state. Both intended;
  kept as-is.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `CustomProgramSpecFactoryConcept`.
- **Custom `compute_program_hash`**: none — default reflection hash; nothing to leave intact.
- **Implementation notes**:
  - Method swap inside the existing `ConvertToCHWProgramFactory`: `create_descriptor` →
    `create_program_artifacts` (returns `ttnn::device_operation::ProgramArtifacts`), and
    `override_runtime_arguments` drops its `Program&` parameter and returns
    `tt::tt_metal::experimental::ProgramRunArgs` (the return type is what selects the custom concept).
  - The translated override returns `tensor_args = {INPUT, OUTPUT}` and no `kernel_run_args` — exactly the
    ported-from set. It keeps both `TT_FATAL`s and ignores `coord`, as the ported-from override ignored
    `mesh_dispatch_coordinate`.
  - No device-op-class edits: no pybound `create_descriptor` (`convert_to_chw_nanobind.cpp:25-26` binds only the
    free function), no pybind-only factory parameter, not direct-descriptor.

## Planned Spec Shape

Names live in an anonymous-namespace nested namespace (`chw`) so they cannot collide with the `convert_to_hwc`
factory's in the unity build (`cnn/CMakeLists.txt:7`, both factories in one target).

- **KernelSpecs** (one per legacy `KernelDescriptor`, same order):
  - `READER` `"reader"` — source `reader`; `dfb_bindings = {IN "in" PRODUCER}`; no CTAs;
    `runtime_arg_schema.runtime_arg_names = {"total_tiles"}`; `hw_config = create_reader_datamovement_config()`
    (resolved triple equals the legacy reader default); `compiler_options` default (`O2` = legacy resolved).
  - `WRITER` `"writer"` — source `writer`; `dfb_bindings = {IN_TRANSPOSE "in_transpose" CONSUMER, OUT "out"
    PRODUCER, OUT "out" CONSUMER}`; `compile_time_args = {{"C", C}}`; RTAs `{"total_tiles"}`;
    `hw_config = create_writer_datamovement_config()`; `compiler_options` default (`O2`).
  - `COMPUTE` `"compute"` — source `compute`; `dfb_bindings = {IN "in" CONSUMER, IN_TRANSPOSE "transpose_in"
    PRODUCER}`; no CTAs; RTAs `{"total_tiles"}`; `hw_config = ComputeHardwareConfig{.fpu_math_fidelity = HiFi4,
    .sfpu_precision_mode = Precise, .enable_32_bit_dest = false, .double_buffer_dest = true}` (Style B, every
    legacy-set field copied through the table; `unpack_modes` empty; `config_1xx` / `config_12x` unset = legacy
    `bfp8_pack_precise = false` / `enable_trisc2_rvv = false`); `compiler_options = {.opt_level = O3}`.
  - Accessor names follow each kernel's own legacy variable (`cb_X` → `dfb::X`): the compute kernel called the
    intermediate `cb_transpose_in`, the writer `cb_in_transpose`, so the one DFB carries `transpose_in` on the
    compute binding and `in_transpose` on the writer's.
- **DataflowBufferSpecs** (one per legacy `CBDescriptor`; no aliasing):
  - `IN` `"in"` (legacy `c_0`): `entry_size = input_tile_size`, `num_entries = total_tiles_per_core`,
    `data_format_metadata = input_format`, `borrowed_from = INPUT`.
  - `OUT` `"out"` (legacy `c_1`): `entry_size = output_shard_shape[1] * element_size`,
    `num_entries = output_shard_shape[0]`, `data_format_metadata = output_format`, `borrowed_from = OUTPUT`.
  - `IN_TRANSPOSE` `"in_transpose"` (legacy `c_2`): `entry_size = intermediary_tile_size`, `num_entries = 16`,
    `data_format_metadata = Float16_b`.
  - `tile_format_metadata` unset on all three (legacy `tile` unset).
- **SemaphoreSpecs**: none.
- **TensorParameters**: `INPUT` (`a.tensor_spec()`), `OUTPUT` (`output.tensor_spec()`); strict (relaxation
  `none`). Both borrow-only: named by `borrowed_from`, no kernel `TensorBinding` (the validator's stated
  exception for borrow-only parameters).
- **WorkUnitSpecs**: one, `"main"`, `{READER, WRITER, COMPUTE}`, `target_nodes = input_core_grid`.
- **ProgramRunArgs**: `kernel_run_args` for all three kernels with `total_tiles = total_tiles_per_core` on every
  node of `input_cores` (the legacy node-first helper `set_runtime_args_for_all_kernels` is kept and retargeted at
  `KernelRunArgs` via `AddRuntimeArgsForNode`); `tensor_args = {INPUT, OUTPUT}`.
- **Per-coordinate programs**: n/a (custom concept; SPMD).
- **Op-owned tensors**: none.

Endpoint census, re-derived from the kernels (agrees with the brief):

| DFB | reader | compute | writer | disposition |
|---|---|---|---|---|
| `in` (`c_0`) | `push_back` (`reader:12`) — locked producer | `wait_front` / `pop_front` (`compute:15, 22`) + LLK reads — locked consumer | — | 1P + 1C |
| `in_transpose` (`c_2`) | — | `reserve_back` / `push_back` (`compute:24, 29`) + pack — locked producer | `wait_front` / `pop_front` (`writer:35, 48, 52, 61`) + peeks — locked consumer | 1P + 1C |
| `out` (`c_1`) | — | — | `reserve_back` / `push_back` (`writer:27, 64`) + raw NoC writes; nothing drains | one toucher → self-loop (DM, Gen1-legal) |

## Preserved Multiplicity

none — no work-split multiplicity in legacy (one `KernelDescriptor` per source, all over one grid).

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `.cpp:108` reader CTA 0; `reader:10` | `cb_in_id` (= `c_0`) | `DFBBinding{IN, "in", PRODUCER}`; kernel uses `dfb::in` |
| `.cpp:109` writer CTA 0; `writer:18` | `cb_in_transpose_id` (= `c_2`) | `DFBBinding{IN_TRANSPOSE, "in_transpose", CONSUMER}`; `dfb::in_transpose` |
| `.cpp:109` writer CTA 1; `writer:20` | `cb_out_id` (= `c_1`) | `DFBBinding{OUT, "out", PRODUCER}` + `{OUT, "out", CONSUMER}`; `dfb::out` |
| `.cpp:109` writer CTA 2; `writer:21` | positional `C` | named CTA `"C"`; `get_arg(args::C)` |
| `.cpp:110` compute CTA 0; `compute:36` | `cb_in_id` (= `c_0`) | `DFBBinding{IN, "in", CONSUMER}`; `dfb::in` |
| `.cpp:110` compute CTA 1; `compute:37` | `cb_in_transpose_id` (= `c_2`) | `DFBBinding{IN_TRANSPOSE, "transpose_in", PRODUCER}`; `dfb::transpose_in` |
| `.cpp:33-37` RTA slot 0 of all three; `reader:9`, `writer:13`, `compute:33` | positional `total_tiles_per_core` | named RTA `"total_tiles"`; `get_arg(args::total_tiles)` |
| `.cpp:85-86`, `.cpp:93-94` | `CBDescriptor::buffer = input_buffer / output_buffer` | `DataflowBufferSpec::borrowed_from = INPUT / OUTPUT` + `TensorArgument`s |
| `.cpp:163-166` (override) | CB-only `ProgramDescriptor` + `apply_descriptor_runtime_args` | returned `ProgramRunArgs::tensor_args = {INPUT, OUTPUT}` |
| `.cpp:7` | `#include …/cnn_cb_descriptor.hpp` (`make_cnn_circular_buffer`) | dropped — the factory no longer builds `CBDescriptor`s (helper itself untouched) |

Not present, so nothing to drop: buffer-address RTAs / `Buffer*` RTA bindings, `TensorAccessorArgs` plumbing,
page-size third-argument CTAs/RTAs, semaphore-ID RTAs.

Kernel-side swaps (whitelist rules 1-4, 7):
- `reader:11` / `writer:24-25`: `experimental::CB` (pool alias for `CircularBuffer`) → `DataflowBuffer` from the
  `dfb::` token. The reader's pool-header include (`reader:6`) supplied only that alias, so it drops with the
  rule-1 sweep; the writer keeps it (it still calls `experimental::set_read_state` / `read_with_state`).
- `compute:9`: `api/dataflow/circular_buffer.h` → `api/dataflow/dataflow_buffer.h`.
- All three add `experimental/kernel_args.h`; DM kernels add `api/dataflow/dataflow_buffer.h`.
- `compute:11-30`: the helper `transpose<BATCH_SIZE>(uint32_t cb_in, uint32_t cb_out)` takes the binding tokens
  instead of CB ids (`DFBBindingToken<P, C>` parameters), builds its `DataflowBuffer` wrappers from them and passes
  them straight to the LLKs — the same shape as legacy (handle in, local wrapper, handle to LLK), with the token as
  the handle. Avoids both the low-level raw-id `DataflowBuffer(uint16_t)` constructor and `get_id()` at LLK sites.
- `compute:39-52` init / uninit LLKs: CB ids → `dfb::in` / `dfb::transpose_in`.
- `writer:19`: `constexpr … get_tile_size(cb_in_transpose)` stays a `constexpr` free-function call with the token,
  `get_tile_size(dfb::in_transpose)` (rule 7 `constexpr` exception; Gen1-only, recorded in the report).

## Applied Patterns

- [Sync-free and single-ended CBs → self-loop DFB](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb):
  `OUT` on the writer (single-ended: the writer reserves/pushes into the resident output shard and nothing drains
  it). DM self-loop — Gen1-legal, Quasar-uplift debt.
- [Self-loop DFB binding](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-self-loop-dfb-binding):
  shared accessor name `"out"` for both endpoints.
- [Pass DFB handles directly to LLKs and kernel-lib helpers](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers):
  compute's `compute_kernel_hw_startup` / `pack_untilize_*` / `transpose_*` calls.
- Borrowed-memory DFBs (migration guide, `DataflowBufferSpec`): `IN` ← `INPUT`, `OUT` ← `OUTPUT`.
- [Translating `override_runtime_arguments`](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#translating-override_runtime_arguments):
  two tensor-address writes → two `TensorArgument`s; no runtime args; guards carried across.
- [Unity-build hygiene for anonymous-namespace symbols](../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-unity-build-hygiene-for-anonymous-namespace-symbols):
  spec names nested in `namespace { namespace chw { … } }`; the `using namespace tt::tt_metal::experimental`
  directive is function-scoped so it cannot reach `convert_to_hwc`'s code in the merged unity TU.

## Deferred / Flagged

- New findings during planning (none blocking; each goes to the report):
  - `DataflowBuffer` FIFO counts are `uint16_t` where `CircularBuffer`'s were `int32_t` (`reader:12` pushes
    `total_tiles`). No behavior change reachable: `in` is borrowed from the input shard, so
    `total_tiles_per_core × tile size` must fit one core's L1, far below 65536 tiles.
  - The factory stops calling `make_cnn_circular_buffer(…, log_cb = true)`, so its three debug-level
    `"Creating CB at index …"` log lines disappear from this op's `LogOp` debug output.
  - The `total_tiles` RTA has the same value on every node — a CRTA candidate for a later pass (not converted:
    RTA→CRTA changes dispatch semantics).
  - Inherited, not re-litigated: the Device 2.0 judgment call on the pool helper's hand-built loopback endpoint
    (brief, first bullet).
