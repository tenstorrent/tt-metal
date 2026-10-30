# Port Plan — deepseek_moe_gate

Port plan for `ttnn/cpp/ttnn/operations/experimental/deepseek/moe/deepseek_moe_gate`, ported from the
`ProgramDescriptor` API (`ProgramDescriptorFactoryConcept` plus a descriptor-style `override_runtime_arguments`)
to Metal 2.0 (`CustomProgramSpecFactoryConcept`).
Written during the inventory and planning steps; committed alongside the port for review.

All `file:line` references are to the pre-port tree (`b5ec776b097`) unless marked otherwise. Shorthand:
`factory.{hpp,cpp}` = `device/deepseek_moe_gate_program_factory.{hpp,cpp}`; `builder.{hpp,cpp}` =
`device/deepseek_moe_gate_program_descriptor_builder.{hpp,cpp}`; `kernel` = `device/kernels/deepseek_moe_gate_kernel.cpp`;
`gate.hpp` / `utils.hpp` = `device/unified_kernels/{deepseek_moe_gate,kernel_utils}.hpp`; `hw_startup.h` =
`device/kernel_includes/tt_metal/include/compute_kernel_api/deepseek_compute_kernel_hw_startup.h`.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept`. `DeepseekMoeGateProgramFactory::create_descriptor` returns a
  `ProgramDescriptor` (`factory.hpp:18-21`, `factory.cpp:34-37`); its body is
  `build_moe_gate_program_descriptor` (`builder.cpp:40-185`). Alongside it, a descriptor-style
  `void override_runtime_arguments(Program&, …, const std::optional<MeshCoordinate>& = std::nullopt)`
  (`factory.hpp:24-29`, `factory.cpp:39-50`).
- Where the factory methods live: a conventional factory struct, the single alternative of
  `program_factory_t = std::variant<program::DeepseekMoeGateProgramFactory>`
  (`device/deepseek_moe_gate_device_operation.hpp:25`). Not direct-descriptor; exception 3 does not apply. The
  port is a method swap inside the existing struct.
- Variants: single. One config axis, `enable_sigmoid`, which is a compute CTA (a template argument of the kernel's
  `ComputeCTArgs`). No defines, no branches in the factory, no runtime kernel-source selection.
- Custom `compute_program_hash`: none — default reflection hash over
  `operation_attributes_t{eps, scaling_factor, enable_sigmoid}` (`device/deepseek_moe_gate_device_operation_types.hpp:13-17`)
  and the five tensors. No backdoor (`attribute_values` / `to_hash`).

*(The Metal 2.0 factory concept was chosen during the audit — see the brief's TTNN factory analysis. Carried
forward in [TTNN ProgramFactory](#ttnn-programfactory) below.)*

### Kernels
One op-owned *unified* source, `kernel` (`builder.cpp:20-21`), instantiated three times; all three over
`core_ranges = all_cores = input_shard.grid` (`builder.cpp:72`). The source splits per RISC at the preprocessor
level (`#if defined(COMPILE_FOR_NCRISC / BRISC / TRISC)`, `kernel:25-56`).

| unique_id | CTAs (positional) | CTAs (named, host emission order) | RTAs | CRTAs | defines | opt_level (resolved) | config |
|---|---|---|---|---|---|---|---|
| reader (`builder.cpp:137-148`) | none | `moe_gate_input_cb` = 0, `moe_gate_bias_cb` = 1, `moe_gate_input_indices_cb` = 3, `moe_gate_is_active_core` = 1 (`builder.cpp:111-116`) | none | none | none | `O2` (unset; DM default, `tt_metal/impl/program/program.cpp:528-548`) | `DataMovementConfigDescriptor{.processor = RISCV_1, .noc = NOC::RISCV_0_default (= NOC_0), .noc_mode = DM_DEDICATED_NOC}` (`builder.cpp:142-147`) |
| writer (`builder.cpp:150-161`) | none | `moe_gate_output_cb` = 2, `moe_gate_output_indices_cb` = 4, `moe_gate_is_active_core` = 1 (`builder.cpp:117-121`) | none | none | none | `O2` (unset) | `DataMovementConfigDescriptor{.processor = RISCV_0, .noc = NOC::RISCV_1_default (= NOC_1), .noc_mode = DM_DEDICATED_NOC}` (`builder.cpp:155-160`) |
| compute (`builder.cpp:163-169`) | none | `moe_gate_input_cb` = 0, `moe_gate_bias_cb` = 1, `moe_gate_input_indices_cb` = 3, `moe_gate_output_cb` = 2, `moe_gate_output_indices_cb` = 4, `moe_gate_eps` = `bit_cast<uint32_t>(eps)`, `moe_gate_scaling_factor` = `bit_cast<uint32_t>(scaling_factor)`, `moe_gate_enable_sigmoid` = 0/1, `moe_gate_is_active_core` = 1 (`builder.cpp:122-132`) | none | none | none | `O3` (unset; `ComputeConfigDescriptor` resolves `O3`, `tt_metal/impl/program/program.cpp:565`) | `ComputeConfigDescriptor{.math_fidelity = HiFi4}` (`builder.cpp:134-135`); everything else at its default (`tt_metal/api/tt-metalium/program_descriptors.hpp:103-114`): `fp32_dest_acc_en = false`, `dst_full_sync_en = false`, `unpack_to_dest_mode = {}`, `bfp8_pack_precise = false`, `math_approx_mode = false`, `enable_trisc2_rvv = false` |

`grep -n opt_level` over the op directory: no hits, so every level above is the resolved default.
`NOC::RISCV_0_default` / `RISCV_1_default` are the enum values 0 / 1, i.e. `NOC_0` / `NOC_1`
(`tt_metal/api/tt-metalium/kernel_types.hpp:34-39`).

Kernel-side reads: `kernel:18` (`moe_gate_is_active_core`, in the file-scope `struct Core`); NCRISC `kernel:29-31`
(three CB-id CTAs); BRISC `kernel:41-43` (two CB-id CTAs as `WriterCTArgs` template arguments); TRISC
`kernel:46-54` (five CB-id CTAs + `eps` / `scaling_factor` / `enable_sigmoid` as `ComputeCTArgs` template
arguments). The CB ids then travel as `uint32_t` template parameters (`gate.hpp:53-79`) into `Op::impl()`
(`gate.hpp:94-161`), and on NCRISC as a `uint32_t cb_id` function parameter into
`unified_kernels::setup_sharded_buffer` (`utils.hpp:66-70`).

### CBs
All five built by `ttnn::cb_descriptor_from_sharded_tensor(index, tensor)` (`builder.cpp:99-103`;
helper at `ttnn/core/tensor/tensor_utils.cpp:42-66`) and then rewritten by `set_cb_page_size_for_tile`
(`builder.cpp:28-36, 105-109`). One format descriptor each; `address_offset = 0`; no `global_circular_buffer`.

| index | total_size | core_ranges | data_format | page_size | tile | backing |
|---|---|---|---|---|---|---|
| 0 `kInputCb` | `input.buffer()->aligned_size_per_bank()` | input shard grid | from dtype (`Float16_b`) | `tile_size` if it divides `total_size`, else `total_size` | input `tensor_spec().tile()` | **borrowed: input buffer** |
| 1 `kBiasCb` | same rule on bias | bias shard grid | `Float16_b` | same rule | bias tile | **borrowed: bias** |
| 2 `kOutputCb` | same rule on output | output shard grid | `Float16_b` | same rule | output tile | **borrowed: output** |
| 3 `kInputIndicesCb` | same rule on input_indices | its shard grid | `UInt16` | same rule | its tile | **borrowed: input_indices** |
| 4 `kOutputIndicesCb` | same rule on output_indices | its shard grid | `UInt16` | same rule | its tile | **borrowed: output_indices** |

`tile_size = tile.get_tile_size(data_format)`. Under the validated 32×32 shard and tile, every CB is 2048 B with a
2048 B page — one entry — including the row-major output tensors the deepseek_v3 decode path passes
(`models/demos/deepseek_v3/tt/moe_gate.py:135-156`), whose 64 B row page `set_cb_page_size_for_tile` overrides to
2048 B. No GlobalCircularBuffer (no `.global_circular_buffer`, `global_cb` parameter or `remote_cb_config`).

### Semaphores
none

### Tensor accessors
none — no kernel constructs a `TensorAccessor`, and no buffer address reaches any kernel argument. Tensor memory
reaches the kernels only through the five borrowed CBs:

| host site (file:line) | originating Tensor | how it reaches the kernel |
|---|---|---|
| `builder.cpp:99` | `tensor_args.input_tensor` | `CBDescriptor::buffer` of CB 0 (not an RTA) |
| `builder.cpp:100` | `tensor_args.bias_tensor` | `CBDescriptor::buffer` of CB 1 |
| `builder.cpp:101` | `tensor_args.output_tensor` | `CBDescriptor::buffer` of CB 2 |
| `builder.cpp:102` | `tensor_args.input_indices_tensor` | `CBDescriptor::buffer` of CB 3 |
| `builder.cpp:103` | `tensor_args.output_indices_tensor` | `CBDescriptor::buffer` of CB 4 |

### Work split
- n/a — no `split_work_to_cores`. Every core of the input shard grid runs all three kernels on its own one-tile
  shard; there are no per-core arguments.

### Override (`override_runtime_arguments`, `factory.cpp:39-50`) — what it refreshes on a cache hit
| write (in call order) | bucket | Metal 2.0 home |
|---|---|---|
| CB 0 address ← `input_tensor.buffer()` (`update_tensor_cb` → 3-arg `UpdateDynamicCircularBufferAddress`) | tensor address | `TensorArgument` for the input parameter |
| CB 1 ← `bias_tensor.buffer()` | tensor address | `TensorArgument` (bias) |
| CB 2 ← `output_tensor.buffer()` | tensor address | `TensorArgument` (output) |
| CB 3 ← `input_indices_tensor.buffer()` | tensor address | `TensorArgument` (input_indices) |
| CB 4 ← `output_indices_tensor.buffer()` | tensor address | `TensorArgument` (output_indices) |
| (no runtime args) | — | no `kernel_run_args` |

Guards inside it (in `update_tensor_cb`, `factory.cpp:21-30`, run once per tensor):
`TT_FATAL(buffer != nullptr, "deepseek_moe_gate tensor buffer is null for CB {}", …)` (`:23`) and
`TT_FATAL(cb_it != cbs.end(), "Missing globally allocated circular buffer {}", …)` (`:28`). It ignores `coord`.

### Shared kernels
none.
- `grep -rl deepseek_moe_gate_kernel ttnn/cpp/ttnn/operations/` hits only `builder.cpp:21` (plus the audit
  `.md` files). No `_metal2` sibling exists. `models/demos/deepseek_v3_b1/micro_ops/deepseek_moe_gate/op.py:190`
  binds a same-named private copy under `models/`, not this file.
- The op-local headers `unified_kernels/{deepseek_moe_gate,kernel_utils,kernel_op_api}.hpp` are included only by
  `kernel` (repo-wide grep for the `deepseek_moe_gate/device/unified_kernels` path: no hits; `generalized_moe_gate`
  carries its own copies in its own directory).
- `device/kernel_includes/**` is reached from outside only by `tt_metal/tt-llk/tests/sources/deepseek_moe_gate_test.cpp:141-147`
  (six LLK headers). The port does not touch `kernel_includes/`.

Convert `kernel`, `gate.hpp` and `utils.hpp` in place.

### Flags
- No unreferenced kernel files. `utils.hpp` holds never-called helpers (`linear_id_in_grid`,
  `get_split_half_core_info`, `semaphore_dec`, `sync_riscs_enter/exit`, `override_cb_rd_ptr`,
  `reconfig_cbs_for_mask`, `reconfig_cb_interfaces`, `get_local_cb_rd_ptr`, `get_local_cb_page_size`,
  `update_local_cb_rd_ptr`) — left untouched per the audit's ruling (they include the
  `get_cb_tiles_*_ptr` calls the whitelist says cannot port; nothing reaches them).
- `hw_startup.h:20`: `compute_kernel_hw_startup(0, 0, 0)` hard-codes legacy CB 0 (the input) as srcA / srcB / pack
  operand. See Planned Spec Shape, DFB declaration order.
- Never-popped FIFOs are intentional: compute `wait_front`s bias and input_indices without popping
  (`gate.hpp:126-127`); the writer `wait_front`s output and output_indices without popping (`gate.hpp:106-107`).

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `CustomProgramSpecFactoryConcept`.
- **Custom `compute_program_hash`**: none — default reflection hash; nothing to leave intact.
- **Implementation notes**:
  - Method swap inside the existing `DeepseekMoeGateProgramFactory`: `create_descriptor` →
    `create_program_artifacts` (returns `ttnn::device_operation::ProgramArtifacts`), and
    `override_runtime_arguments` drops its `Program&` parameter and returns
    `tt::tt_metal::experimental::ProgramRunArgs` (the return type is what selects the custom concept).
  - The body stays split across the two existing files (no `CMakeLists.txt` edit): `factory.cpp` keeps the entry
    points; `builder.cpp` keeps the body, now `build_moe_gate_program_artifacts` (renamed with the API it returns).
  - The translated override returns `tensor_args` for all five `TensorParameter`s, in the legacy refresh order, and
    no `kernel_run_args`. It keeps the per-tensor null-buffer guard (message names the tensor parameter instead of
    a CB index). The `"Missing globally allocated circular buffer"` guard checks the `Program`'s CB list, which the
    Metal 2.0 override no longer receives: recorded as subject-deleted (invoker-approved), flagged in the report.
  - No device-op-class edits: no pybound `create_descriptor` (`deepseek_moe_gate_nanobind.cpp:12-43` binds only the
    free function), no pybind-only factory parameter, not direct-descriptor.

## Planned Spec Shape

Names: the five `TensorParamName`s are shared by both entry points, so they replace the `kXxxCb` constants in
`builder.hpp` (`inline const`). Kernel and DFB names are local to `builder.cpp`'s anonymous namespace (unity build:
`builder.cpp` and `factory.cpp` compile into one TU; `using namespace` stays function-scoped).

- **KernelSpecs** (one per legacy `KernelDescriptor`, same order, all on `kernel`):
  - `"reader"` — `dfb_bindings = {input "moe_gate_input" PRODUCER, bias "moe_gate_bias" PRODUCER,
    input_indices "moe_gate_input_indices" PRODUCER}`; `compile_time_args = {{"moe_gate_is_active_core", 1}}`;
    no RTAs; `hw_config = ttnn::create_reader_datamovement_config()` (resolved triple equals the reader default);
    `compiler_options` default (`O2` = legacy resolved).
  - `"writer"` — `dfb_bindings = {output "moe_gate_output" CONSUMER, output_indices "moe_gate_output_indices"
    CONSUMER}`; `compile_time_args = {{"moe_gate_is_active_core", 1}}`; no RTAs;
    `hw_config = ttnn::create_writer_datamovement_config()`; `compiler_options` default (`O2`).
  - `"compute"` — `dfb_bindings = {input CONSUMER, bias CONSUMER, input_indices CONSUMER, output PRODUCER,
    output_indices PRODUCER}` (same accessor names); `compile_time_args = {moe_gate_eps, moe_gate_scaling_factor,
    moe_gate_enable_sigmoid, moe_gate_is_active_core}` with the legacy values; no RTAs;
    `hw_config = ComputeHardwareConfig{.fpu_math_fidelity = HiFi4, .sfpu_precision_mode = Precise,
    .enable_32_bit_dest = false, .double_buffer_dest = true}` (Style B; `unpack_modes` empty — no Float32 DFB and
    32-bit Dest off; `config_1xx` / `config_12x` unset = legacy `bfp8_pack_precise = false` /
    `enable_trisc2_rvv = false`); `compiler_options = {.opt_level = O3}`.
  - Accessor names drop the legacy CTA names' `_cb` type suffix (the `dfb::` namespace now carries it); the
    `moe_gate_` prefix stays, matching the surviving named CTAs.
- **DataflowBufferSpecs** (one per legacy CB, **declared in legacy CB-index order** — input, bias, output,
  input_indices, output_indices — so they take device slots 0-4, the legacy CB indices; the compute kernel's
  hardware startup `compute_kernel_hw_startup(0, 0, 0)` assumes the input sits on slot 0. Slots are assigned
  lowest-free in declaration order, `tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:2182-2202`,
  `tt_metal/impl/metal2_host_api/program_spec/construction/resource/dfb.cpp:253-254`):
  - names `"input"`, `"bias"`, `"output"`, `"input_indices"`, `"output_indices"`;
  - each built from its tensor by the legacy rule: `total = tensor.buffer()->aligned_size_per_bank()`,
    `tile_size = tile.get_tile_size(format)`, `entry_size = total % tile_size == 0 ? tile_size : total`,
    `num_entries = total / entry_size`; `data_format_metadata` = dtype's format; `tile_format_metadata` =
    `tensor_spec().tile()` (legacy `format_descriptors[0].tile`); `borrowed_from` = the matching parameter.
  - No aliasing, no multi-binding flag, no self-loop.
- **SemaphoreSpecs**: none.
- **TensorParameters**: `"input_tensor"`, `"bias_tensor"`, `"output_tensor"`, `"input_indices_tensor"`,
  `"output_indices_tensor"`, each from its own `MeshTensor`'s `tensor_spec()`; strict (relaxation `none`). All five
  are borrow-only (named by `borrowed_from`, no kernel `TensorBinding`).
- **WorkUnitSpecs**: one, `"main"`, `{reader, writer, compute}`, `target_nodes = input_shard.grid`.
- **ProgramRunArgs**: `tensor_args` for all five parameters (from `tensor_args`; the outputs are echoed in
  `tensor_return_value`); no `kernel_run_args`.
- **Per-coordinate programs**: n/a (custom concept; SPMD).
- **Op-owned tensors**: none.

Endpoint census, re-derived from the kernel (agrees with the brief):

| DFB | reader (NCRISC) | compute (TRISC) | writer (BRISC) | disposition |
|---|---|---|---|---|
| input (0) | `reserve_back` / `push_back` via `setup_sharded_buffer` (`kernel:35`, `utils.hpp:66-70`) — locked producer | `wait_front` / `pop_front` (`gate.hpp:139, 143`) + unpack — locked consumer | — | 1P + 1C |
| bias (1) | same (`kernel:36`) — locked producer | `wait_front` (`gate.hpp:127`), never popped — locked consumer | — | 1P + 1C |
| output (2) | — | `reserve_back` / `push_back` (`gate.hpp:147, 153`) + pack — locked producer | `wait_front` (`gate.hpp:107`), never popped — locked consumer | 1P + 1C |
| input_indices (3) | same (`kernel:37`) — locked producer | `wait_front` (`gate.hpp:126`), never popped — locked consumer | — | 1P + 1C |
| output_indices (4) | — | `reserve_back` / `push_back` (`gate.hpp:148, 157`) + pack — locked producer | `wait_front` (`gate.hpp:106`), never popped — locked consumer | 1P + 1C |

## Preserved Multiplicity

none — no work-split multiplicity in legacy (three `KernelDescriptor`s of one source, one per RISC, over one grid;
each binds a different side of each DFB).

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `builder.cpp:112, 123`; `kernel:29, 47` | named CTA `moe_gate_input_cb` = 0 (reader, compute) | `DFBBinding{input, "moe_gate_input", PRODUCER / CONSUMER}`; `dfb::moe_gate_input` |
| `builder.cpp:113, 124`; `kernel:30, 48` | named CTA `moe_gate_bias_cb` = 1 | `DFBBinding{bias, "moe_gate_bias", …}`; `dfb::moe_gate_bias` |
| `builder.cpp:118, 126`; `kernel:42, 50` | named CTA `moe_gate_output_cb` = 2 (writer, compute) | `DFBBinding{output, "moe_gate_output", CONSUMER / PRODUCER}`; `dfb::moe_gate_output` |
| `builder.cpp:114, 125`; `kernel:31, 49` | named CTA `moe_gate_input_indices_cb` = 3 | `DFBBinding{input_indices, "moe_gate_input_indices", …}`; `dfb::moe_gate_input_indices` |
| `builder.cpp:119, 127`; `kernel:43, 51` | named CTA `moe_gate_output_indices_cb` = 4 | `DFBBinding{output_indices, "moe_gate_output_indices", …}`; `dfb::moe_gate_output_indices` |
| `builder.cpp:99-109` | `cb_descriptor_from_sharded_tensor` + `set_cb_page_size_for_tile` (`CBDescriptor::buffer` = tensor buffer) | `DataflowBufferSpec` with the same sizing rule + `borrowed_from` + `TensorArgument` |
| `factory.cpp:21-30, 45-49` (override) | `update_tensor_cb` → `UpdateDynamicCircularBufferAddress` per CB | returned `ProgramRunArgs::tensor_args` (five entries) |
| `builder.hpp:15-20` | `inline constexpr uint8_t kInputCb … kOutputIndicesCb` | shared `TensorParamName` constants |

Named CTAs that stay named CTAs (1:1): `moe_gate_eps`, `moe_gate_scaling_factor`, `moe_gate_enable_sigmoid`,
`moe_gate_is_active_core` (`get_named_compile_time_arg_val("…")` → `get_arg(args::…)`, still `constexpr`).
Not present, so nothing to drop: positional CTAs, RTAs / CRTAs (and so no buffer-address RTAs or `Buffer*` RTA
bindings), `TensorAccessorArgs` plumbing, page-size third-argument CTAs, semaphore-ID RTAs.

Kernel-side swaps (whitelist rules 1-4, 8):
- `CircularBuffer` → `DataflowBuffer`, constructed from the binding token everywhere (`gate.hpp:104-105, 119-123`,
  `utils.hpp:67`); `api/dataflow/circular_buffer.h` → `api/dataflow/dataflow_buffer.h` (`gate.hpp:11, 27`,
  `utils.hpp:11`); `experimental/kernel_args.h` added to `kernel`.
- `WriterCTArgs` / `ComputeCTArgs` (`gate.hpp:53-79`): the CB-id template parameters become `const auto&`
  parameters bound to the `dfb::` tokens, so the structs carry the tokens and `Op::impl()` builds each
  `DataflowBuffer` from its token. A token cannot be a value template argument (it is not a structural type),
  but a reference to the namespace-scope `constexpr` token can — the same form the framework's
  `LLKOperandFrom<dfb::name>` uses (`tt_metal/hw/inc/api/llk_operand_from_tokens.h:34, 58`). The LLK call sites
  receive the tokens through their implicit `uint32_t` conversion. The `eps` / `scaling_factor` /
  `enable_sigmoid` template parameters stay `uint32_t`.
- `setup_sharded_buffer(uint32_t cb_id, …)` (`utils.hpp:66-70`) takes a `DFBBindingToken<Pap, Cap>` instead.
- Comments that name "CB(s)" for these buffers are reworded to "DFB(s)"; nothing else in them changes.
- Untouched: `hw_startup.h` (and all of `kernel_includes/`), `kernel_op_api.hpp`, the never-called helpers in
  `utils.hpp`.

## Applied Patterns

- Borrowed-memory DFBs (migration guide, `DataflowBufferSpec`): all five DFBs borrow their tensor's L1 shard.
- [Pass DFB handles directly to LLKs and kernel-lib helpers](../../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers):
  every compute LLK call (`reconfig_full_operand`, `pack_reconfig_data_format`, `copy_init`, `copy_tile`,
  `reconfig_data_format_srca`, `deepseek_moe_gate_init`, `deepseek_moe_gate`, `pack_tile`).
- [Translating `override_runtime_arguments`](../../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/port/metal2_port.md#translating-override_runtime_arguments):
  five tensor-address writes → five `TensorArgument`s; no runtime args; null-buffer guard carried across.
- [Unity-build hygiene for anonymous-namespace symbols](../../../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-unity-build-hygiene-for-anonymous-namespace-symbols):
  shared names in `builder.hpp`; per-file helpers with distinct names; function-scoped `using namespace`.

## Deferred / Flagged

- New findings during planning (none blocking; each goes to the report):
  - `compute_kernel_hw_startup(0, 0, 0)` (`hw_startup.h:20`) is a literal CB index outside the port's surface;
    handled by declaration order (invoker-approved), not by editing `kernel_includes/`.
  - The override's program-shape guard (`factory.cpp:28`) has no Metal 2.0 subject; recorded as a deviation from
    the recipe's "override guards are never subject-deleted" (invoker-approved).
  - `DataflowBuffer` FIFO counts are `uint16_t` (legacy `CircularBuffer` took `int32_t`); every count here is 1.
  - The `uint32_t` → `const auto&` change of the CTArgs template parameters has no recipe / catalog entry.
