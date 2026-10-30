# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/experimental/deepseek/moe/deepseek_moe_gate`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** ``c41f7f4ef0a 2026-10-29 docs(metal_2.0): DFBAccessor -> DFBBindingToken; drop the dead `.id` form`` *(carry this line into the port report's Provenance section)*

*Path convention:* paths are op-relative unless they start with `tt_metal/`, `ttnn/`, `models/`, `tests/` or `docs/`.

**Shape of the op, in one paragraph.** There is one DeviceOperation (`DeepseekMoeGateDeviceOperation`) and one factory (`DeepseekMoeGateProgramFactory`). The factory's entry points are in `device/deepseek_moe_gate_program_factory.cpp`; the descriptor body is in `device/deepseek_moe_gate_program_descriptor_builder.cpp`. The factory builds one op-owned unified kernel source, `device/kernels/deepseek_moe_gate_kernel.cpp`, into three `KernelDescriptor`s (NCRISC reader, BRISC writer, TRISC compute), all on the input tensor's shard grid. It also builds five tensor-backed (borrowed-memory) CBs, one per io tensor. There are no runtime args, no semaphores, no `TensorAccessor`s and no NoC traffic. The only config axis is the compute CTA `enable_sigmoid`.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (see `ttnn_factory.md`). Carry them forward:

- **Current concept:** `descriptor`. `create_descriptor(attrs, tensor_args, tensor_return_value)` takes no `MeshCoordinate` (`device/deepseek_moe_gate_program_factory.hpp:18-21`, `.cpp:34-37`).
- **Op-owned tensors:** none.
- **Target concept:** `CustomProgramSpecFactoryConcept`. Translate `DeepseekMoeGateProgramFactory::override_runtime_arguments` @ `device/deepseek_moe_gate_program_factory.cpp:39-50` (decl `.hpp:24-29`).
  - **What it refreshes today:** exactly the five tensor-backed CB addresses, nothing else. Each goes through `update_tensor_cb` (`:21-30`), which finds the globally-allocated CB holding the index and calls the 3-argument `UpdateDynamicCircularBufferAddress(program, cb_id, *buffer)`. The calls are at `:45-49`, in the order input, bias, output, input_indices, output_indices.
  - **Faithful translation:** return a `ProgramRunArgs` whose `tensor_args` has one `TensorArgument` for each of the five `TensorParameter`s. No `kernel_run_args`: the op has no RTAs or CRTAs. No `dfb_run_overrides`: the legacy override never changes a size. Refresh exactly this set, no more and no less (`ttnn_factory.md`, "The cache-hit contract").
  - **`coord`:** unused (SPMD), as it is today (`:44`).
- **Gate-cleared, confirmed absent:** a `TensorParameter relaxation` that is neither `none` nor an analysis pointer (the cell is `none`) · `get_dynamic_runtime_args` (absent).
- **Non-gating device-op facts:**
  - No custom hash. The default reflection hash applies; there is nothing to leave alone.
  - No pybound `create_descriptor`. `deepseek_moe_gate_nanobind.cpp:12-43` binds only the user-facing function, so no pybind line is removed.
  - The op already has a conventional `program_factory_t` (`device/deepseek_moe_gate_device_operation.hpp:25`), so no direct-descriptor conversion is needed.
  - None of the three device-op-class exceptions in `ttnn_factory.md` applies.

## Construct — to do

**Tensor bindings** (per binding): all five are **clean**, as borrowed-memory DFBs. None is Case 1 or Case 2, and no kernel gets a `TensorBinding` or a `TensorAccessor`.

- Declare five `TensorParameter`s from each tensor's own `tensor_spec()`.
- Name each one in exactly one DFB's `borrowed_from`. A borrow-only `TensorParameter` is legal without any kernel binding (`migration_guide.md`, TensorParameter).
- Take all five `MeshTensor`s from `tensor_args`. The outputs are preallocated and passed in, and are echoed in `tensor_return_value` (`device/deepseek_moe_gate_device_operation.cpp:110-113`).

| `TensorParameter` (tensor) | Borrowed DFB (legacy CB, `device/deepseek_moe_gate_program_descriptor_builder.hpp:16-20`) | Class |
|---|---|---|
| `input_tensor` | CB 0 `kInputCb` | clean (borrowed DFB; compute unpacks) |
| `bias_tensor` | CB 1 `kBiasCb` | clean (borrowed DFB; compute unpacks) |
| `output_tensor` | CB 2 `kOutputCb` | clean (borrowed DFB; compute packs into the resident shard) |
| `input_indices_tensor` | CB 3 `kInputIndicesCb` | clean (borrowed DFB; compute copies) |
| `output_indices_tensor` | CB 4 `kOutputIndicesCb` | clean (borrowed DFB; compute packs into the resident shard) |

**TensorParameter relaxation:** `none`. Every parameter is strict.

**TensorAccessor 3rd arg:** none. The op constructs no `TensorAccessor`.

**CB endpoints:** all legal. Each of the five DFBs is a plain 1P+1C whose roles are fixed by FIFO ops, on every kernel node and in both `enable_sigmoid` configs. There is no self-loop, no relabel, no multi-binding flag, no drop and no conditional DFB.

| DFB (legacy CB) | PRODUCER | CONSUMER |
|---|---|---|
| input (0) | reader: `reserve_back` / `push_back` via `setup_sharded_buffer` (`deepseek_moe_gate_kernel.cpp:35`, `unified_kernels/kernel_utils.hpp:66-70`) | compute: `wait_front` / `pop_front` (`unified_kernels/deepseek_moe_gate.hpp:139, 143`) |
| bias (1) | reader (`deepseek_moe_gate_kernel.cpp:36`) | compute: `wait_front` only (`unified_kernels/deepseek_moe_gate.hpp:127`) |
| output (2) | compute: `reserve_back` / `push_back` (`unified_kernels/deepseek_moe_gate.hpp:147, 153`) | writer: `wait_front` only (`:107`) |
| input_indices (3) | reader (`deepseek_moe_gate_kernel.cpp:37`) | compute: `wait_front` only (`unified_kernels/deepseek_moe_gate.hpp:126`) |
| output_indices (4) | compute: `reserve_back` / `push_back` (`unified_kernels/deepseek_moe_gate.hpp:148, 157`) | writer: `wait_front` only (`:106`) |

That gives these per-`KernelSpec` binding sets:
- **reader:** input, bias, input_indices as PRODUCER
- **writer:** output, output_indices as CONSUMER
- **compute:** input, bias, input_indices as CONSUMER; output, output_indices as PRODUCER

**DFB contents.** Carry the five CBs over from the legacy builder:
- **Source:** `ttnn::cb_descriptor_from_sharded_tensor(...)` with `address_offset = 0`, then `set_cb_page_size_for_tile` (`deepseek_moe_gate_program_descriptor_builder.cpp:99-109, 28-36`).
- **Entry size:** reproduce the page-size rule verbatim: `entry_size` = tile size, or the total size if the tile size does not divide it. Under the validated 32×32 shard and 32×32 tile, each DFB is 1 entry × 2048 B.
- **Metadata:** set `data_format_metadata` to `Float16_b` for input, bias and output, and `UInt16` for input_indices and output_indices. Copy `tile_format_metadata` from the legacy `format_descriptors[0].tile`, which is the tensor's 32×32 tile. Compute touches all five DFBs, and a format-less DFB gets no compute JIT metadata.

**KernelSpecs and the WorkUnitSpec.** Use three `KernelSpec`s on the same source, one per legacy `KernelDescriptor`, in one `WorkUnitSpec` over the input tensor's shard grid (`all_cores`, `deepseek_moe_gate_program_descriptor_builder.cpp:72`).

- **Hardware configs** map 1:1 onto the Metal 2.0 helpers in `tt_metal/api/tt-metalium/experimental/metal2_host_api/data_movement_hardware_config.hpp:107-155`:
  - reader: `RISCV_1`, `NOC::RISCV_0_default` (NOC 0), `DM_DEDICATED_NOC` → `CreateReaderDataMovementConfig()`
  - writer: `RISCV_0`, `NOC::RISCV_1_default` (NOC 1), `DM_DEDICATED_NOC` → `CreateWriterDataMovementConfig()`
  - compute: `HiFi4`, every other field at its default (`deepseek_moe_gate_program_descriptor_builder.cpp:134-135`)
- **Named CTAs:** `moe_gate_is_active_core` (= 1) on all three, and `moe_gate_eps`, `moe_gate_scaling_factor`, `moe_gate_enable_sigmoid` on compute (`deepseek_moe_gate_program_descriptor_builder.cpp:111-132`). `eps` and `scaling_factor` are float bit patterns (`std::bit_cast`, `:23-26`); keep the same `uint32_t` values. The five `moe_gate_*_cb` CTAs become DFB bindings.

## Watch for

- **CB endpoints (multi-binding):** none.
- **Cross-op / shared kernels:** none. The kernel and every header it includes are op-local, and this op's single factory is the only binder. A census of `deepseek_moe_gate_kernel` under `ttnn/cpp/ttnn/operations/` hits only this op's builder. `models/demos/deepseek_v3_b1/micro_ops/deepseek_moe_gate/op.py:190` binds a *same-named private copy* under `models/`, not this file. No `_metal2` fork exists and none is needed: convert the kernel in place as ordinary port work. There is no other binding op and no sunset list. "Porting a shared kernel" does not apply.
- **RTA varargs:** none. No RTAs or CRTAs, and every CTA is a distinct named field read at a fixed name.
- **Anything else the porter needs:**
  - **Literal CB index 0 in the compute init.** `deepseek_compute_kernel_init()` (`kernel_includes/tt_metal/include/compute_kernel_api/deepseek_compute_kernel_hw_startup.h:17-21`, called at `deepseek_moe_gate_kernel.cpp:55`) calls `compute_kernel_hw_startup(0, 0, 0)`. That uses legacy CB 0, the input, as srcA, srcB and pack operand. In Metal 2.0 a DFB's device slot, which equals its CB index on Gen1 (`tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3242`), is the lowest free slot among overlapping DFBs, assigned in `ProgramSpec::dataflow_buffers` declaration order (`dataflow_buffer.cpp:2182-2202`; `tt_metal/impl/metal2_host_api/program_spec/construction/resource/dfb.cpp:253-270`). There are two faithful options:
    - (a) Leave the helper untouched and **declare the DFBs in legacy CB-index order**: input, bias, output, input_indices, output_indices. That gives slots 0-4, identical to the legacy indices. This is zero kernel change, but it relies on observed implementation behavior, not a documented contract.
    - (b) Pass the input DFB token into the helper, if the kernel-side whitelist allows that edit to an op-local helper.

    Do not leave the literal resolving to whatever lands on slot 0. Record the choice in the port report.
  - **CB ids reach the wrappers as plain integers.**
    - The writer and compute receive their CB ids as `uint32_t` template arguments (`WriterCTArgs<…>` / `ComputeCTArgs<…>`, `deepseek_moe_gate_kernel.cpp:41-54`; `unified_kernels/deepseek_moe_gate.hpp:53-79`) and build `CircularBuffer`s from them (`:104-105, 119-123`).
    - The reader passes them through `setup_sharded_buffer(uint32_t cb_id, …)` (`unified_kernels/kernel_utils.hpp:66-70`).
    - `dfb::name` works in those positions, because the token's `uint32_t` conversion is `constexpr` (`port_patterns.md`, "Pass DFB handles directly…"). But a `DataflowBuffer` built from that integer goes through the low-level `DataflowBuffer(uint16_t)` constructor, which `tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:155-156` documents as "prefer DFBBindingToken / RelayDFBBindingToken for new kernel code".

    Confirm the construction form against the kernel-side whitelist rather than swapping blind. The `circular_buffer.h` include sites are `unified_kernels/deepseek_moe_gate.hpp:11, 27` and `unified_kernels/kernel_utils.hpp:11`.
  - **Named CTAs in constant-expression positions.** They are read at file scope (`struct Core`, `deepseek_moe_gate_kernel.cpp:17-19`) and as template arguments (`:41-54`). `constexpr auto x = get_arg(args::x)` is legal for CTAs (`tt_metal/hw/inc/experimental/kernel_args.h:80-83`), so these positions survive. Keep them `constexpr`.
  - **The unified kernel's `#if defined(COMPILE_FOR_NCRISC / BRISC / TRISC)` split is preprocessor-level.** Each `KernelSpec`'s generated `dfb::` namespace therefore only needs the tokens its own branch names (the binding sets above). Do not bind a DFB to a kernel just to make a token exist.
  - **Leave the never-called helpers in `unified_kernels/kernel_utils.hpp` untouched.** Do not convert them and do not delete them. They are:
    - `linear_id_in_grid` / `get_split_half_core_info` (`:30-55`)
    - `semaphore_dec` (`:73-75`)
    - `sync_riscs_enter` / `sync_riscs_exit` (`:95-117`)
    - `override_cb_rd_ptr` (`:126-128`)
    - `reconfig_cbs_for_mask` / `reconfig_cb_interfaces` (`:149-220`)
    - `get_local_cb_rd_ptr` / `get_local_cb_page_size` / `update_local_cb_rd_ptr` (`:224-237`)

    In particular, the CB-to-DFB whitelist row "`get_cb_tiles_acked_ptr` / `get_cb_tiles_received_ptr` → no DFB equivalent, cannot be ported today" is **not** triggered by `kernel_utils.hpp:177-178`, because nothing in this op reaches that code. This is an explicit audit ruling. If a scan stops you there, note it in the port report and continue.
  - **Never-popped FIFOs are intentional.** Compute `wait_front`s bias and input_indices without popping; the writer `wait_front`s output and output_indices without popping. Do not "balance" them; per-enqueue DFB state is reinitialized (`migration_guide.md`, DataflowBufferSpec).
  - **Legacy CB placement differs from Metal 2.0 placement, and that is fine.** Legacy CBs sit on each tensor's own shard grid (`ttnn/core/tensor/tensor_utils.cpp:57`). Validation lets CBs 1-4 sit on grids wider than the kernel grid (`device/deepseek_moe_gate_device_operation.cpp:78, 83, 88`). Metal 2.0 derives DFB placement from the binding kernels' `WorkUnitSpec` nodes, so extra-grid instances, which no kernel touches, disappear. Do not try to reproduce the legacy `core_ranges`. Every in-tree caller uses one grid for all five tensors.
  - **The factory body spans two `.cpp` files in a unity-build target** (`CMakeLists.txt:14, 38-42`): `deepseek_moe_gate_program_factory.cpp` (anonymous namespace `:19-32`) and `deepseek_moe_gate_program_descriptor_builder.cpp` (anonymous namespace `:18-38`). Keep new anonymous-namespace names collision-free (`port_patterns.md`, "Unity-build hygiene"). The CB-index constants in `deepseek_moe_gate_program_descriptor_builder.hpp:15-20` exist only so both entry points share CB ids; shared `TensorParamName` / `DFBSpecName` constants are their natural successors. Keeping both file names avoids editing `CMakeLists.txt`, which `metalium-developers-infra` co-owns (`CMakeLists.txt:71-75`).
  - **`device/kernel_includes/**` is not part of the port.** It holds the op-local LLK and compute-API copies for WH and BH. They take `uint32_t` operand ids, which DFB tokens convert to implicitly, so nothing changes. The tt-llk test `tt_metal/tt-llk/tests/sources/deepseek_moe_gate_test.cpp:141-147` includes six of them by path. Do not move or rename them.
  - **Verification:**
    - `tests/ttnn/unit_tests/operations/experimental/deepseek/test_deepseek_moe_gate_program_cache.py` dispatches twice with two live allocations and asserts one cache entry. It directly tests the translated override's `tensor_args` refresh.
    - `models/common/tests/moe/test_tt_moe_gate.py`: the `n_group=8` configs route through this op.
    - The compute path is arch-specific (`kernel_includes/tt_metal/include/compute_kernel_api/deepseek_moe_gate.h:11-19` picks the BH or WH LLK copies), so verify on both WH and BH where possible.
