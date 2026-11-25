# Metal 2.0 Port Brief — `experimental/deepseek/moe/generalized_moe_gate`

> Audit cleared all gates. This is your actionable input; the full record is in `METAL2_PREPORT_AUDIT.md`.

**Gates cleared:** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓

**Recipe docs:** `1256b6608d9a 2026-11-25 docs(metal_2.0): import the op porting recipe from akertesz/op-porting-recipe @ c41f7f4ef0a` *(carry this line into the port report's Provenance section)*

One DeviceOperation (`GeneralizedMoeGateDeviceOperation`) and one factory (`GeneralizedMoeGateProgramFactory`). The factory builds one unified kernel source three times on `input_shard.grid` (`device/generalized_moe_gate_program_descriptor_builder.cpp:156-188`):
- reader: NCRISC, RISCV_1, `NOC::RISCV_0_default`
- writer: BRISC, RISCV_0, `NOC::RISCV_1_default`
- compute: TRISC

The program has ten CBs and **no runtime args of any kind**.

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`). Carry them forward:

- **Current concept:** `descriptor` (`create_descriptor`, `device/generalized_moe_gate_program_factory.cpp:35-38` → `build_moe_gate_program_descriptor`)
- **Op-owned tensors:** none
- **Target concept:** `CustomProgramSpecFactoryConcept`. Translate `override_runtime_arguments` at `device/generalized_moe_gate_program_factory.cpp:40-51`. Today it re-points all five tensor-backed CBs (`input`, `bias`, `output`, `input_indices`, `output_indices`) via `UpdateDynamicCircularBufferAddress`. The Metal 2.0 override returns `ProgramRunArgs` whose `tensor_args` carry a `TensorArgument` for **all five** `TensorParameter`s: no more, no fewer, and no kernel run args. The `coord` parameter is unused today; keep it unused.
- **Gate-cleared, confirmed absent:** a non-clearing `TensorParameter relaxation` (cell = `none`) · `get_dynamic_runtime_args`. Also absent, for the record: custom hash (default reflection hash; leave the device-op untouched) and pybound `create_descriptor` (the nanobind binds only the op function).

## Construct — to do

**Tensor bindings** (per binding): all five are **clean**. Each is a borrowed-memory CB, so it becomes a `TensorParameter` plus a `DataflowBufferSpec::borrowed_from`. There is no `TensorAccessor`, no `get_bank_base_address` bridge and no address arg.

| DFB (legacy CB) | `borrowed_from` | entry / count | Format metadata |
|---|---|---|---|
| `input` (0) | `input_tensor` | tile size × `num_blocks` | BFLOAT16, tensor tile |
| `bias` (1) | `bias_tensor` | tile size × `num_blocks` | BFLOAT16, tensor tile |
| `output` (2) | `output_tensor` | tile size × 1 | BFLOAT16, tensor tile |
| `input_indices` (3) | `input_indices_tensor` | tile size × `num_blocks` | UINT16, tensor tile |
| `output_indices` (4) | `output_indices_tensor` | tile size × 1 | UINT16, tensor tile |

- **Where the sizes come from.** Legacy uses `total_size = aligned_size_per_bank`, with `page_size` overridden to the tile size by `set_cb_page_size_for_tile` (`builder.cpp:29-37`). Every validated config is tile-divisible.
- **`num_blocks` must stay identical.** It is `(input_shard.shape[0]/32) * (input_shard.shape[1]/32)` (`builder.cpp:64`).
- **Output tensors come from `tensor_args`.** The two outputs are caller-preallocated and live in `tensor_args`, which is where the legacy override reads them (`program_factory.cpp:48,50`).

**TensorParameter relaxation:** `none`. Strict matching.

**TensorAccessor 3rd arg:** none.

**CB endpoints:**

- **1P+1C on CBs 0-4, in every config.** These are plain FIFO roles, no flag:
  - NCRISC is the PRODUCER of `input`, `bias` and `input_indices` (`setup_sharded_buffer`, `kernels/generalized_moe_gate_kernel.cpp:46-50`). TRISC is their CONSUMER.
  - TRISC is the PRODUCER of `output` and `output_indices`. BRISC is their CONSUMER (`wait_front` only, `unified_kernels/generalized_moe_gate.hpp:198-201`).
- **CBs 5-9 are conditional DFBs.** Make the `DataflowBufferSpec` for each of `run_scores` (5), `run_idx` (6), `run_bias` (7), `cb_tilize` (8) and `cb_tilize_idx` (9) **conditional on `num_blocks == 2`**:
  - Dead under `num_blocks == 1`: zero touchers, because all uses sit in the discarded `else` of `if constexpr (CTArgs::num_blocks == 1)`, `generalized_moe_gate.hpp:219/254-351`, and in `process_block_to_run`, `:127-184`.
  - Live under `num_blocks == 2`.
  - **Do not drop them.** When present, bind them as a compute **self-loop** (TRISC PRODUCER + CONSUMER).
  - Sizes: `num_blocks` tiles for 5, 6, 7 and 9; `2 * num_blocks` tiles for 8 (`builder.cpp:108-114`). Formats: bf16 from `input_tensor` for 5, 7 and 8; uint16 from `input_indices_tensor` for 6 and 9.
  - Use the conditional-bindings pattern in `port_patterns.md`: a host define with the same condition, plus an `#ifdef` gate on the kernel side (see Watch for).

**Compute hardware config:**
- `fpu_math_fidelity = HiFi4`
- `double_buffer_dest = false` (legacy `dst_full_sync_en = true`, `builder.cpp:154`; load-bearing for the multi-block combine)
- `sfpu_precision_mode = Precise` (legacy `math_approx_mode` default `false`)
- 16-bit DEST (legacy `fp32_dest_acc_en` default `false`)

**Named CTAs:** these carry over by name as `args::`:
- NCRISC: `moe_gate_num_blocks`, `moe_gate_is_active_core`
- BRISC: `moe_gate_is_active_core`
- TRISC: `moe_gate_eps`, `moe_gate_scaling_factor`, `moe_gate_enable_sigmoid`, `moe_gate_topk`, `moe_gate_softmax`, `moe_gate_ungrouped_top8`, `moe_gate_num_blocks`, `moe_gate_is_active_core`

`eps` and `scaling_factor` are float bit patterns in `uint32_t` (`builder.cpp:24-27,134-135`); keep them as such. The `*_cb` CTAs become DFB bindings.

## Watch for

- **CB endpoints (multi-binding):** none.
- **Cross-op / shared kernels:** none. The bound kernel `device/kernels/generalized_moe_gate_kernel.cpp` and its headers `device/unified_kernels/{generalized_moe_gate,kernel_utils,kernel_op_api}.hpp` are this op's **private copies**. No other factory binds or includes them. No `_metal2` fork exists or is needed, so convert in place. The same-named `kernel_utils.hpp` / `kernel_op_api.hpp` under `deepseek_moe_gate/device/unified_kernels/` and `models/demos/deepseek_v3_b1/unified_kernels/` are **separate copies** owned by those ops. Do not edit them, and do not repoint includes at them. Sunset list: none.
- **RTA varargs:** none. There are no RTAs, CRTAs or CTA varargs.
- **DFB declaration order is load-bearing.** `deepseek_compute_kernel_init<false>()` (kernel.cpp:76) runs `compute_kernel_hw_startup(0, 0, 0)` (`tt_metal/hw/inc/api/compute/experimental/deepseek_compute_kernel_hw_startup.h:35`), so device slot 0 must be the bf16 `input` DFB. Metal 2.0 gives each DFB the lowest free slot in `ProgramSpec::dataflow_buffers` order (`tt_metal/impl/metal2_host_api/program_spec/construction/resource/dfb.cpp:239-259`, `tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:2255-2275`). Declare `input, bias, output, input_indices, output_indices` first, in that order, then the conditional five. Leave a comment saying why. The startup header is framework code and out of scope.
- **Conditional DFB tokens: an `#ifdef` gate is required.** kernel.cpp:69-73 passes the five run/tilize CB CTAs to `ComputeCTArgs` unconditionally, inside non-template `kernel_main`. With the DFBs conditional, `dfb::<run/tilize names>` won't exist under `num_blocks == 1`, so those five template arguments need a preprocessor gate fed by the host define. The uses inside `Op<CTArgs, …>` depend on `CTArgs` and need no gate. Note that `ComputeCTArgs` declares **integer defaults** for exactly those parameters (`generalized_moe_gate.hpp:75-80`: `= 5 … = 9`). If you change the CB template parameters' type to carry `dfb::` tokens, decide how the `num_blocks == 1` instantiation fills them. The defaults are otherwise dead: kernel.cpp always passes every argument.
- **Buffer objects are built from integer NTTPs today.** These sites are all `CircularBuffer x(<uint32_t id>)`:
  - `generalized_moe_gate.hpp:128-132, 198-199, 216, 220-223, 255-261`
  - `kernel_utils.hpp:67` in `setup_sharded_buffer(uint32_t cb_id, …)`

  On NCRISC the ids are `constexpr std::uint32_t` locals (kernel.cpp:38-40), so by the constexpr rule in `cb_dfb_api_whitelist.md` they take the token form. Construct `DataflowBuffer` from tokens, not from the low-level `DataflowBuffer(uint16_t)` constructor. The many LLK calls that take a CB id (`copy_tile`, `pack_tile`, `tilize_*`, `transpose_*`, `pack_untilize_*`, `reconfig_*`, `compute_kernel_hw_startup(run_*, cb_tilize*)` at `:273,294`, the `generalized_moe_gate*` LLK API) accept a `dfb::` token directly through its constexpr `uint32_t` conversion. Pass handles straight through; don't build temporary wrappers.
- **Kernel includes.** `api/dataflow/circular_buffer.h` is included at `kernel_utils.hpp:11` and `generalized_moe_gate.hpp:11,31` (the TRISC branch too). It becomes `api/dataflow/dataflow_buffer.h`, and the kernel reads CTAs via `experimental/kernel_args.h` / `get_arg(args::…)`.
- **Leave the dead helpers in `kernel_utils.hpp` alone.** These are `linear_id_in_grid`, `get_split_half_core_info`, `semaphore_dec`, `sync_riscs_*`, `override_cb_rd_ptr`, `reconfig_cbs_for_mask`, `reconfig_cb_interfaces` and `get_local_cb_*`. Nothing in this op calls them. `reconfig_cbs_for_mask` (`:177-178`) uses `get_cb_tiles_received_ptr` / `get_cb_tiles_acked_ptr`, which have no DFB equivalent. Do not convert or work around them; convert only `setup_sharded_buffer`. If the user decides they must go first (audit Q1), that is a separate ops-side change, not port scope.
- **Override guard.** `update_tensor_cb`'s "Missing globally allocated circular buffer" `TT_FATAL` (`program_factory.cpp:29`) searches `program.circular_buffers()`. The Metal 2.0 override gets no `Program`, so that guard becomes structural, from `borrowed_from` naming a declared `TensorParameter`. Keep the null-buffer check if you keep a helper. Record the dropped guard in the port report.
- **DFB placement narrows, harmlessly.** Legacy tensor CBs span each tensor's own shard grid (`ttnn/core/tensor/tensor_utils.cpp:57`), which validation lets be a superset of the input grid (`device_operation.cpp:101,110,115`). DFB placement is derived from the bound kernels, so it covers the input grid only. No kernel runs on the extra cores.
- **Verification hooks:**
  - `tests/ttnn/unit_tests/operations/experimental/deepseek/test_generalized_moe_gate_program_cache.py`: cache reuse/miss; exercises the override.
  - `models/common/tests/moe/test_generalized_moe_gate.py`: 256-expert, `test_…_512_global` (the `num_blocks == 2` path, which exercises the conditional DFBs) and `test_…_grouped`.
  - Model users: `models/common/moe/tt_moe_gate.py` and `models/demos/gemma4_26b_a4b_qb2/tt/optimized_decoder.py`.
  - Hit both `num_blocks` values, and both on cache hit with tensors at new addresses.
- **Do not** read or borrow from `ttnn/cpp/ttnn/operations/experimental/quasar/`.
