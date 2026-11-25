# Metal 2.0 Audit Findings — `experimental/deepseek/moe/generalized_moe_gate`

- **`GeneralizedMoeGateDeviceOperation`** (`device/generalized_moe_gate_device_operation.hpp:18`, `program_factory_t = std::variant<program::GeneralizedMoeGateProgramFactory>` at `:26`)
  - `GeneralizedMoeGateProgramFactory` (`device/generalized_moe_gate_program_factory.{hpp,cpp}`). `create_descriptor` (`program_factory.cpp:35-38`) delegates to `build_moe_gate_program_descriptor` in `device/generalized_moe_gate_program_descriptor_builder.cpp:41-209`.
- **Kernel source:** one unified kernel, `device/kernels/generalized_moe_gate_kernel.cpp`. The builder instantiates it three times over the same core range (`input_shard.grid`): reader on RISCV_1 (NCRISC, `builder.cpp:156-167`), writer on RISCV_0 (BRISC, `:169-180`), and compute (TRISC, `:182-188`). It includes the op-local headers `device/unified_kernels/{kernel_op_api.hpp, kernel_utils.hpp, generalized_moe_gate.hpp}`.
- **Configs that change the program:** `num_blocks` (1 for ≤256 experts, also the only value allowed in grouped mode; 2 for the 512-expert combine, `device_operation.cpp:134-140`, `builder.cpp:64`). Also the named CTAs `topk`, `output_softmax`, `grouped`, `enable_sigmoid`, `eps` and `scaling_factor`. Only `num_blocks` changes which CBs are live.
- No unreferenced kernel files in the op directory.

**Scope:** TTNN op, Gen1 (WH/BH) target — within scope of `audit/metal2_audit.md`.

**Recipe docs:** `1256b6608d9a 2026-11-25 docs(metal_2.0): import the op porting recipe from akertesz/op-porting-recipe @ c41f7f4ef0a`

**Readiness sheet:** a header + one-row extract of the live "Operations analysis" sheet (row 167), fetched 2026-11-25 by the main session and saved to `analyses/ttnn_op_porting_readiness.csv`. This audit treats that extract as the fresh fetch. Per the recipe, the subagent did not fetch the sheet itself.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/deepseek/moe/generalized_moe_gate` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `GeneralizedMoeGateDeviceOperation` → `GeneralizedMoeGateProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | Yes. All data movement in use goes through `CircularBuffer` wrapper methods. There are no NoC calls, address generators or semaphores. Some never-called helpers in `kernel_utils.hpp` use Device 1.0 idioms; they are not exercised (see Gate detail). |
| *Prereqs* — Cross-op escapes | Ok. The only out-of-directory includes are `tt_metal/*` framework/LLK headers. No kernel file is borrowed or lent. |
| *Feature Support* — overall | GREEN (all Appendix A rows N/A) |
| *Feature Support* — Variadic-CTA | Ok. All CTAs are named; there are no CTA varargs. |
| *TTNN Readiness* — `Is able to port?` (the gate) | Yes (cross-check clean) |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Execution Model | `SPMD` (not mesh-shaped, so `Secretly SPMD Workload?` is N/A; the cell is blank) |
| *TTNN Readiness* — Custom hash | No. The default reflection hash is used. A custom hash was removed by #49870; there is no backdoor `attribute_values` / `to_hash`. |
| *TTNN Readiness* — `get_dynamic_runtime_args` | No |
| *TTNN Readiness* — `override_runtime_arguments` | Yes (not a gate; the porter translates it): `GeneralizedMoeGateProgramFactory::override_runtime_arguments`, `device/generalized_moe_gate_program_factory.cpp:40-51` |
| *TTNN Readiness* — Pybind `create_descriptor` | No. `generalized_moe_gate_nanobind.cpp:13` binds only the `ttnn.experimental.deepseek.moe.generalized_moe_gate` function. |
| *TTNN Readiness* — Op-owned tensors | No |
| *TTNN Readiness* — Target concept | `CustomProgramSpecFactoryConcept` |
| *Port work* — Offset base pointer | none (the op has no runtime args at all) |
| *Port work* — Tensor bindings (per binding) | all 5 **clean** (borrowed-memory DFBs) |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) |
| *Port work* — TensorAccessor 3rd arg | none. No `TensorAccessor` exists in the op. |
| *Port work* — CB endpoints | 5 legal 1P+1C (tensor-backed CBs 0-4). CBs 5-9 are a **conditional DFB**: self-loop on compute when `num_blocks == 2`, zero touchers when `num_blocks == 1`. |

## Result

**GREEN → brief issued** (`METAL2_PORT_BRIEF.md`). Every gate-bearing subject clears:
- Device 2.0
- Feature compatibility
- TTNN factory concept (`Is able to port?` = `yes`, cross-check clean)
- Offset base pointers
- TensorAccessor 3rd argument

The target is `CustomProgramSpecFactoryConcept`. The override translates into a `ProgramRunArgs` that refreshes all five tensor bindings.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** GREEN. The sheet's `Is able to port?` is `yes`. `Op Classification` = `PD Op (custom)`, `Diego validation` = `yes`, and `Known op issues` is empty. The primary-column cross-check is clean:

  | Sheet column | Sheet | Code evidence | Match |
  |---|---|---|---|
  | `Factory (variant)` (factory-set match) | `GeneralizedMoeGateProgramFactory` (1 row) | `device_operation.hpp:26`: a variant with exactly that one factory | ✓ |
  | `Concept` | `descriptor` | `create_descriptor(attrs, tensor_args, tensor_return_value)` returns a `ProgramDescriptor` and takes no `MeshCoordinate` (`program_factory.hpp:18-21`). No `create_workload_descriptor` / `create_mesh_workload` / `create()`. | ✓ |
  | `Custom hash` | `no` | no `compute_program_hash` in the op | ✓ |
  | `Backdoor custom hash` | `no` | no `attribute_values` / `to_hash` | ✓ |
  | `Runtime-args update (get_dynamic_runtime_args)` | `no` | absent from the device-op | ✓ |
  | `Override runtime args method?` | `yes` | `program_factory.hpp:24-29`, `program_factory.cpp:40-51` (PD op → target-concept signal) | ✓ |
  | `Pybind descriptor` | `no` | the nanobind file binds only the function (`generalized_moe_gate_nanobind.cpp:13`) | ✓ |
  | `Smuggled pointer` | `no` | the builder sets no runtime args or CRTAs (`builder.cpp:156-188`); tensor addresses reach kernels only as CB backing (`cb_descriptor_from_sharded_tensor`) | ✓ |
  | `Op-owned tensors?` | `no` | `descriptor` concept, no `buffers` | ✓ |
  | `Secretly SPMD Workload?` / `Why secretly SPMD?` | blank | not mesh-shaped, so N/A | ✓ |
  | `TensorParameter relaxation` | `none` | clears | — |

  Cross-column invariants hold: `get_dynamic_runtime_args` is `no`, and `Op-owned tensors?` is `no` on a `descriptor` concept. `Factory definition path` and `Declared in` name the files above.

- **Device 2.0 (every kernel used):** GREEN. The single kernel source is instantiated three ways. Only code reachable from `kernel_main` is exercised:
  - **NCRISC:** `unified_kernels::setup_sharded_buffer` (`kernels/generalized_moe_gate_kernel.cpp:47-49` → `unified_kernels/kernel_utils.hpp:66-70`) uses `CircularBuffer cb(cb_id); cb.reserve_back(n); cb.push_back(n);`.
  - **BRISC:** `CircularBuffer::wait_front` on the two output CBs (`unified_kernels/generalized_moe_gate.hpp:198-201`).
  - **TRISC:** `CircularBuffer` wrapper FIFO ops plus compute LLK calls keyed by CB id (`copy_tile`, `pack_tile`, `tilize_*`, `transpose_*`, `pack_untilize_*`, `reconfig_*`, the `generalized_moe_gate*` LLK API from `tt_metal/hw/inc/api/compute/experimental/generalized_moe_gate.h`). These are compute APIs, outside Device 2.0's data-movement scope.

  There is no `noc_async_*`, no `InterleavedAddrGen*` / `ShardedAddrGen`, no raw semaphore address and no `get_read_ptr(cb_id)` / `get_write_ptr(cb_id)` holdover.

  **Never-called helpers (not exercised, so not gating):** `unified_kernels/kernel_utils.hpp` also defines helpers that no code path of this op calls (grep of the kernel and op header: only `setup_sharded_buffer` is used):
  - `semaphore_dec` (`:73-75`, raw L1 semaphore pointer)
  - `sync_riscs_enter` / `sync_riscs_exit` (`:95-117`, raw L1 semaphore pointer)
  - `reconfig_cbs_for_mask` / `reconfig_cb_interfaces` (`:149-220`). These write `LocalCBInterface` fields, and `reconfig_cbs_for_mask` calls `get_cb_tiles_received_ptr` / `get_cb_tiles_acked_ptr` at `:177-178`. The CB-DFB whitelist lists those two as having *no DFB equivalent*.
  - `override_cb_rd_ptr`, `get_local_cb_rd_ptr`, `get_local_cb_page_size`, `update_local_cb_rd_ptr` (`:126-128`, `:224-237`)
  - `linear_id_in_grid`, `get_split_half_core_info` (`:30-55`)

  They are dead code in the op's **own** header. No other op includes this copy; see Out-of-directory coupling. They do not gate, but they are recorded as an anomaly, a heads-up and a question (Q1).

- **Feature compatibility:** GREEN. No gate fired.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No GCB type, `remote_index` or `CreateCircularBuffer(…, global_cb)`. All five tensor CBs come from `cb_descriptor_from_sharded_tensor`, which sets `.global_circular_buffer = nullptr` (`ttnn/core/tensor/tensor_utils.cpp:65`). The five intermediate CBs are plain `CBDescriptor`s (`builder.cpp:96-107`). |
  | CBDescriptor `address_offset` (non-zero) | N/A | `cb_descriptor_from_sharded_tensor(cb, tensor)` is called with two arguments (`builder.cpp:83-87`), so `address_offset` defaults to `0` (`ttnn/api/ttnn/tensor/tensor_utils.hpp:76`). The intermediate CBs never set it. The offset-base-pointer triage (2026-07-19, Type 3 note) also lists this builder as "offset 0 (benign; no action)". |
  | GlobalSemaphore | N/A | No `GlobalSemaphore` anywhere, and no semaphores of any kind. |

- **CB endpoints (GATE-free):** dispositions below. Nothing blocks.
  - CBs 0-4 are legal 1P+1C in every config.
  - CBs 5-9 (`run_scores`, `run_idx`, `run_bias`, `cb_tilize`, `cb_tilize_idx`) are a **conditional DFB**: dead when `num_blocks == 1`, self-loop on compute when `num_blocks == 2`.

  Census (all three kernels cover every node of `input_shard.grid`, so every node has the same census):

  | CB (index → role) | Backing | `num_blocks == 1` | `num_blocks == 2` | Disposition |
  |---|---|---|---|---|
  | 0 `input` | borrowed: `input_tensor` (`builder.cpp:83`) | NCRISC P (`reserve_back`/`push_back(num_blocks)`, kernel.cpp:47) · TRISC C (`wait_front`/`pop_front`, `generalized_moe_gate.hpp:231,242`) | same; TRISC pops one per block (`:134,182`) | **1P+1C** (legal) |
  | 1 `bias` | borrowed: `bias_tensor` (`:84`) | NCRISC P (kernel.cpp:48) · TRISC C (`:225`; pops `:183` in multi-block) | same | **1P+1C** (legal). Note: the single-block path `wait_front`s but never `pop_front`s bias (`:225`). That is harmless on Gen1, and the port keeps it. |
  | 2 `output` | borrowed: `output_tensor` (`:85`) | TRISC P (`reserve_back`/`push_back`, `:245,249` / `:341,346`) · BRISC C (`wait_front(1)`, no pop, `:201`) | same | **1P+1C** (legal) |
  | 3 `input_indices` | borrowed: `input_indices_tensor` (`:86`) | NCRISC P (kernel.cpp:49) · TRISC C (`wait_front(num_blocks)`, never popped, `:217`; `copy_tile(…, b, …)` `:141,228`) | same | **1P+1C** (legal) |
  | 4 `output_indices` | borrowed: `output_indices_tensor` (`:87`) | TRISC P (`:246,252` / `:342,349`) · BRISC C (`wait_front(1)`, `:200`) | same | **1P+1C** (legal) |
  | 5 `run_scores`, 6 `run_idx`, 7 `run_bias` | L1, `num_blocks` tiles (`builder.cpp:108-110`) | **0 touchers** | TRISC only (P `:156-180`, C `:274-303`) | **conditional DFB**: dead under `num_blocks == 1`, self-loop under `num_blocks == 2` |
  | 8 `cb_tilize`, 9 `cb_tilize_idx` | L1, `2*num_blocks` / `num_blocks` tiles (`:113-114`) | **0 touchers** | TRISC only (P `:278-300`, C `:306-307,339-340`) | **conditional DFB**: dead under `num_blocks == 1`, self-loop under `num_blocks == 2` |

  - **Hunt for the hidden second writer:** none found. No kernel uses `get_write_ptr` / `get_read_ptr` / `fifo_*_ptr` on any CB in an exercised path. `setup_sharded_buffer` says it "can be called from either NCRISC or BRISC", but here only NCRISC calls it (kernel.cpp:46-50).
  - **Why CBs 5-9 are dead under `num_blocks == 1`:** their host indices are fixed constants (`builder.cpp:73-81`). They reach a kernel only through the TRISC named CTAs `moe_gate_run_scores_cb` … `moe_gate_cb_tilize_idx` (`builder.cpp:141-145` → kernel.cpp:69-73 → `ComputeCTArgs` fields). Every *use* of those fields sits in the `else` branch of `if constexpr (CTArgs::num_blocks == 1)` (`generalized_moe_gate.hpp:219/254-351`) or in `process_block_to_run<b>` (`:127-184`), which is instantiated only from that branch (`:269-270`). Both are templates on `CTArgs`, so with `num_blocks == 1` the branch is discarded and nothing touches CBs 5-9. NCRISC and BRISC never receive those indices. No index arithmetic or aliasing reaches 5-9. The LLK API headers take CB ids only as parameters, with no hardcoded indices. The ComputeCTArgs defaults `5..9` (`:76-80`) mirror the host constants but are always overridden by kernel.cpp:69-73.
  - **Legacy behaviour:** the legacy factory allocates CBs 5-9 unconditionally (1 tile each, plus 2 for `cb_tilize`, when `num_blocks == 1`). Making them conditional is the recipe's disposition and changes only L1 footprint. See the heads-up on the kernel-side consequence.

- **Offset base pointers:** GREEN. The op passes **no runtime args and no common runtime args** to any kernel (`builder.cpp:156-188` sets only named CTAs). Tensor bases reach the device only as CB backing (`CBDescriptor::buffer`). There is no `->address()` anywhere in the op, so no fold is possible. Type 3 is N/A (see Feature compatibility). The triage doc (2026-07-19) does not list this op in Types 1/2 and lists it as benign offset 0 under Type 3, which agrees with this scan.
- **TensorAccessor 3rd argument:** N/A. No accessor in the op passes a 3rd argument, because the op constructs no `TensorAccessor` at all. The 3rd-arg triage (2026-07-06) has no entry for the op, which agrees.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding): `input_tensor`, `bias_tensor`, `input_indices_tensor`, `output_tensor`, `output_indices_tensor` are all **clean**. Each is a borrowed-memory DFB read or write through its CB, and becomes a `TensorParameter` plus a `DataflowBufferSpec::borrowed_from`. No Case 1 / Case 2 work.
- **TensorParameter relaxation:** none.
- **TensorAccessor 3rd arg:** none.
- **CB endpoints:**
  - 1P+1C on CBs 0-4 in every config. These are ordinary FIFO roles:
    - NCRISC produces `input`, `bias` and `input_indices`; TRISC consumes them.
    - TRISC produces `output` and `output_indices`; BRISC consumes them.
  - Conditional DFB on CBs 5-9: bind them, as compute self-loops, only when `num_blocks == 2`. They are dead when `num_blocks == 1`.
  - No dead-everywhere CB and no multi-binding.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none.
- **Cross-op / shared kernels:** none. The kernel and its three headers are op-private copies bound only by this factory; no `_metal2` fork exists or is needed. Same-named files in `deepseek_moe_gate/device/unified_kernels/` (and in `models/demos/deepseek_v3_b1/unified_kernels/`) are **separate private copies**, not shared files. Convert in place.
- **RTA varargs:** none. The op has no RTAs or CRTAs, and no CTA varargs.
- **DFB declaration order is load-bearing.** `deepseek_compute_kernel_init<false>()` (kernel.cpp:76) calls `compute_kernel_hw_startup(0, 0, 0)` (`tt_metal/hw/inc/api/compute/experimental/deepseek_compute_kernel_hw_startup.h:35`), which configures unpack/math/pack from slot 0 = `input`. Metal 2.0 assigns each DFB the lowest free device slot in `ProgramSpec::dataflow_buffers` order (`tt_metal/impl/metal2_host_api/program_spec/construction/resource/dfb.cpp:239-259`, `tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:2255-2275`). So declare the DFBs in legacy CB-index order — `input, bias, output, input_indices, output_indices`, then the conditional five — to land `input` on slot 0. The header is framework (`tt_metal/hw/inc`), so it is out of port scope. Do not change it.
- **Conditional DFBs need a kernel-side preprocessor gate.** `dfb::<name>` tokens exist only for bound DFBs. kernel.cpp:69-73 names the five run/tilize CTAs unconditionally, inside non-template `kernel_main`, as `ComputeCTArgs` template arguments. With the conditional DFB, those references must sit behind an `#ifdef` driven by a host define that shares the `num_blocks > 1` condition, per the conditional-bindings pattern in `port_patterns.md` (promote a CTA gate to a define). The uses inside `Op<CTArgs,…>` (`generalized_moe_gate.hpp:127-184`, `:254-351`) are dependent on `CTArgs` and already discarded by `if constexpr`. `ComputeCTArgs` gives those five parameters integer defaults (`:76-80`). If the porter makes the CB parameters token-typed, it must handle those defaults for the `num_blocks == 1` instantiation.
- **CB objects are built from integer template parameters.** Every kernel-side buffer object is `CircularBuffer x(CTArgs::<cb>)`, where `<cb>` is a `std::uint32_t` NTTP:
  - `generalized_moe_gate.hpp:128-132, 198-199, 216, 220-223, 255-261`
  - `kernel_utils.hpp:67` (`setup_sharded_buffer(uint32_t cb_id, …)`)
  - On NCRISC, the ids are `constexpr std::uint32_t` locals (kernel.cpp:38-40), which by the whitelist's constexpr rule take the token form.

  The LLK calls take the id, and a `dfb::` token converts to `uint32_t` implicitly at constexpr time (Gen1-only). The `DataflowBuffer` constructions should take tokens, not the low-level `DataflowBuffer(uint16_t)` constructor.
- **The override's `program.circular_buffers()` lookup has no Metal 2.0 counterpart.** `update_tensor_cb` (`program_factory.cpp:21-31`) finds each globally-allocated CB by index and `TT_FATAL`s on a null buffer or a missing CB. The Metal 2.0 override receives no `Program`. It returns a `TensorArgument` for each of the five tensors, which is exactly the set the legacy override refreshed (`:46-50`). The missing-CB guard becomes structural; the null-buffer guard can stay.
- **Output tensors arrive through `tensor_args`.** The op writes into caller-preallocated `output_tensor` / `output_indices_tensor`. `create_output_tensors` returns them as-is (`device_operation.cpp:151-154`), and the legacy override reads them from `tensor_args` (`program_factory.cpp:48,50`). Keep that source.
- **CB placement differs, harmlessly.** Each tensor CB's legacy `core_ranges` is that tensor's own shard grid (`tensor_utils.cpp:57`). Validation allows bias, input-indices and output grids to be **supersets** of the input grid (`device_operation.cpp:101,110,115`), and the kernels run only on the input grid. A Metal 2.0 DFB's placement is derived from its bound kernels, so it covers the input grid only. No kernel runs on the extra cores, so behaviour is unchanged. The borrowed-storage invariant (tensor storage on every DFB node) holds because of those `contains` checks.
- **DFB sizing.** Legacy tensor CBs take `total_size = buffer->aligned_size_per_bank()` and `page_size = tile_size` (`builder.cpp:29-37`). Every validated config is tile-divisible (32x32 tiles, bf16/uint16, tile-aligned input shard, 32x32 output shard, TILE or ROW_MAJOR output), so `entry_size = tile_size` and `num_entries` = shard tiles:
  - `num_blocks` for input, bias and input_indices
  - 1 for the two outputs

  Every DFB is bound to compute, so each needs `data_format_metadata` (BFLOAT16 / UINT16 from the tensor dtype) and `tile_format_metadata` (the tensor's tile).
- **Compute config:** `HiFi4`, `dst_full_sync_en = true` (`builder.cpp:149-154`), other fields at their defaults. This maps to `fpu_math_fidelity = HiFi4`, `double_buffer_dest = false` and `sfpu_precision_mode = Precise` (legacy `math_approx_mode = false`). Full-sync DEST is load-bearing: the multi-block combine keeps block 0's run in DEST across separate acquires (comment at `builder.cpp:151-153`).
- **Float CTAs.** `eps` and `scaling_factor` travel as `std::bit_cast<uint32_t>` named CTAs (`builder.cpp:24-27,134-135`) and are consumed as NTTPs. Keep them as compile-time `uint32_t` args.
- **Dead helpers in `kernel_utils.hpp`.** These include the `get_cb_tiles_received_ptr` / `acked` user at `:177-178` (see Device 2.0 above). The op never calls them, so the port has no reason to convert them, and changing unexercised code is outside a zero-functional-change port. Converting only the one used helper, `setup_sharded_buffer`, should leave the rest compiling. As a fact, not a pattern: the same helpers compile today in `deepseek_moe_gate`'s separate copy of `kernel_utils.hpp`, which already includes `api/dataflow/dataflow_buffer.h`.

## Team-only

- **Out-of-directory coupling & donor shape:** **✓ clean.**
  - Function-call escapes: every out-of-directory `#include` is `tt_metal/*` (LLK/HAL/framework, class 1, no concern):
    - `api/dataflow/dataflow_api.h`
    - `api/dataflow/circular_buffer.h`
    - `api/debug/kernel_profiler.hpp`
    - `api/compute/*` (incl. `tilize.h`, `transpose.h`, `pack_untilize.h`, `compute_kernel_hw_startup.h`)
    - `api/compute/experimental/generalized_moe_gate.h`
    - `api/compute/experimental/deepseek_compute_kernel_hw_startup.h`

    The two `experimental/` compute headers are also used by `models/demos/deepseek_v3_b1/**` and `api/compute/experimental/sdpa.h`. They are framework API, not op-owned, so they are excluded from the shared-kernel convention.

    | Op kernel | Donor file | Class | Status |
    |---|---|---|---|
    | `generalized_moe_gate_kernel.cpp` (+ op-local `unified_kernels/*.hpp`) | `tt_metal/hw/inc/api/...` headers above | 1 (`tt_metal/*`) | ✓ |

  - **Borrowed kernel files:** none. The factory binds only its own `device/kernels/generalized_moe_gate_kernel.cpp` (`builder.cpp:20-22`).
  - **Shared-kernel census.** I grepped the filenames across `ttnn/cpp/ttnn/operations/` and `models/` (quasar excluded):
    - `generalized_moe_gate_kernel.cpp`: one hit, this op's builder.
    - `unified_kernels/generalized_moe_gate.hpp`: included only by this op's kernel.
    - `kernel_utils.hpp` / `kernel_op_api.hpp`: every other hit resolves, through a relative `../unified_kernels/…` include, to that op's or model's **own same-named private copy**. Those are `experimental/deepseek/moe/deepseek_moe_gate/device/unified_kernels/` and `models/demos/deepseek_v3_b1/unified_kernels/`. `deepseek_prefill/csa_compressor` hits a differently named `csa_compressor_kernel_utils.hpp`.
    - `moe_gate_mm` includes none of these files.

    This op's copies are therefore **not lent**, and nothing is borrowed. Notes:
    - PR #63510 (merged 2026-11-03, merge `38e5ef987aa`) changed only seven files, all under `deepseek_moe_gate/`. It converted *that op's* private `kernel_utils.hpp` in place.
    - Today the two `kernel_utils.hpp` copies differ only in `setup_sharded_buffer`'s signature (DFB token vs `uint32_t`) and two includes. The two `kernel_op_api.hpp` copies are byte-identical.
    - **No `_metal2` fork exists** beside any of this op's kernel files (`ls device/kernels/`, `device/unified_kernels/`), and none is needed. Rung: convert in place (no other binder).
  - The by-shape donor function-call analysis is all class 1, so there is no per-call detail.
- **Relaxation candidates:** none. There is no custom hash to mine.
- **TTNN factory analysis:**

  ### TTNN ProgramFactory

  #### Concept
  `CustomProgramSpecFactoryConcept` (the sheet's `Porting Target`). The override to translate is `GeneralizedMoeGateProgramFactory::override_runtime_arguments`, `device/generalized_moe_gate_program_factory.cpp:40-51`. It is declared at `program_factory.hpp:24-29` with a defaulted, unused `std::optional<ttnn::MeshCoordinate>` parameter. Today it calls `UpdateDynamicCircularBufferAddress` for all five tensor-backed CBs. Translated, it returns `ProgramRunArgs::tensor_args` for all five `TensorParameter`s and no kernel run args.

  #### Fit
  - Execution model: `SPMD`.
  - Op-owned device resources: none.
  - Tensor-arg matching: strict (default).
  - Ported-from shape: 1:1 (`descriptor` → one `ProgramSpec`).

  #### Custom compute_program_hash
  None: the default reflection-based hash over `operation_attributes_t` + `tensor_args_t`. The custom hash was dropped by #49870. There is no backdoor.

  #### Stop signals
  None.

  Other facts:
  - Pybind `create_descriptor`: none.
  - `get_dynamic_runtime_args`: none.
  - Other risky pybind: none.

## Misc anomalies  *(team-only, non-gating)*

- **Dead helpers in `device/unified_kernels/kernel_utils.hpp`.** These are `linear_id_in_grid`, `get_split_half_core_info`, `semaphore_dec`, `sync_riscs_enter/exit`, `override_cb_rd_ptr`, `reconfig_cbs_for_mask`, `reconfig_cb_interfaces`, `get_local_cb_rd_ptr`, `get_local_cb_page_size` and `update_local_cb_rd_ptr` (`:30-55, 73-75, 95-237`). None is called by this op. `reconfig_cbs_for_mask` uses `get_cb_tiles_received_ptr` / `get_cb_tiles_acked_ptr` (`:177-178`), which have no DFB equivalent. Because `reconfig_cb_interfaces` is a non-template inline, that body is still compiled on every NCRISC/BRISC/unpack/pack build. The file appears to be a copy of the `deepseek_v3_b1` unified-kernel helper; deleting the unused helpers would remove the only unportable API from the op's tree.
- **`moe_gate_is_active_core` is always 1** (`builder.cpp:121,126,146`). The `Core::is_active_core` / `Op<…, IsActiveCore>` gating (kernel.cpp:26-28,46,82; `generalized_moe_gate.hpp:111-118`) is constant plumbing with no inactive variant.
- **`ComputeCTArgs` duplicates the host's magic CB ids.** Its template defaults `run_scores_cb_ = 5 … cb_tilize_idx_ = 9`, `topk_ = 8`, `output_softmax_ = 0` (`generalized_moe_gate.hpp:75-82`) repeat the host constants in `builder.cpp:73-81`. kernel.cpp:58-75 always passes all of them explicitly, so the defaults are dead but would silently diverge if the host indices changed.
- **The bias tile is never popped on the single-block path** (`generalized_moe_gate.hpp:225`; contrast `:183` on the multi-block path). It is benign, because the CB is borrowed and re-pushed each dispatch, but it is asymmetric.
- **`set_cb_page_size_for_tile`'s fallback looks unreachable.** `fd.page_size = cb_desc.total_size` (`builder.cpp:36`) appears unreachable under current validation: every validated shard is a multiple of the tile size.
- **Possible packaging gap (unverified).** The `kernels` `FILE_SET` globs only `device/kernels/*.{cpp,hpp}` (`CMakeLists.txt:23-27`). The kernel includes `../unified_kernels/*.hpp`, which that glob does not cover. If no other mechanism ships `device/unified_kernels/`, an installed (non-source-tree) runtime could fail to JIT this kernel. I did not chase this; `deepseek_moe_gate` has the same glob.

## Questions for the user

1. **Should the dead `get_cb_tiles_*_ptr` helper go to the Device 2.0 / ops track before the port, or ride along untouched?** `kernel_utils.hpp:149-220` (`reconfig_cbs_for_mask` / `reconfig_cb_interfaces`) uses `get_cb_tiles_received_ptr` / `get_cb_tiles_acked_ptr`. `cb_dfb_api_whitelist.md` says a kernel that *uses* these "cannot be ported today". This op never calls the helper; it is only compiled. So I judged it not exercised and non-gating, consistent with the recipe's "every kernel this op exercises" scope. If you'd rather not ship a Metal 2.0 kernel whose included header still carries the API, an ops-side PR deleting the unused helpers (about 190 lines, no callers) is a cheap prerequisite. Otherwise the port leaves them as they are.

## Recipe notes

- **Header-path drift.**
  - Appendix A's GlobalSemaphore entry cites `tt_metal/api/tt-metalium/experimental/metal2_host_api/kernel_spec.hpp`. That file now lives at `tt_metal/api/tt-metalium/workload/specification/kernel_spec.hpp` (namespace `tt::tt_metal`). Only `advanced_options.hpp`, `prefetcher_pipe_parameter.hpp` and `sram_arena_parameter.hpp` remain under `experimental/metal2_host_api/`.
  - The `KernelSpec` there now has a `GlobalSemaphoreBinding` / `global_semaphore_bindings` member, and `workload/specification/global_semaphore_parameter.hpp` exists. So the entry's stated evidence ("listed under *coming soon*") no longer matches the header.
  - Per the maintenance rule I treated Appendix A as definitive. It had no effect here, because the op uses no semaphores.
  - The run-args headers are at `workload/run_args/` (`program_run_args.hpp`, `validation_mode.hpp`).
- **Dead code inside an exercised header is not covered.** The Device 2.0 gate covers "every kernel this op exercises", and Scope makes *unreferenced kernel files* out of scope. Neither says what to do with **never-called functions inside a referenced header**. Here that matters because one of them uses an API the CB-DFB whitelist calls unportable. I judged uncalled = not exercised. A one-line rule would settle it: "uncalled helpers in an included header are not exercised; record them as anomalies".
- **Tension between the whitelist and Appendix A.** `cb_dfb_api_whitelist.md:165` says a kernel using `get_cb_tiles_acked_ptr` / `get_cb_tiles_received_ptr` "cannot be ported today". That reads as an unsupported feature, but it is absent from Appendix A, which the audit calls authoritative ("a construct not listed there is supported"). An auditor who finds a *called* use has two contradictory instructions. Suggest either an Appendix A entry or a Device 2.0 sanctioned/unsanctioned note.
- **The dead-CB grep heuristic false-fires.** The recognition says "grep the bound kernels for the index and any named CTA carrying it; zero hits". Here the named CTAs carrying CBs 5-9 *are* read unconditionally (kernel.cpp:69-73), yet the CBs have zero touchers in the `num_blocks == 1` config, because every use sits in a discarded `if constexpr` branch of a class template. The "touch" definition (an actual FIFO or pointer access) is the right test, and the grep heuristic would wrongly report "live". Worth a sentence: a CTA that only carries the index into a template, without an access, is not a toucher.
- **Conditional DFBs plus defaulted NTTPs.** The conditional-bindings pattern in `port_patterns.md` covers `#ifdef`-gating non-dependent `dfb::` references. It does not anticipate CB ids that flow into a **class-template NTTP list with integer defaults**, where a token-typed parameter can't keep the default. That is a likely shape in other unified-kernel ops, the deepseek ones among them.
