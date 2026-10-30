# Metal 2.0 Audit Findings — `ttnn/cpp/ttnn/operations/experimental/deepseek/moe/deepseek_moe_gate`

- **`DeepseekMoeGateDeviceOperation`** (`device/deepseek_moe_gate_device_operation.hpp:18`; `program_factory_t = std::variant<program::DeepseekMoeGateProgramFactory>` at `:25`)
  - `DeepseekMoeGateProgramFactory`: entry points in `device/deepseek_moe_gate_program_factory.{hpp,cpp}` (`create_descriptor` and `override_runtime_arguments`). The descriptor body is in `device/deepseek_moe_gate_program_descriptor_builder.{hpp,cpp}` (`build_moe_gate_program_descriptor`), called from `deepseek_moe_gate_program_factory.cpp:36`.
- **Kernels.** The op owns one *unified* kernel source, `device/kernels/deepseek_moe_gate_kernel.cpp`. The one factory instantiates it three times (`deepseek_moe_gate_program_descriptor_builder.cpp:137-169`):
  - reader on NCRISC (`RISCV_1`, NOC 0)
  - writer on BRISC (`RISCV_0`, NOC 1)
  - compute on TRISC (`HiFi4`, every other compute field at its default)

  All three run over the input tensor's shard grid. The kernel includes only op-local headers: `device/unified_kernels/{kernel_op_api,kernel_utils,deepseek_moe_gate}.hpp`, plus the op-local LLK and compute-API copies under `device/kernel_includes/**` (WH and BH).
- **Config axes.** There is one: `enable_sigmoid`, a compute CTA that selects a compile-time branch. The factory has no branching and no sharding-mode variants: all five tensors must be sharded, with 32×32 shards and 32×32 tiles (`device/deepseek_moe_gate_device_operation.cpp:47-51, 96-99`).
- **Unreferenced files:** none. Every file under `device/` is reached from the one kernel or the factory. `device/unified_kernels/kernel_utils.hpp` does define about 12 helper functions that this op never calls. See the Device 2.0 detail and Misc anomalies.

*Path convention:* paths are op-relative unless they start with `tt_metal/`, `ttnn/`, `models/`, `tests/` or `docs/`.

**Scope:** TTNN op, Gen1 (WH/BH) target, within the scope of `audit/metal2_audit.md`.

**Recipe docs:** ``c41f7f4ef0a 2026-10-29 docs(metal_2.0): DFBAccessor -> DFBBindingToken; drop the dead `.id` form``

**Code audited:** worktree branch `iwrosz/deepseek_moe_gate_2.0` at `ec310e0d286`. That is `origin/akertesz/op-porting-recipe` (`c41f7f4ef0a`) with `origin/main` (`012c19beaae`, 2026-10-30) merged in. The last commit touching the op directory is `27cad27903e 2026-10-10 [Cleanup] Drop bare tt_metal from the JIT include path list (#60172)`.

**Readiness sheet:** Diego's "Operations analysis" sheet (file ID `1KUMj8SyBGlNMZlLFgs1MbAZlO2g6EoUc4KaxSlcy8jw`). The launching session fetched it at **2026-10-30 12:21 UTC** per `ttnn_op_porting_readiness.md` (`download_file_content` as `text/csv`, base64-decoded) into `docs/.../metal_2.0/analyses/ttnn_op_porting_readiness.csv`. That file is git-ignored and not committed. This audit resolved the sheet's 28 columns by header name. There is one row for this op.

## Status summary

| Field | Value |
|---|---|
| **Op directory** | `ttnn/cpp/ttnn/operations/experimental/deepseek/moe/deepseek_moe_gate` |
| **Overall** | **GREEN** |
| **DOps / Factories** | `DeepseekMoeGateDeviceOperation` → `DeepseekMoeGateProgramFactory` |
| *Prereqs* — Device 2.0 (every kernel used) | **Yes.** The one op-owned kernel uses Device 2.0 `CircularBuffer` wrappers for every CB sync on all three RISCs. It makes no NoC, address-generator or semaphore calls, and has no CB-index free-function holdovers. |
| *Prereqs* — Cross-op escapes | **Ok.** Every include is op-local or `tt_metal/*`. No donor kernels and no borrowed kernel files. |
| *Feature Support* — overall | **GREEN.** All three Appendix A entries are N/A. |
| *Feature Support* — Variadic-CTA | **Ok.** No CTA varargs: every CTA is read by name (`get_named_compile_time_arg_val("moe_gate_*")`). |
| *TTNN Readiness* — `Is able to port?` (the gate) | **Yes** (`yes`); the primary-column cross-check is clean |
| *TTNN Readiness* — Concept (current) | `descriptor` |
| *TTNN Readiness* — Execution Model | `SPMD` (`Secretly SPMD Workload?` is blank, as expected for a non-mesh-shaped concept) |
| *TTNN Readiness* — Custom hash | **No.** The op uses the default reflection hash; there is no `compute_program_hash`, `attribute_values` or `to_hash`. |
| *TTNN Readiness* — `get_dynamic_runtime_args` | **No** |
| *TTNN Readiness* — `override_runtime_arguments` | **Yes** (not a gate; the porter translates it): `DeepseekMoeGateProgramFactory::override_runtime_arguments` @ `device/deepseek_moe_gate_program_factory.cpp:39-50` (decl `.hpp:24-29`) |
| *TTNN Readiness* — Pybind `create_descriptor` | **No.** `deepseek_moe_gate_nanobind.cpp:12-43` binds only the user-facing function. |
| *TTNN Readiness* — Op-owned tensors | **No** |
| *TTNN Readiness* — Target concept | `CustomProgramSpecFactoryConcept` (no op-owned tensors) |
| *Port work* — Offset base pointer | **None.** There are no address arguments of any kind. The borrowed CBs sit at offset 0. |
| *Port work* — Tensor bindings (per binding) | **All five clean** (borrowed-memory DFB): `input_tensor`, `bias_tensor`, `input_indices_tensor`, `output_tensor`, `output_indices_tensor` |
| *TTNN Readiness* — TensorParameter relaxation | `none` (clears) |
| *Port work* — TensorAccessor 3rd arg | **None.** The op constructs no `TensorAccessor`. |
| *Port work* — CB endpoints | **Legal.** All five CBs are plain 1P+1C with roles fixed by FIFO ops: no self-loop, no multi-binding, no dead CB. |

**CB endpoints** are dispositions, not gates. Here every CB is a plain one-producer / one-consumer FIFO on every kernel node, in both `enable_sigmoid` configs, so none of the out-of-window resolutions (self-loop, 1P+1C relabel, multi-binding flag, dead-CB drop, conditional DFB) is needed.

## Result

**GREEN → brief issued.** Every gate-bearing subject cleared:

- **Device 2.0:** the op-owned unified kernel already uses `CircularBuffer` wrappers and does no NoC, address-generator or semaphore work.
- **Feature compatibility:** no `GlobalCircularBuffer`, no non-zero `address_offset`, no `GlobalSemaphore`.
- **TTNN factory concept:** `Is able to port?` = `yes`, and the cross-check is clean.
- **Offset base pointers:** the op passes no address arguments at all.
- **TensorAccessor 3rd argument:** the op has no accessors.

The target is `CustomProgramSpecFactoryConcept`. All seven informational subjects were run, because a port follows this audit.

## Gate detail

- **TTNN factory concept (`Is able to port?`):** **GREEN.** The sheet's `Is able to port?` is exactly `yes`.

  Relevant cells, verbatim:
  - `Concept` = `descriptor`
  - `Op Classification` = `PD Op (custom)`
  - `Execution Model` = `SPMD`
  - `Porting Target` = `CustomProgramSpecFactoryConcept`
  - `Custom hash (compute_program_hash)` = `no`; `Backdoor custom hash (attribute_values / to_hash)` = `no`
  - `Runtime-args update (get_dynamic_runtime_args)` = `no`
  - `Override runtime args method? (PD only)` = `yes`
  - `Pybind descriptor (nb::class_ of device op)` = `no`
  - `Smuggled pointer (raw buffer addr in RTA/CRTA)` = `no`
  - `Known op issues` = *(blank)*
  - `Diego validation` = `yes`
  - `TensorParameter relaxation` = `none`; `Provisional relaxation finding (Edwin)` = *(blank)*
  - `Op-owned tensors?` = `no`; `Secretly SPMD Workload?` / `Why secretly SPMD?` = *(blank)*
  - `Model` = `other`; `ProgramFactory used in llama?` = `no`; `Uses llama kernels?` = `no`
  - `Factory definition path` = `…/device/deepseek_moe_gate_program_factory.hpp`; `Declared in` = `…/device/deepseek_moe_gate_device_operation.hpp`

  Cross-check of the primary columns against the code:

  | Column (sheet value) | Code evidence | Match |
  |---|---|---|
  | `Concept` = `descriptor` | `static ProgramDescriptor create_descriptor(attrs, tensor_args, tensor_return_value)` takes no `MeshCoordinate` parameter (`device/deepseek_moe_gate_program_factory.hpp:18-21`). The op has no `create_workload_descriptor`, `create_mesh_workload`, `create_at` or `create_program_artifacts`. | ✓ |
  | `Custom hash` = `no`, backdoor = `no` | Grepping the op for `compute_program_hash`, `attribute_values` and `to_hash` returns nothing. The default reflection hash covers `operation_attributes_t {eps, scaling_factor, enable_sigmoid}` (`device/deepseek_moe_gate_device_operation_types.hpp:13-17`) and the five tensors. | ✓ |
  | `get_dynamic_runtime_args` = `no` | The device-op has no such hook (`device/deepseek_moe_gate_device_operation.hpp:18-44`). | ✓ |
  | `Override runtime args method?` = `yes` | `override_runtime_arguments` exists at `device/deepseek_moe_gate_program_factory.hpp:24-29` / `.cpp:39-50`. On a `descriptor` op this is the target-concept signal, not the legacy signature. | ✓ |
  | `Pybind descriptor` = `no` | `deepseek_moe_gate_nanobind.cpp:13-42` binds only `&ttnn::experimental::deepseek::moe::deepseek_moe_gate`. | ✓ |
  | `Secretly SPMD Workload?` / `Why secretly SPMD?` blank | The concept is not mesh-shaped, so blank is correct. | ✓ |
  | Factory-set match | The sheet has one row, `DeepseekMoeGateProgramFactory`. The code has one factory in `program_factory_t` (`device/deepseek_moe_gate_device_operation.hpp:25`). | ✓ 1:1 |
  | Cross-column invariants | `get_dynamic_runtime_args` is `no`. `Op-owned tensors?` is `no` on a `descriptor` row. | ✓ |

  `Smuggled pointer` = `no` is also borne out: the factory sets no `runtime_args` or `common_runtime_args` on any kernel (`deepseek_moe_gate_program_descriptor_builder.cpp:137-169`), and the kernel reads none.

- **Device 2.0 (every kernel used):** **GREEN.** The op exercises exactly one kernel source, the op-owned `device/kernels/deepseek_moe_gate_kernel.cpp`, compiled for NCRISC, BRISC and TRISC. It also exercises the op-local headers that source includes. No donor, shared-pool or cross-family kernel is involved. Per RISC, on the paths that actually run (`moe_gate_is_active_core` is hard-coded to `1`, so the `IsActiveCore` branches are always taken):
  - **NCRISC (reader):** `unified_kernels::setup_sharded_buffer(cb, 1)` ×3 (`deepseek_moe_gate_kernel.cpp:35-37`) does `CircularBuffer cb(cb_id); cb.reserve_back(n); cb.push_back(n);` (`unified_kernels/kernel_utils.hpp:66-70`). `Op::impl()` is a no-op on NCRISC (`unified_kernels/deepseek_moe_gate.hpp:95-98`). Device 2.0 wrapper only.
  - **BRISC (writer):** `CircularBuffer output_indices_cb(…)`, `CircularBuffer output_cb(…)`, then `.wait_front(1)` ×2 (`unified_kernels/deepseek_moe_gate.hpp:104-107`). Device 2.0 wrapper only.
  - **TRISC (compute):** all CB sync goes through `CircularBuffer` objects (`unified_kernels/deepseek_moe_gate.hpp:119-157`). Every other CB-id call is a compute or LLK API with no `CircularBuffer`-method counterpart, so none is a holdover: `reconfig_full_operand`, `pack_reconfig_data_format`, `copy_init`, `copy_tile`, `reconfig_data_format_srca`, `pack_tile`, `deepseek_moe_gate_init`, `deepseek_moe_gate`, and `compute_kernel_hw_startup` in `kernel_includes/tt_metal/include/compute_kernel_api/deepseek_compute_kernel_hw_startup.h:20`. The op-local LLK files under `kernel_includes/tt_llk/**` and `kernel_includes/tt_metal/hw/**` are compute-only and touch no data-movement API.
  - **No** `noc_async_*`, `Noc`, `InterleavedAddrGen*`, `ShardedAddrGen`, `TensorAccessor`, `get_semaphore`, `Semaphore`, `get_read_ptr` or `get_write_ptr` on any executed path. The sanctioned `get_local_cb_interface` and `get_tile_size` are not called on executed paths either.

  No violations, so there is no table.

  *Not exercised, so not gated (FYI-U):* `unified_kernels/kernel_utils.hpp` also defines helpers this op **never calls**:
  - `semaphore_dec` (`:73-75`) and `sync_riscs_enter` / `sync_riscs_exit` (`:95-117`), which do raw `volatile tt_l1_ptr uint32_t*` atomics
  - `override_cb_rd_ptr` (`:126-128`), `reconfig_cbs_for_mask` (`:149-184`), `reconfig_cb_interfaces` (`:186-220`) and `get_local_cb_rd_ptr` / `get_local_cb_page_size` / `update_local_cb_rd_ptr` (`:224-237`), which write `LocalCBInterface` cursors directly. `reconfig_cbs_for_mask` also resets stream registers through `get_cb_tiles_received_ptr` / `get_cb_tiles_acked_ptr` (`:177-178`).
  - `linear_id_in_grid` / `get_split_half_core_info` (`:30-55`)

  Each would be a Device-2.0-style or CB-to-DFB-whitelist finding if a kernel path ever reached it. None does: the kernel calls only `setup_sharded_buffer` from that header, plus `deepseek_compute_kernel_init` from the hw-startup header. This verdict must be re-run if a future change starts calling any of them. See Misc anomalies and Recipe notes.

- **Feature compatibility:** **GREEN.** No gate fired.

  | Feature | Status | Notes |
  |---|---|---|
  | GlobalCircularBuffer | N/A | No `GlobalCircularBuffer` type, `CreateGlobalCircularBuffer`, GCB include, `remote_index` or `remote_cb_*` in the op. All five CBs come from `ttnn::cb_descriptor_from_sharded_tensor` (`deepseek_moe_gate_program_descriptor_builder.cpp:99-103`), which sets `.global_circular_buffer = nullptr` (`ttnn/core/tensor/tensor_utils.cpp:65`). The override's `UpdateDynamicCircularBufferAddress(program, cb_id, *buffer)` is the `Buffer&` overload (`device/deepseek_moe_gate_program_factory.cpp:29`). |
  | CBDescriptor `address_offset` (non-zero) | N/A | All five `cb_descriptor_from_sharded_tensor` calls omit the third argument, so `address_offset = 0` by default (`ttnn/api/ttnn/tensor/tensor_utils.hpp:75`, assigned at `ttnn/core/tensor/tensor_utils.cpp:64`). The override uses the 3-argument, offset-free `UpdateDynamicCircularBufferAddress` (`device/deepseek_moe_gate_program_factory.cpp:29`). The offset triage doc agrees: "The C++ `deepseek_moe_gate` / `generalized_moe_gate` descriptor builders call the helper at offset 0 (benign; no action)." |
  | GlobalSemaphore | N/A | No `GlobalSemaphore` type, `CreateGlobalSemaphore`, `global_semaphore.hpp` include or semaphore parameter. The only matching text is a comment, "(for global semaphore reset across iterations)", on the never-called `semaphore_dec` raw-pointer helper (`unified_kernels/kernel_utils.hpp:72`). That is not a recognition signal. The `t6_semaphore_init` calls (`…/deepseek_compute_kernel_hw_startup.h:18-19`) set up Tensix hardware semaphores between MATH and PACK, which is unrelated. |

- **CB endpoints (GATE-free):** **All legal.** The census is per CB, per kernel node; all three kernels share one node set. The census is identical for `enable_sigmoid` = 0 and 1: the sigmoid branch (`kernel_includes/tt_metal/include/compute_kernel_api/deepseek_moe_gate.h:27-31, 48-67`) changes which LLK reads CB 0 and CB 1, but the same compute kernel touches the same CBs.

  | Legacy CB | Backing tensor (`borrowed_from`) | Format, size | Locked producer | Locked consumer | Other touchers | Disposition |
  |---|---|---|---|---|---|---|
  | 0 `kInputCb` | `input_tensor` | `Float16_b`, 1 × 2048 B | reader: `setup_sharded_buffer` → `reserve_back` / `push_back` (`deepseek_moe_gate_kernel.cpp:35`, `kernel_utils.hpp:66-70`) | compute: `wait_front` (`deepseek_moe_gate.hpp:139`) / `pop_front` (`:143`). Same kernel also unpacks it in `deepseek_moe_gate_init` / `deepseek_moe_gate` (`:138, :140-141`) and `reconfig_data_format_srca` (`:137`). | none | legal 1P+1C |
  | 1 `kBiasCb` | `bias_tensor` | `Float16_b`, 1 × 2048 B | reader (`deepseek_moe_gate_kernel.cpp:36`) | compute: `wait_front` (`deepseek_moe_gate.hpp:127`), never popped. Unpacked in `:138, :140-141`; `reconfig_full_operand` (`:115`). | none | legal 1P+1C |
  | 2 `kOutputCb` | `output_tensor` | `Float16_b`, 1 × 2048 B | compute: `reserve_back` (`:147`), `pack_tile` (`:152`), `push_back` (`:153`); `pack_reconfig_data_format<true>` (`:117`) | writer: `wait_front` (`deepseek_moe_gate.hpp:107`), never popped | none | legal 1P+1C |
  | 3 `kInputIndicesCb` | `input_indices_tensor` | `UInt16`, 1 × 2048 B | reader (`deepseek_moe_gate_kernel.cpp:37`) | compute: `wait_front` (`deepseek_moe_gate.hpp:126`), never popped. `copy_init` (`:130`), `copy_tile` (`:135`), `reconfig_full_operand` (`:115`). | none | legal 1P+1C |
  | 4 `kOutputIndicesCb` | `output_indices_tensor` | `UInt16`, 1 × 2048 B | compute: `reserve_back` (`:148`), `pack_reconfig_data_format` (`:155`), `pack_tile` (`:156`), `push_back` (`:157`) | writer: `wait_front` (`deepseek_moe_gate.hpp:106`), never popped | none | legal 1P+1C |

  - **Hidden-second-writer hunt:** negative. No executed path makes a raw `get_write_ptr`, `get_read_ptr` or `fifo_*_ptr` access, and the kernels use no semaphores. The cursor-writing helpers in `kernel_utils.hpp` (`:126-128`, `:149-220`, `:234-237`) are never called. The literal `compute_kernel_hw_startup(0, 0, 0)` (`…/deepseek_compute_kernel_hw_startup.h:20`) reads only CB 0's format metadata, and compute is already CB 0's consumer, so it adds no toucher. It does matter for DFB slot numbering; see the Heads-ups.
  - **Dead CBs:** none. Each of the five indices is referenced by two kernels.
  - **Nodes outside the kernel grid:** this is an artifact of legacy placement, not a disposition. `cb_descriptor_from_sharded_tensor` places each CB on *its own tensor's* shard grid (`ttnn/core/tensor/tensor_utils.cpp:57`). For CBs 1-4 validation only requires that grid to *contain* the input grid (`device/deepseek_moe_gate_device_operation.cpp:78, 83, 88`). If a caller passes a wider grid, those CB instances have 0 touchers on the extra nodes. Metal 2.0 derives DFB placement from the binding kernels' `WorkUnitSpec` nodes (`tt_metal/impl/metal2_host_api/program_spec/construction/resource/dfb.cpp:263-267`), so those instances simply do not exist after the port. Nothing is dropped, no spec is conditional, and the porter has nothing to do. Every in-tree caller uses one grid for all five tensors (`models/common/moe/tt_moe_gate.py:532-547`, `tests/ttnn/unit_tests/operations/experimental/deepseek/test_deepseek_moe_gate_program_cache.py:33-39`).

- **Offset base pointers:** **GREEN.** The op has no address runtime args or common runtime args at all:
  - The factory sets no `runtime_args`, `common_runtime_args` or `emplace_runtime_args` on any of the three kernels (`deepseek_moe_gate_program_descriptor_builder.cpp:137-169`).
  - The named CTAs carry CB ids, an always-1 flag and the bit patterns of two floats (`:111-132`). There is no CTA-baked address.
  - The kernels call neither `get_arg_val` nor `get_common_arg_val`.

  The tensors reach the kernels only as base-address borrowed CBs (`.buffer = tensor.buffer()` with `address_offset = 0`, `ttnn/core/tensor/tensor_utils.cpp:63-64`). They are refreshed by the offset-free 3-argument `UpdateDynamicCircularBufferAddress`. There are no Type 1 or Type 2 folds. Type 3 does not apply because the offset is zero. Type 4 does not apply because there is no `narrow`. The dated prior (`2026-07-19_offset_base_pointers.md`) does not list this op under Types 1 or 2, and its Type 3 note names it explicitly as offset 0 and benign. Reconciliation outcome: no fold, op not in the tables → clean.

- **TensorAccessor 3rd argument:** **N/A.** No accessor in the op passes a 3rd argument, because the op constructs no `TensorAccessor` at all (no `TensorAccessor` or `TensorAccessorArgs` in any kernel or in the factory), so the subject never fires. The dated triage (`2026-07-06_tensor_accessor_3rd_arg_triage.md`) does not list this op. Its `deepseek_moe_*` rows are `deepseek_moe_post_combine_reduce` and `deepseek_moe_fast_reduce_nc_fused`, which are different ops.

## Port-work summary  *(mirrors the brief)*

- **Tensor bindings** (per binding): all **clean** (borrowed-memory DFB). None is Case 1 or Case 2, and there is no kernel `TensorBinding`. Each of the five `TensorParameter`s is named only by one DFB's `borrowed_from`; a borrow-only parameter is legal without a kernel binding (`migration_guide.md`, TensorParameter section).

  | Legacy CB | Tensor | Access | Class |
  |---|---|---|---|
  | 0 | `input_tensor` | compute unpacks from the borrowed CB | clean |
  | 1 | `bias_tensor` | compute unpacks from the borrowed CB | clean |
  | 2 | `output_tensor` | compute packs into the borrowed CB (resident output shard) | clean (borrowed write; see Recipe notes) |
  | 3 | `input_indices_tensor` | compute copies from the borrowed CB | clean |
  | 4 | `output_indices_tensor` | compute packs into the borrowed CB | clean (borrowed write) |

- **TensorParameter relaxation:** `none`. The port applies no relaxation; every parameter is strict.
- **TensorAccessor 3rd arg:** none (no accessors).
- **CB endpoints:** all legal. Five borrowed DFBs, each with one locked producer and one locked consumer: reader → compute for CBs 0, 1 and 3; compute → writer for CBs 2 and 4. No self-loop, no 1P+1C relabel, no multi-binding flag, no drop, no conditional DFB.

## Heads-ups  *(mirrors the brief)*

- **CB endpoints (multi-binding shapes to watch):** none.
- **Cross-op / shared kernels:** none. The kernel is op-owned, and its only binder is this op's single factory, which instantiates it three times in one change. A census (`grep -rl deepseek_moe_gate_kernel ttnn/cpp/ttnn/operations/`) hits only `device/deepseek_moe_gate_program_descriptor_builder.cpp:21`. `models/demos/deepseek_v3_b1/micro_ops/deepseek_moe_gate/op.py:190` binds a *same-named private copy* under `models/`, not this file. No `_metal2` fork exists and none is needed. The kernel converts in place as ordinary port work, and there is no sunset list. "Porting a shared kernel" in `port_patterns.md` does not apply: its recognition signal needs a second binder that will not convert in the same change.
- **RTA varargs:** none. The kernels read no RTAs or CRTAs, and every CTA is a distinct named field.
- **Anything else the porter needs:**
  - **Literal CB index 0 in the compute init.** `deepseek_compute_kernel_init()` calls `compute_kernel_hw_startup(0, 0, 0)` (`kernel_includes/tt_metal/include/compute_kernel_api/deepseek_compute_kernel_hw_startup.h:20`, called at `deepseek_moe_gate_kernel.cpp:55`). It hard-codes legacy CB 0 (`kInputCb`, `device/deepseek_moe_gate_program_descriptor_builder.hpp:16`) as srcA, srcB and pack operand for the hardware startup. In Metal 2.0 a DFB's device slot, which on Gen1 is its CB index (`tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3242`), is the lowest slot not used by an overlapping DFB, assigned in `ProgramSpec::dataflow_buffers` declaration order (`dataflow_buffer.cpp:2182-2202`; `…/construction/resource/dfb.cpp:253-270`). There are two faithful options:
    - (a) Leave the helper untouched and declare the DFBs in legacy CB-index order: input, bias, output, input_indices, output_indices. That reproduces slots 0-4 exactly, so the literal keeps meaning "input". This relies on observed implementation behavior, not a documented API contract.
    - (b) Pass `dfb::<input>` into the helper, if the kernel-side whitelist allows that edit to an op-local helper.

    Do not leave the literal pointing at whichever DFB happens to land on slot 0. Record the choice in the port report.
  - **CB ids travel as plain integers before a wrapper is built.** The writer and compute receive CB ids as `uint32_t` template parameters (`WriterCTArgs<…>` / `ComputeCTArgs<…>`, `deepseek_moe_gate_kernel.cpp:41-54`; `unified_kernels/deepseek_moe_gate.hpp:53-79`), and then build `CircularBuffer`s from them (`:104-105, :119-123`). The reader passes them as a `uint32_t cb_id` parameter into `setup_sharded_buffer` (`kernel_utils.hpp:66-67`).
    - Passing `dfb::name` in either position compiles: the token's `uint32_t` conversion is `constexpr` (`port_patterns.md`, "Pass DFB handles directly…").
    - The `DataflowBuffer` built from such an integer would use the low-level `DataflowBuffer(uint16_t)` constructor, which `tt_metal/hw/inc/api/dataflow/dataflow_buffer.h:155-156` documents as "prefer DFBBindingToken / RelayDFBBindingToken for new kernel code".

    Check this against the kernel-side whitelist rather than swapping blind. The `CircularBuffer` include sites are `unified_kernels/deepseek_moe_gate.hpp:11, :27` and `unified_kernels/kernel_utils.hpp:11`; the constructions are `kernel_utils.hpp:67` and `deepseek_moe_gate.hpp:104-105, 119-123`.
  - **Named CTAs are read in constant-expression positions:** at file scope (`struct Core`, `deepseek_moe_gate_kernel.cpp:17-19`) and as template arguments (`:41-54`). Metal 2.0's `constexpr auto x = get_arg(args::x)` is legal for CTAs (`tt_metal/hw/inc/experimental/kernel_args.h:80-83`), so these positions survive. `eps` and `scaling_factor` are float bit patterns packed into CTAs (`deepseek_moe_gate_program_descriptor_builder.cpp:23-26, 128-129`); carry them as the same `uint32_t` values.
  - **Unified kernel, three `KernelSpec`s, one `WorkUnitSpec`.** Keep one `KernelSpec` per legacy `KernelDescriptor`, all on the same source. The per-RISC `#if defined(COMPILE_FOR_*)` partition is preprocessor-level, so each `KernelSpec` needs only the DFB tokens its own branch names:

    | `KernelSpec` | Binds | Role |
    |---|---|---|
    | reader | input, bias, input_indices | PRODUCER |
    | writer | output, output_indices | CONSUMER |
    | compute | input, bias, input_indices | CONSUMER |
    | compute | output, output_indices | PRODUCER |

    The hardware configs map 1:1:
    - legacy reader (`RISCV_1`, `NOC::RISCV_0_default` = NOC 0, `DM_DEDICATED_NOC`) = `CreateReaderDataMovementConfig()`
    - legacy writer (`RISCV_0`, `NOC::RISCV_1_default` = NOC 1, `DM_DEDICATED_NOC`) = `CreateWriterDataMovementConfig()`

    (Both in `tt_metal/api/tt-metalium/experimental/metal2_host_api/data_movement_hardware_config.hpp:107-155`.) Compute is `HiFi4` with every other field at its default (`deepseek_moe_gate_program_descriptor_builder.cpp:134-135`).
  - **Leave the never-called helpers in `kernel_utils.hpp` untouched.** Do not convert their `LocalCBInterface` writes to `evil_set_*`, and do not delete them. In particular, the CB-to-DFB whitelist row "`get_cb_tiles_acked_ptr` / `get_cb_tiles_received_ptr` → no DFB equivalent, cannot be ported today" (`cb_dfb_api_whitelist.md`, section D) is **not** triggered by `kernel_utils.hpp:177-178`, because nothing calls `reconfig_cbs_for_mask` or `reconfig_cb_interfaces`. If a grep stops you there, this is the auditor's ruling: dead code, out of scope.
  - **DFB sizing and metadata:**
    - Reproduce `set_cb_page_size_for_tile` (`deepseek_moe_gate_program_descriptor_builder.cpp:28-36`): `entry_size` = tile size, falling back to the total size if that does not divide evenly. Under the validated 32×32 shard and tile, every DFB is 1 entry × 2048 B.
    - Set `data_format_metadata` (`Float16_b` ×3, `UInt16` ×2) and `tile_format_metadata` (copy the legacy `format_descriptors[0].tile`, which is the tensor's 32×32 tile) on all five DFBs. Compute touches every one of them, and a format-less DFB gets no compute JIT metadata (`tt_metal/impl/dataflow_buffer/dataflow_buffer.cpp:3244-3246`).
    - Do not try to reproduce the legacy per-tensor CB `core_ranges`; placement is derived.
  - **Never-popped FIFOs are intentional.** Compute waits on bias and input_indices without popping; the writer waits on output and output_indices without popping. Keep both as they are, and do not balance them (`migration_guide.md`, DataflowBufferSpec validator note).
  - **Preallocated in-place outputs.** `output_tensor` and `output_indices_tensor` arrive in `tensor_args` and are echoed in `tensor_return_value` (`device/deepseek_moe_gate_device_operation.cpp:110-113`). Take all five `MeshTensor`s from `tensor_args` in both `create_program_artifacts` and the translated override.
  - **The factory body spans two `.cpp` files in a unity-build target** (`CMakeLists.txt:14, 38-42`): `deepseek_moe_gate_program_factory.cpp` (anonymous namespace `update_tensor_cb`, `:19-32`) and `deepseek_moe_gate_program_descriptor_builder.cpp` (anonymous namespace `:18-38`). New anonymous-namespace names must not collide (`port_patterns.md`, "Unity-build hygiene"). The `kInputCb`… constants in `deepseek_moe_gate_program_descriptor_builder.hpp:15-20` exist only to share CB ids between the two entry points; the natural successors are shared `TensorParamName` and `DFBSpecName` constants. Keeping both file names avoids touching `CMakeLists.txt`, which `metalium-developers-infra` co-owns (`CMakeLists.txt:71-75`).
  - **Do not move or rename `device/kernel_includes/**`.** It is LLK-level, compute-only code that takes `uint32_t` operand ids, so DFB tokens convert implicitly and nothing in it changes. The tt-llk test `tt_metal/tt-llk/tests/sources/deepseek_moe_gate_test.cpp:141-147` includes six of those headers by path. The compute path is arch-specific (`kernel_includes/tt_metal/include/compute_kernel_api/deepseek_moe_gate.h:11-19` selects the BH or WH copies).
  - **Verification pointers:**
    - `tests/ttnn/unit_tests/operations/experimental/deepseek/test_deepseek_moe_gate_program_cache.py` runs two live allocations and a cache hit, and asserts one cache entry. This exercises the translated override's whole job.
    - `models/common/tests/moe/test_tt_moe_gate.py` covers the `n_group=8` configs, which take the deepseek grouped path through this op.
    - Verify on both WH and BH where possible.

## Team-only

- **Out-of-directory coupling & donor shape:**
  1. **Op-level roll-up:** ✓ clean.
     - Every `#include` reachable from the kernel resolves either inside the op directory (`device/unified_kernels/**`, `device/kernel_includes/**`) or to `tt_metal/*` (class 1: LLK / HAL / compute and dataflow APIs).
     - No `ttnn/cpp/ttnn/kernel_lib/`, no `ttnn/cpp/ttnn/kernel/`, no `kernel_helper_functions/`, and no in-family or cross-family donor.
     - No borrowed kernel files.
  2. **Summary table:**

     | Op kernel | Donor file(s) | Class | Status |
     |---|---|---|---|
     | `deepseek_moe_gate_kernel.cpp` (all RISCs) | `api/debug/kernel_profiler.hpp`, `api/compute/blank.h` (via `unified_kernels/kernel_op_api.hpp:14-25`) | `tt_metal/*` | ✓ |
     | `deepseek_moe_gate_kernel.cpp` (NCRISC/BRISC) | `api/dataflow/dataflow_api.h`, `api/dataflow/circular_buffer.h` (via `unified_kernels/kernel_utils.hpp:10-11`, `unified_kernels/deepseek_moe_gate.hpp:10-11`) | `tt_metal/*` | ✓ |
     | `deepseek_moe_gate_kernel.cpp` (TRISC) | `api/compute/{compute_kernel_api,common,tile_move_copy,pack,reconfig_data_format,eltwise_binary,transpose}.h`, `api/dataflow/circular_buffer.h` (via `unified_kernels/deepseek_moe_gate.hpp:20-27`, `kernel_includes/…/deepseek_moe_gate.h:7-9`) | `tt_metal/*` | ✓ |
     | `deepseek_moe_gate_kernel.cpp` (TRISC) | `llk_unpack_common_api.h`, `llk_math_common_api.h`, `llk_math_eltwise_unary_sfpu_macros.h`, `ckernel*.h`, `cmath_common.h`, `llk_math_common.h`, `llk_assert.h`, `lltt.h`, `sfpi.h`, `sfpu/ckernel_sfpu_load_config.h`, `ckernel_sfpu_recip.h` (via the op-local `kernel_includes/**`) | `tt_metal/*` (LLK) | ✓ |

  3. **Per-call detail:** omitted, because every row is ✓.
  4. **Borrowed kernel files:** none. The factory binds only its own `device/kernels/deepseek_moe_gate_kernel.cpp` (`deepseek_moe_gate_program_descriptor_builder.cpp:20-21`). It is not lent either; see the census in Heads-ups. There is no `_metal2` fork.
  5. **Reverse coupling (not an escape):** `tt_metal/tt-llk/tests/sources/deepseek_moe_gate_test.cpp:141-147` includes six of this op's `device/kernel_includes/**` LLK headers by path. The port does not touch them.
  6. **Header provenance:** `device/unified_kernels/{deepseek_moe_gate,kernel_op_api,kernel_utils}.hpp` are Device-2.0-migrated, diverged forks of `models/demos/deepseek_v3_b1/unified_kernels/` (the model copies still use `cb_*` free functions). The two sets are independent, and the port touches only the op's.
- **Relaxation candidates:** none. There is no custom hash to mine.
- **TTNN factory analysis:**
  - **Concept:** `descriptor` (`device/deepseek_moe_gate_program_factory.hpp:18-21`). The op already has a conventional factory struct and `program_factory_t` (`device/deepseek_moe_gate_device_operation.hpp:25`), so `ttnn_factory.md` exception 3 (direct-descriptor conversion) does not apply.
  - **Op-owned tensors:** none.
  - **Execution model:** `SPMD`. One program for every coordinate; the override ignores `coord` (`device/deepseek_moe_gate_program_factory.cpp:44`).
  - **Pybind `create_descriptor`:** none, so the port deletes no binding (exceptions 1 and 2 do not apply). No other risky pybind exists: nanobind exposes only the user-facing function.
  - **Custom hash:** none (default reflection hash).
  - **`get_dynamic_runtime_args`:** none.
  - **`override_runtime_arguments`:** present (`device/deepseek_moe_gate_program_factory.cpp:39-50`, decl `.hpp:24-29`). It refreshes exactly the five tensor-backed CB addresses, one per io tensor, through `update_tensor_cb`. That helper finds the globally-allocated CB holding each index and calls the 3-argument `UpdateDynamicCircularBufferAddress` (`:21-30`, `:45-49`). It refreshes nothing else, because the op has no RTAs. The faithful translation returns a `ProgramRunArgs` whose `tensor_args` holds one `TensorArgument` for each of the five `TensorParameter`s, with no `kernel_run_args` and no `dfb_run_overrides` (the legacy override sets no size).
  - *Observation for the sheet owner (not port work):* once translated, that override does exactly what `ProgramSpecFactoryConcept`'s automatic binding refresh would do. Switching to the base concept later is a TTNN or ops decision. The port follows the sheet's `Porting Target` and translates the override rather than deleting it (`ttnn_factory.md`).

## TTNN ProgramFactory

### Concept
`CustomProgramSpecFactoryConcept` (the sheet's `Porting Target`). The override to translate is `DeepseekMoeGateProgramFactory::override_runtime_arguments` @ `device/deepseek_moe_gate_program_factory.cpp:39-50` (decl `device/deepseek_moe_gate_program_factory.hpp:24-29`).

### Fit
- Execution model: `SPMD`. Nothing varies per coordinate (`coord` is unused at `device/deepseek_moe_gate_program_factory.cpp:44`).
- Op-owned device resources: none.
- Tensor-arg matching: strict (relaxation `none`).
- Ported-from shape: 1:1. A `descriptor` factory (`create_descriptor` → one `ProgramDescriptor`, no `MeshCoordinate` parameter) becomes one `ProgramSpec`.

### Custom compute_program_hash
None: the default reflection-based hash. The op has no `compute_program_hash`, `attribute_values` or `to_hash`.

### Stop signals
None.

## Misc anomalies  *(team-only, non-gating)*

- **Dead helper code in the op-local `unified_kernels/kernel_utils.hpp`:**
  - `linear_id_in_grid` / `get_split_half_core_info` (`:30-55`, plus the `extern` firmware globals at `:18-19`)
  - `semaphore_dec` (`:72-75`)
  - `sync_riscs_enter` / `sync_riscs_exit` (`:95-117`)
  - `override_cb_rd_ptr` (`:126-128`)
  - `reconfig_cbs_for_mask` / `reconfig_cb_interfaces` (`:149-220`)
  - `get_local_cb_rd_ptr` / `get_local_cb_page_size` / `update_local_cb_rd_ptr` (`:224-237`)
  - in other headers: `deepseek_compute_kernel_hw_startup<>` (`kernel_includes/tt_metal/include/compute_kernel_api/deepseek_compute_kernel_hw_startup.h:27-37`) and `unified_kernels::SelectByRISCV` (`unified_kernels/kernel_op_api.hpp:35-36`)

  These were carried over from the model's unified-kernel library, and this op calls none of them. Several use idioms that would be audit findings if they were ever reached: raw L1 semaphore atomics; direct `LocalCBInterface` cursor writes; stream-register resets via `get_cb_tiles_received_ptr` / `get_cb_tiles_acked_ptr`, which have no DFB equivalent. The comment at `:72` mentions a "global semaphore" that does not exist in this op. Suggestion for the ops team: trim the op-local copy, which would remove the trap for grep-driven tooling.
- **Always-1 flag:** `moe_gate_is_active_core` is hard-coded to `1` on all three kernels (`deepseek_moe_gate_program_descriptor_builder.cpp:115, 120, 131`). The inactive-core branches (`deepseek_moe_gate_kernel.cpp:34`, `unified_kernels/deepseek_moe_gate.hpp:88-90`) are therefore dead. This is a vestige of the fused-op framework this kernel came from.
- **Literal CB index:** `compute_kernel_hw_startup(0, 0, 0)` (`…/deepseek_compute_kernel_hw_startup.h:20`) uses literal CB 0 rather than the `moe_gate_input_cb` CTA. It is correct today only because `kInputCb == 0` (`device/deepseek_moe_gate_program_descriptor_builder.hpp:16`).
- **Redundant page-size rewrite:** `set_cb_page_size_for_tile` (`deepseek_moe_gate_program_descriptor_builder.cpp:28-36`) rewrites the `page_size` and `tile` that `cb_descriptor_from_sharded_tensor` already set (`ttnn/core/tensor/tensor_utils.cpp:61-62`). Under the validated 32×32 tile and 32×32 shard the two agree (2048 B, one page), and the `total_size` fallback branch is unreachable.
- **Duplicated validation:** the builder re-runs a subset of `validate_on_program_cache_miss` (`deepseek_moe_gate_program_descriptor_builder.cpp:56-91` vs `device/deepseek_moe_gate_device_operation.cpp:28-99`). The subset omits the output-shard shape and orientation equality checks and the 32×32 checks. This is harmless, because validation always runs first.
- **Uncovered output shards:** validation only requires the output shard grid to *contain* the input grid (`device/deepseek_moe_gate_device_operation.cpp:88`). No kernel runs on output cores outside the input grid, so those shards keep their prior contents. That is probably intended for preallocated buffers; noted for completeness.

## Recipe notes

1. **Uncalled functions inside a referenced header.** The Device 2.0 gate is about kernels the op "exercises" ("instantiates or calls into"). *Scope of the audit* excludes unreferenced kernel *files*, but says nothing about uncalled *functions* inside a referenced op-local header. That gap matters here. `unified_kernels/kernel_utils.hpp:177-178` calls `get_cb_tiles_received_ptr` / `get_cb_tiles_acked_ptr`, which `cb_dfb_api_whitelist.md` §D says make a kernel "cannot be ported today". A grep-driven reading would therefore RED this op, or stop the porter, over dead code. I judged by exercise: not called, so not gated, and the ruling is passed to the porter explicitly. A sentence in the Device 2.0 subject would settle it, for example "judge what a kernel path reaches; uncalled helpers in an included header are FYI-U and must be re-audited if they become reachable".
2. **The causal-link gate only covers reads.** *TensorParameter analysis* frames the clean case as a "borrowed-memory DFB **read**". Here two of the five borrowed CBs are **written**: compute packs into the resident output shards. I applied the gate symmetrically and marked them clean. Suggest wording such as "borrowed-memory DFB read or write".
3. **Per-node census when a borrowed CB sits on a wider grid than the kernels.** `cb_descriptor_from_sharded_tensor` places a CB on its tensor's shard grid, which the op's validation lets exceed the kernel grid. A per-node census then finds 0-toucher nodes for a CB that is live on every kernel node. The census table's dead-CB and conditional-DFB rows don't fit this case, and Metal 2.0's derived placement makes it a non-event. A one-line note in *CB endpoints* would keep the next auditor from reporting a phantom dead or conditional CB.
4. **Two templates for the TTNN section.** `ttnn_factory.md` ("Audit report deliverable") tells the auditor to add a `## TTNN ProgramFactory` section. The audit template's Team-only "TTNN factory analysis" bullet covers much of the same ground. I emitted both. Folding one into the other would remove the duplication.
5. **Template drift: the Variadic-CTA row.** The status-summary template still has a "*Feature Support* — Variadic-CTA (Ok / Unsupported)" row, but Appendix A has no such entry, and *RTA varargs* says CTA varargs are supported via `compile_time_varargs`. I filled it as "Ok". The row could be dropped or moved under Port work.
6. **Doc drift in `shared/migration_guide.md`.** The KernelSpec example at line 214 calls `CreateReaderGen1DataMovementConfig()`, which does not exist in the tree. The helpers are `CreateReaderDataMovementConfig()` / `CreateWriterDataMovementConfig()` (`tt_metal/api/tt-metalium/experimental/metal2_host_api/data_movement_hardware_config.hpp:107, 140`).
7. **DFB slot numbering is invisible to the recipe.** This op's compute init hard-codes a CB index (`compute_kernel_hw_startup(0, 0, 0)`). Whether it still means "input" after the port depends on DFB declaration order. The ordering is deterministic in code (`dfb.cpp:253-254`: "deterministic DFB ID assignment based on user-specified order"; lowest-free slot in `dataflow_buffer.cpp:2182-2202`), but neither the audit nor the shared docs mention it. A literal CB index in a kernel or helper may deserve its own audit signal, or a note under *CB endpoints* or the brief's "Anything else" bullet.
