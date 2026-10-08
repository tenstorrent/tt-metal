# Port Plan — topk_router_gpt

Port plan for `experimental/topk_router_gpt`, ported from the `ProgramDescriptor` API (direct-descriptor shape) to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept`, in the **direct-descriptor** shape. `create_descriptor` is a static member of `TopkRouterGptDeviceOperation` (`device/topk_router_gpt_device_operation.hpp:26-29`), body in `device/topk_router_gpt_program_factory.cpp:44-373`. There is no `program_factory_t` and no factory struct, so [exception 3](ttnn_factory.md §3) applies. (Checked at HEAD `5ce297c508a`: no `program_factory_t` has appeared since the audit.)
- Variants: single. There is one host-derived config axis, chosen by the device: `cores_per_group = num_cores >= 12 ? 3 : 2` (WH: 12 cores, 2 senders/group; BH P150: 8 cores, 1 sender/group). It changes values, not structure: the same three kernels, CBs and semaphores in both configs.
- Custom `compute_program_hash`: none — default reflection-based hash. No `attribute_values` / `to_hash` backdoor.
- `override_runtime_arguments`: none (the header comment at `device_operation.hpp:22-25` explains why).

### Kernels
All three kernels run on `all_cores` = every DRAM-bank-aligned worker (`program_factory.cpp:53-56`). All three get the same positional CTA block, the same named CTAs, and the same 20-slot RTA block per core.

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| dm0 | `device/kernels/dm0.cpp` | `all_cores` | `TensorAccessorArgs` × 5: input, weight, bias, indices_rm, weights_rm (`:203-212`) | `num_cores`, `num_groups`, `cores_per_group`, `num_senders`, `collector_physical_x`, `collector_physical_y`, `topk_k`, `k_padded`, `n_tiles`, `tile_size_bf16` (`:216-227`) | 20 per core, `required_cores` ring positions (`:319-366`), see below | none | none | O2 (unset → DM default) | DM `RISCV_1` / `NOC_0` / `DM_DEDICATED_NOC` (default) |
| dm1 | `device/kernels/dm1.cpp` | `all_cores` | same | same | same | none | none | O2 (unset → DM default) | DM `RISCV_0` / `NOC_1` / `DM_DEDICATED_NOC` (default) |
| compute | `device/kernels/compute.cpp` | `all_cores` | same | same | same | none | none | **O3** (unset → compute default) | `HiFi2`, `fp32_dest_acc_en = true`, `dst_full_sync_en = false`, `bfp8_pack_precise = false`, `math_approx_mode = false`, `unpack_to_dest_mode` empty |

RTA block (all three kernels, one fixed `argidx++` run, `dm0.cpp:30-50`, `dm1.cpp:88-108`, `compute.cpp:39-59`):
`[0] dram_bank_id [1] vchannel [2] weight_addr [3] input_addr [4] bias_addr [5] sem_partial_ready [6] is_sender [7] is_worker [8] is_collector [9] num_k_tiles [10] k_tile_offset [11] n_tile_id [12] worker_phys_x [13] worker_phys_y [14] sender_slot [15] worker_gather_slot [16] sem_topk_ready [17] indices_rm_addr [18] weights_rm_addr [19] aligned_page_size`.

### CBs
All Float16_b, `tile` unset, `address_offset` 0, none buffer-backed, no GlobalCircularBuffer. Tile page size = `tile_size(Float16_b)` = 2048 B.

| index | name | total_size | core_ranges | data_format | page_size | tile |
|---|---|---|---|---|---|---|
| c_0 | cb_weight | `max_k_tiles` × tile | all | Float16_b | tile | unset |
| c_1 | cb_input | `max_k_tiles` × tile | all | Float16_b | tile | unset |
| c_2 | cb_partial_recv | `num_senders` × tile | all | Float16_b | tile | unset |
| c_3 | cb_local_out | 1 × tile | all | Float16_b | tile | unset |
| c_4 | cb_bias | 1 × tile | workers | Float16_b | tile | unset |
| c_5 | cb_index | 1 × tile | workers | Float16_b | tile | unset |
| c_6 | cb_topk_val | 1 × tile | workers | Float16_b | tile | unset |
| c_8 | cb_gathered_val | 4 × tile | workers | Float16_b | tile | unset |
| c_9 | cb_gathered_ind | 4 × tile | workers | Float16_b | tile | unset |
| c_10 | cb_intermed_val | 2 × tile | collector | Float16_b | tile | unset |
| c_11 | cb_intermed_ind | 1 × tile | collector | Float16_b | tile | unset |
| c_12 | cb_softmax_mask | 1 × tile | collector | Float16_b | tile | unset |
| c_13 | cb_softmax_tmp | 1 × tile | collector | Float16_b | tile | unset |
| c_14 | cb_reduce_scalar | 1 × tile | collector | Float16_b | tile | unset |
| c_15 | cb_bcast_scaler | 1 × tile | collector | Float16_b | tile | unset |
| c_16 | cb_final_out | 2 × tile | collector | Float16_b | tile | unset |
| c_19 | cb_dispatch | `2·32·k_padded·2` B (1 page) | collector | Float16_b | = total | unset |

### Semaphores
| id | core_type | core_ranges | initial_value |
|---|---|---|---|
| 0 (`sem_partial_ready`, RTA `[5]`) | WORKER | all_cores | 0 |
| 1 (`sem_topk_ready`, RTA `[16]`) | WORKER | all_cores | 0 |

Only dm1 uses them (`dm1.cpp:162, 222, 258, 299`: `Semaphore<> x(<rta>)`), both locally (`wait` / `set`) and remotely (`up(noc, x, y, 1)`).

### Tensor accessors
| host site (file:line) | originating Tensor | RTA slot (host) | kernel use |
|---|---|---|---|
| `program_factory.cpp:203-208, 345` | `input_tensor` | `[3]` | `dm0.cpp:61` |
| `program_factory.cpp:203-208, 344` | `weight_tensor` | `[2]` | `dm0.cpp:62` |
| `program_factory.cpp:203-208, 346` | `bias_tensor` | `[4]` | `dm0.cpp:99` (worker only) |
| `program_factory.cpp:210, 359` | output 0 `indices_rm` | `[17]` | `dm1.cpp:343` (+ 3rd arg `aligned_page_size`) |
| `program_factory.cpp:211, 360` | output 1 `weights_rm` | `[18]` | `dm1.cpp:344` (+ 3rd arg `aligned_page_size`) |

The address RTAs are pushed as `Buffer*` (framework-patched `BufferBinding`s). dm1 and compute read `[2-4]`, dm0 and compute read `[17-18]`, without using them.

### Work split
- n/a — no `split_work_to_cores`. Per-core roles come from the DRAM-bank → worker assignment (`get_optimal_dram_bank_to_logical_worker_assignment(RISCV_0_default)`), a NOC1 ring sort, and the k-tile split `k_tiles_per_core_base` / `k_tiles_remainder` over `cores_per_group`. All of it is host geometry carried over unchanged; it feeds per-node RTA values only. There is one `KernelDescriptor` per kernel, so no work-split multiplicity.

### Shared kernels
none. `grep -rl` for `dm0.cpp` / `dm1.cpp` / `compute.cpp` under `ttnn/cpp/ttnn/operations/` hits many unrelated same-named files in other ops; the only references to *these* paths (`experimental/topk_router_gpt/device/kernels/…`) are this op's factory. Every kernel `#include` is under `tt_metal/*`. Convert in place.

### Flags
- The RTA node set (`required_cores` ring positions, `:319`) and the kernel placement (`all_cores`, `num_cores` cores) are equal on WH (12 = 12) and BH P150 (8 = 8) only; the `TT_FATAL` at `:66-70` checks `>=`. Preserved as is (audit Misc anomalies).
- `compute.cpp:84` declares `CircularBuffer cb_index(cb_index_id)` and never uses it. It has no DFB binding on compute (see Applied Patterns), so the declaration and its `cb_index_id` constant (`:67`) cannot survive the swap; they are deleted as dead code (no behavior).

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept`.
- **Custom `compute_program_hash`**: none.
- **Implementation notes**: exception 3 (direct-descriptor op). Nest `struct TopkRouterGptProgramFactory` with `create_program_artifacts` inside `TopkRouterGptDeviceOperation`, add `using program_factory_t = std::variant<TopkRouterGptProgramFactory>;`, remove the device-op-level `create_descriptor`. The body stays in `device/topk_router_gpt_program_factory.cpp` (already in `sources.cmake`). No pybind change: `topk_router_gpt_nanobind.cpp` binds only the user function.

## Planned Spec Shape

- **KernelSpecs** (3, 1:1 with legacy):
  - `dm0` — `dm0.cpp`, `create_reader_datamovement_config()` (RISCV_1 / NOC_0 / dedicated = reader default), opt O2.
  - `dm1` — `dm1.cpp`, `create_writer_datamovement_config()` (RISCV_0 / NOC_1 / dedicated = writer default), opt O2.
  - `compute` — `compute.cpp`, Style B `ComputeHardwareConfig{.fpu_math_fidelity = HiFi2, .sfpu_precision_mode = Precise, .enable_32_bit_dest = true, .double_buffer_dest = true}`, `bfp_pack_precision_mode` left at its default (Approximate = legacy `bfp8_pack_precise = false`), no `unpack_modes` (no Float32 DFB; legacy vector empty), opt **O3**.
  - Named CTAs on all three: the legacy 10, unchanged.
  - Named RTA schema on all three: `dram_bank_id`, `vchannel`, `is_sender`, `is_worker`, `is_collector`, `num_k_tiles`, `k_tile_offset`, `n_tile_id`, `worker_phys_x`, `worker_phys_y`, `sender_slot`, `worker_gather_slot` (the 12 non-address, non-semaphore, non-page-size legacy slots; read-but-unused fields are carried over per the brief).
- **DataflowBufferSpecs** (17, legacy CB order, so allocation order is unchanged): `weight`, `input`, `partial_recv`, `local_out`, `bias`, `index`, `topk_val`, `gathered_val`, `gathered_ind`, `intermed_val`, `intermed_ind`, `softmax_mask`, `softmax_tmp`, `reduce_scalar`, `bcast_scaler`, `final_out`, `dispatch`. Sizes, formats and entry counts copied from the CB table; `tile_format_metadata` left unset (legacy `tile` unset).
- **SemaphoreSpecs** (2): `partial_ready`, `topk_ready`, `target_nodes = all_cores`, initial value 0 (default). Bound on dm1 only.
- **TensorParameters** (5): `input`, `weight`, `bias`, `indices_rm`, `weights_rm`, each from its `MeshTensor::tensor_spec()`, strict (relaxation `none`). Bound: dm0 ← input, weight, bias; dm1 ← indices_rm, weights_rm.
- **WorkUnitSpecs** (1): `main` = {dm0, dm1, compute} on `all_cores`.
- **Op-owned tensors**: none.

**DFB placement.** Legacy narrowed c_4–c_9 to workers and c_10–c_19 to the collector. Metal 2.0 derives placement from the bound kernels, and every kernel runs on every core, so every DFB lands on every core. That keeps the uniform L1 layout dm1 depends on (it uses its own `partial_recv` / `gathered_val` / `gathered_ind` write pointer as a remote NoC destination). Allocation is in `spec.dataflow_buffers` order (`program_spec.cpp`, "deterministic DFB ID assignment based on user-specified order") over one node set, so offsets are identical on every node. Cost: footprint only — senders now also hold the worker/collector DFBs (see report). No per-role work units (brief, audit Q2).

## Preserved Multiplicity

none — no work-split multiplicity in legacy (one `KernelDescriptor` per kernel source).

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| factory `:203-212` + dm0 `:25-27`, dm1 `:187-192` | `TensorAccessorArgs(*buf).append_to(compile_args)` × 5, kernel `TensorAccessorArgs<0>()` + `next_compile_time_args_offset()` chain | `TensorBinding`s; `TensorAccessor(tensor::<name>)` |
| factory `:344`, all kernels RTA `[2]` | `weight_buffer` (`Buffer*`) | `TensorBinding(weight)` on dm0; read dropped from dm1 / compute |
| factory `:345`, all kernels RTA `[3]` | `input_buffer` | `TensorBinding(input)` on dm0; read dropped from dm1 / compute |
| factory `:346`, all kernels RTA `[4]` | `bias_buffer` | `TensorBinding(bias)` on dm0; read dropped from dm1 / compute |
| factory `:359`, all kernels RTA `[17]` | `indices_rm_buffer` | `TensorBinding(indices_rm)` on dm1; read dropped from dm0 / compute |
| factory `:360`, all kernels RTA `[18]` | `weights_rm_buffer` | `TensorBinding(weights_rm)` on dm1; read dropped from dm0 / compute |
| factory `:296-297, 361`, all kernels RTA `[19]`, dm1 `:343-344` | `aligned_page_size` as `TensorAccessor` 3rd arg | dropped (Class 2, brief); its only consumer was the 3rd arg, so the RTA, its host computation and the stale comment at dm1 `:341-342` go with it |
| factory `:269-274, 347`, all kernels RTA `[5]` | semaphore id 0 | `SemaphoreSpec partial_ready` + `SemaphoreBinding` on dm1 (`sem::partial_ready`); read dropped from dm0 / compute |
| factory `:269-274, 358`, all kernels RTA `[16]` | semaphore id 1 | `SemaphoreSpec topk_ready` + `SemaphoreBinding` on dm1 (`sem::topk_ready`); read dropped from dm0 / compute |
| dm0 `:53-55`, dm1 `:218-227`, compute `:62-77` | hardcoded `constexpr auto cb_*_id = tt::CBIndex::c_N` | `DFBBinding`s; `DataflowBuffer dfb_x(dfb::x)` and `dfb::x` at LLK call sites |
| all kernels, remaining RTAs | positional `get_arg_val<uint32_t>(argidx++)` | named RTAs `get_arg(args::<name>)` |
| all kernels, named CTAs | `get_named_compile_time_arg_val("x")` | `get_arg(args::x)` (same names) |

## Applied Patterns

- [Two-toucher DFB → 1P+1C](port_patterns.md#pattern-two-toucher-dfb--assign-1p1c-dual-instance-work-split) (census re-derived; matches the brief): `weight`, `input`, `bias` (dm0 P, compute C); `partial_recv`, `gathered_val`, `gathered_ind`, `softmax_mask`, `bcast_scaler` (dm1 P, compute C); `local_out`, `topk_val`, `final_out` (compute P, dm1 C).
- [Sync-free / single-ended → self-loop](port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb): `index`, `dispatch` (dm1, DM self-loop — Gen1-only shape); `intermed_val`, `intermed_ind`, `softmax_tmp`, `reduce_scalar` (compute). Shared accessor name per pair.
- [Pass DFB handles directly to LLKs](port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers): every compute LLK call site.
- No multi-binding flag, no conditional bindings, no varargs, no aliasing, no borrowed memory.

## Deferred / Flagged

- New findings during planning: none that block. Recipe/header naming drift noted in the report (the headers have `ComputeHardwareConfig` / `DataMovementHardwareConfig` with `config_1xx`, not `ComputeGen1Config` / `DataMovementGen1Config`; the TTNN DM helpers take no `arch` argument).
