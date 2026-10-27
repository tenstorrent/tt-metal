# Port Plan — experimental/matmul/group_attn_matmul

Port plan for `ttnn/cpp/ttnn/operations/experimental/matmul/group_attn_matmul`, ported from the `ProgramDescriptor`
API to Metal 2.0. Written during the inventory and planning steps; committed alongside the port for review.
Line references into the legacy code are against `origin/main` @ `0df0dde3316` (the pre-port revision, the one the
audit and brief also cite); `factory:` is `device/group_attn_matmul_program_factory.cpp`, `reader:` / `writer:` /
`compute:` are the three kernel files under `device/kernels/`.

## Legacy Inventory

*Filled in during the inventory step.*

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` — `GroupAttnMatmulProgramFactory::create_descriptor` returns a
  `ProgramDescriptor` (`factory:19`); no `override_runtime_arguments`. The factory lives in the `program_factory_t`
  variant (`device/group_attn_matmul_device_operation.hpp:23`), so no direct-descriptor conversion is needed.
- Variants: single.
- Custom `compute_program_hash`: none — default reflection-based hash (the former custom hash was removed by
  PR #47232; two stale comments still refer to it, see Flags).

*(The Metal 2.0 factory concept the port targets was chosen during the audit — see the brief's TTNN factory
analysis section. Carried forward in the [TTNN ProgramFactory](#ttnn-programfactory) section below.)*

### Kernels
| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_mcast_transformer_group_attn_matmul.cpp` (own) | `all_device_cores` = full device compute grid (`factory:60-62`, `:284`) | `[0] transpose_hw_bool`, `[1] row_major`, `[2] out_subblock_w`, then `TensorAccessorArgs(*src1_buffer)` tail (`:248-253`) | none | per core (`:358-400`), 33 scalars + 2 vectors: `[0] has_work_for_mcast_kv_heads`, `[1] has_work_for_q_heads`, `[2] src1_buffer (Buffer*)`, `[3] Mt`, `[4] Nt`, `[5] KV_HEADS`, `[6] in1_CKtNt`, `[7] in1_CKtNt*32`, `[8] num_output_blocks_per_core`, `[9] 0 (in1_start_id)`, `[10] in0_block_w`, `[11] out_block_w`, `[12] in1_num_subblocks`, `[13] in1_num_blocks`, `[14] in1_block_num_tiles`, `[15] Nt_bytes`, `[16] in1_block_w_tile_bytes`, `[17] out_last_subblock_w`, `[18] in1_last_block_w_tile_read_bytes`, `[19] in1_last_block_addr_skip`, `[20..23]` mcast dest NOC start x/y, end x/y (swapped when the reader NOC is NOC_1), `[24] in1_mcast_num_dests`, `[25] mcast_num_cores_for_core`, `[26] mcast_num_cores`, `[27] sender semaphore id (0)`, `[28] receiver semaphore id (1)`, `[29] in1_block_num_tiles*in1_single_tile_size`, `[30] i (sender id)`, `[31] num_x`, `[32] num_y`, then `in1_mcast_sender_noc_x[0..num_x)`, `in1_mcast_sender_noc_y[0..num_y)` (`:396-399`) | none | `IN1_SHARDED=1` when `in1_is_sharded` (`:268-270`) | O2 (resolved; field absent) | `DataMovementConfigDescriptor{RISCV_1, reader_noc}` (`:287-290`), `reader_noc = preferred_noc_for_dram_read(arch)` (`:275`) = NOC_0 on every arch today |
| writer | `device/kernels/dataflow/writer_transformer_group_attn_matmul.cpp` (own) | `all_device_cores` (`:297`) | `[0] output_cb_index (c_5)`, `[1] out_subblock_w`, `[2] intermediate_num_tiles`, then `TensorAccessorArgs(*src0_buffer)` tail, then `TensorAccessorArgs(*dst_buffer)` tail (`:255-261`) | none | per core (`:402-424`): `[0] has_work_for_q_heads`, `[1] src0_buffer (Buffer*)`, `[2] dst_buffer (Buffer*)`, `[3] Mt`, `[4] Kt`, `[5] Nt`, `[6] MtKt`, `[7] num_output_blocks_per_core`, `[8] num_blocks_written*MtKt`, `[9] num_blocks_written*MtNt`, `[10] in0_block_w`, `[11] in1_num_subblocks`, `[12] in1_num_blocks`, `[13] MtNt`, `[14] bfloat16_row_bytes`, `[15] bfloat16_Nt_bytes`, `[16] bfloat16_last_row_bytes_read` | none | `IN0_SHARDED=1` when `in0_is_sharded`, `OUT_SHARDED=1` when `output_is_sharded` (`:265-273`) | O2 (resolved; field absent) | `DataMovementConfigDescriptor{RISCV_0, writer_noc}` (`:300-303`), `writer_noc` = the other NOC (`:276-277`) = NOC_1 today |
| compute | `device/kernels/compute/transformer_group_attn_matmul.cpp` (own) | `all_device_cores` (`:310`) | `[0] transpose_hw_bool`, `[1] out_subblock_w`, `[2] out_subblock_num_tiles`, `[3] intermediate_num_tiles` (`:311-316`) | none | per core (`:426-444`): `[0] has_work_for_q_heads`, `[1] num_output_blocks_per_core`, `[2] Mt`, `[3] kv_heads_id*in1_block_num_tiles_per_kv_heads`, `[4] (KV_HEADS-kv_heads_id)*in1_block_num_tiles_per_kv_heads`, `[5] in0_block_w`, `[6] out_subblock_h (1)`, `[7] in1_num_subblocks (1)`, `[8] in1_num_blocks`, `[9] in0_block_num_tiles`, `[10] in1_block_num_tiles`, `[11] MtNt`, `[12] in0_subblock_num_tiles`, `[13] in1_per_core_w` | none | none | **O3** (resolved; field absent on a `ComputeConfigDescriptor`) | `ComputeConfigDescriptor{.math_fidelity, .fp32_dest_acc_en}` (`:317-320`); `math_approx_mode` / `dst_full_sync_en` resolved at `:34-35` but **not** set (descriptor defaults `false` / `false`); `packer_l1_acc` resolved, no counterpart; no `unpack_to_dest_mode`, no `bfp8_pack_precise` |

*(`opt_level` is the legacy kernel's **resolved** level, not what the source literally says — `grep -n opt_level`
on the factory returns nothing, so the DM kernels resolve to `O2` and the compute kernel to `O3`.)*

Per-core flag values (`:345-356`, cores walked in `grid_to_cores_with_noop` order over the whole device grid,
`:332-337`): `has_work_for_q_heads = i < Q_HEADS`; `has_work_for_mcast_kv_heads = i < num_active_cores`
(`num_active_cores = max(Q_HEADS, 32)`, `:97`); `num_output_blocks_per_core = 1` for `i < num_cores`, else `0`.
Three node classes: **Q** (`i < Q_HEADS`: all three kernels work), **M** (`Q_HEADS <= i < 32`, only when
`Q_HEADS < 32`: the reader runs the full mcast protocol, compute returns at `compute:21`, writer at `writer:16-18`),
**I** (`i >= num_active_cores`: all three return at once).

### CBs
All six on `all_device_cores`; none sets `.address_offset`, `.global_circular_buffer` or a `.tile`.

| index | total_size | core_ranges | data_format | page_size | tile (if set) | `.buffer` |
|---|---|---|---|---|---|---|
| `c_0` (in0) — `:163-176` | `cb0_num_input_tiles * in0_single_tile_size`, `cb0_num_input_tiles = in0_is_sharded ? a.shard numel / TILE_HW : in0_block_w (= Kt)` | all | `in0_data_format` | `in0_single_tile_size` | unset | `src0_buffer` when `in0_is_sharded` |
| `c_1` (in1) — `:178-190` | `2 * in1_block_num_tiles * in1_single_tile_size` | all | `in1_data_format` | `in1_single_tile_size` | unset | none |
| `c_2` (in2) — `:192-205`, **only when `in1_is_sharded`** | `b.shard numel / TILE_HW * in1_single_tile_size` | all | `in1_data_format` | `in1_single_tile_size` | unset | `src1_buffer` |
| `c_3` (intermed0) — `:207-219` | `2 * intermediate_num_tiles * interm_single_tile_size` | all | `interm_data_format` (`fp32_dest_acc_en && in0 == Float32 ? Float32 : Float16_b`, `:39-41`) | `interm_single_tile_size` | unset | none |
| `c_4` (intermed1) — `:221-230` | `MtNt * interm_single_tile_size` | all | `interm_data_format` | `interm_single_tile_size` | unset | none |
| `c_5` (out) — `:232-245` | `num_output_tiles * output_single_tile_size`, `num_output_tiles = output_is_sharded ? output.shard numel / TILE_HW : MtNt` | all | `output_data_format` | `output_single_tile_size` | unset | `dst_buffer` when `output_is_sharded` |

### Semaphores
| id | core_type | core_ranges | initial_value |
|---|---|---|---|
| 0 (`in1_mcast_sender_semaphore_id`, `:140`, `:142-147`) | WORKER | `all_device_cores` | `INVALID` (= 0) |
| 1 (`in1_mcast_receiver_semaphore_id`, `:141`, `:148-153`) | WORKER | `all_device_cores` | `INVALID` (= 0) |

Kernel side: `Semaphore<> sender_sem(id)` / `receiver_sem(id)` from the two id RTAs (`reader:54-55`);
`receiver_sem.set` / `.wait` / `.set_multicast(noc, …)` (`:88`, `:295-301`, `:312`, `:320`), `sender_sem.wait` /
`.set` / `.up(noc, x, y, 1)` (`:234-235`, `:315-316`).

### Tensor accessors
| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| reader `TensorAccessor(in1_args, src1_addr)` (`reader:83`, inside `#ifndef IN1_SHARDED` `:81-84`; CTA tail `factory:253`, kernel `TensorAccessorArgs<3>()` `reader:82`) | `input_tensor_b` (`b`) | reader RTA 2 = `src1_buffer` (`Buffer*`, `:361`) |
| writer `TensorAccessor(in0_args, src0_addr)` (`writer:65`, inside `#ifndef IN0_SHARDED` `:64-66`; CTA tail `factory:260`, kernel `TensorAccessorArgs<3>()` `writer:62`) | `input_tensor_a` (`a`) | writer RTA 1 = `src0_buffer` (`Buffer*`, `:406`) |
| writer `TensorAccessor(out_args, dst_addr)` (`writer:71`, inside `#ifndef OUT_SHARDED` `:70-72`; CTA tail `factory:261`, kernel `TensorAccessorArgs<in0_args.next_compile_time_args_offset()>()` `writer:69`) | `output` (`tensor_return_value`) | writer RTA 2 = `dst_buffer` (`Buffer*`, `:407`) |

No accessor passes a 3rd (page-size) argument. The `Buffer*` slots and the CTA tails are emitted in every
config, including the sharded ones where the kernel never instantiates the accessor.

### Work split
- Driver: `split_work_to_cores(operation_attributes.compute_with_storage_grid_size, num_active_cores, row_major)`
  (`:98-106`), `num_active_cores = max(Q_HEADS, TILE_HEIGHT)`.
- num_cores: `num_cores`; `all_cores` (its bounding box drives the per-core walk, `:331-337`).
- core_group_1: `core_group_1`, count_per_core: `num_output_blocks_per_core_group_1` — forced to `1` by the
  `TT_FATAL` at `:107-110`.
- core_group_2: `core_group_2`, count_per_core: `num_output_blocks_per_core_group_2` — forced to `0`.
- Mcast topology (`:112-137`): receivers = first `Q_HEADS` cores (`num_cores_to_corerangeset`), their bounding box
  gives the mcast rectangle; senders = the first 32 cores, whose grid extents give the two NOC coordinate vectors.

### Shared kernels
none — the three kernels are owned by this op, bound by no other factory on `main` (`grep -rl` over
`ttnn/cpp/ttnn/operations/` finds only this factory), and no `_metal2` fork exists beside any of them. Single
factory, so no intra-op sharing. All three convert in place.

### Flags
- `group_attn_matmul_program_factory.hpp:15-20`, `factory:58-59` and `:155-158` carry comments claiming
  `padded_shape` is folded into `compute_program_hash()`; that custom hash was removed by PR #47232 (the op is on
  the default reflection hash, which keys on the full `TensorSpec`s). Not carried into the port; the substance is
  restated where the shape-sized DFBs are declared.
- `writer:59` `const uint32_t in1_tile_bytes = get_tile_size(cb_id_in1);` is unused and the writer binds no `in1`
  — dropped (audit Question 2, decided by the invoker).
- `reader:71`, `:75`, `:123`, `:239` reference CB 2 outside `#ifdef IN1_SHARDED` although the CB exists only in
  that config — gated in the port (see Applied Patterns).
- No unreferenced kernel files in the directory; no descriptor type outside the audit's scan (`KernelDescriptor`
  × 3, `CBDescriptor` × 6, `SemaphoreDescriptor` × 2).

## TTNN ProgramFactory

*Filled in during the planning step. The concept itself was chosen in the audit; this section carries it forward.*

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept`
- **Custom `compute_program_hash`**: none
- **Implementation notes**: the one public entry point always passes `num_tokens = nullopt`, `transpose_hw = nullopt`;
  the factory is ported as written (the `transpose_hw` CTA and the `in1_Kt` / `in1_KtNt` arithmetic stay) but that
  branch cannot be exercised. `tensor_args_t` carries an optional `preallocated_output`; the `OUTPUT`
  `TensorParameter` / `TensorArgument` is built from `tensor_return_value`, which is that tensor when supplied,
  exactly as the legacy `dst_buffer` was.

## Planned Spec Shape

*Filled in during the planning step.* 1:1 with legacy except for the M-node placement split below (a decision the
invoker made on the audit's Question 1).

- KernelSpecs: four — `READER` (`reader`), `READER_MCAST_ONLY` (`reader_mcast_only`, the same reader source, see
  Preserved Multiplicity), `WRITER` (`writer`), `COMPUTE` (`compute`).
- DataflowBufferSpecs: six — `IN0` (`in0`), `IN1` (`in1`), `IN2` (`in2`, declared only when `in1_is_sharded`),
  `INTERMED0` (`intermed0`), `INTERMED1` (`intermed1`), `OUT` (`out`); `entry_size` / `num_entries` /
  `data_format_metadata` copied from the legacy `page_size` / `total_size ÷ page_size` / `data_format`; no
  `tile_format_metadata` (legacy `.tile` unset); `borrowed_from = INPUT_A` on `IN0` when `in0_is_sharded`,
  `INPUT_B` on `IN2`, `OUTPUT` on `OUT` when `output_is_sharded`; no advanced options.
- SemaphoreSpecs: two — `IN1_MCAST_SENDER_SEM` (`in1_mcast_sender`), `IN1_MCAST_RECEIVER_SEM`
  (`in1_mcast_receiver`), `target_nodes = all_device_cores` (legacy placement; the remote `up` / `set_multicast`
  rely on one address on every node); initial value 0 (= legacy `INVALID`), nothing to carry.
- TensorParameters: three — `INPUT_A` (`input_a`, `a.tensor_spec()`), `INPUT_B` (`input_b`, `b.tensor_spec()`),
  `OUTPUT` (`output`, `output.tensor_spec()`); all three declared in every config (a `borrowed_from` counts as a
  use); strict matching.
- WorkUnitSpecs: two — `main`: kernels `{READER, WRITER, COMPUTE}`, `target_nodes = all_device_cores` minus the
  M nodes (Q + I nodes; idle cores keep running all three kernels with their legacy flag values, as before);
  `mcast_only`: kernels `{READER_MCAST_ONLY}`, `target_nodes` = exactly the M nodes (`Q_HEADS <= i < 32` in the
  legacy core order), emitted only when that set is non-empty (`Q_HEADS < 32`). Both are built from the same
  per-core loop that assigns the legacy flags, so the classification cannot drift from the run-arg values.
- Op-owned tensors: none.

### DFB bindings (endpoint census re-derived from the kernels, per node class)

| DFB | reader (Q + I nodes) | reader_mcast_only (M nodes) | compute (Q + I) | writer (Q + I) | disposition |
|---|---|---|---|---|---|
| `IN0` | — | — | CONSUMER (`wait_front` `compute:74`, `pop_front` `:174`), accessor `in0` | PRODUCER (`reserve_back` `writer:98`, `push_back` `:113`), accessor `in0` | legal 1:1 |
| `IN1` | PRODUCER (`reserve_back` `reader:168`, `get_write_ptr` `:169`, mcast destination `:243-282`, `push_back` `:323`), accessor `in1` | PRODUCER **and** CONSUMER (`push_back` `:327`, `pop_front` `:328`), accessor `in1` | CONSUMER (`wait_front` `compute:79`, `pop_front` `:80` / `:142`), accessor `in1` | — | 1:1 on Q + I; reader self-loop on M (per node: exactly one producer and one consumer instance everywhere) |
| `IN2` (only `IN1_SHARDED`) | PRODUCER + CONSUMER (`get_read_ptr` peek only, `reader:123`), accessor `in2` | PRODUCER + CONSUMER, accessor `in2` | — | — | DM self-loop on the one touching kernel, on both reader specs over disjoint nodes |
| `INTERMED0` | — | — | PRODUCER (`reserve_back` `compute:145`, `push_back` `:151`), accessor `intermed0` | CONSUMER (`wait_front` `writer:131`, `get_read_ptr` `:132`, `pop_front` `:141`), accessor `intermed0` | legal 1:1 |
| `INTERMED1` | — | — | CONSUMER (`wait_front` `compute:162`, `pop_front` `:171`), accessor `intermed1` | PRODUCER (`reserve_back` `writer:115`, `get_write_ptr` `:116`, loopback writes `:133-139`, `push_back` `:153`), accessor `intermed1` | legal 1:1 |
| `OUT` | — | — | PRODUCER (`reserve_back` `compute:164`, `push_back` `:169`), accessor `out` | CONSUMER (`wait_front` `writer:156` / `:176`, `pop_front` `:166`), accessor `out` | legal 1:1 |

No `allow_instance_multi_binding` anywhere. The compute and writer kernels are not placed on M nodes; today they
return at their first statement there, so nothing observable changes. The DFBs they bind (`IN0`, `INTERMED0`,
`INTERMED1`, `OUT`) are therefore not instantiated on M nodes (legacy allocated all six CBs on every core).

### Tensor bindings
| TensorParameter | kernel(s) | accessor | condition | kernel-side construction |
|---|---|---|---|---|
| `INPUT_B` | reader, reader_mcast_only | `src1` | `!in1_is_sharded` (sharded: reached through `IN2.borrowed_from`) | `TensorAccessor(tensor::src1)` inside the existing `#ifndef IN1_SHARDED` (replaces `reader:82-83`) |
| `INPUT_A` | writer | `src0` | `!in0_is_sharded` (sharded: `IN0.borrowed_from`) | `TensorAccessor(tensor::src0)` inside the existing `#ifndef IN0_SHARDED` (replaces `writer:62`, `:65`) |
| `OUTPUT` | writer | `dst` | `!output_is_sharded` (sharded: `OUT.borrowed_from`) | `TensorAccessor(tensor::dst)` inside the existing `#ifndef OUT_SHARDED` (replaces `writer:69`, `:71`) |

### Semaphore bindings
Both semaphores on both reader specs: `IN1_MCAST_SENDER_SEM` as `in1_mcast_sender`, `IN1_MCAST_RECEIVER_SEM` as
`in1_mcast_receiver`; kernel `Semaphore<> sender_sem(sem::in1_mcast_sender)` / `receiver_sem(sem::in1_mcast_receiver)`
(replaces the two id RTAs, `reader:54-55`).

### Named arguments
| kernel | CTAs (named) | RTAs (named) | varargs |
|---|---|---|---|
| reader / reader_mcast_only (identical schema) | `transpose_hw`, `row_major`, `out_subblock_w` | `has_work_for_mcast_kv_heads`, `has_work_for_q_heads`, `Mt`, `Nt`, `num_kv_heads`, `in1_CKtNt`, `in1_CKtNt_mul_32`, `blocks`, `in1_start_id`, `in0_block_w`, `out_block_w`, `in1_num_subblocks`, `in1_num_blocks`, `in1_block_num_tiles`, `Nt_bytes`, `in1_block_w_tile_bytes`, `out_last_subblock_w`, `in1_last_block_w_tile_read_bytes`, `in1_last_block_addr_skip`, `in1_mcast_dest_noc_start_x`, `in1_mcast_dest_noc_start_y`, `in1_mcast_dest_noc_end_x`, `in1_mcast_dest_noc_end_y`, `in1_mcast_num_dests`, `in1_mcast_num_cores`, `in1_mcast_grid_size`, `in1_mcast_sender_size_bytes`, `in1_mcast_sender_id`, `in1_mcast_sender_num_x`, `in1_mcast_sender_num_y` (the kernel's own local names, `reader:17-60`) | `num_runtime_varargs = num_x + num_y`; per node `[noc_x_0 … noc_x_{num_x-1}, noc_y_0 … noc_y_{num_y-1}]`; kernel `get_vararg(x)` / `get_vararg(in1_mcast_sender_num_x + y)` (replaces the `get_arg_addr` pointer walk, `reader:61-64`, `:96-97`, `:107-108`) |
| writer | `out_subblock_w`, `intermediate_num_tiles` | `has_work_for_q_heads`, `Mt`, `Kt`, `Nt`, `MtKt`, `blocks`, `in0_start_id`, `out_start_id`, `in0_block_w`, `in1_num_subblocks`, `in1_num_blocks`, `out_num_tiles`, `bfloat16_row_bytes`, `bfloat16_Nt_bytes`, `bfloat16_last_row_bytes_read` (`writer:15-39`) | none |
| compute | `transpose_hw`, `out_subblock_w`, `out_subblock_num_tiles`, `intermediate_num_tiles` | `has_work_for_q_heads`, `batch`, `Mt`, `num_kv_heads_skip`, `num_kv_heads_remaining`, `in0_block_w`, `out_subblock_h`, `in1_num_subblocks`, `in1_num_blocks`, `in0_block_num_tiles`, `in1_block_num_tiles`, `out_num_tiles`, `in0_subblock_num_tiles`, `in1_per_core_w` (`compute:20-39`) | none |

Every named RTA keeps its legacy per-core value (including the constants `in1_start_id = 0`, `in1_num_subblocks = 1`,
`out_subblock_h = 1` and the zero `blocks` / `batch` on idle cores). The vararg vectors are identical on every node;
they stay per-node RTAs (converting to common runtime varargs would change dispatch, noted for a later pass).

### Hardware configuration
- reader, reader_mcast_only: `ttnn::create_reader_datamovement_config()` — the legacy resolved triple
  (`RISCV_1`, `preferred_noc_for_dram_read(arch)` = `NOC_0`, dedicated NOC) equals the reader default on every
  arch today. The mcast destination start / end swap (`factory:382-385`) is keyed off the NOC the reader actually
  gets, read back from that config, so the two cannot drift apart.
- writer: `ttnn::create_writer_datamovement_config()` — resolved triple (`RISCV_0`, the other NOC = `NOC_1`,
  dedicated) equals the writer default.
- compute (Style A with dropped fields): `to_compute_hardware_config(compute_kernel_config)` then, on the returned
  config, force `sfpu_precision_mode = Precision::Precise` and `double_buffer_dest = true` (the legacy descriptor
  never set `math_approx_mode` / `dst_full_sync_en`, so the op ran at the descriptor defaults regardless of the
  caller's config); `config_1xx.bfp_pack_precision_mode` stays default; `unpack_modes` gets a
  `{<dfb>, UnpackMode::UnpackToSrc}` entry (legacy `Default`) for each DFB the compute kernel consumes whose format
  is `Float32` — `IN0` / `IN1` when the input dtype is Float32, `INTERMED1` when `interm_data_format == Float32` —
  because the validator requires an explicit entry for every Float32 DFB consumed under `enable_32_bit_dest`.
  `INTERMED0` and `OUT` are produced, not consumed, by the compute kernel.
- `compiler_options.opt_level`: compute `O3` explicitly; DM kernels default (`O2`).
- `compiler_options.defines`: `IN1_SHARDED` on both reader specs when `in1_is_sharded`; `IN0_SHARDED` /
  `OUT_SHARDED` on the writer when `in0_is_sharded` / `output_is_sharded`.

## Preserved Multiplicity

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| reader (`factory:279-290`, one descriptor over `all_device_cores`) | `READER`, `READER_MCAST_ONLY` of `reader_mcast_transformer_group_attn_matmul.cpp` (identical CTAs, defines, run-arg schema, varargs, semaphore bindings, hw config) | `main` (Q + I nodes: `READER` with `WRITER` + `COMPUTE`), `mcast_only` (M nodes: `READER_MCAST_ONLY` alone) | `IN1`: `READER` PRODUCER, `READER_MCAST_ONLY` PRODUCER + CONSUMER (`COMPUTE` CONSUMER on the main nodes); `IN2` (sharded `in1`): both PRODUCER + CONSUMER; `INPUT_B` bound on both when interleaved |

This is not a legacy work-split multiplicity (legacy had one descriptor per source); it is the invoker's decision on
the audit's Question 1 — the per-node endpoint roles of `in1` differ between the M nodes and the rest, and bindings
follow `KernelSpec`s, so the M nodes get a second same-source `KernelSpec` in a disjoint `WorkUnitSpec` rather than
`allow_instance_multi_binding`. The `WorkUnitSpec` rules it must satisfy — no overlap in `target_nodes`, at least one
kernel, at most one compute and two DM kernels per work unit — hold by construction.

## Dropped Plumbing

For each legacy RTA / CTA that disappears in the port:

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `factory:361` → reader RTA slot 2 | `src1_buffer` (`Buffer*` binding) | `TensorBinding{INPUT_B, "src1"}` on both reader specs when `!in1_is_sharded`; kernel `TensorAccessor(tensor::src1)` |
| `factory:253` → reader CTA tail; `reader:82` | `TensorAccessorArgs(*src1_buffer).append_to(...)`; `TensorAccessorArgs<3>()` | the binding mechanism (host packs layout metadata; kernel builds from the token) |
| `factory:248-252` → reader CTA slots 0-2; `reader:66-68` | positional `transpose_hw_bool`, `row_major`, `out_subblock_w` | named CTAs `transpose_hw`, `row_major`, `out_subblock_w` (`get_arg(args::…)`) |
| `factory:389-390` → reader RTA slots 27-28; `reader:54-55` | semaphore ids `0`, `1` | `SemaphoreBinding{IN1_MCAST_SENDER_SEM, "in1_mcast_sender"}`, `{IN1_MCAST_RECEIVER_SEM, "in1_mcast_receiver"}`; kernel `Semaphore<>(sem::…)` |
| `factory:396-399` → reader RTA tail; `reader:61-64` | two appended coordinate vectors read through `get_arg_addr` pointers | `advanced_options.num_runtime_varargs = num_x + num_y`, per-node `runtime_varargs = [x…, y…]`; kernel `get_vararg(i)` |
| `factory:358-394` → reader RTA slots 0-1, 3-26, 29-32; `reader:17-60` | positional scalars | the 30 named RTAs listed above |
| `reader:70-71`, `:74-75` | hardcoded CB ids `1`, `2` + `CircularBuffer` objects | `DataflowBuffer dfb_in1(dfb::in1)`; `dfb_in2(dfb::in2)` under `#ifdef IN1_SHARDED` |
| `factory:406` → writer RTA slot 1 | `src0_buffer` (`Buffer*` binding) | `TensorBinding{INPUT_A, "src0"}` when `!in0_is_sharded`; kernel `TensorAccessor(tensor::src0)` |
| `factory:407` → writer RTA slot 2 | `dst_buffer` (`Buffer*` binding) | `TensorBinding{OUTPUT, "dst"}` when `!output_is_sharded`; kernel `TensorAccessor(tensor::dst)` |
| `factory:260-261` → writer CTA tails; `writer:62`, `:69` | `TensorAccessorArgs(*src0_buffer)`, `TensorAccessorArgs(*dst_buffer)`; `TensorAccessorArgs<3>()`, `TensorAccessorArgs<in0_args.next_compile_time_args_offset()>()` | the binding mechanism |
| `factory:256` → writer CTA slot 0; `writer:41` | `output_cb_index` (magic CB index) | `DFBBinding{OUT, "out", CONSUMER}`; kernel `dfb::out` |
| `factory:257-258` → writer CTA slots 1-2; `writer:42-43` | positional `out_subblock_w`, `intermediate_num_tiles` | named CTAs |
| `factory:402-424` → writer RTA slots 0, 3-16; `writer:15-39` | positional scalars | the 15 named RTAs listed above |
| `writer:45-49`, `:52-55` | hardcoded CB ids `c_0`, `c_1`, `c_3`, `c_4` + objects | `dfb::in0`, `dfb::intermed0`, `dfb::intermed1` (+ `dfb::out`); the `c_1` constant drops with the dead `in1_tile_bytes` (`writer:59`) |
| `factory:311-316` → compute CTA slots 0-3; `compute:42-45` | positional scalars | named CTAs `transpose_hw`, `out_subblock_w`, `out_subblock_num_tiles`, `intermediate_num_tiles` |
| `factory:426-443` → compute RTA slots 0-13; `compute:20-39` | positional scalars | the 14 named RTAs listed above |
| `compute:48-58` | hardcoded CB ids `c_0`, `c_1`, `c_3`, `c_4`, `c_5` + objects | `dfb::in0`, `dfb::in1`, `dfb::intermed0`, `dfb::intermed1`, `dfb::out` |
| `factory:140-153` | `SemaphoreDescriptor` × 2 with ids | `SemaphoreSpec` × 2 (ids are host-side names) |

## Applied Patterns

- Conditional / optional resource bindings: `IN2` (`DataflowBufferSpec`, both reader bindings) and the `INPUT_B`
  tensor binding flip with `in1_is_sharded`, paired with the `IN1_SHARDED` define; `INPUT_A` with `IN0_SHARDED`;
  `OUTPUT` with `OUT_SHARDED`. Kernel side: the legacy accessors already sit inside the matching `#ifndef`; the
  `in2` declaration / object / peek (`reader:71`, `:75`, `:123`) move under `#ifdef IN1_SHARDED`, and the
  `mcast_in1_to_local_cb` branch that consumes the peeked address (`:238-254`) is gated the same way (its condition
  is only ever set under `IN1_SHARDED`, `:129-132`). Preprocessor-level only, no logic change.
- Sync-free and single-ended CBs → self-loop DFB: `IN2` (one toucher, raw `get_read_ptr` peek feeding NoC reads)
  bound PRODUCER + CONSUMER on each reader spec; `IN1` on the M nodes (the mcast-only reader both pushes and pops).
- Two `KernelSpec`s of one source over disjoint node sets (the *Why wrong* paragraph of *Anti-pattern: Demoting
  per-group CTA to RTA*): `READER` / `READER_MCAST_ONLY`, no flag.
- Pass DFB handles directly to LLKs: every `compute_kernel_hw_startup` / `matmul_init` / `matmul_tiles` /
  `pack_untilize_*` / `reconfig_data_format_srca` / `pack_reconfig_data_format` / `tilize_*` call takes `dfb::name`
  (`compute:64-172`).
- Kernel-side whitelist rule 7 (metadata via the object): `reader:78` `in1_tile_bytes`, `writer:61` `in0_tile_bytes`,
  `writer:68` `out_tile_bytes` are runtime `const` → `dfb.get_tile_size()` getters. `writer:59` is dropped.
- Transfers stay as-is (recipe rule 1): the reader's sharded `in1` staging copy (`reader:189-196`) and the writer's
  row gather (`writer:133-139`) keep their `UnicastEndpoint` from `my_x[noc.get_noc_id()]` / `my_y[…]` and their
  `CoreLocalMem` operands; the multicasts (`reader:243-282`) keep the `CoreLocalMem` source and take the
  `DataflowBuffer` as destination; only the peeks and the destination object change spelling.
- Caution: avoid varargs — one genuine indexed-collection block (the sender coordinate tables, `reader:96-97`,
  `:107-108`); the two counts stay named. Reported in the port report.

## Deferred / Flagged

- **Validator rule known at planning time.** The DFB validator
  (`tt_metal/impl/metal2_host_api/program_spec/validation/resource/dfb.cpp:605-625`) requires, for any DFB that
  some `KernelSpec` self-loops, that the set of PRODUCER `KernelSpec`s equal the set of CONSUMER `KernelSpec`s. For
  `IN1` the planned shape has producers `{READER, READER_MCAST_ONLY}` and consumers `{COMPUTE, READER_MCAST_ONLY}`,
  so the validator will reject it on every `Q_HEADS < 32` configuration (the `mcast_only` work unit is not emitted
  otherwise). The invoker's instruction is to build this shape, record the exact rejection and the analysis as a
  handoff point, and not fall back to `allow_instance_multi_binding` on my own — so the port proceeds as planned and
  the `METAL2_PORT_REPORT.md` carries the outcome. The in-tree matmul 2D mcast factory avoids the same rule with a
  second same-geometry DFB (`in0_relay`) self-looped by its no-work kernel, which only works because that pair is
  declared first so both land at the allocator base; `in1` here sits after `in0`, so that trick does not transfer
  unchanged (analysed in the report).
- The M nodes no longer host `IN0` / `INTERMED0` / `INTERMED1` / `OUT`. The mcast relies on `in1` having one L1
  address on every node; with a single `IN1` spec bound over the union of both work units that holds (one circular
  buffer over all of its nodes), but any shape that gives the M nodes a *different* buffer must keep the two at the
  same offset.
- The KV-cache attributes (`num_tokens`, `transpose_hw`) are unreachable from the public API; the `transpose_hw`
  branch of the port cannot be tested.
