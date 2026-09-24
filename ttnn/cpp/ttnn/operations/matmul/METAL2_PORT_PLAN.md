# Port Plan — `matmul` / `SparseMatmulMultiCoreReuseMcast1DProgramFactory`

Port plan for `ttnn/cpp/ttnn/operations/matmul` (sparse sub-tree), ported from the
`ProgramDescriptor` API to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

**Unit of port:** one ProgramFactory — `SparseMatmulMultiCoreReuseMcast1DProgramFactory`, the only
alternative in `SparseMatmulDeviceOperation::program_factory_t`. So this port completes the whole
device-operation. `MatmulDeviceOperation`'s factories are a separate unit and untouched.

Line references are to `device/sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp`
at the pre-port revision unless another file is named.

## Legacy Inventory

### Legacy factory shape
- Concept: **`ProgramDescriptorFactoryConcept`** — a lone
  `static tt::tt_metal::ProgramDescriptor create_descriptor(...)`
  (`…mcast_1d_optimized.hpp:24`). No `cached_program_t`, no `create`, no
  `override_runtime_arguments`.
- Where the factory methods live: in a **`program_factory_t` variant** (`…device_operation.hpp:21`),
  one alternative. **Not** the direct-descriptor shape → `ttnn_factory.md` exception 3 does not
  apply.
- Variants: single.
- Custom `compute_program_hash`: **none** — declaration and definition are both *commented out*
  (`…device_operation.hpp:33`, `…device_operation.cpp:504`). Default reflection hash. Those comment
  lines are device-op-class code and stay byte-identical.

*(The Metal 2.0 factory concept the port targets was chosen during the audit — see the brief's TTNN
factory analysis section. Carried forward in [TTNN ProgramFactory](#ttnn-programfactory) below.)*

### Kernels

Four `KernelDescriptor`s. Every source is **shared** with the dense `MatmulDeviceOperation`
factories and every one already has a checked-in `_metal2` fork with live consumers → **rung 1
(reuse)** for all four (see [Shared kernels](#shared-kernels)).

| unique_id | source (legacy → fork) | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| `in0_sender` | `dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp` (`:494`) → `…_metal2.cpp` | `in0_mcast_sender_cores` = `{start_core}` | 28 slots (`:333`–`:370`) + `TensorAccessorArgs(in0)` (`:371`) + `TensorAccessorArgs(sparsity)` (`:372`) + `num_batch_compute` (`:376`) | `cb_in0`=c_0, `cb_in0_sharded`=c_2, `cb_sparsity`=c_6, `num_active` (`:498`–`:503`) | 8 slots (`:769`–`:792`); slots 0 and 7 are `Buffer*` bindings | none | `SKIP_MCAST` iff `in0_mcast_receiver_num_cores == 1` (`:470`) | absent → **O2** | `DataMovementConfigDescriptor{RISCV_0, in0_noc}` (`:505`) |
| `in0_receiver` | `dataflow/reader_bmm_tile_layout_in0_receiver.cpp` (`:509`) → `…_metal2.cpp` | `in0_mcast_receivers` (empty when `num_cores == 1`) | 8 slots (`:436`–`:448`) | `cb_in0`=c_0 (`:513`) | 2 slots (`:796`–`:803`) | none | none | absent → **O2** | `DataMovementConfigDescriptor{RISCV_0, in0_noc}` (`:516`) |
| `in1_sender_writer` | `dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp` (`:521`) → `…_metal2.cpp` | `all_cores_with_work` | 33 slots (`:378`–`:426`) + `TensorAccessorArgs(in1)` (`:429`), `(in1_sparsity)` (`:431`), `(out)` (`:432`), `()` bias placeholder (`:433`) | `cb_in1`=c_1, `cb_bias`=c_3, `cb_out`=c_4, `cb_sparsity`=c_7, `num_active` (`:525`–`:531`) | 26 slots (`:806`–`:875`); slots 0, 6, 7 are `Buffer*` bindings | none | `SKIP_MCAST` always (`:474`) | absent → **O2** | `DataMovementConfigDescriptor{RISCV_1, in1_noc}` (`:533`) |
| `compute` | `compute/bmm_large_block_zm_fused_bias_activation.cpp` (`:579`) → `…_metal2.cpp` | `all_cores_with_work` | 18 slots (`:544`–`:567`) | `cb_in0`=c_0, `cb_in1`=c_1, `cb_bias`=c_3, `cb_out`=c_4, `cb_intermed0`=c_5, `cb_in0_transposed`=c_10 (`:583`–`:590`) | none | none | `FUSE_ACTIVATION`=0 always; `PACKER_L1_ACC` iff `packer_l1_acc_en`; `FP32_DEST_ACC_EN` iff `fp32_dest_acc_en`; plus `add_stagger_defines_if_needed` + `throttle_mm_perf` (`:454`–`:468`) | absent → **O3** (`ComputeConfigDescriptor`) | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en, dst_full_sync_en, unpack_to_dest_mode, math_approx_mode}` (`:593`–`:598`) |

`opt_level`: `grep -n opt_level` over the factory returns **nothing**, so every level is the
descriptor default — `O2` on the three DM descriptors, **`O3`** on the `ComputeConfigDescriptor`.

`in0_noc = preferred_noc_for_dram_write(arch)` = `NOC_1`; `in1_noc = preferred_noc_for_dram_read(arch)`
= `NOC_0` (`:477`–`:478`; both helpers are arch-invariant today). `noc_mode` unset on all three →
`DM_DEDICATED_NOC`.

> **The processor assignment is the mirror image of the dense factory's** — sparse puts the in0
> sender on `RISCV_0` and the in1 sender/writer on `RISCV_1`; dense does the reverse. Carrying the
> dense values would be a silent perf change with no test to catch it.

### CBs

Six `CBDescriptor`s — but **five or six objects** depending on config, because `c_4`/`c_5` collapse
onto one descriptor when their formats match.

| index | total_size | core_ranges | data_format | page_size | tile (if set) | site |
|---|---|---|---|---|---|---|
| `c_0` in0 | `in0_CB_size` | `all_cores` | `in0_data_format` | `in0_aligned_tile_size` | `in0_tile` | `:615`–`:623` |
| `c_1` in1 | `in1_CB_size` | `all_cores` | `in1_data_format` | `in1_aligned_tile_size` | `in1_tile` | `:635`–`:643` |
| `c_6` sparsity | `sparsity_cb_size` | `all_cores` | `dtype(sparsity)` | `sparsity_cb_size` | *(none)* | `:655`–`:662` |
| `c_7` in1 sparsity / indices | `in1_sparsity_cb_size` | `all_cores` | `dtype(in1_sparsity_tensor)` | `in1_sparsity_cb_size` | *(none)* | `:668`–`:676` |
| `c_5` interm0 | `interm0_CB_size` | `all_cores` | `interm0_data_format` | `interm0_single_tile_size` | `output_tile` | `:684`–`:692` — **formats-differ branch only** |
| `c_4` out | `out_CB_size` | `all_cores` | `output_data_format` | `output_single_tile_size` | `output_tile` | `:703`–`:711` — formats-differ branch |
| `c_4` + `c_5` **aliased** | `out_CB_size` | `all_cores` | resp. | resp. | `output_tile` | `:715`–`:728` — **formats-equal branch**: one descriptor, two `format_descriptors` |

`sparsity_cb_size` = `sparsity.buffer()->aligned_page_size()` (`:653`);
`in1_sparsity_cb_size` = `in1_sparsity_buffer->aligned_page_size()` (`:667`). Both are one page.

No `CBDescriptor` sets `.buffer`, `.tensor`, `.global_circular_buffer` or `.address_offset` —
everything is a plain L1 allocation, so there is **no borrowed memory** anywhere in this factory.

### Semaphores

| id | core_type | core_ranges | initial_value | site |
|---|---|---|---|---|
| 0 | *(default)* | `all_cores` | `INVALID` | `:738`–`:739` |
| 1 | *(default)* | `all_cores` | `INVALID` | `:740`–`:741` |

The ids are hand-assigned (`:294`–`:295`) precisely so the values baked into the sender/receiver
CTAs stayed stable across the PD migration. Under Metal 2.0 the ids go away entirely — the
semaphores become `SemaphoreSpec`s bound by name.

### Tensor accessors

| host site | originating Tensor | RTA/CTA slot (host) | kernel construction |
|---|---|---|---|
| `:371` `TensorAccessorArgs(*in0_buffer)` | input A | in0 sender RTA 0 (`Buffer*`, `:790`) | `TensorAccessor(in0_args, in0_tensor_addr)` — in0 sender `:162` |
| `:372` `TensorAccessorArgs(*sparsity_buffer)` | sparsity mask | in0 sender RTA 7 (`Buffer*`, `:791`) | `TensorAccessor(sparsity_args, sparsity_addr)` — in0 sender `:168` |
| `:429` `TensorAccessorArgs(*in1_buffer)` | input B | in1 sender RTA 0 (`Buffer*`, `:872`) | `TensorAccessor(in1_args, in1_tensor_addr)` — in1 sender `:232` |
| `:431` `TensorAccessorArgs(*in1_sparsity_buffer)` | **indices** in indexed mode, else the sparsity mask (`:135`) | in1 sender RTA 6 (`Buffer*`, `:873`) | `TensorAccessor(sparsity_args, sparsity_addr)` — in1 sender `:249` |
| `:432` `TensorAccessorArgs(*out_buffer)` | output | in1 sender RTA 7 (`Buffer*`, `:874`) | `TensorAccessor(out_args, out_tensor_addr)` — in1 sender `:241` |
| `:433` `TensorAccessorArgs()` | *(none — bias placeholder)* | — | `TensorAccessor(bias_args, in3_tensor_addr)` under `FUSE_BIAS`, never taken |

Every construction is **2-arg**; no page-size third argument anywhere.

### Work split

- Driver: **not** `split_work_to_cores`. The grid is derived from the output block decomposition:
  `num_blocks_y = ceil(Mt / per_core_M)`, `num_blocks_x = ceil(Nt / per_core_N)`,
  `num_blocks_total = num_blocks_y * num_blocks_x` (`:172`–`:174`), and
  `num_cores = num_cores_with_work = num_blocks_total` (`:242`, `:245`).
- `all_cores = num_cores_to_corerangeset_in_subcoregrids(start_core, num_cores, matmul_core_rect, row_major)` (`:248`)
- `all_cores_with_work` — same call with `num_cores_with_work` (`:254`), so **identical to
  `all_cores`** here.
- `in0_mcast_sender_cores` — same call with `in0_sender_num_cores = 1` (`:251`) → `{start_core}`.
- `in0_mcast_receivers` — `num_cores - 1` cores anchored one step right (or down, on a 1-wide grid)
  of `start_core` (`:283`–`:286`); **empty when `num_cores == 1`**.
- `TT_FATAL(num_cores_with_work == in0_mcast_receiver_num_cores, …)` (`:265`) pins the grid
  rectangular, so `all_cores == {start_core} ⊎ in0_mcast_receivers` exactly.

No core groups, no per-group CTA variation → no work-split multiplicity.

### Shared kernels

All four sources are **lent**: they live in matmul's own `device/kernels/` tree *and* the dense
`MatmulDeviceOperation` factories bind them. Census via `grep -rl <filename> ttnn/cpp/ttnn/operations/`,
with build-file / comment / quasar hits discarded:

| kernel | other mainline binders | `_metal2` fork beside it | rung |
|---|---|---|---|
| `…in0_sender_padding.cpp` | `matmul_multicore_reuse_mcast_1d_program_factory.cpp:632,1670`; `…mcast_2d…:2600` | ✓ (bound by ported 1D `:3769`,`:4939` and 2D `:1307`) | **1 — reuse** |
| `…in0_receiver.cpp` | `…mcast_1d…:697`; `…mcast_2d…:2656,2690` | ✓ (bound by ported 1D) | **1 — reuse** |
| `…in1_sender_writer_padding.cpp` | `…mcast_1d…:710,1686`; `…mcast_2d…:2617` | ✓ (bound by ported 1D `:3977`,`:5026` and 2D `:1392`) | **1 — reuse** |
| `bmm_large_block_zm_fused_bias_activation.cpp` | `…mcast_1d…:816,1815`; `…mcast_2d…:2792` | ✓ (bound by ported 1D, 2D, dram_sharded, optimized, batched_hs_dram_sharded) | **1 — reuse** |

**Binding vocabulary inherited from the forks** (this is now the constraint, not a free choice):

| fork | `dfb::` | `tensor::` | `sem::` | `#ifdef`s this port must feed |
|---|---|---|---|---|
| in0 sender | `in0`, `in0_sharded`, `sparsity` | `in0`, `sparsity` | `in0_mcast_sender`, `in0_mcast_receiver` | `SPARSITY` (set), `SKIP_MCAST` (conditional); `IN0_SHARDED`/`EXTRACT_SHARD_SUB_BLOCKS`/`FUSE_OP` unset |
| in0 receiver | `in0` | — | `in0_mcast_sender`, `in0_mcast_receiver` | none |
| in1 sender/writer | `in1`, `out`, `bias`, `sparsity` | `in1`, `out`, `bias`, `sparsity` | `in1_mcast_sender`, `in1_mcast_receiver` | `SPARSITY` (set), `SKIP_MCAST` (set); `FUSE_BIAS`/`BIAS_SHARDED`/`IN1_SHARDED`/`IN1_DRAM_*_SHARDED`/`OUT_SHARDED`/`ENABLE_GLOBAL_CB`/`FUSE_OP_*` unset |
| compute | `in0`, `in1`, `out`, `intermed0`, `intermed0_reload_alias`, `bias`, `in0_transposed` | — | — | `PACKER_L1_ACC`, `FP32_DEST_ACC_EN` (conditional); `FUSE_BIAS`/`MM_PARTIALS_RELOAD_ALIAS`/`IN0_TRANSPOSE_TILE`/`IN1_TRANSPOSE_TILE`/`MATMUL_DRAM_SHARDED`/`SFPU_ACTIVATION`/`PACK_RELU`/`SKIP_COMPUTE` unset |

Named-arg sets per fork are transcribed into [Dropped Plumbing](#dropped-plumbing).

> **The forks are read-only to this port.** Each has live consumers. No fork needs a change: every
> name the sparse path requires is already there, including the whole `SPARSITY` region. **This port
> is the first to set `SPARSITY`** — the region is checked in but has never run through a Metal 2.0
> spec.

### Flags

- **`in1`'s sparsity slot is a different tensor from `in0`'s in indexed mode** (`:135`). Needs two
  DFB specs always, and two `TensorParameter`s when `use_indices`.
- Five `->address()` reads (`:771`, `:782`, `:809`, `:818`, `:822`) are dead — overwritten by the
  `Buffer*` assignment at the same index.
- Five trailing in1 RTA slots (indices 21–25, `:864`–`:868`) are dead in this configuration: the
  kernel's last read is `last_num_blocks_w_dim` at index 20. Indices 18–19 are the bias placeholders
  the kernel explicitly steps over (`rt_args_idx += 2;` in the legacy in1 kernel's `#else`) — which
  is why `last_num_blocks_w_dim` correctly lands at 20 and not 18.
- Three named CB args address indices the factory never allocates: `cb_in0_sharded`→`c_2` (`:500`),
  `cb_bias`→`c_3` (`:527`), `cb_in0_transposed`→`c_10` (`:589`).
- `FUSE_ACTIVATION = "0"` (`:454`) is read by no matmul compute kernel.
- Six dead locals (`:273`–`:277` mcast-list/no-work core sets; the four `*_cb_index` log-only
  variables).
- No unreferenced kernel files in the sparse tree.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: **`ProgramSpecFactoryConcept`** — the ported-from factory has
  no `override_runtime_arguments`, so the framework owns the cache-hit tensor refresh and the
  factory writes one method, `create_program_artifacts`. The recipe's
  *Translating `override_runtime_arguments`* step is skipped entirely.
- **Custom `compute_program_hash`**: none (commented out at `…device_operation.hpp:33`,
  `.cpp:504`) — left intact.
- **Implementation notes**: the factory keeps its existing struct; only the method changes
  (`create_descriptor` → `create_program_artifacts`, returning `ProgramArtifacts`). No pybind line
  to delete, no parameter to unwind, no direct-descriptor conversion. The header's existing comment
  block explains why the *descriptor* shape was chosen and how `Buffer*` bindings covered
  cache-hit patching; that reasoning is superseded by the typed binding channel and is rewritten,
  not deleted.

## Planned Spec Shape

1:1 with legacy throughout. No multiplicity, no op-owned tensors.

- **KernelSpecs (4)** — `IN0_SENDER`, `IN0_RECEIVER` (omitted when `in0_mcast_receivers` is empty),
  `IN1_SENDER_WRITER`, `COMPUTE`. Sources are the four `_metal2` forks.
- **DataflowBufferSpecs (5 or 6)** — `IN0_DFB`, `IN1_DFB`, `SPARSITY_DFB` (`c_6`),
  `IN1_SPARSITY_DFB` (`c_7`), `OUT_DFB`, `INTERM0_DFB`. Six always; the *aliasing* between
  `OUT_DFB` and `INTERM0_DFB` is what varies with config (below). No `borrowed_from` anywhere —
  the legacy factory borrows no CB.

  | DFB | entry_size | num_entries | data_format_metadata | tile_format_metadata |
  |---|---|---|---|---|
  | `IN0_DFB` | `in0_aligned_tile_size` | `in0_CB_size / in0_aligned_tile_size` | `in0_data_format` | `in0_tile` |
  | `IN1_DFB` | `in1_aligned_tile_size` | `in1_CB_size / in1_aligned_tile_size` | `in1_data_format` | `in1_tile` |
  | `SPARSITY_DFB` | `sparsity_cb_size` | 1 | `dtype(sparsity)` | *(unset)* |
  | `IN1_SPARSITY_DFB` | `in1_sparsity_cb_size` | 1 | `dtype(in1_sparsity_tensor)` | *(unset)* |
  | `OUT_DFB` | `output_single_tile_size` | `out_CB_size / output_single_tile_size` | `output_data_format` | `output_tile` |
  | `INTERM0_DFB` | `interm0_single_tile_size` | `interm0_total_size / interm0_single_tile_size` | `interm0_data_format` | `output_tile` |

  where `interm0_total_size = separate_out_and_interm0 ? interm0_CB_size : out_CB_size` and
  `separate_out_and_interm0 = (interm0_data_format != output_data_format)` — the exact condition the
  legacy branch at `:681` tests.

- **SemaphoreSpecs (2)** — `SENDER_SEM` ("in0_mcast_sender"), `RECEIVER_SEM`
  ("in0_mcast_receiver"), both `target_nodes = all_cores`. The legacy `initial_value = INVALID` is
  not a `SemaphoreSpec` field; Metal 2.0 initialises mcast semaphores itself and the in0 sender fork
  re-establishes `VALID`/`INVALID` on the receiver semaphore in-kernel exactly as the legacy kernel
  did.
- **TensorParameters (4 or 5)** — `IN0`, `IN1`, `SPARSITY`, `OUTPUT`, plus `INDICES` **only when
  `use_indices`**. All strict (`relaxations` untouched).
- **WorkUnitSpecs (1 or 2)**:
  - `sparse_mm_in0_sender` = `{IN0_SENDER, IN1_SENDER_WRITER, COMPUTE}` on `in0_mcast_sender_cores`
  - `sparse_mm_in0_receivers` = `{IN0_RECEIVER, IN1_SENDER_WRITER, COMPUTE}` on
    `in0_mcast_receivers` — omitted when empty
  - Their union is `all_cores_with_work`, which reproduces the legacy placement of the in1
    sender/writer and compute exactly.
- **Op-owned tensors** — none. The factory allocates no device tensor beyond the op's io.

## Preserved Multiplicity

**none — no work-split multiplicity in legacy.** Each of the four kernel sources is instantiated
exactly once. `IN0_SENDER` and `IN0_RECEIVER` are *different sources* over **disjoint** node sets
(`{start_core}` vs the rest), so each one's `IN0_DFB` PRODUCER binding is a legal single-role
binding per node — the disjoint-node case, not the same-grid two-toucher case, and **not** the
`allow_instance_multi_binding` flag.

## Dropped Plumbing

### in0 sender — positional CTAs → named

| legacy slot | legacy form | Metal 2.0 replacement |
|---|---|---|
| 0–3 | `in0_tensor_stride_w/h`, `next_block_stride`, `next_h_dim_block_stride` | named `in0_tensor_stride_w`, `in0_tensor_stride_h`, **`in0_tensor_next_inner_dim_block_stride`** (fork's name for slot 2), `in0_tensor_next_h_dim_block_stride` |
| 4–8 | `in0_block_w/h`, `in0_block_num_tiles`, `in0_last_ktile_w`, `0` | named, same names + `in0_last_ktile_h` |
| **9** | `false` — `extract_shard_sub_blocks` | **dropped** — the fork gates it on the `EXTRACT_SHARD_SUB_BLOCKS` define inside `IN0_SHARDED`, neither of which this factory sets |
| 10–11 | `0`, `0` — shard dims | named `shard_width_in_tiles`, `shard_height_in_tiles` (read only under `IN0_SHARDED`; carried as `0u`, matching legacy) |
| 12–14 | `num_blocks`, `out_num_blocks_x/y` | named **`num_blocks_inner_dim`**, **`num_blocks_w_dim`**, **`num_blocks_h_dim`** |
| **15–16** | `in0_mcast_sender_semaphore_id`, `in0_mcast_receiver_semaphore_id` | **`SemaphoreBinding`**s → `sem::in0_mcast_sender`, `sem::in0_mcast_receiver` |
| 17–18 | `num_cores-1`, `in0_mcast_receiver_num_cores-1` | named `in0_mcast_num_dests`, `in0_mcast_num_cores` |
| 19–22 | `Mt*Kt`, `batchA`, `batchA`, `false` | named `MtKt`, `in0_B`, `in1_B`, **`in0_reuse_in_dfb`** |
| 23–26 | `batchB`, `sparsity_pagesize`, `bcast_A`, `get_batch_from_reader` | named, same names (`sparsity_pagesize` read under `#ifdef SPARSITY`) |
| **27** | `false` — `fuse_op` | **dropped** — the fork gates the CCL receiver on `FUSE_OP`, which no Metal 2.0 factory may set |
| **`TensorAccessorArgs(in0)`** (`:371`) | host-side accessor-args append + kernel `TensorAccessorArgs<N>()` chain | **`TensorBinding{IN0, "in0"}`**; kernel builds `TensorAccessor(tensor::in0)` |
| **`TensorAccessorArgs(sparsity)`** (`:372`) | ditto | **`TensorBinding{SPARSITY, "sparsity"}`** |
| trailing `num_batch_compute` (`:376`) | positional push after the accessor args | named `num_batch_compute` (read under `#ifdef SPARSITY`) |
| **named `cb_in0`, `cb_sparsity`** | CB index carried by a named CTA | **`DFBBinding`**s → `dfb::in0`, `dfb::sparsity` |
| **named `cb_in0_sharded`** | CB index `c_2`, never allocated | **dropped** — no binding; read only under `EXTRACT_SHARD_SUB_BLOCKS` |
| named `num_active` | already named | stays a named CTA |

### in0 sender — RTAs → named + bindings

| legacy slot | legacy form | Metal 2.0 replacement |
|---|---|---|
| **0** | `Buffer*` binding, `in0_buffer` (`:790`) | **`TensorBinding{IN0}`** (the same binding as the CTA row above — one parameter, one binding) |
| 1 | `Kt * per_core_M * output_idx_y` | named RTA `in0_tensor_start_tile_id` |
| 2–5 | mcast dest NOC start/end x/y | named RTAs, same names |
| 6 | `out_block_h` | named RTA `last_block_h` |
| **7** | `Buffer*` binding, `sparsity_buffer` (`:791`) | **`TensorBinding{SPARSITY}`** |

### in0 receiver

| legacy slot | legacy form | Metal 2.0 replacement |
|---|---|---|
| CTA 0–3 | `in0_block_num_tiles`, `num_blocks`, `out_num_blocks_x/y` | named `in0_block_num_tiles`, `num_blocks_inner_dim`, `num_blocks_w_dim`, `num_blocks_h_dim` |
| **CTA 4–5** | the two semaphore ids | **`SemaphoreBinding`**s |
| CTA 6–7 | `num_batch_compute`, `get_batch_from_reader` | named **`batch`** (the fork's name), `get_batch_from_reader` |
| **named `cb_in0`** | CB index | **`DFBBinding`** → `dfb::in0` |
| RTA 0–1 | `top_left_core_physical.x/y` | named RTAs `in0_mcast_sender_noc_x`, `in0_mcast_sender_noc_y` |

### in1 sender / writer — positional CTAs → named

| legacy slot | legacy form | Metal 2.0 replacement |
|---|---|---|
| 0–3 | in1 strides | named `in1_tensor_stride_w`, `in1_tensor_stride_h`, `in1_tensor_next_block_stride`, `in1_tensor_next_w_dim_block_stride` |
| 4–6 | `in1_block_w`, `in0_block_w`, `in1_block_w*in0_block_w` | named `in1_block_w`, **`in1_block_h`**, `in1_block_num_tiles` |
| 7–9 | `num_blocks`, `out_num_blocks_x/y` | named `num_blocks_inner_dim`, `num_blocks_w_dim`, `num_blocks_h_dim` |
| **10–11** | `0`, `0` — the in1 mcast semaphore ids | **`SemaphoreBinding`**s → `sem::in1_mcast_sender`, `sem::in1_mcast_receiver`. Under `SKIP_MCAST` the kernel constructs both objects but never uses them; they are bound because the construction is unconditional |
| 12–13 | `0`, `0` | named `in1_mcast_num_dests`, `in1_mcast_num_cores` |
| 14–16 | `Kt*Nt`, `batchA`, `true` | named `KtNt`, **`batch`**, `bcast_B` |
| 17–18 | `batchB`, `sparsity_pagesize` | named, same names (`sparsity_pagesize` under `#ifdef SPARSITY`) |
| 19–27 | out strides + subblock geometry | named `out_tensor_stride_w/h`, `out_tensor_next_subblock_stride_w/h`, `out_tensor_next_w_dim_block_stride`, `out_tensor_next_h_dim_block_stride`, `out_subblock_w`, `out_subblock_h`, **`out_subblock_tile_count`** |
| 28 | `Mt*Nt` | named `MtNt` |
| **29** | `0` — `in3_tensor_stride_w` | **dropped** — read only under `FUSE_BIAS` |
| **30–31** | `false`, `false` — `fuse_op`, `fuse_op_reduce_scatter` | **dropped** — `FUSE_OP_ALL_GATHER` / `FUSE_OP_REDUCE_SCATTER` defines, unsettable |
| 32 | `compact_output` | named `compact_output` |
| **`TensorAccessorArgs(in1)`** (`:429`) | accessor-args append | **`TensorBinding{IN1, "in1"}`** |
| **`TensorAccessorArgs(in1_sparsity)`** (`:431`) | ditto | **`TensorBinding{SPARSITY or INDICES, "sparsity"}`** |
| **`TensorAccessorArgs(out)`** (`:432`) | ditto | **`TensorBinding{OUTPUT, "out"}`** |
| **`TensorAccessorArgs()`** (`:433`) | empty bias placeholder | **dropped** |
| **named `cb_in1`, `cb_out`, `cb_sparsity`** | CB indices | **`DFBBinding`**s → `dfb::in1`, `dfb::out`, `dfb::sparsity` (the `c_7` spec) |
| **named `cb_bias`** | CB index `c_3`, never allocated | **dropped** — read only under `FUSE_BIAS` |
| named `num_active` | already named | stays a named CTA |

### in1 sender / writer — RTAs → named + bindings

| legacy slot | legacy form | Metal 2.0 replacement |
|---|---|---|
| **0** | `Buffer*` binding, `in1_buffer` | **`TensorBinding{IN1}`** |
| 1 | `per_core_N * output_idx_x` | named `in1_tensor_start_tile_id` |
| 2–5 | `0,0,0,0` in1 mcast dest NOC | named `in1_mcast_dest_noc_start_x/y`, `..._end_x/y` |
| **6** | `Buffer*` binding, `in1_sparsity_buffer` | **`TensorBinding{SPARSITY or INDICES}`** |
| **7** | `Buffer*` binding, `out_buffer` | **`TensorBinding{OUTPUT}`** |
| 8 | out start tile id | named `out_tensor_start_tile_id` |
| 9 | `last_out_block_w` / `out_block_w` | named `last_block_w` |
| 10–17 | writer padding geometry | named `out_num_nonzero_subblocks_h`, `out_last_subblock_h`, `padded_block_tiles_h_skip`, `out_num_nonzero_subblocks_w`, `out_last_num_nonzero_subblocks_w`, `out_last_subblock_w`, `padded_subblock_tiles_addr_skip`, `padded_block_tiles_w_skip` |
| **18–19** | `0`, `0` — bias placeholders the kernel steps over (`rt_args_idx += 2`) | **dropped** — named args live in their own section, so the skip has nothing to preserve |
| 20 | `last_out_num_blocks_w` / `out_num_blocks_x` | named **`last_num_blocks_w_dim`** (the fork reads it under `#ifndef OUT_SHARDED`, always live here) |
| **21–25** | five `0`s | **dropped** — dead in this configuration; the kernel's last read is slot 20, and the `IN1_DRAM_*_SHARDED` reads that would follow are unsettable here |

### compute — positional CTAs → named

| legacy slot | legacy form | Metal 2.0 replacement |
|---|---|---|
| 0–6 | `in0_block_w`, `in0_num_subblocks`, `in0_block_num_tiles`, `in0_subblock_num_tiles`, `in1_num_subblocks`, `in1_block_num_tiles`, `in1_per_core_w` | named, last one as **`in1_block_w`** |
| 7–9 | `num_blocks`, `out_num_blocks_x/y` | named `num_blocks_inner_dim`, `num_blocks_w_dim`, `num_blocks_h_dim` |
| 10–12 | `out_subblock_h/w`, `out_subblock_num_tiles` | named, same names |
| 13 | `num_batch_compute` | named **`batch`** |
| 14 | `out_block_tiles` | named `out_block_num_tiles` |
| 15 | `false` | named `untilize_out` |
| 16 | `get_batch_from_reader` | named, same name |
| **17** | `false` — `in0_transpose_tile` | **dropped** — the fork selects the transposed buffer with `#ifdef IN0_TRANSPOSE_TILE`, unset here |
| **named `cb_in0`, `cb_in1`, `cb_out`, `cb_intermed0`** | CB indices | **`DFBBinding`**s → `dfb::in0`, `dfb::in1`, `dfb::out`, `dfb::intermed0` |
| **named `cb_bias`, `cb_in0_transposed`** | CB indices `c_3`, `c_10`, neither allocated | **dropped** — read only under `FUSE_BIAS` / `IN0_TRANSPOSE_TILE` |
| **`unpack_to_dest_mode` vector** (`:568`–`:572`) | `vector<UnpackToDestMode>` indexed by CB id | **`unpack_modes` `Table`** keyed by DFB name — see [Applied Patterns](#applied-patterns) |

## Applied Patterns

- **[Caution: Porting a shared kernel] — rung 1 (reuse an existing `_metal2` fork), ×4.** All four
  sources are lent and all four forks exist with live consumers. `KernelSpec::source` points at the
  fork; the fork's binding vocabulary is adopted verbatim; nothing is forked, copied or edited, and
  no pointer comment is added to any legacy original (rung 1 forbids it).
- **[Pattern: Self-loop DFB binding] ×3.** `INTERM0_DFB` on `COMPUTE` (accumulator: the kernel
  packs partials and reloads them across K-blocks); `SPARSITY_DFB` on `IN0_SENDER`;
  `IN1_SPARSITY_DFB` on `IN1_SENDER_WRITER`. Each has exactly **one** toucher, which both produces
  and consumes — so PRODUCER + CONSUMER on the single kernel, sharing one `accessor_name`.
- **[Pattern: Aliased DFBs] — conditional.** When `interm0_data_format == output_data_format`,
  `OUT_DFB` and `INTERM0_DFB` mutually name each other in `advanced_options.alias_with` and both
  size to `out_CB_size`, reproducing the legacy single-`CBDescriptor`-two-`format_descriptors`
  shape. When the formats differ, neither aliases and `INTERM0_DFB` sizes to `interm0_CB_size`. The
  group is a strict clique of two, both non-borrowed, with identical `num_entries * entry_size` —
  the three legality rules hold in both configs.
- **[Pattern: Conditional / optional resource bindings] — `SPARSITY` define.** The sparsity DFB and
  its tensor accessor exist only under the `SPARSITY` define, which this factory sets on both
  readers. The forks already carry the `#ifdef` gating; the port's side of the pattern is emitting
  the define alongside the bindings. *(The condition is a constant `true` for this factory, so no
  host-side branch is needed — but the define is still mandatory, because the forks' default is
  sparsity-off.)*
- **[Pattern: Pass DFB handles directly to LLKs]** — inherited, not authored. The forks already pass
  `dfb::name` into `matmul_block`, `pack_tile` and friends.
- **[Anti-pattern: Demoting per-group CTA to RTA]** — not applicable, and deliberately so: there is
  no work-split multiplicity to demote. Recorded because the `IN0_SENDER` / `IN0_RECEIVER` pair over
  disjoint node sets is the shape the entry's *Constraint* paragraph distinguishes from the same-grid
  two-toucher case.

## Deferred / Flagged

- **`SPARSITY` has never been exercised through a Metal 2.0 spec.** The forks carry the region and
  no factory sets the define. Treat first-run failures there as the most likely location, and
  distinguish "fork bug" (report, do not edit — it has consumers) from "spec bug" (mine).
- **The in1 sender/writer binds two semaphores it never uses.** `SKIP_MCAST` is unconditional for
  this factory, so both `Semaphore` objects are constructed and never touched. They are bound
  because the fork constructs them unconditionally; the legacy factory likewise passed ids `0`/`0`.
  Binding them is faithful, not a leak — but it means the in1 sender appears as a semaphore endpoint
  in the spec where legacy showed a literal `0`.
- **Two `TensorParameter`s can point at one tensor.** When `use_indices` is false, `SPARSITY` is the
  only sparsity parameter and both readers bind it. When true, `INDICES` is a distinct parameter.
  The count of `tensor_parameters` therefore varies with the *attributes*, which the program hash
  covers (`use_indices` is derived from `operation_attributes.use_indices` plus the presence of the
  optional input), so no cache-equivalence question arises.
- **`SemaphoreDescriptor::initial_value = INVALID` has no `SemaphoreSpec` counterpart.** The legacy
  descriptor set it explicitly; `SemaphoreSpec` carries only `unique_id` and `target_nodes`. The in0
  sender fork sets the receiver semaphore to `VALID` at entry (`receiver_sem.set(VALID)`) and the
  receivers reset to `INVALID` per batch, exactly as the legacy kernels did, so the observable
  initial state is unchanged. Flagged because it is the one legacy field with no destination, and a
  reviewer will look for it.
- Nothing else surfaced during planning that the audit had not already named.
