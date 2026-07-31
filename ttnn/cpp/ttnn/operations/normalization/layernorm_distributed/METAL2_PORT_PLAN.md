# Port Plan — `normalization/layernorm_distributed`

Port plan for the two DeviceOperations in `ttnn/cpp/ttnn/operations/normalization/layernorm_distributed`,
ported from `ProgramDescriptor` (`create_descriptor`) to Metal 2.0 (`create_program_artifacts`).
Written during the inventory and planning steps; committed alongside the port for review.

**Unit of work.** All **five** factories port together in one change. They are not independent: the
writer kernel is bound by all five, the pre-allgather reader by two, and the post-allgather reader by
two. Porting one factory at a time would force `_metal2` forks of those three intra-op shared kernels
and leave the op carrying duplicate kernel sources indefinitely. Converting the whole directory at
once needs no intra-op fork at all.

---

## Legacy Inventory

### Legacy factory shape

- Concept: `ProgramDescriptorFactoryConcept` — five `static ProgramDescriptor create_descriptor(...)`
  entry points, no `create()` and no `create_workload_descriptor`.
  - `LayerNormPreAllGatherDeviceOperation` → `LayerNormPreAllGatherProgramFactory`,
    `LayerNormPreAllGather2DProgramFactory`, `LayerNormPreAllGatherWelfordProgramFactory`
    (`device/layernorm_pre_all_gather_device_operation.hpp:19-40`)
  - `LayerNormPostAllGatherDeviceOperation` → `LayerNormPostAllGatherProgramFactory`,
    `LayerNormPostAllGatherWelfordProgramFactory`
    (`device/layernorm_post_all_gather_device_operation.hpp:19-32`)
- Variants: five, listed above. The default post factory additionally carries two work-split configs in
  one body (1D and `use_2d_core_grid`, `device/layernorm_post_all_gather_program_factory.cpp:149`).
- Custom `compute_program_hash`: **none** on either DeviceOperation — already the default
  reflection-based hash. Nothing to delete.
- `get_dynamic_runtime_args` / `override_runtime_arguments` / pybind `create_descriptor`: all absent.

### Configs the port must hold

`is_rmsnorm` (selects a different compute kernel) × `fuse_pre_add` (residual present) ×
`gamma` present × `beta` present × `use_2d_core_grid` × Welford-vs-not.
Two combinations are rejected upstream by validation and need no port coverage:
RMSNorm + Welford (`device/layernorm_post_all_gather_device_operation.cpp:166-171`,
`device/layernorm_pre_all_gather_welford_program_factory.cpp:46`) and Welford + 2D core grid
(`device/layernorm_pre_all_gather_device_operation.cpp:62-69`).

---

### Variant: `LayerNormPreAllGatherProgramFactory` (Pre 1D)

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_unary_interleaved_ln_rm_gb_pre_allgather.cpp` | `all_cores` | `blk`, then `TensorAccessorArgs(a)`, then `TensorAccessorArgs(b)` if fuse | none | `a.buffer()`, `NCHt`, `Wt`, `in_tile_offset`, [`b.buffer()` if fuse] | none | `FUSE_PRE_ADD` = "0"/"1" | `O2` (field absent → DM default) | `ReaderConfigDescriptor{}` |
| writer | `device/kernels/dataflow/writer_unary_interleaved_start_id_blocked.cpp` | `all_cores` | `writer_block_size`, then `TensorAccessorArgs(output)` | none | `output.buffer()`, `num_tile_rows_per_core * out0_tiles`, `out_tile_offset` | none | none | `O2` | `WriterConfigDescriptor{}` |
| compute | `rmsnorm_distributed/.../compute/rmsnorm_pre_allgather.cpp` if `is_rmsnorm`, else `device/kernels/compute/layernorm_pre_allgather.cpp` | `all_cores` | `Wt`, `block_size` | none | `num_tile_rows_per_core` | none | `FUSE_PRE_ADD` = "0"/"1" | **`O3`** (field absent → compute default) | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en, dst_full_sync_en, math_approx_mode}` |

#### CBs

| index | total_size | core_ranges | data_format | page_size | tile |
|---|---|---|---|---|---|
| `c_0` input | `in0_tiles * in_single_tile_size` (`in0_tiles = Wt*2`) | `all_cores` | `in_data_format` | `in_single_tile_size` | unset |
| `c_1` reduce scaler | `1 * scaler_tile_size` | `all_cores` | `scaler_cb_data_format` (fp32 if input fp32 else bf16) | `scaler_tile_size` | unset |
| `c_5` residual *(fuse only)* | `res_tiles * inb_single_tile_size` (`Wt*2`) | `all_cores` | `inb_data_format` | `inb_single_tile_size` | unset |
| `c_3` fused a+b *(fuse only)* | `fused_tiles * single_tile_size` (`Wt`) | `all_cores` | `cb_data_format` (bf16) | `single_tile_size` | unset |
| `c_6` x² | `intermed0_tiles * single_tile_size` (`Wt*2`) | `all_cores` | `cb_data_format` | `single_tile_size` | unset |
| `c_14` output | `out0_tiles * out_single_tile_size` (`out0_tiles` = 2 LN / 1 RMS) | `all_cores` | `out_data_format` | `out_single_tile_size` | unset |

No `.tile` is set on any `CBFormatDescriptor` anywhere in the op; no `.address_offset`; no
`GlobalCircularBuffer`. Exactly one `.buffer` (Pre-Welford `c_2`) — a plain borrowed-memory CB.

#### Semaphores

none

#### Tensor accessors

| host site | originating Tensor | RTA slot (host) |
|---|---|---|
| `device/kernels/dataflow/reader_unary_interleaved_ln_rm_gb_pre_allgather.cpp:34` | `tensor_args.input` | reader slot 0 |
| `…reader_unary_interleaved_ln_rm_gb_pre_allgather.cpp:44` *(fuse)* | `tensor_args.residual_input_tensor` | reader slot 4 |
| `device/kernels/dataflow/writer_unary_interleaved_start_id_blocked.cpp:26` | `tensor_return_value` (output) | writer slot 0 |

#### Work split

- Driver: `split_work_to_cores(grid_size, num_tile_rows, true)`
  (`device/layernorm_pre_all_gather_program_factory.cpp:115`)
- Yields `num_cores`, `all_cores`, `core_group_1` / `core_group_2`,
  `num_tile_rows_per_core_group_1` / `_2`. The per-group count travels as an **RTA**
  (`num_tile_rows_per_core`), not a CTA — so there is **no** multi-`KernelDescriptor` work split.

---

### Variant: `LayerNormPreAllGather2DProgramFactory` (Pre 2D)

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | RTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_layernorm_preallgather_2d.cpp` | `all_cores` (`cores_x` × `cores_y`) | `blk`, `reducer_semaphore_id`, `cores_y`, `TensorAccessorArgs(a)`, [`TensorAccessorArgs(b)`] | `a.buffer()`, `tiles_per_core_x`, `tiles_per_core_y`, `in_tile_offset`, `is_merge_core`, `merge_core.x`, `merge_core.y`, `y`, [`b.buffer()`] | `FUSE_PRE_ADD` | `O2` | `ReaderConfigDescriptor{}` |
| writer | `device/kernels/dataflow/writer_unary_interleaved_start_id_blocked.cpp` | **`merge_cores`** (the `y == 0` row) | `writer_block_size`, `TensorAccessorArgs(output)` | `output.buffer()`, `tiles_per_core_x * out0_tiles`, `out_tile_offset` | none | `O2` | `WriterConfigDescriptor{}` |
| compute | `device/kernels/compute/layernorm_pre_allgather_2d.cpp` (**no `is_rmsnorm` branch**) | `all_cores` | `tiles_per_core_x`, `tiles_per_core_y`, `blk`, `cores_y` | `is_merge_core` | `FUSE_PRE_ADD` | **`O3`** | `ComputeConfigDescriptor{…}` |

#### CBs

`c_0`, `c_1`, `c_5` *(fuse)*, `c_3` *(fuse)*, `c_6` as in Pre 1D, plus:

| index | total_size | core_ranges | data_format |
|---|---|---|---|
| `c_15` cross-core merge | `cores_y * single_tile_size` | `all_cores` | `cb_data_format` |
| `c_16` per-core partial out | `out0_tiles * single_tile_size` (`out0_tiles = 1`) | `all_cores` | `cb_data_format` |
| `c_13` zero tile | `out0_tiles * single_tile_size` | `all_cores` | `cb_data_format` |
| `c_14` final out | `out0_tiles * out_single_tile_size` | **`merge_cores`** | `out_data_format` |

#### Semaphores

| id | core_type | core_ranges | initial_value |
|---|---|---|---|
| 0 | `WORKER` | `all_cores` | 0 |

#### Work split

2D grid: `cores_x = min(grid.y, num_tile_rows)` decremented until it divides `num_tile_rows`;
`cores_y = min(grid.y, Wt)` decremented until it divides `Wt`. No `split_work_to_cores`.
`merge_cores` = the `y == 0` row of that grid.

---

### Variant: `LayerNormPreAllGatherWelfordProgramFactory` (Pre Welford)

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_unary_interleaved_ln_rm_gb_pre_allgather.cpp` *(shared with Pre 1D)* | `all_cores` | `blk`, `TensorAccessorArgs(a)`, [`TensorAccessorArgs(b)`] | none | `a.buffer()`, `NCHt`, `Wt`, `in_tile_offset`, [`b.buffer()`] | `FUSE_PRE_ADD` | `O2` | `ReaderConfigDescriptor{}` |
| writer | `device/kernels/dataflow/writer_unary_interleaved_start_id_blocked.cpp` | `all_cores` | `writer_block_size`, `TensorAccessorArgs(output)` | none | `output.buffer()`, `num_tile_rows_per_core * out0_tiles`, `out_tile_offset` | none | `O2` | `WriterConfigDescriptor{}` |
| compute | `device/kernels/compute/layernorm_pre_allgather_welford.cpp` | `all_cores` | `Wt`, `W`, `block_size` | `welford_unpack_fp32_active` | `num_tile_rows_per_core` | `FUSE_PRE_ADD` | **`O3`** | `ComputeConfigDescriptor{…, .unpack_to_dest_mode = …}` |

`block_size = gcd(Wt, fp32_dest_acc_en ? 4 : 8)`.

`unpack_to_dest_mode` (a `NUM_CIRCULAR_BUFFERS`-long vector, `Default` everywhere except):

| CB | condition | legacy value |
|---|---|---|
| `c_0` | `welford_unpack_fp32_active` | `UnpackToDestFp32` |
| `c_5`, `c_3` | `welford_unpack_fp32_active && fuse_pre_add` | `UnpackToDestFp32` |
| `c_1` | `out_data_format == Float32 && fp32_dest_acc_en` | `UnpackToDestFp32` |
| `c_4`, `c_6` | `fuse_pre_add && fp32_dest_acc_en` | `UnpackToDestFp32` |

where `welford_unpack_fp32_active = (in_data_format == Float32 && fp32_dest_acc_en)`.

#### CBs

| index | total_size | core_ranges | data_format | page_size |
|---|---|---|---|---|
| `c_0` input | `in0_tiles * in_single_tile_size` (`in0_tiles = block_size*2`) | `all_cores` | `in_data_format` | `in_single_tile_size` |
| `c_5` residual *(fuse)* | `res_tiles * inb_single_tile_size` | `all_cores` | `inb_data_format` | `inb_single_tile_size` |
| `c_3` fused *(fuse)* | `fused_tiles * single_tile_size` | `all_cores` | `cb_data_format` (fp32 if `fp32_dest_acc_en`) | `single_tile_size` |
| `c_4` Welford mean spill *(fuse)* | `1 * single_tile_size` | `all_cores` | `cb_data_format` | `single_tile_size` |
| `c_6` Welford M2 spill *(fuse)* | `1 * single_tile_size` | `all_cores` | `cb_data_format` | `single_tile_size` |
| `c_1` transpose scratch | `in0_tiles * out_single_tile_size` | `all_cores` | `out_data_format` | `out_single_tile_size` |
| `c_14` output | `in0_tiles * out_single_tile_size` | `all_cores` | `out_data_format` | `out_single_tile_size` |
| `c_2` reciprocal LUT | `recip_tensor.buffer()->aligned_size_per_bank()` | `all_cores` | `Float32` | same as total | **`.buffer = recip_tensor.buffer()`** — borrowed memory |

#### Tensor accessors

`input` (reader slot 0), `residual` (reader slot 4, fuse), `output` (writer slot 0). The
`recip_tensor` is **not** an accessor — it backs `c_2` and is read through a raw
`get_pointer_to_cb_data<recip_lut_t>(cb_reciprocals, 0)` peek
(`device/kernels/compute/layernorm_pre_allgather_welford.cpp:75`).

#### Work split

`split_work_to_cores(grid_size, num_tile_rows, true)`; per-group count travels as an RTA. No
multi-`KernelDescriptor` split.

---

### Variant: `LayerNormPostAllGatherProgramFactory` (Post default — 1D and 2D configs)

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | RTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|
| reader | `device/kernels/dataflow/reader_unary_interleaved_ln_rm_gb_post_allgather.cpp` | `all_cores` | `blk`, `stats_tiles_cols`, `gamma_stick_size`, `beta_stick_size`, `gamma_is_row_major`, `beta_is_row_major`, `cb_length`, `tiles_per_core_y`, `reduce_factor`, then `TensorAccessorArgs` × 4 (input, stats, gamma-or-nullptr, beta-or-nullptr) | `a.buffer()`, `NCHt`, `Wt`, `tile_offset`, `stats_offset`, `eps_u`, `gamma_buffer`, `beta_buffer`, `stats.buffer()`, `y_offset` | `FUSE_GAMMA` if gamma, `FUSE_BETA` if beta | `O2` | `ReaderConfigDescriptor{}` |
| writer | `device/kernels/dataflow/writer_unary_interleaved_start_id_blocked.cpp` | `all_cores` | `block_size`, `TensorAccessorArgs(output)` | `output.buffer()`, `num_tiles`, `tile_offset` | none | `O2` | `WriterConfigDescriptor{}` |
| compute | `rmsnorm_distributed/.../compute/rmsnorm_post_allgather.cpp` if `is_rmsnorm`, else `device/kernels/compute/layernorm_post_allgather.cpp` | `all_cores` | `tiles_per_core_y`, `block_size`, `stats_tiles_cols`, `do_gamma`, `do_beta`, `fp32_dest_acc_en`, `float32_reduction`, `legacy_rsqrt`, `cb_length` | `num_tile_rows_per_core` | none | **`O3`** | `ComputeConfigDescriptor{…}` |

Reader RTA slot 2 (host writes `Wt` / `tiles_per_core_y`) is **never read by the kernel** — the reader
takes its width from CTA 7 instead. It has no name to carry and is dropped; see *Dropped Plumbing*.

#### CBs

`c_0` input, `c_1` stats, `c_2` gamma *(gamma only)*, `c_3` beta *(beta only)*, `c_4` epsilon
(always `Float16_b`), `c_5` reduce scaler (fp32 if input fp32 else bf16), `c_6` reduced stats,
`c_8` var, **`c_9` var+eps (dead)**, `c_10` 1/√(var+ε), `c_12` x normed, `c_7` mean² *(LN only)*,
`c_11` x−mean *(LN only)*, `c_13` ×gamma intermediate *(LN + beta only — see below)*, `c_14` output.
All over `all_cores`.

Sizes: input / gamma / beta / `c_11` / `c_12` / `c_13` / output are `cb_length` tiles; `c_1` is
`stats_tiles_cols`; `c_6` is `tile_cols_per_device`; `c_4`, `c_5`, `c_7`, `c_8`, `c_9`, `c_10` are 1.

#### Work split

Two configs in one factory body:
- 1D: `split_work_to_cores(grid_size, num_tile_rows, true)`, `tiles_per_core_y = Wt`.
- 2D (`use_2d_core_grid`): `cores_x` × `cores_y` grid, `tiles_per_core_y = Wt / cores_y`.

Both put the per-core row count in an RTA, so neither is a multi-`KernelDescriptor` split.

---

### Variant: `LayerNormPostAllGatherWelfordProgramFactory` (Post Welford)

Same three kernel roles; compute is `device/kernels/compute/layernorm_post_allgather_welford.cpp`
(the `is_rmsnorm` branch selecting `rmsnorm_post_allgather.cpp` is unreachable — validation rejects
RMSNorm + Welford). Compute CTAs: `Wt`, `W`, `block_size`, `stats_tiles_cols`, `do_gamma`, `do_beta`,
`fp32_dest_acc_en`, `cb_length`; `opt_level` **`O3`**. `unpack_to_dest_mode` sets `c_1` →
`UnpackToDestFp32` when `!is_rmsnorm && fp32_dest_acc_en && stats_data_format == Float32`.

CBs as in the default post factory, except `c_5` uses `cb_data_format` (not the scaler format), and
`c_7`, `c_8`, `c_9` are allocated but **dead**.

---

### Shared kernels

| kernel source | kind | consumers | `_metal2` fork exists? | rung |
|---|---|---|---|---|
| `device/kernels/dataflow/writer_unary_interleaved_start_id_blocked.cpp` | **intra-op** | all 5 of this op's factories | no | **n/a — every consumer converts in this change**, so it is converted in place |
| `device/kernels/dataflow/reader_unary_interleaved_ln_rm_gb_pre_allgather.cpp` | **intra-op** | Pre 1D + Pre Welford | no | same — converted in place |
| `device/kernels/dataflow/reader_unary_interleaved_ln_rm_gb_post_allgather.cpp` | **intra-op** | Post default + Post Welford | no | same — converted in place |
| `device/kernels/compute/chain_llk.hpp` | in-directory header | Post default + Post Welford compute | no | same — converted in place |
| `rmsnorm_distributed/device/kernels/compute/rmsnorm_pre_allgather.cpp` | **borrowed** (peer op directory) | this op only | no | **rung 2 — create `rmsnorm_pre_allgather_metal2.cpp` beside it + pointer comment in the original** |
| `rmsnorm_distributed/device/kernels/compute/rmsnorm_post_allgather.cpp` | **borrowed** (peer op directory) | this op only | no | **rung 2 — create `rmsnorm_post_allgather_metal2.cpp` beside it + pointer comment in the original** |

The two borrowed kernels are bound *only* by this op's factories (`grep -rl` across `ttnn/` returns the
three call sites in this directory and nothing else), so their legacy copies become retirable once this
port lands. That is a **sunset list, not authorization** to convert them in place — no bundled-port
assignment was given, so rung 2 applies.

Near-miss basenames that are **different files**, not shared code:
`normalization/layernorm/device/kernels/dataflow/writer_unary_interleaved_start_id_blocked.cpp` and
`experimental/transformer/fused_distributed_rmsnorm/device/kernels/compute/rmsnorm_{pre,post}_allgather.cpp`.
Nothing outside this directory binds any kernel this op owns.

Out-of-scope donors the port passes `dfb::name` into unchanged (all ✓ shapes, none edited):
`dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<dfb_id,…>()`,
`dataflow_kernel_lib::prepare_zero_tile<dfb_id>()`,
`compute_kernel_lib::reduce<…, in, scaler, out, …>()`,
`generate_bcast_col_scalar(CircularBuffer, uint32_t)` (by value),
`norm::…::pre_add::one_row(DataflowBuffer&, …)`,
`norm::…::combine_welford_partials(DataflowBuffer&, …)`,
`norm::…::memory::get_pointer_to_cb_data<To>(uint32_t cb_id, …)`.

### Flags

1. **No unreferenced kernel files.** Every kernel under `device/kernels/` is bound by some factory.
   (`rmsnorm_distributed/…/compute/rmsnorm_pre_allgather_2d.cpp` exists in the sibling family but is
   bound by no factory anywhere; it is outside this port and was not audited.)

2. **The Post-Welford factory's reader argument layout does not match the shared reader — and the path
   is broken on `main` today.** Found during inventory; the audit did not catch it. The shared post
   reader (`device/kernels/dataflow/reader_unary_interleaved_ln_rm_gb_post_allgather.cpp:59-96`) reads
   **9** CTAs (`reduce_factor` at index 8, then `TensorAccessorArgs<9>()`) and **10** RTAs (`eps` at 5,
   `gamma_addr` at 6, `beta_addr` at 7, `stats_addr` at 8, `y_offset` at 9). The Post-Welford factory
   emits **8** CTAs — no `reduce_factor`
   (`device/layernorm_post_all_gather_welford_program_factory.cpp:254-284`) — and **11** RTAs with
   `packed_winv_value` inserted at slot 5 (`…:382-394`), so every RTA from `eps` onward lands one slot
   late and `TensorAccessorArgs<9>()` starts one word inside the input accessor's args. The default post
   factory (`device/layernorm_post_all_gather_program_factory.cpp:214-247,349-360`) matches the kernel
   exactly, so the defect is Post-Welford-only.

   **Confirmed empirically on the pre-port tree:** a single-device `layer_norm_pre_all_gather` →
   `layer_norm_post_all_gather` run with `LayerNormDefaultProgramConfig(use_welford=True)` returns
   `max_abs_err = 10.48` against the torch reference with a NaN cosine similarity — i.e. garbage. The
   path survives because it is exercised only by `mesh_device=(1, 8)` tests, which skip on a
   single-device host.

   Consequence for the port: named arguments carry meaning, not position, so the shift **cannot be
   reproduced** — the ported Post-Welford path binds `eps` to `eps`. This is an unavoidable behavior
   change on that one path (garbage → correct). Recorded as a Handoff point in `METAL2_PORT_REPORT.md`
   rather than resolved here.

3. **`packer_l1_acc` is destructured and then ignored in all five factories** (e.g.
   `device/layernorm_pre_all_gather_program_factory.cpp:55-56`). No `ComputeConfigDescriptor` in the op
   sets it. Preserved exactly — the port does not start honouring it.

4. **`log_debug(tt::LogOp, "device_id: {}", gamma.value().device()…)`** at
   `device/layernorm_post_all_gather_welford_program_factory.cpp:78` dereferences `gamma`
   unconditionally, so a Welford post-allgather call without a weight would fault there. Left exactly
   as-is; reported.

---

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept` — all five factories.
- **Custom `compute_program_hash`**: none (neither DeviceOperation declares one). Nothing to delete.
- **Implementation notes**:
  - Each factory's `create_descriptor` becomes `create_program_artifacts(attributes, tensor_args,
    tensor_return_value)`; the declarations in both `*_device_operation.hpp` files change accordingly,
    and `#include "ttnn/metal_v2_artifacts.hpp"` replaces `<tt-metalium/program_descriptors.hpp>` there.
  - No pybind line references any factory entry point (`layernorm_distributed_nanobind.cpp:20,93` bind
    only the two user-facing functions), so no pybind deletion is forced and no factory parameter needs
    unwinding.
  - `MeshTensor` is extracted once per tensor at the top of each factory and used throughout.
  - Both `program_factory_t` variants flip wholesale, so no per-factory concept mixing is needed.

---

## Planned Spec Shape

Placement is derived from `WorkUnitSpec::target_nodes` throughout; no `core_ranges` on any spec.

### Variant: Pre 1D

- **KernelSpecs** (3): `pre1d_reader`, `pre1d_writer`, `pre1d_compute`.
- **DataflowBufferSpecs** (4, +2 when fuse): `INPUT`(c_0), `REDUCE`(c_1), `X2`(c_6), `OUT`(c_14);
  `RESIDUAL`(c_5) and `FUSED`(c_3) only when `fuse_pre_add`.
- **SemaphoreSpecs**: none.
- **TensorParameters**: `INPUT_T`, `OUTPUT_T`, plus `RESIDUAL_T` when fuse.
- **WorkUnitSpecs** (1): `{reader, writer, compute}` over `all_cores`.
- **Endpoint census / disposition**:

  | DFB | producer | consumer | disposition |
  |---|---|---|---|
  | `INPUT` | reader | compute | 1:1 |
  | `REDUCE` | reader | compute | 1:1 |
  | `RESIDUAL` *(fuse)* | reader | compute | 1:1 |
  | `FUSED` *(fuse)* | compute | compute | **self-loop** (1 toucher) |
  | `X2` | compute | compute | **self-loop** (1 toucher) |
  | `OUT` | compute | writer | 1:1 |

### Variant: Pre 2D

- **KernelSpecs** (4): `pre2d_reader`, `pre2d_writer`, **`pre2d_compute_merge`**, **`pre2d_compute_worker`**.

  The compute kernel is split into **two** `KernelSpec`s of the same source over disjoint node sets.
  This is forced by the Metal 2.0 per-node DFB invariant, not chosen: `c_14` is produced by compute
  (declared over `all_cores`) and consumed by the writer (declared over `merge_cores` only), so on every
  worker node the DFB would have a producer and no consumer — a hard `TT_FATAL` from the per-node census
  in `ValidateProgramSpec`. The merge instance binds `OUT_FINAL`; the worker instance does not.
  `is_merge_core` therefore moves from a runtime arg to a `KernelSpec::compiler_options.defines` entry
  (`IS_MERGE_CORE`), which is what the conditional-binding pattern requires anyway — the gate must run at
  the preprocessor, before `dfb::out_final` enters name lookup. The value is host-known (`y == 0`), so
  this is a dispatch-slot change, not a semantic one.
- **DataflowBufferSpecs** (7, +2 when fuse): `INPUT`(c_0), `REDUCE`(c_1), `X2`(c_6), `X2_MERGE`(c_15),
  `PARTIAL_OUT`(c_16), `ZERO`(c_13), `OUT_FINAL`(c_14); `RESIDUAL`(c_5), `FUSED`(c_3) when fuse.
- **SemaphoreSpecs** (1): `REDUCER` over `all_cores`.
- **TensorParameters**: `INPUT_T`, `OUTPUT_T`, plus `RESIDUAL_T` when fuse.
- **WorkUnitSpecs** (2): `wu_merge` = `{reader, writer, compute_merge}` over `merge_cores`;
  `wu_worker` = `{reader, compute_worker}` over the `y >= 1` rows. The reader belongs to both, so its
  effective node set is `all_cores` — matching legacy.
- **Endpoint census / disposition**:

  | DFB | producer(s) | consumer(s) | disposition |
  |---|---|---|---|
  | `INPUT`, `REDUCE`, `RESIDUAL` *(fuse)* | reader | compute_merge + compute_worker | 1:1 per node (two same-role KernelSpecs over disjoint nodes — legal, no flag) |
  | `FUSED` *(fuse)*, `X2` | compute_merge, compute_worker | same | **self-loop**, one per node |
  | `PARTIAL_OUT` | compute_merge + compute_worker | reader | 1:1 per node |
  | `X2_MERGE`, `ZERO` | reader | compute_merge + compute_worker | 1:1 per node |
  | `OUT_FINAL` | compute_merge | writer | 1:1 on `merge_cores` only |

  `X2_MERGE`'s cross-core fill is **not** a second local endpoint: the
  `noc.async_write(…, {.addr = dfb_x2_merge_buf.get_write_ptr() + worker_offset})` at
  `device/kernels/dataflow/reader_layernorm_preallgather_2d.cpp:120-127` targets a *remote* node's
  instance; the local `get_write_ptr()` is a peek used only to compute the same offset. On a merge node
  the touchers are the local reader (locked producer via its `push_back` at `:137`) and the local
  compute (locked consumer at `:108,124`).

### Variant: Pre Welford

- **KernelSpecs** (3): `prewf_reader`, `prewf_writer`, `prewf_compute`.
- **DataflowBufferSpecs** (4, +4 when fuse): `INPUT`(c_0), `SCRATCH`(c_1), `RECIP`(c_2,
  `borrowed_from = RECIP_T`), `OUT`(c_14); `RESIDUAL`(c_5), `FUSED`(c_3), `MEAN_SPILL`(c_4),
  `M2_SPILL`(c_6) when fuse.
- **SemaphoreSpecs**: none.
- **TensorParameters**: `INPUT_T`, `OUTPUT_T`, `RECIP_T`, plus `RESIDUAL_T` when fuse.
- **WorkUnitSpecs** (1): `{reader, writer, compute}` over `all_cores`.
- **Endpoint census / disposition**:

  | DFB | producer | consumer | disposition |
  |---|---|---|---|
  | `INPUT` | reader | compute | 1:1 |
  | **`SCRATCH`** | **reader** | **compute** | **1P + 1C — see below** |
  | `RESIDUAL` *(fuse)* | reader | compute | 1:1 |
  | `FUSED`, `MEAN_SPILL`, `M2_SPILL` *(fuse)* | compute | compute | **self-loop** |
  | `RECIP` | compute | compute | **self-loop** (1 toucher, raw peek only) |
  | `OUT` | compute | writer | 1:1 |

  **`SCRATCH` (c_1) departs from the brief's disposition, and had to.** The brief calls for
  `advanced_options.allow_instance_multi_binding` because the census has two locked producers: the
  shared reader unconditionally pushes a reduce-scaler tile into c_1
  (`device/kernels/dataflow/reader_unary_interleaved_ln_rm_gb_pre_allgather.cpp:31-32` — correct in
  Pre 1D, where c_1 *is* the scaler CB), and the compute kernel uses c_1 as its post-Welford transpose
  scratch with its own `reserve_back(2)`/`push_back(2)`/`wait_front(2)`/`pop_front(2)`
  (`device/kernels/compute/layernorm_pre_allgather_welford.cpp:215-220,277-283,290-293`).
  Binding that census literally — reader PRODUCER, compute PRODUCER **and** CONSUMER, flag set — is
  **rejected by the framework**: the self-loop rule requires the producer KernelSpec set to equal the
  consumer set, and that check is *not* relaxed by the multi-binding flag. The minimal legal assignment
  is reader = PRODUCER, compute = CONSUMER, no flag: the per-node census is exactly 1P + 1C, no
  self-loop is declared, and on Gen1 the DFB lowers to a plain circular buffer whose FIFO pointers live
  in L1 and are driven by whichever RISC executes the call — so the compute kernel's packer-side
  `push_back` behaves exactly as it does today. Endpoint role also does not affect the per-DFB data
  format or the `unpack_modes` slot, both of which are keyed on the DFB itself.

### Variant: Post default

- **KernelSpecs** (3): `post_reader`, `post_writer`, `post_compute`.
- **DataflowBufferSpecs**: `INPUT`(c_0), `STATS`(c_1), `EPS`(c_4), `REDUCE`(c_5), `VAR`(c_8),
  `RECIP_SQRT_VAR`(c_10), `OUT`(c_14) always; `GAMMA`(c_2) when gamma; `BETA`(c_3) when beta;
  `STATS_REDUCED`(c_6), `MEAN_SQUARED`(c_7) and `X_MINUS_MEAN`(c_11) when `!is_rmsnorm`;
  `X_NORMED`(c_12) when the normalized result still needs gamma or beta applied to it
  (`gamma || beta` for layernorm, `gamma` for RMSNorm, which applies beta only alongside gamma);
  `TIMES_GAMMA_OUT`(c_13) when **both** gamma and beta are present (see below).
  **`c_9` (var+eps) gets no spec — dead CB, dropped.** Three further drops fall out of the conditions
  above and are config-conditional rather than absolute: `c_6` is untouched on the RMSNorm path (that
  kernel reduces the stats straight into `c_8`), `c_12` is untouched when neither gamma nor beta is
  applied, and `c_13` is untouched when beta is present without gamma.
- **SemaphoreSpecs**: none.
- **TensorParameters**: `INPUT_T`, `STATS_T`, `OUTPUT_T`, plus `GAMMA_T` / `BETA_T` when present.
- **WorkUnitSpecs** (1): `{reader, writer, compute}` over `all_cores` (whichever grid the 1D or 2D
  branch produced).
- **Endpoint census / disposition**:

  | DFB | producer | consumer | disposition |
  |---|---|---|---|
  | `INPUT`, `STATS`, `GAMMA`, `BETA`, `EPS`, `REDUCE` | reader | compute | 1:1 |
  | `STATS_REDUCED`, `MEAN_SQUARED`, `VAR`, `RECIP_SQRT_VAR`, `X_MINUS_MEAN`, `X_NORMED`, `TIMES_GAMMA_OUT` | compute | compute | **self-loop** |
  | `OUT` | compute | writer | 1:1 |

- **`TIMES_GAMMA_OUT` (c_13) allocation condition changes — an invoker-authorized functional fix.**
  Legacy allocates c_13 only under `if (!is_rmsnorm)` + `if (beta)`
  (`device/layernorm_post_all_gather_program_factory.cpp:517-545`), but `rmsnorm_post_allgather.cpp`
  sets `cb_times_gamma_out_idx = c_13` whenever `do_gamma && do_beta` and then drives it
  (`:63-65,153-182`), so RMSNorm + gamma + beta drives an unconfigured CB index today (audit *Misc
  anomalies* #1). The port allocates the DFB whenever `beta` is present, RMSNorm included. This is a
  **functional change**, made only because the invoker explicitly directed it after the audit's question
  was raised; it is called out prominently in `METAL2_PORT_REPORT.md`.

### Variant: Post Welford

- **KernelSpecs** (3): `postwf_reader`, `postwf_writer`, `postwf_compute`.
- **DataflowBufferSpecs**: `INPUT`(c_0), `STATS`(c_1), `EPS`(c_4), `REDUCE`(c_5), `STATS_REDUCED`(c_6),
  `RECIP_SQRT_VAR`(c_10), `X_MINUS_MEAN`(c_11), `X_NORMED`(c_12), `OUT`(c_14); `GAMMA`(c_2) when gamma;
  `BETA`(c_3) when beta; `TIMES_GAMMA_OUT`(c_13) when beta.
  **`c_7`, `c_8`, `c_9` get no spec — dead CBs, dropped.**
- **TensorParameters / WorkUnitSpecs**: as the default post factory.
- **Endpoint census / disposition**: as the default post factory, plus:

  | DFB | producer | consumer | disposition |
  |---|---|---|---|
  | `REDUCE` (c_5) | reader | reader | **self-loop** — the shared reader fills it and this factory's compute kernel never reads it (no `c_5` anywhere in `layernorm_post_allgather_welford.cpp`). Orphan producer, one toucher. |

  The `is_rmsnorm` compute-source ternary
  (`device/layernorm_post_all_gather_welford_program_factory.cpp:315-319`) is unreachable — validation
  rejects RMSNorm + Welford before the factory runs. It is kept as written, and the spec is built
  against the Welford kernel.

### Op-owned tensors

**None** in any factory. The Pre-Welford `recip_tensor` looks like one and is not: it arrives through
`tensor_args_t` (`device/layernorm_pre_all_gather_device_operation_types.hpp:25`) from the separate
`ttnn.create_layer_norm_reciprocals` API, so it is an ordinary caller-supplied `TensorParameter` whose
DFB is `borrowed_from` it.

---

## Preserved Multiplicity

**No work-split multiplicity in legacy** — every factory puts its per-group row count in a **runtime**
arg, so no factory emits two `KernelDescriptor`s of one source for a work split, and no per-group CTA
exists to preserve. Nothing here is the demoting-CTA anti-pattern.

One `KernelSpec` multiplicity *is* introduced, for a different reason (the DFB placement invariant, not
a work split):

```
Legacy KernelDescriptor [compute] of source device/kernels/compute/layernorm_pre_allgather_2d.cpp
  → KernelSpecs [pre2d_compute_merge, pre2d_compute_worker] of the same source
  → in WorkUnitSpecs [wu_merge (merge_cores), wu_worker (y >= 1 rows)]
  → sharing DFBs: INPUT (CONSUMER each), REDUCE (CONSUMER each), RESIDUAL (CONSUMER each, fuse),
    FUSED / X2 (PRODUCER + CONSUMER each — self-loop), PARTIAL_OUT (PRODUCER each),
    X2_MERGE / ZERO (CONSUMER each); OUT_FINAL (PRODUCER) on pre2d_compute_merge only.
```

The two instances cover **disjoint** node sets, so each node still sees exactly one instance and each
shared DFB is an ordinary per-node 1:1 — the legal multiple-KernelSpecs-per-endpoint case, not
`allow_instance_multi_binding`.

---

## Dropped Plumbing

### Buffer-address RTAs → `TensorBinding`

| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| Pre 1D / Pre Welford reader RTA slot 0 (`…pre_all_gather_program_factory.cpp:174`, `…welford_…:187`) | `a.buffer()` | `TensorBinding{INPUT_T, "src"}` |
| Pre 1D / Pre Welford reader RTA slot 4 (`…:179`, `…welford_…:192`) | `b->buffer()` | `TensorBinding{RESIDUAL_T, "res_src"}` (fuse only) |
| Pre 2D reader RTA slots 0 / 8 (`…pre_all_gather_program_factory.cpp:429,438`) | `a.buffer()`, `b->buffer()` | `TensorBinding{INPUT_T, "src"}`, `{RESIDUAL_T, "res_src"}` |
| Writer RTA slot 0, all five factories (`…:184`, `…:444`, `…post_…:328,362`, `…welford_…:197`, `…postwf_…:361,396`) | `output.buffer()` | `TensorBinding{OUTPUT_T, "dst"}` |
| Post reader RTA slots 0, 6, 7, 8 (`…post_all_gather_program_factory.cpp:316-325,351-360`; `…welford_…:346-358,382-394`) | `a.buffer()`, `gamma_buffer`, `beta_buffer`, `stats.buffer()` | `TensorBinding`s `INPUT_T` / `GAMMA_T` / `BETA_T` / `STATS_T` |

All are **Case 1** (consumed through a `TensorAccessor`). No Case 2 anywhere — no kernel does raw NoC
arithmetic on a tensor base, so `get_bank_base_address` is never needed.

`gamma` and `beta` are independently optional and today pass `nullptr` (→ `0u`) when absent, with a
matching `TensorAccessorArgs(nullptr)` (`…post_all_gather_program_factory.cpp:111-112,251-254`).
In Metal 2.0 they become **conditionally declared** `TensorParameter`s: the parameter, the binding, and
the accessor's kernel-side construction all appear only when the tensor is present, gated by the same
`FUSE_GAMMA` / `FUSE_BETA` defines the reader already uses. The `nullptr` / `0u` sentinel disappears.

### Magic CB indices → `DFBBinding`

Every `constexpr uint32_t dfb_… = tt::CBIndex::c_N;` in all ten kernels is replaced by a `dfb::<name>`
handle from the binding. There are **no** CB indices in any CTA or RTA list in this op, so no CTA slot
drops on this account — the indices were kernel-side literals only.

### `TensorAccessorArgs` plumbing → binding mechanism

| host site | kernel-side chain |
|---|---|
| `TensorAccessorArgs(a.buffer()).append_to(reader_cta)` — all 5 factories | `TensorAccessorArgs<1>()` (pre), `<3>()` (2D), `<9>()` (post) |
| `TensorAccessorArgs(b->buffer())` — pre ×3, fuse | `TensorAccessorArgs<src_args.next_compile_time_args_offset()>()` |
| `TensorAccessorArgs(output.buffer()).append_to(writer_cta)` — all 5 | `TensorAccessorArgs<1>()` |
| `TensorAccessorArgs(stats.buffer())`, `(gamma…)`, `(beta…)` — post ×2 | the `next_compile_time_args_offset()` chain at `…reader_…post_allgather.cpp:95-98` |

All replaced by `TensorAccessor(tensor::<name>)`.

### Page-size 3rd-argument CTAs

| legacy CTA slot | value | fate |
|---|---|---|
| Post reader CTA 2 (`gamma_stick_size`) | `element_size()*1024` (TILE) or `padded_shape[-1]*element_size()` (RM) | **dropped** — fed only `TensorAccessor(gamma_args, gamma_addr, gamma_stick_size)` (`…reader_…post_allgather.cpp:104`), Class 2 (equals the true aligned page size) |
| Post reader CTA 3 (`beta_stick_size`) | same for beta | **dropped** — `…reader_…post_allgather.cpp:107` |

The host-side `gamma_stick_size` / `beta_stick_size` computations stay — they still produce the
`gamma_is_row_major` / `beta_is_row_major` flags and carry the `is_power_of_two_at_least_32` validation
— but their byte values stop being emitted. No `dynamic_tensor_shape` relaxation is involved.

### Semaphore-ID CTAs → `SemaphoreBinding`

| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| Pre 2D reader CTA slot 1 (`…pre_all_gather_program_factory.cpp:393`) | `reducer_semaphore_id` (0) | `SemaphoreBinding{REDUCER, "reducer"}`; the kernel builds `Semaphore<> reducer_sem(sem::reducer)` |

### Positional CTAs → named CTAs

| kernel | legacy positional list | named CTAs assigned |
|---|---|---|
| pre reader (1D, Welford) | `blk` | `blk` |
| pre 2D reader | `blk`, `reducer_semaphore_id`, `cores_y` | `blk`, `num_cores_to_wait` (the semaphore id becomes a binding) |
| writer (all 5) | `blk` | `blk` |
| pre 1D compute (LN + RMS) | `Wt`, `blk` | `Wt`, `blk` |
| pre 2D compute | `NCHt`, `Wt`, `blk`, `num_cores_y` | same four names |
| pre Welford compute | `Wt`, `W`, `blk` + named `welford_unpack_fp32_active` | `Wt`, `W`, `blk`; the already-named CTA carries over unchanged |
| post reader | `blk`, `stats_tiles_cols`, `gamma_stick_size`, `beta_stick_size`, `gamma_is_row_major`, `beta_is_row_major`, `dfb_length`, `Wt`, `reduce_factor` | `blk`, `stats_tiles_cols`, `gamma_is_row_major`, `beta_is_row_major`, `dfb_length`, `Wt`, `reduce_factor` (the two stick sizes drop) |
| post compute (LN + RMS) | `Wt`, `blk`, `stats_tiles_cols`, `do_gamma`, `do_beta`, `fp32_dtype`, `float32_reduction`, `legacy_rsqrt`, `dfb_length` | `Wt`, `blk`, `stats_tiles_cols`, `fp32_dtype`, `float32_reduction`, `legacy_rsqrt`, `dfb_length`. **`do_gamma` / `do_beta` drop**: their only readers were the `if constexpr` gates that become the `FUSE_GAMMA` / `FUSE_BETA` defines, so nothing is left to read them |
| post Welford compute | `Wt`, `W`, `blk`, `stats_tiles_cols`, `do_gamma`, `do_beta`, `fp32_dtype`, `dfb_length` | `Wt`, `W`, `blk`, `stats_tiles_cols`, `fp32_dtype`, `dfb_length`; `do_gamma` / `do_beta` drop for the same reason |

### Positional RTAs → named RTAs

**No varargs anywhere.** Every kernel reads each runtime arg at a fixed literal index as a distinct
field, so all RTAs become named:

| kernel | named RTAs |
|---|---|
| pre reader (1D, Welford) | `NCHt`, `Wt`, `tile_offset` |
| pre 2D reader | `NCHt`, `Wt`, `tile_offset`, `is_merge_core`, `reduce_core_noc_x`, `reduce_core_noc_y`, `y` |
| writer (all 5) | `num_tiles`, `tile_offset` |
| pre compute (1D, 2D, Welford) | `NCHt` (the 2D kernel's `is_merge_core` RTA becomes the `IS_MERGE_CORE` define — see *Planned Spec Shape*) |
| post reader | `NCHt`, `tile_offset`, `stats_tile_offset`, `eps`, `y_offset` |
| post compute (all) | `NCHt` |

Two legacy RTA slots disappear with nothing to name:
- Post reader RTA slot 2 (host writes `Wt` / `tiles_per_core_y`) — never read by the kernel, which takes
  its width from a CTA instead.
- Post-Welford reader RTA slot 5 (`packed_winv_value`) — the shared reader has no such field; see
  *Flags* #2.

---

## Applied Patterns

- **Self-loop DFB binding** — every compute-private intermediate: `X2`, `FUSED` (pre 1D / 2D / Welford),
  `MEAN_SPILL`, `M2_SPILL` (pre Welford), `STATS_REDUCED`, `MEAN_SQUARED`, `VAR`, `RECIP_SQRT_VAR`,
  `X_MINUS_MEAN`, `X_NORMED`, `TIMES_GAMMA_OUT` (post ×2). Bound PRODUCER **and** CONSUMER on the one
  compute KernelSpec that touches them, under a single shared `accessor_name`.
- **Sync-free / single-ended CB → self-loop DFB** — `RECIP` (pre Welford `c_2`): a borrowed-memory DFB
  read only through a raw base-pointer peek, with no FIFO ops at all; one toucher, so it self-loops.
  `REDUCE` in the Post-Welford factory: an orphan producer (the shared reader fills it, that factory's
  compute kernel never reads it); one toucher, so it self-loops on the reader.
- **Two-toucher DFB → assign 1P+1C** — `SCRATCH` (pre Welford `c_1`), re-derived from the census; see
  *Planned Spec Shape* for why the brief's multi-binding disposition is not expressible.
- **Conditional / optional DFB bindings** — `RESIDUAL` / `FUSED` / `MEAN_SPILL` / `M2_SPILL` behind
  `FUSE_PRE_ADD`; `GAMMA` / `BETA` / `TIMES_GAMMA_OUT` behind `FUSE_GAMMA` / `FUSE_BETA`; `OUT_FINAL`
  behind `IS_MERGE_CORE`. Each is bound conditionally on the host, with a matching
  `compiler_options.defines` entry, and the kernel `#ifdef`-gates both the alias and every use.
  The legacy `FUSE_PRE_ADD` define is always emitted (as `"0"` or `"1"`) and tested with `#if`; the port
  emits it **only when true** and tests with `#ifdef`, because `#if FUSE_PRE_ADD` would still leave
  `dfb::res` in name lookup on the unfused path.
- **Conditional bindings — promote a CTA gate to a define** — the post compute kernels gate their
  gamma/beta chains on the `do_gamma` / `do_beta` **CTAs** via `if constexpr`, which still performs name
  lookup on the discarded branch. Those gates are promoted to `FUSE_GAMMA` / `FUSE_BETA` preprocessor
  conditionals, and the defines (previously emitted to the reader only) are now emitted to the compute
  kernel too. The `do_gamma` / `do_beta` CTAs then have no reader left and drop with the gates.
- **Same-FIFO aliasing (path-dependent variant)** — the file-scope ternaries that resolve one name to
  different CB indices per path: `dfb_inp_id = FUSE_PRE_ADD ? c_3 : c_0` (pre compute ×3),
  `normed_output_cb = (do_gamma||do_beta) ? c_12 : c_14`, `cb_times_gamma_out = do_beta ? c_13 : c_14`,
  `cb_in_beta = do_gamma ? cb_times_gamma_out : normed_output_cb` (post compute ×3). Each becomes an
  `#ifdef`-gated `constexpr auto` **handle alias** — one DFB, one `DataflowBuffer` object, no second
  `DFBBinding` and no `alias_with`.
- **Multi-variant factories** — the default post factory branches on `use_2d_core_grid` inside
  `create_program_artifacts`; the Pre 1D and Post factories branch on `is_rmsnorm` to select the compute
  source. No class hierarchy.
- **Pass DFB handles directly to LLKs and kernel-lib helpers** — `dfb::name` flows unchanged into
  `reduce_init`, `pack_tile`, `mul_tiles`, `reconfig_data_format`, `binary_op_init_common`,
  `compute_kernel_lib::reduce<…>` (template-parameter position),
  `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<…>` / `prepare_zero_tile<…>` (NTTP),
  `get_pointer_to_cb_data(uint32_t, …)`, and `chain_llk`'s `LLK_Node` `uint32_t` fields. No `.id`
  extraction, no temporary `DataflowBuffer` wrappers.
- **Porting a shared kernel — rung 2** — `rmsnorm_pre_allgather.cpp` and `rmsnorm_post_allgather.cpp` in
  the peer `rmsnorm_distributed` directory get `_metal2` forks beside the originals, plus a pointer
  comment in each original. Nothing else in that directory is touched.
- **Unity-build hygiene** — the five factories share one CMake target and are unity-built, so the
  anonymous-namespace spec-name constants are prefixed per factory (`PRE1D_`, `PRE2D_`, `PREWF_`,
  `POST_`, `POSTWF_`).

---

## Deferred / Flagged

New findings that surfaced during inventory / planning and are **not** acted on in the port:

1. **The Post-Welford factory is broken on `main` today** (*Flags* #2) — a reader CTA/RTA layout
   mismatch, confirmed empirically (garbage output, NaN cosine similarity). The audit did not catch it.
   The port cannot reproduce it, because named arguments have no positional shift to reproduce, so the
   ported path is expected to become correct. Recorded as a Handoff point.
2. **The brief's `allow_instance_multi_binding` disposition for Pre-Welford `c_1` is unimplementable**
   as stated; the framework rejects multi-binding stacked on a compute self-loop. Re-derived from the
   census to 1P + 1C. Recorded as Friction (a gap in the audit's census-to-disposition table).
3. **`c_13` allocation condition changes for RMSNorm + gamma + beta** — an invoker-authorized functional
   fix, not port work. Recorded prominently in the report.
4. **The Pre-2D factory ignores `is_rmsnorm`** (audit *Misc anomalies* #6) — it hardcodes
   `layernorm_pre_allgather_2d.cpp` and forces `out0_tiles = 1`, while `compute_output_specs` sizes a
   LAYERNORM output at two tile columns. Untouched by the port; still on the ops team's plate.
5. **The Post-Welford factory's `is_rmsnorm` compute-source branch is unreachable** (validation rejects
   RMSNorm + Welford). Kept as-is; the spec is built against the Welford kernel.
6. **Test coverage gap** — the Pre-2D, Post-2D and Post-Welford paths are exercised only by
   `mesh_device=(1, 8)` tests, which skip on a single-device host. That is why *Flags* #2 went unnoticed.
