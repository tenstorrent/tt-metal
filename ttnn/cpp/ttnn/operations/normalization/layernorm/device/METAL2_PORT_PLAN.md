# Port Plan — LayerNormMultiCoreProgramFactory (interleaved)

Port plan for the **interleaved** factory of `LayerNormDeviceOperation`, ported from `ProgramDescriptor` to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

**Scope:** **both** `LayerNormMultiCoreProgramFactory` (interleaved) and `LayerNormShardedProgramFactory` (sharded) are now ported in this PR. The sharded plan is appended at the end of this document (see [Sharded Port Plan](#sharded-port-plan-layernormshardedprogramfactory)).

## Legacy Inventory

### Factory shape
- Concept: `ProgramDescriptorFactoryConcept` (returns `tt::tt_metal::ProgramDescriptor` from `create_descriptor`).
- Variants: a single factory entry-point that selects kernel paths and configures CBs at run time based on a handful of host-side booleans:
  - `use_welford` (program-config field; toggles Welford-variant compute + reader)
  - `large_tensor_needed` (computed: triggered when CBs don't fit L1; toggles "large tensor" kernels)
  - `input_is_row_major` (input layout; toggles TILIZE_IN/UNTILIZE_OUT paths and RM-staging CBs)
  - `use_row_major_kernel` (gamma/beta layout; toggles the `_rm_gb` reader)
  - `fuse_pre_add` (`b` residual present; toggles FUSE_PRE_ADD define + CB)
  - `gamma.has_value()` / `beta.has_value()` (toggle FUSE_GAMMA / FUSE_BETA defines + CBs)
  - `rms_norm` (toggles RMSNORM define + skips a couple of CBs)
  - `fused_activation` (toggles `ACTIVATION_*` defines on compute)

  Each path resolves to a different `kernel_source` for reader & compute, plus a different define set. The port preserves these as runtime conditionals inside `create_program_spec`.

### Kernels

Single-table representation (all variants build one reader / one writer / one compute KernelDescriptor):

| unique_id | source (one of) | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | config |
|---|---|---|---|---|---|---|---|---|
| reader | `reader_unary_interleaved_ln.cpp` / `reader_unary_interleaved_ln_large_tensor.cpp` / `reader_unary_interleaved_ln_large_tensor_welford.cpp` / `reader_unary_interleaved_ln_rm_gb.cpp` | `all_cores` | `[block_size, (use_welford?), W, TAArgs(a), TAArgs(b), TAArgs(gamma), TAArgs(beta), (elem_size_bytes|gamma_stick|beta_stick|tile_size)]` | `{cb_in, cb_inb, cb_scaler, cb_eps, cb_gamma, cb_beta, cb_out, cb_ex, cb_ex2, cb_xmm2, cb_ex2pe, cb_fusion, cb_x, cb_xmm, cb_reciprocals, cb_accumulate, cb_in_rm, cb_out_rm}` | `[a_addr, NCHt, Wt, start_tile_row, packed_one, eps, gamma_addr, beta_addr, b_addr, (H_logical)?]` | — | `FUSE_PRE_ADD`, `FUSE_GAMMA`, `FUSE_BETA`, `RMSNORM`, `TILIZE_IN` (conditional) | `ReaderConfigDescriptor` |
| writer | `writer_unary_interleaved_start_id_blocked.cpp` / `writer_unary_interleaved_start_id_blocked_rm_output.cpp` | `all_cores` | `[block_size, TAArgs(output), (elem_size_bytes)?]` | (same `cb_named_args`) | `[dst_addr, Wt, num_tile_rows, writer_start, (H_logical)?]` | — | (none beyond cb_named_args) | `WriterConfigDescriptor` |
| compute | `layernorm.cpp` / `layernorm_large_tensor.cpp` / `layernorm_welford.cpp` / `layernorm_large_tensor_welford.cpp` | `all_cores` | `[Wt, block_size, do_gamma, do_beta, fp32_dest_acc_en, (W,TILE_SIZE,rms_norm,fuse_pre_add) [welford] / (float32_reduction, legacy_rsqrt, W, tile_width) [non-welford]]` | (same `cb_named_args`) | `[num_tile_rows_per_core]` | — | `FUSE_PRE_ADD`, `RMSNORM`, `TILIZE_IN`, `UNTILIZE_OUT`, `ACTIVATION_*` | `ComputeConfigDescriptor` |

### CBs

`cb_named_args` aligns CB index → CB id everywhere:

| index | symbol | total_size | core_ranges | data_format | page_size | tile (if set) |
|---|---|---|---|---|---|---|
| `c_0`  | `cb_in`         | `in0_t  * in_single_tile_size`        | all_cores | `in_data_format`        | `in_single_tile_size`       | — |
| `c_1`  | `cb_inb`        | `in1_t  * inb_single_tile_size`       | all_cores | `inb_data_format`       | `inb_single_tile_size`      | — (conditional on `b`) |
| `c_2`  | `cb_scaler`     | `in2_t  * scaler_tile_size`           | all_cores | `Float16_b`             | `scaler_tile_size`          | — (conditional on `!use_welford`) |
| `c_3`  | `cb_eps`        | `in3_t  * bfloat16_tile_size`         | all_cores | `Float16_b`             | `bfloat16_tile_size`        | — |
| `c_5`  | `cb_gamma`      | `in5_t  * gamma_single_tile_size`     | all_cores | `gamma_cb_data_format`  | `gamma_single_tile_size`    | — (conditional on `gamma.has_value()`) |
| `c_6`  | `cb_beta`       | `in6_t  * beta_single_tile_size`      | all_cores | `beta_cb_data_format`   | `beta_single_tile_size`     | — (conditional on `beta.has_value()`) |
| `c_16` | `cb_out`        | `out0_t * out_single_tile_size`       | all_cores | `out_data_format`       | `out_single_tile_size`      | — |
| `c_18` | `cb_ex`         | `im1_t  * single_tile_size`           | all_cores | `cb_data_format`        | `single_tile_size`          | — (conditional on `!rms_norm`) |
| `c_19` | `cb_ex2`        | `im2_t  * single_tile_size`           | all_cores | `cb_data_format`        | `single_tile_size`          | — |
| `c_20` | `cb_xmm2`       | `im3_t  * single_tile_size`           | all_cores | `cb_data_format`        | `single_tile_size`          | — (conditional on `!use_welford`) |
| `c_21` | `cb_ex2pe`      | `im4_t  * single_tile_size`           | all_cores | `cb_data_format`        | `single_tile_size`          | — |
| `c_22` | `cb_fusion`     | `im5_t  * single_tile_size`           | all_cores | `cb_data_format`        | `single_tile_size`          | — (conditional on `gamma|beta`) |
| `c_23` | `cb_x`          | `im6_t  * single_tile_size`           | all_cores | `cb_data_format`        | `single_tile_size`          | — (conditional on `b && !rms_norm`) |
| `c_24` | `cb_xmm`        | `im0_t  * single_tile_size`           | all_cores | `cb_data_format`        | `single_tile_size`          | — (conditional on `!rms_norm || fuse_pre_add || large_tensor_needed`) |
| `c_25` | `cb_reciprocals`| `reciprocal_CB_size_bytes`            | all_cores | `Float32`               | `reciprocal_CB_size_bytes`  | **borrowed** from `recip_tensor` (conditional on `use_welford`) |
| `c_26` | `cb_accumulate` | `large_tensor_acc_tile_size`          | all_cores | `Float32`/`cb_data_format` | `large_tensor_acc_tile_size` | — (conditional on `large_tensor_needed && !use_welford`) |
| `c_27` | `cb_in_rm`      | `in_rm_size`                          | all_cores | `in_data_format`        | `in_single_tile_size`       | — (conditional on `input_is_row_major`) |
| `c_28` | `cb_out_rm`     | `out_rm_size`                         | all_cores | `out_data_format`       | `out_single_tile_size`      | — (conditional on `input_is_row_major`) |

### Semaphores

none

### Tensor accessors

| host site (file:line) | originating Tensor | RTA slot (host) | CTA offset (kernel) |
|---|---|---|---|
| reader CTAs (in `reader_compile_time_args`): `TensorAccessorArgs(a.buffer())` (line 358) | `tensor_args.input` (input `a`) | reader RTA[0] = `a_addr` | reader CTA[3..] = `src0_args` |
| reader CTAs: `TensorAccessorArgs(b ? b->buffer() : nullptr)` (line 359) | `tensor_args.residual_input_tensor` (`b`) | reader RTA[8] = `b_dram_addr` | reader CTA = `src1_args` |
| reader CTAs: `TensorAccessorArgs(gamma ? gamma->buffer() : nullptr)` (line 360) | `tensor_args.weight` (`gamma`) | reader RTA[6] = `gamma_dram_addr` | reader CTA = `gamma_args` |
| reader CTAs: `TensorAccessorArgs(beta ? beta->buffer() : nullptr)` (line 361) | `tensor_args.bias` (`beta`) | reader RTA[7] = `beta_dram_addr` | reader CTA = `beta_args` |
| writer CTAs: `TensorAccessorArgs(output.buffer())` (line 378) | `tensor_return_value` (output) | writer RTA[0] = `dst_addr` | writer CTA[1..] = `dst_args` |

The recip-tensor (`recip_tensor`) is *not* read via `TensorAccessor` — it's exposed as a borrowed-memory CB (CB 25) backed by `recip_tensor.value().buffer()`. Translates to `DataflowBufferSpec::borrowed_from = RECIP_TENSOR`.

### Work split

- Driver: `split_work_to_cores(requested_cores, num_tile_rows, /*row_wise=*/true)`
- `num_cores`: as returned (varies with `num_tile_rows = NC * Ht`)
- `core_group_1`: `num_tile_rows_per_core_group_1` rows per core (typically `ceil(num_tile_rows / num_cores)`)
- `core_group_2`: `num_tile_rows_per_core_group_2` rows per core (one fewer than group_1 when not evenly divisible)

The per-group dimension (`num_tile_rows_per_core`) is currently passed as an **RTA** to all three kernels (`reader_args[1] = NCHt`, `writer_args[2] = num_tile_rows`, `compute_runtime_args = {num_tile_rows_per_core}`). The legacy factory builds **one** `KernelDescriptor` per kernel covering `all_cores`. The [Demoting per-group CTA to RTA](metal2_port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta) anti-pattern only fires when legacy already had two `KernelDescriptor`s per work split — it does not fire here. The port preserves the legacy shape: one `KernelSpec` per kernel covering `all_cores`, with `num_tile_rows_per_core` as a per-node RTA.

### Cross-op kernels

None — every kernel `source` lies under `device/kernels/`.

### Flags

None.

## Planned Spec Shape

- **KernelSpecs**: three — `reader`, `writer`, `compute`. Source path remains a runtime choice (selected from the same legacy table); per-spec defines and CTAs vary per the legacy switches.
- **DataflowBufferSpecs**: one per CB in the legacy table above — created conditionally with the same `if` guards. The `c_25` reciprocal LUT uses `borrowed_from = RECIP_TENSOR`.
- **SemaphoreSpecs**: none.
- **TensorParameters**: up to six, declared conditionally to match the legacy `has_value()` guards:
  - `INPUT_A` — always present (input `a`).
  - `OUTPUT` — always present (output).
  - `RESIDUAL_B` — when `b.has_value()`.
  - `GAMMA` — when `gamma.has_value()`.
  - `BETA` — when `beta.has_value()`.
  - `RECIP` — when `use_welford` (backs the borrowed-memory CB 25).
- **WorkUnitSpecs**: one (`"main"`) covering `{reader, writer, compute}` on `all_cores`.

## Preserved Multiplicity

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (multi-binding) |
|---|---|---|---|

(none — no work-split multiplicity in legacy)

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| `layernorm_op_multi_core.cpp:358` host + `reader_unary_interleaved_ln.cpp:82` kernel | `TensorAccessorArgs(a.buffer()).append_to(reader_cta)` + `TensorAccessorArgs<3>()` + `TensorAccessor(src0_args, src_addr)` | `TensorParameter(INPUT_A)` + `TensorBinding(INPUT_A, "src_a")` on reader + `TensorAccessor(ta::src_a)` |
| `layernorm_op_multi_core.cpp:359` host + reader kernel | `TensorAccessorArgs(b->buffer())` + `TensorAccessor(src1_args, b_addr)` | `TensorParameter(RESIDUAL_B)` + `TensorBinding(RESIDUAL_B, "src_b")` + `TensorAccessor(ta::src_b)` (declared conditionally on `b.has_value()`) |
| `layernorm_op_multi_core.cpp:360` host + reader kernel | `TensorAccessorArgs(gamma->buffer())` + `TensorAccessor(gamma_args, gamma_addr)` | `TensorParameter(GAMMA)` + `TensorBinding(GAMMA, "gamma")` + `TensorAccessor(ta::gamma)` (declared conditionally on `gamma.has_value()`) |
| `layernorm_op_multi_core.cpp:361` host + reader kernel | `TensorAccessorArgs(beta->buffer())` + `TensorAccessor(beta_args, beta_addr)` | `TensorParameter(BETA)` + `TensorBinding(BETA, "beta")` + `TensorAccessor(ta::beta)` (declared conditionally) |
| `layernorm_op_multi_core.cpp:378` host + `writer_unary_interleaved_start_id_blocked.cpp:30` kernel | `TensorAccessorArgs(output.buffer())` + `TensorAccessor(dst_args, dst_addr)` | `TensorParameter(OUTPUT)` + `TensorBinding(OUTPUT, "output")` + `TensorAccessor(ta::output)` |
| reader RTA[0] `a_addr`, RTA[6] `gamma_dram_addr`, RTA[7] `beta_dram_addr`, RTA[8] `b_dram_addr` | per-core buffer-address RTAs | gone — auto-injected by `TensorBinding` |
| writer RTA[0] `dst_addr` | buffer-address RTA | gone — auto-injected by `TensorBinding` |
| Named CTAs `cb_named_args = {{"cb_in", c_0}, ..., {"cb_out_rm", c_28}}` (line 424) duplicated across reader/writer/compute | named CTA → CB index, then kernel reads it back via `get_named_compile_time_arg_val` and constructs `CircularBuffer(cb_id)` | gone — replaced by `DFBBinding(<DFB_NAME>, "<local_accessor_name>", PRODUCER/CONSUMER)` on each kernel + `DataflowBuffer(dfb::<local_accessor_name>)` in kernel code |
| Reader positional CTAs `[block_size, use_welford?, W, ..., (elem_size_bytes | gamma_stick | beta_stick | tile_size)]` | positional CTAs | Named CTAs: `block_size`, `use_welford`, `W`, `last_arg` (the final positional value that varies by branch) |
| Writer positional CTAs `[block_size, (elem_size_bytes)?]` | positional CTAs | Named CTAs: `block_size`, `elem_size_bytes` (conditional) |
| Compute positional CTAs `[Wt, block_size, do_gamma, do_beta, fp32_dest_acc_en, ...]` | positional CTAs | Named CTAs: `Wt`, `block_size`, `do_gamma`, `do_beta`, `FLOAT32_DTYPE`, plus path-specific (`FLOAT32_REDUCTION`/`LEGACY_RSQRT`/`W`/`tile_width` non-welford OR `W`/`TILE_SIZE`/`rms_norm`/`fuse_pre_add` welford) |

## Applied Patterns

- **Borrowed-memory DFB** for `cb_reciprocals` (CB 25): `DataflowBufferSpec::borrowed_from = RECIP_TENSOR` — translates the legacy `CBDescriptor::buffer = recip_tensor.value().buffer()` pattern. Per the [Dynamic CircularBuffer entry](port_op_to_metal2_audit.md#dynamic-circularbuffer-cb-built-on-borrowed-buffer-memory--landed) (LANDED).
- **Multi-variant factory** branching inside `create_program_spec` (legacy already does this via runtime conditionals on `large_tensor_needed`, `use_welford`, `input_is_row_major`, `use_row_major_kernel`; the port preserves them verbatim).
- **Pass DFB handles directly to LLKs and kernel-lib helpers** — kernels call `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<cb_scaler, ...>()` and `generate_bcast_col_scalar(cb_eps, eps)` with raw CB ids today; replace with `<dfb::cb_scaler, ...>` and `(dfb::cb_eps, eps)` via the implicit `DFBAccessor::operator uint32_t()` conversion. No `.id` extraction.

## Deferred / Flagged

- None — audit was GREEN with no yellow items.
- The `recip_tensor` is currently an extra field on `tensor_args_t` (see `layernorm_device_operation_types.hpp`); it must be plumbed into the factory's `tensor_args` (already there) and used as the source for the `RECIP` `TensorParameter` declared conditionally on `use_welford`.

## Kernel-source migration

Each kernel source needs three mechanical changes to be Metal 2.0–native:

1. `#include "experimental/kernel_args.h"` (after `dataflow_api.h` / `compute_kernel_api.h`).
2. Replace `get_compile_time_arg_val(N)` / `get_named_compile_time_arg_val("name")` with `get_arg(args::<name>)`.
3. Replace `TensorAccessorArgs<N>()` + `TensorAccessor(args, addr)` with `TensorAccessor(ta::<accessor_name>)`.
4. Replace `CircularBuffer cb_x(cb_id_x)` with `DataflowBuffer cb_x(dfb::<local_accessor_name>)`. Pass `dfb::<name>` directly in kernel-lib template args and call sites.
5. Drop the buffer-address RTA reads (`get_arg_val<uint32_t>(0)`, `(5)`, `(6)`, `(7)`, `(8)`) — re-index remaining RTAs as named args.

The compute kernel doesn't use `TensorAccessor` (it operates only on CBs), so its migration is changes 1, 2, and 4 only.

---

# Sharded Port Plan — LayerNormShardedProgramFactory

## Legacy Inventory

### Factory shape
- Concept: `ProgramDescriptorFactoryConcept` (returns `tt::tt_metal::ProgramDescriptor` from `create_descriptor`).
- Variants: runtime dispatch over (a) `is_pre_all_gather` / `is_post_all_gather` / normal mode (mutually exclusive); (b) `use_welford` (only in normal mode); (c) `use_row_major_kernel` (gamma/beta layout, only in normal+post modes); (d) `mcast_1d` (vs 2D mcast); (e) `use_two_stage_reduce`; (f) `skip_write_back` (post-allgather only). The legacy code already builds the spec from these flags via helper structs (GridParams / WorkerDistribution / CoreRanges / KernelLayout); the port reuses those structurally and only changes the spec-emission step.

### Kernels

Six logical kernels, with up to two `KernelSpec`s per kernel (sender vs receiver, all-to-all vs not-all-to-all):

| unique_id | source (one of) | core_ranges | role |
|---|---|---|---|
| reader_sender | `reader_mcast_sender_unary_sharded_ln{,_pre_allgather,_post_allgather}.cpp` | `sender_cores` (1×1) | mcast leader |
| reader_receiver_all_to_all | `reader_mcast_receiver_unary_sharded_ln{,_pre_allgather,_post_allgather}.cpp` | `all_to_all_workers_except_sender` (if mcast) | receive + participate in all-to-all reduce |
| reader_receiver | (same as above) | `not_all_to_all_workers` (if any) | receive only |
| writer_sender | `writer_unary_sharded_ln{,_rm_gb,_pre_all_gather}.cpp` | `all_to_all_cores` | global-scaler-aware writer |
| writer_receiver | (same writer source) | `not_all_to_all_workers` | non-scaler writer |
| compute_all_to_all | `compute/layernorm_sharded{,_welford,_pre_allgather,_post_allgather}.cpp` | `all_to_all_cores` | compute with all-to-all |
| compute_not_all_to_all | (same compute source) | `not_all_to_all_workers` | compute without all-to-all |

This is a **preserved-multiplicity** case (two compute KernelSpecs of the same source, differing only on a single CTA `is_all_to_all_worker`) — see [Anti-pattern: Demoting per-group CTA to RTA](metal2_port_patterns.md#anti-pattern-demoting-per-group-cta-to-rta).

### CBs

The sharded factory creates between 7 and 22 CBs depending on flags. Notable cases:

| index | symbol | borrowed memory | gating |
|---|---|---|---|
| `c_0`  | `cb_in` (in0 sharded)              | **yes** — `a.buffer()`                                | always |
| `c_1`  | `cb_inb` (in1 sharded)             | **yes** — `b.buffer()`                                | `has_b` |
| `c_14` | (in0 also under is_pre_all_gather) | **yes** — `a.buffer()`                                | `is_pre_all_gather && has_b` |
| `c_2`  | `cb_scaler` (in2)                  | no                                                    | `!use_welford` |
| `c_3`  | `cb_eps` (in3)                     | no                                                    | `!use_welford` |
| `c_4`  | `cb_scaler_global` (in4)           | no                                                    | `!use_welford` |
| `c_5`  | `cb_gamma`                         | no (sender reads DRAM via TensorAccessor)             | `has_gamma` |
| `c_6`  | `cb_beta`                          | no                                                    | `has_beta` |
| `c_7`  | `cb_stats`                         | **yes** — `stats.buffer()`                            | `is_post_all_gather` |
| `c_8`  | `cb_ex_partial`                    | no                                                    | `!rms_norm` |
| `c_9`  | `cb_ex`                            | no                                                    | `!rms_norm` |
| `c_10` | `cb_ex_external`                   | no                                                    | `!rms_norm` |
| `c_11` | `cb_ex_partial2`                   | no                                                    | `!use_welford` |
| `c_12` | `cb_ex2`                           | no                                                    | `!use_welford` |
| `c_13` | `cb_ex_external2`                  | no                                                    | `!use_welford` |
| `c_15` | `cb_ex_global`                     | no                                                    | always |
| `c_16` | `cb_out`                           | **yes** — `output.buffer()` (normal/pre paths) or none (post + !skip_write_back) | always |
| `c_17` | `cb_out_resharded`                 | **yes** — `output.buffer()`                           | `is_post_all_gather && !skip_write_back` |
| `c_18` | `cb_xmm`                           | no                                                    | always |
| `c_19` | `cb_var`                           | no                                                    | `is_post_all_gather` |
| `c_20` | `cb_ex2pe`                         | no                                                    | `!use_welford` |
| `c_21` | `cb_stats_reduced`                 | no                                                    | `is_post_all_gather` |
| `c_22` | `cb_transpose` (intermed for Welford) | no                                                 | `use_welford` |
| `c_24` | `cb_x`                             | no                                                    | always |
| `c_25` | `cb_reciprocals`                   | **yes** — `recip_tensor.buffer()`                     | `use_welford` |

Several CBs have non-default `core_ranges` (sender_cores only for `c_7`/`c_19`/`c_21`/`c_16` in pre-all-gather, `all_worker_and_storage_cores` for `c_17`). The Metal 2.0 port derives DFB placement from kernel-binding membership; the legacy explicit core_ranges become implicit.

### Semaphores

| id (legacy) | name | core_type | core_ranges | initial_value |
|---|---|---|---|---|
| 0 | reduce_sender    | WORKER | `all_cores` | 0 |
| 1 | reduce_receiver  | WORKER | `all_cores` | 0 |
| 2 | reduce_second_stage | WORKER | `all_cores` | 0 |

All three semaphores have `initial_value = 0` and are bound to all worker cores. Translates cleanly to three `SemaphoreSpec`s with `target_nodes = all_cores`.

### Tensor accessors

Only used in the writer kernels (`writer_unary_sharded_ln{,_rm_gb,_pre_all_gather}.cpp`) for DRAM gamma/beta access. The sharded reader kernels do NOT use TensorAccessor — they read tensor data through the borrowed-memory CBs (c_0/c_1 backed by input/residual shards, c_7 backed by stats shard for post-all-gather).

| host site (file:line) | originating Tensor | RTA slot (host) | CTA offset (kernel) |
|---|---|---|---|
| `sharded_layernorm_factory_helpers.cpp:682,692` writer_sender/receiver CTA `TensorAccessorArgs(gamma_buffer)` | `tensor_args.weight` (`gamma`) | RTA[3] = `gamma_dram_addr` | CTA = TensorAccessorArgs slot after writer's positional CTAs |
| `sharded_layernorm_factory_helpers.cpp:683,693` writer_sender/receiver CTA `TensorAccessorArgs(beta_buffer)` | `tensor_args.bias` (`beta`) | RTA[4] = `beta_dram_addr` | CTA next slot after gamma_args |

### Work split

- Grid driven by `tensor_args.input.shard_spec()` (sharded input). The grid_size, mcast_1d, row_wise, etc. derive from the shard spec via `GridParams::compute`.
- Worker distribution (`WorkerDistribution::compute`) splits cores into:
  - `num_cores_all_to_all` — all-to-all workers (participate in reduction's gather phase)
  - `num_none_all_to_all_workers` — receive-only workers (the rest, if any)
- Two compute kernels are created at the legacy host: one for `all_to_all_cores`, one for `not_all_to_all_workers`. Same source, different CTAs (mostly `is_all_to_all_worker=1` vs `0`).

### Cross-op kernels

None — every kernel source lies under `device/kernels/`.

### Flags

The legacy factory uses helper structs heavily (GridParams, WorkerDistribution, CoreRanges, CBSizeParams, CompileTimeArgs, RuntimeArgsResult). The Metal 2.0 port reuses these structurally for grid/sizing/worker-distribution computation but replaces the spec-emission step (`add_kernel_descriptors`, `add_cb_descriptors`, `build_*_args`) with Metal 2.0 builders (`KernelSpec`, `DataflowBufferSpec`, `TensorParameter`, `WorkUnitSpec`, `ProgramRunParams::KernelRunParams`).

## Planned Spec Shape

- **KernelSpecs**: up to seven (reader_sender, optionally reader_receiver_all_to_all, optionally reader_receiver, writer_sender, optionally writer_receiver, compute_all_to_all, optionally compute_not_all_to_all). Same source paths as legacy.
- **DataflowBufferSpecs**: one per CB in the table above, each with `borrowed_from` set when the legacy CB had `.buffer != nullptr`.
- **SemaphoreSpecs**: three (DONE / RCV / SECOND_STAGE), all with `target_nodes = all_cores` and `initial_value = 0`.
- **TensorParameters**: up to six — INPUT_A (backs `cb_in`), RESIDUAL_B (backs `cb_inb`), GAMMA, BETA, OUTPUT (backs `cb_out` or `cb_out_resharded`), STATS (backs `cb_stats`), RECIP (backs `cb_reciprocals`).
- **WorkUnitSpecs**: up to four — one per (kernel set, core range) pairing. The compute_all_to_all + reader_sender (+ optionally reader_receiver_all_to_all) + writer_sender share the `all_to_all_cores` work unit; compute_not_all_to_all + reader_receiver + writer_receiver share the `not_all_to_all_workers` work unit. Sender_cores has its own work unit when distinct from all_to_all_cores.

## Preserved Multiplicity

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (multi-binding) |
|---|---|---|---|
| 2× compute (all_to_all + not_all_to_all) of `layernorm_sharded.cpp` etc. | `compute_all_to_all` + `compute_not_all_to_all` of same source | `wu_all_to_all` + `wu_not_all_to_all` | `cb_in`, `cb_out`, `cb_xmm`, `cb_x`, `cb_ex_global`, plus `cb_inb`/`cb_gamma`/`cb_beta`/`cb_fusion`/scaler set as conditionally bound |
| 2× reader (sender + receiver) of `reader_mcast_sender/receiver_unary_sharded_ln.cpp` | `reader_sender` + `reader_receiver_all_to_all` + optionally `reader_receiver` | `wu_sender` + `wu_all_to_all` + `wu_not_all_to_all` | same set as above |
| 2× writer (sender + receiver) of `writer_unary_sharded_ln.cpp` | `writer_sender` + `writer_receiver` | `wu_all_to_all` + `wu_not_all_to_all` | `cb_out`, gamma/beta DFBs |

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| writer_sender CT slot N: `TensorAccessorArgs(gamma_buffer)` (`sharded_layernorm_factory_helpers.cpp:682`) | TensorAccessorArgs plumbing | `TensorBinding(GAMMA, "gamma")` on writer KernelSpec + `TensorAccessor(ta::gamma)` in kernel |
| writer_receiver CT slot N: `TensorAccessorArgs(beta_buffer)` | same | `TensorBinding(BETA, "beta")` + `TensorAccessor(ta::beta)` |
| writer RTA[3,4] `gamma_dram_addr, beta_dram_addr` | buffer-address RTAs | gone (auto-injected by TensorBinding) |
| reader_sender CTA[0,1,16]: semaphore IDs | semaphore-ID CTAs (3) | `SemaphoreBinding(RCV/DONE/SECOND_STAGE, "rcv/done/second_stage")` + `Semaphore sem(sem::rcv)` etc. in kernel |
| `CBDescriptor::buffer = a.buffer() / b.buffer() / output.buffer() / output_reshard.buffer() / stats.buffer() / recip.buffer()` (7 sites in `sharded_layernorm_factory_helpers.cpp`) | borrowed-memory CB | `DataflowBufferSpec::borrowed_from = <TensorParameter>` (7 DFBs marked) |
| Named CTAs in `reader_cb_named_args`, `writer_cb_named_args`, `compute_cb_named_args` | named CTA → CB index | gone — replaced by `DFBBinding` on each kernel + `DataflowBuffer(dfb::<name>)` in kernel code |
| Positional CTAs (~19 on reader_sender, ~15 on compute) | positional CTAs | Named CTAs (matched 1:1 by position to legacy index list — see helper code in port) |

## Applied Patterns

- **Borrowed-memory DFB** for `cb_in` / `cb_inb` (and `c_14` aliased) / `cb_out` / `cb_out_resharded` / `cb_stats` / `cb_reciprocals` — see [Dynamic CircularBuffer (LANDED)](port_op_to_metal2_audit.md#dynamic-circularbuffer-cb-built-on-borrowed-buffer-memory--landed).
- **Multi-variant factory** branching inside `create_program_spec` on `is_pre_all_gather`/`is_post_all_gather`/`use_welford`/`use_row_major_kernel`.
- **Preserved multiplicity** for the work-split compute / reader / writer pairs.
- **Self-loop DFB binding** for compute kernel's self-loop CBs (`cb_xmm`, `cb_x`, `cb_ex_global`, etc. — same kernel both produces and consumes).
- **Pass DFB handles directly to LLKs and kernel-lib helpers** — kernel code passes `dfb::xxx` directly to `reduce_init`, `dataflow_kernel_lib::prepare_reduce_scaler`, etc.

## Deferred / Flagged

- None new — audit was GREEN with no yellow items.
- The sharded kernels for pre/post all-gather have specialized roles (multi-device norm). Their kernel code is largely independent of the main norm flow but uses the same DFB / TA binding machinery.
