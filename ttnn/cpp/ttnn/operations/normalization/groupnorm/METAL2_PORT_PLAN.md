# Port Plan — `ttnn/cpp/ttnn/operations/normalization/groupnorm`

Port plan for `groupnorm`, planned for a port from `ProgramDescriptor` to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

> **Status: the port stopped before construction.** Planning uncovered a hard stop that
> applies to every one of the op's three factories — see [Deferred / Flagged](#deferred--flagged)
> and `METAL2_PORT_REPORT.md`. No source file in the op was modified. This plan is therefore
> the groundwork a later port picks up from, not a record of work done.
>
> Depth of this document reflects that: the **sharded** factory (the first factory the port
> would have converted) is inventoried and planned in full; the **mcast** and **no-mcast**
> factories are inventoried at structural granularity — enough for the blocker statement to be
> complete and for a later porter to resume — but their per-DFB endpoint census and dropped-plumbing
> tables were not finished, because the stop makes them unactionable today. Where that is the case
> the section says so rather than implying coverage it does not have.

---

## Legacy Inventory

### Legacy factory shape

- **Concept:** `ProgramDescriptorFactoryConcept` — all three factories are
  `static ProgramDescriptor create_descriptor(...)`
  ([groupnorm_device_operation.hpp:24-46](device/groupnorm_device_operation.hpp#L24-L46)).
- **Variants:** three program factories under one device operation
  `ttnn::prim::GroupNormDeviceOperation`, selected at
  [groupnorm_device_operation.cpp:15-45](device/groupnorm_device_operation.cpp#L15-L45):
  - `GroupNormShardedProgramFactory` — input is sharded.
  - `GroupNormNoMcastProgramFactory` — interleaved input, `batch >= num_virtual_rows`.
  - `GroupNormMcastProgramFactory` — interleaved input, otherwise.
  Each factory additionally has a **welford / non-welford** kernel-set split driven by the
  `use_welford` operation attribute, plus layout sub-configurations (`TILIZE_IN`,
  `UNTILIZE_OUT`, `READER_REPACK`) and optional-tensor gates (`FUSE_GAMMA`, `FUSE_BETA`,
  `FUSE_NEGATIVE_MASK`).
- **Custom `compute_program_hash`:** none — the op uses the default reflection-based hash. A grep
  over the op directory for `compute_program_hash`, `get_dynamic_runtime_args`,
  `override_runtime_arguments` and `WorkloadDescriptor` returns zero hits.

*(The Metal 2.0 factory concept the port targets was chosen during the audit — `ProgramSpecFactoryConcept`.
Carried forward in the [TTNN ProgramFactory](#ttnn-programfactory) section below.)*

---

### Variant: `GroupNormShardedProgramFactory`

Source: [groupnorm_sharded_program_factory.cpp](device/groupnorm_sharded_program_factory.cpp).

Placement vocabulary used below:
- `all_cores` — the input tensor's shard grid (a single merged rectangular `CoreRangeSet`,
  enforced at [:115](device/groupnorm_sharded_program_factory.cpp#L115)).
- `mcast_sender_cores` — one core per (batch × group) mcast group.
- `mcast_receiver_cores` — `all_cores` minus `mcast_sender_cores`; **empty when `use_mcast == false`**.
- `use_mcast = num_cores_per_batch > 1 || num_cores_per_group > 1`
  ([:359](device/groupnorm_sharded_program_factory.cpp#L359)).

#### Kernels

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| reader_mcast_sender | `kernels/dataflow/reader_mcast_sender_unary_sharded_gn_v2.cpp`, or `welford_reader_mcast_sender_unary_sharded_gn_v2.cpp` when `use_welford` | `mcast_sender_cores` | 10 slots ([:494-504](device/groupnorm_sharded_program_factory.cpp#L494-L504)): `reduce_receiver_semaphore_id`, `reduce_sender_semaphore_id`, `num_mcast_cores`, `num_batch_group`, `per_core_N`, `per_core_N_bytes`, `per_core_N_bytes_with_stride`, `datum_size_bytes`, `per_core_M`, `tile_height`; **+3 when `use_welford`**: `block_hw`, `num_groups`, `tile_width` | none | Variable-length, per sender core ([:1136-1215](device/groupnorm_sharded_program_factory.cpp#L1136-L1215)). Fixed head: `has_mcast_first_group`, `has_mcast_last_group`, 4 mid-group NoC coords, `num_mcast_cores_mid_group`; **optionally** 5 first-group fields and 5 last-group fields; then a `2 × num_mcast_cores` block of per-core NoC x then y | none | `FUSE_GAMMA` (if gamma), `FUSE_BETA` (if beta), `READER_REPACK` (if `reader_repack_output`), `TILIZE_IN` (if input row-major), `UNTILIZE_OUT` (if output row-major) | absent → **O2** | `DataMovementConfigDescriptor{RISCV_0, preferred_noc_for_dram_read = NOC_0}` |
| reader_mcast_receiver | `kernels/dataflow/reader_mcast_receiver_unary_sharded_gn_v2.cpp`, or the `welford_` variant | `mcast_receiver_cores`; **descriptor created only when `use_mcast`** ([:546-562](device/groupnorm_sharded_program_factory.cpp#L546-L562)) | 8 slots ([:510-518](device/groupnorm_sharded_program_factory.cpp#L510-L518)): as the sender minus `num_mcast_cores` and `datum_size_bytes`; **+3 when `use_welford`** | none | 2 per receiver core: sender NoC x, y ([:1219-1222](device/groupnorm_sharded_program_factory.cpp#L1219-L1222)) | none | same set as the sender reader | absent → **O2** | `DataMovementConfigDescriptor{RISCV_0, NOC_0}` |
| writer | `kernels/dataflow/writer_unary_sharded_gn_rm_gb_v2.cpp`, or `welford_writer_unary_sharded_gn_rm_gb_v2.cpp` | `all_cores` | 13 slots ([:571-594](device/groupnorm_sharded_program_factory.cpp#L571-L594)): `is_mcast_sender` (**hardcoded `1` on every core**), `fuse_gamma`, `fuse_beta`, `num_cols_tile_gamma_beta`, `per_core_N`, `per_core_N_bytes`, `per_core_N_bytes_with_stride`, `num_groups_per_core`, `num_batches_per_core`, `block_w`, `page_size` (**slot 10, read by neither writer kernel**), `reduce_factor_w`, `reduce_factor_c`; then four `TensorAccessorArgs` blocks appended for gamma / beta / input_mask / negative_mask ([:597-609](device/groupnorm_sharded_program_factory.cpp#L597-L609)) | none | 8 per core ([:1232-1257](device/groupnorm_sharded_program_factory.cpp#L1232-L1257)): `eps`, gamma `Buffer*` or `0u`, beta `Buffer*` or `0u`, input_mask `Buffer*` or `0u`, negative_mask `Buffer*` or `0u`, `gamma_tile_start_id`, `beta_tile_start_id`, `input_mask_tile_start_id` | none | `TILE_HW_VAL=<tile_hw>`, `FUSE_NEGATIVE_MASK` (if negative_mask) | absent → **O2** | `DataMovementConfigDescriptor{RISCV_1, preferred_noc_for_dram_write = NOC_1}` |
| compute_sender | `kernels/compute/groupnorm_sharded_v2.cpp`, or `welford_groupnorm_sharded_v2.cpp` | `mcast_sender_cores` | 25 slots ([:644-677](device/groupnorm_sharded_program_factory.cpp#L644-L677)) with slot 0 `is_mcast_sender = 1`; **+1 before the last when `use_welford`** (`num_cols_per_group`) | `welford_fp32_alias`, `cb_in0_welford`, `cb_in_welford` — **only when `use_welford`** ([:767-774](device/groupnorm_sharded_program_factory.cpp#L767-L774)) | none | none | `READER_REPACK`, `TILIZE_IN`, `UNTILIZE_OUT`, `FUSE_NEGATIVE_MASK`, `FP32_DEST_ACC=true|false` | absent → **O3** | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en, dst_full_sync_en, unpack_to_dest_mode, math_approx_mode}` |
| compute_receiver | same source as compute_sender | `mcast_receiver_cores`; **descriptor is pushed unconditionally** ([:1280](device/groupnorm_sharded_program_factory.cpp#L1280)) even when that set is empty | same 25 (+1) slots with slot 0 `is_mcast_sender = 0` | same as compute_sender | none | none | same as compute_sender | absent → **O3** | same as compute_sender |

Notes carried into planning:

- **`opt_level` is absent on every `KernelDescriptor` in the whole op** (`grep -rn opt_level` over the
  op directory returns nothing), so the resolved levels are the per-kernel-type legacy defaults:
  **O2** for the three DM kernels, **O3** for the two compute kernels. Metal 2.0's single
  `CompilerOptions::opt_level` defaults to `O2`, so both compute `KernelSpec`s must set `O3` explicitly.
- **Neither DM config matches a Metal 2.0 role default.** Resolved triples are
  reader = `(RISCV_0, NOC_0, DM_DEDICATED_NOC)` and writer = `(RISCV_1, NOC_1, DM_DEDICATED_NOC)`.
  The Metal 2.0 reader default is `(RISCV_1, NOC_0)` and the writer default is `(RISCV_0, NOC_1)` — so
  the op's RISC assignment is *swapped* relative to both helpers. `create_reader_datamovement_config` /
  `create_writer_datamovement_config` would silently move each kernel to the other RISC. Both kernels
  are therefore the **custom** case and must be written as explicit `DataMovementGen1Config`s.
  This holds for all three factories.
- `compute_receiver_desc` being pushed with an empty `core_ranges` in the non-mcast configuration is
  legal under the descriptor API but has no Metal 2.0 equivalent: a `WorkUnitSpec` needs a non-empty
  `target_nodes`, and a `KernelSpec` in `ProgramSpec::kernels` must be referenced by some
  `WorkUnitSpec`. The port must gate the receiver compute `KernelSpec` on `use_mcast`, exactly as the
  legacy code already gates the receiver *reader*.

#### CBs

Sizes reference the locals computed at
[:317-345](device/groupnorm_sharded_program_factory.cpp#L317-L345). `all_cores` throughout.
`in0_cb_page_size = reader_repack_output ? a.buffer()->page_size() : in_single_tile_size`.

| index | total_size | core_ranges | data_format | page_size | tile (if set) | notes |
|---|---|---|---|---|---|---|
| `c_0` in0 | `in0_CB_size` = `a.buffer()->aligned_size_per_bank()` | all_cores | `in_data_format` | `in0_cb_page_size` | not set | **borrowed** — `.buffer = a.buffer()` ([:837](device/groupnorm_sharded_program_factory.cpp#L837), `:856`) |
| `c_16` out | shares `c_0`'s descriptor **when `inplace`** ([:823-846](device/groupnorm_sharded_program_factory.cpp#L823-L846)); otherwise its own descriptor of `reader_repack_output ? output.buffer()->aligned_size_per_bank() : out_CB_size` ([:867-878](device/groupnorm_sharded_program_factory.cpp#L867-L878)) | all_cores | `in_data_format` (inplace) / `out_data_format` | `in0_cb_page_size` / `reader_repack_output ? output.buffer()->page_size() : out_single_tile_size` | not set | **borrowed** — `.buffer = a.buffer()` (inplace) or `output.buffer()` |
| `c_29` | third format descriptor on `c_0`'s allocation, **only when `welford_fp32_alias`** ([:839-845](device/groupnorm_sharded_program_factory.cpp#L839-L845)) | all_cores | `in_data_format` | `in0_cb_page_size` | not set | aliased onto `c_0` |
| `c_1` in (tilized) | `in_CB_size` | all_cores | `in_data_format` | `in_single_tile_size` | not set | |
| `c_31` | second format descriptor on `c_1`'s allocation, **only when `welford_fp32_alias`** ([:892-898](device/groupnorm_sharded_program_factory.cpp#L892-L898)) | all_cores | `in_data_format` | `in_single_tile_size` | not set | aliased onto `c_1` |
| `c_30` out | `in_CB_size`; **only when `!negative_mask && untilize_out`** ([:905-914](device/groupnorm_sharded_program_factory.cpp#L905-L914)) | all_cores | `in_data_format` | `in_single_tile_size` | not set | |
| `c_2` scaler | `in2_CB_size` = `single_tile_size * (use_welford ? 3 : 1)` | all_cores | `cb_data_format` | `single_tile_size` | not set | in the welford compute kernel this index is `xmm`, not a scaler |
| `c_3` eps | `single_tile_size` | all_cores | `cb_data_format` | `single_tile_size` | not set | |
| `c_4` scaler-global | `in2_CB_size`; **only when `!use_welford`** ([:944-955](device/groupnorm_sharded_program_factory.cpp#L944-L955)) | all_cores | `cb_data_format` | `single_tile_size` | not set | |
| `c_5` gamma | `in5_CB_size`; **only when gamma** | all_cores | `gamma_beta_cb_data_format` | `gamma_beta_single_tile_size` | not set | |
| `c_6` beta | `in6_CB_size`; **only when beta** | all_cores | `gamma_beta_cb_data_format` | `gamma_beta_single_tile_size` | not set | |
| `c_7` input mask | `in_mask_CB_size`; **only when input_mask** | all_cores | `in_mask_cb_data_format` | `in_mask_single_tile_size` | not set | |
| `c_14` negative mask | `in_negative_mask_CB_size`; **only when negative_mask** | all_cores | `in_negative_mask_cb_data_format` | `in_negative_mask_single_tile_size` | not set | |
| `c_11` repack + `c_12` repack_out | one descriptor of `repack_CB_size`, two format descriptors; **only when `reader_repack_output`** ([:1008-1026](device/groupnorm_sharded_program_factory.cpp#L1008-L1026)) | all_cores | `in_data_format` | `in_single_tile_size` | not set | **aliased pair** |
| `c_13` x | `x_CB_size` | all_cores | `cb_data_format` | `single_tile_size` | not set | |
| `c_8` ex_partial | `ex_partial_CB_size` | all_cores | `cb_data_format` | `single_tile_size` | not set | |
| `c_10` ex_external | `single_tile_size`; **only when `!use_welford`** ([:1052-1063](device/groupnorm_sharded_program_factory.cpp#L1052-L1063)) | all_cores | `cb_data_format` | `single_tile_size` | not set | per-core slot pitch is the shared constant `dfb_ex_external_slot_pitch_bytes` |
| `c_15` ex_global + `c_9` ex | one descriptor of `ex_global_CB_size`, two format descriptors ([:1068-1082](device/groupnorm_sharded_program_factory.cpp#L1068-L1082)) | all_cores | `cb_data_format` | `single_tile_size` | not set | **aliased pair** |
| `c_17` ex2pe | `ex2pe_CB_size` | all_cores | `cb_data_format` | `single_tile_size` | not set | |
| `c_26` ones | `single_tile_size` | all_cores | `cb_data_format` | `single_tile_size` | not set | |

No `CBDescriptor` in this factory sets `tile`, `address_offset`, or `global_circular_buffer`.

#### Semaphores

| id | core_type | core_ranges | initial_value |
|---|---|---|---|
| 0 `reduce_sender` | `tt::CoreType::WORKER` | `all_cores` | 0 |
| 1 `reduce_receiver` | `tt::CoreType::WORKER` | `all_cores` | 0 |

Both ids reach the kernels as **compile-time** args (reader CTA slots 0 and 1), not RTAs.

#### Tensor accessors

All four are on the writer kernel; the readers and compute kernels construct none.

| host site (file:line) | originating Tensor | RTA slot (host) |
|---|---|---|
| [:597](device/groupnorm_sharded_program_factory.cpp#L597) (`TensorAccessorArgs` → CTA) | `gamma` (optional) | writer RTA 1 (`Buffer*`, or literal `0u` when absent) |
| [:599](device/groupnorm_sharded_program_factory.cpp#L599) | `beta` (optional) | writer RTA 2 |
| [:601](device/groupnorm_sharded_program_factory.cpp#L601) | `input_mask` (optional) | writer RTA 3 |
| [:605-609](device/groupnorm_sharded_program_factory.cpp#L605-L609) | `negative_mask` (optional) | writer RTA 4 |

Kernel-side construction: `TensorAccessorArgs<13>()` then the `next_compile_time_args_offset()` chain
([writer_unary_sharded_gn_rm_gb_v2.cpp:47-49](device/kernels/dataflow/writer_unary_sharded_gn_rm_gb_v2.cpp#L47-L49),
`:83-84`). Every construction is the **two-argument** form — no page-size third argument anywhere in
the op.

The `input` and `output` tensors do **not** travel as accessors in this factory; they are borrowed-memory
CBs (`c_0` / `c_16`).

#### Work split

n/a — no `split_work_to_cores`. Placement is the input's shard grid; the sender/receiver division is by
mcast role over **disjoint** core ranges, computed at
[:395-452](device/groupnorm_sharded_program_factory.cpp#L395-L452).

#### Shared kernels

None. All eight kernel sources this factory can select live under
`device/kernels/` and are bound by **no other factory** — the mcast and no-mcast factories use the
non-`_v2` kernel set. There is no `_metal2` fork to reuse or create for this factory.

#### Flags

- The `page_size` value at writer CTA slot 10 is computed on the host
  ([:583-591](device/groupnorm_sharded_program_factory.cpp#L583-L591)) and read by neither sharded
  writer kernel (`// compile_time_arg 10: size (unused here)`).
- `packer_l1_acc` is destructured from the compute-kernel config at
  [:713](device/groupnorm_sharded_program_factory.cpp#L713) and never used.
- `bool block_wt_last = (per_core_Nt + num_groups_per_core - 1) / num_groups_per_core;`
  ([:226](device/groupnorm_sharded_program_factory.cpp#L226)) collapses a tile count to a `bool`.
  Whatever it means, the port carries the value through unchanged.
- Four `get_dataformat(...)` locals across the reader kernels are never read.

All four are pre-existing and out of the port's scope; they are recorded in `METAL2_PORT_REPORT.md`.

---

### Variant: `GroupNormMcastProgramFactory`

Source: [groupnorm_mcast_program_factory.cpp](device/groupnorm_mcast_program_factory.cpp).
**Structural inventory only** — see the status note at the top.

#### Kernels

| unique_id | source | core_ranges | CTAs | RTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|
| reader_mcast_sender | `dataflow/reader_mcast_sender_unary_gn.cpp` / `welford_reader_mcast_sender_unary_gn.cpp` | `mcast_sender_cores_group_1` | **named** via `to_named_args_mcast` ([:407-441](device/groupnorm_mcast_program_factory.cpp#L407-L441)) plus positional `TensorAccessorArgs` for `input` and `output` | variable-length mcast-coordinate block, same shape as the sharded sender | as sharded | absent → O2 | `{RISCV_0, NOC_0}` |
| reader_mcast_receiver | `dataflow/reader_mcast_receiver_unary_gn.cpp` / `welford_` | `mcast_receiver_cores_group_1` | named + `TensorAccessorArgs` for `input`, `output` ([:443-475](device/groupnorm_mcast_program_factory.cpp#L443-L475)) | sender NoC x, y | as sharded | absent → O2 | `{RISCV_0, NOC_0}` |
| writer | `dataflow/writer_unary_gn_rm_gb.cpp` / `welford_writer_unary_gn_rm_gb.cpp` | `all_cores_group_1` | named ([:518-556](device/groupnorm_mcast_program_factory.cpp#L518-L556)) + `TensorAccessorArgs` for `output`, `gamma`, `beta`, `input_mask` | `eps`, four `Buffer*`-or-`0u` slots, tile start ids | `TILE_HW_VAL` | absent → O2 | `{RISCV_1, NOC_1}` |
| compute_sender | `compute/groupnorm.cpp` / `welford_groupnorm.cpp` | `mcast_sender_cores_group_1` | named ([:594-627](device/groupnorm_mcast_program_factory.cpp#L594-L627)) + welford alias args | none | `READER_REPACK`, `TILIZE_IN`, `UNTILIZE_OUT`, `FP32_DEST_ACC` | absent → **O3** | `ComputeConfigDescriptor{...}` |
| compute_receiver | same source | `mcast_receiver_cores_group_1` | named, `is_mcast_sender = 0` | none | same | absent → **O3** | same |

#### CBs

`c_0` in0 (**borrowed**, `.buffer = a.buffer()`), `c_16` out0 (**borrowed**), `c_19` (welford fp32 alias
of `c_0`), `c_29` in (tilized), `c_30` out (untilize only), `c_2` scaler, `c_3` eps, `c_4` scaler-global,
`c_5` gamma, `c_6` beta, `c_28` input mask, `c_26` repack + `c_31` repack_out (aliased pair, repack only),
`c_24` x, `c_25` xmm, `c_23` xmm2, `c_22` xmm3, `c_8` ex_partial, `c_21` ex2_partial (non-welford),
`c_10` ex_external (non-welford), `c_15` ex_global + `c_9` ex (aliased pair), `c_14` ex2_global +
`c_13` ex2 (aliased pair, non-welford), `c_27` ex2pe, `c_18` reciprocals (**borrowed**,
`.buffer = reciprocals.value().buffer()`, welford + reciprocals only).

#### Semaphores

Two, both `WORKER` on `all_cores`, `initial_value = 0`
([:372-381](device/groupnorm_mcast_program_factory.cpp#L372-L381)).

#### Tensor accessors

Six distinct originating tensors: `input` (reader), `output` (reader **and** writer — the reader re-reads
the output on the third pass), `gamma`, `beta`, `input_mask` (writer), and `reciprocals` (borrowed CB,
read via `get_pointer_to_cb_data`, not an accessor). All constructions are the two-argument form.

#### Work split

n/a — `all_cores_group_1` is the full grid; sender/receiver is the mcast-role split over disjoint ranges.

#### Shared kernels

**Intra-op.** Six of the eight sources this factory can select are also bound by
`GroupNormNoMcastProgramFactory` — both sender readers, both writers, and both compute kernels. Only
the two receiver readers (`reader_mcast_receiver_unary_gn.cpp` and its `welford_` sibling) are
mcast-only. Converting either factory in place breaks the other on those six. No `_metal2` fork exists
beside any of them today. The two factories must therefore be co-ported as one unit, or the six shared
sources forked — this is why the scope for a resumed port is "mcast + no-mcast together".

---

### Variant: `GroupNormNoMcastProgramFactory`

Source: [groupnorm_no_mcast_program_factory.cpp](device/groupnorm_no_mcast_program_factory.cpp).
**Structural inventory only.**

Same kernel set as the mcast factory minus the receiver reader: every core is its own mcast group of
size 1, so only the *sender* reader runs.

#### Kernels

| unique_id | source | core_ranges | opt_level | config |
|---|---|---|---|---|
| reader_g1 | `dataflow/reader_mcast_sender_unary_gn.cpp` / `welford_` | `mcast_sender_cores_group_1` | absent → O2 | `{RISCV_0, NOC_0}` |
| reader_g2 | same source | `mcast_sender_cores_group_2` | absent → O2 | `{RISCV_0, NOC_0}` |
| writer_g1 | `dataflow/writer_unary_gn_rm_gb.cpp` / `welford_` | `all_cores_group_1` | absent → O2 | `{RISCV_1, NOC_1}` |
| writer_g2 | same source; **created only when `all_cores_group_2` is non-empty** ([:751-765](device/groupnorm_no_mcast_program_factory.cpp#L751-L765)) | `all_cores_group_2` | absent → O2 | `{RISCV_1, NOC_1}` |
| compute_g1 | `compute/groupnorm.cpp` / `welford_groupnorm.cpp` | `mcast_sender_cores_group_1` | absent → **O3** | `ComputeConfigDescriptor{...}` |
| compute_g2 | same source | `mcast_sender_cores_group_2` | absent → **O3** | same |

All CTAs are named via `to_named_args_no_mcast`, plus positional `TensorAccessorArgs` blocks.

#### CBs

Same index vocabulary as the mcast factory. `c_9` `ex` and `c_13` `ex2` are **allocated but have zero
touchers in every configuration** of this factory (the guards that would use them are
`num_mcast_cores > 1`, always false here) — they are the second format descriptor of the `c_15` / `c_14`
allocations, which stay live.

#### Semaphores

One ([:505](device/groupnorm_no_mcast_program_factory.cpp#L505)), `WORKER`, `initial_value = 0`.

#### Work split

**Two core groups.** When batches do not divide evenly across the grid, the factory splits into
`group_1` (`num_batches_per_core_group_1` batches per core) and `group_2`
(`num_batches_per_core_group_1 - 1`) over disjoint core sets
([:218-236](device/groupnorm_no_mcast_program_factory.cpp#L218-L236)), and instantiates the reader,
writer, and compute kernels **twice each** with per-group CTA values. This is the classic disjoint-node
work split — it maps to two `KernelSpec`s per source in two `WorkUnitSpec`s, never to a CTA→RTA demotion.

#### Shared kernels

**Intra-op** — the same eight sources as the mcast factory. See that section.

---

## TTNN ProgramFactory

- **Concept (inherited from audit):** `ProgramSpecFactoryConcept`, for all three factories.
- **Custom `compute_program_hash`:** none — already the default reflection-based hash. Nothing to delete.
- **Implementation notes:**
  - Each factory's `create_descriptor` becomes
    `static ttnn::device_operation::ProgramArtifacts create_program_artifacts(const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&)`
    in [groupnorm_device_operation.hpp](device/groupnorm_device_operation.hpp). The three factories may
    sit on different concepts while the port lands one at a time; the framework dispatches per factory.
  - No op-owned tensors: the op allocates no device tensor beyond its declared io.
  - `groupnorm_nanobind.cpp` binds only the user-facing `ttnn::group_norm` and the two program-config
    structs — no `create_descriptor` exposure, so no pybind deletion is forced.
  - The three factories all define anonymous-namespace-scope constants under a unity build; the
    per-factory name-prefix discipline from
    [Unity-build hygiene for anonymous-namespace symbols](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-unity-build-hygiene-for-anonymous-namespace-symbols)
    applies from the first factory onward.

---

## Planned Spec Shape

### Variant: `GroupNormShardedProgramFactory`

- **KernelSpecs** — five at most, matching the legacy descriptors 1:1: `READER_SENDER`,
  `READER_RECEIVER` (only when `use_mcast`), `WRITER`, `COMPUTE_SENDER`,
  `COMPUTE_RECEIVER` (only when `use_mcast` — a change from legacy, which pushes the descriptor with an
  empty core range; see the Kernels notes above). The welford / non-welford choice picks the `source`
  path; it does not change the spec's shape.
  - `compiler_options.opt_level = O3` on both compute specs; DM specs leave the `O2` default.
  - `hw_config`: explicit `DataMovementGen1Config{RISCV_0, NOC_0, DM_DEDICATED_NOC}` on all three DM
    specs' readers and `{RISCV_1, NOC_1, DM_DEDICATED_NOC}` on the writer — **not** the role helpers.
  - `hw_config` compute: `to_compute_hardware_config(device->arch(), compute_kernel_config)`, then
    `std::get<ComputeGen1Config>(...).unpack_modes` set from the legacy `unpack_to_dest_mode` vector
    (see below). `bfp_pack_precision_mode` stays default (legacy sets no `bfp8_pack_precise`).
- **DataflowBufferSpecs** — one per legacy `buffer_index`, so up to 21, with three alias groups:
  - `{IN0, OUT}` when `inplace`, plus `IN0_WELFORD` (`c_29`) when `welford_fp32_alias` — all mutually
    listed in `advanced_options.alias_with`.
  - `{IN, IN_WELFORD}` (`c_1` / `c_31`) when `welford_fp32_alias`.
  - `{REPACK, REPACK_OUT}` (`c_11` / `c_12`) when `reader_repack_output`.
  - `{EX_GLOBAL, EX}` (`c_15` / `c_9`).
  - Borrowed: `IN0.borrowed_from = INPUT`; `OUT.borrowed_from = INPUT` when `inplace`, else `OUTPUT`.
- **SemaphoreSpecs** — two, `target_nodes = all_cores`. Both are `initial_value = 0`, so the deprecated
  `SemaphoreAdvancedOptions::initial_value` is not touched. The two reader CTA slots carrying the
  semaphore ids drop; the kernels bind `sem::reduce_sender` / `sem::reduce_receiver`.
- **TensorParameters** — up to six: `INPUT` and `OUTPUT` (backing the borrowed DFBs — declared even
  though no kernel builds an accessor over them, because `borrowed_from` names a `TensorParameter`),
  and `GAMMA` / `BETA` / `INPUT_MASK` / `NEGATIVE_MASK`, each declared only when its optional tensor is
  present.
  - ⚠ **Open question the construction step must answer before it can be written:** the validator
    requires every `TensorParameter` to have ≥1 `TensorBinding` across the program's kernels. `INPUT`
    and `OUTPUT` here are referenced only by `DataflowBufferSpec::borrowed_from`, and no kernel binds
    them as accessors. Whether `borrowed_from` counts as a binding for that check has to be settled
    against the validator; if it does not, the port needs a mechanism the plan does not currently have.
    Flagged in the report.
- **WorkUnitSpecs** — two: `wu_sender` = `{READER_SENDER, WRITER, COMPUTE_SENDER}` on
  `mcast_sender_cores`, and `wu_receiver` = `{READER_RECEIVER, WRITER, COMPUTE_RECEIVER}` on
  `mcast_receiver_cores` (omitted entirely when `use_mcast` is false). `WRITER` belongs to both, which is
  the documented multi-work-unit case; its effective node set is the union, i.e. `all_cores`.
- **Op-owned tensors** — none.

#### DFB endpoint dispositions (re-derived from the kernel-touch census)

Census run per configuration over the three kernels resident on a node. Helper-mediated FIFO ops
(`compute_kernel_lib::reduce` / `tilize` / `untilize`,
`dataflow_kernel_lib::calculate_and_prepare_reduce_scaler`) are counted as touches of the DFBs they
carry as template parameters.

| DFB | census | disposition |
|---|---|---|
| `c_0` in0 | compute (tilize source / `binary_op_init_common`); reader raw-reads it too under `READER_REPACK && TILIZE_IN` | **1P+1C** when both touch, else **self-loop** on compute |
| `c_29`, `c_31` | compute only | **self-loop** |
| `c_1` in | compute produces (tilize dest) and consumes | **self-loop** |
| `c_2` scaler | writer produces via `calculate_and_prepare_reduce_scaler<c_2>`, compute consumes | **1P+1C** |
| `c_3` eps | writer produces via `generate_bcast_col_scalar`, compute consumes | **1P+1C** |
| `c_4` scaler-global | writer always produces (its `is_mcast_sender` CTA is hardcoded `1`); compute consumes only when `num_cores_per_mcast_group > 1` | mcast **1P+1C**; single-core group **self-loop** on the writer |
| `c_5` gamma, `c_6` beta, `c_7` input mask, `c_14` negative mask, `c_26` ones | writer produces, compute consumes | **1P+1C** each |
| `c_11` / `c_12` | reader produces `c_11` and consumes `c_12`; compute consumes `c_11` and produces `c_12` | **1P+1C** each |
| `c_13` x, `c_17` ex2pe, `c_30` out | compute only | **self-loop** |
| `c_16` out | exactly one toucher in every configuration — the reader's raw write under `READER_REPACK && UNTILIZE_OUT`, otherwise compute's untilize / pack target | **self-loop** |
| `c_8` ex_partial | compute produces (`reduce<…, REDUCE_SCALAR>`); reader consumes when mcasting; when `num_cores_per_mcast_group == 1` compute also consumes, because `dfb_ex_global_id` aliases to `c_8` ([groupnorm_sharded_v2.cpp:83](device/kernels/compute/groupnorm_sharded_v2.cpp#L83)) | mcast **1P+1C**; single-core group **self-loop** |
| `c_9` ex | sender node while mcasting: compute produces, reader consumes. Receiver node: untouched. Single-core group: untouched | mcast sender **1P+1C**; **not bound** on receiver specs; **dropped** when `use_mcast == false` |
| `c_15` ex_global | sender node: compute produces and consumes. Receiver node: reader produces, compute consumes | sender **self-loop**; receiver **1P+1C**; **dropped** when `use_mcast == false` |
| `c_10` ex_external | reader produces, compute consumes, both under a `num_mcast_cores > 1` guard | mcast **1P+1C**; **dropped** when `use_mcast == false` |

**No DFB in this factory needs `allow_instance_multi_binding`.** The census never exceeds two distinct
touchers, and never puts two kernels on the same locked FIFO role. This agrees with the brief.

**Dead-CB drops, config-scoped.** When `use_mcast == false` (`num_cores_per_batch == 1 &&
num_cores_per_group == 1`), `c_9`, `c_15`, and `c_10` have zero touchers and no
`DataflowBufferSpec` is built for them. `c_9` and `c_15` share one legacy `CBDescriptor`, so dropping
them drops the whole allocation; `c_10` has its own. In every other configuration all three are live.
On receiver nodes `c_9` is unbound but still allocated (the sender nodes need it), which is expressible
because bindings are per `KernelSpec`.

#### `unpack_modes`

Legacy builds a `NUM_CIRCULAR_BUFFERS`-long `unpack_to_dest_mode` vector, all `Default`, and sets
`UnpackToDestFp32` on `c_29` and `c_31` only when `welford_fp32_alias`
([:752-759](device/groupnorm_sharded_program_factory.cpp#L752-L759)). Translation:

- `welford_fp32_alias` ⇒ `unpack_modes = {{IN0_WELFORD, UnpackToDest}, {IN_WELFORD, UnpackToDest}}`.
- Every other DFB's `Default` becomes `UnpackToSrc`, expressed by **omitting** the entry.
- `welford_fp32_alias` implies `fp32_dest_acc_en` (`enable_32_bit_dest`) **and** `Float32` input, so the
  validator's required-entry rule fires for every Float32 DFB the compute kernel consumes, not just the
  two aliases. The construction step must enumerate those consumed Float32 DFBs and add an explicit
  `UnpackToSrc` entry for each — deriving the value from the legacy vector, which says `Default`.
  This is a *newly required* entry with no legacy counterpart to copy from, and is the one place in the
  sharded factory where Metal 2.0 is stricter than the descriptor API.

### Variants `GroupNormMcastProgramFactory` / `GroupNormNoMcastProgramFactory`

Not planned to DFB granularity — see the status note. What is settled:

- Same `ProgramSpecFactoryConcept` target, same `O3`-on-compute and custom-DM-config rules.
- The no-mcast factory's two core groups become two `KernelSpec`s per source in two `WorkUnitSpec`s
  (see [Preserved Multiplicity](#preserved-multiplicity)).
- `c_9` / `c_13` are dead in **every** no-mcast configuration and get no spec there.
- The `c_16` dead-output-CB case under `!UNTILIZE_OUT && !gamma && !beta` is an **open question to the
  ops team** (audit *Questions for the user* #1) and must be answered before that build drops the
  allocation.
- `reciprocals` (`c_18`) is a borrowed DFB read by base pointer with no FIFO ops — a one-toucher, so
  **self-loop**.

---

## Preserved Multiplicity

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| **Sharded:** `compute_sender_desc` + `compute_receiver_desc`, both of `groupnorm_sharded_v2.cpp` (or the welford source), over the **disjoint** `mcast_sender_cores` / `mcast_receiver_cores` | `COMPUTE_SENDER`, `COMPUTE_RECEIVER` | `wu_sender`, `wu_receiver` | `c_8` (PRODUCER on both, plus CONSUMER on the sender spec in the single-core-group config); `c_15` (PRODUCER+CONSUMER on the sender spec, CONSUMER only on the receiver spec); `c_9` (PRODUCER on the sender spec, **unbound** on the receiver spec); all the writer-fed input DFBs (CONSUMER on both) |
| **Sharded:** `reader_mcast_sender_desc` + `reader_mcast_receiver_desc` — **different sources**, so not a same-source multiplicity; listed because they cover the same disjoint node partition | `READER_SENDER`, `READER_RECEIVER` | `wu_sender`, `wu_receiver` | `c_8` CONSUMER (sender only); `c_9` CONSUMER (sender only); `c_10` PRODUCER (sender only); `c_15` PRODUCER (receiver only) |
| **No-mcast:** `reader_g1`+`reader_g2`, `writer_g1`+`writer_g2`, `compute_g1`+`compute_g2`, each pair one source over disjoint `group_1` / `group_2` core sets, differing only in per-group CTA values | two `KernelSpec`s per source | `wu_group_1`, `wu_group_2` | not enumerated — see the status note |

Every pair above covers **disjoint** node sets, so each node sees exactly one instance and each binding
is a legal single-role binding. None of these is the same-grid dual-instance shape, and none needs
`allow_instance_multi_binding`. No per-group CTA is demoted to an RTA anywhere.

---

## Dropped Plumbing

Sharded factory. (The mcast / no-mcast tables were not built — status note.)

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| reader sender CTA slot 0 ([:495](device/groupnorm_sharded_program_factory.cpp#L495)) | `reduce_receiver_semaphore_id` | `SemaphoreBinding` → `sem::reduce_receiver` |
| reader sender CTA slot 1 ([:496](device/groupnorm_sharded_program_factory.cpp#L496)) | `reduce_sender_semaphore_id` | `SemaphoreBinding` → `sem::reduce_sender` |
| reader receiver CTA slots 0, 1 ([:511-512](device/groupnorm_sharded_program_factory.cpp#L511-L512)) | same two semaphore ids | same two `SemaphoreBinding`s |
| reader sender CTA slots 2-9 (+3 welford) | positional `uint32_t`s | named `compile_time_args` — `num_mcast_cores`, `num_batch_group`, `per_core_N`, `per_core_N_bytes`, `per_core_N_bytes_with_stride`, `datum_size_bytes`, `per_core_M`, `tile_height`, `block_hw`, `num_groups`, `tile_width` |
| reader receiver CTA slots 2-7 (+3 welford) | positional | named, same names minus `num_mcast_cores` / `datum_size_bytes` |
| writer CTA slots 0-9, 11, 12 | positional | named — `is_mcast_sender`, `fuse_gamma`, `fuse_beta`, `num_cols_tile_gamma_beta`, `per_core_N`, `per_core_N_bytes`, `per_core_N_bytes_with_stride`, `num_groups_per_core`, `num_batches_per_core`, `block_w`, `reduce_factor_w`, `reduce_factor_c` |
| writer CTA slot 10 ([:583-591](device/groupnorm_sharded_program_factory.cpp#L583-L591)) | host-computed `page_size`, read by no kernel | **dropped** — no named arg emitted, and the host computation goes with it |
| writer CTA tail ([:597-609](device/groupnorm_sharded_program_factory.cpp#L597-L609)) | four `TensorAccessorArgs(...).append_to(...)` blocks | `TensorBinding`s for `GAMMA` / `BETA` / `INPUT_MASK` / `NEGATIVE_MASK`; kernel-side `TensorAccessorArgs<13>()` and its `next_compile_time_args_offset()` chain both drop |
| writer RTA slots 1-4 ([:1234-1253](device/groupnorm_sharded_program_factory.cpp#L1234-L1253)) | `gamma.value().buffer()` / `0u`, and the same for beta, input_mask, negative_mask | `TensorBinding` per tensor, declared **only when the tensor is present**; the `0u` placeholder disappears rather than becoming a binding carrying zero |
| writer RTA slots 0, 5, 6, 7 | `eps`, `gamma_tile_start_id`, `beta_tile_start_id`, `input_mask_tile_start_id` | named RTAs (kept — genuinely per-node values) |
| compute CTA slots 0-24 (+1 welford) | positional | named `compile_time_args`; the three existing welford named CTAs pass through unchanged |
| kernel-side `constexpr uint32_t dfb_*_id = tt::CBIndex::c_N;` — 18 sites in `groupnorm_sharded_v2.cpp`, 7 in each reader, 6 in the writer | magic CB index constants | `DFBBinding` per DFB → `dfb::<accessor_name>` |
| `get_tile_size(dfb_*_id)` — 6 sites across the sharded kernels | cb-id free function | `dfb.get_tile_size()` member getter |
| `get_dataformat(dfb_ex_partial_id)` — [reader_mcast_sender_unary_sharded_gn_v2.cpp:132](device/kernels/dataflow/reader_mcast_sender_unary_sharded_gn_v2.cpp#L132), [reader_mcast_receiver_unary_sharded_gn_v2.cpp:49](device/kernels/dataflow/reader_mcast_receiver_unary_sharded_gn_v2.cpp#L49) | cb-id free function (result unused at both sites) | `dfb.get_dataformat()` member getter |
| `constexpr uint32_t dfb_ex_id = tt::CBIndex::c_9;` — [reader_mcast_receiver_unary_sharded_gn_v2.cpp:31](device/kernels/dataflow/reader_mcast_receiver_unary_sharded_gn_v2.cpp#L31) | magic index for a CB this kernel never touches | **dropped** — the receiver reader does not bind `c_9`, so no handle exists to replace it |

Nothing in this factory folds a host-computed offset into a base pointer: there is no `->address()`
expression anywhere in the op. Every tensor reaches a kernel as a `Buffer*` the framework injects, so
each is a clean base that becomes a `TensorBinding` directly.

---

## Applied Patterns

Patterns the sharded factory's plan invokes:

- [Self-loop DFB binding](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-self-loop-dfb-binding)
  — `c_1`, `c_13`, `c_17`, `c_30`, `c_29`, `c_31` on the compute `KernelSpec`s; `c_15` on the sender
  compute spec.
- [Sync-free and single-ended CBs → self-loop DFB](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-sync-free-and-single-ended-cbs--self-loop-dfb)
  — `c_16` (one toucher in every configuration: either the reader's raw write or compute's pack target);
  `c_0` and `c_4` in the configurations where only one kernel touches them.
- [Conditional / optional DFB bindings](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-conditional--optional-dfb-bindings)
  — the largest single body of work in this factory. `c_5`, `c_6`, `c_7`, `c_14`, `c_11`/`c_12`, `c_30`,
  `c_4`, `c_10`, `c_9`/`c_15`, `c_29`/`c_31` are each bound on some configurations only. Three of the
  gates are already `#define`s the host emits (`FUSE_NEGATIVE_MASK`, `READER_REPACK`, `UNTILIZE_OUT`),
  but `do_gamma` / `do_beta` / `num_cores_per_mcast_group > 1` are today **CTAs** the kernels test with
  `if constexpr`, and every one of them guards a reference to a conditionally-bound DFB name. Each must
  be promoted from a CTA gate to a preprocessor gate, with the matching define fed to *every* kernel
  that names the DFB. The compute kernels also construct `DataflowBuffer` objects for all of these
  unconditionally at file scope
  ([groupnorm_sharded_v2.cpp:155-173](device/kernels/compute/groupnorm_sharded_v2.cpp#L155-L173)) —
  those constructions need the same `#ifdef` treatment.
- [Same-FIFO aliasing](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-same-fifo-aliasing-one-dfb-multiple-kernel-side-names)
  — `dfb_ex_global_id` resolving to `c_8` when `num_cores_per_mcast_group == 1`
  ([groupnorm_sharded_v2.cpp:83](device/kernels/compute/groupnorm_sharded_v2.cpp#L83)), and the whole
  `dfb_outgamma_id` / `dfb_inbeta_id` / `dfb_outbeta_id` / `dfb_untilize_in_id` / `dfb_out_id` family at
  `:89-144`. These are one buffer under several names — **one** `DataflowBufferSpec`, **one**
  `DFBBinding`, the extra names expressed as `constexpr auto` handle aliases under the matching
  `#ifdef`. They must not be modelled with `alias_with`.
- [Aliased DFBs](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-aliased-dfbs-legacy-aliased-cbs)
  — the genuinely distinct-index groups: `{c_0, c_16}` when inplace, `{c_0, c_29}` and `{c_1, c_31}`
  when `welford_fp32_alias`, `{c_11, c_12}`, `{c_15, c_9}`. Each member lists every other member.
- [Pass DFB handles directly to LLKs and kernel-lib helpers](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-pass-dfb-handles-directly-to-llks-and-kernel-lib-helpers)
  — every `compute_kernel_lib::reduce` / `tilize` / `untilize` NTTP position and every
  `reconfig_data_format_srcb` / `mul_tiles_init` / `binary_op_init_common` argument.
- [Multi-variant factories](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-multi-variant-factories)
  — the welford / non-welford source selection inside `create_program_artifacts`.
- [Unity-build hygiene for anonymous-namespace symbols](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#pattern-unity-build-hygiene-for-anonymous-namespace-symbols)
  — three factory `.cpp`s in one target, all wanting names like `READER`, `WRITER`, `IN0`.
- [Caution: Avoid varargs](../../../../../../docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/ai/shared/port_patterns.md#caution-avoid-varargs-unless-absolutely-necessary)
  — the sender readers' per-core NoC coordinate block is a genuine vararg (its count is
  `num_mcast_cores`, read as `noc_coord_x[i + 1]` in a loop). The scalars ahead of it —
  `mcast_first_group_dest_noc_start_x` and its seven siblings, and the two `num_mcast_cores_*_group`
  counts — are distinct fields read once each and become **named** args, notwithstanding that their
  legacy offsets shift with `has_mcast_first_group` / `has_mcast_last_group`. What actually varies is
  whether each field *set* is populated, which the host already signals through the two booleans at
  slots 0 and 1.

Not invoked: `Two-toucher DFB → assign 1P+1C` (no same-grid dual-instance work split anywhere in the op),
`Demoting per-group CTA to RTA` (avoided by construction), `Removing pybound legacy factory entry points`
(nothing pybound to remove).

---

## Deferred / Flagged

### Stop signal — the port capitulates here

**Every writer kernel in the op calls a donor whose parameter type Metal 2.0 kernel code cannot supply.**

```cpp
generate_bcast_col_scalar(CircularBuffer(eps_dfb_id), eps);
```

Sites:
- [writer_unary_sharded_gn_rm_gb_v2.cpp:148](device/kernels/dataflow/writer_unary_sharded_gn_rm_gb_v2.cpp#L148)
- [welford_writer_unary_sharded_gn_rm_gb_v2.cpp:68](device/kernels/dataflow/welford_writer_unary_sharded_gn_rm_gb_v2.cpp#L68)
- [writer_unary_gn_rm_gb.cpp:156](device/kernels/dataflow/writer_unary_gn_rm_gb.cpp#L156)
- [welford_writer_unary_gn_rm_gb.cpp:106](device/kernels/dataflow/welford_writer_unary_gn_rm_gb.cpp#L106)

The callee is
`FORCE_INLINE void generate_bcast_col_scalar(CircularBuffer cb, uint32_t scalar)`
at `ttnn/cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp:13` — outside the op directory, and outside
the shared-kernel fork convention (that convention covers kernels owned by ops under
`ttnn/cpp/ttnn/operations/`; this is a shared kernel-code pool). The porter may not edit it and may not
fork it.

It takes a `CircularBuffer` **by value**. The named-handle bridge does not reach it: `dfb::eps` converts
implicitly to `uint32_t`, but `CircularBuffer`'s only constructor from an id is `explicit`, so the call
site has to materialise a `CircularBuffer` wrapper. That leaves a live `CircularBuffer` reference — and
its `api/dataflow/circular_buffer.h` include — inside a ported kernel, which the kernel-side whitelist's
rule-1 sweep and the verification checklist both require to be absent. There is no `DataflowBuffer`
overload of this helper and no `DataflowBuffer → CircularBuffer` conversion.

This blocks **all three factories**, because every factory's writer set contains one of the four call
sites, and a factory converts atomically with the kernels it binds.

The needed upstream change is small and is the subject of the report's handoff entry: a `DataflowBuffer`
overload (or a `DataflowBuffer`-templated form) of `generate_bcast_col_scalar` — and, by the same
argument, of `generate_bcast_row_scalar` and `generate_bcast_unary_scalar`, which have the identical
signature shape and are called the same way from 25 other kernel files across 11 other ops.

### Other findings from planning

1. **Unbound `TensorParameter`s for the borrowed input / output.** See the open question under
   [Planned Spec Shape](#planned-spec-shape). `INPUT` and `OUTPUT` in the sharded factory are named only
   by `DataflowBufferSpec::borrowed_from`; no kernel builds a `TensorAccessor` over them. The validator
   rejects a `TensorParameter` with zero `TensorBinding`s. Whether `borrowed_from` satisfies that rule
   has to be established before the sharded factory can be written, and the recipe does not say. This
   is a real unknown that a resumed port hits on its first build.
2. **Newly required `unpack_modes` entries have no legacy counterpart.** Under `welford_fp32_alias` the
   compute kernel sets `enable_32_bit_dest` and consumes Float32 DFBs beyond the two aliases, so the
   validator demands explicit entries where legacy defaulted silently. The values are derivable
   (`Default` → `UnpackToSrc`), but the *set* of DFBs needing one has to be enumerated per configuration.
3. **`preferred_noc_for_dram_read` / `_write` are `tt::tt_metal::detail::` functions carrying the comment
   "only used in op_profiler, are unstable and have not been designed for general use"
   ([kernel_types.hpp:131-146](../../../../../../tt_metal/api/tt-metalium/kernel_types.hpp#L131-L146)),
   yet all three groupnorm factories call them to pick their NOCs. The port would copy the resolved
   values, so nothing changes behaviourally — but a Metal 2.0 factory reaching into `detail::` for a
   hardware-config value is worth a decision by the owners.
4. **`compute_receiver_desc` with an empty core range** has no Metal 2.0 expression and needs an explicit
   `use_mcast` gate the legacy code does not have. Noted in the Kernels table; not a blocker.
