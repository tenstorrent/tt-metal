# Port Plan — `ttnn/cpp/ttnn/operations/matmul`

Port plan for **one** factory — `MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory` — ported
from `ProgramDescriptorFactoryConcept` to Metal 2.0's base `ProgramSpecFactoryConcept`.
Written during the inventory and planning steps; committed alongside the port for review.

The op's other seven factories stay on their legacy concepts. `program_factory_t` dispatches per
factory, so the op keeps building and running with this one ported.

**Stacked on #55961** (`iwrosz/port-mm-dram-sharded`, base commit `a7796e0f567`), which ported the
sibling `MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory` and **created** the shared compute
kernel's `_metal2` fork. This port is therefore at **rung 1 — reuse the fork**, not rung 2 as the
audit brief states; see [Shared kernels](#shared-kernels).

## Legacy Inventory

> Every `factory:NNN` below is a line in the **pre-port** file, i.e. at this branch's base
> commit `a7796e0f567` — the port rewrites it. `matmul_nanobind.cpp` line numbers are also
> as of that base, which is 33 lines earlier than the audit recorded because #55961 removed
> its own pybind block.

### Legacy factory shape

- Concept: `ProgramDescriptorFactoryConcept` — `create_descriptor` returning a `ProgramDescriptor`
  is the factory's only member (`…batched_hs_dram_sharded_program_factory.hpp:14`).
- Variants: single. One code path, plus a bias conditional and one config-dependent CB branch.
- Custom `compute_program_hash`: **none** — default reflection-based hash. A deliberately-renamed
  helper `compute_descriptor_program_hash` sits at `device/matmul_device_operation.hpp:50` with a
  comment saying it is *not* named `compute_program_hash` so the framework does not detect it, plus
  a pybind alias exposing it under that name. Left entirely intact.
- Factory methods live in a `program_factory_t` variant, so the direct-descriptor exception
  (device-op-class edit 3) does not apply.
- Work-split driver: **none**. No `split_work_to_cores`; the worker set comes from
  `get_optimal_dram_bank_to_reader_assignment` (factory:85) and all three kernels are placed over
  the bounding box `all_cores_in_rect_grid` (factory:142).

### Kernels

All three over `all_cores_in_rect_grid` (the bounding box of workers ∪ input-storage ∪
output-storage cores). One `KernelDescriptor` each — no multiplicity.

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | CRTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|---|
| in0 reader | `dataflow/reader_bmm_tile_layout_in0_sender_dram_sharded_height.cpp` | `all_cores_in_rect_grid` | 6: `in0_block_tiles`, `in0_block_tiles*in0_single_tile_size`, `num_blocks`, `batches_per_core`, `in0_batch_stride_bytes`, `in2_CB_size` (factory:383-390) | `cb_in0`→`c_0` (factory:455-457) | worker 4 / idle 1 (factory:532-533, 559-564) | none | none (`reader_defines` stays empty) | unset → resolved **O2** | `DataMovementConfigDescriptor{RISCV_1, in0_noc}` (factory:458-459) |
| in1 writer | `dataflow/reader_bmm_tile_layout_in1_sender_dram_sharded_height.cpp` | `all_cores_in_rect_grid` | 10 + 3 when bias: `in1_buffer_page_size`, `in1_buffer_num_pages`, `per_core_N`, `in1_block_tiles`, `num_blocks`, `out_block_tiles`, `batches_per_core`, `in1_batch_stride_bytes`, `out_batch_stride_bytes`, `out_reshard_CB_size` [, `bias_buffer_page_size`, `bias_buffer_num_pages`, `in3_block_tiles`] (factory:392-408) | `cb_in1`→`c_1`, `cb_bias`→`c_3`, `cb_out`→`c_4` (factory:470-474) | worker 8 / idle 1 (factory:535-536, 567-580) | none | `FUSE_BIAS` (if bias), `SKIP_WRITE_BACK` (hardcoded off), `OUT_SHARDED` (always, factory:444) | unset → resolved **O2** | `DataMovementConfigDescriptor{RISCV_0, in1_noc}` (factory:475-476) |
| compute | `compute/bmm_large_block_zm_fused_bias_activation.cpp` | `all_cores_in_rect_grid` | 18 + 1 when bias (factory:415-437) — see below | 10 + 4 when SFPU activation (factory:487-510) — see below | worker 1 / idle 1 (factory:538-539, 583-586) | none | `MATMUL_DRAM_SHARDED` (always), `FUSE_BIAS`, `PACK_RELU` xor `SFPU_ACTIVATION`, `PACKER_L1_ACC`, `FP32_DEST_ACC_EN`, `SKIP_COMPUTE` (hardcoded off), + stagger/throttle | unset → resolved **O3** | `ComputeConfigDescriptor{math_fidelity, fp32_dest_acc_en, dst_full_sync_en, math_approx_mode}` (factory:512-516) |

Compute positional CTAs, in emission order (factory:415-437) — read from the host, which is
authoritative for slot position:

`0 in0_block_w` · `1 in0_num_subblocks` · `2 in0_block_tiles` · `3 in0_subblock_num_tiles` ·
`4 in1_num_subblocks` · `5 in1_block_tiles` · `6 per_core_N` · `7 num_blocks` · `8 1u` · `9 1u` ·
`10 out_subblock_h` · `11 out_subblock_w` · `12 out_subblock_num_tiles` · `13 batches_per_core` ·
`14 out_block_tiles` · `15 untilize_out` · `16 0u` (`get_batch_from_reader`) ·
`17 0u` (`in0_transpose_tile`) · `18 1u` (`row_broadcast_bias`, only when bias).

Compute named CTAs (factory:487-510): `cb_in0`→`c_0`, `cb_in1`→`c_1`, `cb_bias`→`c_3`,
`cb_out`→`c_4`, `cb_intermed0`→`c_5`, `cb_in0_intermediate`→`c_8`, `cb_in1_intermediate`→`c_9`,
`cb_in0_transposed`→`c_10`, `bias_ntiles`→`per_core_N`, `last_subblock_w_valid`→`out_subblock_w`;
plus `activation_type`, `activation_param0/1/2` when a non-RELU fused activation is present.

`grep -n opt_level` on the factory returns nothing — hence the *resolved* levels above.

### CBs

| index | total_size | core_ranges | data_format | page_size | tile |
|---|---|---|---|---|---|
| `c_0` in0 | `in0_CB_size` = `per_core_M*in0_block_w*2 * in0_single_tile_size` | `all_cores_in_rect_grid` | `in0_data_format` | `in0_single_tile_size` | `in0_tile` |
| `c_1` in1 | `in1_CB_size` = `in0_block_w*per_core_N*3 * in1_single_tile_size` | `all_cores_in_rect_grid` | `in1_data_format` | `in1_single_tile_size` | `in1_tile` |
| `c_2` in0 sharded | `in2_CB_size` = `in0_shard_tiles * in0_single_tile_size` | `input_all_storage_cores` | `in0_data_format` | `in0_single_tile_size` | `in0_tile` | **borrowed** (`cb_desc.tensor = &in0_tensor`, factory:257) |
| `c_3` bias | `in3_CB_size` = `per_core_N * bias_single_tile_size` | `all_cores_in_rect_grid` | `bias_data_format` | `bias_single_tile_size` | `bias_tile` | **conditional** on bias (factory:262) |
| `c_4` out | `out_reshard_CB_size` = `out_shard_tiles * output_single_tile_size` | `all_worker_cores` | `output_data_format` | `output_single_tile_size` | `output_tile` |
| `c_5` interm0 | `interm0_CB_size` = `per_core_M*per_core_N * interm0_single_tile_size` in the split branch; **`out_reshard_CB_size`** in the shared branch | `all_worker_cores` | `interm0_data_format` | `interm0_single_tile_size` | `output_tile` |
| `c_6` out reshard | `out_reshard_CB_size` | `output_all_storage_cores` | `output_data_format` | `output_single_tile_size` | `output_tile` | **borrowed** (`cb_desc.tensor = &out_tensor`, factory:329) |

`c_4` + `c_5` are **config-dependent**: when `interm0_data_format != output_data_format`
(factory:277-300) they are two separate `CBDescriptor`s; otherwise (factory:301-317) **one**
`CBDescriptor` carries both `format_descriptors` — the aliased case. Note the shared branch sizes
the whole region at `out_reshard_CB_size`, so in that branch `c_5`'s backing size is
`out_reshard_CB_size`, not `interm0_CB_size`; the port must reproduce that, since the alias-group
legality rule requires identical total backing size.

### Semaphores

**none** — the factory creates no semaphores at all. There is no cross-core synchronisation to
express.

### Tensor accessors

**No kernel constructs a `TensorAccessor`.** All four addresses arrive as tensor objects pushed
into the runtime-arg lists and are consumed **raw** — every binding is Case 2.

| host site | originating Tensor | RTA slot (host) | kernel use |
|---|---|---|---|
| factory:563 | in0 (`in0_tensor`) | in0 reader slot 3 | `{.noc_x, .noc_y, .addr = input_shard_l1_addr + read_offset}` (in0 reader:57-62) |
| factory:569 | in1 (`in1_tensor`) | in1 writer slot 1 | `{.bank_id, .addr = in1_tensor_addr + in1_batch_offset + curr_dram_offset}` (in1 writer:91-96) |
| factory:571 | bias (`*bias_tensor`) | in1 writer slot 2 | `{.bank_id, .addr = in3_tensor_addr}` (in1 writer:110-115) |
| factory:579 | output (`out_tensor`) | in1 writer slot 7 | `{.noc_x, .noc_y, .addr = output_shard_l1_addr + out_batch_offset}` (in1 writer:126-133) |

### Work split

n/a — no `split_work_to_cores`. One `KernelDescriptor` per kernel, all over the bounding box; the
worker/idle distinction is carried by an RTA (`worker_core_type` / `is_worker_core`), not by
separate specs. No multiplicity to preserve.

### Shared kernels

Census by filename across `ttnn/cpp/ttnn/operations/`, hits disambiguated to genuine
kernel-source bindings:

| kernel | binders | `_metal2` fork beside it? | rung |
|---|---|---|---|
| `dataflow/reader_bmm_tile_layout_in0_sender_dram_sharded_height.cpp` | **1** (this factory) | n/a | **convert in place** |
| `dataflow/reader_bmm_tile_layout_in1_sender_dram_sharded_height.cpp` | **1** (this factory) | n/a | **convert in place** |
| `compute/bmm_large_block_zm_fused_bias_activation.cpp` | **6** | **yes** — `bmm_large_block_zm_fused_bias_activation_metal2.cpp`, created by #55961 | **rung 1 — reuse** |

**The audit brief says rung 2 (create the fork); that is stale.** The locational check it ran —
`ls` of `device/kernels/compute/` — now shows the fork, created by the port this branch is stacked
on. Reuse is the first applicable rung, so this port creates nothing and edits nothing in the
compute kernel: it points its `KernelSpec::source` at the fork and adopts the fork's names. The
original already carries #55961's pointer comment, so there is nothing to add there either.

Remaining unmigrated consumers of the legacy copy after this port (the sunset list — coordination,
not authorization): `matmul_multicore_reuse_optimized_program_factory.cpp`,
`matmul_multicore_reuse_mcast_1d_program_factory.cpp` (hosts two factories),
`matmul_multicore_reuse_mcast_2d_program_factory.cpp`,
`sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_optimized.cpp`. This port removes
`matmul_multicore_reuse_batched_hs_dram_sharded_program_factory.cpp` from #55961's list of six.

**Fork fit check — the fork is read-only to this port, so fit is a precondition, not a choice.**
Every name the fork reads is one this factory already supplies, under the same meaning:

- DFBs: `dfb::in0` ← `c_0`, `dfb::in1` ← `c_1`, `dfb::bias` ← `c_3` (under `FUSE_BIAS`),
  `dfb::out` ← `c_4`, `dfb::intermed0` ← `c_5`.
- Named args: `is_worker_core` (RTA, under `MATMUL_DRAM_SHARDED`); CTAs `in0_block_w`,
  `in0_num_subblocks`, `in0_block_num_tiles`, `in0_subblock_num_tiles`, `in1_num_subblocks`,
  `in1_block_num_tiles`, `in1_block_w`, `num_blocks_inner_dim`, `num_blocks_w_dim`,
  `num_blocks_h_dim`, `out_subblock_h`, `out_subblock_w`, `out_subblock_num_tiles`, `batch`,
  `out_block_num_tiles`, `untilize_out`, `get_batch_from_reader`; `last_subblock_w_valid` (under
  `MATMUL_DRAM_SHARDED`); `bias_ntiles` + `row_broadcast_bias` (under `FUSE_BIAS`);
  `activation_type`, `activation_param0/1/2` (under `SFPU_ACTIVATION`). This maps 1:1 onto legacy
  positional CTAs 0-16 and 18 plus the named CTAs, in order.
- `#ifdef`s the fork gates on that this factory must **not** define: `IN0_TRANSPOSE_TILE` (this
  factory hardcodes `in0_transpose_tile = 0u`, so `dfb::in0_transposed` is never bound and never
  named), `MM_PARTIALS_RELOAD_ALIAS` (this factory never emitted `MM_PARTIALS_RELOAD_ALIAS_CB`, so
  `mm_partials_reload_dfb_id == mm_partials_dfb_id` and the reload alias is never bound),
  `BIAS_FULL_BLOCK` (only `matmul_multicore_reuse_optimized` sets it). Same three branches #55961
  left uncompiled — this port does not change that.

No misfit in either direction, so no handoff point and no capitulation.

### Flags

- **Six dead CTAs and one dead RTA in the two DM kernels** — declared and never used. Not in the
  audit. Verified by word-boundary occurrence count (1 = the declaration only):
  in0 reader `in0_shard_size_bytes` (CTA 5, kernel:27); in1 writer `in1_num_pages` (CTA 1,
  kernel:40), `in1_block_w` (CTA 2, kernel:41), `out_shard_size_bytes` (CTA 9, kernel:48),
  `in3_page_size` (CTA 10, kernel:51), `in3_num_pages` (CTA 11, kernel:52), and RTA `vc`
  (slot 4, kernel:31).
  **Carried across faithfully as named args, with the kernel-side declarations kept.** They are
  read by the kernel today (the `constexpr` / `const` declaration *is* the read), so the binding is
  exercised and the resulting constant is simply unused — exactly the status quo, which compiles
  warning-free today. Dropping them would change the dispatched payload for no port reason.
  Reported as findings, not fixed.
- The three `skip_*` flags (`skip_compute`, `skip_write_back` at factory:716-717, and the absent
  in0-mcast one) are hardcoded `false` at the only call site. Not config axes; censused once.
- `bias_data_format` defaults to `Bfp8_b` when there is no bias (factory:625) and the bias tile
  falls back to `output_tile` (factory:709). Both only feed a CB and CTAs that are not created in
  the no-bias build.

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `ProgramSpecFactoryConcept` (base).
- **Custom `compute_program_hash`**: none — leave the renamed helper and its pybind alias intact.
- **No `override_runtime_arguments`.** The factory has none today and the base concept means the
  framework refreshes tensor bindings on cache hit. Do **not** add one.
- **Implementation notes**: `create_descriptor` → `create_program_artifacts` returning
  `ttnn::device_operation::ProgramArtifacts{.spec, .run_params}`. The `ComputeHardwareConfig` is
  resolved at the entry point with `ttnn::to_compute_hardware_config(device->arch(),
  compute_kernel_config)` and passed down, replacing the four scalars the legacy helper took;
  `fp32_dest_acc_en` and `packer_l1_acc` are still passed separately because the helper body uses
  them to derive `packer_l1_acc_en`, `interm0_data_format` and the subblock params.

## Planned Spec Shape

- **KernelSpecs**: 3 — `in0_reader`, `in1_writer`, `compute`. 1:1 with legacy; no multiplicity.
- **DataflowBufferSpecs**: **5** (6 with bias) — `in0`, `in1`, `out`, `intermed0`
  (+ `bias` when present). Down from 7 legacy CBs: `c_2` and `c_6` are dropped (below).
  `out` + `intermed0` carry mutual `advanced_options.alias_with` **only** in the
  `interm0_data_format == output_data_format` branch.
- **SemaphoreSpecs**: none.
- **TensorParameters**: 3 (4 with bias) — `in0`, `in1`, `output` (+ `bias`).
- **WorkUnitSpecs**: 1 — all three kernels over `all_cores_in_rect_grid`.
- **Op-owned tensors**: none.

### DFB endpoint census and dispositions

Re-derived from the kernel bodies, not transcribed:

| DFB | touchers on a node | roles | disposition |
|---|---|---|---|
| `in0` | in0 reader (`reserve_back`/`push_back`, locked producer), compute (locked consumer) | 1P+1C | plain 1:1 |
| `in1` | in1 writer (locked producer), compute (locked consumer) | 1P+1C | plain 1:1 |
| `bias` | in1 writer (locked producer), compute (locked consumer) | 1P+1C | plain 1:1, **conditional binding** |
| `out` | compute (locked producer), in1 writer (`wait_front`/`pop_front`, locked consumer) | 1P+1C | plain 1:1 |
| `intermed0` | compute only — packs into it and reads it back for accumulation and the bias add | locked on both sides | **compute self-loop**: PRODUCER + CONSUMER on the compute KernelSpec, one accessor name |
| ~~`c_2`~~ | **none** | — | **drop** |
| ~~`c_6`~~ | **none** | — | **drop** |

`intermed0` is a self-loop, not sync-free — the compute kernel runs genuine FIFO machinery against
it (`mm_partials_dfb.wait_front` / `pop_front` / `reserve_back` in the fork). A self-loop is a
statement about endpoints, not about synchronisation.

### The two dropped CBs

`c_2` (`cb_desc.tensor = &in0_tensor`, factory:247-259) and `c_6`
(`cb_desc.tensor = &out_tensor`, factory:319-331) have **zero endpoints**: neither index appears
anywhere in the factory beyond its own `CBFormatDescriptor` (factory:253, factory:325 — one hit
each), no named CTA on any of the three kernels carries either index, and no kernel body references
them. Metal 2.0 rejects a DFB with no producer and no consumer binding, so the drop is the only
expressible outcome.

**Why they exist, which the audit could not determine — they are vestiges of an idiom the sibling
factory still uses live.** In `MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory` the same two
borrowed CBs are the *mechanism* for obtaining a resident shard's L1 base address: its in1 writer
calls `dfb_out_reshard.get_write_ptr()`
(`reader_bmm_tile_layout_in1_sender_dram_sharded.cpp:219`) and its in0 sender constructs
`DataflowBuffer dfb_in2(dfb::in0_sharded)` (same file family, in0 sender:61) — both sync-free
one-touchers, self-looped by #55961 for exactly that reason. BatchedHS instead takes both addresses
from runtime args (in0 reader slot 3, in1 writer slot 7) and never names `c_2` or `c_6`. So the
declarations are what remains of the borrowed-CB idiom after the kernels were rewritten to use an
explicit address; nothing reads them.

Because both are *borrowed* views onto tensors that are already resident, dropping them allocates
and frees nothing: a CB backed by a tensor's buffer points at the tensor's own L1 shard rather than
taking an allocation of its own. Confirmed with the invoker before the drop landed; the evidence
above rides the PR description for the op owner.

### Hardware config

Style A — the factory resolves a TTNN `ComputeKernelConfig` via `get_compute_kernel_config_args`
(factory:673-674), so the port translates with `to_compute_hardware_config(device->arch(),
config)`. All four helper-covered knobs are set on the legacy compute descriptor
(factory:512-516), so there is no resolved-but-unset field to reapply. `packer_l1_acc` has no
Metal 2.0 counterpart but is genuinely consumed host-side (it derives `packer_l1_acc_en` at
factory:161 and hence `interm0_data_format` at factory:163-165), so it stays a parameter.

`unpack_modes`: explicit `UnpackMode::UnpackToSrc` for every DFB the compute kernel **consumes
from** — `in0`, `in1`, `intermed0`, and `bias` when present. `UnpackToSrc` is the framework default
and the legacy `ComputeConfigDescriptor` left `unpack_to_dest_mode` empty, so this is
behaviour-preserving; it is stated explicitly because
`compute_hardware_config.hpp:119-121` requires an explicit mode for any 32-bit-format DFB when
`enable_32_bit_dest` is set, and `fp32_dest_acc_en` forces `interm0_data_format = Float32`
(factory:163-165). Reading the resolved config rather than the tensor dtypes, per the brief.

Both DM kernels keep **explicit custom configs** — `DataMovementGen1Config{RISCV_1, in0_noc}` and
`{RISCV_0, in1_noc}`, field for field. Neither NOC is the role default (`in0_noc` comes from
`preferred_noc_for_dram_write`, `in1_noc` from `preferred_noc_for_dram_read`, factory:81 and 442),
so `create_reader_datamovement_config` / `create_writer_datamovement_config` would silently
substitute a wrong NOC. `opt_level` set explicitly to `O3` on the compute KernelSpec only; both DM
specs carry none, matching their resolved legacy `O2`.

## Preserved Multiplicity

none — no work-split multiplicity in legacy. One `KernelDescriptor` per kernel source, all over one
core range.

## Dropped Plumbing

| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| factory:563 → in0 reader RTA slot 3 | `in0_reader_args.push_back(in0_tensor)`, read as `get_arg_val<uint32_t>(3)` | `TensorParameter`/`TensorBinding` `in0`; base pulled kernel-side via the `get_bank_base_address` bridge |
| factory:569 → in1 writer RTA slot 1 | `in1_writer_args.push_back(in1_tensor)` | `TensorParameter`/`TensorBinding` `in1`, same bridge |
| factory:571 → in1 writer RTA slot 2 | `in1_writer_args.push_back(*bias_tensor)` (or `0u` filler) | `TensorParameter`/`TensorBinding` `bias`, conditional, same bridge |
| factory:579 → in1 writer RTA slot 7 | `in1_writer_args.push_back(out_tensor)` | `TensorParameter`/`TensorBinding` `output`, same bridge |
| factory:456 `cb_in0`→`c_0` | named CTA carrying a magic CB index | `DFBBinding{in0, "in0", PRODUCER}` (reader) / `CONSUMER` (compute) |
| factory:471-473 `cb_in1`/`cb_bias`/`cb_out` | ditto | `DFBBinding`s on the in1 writer |
| factory:488-492 `cb_in0`/`cb_in1`/`cb_bias`/`cb_out`/`cb_intermed0` | ditto | `DFBBinding`s on compute |
| factory:493 `cb_in0_intermediate`→`c_8` | named CTA for a CB no descriptor allocates and no kernel reads | **dropped** — dead on both ends |
| factory:494 `cb_in1_intermediate`→`c_9` | ditto | **dropped** — dead on both ends |
| factory:495 `cb_in0_transposed`→`c_10` | named CTA for a nonexistent CB, kept alive only because compute's parse-time ternary name-looks-up both operands | **dropped**; the fork's `#ifdef IN0_TRANSPOSE_TILE` (undefined here) removes the token entirely |
| factory:415-433 compute CTAs 0-17 | positional `get_compile_time_arg_val(0..17)` | named CTAs; slot 17 (`in0_transpose_tile = 0u`) becomes the absence of the `IN0_TRANSPOSE_TILE` define |
| factory:436 compute CTA 18 | positional | named CTA `row_broadcast_bias` |
| factory:383-408 DM CTAs | positional | named CTAs, names taken from the kernels' own locals |
| in0 reader RTAs 0-2, in1 writer RTAs 0, 3-6 | positional `get_arg_val<uint32_t>(N)` | named RTAs via `runtime_arg_schema.runtime_arg_names` |

`TensorAccessorArgs` plumbing: **none to drop** — no kernel ever constructed a `TensorAccessor`.
Page-size 3rd-argument CTAs: none. Semaphore-ID RTAs: none.

### Short arg lists become zero-fills

The legacy factory emits a **1-element** RTA list on idle cores (factory:532-539) against 4 (in0
reader) and 8 (in1 writer) on workers. Metal 2.0 requires every name in a kernel's
`runtime_arg_schema` to have a value on every node the kernel runs on, so the port zero-fills the
remaining names on idle nodes. Behaviour-preserving — both kernels return on the
`worker_core_type == 0` / `is_worker_core` test before reading anything else (in0 reader:30-33,
in1 writer:21-24) — but it is a real change to the dispatched payload and is called out for
reviewers. Compute is 1 arg on both sides, so nothing to fill.

## Applied Patterns

- **Conditional / optional resource bindings** — `bias`: the DFB, the `TensorParameter`/binding,
  the `FUSE_BIAS` define, and the `in3_*` / `row_broadcast_bias` CTAs are all emitted together or
  not at all. Note the legacy factory hands `cb_bias`→`c_3` to compute and the in1 writer
  *unconditionally* (factory:472, 490) while only creating the CB when bias is present; both
  kernels already gate their reads behind `#ifdef FUSE_BIAS`, so the port simply makes the binding
  conditional too.
- **Conditional / optional resource bindings**, again — for `c_10`: the fork's
  `#ifdef IN0_TRANSPOSE_TILE` is the already-built form of this pattern, and this port satisfies it
  by not emitting the define.
- **Self-loop DFB binding** — `intermed0` on the compute KernelSpec, PRODUCER + CONSUMER under one
  accessor name.
- **Aliased DFBs** — `out` + `intermed0` mutual `alias_with`, in the
  `interm0_data_format == output_data_format` branch only. Derived per instantiation; the group is
  a strict 2-clique, both members sized `out_reshard_CB_size`, both bound to the same kernels,
  neither borrowed.
- **Same-FIFO aliasing** — the fork's `constexpr uint32_t mm_out_dfb_id = mm_partials_dfb_id;`
  (fork:249). Already expressed correctly in the fork as a `constexpr` handle alias over one
  binding. **Not** modelled with `alias_with`; the two are distinct and conflating them would give
  two independent FIFOs at one address. Nothing for this port to do beyond not disturbing it.
- **Pass DFB handles directly to LLKs and kernel-lib helpers** — `get_tile_size(dfb::in1)` /
  `get_tile_size(dfb::out)` in the in1 writer keep the free-function form with the binding token,
  because the `constexpr` declaration is the whole test (in1 writer:59-60). Contrast
  `dfb_in3.get_tile_size()` at in1 writer:113, already the member form in a non-`constexpr`
  context, which stays as it is.
- **Removing pybound legacy factory entry points** — `matmul_nanobind.cpp:1292-1307`, the
  `nb::class_<…BatchedHSDRAMShardedProgramFactory>` block whose only member is the vanishing
  `create_descriptor`. Deleted; recorded under Handoff points. The `nb::class_<MatmulDeviceOperation>`
  block at :1222-1237 is untouched.
- **Dropping a pybind-hook-only parameter** — `create_descriptor`'s fourth argument
  `const std::optional<CoreRangeSet>& core_range_set`, ignored by the body (spelled
  `/*core_range_set*/` at factory:603). Dropped; there is no production default to inline because
  nothing reads it.
- **RTA varargs: none.** Both DM kernels read their args as distinct fields at constant indices in
  a block at the top. Every one is nameable; nothing here justifies `get_vararg`, and no
  `get_arg_addr` / `get_common_arg_addr` construct appears either (the gap #55961 reported in the
  audit's recognition signals — checked for explicitly here, and genuinely absent).

## Deferred / Flagged

- **Fork drift, reported not fixed.** The required first-step diff of the fork against the original
  shows one divergence beyond the Metal 2.0 conversion: the original's `FUSE_BIAS` path calls
  `apply_activation_from_pack<…>(out_subblock_num_tiles)` (original:554) while the fork carries
  that helper's body inlined (fork:571-588). Same four template arguments, so behaviour is
  identical, but the fork is meant to track the original and Ports 4-5 inherit this. A fork with a
  consumer is read-only to a rung-1 port, so this port does not touch it; confirmed with the
  invoker. Written up in the port report.
- The six dead CTAs, the dead `vc` RTA, and the provably-unreachable
  `mm_partials_reload_dfb_id != mm_partials_dfb_id` branch in the shared compute kernel — all
  carried verbatim, all reported.
- The `_UNSUPPORTED_FACTORY` gap in `models/experimental/ops/descriptors/matmul.py` widens by one
  more factory here, exactly as #55224 and #55961 widened it. Left unguarded for the same reason:
  the general fix is a `main`-targeted change belonging with the still-open decision #55224 raised,
  and guarding one config while that gap stays open would be piecemeal. No in-tree caller passes a
  batched DRAM-sharded config to the descriptor `matmul()`.

## Test set (invoker-confirmed)

This factory is **never auto-selected** — `create_matmul_program_config` excludes the config type,
so only an explicitly-constructed `MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig`
reaches it. `tests/ttnn/unit_tests/gtests/test_matmul.cpp:19-29` says so in a comment and
**deliberately excludes** this factory from the gtest smoke set, pointing coverage at the deepseek
pytests.

Baseline, confirmed with the invoker before being relied on:

| test | why |
|---|---|
| `tests/ttnn/unit_tests/operations/matmul/test_matmul_deepseek.py::test_matmul_batched_dram_sharded` (:796) | primary coverage of this factory |
| `…::test_matmul_batched_dram_sharded_program_cache` (:963) | program-cache behaviour, which the port must not change |
| `unit_tests_ttnn --gtest_filter='*MatmulSmoke*'` | #55961's baseline — proves the stacked base still passes with a second consumer on the shared fork |
| `tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_dram_sharded.py` | ditto |
| `tests/ttnn/nightly/unit_tests/operations/matmul/test_matmul_activations.py` | ditto — the fused-bias × activation matrix through the fork |

Not runnable on this box (single Blackhole p150b): the model-level paths in
`models/demos/deepseek_v3/tt/mla/mla1d.py:760,822` and `models/demos/deepseek_v3/tests/test_mla.py`,
which need multi-device. Recorded so the gap is explicit.

Every run with `TT_METAL_WATCHER=10`, once before conversion and once after.
