# Port Plan — `matmul` / `MatmulMultiCoreReuseMcast1DProgramFactory`

Port plan for the mcast-1d matmul factory, ported from the `ProgramDescriptor` API to Metal 2.0.
Written during the inventory and planning steps; committed alongside the port for review.

## Preamble — why this port ran without a `METAL2_PORT_BRIEF.md`

The audit (`METAL2_PREPORT_AUDIT.md`, 2026-09-01) returned **RED** and therefore issued no brief.
Its single gate was the readiness sheet's `Is able to port? = no`, attributed to
`Smuggled pointer = yes`. The audit disagreed with that cell on the evidence and routed the
question to the sheet owner rather than overriding it.

**The sheet has since been reclassified.** Re-fetched live from *TTNN Operations analysis*
(`1KUMj8SyBGlNMZlLFgs1MbAZlO2g6EoUc4KaxSlcy8jw`, owner `dgomez@`, modified 2026-09-16 15:36 UTC),
the row (`matmul`, `MatmulDeviceOperation`, `MatmulMultiCoreReuseMcast1DProgramFactory`) now reads:

| Column | Audit (2026-09-01) | Sheet now |
|---|---|---|
| `Smuggled pointer` | yes | **no** |
| `Op Classification` | Broken Op | **PD Op (custom)** |
| `Diego validation` | no | **yes** |
| **`Is able to port?`** | **no** | **yes** |
| `Porting Target` | (N/A) | **`CustomProgramSpecFactoryConcept`** |

The gate is cleared and the audit's own prediction — that it would clear *sheet-side with the op's
code untouched* — is what happened. Because the audit deliberately ran all seven informational
subjects anyway, its **Port-work summary** is the actionable input this plan builds on, standing in
for the brief. No re-audit was needed: the factory `.cpp`/`.hpp` are byte-identical to the audit's
base commit (`beb2ea8f08a`), so every line reference in the audit still resolves.

**What did move under the audit since then** (branch rebased onto `origin/main` @ `061086bbdec`):
the six kernels were touched by cleanup PRs, and — materially — a `_metal2` fork of the compute
kernel now exists (created by #55961). The audit's "no fork exists for any of the six" is stale;
see *Shared kernels* below, where the compute kernel is now a **rung-1 reuse**, not a rung-2 create.

---

## Legacy Inventory

### Legacy factory shape

- **Concept**: `ProgramDescriptorFactoryConcept` — `create_descriptor` returning `ProgramDescriptor`
  (`device/factory/matmul_multicore_reuse_mcast_1d_program_factory.hpp:35`).
- **Factory methods live in** a `program_factory_t` variant member (`MatmulDeviceOperation`), so
  the direct-descriptor conversion (device-op-class exception 3) does **not** apply.
- **Variants**: two descriptor paths selected inside `create_descriptor` by
  `program_config.mcast_in0`:
  - `create_program_mcast_in0_descriptor` (`:3141`-`:4216`)
  - `create_program_mcast_in1_descriptor` (`:4217`-`:5150`)
  `gather_in0` is rejected with a `TT_FATAL` at `:5525` and is **not** part of this factory's
  descriptor path.
- **Custom `compute_program_hash`**: **none framework-visible.** The device-op declares
  `compute_descriptor_program_hash` (`device/matmul_device_operation.hpp:50`) which is deliberately
  *not* named `compute_program_hash`, so the framework uses its default reflection hash. Nothing to
  touch; matches the sheet's `Custom hash = no`.

> **Scoping note — two factories share one 5,885-line file.** The same `.cpp` hosts
> `MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory`, whose imperative
> `process_*_program_and_create_override_variables` builders occupy `:77`-`:2917` and whose entry
> point is `matmul_multi_core_reuse_mcast_1d_optimized_` (`:5152`). **None of that is in scope.**
> Everything this port rewrites lies in `:3141`-`:5150` plus `:5482`-`:5700`.

### Kernels

Six sources; the block-sharded in0 reader is bound **three times** in `mcast_in0` (work-split
multiplicity, see *Preserved Multiplicity*).

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|
| `in0_sender` (mcast_in0, interleaved) | `dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp` | `in0_mcast_cores_with_work_and_in_receiver_grid` | 28 + TA(in0) + TA(sparsity placeholder) + `num_batch_compute` | `cb_in0`, `cb_in0_sharded`, `cb_sparsity`, `num_active` | 8 | `SKIP_MCAST`? | unset → O2 | `RISCV_1` / `in0_noc` |
| `in0_sender` (mcast_in0, in0 sharded) | `dataflow/…_in0_sender_receiver_padding_block_sharded.cpp` | same | 22 + TA + TA + 1 | `cb_in0`, `cb_in0_sharded`, `cb_l1_array`, `cb_sparsity`, `num_active` | 5 + noc lists | `SKIP_MCAST`? | unset → O2 | `RISCV_1` / `in0_noc` |
| `in0_no_work_in_receiver` | same block-sharded source | `in0_mcast_cores_without_work_and_in_receiver_grid` | same, slots 0/1 = `0,1` | `cb_in0`, `cb_in0_sharded`, `cb_l1_array` | same | same | unset → O2 | `RISCV_1` / `in0_noc` |
| `in0_no_work_not_in_receiver` | same block-sharded source | `in0_mcast_cores_without_work_and_not_in_receiver_grid` | same, slots 0/1 = `0,0` | `cb_in0`, `cb_in0_sharded`, `cb_l1_array` | same | same | unset → O2 | `RISCV_1` / `in0_noc` |
| `in0_receiver` (mcast_in0) | `dataflow/reader_bmm_tile_layout_in0_receiver.cpp` | `in0_mcast_receivers` | 8 | `cb_in0` | 2 | — | unset → O2 | `RISCV_1` / `in0_noc` |
| `in1_sender_writer` | `dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp` | mcast_in0: `all_cores_with_work`; mcast_in1: `in1_mcast_sender` | 33 + TA(in1) + TA(sparsity) + TA(out) + TA(bias)? | `cb_in1`, `cb_bias`, `cb_out`, `cb_sparsity`, `num_active` | 19-21 | `FUSE_BIAS`?, `IN1_SHARDED`?, `BIAS_SHARDED`?, `OUT_SHARDED`?, `SKIP_MCAST` | unset → O2 | `RISCV_0` / `in1_noc` |
| `in1_receiver_writer` (mcast_in1) | `dataflow/reader_bmm_tile_layout_in1_receiver_writer_padding.cpp` | `in1_mcast_receivers` | 19 + TA(out) | `cb_in1`, `cb_bias`, `cb_out` | 13-15 | `FUSE_BIAS`?, `OUT_SHARDED`? | unset → O2 | `RISCV_0` / `in1_noc` |
| `compute` | `compute/bmm_large_block_zm_fused_bias_activation.cpp` | mcast_in0: `all_cores_with_work`; mcast_in1: `all_cores` | 18 (+1 if bias) | `cb_in0`, `cb_in1`, `cb_bias`, `cb_out`, `cb_intermed0`, `cb_in0_transposed`, `bias_ntiles` (+4 activation) | — | `FUSE_BIAS`?, `PACK_RELU`/`SFPU_ACTIVATION`?, `PACKER_L1_ACC`?, `FP32_DEST_ACC_EN`?, `IN1_TRANSPOSE_TILE`?, `MM_PARTIALS_RELOAD_ALIAS_CB`?, stagger/throttle | **absent → resolves O3** | `ComputeConfigDescriptor` |

`grep -n opt_level` over `:3141`-`:5150` returns **nothing** → every kernel's level is the legacy
resolved default: `O2` for the six DM descriptors, **`O3` for the compute `ComputeConfigDescriptor`**.

### CBs

`mcast_in0` (`:3855`-`:4007`):

| index | total_size | core_ranges | data_format | page_size | tile | condition | borrowed from |
|---|---|---|---|---|---|---|---|
| `c_0` in0 | `in0_CB_size` | `all_cores` | `in0_data_format` | `in0_aligned_tile_size` | in0 | always | — |
| `c_1` in1 | `in1_CB_size` | `all_cores` | `in1_data_format` | `in1_aligned_tile_size` | in1 | always | `in1_tensor` iff `in1_is_sharded` |
| `c_2` in0 sharded | `in2_CB_size` | `all_cores` | `in0_data_format` | `in0_single_tile_size` | in0 | `in0_is_sharded` | `in0_tensor` |
| `c_6` scratch | `32*2` = 64 B | `all_cores` | `Float16_b` | 64 | — | `in0_is_sharded` | — (no tensor) |
| `c_4` out | `out_CB_size` | `all_cores` | `output_data_format` | `output_single_tile_size` | output | always | `out_tensor` iff `output_is_sharded` |
| `c_5` interm0 | `interm0_CB_size` | `all_cores` | `interm0_data_format` | `interm0_single_tile_size` | output | always | — |
| `c_7` interm0 alias | (shares `c_5`'s descriptor) | `all_cores` | `interm0_data_format` | `interm0_single_tile_size` | output | `bias_reload_alias` | — |
| `c_3` bias | `in3_CB_size` | `all_cores` | `bias_data_format` | `bias_aligned_tile_size` | bias | `bias_tensor.has_value()` | `*bias_tensor` iff `bias_is_sharded` |
| `c_10` in0 transposed | `in0_CB_size` | `all_cores` | `in0_data_format` | `in0_aligned_tile_size` | in0 | `in0_transpose_tile` | — |

**Sharing axis.** `c_4`/`c_5` (+`c_7`) sit on **one** `CBDescriptor` — i.e. share one L1 region —
unless `do_not_inplace_interm0_out_CB || interm0_data_format != output_data_format ||
(untilize_out && in1_num_subblocks > 1)`, in which case they are two separate descriptors and
`c_7` (when present) aliases only `c_5`.

`mcast_in1` (`:4817`-`:4953`) is the same set with three differences: `c_0` is borrowed from
`in0_tensor` when `in0_is_sharded && !extract_shard_sub_blocks`; `c_2` exists only when
`in0_is_sharded && extract_shard_sub_blocks`; there is **no** `c_6` scratch, and `c_1`/`c_3` are
never borrowed.

### Semaphores

Both paths declare exactly two, ids `0` and `1`, `core_ranges = all_cores`, `initial_value = INVALID`.

| id | path | role |
|---|---|---|
| 0 | mcast_in0 / mcast_in1 | `in0_mcast_sender` / `in1_mcast_sender` |
| 1 | mcast_in0 / mcast_in1 | `in0_mcast_receiver` / `in1_mcast_receiver` |

### Tensor accessors

| host site | originating Tensor | RTA slot (host) |
|---|---|---|
| `:3489` / `:4489` `TensorAccessorArgs(in0_tensor)` | input 0 (`a`) | in0_sender slot 0 (rebound at `:4105` / `:5132`) |
| `:3547` / `:4548` `TensorAccessorArgs(in1_tensor)` | input 1 (`b`) | in1_sender_writer slot 0 (rebound `:4188` / `:5057`) |
| `:3549` / `:4550` `TensorAccessorArgs(out_tensor)` | output | in1_sender_writer slot 7 (rebound `:4189` / `:5058`) |
| `:4591` `TensorAccessorArgs(out_tensor)` | output | in1_receiver_writer slot 2 (rebound `:5109`) |
| `:3551` / `:4552` `TensorAccessorArgs(*bias_tensor)` | optional bias | in1_sender_writer slot 18 (rebound `:4191` / `:5060`) |
| `:3490` / `:4490` / `:4549` `TensorAccessorArgs()` | **none — placeholder** | n/a (sparsity disabled) |

All four real bindings are **Case 1** (consumed through a `TensorAccessor`); no Case 2, so the
`get_bank_base_address` bridge is not needed anywhere. Every accessor construction is **2-arg** —
the third-argument (page-size) subject never fires.

### Work split

Not a `split_work_to_cores` op. Core sets are built with
`num_cores_to_corerangeset_in_subcoregrids(start_core, n, matmul_core_rect, row_major=true)`:

- `num_blocks_y = ceil(M / per_core_M)`, `num_blocks_x = ceil(N / per_core_N)`,
  `num_cores_with_work = num_blocks_y * num_blocks_x`.
- **mcast_in0**: `num_blocks_y == 1` is enforced (`:3301`).
  `num_cores = in0_is_sharded ? max(num_cores_with_work, in0_sender_num_cores) : num_cores_with_work`.
  Four disjoint core sets: `…with_work_and_in_receiver_grid`, `…without_work_and_in_receiver_grid`,
  `…without_work_and_not_in_receiver_grid`, `in0_mcast_receivers`.
- **mcast_in1**: `num_blocks_x == 1` is enforced (`:4380`). `in1_mcast_sender` = `start_core` only;
  `in1_mcast_receivers` = the other `num_cores - 1`.

### Shared kernels

Census via `grep -rl <filename> ttnn/cpp/ttnn/operations/` with build-file, comment-only and
`experimental/quasar/**` hits discarded (quasar `_metal2` files explicitly do **not** count as forks).

| kernel | other binding factories | `_metal2` fork beside it? | rung |
|---|---|---|---|
| `compute/bmm_large_block_zm_fused_bias_activation.cpp` | Optimized, McastDRAMSharded, BatchedHS, Mcast2D, Sparse, **MeshWorkload sibling** | **yes** (`…_metal2.cpp`, added by #55961) | **1 — reuse** |
| `dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp` | Mcast2D, Sparse, **MeshWorkload sibling** | no | **2 — create** |
| `dataflow/…_in0_sender_receiver_padding_block_sharded.cpp` | Mcast2D, **MeshWorkload sibling** | no | **2 — create** |
| `dataflow/reader_bmm_tile_layout_in0_receiver.cpp` | Mcast2D, Sparse, **MeshWorkload sibling** | no | **2 — create** |
| `dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp` | Mcast2D, Sparse, **MeshWorkload sibling** | no | **2 — create** |
| `dataflow/reader_bmm_tile_layout_in1_receiver_writer_padding.cpp` | Mcast2D, **MeshWorkload sibling** | no | **2 — create** |

Every one is **intra-op** or **lent**: all six live in matmul's own directory, so the five new forks
land beside their originals in `device/kernels/{dataflow,compute}/`, and the pointer comment goes
into each legacy original. No peer-op directory is written.

**Rung-1 fit check — the compute fork is a clean match.** Its binding vocabulary is
`dfb::{in0, in1, bias, out, intermed0, intermed0_reload_alias, in0_transposed}` — exactly the seven
CB indices mcast-1d configures (`c_0`, `c_1`, `c_3`, `c_4`, `c_5`, `c_7`, `c_10`). Its `args::` set
covers every positional CTA mcast-1d passes; the three args mcast-1d does *not* pass
(`is_worker_core`, `last_subblock_w_valid`) are behind `#ifdef MATMUL_DRAM_SHARDED`, which
mcast-1d never defines, and `bias_ntiles`/`row_broadcast_bias` / `activation_*` sit behind
`FUSE_BIAS` / `SFPU_ACTIVATION`, matching mcast-1d's own conditional pushes. The legacy
`MM_PARTIALS_RELOAD_ALIAS_CB` *value* define is already re-expressed there as the flag
`MM_PARTIALS_RELOAD_ALIAS` + the `dfb::intermed0_reload_alias` binding — i.e. the fork already
implements the conditional-binding pattern this port needs. **Renaming happens on our side**: the
fork's names are the constraint.

### Flags

- **`fuse_op` is structurally dead on this path.** `create_descriptor` hardcodes
  `fused_op_signaler = std::nullopt` (`:5609`), so every `fuse_op` CTA is emitted `false` and the
  CCL signaler contributes no runtime args. It stays reachable only through the sibling factory.
- **Sparsity is structurally dead on this path.** `num_active` is emitted `0` and the sparsity
  `TensorAccessorArgs()` is a **tensor-less placeholder** (`:3490`, `:4490`, `:4549`); `batchB` and
  `sparsity_pagesize` are `0`. Only the sparse factory ever sets these.
- **Dangling CB indices.** `cb_in0_sharded = c_2` is emitted on the *interleaved* branch where no
  `c_2` CB exists; `cb_sparsity` is emitted as `c_6` (in0 sender) and `c_7` (in1 sender/writer),
  neither of which is a sparsity buffer — `c_6` is the 64-byte scratch and `c_7` the partials alias.
  These are legacy placeholders, not buffers.
- **A resolved compute-config field is dropped.** `dst_full_sync_en` is resolved at `:5558` and is
  **not** a parameter of either descriptor builder, so the descriptor default applies and the op
  ignores the knob whatever the caller passes. Preserved, not repaired — see *Applied Patterns*.
- **Unreferenced kernel files**: none in scope.

---

## TTNN ProgramFactory

- **Concept (realized)**: **`ProgramSpecFactoryConcept`** — the base concept.
- **Sheet's `Porting Target`**: `CustomProgramSpecFactoryConcept`. **The port deviates, with the
  invoker's explicit sign-off**; the reasoning is below and is routed to the sheet owner in the port
  report. It is recorded here rather than silently taken.
- **Custom `compute_program_hash`**: none (see Legacy Inventory) — nothing to leave alone.

### Why the base concept, and why that is not a loss of fidelity

The sheet's target follows the recipe's selector — *does the ported-from factory have an
`override_runtime_arguments`?* This factory's struct does (`.hpp:28`, body `:5482`), which is what
made the audit and the sheet both name the custom concept. Reading the method rather than its
presence changes the answer on two counts:

1. **It is not this factory's cache-hit hook.** Its second parameter is `shared_variables_t` —
   kernel handles, CB handles, core lists — which `create_descriptor` **never produces**. Nothing on
   the descriptor path can call it. It exists for the sibling `MatmulMeshWorkload…` factory in the
   same `.cpp` (`:5836`) and for `all_gather_matmul_async_program_factory.cpp:253`, both of which
   build that state themselves.
2. **What it refreshes is exactly what the base concept refreshes for free.** Walking
   `override_mcast_in0_program_parameters` (`:2996`-`:3071`) and
   `override_mcast_in1_program_parameters` (`:2918`-`:2994`) statement by statement, **every write is
   a tensor address** — in0 (RTA slot 0, or the `c_0`/`c_2` CB address), in1 (slot 0, or the `c_1` CB
   address), out (slot 7 / slot 2, or the `c_4` CB address), bias (slot 18, or the `c_3` CB address).
   There is **no non-tensor refresh anywhere in either helper**. The base concept's framework hit
   path refreshes the tensor bindings and nothing else — the same set, and the borrowed DFBs
   (`IN1`/`IN0_SHARDED`/`IN0`/`OUT`/`BIAS`) resolve their backing address from those same tensor
   arguments, which is precisely what the legacy `UpdateDynamicCircularBufferAddress` calls did.

So a `ProgramRunArgs`-returning translation would have had `tensor_args = {IN0, IN1, OUTPUT, [BIAS]}`
and an empty `kernel_run_args` — byte-for-byte what the framework already does. Cache-hit behaviour
is identical either way; this is not a case of dropping refreshes to reach the simpler concept.

**What the custom concept would have cost.** `CustomProgramSpecFactoryConcept` is keyed on
`&T::override_runtime_arguments` (`operation_concepts.hpp:111-115`), so an *overload* is ill-formed
and leaves the concept unsatisfied — silently. Taking the custom concept therefore means **deleting**
the void method, and repointing its two callers, one of which
(`experimental/ccl/all_gather_matmul_async/…`) is outside this op's directory and so outside the
port's scope boundary. The base concept keeps the diff inside `operations/matmul/` entirely.

The framework anticipates exactly this shape: `operation_concepts.hpp:108` notes that "the legacy
void-returning `override_runtime_arguments` (some matmul factories) doesn't match", which is why
`ProgramSpecFactoryConcept` is satisfied with that method still present. The guards inside the two
helpers are untouched, because the method they live in is untouched.

---

## Planned Spec Shape

Built per path inside `create_program_artifacts`, which keeps `create_descriptor`'s dispatch
(`mcast_in0 ? … : …`) and its `TT_FATAL(!gather_in0, …)`.

### Resource names (function-local — unity-build hygiene)

The six matmul factory `.cpp`s share one unity-build target, so all spec-name constants are declared
**function-local**, matching `matmul_multicore_program_factory.cpp:107-119`.

`DFBSpecName`: `IN0_DFB "in0"`, `IN1_DFB "in1"`, `IN0_SHARDED_DFB "in0_sharded"`,
`SCRATCH_DFB "l1_array"`, `OUT_DFB "out"`, `INTERM0_DFB "intermed0"`,
`INTERM0_ALIAS_DFB "intermed0_reload_alias"`, `BIAS_DFB "bias"`, `IN0_TRANSPOSED_DFB "in0_transposed"`.

`TensorParamName`: `IN0 "in0"`, `IN1 "in1"`, `OUTPUT "output"`, `BIAS "bias"`.

`SemaphoreSpecName`: `MCAST_SENDER_SEM "mcast_sender"`, `MCAST_RECEIVER_SEM "mcast_receiver"`.

### mcast_in0

- **KernelSpecs** (up to 6): `IN0_SENDER`, `IN0_NO_WORK_IN_RECV`?, `IN0_NO_WORK_NOT_IN_RECV`?,
  `IN0_RECEIVER`?, `IN1_SENDER_WRITER`, `COMPUTE`.
- **DataflowBufferSpecs**: `IN0_DFB`, `IN1_DFB`, `IN0_SHARDED_DFB`?, `SCRATCH_DFB`?, `OUT_DFB`,
  `INTERM0_DFB`, `INTERM0_ALIAS_DFB`?, `BIAS_DFB`?, `IN0_TRANSPOSED_DFB`?.
- **SemaphoreSpecs**: `MCAST_SENDER_SEM`, `MCAST_RECEIVER_SEM`, both over `all_cores`, `INVALID`.
- **TensorParameters**: `IN0`, `IN1`, `OUTPUT`, `BIAS`?.
- **WorkUnitSpecs**: one per disjoint core set —
  `wu_with_work_in_recv` {IN0_SENDER, IN1_SENDER_WRITER, COMPUTE},
  `wu_no_work_in_recv` {IN0_NO_WORK_IN_RECV}, `wu_no_work_not_in_recv` {IN0_NO_WORK_NOT_IN_RECV},
  `wu_in0_receivers` {IN0_RECEIVER, IN1_SENDER_WRITER, COMPUTE}.

### mcast_in1

- **KernelSpecs** (up to 4): `IN0_SENDER`, `IN1_SENDER_WRITER`, `IN1_RECEIVER_WRITER`?, `COMPUTE`.
- **DataflowBufferSpecs**: `IN0_DFB`, `IN1_DFB`, `IN0_SHARDED_DFB`?, `OUT_DFB`, `INTERM0_DFB`,
  `INTERM0_ALIAS_DFB`?, `BIAS_DFB`?, `IN0_TRANSPOSED_DFB`?. (No `SCRATCH_DFB`.)
- **SemaphoreSpecs / TensorParameters**: as above.
- **WorkUnitSpecs**: `wu_sender` {IN0_SENDER, IN1_SENDER_WRITER, COMPUTE} over `in1_mcast_sender`;
  `wu_receivers` {IN0_SENDER, IN1_RECEIVER_WRITER, COMPUTE} over `in1_mcast_receivers`.

### DFB endpoint census (re-derived, not transcribed)

Per node, counting **distinct kernels** that touch each buffer and tagging FIFO-locked vs role-free:

| DFB | mcast_in0 touchers | mcast_in1 touchers | disposition |
|---|---|---|---|
| `IN0_DFB` | in0 sender/receiver **produces**; compute **consumes** | in0 sender produces; compute consumes | plain 1P+1C |
| `IN1_DFB` | in1 sender/writer produces; compute consumes | in1 sender/receiver-writer produces; compute consumes | plain 1P+1C |
| `IN0_SHARDED_DFB` | in0 sender only (raw peek, borrowed) | in0 sender only | **self-loop** (1 toucher) |
| `SCRATCH_DFB` (`c_6`) | **zero** — no kernel reads `cb_l1_array` | n/a | **dead CB → drop, no spec** |
| `OUT_DFB` | compute produces; in1 sender/writer consumes | compute produces; in1 sender/receiver-writer consumes | plain 1P+1C |
| `INTERM0_DFB` | compute only (produce + reload) | compute only | **self-loop** (1 toucher) |
| `INTERM0_ALIAS_DFB` | compute only | compute only | **self-loop** (1 toucher) |
| `BIAS_DFB` | in1 sender/writer produces; compute consumes | in1 sender/receiver-writer produces; compute consumes | plain 1P+1C |
| `IN0_TRANSPOSED_DFB` | compute only (transpose in place) | compute only | **self-loop** (1 toucher) |

**`SCRATCH_DFB` (`c_6`, "Local L1 to store temp vars", `:3899`-`:3907`) is a dead CB.** The host
hands its index to the three block-sharded in0 readers as the named CTA `cb_l1_array` (`:3660`,
`:3693`, `:3713`), but `grep -rn cb_l1_array ttnn/cpp/ttnn/operations/matmul/device/kernels/`
returns **nothing** — no kernel reads it. Zero endpoints, so the port builds **no**
`DataflowBufferSpec` for it and drops the dead CTA with it; a bindingless DFB would be rejected by
the validator anyway. The 64 bytes it allocated per core have no behaviour attached, so the drop is
zero-functional-change. Same for the `cb_sparsity` CTA on that kernel, which it also never reads.

**No DFB needs `allow_instance_multi_binding`,** and no DFB is both self-looped and multi-bound.
The three same-source `KernelSpec`s of the block-sharded in0 reader sit on **disjoint** node sets, so
each node sees exactly one instance — a legal single-role binding, *not* the multi-binding flag and
*not* the same-grid two-toucher case.

**Aliasing.** `INTERM0_DFB` ↔ `INTERM0_ALIAS_DFB` form a strict clique of **two** via
`advanced_options.alias_with` when `bias_reload_alias`. When `c_4`/`c_5` additionally share one
legacy descriptor (the "share buffer" branch), `OUT_DFB` joins the clique, making it **three**
members, all naming each other, all at the same total size and bound to the same kernels. The group
size is derived per instantiation from the same two conditions the legacy code branches on.

**Borrowed memory.** `borrowed_from` is set to the matching `TensorParameter` exactly where the
legacy `CBDescriptor::tensor` was: `IN1_DFB ← IN1` (mcast_in0, `in1_is_sharded`),
`IN0_SHARDED_DFB ← IN0`, `IN0_DFB ← IN0` (mcast_in1, `in0_is_sharded && !extract_shard_sub_blocks`),
`OUT_DFB ← OUTPUT` (`output_is_sharded`), `BIAS_DFB ← BIAS` (mcast_in0, `bias_is_sharded`).

## Preserved Multiplicity

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| `in0_sender_kernel_desc`, `in0_no_work_in_receiver_kernel_desc`, `in0_no_work_not_in_receiver_kernel_desc` — all three of `…_in0_sender_receiver_padding_block_sharded_metal2.cpp`, differing only on CTA slots 0/1 (`core_has_output_block_work`, `core_in_in0_receiver_mcast_grid`) | `IN0_SENDER`, `IN0_NO_WORK_IN_RECV`, `IN0_NO_WORK_NOT_IN_RECV` | `wu_with_work_in_recv`, `wu_no_work_in_recv`, `wu_no_work_not_in_recv` (disjoint) | `IN0_DFB` PRODUCER, `IN0_SHARDED_DFB` self-loop, `SCRATCH_DFB` self-loop |

The differing slots stay **CTAs** (they are `if constexpr` gates in the kernel); demoting them to
RTAs would be the *Demoting per-group CTA to RTA* anti-pattern.

## Dropped Plumbing

| legacy location | legacy form | Metal 2.0 replacement |
|---|---|---|
| in0_sender RTA slot 0 (`:4083`, `:5115`) | `in0_tensor.address()` placeholder, variant-rebound | `TensorBinding{IN0, "in0"}` |
| in1_sender_writer RTA slot 0 (`:4122`, `:5013`) | `in1_tensor.address()` | `TensorBinding{IN1, "in1"}` |
| in1_sender_writer RTA slot 7 (`:4135`, `:5026`) | `out_tensor.address()` | `TensorBinding{OUTPUT, "out"}` |
| in1_sender_writer RTA slot 18 (`:4170`, `:5043`) | `bias_tensor->address()` (`// smuggled-rta-ok`) | `TensorBinding{BIAS, "bias"}` (conditional) |
| in1_receiver_writer RTA slot 2 (`:5076`) | `out_tensor.address()` | `TensorBinding{OUTPUT, "out"}` |
| `:3489`/`:3547`/`:3549`/`:3551` and mcast_in1 equivalents | `TensorAccessorArgs(t).append_to(cta)` + kernel `TensorAccessorArgs<N>()` chain | binding mechanism end-to-end |
| `:3490`, `:4490`, `:4549` | `TensorAccessorArgs()` **tensor-less placeholder** for sparsity | **dropped** — no `TensorParameter`; kernel path `#ifdef`-gated off |
| named CTA `cb_in0`, `cb_in1`, `cb_bias`, `cb_out`, `cb_intermed0`, `cb_in0_transposed`, `cb_in0_sharded`, `cb_l1_array` | CB index as a named CTA value | `DFBBinding` + `dfb::<name>` |
| named CTA `cb_sparsity` (`c_6` / `c_7`) | dangling CB index, never a buffer | **dropped** with the sparsity path |
| compute define `MM_PARTIALS_RELOAD_ALIAS_CB=<index>` (`:3803`, `:4765`) | CB **index** carried through a define | `MM_PARTIALS_RELOAD_ALIAS` flag define + `DFBBinding` on `INTERM0_ALIAS_DFB` (already the fork's shape) |
| in0_sender CTA 15/16, in0_receiver CTA 4/5, in1 CTA 10/11 | semaphore **ids** as positional CTAs, `get_semaphore(id)` kernel-side | `SemaphoreBinding` + `sem::<name>`; the raw-pointer read becomes `Semaphore::value()` |
| every positional CTA on all six kernels | `get_compile_time_arg_val(N)` | named CTAs, `get_arg(args::<name>)` |
| every positional RTA on all six kernels | `get_arg_val<uint32_t>(rt_args_idx++)` | named RTAs, `get_arg(args::<name>)` |

**No varargs.** Every argument on every kernel here is a **distinct field read a fixed number of
times**, including the `rt_args_idx++` runs at the top of the in0/in1 readers — a running counter is
not a vararg signal. No kernel reads an argument in a loop or at a data-computed index. The one
apparent exception, the block-sharded reader's `in0_mcast_noc_x` / `in0_mcast_noc_y` **lists**
(`:4064`-`:4065`), *is* a genuine indexed collection with a CTA-derived count — see *Deferred /
Flagged*.

## Applied Patterns

- **[Caution: Porting a shared kernel]** — rung 1 (reuse) for the compute kernel; rung 2 (create)
  for all five dataflow kernels, forks beside their originals plus the pointer comment in each.
- **[Pattern: Aliased DFBs]** — `INTERM0_DFB`/`INTERM0_ALIAS_DFB` (+`OUT_DFB` in the shared-buffer
  branch) as a strict `alias_with` clique, size derived per instantiation.
- **[Pattern: Sync-free and single-ended CBs → self-loop DFB]** — `SCRATCH_DFB`,
  `IN0_SHARDED_DFB`, `INTERM0_DFB`, `INTERM0_ALIAS_DFB`, `IN0_TRANSPOSED_DFB`.
- **[Pattern: Conditional / optional resource bindings]** — `BIAS_DFB`+`BIAS`, `INTERM0_ALIAS_DFB`,
  `IN0_SHARDED_DFB`, `SCRATCH_DFB`, `IN0_TRANSPOSED_DFB`, and the dropped sparsity tensor; each gets
  a matching `compiler_options.defines` flag and `#ifdef`-gated kernel-side alias.
- **[Pattern: Multi-variant factories]** — `mcast_in0` / `mcast_in1` branch inside
  `create_program_artifacts`.
- **[Pattern: Unity-build hygiene]** — all spec-name constants function-local.
- **[Pattern: Removing pybound legacy factory entry points]** — `matmul_nanobind.cpp:1240-1254`.
- **[Anti-pattern avoided: Demoting per-group CTA to RTA]** — the three block-sharded instances keep
  their per-instance CTAs.

### Hardware configuration and compiler options

- **DM kernels — custom, replicate verbatim.** The NOCs come from the factory's own
  `preferred_noc_for_dram_write` / `preferred_noc_for_dram_read`, **not** the reader/writer defaults,
  so each DM `KernelSpec` gets an explicit
  `DataMovementGen1Config{.processor = RISCV_1|RISCV_0, .noc = in0_noc|in1_noc}`. Reaching for
  `create_reader_datamovement_config` / `create_writer_datamovement_config` would substitute the
  default triple and regress silently.
- **Compute — Style A.** The op resolves a TTNN `ComputeKernelConfig`, so
  `to_compute_hardware_config(device->arch(), compute_kernel_config)`, then:
  - **the dropped `dst_full_sync_en` is pinned back to the legacy-default result**:
    `double_buffer_dest = true` (legacy default `dst_full_sync_en = false`). This preserves the
    ignored-knob behaviour; it is **reported, not repaired**.
  - `unpack_modes` reindexed from the legacy `vector<UnpackToDestMode>` (by CB id) to
    `Table<DFBSpecName, UnpackMode>` (by name): the single marked entry becomes
    `{INTERM0_ALIAS_DFB or INTERM0_DFB, UnpackMode::UnpackToDest}` when
    `fp32_dest_acc_en && interm0_data_format == Float32`, gated on the same condition as the
    binding. `Default` entries are expressed by **omission** (`UnpackToSrc`).
  - **The newly-required explicit entry fires here**: `interm0_data_format` becomes `Float32`
    exactly when `fp32_dest_acc_en` is set, and the compute kernel consumes that DFB with
    `enable_32_bit_dest` on — so the entry is mandatory, and its value is derived from the legacy
    vector rather than guessed.
- **`opt_level`** — `O3` set explicitly on **every** compute `KernelSpec` (legacy
  `ComputeConfigDescriptor` resolves to `O3`; Metal 2.0 `CompilerOptions` defaults to `O2`). DM
  kernels set none and keep Metal 2.0's `O2`, matching their legacy resolved default.

## Deferred / Flagged

- **The in0 mcast NOC-coordinate lists are the one genuine vararg.** In the in0-sharded branch the
  sender's RTAs carry `in0_mcast_noc_x` (length `in0_mcast_sender_cores_grid.x`) and
  `in0_mcast_noc_y` (length `…grid.y`) appended as blocks (`:4064`-`:4065`), and the kernel reads
  them inside loops bounded by CTA-supplied counts. That is an *indexed-collection element with a
  non-literal count* — the definition of a vararg. They stay `advanced_options.num_runtime_varargs`
  and are reported as retained varargs. Every other argument is named.
- **The Metal 2.0 fork of the in1 sender/writer will not carry the `ENABLE_GLOBAL_CB` path.** The
  legacy kernel's global-CB ("remote CB") code — `#include "api/remote_circular_buffer.h"`,
  `remote_cb_id = c_31`, `remote_cb_wait_front` / `remote_cb_pop_front` /
  `update_remote_cb_config_in_l1` — is entirely behind `#ifdef ENABLE_GLOBAL_CB`, which this factory
  never defines (factory selection routes GCB configs away from it,
  `matmul_device_operation.cpp:2204`). A GlobalCircularBuffer has no Metal 2.0 analog —
  `GlobalDataflowBuffer` is not implemented — so the fork omits that region rather than carrying a
  legacy CB index into a Metal 2.0 kernel. Zero behaviour change for any consumer of the fork; the
  legacy original keeps serving GCB consumers unchanged. Recorded as a Handoff point.
- **The fused-op (CCL) path is likewise not carried into the forks.** `MatmulOpReceiver` /
  `push_matmul_fused_op_rt_args` live in `ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp`
  — **outside the op directory** — and consume **positional** runtime args through a
  `rt_args_idx` reference. That cannot be expressed against named args without editing an out-of-op
  header, which the scope boundary forbids. `fuse_op` is hardcoded `false` on this factory's
  descriptor path (`:5609`), so the region is unreachable here; the forks `#ifdef FUSE_OP`-gate it
  and the port reports the constraint for whoever ports a CCL-fused consumer.
- **`create_descriptor`'s `core_range_set` parameter is dropped** — accepted and ignored
  (`/*core_range_set*/` at `:5496`); the fixed `create_program_artifacts` signature cannot carry it,
  and nothing reads it. Device-op-class exception 2.
