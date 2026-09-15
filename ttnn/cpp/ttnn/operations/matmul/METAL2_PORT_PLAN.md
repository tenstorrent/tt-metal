# Port Plan — `ttnn/cpp/ttnn/operations/matmul`

Port plan for `MatmulMultiCoreReuseMcast2DProgramFactory`, ported from `ProgramDescriptor` to
Metal 2.0. Written during the inventory and planning steps; committed alongside the port for review.

**Port scope: ONE factory.** The op has eight factories; only this one is in scope. The other
seven keep their current concepts and keep building.

**Scope inside the file.** `matmul_multicore_reuse_mcast_2d_program_factory.cpp` holds two program
builders. Only `create_program_mcast_in0_in1_descriptor` (39–1568) is this factory's dispatch path
and the scope of this port. `create_program_mcast_in0_in1` (1571–3054) is the legacy `Program`
builder reached only from `matmul_multi_core_reuse_mcast_2d_optimized_helper`, which only the two
CCL fused ops call; it is untouched.

## Legacy Inventory

### Legacy factory shape
- Concept: `ProgramDescriptorFactoryConcept` — `create_descriptor` returning a `ProgramDescriptor`
  (`…mcast_2d_program_factory.hpp:41`), plus a void-returning `override_runtime_arguments`
  (`…hpp:34`, defined at `.cpp:3382`).
- Variants: single (one factory; `MatmulDeviceOperation::program_factory_t` holds eight
  alternatives but this port converts one).
- Custom `compute_program_hash`: **none** framework-visible. A deliberately differently-named
  `compute_descriptor_program_hash` sits at `device/matmul_device_operation.hpp:50`; it is not the
  framework's hash hook and is **left alone**.

### Kernels

Up to eight `KernelDescriptor`s, built conditionally. `opt_level` is unset on every one, so it
resolves to the legacy per-kernel-type default (`O2` DM, **`O3` compute**).

| unique_id | source | core_ranges | CTAs (positional) | CTAs (named) | RTAs | defines | opt_level | config |
|---|---|---|---|---|---|---|---|---|
| `in0_sender` (block-sharded) | `…in0_sender_receiver_padding_block_sharded.cpp` | `all_cores_with_work` | 22 (`:396-423`) + fuse flag | `cb_in0`, `cb_in0_sharded`, `cb_l1_array` | sender_id, 4 mcast coords, noc_x[] / noc_y[] lists | — | O2 | `RISCV_1 / in0_noc` |
| `in0_mcast_no_work` | same source | `in0_mcast_cores_without_work_and_not_in_receiver_grid` | same, `[0]=[1]=0` | same | same | — | O2 | `RISCV_1 / in0_noc` |
| `in0_sender` (interleaved) | `…in0_sender_padding.cpp` | `in0_sender_interleaved` | 28 (`:425-461`) + fuse + 2×TensorAccessorArgs + num_batch_compute | `cb_in0`, `cb_in0_sharded`, `cb_sparsity`, `num_active` | addr, start_tile_id, 4 mcast coords, last_block_h, sparsity_addr | `SKIP_MCAST`?, `IN0_SHARDED`? | O2 | `RISCV_1 / in0_noc` |
| `in1_sender_writer` | `…in1_sender_writer_padding.cpp` | `in1_sender` | 33 (`:468-538`) + 4×TensorAccessorArgs (+2 dram-sharded) | `cb_in1`, `cb_bias`, `cb_out`, `cb_sparsity`, `num_active` | see below | `FUSE_BIAS`?, `SKIP_MCAST`?, `IN1_*SHARDED`?, `OUT_SHARDED`? | O2 | `RISCV_0 / in1_noc` |
| `in1_receiver_writer` | `…in1_receiver_writer_padding.cpp` | `in1_receiver` | 19 (`:553-587`) + TensorAccessorArgs | `cb_in1`, `cb_bias`, `cb_out` | see below | `FUSE_BIAS`?, `OUT_SHARDED`? | O2 | `RISCV_0 / in1_noc` |
| `in0_receiver` | `…in0_receiver.cpp` | `in0_receiver_interleaved` | 8 (`:539-552`) | `cb_in0` | sender noc x, y | — | O2 | `RISCV_1 / in0_noc` |
| `in1_receiver_writer_other` | `…in1_receiver_writer_padding.cpp` | `…_other_cores` | same as `in1_receiver_writer` | same | same | same | O2 | `RISCV_0 / **in1_split_noc**` |
| `in0_receiver_other` | `…in0_receiver.cpp` | `…_other_cores` | same as `in0_receiver` | same | same | — | O2 | `RISCV_1 / **in0_split_noc**` |
| `compute` | `…compute/bmm_large_block_zm_fused_bias_activation.cpp` | `all_cores_with_work` | 18 (+1 bias) (`:873-899`) | `cb_in0/in1/bias/out/intermed0/cb_in0_transposed`, `bias_ntiles`, 4 activation params | — | `FUSE_BIAS`?, `PACK_RELU`/`SFPU_ACTIVATION`?, `PACKER_L1_ACC`?, `FP32_DEST_ACC_EN`?, `IN1_TRANSPOSE_TILE`?, `MM_PARTIALS_RELOAD_ALIAS_CB`?, stagger/throttle | **O3** | `ComputeConfigDescriptor` |

NOC values (`:686-689`) are **custom**, not the reader/writer defaults:
`in0_noc = preferred_noc_for_dram_write(arch)`, `in1_noc = preferred_noc_for_dram_read(arch)`,
`in0_split_noc = preferred_noc_for_dram_read(arch)`, `in1_split_noc = preferred_noc_for_dram_write(arch)`.

### CBs

| index | total_size | core_ranges | data_format | page_size | tile | tensor (borrowed) | condition |
|---|---|---|---|---|---|---|---|
| `c_0` in0 | `in0_CB_size` | `all_cores` | `in0_data_format` | `in0_aligned_tile_size` | `in0_tile` | `in0_tensor` **iff `in0_height_sharded`** | always |
| `c_1` in1 | `in1_CB_size` | `all_cores` | `in1_data_format` | `in1_aligned_tile_size` | `in1_tile` | `in1_tensor` iff `in1_is_sharded && !in1_is_dram` | always |
| `c_2` in0_sharded | `in2_CB_size` | `all_cores` | `in0_data_format` | `in0_single_tile_size` | `in0_tile` | `in0_tensor` | iff `in0_block_sharded` |
| `c_6` l1_array | 64 | `all_cores` | `Float16_b` | 64 | — | — | iff `in0_block_sharded` |
| `c_4` out | `out_CB_size` | `all_cores` | `output_data_format` | `output_single_tile_size` | `output_tile` | `out_tensor` iff `output_is_sharded` | always |
| `c_5` intermed0 | `interm0_CB_size` | `all_cores` | `interm0_data_format` | `interm0_single_tile_size` | `output_tile` | — | always |
| `c_7` intermed0 alias | — | — | `interm0_data_format` | `interm0_single_tile_size` | `output_tile` | — | iff `bias_reload_alias` |
| `c_3` bias | `in3_CB_size` | `all_cores` | `bias_data_format` | `bias_aligned_tile_size` | `bias_tile` | — | iff bias |
| `c_10` in0_transposed | `in0_CB_size` | `all_cores` | `in0_data_format` | `in0_aligned_tile_size` | `in0_tile` | — | iff `in0_transpose_tile` |

`c_4` / `c_5` (+ `c_7`) are **one `CBDescriptor` with multiple `format_descriptors`** in the
shared-buffer branch (`:1069-1096`); separate descriptors in the non-shared branch (`:1030-1068`),
where `c_5` may still carry the `c_7` alias.

### Semaphores

| id | core_type | core_ranges | initial_value |
|---|---|---|---|
| 0 in0_mcast_sender | WORKER | `all_cores` | `INVALID` |
| 1 in0_mcast_receiver | WORKER | `all_cores` | `INVALID` |
| 2 in1_mcast_sender | WORKER | `all_cores` | `INVALID` |
| 3 in1_mcast_receiver | WORKER | `all_cores` | `INVALID` |

### Tensor accessors

| host site | originating Tensor | RTA slot (host) |
|---|---|---|
| `:464` `TensorAccessorArgs(in0_tensor)` | in0 | in0_sender RTA 0 (`:1248`, rebound `:1273`) |
| `:465` empty placeholder | *(sparsity — none)* | in0_sender RTA 7 (constant `0`) |
| `:522` `TensorAccessorArgs(in1_tensor)` | in1 | in1_sender RTA 0 (`:1300`, rebound `:1439`) |
| `:523` empty placeholder | *(sparsity — none)* | in1_sender RTA 6 (constant `0`) |
| `:524` `TensorAccessorArgs(out_tensor)` | out | in1_sender RTA 7 (`:1313`, rebound `:1440`) |
| `:526` `TensorAccessorArgs(*bias_mesh)` | bias | in1_sender RTA 18 (`:1349`, rebound `:1442`) |
| `:588` `TensorAccessorArgs(out_tensor)` | out | in1_receiver RTA 2 (`:1457`, rebound `:1530`) |

All **Case 1** (consumed through `TensorAccessor`). No Case 2, no third page-size argument, no
host-folded offset.

### Work split
n/a — this factory does not use `split_work_to_cores`. Placement is a 2D mcast geometry derived
from `num_blocks_x` / `num_blocks_y`, `transpose_mcast`, and the `split_half` receiver split.

### Shared kernels

All six are shared and **none has a `_metal2` fork** (`find` over `matmul/device/kernels/` returned
zero `*_metal2*` dataflow files; the only `_metal2` files are the compute fork below and two
private DRAM-sharded kernels). Rung for each row is therefore rung 2 — **create**, except the
compute kernel which is rung 1 — **reuse**.

| kernel | remaining consumers | rung |
|---|---|---|
| `reader_bmm_tile_layout_in0_sender_padding.cpp` | mcast_1d file (2 factories), sparse device-op, sparse factory | 2 — create |
| `…in0_sender_receiver_padding_block_sharded.cpp` | mcast_1d file (2 factories) | 2 — create |
| `reader_bmm_tile_layout_in0_receiver.cpp` | mcast_1d file (2 factories), sparse factory | 2 — create |
| `reader_bmm_tile_layout_in1_sender_writer_padding.cpp` | mcast_1d file (2 factories), sparse factory | 2 — create |
| `reader_bmm_tile_layout_in1_receiver_writer_padding.cpp` | mcast_1d file (2 factories) | 2 — create |
| `compute/bmm_large_block_zm_fused_bias_activation.cpp` | BatchedHS, Optimized, McastDRAMSharded, sparse, mcast_1d file | **1 — reuse** `…_metal2.cpp` |

**Reused fork's vocabulary (now a constraint, not a choice):**
`dfb::in0`, `dfb::in1`, `dfb::bias`, `dfb::out`, `dfb::intermed0`, `dfb::in0_transposed`,
`dfb::intermed0_reload_alias`; named args `in0_block_w`, `in0_num_subblocks`,
`in0_block_num_tiles`, `in0_subblock_num_tiles`, `in1_num_subblocks`, `in1_block_num_tiles`,
`in1_block_w`, `num_blocks_inner_dim`, `num_blocks_w_dim`, `num_blocks_h_dim`, `out_subblock_h`,
`out_subblock_w`, `out_subblock_num_tiles`, `batch`, `out_block_num_tiles`, `untilize_out`,
`get_batch_from_reader`, `is_worker_core`, `last_subblock_w_valid`, `bias_ntiles`,
`row_broadcast_bias`, `activation_type`, `activation_param0..2`; defines `FUSE_BIAS`,
`SFPU_ACTIVATION`, `PACK_RELU`, `PACKER_L1_ACC`, `FP32_DEST_ACC_EN`, `IN0_TRANSPOSE_TILE`,
`IN1_TRANSPOSE_TILE`, `MM_PARTIALS_RELOAD_ALIAS`, `MATMUL_DRAM_SHARDED`, `SKIP_COMPUTE`.

Two vocabulary shifts the reuse forces on this factory:
- `MM_PARTIALS_RELOAD_ALIAS_CB=<index>` (a value-carrying define) becomes the valueless
  `MM_PARTIALS_RELOAD_ALIAS` plus a `dfb::intermed0_reload_alias` binding.
- `in0_transpose_tile` moves from a positional CTA to the `IN0_TRANSPOSE_TILE` define, because the
  fork gates the `dfb::in0_transposed` binding on it.

### Flags

- **`c_6` (`cb_l1_array`) is a dead CB.** `grep -rn l1_array ttnn/cpp/ttnn/operations/matmul/`
  returns only host-side `named_compile_time_args` entries and the legacy builder's
  `CreateCircularBuffer`; **no kernel source reads it**. Zero endpoints → dead-CB drop
  (allocation `:1020-1027`, named CTAs `:733` and `:754`). This answers the audit's open census
  question on `c_6`.
- **`cb_sparsity` is dead on this factory's paths.** The in0 interleaved sender is handed
  `c_6` (a CB that only exists when block-sharded — i.e. never on that kernel's path) and the in1
  sender is handed `c_7` (which exists only as the intermed0 alias). Both kernels construct a
  sparsity `DataflowBuffer` unconditionally but touch it only under `batchB > 0 || use_indices`,
  and this factory hardcodes `batchB = 0`, `sparsity_pagesize = 0`, `num_active = 0`. No binding.
- **Fused-op (CCL) paths are statically dead here.** `create_descriptor` passes
  `fused_op_signaler = std::nullopt` (`:3485`), so `fuse_op`, `fuse_op_all_gather` and
  `fuse_op_reduce_scatter` are all `0` on every program this factory builds.
- **`ENABLE_GLOBAL_CB`** appears in `reader_bmm_tile_layout_in1_sender_writer_padding.cpp` but is
  never defined by this factory (it belongs to the tensor-prefetcher path of another factory).

## TTNN ProgramFactory

- **Concept (inherited from audit)**: `CustomProgramSpecFactoryConcept`.
- **Custom `compute_program_hash`**: none — default reflection hash. Left untouched.
- **Implementation notes**: the concept keys on `decltype(&T::override_runtime_arguments)`, which is
  ill-formed for an overload set, so the class can carry exactly one such method. Two CCL ops call
  the current void one directly. **Resolved by the invoker (decision D4):** move the legacy
  cache-hit refresh to a public free function beside the existing
  `matmul_multi_core_reuse_mcast_2d_optimized_helper`, repoint the two CCL call sites at it, and
  give the class the `ProgramRunArgs`-returning method the concept requires. CCL behaviour is
  byte-identical — same function body, same `shared_variables_t`.

## Planned Spec Shape

- **KernelSpecs**: 1:1 with the legacy `KernelDescriptor`s (up to eight, built under the same
  conditions). Both receiver NOC-setup twins are preserved as separate specs.
- **DataflowBufferSpecs**: one per surviving legacy `buffer_index` — `IN0`, `IN1`, `IN0_SHARDED`?,
  `OUT`, `INTERMED0`, `INTERMED0_RELOAD_ALIAS`?, `BIAS`?, `IN0_TRANSPOSED`?. `c_6` dropped.
  Declaration order matches the legacy CB creation order so L1 addresses land identically.
- **SemaphoreSpecs**: 4, `target_nodes = all_cores`, `advanced_options.initial_value = INVALID`.
- **TensorParameters**: `IN0`, `IN1`, `OUT`, and `BIAS` when present.
- **WorkUnitSpecs**: derived by grouping nodes by the exact set of kernels the legacy core_ranges
  place on them (see Applied Patterns). Between three and six in practice.

## Preserved Multiplicity

| legacy KernelDescriptors | same-source KernelSpecs | WorkUnitSpecs | shared DFBs (endpoint role each binds) |
|---|---|---|---|
| `in1_receiver_writer` + `in1_receiver_writer_other` of `…in1_receiver_writer_padding.cpp` | `IN1_RECEIVER_WRITER`, `IN1_RECEIVER_WRITER_OTHER` | the left-half and right-half interior WUs (disjoint) | `IN1` PRODUCER, `OUT` CONSUMER, `BIAS` PRODUCER |
| `in0_receiver` + `in0_receiver_other` of `…in0_receiver.cpp` | `IN0_RECEIVER`, `IN0_RECEIVER_OTHER` | same two WUs (disjoint) | `IN0` PRODUCER |
| `in0_sender` + `in0_mcast_no_work` of `…block_sharded.cpp` | `IN0_SENDER`, `IN0_MCAST_NO_WORK` | work-grid WUs and the no-work WU (disjoint) | `IN0` PRODUCER (+CONSUMER on the no-work WU), `IN0_SHARDED` self-loop |

Disjoint node sets, so each is a legal single-role binding — **not** the multi-binding flag.

## Dropped Plumbing

| legacy location (file:line) | legacy form | Metal 2.0 replacement |
|---|---|---|
| in0_sender RTA 0 (`:1248`/`:1273`) | `in0_tensor.address()` → rebound tensor ref | `TensorBinding(IN0)` |
| in1_sender RTA 0 (`:1300`/`:1439`) | `in1_tensor.address()` → rebound | `TensorBinding(IN1)` |
| in1_sender RTA 7 (`:1313`/`:1440`) | `out_tensor.address()` → rebound | `TensorBinding(OUT)` |
| in1_sender RTA 18 (`:1349`/`:1442`) | bias address → rebound | `TensorBinding(BIAS)` |
| in1_receiver RTA 2 (`:1457`/`:1530`) | `out_tensor.address()` → rebound | `TensorBinding(OUT)` |
| in0_sender RTA 7, in1_sender RTA 6 | `sparsity_addr` (constant `0`) | dropped — no sparsity parameter |
| `:464-465`, `:522-527`, `:588` | `TensorAccessorArgs(…).append_to(cta)` | binding mechanism end-to-end |
| named CTAs `cb_in0`, `cb_in1`, `cb_bias`, `cb_out`, `cb_intermed0`, `cb_in0_transposed`, `cb_in0_sharded`, `cb_sparsity`, `cb_l1_array` | CB indices carried by name | `DFBBinding`s (or dropped where dead) |
| in0_sender CTAs 9/10, in1_sender CTAs 10/11, receivers CTAs 4/5 | semaphore ids as positional CTAs | `SemaphoreBinding`s |
| `MM_PARTIALS_RELOAD_ALIAS_CB=<index>` (`:919`) | define carrying a CB index | `MM_PARTIALS_RELOAD_ALIAS` define + `dfb::intermed0_reload_alias` binding |
| every positional CTA on all six kernels | `get_compile_time_arg_val(N)` | named CTAs |
| `num_batch_compute`, `num_active`, sparsity CTAs | sparse-matmul-only plumbing | dropped on this factory's specs (kept `#ifdef`-gated in the forks) |

## Applied Patterns

- **Node-signature WorkUnitSpecs.** Rather than re-deriving the mcast geometry, each node in
  `cores` is tested against the *same* legacy `CoreRangeSet` membership predicates, the resulting
  kernel-set signature is hashed, and one `WorkUnitSpec` is emitted per distinct signature. The
  placement is therefore provably the legacy placement.
- [Self-loop DFB binding](port_patterns) — `IN0_SHARDED` (one toucher: the block-sharded in0
  sender); `IN0` on the no-work nodes (the no-work kernel is the only toucher there).
- [Two-toucher → 1P+1C] — not needed; every DFB instance has a natural producer and consumer.
- [Aliased DFBs] — `OUT` + `INTERMED0` (+ `INTERMED0_RELOAD_ALIAS`) form a strict clique **only in
  the shared-buffer branch**; in the non-shared branch `OUT` and `INTERMED0` are independent and
  only `INTERMED0` + `INTERMED0_RELOAD_ALIAS` alias. Group size derived per instantiation.
- [Conditional / optional resource bindings] — `BIAS`, `IN0_TRANSPOSED`, `IN0_SHARDED`,
  `INTERMED0_RELOAD_ALIAS`, `tensor::in0` (absent when in0 is height-sharded), `tensor::bias`,
  and the whole sparsity/fused-op families. Each host-side conditional binding is paired with a
  `compiler_options.defines` entry and an `#ifdef` gate in the fork.
- [Pass DFB handles directly to LLKs] — `get_tile_size(dfb::in0)` / `get_dataformat(dfb::in0)` keep
  the free-function form because the legacy declarations are `constexpr`.
- [Removing pybound legacy factory entry points] — `matmul_nanobind.cpp` `nb::class_` block for
  this factory.
- **Varargs, two sites (the audit said none — census disagreed and the census wins).** The
  block-sharded in0 sender reads `in0_mcast_noc_x[num_x]` / `in0_mcast_noc_y[num_y]` as
  CTA-bounded indexed collections (`get_arg_addr(increment_arg_idx(...))`, then `arr[i]` inside a
  loop). Those are genuine indexed-collection elements → `num_runtime_varargs`, matching the
  already-ported DRAM-sharded sender. Everything else in every kernel is a distinct field read
  once → named.

## Deferred / Flagged

- **Fused-op CCL helpers cannot be expressed under named args.**
  `MatmulOpReceiver` / `OpSignaler` (`ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp`,
  outside the op directory) consume runtime args positionally through a `uint32_t& rt_args_idx`.
  Converting the kernels' RTAs to named args removes the counter those helpers need. The paths are
  statically dead for this factory, so they are preserved behind `#ifdef FUSE_OP…` gates that no
  Metal 2.0 factory may define. Reported as a Handoff point.
- **`ENABLE_GLOBAL_CB`** is a GlobalCircularBuffer path in the in1 sender/writer. Not portable
  (no `GlobalDataflowBuffer`), never defined by this factory, preserved behind its existing
  `#ifdef`. Reported.
- **Dead placeholder `.address()` calls** at `:1248`, `:1300`, `:1313`, `:1457` disappear with the
  bindings; nothing to carry.
