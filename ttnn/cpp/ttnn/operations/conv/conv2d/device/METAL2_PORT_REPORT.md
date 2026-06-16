# Metal 2.0 Port Report — conv2d `Conv2dWidthShardedProgramFactory` (L1 config path)

## Summary

- **Op / factory**: `ttnn::prim::Conv2dWidthShardedProgramFactory` in
  `ttnn/cpp/ttnn/operations/conv/conv2d`.
- **Scope ported**: the **L1 config-tensor path only** (`!config_tensors_in_dram`).
  The in-DRAM config path stays blocked (see Blockers) and is excluded from the spec build.
- **Concept transition**: `MeshWorkloadFactoryConcept` (`create_workload_descriptor`
  returning `tt::tt_metal::WorkloadDescriptor`) → `ProgramSpecFactoryConcept`
  (`create_program_spec` returning `ttnn::device_operation::ProgramArtifacts`).
- **Custom `compute_program_hash`**: KEPT (shared across both factories of the
  device-op variant; the sharded factory is still on the legacy concept). Per the
  bulk-port instruction, this hash is preserved (NOT deleted).

## Factory chosen

`Conv2dWidthShardedProgramFactory` — the single-program width-sharded factory.
Selected by `select_program_factory` when the activation memory layout is
`WIDTH_SHARDED`. The sister `Conv2dShardedProgramFactory` stays on its current
(WorkloadDescriptor) concept; the variant dispatches per-factory at runtime.

## Kernels

| kernel | role | disposition |
|---|---|---|
| `device/kernels/conv_bmm_tilize.cpp` | compute | **FORKED** → `conv_bmm_tilize_m2.cpp` (shared with `Conv2dShardedProgramFactory`) |
| `device/kernels/activation_reader_width_sharded.cpp` | act reader (RISCV_0) | ported in place (only this factory uses it) |
| `device/kernels/weights_reader_width_sharded.cpp` | weights/bias reader (RISCV_1) | ported in place (only this factory uses it) |

`conv_reader_common.hpp` (shared helper header, in-op) is referenced by the act
reader; its `load_config_tensor_if_in_dram` body is entirely under
`#ifdef CONFIG_TENSOR_IN_DRAM`, never defined on the L1 spec, so it compiles to a
no-op and pulls in none of the DRAM-only CTAs.

## DFB ↔ CB map (width-sharded L1 path)

One `DataflowBufferSpec` per non-zero-page `CBInfo`. Globally-allocated CBs become
**borrowed-memory DFBs** (`borrowed_from` the backing `TensorParameter`):

| Conv2dCb | borrowed_from | producer | consumer |
|---|---|---|---|
| ACT_SHARDED | input `a` | (reader source only) self-loop on act reader |
| OUT | output | compute | weights? no — writer is act/compute; OUT produced by compute |
| MATMUL_PARTIALS | output (iff globally-alloc) OR local | compute self-loop |
| READER_INDICES | **op-owned** conv_reader_indices | self-loop on act reader |
| WEIGHTS | — (local) | weights reader → compute |
| ACT | — (local) | act reader → compute |
| ACT_ROW_MAJOR_BFLOAT16 | — (local) | act reader → compute(tilize) |
| ACT_TILIZED | — (local) | act reader (mcast) ↔ compute |
| BIAS | — (local, conditional) | weights reader → compute |
| L1_ARRAY | — (local scratchpad) | act reader self-loop |
| ACT_SECOND_READER | — (always 0-page in width-sharded → omitted) | n/a |

(See "Open items" for the self-loop / borrowed-DFB endpoint subtleties — conv's
CB usage does not map cleanly onto strict producer/consumer FIFO semantics.)

## Op-owned tensor

`conv_reader_indices` — the host-populated index table from
`sliding_window::move_config_tensor_to_device(...)`. In the L1 path this is an
L1-sharded ttnn::Tensor; it is pushed into `ProgramArtifacts::op_owned_tensors`
and bound as a `TensorParameter` whose buffer backs the READER_INDICES
borrowed-memory DFB.

## Semaphores

The two legacy `SemaphoreDescriptor`s (act mcast sender/receiver, init 0, on
`all_cores`) → two `SemaphoreSpec`s (init value 0, so no
`SemaphoreAdvancedOptions::initial_value` needed). The activation reader binds
both via `SemaphoreBinding` (sem:: tokens); the legacy CTA-12/13 semaphore-id
reads are replaced by `sem::act_mcast_sender` / `sem::act_mcast_receiver`.

## Blockers (precise)

1. **In-DRAM config-tensor path (`config_tensors_in_dram == true`) — BLOCKED.**
   That path threads `conv_reader_indices_buffer->address()` and `->page_size()`
   through CTAs (`activation_kernel_compile_args.push_back(buffer->address())`,
   factory line ~569) plus a `TensorAccessorArgs(buffer)` into the activation
   kernel, consumed by `load_config_tensor_if_in_dram<27,28,29,...>` under
   `#ifdef CONFIG_TENSOR_IN_DRAM`. A raw `buffer->address()` threaded through a
   CTA to a kernel is an enumerated framework blocker (PORT_INSTRUCTIONS / recipe
   §kernel-side rule 5). Excluded from the L1 spec build: the spec never defines
   `CONFIG_TENSOR_IN_DRAM`, so the kernel's DRAM branch is dead. The L1 spec
   factory `TT_FATAL`s if invoked with `config_tensors_in_dram == true`.

## Open items / risks (UNBUILT — no build dir in this worktree)

These are faithfully-translated but unverified, and are the things most likely to
need framework attention or a closer look on first build/run:

1. **Borrowed DFBs on noop cores (top runtime risk).** The single WorkUnit places
   all three kernels on the **bounding box** of `all_cores` (legacy placed the act
   reader on the bounding box and weights/compute on `all_cores`; the shared-local-DFB
   invariant forces one node set). The borrowed DFBs ACT_SHARDED (`a`) and OUT/
   MATMUL_PARTIALS (output) are then bound on bounding-box nodes that may have **no
   shard** of the backing tensor (inactive/noop cores). Legacy tolerated this because
   the kernels early-return on inactive cores before touching those CBs; whether the
   Metal 2.0 borrowed-from address resolution / validator tolerates a borrowed DFB on
   a node where the backing tensor has no shard is **unverified**. For contiguous
   width-sharded grids (the common case) the bounding box == active set and this is
   moot. Flagged for the framework owner.

2. **OUT / MATMUL_PARTIALS / ACT_SHARDED / READER_INDICES / L1_ARRAY self-loops.**
   conv's CB usage does not map onto strict producer/consumer FIFO semantics:
   - OUT is borrowed onto the (sharded) output; compute packs directly into it with
     no FIFO consumer → bound as a compute self-loop (PRODUCER+CONSUMER).
   - ACT_SHARDED / READER_INDICES are base-pointer address sources read by the act
     reader (no FIFO peer) → self-loops on the act reader.
   - L1_ARRAY is an allocated scratch CB that **no width-sharded kernel references**
     (legacy passes its index at CTA 24, which the act reader skips). Bound as an act-
     reader self-loop purely to satisfy the producer+consumer invariant; the kernel
     never constructs `dfb::l1_array`. Candidate for the forthcoming Metal 2.0
     scratchpad resource.
   - MATMUL_PARTIALS is a genuine compute accumulator → self-loop (real).
   All self-loops are interim validator-satisfying devices, not real FIFOs.

3. **MATMUL_PARTIALS borrowed-twice with OUT.** When `partials_cb_uses_output` is
   true, MATMUL_PARTIALS is globally-allocated onto the output buffer — so OUT and
   MATMUL_PARTIALS are two DFBs both `borrowed_from = TP_OUT` (same backing memory).
   This mirrors the legacy "partials reuse output" aliasing, but two DFBs borrowing
   the same TensorParameter is not expressed via `advanced_options.alias_with`; the
   borrowed-twice semantics are unverified.

4. **Activation-reader mcast lookup tables as runtime varargs.** The two variable-
   length per-core lookup tables (`act_mcast_x/y_lookup`, indexed by a runtime value)
   are passed as `num_runtime_varargs` after the three named RTAs and accessed by
   base pointer (`get_arg_addr(3 + ...)`), since the value-returning `get_vararg()`
   helper cannot return a pointer for runtime-indexed array access. The hard-coded
   `i = 3` offset = named-RTA count; if the act reader's named RTA list changes, that
   offset must change. Documented in-kernel.

5. **Dead `get_arg_val<uint32_t>(0)` in compute (skip_compute).** `check_skip_compute`
   is a CTA, always `false` for width-sharded, so the `if constexpr(check_skip_compute)`
   body (the only RTA read) is dead. Left as the legacy positional `get_arg_val<uint32_t>(0)`
   rather than inventing a phantom named RTA the spec would have to declare on every
   node. Never executed on this path.

6. **Conditionally-bound DFBs (BIAS, ACT_SECOND_READER) via `#ifdef`.** BIAS is bound
   (+ `FUSE_BIAS` define) only when `has_bias`; ACT_SECOND_READER is never bound on
   the width-sharded path (`act_block_split_num_tiles == 0`), gated by
   `SECOND_READER_PRESENT` (kept honest if `get_cb_info` ever changes). Kernel-side
   `dfb::bias` / `dfb::act_second_reader` tokens are `#ifdef`-gated so they are not
   name-looked-up on the unbound build; the unbound alias points at a bound DFB to
   keep derived expressions well-formed (all real uses are inside the matching
   `if constexpr`).

7. **Dropped `post_conv2d_op_memory_checks_descriptor` CB-size check.** The legacy
   factory ran a CB-size-equality sanity check against the `ProgramDescriptor`'s
   `desc.cbs`. There is no descriptor in the spec world, so this host-side check is
   dropped (same reasoning the original applied for the L1-allocator half of the
   legacy check, which also needed a realised program). The Metal 2.0 framework does
   its own L1/DFB validation. Flagged for the owner.

## Findings (routed, not changed)

- **Latent bug preserved:** `writer_mcast_sender_defines["SKIP_MCAST"]` is populated
  when `skip_weights_mcast` but **never applied to any kernel** — both in the legacy
  factory and (verbatim) in this port. So `skip_weights_mcast` never reaches the
  weights kernel. Preserved exactly (scope discipline); flagged here for the op owner
  to decide whether the define should be wired to the weights kernel.

## Custom compute_program_hash

KEPT (`Conv2dDeviceOperation::compute_program_hash`) — shared with the still-legacy
`Conv2dShardedProgramFactory`; not deleted, per the bulk-port instruction.

## Pybind

No pybind hook exposes the factory entry point (`conv2d_nanobind.cpp` binds the
user-facing `conv2d` op, not `create_workload_descriptor`/`create_program_spec`).
Nothing to remove.
