# Metal 2.0 Port Report — `ttnn/cpp/ttnn/operations/matmul`

**Factory ported: `MatmulMultiCoreReuseMcast2DProgramFactory` (1 of 8).** The other seven keep their
current concepts and keep building.

## Outcome

**`CAPITULATED`** — and the word needs its shape spelled out, because this is not the usual early
stop. The conversion is **complete**: factory, six kernels, override, pybind and the CCL coupling are
all done, and the op's own correctness tests are green. But **two things outside the porter's reach
stop the diff from merging**, and both have their own Handoff point below:

1. one reachable configuration cannot be expressed under Metal 2.0's DFB endpoint rules, and on it
   the ported factory raises a `TT_FATAL` at spec-validation time where the legacy factory ran;
2. removing the factory's pybound `create_descriptor` — which the port forces — breaks the Python
   descriptor framework's path to this factory, and what that framework should do about a ported
   factory is a decision its owner has to make.

Everything else in this report holds either way.

| | |
|---|---|
| Verified working | interleaved 2D mcast · `transpose_mcast` both ways · block-sharded in0 (matched grid) · block-sharded in0 **and** output · fused bias · fused activation · fp32 dest acc with the partials-reload alias · multi-K-block packer L1 acc · single-row and single-column grids (both `SKIP_MCAST` paths) · cache-hit refresh across reallocation |
| Blocked — Handoff 1 (DFB endpoint rules) | `in0` **BLOCK_SHARDED** whose shard grid is wider along the mcast axis than the output's column blocks (`in0_sender_num_cores_along_width > num_blocks_x`) |
| Blocked — Handoff 4b (descriptor framework) | every `TestMatmulFactories` case in `test_parallel_sequential.py` that reaches this factory through `models/experimental/ops/descriptors/matmul.py` (8 tests) |
| Not runnable here | the two CCL ops' own coverage — multi-device, and this bench has one Blackhole |

Measured on the confirmed test set (Blackhole p150b, `TT_METAL_WATCHER=10`, host-side legality checks
forced on and both `METAL2_CHECKS_FORCED` markers observed):

| suite | result |
|---|---|
| `unit_tests_ttnn --gtest_filter='*Matmul*'` | **24 passed, 5 skipped, 0 failed** — identical to the pre-port baseline |
| `unit_tests/operations/matmul/{test_matmul,test_custom_grids}.py` | **888 passed, 312 skipped, 0 failed** |
| targeted configuration probes (8) | 7 pass; the 8th is Handoff 1 |
| `test_parallel_sequential.py::TestMatmulFactories` | 9 passed, **8 failed** (all Handoff 4b), 17 skipped |
| `unit_tests/operations/ccl/test_new_matmul_reduce_scatter.py` | 8 skipped — needs 8 devices |

## Provenance

- **Recipe docs (this port):** read out of `origin/akertesz/op-porting-recipe` at
  `4bd4bf42bfe 2026-09-14 docs(metal_2.0): state the offset-base wall as a category, not as slice's current state`.
  The port branch is `main` + the audit commit, so
  `git log -1 -- docs/source/tt-metalium/tt_metal/apis/host_apis/metal_2.0/` prints nothing here —
  those docs are deliberately not merged to `main`.
- **Audit docs (inherited):** `058100de698 2026-08-31 docs(metal_2.0): let the sheet gate multi-program ops, and bound what the port covers`

## TTNN ProgramFactory

### Concept realized

`CustomProgramSpecFactoryConcept`, as the audit chose. `create_descriptor` is replaced by
`create_program_artifacts`; `override_runtime_arguments` keeps its name but changes shape to return
`ProgramRunArgs`.

The override returns a `TensorArgument` for **every** io-tensor `TensorParameter` — `in0`, `in1`,
`output`, and `bias` when present — which is exactly the set the ported-from override refreshed (its
eight statements were all address refreshes and nothing else). `kernel_run_args` is empty, because
the ported-from override touched no runtime argument. Nothing is deliberately skipped.

Verified rather than assumed: the same 2D matmul was run four times with the input tensors
reallocated at different addresses in between, and every call was numerically correct. On this
concept the framework refreshes nothing on the factory's behalf, so a missing `TensorArgument` would
have surfaced as wrong numerics on calls 2–4 only.

### Device-op-class edits

- **Pybind entry point removed:** `matmul_nanobind.cpp` — the
  `nb::class_<ttnn::prim::MatmulMultiCoreReuseMcast2DProgramFactory>` block whose only member was
  `create_descriptor`. See Handoff points.
- **Pybind-hook-only parameter dropped:** `create_descriptor`'s fourth argument,
  `const std::optional<CoreRangeSet>& core_range_set`, which production code never set and the
  factory ignored (it was spelled `/*core_range_set*/`). It disappears with the method.
- **Custom `compute_program_hash`:** none. The device-op's deliberately differently-named
  `compute_descriptor_program_hash` (`device/matmul_device_operation.hpp:50`) is untouched.

### Guard census

The op's `TT_FATAL` / `TT_ASSERT` / `TT_THROW` count is unchanged in every file but this factory's,
where it goes 31 → 32. That nets one loss against two additions:

- **Lost (subject deleted):** `TT_FATAL(false, "Fused operation must be either all_gather or reduce_scatter.")`
  in the descriptor builder's per-core loop. `create_descriptor` passed `fused_op_signaler = std::nullopt`
  unconditionally, so the branch it guarded was unreachable on this path; the spec builder takes no
  signaler parameter at all, so the subject is gone with it. The identical guard in the legacy
  `Program` builder (which the CCL ops do reach) is untouched.
- **Added:** the two input/output arity guards the ported-from override carried, reproduced in the
  translated `override_runtime_arguments`. The legacy `override_runtime_arguments_impl` keeps its
  own copies, since it still serves the CCL ops.

## Handoff points

### 1. DFB endpoint rules reject a legal Gen1 topology — *the blocker* (Metal 2.0 runtime)

**What the op does.** When `in0` is BLOCK_SHARDED and its shard grid is wider along the multicast
axis than the output needs in columns, the factory puts an in0 sender on nodes *outside* the work
grid (`in0_mcast_cores_without_work_and_not_in_receiver_grid`). Those nodes run exactly one kernel —
the sender — and it both fills and drains its `in0` buffer (`reserve_back` / `push_back`, then
`pop_front`, because `core_has_output_block_work` is false). On the work-grid nodes the same buffer
has the ordinary shape: the sender fills it, the compute kernel drains it. The buffer must be one
`DataflowBufferSpec`, because the no-work sender multicasts to `.addr = dfb_in0.get_write_ptr()` —
its *own* write pointer — so its L1 address and FIFO cursor have to stay in lockstep with the
receivers'.

**Per node, that satisfies the invariant the header states**
(`dataflow_buffer_spec.hpp`: *"at the node level, a DFB instance has exactly one producer kernel
instance and exactly one consumer kernel instance"*): work nodes have 1 producer (sender) + 1
consumer (compute); no-work nodes have 1 + 1 from the self-looping sender. The per-node census in
`program_spec.cpp` agrees. **Two secondary rules reject it anyway**, and both were confirmed by
running, not just by reading:

| Rule | Site | Message |
|---|---|---|
| Per-role kernel-kind uniformity | `tt_metal/impl/metal2_host_api/program_spec.cpp:1366-1377` | `records[i].kernel->is_compute_kernel() == first_is_compute` — the CONSUMER role holds `compute` (work nodes) and the DM sender (no-work nodes) |
| Self-loop set equality | `tt_metal/impl/metal2_host_api/program_spec.cpp:1515-1521` | `producer_kernels == consumer_kernels` — reached only after setting `allow_instance_multi_binding`, which skips the first rule |

So the flag does not rescue it: rule 1 is skipped under `allow_instance_multi_binding`, and rule 2
is unconditional. The only shapes that get past both are ones that misstate the topology (declaring
`compute` a *producer* of `in0` and the senders *consumers*, stacked with the multi-binding flag) —
which is the stacking the port recipe's self-audit explicitly forbids, so the port stops here
instead.

**Why the rules are conservative here.** Both exist to guarantee a single per-role processor mask,
which matters on Gen2. On Gen1 a DFB lowers to a plain circular buffer with no per-role mask at all,
and the legacy program this port reproduces is correct on Gen1 today. The narrowest fix that would
unblock this port is to scope both checks to Gen2 (or to evaluate role uniformity per node rather
than per spec, which is what the hardware invariant actually needs).

**Reproducer** (fails at program build; passes on `main`):

```python
# in0 [256, 512] BLOCK_SHARDED over a 4x2 grid -> in0_sender_num_cores_along_width = 4
# output N = 128, per_core_N = 4 tiles                        -> num_blocks_x = 1
ttnn.matmul(a_block_sharded, b, program_config=ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
    compute_with_storage_grid_size=(4, 2), in0_block_w=4, out_subblock_h=1, out_subblock_w=1,
    out_block_h=4, out_block_w=4, per_core_M=4, per_core_N=4,
    transpose_mcast=False, fused_activation=None, fuse_batch=True))
```

### 2. The fused-CCL kernel helpers read runtime args positionally (kernel-lib / CCL owners)

`MatmulOpReceiver` and `OpSignaler`
(`ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp`, outside the op directory)
consume runtime arguments positionally through a `uint32_t& rt_args_idx` cursor. A Metal 2.0 kernel
addresses its arguments by name and has no such cursor, so there is nothing to hand them.

This factory never reaches those paths — `create_descriptor` passed `fused_op_signaler = std::nullopt`
unconditionally, so `fuse_op` / `fuse_op_all_gather` / `fuse_op_reduce_scatter` were always 0 — so
the port is not blocked on it. In the four forks that carry the code, the gate is promoted from a
compile-time argument to a `#ifdef FUSE_OP…` that no Metal 2.0 factory may define, and the block is
preserved verbatim underneath with a comment saying so. Enabling the define fails to compile,
deliberately and loudly.

**This blocks the fused-CCL matmul factories outright.** `matmul_multicore_reuse_mcast_1d` and the
sparse factory bind these same kernels *and* use the fused paths; they cannot port until these two
helpers gain a named-argument interface.

### 3. GlobalCircularBuffer in the in1 sender/writer (Metal 2.0 runtime)

`reader_bmm_tile_layout_in1_sender_writer_padding.cpp` carries an `#ifdef ENABLE_GLOBAL_CB` path (the
tensor-prefetcher path, `tt::CBIndex::c_31` plus `experimental::remote_cb_*`). A GlobalCircularBuffer
is not a `DataflowBuffer` and `GlobalDataflowBuffer` is unimplemented, so it cannot be converted.
This factory never defines `ENABLE_GLOBAL_CB`, so the path is preserved verbatim under its existing
gate with a comment; the remaining `cb` vocabulary in the fork is confined to it and is correct
(those really are circular buffers). Whichever factory owns that path is blocked on
`GlobalDataflowBuffer`.

### 4. Removed pybind surface (TTNN / downstream Python consumers)

`ttnn/cpp/ttnn/operations/matmul/matmul_nanobind.cpp`: the
`nb::class_<ttnn::prim::MatmulMultiCoreReuseMcast2DProgramFactory>` block and its single
`create_descriptor` static are deleted. It exposed the factory's `ProgramDescriptor` to Python for
descriptor-framework introspection. Nothing in this repo's Python called it (the only `.py` mentions
of the factory name are docstrings in `test_parallel_sequential.py`), but it was a public surface.
The sibling `nb::class_<MatmulDeviceOperation>` block is untouched.

### 4b. The Python descriptor framework loses its path to this factory (descriptor-framework owner)

`models/experimental/ops/descriptors/matmul.py` picks a factory with
`ttnn.matmul_select_program_factory` (`:97`) and then calls
`factory.create_descriptor(operation_params, tensor_args, [out], core_range_set)` (`:120`). Removing
this factory's `nb::class_` unregisters it, so `matmul_select_program_factory` cannot convert its own
return value any more:

```
E  TypeError: Unable to convert function return value to a Python type! The signature was
E      matmul_select_program_factory(...) -> ttnn::prim::MatmulMultiCoreProgramFactory | ...
   models/experimental/ops/descriptors/matmul.py:97: TypeError
```

**Eight tests fail as a result**, all of them building a `MatmulMultiCoreReuseMultiCastProgramConfig`
in `test_parallel_sequential.py::TestMatmulFactories`: `test_mcast_2d_factory`,
`test_mcast_2d_layernorm_chain`, `test_persistent_mcast_2d_deferred`, and the five mixed-factory
chains with a 2D-mcast leg (`test_parallel_different_factories`,
`test_deep_chain_alternating_factories_and_norms`, `test_parallel_mm_chains_different_factories`,
`test_nested_tree_mixed_factories`). The nine cases in that class that use only
`MatmulMultiCoreReuseProgramConfig` or the 1D config still pass.

**Two facts make this worth resolving once rather than per port:**

- **The debt is already three ports old.** On `origin/main` only three matmul factories are still
  pybound — `MatmulMultiCoreReuseOptimizedProgramFactory`, `…Mcast1DProgramFactory` and this one.
  Ports 1–3 already unregistered theirs, so `matmul_select_program_factory` already raises this same
  `TypeError` for a MultiCore, DRAM-sharded or BatchedHS config today; nothing in the descriptor
  framework's tests exercises those configs, so it went unnoticed. This port is the first whose
  configs that suite does cover.
- **The framework's own guard is inert.** `_UNSUPPORTED_FACTORY` (`:29`) is
  `getattr(ttnn, "MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory", None)` — and that class is
  not pybound, so the attribute resolves to `None` and the `isinstance` check at `:98` never fires.
  The intended "not supported in the descriptor interface" `ValueError` is dead code; every
  unsupported factory reaches the caller as a raw nanobind `TypeError` instead.

Deciding what the framework does about ported factories — and what the affected tests should then do
— is its owner's call, not the porter's. It is the same decision the porting plan raised for
`MatmulMultiCoreReuseOptimizedProgramFactory` (the plan's D3), and `…Mcast1DProgramFactory` will
reach it too. **It was not flagged for this factory:** the audit checked the pybind block and the
readiness sheet's `Pybind descriptor` cell, but not this Python consumer.

### 5. Out-of-directory edit: the CCL consumers of the factory's override (invoker-authorized)

The audit's open question — two CCL device operations calling
`MatmulMultiCoreReuseMcast2DProgramFactory::override_runtime_arguments` directly, while the concept
requires that method to be a single overload returning `ProgramRunArgs` — was resolved by the invoker
in favour of the smallest split:

- **In the op directory:** the legacy body moves to a public free function,
  `ttnn::prim::matmul_multi_core_reuse_mcast_2d_override_runtime_arguments_helper`, beside the
  existing `matmul_multi_core_reuse_mcast_2d_optimized_helper` it partners. Same body (it still
  forwards to `reuse_mcast_optimized_helpers::override_runtime_arguments_impl`), same
  `shared_variables_t`.
- **Outside it (two files, one call each):**
  `experimental/ccl/all_gather_matmul_async/device/all_gather_matmul_async_program_factory.cpp` and
  `experimental/ccl/matmul_reduce_scatter_async/device/matmul_reduce_scatter_async_program_factory.cpp`
  now call that free function. The unused `operation_attributes` argument drops with the rename.

CCL behaviour is unchanged — same function body, same legacy `Program` builder, which this port does
not touch. `MatmulMultiCoreReuseMcast1DProgramFactory::override_runtime_arguments` (called from the
same `std::visit` in `all_gather_matmul_async`) will need the identical split when that factory
ports.

### 6. Audit corrections (Metal 2.0 audit recipe / next auditor)

Two items the audit stated that the code contradicts. Neither blocked the port; both would have
misled a porter who transcribed instead of re-deriving.

- **"RTA varargs: none — name every argument."** Two genuine indexed-collection sites exist. The
  block-sharded in0 sender reads `in0_mcast_noc_x[num_x]` / `in0_mcast_noc_y[num_y]` through
  `get_arg_addr(increment_arg_idx(...))` and then indexes them in a loop bounded by compile-time
  args; the DRAM-width-sharded in1 sender reads a `(stride_bytes, bank_id)` list at an index it
  advances two at a time over a runtime-valued count. Both are varargs, matching what the
  already-ported DRAM-sharded sibling does.
- **"Tensor bindings — four, all Case 1. No Case 2 site."** The in1 sender's two DRAM-sharded paths
  use `in1_tensor_addr` (and, with bias, `in3_tensor_addr`) as **raw base addresses** for
  bank-direct reads — Case 2. They are reachable from this factory whenever `in1` is DRAM
  width- or height-sharded. The port takes them off the binding via
  `TensorAccessor::get_bank_base_address()`, the sanctioned bridge, so nothing is smuggled through a
  runtime arg; but the audit's "the bridge is not needed anywhere" is wrong.

## Successes

- **[Aliased DFBs] read against the declaring header, not the catalog.**
  `advanced_options.hpp:168-171` requires alias members to target the same *node set*; the patterns
  catalog says the same *kernels*. `OUT` is bound by the compute spec **and** the in1 writers while
  `INTERMED0` is compute-only, so the catalog's wording would have forced a bogus capitulation — the
  node sets both resolve to the work grid, and the alias is legal. The recipe's "go to the headers
  first; they are ground truth" earned its place here.
- **[Conditional / optional resource bindings] fired exactly as documented.** Three families —
  sparsity, the fused-CCL paths, and `in0_transpose_tile` — reach their tokens from `if constexpr`
  branches or file-scope ternaries that the legacy kernels gated on compile-time argument *values*.
  Every one of them had to be promoted to a preprocessor gate, and the catalog's "Promote a CTA gate
  to a define" paragraph is what named the move before the compiler did.
- **[`constexpr` metadata keeps the free-function form].** `get_tile_size(dfb::in0)` /
  `get_dataformat(dfb::in0)` in three forks: the whitelist's rule (the legacy *declaration* is the
  whole test) meant no thinking was required, and demoting to a member getter would have silently
  cost constant folding in `pad_last_ktile<in0_data_format, in0_last_ktile_w>`.
- **The op's pytest sweep caught an aliasing bug the gtests could not.** In the branch where the
  output and intermediate share one buffer *and* the output is sharded, legacy put every index on a
  single `CBDescriptor` backed by the output tensor — so the partials views are on the output shard
  too. The first draft borrowed only on `OUT`, which violates the alias group's borrowed-memory
  consistency rule (`program_spec.cpp:1707-1713`). All 68 failures in the first
  `test_matmul.py` run were that one assertion, and none of the C++ smoke tests could have found it:
  every 2D case there writes a **DRAM-interleaved** output, so none reaches the branch. Worth
  knowing for the remaining matmul ports, which share this out/intermed0 shape.

- **The DRAM-sharded sibling's vararg-padding idiom transferred cleanly.** Cores straddle different
  numbers of DRAM shards, so the per-node lists differ in length while `num_runtime_varargs` is one
  number; declaring the max and zero-padding the short lists (the kernel's loop is bounded by its own
  count) is exactly what `matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:836-843`
  already does.

## Friction

### Gaps

- **`AddRuntimeArgsForNode` takes only `std::initializer_list`** (`program_run_args.hpp:189-197`), so
  it cannot be used for a kernel whose runtime-argument *set* is built up conditionally — which is
  the common case for any factory with optional bias / sharded output / sharded weights. Two of the
  five kernels here had to open-code the two-line append instead. A `std::span<const
  std::pair<std::string, uint32_t>>` overload would cover both shapes.
- **No way to declare that two DFBs must share an L1 address.** This is the other side of Handoff
  point 1: `alias_with` is the mechanism, and it requires members to target the same node set, so
  disjoint-node buffers that must coincide in L1 cannot say so. They happen to coincide today
  because the allocator assigns one address per buffer as a max over its cores — but that is an
  implementation detail a port may not lean on.
- **The compute fork's `MM_PARTIALS_RELOAD_ALIAS` path had no consumer until now.** Port 2 wrote
  `dfb::intermed0_reload_alias` into the shared compute fork forward-lookingly; this factory is the
  first to bind it (`mcast_1d` is the only other user, and it is blocked). Worth knowing that the
  path's first exercise is here, not in the port that authored it.

### Confusion

- **The audit's endpoint dispositions were right to be re-derived, and the recipe says so** — but
  the same recipe presents the *varargs* and *binding-case* findings as inherited fact. Both were
  wrong here (Handoff point 6). Extending "re-derive, don't transcribe" to those two subjects would
  have saved a mid-port surprise.
- **`-Wconditional-uninitialized` on carried-over legacy locals.** Two mcast coordinates that the
  legacy declared uninitialized and assigned under `if (in0_block_sharded)` now get read from a loop
  far enough away that clang can no longer see the guard, and `-Werror` rejects it. Zero-initializing
  them is behaviour-preserving but is a diff on a line the port otherwise had no reason to touch;
  worth a line in the recipe so the next porter does not wonder whether it is in scope.

## Open items for downstream

### Shared kernel touches

Six shared kernels; five forks created (rung 2), one fork reused (rung 1). Every fork is beside its
original, and every original got the pointer comment.

| kernel (all under `matmul/device/kernels/`) | rung | remaining unmigrated consumers |
|---|---|---|
| `dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp` | **2 — created** `…_metal2.cpp` | `matmul_multicore_reuse_mcast_1d_program_factory.cpp` (2 factories), `device/sparse/` device-op + factory |
| `dataflow/reader_bmm_tile_layout_in0_sender_receiver_padding_block_sharded.cpp` | **2 — created** | mcast_1d file (2 factories) |
| `dataflow/reader_bmm_tile_layout_in0_receiver.cpp` | **2 — created** | mcast_1d file (2 factories), sparse factory |
| `dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp` | **2 — created** | mcast_1d file (2 factories), sparse factory |
| `dataflow/reader_bmm_tile_layout_in1_receiver_writer_padding.cpp` | **2 — created** | mcast_1d file (2 factories) |
| `compute/bmm_large_block_zm_fused_bias_activation.cpp` | **1 — reused** `…_metal2.cpp` | BatchedHS, Optimized and McastDRAMSharded are already on the fork; mcast_1d file (2 factories) and sparse remain |

The five new forks' binding vocabulary — now fixed for every later consumer — is
`dfb::{in0, in0_sharded, in1, bias, out, sparsity}`,
`sem::{in0_mcast_sender, in0_mcast_receiver, in1_mcast_sender, in1_mcast_receiver}`,
`tensor::{in0, in1, bias, out, sparsity}`, and the gates
`IN0_SHARDED`, `IN0_EXTRACT_SHARD_SUB_BLOCKS`, `SPARSITY`, `FUSE_OP`, `FUSE_OP_ALL_GATHER`,
`FUSE_OP_REDUCE_SCATTER`, alongside the legacy gates the kernels already had (`SKIP_MCAST`,
`FUSE_BIAS`, `BIAS_SHARDED`, `OUT_SHARDED`, `IN1_SHARDED`, `IN1_DRAM_WIDTH_SHARDED`,
`IN1_DRAM_HEIGHT_SHARDED`, `ENABLE_GLOBAL_CB`).

### Dead buffer dropped

`c_6`, the 64-byte "local L1 to store temp vars" buffer the legacy factory allocated on the
block-sharded path (allocation at the pre-port `…mcast_2d_program_factory.cpp:1020-1027`, handed to
the two sender kernels as the `cb_l1_array` named argument at `:733` and `:754`), has **zero
endpoints**: `grep -rn l1_array ttnn/cpp/ttnn/operations/matmul/` finds only host-side
`named_compile_time_args` entries and the legacy builder's `CreateCircularBuffer`, and no kernel
source reads it. Dropped, along with its named argument. This answers the audit's open census
question on `c_6`. **`matmul_multicore_reuse_mcast_1d_program_factory.cpp` carries the same dead
buffer and argument** (`:622`, `:649`, `:668`, `:887-890`, `:3660`, `:3693`, `:3713`) — its porter
can drop them the same way.

### Observations left alone (pre-existing, not port work)

- **Dead placeholder `.address()` calls.** The pre-port descriptor builder computed a buffer address
  at `:1248`, `:1300`, `:1313` and `:1457` only to overwrite it with a tensor reference tens to
  hundreds of lines later. They disappear with the bindings, so nothing is left to fix here — but the
  identical pattern is still in the legacy `Program` builder in the same file and in mcast_1d, where
  it remains the first thing a smuggled-pointer search finds.
- **The two builders in one file keep opposite conventions.** `create_program_mcast_in0_in1`
  (the CCL ops' legacy `Program` builder, untouched by this port) writes raw un-rebound addresses
  and relies on the shared override for cache-hit patching. A reader who lands in it mid-file will
  find code that looks unported and is supposed to be.
- **`create_descriptor`'s `core_range_set` was accepted and silently ignored.** A Python caller
  passing one had it discarded. Removing the entry point resolves it.

### Test coverage notes

- The op's C++ gtests cover this factory well (`MatmulSmoke.Auto2DMcastDefault`,
  `BlockSharded2DAuto`, `BlockSharded2DTransposeAuto`, `MultiBlockPerCore2D`,
  `FusedActivationGelu2D`, `Fp32DestAccConfigGrid`), but **nothing in the confirmed test set covers
  the configuration in Handoff point 1** — which is why it took a hand-written probe to find. A case
  with a block-sharded `in0` whose shard grid is wider than the output's column blocks belongs in
  `tests/ttnn/unit_tests/gtests/test_matmul.cpp` regardless of how the handoff is resolved.
- The two CCL ops that consume this factory's legacy builder are multi-device; their coverage
  (`tests/nightly/t3000/ccl/test_minimal_all_gather_matmul_async.py`) could not run on this
  single-Blackhole bench and needs CI.
