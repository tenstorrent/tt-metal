# Metal 2.0 Port Brief — `ttnn/cpp/ttnn/operations/data_movement/untilize`

> The op as a whole is **RED**, but the gate is **config-scoped**: a clean 7-factory subset of the native `UntilizeDeviceOperation` clears every gate and ports today. This brief covers **only that subset**. The full record — including the two blockers — is in `METAL2_PREPORT_AUDIT.md`.

**Scope of this brief — port these 7 native factories only:**

1. `UntilizeSingleCoreProgramFactory`
2. `UntilizeMultiCoreProgramFactory`
3. `UntilizeMultiCoreNDShardInputProgramFactory`
4. `UntilizeMultiCoreParallelizeColumnProgramFactory`
5. `UntilizeMultiCoreSubCoreGridsProgramFactory`
6. `UntilizeMultiCoreInputAndOutputShardTypeAndShardSpecIdenticalProgramFactory`
7. `UntilizeMultiCoreInputAndOutputNDShardTypeAndShardSpecIdenticalProgramFactory`

**Do NOT port (blocked — see audit):**
- `UntilizeMultiCoreBlockProgramFactory` — readiness sheet `Is able to port? = no` (`Known op issues = "Per-node CB size"`: same CB index sized differently per core group). Tracked by [tenstorrent/tt-metal#51305](https://github.com/tenstorrent/tt-metal/issues/51305) (assigned bbradelTT, OPEN): a **prereq refactor** the ops team owns — buffer size is a *correctness* property in this factory, so **do not improvise the DFB sizing** (a quasar porter shipped silent data corruption doing so). Stays blocked until #51305 lands. **None of the 7 factories in this brief have per-node CB sizing**, so following this brief does not touch #51305.
- `UntilizeCodegenDeviceOperation` (`codegen/`) — no readiness-sheet row yet (coverage gap → sheet owner). Code cross-check is clean and all other gates pass, so it likely joins the port once the sheet is reconciled — but not under this brief.

**Gates cleared (for the 7-factory subset):** Device 2.0 ✓ · Features ✓ · TTNN factory concept ✓ · Offset base pointers ✓ · TensorAccessor 3rd arg ✓ (subset uses the 2-arg form; no sites).

**Recipe docs:** `548e18500b3 2026-08-18 docs(metal_2.0): a direct-descriptor op converts to a real program factory` *(carry into the port report's Provenance section)*

## TTNN factory analysis

These facts feed the port's TTNN ProgramFactory wiring (→ `ttnn_factory.md`); the subset ports to `ProgramSpecFactoryConcept`.

- **Current concept:** `descriptor` (each factory defines `create_descriptor(...) → ProgramDescriptor`).
- **Op-owned tensors:** none.
- **Target concept:** `ProgramSpecFactoryConcept` (`Override runtime args method? = no` → the framework refreshes tensor bindings on cache hit; write the one method).
- **Gate-cleared, confirmed absent** (each would have blocked the brief): a non-`none` `TensorParameter relaxation`; `get_dynamic_runtime_args`. A custom hash, an `override_runtime_arguments`, and a pybound `create_descriptor` are all **absent** here too (none of them gate; noting they simply don't apply).

## Construct — to do

**Tensor bindings** (per binding; all Case 1 or clean — **no Case 2 raw-pointer work anywhere**):

- **Interleaved I/O** — `SingleCore`, `MultiCore` (interleaved config), `ParallelizeColumn`, `SubCoreGrids`: `src0` and `dst` are **Case 1** (kernel builds `TensorAccessor(tensor::name)`; the current `Buffer*` RTA slot + `TensorAccessorArgs` plumbing both disappear).
- **`MultiCore` even-sharded config:** input CB `c_0` is **borrowed-memory** (`.buffer = src0_buffer`, [untilize_multi_core_program_factory.cpp:118](device/factories/untilize_multi_core_program_factory.cpp)) → bind via `DataflowBufferSpec::borrowed_from` (**clean**, no accessor); `dst` → Case 1. Block-reader config (`reader_unary_sharded_blocks.cpp`): `src0` → Case 1, `dst` → Case 1.
- **`NDShardInput`:** `src0` → **Case 1**, bound by **both** the reader and the writer (the writer reads the input buffer for ND-shard page mapping, [writer_unary_stick_layout_split_rows_multi_core_nd_shard.cpp:34-37](device/kernels/dataflow/writer_unary_stick_layout_split_rows_multi_core_nd_shard.cpp)); `dst` → Case 1.
- **`...ShardTypeAndShardSpecIdentical`** and **`...NDShardType...Identical`:** both `c_0`←src0 and `c_16`←dst are **borrowed-memory** (`.buffer = ...`, zero-copy) → bind **both** via `DataflowBufferSpec::borrowed_from` (**clean**). Readers/writers here carry only tile-count RTAs (no `Buffer*` slot).

**TensorParameter relaxation:** none.

**TensorAccessor 3rd arg:** none in this subset (all native kernels use `TensorAccessor(args, addr)`).

**CB endpoints:** all legal 1P+1C — `c_0` = reader(producer)+compute(consumer); `c_16` = compute(producer)+writer(consumer). Nothing to self-loop, flag, or drop. Translate the borrowed-memory CBs (identical-shard, identical-nd-shard, multi_core even-sharded) via `borrowed_from` as noted above.

**WorkUnitSpec split (cliff / per-group compute)** — applies to `MultiCore` (full + cliff compute) and `ParallelizeColumn` (full + cliff compute): reader and writer run over the **union** core range; compute is split across **disjoint** core groups. Give **each core group its own `WorkUnitSpec`** listing reader + writer + that group's compute instance (a kernel's effective placement is the union of the WUs it appears in). Do **not** put reader/writer in one WU over the union and compute in a narrower WU — the `target_nodes` disjointness invariant fires at test time (`program_spec.cpp` "overlap in target nodes").

**Compute hardware config:** set `.compiler_options.opt_level = KernelBuildOptLevel::O3` on every **compute** `KernelSpec` (Metal 2.0 defaults all kernels to O2; legacy built compute at O3 — leaving it default silently regresses perf, and has caused fp32 JIT-compile failures on other ports). Leave reader/writer at the O2 default. Carry the `fp32_dest_acc_en` / `unpack_to_dest_mode` / `DST_ACCUM_MODE` config through unchanged (each factory already sets these).

## Watch for

- **Shared-kernel forks — two already exist; bind them, do NOT create a second fork:**
  - `device/kernels/compute/untilize.cpp` → bind the existing **`untilize_metal2.cpp`** (created by the `data_movement/fold` port). Applies to `SingleCore`, `SubCoreGrids`, `ParallelizeColumn`, `...ShardTypeAndShardSpecIdentical`.
  - `eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp` → bind the existing **`reader_unary_interleaved_start_id_metal2.cpp`**. Applies to `ParallelizeColumn`, `SubCoreGrids`.
- **Shared donors with NO fork yet — this port creates the first `_metal2` fork; the co-borrower list is a *sunset list*, not authorization to convert the original in place:**
  - `data_movement/sharded/.../writer_unary_sharded.cpp` (also used by ~9 op families) — `IdenticalShard`, `IdenticalNDShard`, `MultiCore` (even-shard).
  - `eltwise/unary/.../reader_unary_sharded.cpp` (~7 families) — `IdenticalShard`, `IdenticalNDShard`, `MultiCore` (even-shard).
  - `data_movement/sharded/.../reader_unary_nd_sharded_blocks.cpp` (also untilize_with_unpadding) — `NDShardInput`.
  - Own shared kernels also needing forks: `device/kernels/dataflow/reader_unary_start_id.cpp` (also copy, tilize); `device/kernels/compute/untilize_variable_num_blocks.cpp` (also untilize_with_unpadding) — `MultiCore`, `NDShardInput`, `IdenticalNDShard`.
  - Untilize-only own DM kernels (no external co-borrowers; still fork-per-shared-kernel policy applies only if they gain one): `writer_unary_stick_layout_split_rows_single_core.cpp`, `..._multi_core.cpp`, `..._interleaved_parallel_columns.cpp`, `..._multi_core_nd_shard.cpp`, `reader_unary_sharded_blocks.cpp`.
- **Kernels are already Device-2.0 / DFB-aware.** The DM kernels use `DataflowBuffer dfb(cb_id)`, `Noc`, `TensorAccessor` today — the port is a **binding-layer change** (construct DFBs from `dfb::name` binding tokens, tensors from `tensor::name`), not an idiom rewrite. The writers' `dfb_out.get_read_ptr()` is a legitimate DFB member getter; keep it.
- **Pre-existing bug in `ParallelizeColumn` — do not fix in the port, do not replicate.** The cliff-core writer RTA list passes 7 args but the writer kernel reads 6 (extra `stick_size` at index 2; [factory:227-235](device/factories/untilize_multi_core_parallelize_column_program_factory.cpp) vs [kernel:17-22](device/kernels/dataflow/writer_unary_stick_layout_split_rows_interleaved_parallel_columns.cpp)). It is a latent, likely-unreached ops-team bug (see audit Misc anomalies), out of port scope. Translate the RTAs faithfully as-is; flag it to the ops team, don't silently "correct" it inside the port diff.
- **RTA varargs:** none — every kernel reads a fixed arg set at constant indices. Name each; reach for no vararg mechanism.
