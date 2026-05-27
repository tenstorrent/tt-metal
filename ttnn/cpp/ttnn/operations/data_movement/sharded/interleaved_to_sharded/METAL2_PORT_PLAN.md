# Port Plan — `interleaved_to_sharded` (DEFERRED)

## Status: DEFERRED — structural blocker uncovered during planning

The pre-port audit marked this op GREEN. **Planning revealed that 3 of the 4 kernel-dispatch paths in the legacy factory use donor kernels that are still on the Device 1.0 API and cannot be ported to Metal 2.0 without first migrating the donors.** Per the recipe's stop-signal language ("if planning uncovers a structural issue the audit didn't catch ... stop and report. Do not improvise around it."), this port is deferred until the blockers are resolved.

This document records the legacy inventory and the blocker so the work isn't lost.

## Legacy inventory

The factory `InterleavedToShardedProgramFactory::create_descriptor` dispatches on `{input.layout(), dst_buffer->buffer_type()}` and selects a different (reader, writer) kernel pair per path. The compute kernel (`eltwise_copy.cpp`) is optional, only when input/output dtypes differ.

| Path | Reader | Writer | Donor Device-2.0 status |
|---|---|---|---|
| TILE + sharded-L1 output | `data_movement/sharded/.../reader_unary_sharded_blocks_interleaved_start_id.cpp` | `data_movement/sharded/.../writer_unary_sharded.cpp` | **both GREEN** |
| TILE + DRAM output | same reader | `data_movement/sharded/.../writer_unary_sharded_blocks_start_id.cpp` | reader GREEN, **writer uses `experimental::ShardedAddrGen` + legacy `noc_async_write` — RED** |
| ROW_MAJOR + sharded-L1 output | `data_movement/sharded/.../reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` | `data_movement/sharded/.../writer_unary_sharded.cpp` | **reader uses legacy `noc_async_read` (with scratch-pad TID state machine) — RED**, writer GREEN |
| ROW_MAJOR + DRAM output | same legacy reader | `data_movement/sharded/.../writer_unary_sharded_stick_layout_start_id.cpp` | both RED |

Three RED donor kernels in total:

- `writer_unary_sharded_blocks_start_id.cpp` — uses `experimental::ShardedAddrGen<tensor_shard_info>`. The Device 2.0 kernel API has no `noc_traits_t<ShardedAddrGen<>>` specialization, so this cannot be migrated by simple kernel rewrite. Blocked by the wider kernel-side API gap.
- `writer_unary_sharded_stick_layout_start_id.cpp` — same ShardedAddrGen dependency.
- `reader_unary_stick_layout_sharded_blocks_interleaved_start_id.cpp` — uses legacy `noc_async_read` with a hand-rolled transaction-ID state machine and scratch buffer (alignment workaround for DRAM-input rows whose stride isn't a multiple of NoC alignment). The pattern doesn't have an obvious 1:1 mapping into the Device 2.0 `Noc::async_read_with_state` family without breaking the alignment correctness; needs a careful donor-side port.

## Why partial port isn't appropriate

The legacy factory selects the kernel path *inside* `create_descriptor`. Porting only the TILE-sharded-L1 path would mean:

- Either: keep `create_descriptor` for the legacy paths AND add `create_program_spec` for the new path. But `ProgramSpecFactoryConcept` and the legacy ProgramDescriptor concept aren't simultaneously satisfiable on one factory in the `program_factory_t` variant.
- Or: introduce a second factory type for the new path and route via `select_program_factory`. This is a more invasive refactor than a clean port; the user explicitly scoped the port narrowly ("Fork. Proposed order sounds good. Stop after the first one."), and the structural blocker is not what the user asked me to navigate around.

Both options carry test-surface risk (paths 2-4 would still be on the legacy factory, paths 1 alone would be on the new factory; cache hit/miss behavior asymmetric).

## What unblocks the port

Choose one:

1. **Migrate the three RED donor kernels to Device 2.0 first.** The ShardedAddrGen pair needs Metal kernel-side support added (a `noc_traits_t<ShardedAddrGen<>>` specialization or an equivalent NoC-write helper). The stick-layout reader needs a careful re-expression of its alignment scratch-pad TID machine on top of `Noc::async_read_with_state`.
2. **Drop the DRAM-sharded-output path entirely if it has no live consumers.** If no test or production caller uses sharded-into-DRAM output for this op, deprecating that branch removes 2 of the 3 RED donors at once. The stick-layout reader would still need migration for the ROW_MAJOR path, which IS in active use.
3. **Hold both options until donor migration is part of a wider sharded-kernel pass.** Once `ShardedAddrGen` has a Device 2.0 traits specialization (planned, per `device_api_migration_guide.md` outstanding-work notes), the unblock is mechanical — re-fork the three donors, then re-run the port.

## Donor-kernel finding to feed back into the audit catalog

The pre-port audit's "Device 2.0 DM: GREEN" verdict for this op was reached without spot-checking each cross-op kernel for its Device 2.0 state. Suggest adding a step to the recipe's audit pass:

- **For each kernel-source path in the factory's `kernel_source = "..."` lines, grep the target file for `ShardedAddrGen|InterleavedAddrGen|noc_async_(read|write)[^_]` and `^\s*Noc\b|CircularBuffer\s+cb_|TensorAccessor\(`. If the first set hits and the second set doesn't, the donor is still Device 1.0 — the audit cannot mark Device 2.0 GREEN.**

This op is the witnessing example: 3 of its 4 paths slip past a structural prerequisite the audit's surface scan didn't catch.

## Plan when the port resumes (preserved for reuse)

When the donors are GREEN, the port follows the same shape as `sharded_to_interleaved`:

- Single factory, single `create_program_spec` returning `ProgramArtifacts`.
- `INPUT` and `OUTPUT` TensorParameters; OUTPUT borrows the bound DFB when output is sharded-L1 (as legacy does via `cb.buffer = dst_buffer`).
- The 4-path branching becomes a `dataflow_kernel_specs` switch on `layout × dst_buffer_type`, with `DFBBinding` choosing among `OUT_DFB` / `SRC_DFB` / DFB-less depending on convert_df.
- Reuse the already-forked `reader_unary_sharded_metal2.cpp` (eltwise/unary family — created during `sharded_to_interleaved` port) for the TILE path; donor migration produces the ROW_MAJOR-reader replacement.
- Fork `writer_unary_sharded.cpp → _metal2` (in-family; trivial fork).
- Fork `eltwise_copy.cpp → _metal2` (already done during s2i port — reuse).
