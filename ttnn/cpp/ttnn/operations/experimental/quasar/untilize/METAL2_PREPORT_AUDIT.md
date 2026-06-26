# Metal 2.0 Pre-Port Feasibility Audit — `experimental/quasar/untilize`

**Op:** `ttnn::operations::experimental::quasar::untilize` (device op `ttnn::prim::qsr::UntilizeDeviceOperation`)
**Audited against:** `port_op_to_metal2_audit.md` @ `origin/akertesz/metal2-documentation`
**Factories in scope:** 8 — `untilize_multi_core_program_factory`, `_block`, `_parallelize_column`, `_sub_core_grids`, `_nd_shard_input`, `_input_and_output_{shard,nd_shard}_type_and_shard_spec_identical`, `_single_core`.
**Kernels in scope:** op-owned writers `writer_unary_stick_layout_split_rows_{single_core,multi_core,multi_core_nd_shard,interleaved_parallel_columns}.cpp`; copied-in readers `reader_unary_{start_id,sharded_blocks,interleaved_start_id,interleaved_wh_multicore,sharded,nd_sharded_blocks}.cpp`; copied-in `writer_unary_sharded.cpp`; compute `untilize.cpp`, `untilize_wh.cpp`, `untilize_variable_num_blocks.cpp`.

## Verdict: **GREEN** — port feasible

## Subjects
1. **Prerequisites — ProgramDescriptor (GATE): GREEN.** `create_descriptor()` factory concept; no `override_runtime_arguments`, no imperative builder.
2. **Prerequisites — Device 2.0 (GATE): GREEN.** Every referenced kernel — op-owned and copied-in donor (from `eltwise/unary`, `sharded`) — is Device-2.0 compliant (`Noc`/`CircularBuffer` wrappers; no Device-1.0 addr-gen or raw `noc_async_*`). Sources copied from were already migrated; verified clean.
3. **Feature compatibility: GREEN.** No UNSUPPORTED feature (GlobalCircularBuffer / GlobalSemaphore / `address_offset`≠0 / `UpdateCircularBuffer*` / CTA varargs all N/A — single input tensor, fixed-index CTAs). No caveated LANDED features.
4. **TensorAccessor handling: PORT WORK (Case 1).** Input/output reach kernels via `TensorAccessor`; no Case-2 raw pointer, no `buffer()->address()` RTA smuggling. Mechanical `TensorParameter`/`TensorBinding` conversion.
5. **DFB endpoint legality (SPSC): GREEN.** Reader→CB→compute→CB→writer pipelines; each CB 1-producer/1-consumer per node. No hidden second writer observed (per-config re-trace advisable for the sharded factories at port time).
6. **Out-of-directory coupling: FYI-U.** Donor kernels copied **into** the op dir (self-contained); host uses shared `common.hpp` helpers (`is_enough_space`, `get_max_l1_space`, `MassagedOperation`) and `common_tm_bw_model`. No external kernel references.
7. **Custom program hash: N/A.** No `compute_program_hash` override.
8. **Other signals: none.** RTA varargs present in the split-rows writers (run-length block reps) — supported via Metal 2.0 kernel-side RTA varargs (FYI-P, not a gate).

## Routing
- GATEs cleared. PORT WORK: TensorAccessor Case-1 bindings (input, output) across all 8 factories. FYI-P: writer RTA varargs.
