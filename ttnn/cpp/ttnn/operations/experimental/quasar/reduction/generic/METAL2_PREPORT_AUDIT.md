# Metal 2.0 Pre-Port Feasibility Audit — `experimental/quasar/reduction/generic`

**Ops (bundled — shared factories/kernels):** `ttnn::prim::qsr::ReduceDeviceOperation` + `ttnn::prim::qsr::WelfordReduceDeviceOperation`. Host entry used by quasar `avg_pool2d` is `pool_sum` (→ `ReduceDeviceOperation`, Sum). Bundled per the audit's multi-DeviceOperation rule; per-DeviceOperation attribution retained below.
**Audited against:** `port_op_to_metal2_audit.md` @ `origin/akertesz/metal2-documentation`
**Factories in scope:** `reduce_op_multi_core_h`, `reduce_op_multi_core_w`, `reduce_op_single_core_hw` (ReduceDeviceOperation); `welford_reduce_program_factory` (WelfordReduceDeviceOperation).
**Kernels in scope:** compute `reduce.cpp`, `reduce_h_neg.cpp`, `reduce_hw_neg.cpp`, `reduce_w_neg.cpp`, `reduce_rm.cpp`, `welford_reduce_{h,hw,w}.cpp`; dataflow `reader_unary_reduce_rm.cpp`, `reader_unary_reduce_universal_start_id.cpp`, `reader_unary_transpose_wh_*`, `writer_reduce_rm_scalar.cpp`, `writer_welford_hw.cpp`; copied-in `writer_unary_sharded.cpp`, `writer_unary_interleaved_start_id.cpp`.

## Verdict: **GREEN** — port feasible (both DeviceOperations)

## Subjects
1. **Prerequisites — ProgramDescriptor (GATE): GREEN.** Both DeviceOperations use the `create_descriptor()` factory concept.
2. **Prerequisites — Device 2.0 (GATE): GREEN.** All referenced kernels Device-2.0 compliant. The two stray free `cb_pop_front` calls (`reduce.cpp`, `reduce_w_neg.cpp`) were migrated to `CircularBuffer.pop_front` this session; no Device-1.0 idioms remain in any referenced kernel (compute or dataflow), incl. copied-in donor writers.
3. **Feature compatibility: GREEN.** No UNSUPPORTED feature — `pool_sum`/`reduce` take a single input + scalar (CTA varargs N/A); no GlobalCB/GlobalSemaphore/`address_offset`/`UpdateCircularBuffer*`. Welford path likewise. No caveated LANDED features.
4. **TensorAccessor handling: PORT WORK (Case 1).** Dataflow readers/writers access via `TensorAccessor`; no Case-2 raw pointer, no `buffer()->address()` RTA smuggling. Compute kernels consume CBs only (out of TA scope). Mechanical binding conversion.
5. **DFB endpoint legality (SPSC): GREEN.** reader→CB→compute→CB→writer; scaler CB is 1-producer/1-consumer (waited once, popped once — balanced after this session's fix). Per-config re-trace advisable for the sharded/transpose-partitioned readers.
6. **Out-of-directory coupling: FYI-U.** `reduction_common::ReduceType` is a shared top-level namespace (reused, not copied). Host `reduce_impl` calls non-quasar `ttnn::reshape/transpose/permute/tilize_with_val_padding` on its multi-dim/welford paths (NOT on `pool_sum`'s single-H-reduce hot path) — host-level, not kernel references; deferred-quasar-routing noted but not a gate. Donor writer kernels copied into op dir.
7. **Custom program hash: N/A.** No override on either DeviceOperation.
8. **Other signals: none.** `reduce_op_utils_qsr` namespace (renamed from `reduce_op_utils` to avoid link collision) is incidental; not a port concern.

## Routing
- GATEs cleared for both DeviceOperations. PORT WORK: TensorAccessor Case-1 bindings across reduce + welford factories.
- Recipe note: bundled report covers both DeviceOperations; downstream per-op accounting can split via the attribution above.
