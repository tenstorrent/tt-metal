# Metal 2.0 Pre-Port Feasibility Audit — `experimental/quasar/reshape_view`

**Op:** `ttnn::operations::experimental::quasar::reshape` (device op `ttnn::prim::qsr::ReshapeViewDeviceOperation`)
**Audited against:** `port_op_to_metal2_audit.md` @ `origin/akertesz/metal2-documentation`
**Factories in scope:** `reshape_rm_program_factory.cpp` (`ReshapeViewRMProgramFactory`), `reshape_tiled_program_factory.cpp` (`ReshapeViewTiledProgramFactory`)
**Kernels in scope (referenced by factories):** `device/device/rm_reshape_interleaved.cpp`, `device/device/dataflow/reader_reshape_tiled.cpp`, `device/device/dataflow/writer_reshape_tiled.cpp`

## Verdict: **GREEN** — port feasible

All hard gates clear. Port work and one FYI noted below.

## Subjects

1. **Prerequisites — ProgramDescriptor (GATE): GREEN.** Op uses the ProgramDescriptor factory concept — `create_descriptor()` returning `ProgramDescriptor`; no `override_runtime_arguments()`, no imperative `host_api.hpp` (`CreateKernel`/`CreateCircularBuffer`).
2. **Prerequisites — Device 2.0 (GATE): GREEN.** All 3 referenced kernels are Device-2.0 compliant (`experimental::Noc`, `CircularBuffer` wrappers; reads/writes via `enhanced_noc_async_*`/`tt_memmove(noc,...)`). Migrated this session — no Device-1.0 idioms (`InterleavedAddrGen*`, raw `noc_async_read/write`, bank-id addr-gen) remain.
3. **Feature compatibility: GREEN.** No UNSUPPORTED Appendix-A feature in use — N/A for GlobalCircularBuffer, GlobalSemaphore, CB `address_offset`≠0, per-execution CB-size updates (`UpdateCircularBuffer*`/`UpdateDynamicCircularBuffer*`), CTA varargs (fixed single input + mapping tensor; kernels read CTAs at fixed indices). No LANDED-with-caveat features (no borrowed-mem DFB, aliased CB, dynamic TA, non-zero sem init).
4. **TensorAccessor handling: PORT WORK (Case 1).** Both tensors (input + mapping; output) reach kernels via `TensorAccessor(args, addr)` (`TensorAccessorArgs<...>`). No raw-pointer Case 2; no `buffer()->address()` smuggled through runtime args. Port = express as `TensorParameter`/`TensorBinding`, kernel builds `TensorAccessor(ta::name)`. Mechanical.
5. **DFB endpoint legality (SPSC): GREEN with one FYI-P.** `mapping_cb` and `input_cb` are each 1-producer (reader kernel) / 1-consumer (writer kernel) — SPSC-legal per node. `working_cb` (tiled writer scratch) is **single-ended** (writer `reserve_back`+`push_back`, no FIFO consumer) → sync-free/single-ended CB; port applies the sanctioned interim workaround (DM fabricated consumer). FYI-P, not a gate. No hidden second writer observed, but a per-config SPSC re-trace at port time is advisable.
6. **Out-of-directory coupling: FYI-U.** Host `reshape.cpp` calls quasar `sharded_to_interleaved`/`interleaved_to_sharded` and shared `data_movement::detail::{reshape_rm,reshape_tiled,infer_dims_for_reshape}` + `common.hpp` helpers; kernels include `data_movement/common/kernels/common.hpp`. No cross-op *kernel* borrowing (all 3 kernels are op-owned). Donor helpers are host-side, not gates.
7. **Custom program hash: PORT WORK.** `ReshapeViewDeviceOperation::compute_program_hash` is defined (`reshape_device_operation.hpp/.cpp`). Port deletes it → framework default.
8. **Other signals: none.** No RTA varargs; no dead CBs; no suspicious constants flagged.

## Routing
- GATEs cleared → porter brief may issue.
- PORT WORK: TensorAccessor Case-1 bindings (input, mapping, output); delete custom `compute_program_hash`; single-ended `working_cb` interim workaround.
- Recipe notes: none.
