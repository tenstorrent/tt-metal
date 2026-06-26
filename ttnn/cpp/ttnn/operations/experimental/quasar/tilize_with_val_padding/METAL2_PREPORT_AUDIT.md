# Metal 2.0 Pre-Port Feasibility Audit — `experimental/quasar/tilize_with_val_padding`

**Op:** `ttnn::operations::experimental::quasar::{tilize_with_val_padding, tilize_with_zero_padding}` (device op `ttnn::prim::qsr::TilizeWithValPaddingDeviceOperation`)
**Audited against:** `port_op_to_metal2_audit.md` @ `origin/akertesz/metal2-documentation`
**Factories in scope:** 4 — `tilize_with_val_padding_multi_core_{default,block_interleaved,sharded}_program_factory`, `_single_core_program_factory` (+ `factory_helper.cpp`).
**Kernels in scope:** readers `reader_unary_pad_dims_split_rows{,_multicore}.cpp`, `reader_unary_pad_height_width_sharded.cpp`, `reader_unary_pad_multicore_both_dims.cpp`; copied-in compute `tilize.cpp`, `tilize_wh.cpp`; copied-in writers `writer_unary_sharded.cpp`, `writer_unary_interleaved_start_id{,_wh}.cpp`.

## Verdict: **GREEN** — port feasible

## Subjects
1. **Prerequisites — ProgramDescriptor (GATE): GREEN.** `create_descriptor()` factory concept.
2. **Prerequisites — Device 2.0 (GATE): GREEN.** All referenced kernels (op-owned readers + copied-in compute/writers from `tilize`, `eltwise/unary`, `sharded`, global `kernel/compute/tilize.cpp`) are Device-2.0 compliant. Verified: no Device-1.0 idioms.
3. **Feature compatibility: GREEN.** No UNSUPPORTED feature (single input; fixed-index CTAs → CTA varargs N/A; no GlobalCB/GlobalSem/`address_offset`/`UpdateCircularBuffer*`). No caveated LANDED features.
4. **TensorAccessor handling: PORT WORK (Case 1).** Input/output via `TensorAccessor`; no Case-2, no address-RTA smuggling. Mechanical binding conversion.
5. **DFB endpoint legality (SPSC): GREEN.** Standard reader→CB→compute→CB→writer; 1-producer/1-consumer per node. Per-config re-trace advisable for the sharded factory.
6. **Out-of-directory coupling: FYI-U.** Donor kernels copied into op dir; host uses shared `common.hpp` helpers (`is_enough_space`, `pack_two_uint16_into_uint32`, `MassagedOperation`, `squeeze_*`). No external kernel references.
7. **Custom program hash: N/A.** No override.
8. **Other signals: none.**

## Routing
- GATEs cleared. PORT WORK: TensorAccessor Case-1 bindings across all factories.
