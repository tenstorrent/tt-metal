# binary_ng — Metal 2.0 Port Report

**STATUS: PARTIAL (widened).** The Metal 2.0 spec factory
(`BinaryNgDeviceOperation::ProgramSpecFactory`, in
`device/binary_ng_program_factory_m2.cpp`) previously covered exactly ONE path
(no-broadcast × tile × FPU × tensor-b × interleaved × no-activation × plain ADD/SUB/MUL).
This pass widens Metal 2.0 coverage to **three** operand/compute axes on the same
no-broadcast/tile/interleaved/no-activation/no-typecast/plain-op base:

1. tensor-b present × **FPU**  (unchanged from before)
2. tensor-b present × **SFPU** (new)
3. **scalar-b** (no tensor b) × FPU (new)

All remaining paths stay on the legacy `ProgramFactory::create_descriptor`
(`device/binary_ng_program_factory.cpp`). `select_program_factory()` routes only the three
modeled cases to the spec factory; the gate is conservative — anything not modeled falls through.
The op builds/runs as a mixed-concept device op (`program_factory_t` holds one legacy
`ProgramDescriptorFactoryConcept` factory and one Metal 2.0 `MetalV2FactoryConcept` factory),
dispatched per-call.

The custom `compute_program_hash` is **kept** (shared across both factories; not deleted).

## Factory chosen
`BinaryNgDeviceOperation::ProgramSpecFactory` (`MetalV2FactoryConcept`), one
`create_program_spec` that branches on `is_sfpu` and `b_present` to pick kernels, DFB sizing,
and per-core RTA layout. The legacy `ProgramFactory` handles everything else; the
`program_factory_t` variant dispatches per-factory at runtime.

## Kernels — ported vs forked
All edited kernels are FORKED to `*_m2.cpp` (the legacy copies are shared with the legacy factory
and other broadcast/RM paths, so in-place editing would break them). Conversion is mechanical:
CB id → `dfb::`, `TensorAccessorArgs`/addr → `ta::`, positional `get_arg_val` /
`get_compile_time_arg_val` → `get_arg(args::)`. Logic, `#ifdef`s, loop bounds unchanged.

| Forked file (`device/...`) | Forked from | Bound by |
|---|---|---|
| `kernels_ng/dataflow/reader_interleaved_no_bcast_m2.cpp` | `kernels_ng/dataflow/reader_interleaved_no_bcast.cpp` | tensor-b reader (FPU+SFPU) — *pre-existing* |
| `kernels_ng/dataflow/writer_interleaved_no_bcast_m2.cpp` | `kernels_ng/dataflow/writer_interleaved_no_bcast.cpp` | tensor-b writer (FPU+SFPU) — *pre-existing* |
| `kernels/compute/eltwise_binary_no_bcast_m2.cpp` | `kernels/compute/eltwise_binary_no_bcast.cpp` | tensor-b FPU compute — *pre-existing* |
| `kernels/compute/eltwise_binary_sfpu_no_bcast_m2.cpp` | `kernels/compute/eltwise_binary_sfpu_no_bcast.cpp` | tensor-b SFPU compute — **NEW** |
| `kernels/dataflow/reader_interleaved_no_bcast_m2.cpp` | `kernels/dataflow/reader_interleaved_no_bcast.cpp` (single-tensor `ReaderNoBcast`) | scalar-b reader — **NEW** |
| `kernels/dataflow/writer_interleaved_scalar_m2.cpp` | `kernels/dataflow/writer_interleaved_scalar.cpp` | scalar-b writer (fills scalar tile + writes dst) — **NEW** |
| `kernels/compute/eltwise_binary_scalar_m2.cpp` | `kernels/compute/eltwise_binary_scalar.cpp` | scalar-b FPU compute — **NEW** |

Note: two distinct `reader_interleaved_no_bcast_m2.cpp` files exist — one under `kernels_ng/`
(two-tensor reader) and one under `kernels/` (single-tensor `ReaderNoBcast`); they are different
kernels with the same base name, mirroring the legacy layout.

## DFB / binding shape
- `src` (c_0), `src_b` (c_1), `dst` (c_2). `src_b` is double-buffered (2 entries) on the
  tensor-b path and single-entry (1) on the scalar-b path, matching legacy CB page counts.
- tensor-b: reader PRODUCES src+src_b; compute CONSUMES src+src_b, PRODUCES dst; writer CONSUMES dst.
- scalar-b: reader PRODUCES src; **writer PRODUCES src_b** (fills the scalar tile once) and
  CONSUMES dst; compute CONSUMES src+src_b, PRODUCES dst. The src_b producer/consumer endpoints
  (writer/compute) co-reside in each WorkUnitSpec — local-DFB rule satisfied.
- Activation-intermediate DFBs c_3/c_4 (`post_lhs`/`post_rhs`) are `#ifdef HAS_ACTIVATIONS`-gated
  in every compute kernel but **never bound** by this factory: the routed cases are all
  no-activation/no-typecast/plain ADD-SUB-MUL, for which `OpConfig` injects no process activations
  (verified in `binary_ng_utils.cpp`), so the conditional branches compile out cleanly.

## Findings / notes for downstream
- **SFPU `unpack_to_dest_mode`.** The legacy `ComputeConfigDescriptor` set `UnpackToDestFp32` on
  the operand CBs unconditionally for non-POWER SFPU ops, regardless of `fp32_dest_acc_en`. The
  Metal 2.0 `ComputeHardwareConfig` **rejects** `UnpackToDestFp32` unless `fp32_dest_acc_en == true`
  (header contract). The mode is hardware-inert outside the Float32+consumer+fp32-dest triple, so
  this port sets `UnpackToDestFp32` on `src`/`src_b` **only when `fp32_dest_acc_en` is true** and
  `Default` otherwise — behavior-preserving (only plain ADD/SUB/MUL route here, never POWER, so the
  per-operand-dtype POWER branch is irrelevant).
- **`compute_program_hash` kept** (shared; deliberately not deleted).
- **Per-core dummy/no-op RTA emission dropped.** Metal 2.0 derives the active node set from
  WorkUnitSpec `target_nodes`; unused cores get no per-core args (legacy emitted zero-filled dummy
  arg vectors to size the allocation). Behavior-equivalent.

## Remaining on legacy (NOT yet on Metal 2.0) — for a follow-up pass
- **Row-major (RM) no-bcast** (priority #1). Entirely different RTA layout (26 reader / 14 writer
  slots: page sizes, alignments, stride bytes, row blocks), a different reader/writer pair
  (`reader_interleaved_rm_no_bcast.cpp`, `writer_interleaved_rm_no_bcast.cpp`,
  `reader_interleaved_rm_scalar_op.cpp`), and DFB producer roles that differ from the tile path.
  Larger and faithfulness-risky; deferred so the shipped subset stays correct.
- **Simple broadcast (ROW / COL / SCALAR) on tile** (priority #4). Requires the extra CBs c_5/c_6
  (`cb_llk_post`), the `BCAST_INPUT` define, `calculate_compute_kernel_args` (freq/counter RTAs),
  same-FIFO aliasing (`cb_pre_lhs = cb_llk_post`), and — critically — the ~80-line `use_llk_bcast`
  software-vs-LLK fallback decision (arch/dtype-specific hangs: BH COL bf16→fp32, BH MOVB2D fp32,
  UInt16 SCALAR relational #36217, EXP/BFP rounding). Faithful reproduction is substantial;
  deferred to keep the subset correct.
- where-op, quant-op, scalar-b-on-SFPU, sharded (interleaved-only here), activations, typecast,
  non-plain ops (DIV, comparisons, bitwise, etc.), `ROW_A_COL_B` / `ROW_B_COL_A` mixed broadcast,
  ISCLOSE.

## Routed-cases matrix (broadcast × layout × compute × operand)

Legend: **M2** = on Metal 2.0 spec factory; legacy = on `ProgramFactory::create_descriptor`.
All M2 cells additionally require: interleaved (not sharded), no activations, no typecast
(a.dtype == out.dtype), and plain op (ADD/SUB/MUL). Any cell not meeting those → legacy.

| Broadcast | Layout | Compute | Operand | Routed to |
|---|---|---|---|---|
| NONE | TILE | FPU  | tensor-b | **M2** |
| NONE | TILE | SFPU | tensor-b | **M2** |
| NONE | TILE | FPU  | scalar-b | **M2** |
| NONE | TILE | SFPU | scalar-b | legacy |
| NONE | TILE | where/quant | any | legacy |
| NONE | TILE | any (activations / typecast / non-plain) | any | legacy |
| NONE | TILE | any | sharded | legacy |
| NONE | ROW_MAJOR | any | any | legacy |
| ROW_A / ROW_B | TILE | any | any | legacy |
| COL_A / COL_B | TILE | any | any | legacy |
| SCALAR_A / SCALAR_B | TILE | any | any | legacy |
| ROW_A_COL_B / ROW_B_COL_A | TILE | any | any | legacy |
| any | ROW_MAJOR | any | any | legacy |
| any | any | any | sharded | legacy |
