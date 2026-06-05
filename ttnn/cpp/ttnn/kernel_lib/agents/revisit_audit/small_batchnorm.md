# Revisit Audit — small_batchnorm batch

Read-only audit for MISSED migrations to `compute_kernel_lib::eltwise_chain`.

Capabilities reference: `eltwise_chain.hpp`, `eltwise_convenience.hpp`, `eltwise_math.hpp`
(read in full / skimmed). Chain elements: CopyTile, BinaryFpu(Add/Sub/Mul),
DestReuseBinary, UnaryBcast, PackTile, SFPU DEST-only op structs, Fill*/Rand. Lifecycles
incl. CallerManaged (no wait/pop edge — caller owns), HeldStream, Streaming, Bulk, Chunked.

## Result: ALL FOUR KERNELS ALREADY FULLY MIGRATED — 0 missed stages.

Every CB-touching eltwise compute stage in all four kernels is already expressed through
`eltwise_chain` (or its convenience wrapper). The only raw CB ops remaining are
`cb_wait_front` / `cb_pop_front` calls that are the deliberate caller-side half of a
documented `InputLifecycle::CallerManaged` contract, plus caller-owned BIG init
(`init_sfpu` / `unary_op_init_common` / `binary_op_init_common`) which the chain
contract (eltwise_chain.hpp @file block) explicitly leaves to the caller.

---

## 1. eltwise_where_no_bcast.cpp

File: `ttnn/cpp/ttnn/operations/eltwise/binary_ng/device/kernels/compute/eltwise_where_no_bcast.cpp`

- Lines 67: `init_sfpu(cb_cond, cb_out)` — caller-owned BIG init (per chain contract, NOT a stage).
- Lines 69-99: single `eltwise_chain` over `EltwiseShape::tiles(num_tiles, num_tiles_per_cycle)`:
  CopyTile(cond→D0, Chunked, Block) → CopyTile(tensor→D1/D2, Chunked, Block) →
  OptionalChainElement FillInt/FillBitcast(scalar→other slot) → Where<DF,D0,D1,D2,D0> →
  PackTile(cb_out, Chunked).
- Verdict: **fully migrated. 0 missed stages.** Block + Chunked lifecycle + Optional fill +
  ternary SFPU all already inside the chain.

## 2. ternary_addcmul_int_sfpu.cpp

File: `ttnn/cpp/ttnn/operations/eltwise/ternary/device/kernels/compute/ternary_addcmul_int_sfpu.cpp`

- Line 31: `unary_op_init_common` — caller-owned BIG init (NOT a stage).
- Lines 33-42: single `eltwise_chain(num_tiles, ...)`:
  CopyTile(in0→D0) → CopyTile(in1→D1) → CopyTile(in2→D2) → FillInt(scalar→D3) →
  MulIntBinary(D3*D1→D3) → MulIntBinary(D3*D2→D2) → AddIntBinary(D0+D2→D0) → PackTile(cb_out).
- Verdict: **fully migrated. 0 missed stages.** Int SFPU binary ops + fill all in chain.

## 3. batch_norm_kernel.cpp

File: `ttnn/cpp/ttnn/operations/normalization/batch_norm/device/kernels/compute/batch_norm_kernel.cpp`

- Line 145: `binary_op_init_common` — caller-owned BIG init (NOT a stage).
- Lines 49-59: Stage-1 `eltwise_chain(1, ...)`: BinaryFpu(batch_var + eps, Bulk/CallerManaged) →
  Rsqrt → PackTile(cb_den). Migrated.
- Lines 110: Stage-2..4 fused `eltwise_chain(inner_count, ...)`:
  BinaryFpu Sub(cb_other - cb_bcast, Streaming/CallerManaged) → DestReuseBinary Mul(cb_den) →
  Optional DestReuseBinary Mul(cb_weight) → Optional DestReuseBinary Add(cb_bias) →
  PackTile(cb_output_0). Migrated.
- Lines 65-72 / 112-119 (`cb_wait_front`/`cb_pop_front` for cb_bcast/cb_den/cb_weight/cb_bias),
  and 150 / 179 (`cb_wait_front`/`cb_pop_front` cb_eps in kernel_main): these are the
  CALLER side of `InputLifecycle::CallerManaged` operands held across the chain call
  (eps held across the whole kernel; bcast/den/weight/bias held across the stage-2 chain).
  Per the CallerManaged contract (eltwise_chain.hpp:169,234) the chain intentionally emits
  no wait/pop on these — the caller owns those edges. NOT missed stages; this is the
  designed split for persistent broadcast operands.
- Verdict: **fully migrated. 0 missed stages.**

## 4. prod_nc.cpp

File: `ttnn/cpp/ttnn/operations/reduction/prod/device/kernels/compute/prod_nc.cpp`

- Line 21: `binary_op_init_common` — caller-owned BIG init (NOT a stage).
- Line 22: `cb_in1_obj.wait_front(onetile)` — caller-side wait for the ones-scaler operand,
  consumed via `InputLifecycle::HeldStream` in the chain (chain waits per-iter, no pop; the
  held scaler is the B operand). This is the documented CircularBuffer-method caller edge.
- Lines 37-58: product-reduction expressed entirely via the `mul<>` convenience wrapper:
  - num_input_tiles==1: `mul(cb_in0, cb_in1→cb_out0, Streaming/HeldStream)`.
  - seed: `mul(cb_in0, cb_in1→cb_intermed0, Streaming/HeldStream)`.
  - middle (n-2 iters): `mul(cb_in0, cb_intermed0→cb_intermed0)` — running partial reload.
  - final: `mul(cb_in0, cb_intermed0→cb_out0)`.
- Note: the "reduction by product" is implemented as a reload-and-accumulate FPU-mul loop
  through cb_intermed0 (an L1 intermediate CB), NOT an in-DEST cross-iter accumulation — so
  it is fully chainable and already chained. Not a reduce_tile / OUT-OF-SCOPE case.
- Verdict: **fully migrated. 0 missed stages.**

---

## Summary table

| Kernel | Missed stages | Notes |
|---|---|---|
| eltwise_where_no_bcast.cpp | 0 | single chain (Block/Chunked + Where ternary + Optional fill) |
| ternary_addcmul_int_sfpu.cpp | 0 | single chain (int SFPU mul/add + fill) |
| batch_norm_kernel.cpp | 0 | 2 chains; remaining cb_wait/pop are CallerManaged caller-edges |
| prod_nc.cpp | 0 | mul<> wrappers; reload-accumulate via cb_intermed0, not in-DEST |

No MIGRATABLE remaining stages. No BLOCKED stages. No OUT-OF-SCOPE stages.
