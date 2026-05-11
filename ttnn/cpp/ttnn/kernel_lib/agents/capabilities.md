# Capabilities: eltwise_chain (kernel-side helper library)

> Last updated: 2026-05-11 by incremental-verifier (Type-1 sweep + Reg A/B/C verification pass)
> Branch: `astancov/eltwise_run7_refined` HEAD `7060e1245a3`
> Prior pass: 2026-05-08 (run7 refinement) at `3b0cc6026e8`

This is a **kernel-side helper library**, not a user-facing TTNN op. The capabilities below document what kinds of chain shapes, element types, traits, and modes the helper supports as of the run7 refinement.

---

## Chain shapes supported

| Shape | Helper API | Notes |
|---|---|---|
| Streaming unary (`CopyTile → SFPU → PackTile`) | `eltwise_chain(n, elts...)` or `eltwise_chain_with_init(n, elts...)` | Native. Doxygen example #1, #3. |
| Streaming binary FPU (`BinaryFpu → PackTile`) | `eltwise_chain(...)` | Native. Doxygen example #2. |
| Streaming binary SFPU (`AddBinary` / `SubBinary` / `MulBinary` / `DivBinary`) | `eltwise_chain(...)` | SFPU op-structs in `eltwise_binary_sfpu.hpp`. |
| Block-mode FPU (`BlockBinaryFpu` + `BlockPackTile`) | `eltwise_chain(...)` | Multi-tile DEST scratch. Block elements participate in same prev-CB / prev-fp32 fold (D7). |
| Dest-reuse binary (`DestReuseBinary`) | `eltwise_chain(...)` | Reads + writes same DEST slot. CARRY for fp32. |
| Unary broadcast (`UnaryBcast`) | `eltwise_chain(...)` | `BroadcastDim::{Row,Col,Scalar,None}`. CARRY for fp32. |
| Fan-out (one input → N outputs) | `eltwise_chain(...)` | Doxygen example #4. |
| Multi-stage (different PACK output CB per stage) | Explicit per-stage `compute_kernel_hw_startup` + `eltwise_chain(...)` | NOT covered by `eltwise_chain_with_init`. See `logit_kernel.cpp` for canonical multi-stage pattern. |
| Fill-only chain (e.g. `FillScalar` + `WhereSfpu` + `PackTile`) | `eltwise_chain_with_init(...)` | Reader-less fallback boots `compute_kernel_hw_startup(cb_out, cb_out, cb_out)` per `eltwise_chain.inl:1241`. |
| Optional / conditional element pairs | `OptionalChainElement<COND, Inner>` | Compile-time predicate; `<false, Inner>` is tag-only no-op. See `where_tss_kernel.cpp`. |

---

## Element types — full inventory

### CARRY list (DEST-format-sensitive — carry `EnableFp32DestAcc` template param)

| Element | Header | Notes |
|---|---|---|
| `BinaryFpu<CbA, CbB, CbOut, Op, Bcast, DfReconfig, APolicy, BPolicy, Index, DstSlot, EnableFp32DestAcc>` | `eltwise_chain.hpp:568` | Q4 collapse: single `Index` (was per-side `AIndex`/`BIndex`). |
| `DestReuseBinary<...>` | `eltwise_chain.hpp:582` | |
| `UnaryBcast<...>` | `eltwise_chain.hpp:594` | Bcast = Row, Col, Scalar, or None. |
| `PackTile<...>` | `eltwise_chain.hpp:604` | |
| `PackTileBlock<...>` | `eltwise_chain.hpp:613` | |
| `BlockBinaryFpu<CbA, CbB, Op, BlockSize, CbOut, BaseDst, Bcast, DF, APolicy, BPolicy, Index, BWaitTiles, EnableFp32DestAccV>` | `eltwise_block.hpp:139` | Q4 collapse complete. |
| `BlockPackTile<Cb, BlockSize, BaseDst, Policy, Reconfig, EnableFp32DestAccV>` | `eltwise_block.hpp:260` | |

### SKIP list (no `EnableFp32DestAcc` template param — fold pass-through)

| Element | Header | Notes |
|---|---|---|
| `CopyTile<Cb, DstSlot, Policy, IndexMode, Reconfig>` | `eltwise_chain.hpp:560` | |
| `BlockCopyTile<Cb, BlockSize, BaseDst, Policy, Reconfig>` | `eltwise_block.hpp:76` | D7 — no `OldCb*`. |
| `FillScalar<DstSlot>` | `eltwise_fill.hpp:20` | |
| `FillInt<DF, DstSlot>` | `eltwise_fill.hpp:31` | |
| `FillBitcast<DstSlot>` | `eltwise_fill.hpp:43` | |
| `RandTile<DstSlot, Seed>` | `eltwise_rand.hpp:23` | NTTP seed (F-UX-11 fix carried forward). |
| `OptionalChainElement<COND, Inner>` | `eltwise_optional.hpp:60` | Forwards to Inner; `<false, Inner>` is tag-only no-op. |

### Unary SFPU op-structs (`api/compute/common.h` and operation kernels)

Many SFPU ops (`Exp`, `Tanh`, `Sigmoid`, `Eqz`, `Mish`, `Logit`, etc.) implemented as struct types satisfying `DestOnlyTag`. They're SKIP for the D6 fold today (see deferred F-UX-9 below).

### SFPU binary op-structs in `eltwise_binary_sfpu.hpp`

`AddBinary`, `SubBinary`, `MulBinary`, `DivBinary`, `WhereSfpu`. Currently SKIP via SFINAE under F-UX-9 deferral (see "Deferred / not yet implemented" below).

---

## Traits / policies

| Trait | Values | Notes |
|---|---|---|
| `CopyTilePolicy` | `WaitAndPop`, `WaitNoPop`, `NoWaitPop`, `WaitUpfrontPopAtEnd` | Unified across CopyTile and BinaryFpu A/B sides. |
| `PackTilePolicy` | `PerTileReserveAndPush`, `UpfrontReservePushAtEnd` | |
| `CbIndexMode` | `FirstTile`, `BlockIter`, `Pinned`, `Absolute` | Q4 collapse → single `Index` per element (was per-side AIndex/BIndex on `BinaryFpu` / `BlockBinaryFpu`). |
| `BinaryFpuOp` | `Add`, `Sub`, `Mul` | |
| `BroadcastDim` | `None`, `Row`, `Col`, `Scalar` | |
| `BinaryDataFormatReconfig` | `None`, `Input`, `Output`, `InputAndOutput` | |
| `CopyTileReconfig` | `None`, `Input` | |
| `PackTileReconfig` | `None`, `Output`, `OutputConditional` | OutputConditional currently emits same as Output; future two-arg form deferred (Issue 1). |
| `UnaryBcastReconfig` | `None`, `Input` | |
| `DestReuseReconfig` | `None`, `Input` | |
| `DestReuseType` | `SrcA`, `SrcB` | |
| `Dst` | `D0`–`D15` | DEST_AUTO_LIMIT = 16 on Wormhole. |
| `EnableFp32DestAcc` | `false` (default), `true` | Per-CARRY-element. `static_assert(!EnableFp32DestAcc \|\| DST_ACCUM_MODE)` requires kernel built with `FP32_DEST_ACC_EN`. |

---

## Caller responsibilities (D5 + D8 contract)

The chain helper does **not** wrap BIG inits. Caller responsibilities:

| Init | Owner | When |
|---|---|---|
| `compute_kernel_hw_startup(cb_a, cb_b, cb_out)` | **caller** | First statement of `MAIN()` for chains that read/write CBs. Multi-stage: stage 1 at top of `MAIN()`, stages 2+ immediately before that stage's chain call. Mid-`MAIN()` is **undefined** per `compute_kernel_hw_startup.h:26-30`. |
| `binary_op_init_common(cb_a, cb_b, cb_out)` | **caller** | When kernel mixes raw binary primitives with chain calls; NOT required for chain-only kernels. |
| `mm_init(...)` / `reduce_init<...>(...)` | **caller** | N/A for eltwise chain (chain is eltwise-only). |
| Per-element bootstrap (`add_tiles_init`, `*_tile_init`, `init_bcast`, `copy_tile_to_dst_init_short`, `reconfig_data_format_*`, `enable_fp32_dest_acc()` / `disable_fp32_dest_acc()`, `tile_regs_*` lifecycle) | **chain** | Fold-driven. Caller does NOT call these. |

The convenience wrapper `eltwise_chain_with_init(num_tiles, elts...)` (in `eltwise_chain.inl:1230`) deduces `(cb_a, cb_b, cb_out)` from the chain element pack and emits `compute_kernel_hw_startup` + `eltwise_chain(...)` for **single-stage** kernels. Multi-stage kernels MUST keep the explicit per-stage pattern.

D8 grep gate (run ad-hoc):

```bash
grep -nE 'init_common|compute_kernel_hw_startup|mm_init|reduce_init' \
     ttnn/cpp/ttnn/kernel_lib/eltwise_{chain.hpp,chain.inl,block.hpp}
```

Expected: `#include "compute_kernel_hw_startup.h"` line in `eltwise_chain.hpp:235`, the convenience wrapper at `eltwise_chain.inl:1244`, and doxygen comment hits. Zero call sites in helper bodies beyond the wrapper.

---

## Q4 collapse — asymmetric-mode caller dispositions

After v6 collapse to single `Index` template param on `BinaryFpu`/`BlockBinaryFpu`, asymmetric-mode callers (where A and B sides need different `CbIndexMode`) must follow one of:

- **(a) Symmetric refit to `Pinned`** — uniform `Pinned` works when A walks pre-waited tiles and B is single-tile pinned at index 0 (Pinned with `tile_idx=0` ≡ FirstTile semantics for B). Adopted by 6 moreh kernels (`moreh_adam`, `moreh_softmax_backward_{h,w,c_large,h_large,w_large}`).
- **(b) Helper alternative** — use `DestReuseBinary` or fold the asymmetry into a different element. Not exercised in run7.
- **(c) Regress to raw LLK** — explicit `add_tiles_init` + per-tile `tile_regs_acquire` / `add_tiles(idx_a, idx_b, dst)` / `pack_tile` / release. Adopted by 2 kernels: `deepseek_grouped_gate.cpp::add_bias` (lines 39–66) and `eltwise_binary_scalar.cpp` no-act fast path (lines 74–112). Both inline-commented with the Q4 reasoning.
- **(d) Drop migration** — kernel stays on raw LLK. Not exercised in run7.

---

## Deferred / not yet implemented

| Item | Tracker | Status |
|---|---|---|
| `BinarySfpu` family (`AddBinary`/`SubBinary`/`MulBinary`/`DivBinary` + `WhereSfpu`) — D6 CARRY-list registration | F-UX-9 | Deferred; documented at `eltwise_chain.hpp:86`. SFINAE pass-through today. |
| `pack_reconfig_data_format` two-arg form for `PackTileReconfig::OutputConditional` | Open issue 1 | Deferred; doxygen at `eltwise_chain.hpp:217–218`. |
| `BlockBinaryFpuScalarB` (asymmetric per-side index for block path) | Future | Not started. Would cover the 2 raw-LLK regressed kernels. |
| Cumulative wait policy (`cb_wait_front(base + i)`) | Out of scope | Documented in non-goals. |
| Mid-loop dtype swaps | Out of scope | Reconfig is entry-time per chain element. |
| L1 accumulation (`pack_reconfig_l1_acc`), pack-relu, pack-rows | Future | Future PackTilePolicy extensions. |
| Held-DEST patterns | Out of scope | Zero TSV evidence. |
| `acquire_dst/release_dst` and `ACQ()/REL()` macros | Out of scope | Modern dst-sync only. Kernels migrate their dst-sync as part of adopting the chain. |
| Performance validation of F-PERF-1+2+3+4 | Future pass | Deferred to a separate perf-validation pass. |

---

## Compatibility

- **Architecture:** Wormhole_b0. Block-element `DEST_AUTO_LIMIT = 16` is hard-coded for Wormhole.
- **Kernel build flag dependence:** `FP32_DEST_ACC_EN` required for any CARRY element with `EnableFp32DestAcc=true`. Compile-time `static_assert` rejects opt-in on kernels not built with the flag.
- **Programming model:** modern dst-sync only (`tile_regs_acquire/commit/wait/release`).
- **Test reach:** `tests/ttnn/unit_tests/kernel_lib/test_eltwise.py` — 453 passed / 7 skipped at HEAD `7060e1245a3` (preserved through Reg A + Reg C helper edits).

---

## Helper init-emission state (post Reg A + Reg C)

### Reg A — bcast init pair (commit `be533cfbbf4`)

The chain emits two distinct init patterns for bcast:

| Element | Init form | Why |
|---|---|---|
| `BinaryFpu<...Bcast != None>` | `llk_math_eltwise_binary_init_with_operands<et, bt, FID>(CbA, CbB)` + `llk_unpack_AB_init<bt>(CbA, CbB)` (`eltwise_chain.inl:528-532`) | Short-init pair (math + unpack only) — D8-compliant. The `_with_operands` form reads operand TensorShape from CB metadata via `get_operand_tensor_shape`, avoiding the `DEFAULT_TENSOR_SHAPE` hang the non-operand form caused on softmax LARGE_H partial-tile shapes. |
| `UnaryBcast<...>` | `unary_bcast_init<bt>(Cb, ocb)` (`eltwise_chain.inl:714`) | Full BIG init (hw_configure + pack_dest_init + sync_init). **Technically D8-violating** but required: `bcast_to` kernels (`compute_interleaved_*_bcast_to.cpp`) have no boot init at all — no `binary_op_init_common`, no `compute_kernel_hw_startup` — and rely on `UnaryBcast::init()` as their IMPLICIT BOOT. The redundant MMIO writes are harmless on Wormhole. |

D8 grep gate remains clean — both inits are per-element LLK helpers, not the `compute_kernel_hw_startup` / `binary_op_init_common` / `mm_init` / `reduce_init` family.

### Reg C — PackTile reconfig fold (commit `7ab7da6027c`)

`hoisted_init_for_each` at `eltwise_chain.inl:1031-1045` no longer guards on `!is_pack_tile_op_v<ElemT>`. Pack elements now correctly fire `emit_pre_element_transitions` so `pack_reconfig_data_format(CbOut)` declared via `PackTile<...PackTileReconfig::Output>` actually emits on the hoisted (non-clash) chain path. Pack init() remains a no-op so no side-effects beyond the reconfig.

Prior bug: chains without FPU-clash elements (no `BinaryFpu`/`UnaryBcast`/`DestReuseBinary`) inherited stale pack format from prior chain — affected `moreh_adam` T1.32/T1.33 (CopyTile+Power+PackTile chain) which dropped to 66F until the helper guard was removed.

### D8 strict-grep state

Helper-body emission of BIG inits remains: **1 site** (the intentional convenience wrapper at `eltwise_chain.inl:1254`). Verifier ran:

```bash
grep -nE 'init_common|compute_kernel_hw_startup|mm_init|reduce_init' \
     ttnn/cpp/ttnn/kernel_lib/eltwise_{chain.hpp,chain.inl,block.hpp}
```

All hits are doxygen comment text or the `#include "compute_kernel_hw_startup.h"` directive — except the one wrapper call at line 1254. D8 invariant held through Reg A + Reg C helper edits.

---

## Known limitation (verifier-flagged, NOT addressed by Reg A/B/C)

The chain helper is **DEST-format-correct** when the kernel-side caller opts in via per-element flags:

- `CopyTileReconfig::Input` on `CopyTile`
- `PackTileReconfig::Output` on `PackTile`
- `EnableFp32DestAcc=true` (when kernel built with `FP32_DEST_ACC_EN`)

If the migration drops these flags relative to the pre-migration raw shape (which used `copy_tile_init_with_dt` / `pack_tile_with_dt` for dynamic CB-format-driven reconfig), the `fp32_dest_acc_en=True` numerical path can produce `-inf` / overflow. The Type-1 sweep's `moreh_unary_chain`/`moreh_rexp_chain` templates (in `moreh_softmax_*_large.cpp`) and the `nll_loss_step2`/`nll_loss_backward` migrations exhibit this gap — see verification report for the full regression class.

**Not a helper bug — a migration-side annotation gap.** The helper supports the correct path; migrations must explicitly opt in.
