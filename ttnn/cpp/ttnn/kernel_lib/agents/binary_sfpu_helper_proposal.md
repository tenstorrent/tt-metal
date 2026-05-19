# Helper Update Proposal — `BinarySfpu` op-struct coverage + chain composition pattern

Status: **AWAITING SIGN-OFF (revised)** (Gate 1).
Helpers touched: `compute_kernel_lib::eltwise_binary_sfpu.hpp` (extend with op-structs only).
Unblocks: 6 binary_ng SFPU kernels (`eltwise_binary_sfpu*.cpp`) — the where-variants and legacy binary stay out of scope here.
Pipeline phase: Helper Update (Phase 3 proposal).

Revision history:
- v1: proposed a new `BinarySfpu` CB-reading chain element. REJECTED — duplicates `CopyTile` functionality that already exists.
- **v2 (this doc)**: no new chain element. Composition of two existing `CopyTile` readers + an existing DEST-only SFPU binary op + an existing `PackTile` writer covers the pattern. Helper work narrows to op-struct coverage and a small bit of chain plumbing if composition isn't already plug-and-play.

---

## 1. Problem statement

`eltwise_binary_sfpu_no_bcast.cpp` (and 5 sibling kernels — scalar, col_bcast, row_bcast, row_col_bcast, scalar_bcast) do:

```cpp
tile_regs_acquire();
copy_tile_to_dst_init_short_with_dt(cb_b, cb_a);
for (i=0; i<n; ++i) copy_tile(cb_a, i, 2*i);            // A → even DEST slots
copy_tile_to_dst_init_short_with_dt(cb_a, cb_b);
for (i=0; i<n; ++i) {
    copy_tile(cb_b, i, 2*i + 1);                         // B → odd DEST slots
    BINARY_SFPU_OP(2*i, 2*i + 1, 2*i);                    // SFPU bin: (even, odd) → even
    PROCESS_POST_ACTIVATIONS(2*i);
}
tile_regs_commit();
// pack from even slots → cb_out
```

`BINARY_SFPU_OP` is a host-defined macro expanding to one of ~30 SFPU binary tile-ops (full list at `binary_ng_utils.cpp:get_sfpu_init_fn`, lines 404-553: ADD, SUB, MUL, DIV, DIV_FLOOR, DIV_TRUNC, REMAINDER, FMOD, POWER, RSUB, GCD, LCM, LEFT_SHIFT, RIGHT_SHIFT, LOGICAL_RIGHT_SHIFT, BITWISE_AND, BITWISE_OR, BITWISE_XOR, MAXIMUM, MINIMUM, QUANT, REQUANT, DEQUANT, XLOGY, ATAN2, LT, GT, GE, LE, EQ, NE; WHERE is ternary, excluded).

The chain helper today has op-structs for only 6 of these (`AddBinary` / `SubBinary` / `MulBinary` / `DivBinary` / `BinaryMax` / `BinaryMin` in `eltwise_binary_sfpu.hpp`). The remaining ~25 ops have LLK functions but no helper struct → kernels cannot pick a struct and have to fall back to the raw macro.

## 2. Goal

Make the SFPU binary kernels migrate via the **existing chain composition** — no new chain element:

```cpp
eltwise_chain<BlockSize>(
    num_tiles,
    CopyTile<cb_a, …, Dst::D0, CopyTileReconfig::Input>{},  // A → even slots
    CopyTile<cb_b, …, Dst::D1, CopyTileReconfig::Input>{},  // B → odd slots; fold flips srca cb_a→cb_b
    SfpuBinOp<Dst::D0, Dst::D1, Dst::D0>{},                 // existing DEST-only struct (Add/Sub/…)
    PackTile<cb_out, Dst::D0, …, PackTileReconfig::Output>{}
);
```

`chain_lane_width_v` already folds element `lane_width`s and bounds via `BlockSize * chain_lane_width ≤ DEST_AUTO_LIMIT`. With CopyTile<D1>'s `lane_width = to_u32(D1) + 1 = 2`, `chain_lane_width = 2`, and each lane j writes:

- A → slot `0 + j*2` (D0 base, stride 2).
- B → slot `1 + j*2` (D1 base, stride 2).
- SFPU bin op: `(D0 + j*2, D1 + j*2) → D0 + j*2`.
- Pack: reads slot `0 + j*2`.

The srca reconfig between the two CopyTile loads is emitted by the existing prev-CB fold (D2 mechanism — `reconfig_srca_cb` chain on consecutive CopyTiles).

Two things are needed for that composition to actually work:

1. **Op-struct coverage**: extend `eltwise_binary_sfpu.hpp` with the ~25 missing DEST-only SFPU binary op-structs (each ≈ 4 lines, matching the existing 6).
2. **Composition smoke test**: verify with a unit test kernel that the chain *does in fact* run two CopyTile readers + a DEST-only SFPU bin + PackTile correctly at `chain_lane_width=2`. If anything is missing (fold edge case, lane-width interaction), fix it in `eltwise_chain.inl`.

Non-goals:
- No new chain element.
- Migrating `eltwise_where_sfpu*.cpp` — `WHERE` is ternary; needs a different composition (CopyTile×3 + DEST-only ternary). Filed as a separate proposal later.
- Wrapping `PREPROCESS` / `PROCESS_POST_ACTIVATIONS` macros — activations branch stays raw (PARTIAL migration).
- Legacy `eltwise/binary/device/.../eltwise_binary_sfpu_kernel.cpp` — runtime block size blocks chain entry; separate ticket.

## 3. API changes

### 3.1 New op-structs in `eltwise_binary_sfpu.hpp`

Mechanical extension of the existing `BinaryOp<Self, In0, In1, Out>` CRTP pattern. One struct per `OpConfig::SfpuBinaryOp` enum value not already covered:

| Enum value | New struct | LLK init / call | Notes |
|---|---|---|---|
| ADD (int) | `AddIntBinary<DataFormat DF>` | `add_int_tile_init` / `add_int_tile<DF>` | DataFormat template param |
| SUB (int) | `SubIntBinary<DataFormat DF>` | `sub_int_tile_init` / `sub_int_tile<DF>` |  |
| MUL (int) | `MulIntBinary<DataFormat DF>` | `mul_int_tile_init<DF>` / `mul_int_tile<DF>` |  |
| DIV (int32) | `DivInt32Binary` | `div_int32_tile_init` / `div_int32_tile` |  |
| DIV_FLOOR | `DivInt32FloorBinary` | `div_int32_floor_tile_init` / `div_int32_floor_tile` |  |
| DIV_TRUNC | `DivInt32TruncBinary` | `div_int32_trunc_tile_init` / `div_int32_trunc_tile` |  |
| REMAINDER (float) | `RemainderBinary` | `remainder_binary_tile_init` / `remainder_binary_tile` |  |
| REMAINDER (int32) | `RemainderInt32Binary` | `remainder_int32_tile_init` / `remainder_int32_tile` |  |
| FMOD (float) | `FmodBinary` | `fmod_binary_tile_init` / `fmod_binary_tile` |  |
| FMOD (int32) | `FmodInt32Binary` | `fmod_int32_tile_init` / `fmod_int32_tile` |  |
| POWER | `PowerBinary` | `power_binary_tile_init` / `power_binary_tile` |  |
| RSUB (float) | `RsubBinary` | `rsub_binary_tile_init` / `rsub_binary_tile` |  |
| RSUB (int) | `RsubIntBinary<DF>` | `rsub_int_tile_init` / `rsub_int_tile<DF>` |  |
| GCD | `GcdBinary` | `gcd_tile_init` / `gcd_tile` |  |
| LCM | `LcmBinary` | `lcm_tile_init` / `lcm_tile` |  |
| LEFT_SHIFT | `LeftShiftBinary<DF>` | `binary_shift_tile_init` / `binary_left_shift_tile<DF>` |  |
| RIGHT_SHIFT | `RightShiftBinary<DF>` | `binary_shift_tile_init` / `binary_right_shift_tile<DF>` |  |
| LOGICAL_RIGHT_SHIFT | `LogicalRightShiftBinary<DF>` | `binary_shift_tile_init` / `binary_logical_right_shift_tile<DF>` |  |
| BITWISE_AND | `BitwiseAndBinary<DF>` | `binary_bitwise_tile_init` / `bitwise_and_binary_tile<DF>` |  |
| BITWISE_OR | `BitwiseOrBinary<DF>` | `binary_bitwise_tile_init` / `bitwise_or_binary_tile<DF>` |  |
| BITWISE_XOR | `BitwiseXorBinary<DF>` | `binary_bitwise_tile_init` / `bitwise_xor_binary_tile<DF>` |  |
| MAXIMUM (int32) | `BinaryMaxInt32` | `binary_max_int32_tile_init` / `binary_max_int32_tile` |  |
| MAXIMUM (uint32) | `BinaryMaxUint32` | `binary_max_uint32_tile_init` / `binary_max_uint32_tile` |  |
| MINIMUM (int32) | `BinaryMinInt32` | `binary_min_int32_tile_init` / `binary_min_int32_tile` |  |
| MINIMUM (uint32) | `BinaryMinUint32` | `binary_min_uint32_tile_init` / `binary_min_uint32_tile` |  |
| QUANT | `QuantBinary` (ctor: `uint32_t zero_point`) | `quant_tile_init(zp)` / `quant_tile` | Runtime arg — same shape as existing `Power{exponent}` |
| REQUANT | `RequantBinary` (ctor: `uint32_t zero_point`) | `requant_tile_init(zp)` / `requant_tile` |  |
| DEQUANT | `DequantBinary` (ctor: `uint32_t zero_point`) | `dequant_tile_init(zp)` / `dequant_tile` |  |
| XLOGY | `XlogyBinary` | `xlogy_binary_tile_init` / `xlogy_binary_tile` |  |
| ATAN2 | `Atan2Binary` | `atan2_binary_tile_init` / `atan2_binary_tile` |  |
| LT (float) | `LtBinary` | `lt_binary_tile_init` / `lt_binary_tile` |  |
| LT (int) | `LtIntBinary<DF>` | `lt_int_tile_init<DF>` / `lt_int_tile<DF>` |  |
| GT (float / int) | `GtBinary` / `GtIntBinary<DF>` |  |  |
| GE (float / int) | `GeBinary` / `GeIntBinary<DF>` |  |  |
| LE (float / int) | `LeBinary` / `LeIntBinary<DF>` |  |  |
| EQ (float only) | `EqBinary` | `eq_binary_tile_init` / `eq_binary_tile` |  |
| NE (float only) | `NeBinary` | `ne_binary_tile_init` / `ne_binary_tile` |  |

Total ~25 new structs (a few more counting the integer variants). All template signatures default to `In0=Dst::D0, In1=Dst::D1, Out=Dst::D0` matching the existing 6.

### 3.2 Composition verification (smoke test, NO new helper API)

Before touching production kernels, a kernel_lib unit test exercises the composed chain end-to-end:

```cpp
// tests/eltwise/kernels/binary_sfpu_compose.cpp
eltwise_chain<BLOCK_SIZE>(
    num_tiles,
    CopyTile<cb_a, CopyTilePolicy::WaitAndPopPerBlock, CbIndexMode::BlockIter, Dst::D0,
             CopyTileReconfig::Input>{},
    CopyTile<cb_b, CopyTilePolicy::WaitAndPopPerBlock, CbIndexMode::BlockIter, Dst::D1,
             CopyTileReconfig::Input>{},
    AddBinary<Dst::D0, Dst::D1, Dst::D0>{},
    PackTile<cb_out, Dst::D0, PackTilePolicy::PerBlockReserveAndPush,
             PackTileIndexMode::BlockIter, PackTileReconfig::Output>{}
);
```

Acceptance: bf16 PCC ≥ 0.9999 vs torch golden for `BlockSize ∈ {1, 2, 4}`, `num_tiles ∈ {4, 8, 16, 64}`, `fp32_dest_acc ∈ {False, True}`.

If this fails:
- The most likely culprit is the chain's prev-CB fold not emitting `copy_tile_to_dst_init_short_with_dt(cb_a, cb_b)` between two `CopyTile` elements with different CBs. Fix is local to `emit_pre_element_transitions` / the CopyTile reconfig path in `eltwise_chain.inl` — a chain plumbing fix, NOT a new chain element.

This is the **only** chain code touched by this proposal, and only if the smoke test exposes a gap.

## 4. Caller migration shape

Migrated `eltwise_binary_sfpu_no_bcast.cpp` (no-activations branch):

```cpp
namespace detail {
// Map BINARY_SFPU_OP host-side macro identity to the chain struct.
// Same idea as the existing FpuOpForBinaryType mapping in eltwise_binary_scalar.cpp.
template <typename> struct SfpuBinaryFor;
// One specialization per supported macro/struct pair, populated by the program factory's macro.
}  // namespace detail

#if not(HAS_ACTIVATIONS(LHS) or HAS_ACTIVATIONS(RHS) or HAS_ACTIVATIONS(POST))
    using BinElt = ...;  // a 3-element chain alias: CopyTile A + CopyTile B + SFPU bin struct
    eltwise_chain<num_tiles_per_cycle>(
        num_tiles,
        CopyTile<cb_post_lhs, CopyTilePolicy::WaitAndPopPerBlock, CbIndexMode::BlockIter,
                 Dst::D0, CopyTileReconfig::Input>{},
        CopyTile<cb_post_rhs, CopyTilePolicy::WaitAndPopPerBlock, CbIndexMode::BlockIter,
                 Dst::D1, CopyTileReconfig::Input>{},
        SelectedSfpuBin<Dst::D0, Dst::D1, Dst::D0>{},  // resolved by the macro mapping
        PackTile<cb_out, Dst::D0, PackTilePolicy::PerBlockReserveAndPush,
                 PackTileIndexMode::BlockIter, PackTileReconfig::Output>{}
    );
#else
    // Activations branch — keep raw (PROCESS_POST_ACTIVATIONS / PREPROCESS macros).
    ...
#endif
```

`SelectedSfpuBin` is resolved by a small kernel-side mapping that ties the existing program-factory `BINARY_SFPU_OP_NAME` (or analogous CT identifier) to the right helper struct. Same shape as `eltwise_binary_scalar.cpp`'s `FpuOpForBinaryType` mapping that's already in tree.

## 5. Risk assessment

| Risk | Mitigation |
|---|---|
| Chain composition fails at `chain_lane_width=2` (today untested with two CopyTile readers using different `Dst` slots) | Step 0 smoke test catches this BEFORE any production migration. Fix is local plumbing — not a new chain element. |
| 25 new structs explode header line count | Header already groups op family; mechanical adds ≈ 100 LoC total. Each ≈ 4 lines (decl + init + exec_impl). |
| Integer-format ops template-pollute call sites | Each int variant takes `DataFormat` as the first template param. Macro mapping handles the dtype selection (same way the program factory does today). |
| Quant/Requant/Dequant runtime zero-point | Ctor takes `uint32_t`, identical to the existing `Power{exponent}` runtime-arg pattern. |
| Macro-injected `BINARY_SFPU_OP` doesn't compose with a typed struct | The host-side change is small: program factory emits a `BINARY_SFPU_STRUCT` define naming the helper struct (in addition to or replacing `BINARY_SFPU_OP`). Mapping table in `binary_ng_utils.cpp` mirrors the existing dispatch. Out of scope for the kernel-lib change but documented as a host-side follow-up. |
| `PROCESS_POST_ACTIVATIONS` macro is post-op SFPU chain — does it compose? | Out of scope. Activations branch stays raw. |

## 6. Acceptance criteria

1. Composition smoke test (`binary_sfpu_compose.cpp`) passes `comp_pcc ≥ 0.9999` for `BlockSize ∈ {1, 2, 4}`, `num_tiles ∈ {4, 8, 16, 64}`, `fp32_dest_acc ∈ {False, True}`, using the existing `AddBinary` op.
2. Each new op-struct in `eltwise_binary_sfpu.hpp` compiles and has at least a smoke-test entry exercising the chain composition (one struct per representative dtype group — float, int32, uint32).
3. Migrated `eltwise_binary_sfpu_no_bcast.cpp` (no-activations branch) — `test_binary_ng_*.py` SFPU runs still pass for ADD / SUB / MUL / DIV / MAX / MIN / POWER / REMAINDER (representative dtype coverage).
4. Existing kernel_lib baseline (591 passed + 7 skipped) unchanged.
5. Existing FPU regression set (`test_add.py`, `test_binary_ng_*.py`, `test_group_norm.py`) unchanged.

Per-commit cadence:
- Commit 1: composition smoke test + any plumbing fix uncovered.
- Commit 2: op-struct adds in `eltwise_binary_sfpu.hpp` (one commit; structurally identical to each other).
- Commits 3+: one migrated kernel per commit.

## 7. Files touched (post-sign-off)

- `ttnn/cpp/ttnn/kernel_lib/eltwise_binary_sfpu.hpp` — ~25 new op-structs.
- `ttnn/cpp/ttnn/kernel_lib/eltwise_chain.inl` — local plumbing fix ONLY if the smoke test exposes a gap; no new elements.
- `ttnn/cpp/ttnn/kernel_lib/tests/eltwise/kernels/binary_sfpu_compose.cpp` (new).
- `tests/ttnn/unit_tests/kernel_lib/test_eltwise.py` — new test fns (Gate-2 test plan to follow).
- Migrations (separate commits, post-helper-landing): `eltwise_binary_sfpu_no_bcast.cpp` first.

## 8. Out of scope

- New `BinarySfpu` chain element (explicitly rejected — composition covers the same surface).
- `eltwise_where_sfpu*.cpp` (ternary).
- Activations-branch raw-LLK paths (`PREPROCESS`, `PROCESS_POST_ACTIVATIONS`).
- Legacy `eltwise_binary_sfpu_kernel.cpp` (runtime block size).
- Host-side macro/struct dispatch change (small follow-up in `binary_ng_utils.cpp` to emit the chosen struct identifier).

---

Proposal at `ttnn/cpp/ttnn/kernel_lib/agents/binary_sfpu_helper_proposal.md` (v2). Awaiting sign-off.
