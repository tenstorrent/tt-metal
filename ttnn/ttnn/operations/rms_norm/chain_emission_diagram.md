# eltwise_chain emitted-call diagram — pre and post fix

What the chain DSL actually emits for Stage A
(`CopyTile<cb_input_tiles> + Square + PackTile<cb_x_sq, ..., PackTileReconfig::Output>`)
under each emission strategy. Code blocks are the *effective* C++ that the
chain templates expand into (after constexpr folding, inlining, and the
`MATH(...)` / `UNPACK(...)` / `PACK(...)` macros that route a statement
to one Tensix thread).

## What "hoist" means here

**Hoisting** = moving an instruction out of a loop so it runs once instead
of N times per tile. The chain DSL has two strategies for emitting each
element's `init()` and `emit_pre_element_transitions()`:

- **Hoisted**: emit once before the per-tile loop. Fast, but only safe if
  no two elements' inits clobber each other.
- **Per-tile**: emit on every iteration. Slow but safe; re-issuing the
  init each iter guarantees a later element can't have overwritten it.

`ttnn/cpp/ttnn/kernel_lib/eltwise_chain.inl:946-968` hard-codes
`chain_is_hoist_safe<Chain> = false` for every chain, because some SFPU
inits clobber each other when hoisted. So the per-tile path is what's
actually used today — and that's the path with the missing pack-element
transition emission.

---

## Path 1 — Hoisted (currently disabled; would be correct)

This is what `PackTileReconfig::Output` *claims* to do. Unreachable today
because `chain_is_hoist_safe_v` is forced false.

```cpp
// ── chain entry, BEFORE per-tile loop ────────────────────────────────
// hoisted_init_for_each iterates all elements once.

// Element 0: CopyTile<cb_input_tiles, D0, NoWaitNoPop, BlockIter, Input>
//   emit_pre_element_transitions():
//     curr_a = cb_input_tiles, prev_a = NO_PREV_CB → emit srcA reconfig
UNPACK(( llk_unpack_reconfig_data_format_srca<DST_ACCUM_MODE>(cb_input_tiles) ));
MATH((   llk_math_reconfig_data_format_srca<DST_ACCUM_MODE>(cb_input_tiles)   ));
//   CopyTile::init() → copy_tile_to_dst_init_short(cb_input_tiles)
UNPACK(( llk_unpack_A_init<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE,
                           UnpackToDestEn>(0, false, cb_input_tiles) ));
MATH((   llk_math_eltwise_unary_datacopy_init<DataCopyType::A2D, DST_ACCUM_MODE,
                                              BroadcastType::NONE>(cb_input_tiles) ));

// Element 1: Square<D0>  (DestOnlyTag — touches DEST only)
//   emit_pre_element_transitions(): no CB sides → nothing
//   Square::init() → square_tile_init()
MATH(( llk_math_eltwise_unary_sfpu_square_init() ));

// Element 2: PackTile<cb_x_sq, D0, PerTileReserveAndPush, FirstTile, Output>
//   emit_pre_element_transitions():
//     curr_p = cb_x_sq, prev_p = NO_PREV_CB → emit pack reconfig  ⭐ HERE
PACK(( llk_pack_reconfig_data_format<DST_ACCUM_MODE, /*is_tile_dim_reconfig_en=*/false>
                                    (cb_x_sq) ));
//   PackTile::init() is a no-op (line 297-302 of eltwise_chain.inl)

// ── per-tile loop ────────────────────────────────────────────────────
for (uint32_t i = 0; i < Wt; ++i) {
    tile_regs_acquire();

    // CopyTile.exec: unpack cb_input_tiles[i] → srcA, math A2D → DEST[0]
    copy_tile(cb_input_tiles, /*tile_idx=*/i, /*dst=*/0);

    // Square.exec: SFPU square on DEST[0]
    square_tile(/*dst=*/0);

    tile_regs_commit();
    tile_regs_wait();

    // PackTile.exec: pack DEST[0] → cb_x_sq[0]
    cb_reserve_back(cb_x_sq, 1);
    pack_tile(/*dst=*/0, cb_x_sq, /*out_idx=*/0);
    cb_push_back(cb_x_sq, 1);

    tile_regs_release();
}
```

---

## Path 2 — Per-tile (current behaviour, PRE-FIX)

What the chain actually emits today. Note the absence of any
`llk_pack_reconfig_data_format` call for `cb_x_sq`.

```cpp
// ── chain entry ──────────────────────────────────────────────────────
// chain_is_hoist_safe_v == false → emit_init_per_tile == true
// → hoisted_init_for_each is GATED OFF (eltwise_chain.inl:1236-1238)
// (nothing emitted here)

// ── per-tile loop ────────────────────────────────────────────────────
for (uint32_t i = 0; i < Wt; ++i) {
    tile_regs_acquire();

    // ----- apply_compute_phase, EmitInit=true -----

    // Element 0: CopyTile (CbReaderTag branch, lines 1083-1095)
    //   emit_pre_element_transitions (srcA reconfig)
    UNPACK(( llk_unpack_reconfig_data_format_srca<DST_ACCUM_MODE>(cb_input_tiles) ));
    MATH((   llk_math_reconfig_data_format_srca<DST_ACCUM_MODE>(cb_input_tiles)   ));
    //   CopyTile::init()
    UNPACK(( llk_unpack_A_init<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE,
                               UnpackToDestEn>(0, false, cb_input_tiles) ));
    MATH((   llk_math_eltwise_unary_datacopy_init<DataCopyType::A2D, DST_ACCUM_MODE,
                                                  BroadcastType::NONE>(cb_input_tiles) ));
    //   wait_per_tile / exec / pop_per_tile
    // (NoWaitNoPop policy: wait and pop are no-ops; caller pre-waited Wt tiles)
    copy_tile(cb_input_tiles, /*tile_idx=*/i, /*dst=*/0);

    // Element 1: Square (DestOnlyTag branch, lines 1096-1104)
    //   emit_pre_element_transitions: no CB sides → no emission
    //   Square::init()
    MATH(( llk_math_eltwise_unary_sfpu_square_init() ));
    //   exec
    square_tile(/*dst=*/0);

    // Element 2: PackTile (PackTileTag → lines 1081-1082)
    //   if constexpr (is_pack_tile_op_v<ElemT>) {
    //       (void)elem; (void)i_outer; (void)base_tile;
    //       (void)inner_count; (void)chain_lane_width; (void)n_tiles;
    //   }
    //   ❌ NOTHING EMITTED HERE — pack elements skipped in compute phase

    tile_regs_commit();
    tile_regs_wait();

    // ----- apply_pack_phase -----

    // Elements 0, 1: CopyTile, Square (else branch of elem_apply_pack, line 1122)
    //   (void)elem; ... → nothing

    // Element 2: PackTile (is_pack_tile_op_v branch, lines 1115-1121)
    //   ❌ NO emit_pre_element_transitions
    //   ❌ pack_reconfig_data_format(cb_x_sq) NEVER EMITTED — anywhere
    cb_reserve_back(cb_x_sq, 1);
    pack_tile(/*dst=*/0, cb_x_sq, /*out_idx=*/0);  // uses STALE pack state
    cb_push_back(cb_x_sq, 1);

    tile_regs_release();
}
```

**The hole**: `PackTileReconfig::Output` on Stage A's `PackTile` declares
`reconfig_pack_cb = cb_x_sq` (eltwise_chain.inl:288-292), but no live
emission path reads it. The compute phase explicitly skips pack elements
(line 1081-1082), and the pack phase doesn't call
`emit_pre_element_transitions` at all (lines 1107-1125).

---

## Path 3 — Per-tile (POST-FIX, one-line patch)

```cpp
// ── chain entry ──────────────────────────────────────────────────────
// (unchanged: still no hoisted init)

// ── per-tile loop ────────────────────────────────────────────────────
for (uint32_t i = 0; i < Wt; ++i) {
    tile_regs_acquire();

    // ----- apply_compute_phase -----
    // (identical to PRE-FIX: CopyTile and Square per-iter inits + execs,
    //  PackTile still a no-op here)

    tile_regs_commit();
    tile_regs_wait();

    // ----- apply_pack_phase, PATCHED -----

    // Element 2: PackTile
    if (i == 0) {                                                    // ⭐ NEW
        // emit_pre_element_transitions for the pack element, once.
        // The pack target CB is fixed at the element type level, so
        // re-emitting per iteration would just waste MMIO.
        PACK(( llk_pack_reconfig_data_format<DST_ACCUM_MODE, false>   // ⭐ NEW
                                            (cb_x_sq) ));             // ⭐ NEW
    }
    cb_reserve_back(cb_x_sq, 1);
    pack_tile(/*dst=*/0, cb_x_sq, /*out_idx=*/0);   // uses CORRECT pack state
    cb_push_back(cb_x_sq, 1);

    tile_regs_release();
}
```

`i == 0` guard: the pack target doesn't change mid-chain (chain
invariant — one PackTile element = one fixed `Cb`), so re-emitting every
iter is wasted MMIO. Compute-side inits stay per-tile to preserve the
SFPU-clobber workaround that disabled the global hoist.

---

## Side-by-side: pack_reconfig emission count

For Stage A's chain over `n_tiles = Wt`:

| Strategy | Pre-loop | Per-iter | Total `llk_pack_reconfig_data_format` calls |
|---|---|---|---|
| Hoisted (disabled) | 1 | 0 | **1** ✓ |
| Per-tile PRE-FIX | 0 | 0 | **0** ❌ |
| Per-tile POST-FIX | 0 | 1 (on `i == 0` only) | **1** ✓ |

The `PackTileReconfig::Output` user-facing knob has the same semantic
intent in all three strategies. Only the per-tile path is broken; the
fix gives it one emission point in the pack phase.

---

## Why pack-state was stale in the first place

Pre-fix, when Stage A's `pack_tile` runs, the pack-side hardware state is
whatever Phase 0's helper left behind.

```cpp
// State after boot (compute_kernel_hw_startup → binary_op_init_common):
//   pack_dst_format = bf16
//   pack MOP stride = bf16 (2 bytes per element)

// Phase 0 — helper (ttnn/cpp/ttnn/kernel_lib/tilize_helpers.inl:189-204):

//   Step 1: helper reconfigs pack to gamma's dtype
PACK(( llk_pack_reconfig_data_format<DST_ACCUM_MODE, false>(cb_gamma_tiled) ));
//   → pack_dst_format = gamma_dtype (fp32 in the bug case)

//   Step 2: fast_tilize_init (tt_metal/hw/inc/api/compute/tilize.h:260-269)
UNPACK(( llk_unpack_fast_tilize_init(cb_gamma_rm, /*full_dim=*/Wt) ));
MATH((   llk_math_fast_tilize_init(cb_gamma_rm, /*unit_dim=*/2)    ));
PACK((   llk_pack_fast_tilize_init(cb_gamma_rm, cb_gamma_tiled, /*unit_dim=*/2) ));
//   → DEST_OFFSET regs in fp32 mode, fast-tilize MOP programmed

//   Step 3: tilize loop (writes cb_gamma_tiled)

//   Step 4: fast_tilize_uninit (tt_metal/hw/inc/api/compute/tilize.h:279-288)
UNPACK(( llk_unpack_fast_tilize_uninit<DST_ACCUM_MODE>()         ));
MATH((   llk_math_fast_tilize_uninit<DST_ACCUM_MODE>(cb_gamma_rm) ));
PACK((   llk_pack_fast_tilize_uninit<DST_ACCUM_MODE>(cb_gamma_tiled) ));
//   PACK uninit internally calls (llk_pack.h:484-497):
//     _llk_pack_init_(pack_dst_format = pack_dst_format[cb_gamma_tiled])
//   → DEST_OFFSET regs restored to defaults
//   → MOP reset BUT for gamma's dtype: MOP stride = fp32 (4 bytes)
//   pack_dst_format register still = gamma_dtype (fp32)

// Stage A chain (per-tile path, PRE-FIX):
//   → no pack transitions emitted (the hole)
//   → pack hardware state unchanged from Phase 0's leftovers

// Stage A pack_tile: writes from DEST (bf16, 2B/elem) to cb_x_sq (L1 bf16)
//   using MOP stride = fp32 (4B/elem)
pack_tile(/*dst=*/0, cb_x_sq, /*out_idx=*/0);
//   → L1 receives fp32-shaped writes laid out into bf16-sized slot
//   → raw L1 bytes: 3f800000 3f800000 ... (verified by probe_019.py)

// Stage B matmul reduce: reads cb_x_sq as bf16
//   bytes 0..1 = 0x0000 (bf16 0.0), bytes 2..3 = 0x3f80 (bf16 1.0)
//   → unpacked pattern: (0, 1, 0, 1, ...)
//   → row sum = W/2, mean = 0.5, rsqrt(0.5 + eps) ≈ √2
```

Post-fix: the patched `apply_pack_phase` emits `pack_reconfig_data_format(cb_x_sq)`
on `i == 0`. The high-level wrapper (`tt_metal/hw/inc/api/compute/pack.h:127-132`)
expands to `llk_pack_reconfig_data_format<DST_ACCUM_MODE, /*is_tile_dim_reconfig_en=*/false>(cb_x_sq)`,
which calls `reconfig_packer_data_format` (`tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/cpack_common.h:562`).
That sets `pack_dst_format` to bf16 and resets `PCK_DEST_RD_CTRL_Read_32b_data`
based on the cb_x_sq format — enough that the very next `pack_tile` emits the
correct stride.

---

## One-line patch (verified, then reverted)

```diff
--- a/ttnn/cpp/ttnn/kernel_lib/eltwise_chain.inl
+++ b/ttnn/cpp/ttnn/kernel_lib/eltwise_chain.inl
@@ -1107,6 +1107,9 @@ template <std::size_t I, class ElemT, class... Es>
 ALWI void elem_apply_pack(
     ...) {
     if constexpr (is_pack_tile_op_v<ElemT>) {
+        if (i_outer == 0) {
+            emit_pre_element_transitions<ElemT, I, Es...>();
+        }
         elem.reserve_per_tile(i_outer);
         elem.reserve_upfront(n_tiles);
         for (uint32_t j = 0; j < inner_count; ++j) {
```

Verification (probe_027.py):

| Case | output | status |
|---|---|---|
| matched bf16+bf16 | 1.00000 | ✓ |
| matched fp32+fp32 | 0.99902 | ✓ |
| BUGGY bf16+fp32 | 1.00000 | ✓ (was 1.41406) |
| BUGGY fp32+bf16 | 0.99902 | ✓ (was 0.523±0.476) |

No kernel-side change required.
