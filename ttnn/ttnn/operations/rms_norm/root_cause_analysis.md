# rms_norm — Mixed-precision-gamma + TILE-input Root Cause

## TL;DR

The bug is **NOT in the rms_norm kernel**. It's in the eltwise_chain DSL.
The chain's `PackTile<..., PackTileReconfig::Output>` declaration is silently
**not emitting its `pack_reconfig_data_format` call** when
`chain_is_hoist_safe_v<Chain>` is false — which it always is (hard-coded).

A single-line patch to `ttnn/cpp/ttnn/kernel_lib/eltwise_chain.inl` makes the
pack phase emit pre-element transitions for pack elements, and fixes both
buggy cells with no kernel-side change.

---

## Empirical signature (recap)

With deterministic inputs (x = ones, gamma = ones, expected output = 1.0):

| Case | input dtype | gamma dtype | actual output |
|---|---|---|---|
| matched | bf16 | bf16 | 1.00000 ✓ |
| matched | fp32 | fp32 | 0.99902 ✓ |
| **buggy** | **bf16** | **fp32** | **1.41406 = √2 uniform** |
| **buggy** | **fp32** | **bf16** | **0.523, varies** |

Eps sweep confirms `mean(x²)` is computed as **0.5× the true value** in
bf16+fp32. (See `tests/ttnn/unit_tests/operations/rms_norm/probes/probe_005.py`.)

Race conditions ruled out with 1024 real `TTI_NOP`s per Tensix thread between
Phase 0 and Stage A: no change in output.

---

## Mechanism (concrete chain, verified by DPRINT)

### Step 1 — what we want

`rms_norm_compute.cpp:90-101` runs Phase 0 (gamma tilize) via the helper
`compute_kernel_lib::tilize<..., ReconfigureRegisterDatatypeMode::UnpackAndPackReconfigure>(...)`.

This expands (via `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.inl:189-204`) to:

1. `pack_reconfig_data_format(cb_gamma_tiled)` — pack programmed for **gamma's** format.
2. `fast_tilize_init(cb_gamma_rm, Wt, cb_gamma_tiled)` — pack-side MOP programmed for tilize-of-gamma.
3. (loop) `fast_tilize_block(...)` — runs the tilize.
4. `fast_tilize_uninit(cb_gamma_rm, cb_gamma_tiled)` — `tt_metal/hw/inc/api/compute/tilize.h:279-288`. The PACK uninit is `llk_pack_fast_tilize_uninit<DST_ACCUM_MODE>(cb_gamma_tiled)` — which (`tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_pack.h:484-497`) calls `_llk_pack_init_(pack_dst_format = pack_dst_format[cb_gamma_tiled])`. So the pack MOP is reset to "regular pack" mode — but **for gamma's dtype**.

After Phase 0, the pack MOP is programmed for **gamma's dtype stride**
(e.g. fp32 width when gamma is fp32). pack_dst_format register also holds gamma's format.

### Step 2 — what the kernel declares

`rms_norm_compute.cpp:138-156` runs Stage A through the chain:

```cpp
ckl::eltwise_chain(
    Wt,
    CopyTile<cb_input_tiles, Dst::D0, ..., CopyTileReconfig::Input>{},
    Square<Dst::D0>{},
    PackTile<cb_x_sq, Dst::D0, ..., PackTileReconfig::Output>{});
```

`PackTileReconfig::Output` is supposed to make the chain emit
`pack_reconfig_data_format(cb_x_sq)` before Stage A's first `pack_tile`,
which would reprogram pack format AND (because `is_tile_dim_reconfig_en`
defaults to false in `_llk_pack_reconfig_data_format_` at `tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_pack.h:202-220`) leave the MOP alone.

The chain's intent is documented at `eltwise_chain.inl:288-292`:

```cpp
static constexpr uint32_t reconfig_pack_cb =
    (Reconfig == PackTileReconfig::Output || Reconfig == PackTileReconfig::OutputConditional)
        ? Cb : NO_PREV_CB;
```

…and the emission path at `eltwise_chain.inl:1026-1032`:

```cpp
constexpr uint32_t curr_p = cb_for_side<Side::Pack, E>();
if constexpr (curr_p != NO_PREV_CB) {
    constexpr uint32_t prev_p = prev_cb_for_idx<Side::Pack, I, Es...>();
    if constexpr (curr_p != prev_p) {
        pack_reconfig_data_format(curr_p);
    }
}
```

For Stage A's PackTile, `curr_p = cb_x_sq` and `prev_p = NO_PREV_CB`, so the
fold *would* emit `pack_reconfig_data_format(cb_x_sq)` — IF
`emit_pre_element_transitions` were reached for the pack element.

### Step 3 — where the chain silently drops it

**Bug site 1**: `eltwise_chain.inl:952-968` —
`chain_is_hoist_safe_v<Chain>` is **hard-coded to `false`**:

```cpp
// **Disabled (always false).** ... force per-tile init for every chain.
template <class Chain>
struct chain_is_hoist_safe : std::false_type {};
template <class... Es>
struct chain_is_hoist_safe<EltwiseChain<Es...>> : std::false_type {};
```

The comment explains this was a workaround for a separate SFPU-init issue
(`mish_kernel.cpp`, `logit_kernel.cpp`).

**Bug site 2**: `eltwise_chain.inl:1211` — because `chain_is_hoist_safe_v` is
always false:

```cpp
constexpr bool emit_init_per_tile = !chain_is_hoist_safe_v<Chain>;  // always true
```

**Bug site 3**: `eltwise_chain.inl:1236-1238` — `hoisted_init_for_each`,
the ONLY call site that emits `emit_pre_element_transitions` for pack
elements (lines 1162-1179, with the explicit comment "FIX (Reg C):
previously skipped Pack elements, but emit_pre_element_transitions is
the only emission path for pack_reconfig_data_format declared by
PackTile<...PackTileReconfig::Output>"), is gated on `!emit_init_per_tile`:

```cpp
if constexpr (!emit_init_per_tile) {                       // never true
    detail::hoisted_init_for_each(IdxSeq{}, elts...);      // never executes
}
```

**Bug site 4**: `eltwise_chain.inl:1107-1125` — the per-tile path
`elem_apply_pack` for pack elements **does not** emit pre-element
transitions:

```cpp
template <std::size_t I, class ElemT, class... Es>
ALWI void elem_apply_pack(...) {
    if constexpr (is_pack_tile_op_v<ElemT>) {
        elem.reserve_per_tile(i_outer);
        elem.reserve_upfront(n_tiles);
        for (uint32_t j = 0; j < inner_count; ++j) {
            elem.exec(base_tile + j, j * chain_lane_width);
        }
        elem.push_per_tile(i_outer);
    } else {
        ...
    }
}
```

And the compute-phase path `elem_apply_compute` at
`eltwise_chain.inl:1081-1082` does nothing for pack elements:

```cpp
if constexpr (is_pack_tile_op_v<ElemT>) {
    (void)elem; ...   // intentionally empty in compute phase
}
```

**Consequence**: with hoist-safety disabled (always), there is **no
emission path** for `PackTile::reconfig_pack_cb`. `PackTileReconfig::Output`
declarations are **silently dropped**.

### Step 4 — what the hardware does without the reconfig

`rms_norm_compute.cpp:138-156` Stage A's chain runs `pack_tile(D0, cb_x_sq, ...)`.
Pack uses the **leftover MOP from Phase 0's fast_tilize_uninit** (programmed
via `_llk_pack_init_(pack_dst_format = gamma_dtype)`) and the leftover
`pack_dst_format` register (gamma_dtype).

For bf16 input + fp32 gamma:
- Pack thinks output stride is fp32 (4 bytes/element).
- cb_x_sq in L1 has bf16-sized slots (2 bytes/element).
- Pack writes 32×32 fp32-stride values into a 32×32 bf16-sized L1 region.
- Raw L1 (probed with `CB_RD_PTR` direct uint32 reads in
  `tests/ttnn/unit_tests/operations/rms_norm/probes/probe_019.py`):
  `3f800000 3f800000 3f800000 ...` (fp32(1.0) repeated, every uint32).

### Step 5 — what the matmul reduce sees

`rms_norm_compute.cpp:178-186` Stage B is matmul-based REDUCE_ROW SUM
(`ttnn/cpp/ttnn/kernel_lib/reduce_helpers_common.hpp:16-19` — SUM +
REDUCE_ROW takes the matmul path on Wormhole).

Stage B's `reduce_helpers_compute.inl:226-231` issues
`reconfig_data_format(scaler_cb, input_cb)` — both srcA and srcB are now
correctly bf16. The matmul reads cb_x_sq's L1 bytes as bf16.

For the buggy bf16+fp32 case, each `0x3F800000` uint32 is interpreted as
two bf16 values (little-endian):
- low half (bytes 0-1) = `0x0000` = bf16(0.0)
- high half (bytes 2-3) = `0x3F80` = bf16(1.0)

The matmul reads alternating `(0, 1, 0, 1, ...)`. Sum = W/2. Mean = (W/2)/W = 0.5.
`rsqrt(0.5 + eps) ≈ √2 ≈ 1.41406`. Output = `x * √2 * gamma`. **Bug signature
explained exactly.**

For fp32 input + bf16 gamma, the inverse happens (pack writes bf16 stride
into fp32-sized L1 slots) and the resulting interleaving is more complex,
producing the W-dependent varying output we observed.

---

## Proof: the chain is the bug

Two-pronged verification:

### Workaround 1 — kernel-side (any of these works)

(a) `pack_reconfig_data_format(cb_x_sq)` immediately after the helper's
Phase 0 tilize call (`rms_norm_compute.cpp:101`). The kernel-emitted
reconfig executes BEFORE the chain starts, so the pack state is correct
when Stage A's `pack_tile` runs. The chain's own (silently-dropped)
PackTileReconfig::Output is irrelevant.

(b) Setting Stage A's PackTile reconfig to `PackTileReconfig::None`
(empirically: no change in behavior). Confirms the chain's
`PackTileReconfig::Output` was already a no-op.

### Workaround 2 — chain-side (the real fix)

Patch `ttnn/cpp/ttnn/kernel_lib/eltwise_chain.inl:1107-1125`'s
`elem_apply_pack` to emit pre-element transitions for pack elements
once per chain (gate on `i_outer == 0`):

```cpp
if constexpr (is_pack_tile_op_v<ElemT>) {
    if (i_outer == 0) {
        emit_pre_element_transitions<ElemT, I, Es...>();
    }
    elem.reserve_per_tile(i_outer);
    elem.reserve_upfront(n_tiles);
    for (uint32_t j = 0; j < inner_count; ++j) {
        elem.exec(base_tile + j, j * chain_lane_width);
    }
    elem.push_per_tile(i_outer);
}
```

With the chain patched **and the kernel reverted to its original state**
(no manual pack_reconfig anywhere in `rms_norm_compute.cpp`), both buggy
cases pass:

| Case | output |
|---|---|
| matched bf16+bf16 | 1.00000 ✓ |
| matched fp32+fp32 | 0.99902 ✓ |
| BUGGY bf16+fp32 | 1.00000 ✓ |
| BUGGY fp32+bf16 | 0.99902 ✓ |

This is verified in `tests/ttnn/unit_tests/operations/rms_norm/probes/probe_027.py`.

---

## Where to fix

### Option A — fix the chain (preferred)

`ttnn/cpp/ttnn/kernel_lib/eltwise_chain.inl:1107-1125`. Add the
`emit_pre_element_transitions<ElemT, I, Es...>()` call for pack elements
inside `elem_apply_pack` (gated on `i_outer == 0` so it only fires once
per chain). This is the single missing emission path for the silently-dropped
pack reconfig.

Risk: the hoist-safety predicate at `eltwise_chain.inl:952-968` was
disabled for a different reason (heterogeneous SFPU inits clobbering each
other). Re-enabling the predicate or re-routing pack emission must not
re-introduce that bug. The patch above is conservative: it adds the pack
emission to the per-tile path without touching the SFPU per-tile init
behavior.

A broader audit is warranted: any other chain element type with
`reconfig_pack_cb != NO_PREV_CB` (or `reconfig_srca_cb`, `reconfig_srcb_cb`
on pack-emitting elements) declared as `PackTileTag` is currently
silently dropping its reconfig.

### Option B — work around at the kernel level

If the chain fix isn't acceptable for some reason, the rms_norm kernel
can add `pack_reconfig_data_format(cb_x_sq)` immediately after
`rms_norm_compute.cpp:101` (the helper's Phase 0 tilize call). This
duplicates what the chain claims to do, and works because the kernel-side
call really executes.

---

## Files referenced

| File | Lines | What's there |
|---|---|---|
| `ttnn/ttnn/operations/rms_norm/kernels/rms_norm_compute.cpp` | 90-101 | Phase 0 gamma tilize call |
| `ttnn/ttnn/operations/rms_norm/kernels/rms_norm_compute.cpp` | 138-156 | Stage A chain (CopyTile + Square + PackTile) |
| `ttnn/ttnn/operations/rms_norm/kernels/rms_norm_compute.cpp` | 178-186 | Stage B matmul-based reduce |
| `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.inl` | 189-204 | Helper emits pack_reconfig(cb_gamma_tiled) and runs fast_tilize_init/uninit |
| `tt_metal/hw/inc/api/compute/tilize.h` | 279-288 | `fast_tilize_uninit` on Wormhole — pack uninit IS called |
| `tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_pack.h` | 484-497 | `_llk_pack_fast_tilize_uninit_` → calls `_llk_pack_init_(pack_dst_format=ocb)` — leaves MOP for gamma's dtype |
| `tt_metal/tt-llk/tt_llk_wormhole_b0/llk_lib/llk_pack.h` | 202-220 | `_llk_pack_reconfig_data_format_` — only re-issues MOP when `is_tile_dim_reconfig_en=true` (default false) |
| **`ttnn/cpp/ttnn/kernel_lib/eltwise_chain.inl`** | **952-968** | **`chain_is_hoist_safe_v` hard-coded `false`** |
| **`ttnn/cpp/ttnn/kernel_lib/eltwise_chain.inl`** | **1107-1125** | **`elem_apply_pack` — missing `emit_pre_element_transitions` call for pack elements (THE FIX SITE)** |
| `ttnn/cpp/ttnn/kernel_lib/eltwise_chain.inl` | 1081-1082 | `elem_apply_compute` deliberately skips pack elements |
| `ttnn/cpp/ttnn/kernel_lib/eltwise_chain.inl` | 1162-1179 | `hoisted_init_for_each` — would have emitted, but never called |
| `ttnn/cpp/ttnn/kernel_lib/eltwise_chain.inl` | 1211 | `emit_init_per_tile = !chain_is_hoist_safe_v` (always true) |
| `ttnn/cpp/ttnn/kernel_lib/eltwise_chain.inl` | 1236-1238 | hoisted_init gated on `!emit_init_per_tile` (always skipped) |
| `ttnn/cpp/ttnn/kernel_lib/eltwise_chain.inl` | 1026-1032 | The pack reconfig fold itself (correct, but unreachable) |
| `ttnn/cpp/ttnn/kernel_lib/eltwise_chain.inl` | 288-292 | `PackTile::reconfig_pack_cb` declaration (correct, but the chain doesn't read it on the live path) |
| `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_common.hpp` | 16-19 | `reduce_uses_matmul` — SUM+REDUCE_ROW takes matmul path |

## Probes referenced

| Probe | What it measured |
|---|---|
| `probe_002.py` | First measured output ratios — bf16+fp32 = exactly √2, fp32+bf16 = varied |
| `probe_005.py` | Eps sweep — distinguished "mean(x²) = 0.5× true" from "rsqrt × √2" |
| `probe_013.py` | 1024 TTI_NOPs per thread — race definitively rejected |
| `probe_019.py` | Raw L1 `CB_RD_PTR` byte dump — confirmed pack wrote fp32-stride into bf16 CB |
| `probe_023.py` | Slow tilize (no PACK init/uninit on Wormhole) — bug still present |
| `probe_026.py` | Stage A `PackTileReconfig::None` + manual pack_reconfig after Phase 0 — fix works |
| `probe_027.py` | Original kernel + chain DSL patched — **proves the chain is the bug** |
