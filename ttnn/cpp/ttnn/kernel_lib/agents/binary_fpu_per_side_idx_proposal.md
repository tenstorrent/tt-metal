# Helper Update Proposal — `BinaryFpu` per-side local-vs-absolute index toggle

Status: **AWAITING SIGN-OFF** (Gate 1).
Author: agent run, 2026-05-19.
Helper: `compute_kernel_lib::BinaryFpu` (and `BinaryFpu`-derived chain dispatch).
Pipeline phase: Helper Update (post-Phase-3, pre-Phase-4).

---

## 1. Problem statement

The chain dispatcher in `eltwise_chain.inl` picks the tile-index regime
per-ELEMENT via `element_uses_per_block_index_v<ElemT>`:

```cpp
constexpr bool use_local_idx = element_uses_per_block_index_v<ElemT>;
...
for (uint32_t j = 0; j < inner_count; ++j) {
    const uint32_t i_arg = use_local_idx ? j : (base_tile + j);
    elem.exec(i_arg, j * chain_lane_width);
}
```

`BinaryFpu` has TWO operands (A and B) each with its own `CopyTilePolicy` and
`CbIndexMode`. The element-level toggle collapses both sides to the same `i`:

- A=`WaitAndPopPerBlock` (chunk-local front, wants `j`) +
  B=`NoWaitNoPop` with `BIndex=BlockIter` (absolute caller-managed window, wants
  `base_tile + j`) → impossible to satisfy with a single `i_arg`.

Today this combo is **statically rejected** at
`eltwise_chain.inl:638-651`:

```cpp
static_assert(
    !(APolicy == CopyTilePolicy::WaitAndPopPerBlock &&
      BPolicy != CopyTilePolicy::WaitAndPopPerBlock &&
      (BIndex == CbIndexMode::BlockIter || BIndex == CbIndexMode::BlockIterOffset)),
    "BinaryFpu: A is WaitAndPopPerBlock (chunk-local index) but B uses BlockIter/"
    "BlockIterOffset on a non-per-block policy (absolute index). Mixed indexing "
    "is unrepresentable — switch B to WaitAndPopPerBlock or use FirstTile/Pinned/Absolute on B.");
```

That assert is **correct under the current dispatch**, but it caps what
migrations can do. Specifically (from the migration survey on 2026-05-19):

- `rmsnorm_post_allgather.cpp` stages 2 & 3 (gamma multiply, beta add)
- `layernorm.cpp` analogous gamma/beta stages
- Any "stream the data tile by tile, broadcast a Wt-wide weight upfront" shape

cannot be migrated from `WaitUpfrontPopAtEnd` (input held upfront) to
`WaitAndPopPerBlock` (input streamed in chunks) because the gamma/beta side
needs to keep reading absolute tile indices into its pre-staged Wt-wide window.

## 2. Goal

Allow A and B to independently use chunk-local OR absolute indexing inside
`BinaryFpu`. Both sides today already declare their own `CopyTilePolicy` and
`CbIndexMode`; this proposal extends the dispatch so the index regime tracks
the policy *per side*, not per element.

Non-goals:
- No change to single-operand elements (`CopyTile`, `DestReuseBinary`,
  `UnaryBcast`, `PackTile`, `PackTileBlock`, all DEST-only ops). They keep
  the existing single-`i` `exec` signature.
- No change to existing API surface that callers use — `BinaryFpu`'s template
  parameter list stays the same.
- No change to the per-element `use_local_idx` semantics for any non-BinaryFpu
  element.

## 3. API change

### 3.1 New static traits on `BinaryFpu`

```cpp
static constexpr bool a_uses_local_idx = (APolicy == CopyTilePolicy::WaitAndPopPerBlock);
static constexpr bool b_uses_local_idx = (BPolicy == CopyTilePolicy::WaitAndPopPerBlock);
// True iff the two sides disagree on indexing regime — only then does the
// chain need to thread the second index through. Same-regime is fast-path.
static constexpr bool needs_per_side_idx = (a_uses_local_idx != b_uses_local_idx);
```

The existing accessors (`a_policy()`, `b_policy()`, `a_index_mode`,
`b_index_mode`) stay.

### 3.2 `BinaryFpu::exec` / `exec_2d` — new 3-arg form

The current `exec(uint32_t i, uint32_t slot_offset)` keeps working: when both
sides agree on regime, the chain still passes a single `i` and BinaryFpu uses
it for both `a_idx` and `b_idx` derivation — identical to today.

When the sides disagree, the chain calls a new overload:

```cpp
// 1D form
ALWI void exec(uint32_t i_local, uint32_t i_abs, uint32_t slot_offset) const {
    const uint32_t a_idx = resolve_idx<AIndex>(
        a_uses_local_idx ? i_local : i_abs, a_tile_idx_);
    const uint32_t b_idx = resolve_idx<BIndex>(
        b_uses_local_idx ? i_local : i_abs, b_tile_idx_);
    // ... add_tiles / sub_tiles / mul_tiles or bcast variant, same as today
}

// 2D form (analogous; ht / wt arguments preserved)
ALWI void exec_2d(uint32_t i_local, uint32_t i_abs,
                  uint32_t ht, uint32_t wt, uint32_t slot_offset) const;
```

`resolve_idx<Mode>(i, runtime_k)` is a thin helper that mirrors the existing
inline switch on `CbIndexMode` (FirstTile→0, BlockIter→i, BlockIterOffset→
runtime_k+i, Pinned/Absolute→runtime_k).

The 2-arg `exec(i, slot_offset)` is retained as a forwarder:

```cpp
ALWI void exec(uint32_t i, uint32_t slot_offset) const {
    // i is interpreted as "the single index the chain has decided to pass"
    // — works for matching-regime case identical to today.
    exec(/*i_local=*/i, /*i_abs=*/i, slot_offset);
}
```

This keeps every existing caller binary-compatible.

### 3.3 Chain dispatcher (`elem_apply_compute`, `elem_apply_compute_2d`)

Add a SFINAE detector for the 3-arg overload:

```cpp
template <class E, class = void>
struct has_per_side_exec : std::false_type {};
template <class E>
struct has_per_side_exec<E,
    std::void_t<decltype(std::declval<const E&>().exec(0u, 0u, 0u))>>
    : std::true_type {};
```

In the per-element compute branch:

```cpp
for (uint32_t j = 0; j < inner_count; ++j) {
    if constexpr (has_per_side_exec<ElemT>::value
                  && ElemT::needs_per_side_idx) {
        elem.exec(/*i_local=*/j, /*i_abs=*/(base_tile + j),
                  j * chain_lane_width);
    } else {
        const uint32_t i_arg = use_local_idx ? j : (base_tile + j);
        elem.exec(i_arg, j * chain_lane_width);
    }
}
```

Same structure mirrored in `elem_apply_compute_2d`. Pack-side dispatch
(`elem_apply_pack` / `elem_apply_pack_2d`) is unchanged — single-CB
elements only.

### 3.4 Static-assert relaxation

The two mixed-policy asserts at `eltwise_chain.inl:638-651` are **removed**
or downgraded to "either same regime OR per-side mode is required" — the
new path makes the mixed case legal. The asserts that reject
`RowBcast/ColBcast + WaitAndPopPerBlock` and `BlockIterOffset +
WaitAndPopPerBlock` (via `valid_policy_mode_2d_v`) **stay** — those are
window-shape constraints, not index-mode constraints.

## 4. Caller impact

Existing kernels (the 6 binary_ng FPU kernels migrated in the current diff;
all unary kernels; all chain-using normalization stages) see **zero
behaviour change**:
- Same-regime BinaryFpu hits the 2-arg `exec` exactly as today.
- Non-BinaryFpu elements ignore the new SFINAE branch.

New capability unlocked:
- A=`WaitAndPopPerBlock` + B=`WaitUpfrontPopAtEnd`/`WaitUpfrontNoPop`/
  `NoWaitNoPop` with `BIndex=BlockIter` (or `BlockIterOffset`) becomes
  legal. Gamma/beta-style migrations unblocked.

## 5. Risk assessment

| Risk | Mitigation |
|---|---|
| Inner-loop dispatch divergence regresses existing per-tile / upfront path perf | The `if constexpr` chain is fully compile-time-resolved; the non-per-side path is byte-identical to today's emitted code (verified by reading the generated path: same single-`i` call, no `i_abs` computation). Adds zero runtime cost to chains where `needs_per_side_idx == false`. |
| 2-arg → 3-arg forwarder introduces signature mismatch for user-defined `BinaryFpu`-likes | None exist; `BinaryFpu` is a final concrete struct, not a CRTP base. User code constructs it but doesn't subclass it. |
| SFINAE detector picks up unrelated `exec(uint32_t, uint32_t, uint32_t)` on other tags | Only `BinaryFpu` will declare the 3-arg overload; gated by the additional `ElemT::needs_per_side_idx` constexpr check. |
| Existing kernels using `static_assert`-rejected mixed combos exist as commented-out / TODO code paths | Grepped; none found. |

## 6. Acceptance criteria

1. New `binary_fpu_per_side_idx.cpp` test kernel exercises A=`WaitAndPopPerBlock`
   + `BlockIter` and B=`WaitUpfrontPopAtEnd` + `BlockIter` over `num_tiles ∈
   {4, 16, 64}`, `BlockSize ∈ {2, 4}`, `fp32_dest_acc ∈ {False, True}`. Goldens
   match via `comp_pcc >= 0.9999` (bf16-only path).
2. Symmetric test for A=`WaitUpfrontPopAtEnd` + B=`WaitAndPopPerBlock` (swap
   sides — verifies static_assert was the only thing blocking it).
3. Existing pytest suite unchanged-and-green: `test_eltwise.py` (483p baseline),
   `test_add.py`, `test_binary_ng_bcast_fp32_dest_acc.py`,
   `test_binary_ng_program_cache.py`, `test_group_norm.py`.
4. Migration follow-up (separate commit, NOT part of this proposal):
   `rmsnorm_post_allgather.cpp` stages 2 + 3 switched to `WaitAndPopPerBlock`
   on A keeping `WaitUpfrontPopAtEnd` on B; rmsnorm test green.

## 7. Files touched (implementation, post-sign-off)

- `ttnn/cpp/ttnn/kernel_lib/eltwise_chain.hpp` — minor: doc-comment on
  `BinaryFpu` describing the per-side rule (no template-list change).
- `ttnn/cpp/ttnn/kernel_lib/eltwise_chain.inl` —
  - `BinaryFpu`: add `a_uses_local_idx` / `b_uses_local_idx` /
    `needs_per_side_idx`; add 3-arg `exec` / `exec_2d`; keep 2-arg as
    forwarder.
  - Add `has_per_side_exec` SFINAE detector.
  - `elem_apply_compute` / `elem_apply_compute_2d`: dispatch through
    SFINAE branch.
  - Remove / relax the two mixed-policy `static_assert`s on `BinaryFpu`.
- `ttnn/cpp/ttnn/kernel_lib/tests/eltwise/kernels/binary_fpu_per_side_idx.cpp`
  (new).
- `tests/ttnn/unit_tests/kernel_lib/test_eltwise.py` (new test fns,
  per Gate 2 test plan).

## 8. Out of scope (separate proposals)

- `BinarySfpu` chain element (would unblock the 6 SFPU binary_ng cousins).
- Runtime `BlockSize` chain entry (would unblock legacy `eltwise_binary_kernel.cpp`).
- Per-side toggle on `DestReuseBinary` / `UnaryBcast` — those are single-CB,
  no per-side need.
- Migration commits — each rmsnorm/layernorm kernel moves in its own commit
  after the helper extension lands, per LLK HQ "one kernel per commit".

---

Proposal at `ttnn/cpp/ttnn/kernel_lib/agents/binary_fpu_per_side_idx_proposal.md`. Awaiting sign-off.
