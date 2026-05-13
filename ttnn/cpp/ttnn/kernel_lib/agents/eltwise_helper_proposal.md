# Eltwise Helper — Gate-1 Proposal (single batch)

Closes 11 items from `eltwise_helper_gap_audit.md` in one proposal. Per `llk_helpers_hq.md` Gate 1: **BLOCKING — no `.inl`, `.cpp`, kernel source, or test file written until this proposal is explicitly approved.**

Audit-numbered items dropped: original BLOCKER 1 (auto init-hoist). Investigation of `pack_patterns.tsv` confirmed §3.4's strict "exactly `CopyTile + 1 SFPU op`" precondition was wrong; lessons doc §3.4 has been rewritten with the empirically-grounded gate (`!chain_has_non_copy_tile_fpu_clash_v && chain_copy_tiles_share_cb_v`). Current pipeline gate `has_clash` is a near-miss — it permits multi-CB CopyTile chains that would silently miscompile — and is folded into the proposal as item 7's `chain_loads_share_cb` generalisation.

## Summary — accepted scope

| # | Item | Severity | Where it lands |
|---|---|---|---|
| 1 | Single dispatch contract (drop static `call()` + member `exec(i)` fork) | BLOCKER 2 | `eltwise_chain.{hpp,inl}` + every op-category header |
| 2 | Chain auto-derives BlockSize from element DEST footprint; delete `eltwise_block.hpp` | SHOULD-FIX | `eltwise_chain.{hpp,inl}`; delete `eltwise_block.hpp` |
| 3 | `Fp32DestAcc::Off/On` enum replacing `bool EnableFp32DestAccV` | SHOULD-FIX | `eltwise_chain.hpp` + every CARRY element |
| 4 | Drop stub-default tag fields; SFINAE-detect overrides | SHOULD-FIX | `eltwise_chain.hpp` + chain pipeline |
| 5 | Pipeline state private (`*_idx_` + ctor-only) | SHOULD-FIX | `CopyTile`, `BinaryFpu`, `DestReuseBinary`, `PackTile` |
| 6 | `block_path` wait-late: move upfront wait/reserve inside per-tile loop | SHOULD-FIX | `eltwise_chain.inl` `eltwise_chain()` |
| 7 | Generalise `chain_loads_share_cb` to N-element fold + use as hoist gate | NICE | `eltwise_chain.inl` |
| 8 | Delete dead `chain_is_hoist_safe` specialisations replaced by item 7 | NICE | `eltwise_chain.{hpp,inl}` |
| ~~9~~ | ~~Runtime `ASSERT` for `Pinned` / `Absolute` index bounds~~ | DROPPED | — |
| 10 | `Mask<DF, DataSlot>` op struct in `eltwise_misc.hpp` per §1.4 | NICE | new struct |
| 11 | Reduce↔Broadcast doc table in `BroadcastDim` doc-comment | NICE | `eltwise_chain.hpp` doc |
| 12 | `CumulativeWaitPopAtEnd` policy added (no longer deferred) | DEFERRED→IN-SCOPE | `CopyTilePolicy` + per-tile/upfront emission |

Detail per item below. Each item names: (a) the lesson clause it closes, (b) the source-of-truth file changing, (c) the API delta, (d) the test addition needed for Gate 2.

---

## 1. Single dispatch contract

**Lesson:** §1.5, §4.5, §9 "Two dispatch contracts."

**Current ((BLOCKER 2)):**
```cpp
template <Dst Slot> struct Exp : UnaryOp<Exp<Slot>, Slot> {
    static void init();
    static void call(uint32_t idst);     // body — static
};
template <Dst Slot> struct Power : UnaryOp<Power<Slot>, Slot> {
    uint32_t exponent;
    static void init();
    static void call(uint32_t /*idst*/) {}            // ← empty stub
    void exec(uint32_t /*i*/) const;     // body — member
};
// pipeline:
if constexpr (has_member_exec_v<E>) e.exec(i);
else if constexpr (is_dest_only_op_v<E>) E::exec();   // calls Derived::call via base
```

**Proposed:** every chain element exposes one `void exec(uint32_t i) const`. CRTP bases supply a default that forwards to a `static call(uint32_t idst)` if the derived defines one, but the dispatch surface seen by the pipeline is uniform `e.exec(i)`. Drop `has_member_exec_v` and the static-vs-member fork in `elem_compute_exec`.

```cpp
template <class Derived, Dst Slot> struct UnaryOp : DestOnlyTag {
    static_assert(to_u32(Slot) < DEST_AUTO_LIMIT, "…");
    static constexpr Dst dst_idx = Slot;
    static ALWI void init() { Derived::init_impl(); }
    ALWI void exec(uint32_t /*i*/) const {
        if constexpr (requires { Derived::call(uint32_t{}); }) Derived::call(to_u32(Slot));
        else                                                   static_cast<const Derived&>(*this).exec_impl();
    }
};
```

Or simpler: drop `call()` entirely. Every op defines `static ALWI void exec_impl()` (no-param SFPU) or `ALWI void exec_impl() const` (runtime-param SFPU). CRTP base's `exec(uint32_t)` forwards to `exec_impl()`. Forgetting the override is a compile error.

Op-category headers updated mechanically:
```cpp
template <Dst Slot> struct Exp : UnaryOp<Exp<Slot>, Slot> {
    static ALWI void init() { exp_tile_init<…>(); }
    static ALWI void exec_impl() { exp_tile<…>(to_u32(Slot)); }
};
template <Dst Slot> struct Power : UnaryOp<Power<Slot>, Slot> {
    uint32_t exponent;
    static ALWI void init() { power_tile_init(); }
    ALWI void exec_impl() const { power_tile(to_u32(Slot), exponent); }
};
```

Files touched: `eltwise_chain.{hpp,inl}` (CRTP bases + pipeline), `eltwise_{activations, math, misc, special, scalar, predicates, fill, rand, rounding, trig}.hpp`, `eltwise_binary_sfpu.hpp`.

Gate-2 tests: existing `chain_and_binary` suite plus one new kernel `dispatch_contract.cpp` that mixes static-SFPU + runtime-param SFPU in the same chain to verify the unified dispatch.

---

## 2. Chain auto-derives BlockSize from DEST footprint

**Lesson:** §3.9 "Block-mode is an axis on existing elements, not a parallel sibling kind."

**Direction (user feedback):** BlockSize is **not** a per-element template parameter. Chain computes it from the element pack's DEST footprint and `DEST_AUTO_LIMIT`. Elements never spell BlockSize.

### Mechanic

Each chain element exposes a compile-time `lane_width` trait — the number of consecutive DEST slots it touches per lane of work. Defaults:

| Element | `lane_width` |
|---|---|
| `CopyTile<Cb, DstSlot, …>` | `to_u32(DstSlot) + 1` (highest slot referenced + 1) |
| `BinaryFpu<…, DstSlot, …>` | `to_u32(DstSlot) + 1` |
| `DestReuseBinary<…, DstOut, …>` | `to_u32(DstOut) + 1` |
| `UnaryBcast<…, DstSlot, …>` | `to_u32(DstSlot) + 1` |
| `PackTile<…, DstSlot, …>` | `to_u32(DstSlot) + 1` |
| `Mask<DF, DataSlot>` | `to_u32(DataSlot) + 2` (item 10 — mask uses DataSlot AND DataSlot+1) |
| `UnaryOp` / `BinaryOp` SFPU bases | derived from the highest `Slot` template param |

Chain computes `lane_width = max(E::lane_width for E in Es...)` at compile time.

Chain then picks `BlockSize = DEST_AUTO_LIMIT / lane_width` (compile-time integer divide). Outer loop iterates `n_tiles` in chunks of `BlockSize`; inner emits `BlockSize` lanes per acquire/release window. Each element's `exec(uint32_t i_outer, uint32_t j_lane)` computes DEST slot `dst_slot + j_lane * lane_width` and CB tile index `i_outer * BlockSize + j_lane` (under `BlockIter`) or `i_outer` etc. per the index-mode rules.

Tail: `n_tiles % BlockSize != 0` handled by clamping the inner loop iter count to `min(BlockSize, n_tiles - i_outer)`. Inner iter count becomes runtime — fine, DEST slot index is still compile-time-derivable per `j`.

### Element exec signature change

Currently `exec(uint32_t i)`. New: `exec(uint32_t i_outer, uint32_t j_lane)`. Element computes its slot and index internally.

```cpp
template <uint32_t Cb, Dst DstSlot, …>
struct CopyTile : CopyTileTag {
    static constexpr uint32_t lane_width = to_u32(DstSlot) + 1;
    // …
    ALWI void exec(uint32_t i_outer, uint32_t j_lane) const {
        const uint32_t dst = to_u32(DstSlot) + j_lane * lane_width;
        const uint32_t in_idx = /* index mode × (i_outer, j_lane, BlockSize) */;
        copy_tile(Cb, in_idx, dst);
    }
};
```

Index-mode resolution under auto-block:
- `FirstTile` → always 0 (broadcast operand).
- `BlockIter` → `i_outer * BlockSize + j_lane` (walks the full input tile range).
- `Pinned` / `Absolute` → runtime field as today (caller responsibility — runtime ASSERTs deferred per Q3).

CB lifecycle scales by `BlockSize`:
- `WaitAndPop` per-tile → wait/pop `BlockSize` per outer iter (block-granular streaming).
- `WaitUpfrontPopAtEnd` → wait `n_tiles` upfront, pop `n_tiles` at end. Unchanged (BlockSize is internal).
- `WaitNoPop` / `NoWaitPop` / `NoWaitNoPop` → wait/pop `BlockSize` per outer iter or no-op per policy.

### Deletion

`eltwise_block.hpp` deleted outright in the same PR — there is no parallel `BlockCopyTile`/`BlockBinaryFpu`/`BlockPackTile` to alias to, because the BlockSize axis no longer exists at element level. Every call site using `Block*` migrates to the streaming element with no template-param change.

For existing call sites that today pass `BlockSize` explicitly (e.g. `BlockCopyTile<cb_in, 8, D0, …>{}`), the migration drops the size: `CopyTile<cb_in, D0, …>{}`. The chain produces the same DEST batching at compile time. Behaviour difference: today the kernel picks `BlockSize=8` even if DEST permits more; after auto-derive, the chain picks `BlockSize=DEST_AUTO_LIMIT / lane_width`, which is usually larger. This is a perf improvement, not a behaviour change.

### Chain-wide on/off toggle

Auto-block is chain-wide, not per-element (parallel to `n_tiles` being chain-wide). Single boolean toggle:

```cpp
enum class AutoBlock : bool { Off = false, On = true };

template <AutoBlock Block = AutoBlock::On, class... Es>
ALWI void eltwise_chain(uint32_t n_tiles, Es... elts);
```

- `AutoBlock::On` (default): chain picks `BlockSize = DEST_AUTO_LIMIT / lane_width`.
- `AutoBlock::Off`: chain forces `BlockSize = 1` — every iter is one DEST slot's worth of work (today's streaming behavior). Escape hatch for kernels that depend on a specific outer-loop tile count.

No user-supplied cap value. The toggle is binary because either you want max DEST utilisation (the common case) or you want strict per-tile dispatch. Anything between is a code-smell — pick a different chain shape.

### Resolved clarifications

- **OC-2a (confirmed):** User-specified `DstSlot` lives in lane-0 coordinates. Chain offsets per lane: physical slot = `to_u32(DstSlot) + j_lane * lane_width`. User reads/writes lane-0 slot ids; chain handles replication.
- **OC-2b (confirmed):** No restriction on user `DstSlot`. If user picks `Dst::D7` in a single-op chain, `lane_width = 8` → `BlockSize = 2` (bf16) or `1` (fp32). Accepted as the natural consequence of the slot choice; if the user wants max BlockSize, pick `Dst::D0`.
- **OC-2c (resolved):** Chain-wide `AutoBlock` template arg (above). No `MaxBlockSize` numeric cap.

Files touched: `eltwise_chain.{hpp,inl}`; `eltwise_block.hpp` deleted; every kernel/test referencing `Block*` migrates.

Gate-2 tests: existing `binary_block.cpp` / `pack_block.cpp` rebuilt as streaming chains. New kernel `auto_block_dest_packed.cpp` verifies BlockSize fills DEST when single-slot chain runs (BlockSize should equal DEST_AUTO_LIMIT). New kernel `tail_handling.cpp` verifies `n_tiles % BlockSize != 0` correctness.

---

## 3. `Fp32DestAcc` enum replacing `bool`

**Lesson:** §2.5 "Policy enums are NEVER booleans."

**Decision:** drop-in rename. No alias period.

**Proposed:** add
```cpp
enum class Fp32DestAcc : bool { Off = false, On = true };
```
to `eltwise_chain.hpp`'s named-enum section (`Approx`, `Legacy`, `Dst`). Replace every `bool EnableFp32DestAccV` template parameter on CARRY elements (`BinaryFpu`, `BlockBinaryFpu`, `DestReuseBinary`, `UnaryBcast`, `PackTile`, `PackTileBlock`, `BlockPackTile`) with `Fp32DestAcc EnableFp32DestAccV = Fp32DestAcc::Off`. SFINAE fold probes (`fp32_or_default`, `has_fp32_dest_acc`) read `Fp32DestAcc` not `bool`; one-line patch.

(After item 2 lands `eltwise_block.hpp` is gone, so the `Block*` rows above collapse into the streaming-element rows.)

Migration of `kFp32 ? true : false` patterns in kernels:
```cpp
constexpr auto kFp32 = Fp32DestAcc::On;
// or
#ifdef FP32_DEST_ACC_EN
  constexpr auto kFp32 = Fp32DestAcc::On;
#else
  constexpr auto kFp32 = Fp32DestAcc::Off;
#endif
```

Every call site changes from `…, true>{}` to `…, Fp32DestAcc::On>{}`. Mechanical search-replace; one PR.

Files touched: every header listed in item 1 that carries `EnableFp32DestAccV`.

Gate-2 tests: existing fp32 tests recompiled. No new test needed.

---

## 4. Drop stub-default tag fields

**Lesson:** §1.7 "Stub-default member functions are a silent footgun."

**Current:**
```cpp
struct CbReaderTag { static constexpr uint32_t pack_cb_id() { return 0; } };
struct CbWriterTag { static constexpr uint32_t cb_a_id() { return 0; }
                     static constexpr uint32_t cb_b_id() { return 0; } };
struct DestOnlyTag { static constexpr bool is_upfront = false;
                     static constexpr bool clashes_with_fpu = false;
                     static constexpr uint32_t cb_a_id() { return 0; }
                     static constexpr uint32_t cb_b_id() { return 0; }
                     static constexpr uint32_t pack_cb_id() { return 0; } };
```

**Proposed:** strip the CB-id stubs from the tags. The pipeline already SFINAE-detects `reconfig_srca_cb` / `reconfig_srcb_cb` / `reconfig_pack_cb`; mirror that for `cb_a_id` / `cb_b_id` / `pack_cb_id`:
```cpp
struct CbReaderTag {};   // 0-byte
struct CbWriterTag {};
struct DestOnlyTag {
    static constexpr bool is_upfront = false;
    static constexpr bool clashes_with_fpu = false;
};
// pipeline:
template <class E, class = void> struct has_cb_a_id : std::false_type {};
template <class E> struct has_cb_a_id<E, std::void_t<decltype(E::cb_a_id())>> : std::true_type {};
// …
template <class E> constexpr uint32_t cb_a_of() {
    if constexpr (has_cb_a_id<E>::value) return E::cb_a_id();
    else                                  return 0;     // never reached for is_cb_reader_op_v elements
}
```

Plus a `static_assert` at chain entry that every `is_cb_reader_op_v<E>` element exposes `cb_a_id()` (and `cb_b_id()` for binary), naming the missing override.

Files touched: `eltwise_chain.{hpp,inl}`.

Gate-2 tests: a kernel `missing_cb_id.cpp` is **not** added — this is a compile-time-error path; instead the proposal asserts that removing the override from a test op produces the expected `static_assert` failure (one-liner test in the kernel_lib test driver, not a device test).

---

## 5. Pipeline state private; ctor-only writes

**Lesson:** §4.1 "Pipeline state is private and reset by the pipeline."

**Proposed:** rename `cb_tile_idx → cb_tile_idx_`, etc., mark `private`, expose ctor as the only write surface, optional `ALWI uint32_t tile_idx() const` accessor for tests. The pipeline reads via friend declaration (chain pipeline already lives in the same TU via `eltwise_chain.inl`).

```cpp
struct CopyTile : CopyTileTag {
public:
    constexpr CopyTile() noexcept = default;
    constexpr explicit CopyTile(uint32_t cb_tile_idx) noexcept : cb_tile_idx_(cb_tile_idx) {}
    // … static traits + lifecycle methods …
private:
    uint32_t cb_tile_idx_ = 0;
    template <class... Es> friend ALWI void eltwise_chain(uint32_t, Es...);
    template <class... Es> friend ALWI void eltwise_chain_with_init(uint32_t, Es...);
};
```

Files touched: `eltwise_chain.{hpp,inl}` (`CopyTile`, `BinaryFpu`, `DestReuseBinary`, `PackTile`, `PackTileBlock`); `eltwise_block.hpp` (after item 2's collapse).

Gate-2 tests: existing `copy_upfront.cpp` covers `Pinned` use; no new test.

---

## 6. `block_path` wait-late amendment

**Lesson:** §11 "Wait as late as possible. … Upfront and cumulative shapes are NOT moved to a pre-loop block."

**Current `block_path` (eltwise_chain.inl:1129-1155):**
```cpp
(detail::elem_wait_upfront(elts, n_tiles), ...);       // before loop
(detail::elem_reserve_upfront(elts, n_tiles), ...);    // before loop
for (uint32_t i = 0; i < n_tiles; ++i) { … }
(detail::elem_pop_upfront_end(elts, n_tiles), ...);
(detail::elem_push_at_end(elts, n_tiles), ...);
```

**Proposed:**
```cpp
for (uint32_t i = 0; i < n_tiles; ++i) {
    (detail::elem_wait_upfront(elts, n_tiles), ...);   // idempotent — moves inside loop
    (detail::elem_wait_per_tile(elts, i), ...);
    tile_regs_acquire();
    detail::compute_phase_for_each<…>(IdxSeq{}, i, elts...);
    tile_regs_commit();
    tile_regs_wait();
    (detail::elem_reserve_per_tile(elts, i), ...);
    (detail::elem_reserve_upfront(elts, n_tiles), ...); // idempotent
    (detail::elem_pack_exec(elts, i), ...);
    (detail::elem_push_per_tile(elts, i), ...);
    tile_regs_release();
    (detail::elem_pop_per_tile(elts, i), ...);
}
(detail::elem_pop_upfront_end(elts, n_tiles), ...);
(detail::elem_push_at_end(elts, n_tiles), ...);
```

`cb_wait_front(cb, N)` and `cb_reserve_back(cb, N)` are cumulative-count idempotent — calling every iter is correct and short-circuits after iter 0. The per-tile variants are already policy-guarded internally (no-op for non-PerTile policies). Pop-at-end / push-at-end stay outside the loop.

This also lets `block_path` and the per-tile path share more code; the structural difference reduces to "which lifecycle pair fires at the bottom of the function."

Files touched: `eltwise_chain.inl` `eltwise_chain()` only.

Gate-2 tests: existing `copy_upfront.cpp` + `binary_block.cpp` re-run; one new microbench kernel `wait_late_overlap.cpp` measures push-overlap improvement under a producer-rate-limited reader (optional, not blocking the Gate-2 test plan).

---

## 7. Generalise `chain_loads_share_cb` to N-element fold

**Lesson:** §3.3 chain traits; §3.4 (rewritten); §4.5 single contract.

**Current `eltwise_chain.inl:802-806`:**
```cpp
template <class Chain> struct chain_loads_share_cb : std::false_type {};
template <class A, class B> struct chain_loads_share_cb<EltwiseChain<A, B>>
    : std::bool_constant<is_copy_tile_op_v<A> && is_copy_tile_op_v<B> && (A::cb == B::cb)> {};
```

Only the size-2 chain specialisation works; longer chains silently return false (and current pipeline ignores the trait anyway because the hoist gate is `has_clash`-only).

**Proposed:**
```cpp
namespace detail {
template <class E> constexpr uint32_t copy_tile_cb_of() {
    if constexpr (is_copy_tile_op_v<E>) return E::cb;
    else                                return NO_CB;
}
template <class... Es> struct copy_tiles_share_cb_impl {
    static constexpr bool value = []() {
        constexpr uint32_t cbs[] = { copy_tile_cb_of<Es>()... };
        uint32_t seen = NO_CB;
        for (auto cb : cbs) {
            if (cb == NO_CB) continue;
            if (seen == NO_CB) seen = cb;
            else if (seen != cb) return false;
        }
        return true;
    }();
};
} // namespace detail
template <class... Es> struct chain_loads_share_cb<EltwiseChain<Es...>>
    : std::bool_constant<detail::copy_tiles_share_cb_impl<Es...>::value> {};
```

Plus the hoist gate consumes it:
```cpp
template <class Chain> struct chain_is_hoist_safe
    : std::bool_constant<!chain_has_non_copy_tile_fpu_clash_v<Chain>
                         && chain_loads_share_cb_v<Chain>> {};
…
constexpr bool emit_init_per_tile = !chain_is_hoist_safe_v<Chain>;
```

This closes the latent miscompile in `CopyTile<cbA> + CopyTile<cbB> + AddBinary + PackTile` chains (no test exercises this today, but the lessons-§3.4 rewrite makes it a known unsafe shape).

Files touched: `eltwise_chain.inl` (traits + the `has_clash` line in `eltwise_chain()`).

Gate-2 tests: one new kernel `multi_cb_copy_no_clash.cpp` — `CopyTile<cbA> + CopyTile<cbB> + AddBinary + PackTile` chain, verifies output. Before the gate fix this would be silently wrong; after, hoist falls back to per-tile init and output matches.

---

## 8. Delete dead `chain_is_hoist_safe` size-2 / size-3 specialisations

After item 7 the generic trait replaces both fixed-size specs (`<EltwiseChain<A, B>>`, `<EltwiseChain<A, B, C>>`). Drop them.

Files touched: `eltwise_chain.inl`.

No new tests.

---

## 9. ~~Runtime ASSERT for `Pinned` / `Absolute` index bounds~~ — DROPPED

Per user direction: skip for now. The compile-time `(Policy × IndexMode)` `static_assert` table already rejects the worst structural misuses. Runtime guards revisited if a real bug surfaces.

No files touched. No tests.

---

## 10. `Mask<DF, DataSlot>` op struct

**Lesson:** §1.4 worked example.

**Proposed:** add to `eltwise_misc.hpp`:
```cpp
template <DataFormat DF, Dst DataSlot>
struct Mask : BinaryOp<Mask<DF, DataSlot>, DataSlot,
                       static_cast<Dst>(to_u32(DataSlot) + 1),
                       DataSlot> {
    static_assert(to_u32(DataSlot) + 1 < DEST_AUTO_LIMIT,
                  "Mask requires DataSlot + 1 < DEST capacity (mask tile lives at DataSlot+1).");
    static ALWI void init() { mask_tile_init(); }
    static ALWI void exec_impl() { mask_tile<DF>(to_u32(DataSlot)); }
};
```

Files touched: `eltwise_misc.hpp`.

Gate-2 tests: one new kernel `mask_op.cpp` — fill DataSlot with float, DataSlot+1 with mask, run `Mask<…>`, compare against torch golden.

---

## 11. Reduce↔Broadcast doc table

**Lesson:** §5.3.

**Proposed:** add markdown table to the `BroadcastDim` enum doc-comment in `eltwise_chain.hpp`:
```cpp
/// FPU broadcast dimension. Caller MUST pass explicitly — no inference.
///
/// Reduce↔Broadcast mapping (the surprise that lives where it is needed):
///
/// | Reduce direction | Output shape | Broadcast direction needed downstream |
/// |---|---|---|
/// | REDUCE_ROW        | (N, 1)       | BroadcastDim::Col (broadcast across cols) |
/// | REDUCE_COL        | (1, M)       | BroadcastDim::Row (broadcast across rows) |
/// | REDUCE_SCALAR     | (1, 1)       | BroadcastDim::Scalar |
/// | REDUCE_W (alias)  | (N, 1)       | BroadcastDim::Col |
/// | REDUCE_H (alias)  | (1, M)       | BroadcastDim::Row |
///
/// Mirrors `ckernel::BroadcastType` values (NONE=0, COL=1, ROW=2, SCALAR=3).
enum class BroadcastDim : uint8_t { … };
```

Files touched: `eltwise_chain.hpp` doc-comment only.

No tests.

---

## 12. `CumulativeWaitPopAtEnd` policy

**Lesson:** §2.1 cumulative row; §11 "missing useful cell."

**Proposed:** add the missing enum value, per-side wait emission, and index-mode legality entry.

```cpp
enum class CopyTilePolicy : uint8_t {
    WaitAndPop,
    WaitNoPop,
    NoWaitPop,
    NoWaitNoPop,
    WaitUpfrontPopAtEnd,
    CumulativeWaitPopAtEnd,   // ← new — per-iter wait(i+1), bulk pop at end
};
```

Element behavior:
- `wait_per_tile(uint32_t i_outer)` for `CumulativeWaitPopAtEnd`: `cb_wait_front(Cb, (i_outer + 1) * BlockSize)`. Under `AutoBlock::Off` BlockSize=1 and this reduces to `cb_wait_front(Cb, i_outer + 1)` — the lessons-§11 cumulative shape verbatim. Under `AutoBlock::On` the wait grows in block-sized chunks (block-cumulative), matching how production block kernels stage upstream pushes.
- `wait_upfront(uint32_t n)`: no-op (the wait is per-iter cumulative, not upfront).
- `pop_upfront_end(uint32_t n)`: `cb_pop_front(Cb, n)`.
- `pop_per_tile`: no-op.
- `is_upfront = true` (so the chain pipeline routes it via the upfront-block path, which after item 6 emits both per-tile and end-of-block lifecycle ops).

Compile-time validity table (compatible index modes):
- `FirstTile` ✓ (always tile 0)
- `BlockIter` ✓ — `cb_wait_front(cb, (i_outer+1) * BlockSize)` guarantees tiles `[0..(i_outer+1)*BlockSize)` present; lane index `i_outer * BlockSize + j_lane` falls within that window for every `j_lane < BlockSize`
- `Pinned` ✓ when `k < (i_outer + 1) * BlockSize` at every iter (caller's responsibility; runtime guard deferred per Q3)
- `Absolute` ✓ same caveat

`static_assert` block in `CopyTile` / `BinaryFpu` / `DestReuseBinary` extended to allow `CumulativeWaitPopAtEnd + BlockIter` and `+ Absolute`.

Per-side BinaryFpu support: A and B independently pick `CumulativeWaitPopAtEnd` if their pre-pushed stream needs cumulative semantics; same-CB dedup stays in force (asymmetric `AIndex` vs `BIndex` on a shared CB is still a compile error).

Files touched: `eltwise_chain.{hpp,inl}` (enum + every CB-reader element's wait/pop).

Gate-2 tests: one new kernel `cumulative_wait_block_iter.cpp` — chain `CopyTile<CumulativeWaitPopAtEnd, BlockIter>` + `Exp` + `PackTile<UpfrontReservePushAtEnd>`, verifies producer can push tile `k < N` while consumer is already computing on tiles `[0..k]` (PCC vs torch golden). Parametrise over `num_tiles ∈ {1, 8, 64}` per HQ.

---

## Test-plan summary (Gate 2 input)

Per HQ §"Gate 2 — Test plan requires explicit user approval", the test plan ships as a SEPARATE artifact (`eltwise_helper_test_plan.md`) after Gate 1 is signed. Sketch only:

| Suite | Kernel | Items covered | Shape | dtype matrix |
|---|---|---|---|---|
| existing | `copy_exp_pack`, `binary_fpu`, `fanout`, `multi_chain`, `binary_block`, `pack_block`, `copy_upfront`, `dest_reuse`, `optional_element`, `inplace_accumulate`, `pack_lifecycle` | items 1, 3, 5, 6, 8 (rebuilt against new APIs); `binary_block`/`pack_block` rewritten without `Block*` types (item 2) | unchanged | unchanged |
| new | `dispatch_contract.cpp` | item 1 | static + runtime SFPU mixed | bf16, fp32 |
| new | `auto_block_dest_packed.cpp` | item 2 | single-slot chain, BlockSize fills DEST | bf16, fp32 (different DEST_AUTO_LIMIT) |
| new | `auto_block_off.cpp` | item 2 | `AutoBlock::Off` toggle reproduces today's per-tile dispatch | bf16 |
| new | `tail_handling.cpp` | item 2 | `n_tiles % BlockSize != 0` | bf16 |
| new | `lane_width_high_slot.cpp` | item 2 (OC-2b) | user picks `Dst::D7` → BlockSize collapses; verifies correctness, not perf | bf16, fp32 |
| new | `multi_cb_copy_no_clash.cpp` | item 7 | `CopyTile<cbA> + CopyTile<cbB> + AddBinary + PackTile` | bf16 + `fp32_dest_acc_en` |
| new | `mask_op.cpp` | item 10 | unary mask | bf16 |
| new | `cumulative_wait_block_iter.cpp` | item 12 | cumulative wait | bf16 |

All new kernels parametrised over `num_tiles ∈ {1, 8, 64}` per HQ. PCC ≥ 0.9999 (bf16-only) or ≥ 0.999 (mixed fp32) per HQ.

---

## Order of implementation (Phase 4)

1. Item 11 first (doc only — no risk, 1 commit).
2. Items 4, 8 (mechanical drops — small diff).
3. Item 5 (rename + private — mechanical).
4. Item 3 (enum rename — mechanical, large diff).
5. Item 1 (dispatch contract — touches every op struct; one commit per category header). New element exec signature is `exec(uint32_t i)` for now; item 2 extends it to `exec(uint32_t i_outer, uint32_t j_lane)` in the next step.
6. Items 7 + 6 together (trait generalisation + wait-late — both touch `eltwise_chain.inl`'s main pipeline).
7. Item 2 (chain auto-block — extends element exec signature, deletes `eltwise_block.hpp`, migrates every `Block*` call site to streaming-element spelling). Largest single change; lands after dispatch + trait machinery stabilises.
8. Items 10, 12 (new surface — added after auto-block lands so `lane_width` for `Mask` and `Cumulative` interaction with BlockSize are decided against the final API).

Each numbered group is its own commit. Item 12 enables on-device benchmarking of the wait-late improvement (item 6) — run that microbench at the end.

---

## What this proposal does NOT do

- No new convenience entry point in `eltwise_convenience.hpp`. The audit found none missing.
- No deletion of `eltwise_optional.hpp` — the `OptionalChainElement` surface remains as-is.
- No change to the caller-init contract (D5 / D8 in `eltwise_chain.hpp`). Still caller-side.
- No `CumulativeWaitPopAtEnd` for `PackTile` — packs are output-side; cumulative output policy ships only when a real consumer demands it.

---

## Awaiting sign-off

Proposal at `ttnn/cpp/ttnn/kernel_lib/agents/eltwise_helper_proposal.md`. Awaiting sign-off.
