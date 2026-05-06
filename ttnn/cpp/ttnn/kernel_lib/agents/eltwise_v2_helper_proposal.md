# Eltwise v2 Helper Proposal — Phase 3 / Gate 1

PATTERNS_HEADER: file	line	function	category	heavy_lifting	variant	loop_depth	loop_vars	sig	arg0	arg1	arg2	flow	sync_bucket	sync_seq	sync_style	shape	region_stats

Source artifacts:
- `eltwise_v2_catalog.md` — TSV demand surface + LLK API enumeration.
- `eltwise_v2_pack_analysis.md` — pack_tile usage analysis from TSV (666 rows).
- `eltwise_v2_lifecycle_analysis.md` — CB lifecycle + reconfig analysis on TSV-sampled kernels.
- `agents/eltwise_helper_lessons.md` — gathered design knowledge.
- `agents/llk_helpers_hq.md` — pipeline / general principles.

Status: **proposal**. Gate 1 is BLOCKING — no `.hpp / .inl / .cpp / kernel / test` lands until the user explicitly approves this proposal. Compression mode does not skip the gate.

Explicit anti-source-of-truth: existing `binary_op_helpers.*`, `sfpu_helpers.*`, `copy_tile_helpers.*` are NOT consumed as design input. Lessons may overlap by coincidence — the design path is TSV → catalog → lifecycle → this proposal.

---

## 1. Surface

**One** helper surface: `eltwise_chain` + a chain element type system + thin convenience wrappers. Files ship under the bare `eltwise_*` prefix — no `_v2` suffix. (The `eltwise_v2_*` working name lives only on the agent / planning artifacts in `agents/`; shipped headers drop it.)

```
ttnn/cpp/ttnn/kernel_lib/
  eltwise_chain.hpp        // public API + chain element types + policy enums + traits
  eltwise_chain.inl        // implementation (chain pipeline, init/exec dispatch)
  eltwise_activations.hpp  // SFPU op structs grouped by family
  eltwise_math.hpp
  eltwise_misc.hpp
  eltwise_predicates.hpp
  eltwise_rounding.hpp
  eltwise_scalar.hpp
  eltwise_special.hpp
  eltwise_trig.hpp
  eltwise_helpers.hpp      // aggregator (one include for callers)
```

No standalone `binary_op_helpers`, `sfpu_chain`, or `copy_tile_helpers` peer surfaces. Binary FPU, unary bcast, DEST reuse, fill, rand, copy, AND pack are all chain element types of `eltwise_chain` (per lessons §3.8 + new pack-as-chain-element finding from TSV).

The existing `binary_op_helpers.* / sfpu_helpers.* / copy_tile_helpers.*` are **frozen** during v1: no edits, no deletes. Migration moves call-sites off them tier-by-tier; once every TSV-listed kernel has switched and the files have zero remaining callers, removal lands as a separate cleanup commit (not part of v1).

`uint32_t` cb id at the public boundary; `experimental::CircularBuffer` only inside the helper (per HQ general principle).

---

## 2. Chain element catalog

### 2.1 Marker tag hierarchy (0-byte, organised by data direction)

Three roots by data direction; specific kinds derive from the matching root. Trait machinery either sweeps a root (for CB-collision / fan-out / lifecycle invariants) or a specific kind (for hoist-safety, FPU-clash, copy-load-only logic).

```cpp
// === Root tags (data direction) ===
struct CbReaderTag {};   // element reads ≥1 CB (load and/or compute that consumes a CB)
struct CbWriterTag {};   // element writes to a CB
struct DestOnlyTag {};   // element neither reads nor writes a CB (DEST-internal)

// === CB-reader kinds ===
struct CopyTileTag        : CbReaderTag {};   // pure CB → DEST move (no compute)
struct BinaryFpuTag       : CbReaderTag {};   // 2 CBs → DEST FPU compute (add/sub/mul + bcast)
struct DestReuseBinaryTag : CbReaderTag {};   // 1 CB + DEST → DEST FPU compute
struct UnaryBcastTag      : CbReaderTag {};   // 1 CB → DEST row/col/scalar broadcast

// === CB-writer kinds ===
struct PackTileTag : CbWriterTag {};          // DEST → CB store

// === DEST-only kinds ===
struct FillTileTag : DestOnlyTag {};          // const → DEST
struct RandTileTag : DestOnlyTag {};          // RNG → DEST
// SFPU op structs (UnaryOp / BinaryOp / TernaryOp / QuaternaryOp CRTP bases)
// inherit DestOnlyTag through the base — they operate on DEST slots only.

// === Traits ===
template <class T> inline constexpr bool is_cb_reader_op_v = std::is_base_of_v<CbReaderTag, T>;
template <class T> inline constexpr bool is_cb_writer_op_v = std::is_base_of_v<CbWriterTag, T>;
template <class T> inline constexpr bool is_dest_only_op_v = std::is_base_of_v<DestOnlyTag, T>;

template <class T> inline constexpr bool is_copy_tile_op_v         = std::is_base_of_v<CopyTileTag, T>;
template <class T> inline constexpr bool is_binary_fpu_op_v        = std::is_base_of_v<BinaryFpuTag, T>;
template <class T> inline constexpr bool is_dest_reuse_binary_op_v = std::is_base_of_v<DestReuseBinaryTag, T>;
template <class T> inline constexpr bool is_unary_bcast_op_v       = std::is_base_of_v<UnaryBcastTag, T>;
template <class T> inline constexpr bool is_pack_tile_op_v         = std::is_base_of_v<PackTileTag, T>;
template <class T> inline constexpr bool is_fill_tile_op_v         = std::is_base_of_v<FillTileTag, T>;
template <class T> inline constexpr bool is_rand_tile_op_v         = std::is_base_of_v<RandTileTag, T>;
```

Trait usage cheat-sheet:

| Sweep / decision | Tag predicate |
|---|---|
| Duplicate upfront-CB check across all CB-consumers | `is_cb_reader_op_v` |
| Output-CB collision / fan-out across all writers | `is_cb_writer_op_v` |
| Hoist-safety "chain shape is CopyTile + 1 SFPU op" | `is_copy_tile_op_v` (only pure loads) |
| FPU-clash reinit ("non-load FPU element present") | `is_binary_fpu_op_v` ‖ `is_dest_reuse_binary_op_v` ‖ `is_unary_bcast_op_v` |
| Hoist exclusion: element issues a pack inside the loop | `is_pack_tile_op_v` |
| No CB lifecycle to validate (pure DEST internal) | `is_dest_only_op_v` |

`is_copy_tile_op_v` keeps its existing semantics (pure load only) so hoist-safety and the existing fan-out invariants don't change. The new `is_cb_reader_op_v` is the broader sweep used by collision / lifecycle checks across all CB-consumers.

### 2.2 Compute-side bases (CRTP)

```cpp
template <class Derived, Dst Slot>                                          struct UnaryOp;
template <class Derived, Dst In0, Dst In1, Dst Out>                         struct BinaryOp;
template <class Derived, Dst In0, Dst In1, Dst In2, Dst Out>                struct TernaryOp;
template <class Derived, Dst In0, Dst In1, Dst In2, Dst In3, Dst Out>       struct QuaternaryOp;
```

`Dst` enum capped at `DEST_AUTO_LIMIT` (compile-time, never literal 8). Derived structs supply `init()` + `call()`; bases provide `apply() = init() + exec()`, slot validation, distinctness static_asserts.

### 2.3 Chain elements

| Element | Base / tag | Purpose | Notes |
|---|---|---|---|
| `CopyTile<Cb, DstSlot, Policy, IndexMode, Reconfig>` | `CopyTileTag` (← `CbReaderTag`) | wraps `copy_tile` + init / `*_with_dt` | full lifecycle policy matrix (§3.1) |
| `BinaryFpu<CbA, CbB, BinOp, OutPolicy, ReconfigDF, IndexA, IndexB, BcastDim>` | `BinaryFpuTag` (← `CbReaderTag`) | wraps `add_tiles`/`sub_tiles`/`mul_tiles` + bcast variants | independent A/B policies |
| `DestReuseBinary<Cb, BinOp, ReuseType, DstIn, DstOut, Reconfig, IndexMode>` | `DestReuseBinaryTag` (← `CbReaderTag`) | wraps `binary_dest_reuse_tiles` | DEST_TO_SRCA / DEST_TO_SRCB |
| `BinaryDestSfpu<BinSfpuOp, In0, In1, Out>` | `BinaryOp` (DEST-only via base) | wraps `add_binary_tile`, `mul_binary_tile`, `where`-style | both operands from DEST |
| SFPU op structs: `Exp`, `Log`, `Sqrt`, `Rsqrt`, `Sigmoid`, `Tanh`, `Relu`, `Mask<DF, DataSlot>`, `Where<DF, ...>`, etc. | `UnaryOp` / `BinaryOp` / `TernaryOp` / `QuaternaryOp` (DEST-only via base) | wraps `*_tile` + `*_tile_init` | template params for approx / legacy / iterations |
| `UnaryBcast<Dim, Cb, DstSlot, Policy, Reconfig>` | `UnaryBcastTag` (← `CbReaderTag`) | wraps `unary_bcast<BroadcastType>` | row / col / scalar |
| `FillScalar<float|int, DstSlot>`, `FillBitcast<DstSlot>`, `FillInt<DF, DstSlot>` | `FillTileTag` (← `DestOnlyTag`) + `UnaryOp` | wraps `fill_tile`, `fill_tile_int`, `fill_tile_bitcast` | runtime value via ctor |
| `RandTile<DstSlot>` | `RandTileTag` (← `DestOnlyTag`) + `UnaryOp` | wraps `rand_tile` + seeded init | seed via ctor |
| `PackTile<Cb, DstSlot, Policy, IndexMode, Reconfig>` | `PackTileTag` (← `CbWriterTag`) | wraps `pack_tile` + lifecycle | NEW — see §4 |
| `PackTileBlock<Cb, DstSlots..., ...>` | `PackTileTag` (← `CbWriterTag`) | wraps `pack_tile_block` | multi-slot atomic pack |

CRTP bases (`UnaryOp`/`BinaryOp`/`TernaryOp`/`QuaternaryOp`) inherit `DestOnlyTag` so every SFPU op struct + `BinaryDestSfpu` automatically participates in the `is_dest_only_op_v` sweep without per-op boilerplate.

### 2.4 What is NOT a chain element

- `tile_regs_acquire/commit/wait/release` — owned by `eltwise_pipeline` internally, not a chain element. The chain emits a single canonical modern[ACWR] window per chain invocation by default.
- `compute_kernel_hw_startup` — owned by the helper as a one-time prologue; per HQ rule the helper standardises on this single boot init.
- Held-DEST patterns — out of scope (TSV-sampled survey found 0 genuine cases; lessons §3.7 stays).

---

## 3. Policy enums

### 3.1 `CopyTilePolicy` (input CB lifecycle for CopyTile)

```cpp
enum class CopyTilePolicy {
    WaitAndPop,            // per-tile wait + per-tile pop  (default)
    WaitNoPop,             // per-tile wait + no pop        (fan-out first / persistent)
    NoWaitPop,             // no wait     + per-tile pop    (fan-out last / pre-waited)
    NoWaitNoPop,           // no wait     + no pop          (caller owns lifecycle)
    WaitUpfrontPopAtEnd,   // upfront wait + upfront pop    (block access, BlockIter / Absolute index legal)
};
```

Lessons §2.1 matrix verbatim. Cumulative wait stays unsupported (lessons §2.1 + TSV evidence: only layernorm_pre_allgather, 1 kernel; raw LLK fine).

### 3.2 `CbIndexMode` (per-CB-operand)

```cpp
enum class CbIndexMode {
    FirstTile,    // always tile 0
    BlockIter,    // tile i (loop var)
    Pinned,       // fixed compile-time/runtime k
    Absolute,     // runtime idx
};
```

Compile-time validation matrix (lessons §2.7) — illegal combinations are `static_assert` failures (e.g. `WaitAndPop + BlockIter`).

### 3.3 Binary FPU policies

```cpp
enum class BinaryFpuOutputPolicy {
    PerTile,             // default — release/acquire each tile
    HoistAcquireRelease, // single acquire/release wrapping the whole loop
};

enum class BinaryDataFormatReconfig {
    NONE,
    INPUT,             // srca and/or srcb on entry
    OUTPUT,            // pack reconfig on entry
    INPUT_AND_OUTPUT,  // both (default — safest, no skip)
};

enum class BroadcastDim { NONE, ROW, COL, SCALAR };
```

A and B operand policies (`CopyTilePolicy`) and indexing (`CbIndexMode`) are independent template params on `BinaryFpu`. Same-CB dedup is helper-side (lessons §3.6).

### 3.4 DEST reuse policy

```cpp
enum class DestReuseType {
    DEST_TO_SRCA,  // CB → srcb, DEST → srca
    DEST_TO_SRCB,  // CB → srca, DEST → srcb
};

enum class DestReuseReconfig { None, Input };  // NOT a bool (lessons §2.5)
```

`DestReuseBinary::clashes_with_fpu = true` by default (lessons §3.2).

### 3.5 PackTile policies (NEW)

```cpp
enum class PackTilePolicy {
    PerTileReserveAndPush,    // cb_reserve_back(1); pack; cb_push_back(1)              — default
    PerTileReserveNoPush,     // reserve happens, push deferred to caller                — rare
    NoReservePushAtEnd,       // pack into pre-reserved CB, push later                   — bulk
    NoReserveNoPush,          // caller owns reserve+push                                — full caller control
    UpfrontReservePushAtEnd,  // reserve N upfront, pack sequentially, push N at end     — block
};
```

`PackTileIndexMode` mirrors `CbIndexMode` for the **output tile index** (i.e. `pack_tile(dst, cb, output_index)`):

```cpp
enum class PackTileIndexMode { FirstTile, BlockIter, Pinned, Absolute };
```

`PackTileReconfig` covers the pack-side dtype reconfig (per `pack_reconfig_data_format`):

```cpp
enum class PackTileReconfig { None, Output, OutputConditional /* old_cb → new_cb */ };
```

Justification (TSV evidence):
- 359/666 (54%) `modern-canonical/single` → `PerTileReserveAndPush`.
- 209/666 (31%) multi-pack-in-window → `UpfrontReservePushAtEnd` or chain composition (§4.2).
- 126/666 (19%) variable DEST index → `BlockIter` / `Absolute` index modes.
- 101/157 files (64%) pack into ≥2 CBs → CB is per-element template param, not chain-wide.

### 3.6 Sync style

**Modern only.** No enum. The chain pipeline emits `tile_regs_acquire/commit/wait/release` and nothing else.

```cpp
// (no DestSyncStyle enum — single canonical sync path)
```

Both deprecated styles (`acquire_dst/release_dst` lowercase APIs and the kernel-local `ACQ()/REL()` macros that wrap them) are NOT emitted. Kernels currently on either deprecated style migrate their dst-sync to modern as part of adopting the chain — that surface change rides alongside the helper conversion in the same commit. Carrying a `RawDst` opt-in would freeze the deprecated APIs as a permanent helper surface; the helper exists to drive forward, not to preserve.

TSV impact: 57 raw-dst rows + 43 ACQ-REL-macro rows (≈15% of pack call-sites, concentrated in `ttnn-op:transformer/sdpa` and a handful of moreh/conv kernels) need a sync-style rewrite as a prerequisite to chain adoption. Recorded as a migration-time cost in §9.2 / §9.3 — not a helper-surface gap.

---

## 4. PackTile as chain element

### 4.1 Why a chain element

Same reasoning as `CopyTile`-as-chain-element:

- One pipeline owns `tile_regs_acquire/commit/wait/release` lifecycle. Putting pack inside the chain means the chain combinator can reason about the full window (compute + pack) rather than splitting ownership across helper boundaries.
- Multi-slot pack (TSV: 31% of rows) and multi-CB pack (TSV: 64% of files) are first-class composition needs. Chain composition handles both natively.
- The pack-side dtype reconfig (`pack_reconfig_data_format`) parallels the input-side `*_with_dt` reconfig — same policy enum surface (None / Output / OutputConditional).
- Chain traits already model "what hardware state does this element touch" — pack adds a new trait `is_pack_tile_op_v` that the chain combinator uses for window-boundary decisions.

### 4.2 Composition shapes

```cpp
// Streaming unary (54% baseline):
eltwise_chain(
    CopyTile<cb_in,  Dst::D0, CopyTilePolicy::WaitAndPop>{},
    Exp<>{},                                    // operates on Dst::D0
    PackTile<cb_out, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{}
);

// Multi-pack-in-window (31% — fan-out pack):
eltwise_chain(
    CopyTile<cb_in, Dst::D0, CopyTilePolicy::WaitAndPop>{},
    Exp<>{},                                                   // Dst::D0
    Sigmoid<Dst::D1>{},                                        // ... assume reads from D0 internally — N/A; for fan-out:
    CopyTile<cb_in, Dst::D1, CopyTilePolicy::NoWaitPop>{},     // re-read same CB into D1
    Tanh<Dst::D1>{},
    PackTile<cb_out_a, Dst::D0, PackTilePolicy::PerTileReserveAndPush>{},
    PackTile<cb_out_b, Dst::D1, PackTilePolicy::PerTileReserveAndPush>{}
);

// SDPA-style 4-slot fan-out pack (`fused_max_sub_exp_add_block`):
eltwise_chain(
    CopyTile<cb_prev_max, Dst::D0, NoWaitPop>{},
    CopyTile<cb_worker_max, Dst::D1, NoWaitPop>{},
    CopyTile<cb_prev_sum, Dst::D2, NoWaitPop>{},
    CopyTile<cb_worker_sum, Dst::D3, NoWaitPop>{},
    FusedMaxSubExpAdd<...>{},
    PackTile<cb_cur_max, Dst::D0, ...>{},
    PackTile<cb_exp_max_diff, Dst::D2, ...>{},
    PackTile<cb_cur_sum, Dst::D3, ...>{},
    PackTile<cb_exp_max_diff_2, Dst::D1, ...>{}
);

// Block reduction with upfront reserve (UpfrontReservePushAtEnd):
eltwise_chain<eltwise_chain_options{ .upfront_block_size = N }>(
    CopyTile<cb_in, Dst::D0, WaitUpfrontPopAtEnd, CbIndexMode::BlockIter>{},
    Exp<>{},
    PackTile<cb_out, Dst::D0, UpfrontReservePushAtEnd, PackTileIndexMode::BlockIter>{}
);
```

### 4.3 Atomic vs composable multi-pack

Two evidence-based shapes:

1. **Composable** (default): `PackTile{} >> PackTile{} >> PackTile{}` — N independent chain elements, each with its own DEST slot, CB, policy, and reconfig. Chain combinator emits them sequentially within one tile_regs_wait window. Lessons §3.5 already mandates this for fan-out: do NOT auto-merge.

2. **Atomic block** (`PackTileBlock<Cb, Dst::D0, Dst::D1, Dst::D2, Dst::D3>{}`): wraps `pack_tile_block`. Used when the underlying LLK supports a single multi-tile pack call (consecutive DEST slots, single CB). Smaller code footprint, but constrained to "consecutive DEST slots → one CB". Optional convenience element.

The TSV pattern `pack_tile_block` (`pack.h::pack_tile_block(ifrom_dst, icb, ntiles)`) is rare in the call-site data — most multi-pack windows use N `pack_tile` calls. Default is composable; atomic is an opt-in.

### 4.4 Trait surface

`is_pack_tile_op_v` (`std::is_base_of_v<PackTileTag, T>`) is declared once in §2.1 alongside the rest of the tag predicates and inherits the `CbWriterTag` root, so the broader `is_cb_writer_op_v` sweep already covers PackTile + any future writer kind without re-declaration.

Chain-shape traits specific to PackTile:

```cpp
template <class Chain> inline constexpr bool chain_has_any_pack_tile_v       = ...;  // any element is_pack_tile_op_v
template <class Chain> inline constexpr bool chain_has_no_pack_tile_v        = ...;  // negation; useful for chains the caller packs externally
template <class Chain> inline constexpr bool chain_pack_writes_share_cb_v    = ...;  // 2+ PackTile elements with same Cb (legal but flagged for review)
template <class Chain> inline constexpr bool chain_pack_writes_collide_v     = ...;  // 2+ PackTile elements with same (Cb, output_index, DstSlot) — ILLEGAL
```

Chain combinator validates at compile time:
- No two `PackTile` elements write the same `(Cb, output_index, DstSlot)` triple — `static_assert(!chain_pack_writes_collide_v<Chain>)`.
- `PackTilePolicy` consistent with surrounding `CopyTile` policies (e.g. `UpfrontReservePushAtEnd` requires the loop to be expressed as such — a static_assert checks the chain options).
- `PackTileIndexMode::BlockIter` is only legal under `UpfrontReservePushAtEnd` (parallel to the `CopyTilePolicy + CbIndexMode` matrix from lessons §2.7).

### 4.5 What is explicitly NOT in PackTile

- L1 accumulation flag (`pack_reconfig_l1_acc`) — separate concern; not in scope for v1. If a kernel needs it, add a `PackTilePolicy::L1AccumPushAtEnd` later under a focused proposal.
- ReLU on pack (`pack_relu_config`) — separate concern; deferred to a follow-up proposal.
- Row-mode pack (`pack_rows`, `pack_rows_init`, `pack_rows_uninit`) — different lifecycle (init/uninit pair). Out of v1 scope.

---

## 5. Reconfig surface

Every helper-emitted reconfig is a first-class option, expressed via `with_dt_tree`-style decomposition. Mapping per element type:

| Element | Reconfig policy | Underlying LLK calls |
|---|---|---|
| `CopyTile` | `CopyTileReconfig::{None, Input}` | `copy_tile_to_dst_init_short_with_dt(old_cb, new_cb)` |
| `BinaryFpu` | `BinaryDataFormatReconfig::{None, INPUT, OUTPUT, INPUT_AND_OUTPUT}` | per-side: `reconfig_data_format_srca`, `reconfig_data_format_srcb`, `pack_reconfig_data_format` |
| `DestReuseBinary` | `DestReuseReconfig::{None, Input}` | `reconfig_data_format_srca` OR `reconfig_data_format_srcb` (per `ReuseType`) |
| `UnaryBcast` | `UnaryBcastReconfig::{None, Input}` | `reconfigure_unary_bcast<old, new>(old_icb, new_icb, old_ocb, new_ocb)` |
| `PackTile` | `PackTileReconfig::{None, Output, OutputConditional}` | `pack_reconfig_data_format(new_cb)` or `pack_reconfig_data_format(old_cb, new_cb)` |
| SFPU op structs | (init only — no reconfig surface) | `*_tile_init` |

All reconfigs are entry-time per chain element. Mid-loop dtype swaps (lessons §3.7 corollary): out of scope for v1; flagged as a gap in the gap map.

Reconfig naming matches lessons §2.4 — `Input` (always srca for input-side LLK calls), `Output` (always pack-side), `INPUT_AND_OUTPUT` (binary case).

`with_dt_tree` decomposition is documented inline in `eltwise_chain.hpp`'s doc-comment (lessons §5.4).

---

## 6. Init / startup surface

Single canonical prologue per chain instantiation:

```cpp
eltwise_pipeline_init<Chain>();
```

Emits exactly one `compute_kernel_hw_startup(icb0, icb1, ocb)` (or single-input overload) at compile-time deduced from the chain's first input CB and last pack target. Per HQ rule "standardise on `compute_kernel_hw_startup` as the single boot init."

After the prologue, each chain element's `init()` runs once at chain entry (or per-tile if `chain_is_hoist_safe_v` is false). Hoisting is opt-in, narrowly scoped (lessons §3.4).

`fp32_dest_acc` enable/disable — chain options carried as a compile-time NTTP struct so every field participates in `static_assert` validation and dispatch deduction:

```cpp
struct EltwiseChainOptions {
    bool     enable_fp32_dest_acc = false;
    uint32_t upfront_block_size   = 0;   // > 0 enables UpfrontReservePushAtEnd policies
    // dest sync is always modern — no field, no enum (see §3.6).
};

// Used as a non-type template parameter:
template <EltwiseChainOptions Opts = {}, class... Elts>
ALWI void eltwise_chain(Elts... elts);
```

Compile-time-first design rule (cross-cutting):

- **Default to compile-time.** Every chain-shape decision (lifecycle policy, index mode, reconfig mode, broadcast dim, DEST slot, hoist gating, FPU-clash, fan-out, multi-pack invariants) is a template parameter or NTTP. The chain combinator validates the whole chain at compile time and emits zero per-tile branches for state the type system already pinned.
- **Runtime is the exception, not the default.** A field becomes runtime only when the *value* is genuinely runtime — that is the kernel does not know it at template-instantiation time. The narrow set of legitimately runtime values:
  - `CbIndexMode::Pinned k` / `Absolute idx` — the index itself, when not a compile-time constant. The *mode* is compile-time; the index field is runtime (lessons §4.1, `cb_tile_idx_`).
  - `FillScalar::value`, `FillInt::value` — runtime scalar payloads.
  - `RandTile::seed` — runtime RNG seed.
  - `Power::exponent`, `LogWithBase::base_scale`, etc. — op-struct ctor args specific to ops whose payload is genuinely runtime.
- **No "usually compile-time but occasionally runtime" fields.** Lessons §1.2 strengthening: a field that is sometimes one and sometimes the other gets two op-structs (or a template specialization), not a runtime field with a constexpr default. Mixing the two axes in one struct breaks chain-trait deduction and forces every consumer to handle both arms.
- **No runtime overhead for compile-time-decidable state.** If two policies map to identical LLK sequences, they are still distinct compile-time enum values — the dispatch layer collapses them. Runtime branching on policy is forbidden.

---

## 7. CB lifecycle policy mapping

From `eltwise_v2_lifecycle_analysis.md` Section 6:

| TSV-observed lifecycle | CopyTilePolicy | PackTilePolicy |
|---|---|---|
| PerTile (39%) | `WaitAndPop` | `PerTileReserveAndPush` |
| UpfrontBlock (22% kernels) | `WaitUpfrontPopAtEnd` | `UpfrontReservePushAtEnd` |
| UpfrontKernel / persistent (22%) | `WaitNoPop` (one-shot upfront) | n/a (output is per-tile) |
| Cumulative (1 kernel) | UNSUPPORTED — flag as gap | UNSUPPORTED |
| Caller-managed | `NoWaitNoPop` | `NoReserveNoPush` |
| Pre-pushed streaming | `NoWaitPop` | n/a (input only) |

Cumulative wait remains an open gap (`GAP-CUMULATIVE-WAIT`) — single kernel demand, structural complexity high, lessons §2.1 documents it as unsupported until justified.

---

## 8. Convenience entry points (thin forwarders, not standalone helpers)

Picked by frequency-of-use (lessons §3.8). Each is a one-line `inline` forwarder to `eltwise_chain(...)`. No own policy enums, no own validation kernels. All live in `eltwise_chain.hpp` (or sibling `eltwise_convenience.hpp`).

```cpp
// FPU binary — frequently-invoked patterns
inline void binary_add(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out, uint32_t n_tiles);
inline void binary_sub(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out, uint32_t n_tiles);
inline void binary_mul(uint32_t cb_a, uint32_t cb_b, uint32_t cb_out, uint32_t n_tiles);
inline void binary_add_bcast(uint32_t cb_a, uint32_t cb_b, BroadcastDim dim, uint32_t cb_out, uint32_t n_tiles);

// Unary SFPU — single chain
template <class SfpuOp> void unary_op(uint32_t cb_in, uint32_t cb_out, uint32_t n_tiles, SfpuOp op = {});

// DEST-reuse mul (square-style)
inline void dest_reuse_mul(uint32_t cb_in, Dst dst_slot, uint32_t cb_out, uint32_t n_tiles);

// Pure copy (covers `copy_tile_helpers` use cases without a peer helper)
inline void copy(uint32_t cb_in, uint32_t cb_out, uint32_t n_tiles);
inline void copy_with_dt(uint32_t cb_in, uint32_t cb_out, uint32_t n_tiles);
```

Convenience set for v1 is small; expand only when a pattern recurs in many kernels (per lessons §3.8 picking criterion).

---

## 9. Migration scope

### 9.1 In scope for v1 — helper surface

Helper surface delivered in v1:

- All TSV `(heavy_lifting, flow)` buckets covered by current LLK headers (~624/666 rows = 93.7% per catalog Section "Coverage Summary"):
  - `eltwise-fpu-binary → eltwise-binary→pack` (148 rows)
  - `eltwise-fpu-bcast → eltwise-bcast→pack` (111 rows)
  - `sfpu-unary → copy→sfpu-tile→pack` (169) + sub-flows (45, 21, 16, 15) — all SFPU unary headers
  - `copy → copy→pack` (76 rows)
- Modern dst sync only — `tile_regs_acquire/commit/wait/release`. No deprecated path.
- Entry-only and per-stage dtype reconfig with srca/srcb/pack coverage.

### 9.2 In scope for v1 — mandatory kernel-side migration prerequisites

These are NOT helper-surface items, but they are **mandatory v1 work** that must land before / alongside the corresponding kernel's chain adoption. Treat each as a per-kernel rewrite required by the helper's modern-only sync contract (§3.6).

| Prereq | What | Evidence | Action — required in v1 |
|---|---|---|---|
| `MIGRATE-RAW-DST-SYNC` | `acquire_dst/release_dst` (lowercase, deprecated) | sdpa / transformer (~57 TSV rows) | Rewrite to `tile_regs_acquire/commit/wait/release` as the first step of each kernel's chain-adoption commit. Mandatory for every Tier D kernel. |
| `MIGRATE-ACQ-REL-MACROS` | kernel-local `ACQ()/REL()` macros wrapping the deprecated calls | normalization legacy (~43 TSV rows) | Delete the macro definitions and rewrite call sites to modern dst sync. Mandatory for every Tier E kernel. |

Per-kernel sequencing (the migration commit follows this order):

1. Rewrite dst sync → modern. Run the kernel's existing pytest unchanged. PCC must hold.
2. Adopt the chain (`eltwise_chain(...)`). Run the same pytest. PCC must hold.

Step 1 is a no-op semantically (modern and deprecated APIs map to the same hardware sequence) — it exists to land the syntactic rewrite as a discrete, reviewable change. Step 2 is the actual chain conversion.

The pytest manifest gains a `dst_sync_rewritten:` annotation per kernel as it migrates, so downstream sweeps can detect any kernel still on the deprecated API.

### 9.3 Out of scope for v1 (gap map)

| GAP | What | Evidence | Action |
|---|---|---|---|
| `GAP-CUMULATIVE-WAIT` | cumulative `cb_wait_front(base + i)` policy | layernorm_pre_allgather (1 kernel) | document; helper does not model |
| `GAP-WELFORD` | welford SFPU LREG state | 15 rows | separate helper (already exists / planned), not eltwise scope |
| `GAP-TRANSPOSE` | transpose→pack | 29 rows | separate helper (transpose_block_helpers exists), not eltwise scope |
| `GAP-PACK-L1-ACC` | `pack_reconfig_l1_acc` | unmeasured | future PackTilePolicy extension |
| `GAP-PACK-RELU` | `pack_relu_config` | unmeasured | future PackTilePolicy extension |
| `GAP-PACK-ROWS` | `pack_rows` row-mode | unmeasured | separate proposal (different lifecycle) |
| `GAP-MID-LOOP-RECONFIG` | dtype swap mid-loop | layernorm_sharded_post_allgather class | document; kernels stay raw LLK on those blocks |

### 9.4 Migration tier (per-kernel rollout sequence)

Tier order is "lowest risk → highest" judging by sync_style + shape. **All five tiers are mandatory v1 migration scope** — no tier is deferred or made optional.

1. Tier A (modern + canonical/single, 359 rows / 54%): simplest call sites, single-tile chains. Migrates first.
2. Tier B (modern + single-in-loop, 151 rows / 23%): per-tile loop chains.
3. Tier C (modern + multi-loop / multi-pack-in-window, 56 rows / 8%): exercise PackTile composition + `UpfrontReservePushAtEnd`.
4. Tier D (raw-dst, 57 rows): each kernel runs `MIGRATE-RAW-DST-SYNC` (§9.2 Step 1) then adopts the chain in the same commit. Mandatory.
5. Tier E (ACQ-REL-macro, 43 rows): each kernel runs `MIGRATE-ACQ-REL-MACROS` (§9.2 Step 1) then adopts the chain in the same commit. Mandatory.

v1 ships the helper + validation suite + the kernel migrations across all five tiers. The migration cycle is part of v1, not a follow-up — the deprecated dst-sync APIs do not survive into v2.

---

## 10. Validation strategy (high-level — full plan goes to Gate 2)

Validation kernels exercise the surface of v1 directly. Each kernel pairs with a torch golden + parameterized over `num_tiles ∈ {1, 8, 64}` and `fp32_dest_acc_en ∈ {False, True}` where applicable.

Sketch — validation kernel families (full plan in `eltwise_v2_test_plan.md` after Gate 1 sign-off):

1. **CopyTile lifecycle matrix** — every `CopyTilePolicy × CbIndexMode` legal cell, including `WaitUpfrontPopAtEnd + BlockIter`.
2. **BinaryFpu** — same-CB dedup, independent A/B policies, all four reconfig modes incl. srcA-only and srcB-only paths (lessons §7.1).
3. **PackTile lifecycle matrix** — every `PackTilePolicy × PackTileIndexMode` legal cell. Cross-product validates `pack_reconfig_data_format` paths.
4. **Multi-element chain** — `CopyTile + SFPU + PackTile`, `2× CopyTile + BinaryFpu + PackTile`, `CopyTile + BinaryFpu + SFPU + PackTile`.
5. **DestReuseBinary** — DEST_TO_SRCA + DEST_TO_SRCB, both reconfig variants, FPU-clash reinit.
6. **Fan-out pack** — N CopyTile + N PackTile across multiple CBs, single acquire/release window.
7. **Hoist-safe shape** — `CopyTile + 1 SFPU + PackTile` with hoisting opt-in; verify identical output to per-tile init.
8. **fp32_dest_acc_en sweep** — every binary FPU + DestReuseBinary + PackTile combo under `fp32_dest_acc_en ∈ {False, True}`.

Test home: `ttnn/cpp/ttnn/kernel_lib/tests/eltwise_v2/`. Pytest at `tests/ttnn/unit_tests/kernel_lib/test_eltwise_v2.py`.

The full test list (kernel-by-kernel: shape, num_tiles, dtype matrix, PCC threshold) is the Gate 2 artifact — written separately after Gate 1 approval.

---

## 11. Open questions — RESOLVED

1. ~~**`DestSyncStyle::RawDst` in v1 yes/no?**~~ **RESOLVED — modern-only.** Helper emits `tile_regs_acquire/commit/wait/release` and nothing else. Kernels on the deprecated `acquire_dst/release_dst` API (or the `ACQ()/REL()` macros that wrap it) rewrite their dst-sync to modern as part of adopting the chain — same commit. Mandatory v1 work, all five tiers. (See §3.6, §9.2, §9.4.)
2. ~~**`PackTileBlock` (atomic `pack_tile_block`) ship in v1?**~~ **RESOLVED — yes.** Ships as opt-in convenience element; small surface, low risk. Default multi-pack is composable (N independent `PackTile` chain elements); `PackTileBlock` is the atomic shorthand for "consecutive DEST slots → one CB" (§4.3).
3. ~~**`pack_reconfig_l1_acc` / `pack_relu_config`?**~~ **RESOLVED — defer.** Out of v1; recorded in §9.3 gap map (`GAP-PACK-L1-ACC`, `GAP-PACK-RELU`). Future PackTilePolicy extensions under a focused proposal when demand is measured.
4. ~~**Convenience entry points list?**~~ **RESOLVED — ship.** The nine listed forwarders (binary_add / binary_sub / binary_mul / binary_*_bcast / unary_op / dest_reuse_mul / copy / copy_with_dt) ship in v1 because they wrap the dominant TSV buckets. They stay thin inline forwarders to `eltwise_chain(...)` — no own policy enums, no own validation kernels (§8).
5. ~~**Naming**~~ **RESOLVED — `eltwise_*` (no `_v2` suffix).** Helper ships at `ttnn/cpp/ttnn/kernel_lib/eltwise_chain.{hpp,inl}` + the eight sibling op-family headers (§1). The existing `binary_op_helpers.* / sfpu_helpers.* / copy_tile_helpers.*` are **frozen** — no edits, no deletes — until every migrating kernel has switched to `eltwise_chain` and the old surface has zero call sites. Removal is a separate post-migration cleanup commit, not part of v1 helper land.
6. ~~**`EltwiseChainOptions` runtime vs compile-time?**~~ **RESOLVED — compile-time first.** `EltwiseChainOptions` is a compile-time NTTP struct (§6). Per the new compile-time-first design rule: every chain-shape decision is a template parameter; runtime fields are restricted to genuinely runtime values (Pinned/Absolute index, fill scalar, rand seed, op-struct ctor args). Goal is zero per-tile branches for state the type system already pinned, and `static_assert`-driven validation of the whole chain at instantiation.

---

## 12. Acceptance summary

Approving this proposal authorizes:

- Writing `eltwise_chain.hpp / .inl` plus the eight sibling op-family headers (per §1).
- Implementing the chain element types in §2, policy enums in §3, PackTile design in §4, reconfig surface in §5, init in §6, lifecycle mapping in §7, convenience set in §8.
- Migrating all five tiers of production kernels in v1 (per §9.4) — including the mandatory kernel-side dst-sync rewrites for Tier D (raw-dst) and Tier E (ACQ-REL-macros) per §9.2.
- Beginning Gate 2 (validation test plan) — once Gate 2 is approved, kernels + pytests land per §10.

Approving this proposal does NOT authorize:

- Writing implementation files yet (Gate 1 covers design only — implementation begins after Gate 2 test plan approval, per pipeline).
- Removing the existing `binary_op_helpers / sfpu_helpers / copy_tile_helpers` files (deferred until migration completes).

Mid-implementation deltas to this design re-enter Gate 1 as a new commit — no in-place edits to the approved artifact (HQ rule).

Proposal at `ttnn/cpp/ttnn/kernel_lib/agents/eltwise_v2_helper_proposal.md`. Awaiting sign-off.
