# Eltwise Helper Design — Lessons Learned

Use this when writing the prompt for **creating eltwise helpers** (binary FPU + SFPU chain + DEST reuse).

The order is the order you should think in: type system → policy surface → composition → optimization → testing → migration enablement.

---

## 1. Op struct shape

### 1.1 CRTP bases for boilerplate elimination

Three bases cover every eltwise SFPU op shape. Derived structs only define `init()` + `call()`:

```cpp
template <typename Derived, Dst Slot>                                          struct UnaryOp;
template <typename Derived, Dst In0, Dst In1, Dst Out>                         struct BinaryOp;
template <typename Derived, Dst In0, Dst In1, Dst In2, Dst Out>                struct TernaryOp;
template <typename Derived, Dst In0, Dst In1, Dst In2, Dst In3, Dst Out>       struct QuaternaryOp;
```

The base provides `dst_idx`/`in0`/`in1`/`out`, `max_dst`, `static_assert(<8)`, `exec()`, and `apply()` (which calls `init()` + `exec()`). Derived structs are 4 lines. Don't reintroduce hand-written `apply()`/`exec()` per op.

`QuaternaryOp` covers 4-input SFPU ops (e.g. fused mask + scale + bias). DEST `static_assert` checks `Out + 1 < DEST_AUTO_LIMIT` and that all input slots are distinct.

### 1.2 Op struct templates carry ALL compile-time state

DEST slot, approximation mode, legacy/exact toggles, datatype enums — all template parameters with sane defaults. Runtime fields exist only for genuinely runtime data (e.g. `FillScalar::value`, `LogWithBase::base_scale`, `Power::exponent`). No runtime op dispatch ever leaks into the chain element. Examples:

```cpp
template <Approx approx = Approx::Exact, Approx fast = Approx::Fast, Dst Slot = Dst::D0> struct Exp;
template <Legacy legacy = Legacy::Off, Approx approx = Approx::Exact, Dst Slot = Dst::D0> struct Rsqrt;
template <DataFormat DF = DataFormat::Float16_b, Dst DataSlot = Dst::D0>                  struct Mask;
```

### 1.3 Self-documenting enums for template params

Boolean template params are unreadable at the call site. Use named enums:

```cpp
enum class Approx : bool { Exact = false, Fast = true };
enum class Legacy : bool { Off = false, On = true };
enum class Dst : uint32_t { D0=0, D1=1, ..., D7=7 };
```

Cap the DEST slot enum at `DEST_AUTO_LIMIT`, never a literal `8` — see [llk_helpers_hq.md → DEST capacity is compile-time, never literal](llk_helpers_hq.md#dest-capacity-is-compile-time-never-literal) for the general rule. The `Dst::D0..D7` naming is the eltwise-specific surface; the bound that gates each enum value is the shared `DEST_AUTO_LIMIT` constexpr.

### 1.4 Mirror hardcoded LLK contracts in the type

`mask_tile` always reads the mask from `DataSlot + 1` regardless of what slot index you pass. Encode that at compile time so the user can't violate it:

```cpp
template <DataFormat DF, Dst DataSlot>
struct Mask : BinaryOp<Mask<DF, DataSlot>, DataSlot,
                       static_cast<Dst>(static_cast<uint32_t>(DataSlot) + 1),
                       DataSlot> {
    static_assert(static_cast<uint32_t>(DataSlot) < 7, "...");
};
```

If the underlying LLK has a layout convention, bake it into the struct template instantiation. Don't expose the redundant slot as a free parameter.

### 1.5 Tag types over inheritance trees

`CopyTileTag {}` is a 0-byte marker; `is_copy_tile_op_v<T> = std::is_base_of_v<CopyTileTag, T>`. That's the entire dispatch surface for "the chain treats this element as a CB load." Same pattern for any future cross-cutting category. Don't reach for std::variant, virtual dispatch, or std::any here — compile-time tags are free.

Classification is not dispatch. A tag system that classifies elements into categories and a dispatch surface that calls per-element methods are two contracts, and they can drift — e.g. classification places an element in a category whose pipeline path uses a static call, but the element actually carries runtime state via an instance method. The wrong path runs silently because both methods exist and the compiler can't catch the mismatch. Pick one dispatch shape (e.g. a single member call taking the loop index) for every element; the tag classifies, the call signature is uniform.

### 1.6 `if constexpr` for opt-in capabilities

When a subset of chain elements opt into a behavior (upfront wait/pop, FPU clash, etc.), give every element a static constexpr boolean trait (`is_upfront`, `clashes_with_fpu`, ...) with a sane default, and gate calls in the chain cascade with `if constexpr`:

```cpp
if constexpr (First::is_upfront) {
    first.wait_upfront(n);
}
rest.wait_upfront(n);
```

Every element exposes the same compile-time surface; non-opted-in elements declare the trait `false` (often inherited from a CRTP base default) and the branch compiles out. No SFINAE detectors, no `std::void_t` walls — `if constexpr` on a member trait is enough and it reads the same as the rest of the chain code.

### 1.7 Static-member traits for cross-element invariants

Same idea — static constexpr traits — used to enforce cross-element rules at compile time. Example: "no two upfront CB-input elements share a CB."

`CopyTile` declares `static constexpr bool is_upfront` and `static constexpr uint32_t cb`. `DestReuseOp` declares the same pair. The chain walks every pair of elements and asserts no duplicate `cb` among those with `is_upfront == true`. Any future CB-input op opts in by declaring the same two statics — the `chain_has_duplicate_upfront_cbs_v` check covers it for free.

Stub-default member functions are a silent footgun. Whenever a downstream check short-circuits on a stub return value (an id stub returning 0, a flag stub defaulting false), an element that forgot to override it passes the check by accident — no diagnostic. Force the contract at chain entry via a static_assert that names the missing override. Failures fire at the override site, not at a downstream trait whose meaning depends on the absent value.

### 1.8 Fill / rand tiles tagged distinctly from CB-load ops

`FillTileTag {}` and `RandTileTag {}` are 0-byte markers separate from `CopyTileTag`. They denote chain elements that *write* a DEST slot from constants / RNG state — they do NOT consume from a CB.

Why a separate tag: chain traits like `chain_loads_share_cb_v`, `chain_has_duplicate_upfront_cbs_v`, and `is_copy_tile_op_v` would mis-classify fill/rand under `CopyTileTag` (no CB to wait/pop, no indexing, no fan-out semantics). Conflating the two has shipped double-wait bugs in past chain combinator logic.

`FillScalar`, `RandTile` derive from the new tag. They expose `is_upfront = false`, `clashes_with_fpu = false` by default.

---

## 2. Policy surface

### 2.1 CopyTilePolicy must cover every wait shape the kernels need

`CopyTilePolicy` is two orthogonal axes — **when** the wait happens, and **whether** the pop happens. Cover all real shapes; do not ship a partial matrix and "extend later."

Wait shapes (orthogonal to pop):

| Wait shape | When `cb_wait_front` fires | Used for |
|---|---|---|
| Per-tile | inside the tile loop, count = 1 | streaming input, default |
| Upfront | once before the tile loop, count = N (block size) | block-at-a-time access, indexed reads of a pre-waited region |
| Cumulative | inside the tile loop, count = `base + i` (grows each iteration) | producer-aligned waits where prior tiles must remain visible — **flag explicitly**, structurally hard, ships only when a real consumer demands it |
| None | never (caller waited externally) | sharded / pre-waited tensors, fan-out second-load |

Pop shapes (orthogonal to wait):

| Pop shape | When `cb_pop_front` fires | Used for |
|---|---|---|
| Per-tile | inside the tile loop, count = 1 | streaming consume |
| Upfront-end | once after the tile loop, count = N | block-at-a-time consume after upfront wait |
| None | never | persistent / fan-out first / caller-managed |

The shipped `CopyTilePolicy` enum picks the combinations the kernels actually need:

```
WaitAndPop           per-tile wait    + per-tile pop      streaming, default
WaitNoPop            per-tile wait    + no pop            persistent / fan-out first
NoWaitPop            no wait          + per-tile pop      fan-out last / pre-waited single
NoWaitNoPop          no wait          + no pop            caller owns CB lifecycle
WaitUpfrontPopAtEnd  upfront wait     + upfront-end pop   block access with indexed CopyTile
```

`cb_wait_front` is idempotent so `WaitNoPop + WaitAndPop` works for fan-out — but `WaitNoPop + NoWaitPop` is one fewer NoC roundtrip and the "first already waited" relationship is searchable at the call site. Add a combination the moment you know it can save a wait or clarify intent. Cumulative wait is recorded as a known-unsupported lifecycle in the migration blocker list (§5 / HQ doc) until a kernel justifies the helper-side complexity.

Lifecycle wants to be a type, not an enum, when each value maps to genuinely different code (different wait/pop calls, different valid index modes). A type carries its own valid-companion sets and dispatches via specialization; an enum forces a cartesian static_assert table that grows quadratically with every axis you add. Keep enums for selectors that are 1-1 with a single LLK call (which binary op fires); use types for behavior that varies across multiple calls.

### 2.2 Each chain element owns its CB lifecycle

The pipeline does not override per-tile wait/pop. Two same-CB elements with conflicting policies is a chain-author bug, not something the pipeline silently merges or compacts. Auto-merge of same-CB elements would make fan-out cost "free" in a way that hides double-waits and makes intent unsearchable. Per-element explicit policies — less code in the helper, more code at the call site, intent lives where it is read.

### 2.3 Independent A/B policies for binary ops

The binary FPU chain element exposes independent `input_a_policy` and `input_b_policy` template params (with `input_b_policy = input_a_policy` as the default). This unlocks "A streams, B persists" patterns (`Mul<ROW, WaitAndPopPerTile, WaitUpfrontNoPop>{}` placed inside `eltwise_chain(...)`) without forcing the chain to grow per-pattern element types.

Note: this section talks about the **binary FPU chain element**, not a separate `binary_op` helper. Per §3.8, binary FPU is a chain element; `binary_op` (if it appears in older prose or shipped code) is the historical standalone-helper shape that should be collapsed into the chain.

### 2.4 Reconfig naming parity across enums

When two helpers each have a "reconfig the input-side SRC" mode, give them the same enum value name even if the underlying LLK semantics differ slightly. `CopyTileReconfig::Input` (always srca) and `DestReuseReconfig::Input` (srca or srcb based on `ReuseType`) read identically at the call site; the per-enum doc-comment explains the difference. Don't use cryptic names like `CbSide` — match the existing house style (`BinaryDataFormatReconfig::INPUT`).

### 2.5 Policy enums are NEVER booleans

Don't ship `Reconfig=true/false` template params. `DestReuseReconfig::None / Input` reads correctly at every call site; the bool form requires the reader to remember which way is on. Same for any future binary toggle.

### 2.6 Default to the safe option, opt into the fast one

`BinaryOutputPolicy::PerTile` is default. `BinaryDataFormatReconfig::INPUT_AND_OUTPUT` (the safest, no-skip mode) is default. Defaults are what gets typed when nobody is paying attention; pick the slow-but-correct one.

### 2.7 CB indexing modes — constrained by the wait shape

Every chain element that reads from a CB (CopyTile, binary FPU A/B operands, `DestReuseOp` CB side, any future CB-input op) must support the four indexing patterns kernels actually use:

| Index mode | Tile read each iteration | Used for |
|---|---|---|
| First-tile (default) | always tile `0` of the CB | streaming consume — wait/pop advances the front, tile `0` is the new tile |
| Block-iter `0..N-1` | tile `i` where `i` is the per-tile loop index | block-at-a-time access after `WaitUpfrontPopAtEnd`, or any pre-waited region walked in order |
| Pinned `k` | fixed compile-time/runtime index, same every iteration | scalars, broadcast-once operands, mask tiles, persistent constants |
| Absolute | caller-supplied runtime index (arbitrary tile inside the waited window) | indexed reads driven by reduction/coordinate state, gather-style patterns |

Indexing is **not** orthogonal to wait/pop — the wait shape determines which tiles are guaranteed present. The cartesian product:

| Wait/Pop policy ↓  /  Index → | FirstTile | BlockIter `0..N-1` | Pinned `k` | Absolute `idx` |
|---|---|---|---|---|
| `WaitAndPop` (wait 1, pop 1)              | ✓                  | ✗ only tile 0 waited       | ✓ iff `k == 0`                        | ✗ no stable window beyond 0 |
| `WaitNoPop` (wait 1, no pop)              | ✓ tile 0 persistent | ✗ only tile 0 waited       | ✓ iff `k == 0`                        | ✗ same reason |
| `NoWaitPop` (no wait, pop 1)              | ✓                  | ✗ pop advances front       | ✓ iff `k == 0`                        | ✗ same reason |
| `NoWaitNoPop` (caller-owned window)       | ✓                  | ✓ caller waited block      | ✓ if `k` ∈ caller's window            | ✓ if `idx` ∈ caller's window |
| `WaitUpfrontPopAtEnd(N)` (wait N, pop N)  | ✓                  | ✓ canonical use            | ✓ iff `k < N`                         | ✓ iff `idx < N` |

Two observations from the matrix:

1. The four "single-tile-window" policies (`WaitAndPop`, `WaitNoPop`, `NoWaitPop`) admit only `FirstTile` (or `Pinned k=0`, which is the same tile). `BlockIter` and `Absolute` are structurally unsafe under them — pop-per-tile or wait-of-1 invalidate any index ≠ 0.
2. `BlockIter` and `Absolute` only become legal when the wait shape stages a multi-tile window: explicit `WaitUpfrontPopAtEnd(N)`, or `NoWaitNoPop` where the caller staged it externally.

Surface as a small enum (`CbIndexMode::FirstTile / BlockIter / Pinned / Absolute`) on the element, plus a runtime `cb_tile_idx_` field for `Pinned` / `Absolute`. `FirstTile` and `BlockIter` need no field. Mode is per-CB-operand — the binary FPU element lets A and B pick independently (e.g. `A=BlockIter, B=Pinned` for `block * scalar` under `WaitUpfrontPopAtEnd`).

**Do not collapse per-side `AIndex` / `BIndex` to a single `Index`.** A prior "Q4 collapse" reduced `BinaryFpu` / `BlockBinaryFpu` to one shared index mode driving both sides. That broke the canonical asymmetric broadcast walk — A streams the tile range while B is pinned at the scaler/vector tile (softmax phase 2a `sub_bcast`, phase 4 `mul_bcast`; deepseek_grouped_gate `add_bias`; moreh_layer_norm_backward, moreh_softmax_backward family). Agents discovering no helper variant for the asymmetric pattern fall back to raw `*_tiles_bcast` loops and the helper loses the surface. Keep `AIndex` and `BIndex` as independent template parameters with `BIndex = AIndex` defaulted for back-compat. Per-side wait counts derive from per-side index mode (`a_wait_count`, `b_wait_count`) — FirstTile pins one tile, BlockIter streams `BlockSize` tiles. Same-CB dedup (`CbA == CbB`) requires `AIndex == BIndex` — flag asymmetric indices on shared CBs as a compile-time error since the deduped B-side wait would under-wait.

Validation:

- The chain combinator `static_assert`s the illegal cells whenever both axes are compile-time (most common case — index mode is a template param, wait policy is a template param). A `WaitAndPop` element instantiated with `BlockIter` or `Absolute` fails to compile.
- For runtime-supplied indices (`Pinned k`, `Absolute idx`), the `WaitUpfrontPopAtEnd(N)` path runtime-`ASSERT`s `idx < N`; the single-tile-window policies runtime-`ASSERT` the index is `0` (per §4.1).
- `NoWaitNoPop` is the only escape hatch — caller asserts the window externally; helper trusts it - this can be unsafe, and shouldn't be default.

### 2.8 Unary broadcast is a chain element, not a separate helper

Unary bcast (one-tile-into-many, scalar-into-tile, row/col bcast on a single input) is a **chain element type** that participates in `eltwise_chain` the same way FPU `binary_op` does. It is NOT a separate `unary_bcast_op` helper, and NOT a flag on `CopyTile`.

Surface: a chain element parameterized on `BroadcastDim::{NONE, ROW, COL, SCALAR}` (the same enum already in use by `binary_op`). Caller passes `BroadcastDim` explicitly — per §9, no inference. The element plugs into `eltwise_chain(...)` like any other chain participant; chain traits (`chain_has_any_copy_tile_v`, fan-out, reuse) extend to cover it via the existing trait machinery rather than special-case branches.

---

## 3. Composition

### 3.1 One dispatch path: the chain

There is no "post-op slot," "fusion hook," or "callback" extension point — that vocabulary is a remnant of the standalone-`binary_op` shape (§3.8) where binary FPU was the helper and everything else was an addendum. Once binary FPU is a chain element, "post" is just the next chain element; "fusion" is just two chain elements; "callback" is just a chain element. The chain is the dispatch path.

Don't accept "either a lambda OR a chain OR a single op." Single ops wrap in `eltwise_chain(Rsqrt<...>{})`. One mental model for the user, one set of compile-time guarantees, one path through the implementation. Multiple accepted forms each grow their own corner-case rules and validation static_asserts; collapsing to one wins on every axis.

### 3.2 Express FPU-clobbering as a trait, not a special case

`clashes_with_fpu` is a `static constexpr bool` member that defaults `true` for `CopyTileTag`-marked elements (via `is_copy_tile_op_v<T>`) and is opt-in for non-CopyTile elements like `DestReuseOp`. The chain trait `chain_has_non_copy_tile_fpu_clash_v<Chain>` decides whether the pipeline must reinit per tile. Don't name the flag after its consequence (`needs_parent_reinit`) — name it after its cause. Future readers who don't know the consequence still understand the cause.

### 3.3 Chain traits derive from chain shape

Traits computed once on the chain type:

```cpp
chain_has_any_copy_tile_v<Chain>
chain_has_non_copy_tile_fpu_clash_v<Chain>
chain_loads_share_cb_v<Chain>
chain_has_duplicate_upfront_cbs_v<Chain>
chain_is_hoist_safe_v<Chain> = has_load && !has_non_load_fpu_clash && loads_share_cb
```

The pipeline picks fast / slow / illegal paths from these traits. The chain author writes simple constructors; the optimization decisions are deduced.

### 3.4 Init hoisting — gate on what each element's init reprograms, not on chain length

Per-tile init is the default. Hoisting `init()` out of the loop is a perf optimization with a tight precondition set — get any condition wrong and the chain silently produces wrong output. The default path runs every element's `init()` + `exec()` every tile; readers can answer "what runs per tile" by reading the chain.

#### Empirical baseline — what production kernels actually hoist

Investigation of `pack_patterns.tsv` (667 entries, full L1→L1 traces around `pack_tile` sites) confirms that production kernels hoist substantially more aggressively than the "exactly `CopyTile + 1 SFPU op`" rule a prior draft proposed. Concrete patterns observed:

- **Multi-SFPU after a single CopyTile**: `softmax.cpp` softmax `exp_cb` hoists `copy_tile_init(cb_in)` AND `exp_tile_init<EXP_APPROX>()` outside the per-tile loop; the loop body runs only `copy_tile` + `exp_tile` + `pack_tile`. No per-tile reinit, even though the chain has two `*_init`s.
- **Multi-SFPU after a single FPU bcast**: `softmax_large_tensor.cpp` `exp_cb` (NUMERIC_STABLE path) hoists `sub_bcast_cols_init_short(cb_in, cb_max)` + `exp_tile_init` together; loop runs `sub_tiles_bcast_cols` + `exp_tile`. Both inits hoisted.
- **CopyTile + SFPU pipeline blocks of `ndst` tiles**: `softmax.cpp` non-stable path hoists `copy_tile_to_dst_init_short(cb_in0)` + `exp_tile_init`; the inner DEST-batch loop runs `copy_tile(...)` then `exp_tile(...)` per slot, with no reinit between them.
- **Same-CB CopyTile fan-out**: kernel_lib `fanout.cpp` and `multi_chain.cpp` tests use two `CopyTile<cb_in, …>{}` chain elements sharing the same CB; production code (`pack_tiles_to_output` in `compute_utils.hpp`) hoists `copy_tile_init` once and runs N copies per loop.

#### Patterns that DO require per-iter reinit (and why)

- **Different-CB CopyTile alternation inside one acquire** — `running_statistics_sfpu_kernel.cpp` switches srca between `cb_tmp3` and `cb_tmp2` mid-window and emits `copy_tile_to_dst_init_short_with_dt(last_srca_cb, new_cb)` between every copy. `copy_tile_init` reprograms srca format; if the second CB's tile is read with srca configured for the first CB, the unpacker gets the wrong format.
- **CopyTile mixed with an FPU-class compute element** (`BinaryFpu`, `DestReuseBinary`, `UnaryBcast` in the helper's vocabulary) — FPU op's `*_init_short` reprograms the unpack MOP that `copy_tile` then reads. Layernorm's `mul_bcast → SFPU activation` keeps SFPU init per-iter only because the legacy macro-injection pattern forces it; the FPU/bcast init itself is still hoisted, and SFPU init *would* be safe to hoist if it weren't macro-injected.

#### What the helper's hoist gate actually needs

The real precondition is **"no element's init reprograms hardware state another element's body or hoisted init relies on."** For the chain helper's element vocabulary that resolves to two conjunctions:

1. **No element with `clashes_with_fpu == true` outside the `CopyTile` family** — i.e. no `BinaryFpu`, `DestReuseBinary`, `UnaryBcast`. Those reprogram unpack MOP each iter.
2. **All `CopyTile` (and `BlockCopyTile`) elements share a single srca CB** — or the chain has at most one CB-reader. Multi-CB CopyTile chains require per-iter reinit because each element's hoisted `copy_tile_init` is overwritten by the next one's, leaving srca configured only for the last element seen at boot.

SFPU op inits (`Exp`, `Sqrt`, `Relu`, `Tanh`, …) program SFPU / SFPI state, not unpack MOP, and do not clobber each other or the upstream FPU/CopyTile init. They are always hoist-safe **as a group**; the chain-shape gate covers the elements that *do* touch unpack.

Pack reconfig is fold-driven and emits at most one `pack_reconfig_data_format` per pack-CB transition; it does not participate in the input-side hoist gate.

#### Concrete gate

```
chain_is_hoist_safe_v<Chain> :=
    !chain_has_non_copy_tile_fpu_clash_v<Chain>   // no BinaryFpu / DestReuse / UnaryBcast
 && chain_copy_tiles_share_cb_v<Chain>            // all CopyTiles share one CB (or ≤1 CopyTile)
```

If the gate passes, the chain combinator emits every element's `init()` once at chain entry; the per-tile loop runs only the lifecycle and `exec()`s. If the gate fails, every element's `init()` runs per tile.

Examples under this gate:

- `CopyTile + Exp` — hoist-safe.
- `CopyTile + Exp + Sqrt` — hoist-safe (multi-SFPU is fine; production already does this).
- `CopyTile<cbA> + CopyTile<cbA> + BinarySfpu`  — hoist-safe (single CB).
- `CopyTile<cbA> + CopyTile<cbB> + BinarySfpu` — **not** hoist-safe (different CBs), per-iter.
- `CopyTile + DestReuseBinary` — **not** hoist-safe (FPU clash), per-iter.
- `BinaryFpu + Exp` — **not** hoist-safe (FPU clash), per-iter.

Wrong output from an over-eager hoist is not a perf optimization. When the gate is unclear (new clobber-capable element added) — default to per-iter init.

Hoist-safety generalizes. Both gate conditions are uniform — predicate on the full element pack — so they hold for any chain length. Express both as recursive walks over the element list, not fixed-size specialisations.

### 3.5 Fan-out is N CopyTiles, one per DEST slot

The hardware has no instruction to copy one CB tile into multiple DEST slots in a single op — `copy_tile` writes one source tile to one destination slot. A `CopyTile` chain element therefore stays one-to-one (one CB tile → one DEST slot), and fan-out (the same CB tile reused in multiple DEST slots) is written as N `CopyTile` elements, each with its own DEST slot and its own wait/pop policy. The chain combinator must NOT hide this as "one CopyTile expanded to N." That kind of compaction makes fan-out cost look free and hides double-wait / extra-pop bugs that only surface at specific CB capacities. The chain user writes the actual N copies, with explicit wait once + skip-wait-but-pop semantics:

```cpp
eltwise_chain(
    CopyTile<cb_input, Dst::D0, CopyTilePolicy::WaitNoPop>{},
    CopyTile<cb_input, Dst::D1, CopyTilePolicy::NoWaitPop>{},
    Op{}, ...);
```

Naming: call the element `CopyTile`, not `Load`. The element wraps the `copy_tile` LLK directly — the name should match. "Load" is generic and obscures the one-tile-one-slot reality.

### 3.6 Same-CB dedup lives in the helper

When the user passes the same CB for both operands of a binary op (e.g. `square` is just `mul(cb_x, cb_x, ...)`), the helper must dedup `cb_wait_front` / `cb_pop_front` so the same CB is waited and popped exactly once per iteration. Don't ship a separate `square` alias to "avoid repeating the same CB" — that's a legacy ergonomic patch. Just pass the same CB twice; the helper detects `icb_a == icb_b` and dedups. Correct same-CB semantics is the helper's responsibility — user-side dedup is fragile and produces double-traffic bugs.

### 3.7 In-DEST hold loops break the eltwise acquire/release pattern

The eltwise chain pipeline wraps a full `tile_regs_acquire / commit / wait / release` inside one helper invocation. Some kernels keep DEST values alive across multiple acquire/release boundaries — e.g. a reduction accumulates partial sums in DEST tile `k`, an outer loop runs an FPU op, then reads slot `k` back. The release at end-of-invocation discards the accumulator, so the eltwise helper as it stands cannot model this lifecycle.

Investigate before refusing: confirm the kernel genuinely holds DEST (vs. re-loading from a CB each iteration). If it does, leave that block on raw LLK and flag the missing policy (split-acquire, `DestRetainPolicy`, or equivalent — caller takes the final release). Hoisting the outer loop into the helper just relocates the problem. Mark blocked kernels in the gap map until the policy lands.

### 3.8 One eltwise helper, specialised convenience entry points — not parallel helpers

Eltwise has **one** helper surface (`eltwise_chain` + the chain element type system). Frequently-used patterns ship as **specialised convenience entry points** that wrap the chain — not as parallel standalone helpers (`binary_op`, `unary_bcast_op`, `dest_reuse_op`, etc. living as peer top-level APIs).

Pattern:

```cpp
// Underlying primitive (always available, fully expressive):
eltwise_chain(CopyTile<cb_a>{}, CopyTile<cb_b>{}, Add{}, ...);

// Convenience wrapper for the common case — thin, expands to the chain:
binary_add(cb_a, cb_b, cb_out);              // == eltwise_chain(...)
unary_bcast_mul(cb_in, BroadcastDim::ROW);   // == eltwise_chain(...)
dest_reuse_mul(cb_in, dst_slot);             // == eltwise_chain(...)
```

Why:

- Shared trait machinery (FPU clash, hoist safety, fan-out, CB lifecycle, reconfig deduction) lives once on the chain. Parallel helpers each grow their own copy and drift.
- Fast call sites stay one-liners (the convenience wrapper is the same ergonomics as a standalone helper would be).
- The "drop down to chain when convenience doesn't fit" path is trivial — convenience and chain are the same machinery.
- Helper-design rules (Gate 1 / Gate 2, validation suite, gap map) apply to one surface, not N.

Convenience entry points:

- Are picked by frequency-of-use, not "every binary op gets one." A one-line wrapper for a pattern used in three kernels is noise; one for the pattern used in fifty kernels saves real call-site code.
- Are pure inline forwarders to `eltwise_chain` — no logic of their own, no parallel policy enums, no separate validation kernels (the chain validation already covers them).
- Live in the same `eltwise_chain.{hpp,inl}` (or a sibling `eltwise_convenience.hpp` aggregator), not in standalone files named after the pattern.

**Binary FPU is a chain element, not a standalone helper.** The same rule that disqualified `unary_bcast_op` (§2.8) applies to `add_tiles` / `sub_tiles` / `mul_tiles` / every binary FPU op: each one is a chain element type that participates in `eltwise_chain` next to `CopyTile`, `DestReuseOp`, and the SFPU op structs. The only thing that ships *outside* `eltwise_chain` is the one-line convenience wrapper (`binary_add(...)`) — and that wrapper is a forwarder, not a separate helper API.

Anti-pattern: shipping `binary_op` as a top-level helper with its own policy enums, its own reconfig surface, its own validation suite, parallel to `eltwise_chain`. That is the historical shape and it is wrong even though it shipped — it splits the trait machinery, splits the test surface, and forces every kernel that mixes binary FPU + SFPU + DEST reuse to know which helper owns which CB. Same critique applies to a hypothetical `unary_bcast_op`, `dest_reuse_op`, or `ternary_op` peer. There is one eltwise helper. Everything else is a chain element or a thin convenience wrapper.

### 3.9 Block-mode is an axis on existing elements, not a parallel sibling kind

When demand surfaces for a multi-tile-DEST variant of an existing element (multiple inner ops per acquire/release window), the temptation is to ship it as a parallel sibling type. Resist. If the new shape shares lifecycle, dispatch order, and init contract with the existing element and only differs in iteration count, the missing piece is a template parameter on the existing element, not a new kind. Parallel sibling types duplicate scaffolding and force every chain trait to be redefined to cover them.

The test that distinguishes "missing parameter" from "missing kind": if the proposed sibling shares the lifecycle, dispatch hooks, and init contract with its non-block counterpart and only differs in iteration count, it's a parameter. If it has a structurally different lifecycle (different wait/pop shape, different reconfig timing) it's a kind.

### 3.10 Reconfig attaches to the element that owns the CB

Reconfig is a per-element capability. Every CB-touching element (CB reader, pack writer) carries a small reconfig policy that selects which side fires (input, output, both, none). The element already knows the CB id and which side it consumes; the policy slots in cleanly without threading previous-CB state across template parameters and without making the caller insert a separate reconfig element at the right spot.

Anti-shapes that fail:

- Reconfig as a wall of `OldCb`-style template parameters threaded across every op element. Caller threads previous-CB state by hand; omissions are silent; the parameter list explodes with reconfig bookkeeping unrelated to the op shape.
- Reconfig as standalone chain elements the caller orders manually. Sequencing is fragile; convenience wrappers have to inject them at every call site; forgetting one ships silent wrong dtype.
- Reconfig as a chain-wide enum the pipeline alone controls. No override path when a specific kernel needs non-standard reconfig timing.

Right shape: attach a small reconfig enum to whichever element owns the CB, defaulted to a safe value. The element wires it into its own init. Convenience wrappers set the policy; advanced callers override it on the specific element that needs the override. The decision lives next to the LLK call that consumes it.

---

## 4. Internal state hygiene

### 4.1 Pipeline state is private and reset by the pipeline

`CopyTile::cb_tile_idx_` is `mutable uint32_t` and `private`. Only `WaitUpfrontPopAtEnd` reads/writes it, and the pipeline zeroes it at end-of-block. Callers cannot override it. Internal state externally settable becomes a footgun the moment a second consumer exists.

### 4.2 Don't ship fast paths that don't carry their weight

A fast path (e.g. "skip `init()` when the chain has only one compute op", "batched apply", "exec-only mode") is justified only when a real workload's profile shows it matters. Without that evidence, the fast path's cost is the obscured per-tile semantics: readers can no longer answer "what runs per tile" by reading the code. Default to uniform "init + exec each element, every tile" and only branch when measurement requires it.

### 4.3 Compile-time validation over runtime checks

```cpp
static_assert(static_cast<uint32_t>(Slot) < DEST_AUTO_LIMIT, "DEST slot exceeds compile-time DEST capacity");
static_assert(static_cast<uint32_t>(DataSlot) + 1 < DEST_AUTO_LIMIT, "Mask requires DataSlot + 1 < DEST capacity...");
chain_has_duplicate_upfront_cbs_v<Chain> // static_assert in pipeline
```

Catch bad chains at compile time. Runtime asserts are the fallback for things the type system genuinely can't see (e.g. `WaitAndPop` policy + non-zero `cb_tile_idx` → ASSERT, since streaming pop-per-call can't index into a batch).

### 4.4 Pipeline lifecycle: every path emits every per-tile call, policy-guarded

The chain combinator has multiple traversal paths (per-tile path, block_path for `is_upfront` elements, future variants). Each path must emit the full set of per-tile lifecycle helpers — `elem_wait_per_tile`, `elem_reserve_per_tile`, `elem_pop_per_tile`, `elem_push_per_tile` — even when the dominant elements in that path don't trigger them. The per-tile helpers are policy-guarded internally (`if constexpr (Policy == PerTile...)` — no-op for upfront policies). Skipping them in a non-default path is a lifecycle gap waiting to be hit by mixed-policy chains.

Real bug: `block_path` historically only emitted `elem_wait_upfront` / `elem_reserve_upfront` / `elem_pop_upfront_end` / `elem_push_at_end`. A chain mixing `CopyTile<WaitUpfrontPopAtEnd>` (triggers `block_path = true`) with `PackTile<PerTileReserveAndPush>` (streaming output) lost its pack reserve/push entirely. `pack_tile` wrote to unreserved slots, no `cb_push_back` fired, downstream consumer hung at `cb_wait_front`. The "block_path is for fully-upfront chains" assumption is wrong — the canonical softmax phase 2b pattern (upfront input + streaming output) needs both per-tile and upfront calls in the same loop.

Order matters: emit pack-output reserve as late as possible (right before `pack_exec`) and pack-output push as early as possible (right after `pack_exec`) so the downstream consumer sees pushed tiles before this iteration releases DEST. Wait/pop on the input side stays at loop top/bottom — they pair with the data dependency, not the pack window. The same ordering applies to per-tile path and block_path.

### 4.5 One dispatch signature for every element

The classification surface (tag inheritance, trait constants) and the dispatch surface (the methods the chain pipeline calls per phase) are different contracts. If they drift — for example, classification places an element in a category whose pipeline path uses a static call, but the element actually carries runtime state via an instance method — the wrong path runs silently. The compiler can't catch it because both methods exist. Pick one dispatch shape (a single member method that takes the loop index) for every element. Stateless ops cost nothing — the compiler folds zero-state members into static dispatch. Stateful ops capture their fields naturally. SFINAE detectors that pick between dispatch shapes are a band-aid for letting two contracts coexist; one contract is the fix.

---

## 5. Header layout

### 5.1 One file per op category, one aggregator header

Original monolithic `eltwise_helpers.{hpp,inl}` (~2600 lines, 1700 of header) was split into:

```
eltwise_chain.{hpp,inl}        # core: Dst, policies, CRTP bases, CopyTile, EltwiseChain, eltwise_pipeline,
                               # all chain element types (binary FPU, unary bcast, dest reuse, fill, rand)
eltwise_activations / math / misc / predicates / rounding / scalar / special / trig
eltwise_helpers.hpp            # aggregator; backward-compat include
```

Each file pulls in only the LLK headers it needs. Build time and "where does this op live" both improve. Keep the aggregator so existing call sites don't break.

Note: the older split also carried separate `binary` / `ternary` files for `binary_op` / `ternary_op` standalone helpers. Per §3.8, binary FPU and ternary FPU are chain element types — they live in `eltwise_chain.{hpp,inl}` next to `CopyTile`, not in their own files.

### 5.2 Examples in the header doc-comment, not a separate guide

`eltwise_chain.hpp`'s opening doc-comment carries the worked examples — basic `Add`, broadcast variants, `WaitUpfrontNoPop` for softmax, `NoWaitNoPop` for sharded, independent A/B policies, skip-reconfig, multi-element chains (binary FPU + SFPU + DEST reuse). Examples in the file the user already opened beat external docs that decay.

(Historical note: examples used to live in `binary_op_helpers.hpp`. After the §3.8 collapse they move into `eltwise_chain.hpp` alongside the chain element types.)

### 5.3 BroadcastDim and Reduce↔Broadcast tables in the header

`BroadcastDim::{NONE,ROW,COL,SCALAR}` + companion table mapping `REDUCE_ROW → COL`, `REDUCE_COL → ROW`, etc. The "reduce-row produces column-shaped output" surprise lives where it is needed. Don't bury it in a separate doc.

### 5.4 Per-helper blockers in per-helper doc-comments

Helper-specific blockers (e.g. "DestReuseOp's `WaitAndPop` requires `cb_tile_idx == 0`", "Auto+PerTile deadlock at small CB capacity") live in the helper's `.hpp` doc-comments. The general migration HQ doc only carries the helper-agnostic blockers. Caller reads the helper header before doing the audit.

---

## 6. Naming evolution is part of the design

These all happened mid-branch and were worth doing:

```
CopyTileReconfig::Srca         → CopyTileReconfig::Input          (parity with DestReuseReconfig)
DestReuseReconfig::CbSide  → DestReuseReconfig::Input     ("CbSide" too cryptic)
needs_parent_reinit        → clashes_with_fpu              (cause, not consequence)
needs_parent_reinit (chain)→ chain_has_non_copy_tile_fpu_clash_v
EltwiseChain::num_compute_ops → (deleted)
EltwiseChain::apply_batched   → (deleted)
EltwiseBatching enum          → (deleted entirely)
```

Rename when the new name reveals what the thing IS. Don't keep cryptic names "for stability." Keep aliases for one release if external consumers exist.

---

## 7. Validation suite — non-negotiable

Every helper change (new op, new policy, new reconfig path, new enum value) must land alongside a pytest that runs a custom compute kernel on device against a torch golden. Build-only is not enough — kernels JIT.

The eltwise validation suite at `ttnn/cpp/ttnn/kernel_lib/tests/chain_and_binary/` covers:

1. `chain_fanout_x_times_exp_x` — fan-out CB read with `WaitNoPop + NoWaitPop`, single-compute-op chain.
2. `binary_same_cb_add` — same-CB wait/pop dedup. A broken dedup deadlocks or skips tiles.
3. `binary_dest_reuse_mul` — `DestReuseOp` as chain element, FPU-clash reinit.
4. `chain_binary_then_copy_tile_reinit` — `CopyTile` chained after a binary FPU element (multi-element chain, FPU-clash reinit).

Each parameterized over `num_tiles ∈ {1, 8, 64}` (single tile, fits in DEST, multi-DEST window). Tolerance accommodates compounded bf16 ULPs (rtol≈5e-2, atol≈1e-1 for products on `[-3, 3]`); a too-tight tolerance flags legit rounding.

Add a sister "raw-LLK" diagnostic kernel for any new compute path. If both helper and raw hang, the bug is in the LLK or the test scaffolding (reader push counts, CB capacity, persistent-tile lifecycle). If only the helper hangs, the helper is wrong. The diagnostic kernel saves hours.

The dtype-matrix and untestable-locally rules are general — see [llk_helpers_hq.md → Step 4](llk_helpers_hq.md#step-4--verify-on-device).

### 7.1 srcB reconfig path is its own dimension (eltwise-specific)

Standard FPU binary (`add_tiles`, `sub_tiles`, `mul_tiles`) consumes srcA *and* srcB; either operand can drive a format reconfig on entry. `DestReuseOp` adds the `DEST_TO_SRCB` path (DEST → srcB, CB → srcA). In every case srcA-reconfig and srcB-reconfig are **separate LLK paths** even when they share a single helper enum value (e.g. `BinaryDataFormatReconfig::INPUT_AND_OUTPUT`, `DestReuseReconfig::Input`, see §2.4). A migration that only validates srcA-reconfig has not validated the srcB branch.

Test matrix must cover:

- binary FPU element srcA-reconfig: A operand changes dtype, B unchanged.
- binary FPU element srcB-reconfig: B operand changes dtype, A unchanged.
- binary FPU element both reconfigured: A and B both change dtype between calls.
- `DestReuseOp` srcA-reconfig (`CB_TO_SRCA`-style ReuseType): CB operand changes dtype.
- `DestReuseOp` srcB-reconfig (`DEST_TO_SRCB`): DEST → srcB path with CB-side dtype change.

Add a dedicated test variant per srcB path the helper supports, or document explicitly that srcB-reconfig is untested and untouched in the migration. Don't assume "srcA worked, so srcB works" — they hit different unpacker MOPs.

The combined LLK call is literally the sum of the two single-side calls — no fused fast path under the hood:

```
reconfig_data_format(a, b)  ≡  reconfig_data_format_srca(a) + reconfig_data_format_srcb(b)
reconfig_data_format(a_old, a_new, b_old, b_new)
                            ≡  reconfig_data_format_srca(a_old, a_new)
                             + reconfig_data_format_srcb(b_old, b_new)
```

Source: `tt_metal/hw/inc/api/compute/reconfig_data_format.h` — the combined overloads expand to the same `llk_unpack_reconfig_data_format` + `llk_math_reconfig_data_format` pair that the per-side overloads issue independently. Two consequences:

- **Helper-side**: a binary op that only changes one operand dtype must call the single-side variant, not the combined one with the unchanged side. Combined call is convenience, not optimization — it does the same two MOPs you'd issue by hand.
- **Test-side**: this confirms §7.1's "two genuinely separate paths" — covering only srcA exercises only half the MOP issue path. The combined-call variant is not a third path to test, just the union of the two.

---

## 8. Eltwise-specific migration enablers

The pattern is fixed: real kernel can't migrate → identify the missing primitive (op struct, policy, reconfig path, index mode) → small CRTP/policy addition → kernel migrates. Not features added speculatively. Don't enumerate the historical fixes here — they rot the moment the helper grows or kernels rename.

Track open work in a `feature_gap_map` keyed by `GAP-N`. Each gap entry pins:

- The missing primitive (policy / reconfig path / index mode).
- The list of currently blocked kernels.
- Fix complexity (LOC, new template params, new test variants).
- Whether the gap is real (per §10.3) or fix-and-continue (per §10.2 — missing op struct alone is not a gap).

Close gaps by yield: kernels-unblocked-per-LOC. The gap map is the only roadmap; closed entries get deleted, not archived in the doc.

Free-text skip rationale drifts. The same blocker gets phrased three different ways across cycles, defeats aggregation, and the gap map can't auto-derive from the log. One structured row per kernel — migrated stages, skipped stages, named blocker id per skip — closes the loop. The gap map becomes a query over the log, not a hand-curated parallel document.

---

## 9. Things to avoid in the eltwise helper

- **Auto fast paths.** Single-op fast paths in chains, auto-batching, auto-merge of same-CB CopyTiles. Make optimizations explicit, never silent.
- **Default init hoisting.** Hoisting is opt-in and gated on the strict precondition set in §3.4 (chain = `CopyTile + 1 SFPU op`, single CB input, no clobbering element). Never the default — wrong output is not a perf win.
- **Lambdas / functors / "PostOp" extension points.** None of these exist as separate concepts. The chain is the dispatch path; "post" is just the next chain element (§3.1). Single ops wrap in `eltwise_chain`.
- **Hidden broadcast inference.** `BroadcastDim` is always passed explicitly by the caller — the helper does NOT default-pick based on operand shape. Every value of the enum (e.g. `NONE` for no broadcast, `ROW` / `COL` for 1-D broadcasts, `SCALAR` for a 1×1 tile, plus any others the helper supports) is a distinct dispatch path the caller chooses; "infer it from `Ht`/`Wt`" is forbidden.
- **Mid-loop dtype swaps without policy support.** Helpers do one entry-time reconfig. If the kernel switches dtypes mid-loop, that path stays raw or grows a mid-chain reinit policy.
- **Skipping `fp32_dest_acc_en=True` testing.** Every binary migration runs against `fp32_dest_acc_en ∈ {False, True}` and any mixed-dtype combo the original supported.
- **Hand-coding around a missing op struct in a kernel.** Add the op struct to the helper, rebuild, then migrate (see §10.2 — missing op struct is fix-first, not a blocker). The kernel never gets a workaround copy of the LLK call.
- **Two dispatch contracts.** Classification by tag and dispatch by method signature drift unless held to one shape. SFINAE detectors that pick between dispatch shapes are a band-aid for letting two contracts coexist; one contract is the fix.
- **Stub defaults that silently pass downstream checks.** A stub return value (id 0, flag false) absorbed by a downstream early-out hides missing overrides. Force the contract at the override site via a static_assert that names the missing member.
- **Reconfig divorced from its element.** Threading reconfig through the call site (template params on every op, or standalone elements the caller orders) is fragile. Pipeline-deduced reconfig with no override path is also wrong. Attach it to the element that owns the CB.
- **Parallel sibling kinds for an axis on an existing element.** When the proposed sibling shares lifecycle, dispatch order, and init contract and only differs in iteration count, the missing piece is a template parameter, not a new kind.

Migration-pipeline rules (test-change approvals, partial-migration log format, untestable-locally handoff, HQ doc audit, Phase-2 handoff) live in [llk_helpers_hq.md → Pipeline Self-Maintenance](llk_helpers_hq.md#pipeline-self-maintenance). General helper-design rules (helper owns CB lifecycle, DEST capacity is compile-time) live in [llk_helpers_hq.md → Helper Design Principles](llk_helpers_hq.md#helper-design-principles-general).

---

## 10. Migration triage — what to skip, what to fix-and-continue

A kernel showing up in the survey is not automatically a migration target. Triage before touching it:

### 10.1 Skip macro-injection kernels

Kernels that build their compute path via `#define`-macro injection — single source file compiled once per op via per-op compile flags (`#define ELTWISE_OP add_tiles`, `#define SFPU_INIT exp_tile_init`, etc.) — are out of scope for eltwise-helper migration. The helper produces a typed chain at compile time; macro-injection produces an opaque text substitution at preprocess time. Fitting one inside the other either re-implements the macro dispatch as templates (huge change, often per-op) or strips the dispatch entirely (breaks every consumer flag). Neither belongs in a per-kernel migration PR.

Mark macro-injection kernels `skipped:macro-injection` in the survey and move on. Revisit only as a separate workstream that converts the dispatch surface itself, not as part of the eltwise migration sweep.

### 10.2 Missing op struct ≠ blocker

If the only thing standing between a kernel and the helper is a missing op struct (the LLK call exists, the SFPU/FPU primitive exists, no new policy is needed — just no `Foo {}` in the helper headers yet), that is **not** a blocker. The fix is:

1. Add the op struct to the appropriate `eltwise_*` header (4 lines via the CRTP base, per §1.1).
2. Rebuild.
3. Continue the migration in the same PR.

Real blockers are missing **policies** (cumulative wait, in-DEST hold, mid-loop dtype reconfig, a wait/pop combination not in §2.1) or missing **primitives** (a hardware op nobody has wrapped yet). A missing op struct is template boilerplate, recorded under §8's enabler table, not in the gap map.

### 10.3 Genuine blockers

- Missing policy (see §2.1, §3.7, §9 mid-loop dtype).
- Missing primitive — LLK call doesn't exist yet.
- Index mode the helper doesn't support (per §2.7) — add the mode rather than skipping, unless the mode itself requires new primitives.
- Macro-injection (per §10.1) — the only "skip permanently" category.

## 11. CB lifecycle ordering — wait late, pop early; reserve late, push early

CB sync primitives are idempotent and cheap. The right place to emit them is the position that **maximizes producer/consumer overlap** and **enables in-place reuse of CB slots**:

- **Wait as late as possible.** The latest valid wait position is the instruction just before the unpacker reads the tile. Earlier waits force the consumer to block when the data isn't strictly needed yet.
- **Pop as early as possible.** The earliest valid pop position is the instruction right after the unpacker stops reading. Earlier pop frees the slot for the producer to refill, enabling in-place writes when consumer and producer share the CB.
- **Reserve as late as possible.** The latest valid reserve position is right before `pack_tile`. Earlier reserves hold the slot longer than needed and block producers writing into the CB upstream.
- **Push as early as possible.** The earliest valid push position is right after `pack_tile` (before `tile_regs_release`). Earlier push lets the downstream consumer wake on the new tile while the current iteration is still tearing DEST down.

The pair `pop early + reserve late` is what enables an **in-place CB**: the consumer pops slot N, freeing it, and the producer (or a later step in the same kernel) reserves slot N for write — same physical slot, no second buffer. The four-rule discipline is the minimal contract for in-place lifecycles.

### Per wait-shape positioning

All three wait shapes are emitted **inside** the per-tile loop. Upfront and cumulative shapes are NOT moved to a pre-loop block — `cb_wait_front` is cumulative-count idempotent so calling it in-loop is correct, and emitting late ensures no premature blocking on tiles a later loop iteration may not even reach if an outer condition exits early.

| Wait shape | Where to emit | Argument shape |
|---|---|---|
| Per-tile | inside the loop, just before the unpacker uses the tile | `cb_wait_front(cb, 1)` |
| Cumulative | inside the loop, just before the unpacker uses the tile range | `cb_wait_front(cb, i+1)` (or `base + (i+1)*step`) |
| Upfront | inside the loop, just before the unpacker first reads tile 0 — on iter 0 blocks for full N tiles, iter 1+ returns immediately (cumulative-count idempotent) | `cb_wait_front(cb, N)` |

Same logic for pop / reserve / push: emit at the latest-still-valid (or earliest-still-valid for pop/push) position. The wait shape determines what's waited; the *position* in the loop is independent and follows the late/early rules above.

The historical pattern of emitting `cb_wait_front(cb, N)` once **before** the per-tile loop is functionally equivalent for the N-tile case but is strictly less overlap-friendly: a producer running ahead by k < N tiles can't start filling the CB while the consumer is already computing on the k tiles that did arrive, because the consumer is still blocked on tile N. Moving the wait inside the loop (with full count N on every iter) recovers that overlap for free — the wait short-circuits once the producer has caught up, and the producer wasn't blocked from pushing in the meantime.

### Concrete example

Streaming-friendly bcast-Sub (softmax phase 2-style):

```cpp
// boot — caller's responsibility, NOT in the per-tile loop:
binary_op_init_common(cb_input, cb_max_scaler, cb_output);

for (uint32_t i = 0; i < N; ++i) {
    cb_wait_front(cb_input, N);              // wait late (in-loop), upfront count — idempotent
    cb_wait_front(cb_max, 1);                // wait late, single-tile

    tile_regs_acquire();
    sub_bcast_cols_init_short_with_dt(cb_input, cb_max);
    sub_tiles_bcast<BroadcastType::COL>(cb_input, cb_max, /*a_idx=*/i, /*b_idx=*/0, /*dst=*/0);
    exp_tile_init();
    exp_tile(0);
    tile_regs_commit();

    tile_regs_wait();

    cb_reserve_back(cb_exp, 1);              // reserve late — right before pack
    pack_tile(0, cb_exp);
    cb_push_back(cb_exp, 1);                 // push early — right after pack
    tile_regs_release();
}

cb_pop_front(cb_input, N);                   // pop at end (CB lives across phases)
cb_pop_front(cb_max, 1);
```

Three wins over a pre-loop-wait shape:
1. Reader pushing tile `k < N` of `cb_input` is not blocked by the consumer — `cb_wait_front(cb_input, N)` short-circuits on iter 1+, reader pushes proceed in parallel with compute on tiles `[0..k]`.
2. `cb_reserve_back` is the LAST thing before pack, so a producer writing to `cb_exp` from another core/phase has the maximum window to do so.
3. `cb_push_back` runs before `tile_regs_release`, so the downstream consumer (Phase 3 SUM reduce) can start waking on `cb_exp[i]` while DEST is still being released.

### Helper compliance

Every chain element's `wait_per_tile` / `wait_upfront` / `pop_per_tile` / `pop_upfront_end` / `reserve_per_tile` / `reserve_upfront` / `push_per_tile` / `push_at_end` is emitted at the loop position prescribed by these rules. The chain framework calls them at the right loop position; the element gates by its own policy and emits no-op when not applicable.

A helper that emits `cb_wait_front` early (e.g. at chain boot, before the loop) for an upfront-policy CB is correct but suboptimal — leaves overlap on the table. The current chain code does exactly this for `WaitUpfrontPopAtEnd` via the boot-time `elem_wait_upfront` fold. Moving it inside the loop (at the late-as-possible position) is the structural overlap fix that supersedes a separate `Cumulative` policy for most real cases.

### Granularity — "loop iter" is the unit of work, not always a single tile

The wait-late / pop-early / reserve-late / push-early rules apply at whatever **granularity** the chain's loop body operates on. The "tile" in the rules above is shorthand for "one acquire/commit/wait/release cycle's worth of work". Eltwise chains see only two granularities; everything else lives in the outer kernel.

| Chain granularity | One chain-loop iter processes | Example families |
|---|---|---|
| **Tile** | 1 tile in 1 DEST slot | eltwise unary/binary, ternary, copy/transpose, the bulk of the chain inventory |
| **DEST-batch** | N (>1) tiles into N DEST slots, indexed inside one acquire | binary-ng block paths, bcast, CCL reduction inner loops, rotary embedding |

Restated four rules:

- **Tile granularity** — unit = 1 tile. Wait per-tile inside loop; pop per-tile right after unpack stops; reserve per-tile right before pack; push per-tile right after pack. **§11 verbatim.**
- **DEST-batch granularity** — unit = the batch of N tiles. Wait for `cb, N` once **inside** the outer iter (idempotent, so still in-loop-late); pop `cb, N` once after the last unpack of the batch (not after each tile — popping mid-batch shifts indices and corrupts indexed reads); reserve `out, N` once before the first pack of the batch; push `out, N` once after the last pack. **Per-tile waits/pops inside the inner batch loop are illegal.**


### Acquire / commit / wait / release sit AT the granularity boundary

`tile_regs_acquire` and `tile_regs_release` mark the unit-of-work boundary. Everything between them is "this iteration"; wait/reserve/pop/push positions are computed relative to that window:

- **Wait late** = last position before the unpacker (inside acquire window) first reads.
- **Pop early** = first position after the unpacker (inside acquire window) last reads.
- **Reserve late** = last position before pack (after `tile_regs_wait`).
- **Push early** = first position after pack (before `tile_regs_release`).

When the acquire/release window wraps multiple tiles (DEST-batch), "first read" and "last read" are over the WHOLE window — wait sits at the top of the batch iter, pop at the bottom of the batch iter, reserve at start of pack phase, push at end.

### Per-CB policy — how each input CB declares its lifecycle

Granularity sets the loop unit; **per-CB policy** sets each input CB's lifecycle shape WITHIN that unit. Each input CB picks one independently.

The chain already encodes this as `CopyTilePolicy` (misnamed — the same enum is consumed by every CB-reader chain element: `CopyTile`, both sides of `BinaryFpu`, `DestReuseBinary`, `UnaryBcast`. Rename to `CbReaderPolicy` planned).

Wait shape × Pop shape is a true cartesian. Each axis is independent:

| Wait shape | Pop shape | Enum value | Use case |
|---|---|---|---|
| per-tile `cb_wait_front(cb, 1)` | per-tile `cb_pop_front(cb, 1)` | `WaitAndPop` | Streaming producer/consumer in lock-step. The default. |
| per-tile, idempotent | none in this chain | `WaitNoPop` | Pinned single-tile operand (scaler, mask, bcast tile). Caller or a later step pops. Used for fan-out FIRST loader (first reader of a shared CB waits; subsequent readers `NoWaitPop`). |
| none | per-tile | `NoWaitPop` | Fan-out LAST loader (someone already waited; this loader pops). Or pre-waited streaming (caller waited outside, this chain pops). |
| none | none | `NoWaitNoPop` | Caller owns the full lifecycle. CB persistent across multiple chain calls. Sharded tensors. Softmax `cb_input` reused Phase 1 → Phase 2 chain → kernel pops at end of row. |
| upfront `cb_wait_front(cb, N)` | bulk `cb_pop_front(cb, N)` at end | `WaitUpfrontPopAtEnd` | Block-at-a-time access via `BlockIter` index. The wait-late amendment moves the upfront wait INSIDE the loop (idempotent), recovering producer/consumer overlap. End-of-loop bulk pop stays. |

What this enum buys over a flatter "4 semantic policies" collapse:

1. **Fan-out**: `WaitNoPop` + `NoWaitPop` paired on the same CB lets element A wait once, element B pop once. Cleaner than both elements either waiting or popping. Two same-CB elements with conflicting policies is the chain author's bug (the lessons file §2.1 calls this out).
2. **Fan-out-second + streaming**: `NoWaitPop` on a CB that the caller pre-waited but the chain still pops per-tile — a real shape used by sharded ops.
3. **Per-side independent on `BinaryFpu`**: A-side `WaitUpfrontPopAtEnd` + B-side `WaitNoPop` (softmax phase 2). Two independent CBs in one element, each with its own (wait, pop) pair.
4. **Index-mode constraint**: only `WaitUpfrontPopAtEnd` and `NoWaitNoPop` stage a multi-tile window that justifies `BlockIter`/`Absolute` indexing. Other shapes are `FirstTile`-only. The enum captures this constraint directly.

The missing useful cell is **Cumulative**: per-iter `cb_wait_front(cb, i+1)` (or `base + (i+1)*step`) with bulk pop at end. Note that the *naive* "streaming wait + bulk pop" shape — per-iter `cb_wait_front(cb, 1)` without pop — is NOT a new policy: with no pop between iters, the CB front never moves, so `cb_wait_front(cb, 1)` returns the same tile-0-present answer every call and adds zero new blocking. For `BlockIter`-style indexed access at iter i, the wait must grow with i; only the cumulative shape pipelines correctly.

Today this shape is hand-rolled by callers outside the chain. Adding `CumulativeWaitPopAtEnd` (or equivalent) as an enum value covers the gap. The mechanical effect of moving `cb_wait_front(cb, N)` from boot-time into the per-iter loop (the late-wait amendment for `WaitUpfrontPopAtEnd`) is functionally equivalent when N is constant: `cb_wait_front(cb, N)` is also idempotent, blocks once on iter 0 until N tiles present, no-op thereafter. The distinction:

| Shape | Per-iter wait call | Blocks until |
|---|---|---|
| `WaitUpfrontPopAtEnd` (late-emitted) | `cb_wait_front(cb, N)` | full N tiles present (heavyweight block on iter 0) |
| `CumulativeWaitPopAtEnd` | `cb_wait_front(cb, i+1)` | just `i+1` tiles present (per-iter incremental block) |

Both pop bulk at end. Both allow `BlockIter`. The cumulative shape lets the consumer start computing on tile 0 as soon as tile 0 is pushed, regardless of whether tiles 1..N-1 have arrived. The constant-count shape waits for all N before iter 0 can start.

### Where the four rules forbid what looks like a legal optimization

Two traps when applying the rules at Tile granularity to DEST-batch chains:

1. **Per-tile pop inside DEST-batch** — looks like "pop early" but the CB front moves underfoot. Indexed reads of the remaining batch tiles return wrong data. Correct: batch-granular bulk pop after the last unpack.
2. **Per-tile reserve inside DEST-batch output** — looks like "reserve late" but reserving 1 tile when N tiles will pack into N consecutive slots produces partial reservation. Correct: bulk reserve of N before the pack loop.

The trap is the same one in both directions: confusing the chain's INTERNAL DEST batch loop (inner) with the chain's OUTER iter (outer). Lifecycle ops sit at the OUTER iter, never inside the inner DEST loop.

### Helper compliance — granularity- and policy-aware

A chain element's lifecycle methods MUST declare both:
1. Its **granularity** (Tile or DEST-batch) — drives where the chain framework emits the method (per-inner-tile or per-batch-iter).
2. Each CB's `CopyTilePolicy` value (rename pending → `CbReaderPolicy`) — drives whether the method fires at all and with what args.

The chain framework's job is to call each method at the right loop position. The element's job is to gate by policy and emit no-op when not applicable. The four lifecycle rules bind unchanged across all granularity × policy combinations.
