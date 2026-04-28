# Eltwise Helper Design — Lessons Learned

Use this when writing the prompt for **creating eltwise helpers** (binary FPU + SFPU chain + DEST reuse).

The order is the order you should think in: type system → policy surface → composition → optimization → testing → migration enablement.

---

## 1. Op struct shape

### 1.1 CRTP bases for boilerplate elimination

Three bases cover every eltwise SFPU op shape. Derived structs only define `init()` + `call()`:

```cpp
template <typename Derived, Dst Slot>                                 struct UnaryOp;
template <typename Derived, Dst In0, Dst In1, Dst Out>                struct BinaryOp;
template <typename Derived, Dst In0, Dst In1, Dst In2, Dst Out>       struct TernaryOp;
```

The base provides `dst_idx`/`in0`/`in1`/`out`, `max_dst`, `static_assert(<8)`, `exec()`, and `apply()` (which calls `init()` + `exec()`). Derived structs are 4 lines. Don't reintroduce hand-written `apply()`/`exec()` per op.

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

Always cap DEST slot enum at the hardware limit (8 in half-sync fp16 mode) and `static_assert(<8)` in every base.

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

### 2.2 Each chain element owns its CB lifecycle

The pipeline does not override per-tile wait/pop. Two same-CB elements with conflicting policies is a chain-author bug, not something the pipeline silently merges or compacts. Auto-merge of same-CB elements would make fan-out cost "free" in a way that hides double-waits and makes intent unsearchable. Per-element explicit policies — less code in the helper, more code at the call site, intent lives where it is read.

### 2.3 Independent A/B policies for binary ops

`binary_op` has independent `input_a_policy` and `input_b_policy` template params (with `input_b_policy = input_a_policy` as the default). This unlocks "A streams, B persists" patterns (`mul<ROW, WaitAndPopPerTile, WaitUpfrontNoPop>`) without forcing the helper to grow per-pattern entry points.

### 2.4 Reconfig naming parity across enums

When two helpers each have a "reconfig the input-side SRC" mode, give them the same enum value name even if the underlying LLK semantics differ slightly. `CopyTileReconfig::Input` (always srca) and `DestReuseReconfig::Input` (srca or srcb based on `ReuseType`) read identically at the call site; the per-enum doc-comment explains the difference. Don't use cryptic names like `CbSide` — match the existing house style (`BinaryDataFormatReconfig::INPUT`).

### 2.5 Policy enums are NEVER booleans

Don't ship `Reconfig=true/false` template params. `DestReuseReconfig::None / Input` reads correctly at every call site; the bool form requires the reader to remember which way is on. Same for any future binary toggle.

### 2.6 Default to the safe option, opt into the fast one

`BinaryOutputPolicy::PerTile` is default. `BinaryDataFormatReconfig::INPUT_AND_OUTPUT` (the safest, no-skip mode) is default. Defaults are what gets typed when nobody is paying attention; pick the slow-but-correct one.

---

## 3. Composition

### 3.1 One dispatch path, one extension point

When the helper exposes an extension point (post-op slot, fusion hook, callback), give it a single canonical type. Don't accept "either a lambda OR a chain OR a single op." Pick one — the chain — and require single-op uses to wrap (`eltwise_chain(Rsqrt<...>{})`). One mental model for the user, one set of compile-time guarantees, one path through the implementation. Multiple accepted forms each grow their own corner-case rules and validation static_asserts; collapsing to one wins on every axis.

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

### 3.4 Hoist init based on hardware state each element touches

Each chain element's `init()` programs specific hardware state — unpack MOP, math MOP, ADDR_MOD, srca/srcb format reconfig, packer reconfig, etc. Hoisting an `init()` out of the per-tile loop is safe only when no other element in the chain runs an `init()` (or `exec()`) that clobbers that same state between iterations. The decision is per-piece-of-hardware-state, not per-chain-shape:

- A `CopyTile`'s `init()` programs `copy_tile_to_dst_init_short` (unpack MOP) + optional srca reconfig. Hoist it iff no later element reprograms unpack MOP or srca format.
- An FPU dest-reuse element's `init()` programs `binary_dest_reuse_tiles_init`, which clobbers unpack MOP. A chain mixing it with a `CopyTile` cannot hoist the `CopyTile` init across iterations — the dest-reuse step in iteration `i` invalidates it for iteration `i+1`.
- A purely-SFPU element's `init()` programs SFPU MOPs only and does not touch unpack/math FPU state, so it neither blocks hoisting nor needs hoisting itself.

Encode each element's "what hardware state I touch in init/exec" as a static-constexpr trait (`clashes_with_fpu`, plus finer-grained traits if needed: `clobbers_unpack_mop`, `reconfigs_srca`, etc.). The chain combinator decides hoisting by looking at which traits the elements expose, not by inferring from the chain's shape. Wrong output from an over-eager hoist is not a perf optimization — when in doubt, init per tile.

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

---

## 4. Internal state hygiene

### 4.1 Pipeline state is private and reset by the pipeline

`CopyTile::cb_tile_idx_` is `mutable uint32_t` and `private`. Only `WaitUpfrontPopAtEnd` reads/writes it, and the pipeline zeroes it at end-of-block. Callers cannot override it. Internal state externally settable becomes a footgun the moment a second consumer exists.

### 4.2 Don't ship fast paths that don't carry their weight

A fast path (e.g. "skip `init()` when the chain has only one compute op", "batched apply", "exec-only mode") is justified only when a real workload's profile shows it matters. Without that evidence, the fast path's cost is the obscured per-tile semantics: readers can no longer answer "what runs per tile" by reading the code. Default to uniform "init + exec each element, every tile" and only branch when measurement requires it.

### 4.3 Compile-time validation over runtime checks

```cpp
static_assert(static_cast<uint32_t>(Slot) < 8, "DEST slot exceeds maximum capacity (8)");
static_assert(...DataSlot < 7..., "Mask requires DataSlot + 1 < 8...");
chain_has_duplicate_upfront_cbs_v<Chain> // static_assert in pipeline
```

Catch bad chains at compile time. Runtime asserts are the fallback for things the type system genuinely can't see (e.g. `WaitAndPop` policy + non-zero `cb_tile_idx` → ASSERT, since streaming pop-per-call can't index into a batch).

---

## 5. Header layout

### 5.1 One file per op category, one aggregator header

Original monolithic `eltwise_helpers.{hpp,inl}` (~2600 lines, 1700 of header) was split into:

```
eltwise_chain.{hpp,inl}        # core: Dst, policies, CRTP bases, CopyTile, EltwiseChain, eltwise_pipeline
eltwise_activations / binary / math / misc / predicates / rounding / scalar / special / ternary / trig
eltwise_helpers.hpp            # aggregator; backward-compat include
```

Each file pulls in only the LLK headers it needs. Build time and "where does this op live" both improve. Keep the aggregator so existing call sites don't break.

### 5.2 Examples in the header doc-comment, not a separate guide

`binary_op_helpers.hpp`'s opening doc-comment has 10 worked examples covering basic add, broadcast variants, `WaitUpfrontNoPop` for softmax, `NoWaitNoPop` for sharded, independent A/B policies, skip-reconfig, `eltwise_chain(Rsqrt)` post-op, low-level `binary_op<>` form. Examples in the file the user already opened beat external docs that decay.

### 5.3 BroadcastDim and Reduce↔Broadcast tables in the header

`BroadcastDim::{NONE,ROW,COL,SCALAR}` + companion table mapping `REDUCE_ROW → COL`, `REDUCE_COL → ROW`, etc. The "reduce-row produces column-shaped output" surprise lives where it is needed. Don't bury it in a separate doc.

### 5.4 Per-helper blockers in per-helper doc-comments

Helper-specific blockers (e.g. "DestReuseOp's `WaitAndPop` requires `cb_tile_idx == 0`", "Auto+PerTile deadlock at small CB capacity") live in the helper's `.hpp` doc-comments. The general migration HQ doc only carries the helper-agnostic blockers. Caller reads the helper header before doing the audit.

---

## 6. Generalize on second use, not first

| First version | Second consumer appeared | Generalized to |
|---|---|---|
| `DestReuseMul<CB>` | `batch_norm` stage 2 needed `ELWSUB` with `DEST_TO_SRCB` | `DestReuseOp<CB, OpType, ReuseType, Slot>` + `DestReuseMul` alias |
| `CopyTile` only had hardcoded tile 0, default policy | Multi-CB upfront + indexed access | `cb_tile_idx`, `CopyTilePolicy`, `Reconfig` template params + `WaitUpfrontPopAtEnd` |
| `binary_op` PostOp = lambda | `DestReuseMul` PostOp pattern | `PostOp = NoOp \| EltwiseChain<...>`, single ops wrap |
| 3-corner CopyTilePolicy | Fan-out second-load explicit | 4-corner `{Wait,NoWait} × {Pop,NoPop}` |
| `chain_has_any_copy_tile_v` | `DestReuseOp` chains needed reinit | `clashes_with_fpu` + `chain_has_non_copy_tile_fpu_clash_v` |

Don't preemptively generalize. Don't refuse to generalize past the second consumer. The compile-time interface (template params + traits) absorbs the new dimension cleanly when the trigger is real.

---

## 7. Naming evolution is part of the design

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

## 8. Validation suite — non-negotiable

Every helper change (new op, new policy, new reconfig path, new enum value) must land alongside a pytest that runs a custom compute kernel on device against a torch golden. Build-only is not enough — kernels JIT.

The eltwise validation suite at `ttnn/cpp/ttnn/kernel_lib/tests/chain_and_binary/` covers:

1. `chain_fanout_x_times_exp_x` — fan-out CB read with `WaitNoPop + NoWaitPop`, single-compute-op chain.
2. `binary_same_cb_add` — same-CB wait/pop dedup. A broken dedup deadlocks or skips tiles.
3. `binary_dest_reuse_mul` — `DestReuseOp` as chain element, FPU-clash reinit.
4. `binary_postop_with_copy_tile` — `CopyTile` inside a PostOp chain.

Each parameterized over `num_tiles ∈ {1, 8, 64}` (single tile, fits in DEST, multi-DEST window). Tolerance accommodates compounded bf16 ULPs (rtol≈5e-2, atol≈1e-1 for products on `[-3, 3]`); a too-tight tolerance flags legit rounding.

Add a sister "raw-LLK" diagnostic kernel for any new compute path. If both helper and raw hang, the bug is in the LLK or the test scaffolding (reader push counts, CB capacity, persistent-tile lifecycle). If only the helper hangs, the helper is wrong. The diagnostic kernel saves hours.

---

## 9. Eltwise-specific migration enablers

The features below were each driven by a specific blocked kernel. The pattern: real kernel can't migrate → identify the missing primitive → small CRTP/policy addition → kernel migrates. Not features added speculatively.

| Feature | Triggered by | Enables |
|---|---|---|
| `FillScalar<Slot>{value}`, `FillConst<Bits, Slot>` | `hardshrink` | runtime/compile-time `fill_tile` in chain |
| `TanhDerivative<Approx, Slot>` | `tanh_bw` | `1 - tanh²` numerically-stable form |
| `Logsigmoid<In0, In1, Out>` (3-DST binary) | `logsigmoid` | exp(-x) and x in two DEST slots, fused output |
| `CopyDest<Src, Dst, DF>` | `gelu_backward` | save tanh result before squaring |
| `WaitUpfrontPopAtEnd` CopyTilePolicy | `tanh_bw`, `gelu_poly` | bulk-wait + indexed CopyTile + bulk-pop |
| `CopyTileReconfig::Input`, `DestReuseReconfig::Input` | mixed-dtype paths | per-element srca/srcb format reconfig |
| `BinaryDataFormatReconfig::SRCA_ONLY` etc. | small-window perf tuning | partial unpacker reconfig |
| `Max`, `Min` | `moreh_adam` AMSGRAD | binary SFPU max/min |
| `Mask<DF>`, `MaskPosInf<>` | softmax / moreh masking | attention masking with correct DataSlot+1 contract |

Eltwise helper roadmap: track open gaps in a `feature_gap_map` keyed by GAP-N, each gap pinning the list of blocked kernels and the fix complexity. Close gaps by yield (kernels-unblocked-per-LOC).

---

## 10. Things to avoid in the eltwise helper

- **Auto fast paths.** Single-op fast paths in chains, auto-batching, auto-merge of same-CB CopyTiles. Make optimizations explicit, never silent.
- **Lambda/functor PostOp.** One dispatch path through `binary_op`. Single ops wrap in `eltwise_chain`.
- **Hidden broadcast inference.** `BroadcastDim` is always passed explicitly by the caller — the helper does NOT default-pick based on operand shape. Every value of the enum (e.g. `NONE` for no broadcast, `ROW` / `COL` for 1-D broadcasts, `SCALAR` for a 1×1 tile, plus any others the helper supports) is a distinct dispatch path the caller chooses; "infer it from `Ht`/`Wt`" is forbidden.
- **Mid-loop dtype swaps without policy support.** Helpers do one entry-time reconfig. If the kernel switches dtypes mid-loop, that path stays raw or grows a mid-chain reinit policy.
- **Skipping `fp32_dest_acc_en=True` testing.** Every binary migration runs against `fp32_dest_acc_en ∈ {False, True}` and any mixed-dtype combo the original supported.
- **Hand-coding around a missing op struct in a kernel.** Add the op struct to the helper, rebuild, then migrate. The kernel never gets a workaround copy of the LLK call.
