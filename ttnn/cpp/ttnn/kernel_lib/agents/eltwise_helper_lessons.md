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

### 2.7 CB indexing modes — constrained by the wait shape

Every chain element that reads from a CB (CopyTile, `binary_op` A/B operands, `DestReuseOp` CB side, any future CB-input op) must support the four indexing patterns kernels actually use:

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

Surface as a small enum (`CbIndexMode::FirstTile / BlockIter / Pinned / Absolute`) on the element, plus a runtime `cb_tile_idx_` field for `Pinned` / `Absolute`. `FirstTile` and `BlockIter` need no field. Mode is per-CB-operand — `binary_op` lets A and B pick independently (e.g. `A=BlockIter, B=Pinned` for `block * scalar` under `WaitUpfrontPopAtEnd`).

Validation:

- The chain combinator `static_assert`s the illegal cells whenever both axes are compile-time (most common case — index mode is a template param, wait policy is a template param). A `WaitAndPop` element instantiated with `BlockIter` or `Absolute` fails to compile.
- For runtime-supplied indices (`Pinned k`, `Absolute idx`), the `WaitUpfrontPopAtEnd(N)` path runtime-`ASSERT`s `idx < N`; the single-tile-window policies runtime-`ASSERT` the index is `0` (per §4.1).
- `NoWaitNoPop` is the only escape hatch — caller asserts the window externally; helper trusts it - this can be unsafe, and shouldn't be default.

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

### 3.4 Init hoisting is opt-in, narrowly scoped, never the default

Per-tile init is the default. Hoisting `init()` out of the loop is a perf optimization with a tight precondition set — get any condition wrong and the chain silently produces wrong output. The default path runs every element's `init()` + `exec()` every tile; readers can answer "what runs per tile" by reading the chain.

Hoisting is only allowed when **all** of the following hold:

1. Chain shape is exactly `CopyTile + 1 SFPU op` (one CB load + one SFPU compute). Longer chains, multiple CopyTiles, or any FPU-clobbering element disqualify.
2. No element between iterations reprograms hardware state the hoisted `init()` set up — for the `CopyTile + 1 SFPU` shape this reduces to "the SFPU op doesn't touch unpack MOP / srca format," which is the common case but still must be validated by the SFPU op's traits.

Each chain element's `init()` programs specific hardware state — unpack MOP, math MOP, ADDR_MOD, srca/srcb format reconfig, packer reconfig, etc. Encode "what hardware state I touch in init/exec" as static-constexpr traits (`clashes_with_fpu`, plus finer-grained traits if needed: `clobbers_unpack_mop`, `reconfigs_srca`, etc.). The chain combinator computes a conservative `chain_is_hoist_safe_v<Chain>` that requires all three conditions above; if any fails, init runs per tile.

Examples:

- `CopyTile + Exp` (single input) — hoist-safe. Both inits hoisted out, loop runs `exec` only.
- `CopyTile + Exp + Sqrt` — chain length > 2, **not** hoist-safe even though all SFPU, **is** hoist safe for copy tile.
- `CopyTile<cbA> + CopyTile<cbB> + Add` — multiple CB inputs, **not** hoist-safe. Init per tile.
- `CopyTile + DestReuseOp` — FPU dest-reuse clobbers unpack MOP each iteration, **not** hoist-safe.

Wrong output from an over-eager hoist is not a perf optimization. When in doubt, init per tile.

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
4. `binary_postop_with_copy_tile` — `CopyTile` inside a PostOp chain.

Each parameterized over `num_tiles ∈ {1, 8, 64}` (single tile, fits in DEST, multi-DEST window). Tolerance accommodates compounded bf16 ULPs (rtol≈5e-2, atol≈1e-1 for products on `[-3, 3]`); a too-tight tolerance flags legit rounding.

Add a sister "raw-LLK" diagnostic kernel for any new compute path. If both helper and raw hang, the bug is in the LLK or the test scaffolding (reader push counts, CB capacity, persistent-tile lifecycle). If only the helper hangs, the helper is wrong. The diagnostic kernel saves hours.

The dtype-matrix and untestable-locally rules are general — see [llk_helpers_hq.md → Step 4](llk_helpers_hq.md#step-4--verify-on-device).

### 7.1 srcB reconfig path is its own dimension (eltwise-specific)

Standard FPU binary (`add_tiles`, `sub_tiles`, `mul_tiles`) consumes srcA *and* srcB; either operand can drive a format reconfig on entry. `DestReuseOp` adds the `DEST_TO_SRCB` path (DEST → srcB, CB → srcA). In every case srcA-reconfig and srcB-reconfig are **separate LLK paths** even when they share a single helper enum value (e.g. `BinaryDataFormatReconfig::INPUT_AND_OUTPUT`, `DestReuseReconfig::Input`, see §2.4). A migration that only validates srcA-reconfig has not validated the srcB branch.

Test matrix must cover:

- `binary_op` srcA-reconfig: A operand changes dtype, B unchanged.
- `binary_op` srcB-reconfig: B operand changes dtype, A unchanged.
- `binary_op` both reconfigured: A and B both change dtype between calls.
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

---

## 9. Things to avoid in the eltwise helper

- **Auto fast paths.** Single-op fast paths in chains, auto-batching, auto-merge of same-CB CopyTiles. Make optimizations explicit, never silent.
- **Default init hoisting.** Hoisting is opt-in and gated on the strict precondition set in §3.4 (chain = `CopyTile + 1 SFPU op`, single CB input, no clobbering element). Never the default — wrong output is not a perf win.
- **Lambda/functor PostOp.** One dispatch path through `binary_op`. Single ops wrap in `eltwise_chain`.
- **Hidden broadcast inference.** `BroadcastDim` is always passed explicitly by the caller — the helper does NOT default-pick based on operand shape. Every value of the enum (e.g. `NONE` for no broadcast, `ROW` / `COL` for 1-D broadcasts, `SCALAR` for a 1×1 tile, plus any others the helper supports) is a distinct dispatch path the caller chooses; "infer it from `Ht`/`Wt`" is forbidden.
- **Mid-loop dtype swaps without policy support.** Helpers do one entry-time reconfig. If the kernel switches dtypes mid-loop, that path stays raw or grows a mid-chain reinit policy.
- **Skipping `fp32_dest_acc_en=True` testing.** Every binary migration runs against `fp32_dest_acc_en ∈ {False, True}` and any mixed-dtype combo the original supported.
- **Hand-coding around a missing op struct in a kernel.** Add the op struct to the helper, rebuild, then migrate (see §10.2 — missing op struct is fix-first, not a blocker). The kernel never gets a workaround copy of the LLK call.

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
