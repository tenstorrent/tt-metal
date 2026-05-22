# Policy alias collapse — proposal

## Status

DRAFT — awaiting sign-off. No code changes have landed for this proposal yet.

## Problem

`eltwise_chain.hpp` currently exposes **two parallel spellings** for the same
chain-element template arguments:

| Concept | Legacy alias (struct + `static constexpr` members) | Low-level constant / enum |
|---|---|---|
| Input wait/pop lifecycle | `CopyTilePolicy::WaitAndPop`, `CopyTilePolicy::WaitUpfrontPopAtEnd`, `CopyTilePolicy::NoWaitNoPop`, … (9 names) | `Streaming`, `Bulk`, `CallerManaged`, … (9 `InputLifecycle` values) |
| Input index kind | `CbIndexMode::FirstTile`, `CbIndexMode::BlockIter`, `CbIndexMode::RowBcast`, `CbIndexMode::ColBcast` | `OperandKind::Scalar`, `OperandKind::Block`, `OperandKind::Row`, `OperandKind::Col` |
| Output reserve/push lifecycle | `PackTilePolicy::PerTileReserveAndPush`, `PackTilePolicy::UpfrontReservePushAtEnd`, … (6 names) | `OutStreaming`, `OutBulk`, … (6 `OutputLifecycle` values) |
| Output index kind | `PackTileIndexMode::FirstTile`, `PackTileIndexMode::BlockIter` | `OperandKind::Scalar`, `OperandKind::Block` |

Both spellings compile to the same value. The legacy aliases are documented as
"existing call sites continue to compile" — i.e. a back-compat shim, not the
intended API.

Concrete costs of keeping both:

1. **Reader cost.** A kernel template parameter list typed as `InputLifecycle`
   shows the value `CopyTilePolicy::WaitAndPop` at the call site. The reader has
   to mentally rewrite `CopyTilePolicy::WaitAndPop → InputLifecycle{PerTile,
   PerTile} → "Streaming"` to reason about the chain. The low-level name
   `Streaming` is identity-mapped to the type.
2. **Inconsistent migrations.** New kernel migrations on this branch use the
   low-level constants (`rmsnorm_pre_allgather.cpp`, `rmsnorm_post_allgather.cpp`).
   The ref branch (`astancov/eltwise_run7_refined_rebase_v2`) deleted the legacy
   struct wrappers entirely. We are sitting on a forked state where two styles
   coexist with no documented preference.
3. **No room to grow.** `CopyTilePolicy::*` is a fixed enumeration of nine
   pre-baked combos. The lifecycle is actually a 2D space (`WaitPolicy` ×
   `PopPolicy`); the low-level `InputLifecycle{...}` struct exposes that
   directly. Any future lifecycle (e.g. a deferred-bulk variant) has to be
   bolted into `CopyTilePolicy` as a new alias, or skipped entirely — but on
   the low-level side it falls out of the existing types for free.
4. **Search surface.** `grep CopyTilePolicy` finds use-sites; `grep Streaming`
   finds use-sites; both together miss neither, but a reader does one search,
   not both. Searches return half the relevant hits depending on which name
   they happen to type.

## Audit — what's exposed to outside callers today

```
$ grep -rln "CopyTilePolicy::|CbIndexMode::|PackTilePolicy::|PackTileIndexMode::" \
    ttnn/cpp --include="*.cpp" --include="*.hpp"

ttnn/cpp/ttnn/kernel_lib/eltwise_convenience.hpp
ttnn/cpp/ttnn/kernel_lib/eltwise_optional.hpp
```

- **0 kernel files** use the legacy aliases.
- **0 test files** use the legacy aliases.
- **2 internal helper files** use them — `eltwise_convenience.hpp` (~15 sites)
  and `eltwise_optional.hpp` (~2 sites).

The 4 already-migrated kernels on this branch (`batch_norm_kernel`,
`layernorm_pre_allgather`, `rmsnorm_pre_allgather`, `rmsnorm_post_allgather`)
all use the low-level constants.

In other words: the legacy alias surface only has internal users. Removing it
costs ~17 lines of mechanical rename in two files we already own.

## Recommendation

**Collapse to the low-level constants. Delete the four struct wrappers.**

Why this direction, not the reverse:

- The low-level constants are the actual types in the chain element template
  parameter list (`InputLifecycle`, `OperandKind`, `OutputLifecycle`). The
  alias structs add a layer of indirection over a name that already exists.
- The ref branch already removed the struct wrappers. Aligning lets future
  pull-overs of ref-branch migrations apply cleanly.
- Composition with `TileBase` and other policy attributes is uniform on the
  low-level side; the alias wrappers don't compose with anything.
- The legacy alias name `CopyTilePolicy::CumulativeWaitNoPop` doesn't
  obviously communicate "the chain holds the wait position across iterations";
  the low-level name `HeldCumulative` says exactly that.
- Future lifecycles (e.g. `BulkDrain`, `DeferredPop` — already named on the
  low-level side) don't need a corresponding `CopyTilePolicy::` alias.

## Rename mapping

The mechanical mapping the migration script + kernel-author cheat sheet shares:

| Legacy alias | Low-level | Type |
|---|---|---|
| `CopyTilePolicy::WaitAndPop` | `Streaming` | `InputLifecycle` |
| `CopyTilePolicy::WaitNoPop` | `HeldStream` | `InputLifecycle` |
| `CopyTilePolicy::NoWaitPop` | `NoWaitPop` (constant of the same name) | `InputLifecycle` |
| `CopyTilePolicy::NoWaitNoPop` | `CallerManaged` | `InputLifecycle` |
| `CopyTilePolicy::WaitUpfrontPopAtEnd` | `Bulk` | `InputLifecycle` |
| `CopyTilePolicy::WaitUpfrontNoPop` | `HeldBulk` | `InputLifecycle` |
| `CopyTilePolicy::CumulativeWaitPopAtEnd` | `Pipelined` | `InputLifecycle` |
| `CopyTilePolicy::CumulativeWaitNoPop` | `HeldCumulative` | `InputLifecycle` |
| `CopyTilePolicy::WaitAndPopPerBlock` | `Chunked` | `InputLifecycle` |
| `CbIndexMode::FirstTile` | `OperandKind::Scalar` | `OperandKind` |
| `CbIndexMode::BlockIter` | `OperandKind::Block` | `OperandKind` |
| `CbIndexMode::RowBcast` | `OperandKind::Row` | `OperandKind` |
| `CbIndexMode::ColBcast` | `OperandKind::Col` | `OperandKind` |
| `PackTilePolicy::PerTileReserveAndPush` | `OutStreaming` | `OutputLifecycle` |
| `PackTilePolicy::PerTileReserveNoPush` | `OutHeldReserve` | `OutputLifecycle` |
| `PackTilePolicy::NoReservePushAtEnd` | `OutDeferredReserve` | `OutputLifecycle` |
| `PackTilePolicy::NoReserveNoPush` | `OutCallerManaged` | `OutputLifecycle` |
| `PackTilePolicy::UpfrontReservePushAtEnd` | `OutBulk` | `OutputLifecycle` |
| `PackTilePolicy::PerBlockReserveAndPush` | `OutChunked` | `OutputLifecycle` |
| `PackTileIndexMode::FirstTile` | `OperandKind::Scalar` | `OperandKind` |
| `PackTileIndexMode::BlockIter` | `OperandKind::Block` | `OperandKind` |

Naming nit: the legacy `CopyTilePolicy::NoWaitPop` and the low-level constant
are spelled the same. After the collapse, callers just drop the
`CopyTilePolicy::` prefix.

## Migration steps

One commit per step, so any one of them is trivially revertible.

1. **Port internal helper users.**
   - `eltwise_convenience.hpp`: ~15 sites of `CopyTilePolicy::WaitAndPop` and
     `PackTilePolicy::PerTileReserveAndPush` → `Streaming`, `OutStreaming`.
   - `eltwise_optional.hpp`: 2 sites of `CopyTilePolicy::NoWaitNoPop` → `CallerManaged`.
   - Build the kernel_lib pytest suite and a sample migrated kernel to confirm
     no regression.
2. **Add a `[[deprecated]]` annotation** on each of `CopyTilePolicy`,
   `CbIndexMode`, `PackTilePolicy`, `PackTileIndexMode` (or on each individual
   `static constexpr` member). This is the "release-cycle" safety net — if any
   downstream branch picks this up and finds an external caller, the compiler
   warns instead of breaking. Land this commit and pause for a cycle if the
   project policy demands it; skip straight to step 3 if not.
3. **Delete the four struct wrappers** from `eltwise_chain.hpp` (currently
   ~lines 619-748):
   - `struct CopyTilePolicy { ... };`
   - `struct CbIndexMode { ... };`
   - `struct PackTilePolicy { ... };`
   - `struct PackTileIndexMode { ... };`
   - Drop the "Legacy use-site syntax mapped to …" doc paragraphs that
     accompany each struct.
4. **Update the chain header doc-comment** to reference only the low-level
   types in examples. The header currently shows
   `CopyTilePolicy::WaitUpfrontPopAtEnd` in its top-of-file example — change
   to `Bulk`.
5. **Update `llk_helpers_conventions.md` and `eltwise_taxonomy.md`** if either
   document mentions the alias names. Quick `grep CopyTilePolicy
   ttnn/cpp/ttnn/kernel_lib/agents/` and similar should catch them.
6. **Note in `llk_helpers_hq.md` § Kernel Migration Steps**: "use low-level
   constants (`Streaming`, `Bulk`, `OperandKind::*`, etc.); legacy
   `CopyTilePolicy::*` aliases were removed in <commit>."

Mechanical replacement script (for step 1 and the eventual external-repo sweep):

```bash
# Rough sed map; review each diff.
sed -i \
  -e 's/CopyTilePolicy::WaitAndPop\b/Streaming/g' \
  -e 's/CopyTilePolicy::WaitNoPop\b/HeldStream/g' \
  -e 's/CopyTilePolicy::NoWaitPop\b/NoWaitPop/g' \
  -e 's/CopyTilePolicy::NoWaitNoPop\b/CallerManaged/g' \
  -e 's/CopyTilePolicy::WaitUpfrontPopAtEnd\b/Bulk/g' \
  -e 's/CopyTilePolicy::WaitUpfrontNoPop\b/HeldBulk/g' \
  -e 's/CopyTilePolicy::CumulativeWaitPopAtEnd\b/Pipelined/g' \
  -e 's/CopyTilePolicy::CumulativeWaitNoPop\b/HeldCumulative/g' \
  -e 's/CopyTilePolicy::WaitAndPopPerBlock\b/Chunked/g' \
  -e 's/CbIndexMode::FirstTile\b/OperandKind::Scalar/g' \
  -e 's/CbIndexMode::BlockIter\b/OperandKind::Block/g' \
  -e 's/CbIndexMode::RowBcast\b/OperandKind::Row/g' \
  -e 's/CbIndexMode::ColBcast\b/OperandKind::Col/g' \
  -e 's/PackTilePolicy::PerTileReserveAndPush\b/OutStreaming/g' \
  -e 's/PackTilePolicy::PerTileReserveNoPush\b/OutHeldReserve/g' \
  -e 's/PackTilePolicy::NoReservePushAtEnd\b/OutDeferredReserve/g' \
  -e 's/PackTilePolicy::NoReserveNoPush\b/OutCallerManaged/g' \
  -e 's/PackTilePolicy::UpfrontReservePushAtEnd\b/OutBulk/g' \
  -e 's/PackTilePolicy::PerBlockReserveAndPush\b/OutChunked/g' \
  -e 's/PackTileIndexMode::FirstTile\b/OperandKind::Scalar/g' \
  -e 's/PackTileIndexMode::BlockIter\b/OperandKind::Block/g' \
  <files>
```

Each call site must add `compute_kernel_lib::` qualification (or rely on
existing qualification at the site) — the alias structs were inside the
namespace, the constants are too.

## What does NOT change

- The chain element template parameter **types** (`InputLifecycle`,
  `OperandKind`, `OutputLifecycle`) stay. Only the named values change spelling.
- The constants themselves (`Streaming`, `Bulk`, `OperandKind::Scalar`, …)
  are already present and used; they're the destination, not new surface.
- No behavioural change. Every value is identity-equal pre/post.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| External branch / fork uses `CopyTilePolicy::*` | Step 2 (`[[deprecated]]` annotation pass) gives one cycle of warnings before deletion. Skip step 2 only if a sweep of every active personal branch confirms no usage. |
| Future helper PR mentally types `CopyTilePolicy` from muscle memory | A short row in `llk_helpers_conventions.md` and the deletion-commit message explicitly call out the rename. |
| Diff in step 3 conflicts with other in-flight branches | Step 3 touches only `eltwise_chain.hpp`. Conflict resolution is mechanical — accept the deletion. |
| `NoWaitPop` clash between the alias and the low-level constant | Both spell the same name. After the alias is gone, the constant remains. No call site changes meaning. |

## Decision required

- **Approve "collapse to low-level, with one-cycle deprecation"** (steps 1-6 as written).
- **Approve "collapse to low-level, immediate deletion"** (skip step 2 — fastest, fine because internal-only).
- **Reject "collapse"** and instead document the dual API as supported. Pick one style as preferred-for-new-code in `llk_helpers_conventions.md` and leave the other as a back-compat. (This is the status quo; the proposal exists because the status quo is what we're trying to leave.)

The audit makes "immediate deletion" plausible — there are no external
callers to warn. If the project policy is "no deprecation cycle" and the
team is willing to accept the small risk of a personal-branch surprise,
that's the cheapest landing.
