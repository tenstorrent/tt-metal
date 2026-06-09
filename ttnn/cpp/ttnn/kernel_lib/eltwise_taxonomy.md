# Eltwise helper taxonomy (draft)

Goal: a small expressible vocabulary for the eltwise chain helpers that covers
the patterns observed in `pack_patterns.tsv` without exposing footguns
(arbitrary indexing, mismatched wait/pop counts, hidden DST stride math).

## Axes

```
Shape       EltwiseShape::of(Ht, Wt)
                factories: single() | row(c) | col(r) | from_n_tiles(n) | of(r, c)

Per input   (kind, lifecycle)        on the chain's input element (BinaryFpu, CopyTile, …)
Per output  (kind, lifecycle)        on PackTile elements
                lifecycle is a two-axis struct (WaitPolicy × PopPolicy on the
                input side; ReservePolicy × PushPolicy on the output side) —
                see "Lifecycle as a two-axis struct" below.

Chunk K     uint32_t, default DEST_AUTO_LIMIT      (Chunked lifecycle only)
                Tunable per-input AND per-output. Compile-time or runtime value.
                Variable last-chunk size supported (chain handles `min(K, remaining)`).
                Catalog evidence: block_size, num_tiles_per_cycle, ndst, subblock_w,
                blk, tile_granularity, REDUCE_GRANULARITY, SUB_EXP_GRANULARITY,
                STATS_GRANULARITY, DHT_GRANULARITY — none equal DEST_AUTO_LIMIT.
```

Derived (NEVER a caller knob): wait/pop count per CB, tile_id at iteration (r, c),
internal DST batching stride, broadcast b_idx.

## Input kinds — 4

```
Kind     Operand size    Iteration access (at (r, c))    Wait/pop count
──────   ─────────────   ────────────────────────────    ──────────────
Block    Ht × Wt         tile_id = r·Wt + c              Ht·Wt
Row      1  × Wt         tile_id = c                     Wt
Col      Ht × 1          tile_id = r                     Ht
Scalar   1  × 1          tile_id = 0                     1
```

Anything outside these four (arbitrary indexing, runtime offsets, closures) is
not the chain's job — the reader slices the CB upstream so the chain sees one
of these four shapes.

## Output kind — 1

```
Kind     Per chain element    Per iter at (r, c)
──────   ──────────────────   ───────────────────────────
Block    one PackTile         pack DST → CbOut, slot derived from lifecycle
```

Multi-CB fan-out is composition: N PackTile elements in one chain, each writing
to its own CbOut. They share the chain's DST acquire window.

## Lifecycle as a two-axis struct

Each input's lifecycle is `(WaitPolicy, PopPolicy)`. Each output's lifecycle is
`(ReservePolicy, PushPolicy)`. Named constants compose the legal pairs; custom
pairs are also constructible but validated by `is_legal_lifecycle()`.

```cpp
enum class WaitPolicy : uint8_t {
    None,        // chain emits no cb_wait_front
    PerTile,     // chain waits 1 per iter
    PerChunk,    // chain waits K per K-iter chunk
    Upfront,     // chain waits M once at entry  (M = kind's tile count)
    Cumulative,  // chain waits (i+1) per iter / chunk
};

enum class PopPolicy : uint8_t {
    None,        // chain emits no cb_pop_front
    PerTile,     // chain pops 1 per iter
    PerChunk,    // chain pops K per K-iter chunk
    AtEnd,       // chain pops M once at exit
};

struct InputLifecycle { WaitPolicy wait; PopPolicy pop; };

// Full-edge — chain owns both wait and pop (or neither for CallerManaged).
inline constexpr InputLifecycle Streaming      = {WaitPolicy::PerTile,    PopPolicy::PerTile};
inline constexpr InputLifecycle Chunked        = {WaitPolicy::PerChunk,   PopPolicy::PerChunk};
inline constexpr InputLifecycle Bulk           = {WaitPolicy::Upfront,    PopPolicy::AtEnd};
inline constexpr InputLifecycle Pipelined      = {WaitPolicy::Cumulative, PopPolicy::AtEnd};
inline constexpr InputLifecycle CallerManaged  = {WaitPolicy::None,       PopPolicy::None};
inline constexpr InputLifecycle BulkDrain      = {WaitPolicy::Upfront,    PopPolicy::PerTile};   // bulk wait + drain per-tile

// Half-edge — chain owns exactly one edge. Load-bearing for persistent
// broadcast operands (mean, recip_std, gamma, beta) that outlive the chain
// call. Catalog audit: ~110 sites use these.
inline constexpr InputLifecycle HeldBulk       = {WaitPolicy::Upfront,    PopPolicy::None};      // chain holds; caller pops
inline constexpr InputLifecycle HeldCumulative = {WaitPolicy::Cumulative, PopPolicy::None};      // cumulative hold
inline constexpr InputLifecycle HeldStream     = {WaitPolicy::PerTile,    PopPolicy::None};      // per-call hold (idempotent)
inline constexpr InputLifecycle DeferredPop    = {WaitPolicy::None,       PopPolicy::AtEnd};     // caller waited, chain pops
```

```cpp
enum class ReservePolicy : uint8_t { None, PerTile, PerChunk, Upfront };
enum class PushPolicy    : uint8_t { None, PerTile, PerChunk, AtEnd };

struct OutputLifecycle { ReservePolicy reserve; PushPolicy push; };

inline constexpr OutputLifecycle OutStreaming             = {ReservePolicy::PerTile,  PushPolicy::PerTile};
inline constexpr OutputLifecycle OutChunked               = {ReservePolicy::PerChunk, PushPolicy::PerChunk};
inline constexpr OutputLifecycle OutBulk                  = {ReservePolicy::Upfront,  PushPolicy::AtEnd};
inline constexpr OutputLifecycle OutBulkReservePerTile    = {ReservePolicy::Upfront,  PushPolicy::PerTile};
inline constexpr OutputLifecycle OutBulkReservePerChunk   = {ReservePolicy::Upfront,  PushPolicy::PerChunk};
inline constexpr OutputLifecycle OutCallerManaged         = {ReservePolicy::None,     PushPolicy::None};
```

The two `OutBulkReserve*` cells cover the SDPA `reduce_c` pattern (~17 catalog
sites) where downstream needs incremental visibility. `OutCallerManaged` covers
the tt-train L1-accumulator pack helper (4 sites) where the chain emits only
the pack_tile and the caller wraps reserve/push externally.

## Legal cells (named + validated)

`is_legal_input_lifecycle()` / `is_legal_output_lifecycle()` accept the named
constants above. Any custom struct literal whose `(wait, pop)` pair is not in
the legal set static_asserts at chain composition.

```
Input legal cells (10):
  Streaming        {PerTile,    PerTile}        full-edge
  Chunked          {PerChunk,   PerChunk}       full-edge
  Bulk             {Upfront,    AtEnd}          full-edge
  Pipelined        {Cumulative, AtEnd}          full-edge
  CallerManaged    {None,       None}           no edges
  BulkDrain        {Upfront,    PerTile}        full-edge (~5 sites, in-place patterns)
  HeldBulk         {Upfront,    None}           half-edge (~52 sites)
  HeldCumulative   {Cumulative, None}           half-edge (~33 sites)
  HeldStream       {PerTile,    None}           half-edge (~14 sites)
  DeferredPop      {None,       AtEnd}          half-edge (~7 sites)

Output legal cells (6):
  OutStreaming               {PerTile,  PerTile}
  OutChunked                 {PerChunk, PerChunk}
  OutBulk                    {Upfront,  AtEnd}
  OutBulkReservePerTile      {Upfront,  PerTile}    SDPA reduce_c (~17 sites)
  OutBulkReservePerChunk     {Upfront,  PerChunk}   layernorm welford (~3 sites)
  OutCallerManaged           {None,     None}       L1-acc helper (4 sites)
```

Combinations not in the legal set are static_asserted. Specifically rejected:
- `{None, PerTile}` (caller-wait + chain-per-tile-pop — no catalog evidence)
- `{None, PerChunk}` (same)
- `{PerTile, PerChunk}` / `{PerChunk, PerTile}` (mismatched rates)
- `{PerTile, AtEnd}` (1 niche site at topk_final — accepted as struct literal, not named)
- `{Cumulative, PerTile}` / `{Cumulative, PerChunk}` (no catalog evidence)
- `{Cumulative, Cumulative}` (reduce-into-DST — OOS, fan-in cardinality)

## Input matrix — 25 legal cells / 40

```
                Streaming  Chunked  Bulk  Pipelined  CallerManaged  BulkDrain  HeldBulk  HeldCumulative  HeldStream  DeferredPop
                ─────────  ───────  ────  ─────────  ─────────────  ─────────  ────────  ──────────────  ──────────  ───────────
  Block         ✓          ✓        ✓     ✓          ✓              ✓          ✓         ✓               ✓           ✓
  Row           ✗ E1       ✗ E1     ✓     ✗ E1       ✓              ✗ E1       ✓         ✓               ✗ E1        ✓
  Col           ✗ E1       ✗ E1     ✓     ✗ E1       ✓              ✗ E1       ✓         ✓               ✗ E1        ✓
  Scalar        ✗ E1       ✗ E1     ✓ N2  ✗ E1       ✓              ✗ E1       ✓         ✓ N3            ✗ E1        ✓
```

```
E1  static_assert: "Row/Col/Scalar kinds have operand size M < Ht·Wt iterations.
                    Lifecycles that pop more than M times (Streaming, Chunked,
                    Pipelined, BulkDrain, HeldStream) would over-consume.
                    Use a Bulk-family, Held*, DeferredPop, or CallerManaged
                    lifecycle whose pop count equals M (or zero)."

N2  Scalar + Bulk degenerates to wait(1) + pop(1).

N3  Scalar + HeldCumulative degenerates to Scalar + HeldBulk (M=1).
    Kept legal for naming symmetry across kinds.
```

## Output matrix — 5 legal cells

```
                OutStreaming  OutChunked  OutBulk  OutBulkReservePerTile  OutBulkReservePerChunk
                ────────────  ──────────  ───────  ─────────────────────  ──────────────────────
  Block         ✓             ✓           ✓        ✓                      ✓
```

The two `OutBulkReserve*` cells cover the SDPA `reduce_c` family (4 catalog
sites). They share the bulk reserve discipline with `OutBulk` but emit
push edges incrementally for downstream pipelining.

## Half-edge ownership — supported, with limits

Per catalog audit (~110 sites use half-edges), the chain supports five
half-edge cells where caller owns one edge:

```
Cell              Chain emits           Caller emits         Use case
────────────────  ──────────────────    ──────────────────   ─────────────────────────────────
HeldBulk          cb_wait_front(M)      cb_pop_front(M)      Reduction-result held by chain
HeldCumulative    cumulative wait       cb_pop_front(M)      gamma/beta persistent
HeldStream        per-call wait(1)      cb_pop_front(M)      moreh helper `pop=0` convention
DeferredPop       —                     cb_wait_front(N)     Caller pre-waited (prev phase),
                                        + chain does pop     chain owns cleanup
```

Half-edge cells NOT in the legal set (catalog showed zero benefit):
- `{None, PerTile}` / `{None, PerChunk}`: caller-wait + chain-per-tile-pop has
  no observed use case. The asymmetric direction (caller waits big, chain pops
  small slices) doesn't show up because per-slice pops are usually paired with
  per-slice waits (Chunked).
- `{Cumulative, PerTile}` / `{Cumulative, PerChunk}`: no catalog evidence.

These rejected combinations static_assert at chain composition with a
diagnostic naming a legal alternative.

## Compatibility rules enforced by static_assert

```
R1  lifecycle validity:  is_legal_lifecycle(input.lifecycle) for every input
                         is_legal_lifecycle(output.lifecycle) for every output
                         (the named constants are the legal set; custom struct
                         literals are validated against the table.)
R2  kind × lifecycle:    no E1 cells (Row/Col/Scalar can only be Bulk or
                         CallerManaged on the input side)
R3  single declaration:  any CB referenced by multiple chain elements must
                         agree on (kind, lifecycle)
R4  producer/consumer:   for an intermediate CB (one element packs, another
                         reads) the kind/lifecycle on both sides must be
                         identical
R5  chunk K constraint:  K parameter only meaningful when WaitPolicy/PopPolicy
                         is PerChunk (or ReservePolicy/PushPolicy is PerChunk)
R6  output kind == Block: single-cell column; no other kind values legal
```

## What the chain emits per cell

### Inputs

```
Block + Streaming
   for r in 0..Ht: for c in 0..Wt:
       cb_wait_front(CB, 1); access tile 0; (math); cb_pop_front(CB, 1)

Block + Chunked<K>
   for chunk in chunks_of(Ht·Wt, K):
       cb_wait_front(CB, len(chunk));
       for i in 0..len(chunk): access tile i; (math into DST[i])
       cb_pop_front(CB, len(chunk))

Block + Bulk
   cb_wait_front(CB, Ht·Wt);
   for chunk in chunks_of(Ht·Wt, K):                  # K = DEST_AUTO_LIMIT internal
       for i in 0..len(chunk): access tile (chunk_start + i); (math)
   cb_pop_front(CB, Ht·Wt)

Block + Pipelined
   cumulative = 0
   for chunk in chunks_of(Ht·Wt, K):
       cumulative += len(chunk); cb_wait_front(CB, cumulative);
       for i in 0..len(chunk): access tile (chunk_start + i); (math)
   cb_pop_front(CB, Ht·Wt)

Block + CallerManaged
   no edges; for each iter access tile (r·Wt + c) within caller's window

Row + Bulk
   cb_wait_front(CB, Wt);
   for r in 0..Ht: for c in 0..Wt: access tile c
   cb_pop_front(CB, Wt)

Col + Bulk
   cb_wait_front(CB, Ht);
   for r in 0..Ht: for c in 0..Wt: access tile r
   cb_pop_front(CB, Ht)

Scalar + Bulk
   cb_wait_front(CB, 1);
   for r in 0..Ht: for c in 0..Wt: access tile 0
   cb_pop_front(CB, 1)

{Row|Col|Scalar} + CallerManaged
   no edges; per-iter access tile (c|r|0) within caller's window
```

### Outputs

```
Block + Streaming
   acquire_dst; ... math ...; commit/wait
   cb_reserve_back(OUT, 1); pack_tile(SrcDst, OUT); cb_push_back(OUT, 1)
   release_dst

Block + Chunked<K>
   for chunk in chunks_of(Ht·Wt, K):
       acquire_dst; for i in 0..K: math into DST[i]; commit/wait
       cb_reserve_back(OUT, K)
       for i in 0..K: pack_tile(SrcDst[i], OUT, i)
       cb_push_back(OUT, K)
       release_dst

Block + Bulk
   cb_reserve_back(OUT, Ht·Wt)
   for chunk in chunks_of(Ht·Wt, K):
       acquire_dst; for i in 0..K: math; commit/wait
       for i in 0..K: pack_tile(SrcDst[i], OUT, absolute_slot)
       release_dst
   cb_push_back(OUT, Ht·Wt)
```

## Catalog → taxonomy mapping

```
sync_seq label          shape label                  Maps to
─────────────────────   ──────────────────────────   ──────────────────────────────
ACWPR                   modern-canonical/single      Block + Streaming  (input & output)
CWPR / CWPRA            modern-canonical/single      Block + Streaming with pop-ordering variation
CWPRACWPR (etc.)        modern-canonical/single      sequential PerTileReserveAndPush regions
                                                     (still Block + Streaming, just twice)
ACWPR                   single-in-loop               Block + Bulk + per-tile internal stride
                                                     (waits/pushes wrap an outer block loop)
aPr (raw-dst)           raw-dst/single               legacy acquire_dst — out of scope (R1 mismatch)
QPL / QPLQPL            ACQ-REL-macro/single         macro sugar over Block + Streaming
CumulativeWait usage    welford / sharded reductions Block + Pipelined
PerChunk (legacy)       binary_op_helpers WaitAndPop Block + Chunked  (input & output)
                        PerChunk + PerChunk
```

## Deliberate exclusions

```
What the catalog had              Why we don't model it
─────────────────────────────     ─────────────────────────────────────────────
Custom index closures (rotary,     Refactor the reader to slice the CB so the
 rmsnorm gamma+offset)             chain sees Row + Bulk (or Block + Streaming)
                                   per outer iter. Chain stays footgun-free.

Absolute(runtime k) index mode     Same — push runtime index logic upstream.

Pinned to non-zero k               Same — slice the CB so tile 0 is the right one.

Raw-dst (acquire_dst/release_dst)  Legacy half-DST API; ~30 catalog sites
                                   migrate to tile_regs_acquire/commit/wait/release
                                   when adopting the chain.

Output CallerManaged               Matmul epilogue's 3 sites refactor outside
                                   the eltwise chain (they're matmul-tail,
                                   not eltwise compute).

In-DST persistent accumulator      Welford and friends. ORTHOGONAL to all three
                                   axes — needs its own compute mode, not a
                                   new input kind.

Per-pack pack_tile_with_dt         Will be an orthogonal flag on PackTile
                                   (`DtReconfig = false` default), not a new
                                   lifecycle.

Pack-only helpers (no input)       Not eltwise compute; they're pack helpers.
                                   Live outside the chain entry points.
```

## Coverage audit criteria (for pack_patterns.tsv)

For each TSV row, classify the site as:

```
COVERED               site's (input wait/pop, output reserve/push, index access)
                      maps cleanly to one of the matrix cells.

COVERED_AFTER_REFACTOR site needs reader-side slicing or kernel restructuring
                      but the resulting compute fits the matrix. Acceptable.

OUT_OF_SCOPE          raw-dst, pure pack helpers, matmul epilogue, welford —
                      not eltwise compute; live in a different primitive.

NOT_COVERED           the chain genuinely cannot express this even after
                      reasonable refactor. These are the sites that decide
                      whether the taxonomy is adequate.
```

Target: ≥ 95% of sites in (COVERED ∪ COVERED_AFTER_REFACTOR ∪ OUT_OF_SCOPE).
Any NOT_COVERED count motivates adding to the matrix.

## Final audit result (666 sites)

```
SUPPORTABLE                                ~660 / 666   ~99%
  Full-edge (Streaming/Chunked/Bulk/        357         54%
   Pipelined/CallerManaged + BulkDrain)
  Half-edge (HeldBulk/HeldCumulative/      ~110        17%
   HeldStream/DeferredPop)
  COVERED_AFTER_REFACTOR (reader-side       ~10         1.5%
   CB slicing)
  OUT_OF_SCOPE                              183        27.5%

NOT SUPPORTABLE                              ~6 / 666   ~0.9%
  intimg conditional pack                   2           PackTile::when(predicate) — not modeled
  intimg pinned absolute output slot        1           caller-pinned slot — deliberately excluded
  groupnorm in-place CB rewrite             1           same CB in & out — deliberately excluded
  bcast-fill unary_bcast LLK                6 (OOS)     1 DST → N CB slots (special LLK, OOS)
  Drained {Cumulative, Cumulative}          4 (OOS)    reduce-into-DST, fan-in cardinality
  topk_final staged-flush                   1 (legal   {PerTile, AtEnd} via struct literal,
                                                       no named constant)
```

Coverage progression:
- Single-axis design (pre-collapse 9-cell `CopyTilePolicy` alias surface): 97.7%
- Two-axis, named full-edge only:               97.7% (same)
- Two-axis + 5 half-edge named constants:       ~99%
- Two-axis + half-edge + OutCallerManaged:      ~99% (one more cell named)

The 5 half-edge cells reclaim ~110 sites without compromising the static_assert
discipline — rejected combinations are still caught at chain composition with
specific diagnostics naming legal alternatives.

## Open questions for the audit

```
Q1  Are there sites that need INPUT Chunked + non-DEST_LIMIT K (e.g., K=4 for
    L1 budget reasons)? If yes, K-on-Chunked-input must be tunable.

Q2  Are there sites that use multiple PackTile elements with DIFFERENT
    lifecycles (one Streaming, one Bulk) on the SAME chain? If yes, confirm
    the chain can emit both schedules from one DST acquire window.

Q3  How many "asymmetric" sites (bulk reserve + per-tile push) reduce cleanly
    to Block + Bulk, vs. genuinely needing intermediate visibility?

Q4  Sites where input lifecycle is Streaming but output is Bulk (or vice
    versa) — should the chain emit independent edge schedules, or require
    matching lifecycles?
```
