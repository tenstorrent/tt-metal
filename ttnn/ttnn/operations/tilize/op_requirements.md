# Operation Requirements: tilize

## Definition

- **Formula**: pure permutation of byte positions — `out_tile[r, c][i, j] = in[r*TILE_H + i, c*32 + j]`
  for every output tile `(r, c)` (faces TL/TR/BL/BR, face-row-major inside a tile). No arithmetic.
  Under a pad request, every output position with no corresponding input element is `pad_value`.
  Logical shape is preserved; the *padded* shape becomes the pad target.
- **PyTorch Reference** (identity on the data region, `F.pad` on the padded view):

```python
def torch_tilize(x: torch.Tensor, padded_shape=None, pad_value=None) -> torch.Tensor:
    """tilize reorders bytes only. `to_torch(out)` == x; the padded readback == this."""
    if padded_shape is None:
        return x
    pads = []
    for got, want in zip(reversed(x.shape), reversed(padded_shape)):
        pads += [0, want - got]
    return torch.nn.functional.pad(x, pads, value=pad_value)
```

- **Import Path**: `from ttnn.operations.tilize import tilize`
- **Function Signature**:

```python
tilize(
    input_tensor: ttnn.Tensor,                       # ROW_MAJOR_LAYOUT input
    memory_config: ttnn.MemoryConfig | None = None,  # output mem config (default: input's)
    *,
    dtype: ttnn.DataType | None = None,              # output dtype (default: input's)
    use_multicore: bool = True,                      # multi-core work distribution
    use_double_buffer: bool = True,                  # depth-2 CBs (True) / depth-1 (False)
    output_padded_shape: list[int] | ttnn.Shape | None = None,  # pad target
    pad_value: float | int | None = None,            # fill for padded positions
    tile: ttnn.Tile | None = None,                   # output tile shape (default: 32x32)
) -> ttnn.Tensor
```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases — and must not regress **performance**. Check the refinement's `changelog.md` perf table against the cumulative bench set recorded by prior phases (Phase 0's table: `[1,1,2048,2048]`, `[1,1,32,16384]`, `[1,1,8192,1024]`, `[1,1,32,64]` × {bf16, fp32}); no prior bench shape's device kernel duration may regress beyond the measurement noise margin (Phase 0 measured ≤3% run-to-run spread — that is the band a "win" must clear). If the refinement changed a shape-dependent code path (work distribution, blocking, CB geometry, dtype branch, placement) but the changelog benched only a single point on that axis, flag the missing coverage — a non-regression gate that never measured the other regime cannot have cleared it.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N`. When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … Order follow-ups immediately after their parent. The runner's parser matches exactly `Refinement \d+[a-z]?`.
> **Perf-1 anchor (verifier note on ordering)**: the perf target configuration — the two mandatory bench regimes `[1,1,2048,2048]` (grid-filling square) and `[1,1,32,16384]` (wide/short, `nt_h=1`), bf16 **and** fp32, interleaved DRAM→DRAM, `use_multicore=True`, `use_double_buffer=True`, `tile_aligned`, `pad_mode="none"` — is **already fully inside Phase-0 SUPPORTED** (no `feature_spec.LOOSE_CASES` exists for this op, so the perf region is verifier-selected from the prompt's mandatory bench regimes). Nothing has to be unlocked before Refinement 3, so Refinements 1–2 are ordered **hardest-first** per the difficulty ranking instead. They must still ship at the performance-conformance bar (full grid, both dataflow halves batched, CBs deep enough to overlap) — Refinement 3 measures a path they are not allowed to have regressed.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16, float32] (input) · **output_dtype**: [bfloat16, float32, bfloat8_b]
- **SUPPORTED layout**: input ROW_MAJOR only (`in_layout=[ROW_MAJOR_LAYOUT]`, `in_tile_height=["none"]`), output TILE
- **SUPPORTED shape-derived axes**: `rank ∈ {2,3,4,5}`, `alignment ∈ {tile_aligned, w_non_aligned, h_non_aligned, hw_non_aligned}`
- **SUPPORTED op-specific axes**: `use_multicore ∈ {False, True}`, `double_buffer ∈ {False, True}`,
  `buffer ∈ {dram_to_dram, dram_to_l1, l1_to_l1, l1_to_dram}`, `pad_mode ∈ {none, auto, explicit}`,
  `pad_value ∈ {none, zero, positive, negative}`, `tile_height=[32]`,
  `shard_api=["none"]`, `out_scheme=["interleaved"]`, `orientation=["none"]`
- **EXCLUSIONS**: `bfloat8_b` output × `pad_mode ∈ {auto, explicit}`; `bfloat16 → float32` × `pad_value ∈ {positive, negative}`
- **Blocking model**: block = 1 tile-row × `WT_CHUNK` tile-columns; `WT_CHUNK` / `NT_BLK` / `CB_DEPTH` / `NUM_CORES`
  are named knobs with one source in `derive_blocking()` + `cb_pages()`/`cb_bytes()`; both CBs bounded in W.
- **Cores**: full grid (`split_work_to_cores(..., row_wise=True)` over W-chunk-major blocks); single-core on `use_multicore=False`
- **Compute config**: `fp32_dest_acc_en` + `UnpackToDestFp32` on fp32→fp32 only (bit-exact); `Fp32Mode::Fast` always
- **Golden baseline**: `supported_pass=102`, `supported_fail=0`, `xpass_drift=0`, `xfail_wrong_mode=0`,
  `xfail_expected=246` (216 unbuilt axes + 30 EXCLUSIONS), `invalid_skipped=568`, 24 retile cells arch-skipped (per `verifier_report.json`)

---

### [ ] Refinement 1 — Sharded placement: same-spec zero-copy + interleaved↔sharded crossover

**Goal**: add the sharded placement surface to SUPPORTED —
`shard_api += ["legacy_2d", "nd"]`, `out_scheme += [HEIGHT_SHARDED, WIDTH_SHARDED, BLOCK_SHARDED, "nd"]`,
`orientation += [ROW_MAJOR, COL_MAJOR]` — for the cases where the input and output shard specs are the
**same**, plus the two crossovers (DRAM-interleaved RM → sharded TILE, sharded RM → DRAM-interleaved
TILE). On the sharded side the CB must be **placed on the resident L1 shard** with
`ttnn.cb_descriptor_from_sharded_tensor` so `tilize_block` packs straight into (or unpacks straight out
of) the local shard — **zero NoC traffic on that side**. Keep the wide-W CB bound: a wide HEIGHT shard
must reuse `derive_blocking()`'s `WT_CHUNK` so per-core CB L1 stays constant in W instead of OOMing.
This is the largest single gap in the suite: 104 of the hidden grader's 144 failures and ~150 of the
246 xfail cells are blocked on `shard_api` / `out_scheme` / `orientation` alone.

**Implementation skill**: /memory-budget-metal

**Verifier notes**: the skill pointer covers **only** the wide-W CB-bound sub-item (a wide HEIGHT shard
is the L1-OOM shape); sharded *placement* has no skill in the inventory yet, so the mechanism is stated
here: `ttnn.cb_descriptor_from_sharded_tensor` CB placement (design lamp **L1**, `program_descriptors.cpp:517-556`).
Do **not** attach `/memory-layouts` — that is RM/TILE layout, not placement.
Blocking-model class: **knob-turn**. Per `op_design.md` §1.2 tilize has *no dependent axis*, so a local
shard only pins `NUM_CORES`, the per-core row range and `WT_CHUNK = WT_shard`; the loop nest is unchanged.
**A core must never re-read its own local shard through a `TensorAccessor`** — an accessor read of the
resident shard means the `memory_layout` value was never implemented, only tolerated, and green golden
cells do not clear it. Two other consequences to carry: (1) sharded input is inherently multi-core, so
`use_multicore=False` × sharded belongs in `EXCLUSIONS`, not in SUPPORTED; (2) design lamp **L4**
(`split_reader`) becomes *applicable* for the first time on the sharded-output path (BRISC is free once
the writer does no NoC work) — record it in `lever_ledger.json`, but it is Refinement 3's business, not a
gate here. Cross-spec reshards stay excluded here and are Refinement 2.

**Done when**: the same-spec and crossover sharded golden cells pass for HEIGHT / WIDTH / BLOCK / nd in
both orientations; no DRAM traffic on the sharded side; a wide-W HEIGHT crossover keeps per-core CB L1
constant in W; `use_multicore=False` × sharded is refused by `validate()`; the Phase-0 cumulative bench
set shows no regression.

---

### [ ] Refinement 2 — Cross-spec reshard (general cross-core L1 path) + padded sharded cells

**Goal**: move the cells Refinement 1 left in `EXCLUSIONS` into passing: input shard spec ≠ output shard
spec (including `nd ↔ legacy_2d`, different schemes, uneven/padded shard grids) via a **general
`TensorAccessor` cross-core L1 gather** — the source shard lives on core A, the destination shard on core
B, and the data moves L1→L1 with **zero DRAM staging**. Then add padding on top of the sharded paths
(the `pad_mode ∈ {auto, explicit}` × sharded golden cells), which reuses the existing R_PAD reader fill
against the new placement.

**Verifier notes**: no skill in the inventory covers this. Blocking-model class: **scheme-change**
(design lamp **L2**) — the new data-placement topology *is* the work, which is why it is not bundled into
Refinement 1. Still **no cross-core combine**: nothing is reduced, so no semaphores/mcast are needed;
the reader is already accessor-driven and already takes an arbitrary `start_page`, so the gather is a
different accessor over an L1-sharded source, not a different loop nest.
Gating caveat found during verification: cross-spec is only *partly* expressible on the current tagger
set — `shard_api=legacy_2d ∧ out_scheme=nd` distinguishes an API crossover, but a legacy→legacy *scheme*
change (HEIGHT in → WIDTH out) projects to the same axis tuple as the same-spec WIDTH cell, so it cannot
be excluded by an axis dict. Consequences: (a) if the general path is out of reach in one pass, ship
`[~]` and file `Refinement 2b` — do not try to fake the gate; (b) the one sanctioned way to make it
gateable is a new tagger (e.g. `reshard ∈ {same_spec, cross_spec}`), which is legitimate here *because
the kernel really has two code paths* (local zero-copy vs cross-core gather) — add it to
`INPUT_TAGGERS` + `SUPPORTED` together if you take that route.

**Done when**: the cross-spec / `nd ↔ legacy` golden cells and `test_regression.py`'s reshard cases pass
with no hangs and no DRAM staging on either side; the padded × sharded cells pass (data region identical,
pad region exactly the fill); every cell Refinement 1 put in `EXCLUSIONS` for cross-spec reasons is gone
from `EXCLUSIONS`; Phase-0 + Refinement-1 bench sets show no regression.

---

### [ ] Refinement 3 — Close the DRAM-bandwidth gap on the interleaved aligned path

**Type**: perf

**Goal**: the aligned interleaved path is **DM-bound** (Phase 0's ablation: removing compute moves the
wall ≤4.5%, removing DM collapses it 88–95%) and sits at **0.63 / 0.55 / 0.68 of its DRAM-floor target**
(180.7 / 157.5 / 196.8 GB/s of a 288 GB/s peak) on the three real-work bench regimes. The recorded
largest headroom is the ~92%-of-peak recipe the op does *not* yet build: **A3** (reader adjacent to its
DRAM bank, one reader ↔ one bank — stop stacking readers onto shared routes) plus **B10** (per-reader VC
assignment, to break first-come-first-serve serialization on shared routes), predicted ~35% in
`lever_ledger.json`. Target shapes: `[1,1,2048,2048]` (grid-filling square) and the **mandatory**
`[1,1,32,16384]` (`nt_h=1`, grid-fill regime), bf16 and fp32. If the reader is rewritten anyway, the
second lever to price is **B8** (trid double-issue — barrier on the *previous* transaction id), which is
the `NT_BLK > 1` knob-turn design lamp **L3** already sized the CB formula for; it is only measurable on
a ≥2-blocks/core shape, so bench it on `[1,1,8192,1024]` (4 blocks/core), never on a one-block shape.
Levers, examples and the B0 small-regime caveat: `ttnn/ttnn/operations/examples/master.md` (Part 1
`dram_saturation` / `noc_placement`; Part 2 A3, B8, B10). No SUPPORTED change.

**Verifier notes**: A3+B10 is a cross-core **restructure** (a ⭐⭐⭐-tier lever), so it is a whole phase —
do not pack unrelated cheap knobs in beside it. Per master.md **B0**, both A3/B10 and B8 add fixed
per-core setup, so each must be counterfactualed on the *smallest* regime it will run in (`[1,1,32,64]`,
which Phase 0 measured as overhead-bound at a ~660 ns sync floor) as well as on the big shapes — a lever
that wins on (a) and regresses (d) must be gated on work-per-core, not applied globally. Also fold the
sharded shapes Refinement 1/2 introduced into the cumulative bench set here and **re-target** their
ceiling (a local-shard side is L1 loopback, not DRAM — the DRAM floor does not describe it); measure and
record them, but do not gate this phase on them. `tt_npe.sh` is **absent from this checkout**, so the
prompt's tt-npe pin cannot be produced — record the `/perf-ceiling-dm` bracket + the device measurement
and say so explicitly, rather than silently omitting it.

**Done when**: measured device kernel ns improves on `[1,1,2048,2048]` and `[1,1,32,16384]` (bf16 and
fp32) with the achieved-vs-target ratio recorded per shape and moving up from 0.63 / 0.55; every landed
lever passes its Mode-C used-optimization audit (predicted → measured → keep/drop, including a
smallest-regime check per B0) and is written into `lever_ledger.json` + `changelog.md`; the golden suite
is still green; and no regression across the config-spanning guard set (one representative per distinct
kernel path × placement × dtype: aligned interleaved DRAM→DRAM, padded (R_PAD reader), L1→L1, sharded
same-spec, sharded crossover, bf16 + fp32, depth-1 + depth-2, single-core).

---

### [ ] Refinement 4 — Integer dtype family, rank 0, and the two padding EXCLUSIONS

**Goal**: three bundled numeric-surface items.
(1) `dtype`/`output_dtype += [uint32, uint8]` (plus `uint16` / `int32`, which
`eval/golden_tests/tilize/test_regression.py` exercises and which currently accounts for 10 of its 12
failures). `uint8` is **not** just another width: an 8-bit datum needs the standard **per-face** row dim
rather than the full-tile row dim the 16/32-bit formats use, or the tile comes out *strided* (every other
row zero — shape-correct, value-wrong), and a narrow (<64 B) stick needs the alignment-aware reader.
(2) `rank += [0]` — a scalar padded out to a single tile (reachable only through the pad path).
(3) Delete both current `EXCLUSIONS`: `bfloat8_b` output × `pad_mode ∈ {auto, explicit}` (the block-float
shared exponent is defined over the 16×16 face structure, and the fill is materialized pre-pack), and
`bfloat16 → float32` × `pad_value ∈ {positive, negative}` (the fill is packed in the *input* element
format, so an inexact-in-bf16 fill lands bf16-rounded in an fp32 output — the fix is a second fill word
in the OUTPUT format applied after the cast, keeping the input-format fill for the no-cast case).

**Implementation skill**: /numeric-formats-metal

**Verifier notes**: cheapest tier (precision knobs / dtype lists over an existing path), hence after the
structural work — an integer dtype added *before* Refinement 1–2 would have to be re-extended over the
sharded kernels afterwards. Two hard rules from `eval/prompts/tilize.txt` **re-arm** with this refinement
and must be honoured, not rediscovered: the fill word is packed in the **input** element format (already
true in `_pack_pad_word`, and item 3 must not break it — add a second word, do not repurpose the first),
and sub-word fills must be **replicated across the 32-bit store word** (already true in
`fill_l1_with_val<elem_bytes>`, which already handles `elem_bytes == 1` — verify it, extend if not).
`uint8` correctness must be eyeballed for the every-other-row-zero signature: the identity is exact for
integers, so compare **exactly** and do not lean on PCC.

**Done when**: the `uint32` / `uint8` golden cells pass with an exact identity (and `test_regression.py`'s
`uint16` / `int32` cases pass); the rank-0 scalar cell's data region is the single input value with every
other position the fill; `EXCLUSIONS` is empty and the 30 cells it was gating pass; no regression on any
prior bench.

---

### [ ] Refinement 5 — Tile geometry: tiny tiles and (arch-gated) retile

**Goal**: `tile_height += [16, 8, 4, 2, 1]` for a ROW_MAJOR input (the tile stays 32 wide; the packer's
face geometry changes, so this is a distinct LLK path, not a reshape), interleaved **and** sharded; then
the retile path — `in_layout += [ttnn.TILE_LAYOUT]` and `in_tile_height += [32, 16, 8, 4, 2, 1]` (keeping
`"none"`) — where an already-tiled input is re-tiled to a *different* tile height and the reader walks
faces rather than sticks. Note the host-side gap this exposes: `tilize()` currently allocates the output
with `ttnn.allocate_tensor_on_device(..., ttnn.TILE_LAYOUT, ...)` and therefore the **default 32×32
tile** — the requested `tile` has to be threaded into the output tensor spec as well as into the CB
`TileDescriptor` and `derive_blocking()`.

**Implementation skill**: /memory-layouts

**Verifier notes**: `in_layout` is a genuine `Layout` axis (ROW_MAJOR vs TILE) with an in-kernel
data-access change, which is what the skill covers; the tiny-tile *face geometry* is LLK-level and beyond
it, so the reader/packer detail above stands as the spec. Ordered last of the generality work: it is a
tile-geometry axis that will have to be extended over whatever Refinements 1–2 build, so it must land
*after* them, not before. Two rules re-arm here: **a tiny tile redefines H-alignment** — the multiple is
the requested `tile_height`, not 32 (`tag_alignment` already measures it that way, so a kernel that
hardcodes 32 will mis-gate its own cells), and **retile and padding are mutually exclusive** (a TILE
input is tile-aligned by construction; a call passing both must be refused — `feature_spec.INVALID`
already prunes those cells, so the refusal is belt-and-braces for direct callers).
**Retile is Blackhole-only**: on Wormhole those 24 golden cells must *skip*, not fail — the harness's
`helpers.skip_if_retile_unsupported` already does this, and a skip there is the correct outcome, never
"missing support".

**Done when**: the tiny-tile golden cells pass at every height including the degenerate 1×32 (one stick
per tile), interleaved and sharded, with the alignment axis measured against the requested height; the
retile cells pass on Blackhole and skip cleanly on Wormhole; a pad + TILE-input call is refused; no
regression on any prior bench.

---

### [ ] Refinement 6 — Perf completeness audit (run-closing)

**Type**: perf

**Goal**: the run-closing `/perf-ceiling-dm` **Mode D** completeness audit over the FULL lever list in
`ttnn/ttnn/operations/examples/master.md` (Part 1 examples + Part 2 propositions), now that every
capability and perf phase has landed. Account for **every lever not applied** — tagged
not-applicable / deferred / measured-no-payoff / missed — with a counterfactual predicted delta for the
ones that are not clearly not-applicable, and rank the real remaining opportunities. Phase 0 already
carries the open list to start from: **B13** stateful writer (must be *swept* across transaction size,
not argued), **B6** one-packet fast path (pulls against B5 — only a sweep can price the pair), **F24**
`bfp8_pack_precise` (bf8b is emitted but has no bench arm), **D18/D19/D21** (applied, off-arms never
built), **A2** launch-only-on-cores-that-hold-data (becomes live once sharding exists), **L4**
`split_reader` (live on the sharded-output path), **E22** (whole-model, out of scope). No SUPPORTED
change, no new capability.

**Verifier notes**: this is the one queue entry that exists to answer "what perf did we leave on the
table, and why" rather than to unlock a cell — file exactly one, ordered last. It must audit the
*finished* op, so it runs after Refinement 5. Anything **missed** or **deferred** with a large predicted
delta is surfaced as a concrete follow-up for the next run, never silently dropped.

**Done when**: `changelog.md` carries a completeness ledger covering every master.md lever
(`lever → status → predicted delta if applied → reason`) plus a ranked list of remaining opportunities;
`python3 -m eval.verify_levers ttnn/ttnn/operations/tilize/lever_ledger.json --bench
tests/ttnn/unit_tests/operations/tilize/_bench_tilize.py` is clean; no regression on any prior bench;
golden suite green.
