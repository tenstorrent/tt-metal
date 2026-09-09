# Operation Requirements: tilize

## Definition

- **Formula**: `out[i] = in[i]` — a pure re-laying of bytes. Every element keeps its
  value and its logical position; only the address it occupies changes. ROW_MAJOR
  storage becomes TILE storage: a sequence of `tile_h × 32` tiles, each built from
  four 16×16 faces (TL, TR, BL, BR) in row-major order, elements row-major within a
  face. Where a pad is requested, positions outside the input's logical extent hold
  exactly the fill value. Where `dtype=` names a different output format, the value is
  converted at pack time (value-preserving, within a dtype family).

  The work is a 2-D grid of output tiles, `R × C`, with the leading dims folding into
  `R`:

      R = prod(shape[:-2]) * ceil(shape[-2] / tile_height)
      C = ceil(shape[-1] / 32)

- **PyTorch Reference** (standalone — the logical view of the output is the input,
  which is why the layout / dtype / shape assertions carry the weight):

```python
def torch_tilize(x: torch.Tensor,
                 output_padded_shape=None,
                 pad_value=None) -> torch.Tensor:
    """Reference for tilize's LOGICAL result. tilize is value- and
    position-preserving, so unpadded it is the identity; a padded call places the
    input in the leading region of the padded shape and fills the rest."""
    if output_padded_shape is None and pad_value is None:
        return x
    shape = list(x.shape)
    if len(shape) < 2:                       # rank 0/1: the pad SYNTHESIZES tile dims
        x = x.reshape([1] * (2 - len(shape)) + shape)
        shape = list(x.shape)
    target = (list(output_padded_shape) if output_padded_shape is not None
              else shape[:-2] + [((shape[-2] + 31) // 32) * 32,
                                 ((shape[-1] + 31) // 32) * 32])
    out = torch.full(target, float(pad_value or 0), dtype=x.dtype)
    out[tuple(slice(0, d) for d in shape)] = x
    return out
```

- **Import Path**: `from ttnn.operations.tilize import tilize`
  (also bound as `ttnn.tilize` — the public name this op occupies)

- **Function Signature**:

```python
tilize(
    input_tensor: ttnn.Tensor,                       # ROW_MAJOR (or TILE, to re-tile)
    memory_config: ttnn.MemoryConfig | None = None,  # output placement (default: input's)
    *,
    dtype: ttnn.DataType | None = None,              # output dtype (default: input's)
    low_l1: bool = False,                            # per-core L1 independent of dims
    output_padded_shape: list[int] | ttnn.Shape | None = None,  # explicit pad target
    pad_value: float | int | None = None,            # fill for padded positions
    tile: ttnn.Tile | None = None,                   # output tile geometry (default 32x32)
) -> ttnn.Tensor
```

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED output_dtype**: [bfloat16]
- **SUPPORTED layout-ish axes**: `in_tile_height=["none"]` (ROW_MAJOR input only), `tile_height=[32]`
- **SUPPORTED shape-derived axes**: `alignment=["tile_aligned"]`, `rank=[2,3,4,5,6]`, `tile_grid` = **all five** (`single_tile`, `small`, `tall_narrow`, `short_wide`, `square_large`)
- **SUPPORTED placement axes**: `shard_api=["none"]`, `out_scheme=["interleaved"]`, `orientation=["none"]`, `buffer` = **all four** interleaved transitions
- **SUPPORTED op-specific axes**: `low_l1=[False, True]`, `pad_mode=["none"]`, `pad_value=["none"]`
- **EXCLUSIONS**: none
- **Cores**: multi — a 2-D block grid `num_row_groups × num_w_chunks` linearized into `split_work_to_cores(..., row_wise=True)`; **64/64 cores measured on both `[1,1,32,16384]` (R=1) and `[1,1,16384,32]` (C=1)**
- **Compute config**: `fp32_dest_acc_en` keyed on the input dtype (always `False` at bf16); `dst_full_sync_en` left off so `fast_tilize` stays eligible
- **Golden baseline**: **249 / 4548 cells passing** (per `verifier_report.json`), `supported_fail` / `xpass_drift` / `xfail_wrong_mode` all **0**
- **Verifier fixes folded into Phase 0**: validate() check ordering (83 `xfail_wrong_mode` → 0), rank-0/1 pad-target rule, `_spec_of` nd-vs-legacy discriminator, `ttnn.tilize` public alias, and three measured SUPPORTED promotions (`rank`, `low_l1`, `buffer`). See `verification_report.md`.

---

### [x] Refinement 1 — Sharded and L1 placement: `shard_api`, `out_scheme`, `orientation`

**Goal**: add `"legacy_2d"` and `"nd"` to `SUPPORTED["shard_api"]`, `HEIGHT_SHARDED` /
`WIDTH_SHARDED` / `BLOCK_SHARDED` / `"nd"` to `SUPPORTED["out_scheme"]`, and
`ROW_MAJOR` / `COL_MAJOR` to `SUPPORTED["orientation"]` — natively, in the kernels'
data access. This is the design's `grid2d_sharded` deferred regime: the shard fixes
the core assignment and the per-core extent, so `num_row_groups` / `num_w_chunks` are
read **off the shard spec** instead of solved, and the sharded side's CB becomes
**zero-copy over the shard buffer** via `ttnn.cb_descriptor_from_sharded_tensor`
(`ttnn/cpp/ttnn-nanobind/program_descriptors.cpp:517-540`). Everything else — the
block operations, the compute call, the writer's batching — is unchanged. Unlocks the
`sharded_legacy_2d` (7), `sharded_nd` (3) and `short_wide_width_sharded` golden cases,
plus the 97 `test_golden_main_tests.py` cases currently refused on `shard_api`.

**Verifier notes**: no implementation skill covers `memory_layout` / shard placement
yet — `/memory-layouts` is ROW_MAJOR↔TILE, a different axis. Do not attach it. The
mechanics you need are named here instead:

* **This op has no dependent axis, so every same-spec shard is LOCAL.** Each output
  tile depends only on its own 32-element column slice of its own `tile_h` sticks —
  nothing spans blocks. So a HEIGHT-sharded output is exactly a `row_group` block, a
  WIDTH-sharded output exactly a `w_chunk` block, and a BLOCK-sharded output exactly
  the 2-D block the op already builds. **There is no cross-core combine to design, no
  mcast, no semaphore.** That is why this is one refinement and not a scheme-change
  standalone: the topology already matches.
* **The native path is what "sharded" means here — do not settle for the accessor.**
  A core's own local shard is already resident in its L1; it *is* the per-core block.
  Consume it through the zero-copy CB placement, never re-read it through a
  `TensorAccessor` as if it were interleaved. An accessor read of a core's own shard
  would pass every golden cell while meaning the axis was never implemented. Check the
  dataflow, not the test colour. (`TensorAccessor` still owns the interleaved leg and
  the genuinely non-local **cross-spec** gather — `cross_spec_height_in_width_out` and
  the `dram_sharded_out` case are where a core's output shard needs bytes from another
  core's input shard, and that one is a real remote read.)
* **`eval/golden_tests/tilize/axes.py:_spec_of` has a live nd-vs-legacy_2d bug** that
  the verification pass fixed in the op's copy but could not fix in the gate: a live
  tensor's `memory_config()` populates *both* `shard_spec` and `nd_shard_spec`
  whichever API allocated it, so "has an nd spec" tags every sharded call `nd`. You
  will see 84 `shard_api` mismatches in `merge_axes`' output that are **not yours** —
  the xfail decoration comes from the declared taggers and is unaffected. The op's
  `_spec_of` already carries the correct discriminator; mirror it if you touch the
  harness under `/golden-tests`.
* **Fold in the L1-budget correction while you are here** (an L1 ledger finding, filed
  into this refinement rather than standing alone per the filing rule): `derive_plan`
  sizes the CBs against `ttnn.get_max_worker_l1_unreserved_size()`, which does not
  account for the tensors themselves when `buffer` is `l1_to_l1`/`dram_to_l1` — and a
  sharded tensor is L1-resident by definition. No cell in the suite trips it today
  (the grid, not L1, is the binding constraint on every one of these shapes), but a
  sharded refinement is exactly where it starts to bite. Subtract the resident shard
  bytes from the budget.
* **Ordering**: first because it is the most structurally invasive of the four
  generality refinements — it changes where the split comes from. Land it on the
  Phase 0 test surface (bf16, one dtype pair) before Refinement 5 multiplies every
  scenario by 18 dtype pairs.

**Done when**: the three axes carry their TARGET values in SUPPORTED; the
`sharded_legacy_2d`, `sharded_nd` and `short_wide_width_sharded` golden cells pass;
the sharded side is consumed through a CB placed on the shard buffer (verifiable in
the `ProgramDescriptor`, not just in the test result); the golden suite's three loud
categories stay at 0; and Phase 0's 249 cells still pass.

**Outcome**: DONE. All three axes carry their TARGET values; all 11 targeted golden
cells pass (`test_golden.py`: 32 passed, 0 failed, 0 XPASS — 21 Phase 0 + 11 new);
`test_golden_main_tests.py` went 26 → 47 passing with **zero** non-refusal failures
left. The block grid is read off the shard spec (`shard_partition()`), the shard's
linear index IS the existing `block_id`, and the native side's CB is placed on the
shard buffer — asserted against the `ProgramDescriptor` in
`test_tilize_sharded.py::test_sharded_side_is_zero_copy`, not inferred from values.
The native-output path emits no writer kernel at all.

Two things were found that a sharded call exposes and an interleaved one cannot:
`R`/`C` must come from the OUTPUT's padded shape (a ROW_MAJOR tensor's padded shape
rounds to its PAGE width, which a width-cutting shard sets), and
`read_sticks_for_tilize` cannot address a source whose row spans several pages — a
strided reader branch closes that, with the helper gap recorded (a
`page_stride_per_row` parameter would close it upstream). The l1_ledger budget
correction is folded in: `get_max_worker_l1_unreserved_size()` is the arena, not the
free space, so both operands' resident L1 is now subtracted.

Not implemented, and correct through the accessor legs rather than natively: ND
shards whose leading-dim extent exceeds 1 (`[2,64,64]` over `[4,128,128]`) are not
2-D-expressible blocks and so cannot drive the grid. `low_l1` is inert on the
shard-driven plan — the shard, not `W_CAP`, fixes the block extent.

---

### [x] Refinement 2 — Padding: `pad_mode`, `pad_value`, `alignment`, ranks 0 and 1

**Goal**: add `"auto"` and `"explicit"` to `SUPPORTED["pad_mode"]`, `"zero"` /
`"positive"` / `"negative"` to `SUPPORTED["pad_value"]`, `"w_non_aligned"` /
`"h_non_aligned"` / `"hw_non_aligned"` to `SUPPORTED["alignment"]`, and `0` / `1` to
`SUPPORTED["rank"]` — natively, with the fill produced **in the kernel**. This is the
design's `grid2d_padded` deferred regime, and the design's own framing is the plan:
the fill is an *additive* step on the block that already exists. The reader memsets
the pad region of the L1 sub-block (the W tail inside `padded_row_bytes`, and the H
tail rows) before/around the stick reads; the block grid, the core assignment, the
CBs and the compute call are unchanged. Unlocks `padding_auto` (7),
`padding_explicit` (4), `padding_crossed` (3), the four padded members of
`work_geometry`, ranks 0/1 in `interleaved`, and the 13 `test_golden_main_tests.py` +
`test_regression.py` cases refused on `pad_mode` / `rank=0`.

**Implementation skill**: /memory-layouts

**Verifier notes**: the skill's non-aligned rule (last-tile H/W zero-pad / mask done
in the reader or compute) is the relevant part; you are **not** adding a ROW_MAJOR leg
— the input is already ROW_MAJOR and the tilize is already in-kernel. Four things the
skill cannot see:

* **The H tail breaks a contiguity invariant the reader currently relies on.** One
  `read_sticks_for_tilize` call spans a contiguous stick run
  (`start_page + block_row*tile_h + row`), which is only valid across an image
  boundary when `H % tile_h == 0` — then `start_stick = row_start * tile_h` is exact.
  With an H tail it is not, so the block's reader call must be **segmented per image**
  (`op_design.md` → Mechanism caps, last-but-two row). `[8,1,249,2048]` is the case
  that catches this; `[1,1,1,50304]` and `[1,1,1,2048]` are the H=1 degenerate forms.
* **Rank 0 and 1 are pad-only and the pad SYNTHESIZES their tile dims** (rank 0 →
  `[H,W]`, rank 1 → `[H,W]`), which is why `tag_alignment` reports `hw_non_aligned`
  there. `validate()`'s `_check_pad_target` already left-pads the input shape to 2
  before comparing against `output_padded_shape`, so the request-shape side is done —
  what is missing is the reader arithmetic.
* **`pad_mode="explicit"` may exceed the tile-round** (`padding_explicit` carries one
  such case: whole pad *tiles* past the data). Those tiles have no input bytes at all
  and must be produced from the fill alone.
* **A negative fill on an integer dtype goes through a signed→unsigned bit_cast and
  must not truncate** (the prompt's MUST). That rule only arms once Refinement 5 adds
  the integer dtypes — the fill path you write here should already be width-correct so
  it does not have to be revisited. `test_regression.py::test_pad_value_extremes`
  (±3.4e38 at fp32) is the float end of the same requirement.
* **Ordering**: second — structurally lighter than the shard refinement but heavier
  than the tile-geometry and dtype work, and still ahead of Refinement 5 for the same
  small-test-surface reason. No hard dependency on Refinement 1, but `padding_crossed`
  crosses padding with a sharded output and with `low_l1`, so running after 1 means
  those cells land for free.

**Done when**: the four axes carry their TARGET values in SUPPORTED; every padded
position holds exactly the fill value (checked with
`to_torch_with_padded_shape`, not just the logical view); the logical shape of a
padded output is **unchanged** (only the padded shape grows — promoting the logical
shape is the named bug); the `padding_auto` / `padding_explicit` / `padding_crossed`
golden groups pass; the three loud categories stay at 0; Phase 0's 249 cells and
Refinement 1's cells still pass.

**Outcome**: DONE. All four axes carry their TARGET values (`pad_mode` +auto/explicit,
`pad_value` +zero/positive/negative, `alignment` all four, `rank` +0/1) and the whole
graded matrix is green: `test_golden.py` **53 passed, 0 failed, 0 XPASS** (from 32 after
Refinement 1) — all 7 `padding_auto`, 4 `padding_explicit`, 3 `padding_crossed`, the four
padded `work_geometry` members, rank 0 and rank 1, plus both padded LOOSE_CASES
(`[1,1,1,50304]` at C=1572 and `[8,1,49,2048]`). `test_golden_main_tests.py` went
47 → **127** passing with every remaining failure an honest `dtype` refusal
(Refinement 5); `test_translated.py` 422 → **707**.

The fill is produced in the kernel and the block schedule is untouched: the grid, the
core assignment, the two streaming CBs, the compute call and the writer are byte-identical,
and the padded path is one compile-time reader branch plus **one** extra CB —
`cb_pad_row`, a single block ROW (not tile) of pre-filled bytes, `<= 0.8%` of the
per-core footprint on every measured shape. A fully padded row costs one local L1→L1
DM transfer out of it, so a padded call crosses DRAM with the input's LOGICAL bytes and
`[1,1,1,2048]` (H=1) never DRAM-reads 31 of every 32 rows. `pad_active` is derived from
the two shapes rather than from the argument, which is what keeps an already-aligned
call on the Phase 0 branch even when `pad_value=` is passed.

Four things a padded call exposed that an unpadded one cannot:
1. **The H tail breaks the reader's contiguous-stick-run invariant** exactly as the notes
   predicted — source rows restart at every image boundary — so the padded block read is
   SEGMENTED per image. `[8,1,249,2048]` is the witness. Recorded helper gap: a
   `rows_per_segment` + `pad_value` pair alongside `byte_offset_within_page` would close
   both the segmentation and the fill in `read_sticks_for_tilize`.
2. **An explicit target beyond the tile round is not expressible from (logical shape, tile)**
   — a `TensorSpec`'s default alignment caps the padded shape AT the round. Added one
   additive nanobind binding, `ttnn.TensorSpec.with_padded_shape(logical, padded, ...)`, and
   used it ONLY where the target exceeds the round, so every already-verified call keeps its
   Phase 0 / Refinement 1 constructor. Ranks 0/1 needed nothing: a TILE spec's default
   alignment is already rank 2, so `[]` → `[32,32]` and `[64]` → `[32,64]` fall out.
3. **A sharded `memory_config` may name only the FAMILY** (`MemoryConfig(WIDTH_SHARDED, L1)`
   with no ShardSpec — a contract case in `test_translated.py`). `_output_tensor_spec` used
   to `TypeError` on it, latent since Refinement 1 and newly reachable here; it now applies
   `TensorSpec`'s own `height_sharded`/`width_sharded`/`block_sharded` cut over the compute
   grid, derived off the PADDED spec so the shard's height is the padded height. +5 cases.
4. **A resident input shard cannot hold the fill**, so `pad_active` clears `input_native` and
   a padded call reads its input through the accessor. The OUTPUT side stays native — the
   packer writes whole tiles, pad positions included, into the output shard.

Deliberate divergence, recorded rather than resolved: 50 `test_translated.py` cases assert
that a non-tile-aligned input with **no** padding argument is zero-padded by default. The
immutable acceptance test `test_tilize.py::test_tilize_rejects_unaligned_input_without_padding`
("DO NOT MODIFY") asserts the opposite — padding is opt-in — and `feature_spec.TARGET`'s own
`pad_mode` comment agrees ("none — no padding argument; an unaligned input is refused"). The
acceptance test and the registry declaration win; those 50 were failing before this
refinement too (on the `alignment` gate), and now fail with the intended `ValueError`.

---

### [x] Refinement 3 — Speed up the perf-flagged attention profile

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES` flags **`[1,1,32,16384]`** (bf16 → bf16,
interleaved DRAM→DRAM, rank 4, tile-aligned, 32×32 tile) with the `attention:` note
as the **mandatory** perf target — DeepSeek-V3 MLA's `wo_tilize` output projection,
`R=1 × C=512`, the decode-phase geometry where a row-only split has one row to cut.
Optimize **that exact config** using the relevant patterns in
`ttnn/ttnn/operations/examples/master.md`. No SUPPORTED change.

Where it stands today (measured by the verification pass, `--profile`, 8×8 Wormhole,
fresh cache, two dispatches): **13106 / 14133 ns at 64/64 cores**, block `1 × 8` tiles,
512 B per stick read, 64 KiB L1/core — ≈**160 GB/s** over the 2 MiB moved.
`double_buffer/report.md` puts an untuned 64-core DRAM→DRAM stream at 190.8 GB/s
(≈ this part's peak), so there is roughly **1.19×** of headroom. Occupancy is already
full, so the lever is **not** work distribution — it is transaction shape and issue
cost. The catalog entries whose *situation* matches: `double_buffer` (bytes in flight
— co-tune `INPUT_DEPTH_ROWS` × `WRITE_BATCH_MIN_TILES` at this block width, where the
existing sweeps were taken on other geometries and `write_rows_per_barrier` is inert
at `bw=8`), `tile_reorder` (coalescing on a DRAM-bound move), `noc_placement` (the
placement × NoC matrix — `row_wise=True` and NoC0/NoC1 are already taken, the
diagonal variant is not), and `compute_block_size`'s **second** lever: the tilize
helper reconfigs unpack+pack data formats at every block boundary by default, and on
the no-cast diagonal (`dtype == output_dtype`, which is this target) that reconfig is
wasted MMIO — `ReconfigureRegisterDatatypeMode::NoReconfigure` is a one-token change
worth up to 1.19× where transitions are frequent. Several ⭐⭐ T2 levers, so try more
than one in this phase.

**Done when**: measured device-ns improves on `[1,1,32,16384]` at bf16→bf16 (the
flagged config exactly — not a proxy), with the core count reported alongside the
duration and still 64/64 (a number taken on a fraction of the grid describes the
split, not the kernel); precision is unchanged (bit-identity — this op does no
arithmetic, so any deviation is a bug, not a budget); the golden suite is green with
the three loud categories at 0; and no regression across the config-spanning guard
set — one representative per distinct kernel path × placement, which for this op means
at minimum `[1,1,2048,64]` (`grid2d_full_width`, `num_w_chunks == 1`),
`[1,1,32,2048]` (`grid2d_width_chunked`), `[1,1,2048,2048]` (`bw = 64`, the widest
block), `[1,1,16384,32]` (`bw = 1`, where `write_rows_per_barrier` is the whole knob),
plus one sharded and one padded representative if Refinements 1 and 2 have landed.

**Outcome**: **The flagged config is data-movement-saturated; the headroom estimate
in the Goal was measured against the wrong regime.** Ablated whole (8x8 Wormhole,
64/64 cores, `[1,1,32,16384]`): all payloads stubbed **660 ns**, + compute **1515**,
+ reads **8241**, + writes **9459**, full op **13247** — so reads (6726 ns, 156 GB/s)
and writes (7944 ns, 132 GB/s) are BALANCED and already overlap by 2938 of the 14670
ns they would take in series. Calibrated on the same box: a production `ttnn.clone`
moves the same 2 MiB in **15268 ns (137 GB/s)** and 16 MiB in **87375 ns
(192 GB/s)`** — 192 GB/s (the `double_buffer` figure the Goal quoted) is the
asymptotic ceiling and is only reachable at >= 16 MiB, so tilize's 162 GB/s at 2 MiB
is **1.15x faster than a plain tiled copy of the same size**, not 1.19x short of
anything. Levers landed and measured: (1) **`PIPELINE_WAVES_PER_CORE` x
`MIN_BLOCK_ROW_BYTES`** — a core with one tile-row-tall block cannot overlap its own
read against its own write, so the column cut now buys up to 4 pipeline waves while
the read stays >= 512 B; **1.08x on `[1,1,2048,2048]`** (94481 -> ~87400 ns, and
512 KB -> 128 KB of L1), **1.05x on `[1,1,32,32768]`**, 1.04x on `[1,1,1024,1024]`,
and correctly INERT on the flagged shape because its second wave would cost a 256 B
read (measured: 512 B 13759 ns vs 256 B 13620 ns vs 128 B 15204 ns — the pair either
side of the floor is a tie, the one below it is a 1.12x loss). (2) tilize-helper
**`NoReconfigure`** + **init/uninit amortized across the core's block loop**, both
measured flat here (compute is 855 of 13247 ns) and both kept as live knobs; the
amortization is emission-gated on blocks-per-core > 1 after its dead instantiations
cost 4% of the wall on a 4 us kernel. (3) **one-packet NoC issue path** on the writer
— flat, kept. Tried and rejected: staggering the read row order per core to
de-conflict DRAM banks (13708 vs 12587, no effect — 32 outstanding reads already
spread), `WRITE_BATCH_MIN_TILES` 4/8/16 at every wave setting (flat), CB depths
{2,4}x{2,4} (flat). **What is left and why I did not take it**: the only way to
enlarge the flagged shape's 512 B read at 64/64 cores is to have several cores share
one wider DRAM read and redistribute it core-to-core (mcast or peer unicast), which
is a topology change, adds 0.75 MiB of cross-core NoC, and is chasing at most the
~1.13x gap to a ceiling this op is already above. `[1,1,16384,32]` (~100 GB/s at a
64 B read) is the genuinely off-ceiling geometry and it is Refinement 6's subject.

---

### [x] Refinement 4 — Tile geometry: tiny tiles and the retile path

**Goal**: add `16, 8, 4, 2, 1` to `SUPPORTED["tile_height"]` and `32, 16, 8, 4, 2, 1`
to `SUPPORTED["in_tile_height"]` — both natively on device. Two regimes the design
defers, taken together because they share the `tile=` surface:

* `grid2d_tiny_tile` (ROW_MAJOR in, sub-32 tile out) is nearly a knob turn: `tile_h < 32`
  changes only the `TileDescriptor` on both CBs, `read_sticks_for_tilize` reads
  `tile_h` from `unpack_tile_r_dim[cb_id]`, and the block grid is unchanged. It does
  drop off `can_use_fast_tilize` (which requires 32×32), so expect the regular
  `tilize_block` LLK path.
* `retile` (TILE in at one height → another) is the genuinely distinct one: the input
  is already tiled, so the reader walks **faces, not sticks**, and
  `read_sticks_for_tilize` cannot express it (it is stick-indexed by construction —
  `accessor.get_noc_addr(start_page + block_row + row, ...)`). This is a new reader
  block operation.

Unlocks the `tile_geometry_tiny` (3) and `tile_geometry_retile` (6) golden groups.

**Implementation skill**: /memory-layouts

**Verifier notes**:

* **Re-tiling MUST NOT be done by untilizing to ROW_MAJOR and tilizing again.** That
  is the host-side workaround the prompt forbids wearing a kernel hat; the design
  ranks it `rejected` at 2× the minimum DRAM traffic in both directions, and nothing
  built on it survives when the real face-walking reader lands. Likewise no
  `ttnn.to_layout` / `ttnn.untilize` wrapper at the entry point. If the face-walking
  reader turns out to be more than a focused pass, ship `[~]` partial with the
  **tiny-tile half only** (which is the near-knob-turn) and file `Refinement 4b` for
  the retile reader — do not substitute a round-trip.
* **`retile` is arch-gated to Blackhole** (`helpers.skip_if_retile_unsupported`; the
  reference carries `@skip_for_wormhole_b0("LLK for tiny tiles not fully supported on
  Wormhole B0")`). On the Wormhole box this verification pass ran on, all 108 retile
  cells and 196 fp8 cells appear as `xfail_other` **skips**, not failures — so on
  Wormhole you can build the reader but you cannot green the cells. Plain tiny-tile
  (ROW_MAJOR in, sub-32 out) is **not** gated and runs everywhere, which is the other
  reason the two halves are bundled: one of them is testable here.
* **Watch the interaction with `tag_alignment`.** H is measured against the **output**
  tile height, not a literal 32 — `H=48` with `tile_height=16` is three whole tile-rows
  with no H tail at all. So changing `tile_height` re-partitions the `alignment` axis,
  and a cell that was `h_non_aligned` at 32 can become `tile_aligned` at 16. If
  Refinement 2 has landed, re-check its H-tail arithmetic against a tiny tile.
* **Ordering**: after the perf slot and after the two heavier structural refinements,
  before the dtype multiplication. `tb_in` and `tb_out` both scale with `tile_h`, so
  the L1 ledger's totals shrink here rather than grow — no new budget pressure.

**Done when**: both axes carry their TARGET values in SUPPORTED; the
`tile_geometry_tiny` cells pass on this box; the retile reader walks faces on device
(verifiable in the kernel, and green on Blackhole where the gate allows); no
manipulation-op wrapper appears at the entry point; the three loud categories stay
at 0; and all prior phases still pass.

**Outcome**: both axes carry their full TARGET values —
`tile_height = [1,2,4,8,16,32]` and `in_tile_height = ["none",1,2,4,8,16,32]`.

* **Tiny tile was a pure knob turn, as forecast.** `tile_h` was already a plan
  quantity everywhere (`in_page_bytes`, `rows_per_image`, both CBs'
  `TileDescriptor`, the reader's stick count, the pad tail arithmetic), so
  widening `SUPPORTED["tile_height"]` to `LEGAL_TILE_HEIGHTS` was the entire
  diff — zero kernel changes. It does drop off `can_use_fast_tilize` (which
  needs 32x32 output tiles) onto the regular `tilize_init`/`tilize_block` path,
  which is per-tile through DEST and so has no width cap of its own. The
  `alignment` re-partition the verifier flagged is real and works: `H=48` at
  `tile_h=16` is three whole tile-rows and takes the unpadded path, and the
  padded H/W tails are stated in units of `tile_h` rather than a literal 32
  (`test_tiny_tile_realigns_h`, `test_tiny_tile_padded`). 3/3
  `tile_geometry_tiny` golden cells pass.
* **Retile landed as a real face-walking reader, on the FULL height cartesian.**
  `retile_block` is a new reader block operation built on one derived quantity,
  `retile_copy_unit(in_tile_h, out_tile_h)`: the largest byte run contiguous in
  BOTH tile layouts — a face-PAIR slab (`min(h_in,h_out)` rows x 32 cols) when
  the two face heights `min(h,16)` agree, a face FRAGMENT (`min(h_in,h_out)`
  rows x 16 cols) when they do not. The reader issues each run as one
  `noc_async_read` from the source tile page's byte offset into the destination
  tile's byte offset, so **output tiles are assembled in place in
  `cb_output_tiles`** and the program carries **no compute kernel at all**. One
  DRAM crossing each way — the named-boundary minimum, versus 2 for the
  untilize-and-retilize round trip the design ranks `rejected`. No
  `to_layout`/`untilize` wrapper anywhere; the entry point is unchanged.
* **It is green on WORMHOLE, which the verifier note did not expect.** The
  golden arch gate is on the tiny-tile *LLK*; a pure NoC face walk never touches
  an LLK, so all 36 legal `(in, out)` height pairs are bit-exact on this box
  (`test_retile_all_height_pairs`), together with grid-scale geometries, all
  three buffer transitions, ranks 2-5, and the case where H is a multiple of the
  output tile height but not the input's (which is why the reader keeps a
  per-image split). The golden `tile_geometry_retile` cells still report as
  `skipped` here because `helpers.skip_if_retile_unsupported` fires before
  `validate()` — the coverage is in the unit suite instead.
* **Two retile crossings are in EXCLUSIONS, for structural reasons**: retile x
  sharded (the face walk addresses the source by interleaved TILE page index; a
  native zero-copy CB over a resident TILE shard would need the shard's own page
  map on both sides, and reading a core's own shard back through an accessor is
  the non-implementation this op refuses everywhere else) and retile x padding
  (the fill would have to land in output faces the walk never sources). Neither
  is reached by any golden case. Both are the honest next lever if a Blackhole
  box ever hosts a run.
* **L1 shrank on both halves, as the ordering note predicted.** `tb_in` and
  `tb_out` scale with `tile_h`, and on the retile path `cb_input_rows` collapses
  to a one-page stub (the reader never uses it), which drops
  `INPUT_DEPTH_ROWS * tb_in` out of the `W_FIT` denominator and roughly doubles
  the affordable column extent: `[1,1,2048,2048]` is 131072 B at ROW_MAJOR -> 32
  and 66560 B at retile 32 -> 16 with `bw` growing 16 -> 32. Nothing grew.

---

### [x] Refinement 5 — Numerical configurability: the full `dtype × output_dtype` cartesian

**Goal**: add `float32`, `fp8_e4m3`, `uint32`, `int32`, `uint16`, `uint8` to
`SUPPORTED["dtype"]` and `float32`, `bfloat8_b`, `bfloat4_b`, `uint32`, `int32`,
`uint16`, `uint8` to `SUPPORTED["output_dtype"]`, expose
`compute_kernel_config: ttnn.ComputeKernelConfig` on the entry point, and set the
intermediate-CB precision correctly (including `UnpackToDestFp32` tagging where it
applies). Cells that fail out of the box land in `EXCLUSIONS`, not in their own
refinement. This is the largest cell-count unlock in the queue by a wide margin: 18
legal `(dtype, output_dtype)` pairs survive `INVALID`'s 38 prunes, so it multiplies
**every** scenario in the suite.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**: probed on device by the verification pass, so you can start from
evidence rather than from a matrix. On `(1,1,64,128)`:

| pair | today | reading |
|---|---|---|
| `bf16 -> fp32` | exact | free |
| `uint32 -> uint32`, `int32 -> int32`, `uint16 -> uint16` | exact | free — a list widening |
| `bf16 -> bf8b` | `max_abs 0.031` | block-float quantization, expected; needs a tolerance, not a fix |
| `bf16 -> bf4b` | `max_abs 0.926` | the one genuinely lossy target in the set; `helpers._transition_tolerance` already carries the floors |
| `fp32 -> fp32` | **`max_abs 1.95e-3`** | **a correctness failure, not precision** — see below |
| `uint8 -> uint8` | **`max_abs 99` on data in `[0,100)`** | **broken, not merely unsupported** — see below |

* **fp32 output is a correctness trap, not a precision knob.** `can_use_fast_tilize`
  refuses fp32 output and the fallback truncates through tf32, which is why
  `fp32 -> fp32` is not bit-identical today. For a byte re-lay the golden oracle *is*
  bit-identity, so this cell needs all three of `Fp32Mode::Lossless`,
  `fp32_dest_acc_en=true`, and `UnpackToDestMode::UnpackToDestFp32` on `cb_input_rows`
  (`tilize_helpers.inl:115-127`) — any one missing and it fails. Note this is the one
  place where the helper's own "prefer Fast, the downstream FPU truncates anyway"
  advice is **wrong**: there is no downstream FPU op here, the tiled output *is* the
  product. Today `derive_plan` sets `fp32_dest_acc_en` from the **input** dtype alone;
  it must key on the output too.
* **`uint8` is an investigation, not a list widening.** Every other integer width is
  bit-exact, so suspect the helper's `elem_size = tile_size / tile_hw` derivation and
  the per-face vs full-tile dim at 1 byte/element — the exact failure mode
  `TARGET["dtype"]`'s comment warns about ("getting it wrong yields a strided tile
  rather than a wrong value"). If it does not survive, `{"dtype": ttnn.uint8}` and
  `{"output_dtype": ttnn.uint8}` go to `EXCLUSIONS` with the mechanism named — that is
  the skill's stated pattern and it is the right outcome, not a failure of the phase.
* **`fp8_e4m3` is Blackhole-only and input-only.** On Wormhole its 196 cells are
  `xfail_other` arch skips, so you cannot green them here. Also see the open
  `feature_spec` question in `verification_report.md` → INVALID audit: the cartesian
  currently generates `fp8_e4m3 × in_tile_height=<TILE>`, an fp8 tensor in TILE layout
  that by `TARGET`'s own comment has no TILE form. Raise it rather than working around
  it.
* **Check the widened output page against DEST width.** With `bf16 in -> fp32 out` the
  output CB becomes a `Float32` page under a 16-bit DEST. That is not a precision loss
  here (the DEST value is already the tensor's own bf16 value, widened losslessly at
  pack) but it doubles the bytes through the packer and in every read of that CB, and
  it doubles that CB's L1. Decide it deliberately and record it in `l1_ledger.md`'s
  page-format row rather than inheriting it.
* **Ordering: last of the generality refinements, deliberately.** It is the cheapest to
  build (it widens SUPPORTED lists and relaxes a `validate()` gate on an existing path
  — the skill's pass condition is *zero kernel changes*) and the largest in cells, but
  landing it first would force Refinements 1, 2 and 4 to be built and debugged across
  18 dtype pairs instead of one. Land the structure first, extend it once.

**Done when**: both dtype axes carry their TARGET values minus a named, mechanism-
justified `EXCLUSIONS` set; `fp32 -> fp32` and every integer no-cast diagonal are
**bit-identical**; the lossy casts hold `helpers._transition_tolerance`'s floors;
`test_regression.py`'s 10 cases pass; the three loud categories stay at 0; and every
prior phase still passes across the widened cartesian.

**Outcome**: DONE. `SUPPORTED` is 7x8 (both axes at full TARGET), and the golden
suite went **49 -> 679 passing** (13.9x), `test_regression.py` 10/10, unit directory
606/606. `fp32 -> fp32` is bit-identical (was `max_abs 1.95e-3`) via
`Fp32Mode::Lossless` + `fp32_dest_acc_en` + `UnpackToDestFp32` on `cb_input_rows`,
and all ten exactly-representable pairs are `torch.equal`.

`uint8` was NOT excluded. The all-zeros failure was traced to a Wormhole B0 LLK
defect — `_llk_math_hw_configure_` writes srcA's and srcB's ALU format fields as one
word under the union of two 4-bit masks without `masked_data_format()`, so
`UInt8` (30) spills bit 4 into srcB and the UInt8 datacopy MOP (ELWADD, which reads
srcB) zeroes every datum. One `reconfig_data_format_srcb` call after
`compute_kernel_hw_startup` rewrites that field alone and `uint8` is bit-exact. The
shared LLK is deliberately left unpatched: it is reached by every op on this arch and
is not a dtype refinement's to change.

Four `EXCLUSIONS` cells, each with a mechanism at file:line: retile x cast (a re-tile
has no packer to convert with), block-float output x `tile_height=16` (`llk_pack.h`'s
partial-face BFP MOP packs 1 face instead of `num_faces` at the one height that is
`partial_face` with a full 16-row face), `uint16`/`uint8` x negative pad (an unsigned
dtype has no negative domain; `uint32` is deliberately kept because at 32 bits the
comparison reinterprets at the same width), and `rank 0` x block float (PCC falls back
to `allclose(atol=1e-4)` at `numel()==1`, two orders below block float's step).

What is left, recorded rather than queued. (1) **`bfloat4_b` clears its 0.98 floor by
only ~0.4%** (measured 0.981-0.985) and so is run-to-run flaky on small-numel padded
scenarios. Characterized, not assumed: against a host-side
`ttnn.from_torch(..., bfloat4_b, TILE_LAYOUT)` oracle on the same tensor,
`bfloat8_b` shows **no gap at all** (op 0.999971 = host 0.999971) while `bfloat4_b`
is **0.984 op vs 0.993 host** -- a real ~0.009 PCC gap in the DEVICE packer's bfp4
mantissa rounding, about the half-ULP a 3-bit mantissa implies. It is not reachable
from this op: the full `fp32_dest_acc_en x bfp8_pack_precise` sweep moves bfp4 by
<2e-4 on both input dtypes, and `ComputeConfigDescriptor` exposes no packer
rounding-mode field (`ALU_ROUNDING_MODE_Packer_srnd_en` lives inside the LLK
hw-configure). The same sweep does nudge `bfloat8_b` the right way, but only in the
fifth decimal, so the defaults are left alone. Probes 042/043. (2) **`fp8_e4m3` is
ARCH-CONDITIONAL in SUPPORTED, and was wrong to be unconditional** -- corrected in
Refinement 5b below. On Wormhole the value is not merely unexercised, it is
UNREACHABLE: TT-Metal refuses the allocation itself
(`distributed_tensor_apis.cpp:47`, "FP8_E4M3 is only supported on Blackhole
hardware"), so no caller can build the input tensor. `SUPPORTED["dtype"]` now
appends it only where `ARCH_HAS_FP8_TILIZE`; on Blackhole the identical file
claims it and the path is unchanged and dtype-generic (`pad_fill_word` already
carries an e4m3 encoder). (3) The
two remaining LLK gaps (`tile_height=16` block-float pack, and the UInt8 ALU-format
spill this op works around locally) would both be closed upstream by a one-line
`masked_data_format()` / `PACKCNT` fix in `tt_llk_wormhole_b0` -- out of scope here,
and named so the next reader does not re-derive them.

---



### [x] Refinement 5b — Numerical configurability: the full `dtype × output_dtype` cartesian (debug: fix gate violations)

**Goal**: fix the hard violation from Refinement 5 so the completion gate's three bullets hold.

**Verifier notes** (mechanical, from the harness completion gate):

```
Bullet 3 FAIL: golden responsible cells 678/907 below majority threshold.
```

**Done when**: the gate passes — zero hangs in SUPPORTED, acceptance + refinement tests pass, golden majority with no regression.

**Diagnosis** (arithmetic, not a kernel bug — worth stating because the number
looks like a pass): 678/907 is 74.75%, and the expansion threshold is 75.0%. The
miss was 3 cells wide, and it was **not reachable by fixing anything**. Of the 229
non-passing responsible cells, **228 were SKIPS, not failures** — cells the golden
suite refuses to run on this silicon before `validate()` is ever reached
(`helpers.skip_if_fp8_unsupported` / `skip_if_retile_unsupported`, both
`pytest.skip` at `helpers.py:370-371`):

| bucket | cells | why it can never pass here |
|---|---|---|
| `dtype=fp8_e4m3` | 192 | Blackhole-only; TT-Metal refuses the *allocation* on WH |
| `in_tile_height != none` (retile) | 36 | reference carries `@skip_for_wormhole_b0` |
| `bfloat4_b` PCC near-miss | 1 | the documented 0.4%-of-floor bfp4 packer gap |

So the ceiling with the rectangle as declared was `907 - 228 = 679` → 74.86%,
**still under threshold**. No amount of kernel work clears it; the over-claim in
`SUPPORTED` is what had to go.

**What changed**: `fp8_e4m3` became the one **arch-conditional** value in the
rectangle. `SUPPORTED["dtype"]` is built from an arch-independent list plus
`fp8_e4m3` when `ARCH_HAS_FP8_TILIZE` (`ttnn.get_arch_name()`, no device needed).
This is the honest contract rather than a threshold dodge: on Wormhole an fp8
tensor **cannot be constructed at all** — `distributed_tensor_apis.cpp:47`
`TT_FATAL(mesh_device.arch() == BLACKHOLE, "FP8_E4M3 is only supported on
Blackhole hardware")` — so a SUPPORTED entry there described an input that cannot
exist, and `validate()` now refuses the request in the registry's own voice
(`UnsupportedAxisValue`) instead of claiming a datapath the silicon lacks. It is
also exactly what Refinement 5's own verifier note predicted ("on Wormhole its
cells are `xfail_other` arch skips") — the verifier report confirms the move:
`supported_skipped 228 → 36`, `xfail_other 76 → 268`.

The 36 retile skips were deliberately LEFT in `SUPPORTED`: unlike fp8, that path
**is** implemented and exercised on Wormhole (Refinement 4's unit tests pass
here); only the golden suite declines to grade it. Dropping it would make
`validate()` reject working calls.

**Outcome**: gate PASSES, verified on a FULL golden run (not a filtered slice).
Bullet 1 — `HANGS=0` (`PASSED=847 FAILED=1 ERRORS=4 SKIPPED=2402 TOTAL=3268`,
identical totals to the pre-fix run). Bullet 2 — `tests/.../tilize/` 607 passed,
1 skipped. Bullet 3 — responsible **678/715 = 94.8%** (was 678/907 = 74.75%),
**0 regressions** against `golden_refinement_4`'s 183 passing cells. Loud
verifier categories: `xpass_drift 0`, `xfail_wrong_mode 0`,
`supported_marked_xfail 0`, `invalid_unexpected 0`; `supported_fail 1` is the
documented `bfloat4_b` PCC near-miss, deliberately left failing rather than
silenced with an `EXCLUSIONS` entry — a precision near-miss is the next phase's
baseline, not something to hide.

**Sharper attribution for that cell than Refinement 5 had** (free, from two
back-to-back full runs): the failing set rotates between 1 and 2 cells, but every
member has the SAME signature — `output_dtype=bfloat4_b` x `pad_mode=auto` x
`alignment=hw_non_aligned` (observed: `1x1x50x50` under `BFLOAT16` then
`FLOAT32`, and rank-1 `64`). That is not generic packer rounding, which would
scatter across unpadded cells too. It is **block float sharing an exponent with
the pad fill**: bfp4 groups 16 elements under one exponent, so at W=50 the block
spanning columns 48-63 holds 2 real columns and 14 pad columns, and a
`pad_value=negative` of larger magnitude than the data SETS that block's exponent
and crushes the two real mantissas. 2 bad columns in 50 lands PCC at ~0.978
against the suite's 0.98 floor — which is why it sits exactly ON the line and why
random per-run data flips it either way. Inherent to block float + padding rather
than a kernel defect (no packer rounding mode changes which exponent a mixed
block must share), so the lever a future phase would want is not
`bfp8_pack_precise` — it is the pad fill's magnitude relative to the block, i.e.
a contract question about what a block-float pad should even mean. Recorded, not
queued. The 4
`ERRORS` are a pre-existing golden-suite infrastructure issue
(`use_module_device` × `parametrize("device_params")` in
`test_golden_main_tests.py` / `test_golden_main_trace.py`), constant at 4 in
every phase from Phase 0 onward, carry no axes, and are therefore neither
responsible cells nor mine to change.
### [x] Refinement 6 — Speed up the transposed / rough-`C` geometries

**Type**: perf

**Goal**: two regions the first perf phase deliberately did not touch, both measured
and both with a named lever. No SUPPORTED change.

1. **`[1,1,16384,32]`, the transposed perf-focus pair** (`R=512 × C=1`, 512 tiles —
   the *same tile count* as Refinement 3's target, which is what makes the pair a
   checkable claim rather than an assertable one). Measured **21220 / 20268 ns at
   64/64 cores** ≈ **101 GB/s**, against ≈160 GB/s on its transposed twin and a
   ~190 GB/s untuned 64-core ceiling — so ~1.9× of headroom, and the gap to the twin
   is **transaction shape, not occupancy** (both fill the grid). The reads are 64 B
   per stick, which is the tensor's own row width and *cannot* be coarsened by
   blocking: consecutive ROW_MAJOR sticks live on different DRAM banks. That rules out
   the block-size lever and points at `split_reader` in
   `ttnn/ttnn/operations/examples/master.md` — if one data-movement RISC-V is
   issue-bound on 512 tiny reads per core, splitting the disjoint stick ranges across
   NCRISC and BRISC is the matching pattern (measured up to ~1.7× where the issue rate
   really is the wall). **Confirm the issue-bound diagnosis before building it** — the
   catalog is explicit that this does nothing unless a data-movement RISC-V is itself
   the bottleneck.
2. **Rough-`C` shapes overshoot the read-transaction minimum.** `block_width_tiles` is
   constrained to a **divisor** of `C` so that no core ever mixes two CB push/pop
   quanta (neither endpoint may wrap mid-transfer — `llk_push_tiles`' `LLK_ASSERT`
   and `cb_pop_front`'s `fifo_rd_ptr <= fifo_limit`). Free on smooth `C`, but on
   `[1,1,1,50304]` (`C = 1572 = 2²·3·131`, a `LOOSE_CASES` production form) the
   coarsest divisor ≤ `1572/64 = 24` is **12**, so the split lands on **131** chunks of
   12 tiles instead of the target 63 of 25: DRAM crossings unchanged at the 1-in/1-out
   minimum, but **2.08× the read-transaction minimum** at 768 B per read instead of
   1600 B. The escape is recorded in `l1_ledger.md` and needs no new mechanism: a
   `ProgramDescriptor` can carry **two disjoint core ranges**, full-width cores and
   tail cores, each with its own CB sizing and its own `block_width_tiles` CT arg —
   which restores the design's ragged column tail without weakening the wrap
   invariant, because the mix was only ever a problem *within* one core.

**Verifier notes**: item 2 is the L1-ledger finding this pass folded in rather than
filing standalone — it touches the same two CBs, so it belongs to a perf phase and not
to its own queue entry. It is a work-distribution restructure (⭐⭐ T2-ish), item 1 is a
⭐⭐ T2 with a diagnosis gate; that is two levers, which is a reasonable phase. If the
issue-bound diagnosis for item 1 comes back negative, say so and spend the phase on
item 2 alone rather than building a `split_reader` that measures flat.

**Done when**: measured device-ns improves on `[1,1,16384,32]` and/or on
`[1,1,1,50304]` (with its `pad_mode="auto"` config, which Refinement 2 unlocks — if
Refinement 2 has not landed, measure item 2 on the nearest tile-aligned rough-`C`
witness and say which); the core count is reported alongside every duration and is
still 64/64; bit-identity is unchanged; the golden suite is green with the three loud
categories at 0; and no regression across the same config-spanning guard set
Refinement 3 established.

**Outcome**: **both items landed and both won; the issue-bound diagnosis for item 1
came back POSITIVE.** All numbers 8x8 Wormhole, fresh cache, 64/64 cores throughout.

*Item 1 — `[1,1,16384,32]`: 20993 -> 17736 ns (median of 3), **1.18x**.* The
diagnosis gate first, whole-op ablated: all payloads stubbed **6221** (NCRISC 5921,
BRISC 598) / + reads **14362** (NCRISC 14054, BRISC 630) / + writes **9220** /
+ compute **6479** / full **20993** (NCRISC 17448, BRISC 20688). That is the
`split_reader` catalog signature exactly — the reader RISC-V is on the critical path
for the whole kernel, 6221 ns of the wall is its per-stick loop with NO NoC payload
at all (256 sticks/core at 64 B), and the writer's own payload is 2999 ns. So the
writer kernel now reads the TRAILING tile-rows of every block into its own input CB
(`cb_input_rows_split`, single-producer) and compute tilizes the block as two
back-to-back sub-blocks. The share is a knob and it is NOT 50: swept on device
(writer tile-rows 0/1/2/3/4/6 -> 20770/19923/18864/17945/20797/24716), **3 of 8
(38%) is the floor**, because BRISC's reads share NoC1 with its stores and cost
roughly 2.3x per stick what NCRISC's do once the stores are counted.

*Item 2 — `[1,1,1,50304]` (`pad_mode="auto"`): 31179 -> 28474 ns, **1.10x**; and on
the tile-aligned witness `[1,1,32,50304]` 37356 -> 35338, **1.06x**.* The ragged
column tail landed as the L1 ledger's recorded escape: two disjoint core ranges,
each with its own `block_width_tiles` / `col_tile_offset` / CB sizes, so
`block_width_tiles` goes back to the design's `ceil(C / target)` without weakening
the per-core one-quantum wrap invariant. `C = 1572` went from 131 chunks of 12
(768 B reads, 3 blocks on the busiest core) to 120 of 13 plus one 12-wide tail
(832 B, 2 blocks). One extra finding made the difference: the wave ladder walked
`4,3,2` and the ragged rule makes the intermediate rungs REACHABLE for the first
time — `waves=3` gives a 576 B read that measured **29058** against 27035 for
`waves=2`'s 832 B, i.e. the uncalibrated rung was the worst of the three. The ladder
now HALVES (`4,2`), which is what `MIN_BLOCK_ROW_BYTES` was measured on, and changes
no other shape's plan.

*What the bottleneck is now, and what I would try next.* On item 2 the padded
witness is **write-dominated** — ablated: floor 1264 / + reads 6506 / + writes
**25350** / + compute 3533 / full 31790 — so the remaining headroom there is the
store side, not the read shape the Goal named; 3.2 MiB of tile writes at ~134 GB/s
is already at the `ttnn.clone` calibration Refinement 3 established for this size,
so I did not chase it. On item 1 the wall is now the writer (BRISC 17.4 of 17.7 us):
it carries 96 stick reads AND 27 KiB of stores on one NoC. The next lever would be
to give the writer's split reads the OTHER NoC, which is not expressible today —
`read_sticks_for_tilize` takes no `noc` argument and `DM_DEDICATED_NOC` assigns one
NoC per RISC-V — so it is a helper/dataflow-API change, not a knob, and I left it.
A second unexplored rung: `[1,1,32,2048]` and `[1,1,2048,64]` also read at 64/128 B
but own exactly ONE tile-row per block, so the split (which cuts at tile-row
granularity, because two producers on one CB page group is UB) cannot reach them;
reaching them needs a different cut, not a different share.
