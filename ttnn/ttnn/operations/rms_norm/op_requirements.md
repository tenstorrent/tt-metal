# Operation Requirements: rms_norm

## Definition

- **Formula**: `output[..., h, w] = input[..., h, w] * rsqrt( (1/W) * Σ_{w'=0}^{W-1} input[..., h, w']² + epsilon ) * gamma[w]`
  (the mean is over the **true, unpadded** `W`; `gamma` is optional)
- **PyTorch Reference**:

```python
def torch_rms_norm(x, gamma=None, eps=1e-6):
    """Reference RMSNorm over the last dimension, always computed in fp32."""
    x = x.float()
    out = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    if gamma is not None:
        out = out * gamma.float().reshape(-1)
    return out
```

- **Import Path**: `from ttnn.operations.rms_norm import rms_norm`
- **Function Signature**:

```python
def rms_norm(
    input_tensor: ttnn.Tensor,
    *,
    gamma: Optional[ttnn.Tensor] = None,
    epsilon: float = 1e-6,
    compute_kernel_config: Optional[ttnn.ComputeConfigDescriptor] = None,
    memory_config: Optional[ttnn.MemoryConfig] = None,
) -> ttnn.Tensor
```

`default_compute_kernel_config()` is exported from the same module and is the single source for the
`compute_kernel_config=None` case (HiFi4 / `fp32_dest_acc_en=True` / `math_approx_mode=False`).

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [float32, bfloat16]
- **SUPPORTED fp32_dest_acc_en**: [True] (the maxed-out precision corner)
- **SUPPORTED layout**: [TILE, ROW_MAJOR] — both native, no host-side `to_layout` / `pad` / `slice`
- **SUPPORTED shape-derived axes**: alignment ∈ {tile_aligned, w_non_aligned, h_non_aligned}; rank ∈ {2, 3, 4}
- **SUPPORTED op-specific axes**: gamma_mode ∈ {gamma, no_gamma}; gamma_dtype ∈ {float32, bfloat16, "none"}; gamma_layout ∈ {TILE, ROW_MAJOR, "none"}
- **SUPPORTED memory_layout**: [INTERLEAVED]
- **EXCLUSIONS**: `{dtype: float32, fp32_dest_acc_en: False}` (the canonical precision exclusion; currently unreachable because `fp32_dest_acc_en=False` is outside SUPPORTED entirely)
- **Cores**: multi-core from day 1 — the grid is partitioned into `num_row_groups` rectangles; the
  `row` axis is split across groups and the **dependent** `hidden` axis is split *within* a group,
  with partials combined over the NoC (gather-to-root + multicast-back). Both regimes (R1
  `w_group_size == 1`, R2 `w_group_size > 1`) ship.
- **Compute config**: caller-supplied `ttnn.ComputeConfigDescriptor` passed through unchanged;
  `math_fidelity` / `math_approx_mode` ungated, `fp32_dest_acc_en` gated to True.
- **Golden baseline**: 737 / 737 supported cells passing; 6174 xfail_expected; 33900 invalid_skipped;
  0 in every loud category (per `verifier_report.json`).

### [x] Refinement 1 — Numerical configurability expansion (`fp32_dest_acc_en=False` + `bfloat8_b`)

**Goal**: add `False` to `SUPPORTED["fp32_dest_acc_en"]` and `ttnn.bfloat8_b` to both
`SUPPORTED["dtype"]` and `SUPPORTED["gamma_dtype"]`, keeping `{float32, fp32_dest_acc_en=False}` in
`EXCLUSIONS`. This is the single largest cell unlock in the queue (≈ 1100 interleaved xfail cells:
818 that differ from Phase 0 only by `fp32_dest_acc_en=False`, 100 by `dtype=bfloat8_b`, 140 by
`gamma_dtype=bfloat8_b`, plus their combinations) **and it is the gate on every perf refinement**:
every `group="perf"` loose case in `feature_spec.LOOSE_CASES` runs at `fp32_dest_acc_en=False` +
`math_fidelity=HiFi2`, as does every `resilience` and `pad_poison` loose case. Work items:
the DEST-capacity derivation must follow `DEST_AUTO_LIMIT` in both settings (it doubles when fp32
DEST accumulation is off — the helpers already read it, so verify nothing in the kernel or the L1
solve assumes the halved value); the `Σ x²` accumulator and the whole `cb_stat_*` path must stay
**fp32 in L1 regardless**, because bf16 accumulation error on an all-positive monotonically growing
sum is exactly the failure mode `row_reduce_accumulate` documents; and `bfloat8_b` needs the
intermediate-CB formats and `cb_scaler` left at bf16 (`reduce_helpers_dataflow.inl:185-187`
`static_assert`s anything else). Cells that fail out of the box land in `EXCLUSIONS`, not in their
own refinement.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**: **must land first** — Refinement 3 (the first perf phase) measures the
`fp32_dest_acc_en=False` + HiFi2 configuration, and a perf number taken at `fp32_dest_acc_en=True`
says nothing about it (different DEST datapath, different DEST capacity, different chosen `R`).
Do **not** proxy it. Two collisions to be deliberate about: (1) `bfloat8_b` × non-tile-aligned is
already parked in `feature_spec.INVALID` (author-scoped), so those cells stay *skipped* — do not
add an `EXCLUSIONS` entry for them expecting cell movement; (2) the golden `pad_poison` cases run at
`fp32_dest_acc_en=False`, so this refinement is the first time the ragged-hidden-tile mask
(`mask_tail_block`) is exercised with a bf16 DEST accumulator — the mask-before-square identity
`(x·mask)² == x²·mask` still holds, but the *poisoned* magnitude (1000.0) squared in a bf16 DEST is
where a lost mask now shows up as a large error rather than a small one. Keep `has_tail`'s stat
column fp32.

**Outcome**: landed in full, **zero kernel changes** — the three `.cpp` files are byte-identical to
Phase 0. Every named work item was already satisfied by construction and was verified rather than
built: DEST capacity is only ever read through `DEST_AUTO_LIMIT` / `Dst::D0` (no literal assumes the
halved fp32 value, so the 4 → 8 doubling needs nothing); the whole `cb_stat_*` chain plus
`cb_zero_tile` were already pinned to fp32 **unconditionally** rather than following
`fp32_dest_acc_en`, and `cb_scaler` / `cb_wmask` already stay bf16; and `_cb_specs()` already derived
every input/output/gamma CB format from the corresponding tensor's dtype, so `bfloat8_b` flowed
through as a third `tile_size` value (1088 B, which only *narrows* the L1 footprint). The sole real
blocker was host-side: `Tensor.element_size()` **raises** for block-float dtypes ("datum for bfp2,
bfp4, bfp8 is invalid"), fixed with `_elem_bytes()` returning 0 for `bfpN` — only the ROW_MAJOR stick
legs consume an element size and block-float has no ROW_MAJOR form (ttnn itself `TT_FATAL`s on
`layout == Layout::TILE`, confirming `feature_spec.INVALID` is right to *skip* those cells).
**Nothing landed in EXCLUSIONS**: every cell the refinement named measured green, well inside the
golden `TOLERANCES` — bf16 @ `fp32_dest_acc_en=False` PCC ≥ 0.99993 / rel-RMS ≤ 0.012 (gate
0.995/0.04), `bfloat8_b` PCC ≥ 0.99985 / rel-RMS ≤ 0.021 (gate 0.99/0.10), `gamma_dtype=bfloat8_b`
against bf16 and fp32 activations PCC ≥ 0.99994. The `pad_poison` collision the notes flagged is
clean: all 6 interleaved poisoned shapes pass at `fp32_dest_acc_en=False` with PCC ≥ 0.99998, so the
mask-before-square identity does hold under a 16-bit DEST. One incidental finding worth recording,
not a defect: at **LoFi** the FPU's 5-bit truncating (not rounding) mantissa biases both the
`x · rstd` and `· gamma` multiplies low, giving a measured ~3.5 % systematic shrink (got/true ratio
median 0.965 at bf16) while PCC stays ≥ 0.9995 — the same mechanism Phase 0 recorded as a ~0.1 %
shrink at fp32/HiFi4, so `test_rms_norm_precision_matrix` derives its scale-bug band from
`math_fidelity` instead of firing on it. **This unblocks Refinements 3/4**: all 8 interleaved
`group="perf"` cases now run for real at their pinned `fp32_dest_acc_en=False` + HiFi2 config, so
those phases can measure the configuration they are specified against instead of proxying it.

### [~] Refinement 2 — Sharded input/output placements (all three schemes)

**Goal**: add `ttnn.TensorMemoryLayout.HEIGHT_SHARDED`, `WIDTH_SHARDED` and `BLOCK_SHARDED` to
`SUPPORTED["memory_layout"]`, consuming the resident shard **in place** and writing a matching
sharded output (the golden harness requests `memory_config=input.memory_config()` for every sharded
cell). ≈ 1800 interleaved-only xfail cells move (561 HEIGHT / 586 WIDTH / 586 BLOCK at the Phase 0
corner, and ~2000 more once Refinement 1 has landed `fp32_dest_acc_en=False`), plus 3 `_SHARDED`
and 5 `perf` loose cases.

**Verifier notes**: this is one refinement, not three, because `op_design.md`'s lamp S1 already
classifies all three as **placement unlocks of the scheme Phase 0 built** — the cross-core combine
exists, so the shard spec *supplies* the block geometry instead of the selection function choosing
it: HEIGHT ⇒ `w_group_size = 1` and the reduction is local; WIDTH ⇒ `w_group_size` = the shard
grid's core count and `core_w_tiles` = the shard width in tiles; BLOCK ⇒ one grid row of the shard
grid *is* the `Mcast2D` rectangle. Read the extents **off the spec**; do not re-run
`_select_regime` for a sharded input.
Two hard requirements, both native:
(1) point `cb_input_tiles` / `cb_output_tiles` at the resident L1 shard with
`ttnn.cb_descriptor_from_sharded_tensor(...)` (zero-copy) and drop the DRAM leg — re-reading a
core's *own* shard through a `TensorAccessor` is **not** an acceptable slow path: it would mean the
axis value is merely tolerated, not implemented, and this refinement would be sent back. The
accessor keeps owning interleaved I/O only.
(2) the multicast/gather L1 map must stay identical group-wide (`mcast_pipe.hpp:44-45`), which is
why Phase 0 sizes every CB to the group-uniform `C`; a shard grid that does not divide the tensor
evenly (the `_RESILIENCE_SHAPES` block is full of prime tile counts) must keep that invariant.
There is no implementation skill for placement yet — `/memory-layouts` is RM/TILE layout, **not**
placement, so do not invoke it here. Order: after Refinement 1 (so the sharded cells land at both
`fp32_dest_acc_en` values in one pass) and before Refinement 5, which measures the sharded perf
geometries this creates.

**Outcome**: all three placements landed in `SUPPORTED["memory_layout"]` and are consumed
**natively** — `_sharded_cb` pins `cb_input_tiles` / `cb_output_tiles` zero-copy over the resident
L1 buffers on the TILE legs (the reader's and writer's DRAM legs disappear entirely; asserted
structurally by `test_rms_norm_tile_shard_is_consumed_in_place`, since a `TensorAccessor` read of a
local shard is numerically indistinguishable and would pass every value check). The ROW_MAJOR legs
cannot pin — the block CB there is the `tilize`/`untilize` staging buffer, whose group-uniform
tile-row stride is not the shard's stick stride — so they re-stride the sticks core-locally
(L1 → L1, one bulk transfer per 32-row group when the strides already agree); still zero DRAM
crossings for the activations. The geometry is **read off the spec, not re-chosen** (`_ShardView`):
HEIGHT ⇒ `G = 1` and the combine degenerates to a local copy; WIDTH ⇒ the whole shard grid is one
group; BLOCK ⇒ one grid row of the shard rectangle. `_select_regime` is not called for a sharded
input. Two structures the spec forced: a WIDTH shard grid is often **not a rectangle** (16 slices on
an 11-wide grid = a full row + a 5-core row) while the `rstd` broadcast needs one, so the group
rectangle is the bounding box and the shard-less cores in it stay program cores holding the
identical CB map — receiving the broadcast, never acking, which is why the mcast now carries an
explicit `num_active = G − 1`; and a ROW_MAJOR WIDTH/BLOCK shard's width granule is the **L1
alignment**, not a tile, so `partial_w` became **per-core** (every core can carry a ragged tail, not
just the one owning the tensor's last hidden tile). Two real bugs came out of that second one, both
fixed and both pinned by tests: a DRAM read **truncates** its source address to the DRAM alignment,
so every core was reading gamma from the 64-byte-aligned offset *below* its own slice (PCC 0.28);
and the shifted re-read then hit a sub-L1-alignment offset once the gamma dtype differed from the
activation's (PCC 0.44). Measured: PCC ≥ 0.99996 across all three schemes × TILE/ROW_MAJOR ×
{bf16, bfloat8_b, fp32}; the `pad_poison` shapes are green on all three placements (PCC ≥ 0.99998,
got/true ratio ≈ 1.0, so the mask-before-square and the true-`W` divisor both survive sharding);
all five pinned `group="perf"` shard geometries run, which **unblocks Refinement 5**. Golden slices:
loose-sharded 269/281 pass (was 263 before the two L1 levers below), cartesian slices 516/516 and
761/777. Two L1 levers landed alongside: `_DEPTH_LADDER` (a shard spec pins both `G` and `C`, so `R`
and the buffer depths are the only residency knobs left — depth is spent first and surrendered last,
and no interleaved geometry ever leaves step 0) and the `cb_gamma_rm` → `cb_input_rm` alias (Rule 3
pattern 3, same producer and consumer, guarded on equal page formats). **Deferred to 2b**: every
remaining failure — 12 loose + 16 cartesian in the slices run — is the *same* capacity limit, not a
correctness gap. See Refinement 2b.

### [x] Refinement 2b — HEIGHT_SHARDED wide-`W`: chunk the hidden axis inside a core

**Goal**: close the one class Refinement 2 left failing. `HEIGHT_SHARDED` cuts the **independent**
`row` axis, so `w_group_size == 1` by construction and `core_w_tiles == tensor_w_tiles` — the caller
pinned the very knob the residency solve uses to bound `C`, so the escape hatch "raise `G` until the
slice fits" is gone. Past `C ≈ 127` (bf16) or `C ≈ 64` (fp32 activations, or an fp32 gamma) the
resident input shard + resident output shard + `cb_gamma_tiles` (`C` tiles) **alone** exceed the
1.44 MiB budget and `_max_block_row_tiles` returns 0, so the program descriptor raises with
"the per-core CB working set … does not fit L1 … the shard spec pins both". Confirmed exhaustively
at the ladder's last step: no depth or block setting reaches it (`R` is already 1 — a HEIGHT shard
is often one tile-row), which is exactly why this is a scheme item and not a knob-turn.

Failing cells observed: loose `HEIGHT_SHARDED` at `W ∈ {3000, 4064, 4095, 5119, 5120, 6144, 11008}`
(12 of 281 in the loose-sharded slice) plus the cartesian `1x1x32x4096` HEIGHT column at fp32 or
with an fp32 gamma (16 of 777). WIDTH and BLOCK are unaffected at every width tested up to
`W = 11008` — they split `hidden` and so shrink `C` themselves. One WIDTH cell fails for the mirror
reason (`13x777x1023` puts a 650 KiB full-height shard on each of 32 cores, leaving 110 KiB for a
`R·G = 32`-tile `cb_stat_gather`).

**The lever, and why it is cheap here**: regime R3 (`op_design.md`, streaming two-pass) chunks the
hidden axis into `w_chunk_tiles` so the working set is `O(chunk)` instead of `O(C)`. R3 was rejected
for Phase 0 because its cost is **one extra whole-tensor DRAM read** for the apply pass — by far the
largest term in the traffic ranking. For a *resident* shard that cost is **zero**: the apply pass
re-reads the same L1 the first pass read, so R3's only remaining cost is the per-chunk
`Accumulate` reload (`reduce_helpers_compute.hpp:328`) and per-chunk phase-boundary reconfig. The
design already says `core_w_tiles` is the only extent the CBs are sized against, so `w_chunk_tiles`
slots underneath it. Gate the chunked path on `sharded && C > (whatever the solve admits)` so every
currently-green geometry keeps the single-chunk path byte-identical. Note `cb_gamma_tiles` must
chunk too — at `C = 344` it is 688 KiB on its own, which is most of the overrun.

**Done when**: the `HEIGHT_SHARDED` cells above pass (or, where even a one-tile chunk cannot fit,
fail for a *different*, documented reason), no currently-passing cell regresses, and
`test_rms_norm_sharded.py` gains a wide-`W` HEIGHT case per layout.

**Outcome**: landed as **one new block factor**, `w_chunk_tiles` (WC), threaded as a single
compile-time arg (`CB_CHUNK_TILES`) into all three kernels and defaulted to `C` — at which every
loop it introduces runs exactly once, so the interleaved and already-fitting sharded schedules are
byte-identical (measured: the 8 interleaved perf shapes are within ±2 % of their recorded numbers,
prefill slightly *faster*: 8192×7168 594 707 ns vs 597 240 ns recorded). WC sizes only the buffers
that **stream** over `hidden` (`cb_gamma_tiles`, `cb_normed`, `cb_output_*`, `cb_input_rm`); the
block itself stays whole-resident in `cb_input_tiles`, so R3's headline cost — a second whole-tensor
read for the apply pass — is **not** paid: the shard is already in L1 and is never re-fetched. Three
structural pieces the scheme needed: (1) each chunk's partial `Σ x²` packs into its **own**
`cb_stat_sq` column, because `reduce_stat_block` already sums a tile-row's columns (the tail column
proved the pattern) — `eltwise_chain` forbids `L1Accumulation` together with `DestAccumulation`
(`eltwise_chain.inl:1034`), so an L1 read-modify-write accumulator was not an option and this costs
nothing extra; (2) a ROW_MAJOR block becomes **chunk-major** (`tilize<WC>` emits chunks back to
back) while a TILE block stays row-major-`C`, so one `in_ref(g, rows)` helper is the only place that
knows the layout; (3) with a **pinned** output shard the apply packs at a strided offset under a
caller-managed reserve/push, since the shard's layout is not the compute order. The residency solve
takes the **coarsest** WC that fits, only after the depth ladder has failed at WC == C, and only on
a resident shard whose per-core hidden geometry is uniform (`_chunking_supported`).
**Measured**: 10 of the 12 named loose `HEIGHT_SHARDED` failures now pass — `W ∈ {3000, 4064, 4095,
5119, 5120, 6144}` in both layouts and `11008` in ROW_MAJOR — at PCC ≥ 0.99987; the golden HEIGHT
loose slice is 91/93 (was 81/93), the cartesian `1x1x32x4096` HEIGHT column is 39/39 including all
19 fp32-activation / fp32-gamma cells (was 16 failing), WIDTH+BLOCK 187/188 and INTERLEAVED loose
103/103 unchanged. Chunking was also verified against a real cross-core combine (a pinned
`[32, 8192]` WIDTH shard on 2 cores and the BLOCK equivalent, PCC 1.000000), so it is not
HEIGHT-only.
**What is left, and why it is not a follow-up**: the two `W = 11008` **TILE** cells
(`1x1x160x11008`, `1x224x11008`) still fail, now with an explicit byte accounting instead of a
generic refusal — and for a *different* reason than chunking addresses. Their `32 × 11008` bf16
input and output shards occupy 1 409 024 B of the 1 441 792 B budget, leaving 32 768 B, while the
per-core **fixed** statistics pipeline alone (scaler, zero tile, `cb_stat_partial`, `_gather`,
`_sum`, `cb_rstd_send`, `cb_rstd`) is 26 624 B and the chunked working set bottoms out near
`26 624 + 4096·(⌈344/WC⌉ + WC) ≈ 182 KiB` at its optimum `WC ≈ 19`. No chunk size closes a 5.5×
gap: the lever that would is collapsing the **degenerate `G == 1` combine** (at `w_group_size = 1`
the gather is a self-write and the multicast a local copy, so four of those seven fp32 buffers are
copies of each other), which is a combine-topology change, not a chunking one, and it touches the
much-travelled interleaved R1 path. Left as a recorded finding. One further known bound, also not a
regression: on the ROW_MAJOR legs `cb_input_tiles` stays `O(R·C)` (the tilized block), so a
ROW_MAJOR shard wider than that CB alone cannot be rescued by chunking either — the true two-pass
R3 (re-stride each chunk from the resident shard twice, once per pass) would make it `O(WC)` at the
cost of the block residency, and no golden cell needs it today.

### [x] Refinement 3 — Speed up the perf-flagged wide interleaved decode profile

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES` marks `(1, 1, 32, 7168)` at **INTERLEAVED** placement as the
one perf case carrying `minimum_expected_speedup = 7.0` — "expected to expose a decisively better
architecture" — i.e. a goal of `104259 ns × 1350/actual_aiclk_mhz / 7` (≈ 14.9 µs at 1350 MHz),
with its soft `pcc_threshold = 0.9995` still holding. Optimize **that** shape at **its exact
config**: bf16 / TILE / INTERLEAVED / gamma bf16 TILE / `fp32_dest_acc_en=False` /
`math_fidelity=HiFi2`. Take the other three interleaved decode cases (`W` = 1024 / 2304 / 5120,
same config) along as secondary targets. Relevant patterns in
`ttnn/ttnn/operations/examples/master.md`: `width_split` (this is a one-tile-row tensor — the whole
regime), `tensix_all_reduce` (the combine topology: its measured rule of thumb is *grid two-stage
when the grid is busy or the payload is tiny*, and our payload is `block_row_tiles = 1` tile, while
Phase 0 uses flat root-gather + mcast — 1.45–1.60× is claimed there), and `double_buffer` /
`compute_block_size` for the co-tune below. No SUPPORTED change.

**Done when**: measured device-kernel ns improves on `(1,1,32,7168)` at the config above and moves
it toward the clock-scaled `achievable_ns / 7` goal, its `pcc_threshold=0.9995` gate still holds,
the golden suite is still green, and there is no regression across the config-spanning guard set
(one representative per distinct kernel path × layout × placement: TILE-in/TILE-out interleaved,
RM-in/RM-out interleaved, `w_group_size == 1` row-parallel, `w_group_size > 1` cross-core, a
`w_non_aligned` shape, and — once Refinement 2 has landed — one sharded cell).

**Verifier notes** (queue-level facts measured during verification, not levers I am prescribing):
`MAX_W_GROUP_SIZE = 32` is the *only* place the core-assignment knob is turned, and it was measured
**only at `tensor_row_tiles == 1`** (where capping 110 → 22 cores won 1.53–1.59× on these very
shapes). Its consequence for neighbouring shapes is unmeasured and visible in the selection
function: at `tensor_row_tiles = 2` the cap leaves **44 of 110** cores active, at 3 it leaves 66, at
4 it leaves 88 (computed from `_select_regime` on an 11×10 grid) — `(1,1,64,12288)` is exactly such
a `_WIDE` loose case. Both the cap and `MIN_PIPELINE_BLOCKS` live in one host constant each, so
re-tuning them is a knob-turn; replacing flat root-gather with a two-stage grid combine is a
scheme-change and the bigger of the two pieces of work — size the phase accordingly (one T3 lever
is a whole phase).

**Outcome**: **(1,1,32,7168) 12467 → 8987 ns (1.39×)** at its exact pinned config (bf16 / TILE /
INTERLEAVED / gamma bf16 TILE / `fp32_dest_acc_en=False` / HiFi2, blackhole_p150b), against a goal
of `104259/7 ≈ 14894 ns` that is now cleared by 1.66×; the three secondary decode cases came along
at 1.32–1.34× (1024 9101 → 6882, 2304 9730 → 7299, 5120 11219 → 8350) and the four prefill cases are
unchanged to within noise (99 559 / 212 595 / 425 990 / 592 081 vs 103 076 / 220 005 / 425 343 /
591 707). Golden: `-m perf` 13/13, `-m pad_poison` 24/24, the `1x1x64x128` cartesian slice 165/165
with no xpass drift, `test_op_loose` 381/384 — the 3 failures are the exact cells Refinement 2b
recorded as remaining (two `W=11008` TILE HEIGHT cells + `13x777x1023` WIDTH), not regressions. Unit
directory 503 passed / 0 failed.
Three levers landed, each measured by a `DeviceZoneScopedN` timeline rather than guessed:
(1) the fp32 `cb_zero_tile` fill (2363 ns of scalar stores) sat on EVERY core's reader ahead of the
input block although only a combine root reads it — moved to the leader's writer, whose BRISC idles
~6 µs; (2) the gamma slice was read ahead of the input block although it is first consumed after the
combine — moved behind it (TILE gamma only: a ROW_MAJOR gamma is tilized before the block loop and
aliases `cb_input_rm`, and reordering it gave PCC ~ 0, caught by the unit suite); (3) the
scheme-change — a **two-stage grid combine** replacing flat root-gather, which cut the root's
combine chain 3630 → 1360 ns and, more importantly, made a full-grid reduction group affordable, so
`MAX_W_GROUP_SIZE` went 32 → 0 (`G = 110` now costs 9210 ns where the flat combine cost 19133).
**What binds now, and what I did not do.** The decode profile is a LATENCY CHAIN, not a bandwidth
wall: on the final timeline the member core spends 2.5 µs waiting for its first input tile, 0.8 µs
on the statistics, ~3 µs waiting for the combine round, 1.3 µs on the apply and ~1 µs draining. Two
terms inside the combine are now the largest single items and both are hop/latency, not arithmetic:
the level-2 rendezvous (~1.1 µs for one 4 KiB tile plus a semaphore) and the root's **finalize
chain** — `CopyTile → MulUnary → AddUnary → Rsqrt` on ONE fp32 tile — at ~1.3 µs, i.e. ~1.3 µs of
SFPU work to produce 32 useful numbers (a REDUCE_ROW result is column-0-valid, so 31/32 of every
stat tile is wasted work). The next lever I would try is a **narrower stat payload** (bf16
`cb_stat_gather`, design lamp P6 — halves every hop's bytes and the tile-add cost, at a PCC risk the
`row_reduce_accumulate` note explains) or moving the finalize off the root by broadcasting the SUM
and letting each core finalize in parallel (it is ~1.3 µs on the root's critical path today, and
would become ~1.3 µs on every core's post-mcast path, so it only wins if the apply can overlap it).
I did not take either: both are new schemes with their own precision/serialization questions, and
this phase's named lever (the combine topology) is landed and measured.

### [ ] Refinement 4 — Speed up the prefill (bandwidth-bound) profiles

**Type**: perf

**Goal**: the four `group="perf"` interleaved prefill cases — `(1, 1, 8192, W)` for
`W ∈ {1024, 2304, 5120, 7168}` with `achievable_ns` = 96744 / 211345 / 738307 / 1032281 — at their
exact config (bf16 / TILE / INTERLEAVED / gamma bf16 TILE / `fp32_dest_acc_en=False` / HiFi2).
These are aggregate-DRAM-bandwidth cases (the widest moves 33.5 MB), so the relevant
`ttnn/ttnn/operations/examples/master.md` patterns are the ones that raise achieved bandwidth and
reduce per-block overhead rather than add cores: `double_buffer` (bytes in flight per barrier,
buffer depth), `compute_block_size` (coarser blocks amortize init/reconfig; the granularity floor is
whole tiles, and coarser amortizes up to the L1 bound), and `noc_placement`. Co-tune
`block_row_tiles` against `input_cb_depth` / `output_cb_depth` — they trade the same L1 — and check
the tile-row imbalance the selection leaves (256 tile-rows over 110 cores is 3-vs-2). No SUPPORTED
change.

**Done when**: measured device-kernel ns improves on at least the two most-impacted prefill shapes
at the config above, with the golden suite green and no regression across the same config-spanning
guard set as Refinement 3.

**Verifier notes**: two facts to start from rather than re-measure. (1) `MIN_PIPELINE_BLOCKS` was
already swept 1/2/3/4 on these shapes at `fp32_dest_acc_en=True` and came out **flat** (±2%) — the
wall was aggregate DRAM bandwidth (~330 GB/s on the widest case), so a *pure* block-count change is
predicted to do nothing until the DRAM floor moves; re-measure at `fp32_dest_acc_en=False`, where
DEST capacity doubles and the selected `R` changes. (2) The writer's TILE-path drain is one
`noc_async_write_barrier` per **tile-row** (`core_w` tiles), which is inside `double_buffer`'s
measured 4–8-tile sweet spot for the perf shapes (`C` = 11–112) but **below** it for narrow-hidden
shapes (`C` = 2–3 on e.g. `(99991, 64)`, `(1,1,3232,96)`); batching several tile-rows behind one
barrier requires raising `output_cb_depth`, which is one host constant. That is L1-for-overlap, so
measure it against the block-size co-tune rather than stacking both blind.

### [ ] Refinement 5 — Speed up the sharded perf geometries

**Type**: perf

**Goal**: the five `group="perf"` **sharded** loose cases, whose `achievable_ns` are that exact
geometry's measured latencies and are 2–20× tighter than their interleaved twins:
`(1,1,32,1024)` WIDTH_SHARDED `[32,128]` on `(8,1)` → 4110 ns; `(1,1,32,2304)` WIDTH `[32,256]` on
`(9,1)` → 4617 ns; `(1,1,32,5120)` WIDTH `[32,160]` on `(8,4)` → 5267 ns; `(1,1,32,7168)` WIDTH
`[32,256]` on `(7,4)` → 5481 ns; `(1,1,8192,1024)` BLOCK_SHARDED `[1024,128]` on `(8,8)` → 25640 ns.
Same fixed config as the other perf cases (bf16 / TILE / `fp32_dest_acc_en=False` / HiFi2), with the
shard spec pinned by `extras`. The lever set is the zero-copy resident-shard path Refinement 2
builds plus the combine topology (`tensix_all_reduce`: 28–32-core 2-D groups under contention are
exactly the regime where its measured grid-two-stage reducer beats flat root-gather); `noc_placement`
applies because a sharded input removes the DRAM read entirely and leaves the NoC budget to the
combine. No SUPPORTED change.

**Done when**: measured device-kernel ns improves on at least three of the five flagged sharded
shapes at their pinned geometries, the golden suite is green, and no regression across the
config-spanning guard set (which by this phase must include one cell per sharded scheme).

**Verifier notes**: hard dependency on Refinement 2 — these cells cannot even run before it, and the
`achievable_ns` here are config-matched to the *sharded* placement, so they must never be compared
against an interleaved measurement. Expect the binding cost after the DRAM read disappears to be the
per-block combine round (~1.3–1.5 µs measured for a 1-tile payload on a 16-core group in
`tensix_all_reduce`) against a 4–5.5 µs total budget: at these targets the combine *is* the op, which
is why this is a separate phase from Refinements 3 and 4 rather than a continuation of them.
