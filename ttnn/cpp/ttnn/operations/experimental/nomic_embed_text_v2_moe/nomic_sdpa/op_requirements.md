# Operation Requirements: nomic_sdpa

## Definition
- **Formula**:
  `out[b, 0, i, 64h + d] = Σ_{j < L_b} softmax_j(scale · q[b,h,i,:]·k[b,h,j,:]) · v[b,h,j,d]`,
  where `L_b = key_lengths[b]`, or `S` when `key_lengths` is None.
- **PyTorch Reference**: `reference(q, k, v, scale, key_lengths)` in `eval/golden_tests/nomic_sdpa/helpers.py`
  (float64; keys `j ≥ L_b` masked to −inf; heads concatenated into (B, S, 768)).
- **Import Path**: `from ttnn.operations.nomic_sdpa import nomic_sdpa`
- **Function Signature**:
  ```python
  nomic_sdpa(query, key, value, *, scale: float = 1.0, key_lengths: list[int] | None = None,
             memory_config: ttnn.MemoryConfig | None = None,
             compute_kernel_config: ttnn.ComputeConfigDescriptor | None = None) -> ttnn.Tensor
  ```
  q, k, v: (B, 12, S, 64) bf16 TILE interleaved (L1 or DRAM). Output: (B, 1, S, 768) bf16 TILE. Also exports
  `default_compute_kernel_config()` and `circular_buffer_bytes(batch, seqlen)`.

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

> **Queue shape (verifier)**: TARGET − SUPPORTED is `{mask_mode: key_padding, alignment: ragged}`. Both are the same
> knob-turn mechanism (a per-segment valid key extent plus a boundary mask), so they form **one** generality bundle
> (Refinement 1). The perf-focus contract (8x512, bf16, `buffer=l1`, `mask_mode=none`, tile_aligned, default compute
> config, 176 KB of L1 reserved) is **already fully in SUPPORTED**, so no generality refinement is needed to anchor
> perf 1. With generality exhausted after R1, the remaining slots are all perf: R2–R4, ordered hardest-first (two
> T3 restructures, then the knob co-tune measured against the final structure). Phase 0 is 1.8–2.6× slower than the
> incumbent on every perf shape, so the perf entries are the op's main outstanding debt.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED buffer**: [l1, dram] (q, k, v and the output, interleaved)
- **SUPPORTED shape-derived axes**: alignment = tile_aligned only
- **SUPPORTED op-specific axes**: mask_mode = none only (`key_lengths` given → `UnsupportedAxisValue`)
- **Domain refusals** (SupportRefusal → stock SDPA): heads ≠ 12, head dim ≠ 64, S > 512, B·S_t·32 > 4096, grid
  ≠ 11x10, scale not finite or ≤ 0, LoFi, bf16 DEST + approx exp at S > 384, CBs not fitting below live L1.
- **Cores**: up to 110. Regime "row-split, per-core K/V read": contiguous balanced query-tile-row ranges from a
  common-RT table indexed by core; K/V of each (b, h) pair are read once per segment.
- **Compute config**: default HiFi2, **fp32 DEST**, approximate exp (unclamped + pack ReLU), half-sync. Every CB is
  bf16.
- **Golden baseline**: 45 / 90 (45 supported_pass, 40 xfail_expected, 5 regression self-xfails; 0 in every loud
  category).
- **Perf (device µs, L1)**: 8x512 ≈ 130 (incumbent 72.1, target 58); 4x512 ≈ 79 (38.2); 2x512 ≈ 54 (24.5).
- **Precision**: real rel-RMS 0.0255 at S = 512 against a bar of 0.0265.

### [x] Refinement 2 — Read K/V once per pair and multicast it to the cores sharing that pair

**Type**: perf

**Goal**: the perf-flagged `feature_spec.LOOSE_CASES` entry is **8x512**: bf16, `buffer=l1`, `mask_mode=none`,
profile real, `l1_reserved_kb=176`, default compute config. Its targets are ≤ 58 µs acceptance (62.6 µs model gate),
from `achievable_ns = 72100` and `minimum_expected_speedup = 1.243` at 1350 MHz, with the golden real-profile
precision bar as the soft precision gate. It runs ~130 µs today.

The binding data-movement cost is aggregate NoC bytes: the implementer's ablation found ~450 GB/s chip-wide, the same
for L1 and DRAM placement. K and V are each read 2.00× / 3.15× / 4.88× the minimum at 8x512 / 4x512 / 2x512, because
every core holding rows of a (b, h) pair reads that pair's whole K and V.

The lever is the deferred regime **row-split + pair multicast** (op_design.md § Regimes, § Dataflow "Deferred
multicast contract"). It is a scheme-change:
- Cores sharing a pair are consecutive in `core_row_start` and land K/V at identical CB addresses, so each group's
  first core reads K/V once and multicasts them into the group's reserved `cb_key_t` / `cb_value_aug` slots.
- Use `ttnn.Mcast` + `McastArgs` (handshake; `ChainUnicast` for irregular group sizes).
- Relevant catalog entries in `ttnn/ttnn/operations/examples/master.md`: `shared_input_reuse`, `mcast_topology`.

The io-boundary bytes this should save: 36 → 24 MiB at 8x512, 24.9 → 12 MiB at 4x512, 17.6 → 6 MiB at 2x512. The
guards (2x512 / 4x512) are DM-dominated and should gain most. No SUPPORTED change.

**Verifier notes**:
- **This entry exists because of the L1 ledger audit.** The design deferred this regime on a "per-core ingress is
  byte-identical" premise that the implementer's measurement falsified (`l1_ledger.md` § Measured data movement,
  § Verifier audit).
- **Validate by building and measuring.** Build the multicast variant and measure device-ns; do not infer its value
  from the remove-one-stage ablation (a broadcast's benefit can be under-shown by one).
- **Composes with R1.** Each segment multicasts only `key_tiles_valid` rows.
- **Watch the first-segment exposure.** Every core loads its first segment at t = 0, chip-wide. Multicast shrinks
  that burst; sender ordering or staggering is the implementer's call.
- **Keep both placements.** DRAM placement (`buffer = dram`) is in SUPPORTED and must keep working; it gains the same
  byte cut.

**Done when**: the 8x512 perf-flagged loose case improves measurably in device-ns, its real-profile precision gate
still holds, the golden suite is green, and there is no regression across the config-spanning guard set, one
representative per kernel path × placement:
- L1 and DRAM;
- mask none and key_padding;
- tile-aligned and ragged;
- 4x512 and 2x512;
- a multi-segment shape such as 32x128.

**Outcome**: built (row-split + pair multicast; K/V read once per sharing group, multicast forward on NOC0, ring slot
`p % kv_depth`; `kv_depth` cap 1 on L1 / 2 on DRAM with a live-L1 fallback; off below `S_t = 8`). Device us, median of
3, Phase 0 → now: L1 2x512 59.1 → 30.5, 4x512 76.0 → 53.8, 1x512 49.5 → 17.8, 8x384 82.0 → 74.3; DRAM 2x512 60.7 →
34.4, 4x512 86.2 → 63.6, 8x512 134.7 → 117.8. **The perf-flagged L1 8x512 case did not move** (7-sample medians
120-127 us for every variant, Phase 0 122): the 176 KB-reserved case leaves 914 KiB, so only the 1-deep ring fits and
its pair groups are singletons there, and even the 2-deep ring (no reservation) measured 120.3 — 8x512 is bound by
the t = 0 first-segment burst (compute's first Q + K wait ends at 38.5 us median / 54 us max; ~9 MiB of K + Q chip-wide
on NOC0) plus a 74 us compute floor, and the multicast only removes second-segment bytes that already overlapped
compute. Next: shrink the burst — key-chunked K delivery (R4 L1) or the t = 0 reads split across both NoCs (catalog
`split_reader`, the writer's NOC1) — and the compute floor (R3). Not done here: both are separate levers outside this
heading.

### [x] Refinement 3 — Overlap the SFPU exp with the FPU matmuls (row-group interleave)

**Type**: perf

**Goal**: same perf-flagged target as Refinement 2 (8x512 loose case toward ≤ 58 µs, with the same soft precision
gate), now on the compute side. The ablation's compute floor is ~73 µs at 8x512, about 2× the 34.9 µs speed of light.

The verifier's measurements say the SFPU exp is a large share:
- the exact exp adds ~60% device time;
- fidelity (HiFi3 ≈ HiFi2) and DEST capacity (fp32 full-sync, 8 tiles) do not move it.

`probs_block` runs block-sequentially after `qk_block`. Lever: op_design.md perf lamp **L2**, a row-group interleave.
Query rows are independent, so row group i's `exp(scale·(s − m))` can run on the pack thread
(`exp_packthread_tile`, as in the incumbent's `compute_streaming.hpp:625`) while the FPU computes Q·K^T (or P·V) of
row group i+1. Relevant catalog entries: `compute_fusion` (which engine consumes the result), `compute_block_size`.
One T3 lever, so this is a whole phase. No SUPPORTED change.

**Verifier notes**:
- **Ordering.** This lands after R2 so the compute restructure is measured with data movement no longer dominating.
  It is independent of R2's reader topology.
- **Row groups are a schedule inside the per-core block, not a smaller block.** The CB live sets stay the whole
  block. Any new intermediate buffer needs its `l1_ledger.md` row.
- **Precision is the risk.** Keep the exp in the incumbent's numeric class: the approximate exp with the pack ReLU,
  bf16 P, and the row sum from the same matmul as O. The real-profile margin at S = 512 is only 4% (0.0255 against
  0.0265); re-measure all three profiles at S = 512.

**Done when**: measured device-ns improves on the 8x512 perf-flagged case, all three profiles still meet
PROFILE_TOLERANCES at S = 512, the golden suite is green, and there is no regression on the same guard set as
Refinement 2.

**Outcome**: built (row-group schedule, G = 1, exp on the pack thread, each probs window followed by a P·V window of
the previous group, then Q·K^T of the next; separate 2-group scores / P rings; `pack_sfpu_fence` before the math-thread
recip; gated to `q_block_rows ≥ 3` and the approximate exp). 8x512 L1 128.2 → 111.7 us (median of 7, A/B on
`EXP_OVERLAP`); 8x384 79.3 → 69.9, 16x256 66.0 → 56.7, 32x128 38.2 → 34.1, DRAM 8x512 118.2 → 114.4, full-sync
134.2 → 119.9; no guard slower beyond noise. Precision is unchanged (real rel-RMS 0.0255 at 8x512). The per-core CBs
halve (834 → 410 KiB at 8x512). The compute floor moved only 74.0 → 72.3: the exp is mostly hidden (13.6 → ~6.7 us
visible), but per-window re-inits (`matmul_block_init`, `sub_bcast_cols_init`, pack-ReLU toggles) cost ~5 us. The wall
gain is mostly from the output now streaming row by row. 8x512 is still bound by the t = 0 first-segment read burst
plus a ~72 us compute floor. Next: a cheaper per-window matmul re-init (the incumbent's replay-buffer
`mm_no_mop_reinit_short`; experimental raw LLK outside kernel_lib, so not taken here) and `kv_depth = 2` at L1, which
the halved footprint now makes room for (Refinement 4, L6).

### [x] Refinement 4 — Block, buffer-depth and cheap-lever co-tune on the restructured kernel

**Type**: perf

**Goal**: finish the 8x512 perf-flagged case toward ≤ 58 µs (stretch 48 µs), and the guards toward their
acceptance targets (4x512 ≤ 30 µs, 2x512 ≤ 18 µs), with several T1/T2 levers on the surface the planner exposed.
Measure each lever on its own; keep the ones that win.
- **L6 `kv_depth = 2`**: hide the next segment's K/V. It runs correctly today, with no wall win while math bound.
- **L1 key-chunked K delivery**: start Q·K^T on the first K chunk.
- **L7 pair-aligned `core_row_start` table**: fewer K/V segments, the same kernels.
- **L9 / catalog `bank_stagger`**: rotate each core's K/V page-issue start within its pair group.
- **L5 normalize in the P·V `matmul_block` `PostComputeFn`**: the recip of the sum tile and the multiply of the two O
  tiles on DEST. This removes `cb_pv_out` and its round trip (the L1 ledger fold-in).
- **L8 recip with `VectorMode::C`**: column 0 only (catalog `sfpu_tile_scope`).

Use `ttnn/ttnn/operations/examples/master.md` (`double_buffer`, `trid_double_issue`, `bank_stagger`,
`sfpu_tile_scope`, `compute_fusion`) to bound the search. Chunks stay whole tiles at minimum, and coarser chunks
amortize fixed costs. No SUPPORTED change.

**Verifier notes**:
- **Last on purpose.** These knobs are best tuned against R2's delivery topology and R3's compute schedule; tuning
  them first would be re-done.
- **L1 headroom.** At 8x512 with 176 KB reserved, ~1055 KiB is free at the call and the footprint is 834 KiB.
  `kv_depth = 2` takes it to 994 KiB, so check `circular_buffer_bytes()` against the reserved-L1 loose case.
- **Precision gates.** L4 (row sum via `reduce<SUM>`) and L3 (the exp form) are precision-sensitive and stay out of
  this phase unless re-measured on all three profiles. L4 loses the HiFi2-truncation cancellation between O and l.

**Done when**: measured device-ns improves on the 8x512 perf-flagged case (and on 4x512 / 2x512, which must end no
slower than the incumbent's 38.2 / 24.5 µs), the precision bar holds on all three profiles at S = 512, the golden
suite is green, and there is no regression across the guard set of Refinement 2.

**Outcome**: device us, L1, 176 KB reserved, default config, median of 9, all R4 knobs off → on: **8x512 115.1 →
99.8**, 4x512 53.0 → 49.3, 2x512 30.1 → 27.8. No guard is slower beyond noise across 8 shapes × 4 configs × 2
placements. Precision is unchanged (real rel-RMS 0.0255 / 0.0251 at 8x512 / 2x512).

Device zones showed the wall was startup plus compute in series: compute's first V landed at 46 us, behind each core's
whole Q block. Levers:
- **Kept**:
  - Q lead rows (`Q_LEAD_GROUPS = 2`, the chunked-delivery lever on the operand that gated): first K 31.5 → 18 us,
    first V 46 → 35 us;
  - `Q_DEPTH_L1 = 2` (DRAM stays 1: 16x256 DRAM was slower at 2);
  - L7, a pair-aligned table, path-gated to no larger per-core range (2x512, 1x512): 2x512 −1.4 us;
  - L8, a column-0 recip: compute floor −0.3 us.
- **Parked as live knobs**:
  - L6 `kv_depth = 2`: 8x512 103.6 → 109.1. It puts a store-and-forward multicast hop on every core's first K / V;
  - L9 bank stagger: flat.
- **Not built**:
  - L1 K chunking: V, not K, gates compute's start, so chunking K would only overlap row 0's Q·K^T, which already
    finishes ahead of the V wait;
  - L5: the P·V windows are the pack-thread exp's partners (R3), so a math-thread recip / mul in their `PostComputeFn`
    would contend for the shared SFPU mid-exp. That is the reprogramming hazard `pack_sfpu_fence` exists for.

What binds 8x512 now:
- compute starts at full rate only when the first-segment K + V burst (~12.6 MiB chip-wide, ~35 us) lands;
- then the ~72 us compute floor runs (the speed of light is 34.9 us).

Next, not done here because each is a separate lever outside this heading:
- issue half of the t = 0 K / V reads on NOC1 (catalog `split_reader`; all reads ride NOC0 today, at ~400 GB/s);
- deepen the row-group pipeline so more Q·K^T + exp work runs before the first P·V needs V;
- a cheaper per-window matmul re-init (R3's finding).

### Deferred (supervisor: until the op is faster than stock SDPA) — Key padding and ragged sequence lengths (boundary-masked key extent)

**Goal**: add `"key_padding"` to `SUPPORTED["mask_mode"]` and `"ragged"` to `SUPPORTED["alignment"]`, with native
in-kernel support, following op_design.md § "TARGET axes against the scheme":

- **Valid key extent.** Each segment gets a runtime `key_tiles_valid = ceil(L_b / 32)`, where `L_b = key_lengths[b]`
  under key_padding and `L_b = S` for every sequence under ragged. This extent narrows N of `qk_block`, the
  reduce width, the exp width, and K of `pv_block`.
- **Reads.** The reader reads only `key_tiles_valid` K/V tile-rows. Pushes stay nominal (ring discipline).
- **Boundary mask.** The boundary score column gets an additive mask before `row_max_block`: 0 for columns
  `< L_b mod 32`, a large negative finite value elsewhere. Realize it in the `qk_block` `PostComputeFn` on DEST, or as
  a chain with real CB credit.
- **Ragged geometry.** `S_t = ceil(S/32)` everywhere. Padded query rows read the producer's zero q rows and must
  come out finite.

The cells to move are all 40 `xfail_expected` golden cells, plus `test_key_padding[2-512|8-512|3-400]`,
`test_key_padding_lengths_are_per_call` and `test_finite_padding_rows_of_a_ragged_sequence` in `test_regression.py`.

**Implementation skill**: /memory-layouts

**Verifier notes**:
- **Scope of the skill.** Only the skill's non-aligned (last-tile mask) section applies. Layout stays TILE; no
  ROW_MAJOR leg.
- **The prompt's conditional MUST rules arm here:**
  - padded keys take **exactly zero** weight in the max, the exp, and both P·V columns (O and the row sum);
  - padded query rows stay finite;
  - `key_lengths` are common runtime args rewritten by `bind_call` on **every** call. Only `mask_mode` goes in the
    cache key, never the lengths.
- **Exact zero under the exp.** The default exp is the unclamped fast form with `PackRelu::Zero`, which zeroes very
  negative arguments (and so does exact zero). Verify the masked weights are exactly 0 with the peaky-profile
  key_padding tests. Masked keys also need `L_b ≥ 1`, so no row is all-masked (the max stays finite).
- **Build it performantly** (it extends the perf-focus kernels): don't read masked K/V tiles (that is a byte saving
  on the binding resource), and keep tile-aligned / mask-none shapes on their current code path with no added
  per-tile work. The 8x512 loose case must not regress.
- **Subblocks for prime extents.** Runtime `key_tiles_valid` can be prime (S = 400 gives 13). Derive matmul subblock
  widths at runtime (the divisor rule, or the helper's `last_in1_subblock_w_valid` narrowing per op_design.md §
  Mechanism caps), with CB capacities nominal in `key_block_tiles = S_t`.
- **Precision.** Ragged shapes with real S ≤ 512 must still meet PROFILE_TOLERANCES on all three profiles.
- **Placement in the queue.** This lands before the perf refinements. R2's multicast then carries only
  `key_tiles_valid` rows per segment, so the two compose; design the reader so the K/V delivery range is a single
  per-segment value.

**Done when**: `verify_supported` reports the 40 former `xfail_expected` cells as `supported_pass` (0 loud
categories); the 5 regression tests above pass; acceptance + precision baseline are green; and the 8x512 / 4x512 /
2x512 loose-case device time does not regress.

## Perf leads (supervisor notes, 2026-11-05, after Refinement 4)

- Where it stands at 8x512 with q, k, v in L1: 99.8 us a call against stock SDPA's 72.1; the acceptance bar is 58 and
  the model keeps the op only at 62.6 or less. 4x512 and 2x512 must not be slower than stock (38.2 / 24.5).
- Per core: this op gives a core 14 query tile-rows at 8x512 and its compute alone took about 72 us after Refinement 3,
  about 5.2 us a tile-row. Stock SDPA gives each of 96 cores one (sequence, head) pair of 16 tile-rows and finishes in
  72.1 us with its data movement included: about 4.5 us a tile-row all in. 58 us at 14 rows is about 4.1 us a tile-row
  all in: the compute per tile-row has to get well below the incumbent's, not only the work split better.
- Precision path: stock keeps a bfloat16 DEST at HiFi2 and is inside the golden bar on every profile (the bar is its
  own error plus a margin). It accumulates the softmax row sums in fp32 by packer L1 accumulation, outside DEST
  (ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_streaming.hpp, around lines 625-660).
  This op moved to an fp32 DEST because its row sum (the ones column of P [V | 1]) accumulates in DEST. A bfloat16
  DEST with the row sums accumulated in fp32 elsewhere may get back the bfloat16-DEST speed (twice the DEST tiles a
  sync, cheaper packs) within the bar; measure the bar on all three profiles first.
