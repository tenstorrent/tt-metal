# Operation Requirements: g6_typecast

## Definition
- **Formula**: `out[i] = conv_{src→dst}(x[i])` for every logical index `i` — round-to-nearest-even where float
  bits are dropped, truncate toward zero float→integer, int32→bfloat16 rounded through float32, uint32↔int32
  keep all 32 bits; bfloat8_b block-quantized (16 values of a tile row share an exponent).
- **PyTorch Reference**: `torch.Tensor.to(torch_dtype(dtype))` (`eval/golden_tests/g6_typecast/helpers.py::pytorch_g6_typecast`;
  bfloat8_b: exact float32 values, rendered by the host quantizer for grading).
- **Import Path**: `from ttnn.operations.g6_typecast import g6_typecast`
- **Function Signature**: `g6_typecast(input_tensor: ttnn.Tensor, dtype: ttnn.DataType, *, sub_core_grids: ttnn.CoreRangeSet | None = None) -> ttnn.Tensor`
  (output keeps the input's shape, layout and interleaved memory config; one device program per call).

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.
> **Prompt rules** (`eval/prompts/g6_typecast.txt ## Rules`) are hard constraints in every refinement: EXCLUSIONS stays empty, no size/width/page-size axis, no ttnn conversion/layout op at the entry point, each tensor crosses DRAM once, one device program, 32-bit values converted with all 32 bits.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED output_dtype**: [float32, bfloat8_b, int32]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED buffer**: [dram, l1]
- **SUPPORTED alignment**: [tile_aligned, w_non_aligned, h_non_aligned]
- **SUPPORTED rank**: [3, 4]
- **Regime**: R1 `tile_stream` — contiguous flat-tile ranges over the full grid (`split_work_to_cores`, row-wise), `block_tiles` = whole per-core range clamped to the 1347776 B L1 grant, `buffer_depth` knob 2 (effective 1 for every production shape)
- **Compute config**: `fp32_dest_acc_en` for 32-bit destinations; `bfp8_pack_precise` for bfloat8_b; no user-facing compute config (the spec has none)
- **Golden baseline**: 42 supported_pass / 27 xfail_expected / 0 loud (44/71 incl. the 2 untagged regression tests)

### [x] Refinement 1 — ROW_MAJOR regime (R2 `rm_pseudo_tile_stream`) with its conversion arms

**Goal**: add `ttnn.ROW_MAJOR_LAYOUT` to `SUPPORTED["layout"]`, `1, 2` to `SUPPORTED["rank"]`,
`ttnn.uint32, ttnn.int32` to `SUPPORTED["dtype"]`, and `ttnn.uint8, ttnn.uint32` to `SUPPORTED["output_dtype"]`
— natively in-kernel, per the design's deferred regime R2: the reader gathers row-major pages densely into L1,
compute converts each 1024-value run as a pseudo-tile (same intra-tile permutation on unpack and pack, so an
elementwise op preserves positions), the writer scatters rows back. Conversion arms landed here:
bf16→uint8 (truncation; F317/F322), uint32→int32 and int32→uint32 (all 32 bits kept; Q2189…Q2234 RM, F321
rank 1, and the TILE case Q2230 which becomes supported per-axis and must pass on the R1 path with the same arm).
32-bit sources need `UnpackToDestFp32` on the input CB (SRCA would cut them to 19 bits).
Unlocks: Q2189, Q2196, Q2199, Q2202, Q2205, Q2208, Q2211, Q2214, Q2220, Q2223, Q2226, Q2234, Q2230, F317,
F321, F322 and the two loose cases (F322, F317).

**Implementation skill**: /memory-layouts, /numeric-formats-metal, /interleaved-parallel

**Verifier notes**:
- First because it is the hardest generality axis (layout tier) **and** it carries the full contract of perf
  Refinement 3's target (F322: bf16 → uint8, ROW_MAJOR, DRAM, w_non_aligned, rank 4). After this refinement the
  F322/F317 loose cases must be supported and passing — not xfail, not excluded. The RM arms ship here because
  no RM golden cell uses a Phase 0 arm (a true dependency, not a convenience).
- Build it performantly — Refinement 3 optimizes exactly this path. F322 has 4 194 304 input pages of 6 B
  (3 B out); a per-row `noc_async_read_page` + per-row write is transaction-bound by orders of magnitude.
  The reader/writer must move **bank-contiguous runs** of pages (consecutive pages of one bank are adjacent at
  the aligned-page stride) and compact/expand them in L1, with `block_tiles` (pseudo-tiles per block, the
  existing single-source knob) setting rows per block; the split stays the full grid, contiguous page ranges
  (input and output share the page→bank map). Batch both halves; one barrier per block.
- Each tensor crosses DRAM once; no `ttnn.to_layout`/`tilize`/`untilize` wrapper. If the native RM path is
  out of reach in one pass, a wrapper is NOT an acceptable partial-tick here (prompt rule: MUST NOT) — partial-tick
  by deferring named axis values instead and file `Refinement 1b`.
- Regime selection by `layout` (host-checkable) must be reported in `last_plan["regime"]`, and the acceptance
  test must pin one TILE and one ROW_MAJOR case (design § Regimes).
- `validate()`: after this refinement the per-axis product admits pairs with no arm (e.g. uint32→uint8);
  reject those on the host with `ValueError` or implement the arm — never run a wrong arm, never use EXCLUSIONS.

**Done when**: every golden cell listed above passes (`supported_fail = 0`), including both loose cases, with
0 xpass_drift / xfail_wrong_mode; Phase 0 cells still pass.

### [x] Refinement 2 — 32-bit and 16-bit sources on TILE: float32 / int32 / uint16 sources, bfloat16 destination

**Goal**: add `ttnn.float32, ttnn.uint16` to `SUPPORTED["dtype"]` and `ttnn.bfloat16` to
`SUPPORTED["output_dtype"]`, with the TILE-path arms float32→bfloat16 (RNE, exact ties: F315, F316, F318,
F319, F320, F323, F324 incl. w_non_aligned), int32→bfloat16 (rounded through float32 — the two-rounding
witnesses 33685505, −33685505, 2²⁴+2¹⁶+1; Q2228) and uint16→int32 (Q2231). Input-CB `UnpackToDestFp32` for
32-bit sources; output-CB format = destination; `fp32_dest_acc_en` per the arm (32-bit DEST wherever a 32-bit
value sits in DEST).

**Implementation skill**: /numeric-formats-metal

**Verifier notes**: precision-knob tier — after the RM regime so the arms are added once to a kernel whose
structure no longer changes. Zero data-movement change expected: the R1 `tile_stream` reader/writer already
take page size and format from the tensors. The int32→bf16 LLK arm (`typecast.h`) must be checked against the
two-rounding witnesses — a single RNE from int32 is wrong for them (prompt rule). Same pair-gating note as R1.
No new EXCLUSIONS (prompt rule).

**Done when**: F315, F316, F318, F319, F320, F323, F324, Q2228, Q2231 pass bit-exact; golden suite has
`xfail_expected = 0` and 0 in every loud category; prior cells still pass.

### [x] Refinement 3 — Speed up the perf-flagged profile (F322, with F317)

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES` flags F322 — bfloat16 → uint8, ROW_MAJOR, DRAM interleaved,
`[1, 2048, 2048, 3]`, gate `exact`, no compute config — as the mandatory perf target: faster than stock ttnn
(14563.31 µs) and ≤ 1.35 × the 81.92 µs DRAM roofline = **110.59 µs** device kernel duration. Optimize this
exact config (the R2 path built by Refinement 1) using the relevant patterns in
`ttnn/ttnn/operations/examples/master.md` — `read_coalescing` (6 B / 3 B pages: bytes per transaction is the
whole game), `bank_stagger` / `noc_placement` (bank-local page runs per core), `trid_double_issue` /
`double_buffer` (keep reads in flight), and a `block_tiles` × `buffer_depth` co-tune (coarser blocks amortize;
whole pseudo-tiles minimum). F317 (`[1, 1024, 1024, 3]`, same regime, flagged p2) is the same lever — measure
it alongside. No SUPPORTED change.

**Done when**: measured device-ns on F322 improves toward ≤ 110.59 µs (and F317 improves), the exact gate still
holds, the golden suite is green, and no regression across the config-spanning guard set (one representative
per distinct kernel path × layout × placement: R1 TILE DRAM, R1 TILE L1, R2 ROW_MAJOR DRAM; one per arm family).

**Outcome**: device kernel ns (Blackhole, 110 cores, one fresh run each): F322 2 320 µs → **1 562 µs** (1.49×),
F317 958 µs → **361 µs** (2.65×); exact gate holds; TILE guards unchanged (Q2185 35.9 µs, P39 68.9 µs). Levers, all
in the tree: (1) word-funnel row movers (`kernels/g6_typecast_rm_rows.hpp`, reader compaction and writer expansion
alike), (2) `STAGING_DEPTH = 2` double staging so the bank-run DMA overlaps the row moves (reader and writer;
alone at the default block it was 1 673 µs vs 1 780 µs; at the final block 1 562 µs vs 1 593 µs), (3)
`RM_BLOCK_TILES = 2`, from a sweep of 1/2/3/4/None that measured 1 568/1 572/1 604/1 631/1 673 µs on F322 (shorter
pipeline fill and drain). F322 ablation: the row moves alone take 782 µs and the NoC alone 1 476 µs, so the op
is now **NoC/DRAM-bound**. Each 6 B / 3 B RM row is its own 64 B-aligned DRAM page, so F322 moves 4 194 304 ×
(64 + 64) B = 537 MB physical, not the 36 MB the 81.92 µs roofline assumes. At the BH DRAM peak (~512 GB/s)
the floor is ≈1.05 ms, so the 110.59 µs target is unreachable for an interleaved RM tensor with this page
geometry; 1 562 µs ≈ 344 GB/s of physical traffic. Next, not tried: read only the needed 32 B
granule of each slot / write only a 32 B partial slot. That needs per-row NoC transactions (≈38 K per core),
which the transaction issue cost makes slower. Trid multi-block issue is not needed: one ≈43 KB block per
core in flight already exceeds the per-core bandwidth Little's law asks for at this aggregate rate.

### [x] Refinement 4 — Speed up the flagged TILE profiles (Q2185, P39): read/convert/write overlap

**Type**: perf

**Goal**: flagged loose cases p3 Q2185 (bf16 → float32, TILE, DRAM, `[16, 4096, 12]`, exact) and p4 P39
(bf16 → bfloat8_b, TILE, L1, `[1, 128, 256, 256]`, bfp8_group) run on R1 `tile_stream` with one block per core
(the default block is the whole per-core range ⇒ effective depth 1 ⇒ read, convert, write serialize on each
core). Co-tune the existing knobs `BLOCK_TILES_CAP` (smaller block, several blocks per core) × `BUFFER_DEPTH`
(2–3) per the design's "Overlap" perf lamp and master.md `double_buffer` / `compute_block_size`; also measure the
design's bank-affine/staggered assignment (`bank_stagger`, `noc_placement`) and `UnpackToDestFp32` on the
no-SFPU arms. All are ⭐/⭐⭐ knob-turns on surfaces the planner already exposed — several fit one phase. If a
cap wins, derive it in `plan_blocks` (single source), not as a literal in the kernels. No SUPPORTED change.

**Done when**: measured device-ns improves on Q2185 and P39 with their gates intact, the golden suite is green,
and no regression across the config-spanning guard set (R1 TILE DRAM/L1 per arm family, R2 ROW_MAJOR DRAM, and a
decode shape such as Q2191/Q2227 where a smaller block cannot help).

**Outcome**: the R1 block is now capped at `TILE_BLOCK_TILES = 4` (in `plan_blocks`), with depth 2. Device-kernel ns
(BH, 110 cores, one fresh run each): P39 69.5 → 17.3 µs (4.0×); Q2185 36.6 → 35.2 µs (≈4%, near the noise band).
Guards: Q2187 7.6 → 7.5, i32 [8,1024,256] DRAM 38.8 → 35.3, f32 L1 [4,512,512] 9.3 → 8.4, Q2191 1.28 → 1.25 µs.
F322 1.55 ms and F317 ≈ 0.37 ms are unchanged (R2 is untouched).
Bottlenecks now:
- Q2185 is DRAM-bound. Ablation: writes alone 28.9 µs, reads alone 11.9 µs, both stubbed 1.5 µs. The 12.6 MB of
  physical traffic moves at about 366 GB/s; 8 MB of it is fp32 tile padding of W=12 that the TILE ABI fixes.
- P39 is NoC-congestion-bound on the L1 banks. More reads in flight makes it slower: trid window 2 → 19.9 µs, 3 → 25.2 µs.
  The 69.5 µs single-block run is the extreme of the same effect.
Measured and not kept:
- `READ_INFLIGHT` trid window: no win at 2–3. Parked at 1 as a live knob; that path issues the same reads and barriers
  as before.
- Swapping the NoCs (reads on NoC1, writes on NoC0): Q2185 53.0 µs, a big regression.
- `UnpackToDestFp32` on the bf16 source: flat at 35.6 µs (compute is off the critical path).
- `bank_stagger`: not implemented. Per-core tile counts (18/19, 74/75) are not multiples of 8 DRAM / 110 L1 banks, so
  core starts already fall on spread-out banks.
Next to try: on P39, pace the reads, for example by striding each core's walk across banks to reduce hot spots. On
Q2185, nothing short of fewer bytes.
