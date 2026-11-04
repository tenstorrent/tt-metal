# Operation Requirements: nomic_qkv_rotary_heads

## Definition
- **Formula**:
  - `qkv = x @ W^T + b`, accumulated in fp32.
  - Split into q, k, v (768 columns each), heads of 64, permuted to (B, 12, S, 64).
  - `q *= 1/8`.
  - NeoX rotate-half rotary (base 10000) on q and k, with `theta = s * 10000^(-2i/64)`:
    `out[..., i] = t[..., i]·cos θ − t[..., i+32]·sin θ` and `out[..., i+32] = t[..., i+32]·cos θ + t[..., i]·sin θ`.
  - v is not rotated.
  - Padding rows (s ≥ S) of q, k and v are exactly 0, by selection.
- **PyTorch Reference**: `reference_qkv` in `eval/golden_tests/nomic_qkv_rotary_heads/helpers.py` (float64):
  ```python
  def reference_qkv(x, weight, bias):
      batch, _, seqlen, _ = x.shape
      qkv = x.double().reshape(batch, seqlen, 768) @ weight.double().T + bias.double()
      q, k, v = (t.reshape(batch, seqlen, 12, 64).permute(0, 2, 1, 3) for t in qkv.split(768, dim=-1))
      cos, sin = rotary_cos_sin(seqlen)          # (S, 64): [c0..c31, c0..c31], [s0..s31, s0..s31]
      q = q * 64**-0.5
      return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin, v
  ```
- **Import Path**: `from ttnn.operations.nomic_qkv_rotary_heads import nomic_qkv_rotary_heads`
- **Function Signature**:
  ```python
  nomic_qkv_rotary_heads(
      input_tensor: ttnn.Tensor,     # (B, 1, S, 768) bf16 TILE, L1, in a supported placement; 32 <= B*ceil(S/32) <= 128
      weights: QkvWeights,           # prepare_qkv_weights(weight, bias, device)
      tables: RotaryTables,          # prepare_rotary_tables(S' >= S, device)
      *,
      memory_config: ttnn.MemoryConfig | None = None,          # L1_MEMORY_CONFIG (default) or DRAM_MEMORY_CONFIG
      output_tensors: list[ttnn.Tensor] | None = None,         # [q, k, v] preallocated in memory_config
      compute_kernel_config: ttnn.ComputeConfigDescriptor | None = None,  # fp32_dest_acc_en must be True
  ) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]
  ```
  Module exports: `prepare_qkv_weights`, `prepare_rotary_tables`, `head_dim_order`, `input_memory_config`,
  `default_compute_kernel_config`, `circular_buffer_bytes`, `INPUT_TAGGERS`, `SUPPORTED`, `EXCLUSIONS`, `validate`.

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED fp32_dest_acc_en**: [True]
- **SUPPORTED layout**: [TILE_LAYOUT]
- **SUPPORTED placement**: [ln_layout, op_layout]. `op_layout` is identical to `ln_layout` by construction, so the
  golden harness skips its cells.
- **SUPPORTED output_buffer**: [l1, dram]
- **SUPPORTED output_mode**: [new, preallocated]
- **SUPPORTED seq_alignment** (tagger): [tile_aligned, seq_padded]
- **Shape gate**: last dim 768, 32 ≤ M = B·ceil(S/32) ≤ 128, 11 x 10 grid.
- **Cores**: multi-core, `num_columns` x 9 = 81–99 cores. Regime `mcast2d_fused_rotary`:
  - x is gathered from the norm's ND shards by each column's top core and multicast down the column.
  - W (with the bias as K tile 24) is read once per row from DRAM, multicast west, and resident after M block 0.
  - K runs in 5 blocks with packer L1 accumulation.
  - q/k rotary runs as packer-thread SFPU on fp32 DEST.
  - The writer scatters each tile straight to its q/k/v page.
- **Compute config**: HiFi2 default (fidelity taken as given); fp32_dest_acc_en required; half-sync DEST forced.
- **Golden baseline**: 68 / 180 cells pass.
  - supported_pass 68: 59 golden incl. 3 LOOSE_CASES, 9 regression.
  - xfail_expected 56 (`model_layout`).
  - supported_skipped 56 (`op_layout` == `ln_layout`).
  - All loud categories 0.
- **Precision**: q/k PCC 0.999994 / relative RMS 0.00477; v 0.999994 / 0.00457 (gate 0.99998 / 0.0050).
- **Perf** (device kernel, 1350 MHz, ln_layout, q/k/v preallocated L1):
  - 28.96 µs at 2x512, 50.26 at 4x512, 93.07 at 8x512.
  - **Every focus shape is above its acceptance ceiling** (26 / 45 / 85 µs).

**Queue accounting.**
- TARGET − SUPPORTED = {placement: `model_layout`}. That is Refinement 1, the only generality candidate.
- The flagged perf target (LOOSE_CASES 8x512: bf16, TILE, ln_layout, output_buffer l1, preallocated,
  fp32_dest_acc_en=True, HiFi2) is **fully inside SUPPORTED already**. No unlock is needed before the perf slots.
- Once Refinement 1 lands, the generality candidates are exhausted. Refinements 2–4 are perf, all targeting the
  flagged shape. Each one names the region its ablation measured (verification_report.md § Performance baseline).

### [x] Refinement 1 — Model split layout placement (`model_layout`)

**Goal**: add `"model_layout"` to `SUPPORTED["placement"]`. This moves the 56 `xfail_expected` cells (14 INPUTS shapes
x {l1, dram} x {new, preallocated}) to passing. `model_layout` is the model's current split layout
(`models/experimental/nomic_embed_text_v2_moe/tt/common.py`: `sharded_norm_split`, `sharded_norm_configs`):
- Legacy `BLOCK_SHARDED` L1, `COL_MAJOR`, `shard_spec` set and `nd_shard_spec` **None**.
- Grid of 8 core rows x C core columns.
- Shard `(32·bh, 96)`, with `bh = ceil(M/11)` and `C = ceil(M/bh)`.
- Probed: 2x512 → shard (96, 96) on 11x8; 3x400 → (128, 96) on 10x8; 8x512 → (384, 96) on 11x8.

The op reads it natively with the existing gather: a host-side x source table for this shard geometry. The reader
kernel already handles any `x_part_tiles` / `x_num_parts`: here `kp = 3` and P = 8, so a 5-tile K block splits into at
most 3 runs per row. The table is 16 words per row, at most 196 RT words with `core_m_tiles ≤ 12`, which is under the
341-word budget.

**Verifier notes**:
- **Knob-turn** per op_design.md ("TARGET → scheme": host table builder only, kernels unchanged). It is the only
  generality item, so it comes first. The perf refinements then measure an op whose reader already serves both
  layouts.
- `_x_source_table` currently asserts an ND `ROUND_ROBIN_1D` spec. It needs a BLOCK_SHARDED branch that maps the
  flattened tile row m (= b·S_t + s_t, i.e. `padded_shape` height / 32) and K part j to the shard at grid
  (x = m // bh, y = j) under COL_MAJOR. The branch must read bh, kp and the orientation off the actual `shard_spec`,
  never off the numbers above, and keep the host self-check (every (row, part) has exactly one entry).
- In this layout column c's rows sit in grid column c, so the gather becomes in-column reads. Record the 8x512
  `model_layout` device-ns next to `ln_layout` in the changelog. That is the measurement op_design.md's
  "op_layout decision" asks for. It is information, not a gate.
- `validate()` must tag the placement exactly, by equality with the model's config builder as the golden tagger does
  (`axes.py`), so SUPPORTED stays honest. Anything else stays `"other"` → refused.
- x's shard sits beside the 628 KiB of CBs on 80–88 cores (up to 72 KiB at 8x512). If a cell OOMs, it is a
  `supported_fail` to fix, not an EXCLUSIONS entry.
- Local-shard rule: no core consumes its own x shard as its per-core block (the matmul's per-core block is gathered
  and multicast), so reading the holders over the NoC is the native mechanism here, not an accessor fallback.
- No implementation skill covers this: `/memory-layouts` is RM/TILE layout, not placement.

**Done when**:
- All 56 `model_layout` cells pass, and `verify_supported` shows xfail_expected 0, supported_fail 0, xpass_drift 0
  and xfail_wrong_mode 0.
- `ln_layout` cells, acceptance (35/35) and the precision baseline are unchanged.
- 8x512 `ln_layout` device-ns does not regress.

**Outcome**: `model_layout` added to SUPPORTED. 56/56 golden `model_layout` cells pass. Device kernel ns (1350 MHz,
preallocated L1) for ln_layout vs model_layout: 2x512 29,004 / 29,174; 4x512 50,193 / 50,027; 8x512 93,171 / 93,211.
Phase 0's 8x512 ln_layout baseline was 93,070 (+0.1%, within noise). The gather is off the critical path in both
layouts, so the layout choice does not move this op. The perf refinements below bind the time.



### [x] Refinement 1b — Model split layout placement (`model_layout`) (debug: fix gate violations)

**Goal**: fix the hard violation from Refinement 1 so the completion gate's three bullets hold.

**Verifier notes** (mechanical, from the harness completion gate):

```
Bullet 3 FAIL: golden responsible cells 124/180 below majority threshold.
```

**Done when**: the gate passes — zero hangs in SUPPORTED, acceptance + refinement tests pass, golden majority with no regression.

**Outcome**: the 56 non-passing "responsible" cells were the `op_layout` cells, which the golden helper skips because
`input_memory_config()` returns the norm's layout ("covered by placement ln_layout"). They are not failures. But they
were in SUPPORTED, so the gate counted them as unpassed: 124/180 < 75%. `"op_layout"` was a dead SUPPORTED value:
`_placement()` tags such a tensor `"ln_layout"` and can never produce `"op_layout"`. So it is removed from
`SUPPORTED["placement"]` (now `[ln_layout, model_layout]`). No kernel or host-path change. The op_design.md
"op_layout decision" (the norm keeps its own layout; model_layout would cost the norm +17 µs for a gather that is
off this op's critical path) is unchanged, so the cells are served under `ln_layout`.

Full golden run: test_golden.py 115 passed / 56 skipped (op_layout, now xfail-decorated → `xfail_other`, not loud) /
0 failed; test_regression.py 9/9. Responsible cells 124/124. Unit directory 57 passed / 6 skipped / 0 failed.
### [x] Refinement 2 — q/k rotary on the last K block (perf focus 8x512)

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES` flags 8x512 as the mandatory perf target:
- The full contract: bf16, TILE, ln_layout, q/k/v preallocated L1, `fp32_dest_acc_en=True`, HiFi2,
  `l1_reserved_kb=256`.
- Perf goal: ≤ 85 µs accept (from 93.07 µs), 76 µs stretch, at 1350 MHz.
- Soft precision gate: PCC ≥ 0.99998 and relative RMS ≤ 0.0050 per output.

The measured region is the **q/k cores' last K block** (grid rows 0–5 are the critical path; v rows have no rotary):
- Removing the rotary saves **12.0 µs**: 9.4 µs is the packer-thread SFPU `rotate_head_pair` and 2.7 µs the per-head
  C/S `copy_tile` pair with its srcA reconfig.
- So the packer-thread SFPU is **not** hiding behind the next head's FMAs: the last K block is pack-bound (lamp L2).
- With the rotary off, 1x4 q/k subblocks in K blocks 0..3 are worth another ~1.4 µs (lamp L1).

Speed this region up using the relevant compute patterns in `ttnn/ttnn/operations/examples/master.md`:
- `sfpu_tile_scope`: SFPU per-call fixed cost and instruction count. The rotate is 4 loads + 4 mul + 2 add/sub +
  2 stores per DEST row; fused multiply-add form, per-head semaphore/stall overhead.
- `matmul_output_subblock`: a per-phase subblock, 1x4 / 2x2 in K blocks 0..3 and 1x2 only in the last K block, via
  a per-phase `MatmulBlockShape` or a custom block helper.
- `compute_block_size`: reconfig / `copy_init` count per head.

Rebalancing rotary work between the math and packer threads is also in scope. No SUPPORTED change.

**Verifier notes**:
- Precision headroom on q/k is 4.6% (relative RMS 0.00477 vs 0.0050). The rotary must stay on fp32 DEST values with
  one bf16 rounding; C/S must not drop below their current width.
- The ablation knobs to reproduce the region are in `nomic_qkv_rotary_heads_program_descriptor.py::_env`
  (`QKV_ABLATE_ROTARY=1|2`, `QKV_QK_SUBBLOCK_W`, `QKV_ROTARY_ON_PACK`).
- The DEST cap still pins one head (2 tiles) plus C/S per half-sync fp32 DEST half in the last K block.

**Done when**:
- Device-ns on the flagged 8x512 case improves measurably and moves toward ≤ 85 µs.
- The soft precision gate holds on all outputs, and the golden suite stays green (68 pass, loud categories 0).
- No regression across the config-spanning guard set (one representative per distinct kernel path x layout x
  placement):
  - 2x512 and 4x512 (the LOOSE_CASES guards; 1 vs 2 M blocks).
  - One `seq_padded` shape (8x500).
  - One `output_buffer=dram` cell.
  - One `model_layout` cell.

**Outcome**: 8x512 ln_layout device kernel time went from 93,173 ns to 83,407 ns (−10.5%; 83.4–83.7 µs across runs and
placements), under the 85 µs accept. 4x512 went 50,273 → 45,713 and 2x512 28,868 → 28,189–28,298. There are three levers:
- **Static SFPU addressing** (−8.1 µs): `rotate_head_pair` indexed `dst_reg[i]` with a loop variable. The compiler
  therefore built every SFPLOAD / SFPSTORE at run time and pushed it through the instruction-buffer MMIO (add + sw), so
  the packer TRISC, not the SFPU, bound the rotate. Constant tile offsets plus `dst_reg++` make it fully static.
- **Lamp L1, two-phase q/k K loop** (−0.8 µs): 1x4 subblocks for K blocks 0..3 (`LastBlockTarget::Interm`), then 1x2
  for the last block. The last block reloads the row-grouped partials through a new `MatmulBlockShape::reload_on_first_k_block`.
- **C/S copy on the reload's datacopy config** (−0.8 µs): a new opt-in `post_reload` hook on `matmul_block`'s
  PostComputeFn. It removes one SrcA reconfig and one copy_init per head.

The rotary region now costs 2.7 µs at 8x512 (SFPU 0.7, C/S copy 2.0), down from 12.0. The rest of the q/k critical path
is the matmul K loop (Refinement 4) and the M-block-0 weight stream (Refinement 3). Next for this region would be keeping
C/S resident in DEST across a row's 4 heads, which avoids 3 of 4 fp32 C/S unpacks. It was not tried: the packer's
half release (ZEROACC) clears the whole half, so it needs a custom release protocol, and the gain is at most ~1.5 µs.

### [x] Refinement 3 — M-block-0 weight stream and multicast fill (perf focus 8x512; binds the 2x512 / 4x512 guards)

**Type**: perf

**Goal**: speed up the **in1 stream of M block 0** on the flagged 8x512 case (same contract and ≤ 85 µs / 76 µs goal as
Refinement 2) and its two LOOSE_CASES guards (2x512 ≤ 26 µs, 4x512 ≤ 45 µs accept). Measured:
- Removing the DRAM weight reads saves **3.3–3.4 µs at every shape**: one sender per grid row reads the lane's
  410 KB as 4 KB head pages with one K block of read-ahead.
- With compute stubbed, the data-movement floor is **23.8 µs at 2x512**, against the 26 µs ceiling (30.6 at 4x512,
  46.9 at 8x512). The fill is 5 row multicasts of 80 KB per lane plus the first-block DRAM latency. At small M the op
  is data-movement-bound.

Relevant patterns in `ttnn/ttnn/operations/examples/master.md`:
- `trid_double_issue`: read-ahead depth 2–4 with transaction ids (lamp L4).
- `read_coalescing`: 16 KB lane pages. Re-lay `prepare_qkv_weights` so one (K row, lane) is one page; bump
  `QkvWeights.layout_tag`.
- `mcast_topology`: a rotating in1 sender, `McastRotatingSenderConfig`, with K block b sent by column b mod cols
  (lamp L5).
- `noc_placement`.

No SUPPORTED change.

**Verifier notes**:
- Comes after Refinement 2 because the focus-shape headroom there is 4x larger. It comes before Refinement 4 because
  the guards cannot meet their ceilings without it: at 2x512 the data-movement floor alone is 92% of the ceiling.
- The resident-ring contract (`cb_weights` = `K_TILES * LANE_N_TILES`, re-pushed with no reads for M blocks ≥ 1) and
  equal CB addresses on every core must survive any sender change.
- The rotating-sender variant changes the in1 channel topology. Validate it by building it and measuring device-ns,
  not by inference from the `ABLATE_WREAD` number.

**Done when**:
- Device-ns improves measurably on the flagged 8x512 case **and** on the 2x512 and 4x512 guards, moving them toward
  their ceilings.
- Precision and golden are unchanged (68 pass plus Refinement 1's cells, loud categories 0).
- No regression across the guard set (8x500 `seq_padded`, a `dram` output cell, a `model_layout` cell, an M = 39
  ragged / 10-column shape).

**Outcome**: device kernel ns, 3-run medians, 1350 MHz, ln_layout, preallocated L1. Before → after: 2x512 28,024 →
26,935 (−1.09 µs), 4x512 45,775 → 45,227 (−0.55), 8x512 83,487 → 82,702 (−0.79). model_layout: 27,007 / 44,792 /
82,531 (Refinement 2: 28,156 / 45,504 / 83,637). The 2x512 and 4x512 guards are not yet at their 26 / 45 µs ceilings.
- **Kept win**: 16 KB lane pages (`WEIGHT_PAGE_TILES`, layout tag `n_pages_8_v1`; catalog `read_coalescing`).
- **Built and measured, parked as knobs**:
  - Rotating / diagonal in1 sender (`W_SENDER`): rotating is flat at one block of read-ahead, diagonal +0.3–1.3 µs.
  - Read-ahead with transaction ids (`WEIGHT_READ_AHEAD_CHUNKS`): 2 blocks in flight cost +2–3 µs.
  - Chunked multicast of a K block (`W_MCAST_CHUNK_K_ROWS`, new `send_payload` on the mcast sender pipe): it beats
    the default by ~0.6 µs at every shape with output writes stubbed, and ties it with them on.
- **What binds now** (2x512 zones): weight block 0 reaches compute at ~5.5 µs (x at ~1.7). That is a ~3.1 µs DRAM read
  of 720 KB grid-wide (~240 GB/s, the same in a column or a diagonal) plus a 1.5 µs row multicast. After block 0 the
  stream (~3.0 µs a block) outpaces compute (~3.3–3.7 µs a block). At 2x512 the op's end is set by the output scatter:
  the v rows (bottom grid rows) finish compute first and write last, 3–4.5 µs after compute against ~1 µs on q/k
  rows. That tail absorbs the chunked-multicast gain.
- **Next**: shorten the output-write tail (perf lamps L6/L7: write-order stagger, two-RISC drain), then flip
  `W_MCAST_CHUNK_K_ROWS = 1`. A smaller first K block (a compute change) would cut the DRAM fill itself. Neither was
  done here: one is the drain and the other is the K loop, both outside this refinement's weight stream.

### [x] Refinement 4 — K-loop efficiency and M/K block co-tune (perf focus 8x512, toward stretch)

**Type**: perf

**Goal**: on the flagged 8x512 case, raise the **matmul K loop** toward the HiFi2 ceiling and close the gap to the
76 µs stretch.
- **Measured gap**: with the rotary off and 1x4 subblocks the op runs 79.6 µs. 99 cores x 2400 tile products x 32
  cycles at HiFi2 is ≈ 56.9 µs, so the K loop is at about 71%. The data-movement floor (46.9 µs) does not bind.
- **Fixed costs to amortize**: 4 M blocks x 5 K blocks per core. Each K block spills and reloads fp32 partials
  (packer L1 accumulation) and does an in0 multicast handshake. Each M block re-inits `matmul_block` and handshakes
  the table.

This is a co-tune of the block surface the planner exposed:
- **`BLOCK_M_TILES_MAX`** (lamp L3; capped at 3 by L1). Free L1 for it by shrinking `cb_heads_out` to a row window:
  verifier ledger finding, ~16 KiB a core, see `l1_ledger.md` § Verifier audit.
- **`K_BLOCK_TILES`**: pinned to 5 because the bias is K tile 24 (K_TILES = 25). A coarser K block means re-placing
  the bias, so `K_TILES` admits a larger divisor.
- **`X_BLOCK_DEPTH`**.
- **The last-block tail**: two-RISC drain, lamp L7.
- **Ragged-block garbage rows** at M = 48 / 80 / 112 (lamp L8): guard-set shapes.

Patterns: `compute_block_size`, `eltwise_l1_vs_dest_accumulate`, `double_buffer`, `matmul_output_subblock` in
`ttnn/ttnn/operations/examples/master.md`. A performant block is whole tiles at minimum, and coarser amortizes until
L1 binds. No SUPPORTED change.

**Verifier notes**:
- All knob-turns (no new topology), so they bundle into one phase of T2 levers. Re-measure the regions first: after
  Refinements 2–3 the binding region may have moved.
- L1 margin is 12 KiB at `block_m_tiles = 3` (628 of 640 KiB). Every footprint change must keep `circular_buffer_bytes`
  ≤ 640 KiB at M = 128 and update `l1_ledger.md` (rows, closed form, symbol bounds).
- Moving the bias out of K changes `prepare_qkv_weights` (bump the layout tag) and the reader's ones-tile write. The
  bias must stay an fp32 add in DEST before the rotary (precision headroom 4.6%).

**Done when**:
- Device-ns on the flagged 8x512 case improves measurably (toward the 76 µs stretch).
- The soft precision gate holds, and the golden suite stays green.
- `circular_buffer_bytes` ≤ 640 KiB at every INPUTS shape.
- No regression across the guard set (2x512, 4x512, 8x500 `seq_padded`, one ragged shape among M = 48 / 80 / 112,
  a `dram` output cell, a `model_layout` cell).

**Outcome**:
- **Measured**: 8x512 ln is 82.65–83.28 µs (median ~82.75) against 82.70 (Refinement 3) and 83.44 (this session's
  first run). 4x512 is 44.8–45.4 and 2x512 26.9–27.3: within ±0.5 µs noise, no regression. The stretch was not
  reached.
- **Kept**:
  - `TABLE_LOAD_K_BLOCK = n-2`: the C/S request no longer stalls the column's x stream (−0.3 to −0.7 µs).
  - `OUT_DRAIN_TILES = HEAD_TILES`: per-head push and drain halve the median tail (0.98 → 0.52 µs); the wall ties.
- **Swept**:
  - `K_BLOCK_TILES`: 1 → 162 µs.
  - `BLOCK_M_TILES_MAX`: 1 / 2 / 3 → 100.2 / 87.6 / 83.4 µs; 4 → 81.7 µs, over L1 at 704 KiB.
  - `X_BLOCK_DEPTH = 3`: flat.
  - A whole-K-per-row prototype: v 5.1 µs a row, q/k 6.2 µs a row, so no gain where q/k binds.
- **The bottleneck now** is q/k compute. Op time ≈ math + 27 µs at every fidelity ≥ HiFi2. Most of the excess is the
  finish phase's fixed ~0.23 µs a head (fp32 partials reload + C/S datacopy + matmul re-init; 48 heads ≈ 11 µs). The
  rest is the LLK's srcA-per-product unpack (85–91% of math) and the ~5 µs weight fill.
- **Next would be** keeping C/S resident in DEST across a row's heads, which needs a release that does not ZEROACC
  the whole half (LLK work). After that, `BLOCK_M_TILES_MAX = 4` (−0.9 µs) once ~64 KiB is reclaimed: a 1-row
  `cb_heads_out` plus a per-row C/S ring. Neither was done here: the first is below the helper layer and the second is
  a large reader-schedule change for under 1 µs.
