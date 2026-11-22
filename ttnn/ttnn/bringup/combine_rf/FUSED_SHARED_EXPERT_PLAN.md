# Fused shared expert: plan (2026-10-20)

The shared expert of the token-split MoE block (GLM-5.3-Flash, `models/demos/deepseek_v3_d_p/tt/moe/tt_k3_token_split_moe.py`
with `shared_tp=True`) as one op, then hidden under the dispatch. Context and measurements:
`ttnn/ttnn/bringup/flat_routed_expert_ttnn/FLAT_COMBINE_OVERLAP_WORKLOG.md` (sections "Folded shared expert ...").

## Where we are

The shared expert is folded into the block with no collectives of its own:

- **x all-gather -> dispatch.** `dispatch_prep` ORs the token's group-mates (its mesh row, `folded`, or column,
  `folded_sp`) into every own token's destination mask, so the dispatch lands the group's G x S tokens in every
  group chip's dispatch buffer (slot `origin x S + t`).
- **The chip's slice** (I_s / G columns of gate / up, the matching rows of down) runs on those rows between the dispatch
  and the fused flat + combine op, with all of L1 (the fused op has its own cross-chip barrier, so the arena is
  allocated after the shared expert).
- **Reduce-scatter -> combine.** The partial rows of the group-mates' tokens go home through the combine
  (`shared_rows`: same rings and credits, weight 1, static counts on top of the device plan); the accumulators add
  the own block from DRAM.

Today's shared expert is six programs (`TtSharedTpRow`, `models/demos/glm53_flash_d_p/tt/shared_tp_row.py`): the
gather-tilize, the gate / up matmul (ttnn 2D multicast), `glu_act`, the down matmul, the untilize. bf8 weights, HiFi3
(owner: HiFi2 shrank the output ~0.8% vs the golden), fp32 dest, bf16 gate / up and h.

| L4 MoE block, LoudBox 2 x 4, folded_sp (G 2) | chunk 5120 | chunk 2048 |
|---|---|---|
| shared expert today: tilize + gate/up + glu + down + untilize | 50 + 140 + 30 + 91 + 52 = **363 us** | 21 + 72 + 20 + 46 + 22 = **181 us** |
| block today (HiFi3) | ~2860 us | ~1820 us |
| the dispatch it could hide under | 482 us (link bound, ~20 of 120 cores) | 207 us |

Tried and not kept: the flat expert op as the shared expert (one I 1024 expert has too few gate / up cores: 424 us
at HiFi2), the bring-up unified op (366 + 51 untilize at 5120: 88-core grid, the weights re-streamed per block),
a row-major down output from ttnn's 1D config (`untilize_out`: rows written, tensor labelled TILE).

## Roofline of one fused op

Per chip, LoudBox folded_sp chunk 5120: M = G S = 1280 rows, H = 4096, slice I = 1024. 120 cores, LoFi 664 TFLOP/s
(HiFi2 / 3 / 4: 1/2, 1/3, 1/4), DRAM 512 GB/s.

| | chunk 5120 (M 1280) | chunk 2048 (M 512) | Galaxy 8 x 4, rows G 4 (M 640, I 512) |
|---|---|---|---|
| FLOPs (gate / up + down) | 32.2 G: HiFi3 146 us, HiFi2 97 | 12.9 G: HiFi3 58 us | 8.1 G: HiFi3 37 us |
| DRAM: weights + x in + partial out | 13.4 + 10.5 + 10.5 MB: 67 us | 21.8 MB: 43 us | 17.1 MB: 33 us |
| floor | ~146 us (compute) | ~58 us | ~37 us |
| target (~75-80%) | **~180-190 us** (HiFi3) | **~70 us** | **~45 us** |

## Phase 1: the fused op, standalone

`se_ffn(dispatch buffer, block slots, W_gu, W_down) -> y_se` (combine_rf, one generic_op per chip; static M from the
block's S and G):

- **Grid 12 x 10, 2D.** Grid row cy owns 4 row tiles (128 tokens); columns split the gate / up output columns.
- **x in, row-major.** Each grid row's sender reads its tokens' row-major x from the dispatch buffer at the block
  slots (the `layout_ops` gather-read, 1 KB pieces), tilizes on its compute (~33 cycles a tile) and multicasts the K
  blocks along the row: no separate tilize, no tiled copy of x in DRAM.
- **Gate / up (2D SUMMA).** W_gu in gate / up column pairs (each core's columns are matching gate and up columns),
  streamed down each column from bank-adjacent readers and multicast (lin2d's W path, `combine_rf/lin2d.py`); K
  accumulated in fp32 dest + packer L1 accumulation.
- **Activation in place.** The clamped SiLU GLU (`glu_act`'s SFPU call, `calculate_clamped_silu_glu`) on the core's
  own gate / up tiles: h never leaves L1.
- **h exchange.** Each core multicasts its h tiles along its grid row; every core then holds h for its 4 row tiles
  over the full K of the down projection (4 x 32 tiles bf16, 256 KB: fits with the arena not allocated).
- **Down.** The core's 4 row tiles x ~11 output columns, W_down multicast down the columns; the packer untilizes and
  the core writes its 128 token rows as ~700 B row pieces straight into y_se (row-major, the combine's input).
- **Traffic.** Weights read once (13.4 MB), x once (10.5 MB), y once (10.5 MB); gate / up and h stay in L1.

Expected: 363 -> ~180-190 us at chunk 5120 (HiFi3), 181 -> ~70 us at 2048; Galaxy 8 x 4 ~45 us.

Work and risk: new are the h row exchange and the two-phase sequencing (cross-core synchronization: where hangs come
from); reused are lin2d's W streams / multicast scheme, glu_act's compute call, layout_ops' gather-read and tilize.
Method: a traffic-only skeleton first (x reads + tilize, W streams, h exchange, row writes; no math) to establish the
data path and its time, then compute; accuracy against the golden with `zz_scratch_se_fold_golden.py`, perf in the
block with `/localdev/mstaletovic/blockops/run.sh`. ~3-5 days.

## Phase 2: overlap with the dispatch

The dispatch is link bound and uses only its port and copy cores; the fused op on the remaining cores fits under it
(LoudBox 5120: 482 vs ~190 us; Galaxy 8 x 4: ~140-180 vs ~45 us).

- **Merge, not sub-devices.** `sparse_all_gather_2d` takes `extra=` (another op's kernels and CBs on the cores it leaves
  free; used before for an untilize). One program: no sub-device rectangles, no fabric channel conflict (the shared
  expert uses no fabric), and L1 fits (the dispatch's CBs and ~100 KB manifest; the arena comes after).
- **Own block first.** Its x is local (the dispatch copies own rows into their slots before sending): the shared
  expert starts at once on 1 / G of its rows.
- **Group-mates' rows first in the dispatch.** The dispatch sends them before the routed rows (LoudBox: one Y hop,
  ~105 us at link rate), so they land early in the window.
- **Arrival counters.** The dispatch receivers bump a per-source-chip counter (today the only completion signal is the
  end markers); the shared expert's x readers poll it before reading a group-mate's block.
- **Out.** Partials into y_se as now; the fused op starts after the merged program.

Expected: the shared expert off the critical path (LoudBox 5120 block ~2650 -> ~2500 us). Risk: fabric ops slow down
under co-scheduled DRAM / NoC traffic (an untilize merged into the dispatch once took it 230 -> 284 us): measure the
dispatch with the standalone op as background load before building the merge.

## Phase 3, only if needed

- The routing phase (router -> routing gather -> planner -> prep, ~140 us on ~40 cores) as an alternative home for
  the own block, if the dispatch is too sensitive to share.
- The partials' cost inside the fused op (~136 us on the LoudBox, SP pairs): the port senders' DRAM reads of y_se
  (43 us; NOC0 reads, or y_se placed nearer the ports) and their per-row time.

## Expected totals (L4 MoE block, LoudBox 2 x 4, chunk 5120, HiFi3)

original 4058 us -> today ~2860 -> phase 1 ~2650 -> phase 2 ~2500 (about -38%). On Galaxy stages the shared expert
(~45 us) disappears under the dispatch.
