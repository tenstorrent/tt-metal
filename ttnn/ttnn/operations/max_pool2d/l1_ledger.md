# L1 Ledger: max_pool2d

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking axes are named in `op_design.md` → Blocking Model: **positions** (block extent `block_tiles` stream tiles per NEO), **channels** (`C_core`, pitch `stick_pitch`), **taps** (`taps_per_batch`), **tile elements** (1024 per stream tile). Worked numbers are for `quasar` (C = 4 NEOs, bf16, `tile_bytes` = 2048).

Inventory reasoning (Rule 1, before any solve):

- **No tilize/untilize buffer.** The stream-tile layout makes a row-major block the compute operand.
- **No tiled output buffer.** The fold packs straight into the ring the writer drains (Rule 3.1). The output shard itself is not borrowed; see `zero-copy-output-pack` in the regime table.
- **No accumulator ring.** The running max lives in DEST (Rule 3.4).
- **No padding buffer.** Padding taps re-read a real tap.
- **No staging on IN_PLACE cores.** The shard is the staging.

That leaves three buffers. The one budget predicate (`block_tiles`) exists only for the staging, and no PHASE0 or TARGET input reaches it.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_window_taps` | `tap_ring_depth · taps_per_batch · C` pages of `tile_bytes` (Phase 0: `2 · K · 4`; K=25 → 200 pages = 409,600 B; K=9 → 72 pages = 147,456 B) | `taps_per_batch · C` pages (one window per NEO). The second window is the double buffer. | {positions: streams → 1 stream tile per window; channels: streams (within the 1024-element entry); taps: spans → `taps_per_batch`; tile elements: spans → 1024} | input dtype (bf16). The entries are NoC copies of input bytes, produced by no pack from DEST, so there is no DEST precision to keep; the width follows the input, not `fp32_dest_acc_en` (audit 3 mechanism reason: a dataflow-produced payload). | reader | compute | whole kernel | none. It runs concurrently with `cb_pooled_tiles` (compute reads one while packing the other) and with `staging_window` (the gather reads staging while filling it). Capacity > live set by exactly `tap_ring_depth = 2`: double buffering, reader landing tile u+1 while the NEO folds tile u. |
| `cb_pooled_tiles` | `out_ring_depth · C` = 8 pages of `tile_bytes` (16,384 B) | `C` pages (one per NEO). The second is the double buffer. | {positions: streams → 1 stream tile; channels: streams; taps: streams (folded away in DEST before pack); tile elements: spans → 1024} | output dtype (= input, bf16) under fp32 DEST. Audit 2 *under* answered: the output dtype is fixed by the spec (R11: the input's dtype), and the max selects an input element exactly, so there is nothing to truncate. | compute | writer | whole kernel | none. Concurrent with `cb_window_taps` (above). Cannot alias the output shard: R9's exact shard shape is not a whole number of tiles (e.g. `[400, 32]` = 12.5), so the last pack would overrun it. Capacity > live set: double buffering (`out_ring_depth = 2`). |
| `staging_window` (scratchpad, DM-only) | bytes, not pages: ROW_MAJOR `(s_hi − s_lo) · stick_pitch · esize`; TILE `(tr_hi − tr_lo + 1) · (C_core/32) · tile_bytes`. The max over the core's blocks, bounded by `staged_sticks(block_tiles) ≤ ((rows_out(block_tiles) − 1)·sh + kh_eff) · W` (TILE: `+ 62` sticks of tile rounding) | the whole receptive field of the current block (every tap of every tile in the block may read any staged stick, until the block's gathers end) | {positions: spans → the block's receptive-field sticks, affine in `block_tiles`: `rows_out = ceil(C·block_tiles·1024 / (stick_pitch·W_out)) + 1`, × sh, + kh_eff, × W; channels: spans → `stick_pitch`; taps: streams (all taps read it, none stored); tile elements: n/a (stick-addressed)} | none (DM-only scratchpad; holds raw input bytes) | reader (R threads, disjoint quarters) | reader (R threads, after `sync_threads()`) | one block (Phase 0: the whole kernel, one block) | none. Live concurrently with both rings. On IN_PLACE cores it is unused: allocation is per program, so it is allocated on every engaged core, with size = the max over STAGED cores (one L1 alignment unit, 16 B, when every core is IN_PLACE, so the reader's binding exists). Not merged with `cb_window_taps`: the staging is random-access and read-only for a whole block, while the ring's entries recycle per tile. |

## Symbol table

| Symbol | Meaning | Bound | Predicate that establishes it |
|---|---|---|---|
| `C` | compute threads per core (THREAD_PLAN) | 4 (∈ {1,2,4}) | host constant; the target page's thread-plan limits |
| `tile_bytes` | one stream tile = 1024 elements × `esize` | 2048 (bf16), 4096 (fp32) | dtype ∈ SUPPORTED |
| `K` | taps `kh·kw` | unbounded by the spec; INPUTS max 25 | — (bounded through `taps_per_batch` below) |
| `taps_per_batch` | taps per reader batch / ring window | largest divisor of K that is ≤ `floor(1,048,560 / (tap_ring_depth · C · tile_bytes))` = 63 (bf16, C=4) | ring-size cap (target page) |
| `tap_ring_depth`, `out_ring_depth`, `staging_depth` | depth knobs | 2, 2, 1 | host constants |
| `stick_pitch` | elements per position in staging / streams | `align_up(C_core·esize, ttnn.get_l1_alignment())/esize`; ≤ C (+ alignment) | output shard spec |
| `C_core` | channel run per assignment | ≤ C | output shard spec (R9) |
| `W`, `W_out`, `sh`, `kh_eff = dh(kh−1)+1` | geometry | spec arguments | validate (X1–X4) |
| `block_tiles` | stream tiles per NEO per staging block | `Q = ceil(T_core / C)` when `staging_bytes(Q) + ring_bytes ≤ budget`, else the largest value satisfying it (≥ 1) | budget predicate (below) |
| `budget` | per-core L1 for this op | `ttnn.get_max_worker_l1_unreserved_size() − l1_reserve_bytes` = 3,963,328 − 65,536 = 3,897,792 B on quasar (query value read on ttsim 2026-11-21). Tests may lower it with `_STAGING_BUDGET_OVERRIDE`. | device query, `l1_reserve_bytes` = 65,536 (host knob) |

## Total per-core footprint

```
L1(core) = C · tile_bytes · (tap_ring_depth · taps_per_batch + out_ring_depth)      # rings: scale with K (taps) and C
         + staging_bytes(block_tiles)                                             # scales with block_tiles (positions), stick_pitch (channels), W
```

- Ring terms scale with `taps_per_batch` (the taps knob) and `C` (the plan). They do not scale with positions or channels.
- The staging term scales affinely with `block_tiles` through `rows_out`, linearly with `stick_pitch` and `W`, and is 0 on IN_PLACE cores.

Worked (quasar, bf16, Phase 0):

| Case | Rings | Staging (max over cores) | Total | Budget |
|---|---|---|---|---|
| SPPF 1×20×20×256, K=25, interleaved, 32 cores | 4·2048·(50+2) = 425,984 B | 11 rows × 20 × 64 B = 14,080 B | ≈ 440 KB | 3,897,792 B |
| ResNet 1×32×32×64, K=9, TILE height-sharded on 32 cores | 4·2048·(18+2) = 163,840 B | 3 tile rows × 2 × 2048 = 12,288 B | ≈ 176 KB | 3,897,792 B |
| ResNet 1×32×32×64 on one core (R10), `position_splits = 4` | 163,840 B | ≤ 10 tile rows × 2 × 2048 = 40,960 B | ≈ 205 KB | 3,897,792 B |
| Width-sharded ROW_MAJOR 1×32×32×32, 32 shards of 1 ch | 163,840 B | 0 (IN_PLACE) | 163,840 B | 3,897,792 B |

## Data-movement budget

The chosen split is output shards (channels primary, the given position shards for height-sharded inputs), × whole-round position sub-ranges on spare cores. Bytes are input/output tensor sizes `|in| = N·H·W·C·esize`, `|out| = P·C·esize`.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| input, DRAM-interleaved | `1 + halo_fraction`, where `halo_fraction = Σ_cuts (kh_eff − sh)·W / (N·H·W)`. Worked SPPF: 33/20 = 1.65 (4 sub-ranges per channel group); 1.0 with `position_splits = 1`. | the receptive field is staged once per block and every tap is gathered from L1. Channels never overlap, so channel cuts add nothing. | 0 |
| input, L1-interleaved | 0 (it is in L1) | as above | `1 + halo_fraction` × \|in\| over the NoC (banks → staging) |
| input, height-sharded | 0 | staged once per assignment: own rows core-local, halo rows from neighbours | halo: `(kh_eff − sh)·W·C·esize` per shard boundary (worked ResNet on 32 shards: each core stages 3 image rows for 0.5 output rows, about 3 × \|in\| moved in total, mostly own-or-neighbour 2 KB tiles). Spare cores (one-core case) pull their whole receptive field cross-core. |
| input, width-sharded | 0 | IN_PLACE on owners: read where it lies | 0 on owners; a spare-core sub-range pulls its receptive field once from the owner |
| tap streams (intermediate) | 0 | core-local: K stream tiles gathered per output tile (`K·|out_core|` bytes of L1→L1 per core) | 0 |
| output | 0 | written once, by its computing core | 0 on owner sub-ranges; `|out| · (position_splits − 1)/position_splits` on spare-core sub-ranges (worked SPPF: 0.75 × \|out\| = 150 KB) |

Totals per tier (worked SPPF, 1×20×20×256 bf16, DRAM-interleaved, `position_splits = 4`): DRAM 1.65 × 200 KB = 330 KB read; cross-core 150 KB output writes; core-local 25 × 200 KB of gathers + 50 KB of output-owner writes.

> Cheapest-traffic split considered: **channels only** (`position_splits = 1`): for SPPF, 0.65 × |in| = 130 KB less DRAM and 150 KB less cross-core. Implemented: channels × whole-round position sub-ranges. Deferred because: it is not a separate algorithm. It is the same structure with the knob `position_splits` set to 1, so it stays reachable by one host constant. The default spends those bytes to engage 4× the NEOs on an op bound by compute (K = 25 SPPF folds per output tile). The grid-synchronization perf lamp in `op_design.md` names this exact measurement. Where the output shard count already fills the grid (height-sharded on 32 cores, width-sharded on 32), the implemented split **is** the cheapest.

## Implementation notes (implementer, phase 0)

- Built as inventoried: `window_taps` (`TAP_RING_DEPTH · taps_per_batch · C` entries), `pooled_tiles` (`OUT_RING_DEPTH · C`), `staging` scratchpad. Knobs are the module constants of `max_pool2d_program_artifacts.py` (`THREAD_PLAN`, `TAP_RING_DEPTH`, `OUT_RING_DEPTH`, `STAGING_DEPTH`, `MIN_ROUNDS_PER_CORE`, `MAX_POSITION_SPLITS`, `L1_RESERVE_BYTES`, `_STAGING_BUDGET_OVERRIDE`); every ring size, loop count and grid size derives from them.
- `staging` when every engaged core is IN_PLACE: allocated at one L1 alignment unit (16 B) instead of 0 B, so the reader's scratchpad binding exists; it is never touched there.
- `block_tiles` is chosen per assignment (RT arg), the largest value whose staging fits the budget; the scratchpad is sized to the maximum over STAGED assignments. The staged-streamed regime is pinned by `test_max_pool2d_plan.py::test_staged_streamed` (budget override, one core per shard).
- The data-movement budget above holds as written: the receptive field crosses its home tier once per block, taps are core-local copies, the output is written once.

## Verifier audit (phase 0)

- Currency: the three rows match the code (`max_pool2d_program_artifacts.py`: `window_taps` = `TAP_RING_DEPTH · taps_per_batch · C`, `pooled_tiles` = `OUT_RING_DEPTH · C`, `staging` = max over STAGED assignments, at least 16 B). Two statements corrected: the IN_PLACE-only staging size (16 B, not 0 B) and the reason for `window_taps`' page width (a dataflow-produced payload, not one that "never passes through DEST": it is unpacked into DEST).
- In-place condition: an owner core is IN_PLACE only when it also holds the matching input shard. With a col-major width-sharded input, rule R9's row-major output puts output shard g on a different core from input shard g, and that owner then stages (verifier fix, `test_col_major_input_gives_row_major_result`).
- Cheapest-traffic split: channels only (`MAX_POSITION_SPLITS = 1`) is reachable through one knob. The default spends 0.65 × |in| more DRAM reads on SPPF to put 4× the NEOs to work, and that justification rests on occupancy plus the op being compute-bound at K = 25. The decision is folded into Refinement 3 (op_requirements.md), which changes the per-output-tile compute cost and must re-record this row with its counts.
