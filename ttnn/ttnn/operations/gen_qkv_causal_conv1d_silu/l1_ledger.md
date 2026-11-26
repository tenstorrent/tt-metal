# L1 Ledger: gen_qkv_causal_conv1d_silu

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes (from `op_design.md` →
Blocking Model): **token** (`block_rows` tile-rows, +3-row halo), **channel** (`block_ch_tiles`),
**tap** (4, whole), **batch** (1). Below `w = block_ch_tiles`, `M = block_rows`, `M_b` = rows of the
current block (`= M` except a core's last block), tile page = 2048 B, stick page = `64·w` B.
Every CB is bf16 data (`Float16_b`) under both DEST widths — see audit 2 below.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_rows` | `ROWS_DEPTH·(32·M + 3)` stick pages of `64·w` B | `2·(32·M + 3)`: the slot compute is windowing (all `32·M_b + 3` sticks stay live until the block's last tile-row is tilized — every window of tile-row m reads sticks `[32m, 32m+35)`) + the slot the reader fills | `{token: spans → 32·M + 3 (block rows + HALO_ROWS), channel: spans → w (page width), tap: streams → the 4 windows re-read the same sticks, batch: spans → 1}` | Float16_b — raw bf16 input copy, never passes through DEST | reader | compute | whole kernel | Not shared: concurrent with `cb_window_tiles` (windows are produced from it) and with `cb_output_tiles`; capacity above one slot is the double buffer (reader fills block b+1 during compute on b); the previous slot must also survive as the L1 halo source of the next block, which depth 2 provides for free |
| `cb_tap_tiles` | `w` tile pages | `w` (resident: every output tile of every block reads tap tile c) | `{token: streams → constant along tokens, read by every tile-row, not scaled by M, channel: spans → w, tap: spans → 4 taps packed in rows 0,2,4,6 of each page, batch: spans → 1}` | Float16_b — bf16 operand, only row `2p` valid | reader | compute | whole kernel (loaded once, popped at kernel end) | Not shared: live for the whole kernel; packing 4 taps per page (bcast_row_idx) already cut it from `4·w` to `w` pages |
| `cb_window_tiles` | `TAP_COUNT·w·M` tile pages | `4·w·M_b` = capacity: the tilize helper and the MAC block op both own all three TRISCs, so the tilize pass produces the whole block before the MAC pass consumes it (sequential-helper intermediate) | `{token: spans → M, channel: spans → w, tap: spans → 4 windows, batch: spans → 1}` | Float16_b — exact copies of bf16 input (tilize = data movement); the fp32 accumulation never leaves DEST | compute (windowed tilize) | compute (conv_silu) | windowed_tilize_block → conv_silu_block, per block | Not aliased with `cb_rows` (concurrent: it is filled while the slot is read) nor `cb_output_tiles` (writer drains the previous block's tile-rows while this one is filled) nor `cb_untilized_rows` (the reader may still be copying the last untilized tile-rows when windowed tilize starts: overlapping lifetimes at the phase boundary). It exists only because the shift is a data-movement step before the MAC; the `tilize_fused_mac` deferred regime removes it |
| `cb_output_tiles` | `OUT_DEPTH·w` tile pages | `w` being packed + `w` being written = `2·w` | `{token: streams → one tile-row at a time, channel: spans → w, tap: streams → reduced in DEST before pack, batch: spans → 1}` | Float16_b — the output ABI is bf16; the sum was rounded once at pack | compute | writer | conv_silu_block → store_block | This IS the pack-into-destination buffer (Rule 3.1): outputs are interleaved tensors on other banks, so no zero-copy is possible; depth 2 = double buffer against the writer |
| `cb_input_tiles` (TILE regime) | `IN_DEPTH·w` tile pages | `w` being untilized + `w` landing = `2·w` | `{token: streams → one tile-row, channel: spans → w, tap: streams → n/a (raw input), batch: spans → 1}` | Float16_b — bf16 input tiles | reader | compute | load → untilize_rows_block | Not shared: concurrent with `cb_untilized_rows` and `cb_rows`; depth 2 = reader fetch of row j+1 during untilize of row j. Absent in the RM regime |
| `cb_untilized_rows` (TILE regime) | `UNTILIZED_DEPTH·w` tile pages (`w` pages = one tile-row = 32 sticks) | `w` being copied by the reader + `w` being packed = `2·w` | `{token: streams → one tile-row, channel: spans → w, tap: streams → n/a, batch: spans → 1}` | Float16_b — exact bf16 copy | compute (untilize) | reader (L1→L1 copy into `cb_rows`) | untilize_rows_block → load_rows_block | Cannot be folded into `cb_rows`: `cb_rows` needs a single producer and the halo sticks come from the reader, so the untilized body hops through the reader; aliasing it with `cb_window_tiles` rejected (lifetimes overlap at the phase boundary, see above). Absent in the RM regime |

Audit notes:

| Audit | Result |
|---|---|
| 1. Capacity vs live set | Every capacity equals its live set (double-buffer factors stated). Under-scaling: `cb_tap_tiles` spans tap without a `·4` factor because the 4 taps share one page (rows 0,2,4,6) — capacity still covers the live set. |
| 2. Page format vs DEST width | All pages are Float16_b while `fp32_dest_acc_en` may be True. Mechanism reason for each: `cb_rows`, `cb_input_tiles`, `cb_untilized_rows`, `cb_window_tiles` are dataflow / data-movement copies of bf16 operands (no value in them was ever accumulated); `cb_tap_tiles` is a bf16 operand; `cb_output_tiles` is the bf16 output ABI and receives the single post-SiLU rounding. No partial sum is ever packed. |
| 3. Disjoint lifetimes | No CB pair has disjoint lifetimes within a block's steady state; the one candidate (`cb_untilized_rows` / `cb_window_tiles`) overlaps at the untilize→tilize boundary. |
| 4. Bounds and closed form | below. |

## Symbol table

| Symbol | Meaning | Bound | Predicate establishing it |
|---|---|---|---|
| `w = block_ch_tiles` | channel tiles per block (= per core) | `1 ≤ w ≤ min(255, tensor_ch_tiles)`, `w | tensor_ch_tiles` | `select_plan` candidate filter (`footprint(w,1) ≤ L1_BUDGET`, `G ≤ N`) |
| `M = block_rows` | token tile-rows per block | `1 ≤ M ≤ core_seq_tiles_max` | `select_plan`: largest balanced `M` with `footprint(w, M) ≤ L1_BUDGET` |
| `M_b` | rows of the current block | `1 ≤ M_b ≤ M` | last block of a core only |
| `ROWS_DEPTH, OUT_DEPTH, IN_DEPTH, UNTILIZED_DEPTH` | buffer depths | 2 each | host constants |
| `TAP_COUNT = 4`, `HALO_ROWS = 3` | op math constants | fixed | op definition |
| `L1_BUDGET` | per-core L1 the CBs may use | `1,048,576 − out_l1_share` | BB-05; `out_l1_share = Σ_{q,k,v} ceil(T_t·X_t / num_l1_banks)·2048` when outputs are in L1, else 0 |
| `tensor_ch_tiles`, `tensor_seq_tiles` | `C/32`, `T/32` | unbounded op dims — appear in NO capacity expression | — |

## Total per-core footprint

```
footprint(w, M) = ROWS_DEPTH·(32·M + 3)·64·w          # cb_rows        — scales with w and M
                + TAP_COUNT·w·M·2048                   # cb_window_tiles — scales with w and M
                + w·2048                               # cb_tap_tiles   — scales with w only
                + OUT_DEPTH·w·2048                     # cb_output_tiles — scales with w only
                + [TILE] (IN_DEPTH + UNTILIZED_DEPTH)·w·2048   # TILE hop CBs — scale with w only
    (depths 2)  = RM:   w·(12288·M + 6528) bytes
                  TILE: w·(12288·M + 14720) bytes
```

`M`-dependent terms: `cb_rows`, `cb_window_tiles` (12288 B per tile-row per channel tile). Smallest
plan (w = 1, M = 1): 18,816 B (RM) / 27,008 B (TILE) — always inside the budget, so no golden shape
is refused for L1. Largest Phase-0 golden plan (004 on 11×10: w = 16, M = 3): 694,272 B.

## Data-movement budget

Chosen split: channel groups × token ranges (`num_ch_groups × cores_per_group` cores). Numbers for
the DRAM-bound target 004 (T 4096, C 2560, 11×10 grid: w 16, G 5, U 22) and the latency target 001
(T 32: w 1, G 80, U 1).

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| input (RM) | 1× + 3 sticks per core whose range starts at `t0 > 0` (004: 105 cores × 3 KiB = 315 KiB, +1.5%; 001: 0) | the whole core range streams once through `cb_rows`; later blocks' halo is copied from the previous slot in L1, only a core's first halo is re-read | none (core-local halo copies: 3 sticks per later block) |
| input (TILE) | 1× + one tile-row (`w` tiles) per core with `t0 > 0` | same; the halo needs rows 29..31 of the preceding tile-row | core-local: one L1→L1 copy of every untilized tile-row (1× input bytes, same core) |
| history | 1× per channel (only the `t0 = 0` core of each group reads it) | only the first tile-row needs it | none |
| taps | `cores_per_group`× of the tap rows (004: 440 KiB vs 20 KiB logical; 001: 1×) | reuse-shared across the token split; each core reads only row 0 of each tap tile (2 × 32 B per tap per channel tile), never the whole tile | none (multicast deferred) |
| q, k, v | 1× | written once from `cb_output_tiles` | none |
| actual_start, predecessor_carry | 0 | never read (one sequence-parallel rank) | none |

Totals, 004: DRAM ≈ 20,480 KiB in + 20,480 KiB out + 15 KiB history + 440 KiB taps + 315 KiB halo
= 41,730 KiB vs the 40,995 KiB minimum (+1.8%); cross-core NoC traffic 0. 001: 160 KiB + 160 KiB +
15 KiB + 20 KiB = the minimum.

> Cheapest-traffic split considered: channel groups only (`cores_per_group = 1`) — 004: −735 KiB DRAM
> (−1.8%: no halo re-reads, taps 1×), 0 cross-core. Implemented: channel groups × token ranges with
> `cores_per_group` chosen by min makespan. Not deferred: the cheapest split is the
> `cores_per_group = 1` value of the same built knob, and `select_plan` selects it whenever it is also
> min-makespan (001); for 004 it would leave 30 of 110 cores idle and raise the per-core makespan from
> 96 to 128 tiles for a 1.8% byte saving. The tap redundancy alone (−420 KiB) is the
> `tap_multicast_per_group` deferred regime row.
