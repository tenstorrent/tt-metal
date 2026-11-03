# L1 Ledger: moe_down_proj

Schema: `.claude/references/l1-footprint-discipline.md`. Block axes (from `op_design.md` Blocking Model):
`m` (rows, extent `core_row_tiles`), `wave` (`num_waves`), `n` (`core_n_tiles`), `k` (`k_block_tiles`),
`e` (expert slots, `NUM_EXPERTS`).

Every CB has the same index and size on every core of the layout (multicast slot addresses must agree).

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_x_round` | `X_DEPTH * core_row_tiles_max * k_block_tiles` | 2 rounds: the round being computed plus the round landing (`2 * core_row_tiles * k_block_tiles` filled pages; capacity is nominal `core_row_tiles_max`) | `{m: spans -> core_row_tiles_max, k: spans -> k_block_tiles, n: streams (one x round serves every sub-block of the core's n-slice), e: streams (sub-blocks of all experts share the round), wave: streams}` | Bfp8_b | reader | compute | whole kernel, recycled per round | none: concurrent with `cb_w_round` and `cb_partials` in every round. Capacity above live set only on ragged cores (nominal pushes keep the ring aligned to whole rounds). Depth 2 is the overlap mechanism. |
| `cb_w_round` | `W_DEPTH * NUM_EXPERTS * core_n_tiles * k_block_tiles` | 2 rounds of in-wave expert chunks; empty / out-of-wave slots are pushed but hold no data | `{e: spans -> NUM_EXPERTS slots, n: spans -> core_n_tiles, k: spans -> k_block_tiles, m: streams (one chunk serves every row of the core's sub-block), wave: streams}` | Bfp8_b | writer RISC (w channel, see Implementation note) | compute | whole kernel, recycled per round | none: concurrent with `cb_x_round` and `cb_partials`. Gap between capacity and live set: unused slots of experts that are empty, outside the wave, or not touching this core. Slots stay nominal because the multicast landing address must be identical on every line core (R2 does not change this). |
| `cb_partials` | `core_row_tiles_max * core_n_tiles` | the whole wave block, `core_row_tiles * core_n_tiles` tiles, live from round 0 to the untilize | `{m: spans -> core_row_tiles_max, n: spans -> core_n_tiles, k: streams (folded by packer L1 accumulation each round), e: streams (sub-blocks of every expert occupy disjoint rows of the same block), wave: streams (recycled per wave)}` | Float32 (`fp32_dest_acc_en = True`, the only supported value) | compute (pack) | compute (unpack, untilize) | one wave | none: live across every round of the wave, so it overlaps `cb_x_round` / `cb_w_round` for its whole life; at the untilize it overlaps `cb_output_rows`. Not aliased with the next wave's x/w because the reader prefetches wave p+1 rounds 0-1 during wave p's untilize (explicit pipelining decision). |
| `cb_output_rows` | `out_depth_blocks * core_n_tiles`: `max(OUT_DEPTH, core_row_tiles_max)` blocks when the store is paced (`num_waves >= 2`), else `OUT_DEPTH` = 2 | paced: the whole untilized wave block, drained over wave p + 1's rounds; unpaced: 2 tile rows (one being written, one being packed) | `{n: spans -> core_n_tiles, m: spans -> core_row_tiles_max (paced) / streams (unpaced), k: none (after accumulation), e: none, wave: streams}` | Float16_b (output dtype) | compute | writer RISC (L1 y) or reader RISC (DRAM y, `store_on_reader`) | untilize of wave p to the end of wave p + 1's store | none: concurrent with `cb_partials` during the untilize and with every x / w round of wave p + 1 while the paced store drains. Cannot alias `cb_partials` (format, overlapping lifetime). The whole wave block is what lets compute start wave p + 1 without waiting on the store (Refinement 2: t = 4096 172 -> 163 us). |
| `cb_x_stage` (implementer) | `core_row_tiles_max * k_block_tiles` | 1 step: the next owned x step's rows, prefetched by the rotating sender | `{m: spans, k: spans, n/e/wave: streams}` | Bfp8_b | reader (NoC reads) | reader (multicast source) | whole kernel, RISC-private, no CB credits | none: live from the prefetch issue (one step after the sender's previous send) to its next send, overlapping `cb_x_round`'s two slots. It decouples the sender's L1 reads from the slot ring (X_DEPTH alone would cap the lookahead at one round). |
| `cb_w_stage` (implementer) | `NUM_EXPERTS * core_n_tiles * k_block_tiles` | 1 step of in-wave expert chunks | `{e, n, k: spans; m, wave: streams}` | Bfp8_b | DRAM reads split by tile: `w_reader_share / W_SHARE_DEN` by the reader RISC (NoC1, CB credits: reserve / push one step before the owned step), the rest by the writer RISC (NoC0, RISC-private writes into the same slots) | writer RISC (multicast source; waits / pops when `w_reader_share > 0`) | whole kernel | none: same reason as `cb_x_stage`; lets every line keep several steps of w DRAM reads in flight. Refinement 3: the reader share takes part of the read responses off NoC0, where they contended with the w multicast (t = 1024 102.8 -> 85 us). With share 0 (any blocking with >= 2 waves) it stays writer-private, no credits move. |
| `cb_counts_compute` | 1 (32 B page, constant) | 1 | `{m, n, k, e, wave: none}` (constant) | UInt32 | reader | compute | whole kernel | none: a second consumer (writer) would violate the one-consumer rule, hence `cb_counts_writer`. 32 B. |
| `cb_counts_writer` | 1 (32 B page, constant) | 1 | `{m, n, k, e, wave: none}` (constant) | UInt32 | reader | writer | whole kernel | none: see `cb_counts_compute`; the reader issues one 32 B DRAM read per CB. |
| `cb_w_owner` (Refinement 3) | 1 (16 B page, constant) | 1 | `{m, n, k, e, wave: none}` (constant) | UInt32 | writer (the w sender's first owned step and rotation period, from the mcast helper's `should_send`) | reader (its w share's step ownership) | kernel start, used only when `w_reader_share > 0` | none: written before either RISC's main loop. A separate CB because the mcast host adapter refuses a sender channel on a kernel of another NoC, so the reader cannot carry the w channel args. 16 B. |

Pattern check (Rule 3):

| Pattern | Applied / why not |
|---------|-------------------|
| Pack into the destination | The matmul packs straight into `cb_partials`, the K accumulator. No per-round scratch. |
| Transform in place | Untilize changes layout and format (fp32 tile -> bf16 RM), so it needs its own output CB. |
| Alias disjoint lifetimes | No two CBs have disjoint lifetimes: x / w / partials are live together every round, and output / partials overlap at the untilize. |
| Fold into the accumulator | Within a round, the `k_block_tiles` products of a subblock accumulate in DEST; across rounds the packer folds them into `cb_partials` in L1. A per-subblock DEST accumulator across rounds is impossible (a core's block is up to 30 tiles, DEST holds 4 fp32 tiles). |

## Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `tensor_row_tiles` | `R / 32 <= 263` | `validate()`: R % 32 == 0 and R <= 8416 |
| `num_m_groups` | 9 (L layout, p100a), 11 (Rect, p100a), `grid.x` or `grid.y - 1` in general | `select_layout(grid, tensor_row_tiles)` |
| `num_n_lines` | divisor of 24: 12 (L), 8 (Rect) | `select_layout` |
| `core_n_tiles` | `24 / num_n_lines` in {2, 3} on p100a, <= 24 always | `select_layout` |
| `num_waves` | 1..`MAX_WAVES` = 4 | `select_blocking`: with y in L1, the fewest waves at `k_block_tiles >= K_BLOCK_ONE_PASS_MIN` = 6 and `core_row_tiles_max <= ONE_PASS_MAX_ROWS` = 24 that fit; otherwise the largest `k_block_tiles`, then the fewest waves, one wave with y in DRAM only up to `DRAM_Y_ONE_PASS_MAX_ROWS` = 16 rows per core (Refinement 3: t = 3072 L1 / L1 is 1 wave at k = 6; Refinement 4: t = 4096 under the model's occupancy is 2 waves at k = 6) |
| `core_row_tiles_max` | `ceil(ceil(tensor_row_tiles / num_waves) / num_m_groups)` <= 24 on p100a (23 at t = 3072 one wave) | `select_blocking`; device `core_row_tiles <= core_row_tiles_max` because the device clamps `active_row_tiles <= tensor_row_tiles` |
| `k_block_tiles` | divisor of 96, <= `K_BLOCK_MAX` = 8 | `select_blocking` |
| `x_step_rounds` | 1..`X_STEP_ROUNDS_MAX` = 1 (parked knob: an x step carries that many compute K rounds; `cb_x_round` / `cb_x_stage` scale with it) | `select_blocking` tries 1 first, then splits the x K block |
| `NUM_EXPERTS` | 8 | `validate()`: E == 8 |
| `X_DEPTH`, `W_DEPTH`, `OUT_DEPTH` | 2 each (`cb_output_rows` grows to the wave block when `STORE_PACED` and `num_waves >= 2`) | host constants; `out_depth_blocks()` |
| CB budget | per-bank bytes from the L1 allocator base to the lowest live L1 buffer at call time (Refinement 4; replaces the 976 KiB `CB_BUDGET_BYTES` constant). Empty L1, R = 8416, x and y in L1 on p100a: 1,092,352 B. Nomic 8x512 model occupancy: 853,568 B. | `cb_budget_bytes(device)` reads `ttnn.get_memory_view` (block addresses are relative to the allocator base, where the CB region starts). `validate()` predicts the output's placement (`output_l1_bank_bytes`; charged unless a free block above the lowest live buffer holds it) and refuses (`UnsupportedAxisValue`) when no blocking fits. `footprint_bytes()` rounds each CB up to `CB_ADDR_ALIGN` = 64 B. |

## Footprint

Host source of truth: `cb_plan()` in `moe_down_proj_program_descriptor.py` (the budget solve and the CB allocation
both read it).

```
footprint = (X_DEPTH + 1) * core_row_tiles_max * k_block_tiles * 1088      # + cb_x_stage
          + (W_DEPTH + 1) * NUM_EXPERTS * core_n_tiles * k_block_tiles * 1088  # + cb_w_stage
          + core_row_tiles_max * core_n_tiles * 4096
          + out_depth_blocks * core_n_tiles * 2048         # max(OUT_DEPTH, core_row_tiles_max) when paced
          + 2 * 32 + 16                                     # counts, cb_w_owner
```

| Term | Scales with |
|------|-------------|
| `cb_x_round` | `k_block_tiles` x `core_row_tiles_max` (so with `1 / num_waves` and `1 / num_m_groups`), linear in `X_DEPTH` |
| `cb_w_round` | `k_block_tiles` x `core_n_tiles`, linear in `W_DEPTH`; independent of rows |
| `cb_x_stage`, `cb_w_stage` | one round of `cb_x_round` / `cb_w_round` each (depth 1, sender-private) |
| `cb_partials` | `core_row_tiles_max` x `core_n_tiles`; independent of `k_block_tiles` |
| `cb_output_rows` | `core_n_tiles` x (`core_row_tiles_max` when paced, else `OUT_DEPTH`) |
| counts, `cb_w_owner` | constant |

p100a, L layout (`core_n_tiles` = 2, `num_m_groups` = 9, `k_block_tiles` = 8): t = 4096 (2 waves, `core_row_tiles_max` = 15) 391,680 + 417,792 + 122,880 + 8,192 + 64 = 940,608 B (staging included); with the paced store `cb_output_rows` is 15 blocks (61,440 B), so 993,856 B. One wave would exceed the budget, which is why `num_waves` = 2 there.

Implementation note: the w channel runs on the writer RISC (NoC0) and x on the reader RISC (NoC1), each sender multicasting from its staging buffer; w moves as one multicast per K round covering the wave's expert slot range. With one wave, `w_reader_share()` eighths of the w DRAM reads are issued by the reader RISC on NoC1 (4 down to 0 as rows per core grow, see the descriptor).

p100a, t = 4096 under the model's occupancy (853,568 B free): 2 waves, `k_block_tiles` = 6, `core_row_tiles_max` = 15: 293,760 + 313,344 + 122,880 + 61,440 + 192 = 791,616 B (64 B-aligned CBs). 2 waves at k = 8 (993,984 B) does not fit, and every other fitting blocking measured slower (3 waves k = 8 179.6 us; 2 waves k = 4 with x steps of k = 8 179.6 us; 2 waves k = 4 186 us; vs 171-172 us).

p100a, t = 3072 L1 / L1 (1 wave, `k_block_tiles` = 6, `core_row_tiles_max` = 23): 450,432 + 313,344 + 188,416 + 8,192 + 80 = 960,464 B.

## Data-movement budget

Per call at t = 4096, R = 8416, L layout, 2 waves, layer-0 counts (`active_row_tiles` = 259).

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| w (20.05 MB) | 1 + at most 1 straddling expert (<= 2.5 MB): 22.5 MB worst case | each line reads each in-wave expert's n-slice once per wave; an expert spanning the wave boundary is in both waves | multicast along each n-line to `num_m_groups - 1` = 8 receivers; every core receives every in-wave chunk (about 1.9 MB per core) |
| x (27.5 MB) | 0 (L1, Phase 0); 1 when x is in DRAM | each x tile is read by exactly one sender (its m-group's, rotating per round) | NoC reads 27.5 MB into the senders; multicast to `num_n_lines - 1` = 11 receivers (row multicast plus the strip copy) |
| counts (32 B) | 2 per core (216 reads of 32 B) | one read per counts CB | none |
| y (12.9 MB) | 0 (L1, Phase 0); 1 when output is in DRAM | each row segment written once by its owner | NoC writes 12.9 MB in 128 B segments (DRAM y: issued by the reader RISC on NoC1) |

One-wave shapes (t <= 3072 with y in L1, t <= 2048 otherwise) read w exactly once (20.05 MB, no straddling expert). With
`w_reader_share` > 0, that share of the w read responses travels NoC1 instead of NoC0.

Totals per tier: DRAM 20.05 to 22.5 MB (w), plus 27.5 MB (x) and/or 12.9 MB (y) for the DRAM placements. L1-to-NoC reads 27.5 MB (x). Multicast payload received: about 1.9 MB w + 3.1 MB x per core. Writes 12.9 MB.

> Cheapest-traffic split considered: 2D split with range-pruned w multicast (R2), same DRAM (22.5 MB) and same link bytes; it cuts w bytes received per core from about 1.9 MB to about 0.3 MB. Implemented: R1 (2D split, static line multicast). Deferred because the only difference is receiver ejection bytes, R1's predicted per-core receive (about 5.0 MB over at least 68 us of compute) is under two NoCs' ejection capacity, and R2 needs a raw per-expert semaphore protocol that `ttnn.Mcast` static groups cannot express. R2 keeps R1's slot layout and push sequence, so it replaces only the multicast destination and handshake.

## Verifier notes (Phase 0)

- The R2 deferral reason above ("receive bytes under ejection capacity") is a prediction. The implementer's ablation
  measured a 130 us sync + multicast floor at t = 4096 (compute about 15 us of it), so the multicast scaffold, not
  the DRAM stream, binds there. The receive-byte reduction (R2 plus a layout re-solve) is filed as perf
  Refinement 2 in `op_requirements.md`.
- Refinement 4: the budget is now read from the allocator at call time, so the clash below cannot happen; an occupancy
  with no fitting blocking is refused in `validate()`.
- The CB region ends at about 1,052,640 B (960 KiB budget above the unreserved base); after Refinement 2 (976 KiB
  budget, 993,856 B used at t = 4096) at about 1,063,500 B. Any L1 tensor below that
  address at call time makes program creation throw `Statically allocated circular buffers ... clash with L1
  buffers`. With x and y in L1 at R = 8416 that leaves no room for other L1 residents.
