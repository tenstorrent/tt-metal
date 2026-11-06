# L1 Ledger: nomic_sdpa

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes (from `op_design.md` Blocking
Model): `query_row` (extent `Q = q_block_rows`), `key` (extent `N = key_block_tiles`), `head_dim`
(`HD = head_dim_tiles = 2`), `pv_cols` (`PV = pv_out_tiles = HD + 1 = 3`). Tile = 2048 B (Float16_b, 32x32).
Compute config: **fp32 DEST** by default (`fp32_dest_acc_en = True`, implementer change: the bf16-DEST P·V accumulation
misses the golden bar at S = 512, measured). Every CB page stays Float16_b: DEST is wider than every page, so pack
narrows and unpack widens; no page needs fp32 (audit 2). DEST capacity halves (4 tiles half-sync): the qk subblock
width derives from `DEST_AUTO_LIMIT` in-kernel.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_query` | `Q*HD*q_depth` (`q_depth = 2` on L1 placement, 1 on DRAM; falls back to 1 when it does not fit — Refinement 4) | `block_rows*HD` (≤ capacity; ragged blocks pad) | `{query_row: spans → Q, key: n/a (not an axis of Q) → none, head_dim: spans → HD, pv_cols: n/a → none}` | Float16_b | reader | compute | `load_query_block` → `qk_block` | none: concurrent with every compute-internal CB (the next block's Q loads while this block runs); different producer (reader) from every candidate partner |
| `cb_key_t` | `N*HD*kv_depth` (`kv_depth` = min(cap, most segments a core has); cap 1 for `buffer = l1`, 2 for `buffer = dram`; falls back to 1 when 2 does not fit live L1 — Refinement 2) | `N*HD` (whole pair, K^T order); at `kv_depth = 2` the second slot holds the next pair, pre-sent / multicast at t = 0 |  `{query_row: streams (reused by every block of the segment) → none, key: spans → N, head_dim: spans → HD, pv_cols: n/a → none}` | Float16_b | reader | compute | segment: `load_kv_block` → last `qk_block` | none: live while `cb_value_aug` is live and while the next segment's K is being loaded; aliasing with `cb_value_aug` impossible (both resident during `qk_block`..`pv_block` of one segment). Pair p sits in ring slot `p % kv_depth` on every core (reader pushes `p0 % kv_depth` data-less phantom slots first, compute pops them), so a multicast lands at the same address on every group member |
| `cb_value_aug` | `N*PV*kv_depth` (same `kv_depth` as `cb_key_t`) | `N*PV` (V plus the ones column; ones slots `st*3+2` permanent; the multicast moves only the HD value tiles of each row, never the ones tiles) | `{query_row: streams → none, key: spans → N, head_dim: spans → HD (inside PV), pv_cols: spans → PV}` | Float16_b | reader | compute | segment: `load_kv_block` → last `pv_block` | none: resident across the whole segment; the ones column must stay at fixed slots, which forbids aliasing |
| `cb_scores` | block-sequential: `Q*N`; **row-group (Refinement 3, default): `2*G*N`** | block-sequential: `block_rows*N`, scores then P **in place** (Rule 3 pattern 2). Row-group: group t's scores (draining one probs window at a time) plus group t+1's (filling), `≤ 2*G*N` | `{query_row: spans → Q (row-group: streams in groups of G → 2G), key: spans → N, head_dim: n/a (contracted) → none, pv_cols: n/a → none}` | Float16_b | compute | compute | `qk_block` → `pv_block` (row-group: `qk_window` → `probs_window`) | block-sequential: scores and P are one buffer. Row-group: P can no longer share it — the ring is a FIFO read from its front by `reduce` / the matmul, and group t+1's scores must sit behind group t's while group t's P waits for P V, so P gets `cb_probs` |
| `cb_probs` | **row-group only (Refinement 3): `2*G*N`** | group t's P (filling, window by window) plus group t-1's (draining into the P V windows), `≤ 2*G*N` | `{query_row: streams in groups of G → 2G, key: spans → N, head_dim: n/a → none, pv_cols: n/a → none}` | Float16_b | compute (pack thread, after the pack-thread exp) | compute (P V windows) | `probs_window` → `pv_window` | none: concurrent with `cb_scores` (the probs window reads scores while it packs P) and with `cb_pv_out` (in0 of the matmul that packs it). Two scores+P rings of `2*G*N` replace the block-sized `Q*N`: 64 vs 224 pages at 8x512 |
| `cb_row_stats` | `Q` (row-group: `G`) | `block_rows` (row-group: the group's rows) | `{query_row: spans → Q, key: n/a (reduced) → none, head_dim: n/a → none, pv_cols: n/a → none}` | Float16_b | compute | compute | two disjoint uses a block: row max (`row_max_block` → `probs_block`), then `1/l` (`normalize_block`) | shared by **reuse** of one CB for both disjoint lifetimes (Rule 3 pattern 3, same format, same producer/consumer). Folding `1/l` into DEST instead is lamp L5 |
| `cb_pv_out` | `Q*PV` (row-group: `G*PV`) | `block_rows*PV` (`O` 2 tiles + `l` 1 tile a row; row-group: the group's rows) | `{query_row: spans → Q, key: n/a (contracted) → none, head_dim: spans → HD (inside PV), pv_cols: spans → PV}` | Float16_b | compute | compute | `pv_block` → `normalize_block` | none: concurrent with `cb_scores` (in0 of the matmul that packs it) and with `cb_row_stats` / `cb_out` (inputs and output of the normalize chains). Pack-into-destination is blocked because `cb_out`'s consumer is the writer: an in-place normalize there would give `cb_out` two consumers. Eliminating it via the matmul `PostComputeFn` is lamp L5 |
| `cb_out` | `Q*HD*out_depth` (`out_depth = 1` block) | 2 pages per row in flight; up to `block_rows*HD` when the writer lags | `{query_row: spans → Q, key: n/a → none, head_dim: spans → HD, pv_cols: n/a → none}` | Float16_b | compute | writer | `normalize_block` → `store_block` | none: concurrent with the next block's compute-internal CBs (the writer drains while compute proceeds). Capacity above the 2-page live set is a deliberate decoupling choice: compute never waits on the writer |
| `cb_reduce_scaler` | `1` (constant) | 1 | `{query_row: none, key: none, head_dim: none, pv_cols: none}` | Float16_b (bf16 1.0, MAX/REDUCE_ROW layout) | writer | compute | whole kernel | none: constant read by every `row_max_block`; 1 tile |

## Symbol table

| Symbol | Meaning | Bound | Predicate establishing it |
|--------|---------|-------|---------------------------|
| `S_t` | `ceil(S/32)` | `1 ≤ S_t ≤ 16` | validate(): `S ≤ 512` |
| `B` | batch | `B*S_t ≤ 128` (so `B ≤ 128`; INPUTS reach 32) | validate(): `B*S_t*32 ≤ 4096` |
| `tensor_query_rows` | `12*B*S_t` | `≤ 1536` | same |
| `num_cores` | `min(110, tensor_query_rows)` | `≤ 110` | validate(): grid 11x10 |
| `Q = q_block_rows` | `ceil(tensor_query_rows / num_cores)` | `≤ 14` | the two predicates above (`ceil(1536/110) = 14`) |
| `N = key_block_tiles` | `= S_t` | `≤ 16` | `S ≤ 512` |
| `HD` | head-dim tiles | `= 2` | validate(): head dim 64 |
| `PV` | `HD + 1` | `= 3` | derived |
| `q_depth`, `out_depth` | buffer depths | `= 1` | host `plan` constants |
| `kv_depth` | K/V ring depth (pairs) | `1 ≤ kv_depth ≤ 2`: `min(cap, max segments a core)`, cap `KV_DEPTH_L1 = 1` / `KV_DEPTH_DRAM = 2` | host `plan`; entry point falls back to 1 when the CBs at 2 do not fit below live L1 |
| `block_rows` | runtime rows of a block | `1 ≤ block_rows ≤ Q` | segment construction |
| `G = row_group_rows` | rows a group of the row-group schedule (Refinement 3) | `1 ≤ G ≤ Q`, default 1 | host `ROW_GROUP_ROWS`, clamped in `plan`; the schedule runs only when `Q ≥ EXP_OVERLAP_MIN_ROWS = 3` and the exp is the approximate one |
| tile bytes | Float16_b 32x32 | `2048` | every CB is Float16_b under either DEST width |

## Total per-core footprint

**Row-group schedule (Refinement 3, the default under the approximate exp when `Q ≥ 3`)**: `cb_scores` and
`cb_probs` are `2*G*N` each, `cb_row_stats` `G`, `cb_pv_out` `G*PV`:

```
pages = Q*HD*q_depth + N*HD*kv_depth + N*PV*kv_depth + 4*G*N + G + G*PV + Q*HD*out_depth + 1
      = 4*Q + 5*N + 4*G*N + 4*G + 1                 (HD = 2, PV = 3, all depths 1)
```

| Shape | Q | N | G | KiB (row-group) | KiB (block-sequential) |
|-------|---|---|---|-----------------|------------------------|
| 8x512 | 14 | 16 | 1 | 410 | 834 |
| 4x512 | 7 | 16 | 1 | 354 | 498 |
| 2x512 | 4 | 16 | 1 | 330 | 354 |
| 32x128 | 14 | 4 | 1 | 194 | 378 |
| 16x256 | 14 | 8 | 1 | 266 | 530 |

The `Q*N` term is gone: the footprint now scales with `Q + N`, and 8x512 leaves ~500 KiB of the 914 KiB free at the
176 KB-reserved perf case. **Refinement 4** spends `+Q*HD` pages of it on `q_depth = 2` at L1 placement (8x512: 410 →
466 KiB; 4x512 382, 2x512 346 KiB); DRAM placement keeps `q_depth = 1` (16x256 DRAM measured slower at 2). `kv_depth =
2` (+160 KiB) fits too but measured slower and stays parked at `KV_DEPTH_L1 = 1` (see *Measured data movement,
Refinement 4*). Block-sequential (exact exp, `Q ≤ 2`, or `EXP_OVERLAP = False`):

```
pages = Q*HD*q_depth + N*HD*kv_depth + N*PV*kv_depth + Q*N + Q + Q*PV + Q*HD*out_depth + 1
      = Q*N + 8*Q + 5*N + 1                         (HD = 2, PV = 3, all depths 1)
bytes = 2048 * pages
```

| Term | Scales with |
|------|-------------|
| `Q*N` (`cb_scores`) | `q_block_rows` x `key_block_tiles` — the dominant term (54% at the largest shape) |
| `8*Q` (`cb_query` 2Q, `cb_row_stats` Q, `cb_pv_out` 3Q, `cb_out` 2Q) | `q_block_rows` (and `q_depth`, `out_depth`) |
| `5*N` (`cb_key_t` 2N, `cb_value_aug` 3N) | `key_block_tiles` x `kv_depth` |
| `1` | constant |

| Shape | Q | N | pages | KiB | Free L1 below live tensors at the call |
|-------|---|---|-------|-----|----------------------------------------|
| 8x512 (max over the domain) | 14 | 16 | 417 | 834 | 1055 KiB |
| 4x512 | 7 | 16 | 249 | 498 | 1179 KiB |
| 2x512 | 4 | 16 | 177 | 354 | 1217 KiB |
| 32x128 | 14 | 4 | 189 | 378 | — |
| 16x256 | 14 | 8 | 265 | 530 | — |
| 8x384 | 11 | 12 | 281 | 562 | — |

No budget solve: the inventory (8 CBs, one of them reused for two lifetimes, scores→P in place) fits at the
domain maximum with 221 KiB to spare at the perf focus. `kv_depth = 2` (lamp L6; Refinement 2's DRAM default) adds
`5*N` pages (160 KiB at `N = 16`: 994 KiB at 8x512, 658 at 4x512, 514 at 2x512). **Correction (Refinement 2,
measured):** the perf-flagged case (8x512, L1, 176 KB reserved) leaves **914 KiB** below the live tensors once the
output is allocated, not 1055 — 994 KiB does **not** fit there. Hence the L1 cap of 1 and the fit fallback in the
entry point; `q_block_rows = 12` would fit `kv_depth = 2` (898 KiB) but measured no faster (120.0 vs 120.3 us).

## Data-movement budget

Boundary: the io-tensor memory (L1 banks over the NoC for `buffer = l1`, DRAM for `buffer = dram`). The
"DRAM crossings" column counts crossings of that boundary; under `buffer = dram` they are literal DRAM crossings.
Each tensor is `12*B*S_t*2` tiles (6 MiB at 8x512, 3 MiB at 4x512, 1.5 MiB at 2x512).

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| query | 1 | each query row is owned by exactly one core and read once | none |
| key | Phase 0: `segments / pairs` = 2.00 (8x512), 3.15 (4x512), 4.88 (2x512), 9.17 (1x512), 1.15 (32x128), 1.43 (16x256), 2.00 (8x384). **Refinement 2 (pair multicast)**, reads per pair: `buffer = dram` (kv_depth 2) 1.00 / 1.00 / 1.00 / 1.00 / 1.15 / 1.29 / 1.00; `buffer = l1` (kv_depth 1) 1.85 / 1.85 / 1.29 / 1.00 / 1.15 / 1.43 / 1.85. Multicast off below `S_t = 8` (`KV_MCAST_MIN_SEQ_TILES`) | one read per pair GROUP (the cores holding it at a segment of index `< kv_depth`), by the group's first core; the cores holding it at a later segment read their own copy | the group's first core multicasts K (one `N*HD`-tile block) and V (`N` payloads of HD tiles) to the rest of its consecutive core run on NOC0 (forward routing only: up to 3 rectangles) + one Flag semaphore per (ring slot, K / V). Receiver ingress unchanged |
| value | same as key | same; the ones column is generated in L1 (0 bytes) and never multicast | same as key |
| output | 1 | each output tile written once, straight to its concatenated-heads page | none |
| scores / P | 0 | resident in `cb_scores` (one core's block) | none |

| Shape | q | k | v | out | Total (Phase 0) | Total (R2, `buffer = l1`) | Total (R2, `buffer = dram`) | Minimum |
|-------|---|---|---|-----|-----------------|---------------------------|-----------------------------|---------|
| 8x512 | 6 MiB | 12 MiB | 12 MiB | 6 MiB | 36 MiB | 34.2 MiB | 24 MiB | 24 MiB |
| 4x512 | 3 MiB | 9.4 MiB | 9.4 MiB | 3 MiB | 24.9 MiB | 17.1 MiB | 12 MiB | 12 MiB |
| 2x512 | 1.5 MiB | 7.3 MiB | 7.3 MiB | 1.5 MiB | 17.6 MiB | 6.9 MiB | 6 MiB | 6 MiB |

(k / v columns are Phase 0; the R2 totals use the reads-per-pair factors in the budget table.)

Per-core ingress (bounds a core's load time): at most `Q*HD` Q tiles + 2 segments x (`N*HD` K + `N*HD` V) tiles =
56 KiB + 256 KiB at 8x512; 16 KiB + 256 KiB at 2x512. Core-local re-reads: K and V are read from L1 by the
unpacker once per block (one block per segment in the domain); P once by `pv_block`; `cb_pv_out` twice
(recip column, then the two O columns).

> Cheapest-traffic split considered: **pair-per-core** — 24 MiB vs 36 MiB at 8x512 (−12 MiB across the io
> boundary, 0 cross-core), but makespan 16 rows a core vs 14 (+14%) and 14 / 62 / 86 idle cores at 8x512 /
> 4x512 / 2x512; rejected in the regime table. Cheapest-traffic split that keeps full occupancy: **row-split +
> pair multicast** — 24 MiB at the source (−12 MiB at 8x512, −12.9 at 4x512, −11.6 at 2x512), receiver ingress
> unchanged. Implemented: **row-split, per-core K/V read**. Deferred because under L1 placement (every perf
> shape) the per-core ingress that bounds the K/V load is byte-identical with or without multicast — the saving
> is source-bank egress spread over 110 banks — while DRAM placement, where the saving is real, is not a
> placement any perf case or the model's 1k-4k passes use. The structure keeps it reachable: cores sharing a pair
> are consecutive in `core_row_start` and land K/V at identical CB addresses.

## Measured data movement (implementer, Phase 0, Blackhole p300c)

Ablation (`NOMIC_SDPA_ABLATE`, device kernel time, L1 placement; DRAM placement within noise of it):

| Shape | full | DM floor (compute stubbed) | compute floor (reads+writes stubbed) | target |
|-------|------|----------------------------|--------------------------------------|--------|
| 2x512 | 53-63 us | 39-49 us | 22 us | 18 us |
| 4x512 | 71-83 us | 45-57 us | 38 us | 30 us |
| 8x512 | 125-137 us | 70-80 us | 73 us | 58 us |

The DM floor moves the budget above (17.6 / 24.9 / 36 MiB) at an aggregate ~430-460 GB/s for **every** shape and for
L1 and DRAM placement alike, while each core ingests only ~6-9 GB/s. The load time is therefore bounded by the total
bytes on the NoC, not by per-core ingress as the regime table assumed: the K/V re-reads (2.00x / 3.15x / 4.88x) are
the binding cost, and the deferred **row-split + pair multicast** regime (or pair-aligned assignment, lamp L7) is what
moves the DM floor. `kv_depth = 2` (lamp L6) is correct (acceptance 39/39) and frees the reader earlier (8x512: 81 vs
107 us) but does not move the wall (the math thread then binds); parked at 1 as a live knob.

## Measured data movement (implementer, Refinement 2 — pair multicast)

Device kernel us, median of 3 (Blackhole p300c, default HiFi2 / fp32 DEST), Phase 0 → Refinement 2:
L1 2x512 59.1 → 30.5, 4x512 76.0 → 53.8, 8x512 ~122-134 → ~120-127 (7-sample medians overlap: unchanged);
DRAM 2x512 60.7 → 34.4, 4x512 86.2 → 63.6, 8x512 134.7 → 117.8; 1x512 L1 49.5 → 17.8, 8x384 L1 82.0 → 74.3.
8x512 ablation after the multicast: DM floor (compute stubbed) 50 us (was 70-80), compute floor 74 us, bank reads
stubbed 78 us. Device zones on compute's input waits: the first block waits for Q + K until **38.5 us median / 54 us
max** after kernel start — the chip-wide t = 0 burst (every core's first-segment K + Q, ~9 MiB, at ~2.4 GB/s a core
on NOC0) is what binds 8x512 now, not total bytes: the multicast removes only second-segment bytes, which already
overlapped compute. A multicast whose receivers sit *behind* its sender on NOC0 (east/south routing) wraps the torus
row: the first build (sender = the first core whose first segment is the pair) measured 8x512 126 → 175 us; the
shipped topology sends from the run's first core only.

## Measured data movement (implementer, Refinement 4 — startup burst)

Device zones at 8x512 L1 (176 KB reserved) found the wall was **startup + compute in series**: compute's first K landed
at 31.5 us median and its first V at 46 us. Before that, each core's reader moved its whole 14-row Q block (~56 KiB),
which compute consumes over the next ~70 us. The chip-wide t = 0 burst was Q + K + V ≈ 19 MiB at ~400 GB/s. With
`Q_LEAD_GROUPS = 2` (two Q rows ahead of K / V, the rest after V), the first K lands at 18 us and the first V at 35 us.
The burst is now K + V of the first segments (~12.6 MiB), the minimum for this regime. Bytes moved per call are unchanged.
`kv_depth = 2` (each pair read once, ~12 MiB less per call at 8x512) measured slower: 8x512 L1 103.6 → 109.1 us. Under it
almost every core gets its first-segment K / V through the group sender's read-then-multicast, a store-and-forward hop
on the startup's critical path (first V at 55 us vs 46 us at depth 1). The pair-aligned table (lamp L7) changes
no byte count. It gives each core one segment, so its K / V comes from one group read at t = 0.

## Verifier audit (Phase 0, 2026-11-05)

- **Currency**: the 8 rows match `circular_buffer_specs()` (the single sizing source for the descriptor and
  `circular_buffer_bytes()`): indices, page counts and the Float16_b format all agree with the code.
- **Capacity vs live set**: no unexplained over-capacity. `cb_out` (capacity `Q*HD`, live set 2 pages a row) is a
  stated decoupling choice. `cb_pv_out` exists only because normalize runs as two chains after the matmul. Removing it
  (normalize in the P·V `PostComputeFn`, lamp L5) is folded into **Refinement 4** in `op_requirements.md`, not filed
  on its own. No collapsed extent: every buffer that spans `query_row` scales with `Q`, and every buffer that spans
  `key` scales with `N`.
- **Page format vs DEST (fp32 DEST, 16-bit pages, the "under" direction)**: justified. No accumulation crosses a
  phase boundary. Q·K^T (K = `HD`) and P·[V|1] (K = `N`) each finish in DEST in a single K block
  (`num_k_blocks = 1`, `packer_l1_acc = false`), so each pack rounds a *finished* value to bf16. `cb_scores` is the
  incumbent's bf16-score class. The one remaining rounding, `[O | l]` to bf16 before `O / l`, is ~2^-9 relative and
  sits below the measured error, which is dominated by the approximate exp (see `verification_report.md`).
- **Disjoint lifetimes**: the only disjoint pair, the row max and `1/l`, already shares `cb_row_stats`. Every other
  pair is concurrent, with a stated reason.
- **Bounds / closed form**: complete (symbol table above).
- **Cheapest-traffic split**: the deferral reason for **row-split + pair multicast** ("per-core ingress is
  byte-identical, so multicast only saves source egress") is **falsified by the implementer's own measurement
  above**. The DM floor is bound by aggregate NoC bytes, so the K/V re-reads (2.00x / 3.15x / 4.88x) are the binding
  cost. The deferred row therefore no longer has a positive reason, and the regime is filed as **Refinement 2**
  (perf, reuse multicast).
- **Block-size defaults**: held. Interleaved work spreads over the full grid (110 cores for every shape with at least
  110 query rows), and `q_block_rows` is the whole per-core assignment (the coarsest block). No budget-driven split
  was taken.
