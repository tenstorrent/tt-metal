# L1 Ledger: groupnorm_sc_N_1_HW_C

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes (from `op_design.md` → Blocking Model): `n`, `hw`, `ct`, `g`, `lane`, `c_affine`. `fp32_dest_acc_en = True` (default config) ⇒ every compute-produced statistic page is `Float32`.

Symbols: `x_page` input tile bytes (2048 bf16 / 4096 fp32 / 1088 bf8b — all three dtypes are SUPPORTED after verification); `y_page` output tile bytes; `g_page` affine tile bytes; `F32 = 4096`; `cols = cols_per_group`; `chunk = chunk_rows · cols`; `Kg = ceil(G/32)`; `P_max = max_images(P_used)`; `blk = Ht_core · Ct_core`; `out_block = largest divisor of chunk ≤ out_block_tiles_target (8)` (writer store block, independent of `cols`).

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_x_pass1` | `resident ? blk : x_depth · chunk` | streaming: `chunk` (+1 chunk in flight); resident: `blk` (the point of the regime) | `{n: streams → 1 image, hw: spans → chunk_rows (resident: Ht_core), ct: spans → cols (resident: Ct_core), g: streams, lane: spans → 32, c_affine: streams}` | input dtype | reader (TILE) / compute tilize (RM) | compute | pass 1 | **aliased with `cb_x_pass2`** (same region, same page size, identical quanta from base). Capacity > live set in streaming by one chunk = `x_depth` double-buffering |
| `cb_x_pass2` | resident: same as `cb_x_pass1` (alias); **streaming: own ring of `x_depth · chunk`** | as above | as above | input dtype | same producer | compute (apply) | pass 2 | resident: alias of `cb_x_pass1` — zero additional bytes; the two credit counters give pass 2 a real `push→wait` edge over bytes pass 1 did not evict. **Streaming (implementer deviation): a separate region** — with one aliased ring the reader's pass-2 prefetch (which the stall-shadow analysis wants) could reserve slots pass 1's counter had already freed while compute was still reducing those bytes; `x_depth · chunk · x_page` (128 KiB at the defaults) buys a race-free prefetch |
| `cb_x_rm` | RM only: `x_rm_depth · cols` | `cols` (one tile-row of sticks) | `{n: streams, hw: streams → 32 sticks, ct: spans → cols, g: streams, lane: spans, c_affine: streams}` | input dtype | reader | compute | pass 1 (+2 streaming) | not shared: concurrent with `cb_x_pass1` (tilize reads it while the ring holds tiles). Capacity = live set × depth 2 |
| `cb_xsq` | `chunk` | `chunk` | `{n: streams, hw: spans → chunk_rows, ct: spans → cols, g: streams, lane: spans, c_affine: streams}` | `Float32` (lamp: `Float16_b`) | compute | compute | pass 1 | not shared with `cb_x_pass1` (both fronted during `colsum_block`); disjoint lifetime with `cb_a_full`/`cb_b_full`/`cb_out` (pass 2) → **could alias with `cb_a_full + cb_b_full`** (`2·cols·F32 ≤ chunk·F32` when `chunk_rows ≥ 2`); not done in Phase 0 because the page counts differ (`chunk` vs `cols`) and the reuse saves `≤ 64 KiB` — recorded decision |
| `cb_scaler` | `1` | 1 | `{all axes: streams}` constant | `Float16_b` | reader | compute | kernel | constant; nothing to share |
| `cb_colsum` | `2 · cols` | `2 · cols` | `{n: streams, hw: streams (reduced), ct: spans → cols, g: streams, lane: spans, c_affine: streams}` | `Float32` | compute | compute | column group of pass 1 | capacity == live set. Doubles as the `Accumulate` accumulator (pop `cols` / push `cols` per statistic per chunk) — no separate accumulator CB. Disjoint with pass-2 CBs; not aliased (different page counts, ≤ 64 KiB) |
| `cb_membership` | `membership_depth · cols · Kg` | `cols · Kg` | `{n: streams, hw: streams, ct: spans → cols, g: spans → Kg tiles, lane: spans, c_affine: streams}` | `Float32` | reader | compute | per column group (both passes) | reused across passes (same CB carries `Eᵀ` then `E`); not shared with anything else (live in both passes) |
| `cb_agg_interm` | `2 · Kg` | `2 · Kg` | `{g: spans → Kg, others: streams}` | `Float32` | compute | compute | pass 1 (between K-blocks) | `matmul_block` requires its own region for spill/reload distinct from in0/in1/out (`matmul_block_helpers.hpp:241-252`); disjoint from pass-2 → aliasable with `cb_stats_T` (same `2` pages when `Kg = 1`); not done — 8 KiB |
| `cb_partial` | `2 · Kg` | `2 · Kg` | `{g: spans → Kg, others: streams}` | `Float32` | compute | writer | end of pass 1 → sent | cross-kernel handoff CB — must be distinct from any compute accumulator (`ttnn-cb-memory-fundamentals.md` → CB Ownership) |
| `cb_gather` | `2 · Kg · ceil(P_max/32)` | same (root only; allocated on every core because CB allocation is per core range and the landing address must be identical) | `{g: spans → Kg, others: streams; + cores: spans → ceil(P_max/32)}` | `Float32` | writer (root) | compute (root) | combine | not shared: written by remote cores at an address every core must reserve identically (`examples/tensix_all_reduce` README → "symmetric allocation is the addressing mechanism") |
| `cb_totals_src` | `2 · Kg` | `2 · Kg` | `{g: spans → Kg, others: streams}` | `Float32` | compute (root) | writer (root) | combine | handoff (multicast source) — distinct by the ownership rule |
| `cb_totals_recv` | `2 · Kg` | `2 · Kg` | `{g: spans → Kg, others: streams}` | `Float32` | writer | compute | combine → finalize | multicast landing; same address on every core; **not aliased with `cb_stats_row`** (implementer deviation): the finalize chain reads tile `k` of the totals and packs tile `k` of the row-form statistics in the same DEST window, so an alias would overwrite an operand the unpacker may not have consumed yet — `2·Kg·F32` (8 KiB at `Kg = 1`) buys the ordering |
| `cb_stats_row` | `2 · Kg` | `2 · Kg` | `{g: spans → Kg, others: streams}` | `Float32` | compute | compute | finalize | own region (see `cb_totals_recv`) |
| `cb_stats_g_full` | `2 · Kg` | `2 · Kg` | `{g: spans → Kg, lane: spans, others: streams}` | `Float32` | compute | compute | pass 2 | resident through pass 2 (retained in0 of every expansion) — cannot share with any pass-2 CB |
| `cb_gamma_row` | `has_gamma ? cols : 0` | `cols` | `{ct: spans → cols, c_affine: spans → cols·32, others: streams}` | affine dtype | reader | compute | column group of pass 2 | not shared: format differs from every `Float32` CB |
| `cb_beta_row` | `has_beta ? cols : 0` | `cols` | as gamma | affine dtype | reader | compute | column group of pass 2 | as gamma |
| `cb_stats_T` | `2` | `2` | `{ct: streams → 1 tile, lane: spans, others: streams}` | `Float32` | compute | compute | per T | transient; aliasable with `cb_agg_interm` (disjoint passes) — not done, 8 KiB |
| `cb_beta_full` | `has_beta ? 1 : 0` | 1 | `{lane: spans, others: streams}` | `Float32` | compute | compute | per T | transient 4 KiB; not shared |
| `cb_a_full` | `cols` | `cols` | `{ct: spans → cols, lane: spans, others: streams}` | `Float32` | compute | compute | column group of pass 2 | see `cb_xsq` (disjoint-lifetime alias candidate, not taken in Phase 0) |
| `cb_b_full` | `cols` | `cols` | as `cb_a_full` | `Float32` | compute | compute | column group of pass 2 | as `cb_a_full` |
| `cb_out` | `out_depth_factor · out_block` (= `2 · out_block`) | `out_block` | `{hw: streams → out_block/cols rows, ct: streams → min(out_block, cols) tiles, lane: spans, others: streams}` | output dtype | compute | writer | pass 2 | capacity > live set by one window = writer double buffering; format differs from the `Float32` CBs. **Verifier change**: the store block was `cols` tiles per barrier, which degenerates to ONE tile per barrier whenever the `ct` split leaves a core a single tile-column (every SD/SDXL shape at ≥ 80 cores); it is now `out_block = largest divisor of chunk ≤ 8`, a host knob (`OUT_BLOCK_TILES_TARGET`) |

## Symbol table

| Symbol | Bound | Predicate establishing it |
|---|---|---|
| `cols_per_group` | `≤ DEST_AUTO_LIMIT` (8) | mechanism cap (REDUCE_COL bulk chunk), host clamp |
| `chunk_rows` | `≤ max(1, chunk_tiles_target / cols_per_group)` and `≤ Ht_core` | host clamp; `chunk_tiles_target = 32` default ⇒ `chunk ≤ 32` tiles |
| `Kg` | `≤ ceil(C / 32)`; every INPUTS shape has `Kg = 1` | `G ≤ C`; `Kg ≤ DEST_AUTO_LIMIT` for a single output subblock, else `in1_num_subblocks` grows (host) |
| `P_max` | `≤ max_cores` (64 on an 8×8 WH grid, 110 on the 11×10 BH grid this was verified on, 130 on a full BH) | grid bound |
| `blk = Ht_core · Ct_core` | `≤ (l1_budget_bytes − fixed_bytes) / x_page` | the `resident` predicate — when false the ring is `x_depth · chunk` |
| `x_depth`, `x_rm_depth`, `membership_depth`, `out_depth_factor`, `out_block_tiles_target` | small integers (2, 2, 1, 2, 8) | host knobs; `out_block = largest divisor of chunk ≤ out_block_tiles_target` so it always divides the chunk |

## Total per-core footprint

```
fixed_bytes = chunk·F32                                   # cb_xsq            (∝ chunk_rows·cols)
            + 2048                                        # cb_scaler (TILE-padded rows/lanes are zero: no partial scaler needed for hw/c_non_aligned)
            + 2·cols·F32                                  # cb_colsum         (∝ cols)
            + membership_depth·cols·Kg·F32                # cb_membership     (∝ cols·Kg)
            + 2·Kg·F32 · (1 + 1 + ceil(P_max/32) + 1 + 1 + 1)   # agg_interm, partial, gather, totals_src, totals_recv(+stats_row alias), stats_g_full
            + has_gamma·cols·g_page + has_beta·cols·g_page       # gamma/beta rows
            + 2·F32 + has_beta·F32                        # stats_T, beta_full
            + 2·cols·F32                                  # a_full + b_full   (∝ cols)
            + out_depth_factor·out_block·y_page           # cb_out
            + is_rm·x_rm_depth·cols·x_page                # cb_x_rm
x_bytes     = resident ? blk·x_page : 2·x_depth·chunk·x_page   # streaming: cb_x_pass1 + separate cb_x_pass2 ring
            + 2·Kg·F32                                           # cb_stats_row (own region, not aliased)
total       = fixed_bytes + x_bytes
```

Implemented block extents: `cols = largest divisor of Ct_core ≤ DEST_AUTO_LIMIT`, `chunk_rows = largest divisor of Ht_core ≤ chunk_tiles_target / cols` (Pr | Ht, Pc | Ct, so no block is ragged and `blk = Ht_core · Ct_core` exactly). Worked default (bf16 in/out, `cols = 8`, `chunk_rows = 4` ⇒ `chunk = 32`, `Kg = 1`, `P_max = 64`, gamma+beta bf16, TILE input): `fixed ≈ 128 KiB (xsq) + 2 KiB + 64 KiB + 32 KiB + 6·8 KiB… (48 KiB + 8 KiB gather extra) + 32 KiB + 12 KiB + 64 KiB + 32 KiB ≈ 422 KiB`; streaming `x` ring `128 KiB` → `≈ 550 KiB`; resident regime accepts `blk ≤ (1 000 000 − 432 000) / 2048 ≈ 277` tiles — every `feature_spec.INPUTS` shape is resident at ≥ 56 cores (largest: `(1,1,16384,320)` → 80 tiles/core at 64 cores). fp32 input: `x` ring `256 KiB`, resident `blk ≤ 138` tiles (`(1,1,16384,320)` fp32 → 80 tiles ✓).

Terms scaling with knobs: `chunk_rows·cols` → `cb_xsq`, streaming `x` ring; `cols` → `cb_colsum`, `cb_membership`, `cb_gamma/beta_row`, `cb_a/b_full`; `out_block` → `cb_out`; `Kg` → all statistic CBs; `P_max` → `cb_gather` only; `blk` → resident `x` only (predicate-guarded).

## Data-movement budget (chosen split: 2-D `hw × ct` per image rectangle, lane-form root combine)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `x` (input) | **1** in `resident_2d`; **2** in `streaming_2d` | resident: the aliased ring keeps the whole per-core block through both passes; streaming: the block exceeds `l1_budget_bytes`, pass 2 re-reads it | none (`x` varies along both cut axes) |
| `y` (output) | 1 | each core writes only its own block's tiles | none |
| `gamma`, `beta` | `Pr` reads of each `Ct_core` slice per image (`≤ 128 B` per tile for bf16/fp32; 1088 B for bf8b tiles) | reuse-shared by construction of the `hw` cut; the broadcast regime is rejected on payload size | none |
| statistics | 0 | never touch DRAM | per image: `P_used` unicast records of `2·Kg·128 B` (+ `P_used` semaphore increments), one gather-ready signal multicast, one totals multicast of `2·Kg·4096 B` to `P_n` cores |

Totals per tier (per image, bf16, `Kg = 1`, 64 cores, `(1,1,16384,320)`): DRAM `10 MiB` read + `10 MiB` written (resident) vs `20 MiB + 10 MiB` (streaming); gamma/beta `64 × 10 × 128 B ≈ 80 KiB`; cross-core `64 × 256 B + 8 KiB ≈ 24 KiB` plus two multicast flags and 64 semaphore increments.

> Cheapest-traffic split considered: 2-D `hw × ct` with residency — `x` once, `y` once, `+≈24 KiB` cross-core. Implemented: the same split (`resident_2d` when the predicate holds; `streaming_2d` adds one input read when it does not). The implemented split is also the cheapest; the only deferred alternative on the traffic axis (`shifted_two_pass_variance`) would *add* a read.
