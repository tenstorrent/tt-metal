# L1 ledger: all_gather_rms_norm

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking axes (from `op_design.md`):
**row** (tile-rows of the local slice), **W** (W_local tile-columns), **chip** (group position), **part**
(stats partition j = q·k + m), **pkt** (bulk packet), **batch** (port batch). "streams → 0" = the buffer does not
touch that axis.

Every CB is declared on the **full program core range** (workers ∪ ports) so that addresses are identical on
every core of a chip. Worker-only CBs therefore also occupy port cores and vice versa (counted in the totals).

## CB table

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_x` | `crt · cwt` (sharded: the shard, zero-copy, 0 extra bytes) | `crt · cwt`: the whole block is resident from pass 1 until the end of normalize | row: spans → crt; W: spans → cwt; chip: streams → 0; part: streams → 0; pkt: streams → 0; batch: streams → 0 | input dtype (T_in) | reader | compute | program | none: live for the whole call (it must outlive the stats round, which is the reason the block is the whole assignment) |
| `cb_gamma` | `cwt` (0 if no gamma) | `cwt` compact pages | row: streams → 0 (row-invariant); W: spans → cwt; others: streams → 0 | gamma dtype, compact row-0 faces 0/1 page (T_g) | reader | compute | program | none: read by every normalize tile; concurrent with `cb_x` |
| `cb_beta` | `cwt` (LN with beta only) | `cwt` | as `cb_gamma` | gamma dtype (T_g) | reader | compute | program | none: concurrent with `cb_gamma` in the same normalize chain |
| `cb_reduce_scaler` | 2 (constant: scaler 1.0, W-tail 0/1 mask) | 2 | all axes: streams → 0 (constant) | Float16_b (pool-type-aware scaler, packed bf16) | reader | compute | program | none: both pages read by reduces throughout |
| `cb_sumsq_tiles` (RMS) | `crt` | `crt` (one DEST-accumulated tile per tile-row, packed by the chain, then reduced as a block) | row: spans → crt; W: streams → 0 (folded in DEST); others: streams → 0 | T_acc (Float32 if fp32 DEST, else Float16_b) | compute | compute | pass 1 → collapse | could alias `cb_ln_sq`, but the two never coexist in one compiled kernel (CT `is_layer_norm`), so it is allocated only for RMS |
| `cb_ln_mean_local` (LN) | `crt` | `crt` (m_i col 0, kept until K is built and the records are copied) | row: spans → crt; others: streams → 0 | Float32: the wire contract and the exactness of K_i = tf32(m_i) | compute | compute | stats pass | aliasing with `cb_inv` (disjoint lifetimes: stats pass vs combine) is possible; not taken in Phase 0 because both are `crt` pages of 4 KiB, ≤ 1 % of L1 at the largest golden `crt` (64) on a worker |
| `cb_ln_shift` (LN) | `crt` | `crt` (K_i broadcast tile per tile-row, used by T's chain and copied into the records) | row: spans → crt; others: streams → 0 | Float32 (must hold tf32-exact values without re-rounding) | compute | compute | stats pass | could alias `cb_mean_bcast` (disjoint: stats pass vs combine); not taken, same reason as above |
| `cb_ln_sq` (LN) | `ln_sq_depth` = 2 | 1 (one centered-square tile in flight) | row: streams → 2; W: streams → 2; others: streams → 0 | T_acc | compute | compute | per tile | capacity 2 > live set 1: double buffer between the chain's pack and the reduce's unpack (mechanism) |
| `cb_partial` | `crt · nstats` | `crt · nstats` (all records of the block are handed to the writer at once) | row: spans → crt; part: streams → 0 (this member's partition only); others: streams → 0 | **Float32** with 16-bit DEST too: mechanism reason = the op's precision contract (partials fp32 on the wire) | compute | writer | stats pass → publish | none: concurrent with `cb_x`; it is the only fp32 record source |
| `cb_record_staging` | `crt · nstats` × 128 B | same | row: spans → crt; others: streams → 0 | raw fp32 words (dataflow-only, never through DEST) | writer | writer (self scratch) | publish | could reuse `cb_partial`'s space after the pop, but it must stay valid until the NoC write barrier while `cb_partial` is already popped. Kept separate: ≤ 24 KiB at crt = 64 × 3 stats. |
| `cb_record_landing` | `G · k · nstats` × 128 B | one tile-row's records | row: streams → 1; chip: spans → G; part: spans → k; others: streams → 0 | raw fp32 (dataflow-only) | reader | reader (self scratch) | gather, per tile-row | none: reused per tile-row (ring of 1) |
| `cb_gathered` | `crt · nstats · gather_tiles` | same (the combine reduces the whole block per stat) | row: spans → crt; part: spans → gather_tiles (= ceil(G·k/32)); chip: spans (inside part); others: streams → 0 | Float32 (records are fp32; Accurate combine) | reader | compute | gather → combine | none: concurrent with `cb_x` |
| `cb_stat_consts` (LN) | 1 | 1 (col j = n_j) | part: spans → 1 tile (P ≤ 32 per tile, P_CAP ≤ 4 tiles → capacity `gather_tiles` when P > 32); others: streams → 0 | Float32 | reader | compute | program | none: constant |
| `cb_inv` | `crt` | `crt` | row: spans → crt; others: streams → 0 | T_acc | compute | compute | combine → end of normalize | none: live across normalize |
| `cb_mean_bcast` (LN) | `crt` | `crt` | row: spans → crt; others: streams → 0 | Float32 (exact mean; unpacked to DEST fp32) | compute | compute | combine → end of normalize | none: concurrent with `cb_inv`, `cb_rstd_bcast` |
| `cb_rstd_bcast` (LN) | `crt` | `crt` | row: spans → crt; others: streams → 0 | T_acc | compute | compute | combine → end of normalize | none: concurrent with `cb_mean_bcast` |
| `cb_ln_scratch` (LN) | 2 | 2 (m_hi, m_lo of one tile-row) | row: streams → 1; others: streams → 0 | Float32 | compute | compute | per tile-row in the combine | none: both are operands of one SFPU add |
| `cb_out_block` | `crt · cwt` | `crt · cwt`: the whole block must stay intact until every port has pulled it (end of call) | row: spans → crt; W: spans → cwt; chip: streams → 0 (own slice only); pkt: streams → 0; batch: streams → 0 | output dtype (T_out) | compute | writer (port cores read the region by NoC; they are not CB consumers) | normalize → program end | none: concurrent with `cb_x` during normalize. Packing into `cb_x` in place is rejected: the input dtype page ≠ the output page in general (bf8b out of fp32 DEST), and sharded `cb_x` is the caller's shard, which must not be mutated. |
| `cb_assemble_scratch` (R_U) | 3 | 3 (two source tiles, one output tile) | row: streams → 1; W: streams → 1 output tile; chip: streams → ≤ 2 sources; others: streams → 0 | output dtype (dataflow-only) | reader | writer | per output tile | could alias `cb_port_chunks` on workers (ports never assemble), but CB addresses must be identical on every core and a per-core alias is not expressible with one CB list. 3 tiles. |
| `cb_port_chunks` (ports) | `2 · batch_chunks` × `payload_pages · T_out` | `2 · batch_chunks` packets (reader fills one batch while the sender drains the other) | pkt: spans → 2·batch_chunks; batch: spans → 2; row / W / chip / part: streams → 0 | output dtype pages (dataflow-only) | port reader | port sender | per batch | none on ports. On workers it is unused but allocated (identical-address rule): ≈ 112 KiB per worker at 14 KiB payload, an accepted cost (see total). |

Non-CB L1 (mesh-allocated, same address on every chip):

| Buffer | Bytes | Where | Bound |
|---|---|---|---|
| ~~stats table~~ | — | **now `cb_stats_table` (CB 25), see the As-built table** | — |
| global semaphores | 3 kinds × 16 B | port cores (union) | constant |
| R_U staging | `G · R · Wlt · T_out` | **DRAM** interleaved (not L1) | — |

## Symbol table

| Symbol | Meaning | Bound | Predicate establishing it |
|---|---|---|---|
| `crt` | `core_row_tiles_max` | ≤ R; sized so the footprint ≤ budget | `select_split` + footprint check (RuntimeError otherwise) |
| `cwt` | `core_w_tiles_max` = ceil(Wlt / k) | ≤ Wlt | `select_split` |
| `k` | W members per row-group | ≤ min(Wlt, P_CAP / G) | `select_split`; WIDTH_SHARDED: #shards ≤ P_CAP / G (else RuntimeError) |
| `G` | group size | 2 … 8 (mesh axis length) | mesh shape |
| `P`, `gather_tiles` | G·k, ceil(P/32) | P ≤ 128 → ≤ 4 | `P_CAP` |
| `nstats` | 1 (RMS) / 3 (LN: m, T, K) | ≤ 3 | norm_type |
| `R` | tensor_row_tiles | stats table ≤ 256 KiB → `R ≤ 256 KiB / (G·k·nstats·128)` | capacity check |
| `payload_pages` | floor(max_payload / T_out) | ≤ 14336 / 1088 = 13 (bf8b) | fabric config |
| `batch_chunks` | 4 | constant knob | CT |
| `T_in`, `T_out` | 4096 (fp32) / 2048 (bf16) / 1088 (bf8b) | — | dtype |
| `T_g` | compact gamma page: row 0 of faces 0/1 (base op) | ≤ 192 B stride | gamma dtype |
| `T_acc` | 4096 if fp32 DEST else 2048 | — | `fp32_dest_acc_en` |

## Total per-core footprint

```
worker/port (one CB list) =
    crt·cwt·(T_in + T_out)                               # cb_x + cb_out_block      — scales with block (row × W)
  + (has_gamma + has_beta)·cwt·T_g                        # gamma / beta             — scales with W
  + 2·2048                                                # scaler + mask            — constant
  + crt·( T_acc·[RMS] + 4096·nstats + 128·nstats          # sumsq / partial / record staging
        + 4096·nstats·gather_tiles + T_acc                # gathered / inv
        + [LN]·(4096·3 + T_acc) )                         # LN: mean_local, shift, mean_bcast, rstd_bcast
  + G·k·nstats·128                                        # record landing           — scales with part
  + [LN]·(2·T_acc + 2·4096 + 4096·gather_tiles)           # ln_sq, ln_scratch, stat_consts
  + [R_U]·3·T_out                                         # assemble scratch
  + 2·batch_chunks·payload_pages·T_out                    # port chunks              — constant (≈ 112 KiB)
  + G·R·k·nstats·128                                      # cb_stats_table (CB 25, every core; read/written on the hubs only)
```

Terms scaling with the block knobs: `crt·cwt` (x, out). With `crt` alone: the per-row stat CBs. With `k` and `G`:
the gathered partition tiles and the record landing. Focus case (640×384 bf16, RMS, k = 4: crt = 1, cwt = 3):
≈ 12 KiB block + ~25 KiB stats + 112 KiB port chunks ≈ 150 KiB per core. Largest golden block (2048×1280 bf16,
k = 1, crt = 1, cwt = 40, LN fp32 DEST): 80 KiB + 80 KiB + ~70 KiB + 112 KiB ≈ 0.35 MiB, under the ~1.3 MiB budget.

## Data-movement budget (focus case: G = 4, balanced ring, interleaved, S = 480 KiB slice)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| x[p] | 1 read | resident in `cb_x` from pass 1 to normalize (one stats round per call) | — |
| gamma (+ beta) | 1 read of the W_local columns per row-group (20 groups × 768 B) | reuse-shared over rows, member slice only | — |
| own normalized slice | 1 write (own output) | written by the workers | ports pull it from worker L1: 2·S over the NoC (fwd + bwd ports), no DRAM re-read |
| remote slices (3 per chip) | 3 writes (router writes at final pages) | land at final addresses | fabric: busiest link direction 1.5 S / L links |
| relayed slices | 1·S read per chip (½ slot fwd + ½ slot bwd) | relay forwards from the local output (store-and-forward) | — |
| stats records | 0 | L1 tables on the hubs | per chip: 80 workers × 2 hubs × 128 B (records); per worker G reads of k·128 B; fabric: 3 slots × 2.5·k KiB |

Total per chip: DRAM ≈ 6S (x 1, own write 1, landed 3, relay read 1) against the roofline minimum of 5S. Links:
1.5 S on the busiest direction (minimum for a balanced ring). NoC: 2S pulled by the ports.

> Cheapest-traffic split considered: `SF-GN with fabric chip-multicast` (removes the relay read-back): 5S DRAM
> instead of 6S per chip, links equal. Implemented: `SF-GN with one-hop relay (R_A)`. Deferred because the
> multicast route is unmeasured on the ring's wrap hop (`blackhole-ccl-kernels.md` §4: "no wrap handling in the
> route setter"), while the relay is the measured transport (`fabric_all_gather`). The `fabric_mcast` regime row
> keeps it reachable: an entry's pages and source are unchanged, only its route header differs.

## As built (implementer, Phase 0) — departures from the inventory above

| CB | Index | As built | Why it differs / shares with |
|----|-------|----------|------------------------------|
| `cb_mask` | 4 | **Float32**, 1 tile (LN tail member only) | The reduce helper's AccumulateViaAdd masked fold (`llk_unpack_AB` with the mask as srcB) does not reconfigure srcB to the mask CB's format: a bf16 mask was read as fp32 garbage (LN `w_non_aligned` off by 5 % rel RMS). Same format as `cb_ln_sq` fixes it. |
| `cb_ln_sq` | 8 | `crt_max · cw_max` Float32 | The masked AccumulateViaAdd reduce is BulkWaitBulkPop over a whole tile-row (no streaming with a partial mask); one chain + one reduce per block. Shares: none (concurrent with `cb_x`/`cb_ln_shift`). |
| `cb_ln_m` | 6 | `crt_max` Float32 | Holds m (shift source), then — disjoint lifetime — `d = mean − K` (alias, pattern 3: m is consumed by the shift broadcast before d is produced). |
| `cb_ln_shift` | 7 | `crt_max` Float32 | Kept until the end of normalize: LN centers as `(x − K) − d` (no exact fp32 mean broadcast). |
| `cb_mean_col` | 15 | `crt_max` Float32, UnpackToDestFp32 | exact mean (combine reduce output), read once by the `d = mean − K` chain |
| `cb_mean_b` | 16 | `crt_max` Float32 | **d = mean − K** broadcast to every column (FPU UnaryBcast; d is small, its tf32 truncation is ~1e-3 of d). The reader's word-by-word fp32 mean broadcast (~4 µs) is gone. |
| `cb_ln_scratch` | — | not built | `cb_ln_tmp` (18, `crt_max` Float32, UnpackToDestFp32) carries both combine intermediates (n·m, then M2 terms) in sequence. |
| `cb_x32` | 22 | `crt_max · cw_max` Float32, UnpackToDestFp32 — **LN with non-Float32 x only** | the exact SFPU LN paths (Accurate reduce, centering) need a Float32 source CB; an exact copy of the bf16 / bfp8 block. Shares: none (concurrent with the whole LN block). For Float32 x, `cb_x` itself is tagged and no copy exists. |
| `cb_gamma_rows` | 23 | `(cw_max + 2) · 128 + 64` B | reader scratch for the gamma / beta source rows (64 B-aligned), copied element-wise into the compact pages (handles R_U column offsets). |
| `cb_gather_rows` | 24 | `crt_max · nstats · gather_tiles` Float32, UnpackToDestFp32 | **new**: records land as ROWS (row j = partition j) via local NoC copies from `cb_record_landing`; compute transposes them exactly in DEST (`copy_tile` + `transpose_dest`) into `cb_gathered`. Replaces the RISC-V column scatter (3.6 µs on the focus case). Shares: none (concurrent with `cb_gathered` during the transpose). |
| `cb_record_landing` | 11 | `G · k · nstats · 128` B | one contiguous hub read per slot per tile-row (the table stores each (slot, row, stat) block half-split), then local copies into `cb_gather_rows`. |
| `cb_gathered` | 12 | `crt_max · nstats · gather_tiles` Float32 (UnpackToDestFp32 for LN only) | compute → compute (transpose output), as designed otherwise. |
| `cb_assemble` | 20 | `src_max + 1` pages (R_U) | `src_max` = max distinct staging tiles per output tile (host-computed; 3 for W_local = 50). |
| `cb_port_chunks` | 21 | `2 · batch_chunks · run_pages` pages | as designed; `run_pages = floor(payload / page)` (7 for bf16 at 14 KiB). |
| `cb_stats_table` | 25 | `G · R · k · nstats` × 128 B pages, full core range | **Verifier fix** (was a persistent mesh L1 tensor cached per table size, which leaked one L1 buffer per distinct size and collided with the CB region; the golden conftest's frame-tensor deallocation also freed it under the cache). A program-scoped CB has the same address on every core of every chip (identical CB lists) and needs no persistent L1. Live set = capacity on the 2 hubs; on every other core it is unused but allocated (identical-address rule, as `cb_port_chunks`): ≤ `STATS_TABLE_CAP` = 256 KiB, focus case 4·20·4·1·128 = 40 KiB. Bounded by `stats_k_cap` in `select_split`. Shares: none (live the whole call on the hubs). |

Data-movement budget: unchanged from the table above (x 1, own write 1, landed 3, relay read 1 per chip; the ports
pull the own slice from worker L1). Measured probe: re-reading the own slice from the own DRAM output instead is only
0.7 µs faster on the focus case (34.85 vs 35.54 µs), i.e. the pull is not the transport bottleneck.

## Refinement 1 (sharded input, 16-bit DEST) — ledger deltas

| CB | Change | Shares with / why not |
|----|--------|-----------------------|
| `cb_x` | sharded input: `cb_descriptor_from_sharded_tensor` on the input (zero-copy, 0 CB bytes), declared on the full range (the shard address is valid on every core; nothing else lives there) | is the caller's shard: never written (W tail masked in compute) |
| `cb_out_block` | HEIGHT in + HEIGHT out, one tile-row per shard on the same cores: backed on the own slice of the output shard (`address_offset = p·Wlt·T_out`, `Wlt` pages, 0 CB bytes) | is the final output: the own write disappears, the ports pull from it as before |
| `cb_mask` | sharded W tail: `cwt` row-0 masks (the mask block), T_acc; else 1 page (LN T fold) | none: read by every masked chain of the block |
| `cb_ln_sq` | also allocated for RMS on a sharded W tail (masked x² block); bf16 for sharded LN on a non-fp32 x; T_acc otherwise | reused by S1 (twice) and T in sequence on the masked path — each chain's output is consumed (BulkWaitBulkPop) before the next fills it |
| `cb_x32` | not allocated on a sharded input or under 16-bit DEST (LN reads `cb_x`) | — |
| `cb_inv`, `cb_rstd_b` | T_acc (was Float32 under 16-bit DEST too) | — |
| `cb_stat_consts`, `cb_ln_tmp` | `gather_tiles` / `crt · gather_tiles` pages (LN with P > 32, WIDTH only) | — |

Data-movement budget, sharded input: x never crosses DRAM (consumed in place); a sharded output is written once in
L1 (own slice in place for HEIGHT one-row shards, local / NoC writes otherwise) and the remote slices land at their
final shard addresses over the fabric. Interleaved budget unchanged.
