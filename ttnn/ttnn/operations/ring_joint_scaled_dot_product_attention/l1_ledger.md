# L1 Ledger: ring_joint_scaled_dot_product_attention

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes (from `op_design.md` Blocking Model):
`batch` (extent 1), `q_head` (extent 1), `q_rows` (Sq = `q_block_tiles`), `keys` (Sk = `k_block_tiles`, runtime
`Sk_valid ≤ Sk`), `head_dim` (DHt, whole), `v_head_dim` (DVt, whole), `transport_entry` (1 entry),
`transport_chunk` (`run_pages`). `T_acc` = Float32 page (4096 B) when `fp32_dest_acc_en`, Float16_b (2048 B) when off.

## Compute cores

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_q_block` | `q_block_depth·Sq·DHt` | `Sq·DHt` (Q retained by `score_block` for the whole key walk) | {batch: streams → 1, q_head: streams → 1, q_rows: spans → Sq, keys: streams → (not spanned), head_dim: spans → DHt, v_head_dim: streams → —, transport_entry: streams → —, transport_chunk: streams → —} | Q dtype (Float16_b; Bfp8_b later) | reader | compute | per unit | none: live for the entire unit, concurrent with every other CB; depth 1 = live set |
| `cb_k_block` | `kv_block_depth·Sk·DHt` | `Sk_valid·DHt` per block; 2 blocks resident for prefetch | {batch: 1, q_head: 1, q_rows: streams → —, keys: spans → Sk, head_dim: spans → DHt, v_head_dim: streams → —, transport_entry: streams, transport_chunk: streams} | K dtype | reader | compute | per key block | none: concurrent with `cb_v_block` (same block), different dtype possible; capacity = 2× live set for double buffering (reader one block ahead) |
| `cb_v_block` | `kv_block_depth·Sk·DVt` | `Sk_valid·DVt`; 2 blocks for prefetch | {batch: 1, q_head: 1, q_rows: streams, keys: spans → Sk, head_dim: streams, v_head_dim: spans → DVt, transport_entry: streams, transport_chunk: streams} | V dtype | reader | compute | per key block | none: V of block i is live from load to `pv_block` while K of block i+1 loads; capacity 2× for double buffering |
| `cb_key_mask` | `mask_tiles` (= 1 Phase 0; 2 once a non-aligned joint tail exists) | `mask_tiles` | {all axes: streams — constant tile(s)} | Float16_b (−inf / 0 only: 16-bit is exact) | writer | compute | whole program | none: never popped, read by every masked block; a constant cannot alias a streaming CB |
| `cb_max_scaler` | 1 | 1 (constant) | {all axes: streams — constant} | Float16_b (pool-type-aware bf16 scaler) | writer | compute | whole program | not shared with `cb_sum_scaler`: produced by different pool-type-aware overloads (MAX vs SUM, `reduce_helpers_dataflow.hpp:97-99`) whose fill patterns may differ; 1 tile each |
| `cb_sum_scaler` | 1 | 1 (constant) | {all axes: streams — constant} | Float16_b | writer | compute | whole program | see `cb_max_scaler` |
| `cb_scores` | `Sq·Sk` | `Sq·Sk_valid` (S, then P in place) | {batch: 1, q_head: 1, q_rows: spans → Sq, keys: spans → Sk, head_dim: streams (contracted), v_head_dim: streams, transport_entry: streams, transport_chunk: streams} | T_acc | compute | compute | per key block (score → mask → max → exp in place → consumed by pv) | none: S→P is already in place (Rule 3.2); live concurrently with stats and both O buffers; cannot alias `cb_out_tiles` because the writer drains the previous unit's output while this unit's scores are live |
| `cb_max_a` | `Sq` | `Sq` | {q_rows: spans → Sq; batch, q_head: 1; keys, head_dim, v_head_dim, transport_*: streams} | T_acc | compute | compute | ping-pong across key blocks | pair with `cb_max_b`: α = exp(s(m_prev − m_cur)) reads both at once (concurrent lifetime) |
| `cb_max_b` | `Sq` | `Sq` | same as `cb_max_a` | T_acc | compute | compute | ping-pong | see `cb_max_a` |
| `cb_sum_a` | `Sq` | `Sq` (32-wide row partials until finalize) | {q_rows: spans → Sq; batch, q_head: 1; others: streams} | T_acc | compute | compute | ping-pong | pair with `cb_sum_b`: `l_cur += α·l_prev` reads both (concurrent); a single in-place sum would need L1-accumulating packs onto a fronted CB — lamp "single accumulator" |
| `cb_sum_b` | `Sq` | `Sq` | same | T_acc | compute | compute | ping-pong | see `cb_sum_a` |
| `cb_row_factor` | `Sq` | `Sq` | {q_rows: spans → Sq; others: streams/1} | T_acc | compute | compute | α during each key block; 1/l at finalize | **shared by role**: α (key walk) and 1/l (finalize) have disjoint lifetimes, so one CB serves both — Rule 3.3 applied |
| `cb_out_acc_a` | `Sq·DVt` | `Sq·DVt` | {q_rows: spans → Sq, v_head_dim: spans → DVt; batch, q_head: 1; keys, head_dim, transport_*: streams} | T_acc | compute | compute | ping-pong across key blocks | pair with `cb_out_acc_b`: `O_cur = P·V` is packed while `O_prev` is still needed for `O_cur += α·O_prev` (concurrent); folding into one buffer needs PV to L1-accumulate onto a rescaled O — lamp "single O accumulator" |
| `cb_out_acc_b` | `Sq·DVt` | `Sq·DVt` | same | T_acc | compute | compute | ping-pong | see `cb_out_acc_a` |
| `cb_out_tiles` | `out_block_depth·Sq·DVt` | `Sq·DVt` | {q_rows: spans → Sq, v_head_dim: spans → DVt; others: streams/1} | Float16_b (the output dtype; the final rounding of an fp32 DEST result — not an intermediate) | compute | writer | per unit | none: the writer drains it during the next unit's key walk, concurrent with every compute CB; cannot be `cb_out_acc_*` (different format, and the writer would become a second consumer of a compute-owned CB) |

### Port cores

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_port_chunks` | `2·port_batch_chunks·run_pages` (= 2·4·7 = 56 bf16 tiles, 112 KiB) | `port_batch_chunks·run_pages` in flight + the next batch being read | {transport_chunk: spans → run_pages, transport_entry: streams, batch, q_head, q_rows, keys, head_dim, v_head_dim: streams} | K/V dtype (pages are forwarded byte-for-byte) | port reader | port sender | whole call | only CB on the port core; 2 batches so a reservation never wraps (example's reader rule) |

## Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `Sq` (`q_block_tiles`) | power of two, `1 ≤ Sq ≤ pow2_ceil(tensor_q_tiles)`, `≤ Sq_occ` | `choose_block_extents()`; `footprint ≤ L1_CB_BUDGET` |
| `Sk` (`k_block_tiles`) | power of two, `1 ≤ Sk ≤ pow2_ceil(tensor_kv_tiles)` | same |
| `Sk_valid` | `1 ≤ Sk_valid ≤ Sk` | `key_block_plan` (ragged last block of a segment) |
| `DHt` | `dh/32`; TARGET dh ≤ 576 → ≤ 18 | validate(): `dh % 32 == 0`; footprint predicate rejects what does not fit (ValueError) |
| `DVt` | `dv/32`; TARGET ≤ 8 | same |
| `q_block_depth`, `out_block_depth` | 1 | host constants |
| `kv_block_depth` | 2 | host constant |
| `mask_tiles` | ≤ 2 | number of key segments with a partial last tile (main tail; joint tail) |
| `run_pages` | `max(1, payload // page_bytes)`; 7 bf16 / 13 bfp8 at 14,336 B | fabric config, read once |
| `port_batch_chunks` | `max(1, min(8, 112 KiB // (2·chunk_bytes)))` = 4 | example's `group` rule |
| `T_q, T_kv` | 2048 B (bf16), 1088 B (bfp8) | dtype |
| `T_acc` | 4096 / 2048 B | `fp32_dest_acc_en` |
| `L1_CB_BUDGET` | per-core CB region of the device (≈ 1.3 MB on Blackhole) | queried from the device once per process |

## Total per-core footprint

Compute core:

```
F(Sq, Sk) = q_block_depth·Sq·DHt·T_q                       # scales with Sq, DHt
          + kv_block_depth·Sk·(DHt + DVt)·T_kv             # scales with Sk, DHt, DVt
          + Sq·Sk·T_acc                                    # scales with Sq·Sk (scores)
          + 5·Sq·T_acc                                     # stats: max a/b, sum a/b, row_factor — Sq
          + 2·Sq·DVt·T_acc                                 # O ping-pong — Sq, DVt
          + out_block_depth·Sq·DVt·2048                    # output — Sq, DVt
          + (2 + mask_tiles)·2048                          # constants
```

dh = dv = 128: fp32 DEST, (Sq, Sk) = (8, 8) → 1,062 KB; bf16 DEST (FOCUS), (16, 8) → 1,184 KB.
Port core: `2·port_batch_chunks·run_pages·page_bytes` = 112 KiB (+ packet-header pool).

## Data-movement budget (chosen split R1, per device per call)

FOCUS (Wan2.2 480p: n = 8192, H = 40, dh = dv = 128, bf16, G = 4 ring balanced, 2 links, Sq = 16, `Uq = 16` Q blocks
per head). B = local K+V block = n·H·(dh+dv)·2 = 167.8 MB; Q = out = 83.9 MB.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| Q (main, joint) | 1 read | resident for its unit's whole key walk | — |
| local K/V | `Uq` compute reads + 2 transport reads (sent in both ring directions) = 18 × 167.8 MB | each Q block of a head re-streams the head's K/V (R1 is Q-stationary; R2 removes it) | — |
| remote K/V (G−1 blocks) | 1 landing write + `Uq` compute reads each; relayed share (1 block-equivalent on a balanced 4-ring) +1 read | landing in DRAM is required (R10 rejected); re-read per Q block as above | — |
| joint K/V (replicated) | `Uq` reads (Phase 0 cases) | local source, streamed per unit like local K/V | — |
| out / joint_out | 1 write | normalized once per unit | — |
| landing counters | — | — | ≤ 2·num_links 4-byte polls per (unit, remote source) when not yet landed; `C` 4-byte increments per port per call |

Totals (FOCUS): DRAM reads ≈ 83.9 MB (Q) + 64 × 167.8 MB (compute K/V) + 3 × 167.8 MB (transport) ≈ 11.33 GB;
DRAM writes ≈ 503 MB (landing) + 83.9 MB (out) ≈ 0.59 GB; links: busiest direction (G²/8)·B = 335.6 MB over
2 links; NoC cross-core: negligible.

> Cheapest-traffic split considered: R2 `qstationary_kv_row_mcast` (row group ≈ 10 compute cores per head per wave) —
> DRAM K/V reads ≈ 10.7 GB → ≈ 1.1 GB per call (−9.7 GB DRAM); adds ≈ 1.1 GB of NoC multicast payload (each K/V
> byte once per row) + one handshake per key block per row. Implemented: R1 `qstationary_dram_kv`. Deferred
> because R2 layers onto R1 unchanged (stepping stone, `blocking-model.md` §5): R1's units are already head-major
> round-robin, so a grid row holds consecutive Q blocks of one head and the K/V reader becomes a ReceiverPipe
> without touching compute, the CB inventory or the transport. It is refinement #3 in `op_design.md`'s queue,
> after the fp32-off enabler and the OVERLAP measurements, and moves ahead of them if the FOCUS breakdown shows
> DRAM bound (11.9 GB at the measured 447 GB/s ≈ 26.6 ms vs the 24.1 ms target).
