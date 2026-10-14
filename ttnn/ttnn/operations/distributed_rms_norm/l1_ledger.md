# L1 Ledger: distributed_rms_norm (Phase 0, regime R1)

Schema and audits are defined in `.claude/references/l1-footprint-discipline.md`. Block axes come from `op_design.md`.

Block axes:
- `rows` (extent `block_rows`)
- `w_local` (extent `core_w_tiles`)
- `sources` (extent `n_src_tiles`, stage data)

`group` (G) is a device axis, so no worker CB spans it. Its only footprint is through `n_src_tiles` and the port scratch.

Phase 0 runs with `fp32_dest_acc_en = True`, so every DEST-produced intermediate is Float32.

## Worker cores

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| cb_x_resident | `block_rows · core_w_tiles` | `block_rows · core_w_tiles`. The whole block is resident from load until it is popped by prescale (gamma) or normalize (no gamma). | `{rows: spans → block_rows, w_local: spans → core_w_tiles, sources: streams → 0}` | x dtype (Float32 / Float16_b) | reader | compute | load → prescale / normalize | **No share.** Its residency across the exchange is the whole reason x crosses DRAM once. It cannot alias cb_xg: prescale reads x tile i while packing xg tile i, both live, and an in-place transform would make compute a second producer of a reader-produced CB. |
| cb_gamma_tiles | `core_w_tiles` (allocated only when `HAS_GAMMA`) | `core_w_tiles`, held for the whole prescale | `{rows: streams → reused for every row, w_local: spans → core_w_tiles, sources: streams → 0}` | gamma dtype (Float32 / Float16_b). This is input data, not DEST output, so it keeps its own format. | reader | compute | whole call | **No share.** It is reuse-shared across rows (read once per call). Every lifetime it could share overlaps it. |
| cb_scaler | 1 (constant) | 1 | `{rows: streams, w_local: streams, sources: streams}` (constant) | Float16_b (scaler convention; 1.0 is exact) | reader | compute | whole call | **No share.** It is read by two reduces (sumsq and combine), so it must persist across both. |
| cb_sq_acc | `block_rows` | `block_rows`. One DEST-accumulated Σ_w x² tile per tile row. | `{rows: spans → block_rows, w_local: streams → collapsed in DEST, sources: streams → 0}` | Float32 | compute | compute | sumsq | **Disjoint lifetime with cb_rstd** (same size and format), but kept separate on purpose. Deferred regime R2 (multi-round) runs round b+1's sumsq while round b's rstd is live, and aliasing would foreclose that. It totals ≤ `block_rows · 4 KiB`. Folding into DEST is already done: W is collapsed in DEST, and only one tile per row is packed. |
| cb_partial | `block_rows` | `block_rows` | `{rows: spans → block_rows, w_local: streams → 0, sources: streams → 0}` | Float32 (stats stay fp32 on the wire) | compute | writer | sumsq → publish | **No share with cb_sq_acc:** the reduce reads cb_sq_acc while it packs cb_partial (concurrent), and the consumers differ (compute vs writer). **No share with cb_rstd:** different consumer kernel, and R2 overlap. |
| cb_xg | `block_rows · core_w_tiles` (allocated only when `HAS_GAMMA`) | `block_rows · core_w_tiles` from prescale until normalize | `{rows: spans → block_rows, w_local: spans → core_w_tiles, sources: streams → 0}` | Float32 (DEST width; the precision gate for the stall-shadow reorder) | compute | compute | prescale → normalize | It exists only because of the **explicit stall-shadow decision**: x·gamma is computed during the exchange wait. Its lifetime overlaps cb_x_resident (prescale) and cb_output_tiles (normalize). The perf lamp "gamma prescale" measures the version without it. |
| cb_stats_combine | `n_src_tiles · block_rows` | `n_src_tiles · block_rows`. The reduce waits for the whole block (BulkWait). | `{rows: spans → block_rows, w_local: streams → 0, sources: spans → n_src_tiles}` | Float32 | reader | compute | gather → finalize | **No share.** It could alias cb_sq_acc only when `n_src_tiles = 1`, and that also forecloses R2. It totals ≤ 4 KiB · `n_src_tiles · block_rows`. |
| cb_rstd | `block_rows` | `block_rows`. The Col-mapped Upfront window is held across normalize. | `{rows: spans → block_rows, w_local: streams → broadcast along it, sources: streams → 0}` | Float32 | compute | compute | finalize → normalize | See cb_sq_acc (R2 overlap). |
| cb_output_tiles | `2 · out_batch_tiles` (`out_batch_tiles = core_w_tiles`) | `out_batch_tiles` per stage | `{rows: streams → 1 tile row at a time, w_local: spans → core_w_tiles per batch, sources: streams → 0}` | x dtype | compute | writer | normalize → store | Capacity exceeds the live set by 2×. That is **double buffering**: the pack of tile-row j+1 overlaps the write of j (catalog `double_buffer`). It cannot pack into the destination because the destination is interleaved DRAM. It cannot alias cb_x_resident: compute would become a second producer. |

### Symbol table (worker)

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `block_rows` | ≤ `core_rows = ceil(tensor_row_tiles / S_r)` | `_plan_blocks()`; Phase 0 sets `block_rows = core_rows` under `P_resident` |
| `core_w_tiles` | ≤ `ceil(tensor_w_tiles / S_w)`; ≥ `min(tensor_w_tiles, min_core_w_tiles = 4)` | `_choose_split()` |
| `n_src_tiles` | `ceil(G · S_w / 32)`; G ≤ 8 (Galaxy axis 0), `S_w ≤ ceil(tensor_w_tiles/4)` | `_choose_split()` plus mesh shape |
| `out_batch_tiles` | `= core_w_tiles` | host constant derived from `core_w_tiles` |
| `T_x` | x tile bytes: 4096 (fp32) / 2048 (bf16) / 1088 (bfp8, TARGET) | dtype |
| `T_g` | gamma tile bytes: 4096 / 2048 | gamma dtype |
| `T_f` | 4096 (Float32 tile) | — |
| `HAS_GAMMA` | 0 / 1 | gamma given |
| `L1_BUDGET` | 1 MiB per worker for CBs (host constant `WORKER_L1_BUDGET`) | — |

### Footprint (per worker, bytes)

```
F_worker = block_rows · core_w_tiles · (T_x + HAS_GAMMA · T_f)        # cb_x_resident + cb_xg       -> scales with block_rows x core_w_tiles
         + HAS_GAMMA · core_w_tiles · T_g                              # cb_gamma_tiles              -> scales with core_w_tiles
         + 2048                                                       # cb_scaler                   -> constant
         + block_rows · T_f · (3 + n_src_tiles)                        # sq_acc, partial, rstd, combine -> scales with block_rows (x n_src_tiles)
         + 2 · core_w_tiles · T_x                                      # cb_output_tiles             -> scales with core_w_tiles
P_resident :=  F_worker(block_rows = core_rows) <= L1_BUDGET
```

The largest TARGET INPUTS case is 2048×2048 fp32 with fp32 gamma: `S_w = 5`, `core_rows = 3`, `core_w_tiles = 13`, `n_src_tiles = 1` (G = 2):

```
F = 39·8192 + 13·4096 + 2048 + 3·4096·4 + 26·4096 ≈ 0.53 MiB  ≤ 1 MiB
```

## Port cores

No CBs. Each port core holds:
- its shard of the per-call **exchange scratch** tensor: `G · slot_bytes`, with `slot_bytes = 128 · max_l recs_link[l]`;
- the fabric packet-header pool (firmware-owned, 12 headers per DM RISC);
- global semaphores `arrival` and `ready`, 4 B each.

| Buffer | Capacity | Live set | Axis accounting | Format | Producer | Consumer | Lifetime | Shares with / why not |
|--------|----------|----------|-----------------|--------|----------|----------|----------|-----------------------|
| exchange scratch (L1 tensor shard) | `G · slot_bytes` | all G slots until the stats-ready mcast. Workers pull after it, so the slots must hold for the whole call. | `{rows: spans → link's rows, w_local: spans → S_w records per row, sources: spans → G slots}` | raw fp32 records (128 B) | local workers (slot p, NoC write) plus the upstream port (Fabric) | peer port (relay source) plus workers (NoC read) | one call | **No share.** It is call-scoped by design: no credits and no round trips. It is a per-call tensor (freed after the call), not a persistent allocation. |

`P_scratch := G · slot_bytes ≤ 1 MiB`. The worst INPUTS case is 4096×1024 on Galaxy axis 0 (G = 8, L = 1, `S_w = 4`, 128 rows): 8 · 512 · 128 B = 512 KiB.

## Data-movement budget

Per device, per call, for split R1. B = x bytes = `batch·ceil(H/32)·ceil(W_local/32)·T_x`.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| x | 1 (read) | x is resident in cb_x_resident across the exchange (`P_resident`). No second pass. | — |
| output | 1 (write) | written once from cb_output_tiles | — |
| gamma | S_r (each worker reads its W segment once per call) | Stepping stone. The broadcast (R8) would make it 1. | — |
| stats records | 0 | They live in L1 only. The stats never round-trip through DRAM, unlike today's composite. | NoC: `Ht·S_w·128 B` worker→port (×2 ports per link at interior devices) plus `S_r·S_w·G·S_w·128 B·block_rows/S_r` port→worker pulls. Fabric: per link direction, ≤ (G−1) entries of `recs_link·128 B`. |

Totals at the perf-focus shape (640×1792 bf16, G = 2, L = 2, 117 workers, S_w = 5):

| Tier | Bytes |
|------|-------|
| DRAM | 2.29 MB read + 2.29 MB written (minimum) + gamma 20 × 3.6 KB = 72 KB |
| NoC | 12.8 KB of records to the ports, 100 workers × 10 records × 128 B = 128 KB of pulls |
| Fabric | 6.4 KB per link direction, one packet each |

> Cheapest-traffic split considered: rows × W-chunks with x resident and flat records (R1). The delta is 0: DRAM is at the x-once and out-once minimum. Implemented: R1. The implemented split **is** the cheapest-traffic split. The only above-minimum term is gamma (S_r reads of a W_local stick); its broadcast is deferred as R8 because it is ≤ 3 % of DRAM bytes.
