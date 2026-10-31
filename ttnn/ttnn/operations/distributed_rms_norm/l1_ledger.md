# L1 Ledger: distributed_rms_norm

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking axes are those of `op_design.md` → Blocking Model:
- `batch` (folded into `tile_rows`);
- `tile_rows` (block extent `R`);
- `wave` (`Rw`);
- `width`: the slice `S` (reduction) and the chunk `C` (streaming);
- `device_slot` (`G`);
- `link` (`L`).

Every row lists every axis.

The base op's buffers keep its sizing (`base_ops/rms_norm/op_design.md` → Circular Buffers, `rms_norm_program_descriptor.py` `footprint`). This ledger restates them only so the total is closed-form here. The rows marked **new** are what this op adds.

## Compute cores (every active core; root-only buffers are declared on every core for a uniform layout)

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_x_tiles` | `x_depth · R · WS` (sharded: the shard, zero-copy) | `R · WS` per block; `x_depth = 2` adds the prefetched block | `{batch: folded, tile_rows: spans → R (×x_depth), wave: spans → all waves of the block, width: spans → WS (= Wp ≥ S), device_slot: none, link: none}` | input dtype | reader | compute | load → release, per block | none. It is the residency that makes x cross DRAM once, so it cannot alias anything live across the block. `x_depth = 2` is the explicit pipelining decision (block k+1 prefetch during block k's exchange). |
| `cb_reduce_scaler` | 1 (2 with the sharded W-pad mask) | 1 (2) — constant | `{batch: none, tile_rows: none, wave: none, width: none, device_slot: none, link: none}` | bfloat16 | reader | compute | kernel | none: resident for the kernel's life, read by every collapse |
| `cb_sumsq_partial` | `Rw` (implemented; was `R`) | `Rw` (one wave's raw Σx² tiles between the two helpers) | `{batch: folded, tile_rows: streams → Rw per wave, wave: spans → Rw, width: none (collapsed in DEST), device_slot: none, link: none}` | interm | compute | compute | square → collapse | Sized to the live set `Rw`: square pushes one page per tile-row and collapse pops the wave, so `Rw` never blocks. Kept separate from `cb_partial_col`: concurrent within a wave (collapse reads one while it writes the other). |
| `cb_inv_rms` | `R` | `R` (every wave's rows stay until `release_block`, read by every chunk) | `{tile_rows: spans → R, wave: spans → all, width: none (Col-broadcast operand), device_slot: none, link: none, batch: folded}` | interm | reader | compute | distribute → release | none: live across every chunk of the block |
| ~~`cb_gamma_row`~~ (RM γ) | **deleted** (implemented) | — | — | — | — | — | — | Not a helper input: the writer reads stick `j` (`32·es_g` B, DRAM-aligned) straight into `cb_gamma_tiles` page `j` (values 0..15 land in face-0 row 0) and moves values 16..31 to face-1 row 0 with 16 RISC-V word copies. No staging buffer. |
| `cb_gamma_tiles` (γ) | resident `Wp` when the blocking fits with it; else streaming `STREAM_DEPTH·C` (Refinement 1: base op's L1 fallback, `gamma_tiles_pages()`) | resident: same as capacity; streaming: `C` per (block, chunk) step | `{width: spans → Wp (resident) / C-chunk ring (streaming), tile_rows: none, wave: none, device_slot: none, link: none, batch: none}` | γ dtype | writer | compute | kernel (resident) / per chunk (streaming) | none: Row-broadcast operand of every scale_by_gamma call. Streaming re-reads the window from DRAM once per block (`num_blocks·V` γ tiles instead of `V`); chosen only when resident fails to fit (wide-row shards). |
| `cb_x_gamma` (γ) | `R · C` | `R · C` (chunk 0 hoisted for the whole block, then per chunk) | `{tile_rows: spans → R, wave: spans → all (hoisted), width: streams → C, device_slot: none, link: none, batch: folded}` | interm | compute | compute | per (block, chunk) | none: chunk 0's x·γ is held across the exchange wait by design (stall shadow). Packing x·γ back into `cb_x_tiles` in place would round it to the input dtype before the inv_rms multiply. |
| `cb_partial_col` (R3) | `R` | `Rw` | `{tile_rows: streams → Rw, capacity R, wave: spans → Rw, width: none, device_slot: none, link: none, batch: folded}` | interm | compute | reader | collapse → combine | Capacity `R`: compute may collapse every wave of the block before the reader sends (the reader is busy with publish/collect). Concurrent with `cb_sumsq_partial` within a wave. |
| `cb_gather` (R3 root) | `nwpb · G_w · Rw` (= `G_w · R`) | `G_w · Rw` per wave; slots are block-indexed (base slot safety) | `{tile_rows: spans → R, wave: spans → all, width: none (one partial per slice core: G_w), device_slot: none, link: none, batch: folded}` | interm | reader | compute | per block | none. Contributors write wave slots remotely at fixed offsets `index·R + r`, so the whole block's slot range must exist (base Refinement 4 contract). |
| `cb_rms_root` (root, R3 only) | `nwpb · Rw` (= `R`) | `Rw` | `{tile_rows: streams → Rw, capacity R, wave: spans → Rw, width: none, device_slot: none (already summed), link: none, batch: folded}` | interm | compute | reader | finalize → distribute | Capacity `R` keeps the base's block-wide root CB. Cannot alias `cb_dev_partial`: wave w's inv_rms and wave w+1's device partial are live together. |
| `cb_sumsq_last`, `cb_sumsq_acc` (sharded, W % 32 ≠ 0) | base | base | `{width: last tile only, tile_rows: streams → 1, wave: none, device_slot: none, link: none, batch: folded}` | interm | compute | compute | per tile-row | base: concurrent operands of the masked collapse |
| `cb_output` | `stream_depth · C` (native — output shard == input shard: the output shard) | `C` + one in flight | `{tile_rows: streams → 1 per quantum, wave: streams, width: streams → C, device_slot: none, link: none, batch: folded}` | output dtype | compute | writer | per chunk | none: double-buffered compute→writer stream (`stream_depth = 2` is the overlap decision) |
| **new** `cb_dev_partial` (root) | `nwpb · Rw` (= `R`; ≤ `2 · Rw` at `NUM_WAVES = 2`) | `Rw` | `{tile_rows: streams → Rw, wave: spans → Rw (×2 waves in flight), width: none (device-summed), device_slot: none (own slot only), link: none, batch: folded}` | interm | compute | reader | collapse/combine → publish | Capacity `nwpb·Rw`: compute packs wave w+1 while the reader still publishes wave w (`partial_depth = nwpb`, so every per-wave push divides the capacity). Cannot reuse `cb_partial_col`: a different consumer protocol on the same core (`cb_partial_col` goes to the root's gather, which on the root is itself), and both are live in the same wave on the root. Cannot be written straight into `cb_stats_gather` slot p either: that is a CB with a different producer (reader). The single-producer rule forbids compute and reader both pushing it. |
| **new** `cb_stats_gather` (root) | `G · Rw` | `G · Rw` | `{tile_rows: spans → Rw, wave: spans → Rw, width: none (column 0 only), device_slot: spans → G, link: none, batch: folded}` | interm, `UnpackToDestFp32` under fp32 DEST | reader | compute | collect → finalize | none: it is the only place the G slots of a row meet. Depth 1 wave (`stats_depth = 1`): compute consumes it immediately, since finalize is the only thing waiting on it. Could share with `cb_gather` (both root-side reader→compute)? No: `cb_gather`'s fixed remote-write slot offsets are block-wide and live while waves are collected. |
| **new** `cb_stats_record` (root) | `(G + 1) · Rw` pages of 128 B | `(G + 1) · Rw` | `{tile_rows: spans → Rw, wave: spans → Rw, width: none, device_slot: spans → G (+1 own staging), link: none, batch: folded}` | uint32 (fp32 words) | reader | reader (scratch, never pushed) | per wave | none: records must be contiguous for one NoC run per slot, but `cb_stats_gather` holds them face-strided (column 0). At most `9 · Rw · 128 B` (G = 8). |

## Port cores (`2L` per chip; no CBs)

| Buffer | Capacity | Live set | Axis accounting | Format | Producer | Consumer | Lifetime | Shares with / why not |
|--------|----------|----------|-----------------|--------|----------|----------|----------|-----------------------|
| port landing (L1 `HEIGHT_SHARDED` uint32 tensor over the whole grid, **allocated per call**, freed right after the dispatch; ports use their own shard) | `K · G · seg_stride` B (`K = num_waves` on the credit-free fast path, else `K = max(LANDING_RING_WAVES, nwpb, 2)` wave slots reused under credits, Refinement 4), with `seg_stride = align_up(n_roots_l · Rw · 128, 64)` (bucketed to a power of two of words, ≥ 16 KiB) | the whole capacity: every (wave, slot) is written once per call, read by relays and roots, never reused within a call | `{tile_rows: spans → this link's rows (n_roots_l · Rw per wave), wave: spans → num_waves, width: none, device_slot: spans → G, link: one link per port, batch: folded}` | uint32 (fp32 words) | roots (own slot, NoC write) + fabric router (peer slots) | port (relay source) + roots (collect) | per call | fast path: none, the reuse-free layout removes the credit round trip (`blackhole-ccl-kernels.md` §7); long calls whose full landing exceeds 512 KiB reuse K wave slots across waves (credit per wave). Per call (not persistent): a persistent one pinned the middle of every bank (verifier). Its address range is reserved in every bank, so it is in every compute core's budget for that call. |
| global semaphores (ready + G−1 arrival counters) | 16 B each, persistent, one grow-only set per mesh over the whole grid | — | — | uint32 | port / peer ports | port | mesh lifetime (re-armed to 0 each call) | persistent by necessity: a peer increments the ready counter before this chip's program for the call starts |
| packet headers | `PacketHeaderPool` (static, ≤ 12) | 1 (one destination) | `{all: none}` | header | port | port / router | kernel | n/a (fabric-owned pool) |

## Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `R` (`block_rows`) | `1 ≤ R ≤ rows_this_core` | `choose_blocking` closed form (base) with this ledger's extra terms; refusal (c) if `R = 1, x_depth = 1` does not fit |
| `Rw` (`wave_rows`), `nwpb` | `Rw = R / nwpb`, `nwpb` = the largest divisor of `R` ≤ `min(NUM_WAVES, MAX_WAVES_PER_BLOCK)`; `1 ≤ Rw ≤ R` | `NUM_WAVES = 2`, `MAX_WAVES_PER_BLOCK = 8` host constants (`wave_split`) |
| `S`, `WS`, `Wp`, `C` | `C ≤ W_CHUNK_TILES = 32`; `WS ≤ Wp ≤ S + C − 1` | base `compute_geometry`; R3 `S` from `_r3_interleaved_layout`, or the shard width |
| `G_w` | `1 ≤ G_w ≤ R3_MAX_GROUP_CORES = 24` (interleaved); shard columns (sharded) | base layout rule / shard spec |
| `G` | `2 ≤ G ≤ mesh axis size` (≤ 8 on a Galaxy axis) | `mesh_shape[cluster_axis]`, validated ≥ 2 |
| `L` | `1 ≤ L ≤ L_usable` (≤ 4 on Blackhole boxes) | validation |
| `n_roots_l` | `≤ ceil(roots / L)`, `roots ≤ N_fused` | plan |
| `num_waves` | `num_blocks_max · nwpb` | plan |
| `T_in, T_out, T_g` | ≤ 4096 B | dtype |
| `T_i` | 4096 (fp32 DEST) / 2048 (16-bit DEST) | `intermediate_format(fp32_dest_acc_en)` |
| `x_depth`, `stream_depth`, `partial_depth`, `stats_depth` | 1–2 / 2 / `nwpb` / 1 | `x_depth`: 2 only when a core walks > 1 block and it fits (`choose_block_rows`); `STREAM_DEPTH = 2`; `partial_depth = nwpb` |

## Total per-core footprint

Compute core (worst case: an R3 root with γ):

```
F(R, d) = d·R·WS·T_in                                      # cb_x_tiles (interleaved only) — scales with R, S, x_depth
        + R·T_i·(1 + C·has_gamma)                          # cb_inv_rms, cb_x_gamma — scale with R (and C)
        + R·T_i·(2 + G_w)·combine                          # cb_partial_col, cb_rms_root, cb_gather (R3) — scale with R, G_w
        + (Wp | STREAM_DEPTH·C)·T_g·has_gamma              # cb_gamma_tiles (resident window | streaming ring) — scales with S | C
        + STREAM_DEPTH·C·T_out·(1 − out_native)            # cb_output — fixed in C (out_native: same shard as input)
        + scaler_pages·2048                                # cb_reduce_scaler
        + Rw·T_i·(1 + 2·masked)                            # cb_sumsq_partial (+ cb_sumsq_last / _acc) — scales with Rw
        + R·T_i + G·Rw·T_i + (G + 1)·Rw·128                # NEW: dev_partial, stats_gather, stats_record — scale with R, Rw, G
```

**Single source:** this expression is exactly `footprint()` in `distributed_rms_norm_program_descriptor.py`, which sums
`cb_inventory()`, the same list the program's CB descriptors are built from (verifier fix: the two used to restate
every page count independently).

The new terms are bounded by `R·T_i + G·Rw·T_i + (G+1)·Rw·128`: at the focus shape (`Rw = 1, G = 4, fp32 DEST`) that is 24.6 KiB, and ≤ 41 KiB at `G = 8`. They enter `choose_blocking`'s fixed term, scaled by `Rw ≤ R`. No buffer is unconditionally op-sized: every capacity is a block extent (`R`, `Rw`, `S`, `C`) times a bounded constant (`G`, `G_w`, depths), or the caller's shard.

**Budget (Refinement 1).** `l1_budget()` is the largest free block, bounded by the lowest address of every L1 buffer
the call knows (input / output / γ / landing tensors and the persistent exchange semaphores) minus the allocator base:
the static CB region must end below the lowest L1 buffer, and a freed hole above a live buffer made the largest free
block over-state it (measured: the global semaphores, created on a mesh's first call just below that call's 512 KiB
shard at 524160 B, clashed with a later call's CB region ending at 527744 B).

Port core: `num_waves · G · seg_stride` ≤ 128 KiB for every INPUTS shape (credit-free). Above `LANDING_FULL_MAX_BYTES` (512 KiB, bucketed to a power of two of words) the landing becomes a ring of `K` wave slots (Refinement 4); only a ring above `PORT_LANDING_MAX_BYTES` is refused before allocation. The landing's L1 is reserved in every core's bank, so the compute plan is solved against `l1_budget − landing bytes` (re-solved when the landing exceeds `LANDING_L1_RESERVE`). One grow-only landing per (mesh, port set): a bigger call replaces it and smaller calls reuse it.

## Data-movement budget (chosen split; per device, per call)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| x | 1 read (0 when sharded) | the block is resident in `cb_x_tiles` for both passes (square and scale) | none |
| out | 1 write (0 DRAM when sharded; native same-placement shard: packed in place, no NoC; any other placement: 1 NoC write per tile via the output TensorAccessor) | written once from `cb_output` | none |
| γ | one S-window per active core: TILE γ reads one 544 B row-0 prefix per tile (bf16), so `active_cores · S · 544 B` (focus: 100 × 6 × 544 B ≈ 0.33 MB). RM sticks: `active_cores · S · 32 · es_g`. | per-core window read (G-MC deferred) | none |
| Σx² partials (R3) | 0 | on chip only | per tile-row: `G_w − 1` faces-0/2 tiles to the root (2 KiB fp32 each) |
| inv_rms (R3) | 0 | on chip only | per tile-row: `G_w − 1` faces-0/2 tiles from the root |
| device records | 0 | L1 landing on the port cores | per wave: `Rw·128 B` to ≤ 2 ports per root, then `(G−1)·Rw·128 B` read back per root |
| records over the fabric | — | — | received per device: `(G−1)·T·128 B` (focus: 3 × 2.5 KiB = 7.5 KiB). Busiest link per call: Ring `kf·T·128 / L` B (focus at L = 2: 2 × 2.5 KiB / 2 = 2.5 KiB). Line: `(G−1)·T·128 / L` B. |

Totals at the focus case (640 × 1792 bf16, G = 4 Ring, L = 2, R3-i):
- DRAM: 2.29 MB read + 2.29 MB written + ≈ 0.33 MB γ prefixes.
- On-chip NoC: 20 tile-rows × 9 × 2 × 2 KiB ≈ 0.74 MB (partials + inv_rms), plus 20 × 4 × 128 B of records (10 KiB).
- Fabric: 7.5 KiB received per chip.

> Cheapest-traffic split considered: `tile_rows` over cores, rows resident (R1). Delta vs R3-i: −0.74 MB of on-chip NoC traffic at the focus case, same DRAM and fabric bytes. Implemented: R1 wherever the base predicate selects it (2048 × 1280, every `T ≥ N_fused` shape), and R3-i where `T < N_fused`. At the focus case R1 puts 20 of 106 cores on DRAM, and the base measured R3 15 % faster on 640-row shapes (`changelog.md` Refinement 4: 640 × 6144 49.9 → 43.3 µs). The extra NoC bytes buy DRAM occupancy, a measured trade, not an unbuilt alternative. Both regimes are built and selected by an exact predicate.


## Implementation deltas (Phase 0, ttnn-implementer)

- `cb_sumsq_partial` resized `R → Rw`; `cb_gamma_row` deleted; γ resident only (rows above).
- `cb_dev_partial`, `cb_rms_root`: `nwpb · Rw` (= `R`); `cb_gather`: `nwpb · G_w · Rw` (= `G_w · R`). Every full block's
  waves are exactly `Rw` rows (`nwpb` = the largest divisor of `R` ≤ `NUM_WAVES`), so every per-wave push quantum divides
  its capacity and no push straddles the ring end.
- R1 / R1-HS: `finalize_wave` packs straight into `cb_inv_rms` (the core is its own group); `cb_rms_root`,
  `cb_partial_col`, `cb_gather` exist only when `G_w > 1`.
- Interleaved pad tiles `[V, Wp)` of `cb_x_tiles` are not zeroed: Σx² runs over the `V` valid tiles only; pad output tiles
  are computed on stale data and never written. The W-pad columns of the last valid tile are still zeroed per row.
- `MIN_BLOCKS_PER_CORE = 1` (base op: 2) and `x_depth = 1` whenever a core walks one block. Data-movement budget is
  unchanged (x read once, output written once); at 640×1792 every core now holds its 2 tile-rows in one block
  (`R = 2`, 2 waves of 1 row).
- Program semaphores per compute / port core: `SEM_INV, SEM_OWN_STATS, SEM_LANDED_FWD, SEM_LANDED_BWD,
  SEM_OWN_COUNTED` + one `SEM_GATHER_k` per wave of a block (≤ 8), then the fabric connection's 2 on port cores (≤ 16).
