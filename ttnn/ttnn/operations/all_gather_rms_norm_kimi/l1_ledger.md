# L1 Ledger: all_gather_rms_norm_kimi

Schema: `.claude/references/l1-footprint-discipline.md`. Block axes (from `op_design.md` → Blocking Model):

- `rows`: `block_h` tile rows
- `width`: `block_w` tile columns of one slice
- `slices`: G group positions; a block never straddles them
- `chunks`: the landed-chunk stream, quantum `chunk_tiles`

Tile = 2048 B (bf16) / 4096 B (fp32). Phase 0 values: `block_h = 1`, `block_w = 12`, `chunk_tiles = 6`, `G = 4`, `port_batch_chunks = 4`, `out_depth_chunks = 2`.

## Compute cores (80)

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_x_block` (backed on the `landing` scratch shard) | `block_h·block_w` | `block_h·block_w`: the whole block stays resident (partial phase, port own-sends / relays and the final multiply all read it) | {rows: spans → block_h, width: spans → block_w, slices: one slice, chunks: streams → waited in chunk_tiles quanta} | Float16_b | reader (own: DRAM load; remote: zero-copy push after landing) | compute | whole call | No CB allocation of its own: it **is** the `landing` scratch (zero-copy). Cannot host `cb_out` in place, because the ports read x from it concurrently with compute. |
| `cb_gamma` (Refinement 2: own allocation) | `block_w` | `block_w` (row 0 of each) | {rows: streams (shared by all rows) → 1, width: spans → block_w, slices: one, chunks: n/a} | Float16_b | reader (copied from the hub's γ multicast in the scratch tail) | compute | t0 → the last chunk's xg | No share since Refinement 2: γ is now read per chunk (xg = x ⊙ γ) right up to the last chunk, so it is live together with `cb_out`. |
| `cb_out` (Refinement 2: own allocation) | `out_depth_chunks·chunk_tiles` | `2·chunk_tiles` | {rows: streams, width: streams → 2·chunk_tiles window, slices: one, chunks: spans → 2} | Float16_b | compute | writer | first normalized chunk → end | No share: live with `cb_gamma` (above), `cb_xg` and `cb_x_block`. |
| `cb_partials` | `G·block_h` | `G·block_h` (the combine sums all G in fixed order) | {rows: spans → block_h, width: n/a (col 0), slices: spans → G, chunks: n/a} | Float32 (`UnpackToDestFp32`) | reader | compute | `wait_stats` → `combine_stats_block` | No share. It is live at the same time as `cb_scale`'s producer inputs (`cb_rstd`) on the critical path, and the 1-tile fp32 CBs it could take are already paired below. Disjoint from `cb_xsq_sum`, but different page count and UnpackToDest mode, and sharing would serialize the own-core partial phase against the stats receive. |
| `cb_xsq_sum` / `cb_rstd` (one allocation) | `block_h` | `block_h` | {rows: spans → block_h, width: collapsed, slices: one, chunks: n/a} | Float32 | compute | compute | xsq: `partial_block`. rstd: `combine` → `expand` | **Shared**: on own cores the partial phase (xsq) ends before the hub multicast that `combine` waits on. Same format and size. |
| `cb_partial` / `cb_rstd_full` (one allocation) | `block_h` | `block_h` | {rows: spans → block_h, width: collapsed (partial) / spans one tile (rstd_full), slices: one, chunks: n/a} | Float32 | compute | writer (partial) / compute (rstd_full) | partial: until `export_partial` pops it, which precedes the hub's publish of slot p. rstd_full: `expand` → the last chunk's normalize | **Shared**: same reasoning; the combine needs every slot, including the own slot p, which the hub publishes only after every own core exported |
| `cb_xg` (Refinement 2, replaces `cb_scale`) | `block_h·block_w` | own cores: the whole block (xg computed before the stats); remote cores: the chunks the host's look-ahead table puts before the combine (≤ the block), then one chunk at a time | {rows: spans → block_h, width: spans → block_w, slices: one, chunks: streams (pushed and popped in chunk quanta)} | Float32 | compute | compute | first xg chunk → its normalize | No share: live with `cb_x_block`, `cb_gamma` and `cb_out` through the normalize, and fp32 vs the bf16 CBs. Same size as the `cb_scale` it replaces. |
| `cb_scaler` | 1 (constant) | 1 | {rows: n/a, width: n/a, slices: n/a, chunks: n/a}: a constant | Float16_b | reader | compute | own cores: whole call | No share: 2 KiB, live with everything in the partial phase |
| `stats_scratch` (sharded scratch, not a CB) | `G·Mt·128 B` + 64 B align + `Wt·64 B` (γ sticks) + `16·G B` (hub mask words) | the stats slots as the hub publishes them; each core reads its rows. The γ tail holds all Wt sticks; each core copies its block_w | {rows: spans → all Mt rows (multicast convenience), width: γ spans → Wt sticks, slices: spans → G, chunks: n/a} | uint32 (fp32 bits / bf16 sticks) | hub multicast (per slot; γ once) + own writer for its own row | reader | whole call | No share: the hub multicasts into it at a fixed address, and it must exist on every op core. The bound is set by a predicate (symbol table). |

## Port cores (2L = 4)

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_port_chunks` | `2·port_batch_chunks·chunk_tiles` | `2·port_batch_chunks·chunk_tiles`: one batch read while the other is sent | {rows: streams, width: streams, slices: streams (own then relay entries), chunks: spans → 2·port_batch_chunks} | Float16_b | port reader | port sender | whole call | No share: the only CB on the core. Capacity = 2 × batch by design: double buffering, measured best in `fabric_all_gather`. |
| `stats_scratch` (sharded scratch) | `G·Mt·128 B` | slot p (+ slot p−1 on link-0 forward) | {rows: spans → Mt, slices: spans → ≤ 2 used, others: n/a} | uint32 | hub (NoC write) | port sender (fabric source) | whole call | Same tensor as on the compute cores (one allocation over every op core, same address). |

## Receivers (2L) and hub (1)

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `stats_scratch` (hub) | `G·Mt·128 B` | all G slots | {rows: spans → Mt, slices: spans → G, width/chunks: n/a} | uint32 | own writers (slot p), fabric (other slots) | hub (multicast, port forwarding) | whole call | Same tensor (shard). Receivers carry an unused shard of it because the tensor is allocated over every op core. |

## Symbol table

| Symbol | Bound | Predicate that establishes it |
|--------|-------|-------------------------------|
| `block_h` | `1 ≤ block_h ≤ Mt`; Phase 0 = 1 | `block_h = ceil(Mt / floor((grid − 4L − 1)/(G·Wl_t/block_w)))`. Raise if `G·Wl_t/block_w > grid − 4L − 1`. |
| `block_w` | divides `Wl_t`; Phase 0 = `Wl_t` = 12 | contract (`W_local` multiple of 32) |
| `chunk_tiles` | `≤ floor(max_payload / 2048)` = 7; divides `block_h·block_w`; Phase 0 = 6 | router payload cap (fabric-wide) |
| `G` | mesh `shape[cluster_axis]`, = 4 | validation |
| `Mt` | `G·Mt·32·4 B ≤ 64 KiB` (Mt ≤ 128 at G = 4) | stats_scratch cap (raise outside it) |
| `L` | `≤` links discovered on every hop; = 2 | validation |
| `port_batch_chunks` | `≤ 8` (header pool, 12 per RISC) | host clamp; Phase 0 = 4 |
| `out_depth_chunks` | 2 | constant |

## Footprint per core

Compute core (as built after Refinement 2):
`2048·block_h·block_w` (landing)
`+ 2048·block_w` (γ) `+ 2048·out_depth_chunks·chunk_tiles` (out)
`+ 4096·G·block_h` (partials)
`+ 2·4096·block_h` (the two shared 1-tile fp32 pairs)
`+ 4096·block_h·block_w` (xg)
`+ 2048` (scaler)
`+ G·Mt·128 + 64 + Wt·64 + 16·G` (stats scratch with the γ tail and the hub mask words)

At the PERF FOCUS cell: 24 + 24 + 24 + 16 + 8 + 48 + 2 + 13 = **159 KiB** (Phase 0: 132 KiB), of ~1.5 MB L1. The `block_h·block_w` terms scale with the block knobs. `G·Mt·128` scales with the op dimension M and is bounded by the predicate above. Nothing scales with W beyond `block_w ≤ Wl_t`.

Port core: `2·port_batch_chunks·chunk_tiles·2048 + G·Mt·128` = 96 + 10 = **106 KiB** (scales with `port_batch_chunks`, `chunk_tiles`).

Hub / receiver: `G·Mt·128` = **10 KiB**.

No budget solve, safety fraction or blocking search was introduced: the inventory fits at the coarsest block (the whole per-core assignment) by a factor of ~10.

## Data-movement budget (per device, per call; 640 × 384 → 640 × 1536, G = 4, L = 2)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| input x (own slice, 240 tiles, 480 KiB) | 1 read | Own cores load it into `landing` once. Ports read it from there (L1), not from DRAM (regime R8 rejected). | NoC: L1 → the 4 port cores, 1 slice in total. Fabric out: 2 directions × 1 slice over 2L ports. |
| remote slices (720 tiles, 1.44 MiB) | **0** | They land straight in compute-core `landing` (L1) and are normalized there (regime R2 rejected). | Fabric in: 3 slices. The busiest link direction carries 1.5 slices / L. Relays: 0.5 slice of NoC L1 → port reads. |
| γ (3 KiB) | 1 read (Refinement 2: regime R6 built, `GAMMA_HUB_MCAST`) | the hub reads the Wt = 48 sticks once per chip; Phase 0 had every compute core read its 12 (960 small reads on the same pages, ~4.5 µs) | NoC: one 3 KiB hub multicast to the compute rectangles + a `gamma_go` multicast set |
| output (960 tiles, 1.875 MiB) | 1 write | every tile is written once by the core that normalized it | — |
| partials / stats (G × 2.5 KiB) | 0 | fp32, L1 → fabric → L1 | Fabric: 5 KiB on link-0 forward, 2.5 KiB on link-0 backward. NoC: 20 × 128 B to the hub, the hub → 2 ports (7.5 KiB), and per slot (Refinement 2) a 2.5 KiB hub multicast to 80 cores + a mask-flag multicast set: 10 KiB in total as before. |

Totals per device:
- DRAM: 0.48 MiB read + 3 KiB γ + 1.875 MiB write ≈ **2.36 MiB** (roofline 2.40 MiB; Phase 0 2.42 MiB with the per-core γ reads).
- Fabric: busiest link direction 0.72 MiB / 2 links (+5 KiB stats).
- On-chip NoC: ~1.5 MiB of L1 → port reads + 10 KiB multicast + 1.875 MiB of DRAM writes.

> Cheapest-traffic split considered: W-split by slice × rows with L1 landing and fp32 stats (R1). It is tied on bytes with the rows-only split (R5): same DRAM and fabric bytes, minus nothing. Implemented: R1. It **is** the cheapest-traffic split. The tie with R5 is broken by occupancy (80 vs 20 cores) and the per-core tail, as `op_design.md` states.

## Implementation notes (Phase 0, after verification)

The three planned shares are now built (verifier, Phase 0). Each pair is one `CBDescriptor` with two
`CBFormatDescriptor`s (one allocation, two indices with independent FIFO pointers), so the table above matches
the code:

| Share | Allocation | Why the alias is safe in the built schedule |
|-------|------------|---------------------------------------------|
| `cb_gamma` / `cb_out` | `ceil(max(block_w, out_depth_chunks·chunk_tiles) / chunk_tiles)·chunk_tiles` bf16 pages (= 12 at Phase 0). The round-up keeps `cb_out`'s capacity a multiple of its push quantum. | The unpacker finishes reading γ in `scale_block` before it unpacks anything of `normalize_block`, and the packer can only write `cb_out` after that math. The reader writes γ at t0, before any `cb_out` reservation. |
| `cb_xsq_sum` / `cb_rstd` | `block_h` fp32 pages | The reduce pops `cb_xsq_sum` before `combine_stats_block` packs `cb_rstd` (both on the one compute thread, in order). |
| `cb_partial` / `cb_rstd_full` | `block_h` fp32 pages | The writer pops `cb_partial` before it increments the hub's `own_partials`. The hub multicasts the stats only after every own core did that, and `expand_rstd_block` runs only after those stats arrive. |

Footprint as built, compute core: **132 KiB** at Phase 0 (block 1 × 12, chunk 6), equal to the planned figure above.
In the tall regime (block 3 × 12, chunk 6) it is 72 + 24 + 48 + 24 + 144 + 2 + 32 = **346 KiB**. Port core: 106 KiB.

`cb_xsq_sum` and `cb_partials` are tagged `UnpackToDestFp32`: they are reloaded losslessly through `copy_tile` and
are never an FPU operand. Every fp32 page sits under `fp32_dest_acc_en=True` (page format vs DEST width: no
finding).

Placement as built: the 2L ports, 2L receivers and the hub sit on logical row 0, with the ports centred. The
compute cores fill rows 1.. in row-major order (rectangles: full rows + one partial row). There is no
Ethernet-probe placement, because the example's probe is a second device dispatch on the first call, which the
one-dispatch-per-invocation rule forbids. Placing the ports by link without a probe is perf Refinement 1. The
data-movement budget above is unchanged: placement moves no bytes, only NoC hop counts.

## Refinement 2 changes (OVERLAP)

| Change | L1 effect | Why |
|--------|-----------|-----|
| `cb_scale` → `cb_xg` (same size, fp32) | 0 | x ⊙ γ is computed before the stats; after them only xg ⊙ rstd_full is left per chunk |
| `cb_gamma` / `cb_out` alias split into two allocations | +24 KiB | γ is read per chunk until the last one, so it is live with `cb_out` |
| `cb_rstd_full` alias kept with `cb_partial` | 0 | `expand` runs after the combine, which needs the own slot, published only after every export |
| `stats_scratch` + γ tail (`Wt·64 B` + 64 B align) + hub mask words (`16·G B`, one 16 B L1 slot each: a NoC write source must share the destination's 16 B alignment) | +3.1 KiB on every op core | regime R6 (γ read once per chip) and the per-slot stats publish |

Compute-core footprint: 132 → 159 KiB at the PERF FOCUS cell; tall regime (block 3 × 12): 346 → 373 KiB.
