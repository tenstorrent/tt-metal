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
| `cb_gamma` / `cb_out` (one allocation, two indices) | `max(block_w, out_depth_chunks·chunk_tiles)` | γ: `block_w` (row 0 of each). out: `2·chunk_tiles` | γ {rows: streams (shared by all rows) → 1, width: spans → block_w, slices: one, chunks: n/a}. out {rows: streams, width: streams → 2·chunk_tiles window, slices: one, chunks: spans → 2} | Float16_b | reader (γ) / compute (out) | compute (γ) / writer (out) | γ: t0 → end of `scale_block`. out: `normalize_block` → end | **Shared**: disjoint lifetimes, same format. γ is dead once S is computed, and `normalize_block` cannot start before `scale_block` ends because it needs `cb_scale`. |
| `cb_partials` | `G·block_h` | `G·block_h` (the combine sums all G in fixed order) | {rows: spans → block_h, width: n/a (col 0), slices: spans → G, chunks: n/a} | Float32 (`UnpackToDestFp32`) | reader | compute | `wait_stats` → `combine_stats_block` | No share. It is live at the same time as `cb_scale`'s producer inputs (`cb_rstd`) on the critical path, and the 1-tile fp32 CBs it could take are already paired below. Disjoint from `cb_xsq_sum`, but different page count and UnpackToDest mode, and sharing would serialize the own-core partial phase against the stats receive. |
| `cb_xsq_sum` / `cb_rstd` (one allocation) | `block_h` | `block_h` | {rows: spans → block_h, width: collapsed, slices: one, chunks: n/a} | Float32 | compute | compute | xsq: `partial_block`. rstd: `combine` → `expand` | **Shared**: on own cores the partial phase (xsq) ends before the hub multicast that `combine` waits on. Same format and size. |
| `cb_partial` / `cb_rstd_full` (one allocation) | `block_h` | `block_h` | {rows: spans → block_h, width: collapsed (partial) / spans one tile (rstd_full), slices: one, chunks: n/a} | Float32 | compute | writer (partial) / compute (rstd_full) | partial: until `export_partial` pops it, which precedes the hub multicast. rstd_full: `expand` → `scale_block` | **Shared**: same reasoning; disjoint across the hub multicast |
| `cb_scale` | `block_h·block_w` | `block_h·block_w` (precomputed before x lands, consumed tile by tile, popped at the end) | {rows: spans → block_h, width: spans → block_w, slices: one, chunks: streams (read in chunk quanta)} | Float32 | compute | compute | `scale_block` → end | No share. It is live together with `cb_x_block` and `cb_out` through `normalize_block`, and its fp32 format differs from the bf16 CBs. |
| `cb_scaler` | 1 (constant) | 1 | {rows: n/a, width: n/a, slices: n/a, chunks: n/a}: a constant | Float16_b | reader | compute | own cores: whole call | No share: 2 KiB, live with everything in the partial phase |
| `stats_scratch` (sharded scratch, not a CB) | `G·Mt·128 B` | the whole vector once multicast; each core reads its rows | {rows: spans → all Mt rows (multicast convenience), width: n/a, slices: spans → G, chunks: n/a} | uint32 (fp32 bits) | hub multicast (+ own writer for its own row) | reader | whole call | No share: the hub multicasts into it at a fixed address, and it must exist on every op core. The bound is set by a predicate (symbol table). |

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

Compute core:
`2048·block_h·block_w` (landing)
`+ 2048·max(block_w, 2·chunk_tiles)` (γ / out)
`+ 4096·G·block_h` (partials)
`+ 2·4096·block_h` (the two shared 1-tile fp32 pairs)
`+ 4096·block_h·block_w` (scale)
`+ 2048` (scaler)
`+ G·Mt·128` (stats)

At Phase 0: 24 + 24 + 16 + 8 + 48 + 2 + 10 = **132 KiB**, of ~1.5 MB L1. The `block_h·block_w` terms scale with the block knobs. `G·Mt·128` scales with the op dimension M and is bounded by the predicate above. Nothing scales with W beyond `block_w ≤ Wl_t`.

Port core: `2·port_batch_chunks·chunk_tiles·2048 + G·Mt·128` = 96 + 10 = **106 KiB** (scales with `port_batch_chunks`, `chunk_tiles`).

Hub / receiver: `G·Mt·128` = **10 KiB**.

No budget solve, safety fraction or blocking search was introduced: the inventory fits at the coarsest block (the whole per-core assignment) by a factor of ~10.

## Data-movement budget (per device, per call; 640 × 384 → 640 × 1536, G = 4, L = 2)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| input x (own slice, 240 tiles, 480 KiB) | 1 read | Own cores load it into `landing` once. Ports read it from there (L1), not from DRAM (regime R8 rejected). | NoC: L1 → the 4 port cores, 1 slice in total. Fabric out: 2 directions × 1 slice over 2L ports. |
| remote slices (720 tiles, 1.44 MiB) | **0** | They land straight in compute-core `landing` (L1) and are normalized there (regime R2 rejected). | Fabric in: 3 slices. The busiest link direction carries 1.5 slices / L. Relays: 0.5 slice of NoC L1 → port reads. |
| γ (3 KiB) | 20 reads of 768 B per slice group = 60 KiB in total | each compute core reads its 12 sticks (deferred broadcast, regime R6) | — |
| output (960 tiles, 1.875 MiB) | 1 write | every tile is written once by the core that normalized it | — |
| partials / stats (G × 2.5 KiB) | 0 | fp32, L1 → fabric → L1 | Fabric: 5 KiB on link-0 forward, 2.5 KiB on link-0 backward. NoC: 20 × 128 B to the hub, the hub → 2 ports (7.5 KiB), and a hub multicast of 10 KiB to 80 cores. |

Totals per device:
- DRAM: 0.48 MiB read + 0.06 MiB γ + 1.875 MiB write ≈ **2.42 MiB** (roofline 2.40 MiB).
- Fabric: busiest link direction 0.72 MiB / 2 links (+5 KiB stats).
- On-chip NoC: ~1.5 MiB of L1 → port reads + 10 KiB multicast + 1.875 MiB of DRAM writes.

> Cheapest-traffic split considered: W-split by slice × rows with L1 landing and fp32 stats (R1). It is tied on bytes with the rows-only split (R5): same DRAM and fabric bytes, minus nothing. Implemented: R1. It **is** the cheapest-traffic split. The tie with R5 is broken by occupancy (80 vs 20 cores) and the per-core tail, as `op_design.md` states.

## Implementation notes (ttnn-implementer, Phase 0)

The Phase 0 kernels allocate the compute-core CBs **without** the three planned shares. Every CB still has
exactly one producer and one consumer; only the "Shares with" column changes:

| Planned share | As built | Why not (yet) |
|---------------|----------|---------------|
| `cb_gamma` / `cb_out` (one allocation, two indices) | two allocations: `cb_gamma` `block_w` pages, `cb_out` `OUT_DEPTH_CHUNKS·chunk_tiles` pages | Two CB indices on one allocation keep independent FIFO pointers. The alias is legal by the lifetimes above, but it was not needed to fit: Phase 0 compute cores use ~170 KiB of ~1.5 MB. Candidate for a later L1 refinement. |
| `cb_xsq_sum` / `cb_rstd` | two allocations of `block_h` fp32 pages | as above (+`4096·block_h` B) |
| `cb_partial` / `cb_rstd_full` | two allocations of `block_h` fp32 pages | as above (+`4096·block_h` B) |

Footprint as built, compute core: 132 KiB (planned) + `2048·max(0, 2·chunk_tiles)` + `2·4096·block_h` ≈ **164 KiB**
at Phase 0 (block 1 × 12, chunk 6); tall regime (block 3 × 12, chunk 6) ≈ **330 KiB**. Port core unchanged (106 KiB).
`cb_xsq_sum` and `cb_partials` are tagged `UnpackToDestFp32` (lossless fp32 reload through `copy_tile`; neither is
ever an FPU operand).

Placement as built: the 2L ports, 2L receivers and the hub sit on logical row 0 (ports centred), compute cores fill
rows 1.. in row-major order (rectangles: full rows + one partial row). No Ethernet-probe placement: the probe is a
second device dispatch on the first call, which the one-dispatch-per-invocation rule forbids. The data-movement
budget above is unchanged (placement moves no bytes, only NoC hop counts).
