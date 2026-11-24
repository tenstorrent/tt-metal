# L1 Ledger: all_to_all_matmul (regime R1, Phase 0)

Schema: `.claude/references/l1-footprint-discipline.md`. Block axes (from `op_design.md` Blocking Model):
`head` (extent `block_heads` = 1), `rows` (`block_m_tiles = B*Mq_t`, batch folded in), `n` (`block_n_tiles`),
`k` (`block_k_tiles = Kt`), and the exchange axis `entry` (one (destination, head) block, carried in packets of
`packet_tiles`).

## Compute cores (C = grid − the union over the group's chips of their port cores; those owning ≥ 1 head)

Implementer note: ports are placed per chip next to their link's Ethernet core (host-only map), so their logical
cores differ between chips; the compute core set (one landing-scratch shard spec, one head→core map) is the grid
minus the union of every chip's port cores — 101 compute cores on the QuietBox 2 ring (9 port cells), not 106.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_a_landed` | `scratch_slot_capacity_tiles` = shard of `landing_scratch`, `≥ heads_per_core_max * block_m_tiles * Kt` (grow-only; equals it on the largest call seen) | `heads_this_core * block_m_tiles * Kt` — every slot of the core is resident at once: remote routers write any slot at any moment of the call | `{head: spans → heads_per_core_max slots, rows: spans → block_m_tiles, k: spans → Kt, n: streams (retained across n-blocks, not scaled by them), entry: spans → one entry per slot}` | input dtype tiles (Float16_b): a dataflow payload landed bit-exact, never through DEST | reader | compute | whole call (persistent buffer across calls; contents per call) | Cannot share: written by remote chips concurrently with every other CB's lifetime; it **is** the destination (pack-into-destination pattern for the exchange: no staging copy exists). Capacity above live set on cores with fewer heads than `heads_per_core_max` or after a grow: one shard size per sharded tensor |
| `cb_weight` | `WEIGHT_DEPTH * Kt * block_n_tiles` (`WEIGHT_DEPTH` = 2) | `2 * Kt * block_n_tiles` (one block multiplied while the next loads; on a 1-block core, the one block) | `{head: streams, rows: streams (weight independent of rows), k: spans → Kt, n: spans → block_n_tiles, entry: streams}` | weight dtype (Float16_b; Bfp8_b refinement) — input operand | reader | compute | whole call | Cannot share with `cb_out` / `cb_a_landed`: concurrent (weight of block i+1 loads while block i packs out). Over-capacity on a single-block core (Kimi: 1 head, 1 n-block → second half unused) is the depth knob's price; perf lamp |
| `cb_out` | `OUT_DEPTH * out_subblock_h * block_n_tiles` (`OUT_DEPTH` = 2) | `2 * out_subblock_h * block_n_tiles` | `{head: streams, rows: streams in out_subblock_h row groups, k: streams (accumulated in DEST, never stored), n: spans → block_n_tiles, entry: streams}` | output dtype Float16_b — output ABI (the final rounding of the fp32 DEST sum to the bf16 output); not an intermediate, so audit 2's under-width finding does not apply | compute | writer | whole call | Cannot share: concurrent with both inputs. Also serves as `matmul_block`'s interm placeholder (one K-block: no spill, `matmul_block_helpers.hpp:247-249`), so no partials CB exists (fold-into-DEST pattern: the K-sum never leaves DEST) |

## Port cores (2·num_links per chip)

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_port_chunks` | `2 * PORT_BATCH_CHUNKS` pages of `packet_tiles * tile_bytes` B (`PORT_BATCH_CHUNKS` = 8; implementer: 8 measured 94.5 vs 99.4 us at 4 on the Kimi Phase 0 case, 7232 B) | `2 * PORT_BATCH_CHUNKS` (one batch read + one batch in flight on the fabric) | `{head: streams, rows: streams, k: streams, n: — (not present on ports), entry: streams → PORT_BATCH_CHUNKS packets per batch}` | raw input bytes (input dtype), never through DEST | reader (port) | writer = port sender | whole call | The only CB on the port core; nothing to share with. Packet headers come from the fabric header pool, not a CB |

Inventory check (Rules 1 and 3): no intermediate CB — the exchange packs into the destination (`cb_a_landed` is the landing buffer itself), the K accumulation folds into DEST (one K-block), and the output packs straight into the writer's CB. The local quarter is read straight into its slot. Removing any listed CB would remove an operand or the output path.

## Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `Kt = K/32` | `Kt * weight_tile_bytes ≤ WEIGHT_BLOCK_BYTES` (256 KiB → Kt ≤ 128 bf16) | R1 predicate (else R2) |
| `block_n_tiles` | largest divisor of `Nt` with `Kt * block_n_tiles * weight_tile_bytes ≤ WEIGHT_BLOCK_BYTES` | host derivation |
| `block_m_tiles = B * ceil((S/G)/32)` | `heads_per_core_max * block_m_tiles * Kt * 2048 ≤ A_SLOT_BUDGET_BYTES` (384 KiB) | `A_RESIDENT` (else R3) |
| `heads_per_core_max = ceil(H / C)` | ≤ 8 | validate (semaphore count cap) |
| `out_subblock_h` | divides `block_m_tiles`; `out_subblock_h * out_subblock_w ≤ DEST_AUTO_LIMIT` (4 fp32 / 8 bf16) | host subblock rule |
| `packet_tiles = floor(max_payload / tile_bytes)` | 1 ≤ packet_tiles ≤ 7 (14400 B, bf16) | validate (`≥ 1`); Blackhole max payload |
| `WEIGHT_DEPTH`, `OUT_DEPTH`, `PORT_BATCH_CHUNKS` | 2, 2, 8 | host constants |

## Total per-core footprint

Compute core:
`L1_compute = scratch_slot_capacity_tiles*2048 + WEIGHT_DEPTH*Kt*block_n_tiles*weight_tile_bytes + OUT_DEPTH*out_subblock_h*block_n_tiles*2048`
≤ 384 KiB (A, scales with heads_per_core and block_m) + 2 × 256 KiB (weight, scales with Kt·block_n and WEIGHT_DEPTH) + `OUT_DEPTH*out_subblock_h*block_n_tiles*2048` (scales with block_n and out_subblock_h; 128 KiB at block_n = 32, sb_h = 1, the largest over INPUTS).

| Case | A slots | weight | out | total |
|------|---------|--------|-----|-------|
| Kimi (1 head, Kt 4, Nt 16, sb 1x4) | 40 KiB | 256 KiB | 64 KiB | 360 KiB |
| DeepSeek (2 heads) | 80 KiB | 256 KiB | 64 KiB | 400 KiB |
| GLM (Kt 6) | 60 KiB | 384 KiB | 64 KiB | 508 KiB |
| deep K (Kt 128, block_n 1) | 256 KiB | 512 KiB | 4 KiB | 772 KiB |
| wide N 8192 (block_n 32) | 8 KiB | 512 KiB | 128 KiB | 648 KiB |

Port core: `L1_port = 2*PORT_BATCH_CHUNKS*packet_tiles*tile_bytes` = 96 KiB at 7232 B, 224 KiB at 14400 B.

An L1 output (refinement) adds its interleaved per-bank share (Kimi: 15.7 MB over the L1 banks ≈ 143 KiB) to every core.

## Data-movement budget (per device, per call; chosen split R1, Kimi focus shape)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| input (3.93 MB) | 1 (read) | remote quarters read once by the ports, local quarter once by its compute cores; the landed rows stay resident in L1 for every n-block (no DRAM landing) | link: each remote quarter crosses its route once (far quarter 2 hops) → 2.95 MB leave the chip; busiest directed link 1.97 MB = input/2 (routes' first hops alternate); NoC: router → compute core landing writes 2.95 MB |
| weight (12.58 MB) | 1 (read) | each head's weight is read by the one core that owns the head, once per head (batch and rows folded into the block) | 0 |
| output (15.73 MB) | 1 (write) Phase 0 / 0 (L1 out) | written once by the core that computed it | 0 (L1 out: the write goes to the L1 banks over the NoC) |

Totals per tier — DRAM: 32.2 MB (Phase 0) / 16.5 MB (L1 out); links: 1.97 MB on the busiest directed link (2 links per direction → 0.98 MB per link); cross-core NoC: 2.95 MB landing writes.

> Cheapest-traffic split considered: per tier there are two. On DRAM, R6 (compute at the source, exchange outputs): −9.4 MB of weight reads (16.5 → 7.0 MB at L1 out), at the price of 4x the link bytes (busiest link 7.9 MB vs 1.97 MB). On links and the NoC, the implemented split R1 is the cheapest. Implemented: R1, the cheapest split on the tier that binds at every perf case (the links: 194 GB/s egress per chip against DRAM's 512 GB/s). R6 is deferred: it pays only when `N < K`, and the one INPUTS shape with `N < K` is a correctness cell that R1 serves correctly. R1's structure keeps it reachable: same ports, fence, counters and landing; only the entry payload and the computing chip change.
