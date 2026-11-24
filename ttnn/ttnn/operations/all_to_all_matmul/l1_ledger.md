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
| `cb_a_landed` | `scratch_slot_capacity_tiles` = shard of `landing_scratch`, `≥ slots_per_core_max * block_m_tiles * Kt` (Refinement 3: a core's slots = its own heads + its tail-split helper slots, ≤ own + 1) (grow-only; equals it on the largest call seen) | `heads_this_core * block_m_tiles * Kt` — every slot of the core is resident at once: remote routers write any slot at any moment of the call | `{head: spans → heads_per_core_max slots, rows: spans → block_m_tiles, k: spans → Kt, n: streams (retained across n-blocks, not scaled by them), entry: spans → one entry per slot}` | input dtype tiles (Float16_b): a dataflow payload landed bit-exact, never through DEST | reader | compute | whole call (persistent buffer across calls; contents per call) | Cannot share: written by remote chips concurrently with every other CB's lifetime; it **is** the destination (pack-into-destination pattern for the exchange: no staging copy exists). Capacity above live set on cores with fewer heads than `heads_per_core_max` or after a grow: one shard size per sharded tensor |
| `cb_weight` | `weight_depth * Kt * block_n_tiles`, `weight_depth = min(WEIGHT_DEPTH, max over cores of the n-blocks it processes per call)` (`WEIGHT_DEPTH` = 2; verifier: capped at the blocks a core processes per call; Refinement 3: with the tail split, `block_n` halves when `num_n_blocks` was 1, so a whole-head core processes 2 half blocks) | `weight_depth * Kt * block_n_tiles` (one block multiplied while the next loads; on a 1-block core — Kimi, GLM — the one block) | `{head: streams, rows: streams (weight independent of rows), k: spans → Kt, n: spans → block_n_tiles, entry: streams}` | weight dtype (Float16_b; Bfp8_b refinement) — input operand | reader | compute | whole call | Cannot share with `cb_out` / `cb_a_landed`: concurrent (weight of block i+1 loads while block i packs out). Verifier fix: the second buffer is no longer allocated when no core has a second block (Kimi / GLM: 1 head x 1 n-block): measured neutral (fastest chip 94.47 → 94.42 us, within noise), L1 peak 368672 → 270368 B (Kimi), 458752 → 262144 B (GLM) |
| `cb_out` | `OUT_DEPTH * out_subblock_h * block_n_tiles` (`OUT_DEPTH` = 2) | `2 * out_subblock_h * block_n_tiles` | `{head: streams, rows: streams in out_subblock_h row groups, k: streams (accumulated in DEST, never stored), n: spans → block_n_tiles, entry: streams}` | output dtype Float16_b — output ABI (the final rounding of the fp32 DEST sum to the bf16 output); not an intermediate, so audit 2's under-width finding does not apply | compute | writer | whole call | Cannot share: concurrent with both inputs. Also serves as `matmul_block`'s interm placeholder (one K-block: no spill, `matmul_block_helpers.hpp:247-249`), so no partials CB exists (fold-into-DEST pattern: the K-sum never leaves DEST) |
| `cb_retile_stage` (R1' only, `Mq % 32 != 0`) | `(Mq_t + 1) * Kt` tiles (input dtype; ≤ 12 tiles = 24 KiB on the golden R1' INPUTS) | one image's source tile rows of the local quarter (`nsrc ≤ Mq_t + 1`) x Kt | `{head: streams, rows: spans → nsrc ≤ Mq_t + 1 source tile rows of one image, k: spans → Kt, n: —, entry: streams (one image at a time)}` | raw input bytes, never through DEST | reader (DRAM read) | reader (core-local NoC copies into `cb_a_landed`) | per image of the local slot (scratch; never pushed) | Cannot share with `cb_a_landed` (it is the copy destination) or `cb_weight` (weight block 0 is resident while the local slot is re-tiled). Not allocated on aligned calls. Separate descriptor per core set: one descriptor over the union of compute + port cores was placed inside `cb_port_chunks` on 2 of 4 chips (DEVICE_PRINT) |

## Port cores (2·num_links per chip)

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_port_chunks` | `2 * batch_chunks` pages of `packet_tiles * tile_bytes` B, `batch_chunks = PORT_BATCH_BYTES // packet_bytes` (Refinement 4: 87 KiB → 12 packets at 7232 B, 6 at 14 KiB; was a fixed 8: 8 beat 4 on Kimi Phase 0, and 12 beat 8 by ~1 us on the Kimi model case with the run send order) | `2 * batch_chunks` (one batch read + one batch in flight on the fabric) | `{head: streams, rows: streams, k: streams, n: — (not present on ports), entry: streams → batch_chunks packets per batch; a packed packet may hold the end of one entry and the start of the next (same destination run)}` | raw input bytes (input dtype), never through DEST | reader (port) | writer = port sender | whole call | The only CB on the port core on aligned calls (R1' adds `cb_retile_stage`). Packet headers come from the fabric header pool, not a CB |
| `cb_retile_stage` (R1' only) | `(Mq_t + 1) * Kt` tiles | one image of one entry's source tile rows x Kt | `{head: streams, rows: spans → nsrc ≤ Mq_t + 1, k: spans → Kt, n: —, entry: streams}` | raw input bytes | port reader (DRAM read, one barrier per image) | port reader (core-local NoC copies into the packet pages) | per image of an entry (scratch) | Cannot share with `cb_port_chunks`: a re-tiled packet's page is written while earlier pages are still in flight on the fabric; an in-place shift would need the next tile row of the same entry, which crosses packet / batch boundaries |

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
| `WEIGHT_DEPTH`, `OUT_DEPTH`, `PORT_BATCH_BYTES` | 2, 2, 87 KiB (→ `batch_chunks` 12 at 7232 B, 6 at 14 KiB) | host constants |
| `weight_depth` | `min(WEIGHT_DEPTH, max n-blocks per core)` ≤ 2 | host derivation |
| `slots_per_core_max` | own heads + tail-split helper slots (≤ `TAIL_SPLIT_MAX_HEADS` = 8 per chip, ≤ 1 per helper) ≤ 8 | validate (semaphore count cap) |

## Total per-core footprint

Compute core:
`L1_compute = scratch_slot_capacity_tiles*2048 + weight_depth*Kt*block_n_tiles*weight_tile_bytes + OUT_DEPTH*out_subblock_h*block_n_tiles*2048`
≤ 384 KiB (A, scales with heads_per_core and block_m) + 2 × 256 KiB (weight, scales with Kt·block_n and WEIGHT_DEPTH) + `OUT_DEPTH*out_subblock_h*block_n_tiles*2048` (scales with block_n and out_subblock_h; 128 KiB at block_n = 32, sb_h = 1, the largest over INPUTS).

| Case | A slots | weight | out | total |
|------|---------|--------|-----|-------|
| Kimi (1 head, Kt 4, Nt 16, sb 1x4; weight_depth 1) | 40 KiB | 128 KiB | 64 KiB | 232 KiB |
| DeepSeek (2 heads; weight_depth 2) | 80 KiB | 256 KiB | 64 KiB | 400 KiB |
| *Refinement 3 (tail split on; block_n halved to 8 at every perf case):* | | | | |
| Kimi (≤ 2 slots, 2 half blocks, sb 1x8 / 1x4) | 80 KiB | 128 KiB | 32 KiB | 240 KiB |
| DeepSeek (≤ 3 slots on a 2-head helper) | 120 KiB | 128 KiB | 32 KiB | 280 KiB |
| GLM (Kt 6, ≤ 2 slots) | 120 KiB | 192 KiB | 32 KiB | 344 KiB |
| GLM (Kt 6; weight_depth 1) | 60 KiB | 192 KiB | 64 KiB | 316 KiB |
| deep K (Kt 128, block_n 1) | 256 KiB | 512 KiB | 4 KiB | 772 KiB |
| wide N 8192 (block_n 32) | 8 KiB | 512 KiB | 128 KiB | 648 KiB |

Port core: `L1_port = 2*batch_chunks*packet_tiles*tile_bytes` = 170 KiB at 7232 B (12 x 7 KiB pages x 2; was 113 KiB at 8), 168 KiB at 14 KiB (6 x 14 KiB x 2; was 224 KiB at 8). Global semaphores on the port cores: one ready counter per destination chip of a port (≤ G - 1 = 3; Refinement 4, was 1).

An L1 output (Refinement 1, landed) adds its interleaved per-bank share (Kimi: 15.7 MB over the L1 banks ≈ 143 KiB) to every
core, port cores included; the allocator places it above the CB region and beside the persistent `landing_scratch`. Tightest
case DeepSeek (400 KiB CBs + 80 KiB A + ~188 KiB output share) fits and passes (golden + LOOSE_CASES).

Refinement 1 — bfloat8_b weight: `cb_weight` pages are Bfp8_b tiles (1088 B, format = `weight.dtype`); `WEIGHT_BLOCK_BYTES`
is in bytes, so `block_n` is the largest divisor of Nt within it (Kimi: block_n stays Nt = 16, the block shrinks
128 → 68 KiB; deep K 4096: block_n stays 1, Kt·2·1088 B > 256 KiB, block 256 → 136 KiB). 16-bit DEST: `cb_out` is the output ABI (bf16) either way; the subblock limit
becomes 8 tiles (host rule). No CB added, resized by formula only.

## Data-movement budget (per device, per call; chosen split R1, Kimi focus shape)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| input (3.93 MB) | 1 (read) | remote quarters read once by the ports, local quarter once by its compute cores; the landed rows stay resident in L1 for every n-block (no DRAM landing) | link: each remote quarter crosses its route once (far quarter 2 hops) → 2.95 MB leave the chip; busiest directed link 1.97 MB = input/2 (routes' first hops alternate); NoC: router → compute core landing writes 2.95 MB |
| weight (12.58 MB) | 1 (read) | each head's weight is read by the one core that owns the head, once per head (batch and rows folded into the block) | 0 |
| output (15.73 MB) | 1 (write) Phase 0 / 0 (L1 out) | written once by the core that computed it | 0 (L1 out: the write goes to the L1 banks over the NoC) |

Totals per tier — DRAM: 32.2 MB (Phase 0) / 16.5 MB (L1 out); links: 1.97 MB on the busiest directed link (2 links per direction → 0.98 MB per link); cross-core NoC: 2.95 MB landing writes + (Refinement 3) ≤ 8 x 40 KB = 0.33 MB of tail-split forwards (primary → helper, core-local). DRAM crossings unchanged: the split partitions each split head's weight columns between the two cores. Refinement 4 (TRANSPORT): same link bytes in fewer packets — destination runs of 3 entries packed into full packets (Kimi, 7232 B: 168 → 160 packets per link).

> Cheapest-traffic split considered: per tier there are two. On DRAM, R6 (compute at the source, exchange outputs): −9.4 MB of weight reads (16.5 → 7.0 MB at L1 out), at the price of 4x the link bytes (busiest link 7.9 MB vs 1.97 MB). On links and the NoC, the implemented split R1 is the cheapest. Implemented: R1, the cheapest split on the tier that binds at every perf case (the links: 194 GB/s egress per chip against DRAM's 512 GB/s). R6 is deferred: it pays only when `N < K`, and the one INPUTS shape with `N < K` is a correctness cell that R1 serves correctly. R1's structure keeps it reachable: same ports, fence, counters and landing; only the entry payload and the computing chip change.
