# L1 ledger: a2a_wkv_b2

Schema: `.claude/references/l1-footprint-discipline.md`. Block axes (from `op_design.md`): `h`
(heads), `m` (token tile rows), `k` (latent tiles), `n` (v tiles), `j` (destination / stream).
Every CB descriptor spans all program cores (workers ∪ fabric_tx) so addresses are uniform; the
footprint below is therefore the same on every core.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_weight` | `Kt · Nt` = 64 × 1088 B | 64 (whole W[h] is needed by every mt block) | `{h: streams → 1 head, m: streams (resident across), k: spans → Kt, n: spans → Nt, j: streams}` | Bfp8_b (the DRAM format; no conversion) | worker reader | worker compute | one head, across all its mt blocks | cannot share: live for the whole head, concurrent with every other worker CB; double-buffering across heads not needed at heads_per_core = 1 |
| `cb_attn_rm` | `attn_rm_depth · mt_block · Kt` = 2 · 1 · 16 = 32 × 2048 B | 16 (one block); +16 for double buffering | `{h: streams, m: streams → mt_block, k: spans → Kt, n: n/a (input has no n) streams, j: streams}` | Float16_b (bf16 sticks, tile-sized pages) | worker reader | worker compute (tilize) | per mt block | capacity > live set by the double buffer (reader fetches block mt+1 during tilize of mt); cannot alias `cb_attn_tiled`: tilize needs distinct in/out CBs (`tilize_helpers.inl:89`) and both are live during the tilize |
| `cb_attn_tiled` | `mt_block · Kt` = 16 × 2048 B | 16 | `{h: streams, m: streams → mt_block, k: spans → Kt, n: streams, j: streams}` | Float16_b (matmul in0; bf16 is the input precision, not widened) | worker compute (tilize) | worker compute (matmul) | per mt block | not shared: live from tilize to the end of the matmul of the same block while `cb_attn_rm` already holds the next block (pipelining decision) |
| `cb_out` | `heads_per_core · Mt · Nt` = 1 · 5 · 4 = 20 × 1088 B | 20 (every block stays until fabric_tx has pulled it; no back-pressure) | `{h: spans → heads_per_core, m: spans → Mt, k: reduced (absent), n: spans → Nt, j: streams}` | Bfp8_b (output format, packed directly; 16-bit DEST) | worker compute | worker writer (CB credit); fabric_tx reads by NoC address after the head slot's block counter (GlobalSemaphore, R2) | whole program | not shared: read remotely until the fabric_tx stream ends; the alternative (back-pressure to free pages) adds a per-block NoC handshake to save 17 KB |
| `cb_tx_stage` | `tx_buffers · tx_batch · tiles_per_packet` = 2 · 8 · 4 = 64 × 1088 B | 32 (one batch of 8 packets being sent); +32 for the next batch's pull (Refinement 1) | `{h: streams, m: streams, k: absent, n: spans → tiles_per_packet = Nt, j: spans → 1 stream}` | Bfp8_b | fabric_tx reader | fabric_tx writer | whole program (fabric_tx only uses it) | capacity > live set by the double buffer (pull overlaps send); allocated on workers too only for uniform addresses (69.6 KB) |
| `cb_tx_addrs` (R3) | `max_stream_packets` = 24 · 5 = 120 pages × `tiles_per_packet · 8` = 32 B (3,840 B) | 120 (the whole stream's destination NoC addresses, filled once) | `{h: spans → stream heads, m: spans → Mt / mt_block blocks, k: absent, n: spans → tiles_per_packet, j: spans → 1 stream}` | uint32 scratch (8 B NoC address per tile slot; not a FIFO, never pushed) | fabric_tx writer (fills) | fabric_tx writer (reads per packet) | whole program | not shared: read on every packet until the stream ends while `cb_tx_stage` / `cb_tx_meta` stream; recomputing instead costs ~0.075 us per packet on the send path (R3 ablation); on workers only for uniform addresses |
| `cb_tx_meta` (R3) | `tx_buffers` = 2 pages × `align16((1 + tx_batch) · 4)` = 48 B | 1 page per batch in flight (paired with its `cb_tx_stage` batch region) | `{h, m: per packet code in the batch, j: 1 stream}` | uint32 `[k, code_0 .. code_{k-1}]`, code = (mt block, head, packet) | fabric_tx reader | fabric_tx writer | per batch | not merged into `cb_tx_stage`: the stage slots hold exactly the scatter payload sent from L1; the page count and depth follow `tx_buffers` so meta and stage stay in lockstep |

Symbol table:

| Symbol | Value here | Bound | Predicate that establishes it |
|---|---|---|---|
| `Kt` | 16 | ceil(L/32), L = 512 | shape contract (validate: L = 512) |
| `Nt` | 4 | ceil(V/32), V = 128 | shape contract |
| `Mt` | 5 | ceil(T/32), T = 160 | shape contract; T % 32 == 0 |
| `heads_per_core` | 1 | ceil(H / num_workers) | num_workers = min(H, 110, grid_cores − num_streams) = 96 on a 120-core grid; ≤ 2 for any grid ≥ 52 cores |
| `mt_block` | 1 | 1 ≤ mt_block ≤ Mt, mt_block · Nt ≤ 8 (DEST, half-sync bf16) | host constant |
| `attn_rm_depth`, `tx_batch`, `tx_buffers` | 2, 8, 2 | ≥ 1 | host constants (tx_batch 4–8 measured tied; 12/24 slower) |
| `max_stream_packets` | 120 | max stream heads · Σ_mb ceil(block tiles / tiles_per_packet) | stream plan (`_ring_streams`); fewer with `MUX_OVERFLOW_HEADS` > 0 |
| `tiles_per_packet` | 4 | min(Nt, 4, floor(max_payload / 1088)) | fabric scatter cap, `get_tt_fabric_max_payload_size_bytes()` |

Total per core (all CBs on all cores):

`Kt·Nt·1088 + attn_rm_depth·mt_block·Kt·2048 + mt_block·Kt·2048 + heads_per_core·Mt·Nt·1088 + tx_buffers·tx_batch·tiles_per_packet·1088 + max_stream_packets·tiles_per_packet·8 + tx_buffers·48`
= 69,632 + 65,536 + 32,768 + 21,760 + 69,632 + 3,840 + 96 = **263,264 B** (of 1,461,120 B allocatable). `mt_block`
scales the two attn terms; `heads_per_core` scales `cb_out`; `tx_buffers · tx_batch` scales the stage. The L1
interleaved output (1920 × 1088 B = 2,088,960 B per chip) is spread over the L1 banks (~17 KB per bank)
outside the CB region.

## Data-movement budget

Per chip, chosen split (heads over 96 worker cores, matmul first, 4 fabric streams).

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| attn (15,728,640 B) | 1 | each head's 160 sticks read once by its one worker; tilized in L1 | none |
| W (6,684,672 B) | 1 | each head's W read once by its one worker and kept resident across the head's 5 mt blocks; all 96 heads needed because the matmul runs before the exchange | none |
| out (2,088,960 B per chip, L1) | 0 (L1 tensor) | local 24 heads written by their workers over the NoC (522,240 B); 72 remote heads pulled by fabric_tx (1,566,720 B over the NoC) and sent | fabric: 3 chunks × 522,240 B leave each chip; byte-hops 2,088,960 (antipode 2 hops); each of the 4 link directions carries 522,240 B |

Totals per chip: DRAM 22,413,312 B; on-chip NoC 1,566,720 B pulled + 522,240 B local writes + 1,566,720 B
landing from the fabric; Ethernet 2,088,960 byte-hops (522,240 B per link direction).

> Cheapest-traffic split considered: heads over cores, matmul first (this one) — on links 7.5× fewer
> bytes than transfer-first (7,864,320 vs 1,044,480 B per direction), at +5.0 MB DRAM (all 96 heads of W
> instead of 24). Implemented: heads over cores, matmul first. It is also the cheapest on the binding
> resource once both tiers are priced (DRAM 43.8 us vs fabric 78.6 us for transfer-first), so nothing is
> deferred on traffic grounds.

**Refinement 3 (R6 mux, parked at `MUX_OVERFLOW_HEADS = 0`).** When enabled, each mux core carries the fabric
mux's fixed-address L1 map at the allocator base (`ttnn.experimental.fabric_mux.Config`, 2 channels ×
`MUX_NUM_BUFFERS` = 8 buffers × the fabric channel buffer size). Mux cores are outside every CB core range, so the
map does not overlap a CB. Data-movement budget: unchanged (same bytes; the overflow heads cross link 1
instead of link 0).
