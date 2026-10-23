# L1 Ledger: distributed_rms_norm (Phase 0, regime R1 resident-flat)

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes (from `op_design.md`):
`rows` (extent `block_h`), `w_local` (extent `block_w`), `sources` (= `G · num_w_chunks` partial vectors per tile row),
`exchange_round` (one round = one row-block of `block_h` tile rows; `num_rounds` rounds per core, refinement 1). Tile bytes: `T_x` (input dtype tile: 2048 bf16,
4096 fp32), `T_g` (gamma dtype tile), `T_32 = 4096` (Float32 tile), `T_16 = 2048` (Float16_b tile).
`fp32_dest_acc_en = True` (default) → every compute-produced intermediate is Float32. Refinement 5, `fp32_dest_acc_en = False`: `cb_xg` drops to Float16_b (`T_16`, audit 3: 16-bit DEST already rounded it); `cb_sumsq*`, `cb_partial_row`, the board, `cb_gather`, `cb_pregather`, `cb_inv_row*` keep Float32 pages in both modes (the exchanged partials stay fp32 on the wire and when summed, per the prompt; the reader/writer/port byte arithmetic and the Fabric slot layout assume one fp32 vector format, so one page format serves both modes). The solve charges `cb_xg` at `T_32` in both modes (upper bound: the block plan does not depend on the DEST mode).

## Worker cores

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_x_block` | `block_h·block_w` | `block_h·block_w` (one round's block resident: read once, used by sumsq and by scale/normalize; the reader pushes the whole capacity each round, pad pages popped unread, so every round's block starts at the FIFO base) | `{rows: spans → block_h, w_local: spans → block_w, sources: n/a (none), exchange_round: streams (one block per round, re-filled after scale_by_gamma(r) / normalize(r) pops it)}` | input dtype (reader-produced, not a DEST product) | reader | compute | load → scale_by_gamma (gamma) / normalize (no gamma) | Not shared: live across the whole exchange (it is the residency that removes the second DRAM read); concurrent with every other CB. Not aliased with `cb_xg`: scale_by_gamma reads x tile i while packing xg tile i, and the formats differ (input dtype vs Float32) |
| `cb_gamma` | `block_w` (0 when no gamma) | `block_w` (Row mapping re-reads the same tile row for each of `block_h` rows; held across all rounds, popped once at the end) | `{rows: streams (re-read per row from L1), w_local: spans → block_w, sources: n/a, exchange_round: resident (read once per call)}` | gamma dtype | reader | compute | load → scale_by_gamma | Not shared: concurrent with `cb_x_block` and `cb_xg` during scale_by_gamma; only row 0 of each page is written (slivers), the rest of the page is never read |
| `cb_scaler` | 1 (constant) | 1 | `{rows: n/a, w_local: n/a, sources: n/a, exchange_round: n/a}` — constant SUM scaler 1.0 | Float16_b (packed bf16 scaler, pool-type-aware helper) | reader | compute | whole program (both reduces) | Not shared: read by both reduces (sumsq and combine) whose lifetimes bracket every other CB |
| `cb_sumsq` | `block_h` (alias group A, `total_size = block_h·T_32`) | `block_h` (one S tile per tile row; DEST-accumulated over `w_local`) | `{rows: spans → block_h, w_local: streams (folded into DEST), sources: n/a, exchange_round: spans → rows_per_round}` | Float32 | compute | compute | sumsq (square-accumulate → transpose) | **Shares** alias group A with `cb_sumsq_t` (in-place transpose, lockstep quanta `block_h`) and `cb_inv_row_out` (disjoint lifetime) |
| `cb_sumsq_t` | `block_h` (alias group A) | `block_h` | `{rows: spans → block_h, w_local: n/a (already folded), sources: n/a, exchange_round: spans}` | Float32 | compute | compute | sumsq (transpose → REDUCE_COL) | **Shares** alias group A (same bytes as `cb_sumsq`, transform in place with real credit edges) |
| `cb_inv_row_out` | `block_h` (alias group A) | `block_h` (leader only) | `{rows: spans → block_h, w_local: n/a, sources: n/a (reduced), exchange_round: spans}` | Float32 | compute | writer | combine → fanout (leader) | **Shares** alias group A: its first reserve follows (same compute thread, program order) the REDUCE_COL that pops the last `cb_sumsq_t` page, so lifetimes are disjoint; all three ids start at base with quantum `block_h` |
| `cb_partial_row` | `block_h` | `block_h` | `{rows: spans → block_h, w_local: n/a (reduced), sources: n/a (this worker is one source), exchange_round: spans}` | Float32 (wire stays fp32) | compute | writer | sumsq → publish | Not shared with group A: produced by the REDUCE_COL while that reduce is still reading `cb_sumsq_t` (concurrent); not shared with `cb_gather`: the leader's reader zero-fills/fills `cb_gather` without any ordering edge to the writer's pop of `cb_partial_row` |
| `cb_gather` | `n_gather_tiles·block_h`, `n_gather_tiles = ceil(G·num_w_chunks/32)` | same (all sources of all tile rows needed by one REDUCE_COL call) | `{rows: spans → block_h, w_local: n/a, sources: spans → n_gather_tiles·32 rows, exchange_round: spans}` | Float32 | reader | compute | combine (leader); zero-filled at kernel start | Not shared: the reader zero-fills it at kernel start, concurrently with sumsq (no cross-kernel edge would order a reuse of group A's or `cb_partial_row`'s bytes); allocated on non-leaders too so the CB address is uniform (cost `n_gather_tiles·block_h·T_32`, 4 KB at perf focus) |
| `cb_inv_row` | `inv_depth·block_h` (alias group B, `total_size = inv_depth·block_h·T_32`; `inv_depth = 2` iff gamma and `num_rounds > 1`, else 1) | `block_h` per round; 2 rounds in flight on the gamma multi-round path | `{rows: spans → block_h, w_local: n/a, sources: n/a, exchange_round: inv_depth rounds (the leader fans round r+1 out while round r's normalize may still read cb_inv_col; fan-out target = half r % inv_depth)}` | Float32 | reader (semaphore-gated push of bytes written by the leader) | compute | fanout → normalize | **Shares** alias group B with `cb_inv_col` (transpose in place) |
| `cb_inv_col` | `block_h` (alias group B) | `block_h` | `{rows: spans → block_h, w_local: n/a (Col bcast re-reads per column), sources: n/a, exchange_round: spans}` | Float32 | compute | compute | normalize | **Shares** alias group B |
| `cb_xg` | `block_h·block_w` (0 when no gamma) | `block_h·block_w` (stall-shadow product held until inv_rms arrives) | `{rows: spans → block_h, w_local: spans → block_w, sources: n/a, exchange_round: spans → rows_per_round·block_w}` | Float32 with fp32 DEST, Float16_b with 16-bit DEST (DEST product) | compute | compute | scale_by_gamma → normalize | Not shared: concurrent with `cb_x_block` while being produced; not packed back into `cb_x_block` because that page format is the input dtype and a 16-bit page under fp32 DEST would truncate (audit 2); this is the explicit cost of the stall-shadow reorder |
| `cb_out_tiles` | `min(OUT_DEPTH, block_h·num_rounds)·block_w`, `OUT_DEPTH = 2` (refinement 3: clamped to the core's rows across rounds, not to `block_h`) | `block_w` (one block row in flight while the writer drains the previous) | `{rows: streams (window of OUT_DEPTH rows), w_local: spans → block_w, sources: n/a, exchange_round: streams}` | input dtype (the output tensor's format; the pack rounds fp32 DEST to it) | compute | writer | normalize → store | Not shared: concurrent with `cb_xg`/`cb_inv_col` during normalize; capacity exceeds live set by one block row = double buffering (mechanism reason) |

### Port cores

| Buffer | Capacity | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|--------|----------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| board scratch (persistent L1 mesh tensor, not a CB) | `num_rounds · G · slot_bytes_l`, `slot_bytes_l = rows_l · num_w_chunks · 128 B` (one disjoint region per round: no intra-call fence) | own slot + every slot that arrives through this direction | `{rows: spans → rows_l (all row-groups of link l), w_local: spans → num_w_chunks vectors, sources: spans → G slots, exchange_round: spans}` | fp32 bytes (no CB) | local workers (own slot) and the peer port over Fabric (arrived slots) | Fabric sender (relays) and leaders (NoC reads) | whole call; persistent across calls (fenced); one board per (mesh, port-core set), grown to the largest size seen and never shrunk (a size-keyed cache leaked one board per size onto the port cores until it hit a later program's CB region) | Not shared: must sit at the same address on every chip, which only a persistent lock-step mesh allocation guarantees; sized for `G` slots so every port has the same layout (a receive-only port wastes its unused slots, ≤ `BOARD_CAP_BYTES`) |

## Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `block_h` | `1 ≤ block_h ≤ tensor_h_tiles`; `num_row_groups = ceil(tensor_h_tiles / block_h)` row-blocks dealt over `num_slots = min(num_row_groups, num_workers // num_w_chunks)` row-slots | solve; `fits_l1`. Refinement 1: the per-core footprint is one block (per round), independent of `tensor_h_tiles`; one block per core (`num_rounds = 1`) stays the solve's choice whenever it fits |
| `num_rounds` | `ceil(num_row_groups / num_slots)`, `≥ MIN_ROUNDS` (host knob, default 1) | solve; `G·num_rounds·slot_bytes_l ≤ BOARD_CAP_BYTES` |
| `block_w` | `MIN_BLOCK_W (4) ≤ block_w ≤ tensor_w_tiles` (or `= tensor_w_tiles` when smaller) | solve |
| `num_w_chunks` | `1 ≤ num_w_chunks ≤ ceil(tensor_w_tiles / MIN_BLOCK_W)` and `G·slot_bytes_l ≤ BOARD_CAP_BYTES` | solve (board-cap clamp) |
| `G` | `2 ≤ G ≤ mesh_shape[cluster_axis]` (8 on a Galaxy axis 0) | mesh shape |
| `n_gather_tiles` | `ceil(G·num_w_chunks/32)` (≤ 4 for every `INPUTS` shape at G ≤ 8) | derived |
| `rows_l` | `≤ ceil(num_slots / num_links) · block_h` (per round) | derived |
| `OUT_DEPTH` | 2 | host constant |
| `L1_BUDGET` | `0.9 · (1536 − 256) KiB` = 1 179 648 B (host constant) | conservative Blackhole worker L1 minus a 256 KiB reserved/kernel-config allowance. Deliberately **not** `ttnn.get_max_worker_l1_unreserved_size()` (1 531 904 B here): that is the static HAL ceiling, not this device's allocator base, so it would over-promise (verifier, Phase 0). |

## Total per-core footprint

```
worker_bytes(block_h, block_w) =
      block_h·block_w·T_x                       # cb_x_block           (scales with block_h·block_w)
    + [gamma]  block_w·T_g                       # cb_gamma             (scales with block_w)
    + T_16                                       # cb_scaler            (constant)
    + block_h·T_32                               # alias group A        (scales with block_h)
    + block_h·T_32                               # cb_partial_row       (block_h)
    + n_gather_tiles·block_h·T_32                # cb_gather            (block_h · sources)
    + inv_depth·block_h·T_32                     # alias group B        (block_h; ×2 on the gamma multi-round path)
    + [gamma]  block_h·block_w·T_32              # cb_xg                (block_h·block_w)
    + min(OUT_DEPTH, block_h·num_rounds)·block_w·T_x  # cb_out_tiles   (block_w, depth knob)
    + [COMBINE_AHEAD] 2·block_h·T_32            # own cb_inv_row_out   (knob, default off: else alias group A)

fits_l1(block_h, block_w) := worker_bytes(block_h, block_w) ≤ L1_BUDGET

port_bytes = num_rounds · G · rows_l · num_w_chunks · 128 B ≤ BOARD_CAP_BYTES (256 KiB)
```

Perf focus (640×1792 bf16, gamma bf16, block_h=1, block_w=12, n_gather_tiles=1): 24 KB + 24 KB + 2 KB + 4 KB + 4 KB + 4 KB + 4 KB + 48 KB + 24 KB = **138 KB** per worker. Board per port (G=4, 2 links → rows_l = 10): 4 · 10 · 5 · 128 B = **25.6 KB**. Worst `INPUTS` worker (2048×2048 fp32 x and gamma: block_h=2, block_w=22, 3 chunks): x 176 KB + gamma 88 KB + xg 176 KB + out 176 KB + ≈ 30 KB stats/scaler ≈ **0.65 MB** — fits.

## Data-movement budget

Per device, perf focus 640×1792, bf16 x and gamma, G = 4 ring, 2 links, `num_w_chunks = 5`, `num_row_groups = 20`.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| x (2.29 MB) | 1 | resident `cb_x_block` serves sumsq and normalize | none |
| out (2.29 MB) | 1 | written once from `cb_out_tiles` | none |
| gamma (115 KB tensor) | `num_row_groups` × slivers = 20 · 56 · 128 B = 143 KB (≈ 1.25 tensor-equivalents, but 6 % of x) | reuse-shared across row-groups, read per worker, row 0 only | none by default (R4 mcast built in refinement 6 as the parked knob `GAMMA_MCAST`; on: the slivers cross DRAM once per W-chunk, 56 · 128 B = 7 KB, then multicast down each chunk column) |
| partials (fp32, `num_w_chunks · M · 4 B` = 12.8 KB per device) | 0 (never in DRAM) | L1 only | worker → 2 ports: 25.6 KB; Fabric: 3 slots sent per device (ring: 2 fwd incl. 1 relay, 1 bwd) = 38.4 KB, 3 received; leader board reads `G·num_w_chunks·M·4` = 51.2 KB |
| inv_rms (fp32, `M · 4 B` per row-group, fanned out) | 0 | L1 only | leader → row-group: `num_w_chunks · M · 4` = 12.8 KB |

Totals per device: DRAM 4.58 MB + 0.14 MB gamma; intra-chip NoC ≈ 0.09 MB; Fabric ≈ 38 KB sent (latency-bound: one packet per slot).

> Cheapest-traffic split considered: rows-only (`num_w_chunks = 1`, R5) — DRAM identical (4.58 MB); Fabric 3 × 2.5 KB instead of 3 × 12.8 KB (−31 KB), intra-chip −0.07 MB. Implemented: rows × W (R1) with the solve choosing `num_w_chunks`; R5 is the `num_w_chunks = 1` value of the same structure, selected whenever rows alone fill the grid. Not selected at perf focus because DRAM — the dominant tier — is identical and the extra bytes (≈ 2 % of x) buy 100 busy workers instead of 20. The cheaper-on-the-wire variant that keeps the W split, local pre-combine (R3), is `deferred`: every `INPUTS` slot is ≤ 2 Fabric packets, so the exchange is latency-bound and R3's saved bytes would not shorten it while its extra serial stage would.

## Implementation notes (ttnn-implementer, Phase 0)

Where the implementation departs from the inventory above (no CB added, removed or resized):

| Item | Design | Implemented | Why |
|------|--------|-------------|-----|
| Ragged blocks | nominal pushes, reader zero-fills pad pages | **actual extents**: `h_act`/`w_act` RT args flow into the same block operations (chain/reduce shapes, push/wait counts); CB capacities stay nominal | blocking-model ragged-tail rule ("a short last block passes a smaller runtime extent"); no pad tiles are read, computed or zeroed. Board rows beyond `h_act` are never written and never read (every chunk of a row-group has the same `h_act`) |
| `cb_out_tiles` drain | batches of `block_w` | batches of `w_act`; the writer wraps the read pointer explicitly (`w_act` need not divide the capacity) and pops a batch that straddles the FIFO end in two parts (refinement 3: `cb_pop_front` only wraps on an exact landing) | wrap-safe linear access for ragged chunks |
| `cb_gather` zero-fill | local read from the zero region | `noc.async_write_zeros` over the whole `n_gather_tiles·h_act` window, before the board reads | DM-engine fill, no RISC loop |
| Unpack mode | — | `UnpackToDestFp32` on `cb_sumsq`, `cb_sumsq_t`, `cb_gather`, `cb_inv_row` (the two transposes' and two Accurate reduces' inputs) | lossless fp32 reloads (default unpack truncates fp32 to tf32); `cb_xg` / `cb_inv_col` stay Default (FPU srcA / srcB). Only with `fp32_dest_acc_en = True`: a 16-bit DEST cannot hold an fp32 unpack, so with it off every CB is Default (Float32 pages unpack to Float16_b in srcA) and the three reduces take the FPU path (`ReduceFp32Mode::Fast`) |
| Port placement | Ethernet probe (one probe program per mesh, cached) | `simple`: 2 ports per link on logical row 0, x = 0..2L-1, identical on every chip | the probe is a second device dispatch on the first call, which the one-dispatch-per-invocation rule forbids; design's port-placement lamp lists `simple` as the alternative (payload ≤ 2 packets) |
| Board | sized per `(mesh, port set, bytes)` | same; HEIGHT_SHARDED uint32 L1 tensor over all `2·num_links` port cores, `G·slot_bytes` per core rounded to 16 B | — |
| Links with no row-group | — | `L_eff = min(num_links, num_row_groups)` links carry traffic; ports of unused links run no kernel | decode (1 row-group) at 2 links |

Data-movement budget: unchanged (x and out cross DRAM once each; gamma slivers per row-group).

## Refinement 1 — multi-block per-core schedule

| Item | Before | After | Why |
|------|--------|-------|-----|
| Rounds per core | 1 (solve refused `num_row_groups > num_workers`) | `num_rounds = ceil(num_row_groups / num_slots)`; row-block `b` → slot `b % num_slots`, round `b // num_slots` (the ragged last row-block is its slot's last round; slots past it run an empty last round that still counts in the port/leader counters) | keeps the footprint at one block for any `tensor_h_tiles` |
| `cb_x_block` push | `h_act·w_act` | whole capacity `block_h·block_w` per round, pad pages popped unread by compute | sumsq reads the block as one contiguous run from the FIFO base; a ragged chunk would otherwise leave the pointer mid-FIFO from round 2 on |
| alias group B | `block_h·T_32` | `inv_depth·block_h·T_32` (`inv_depth = 2` only on gamma + multi-round) | gamma order overlaps round r+1's exchange with round r's normalize, so round r+1's inv lands while round r's `cb_inv_col` may still be read |
| Board | `G·slot_bytes` | `num_rounds·G·slot_bytes` | disjoint region per round, no fence between rounds |
| Semaphores | port-local, board-ready, inv-ready | + port-local-odd, board-ready-odd | multi-source counters per round parity (skew ≤ 1 round); single-source ones cumulative |
| Stall-shadow-skip fallback (no `cb_xg`) | — | not built | not needed for a fit: multi-round already bounds the footprint by one block, and dropping the reorder would lengthen the post-stats critical path for shapes that fit today |

Worked fit, per device (2,1,2048,4096) bf16 + bf16 gamma, 106 workers, G = 2, 2 links: `block_h = 5`, `block_w = 16` (8 chunks), 13 slots × 2 rounds: x 160 KiB + xg 320 KiB + out 64 KiB + gamma 32 KiB + stats/scaler 102 KiB = **678 KiB** (694 272 B) per worker; board 2 rounds · 2 · 7 slots·5 rows·8·128 B = 143 KB per port.

Data-movement budget: unchanged — x and out cross DRAM once each (x resident per round); gamma slivers read once per worker per call (not per round).

## Refinement 2 — sharded input / output (R9 HEIGHT, R10 WIDTH)

| CB / item | Capacity (pages) | Change | Producer → consumer | Shares with / why not |
|-----------|------------------|--------|---------------------|-----------------------|
| `cb_x_block` (sharded x) | the core's whole shard (`sh·sw` tiles), backed on the input buffer (`cb_descriptor_from_sharded_tensor`) | no CB bytes: the shard is the tensor's own L1; a round's "load" is a push of `h·w_act` pages (rounds walk the shard from the FIFO base, no pad pages) | reader (push only, no NoC read) → compute | Not shared: it is the input tensor |
| `cb_out_tiles` (sharded out) | the core's whole output shard, backed on the output buffer | compute packs straight into the shard (`h·w_act` per round, in order); the writer stores nothing | compute → (nobody: the shard is the result) | Not shared: it is the output tensor |
| `cb_pregather` (pre-combine only) | `2·n_pre·block_h`, `n_pre = ceil(num_w_chunks/32)` | new | the row-group's writers (NoC writes of half-rows, row = chunk, half `r % 2`) + the leader's reader (zero-fills rows ≥ `num_w_chunks`, semaphore-gated push) → leader compute (REDUCE_COL, Accurate) | Not shared: written remotely while every other CB is live; two halves because with the stall shadow a writer publishes round r+1 before the leader consumes round r (round r+2 needs inv(r), so two halves suffice). Allocated on every worker so its address is uniform |
| `cb_partial_row` (pre-combine) | `block_h` | every push padded to `block_h` pages (the writer pops `block_h`) | compute → writer | the leader's stream interleaves its own partials with the combined vectors; the no-pop reduce packs its outputs linearly from the write pointer, so a short last-round push followed by a full block would straddle the FIFO end |
| `cb_gamma` (streamed) | `2·gamma_batch` (`gamma_batch ≤ GAMMA_BATCH = 8`, halved by the solve when L1 is tight) | streamed variant only | reader (slivers, re-read per tile row per round) → compute (fused normalize, PerTile) | Not shared |
| `cb_inv_full` (streamed) | alias group B | new alias id | compute (`UnaryBcast` Col, in place over `cb_inv_col`) → compute (`DestReuseBinary`) | **Shares** alias group B (lockstep quantum `h` with `cb_inv_row` / `cb_inv_col`) |
| `cb_xg` | 0 on the streamed variant | dropped there | — | the streamed variant exists because resident gamma + `cb_xg` do not fit next to wide shards |

Budget: the sharded solve does not use `L1_BUDGET`. It measures the real room (`ttnn.get_memory_view`: allocator base → lowest live L1 buffer, x and out shards already allocated) and charges any board growth, or re-sizes a stale larger board to the plan's need. Search order: per-chunk board (no pre-combine) before pre-combine; resident gamma + stall shadow before streamed gamma (batch 8, 4, 2, 1); then the largest `block_h` (fewest rounds).

Board: one board per mesh (an L1 allocation reserves its address range on every core's bank), following the current port-core set; the sharded path's ports are the last `2·num_links` grid cores.

Worked fits (G = 2, 2×2 mesh):
- HEIGHT 1×1×32×4096 fp32 + fp32 gamma: 1 core; x 512 KiB + out 512 KiB of shard. Streamed gamma, `block_h = 1`, CBs ≈ 28–82 KiB depending on `gamma_batch`.
- WIDTH 1×1×4096×1024 fp32: 32 shards of 128×1 tiles (512 KiB each for x and out). Pre-combine, board `G·R·block_h·128 B` = 32 KiB, `block_h` ≈ 4–9.

Data-movement budget (sharded): x and out cross no NoC at all; partials and inv as in R1 (with pre-combine: `num_w_chunks` half-row pairs per row to the leader, and one vector per row on the board and Fabric); gamma slivers once per worker (resident) or once per tile row per round (streamed).

## Refinement 3 — perf (640 × 1792, G = 4 ring)

| Item | Before | After | Why |
|------|--------|-------|-----|
| Rounds | one round whenever it fits (`MIN_ROUNDS = 1`) | gamma path (`AUTO_ROUNDS`): the plan with the fewest per-core tiles, then the most rounds whose per-round block is ≥ `MIN_ROUND_TILES` (5) tiles. Perf focus: 2 rounds of 1×7 tiles over 8 chunks × 10 row-slots (refinement 6: `TILE_SLACK` + fewest gather tiles + balanced row-slots; refinement 3 had 1×6 over 10 chunks, before that 1 round of 1×12, 5 chunks × 20). Gamma mcast (R4, `GAMMA_MCAST`, parked off) adds no CB: the multicast lands in `cb_gamma` itself | round r+1's load / sumsq / exchange overlap round r's normalize / store; one round resident. Perf focus per-core block 1×7 (was 1×6): +1 x / out / xg tile each, ≈ +10 KB per worker, 80 workers instead of 100 |
| `cb_out_tiles` | `min(OUT_DEPTH, block_h)·block_w` | `min(OUT_DEPTH, block_h·num_rounds)·block_w` | with one tile row per round, a one-row FIFO capped the knob at 1; measured neutral (OUT_DEPTH 1/2/4 within noise), kept so the knob means what it says |
| `cb_inv_row_out` | alias group A | unchanged by default; own `2·block_h` buffer only with `COMBINE_AHEAD` (off) | combine-ahead lets round r+2's sumsq run while round r+1's inv still waits for the fan-out, which alias group A cannot hold |
| Semaphores | ready fence (global, 0 at rest; one ready sent at call start) | ready **credit** (global, 1 at rest, consumed at call start, returned at call end) + program sems `sem_gather_done` (leaders → both ports of the link) and `sem_rx_free` (receiving port → opposite port) on the port cores | the next call's sends need no round trip to the peer |
| Board (re)allocation | allocate new, free old | host `synchronize_device` first, then allocate / free | a peer may send on the credit before this chip runs the call, so the board's range must not be in use by work still in flight |

Data-movement budget: unchanged (x and out cross DRAM once each). The fan-out is one 1088 B write per tile row per member instead of two 64 B writes; DRAM reads and stores of tile row t start at column `t % w_act` (`DRAM_STAGGER`).
