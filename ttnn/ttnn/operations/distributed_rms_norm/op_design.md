# Operation Design: distributed_rms_norm

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (compute + intra-chip NoC combine + inter-chip Fabric 2D exchange) — CCL |
| Goal | One device program per call that computes RMSNorm over a hidden dim sharded across one mesh axis: local sum of squares, a Fabric exchange of the fp32 per-row partials inside each collective group, the cross-device combine, and the normalize. It replaces today's three programs (pre → all_gather → post). |
| Math | `ss[r] = Σ_{g∈group} Σ_w x[g][r,w]²` ; `out[p][r,w] = x[p][r,w] · rsqrt(ss[r]/W + eps) · gamma[p][w]` with `W = G·W_local`; the gamma factor is dropped when gamma is absent. |
| Mode | Derivative (the transport is adapted from `fabric_all_gather`; the compute is new) |
| References | `ttnn/ttnn/operations/examples/fabric_all_gather/` (plan(), port placement, ready fence, arrival counters, sender kernel); `ttnn/ttnn/operations/examples/fabric_reduce_scatter/` (arrival counters, fence); `references/blackhole-fabric.md`; `references/precision_convention.md`; `.claude/references/blocking-model.md`; `.claude/references/l1-footprint-discipline.md`; catalog `ttnn/ttnn/operations/examples/master.md` (entries: `width_split`, `tensix_all_reduce`, `row_reduce_accumulate`, `eltwise_l1_vs_dest_accumulate`, `compute_fusion`, `compute_block_size`, `double_buffer`, `noc_placement`, `mcast_topology`); `eval/golden_tests/distributed_rms_norm/{feature_spec,helpers,roofline}.py` |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| input_tensor | ttnn.Tensor (mesh) | yes | TILE, rank ≥ 2, identical per-device shape/dtype/placement | — | addresses RT |
| cluster_axis | int (keyword) | yes | 0 or 1; mesh size along it ≥ 2 | — | host only (selects groups) |
| gamma | ttnn.Tensor \| None | no | TILE `(1,1,1,W_local)` (Phase 0); ROW_MAJOR `(1,1,ceil(W_local/32),32)` (refinement); dtype float32/bfloat16 | None | presence/dtype CT, address RT |
| epsilon | float | no | > 0 | 1e-6 | RT (fp32 bits) to the leader compute |
| topology | ttnn.Topology (`from ttnn.operations.ccl import Topology`) | no | Linear, Ring | Linear | host only (port schedule) |
| num_links | int \| None | no | 1 ≤ n ≤ min over the group's hops of `len(ttnn.get_forwarding_link_indices(src,dst))`; None → that minimum | None | host only (#ports) |
| compute_kernel_config | ttnn.ComputeConfigDescriptor \| None | no | None → `default_compute_kernel_config()` = `(math_fidelity=HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)` | None | compute config |
| memory_config | ttnn.MemoryConfig \| None | no | Phase 0: interleaved (DRAM or L1) | DRAM interleaved | host only |

Canonicalization before `validate()` reads axes: `num_links=None` is resolved to the discovered count; `cluster_axis` has no sign convention (only 0/1 exist); `gamma_dtype`/`gamma_layout` read `"none"` when gamma is absent. `default_compute_kernel_config()` is exported from the op package and is the only place the default lives (the golden `axes.classify_call` reads it).

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | per device `(..., M_h, W_local)`, rank 2–4; rows = `prod(leading) · H` |
| Dtype | float32, bfloat16 (Phase 0); bfloat8_b (refinement) |
| Layout | TILE |
| Memory | DRAM interleaved (Phase 0); L1 HEIGHT_SHARDED / WIDTH_SHARDED (refinements) |

### Output

| Property | Value |
|----------|-------|
| Shape | identical to the input's per-device shape |
| Dtype | input dtype |
| Layout | TILE |
| Memory | `memory_config` (default DRAM interleaved); sharded output follows a sharded input (refinement) |

### gamma (optional)

| Property | Requirement |
|----------|-------------|
| Shape | TILE `(1,1,1,W_local)` → one tile row, `tensor_w_tiles` tiles, only row 0 meaningful |
| Dtype | float32 or bfloat16, may differ from the input |
| Memory | DRAM interleaved |

## Blocking Model

Semantics: `.claude/references/blocking-model.md`. All names below are scope-qualified (`tensor_*`, `core_*`, `block_*`).

Derived tensor extents (alignment-aware, per image, from day 1):

| Symbol | Definition |
|--------|-----------|
| `tensor_h_tiles` | `prod(shape[:-2]) · ceil(H / 32)` — tile rows of all images, images tile-padded independently |
| `tensor_w_tiles` | `ceil(W_local / 32)` |
| `G` | group size = `mesh_shape[cluster_axis]` |
| `W_global` | `G · W_local` (true width — never a padded width) |
| `num_workers` | `|allowed_cores| − |union over chips of port cores|` (identical worker set on every chip) |

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `device_group` (the G devices along `cluster_axis`) | **dependent** — `ss[r]` sums over every device's slice | none per device (the caller fixes one slice per device) | 1 slice / device | mesh shape | one device per group position; combined by the Fabric exchange of fp32 partials | — (fixed by the contract) |
| `other_mesh_axis` (groups) | **independent** — groups never exchange | — | whole | mesh shape | every group runs concurrently, own ports | — |
| `rows` (`tensor_h_tiles`, leading dims folded) | **independent** — rows normalize in isolation | `block_h` (tile rows per row-group) | host work-split solve (see Work Distribution); perf focus 640×1792: 1 | host `plan_blocks()` → per-core CT/RT `block_h` | row-groups `rg = 0..num_row_groups−1` spread over the worker grid, `num_row_groups = ceil(tensor_h_tiles / block_h)` | knob-turn |
| `w_local` (`tensor_w_tiles`) | **dependent** — `ss[r]` spans it | `block_w` (tiles per W-chunk); `num_w_chunks = ceil(tensor_w_tiles / block_w)` | host solve; perf focus: 12 (5 chunks) | host `plan_blocks()` | the `num_w_chunks` cores of a row-group each own one chunk; the combine (gather on the row-group leader) is **built in Phase 0** | knob-turn (combine exists) |
| `gamma` operand over `rows` | **reuse-shared** — gamma does not vary along `rows`, the axis split across cores | `block_w` (same chunk) | read per worker (row-0 slivers only) | — | every worker reads its chunk's slivers | scheme-change (mcast per W-chunk column) — `deferred` regime row |
| `exchange_round` (rows of a block exchanged together) | **dependent** coupling of `rows` to the exchange (each round pays one exchange latency) | `rows_per_round` (tile rows exchanged per round); `num_blocks_this_core = ceil(block_h / rows_per_round)` | `rows_per_round = block_h` → 1 round | host constant `ROWS_PER_ROUND=None` (=block_h) | same as rows | knob-turn (overlap lamp) |
| stats intermediate `partial` (`block_h × num_w_chunks` 32-float vectors per device) | **dependent** on `w_local` and `device_group` | one vector = 32 fp32 = 128 B per (tile row, chunk) | — | derived | produced by every worker, gathered by its row-group leader | — |
| stats intermediate `ss / inv_rms` (`block_h` 32-float vectors per row-group) | **independent** over rows | `block_h` | — | derived | **computed on each row-group's leader in parallel** (num_row_groups leaders), not on a hub core; fanned out to the row-group | — |

### Knob table (one row per axis, no blanks)

| Axis | Character | Block factor (Phase 0) | Core assignment | Buffer depth | Why |
|------|-----------|------------------------|-----------------|--------------|-----|
| device_group | dependent | whole slice per device | one device per position | — (Fabric: one slot per source, `G` slots resident) | contract fixes the slice; exchange is the combine |
| other_mesh_axis | independent | whole | concurrent groups | — | no coupling |
| rows | independent | `block_h` from the solve (coarsest that keeps the grid full) | row-groups over the worker grid, contiguous row-major core order | x resident (depth = whole block) | occupancy first, then coarsest block (`width_split`, `compute_block_size`) |
| w_local | dependent | `block_w` from the solve, `≥ MIN_BLOCK_W = 4` tiles | `num_w_chunks` cores per row-group | x resident | fills the grid when rows alone cannot (decode: 1 tile row) |
| gamma over rows | reuse-shared | `block_w` slivers | each worker | whole chunk resident | slivers make the re-read 2×64 B per tile (6 % of x at perf focus) |
| exchange_round | dependent | `rows_per_round = block_h` (1 round) | — | — | one exchange latency per call; overlap lamp measures 2 |
| partial (intermediate) | dependent | one 128 B vector per (tile row, chunk) | produced on every worker, gathered on the leader | board slot per source (`G` slots) | wire stays fp32 (spec) and compact |
| ss/inv_rms (intermediate) | independent over rows | `block_h` tiles per leader | **one leader per row-group** — the work is spread over all `num_row_groups` leaders; ports never compute | `block_h` | prevents a hub core from computing while the grid waits |

Every knob is a parameter: `block_h`, `block_w`, `num_w_chunks`, `num_row_groups`, `rows_per_round`, the worker grid, `num_links`, and the out-buffer depth are host values threaded into CT/RT args; no kernel inlines a tile count. The grid is passed as the worker list (RT args per core), never baked.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_x_block` | none — resident | whole block (`block_h·block_w`) | x read once: sum-of-squares and normalize share one DRAM read |
| `cb_gamma` | none — resident | `block_w` | gamma re-read per output row avoided (Row mapping re-reads from L1) |
| `cb_xg` | none — resident | whole block | stall-shadow product kept until stats arrive |
| `cb_out_tiles` | `OUT_DEPTH` (in units of `block_w` tiles) | 2 (clamped to `block_h`) | writer drains one block row while compute packs the next |
| stats CBs | none — single block | `block_h` (·`n_gather_tiles`) | single pass |
| Fabric packet headers | `HEADER_RING` | 4 (as `fabric_all_gather` `group`) | packets in flight without flush |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Fabric max packet payload (`ttnn.get_tt_fabric_max_payload_size_bytes()`, 14 336 B on the golden config) | bytes per packet of a stats slot | a slot is sent as `ceil(slot_bytes / payload)` packets; arrival increment on each slot's last packet | router rejects / truncates an oversize packet |
| Board scratch per port core (`BOARD_CAP_BYTES = 256 KiB`) | `G · slot_bytes_l` with `slot_bytes_l = rows_l · num_w_chunks · 128 B` (`rows_l` = nominal tile rows owned by link l) | the solve rejects `(block_h, num_w_chunks)` pairs that exceed it (lower `num_w_chunks` first) | L1 overflow on the port core / scratch allocation failure |
| Gather tile = 32 rows | sources per gather tile | `n_gather_tiles = ceil(G · num_w_chunks / 32)` tiles per tile row; rows beyond `G·num_w_chunks` zero-filled | garbage rows enter the REDUCE_COL sum → wrong `ss` |
| DEST capacity (`DEST_AUTO_LIMIT`, 4 tiles at fp32 half-sync, `dest_helpers.hpp:103`) | none on the block: every chain uses slot `D0`; reduce owns its own DEST walk | — | static_assert in the chain |
| NoC write granularity (L1 16 B, DRAM 64 B alignment on Blackhole) | stats half-row = 16 fp32 = 64 B; gamma sliver = 64 B at tile offsets 0 and `face_bytes` (512 B bf16 / 1024 B fp32) | all addresses are multiples of 64 B by construction | misaligned DRAM read returns wrong bytes |
| One producer per router channel | ports per (direction, link) | exactly one port core per (direction, link) | router channel corruption / hang |
| Residency (L1 working set of a worker) | `block_h · block_w` | the solve's L1 predicate (see `l1_ledger.md`); if no feasible pair exists the op raises `RuntimeError("working set exceeds L1")` (streaming regime is `deferred`) | CB allocation clash |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum (boundary: DRAM per device, Fabric per hop) | What a bigger block buys |
|--------|--------|-----------|-------|------------------------------------------------------------------------|--------------------------|
| **R1 resident-flat** — x resident per worker per round (row-block `b` → row-slot `b % num_slots`, round `b // num_slots`; one board region and one exchange per round); every worker sends its compact fp32 partial to the ports; ports exchange all partials (Ring: both directions, ≤ floor(G/2) hops; Linear: toward both ends); each row-group leader sums `G·num_w_chunks` partials, computes inv_rms, fans it out to its row-group | **built** | `fits_l1(block_h, block_w, num_rounds)` for the solved triple; multi-round keeps the footprint at one block, so a feasible triple exists for any `tensor_h_tiles` (e.g. per device (2,1,2048,4096) bf16 → 2 rounds of 5×16 tiles on 106 workers) | `block_h × block_w` tiles per worker per round; `num_rounds = ceil(num_row_groups / num_slots)` rounds per core (1 whenever one block per core fits L1 — refinement 1) | **minimum**: x crosses DRAM once, out once. Gamma: `num_row_groups·tensor_w_tiles·128 B` (slivers; 143 KB at perf focus = 6 % of x). Fabric: each device's slot (`num_w_chunks·rows·4 B`) crosses each hop once per direction it travels | per worker one set of chain/reduce inits, one CB handshake per phase, one exchange round (fixed ≈ Fabric latency + 2 local NoC hops) — a larger block spreads these over more tiles; extent is otherwise bytes-neutral |
| **R2 streaming two-pass** — x read twice (sum of squares pass, normalize pass) for working sets that do not fit L1 | **deferred** — no TARGET `INPUTS` shape violates R1's predicate (worst case: a 44-tile x block, 176 KB at fp32, total worker footprint ≈ 0.65 MB — see `l1_ledger.md`), so it would serve no exercised shape; reachable because R1's per-core schedule already separates `load_block`/`sumsq_block` from `normalize_block`: R2 only replaces the resident `cb_x_block` with a streaming CB and re-runs `load_block` before `normalize_block` | `!fits_l1` | `block_h × block_w_stream` | x crosses DRAM 2× (above minimum by one read) | — |
| **R3 local pre-combine before the exchange** — the leader sums its row-group's `num_w_chunks` partials first, ports exchange `rows·4 B` per device instead of `num_w_chunks·rows·4 B` | **built for the board cap** (refinement 2: selected by the sharded solve when the per-chunk board would exceed `BOARD_CAP_BYTES`, host knob `PRE_COMBINE`; workers write their partial half-rows into the leader's `cb_pregather` (row = chunk, two round halves, per-parity `sem_pre_ready`), the leader reduces them into a second `cb_partial_row` push and publishes that). As a perf lever on interleaved shapes still **deferred** — refinement 6 measured `PRE_COMBINE=True` at 640×1792: 18.8 → 26–27 µs. Every `INPUTS` slot is ≤ 2 packets (largest 16 KB) so the Fabric leg is latency-bound and the saved bytes (perf focus: (5−1)·2.5 KB per hop) do not shorten it, while R3 inserts a serial local gather + reduce before the first send. Reachable: same board layout with `num_w_chunks` replaced by 1 on the wire; the leader's combine already exists | slot bytes > `BOARD_CAP_BYTES / G` | as R1 | Fabric bytes ÷ `num_w_chunks`, board reads ÷ `num_w_chunks` | — |
| **R4 gamma broadcast** — one worker per W-chunk column reads the gamma slivers once and multicasts them down its chunk column | **built as a parked knob (refinement 6, `GAMMA_MCAST = False`)**: a `McastFamily` of one group per W-chunk column (Counter data-ready, no handshake: one block per call into an empty `cb_gamma`), the sender fills its `cb_gamma` (sliver reads issued ahead of its `x(0)` reads) and multicasts the pages; placement goes 2-D when on (worker (rg, wc) on core (1+wc, rg), ports in column 0, so a chunk column is one rectangle) and applies only if the plan fits that box. Measured 640×1792 ring 2-link: ≈ +0.15 µs (flat to 1 % slower) vs per-worker slivers; 2048×1280 / 2048×2048 within noise. A row-major placement needed chunk-major groups (≤ 3 rectangles each), which spread row-groups and cost +0.6 µs; a single all-gamma sender serializes 112 sliver reads (+2 µs). Original note: saves ≤ `(num_row_groups−1)·tensor_w_tiles·128 B` (≤ 137 KB at perf focus, ~6 % of x) but puts a multicast rendezvous ahead of the stall-shadow multiply; the sliver read already removes 31/32 of the naive re-read. Reachable: gamma lands in `cb_gamma` at the same address on every worker, so a `mcast_pipe` sender per chunk column is a reader-only change | `num_row_groups ≥ 2` and gamma present | as R1 | gamma DRAM bytes ÷ `num_row_groups` | — |
| **R5 rows-only split** (today's pre/post split, whole `tensor_w_tiles` per core) | **built as the `num_w_chunks = 1` point of R1** — the solve selects it whenever rows alone fill the grid; never a separate code path | solve returns `num_w_chunks = 1` | `block_h × tensor_w_tiles` | minimum; least stats traffic | — |
| **R6 hub combine** — port core (or one hub) sums the whole board and broadcasts inv_rms | **rejected** — superseded by R1's per-row-group leaders: R6 computes `tensor_h_tiles` combines serially on one core while every worker waits | — | — | same bytes | — |
| **R7 every worker reads the board itself** (no leader / fan-out) | **rejected** — superseded by R1: it multiplies port-core L1 reads by `num_w_chunks` (decode: 14× → ~200 KB from two port cores on the critical path) | — | — | board reads × `num_w_chunks` | — |
| **R8 tiles on the wire** (send each partial as a 4 KB fp32 tile) | **rejected** — 32× the bytes of the compact 128 B vector (perf focus 80 KB per slot vs 2.5 KB·chunks); superseded by the compact half-row wire format | — | — | Fabric bytes × 32 | — |
| **R9 HEIGHT_SHARDED input** | **built** (refinement 2) — worker k = shard k (shard order, ROW_MAJOR): one row-slot per shard, `num_w_chunks = 1`, rounds walk the shard's tile rows contiguously (`round_stride = block_h`, per-core `[row0, row_end)`); `cb_x_block` = `ttnn.cb_descriptor_from_sharded_tensor` (zero-copy), a sharded output backs `cb_out_tiles` (compute packs into the shard, the writer stores nothing). Ports on the last `2·num_links` grid cores (past the row-major shard grid). Wide shards where resident gamma + the fp32 stall shadow do not fit L1 (4096/8192-wide rows) take the **streamed-gamma** variant: no `cb_xg`, gamma slivers re-read per tile row in `gamma_batch` batches, normalize fused as `(x·gamma[Row]) · inv_full` (`UnaryBcast` Col expands inv; `DestReuseBinary`) | input HEIGHT_SHARDED | `block_h ≤` shard tile rows (largest that fits next to the resident shards) × `tensor_w_tiles` | x and out never cross the NoC; gamma slivers once per worker (resident) or once per tile row per round (streamed) | — |
| **R10 WIDTH_SHARDED input** | **built** (refinement 2) — worker k = shard k = W-chunk k (`num_w_chunks` = shards, one row-slot), rounds walk all tile rows; a padded last shard (`Split.PAD`) computes its whole shard width with the pad columns zero-filled once by the reader. When the per-chunk board (`G·tensor_h_tiles·num_w_chunks·128 B`) exceeds `BOARD_CAP_BYTES` (tall shapes: 640×1792, 2048×1280, 2048×2048, 4096×1024) the solve switches on **R3 local pre-combine** | input WIDTH_SHARDED | `block_h ≤ tensor_h_tiles` (largest that fits) × shard width | x and out never cross the NoC; with pre-combine the board/Fabric carry one vector per row instead of `num_w_chunks` | — |

Selection predicate: R1 always. Refinement 3: on the gamma (stall-shadow) path `AUTO_ROUNDS` searches every `(block_h, num_w_chunks)` first, keyed (refinement 6) over the candidates within `TILE_SLACK` (25 %) of the fewest per-core tiles by (per-round block ≥ `MIN_ROUND_TILES` = 5, fewest gather tiles `ceil(G·num_w_chunks/32)`, several rounds over one, fewest tiles, most rounds, chunks, `block_h`); every multi-round candidate uses balanced row-slots `ceil(num_row_groups / num_rounds)`. The perf focus solves to 2 rounds of 1×7 over 8 chunks × 10 row-slots (G·C = 32: one gather tile; was 1×6 over 10 chunks, two gather tiles). Without gamma rounds run back to back (no overlap), so the plain solve stays. Otherwise the solve first runs the Phase 0 one-round search (`num_row_groups ≤ num_workers`, unchanged key) and takes it whenever any pair fits; only if none does (or `MIN_ROUNDS > 1`) does it search `(block_h, num_w_chunks)` with several row-blocks per core, keyed by (per-core tiles = `num_rounds·block_h·block_w`, `num_rounds`, chunks, `block_h`). The stall-shadow-skipping fallback (normalize as one `x·inv·gamma` pass, no `cb_xg`) was not needed: multi-round already bounds the footprint by one block, so it is not built (no regime row). If `fits_l1` fails for every candidate the op raises `RuntimeError`. Pinned coverage of the solve's three structural cases (acceptance test): `(1,1,32,1792)` → `num_row_groups = 1`, 14 chunks; `(1,1,640,1792)` → 20 row-groups × 5 chunks; `(1,1,32,32)` → `tensor_w_tiles = 1` forces `num_w_chunks = 1` on any grid (R5 point).

### Traffic ranking

Per device, perf focus 640×1792 bf16, G = 4 ring (numbers are structural byte counts, not times).

| Candidate split | DRAM | Fabric (per device, sent) | Intra-chip NoC | Rank |
|-----------------|------|---------------------------|----------------|------|
| rows-only (R5, 20 workers) | x 2.29 MB once, out once, gamma whole-tile per core avoided by slivers | 3 slots × 2.5 KB | partials 2×2.5 KB, board reads 10 KB | 1 (least bytes) — but leaves 90 of 110 cores idle (occupancy, not ranked here) |
| rows × W (R1, 20 × 5) | identical DRAM; gamma slivers 143 KB | 3 slots × 12.8 KB | partial writes 25.6 KB, board reads 51 KB, inv fan-out 12.8 KB | 2 — DRAM identical, extra bytes ≤ 0.1 MB (4 % of x) |
| rows × W with local pre-combine (R3) | identical | 3 × 2.5 KB | gather 12.8 KB + board 10 KB + fan-out | between 1 and 2 — deferred (latency, see row) |
| streaming two-pass (R2) | x read 2× (+2.29 MB) | as R1 | as R1 | last |

Chosen: R1 with the solve choosing `num_w_chunks`. DRAM crossings are identical across R1/R3/R5 (the dominant tier), so the tiebreaker is occupancy-then-coarsest-block, which R1's solve implements; R5 is reached by the same structure whenever rows fill the grid.

### Block schedule

Logical schedule per worker (reader / compute / writer realize their parts asynchronously). `num_blocks_this_core = num_rounds` (host knob `MIN_ROUNDS` forces ≥ N rounds; default 1 = one block per core whenever it fits). Realized order (refinement 1): **with gamma** x(r) is consumed by the stall shadow, so round r+1's `load_block`/`sumsq_block`/`publish_partials` run before round r's combine/normalize — round r+1's exchange overlaps round r's normalize/store (this is why `cb_inv_row`/`cb_inv_col` hold 2 rounds when `num_rounds > 1`); **without gamma** x(r) is held until `normalize_block(r)`, so rounds run back to back. Multi-source counters (port-local, board-ready) are per round parity (a worker is at most one round ahead: publish(r+2) needs inv(r)); single-source counters (arrival, inv-ready) are cumulative; all are re-armed at the end of the call.

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_block(block_idx);            // reader: x block + gamma slivers (+ zero-fill ragged pads)
    sumsq_block(block_idx);           // compute: S_rt = Σ_w x², transpose, REDUCE_COL → partial row vectors
    publish_partials(block_idx);      // writer: 2×64 B per tile row → own slot of each port of this link
    exchange_partials(block_idx);     // ports: Fabric send own slot + relays; arrivals; signal leaders
    scale_by_gamma_block(block_idx);  // compute (stall shadow, during the exchange): xg = x ⊙ gamma
    combine_block(block_idx);         // leader only: gather G·chunks vectors → REDUCE_COL → ×1/W, +eps, rsqrt
    fanout_inv_rms(block_idx);        // leader writer → every worker of the row-group (incl. itself)
    normalize_block(block_idx);       // compute: transpose inv row→col; out = xg ⊙ inv_rms (Col bcast)
    store_block(block_idx);           // writer: out tiles → DRAM
}
```

| Block operation | Block shape | Resident across it | Intended frequency of inits / sync |
|-----------------|-------------|--------------------|------------------------------------|
| `load_block` | `block_h × block_w` x tiles, `block_w` gamma slivers | — | reads issued in batches with one barrier per batch (`double_buffer`); one push of the whole block; pads zero-filled once |
| `sumsq_block` | `block_h × block_w` → `block_h` S tiles → `block_h` partial row tiles | `cb_x_block` (not popped) | one chain init for the square-accumulate, one transpose init, one reduce init — per block |
| `publish_partials` | `block_h` row tiles × 2 half-rows × (2 ports) | — | one flush + one semaphore increment per port per block |
| `exchange_partials` | one slot per source (`rows_l·num_w_chunks·128 B`) | board scratch on the ports | one ready fence per port per call; one arrival increment per slot (every 8th packet + last); one board-ready increment per leader per port |
| `scale_by_gamma_block` | `block_h × block_w` | `cb_gamma`, writes `cb_xg` | one chain init per block; runs while the leader waits on the board |
| `combine_block` | `n_gather_tiles × block_h` gather tiles → `block_h` inv row tiles | — | gather zero-fill once; `G · block_h · 2 · ceil-bands` reads; one reduce init + one SFPU post-op init per block |
| `fanout_inv_rms` | `block_h` row tiles × one 1088 B write per member rectangle (multicast; ordered unicast for a 1-core rectangle) | — | multicast SET of `sem_inv_ready` to r + 1 behind the data (no barrier), one flush (Perf 2) |
| `normalize_block` | `block_h` inv tiles; `block_h × block_w` multiply | `cb_xg` (or `cb_x_block`), `cb_inv_col` | one transpose init, one chain init per block |
| `store_block` | `block_h × block_w` out tiles | — | writes in batches of `block_w` with one barrier per batch |

**Stall-shadow reorder (design decision).** The leader's wait for the board (the Fabric exchange) idles every worker. `x ⊙ gamma` does not depend on the stats and multiplication commutes and associates, so `out = (x ⊙ gamma) ⊙ inv_rms` is computed with the gamma product lifted *before* the wait. It is packed to an fp32 CB (`cb_xg`), so the reorder adds no bf16 rounding versus `(x ⊙ inv) ⊙ gamma` with an fp32 intermediate; the only floating-point change is fp32 product association (≤ 1 ulp fp32). The post-stats critical path is then transpose + one multiply pass. With no gamma, `normalize_block` reads `cb_x_block` directly and `cb_xg` is not allocated. Nothing else is independent of the wait: the store needs inv_rms.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Overlap (one exchange round per call) | With one block per core the DRAM write of `out` starts only after the exchange; the read→exchange→write chain is serial | **taken (refinement 3)**: `AUTO_ROUNDS` on the gamma path — 640×1792 21.7 → 19.8 µs, 2048×1280 47 → 38 µs |
| Grid synchronization (`num_w_chunks`) | Decode (1 tile row) solves to 14 chunks of 4 tiles; the leader's board read grows with chunks and the fan-out is 14 unicast rounds | `MIN_BLOCK_W = 8` (7 chunks) vs 4 (14) at `(1,1,32,1792)`; also fan-out via `mcast_pipe` when a row-group is a contiguous grid-row segment |
| Board-ready granularity | Leaders wait for the whole board; the latest source decides | leader combines sources as each port's arrival counter covers them (per-source waits). Refinement 3 took the fence off the critical path instead (ready credit: decode 10.4 → 8.2 µs); a non-blocking port scheduler (own(r+1) ahead of relay(r)) measured flat and was not kept |
| Square-accumulate realization | DEST-accumulated `Mul` (`DestAccumulation::PerRow`) is the catalog's fastest path (`row_reduce_accumulate`: 2.9× at ≥4 tiles) but `Mul` + DEST accumulation is untested in kernel_lib | full-block `square` into an fp32 CB + `reduce<SUM, REDUCE_ROW>` (also the correctness fallback — see Key Risks) |
| SFPU scope of rsqrt | rsqrt runs on all 32 vectors though only row 0 is meaningful (`sfpu_tile_scope`) | `VectorMode::R` + `ITERATIONS=2` (row-0 scope) — compute-only, likely invisible here |
| Port placement | port cores next to each link's Ethernet core (`fabric_all_gather` probe) cost one probe program per mesh (cached) | `placement="simple"` (any free row-0 core) — payload is ≤ 2 packets so NoC1 distance may not matter |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| x DRAM → worker L1 | input dtype tiles | reader (RISCV_1, NoC0) `TensorAccessor` page reads, batched, one barrier per batch | contiguous row-major core order over row-groups; `split_work`-style assignment is host-computed (`noc_placement`: row-wise spread) |
| gamma DRAM → worker L1 | gamma dtype; only face0-row0 and face1-row0 of each tile (2 × 64 B at tile offsets 0 and `face_bytes`) | reader, 2 small reads per tile | Row broadcast reads only row 0 of srcB, the rest of the CB page is never consumed |
| compute: S, transpose, REDUCE_COL | fp32 | compute | partial row vector: tile row 0 = `[v0..v15]` at byte 0 (face0 row0), `[v16..v31]` at byte 1024 (face1 row0); `v_i` = Σ over this chunk of `x[row i]²` |
| worker → port board (own slot) | fp32 half-rows, 64 B each | writer (RISCV_0, NoC1) `noc_async_write` to port core `(x,y)` at `board_addr + own_slot + ((rt_local·2 + h)·num_w_chunks + wc)·64`; then flush; `noc_semaphore_inc(port.sem_local, 1)` | to BOTH ports (fwd, bwd) of the row-group's link `l`; receive-only ports included so every port holds the own slot |
| port → peer port (Fabric) | slot bytes, contiguous | port sender (RISCV_0, NoC1), `WorkerToFabricEdmSender`, 2D unicast route by destination node (`fabric_set_unicast_route`), `to_noc_unicast_write` / `to_noc_fused_unicast_write_atomic_inc` into the peer port's board slot `g` (same address on every chip: persistent scratch); ready fence per call (copied from `fabric_all_gather` sender); arrival counter = global semaphore | schedule = `fabric_all_gather._schedule(p, G, ring, balance=False)`: Ring toward p+1 sends `[p, p−1, …]` (`G//2` slots), toward p−1 `[p, p+1, …]` (`G−1−G//2`); Linear toward p+1 `[p..0]`, toward p−1 `[p..G−1]`. Relays wait on the arrival counter for that slot's last packet |
| port → leaders | semaphore only | after (local count reached ∧ all expected arrivals landed): `noc_semaphore_inc(leader.sem_board_ready)` for every leader of link `l` | then the port re-arms its global arrival counter (subtract expected), as in the example |
| port board → leader `cb_gather` | fp32 half-row runs | leader reader `noc_async_read` from the port holding source `g`: per (g, rt, half) one run of `num_w_chunks·64 B`, split at 16-row band boundaries, into gather tile `t = s/32` (row `s = g·num_w_chunks + wc`), face `(s%32 < 16 ? 0 : 2) + h`, row `s%16` | `cb_gather` zero-filled at kernel start (local read from the zero region); board holds own slot in every port, remote slot `g` in the port of the direction it arrived from (host table) |
| leader `cb_inv_row_out` → row-group | fp32 half-rows | leader writer: per member rectangle (row-major runs, high corner first), per rt one linked multicast of face0 + face1 row 0 (1088 B) into `cb_inv_row` page `rt` (same CB address on all workers), then a multicast SET of `sem_inv_ready` from the leader's `sem_inv_sig` scratch (= r + 1) on the same NoC/VC/path, so it lands after the data (Perf 2) | includes the leader itself (local NoC write), so `cb_inv_row` has one producer (the reader) on every core |
| worker `cb_inv_row` | fp32 | reader: `cb_reserve_back(block_h)`, `noc_semaphore_wait(sem_inv_ready, 1)`, `cb_push_back(block_h)` | |
| out L1 → DRAM | input dtype tiles | writer `TensorAccessor` writes, batched per `block_w`, only actual (non-pad) tiles | |

Physical shards (refinements): HEIGHT_SHARDED — knob-turn (the shard is `block_h × tensor_w_tiles`, `num_w_chunks = 1`); WIDTH_SHARDED — knob-turn because the W axis's combine already exists (`num_w_chunks` = shards); both consume the shard through a CB backed on the sharded buffer (`ttnn.cb_descriptor_from_sharded_tensor`, zero-copy), never a `TensorAccessor` re-read.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one block = `block_h` tile rows × `block_w` tile columns of one device's x |
| Grid | `allowed_cores` (whole compute grid) minus the union over chips of port cores (`2 · num_links` per chip, placed by `fabric_all_gather`'s Ethernet probe); identical worker list on every chip |
| Per-core work | worker `k` (row-major over the worker list) → `rg = k // num_w_chunks`, `wc = k % num_w_chunks`; leader = the row-group's LAST row-major core (the NoC1-start corner of its fan-out rectangle; Perf 2 — was `wc == 0`); rows `[rg·block_h, min(+block_h, tensor_h_tiles))`, tiles `[wc·block_w, min(+block_w, tensor_w_tiles))` |
| Remainder | ragged last row-group / chunk: nominal block pushes; reader zero-fills pad pages; writer writes only actual tiles; partials of pad rows are written (zeros) so the board layout is uniform |
| Links | row-groups split into `num_links` contiguous ranges; link `l` owns `rows_l` nominal tile rows; ports `(fwd,l)`, `(bwd,l)` serve them |

**Work-split solve (`plan_blocks`, host, deterministic):**

```
best = None
for block_h in 1..tensor_h_tiles:
    num_row_groups = ceil(tensor_h_tiles / block_h)
    if num_row_groups > num_workers: continue
    chunks = min(num_workers // num_row_groups, max(1, tensor_w_tiles // MIN_BLOCK_W))
    block_w = ceil(tensor_w_tiles / chunks); chunks = ceil(tensor_w_tiles / block_w)
    while chunks > 1 and board_bytes(block_h, chunks) > BOARD_CAP_BYTES: chunks -= 1 (recompute block_w)
    if not fits_l1(block_h, block_w): continue
    key = (block_h * block_w, chunks, block_h)          # tiles/core, then fewer chunks, then fewer rows
    best = min(best, key)
```

Resolved Phase 0 values (110-core grid, 4 port cores at 2 links → 106 workers; 2 port cores at 1 link → 108):

| Shape (per device) | tensor_h_tiles × tensor_w_tiles | block_h | num_row_groups | block_w | num_w_chunks | workers busy |
|---|---|---|---|---|---|---|
| 640×1792 (PERF FOCUS) | 20 × 56 | 1 | 20 | 12 | 5 | 100 |
| 32×1792 (decode) | 1 × 56 | 1 | 1 | 4 | 14 | 14 |
| 2048×1280 | 64 × 40 | 2 | 32 | 14 | 3 | 96 |
| 4096×1024 (G=4, 2 links) | 128 × 32 | 5 | 26 | 8 | 4 | 104 |
| 4096×1024 (G=4, 1 link: board cap binds at 4 chunks) | 128 × 32 | 4 | 32 | 11 | 3 | 96 |

Regime selection: single built regime (R1); the solve is the only shape-dependent decision. Required regime-pinned coverage: one shape with `num_row_groups = 1` (decode), one with `num_w_chunks = 1` reachable on the grid used, one with both > 1 (perf focus).

## Circular Buffers

Worker cores (all CBs declared on the whole worker set so every address is identical across workers; leader-only CBs cost ≤ `2·block_h` fp32 pages on non-leaders):

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_x_block` | 0 | x tile | `block_h·block_w` | spans rows & w_local (resident) | input dtype | reader | compute | load → normalize (no gamma) / scale_by_gamma |
| `cb_gamma` | 1 | gamma tile | `block_w` (only when gamma) | spans w_local; streams nothing | gamma dtype | reader | compute | load → scale_by_gamma |
| `cb_scaler` | 2 | bf16 tile | 1 | constant (SUM scaler 1.0) | Float16_b | reader | compute | whole program |
| `cb_sumsq` | 3 | fp32 tile | `block_h` | spans rows (one S tile per tile row) | Float32 | compute | compute | sumsq (square-accumulate → transpose) |
| `cb_sumsq_t` | 4 (alias of 3) | fp32 tile | `block_h` | same bytes as `cb_sumsq` (transpose in place) | Float32 | compute | compute | sumsq (transpose → reduce) |
| `cb_inv_row_out` | 5 (alias of 3) | fp32 tile | `block_h` | same bytes, disjoint lifetime (after the reduce has popped `cb_sumsq_t`) | Float32 | compute | writer | combine → fanout (leader only) |
| `cb_partial_row` | 6 | fp32 tile | `block_h` | spans rows | Float32 | compute | writer | sumsq → publish |
| `cb_gather` | 7 | fp32 tile | `n_gather_tiles·block_h` | spans rows × sources | Float32 | reader | compute | combine (leader only; zero-filled early) |
| `cb_inv_row` | 8 | fp32 tile | `block_h` | spans rows | Float32 | reader | compute | fanout → normalize |
| `cb_inv_col` | 9 (alias of 8) | fp32 tile | `block_h` | same bytes as `cb_inv_row` (transpose in place) | Float32 | compute | compute | normalize |
| `cb_xg` | 10 | fp32 tile | `block_h·block_w` (only when gamma) | spans rows & w_local | Float32 | compute | compute | scale_by_gamma → normalize |
| `cb_out_tiles` | 11 | x tile | `min(OUT_DEPTH, block_h)·block_w` | streams rows, spans one block row of w_local | input dtype | compute | writer | normalize → store |

Alias groups are single `CBDescriptor`s with several `CBFormatDescriptor`s (`ttnn-cb-memory-fundamentals.md` "Aliased Buffers"): {3, 4, 5} (`total_size = block_h·4096`), {8, 9} (`total_size = block_h·4096`). All members use identical page size and quanta (`block_h`), starting from base.

Port cores: no CBs. The board is a persistent L1 scratch (mesh tensor, L1, HEIGHT_SHARDED over the union of port cores, `G · slot_bytes_l` per core, same address on every chip), cached per `(mesh, port-core set, board bytes)` alongside the global semaphores. Fabric packet headers come from `PacketHeaderPool`.

Semaphores:

| Name | Kind | Cores | Incremented by | Waited by | Reset |
|------|------|-------|----------------|-----------|-------|
| `sem_port_local` | program `SemaphoreDescriptor` | ports | workers (1 per worker per port of its link) | port (expected = #workers of link l) | program init (per dispatch) |
| `sem_board_ready` | program | workers | ports (1 per port of the leader's link) | leader (expected = 2) | program init |
| `sem_inv_ready` | program | workers | leader (1 per core of its row-group) | every worker (expected 1) | program init |
| `arrival` | global (`ttnn.create_global_semaphore`, cached) | ports | peer port's fused write+inc | port | port subtracts its expected count at the end of the call |
| `ready` | global (cached, **1 at rest** since refinement 3) | ports | peer port, at the END of its call (credit return) | port, at the start of the call | consumed (−1) each call, returned (+1) by the board's owner once its leaders have read it |
| `sem_gather_done` | program | ports | every leader of the link, after its last gather | receiving port (expected = #leaders) | program init |
| `sem_rx_free` | program | ports | the opposite port of the chip, after its board is read and its arrival counter re-armed | the port that returns the credit | program init |

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_block` | `block_h × block_w` | dataflow (raw NoC + TensorAccessor); scaler via `prepare_reduce_scaler` | DRAM x, gamma | `cb_x_block` (block), `cb_gamma` (`block_w`), `cb_scaler` (1) | all pushed once |
| 2a | `sumsq_block`: square-accumulate | `grid(block_h, block_w)` | `eltwise_chain` `BinaryFpu<Mul, x, x, D0, DestAccumulation::PerRow>` + `PackTile` | `cb_x_block` (Upfront wait, **no pop**, Block, both operands) | `cb_sumsq` (`block_h`, PerOuter/PerOuter, PerRow) | x still resident |
| 2b | `sumsq_block`: transpose | `block_h` tiles | raw `transpose_init` / `transpose_block` | `cb_sumsq` (`block_h`, wait/pop) | `cb_sumsq_t` (`block_h`, reserve/push; aliased, in place) | — |
| 2c | `sumsq_block`: column reduce | `ReduceInputBlockShape::of(1, block_h)` (REDUCE_COL, 1 row × `block_h` cols) | `reduce<SUM, REDUCE_COL, …, ReduceFp32Mode::Accurate>` | `cb_sumsq_t` (`block_h`), `cb_scaler` | `cb_partial_row` (`block_h`) | sumsq group free |
| 3 | `publish_partials` | `block_h` × 2 halves × ports | dataflow | `cb_partial_row` (wait `block_h`, pop `block_h`) | port board own slot | — |
| 4 | `exchange_partials` | slots | port sender kernel (Fabric) | board | peer boards | counters re-armed |
| 5 | `scale_by_gamma_block` | `grid(block_h, block_w)` | `eltwise_chain` `BinaryFpu<Mul, x, gamma(BroadcastDim::Row, Row mapping)>` + `PackTile` | `cb_x_block` (PerTile wait/pop — already resident), `cb_gamma` (Upfront, AtEnd) | `cb_xg` (`block_h·block_w`, PerTile) | x and gamma consumed |
| 6 | `combine_block` (leader) | `ReduceInputBlockShape::of(n_gather_tiles, block_h)` | `reduce<SUM, REDUCE_COL, cb_gather, cb_scaler, cb_inv_row_out, BulkWaitBulkPop, INPUT_AND_OUTPUT, Accurate>` with `post_reduce_op = ×(1/W_global), +eps, rsqrt` | `cb_gather` (`n_gather_tiles·block_h`) | `cb_inv_row_out` (`block_h`) | — |
| 7 | `fanout_inv_rms` (leader) | `block_h` × 2 halves × `num_w_chunks` cores | dataflow | `cb_inv_row_out` (wait/pop `block_h`) | every row-group core's `cb_inv_row` bytes + `sem_inv_ready` | — |
| 8a | `normalize_block`: transpose | `block_h` tiles | raw `transpose_init` / `transpose_block` | `cb_inv_row` (`block_h`) | `cb_inv_col` (`block_h`, aliased, in place) | — |
| 8b | `normalize_block`: multiply | `grid(block_h, block_w)` | `eltwise_chain` `BinaryFpu<Mul, xg, inv_col(BroadcastDim::Col, Col mapping)>` + `PackTile` | `cb_xg` or `cb_x_block` (PerTile), `cb_inv_col` (Upfront, AtEnd, Col) | `cb_out_tiles` (PerTile) | all compute CBs empty |
| 9 | `store_block` | `block_h × block_w` | dataflow | `cb_out_tiles` | DRAM out | — |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| startup | raw_api | `compute_kernel_hw_startup(cb_x_block, cb_out_tiles)` | required by `eltwise/api/chain.hpp:24-46` (caller owns hw startup) | once at kernel top | — | — | — |
| 1 scaler | helper | `dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_COL>(1.0f)` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp:58-60` | pool-type-aware overload; scaler 1.0 (the 1/W is applied in the post-op) | — | `cb_scaler` | — |
| 2a | helper | `eltwise_chain(IterationShape::grid(block_h, block_w), BinaryFpu<input(cb_x_block, Upfront, None, Block), input(cb_x_block, Upfront, None, Block), Mul, None, D0, DestAccumulation::PerRow>, PackTile<output(cb_sumsq, PerOuter, PerOuter, Enabled, Direct, DestAccumulation::PerRow)>)` | `eltwise/api/chain.hpp:532-533` (chain), `:485-494` (BinaryFpu), `:503` (PackTile), `:283` (DestAccumulation), `:356-391` (input/output), `core/chain.inl:1178-1184` (mul_init receives acc_to_dest) | `grid(block_h, block_w)` | `cb_x_block` | `cb_sumsq` | `block_h`, `block_w` |
| 2b, 8a | raw_api | `transpose_init(icb)`; `tile_regs_acquire`; `transpose_block(icb, 0, 0, n)` / `transpose_tile`; pack to the alias id | `tt_metal/hw/inc/api/compute/transpose.h:39, :109, :164` | `n ≤ DEST_AUTO_LIMIT` per acquire (walk `block_h` in DEST-sized groups) | `cb_sumsq` / `cb_inv_row` | `cb_sumsq_t` / `cb_inv_col` | `block_h` |
| 2c | helper | `compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_COL, cb_sumsq_t, cb_scaler, cb_partial_row, ReduceInputPolicy::BulkWaitBulkPop, ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT, ReduceFp32Mode::Accurate>(ReduceInputBlockShape::of(1, block_h))` | `reduce_helpers_compute.hpp:616-637` (signature), `:290-301` (shape), `:112` (policy); `reduce_helpers_common.hpp:14-18` (Accurate = lossless fp32 SFPU path) | — | `cb_sumsq_t` | `cb_partial_row` | `block_h` |
| 5 | helper | `eltwise_chain(grid(block_h, block_w), BinaryFpu<input(cb_x_block, PerTile, PerTile, Block), input(cb_gamma, BroadcastDim::Row, Upfront, AtEnd, Row), Mul>, PackTile<output(cb_xg)>)` | `eltwise/api/chain.hpp:232` (Row mapping), `:305` (BroadcastDim), `:367-377` (bcast input) | — | `cb_x_block`, `cb_gamma` | `cb_xg` | `block_h`, `block_w` |
| 6 | helper (+ raw SFPU calls inside the helper's `post_reduce_op` slot) | `reduce<SUM, REDUCE_COL, cb_gather, cb_scaler, cb_inv_row_out, BulkWaitBulkPop, INPUT_AND_OUTPUT, Accurate>(ReduceInputBlockShape::of(n_gather_tiles, block_h), contiguous(), NoAccumulation{}, post)`; `post(dst) = binop_with_scalar_tile_init(); mul_unary_tile(dst, bits(1/W_global)); add_unary_tile(dst, bits(eps)); rsqrt_tile_init(); rsqrt_tile<RsqrtMode::Default>(dst);` | `reduce_helpers_compute.hpp:616-637`; post-op calls mirror `eltwise/unary/scalar.inl:44-56` (`MulUnary`/`AddUnary`) and `eltwise/unary/math.inl:62-68` (`Rsqrt`) | `1/W_global`, eps as fp32 bit patterns (RT args) | `cb_gather` | `cb_inv_row_out` | `block_h`, `n_gather_tiles` |
| 8b | helper | `eltwise_chain(grid(block_h, block_w), BinaryFpu<input(cb_xg or cb_x_block, PerTile, PerTile, Block), input(cb_inv_col, BroadcastDim::Col, Upfront, AtEnd, Col), Mul>, PackTile<output(cb_out_tiles)>)` | `eltwise/api/chain.hpp:219-232` (Col mapping semantics) | — | `cb_xg`/`cb_x_block`, `cb_inv_col` | `cb_out_tiles` | `block_h`, `block_w` |
| 1, 3, 7, 9 | raw_api (dataflow) | `TensorAccessor`, `noc_async_read/write`, `noc_semaphore_*` | `tech_reports/tensor_accessor/tensor_accessor.md`; `fabric_all_gather/program_descriptor_with_inline_kernels.py:172-304` (sender pattern) | — | — | — | batch sizes = `block_w` |
| 4 | raw_api (fabric) | `WorkerToFabricEdmSender::build_from_args`, `PacketHeaderPool::allocate_header`, `fabric_set_unicast_route`, `to_noc_unicast_write`, `to_noc_fused_unicast_write_atomic_inc`, `to_noc_unicast_atomic_inc`, `send_current_slot_non_blocking`, `send_payload_flush_blocking_from_address` | `references/blackhole-fabric.md` §5; example sender `program_descriptor_with_inline_kernels.py:172-304` | `HEADER_RING = 4`, inc every 8th packet + slot last | board | peer board | — |

**Helpers considered and rejected (raw_api rows):**

| Raw entry | Candidate helper | File:line of the mismatch | Concrete reason |
|-----------|------------------|---------------------------|-----------------|
| 2b / 8a transpose | `eltwise_chain` elements; `reduce` | no transpose element exists in `ttnn/cpp/ttnn/kernel_lib/` (grep: only `matmul_block_helpers.hpp:277` B-operand transpose flag and `reduce_helpers_compute.inl:699-713` internal reload transpose) | a 32×32 tile transpose is not expressible through any kernel_lib entry point; gap: a `Transpose<InputSpec, Dst>` chain element would close it |
| 6 post-op SFPU calls | `AddUnary`/`MulUnary`/`Rsqrt` chain elements | `reduce_helpers_compute.hpp:632-637` — `post_reduce_op` is a `void(uint32_t dst)` callback, not a chain | chain elements cannot be composed into the reduce's post-op slot; the calls are the elements' own bodies |
| dataflow / fabric | `mcast_pipe` (`mcast/kernel/mcast_pipe.hpp`), `ccl/ccl_helpers_dataflow.hpp` | mcast_pipe moves one payload to a rectangle with a sender/receiver CB handshake; the fan-out here is 64 B half-rows into a CB on each row-group core and the board reads are strided runs from two port cores | not a multicast-shaped transfer in Phase 0; R4 and the fan-out lamp are where `mcast_pipe` enters |

## Broadcast Verification

| Phase | Op | CB_A (semantic name) Valid Region | CB_B (semantic name) Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| 2a square-accumulate | Mul | `cb_x_block` All | `cb_x_block` All | None |
| 5 scale_by_gamma | Mul | `cb_x_block` All | `cb_gamma` Row0 (only row 0 read from DRAM) | Row |
| 8b normalize | Mul | `cb_xg` / `cb_x_block` All | `cb_inv_col` Col0 (transpose of a Row0-valid tile) | Col |

## Reduce Direction Verification

Both reduces are REDUCE_COL (single reduce dimension per call; listed because two different reduces exist).

| Logical Dim | Tile ReduceDim | Output Valid Region | BroadcastDim | ReduceInputBlockShape | EltwiseShape |
|-------------|----------------|--------------------|--------------|-----------------------|--------------|
| x columns within a chunk (after transposing S) | REDUCE_COL | Row0 (`v_i` at column i) | — (wire format) | `of(1, block_h)` | — |
| sources (G devices × chunks) stacked as gather-tile rows | REDUCE_COL | Row0 → transposed to Col0 | Col (normalize) | `of(n_gather_tiles, block_h)` | `grid(block_h, block_w)` |

## Validation and contract notes (for the implementer)

| Check | Error |
|-------|-------|
| rank < 2 | ValueError |
| gamma shape ≠ `(1,1,1,W_local)` (TILE) or `(1,1,ceil(W_local/32),32)` (ROW_MAJOR) | ValueError |
| `cluster_axis` mesh size 1 | ValueError |
| explicit `num_links` > discovered on any hop of any group, or ≤ 0 | ValueError |
| Ring on an axis whose fabric config does not wrap it (`TORUS_Y` wraps axis 0, `TORUS_X` axis 1, `TORUS_XY` both) or no direct link from the last to the first device | ValueError |
| support axes outside SUPPORTED / EXCLUSIONS (incl. `float32` with `fp32_dest_acc_en=False`) | `UnsupportedAxisValue` / `ExcludedCell` — `validate()` is the entry point's first line |

The op never changes the fabric config; it reads `ttnn.get_fabric_config()` and the mesh shape. A 2-device Ring is run as a Line (same link pair). Ring is written for any `G ≥ 3`; nothing special-cases `G = 4`.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| `Mul` with DEST accumulation | `mul_init` receives `acc_to_dest` (`core/chain.inl:1184`) but no kernel_lib test covers Mul+PerRow; `eltwise_binary.h:530` says acc flag is unused on WH/BH | implementer's first probe: one worker, 1×4 tiles, compare `S` against torch. Fallback (same block shape, more L1): `square<cb_x_block (Upfront, None), cb_xsq>` over the block into an fp32 `block_h·block_w` CB, then `reduce<SUM, REDUCE_ROW>(of(block_h, block_w))` (col vector) and transpose to row — the wire format is unchanged |
| Garbage rows in the gather tile | REDUCE_COL sums all 32 rows; L1 is not zero between programs | leader reader zero-fills `cb_gather` before any board read (barrier between the two) |
| Pad tiles of a ragged block | stale L1 in pad pages would enter `S` | reader zero-fills pad pages of `cb_x_block` and `cb_gamma`; nominal pushes |
| Board address identical on every chip | Fabric writes target the peer's board at *this* chip's address | board is a mesh L1 tensor (lock-step allocation), cached; RT args carry its address |
| Ordering write → semaphore | a semaphore increment overtaking its data | `noc_async_writes_flushed()` (or write barrier) before every `noc_semaphore_inc`; Fabric uses the fused write+inc on the slot's last packet |
| Back-to-back calls | call N+1's sends could overwrite a board a leader of call N still reads | ready credit (refinement 3, replaces the start-of-call fence): returned at the end of call N only after every leader of the link has read the board and the arrival counter is re-armed, so call N+1's sends need no round trip; the board is only (re)allocated after a host `synchronize_device` (a peer may send before this chip reaches the call) |
| Determinism | the regression suite checks bit-identical repeats | gather row order fixed by `(g, wc)`; Accurate fp32 reduce; every leader on every chip sums the same values in the same order |
| Mean divisor | must be the true `W_global`, never padded | `1/W_global` computed on host from `G·W_local` |
| Width padding in the sum (refinement `w_non_aligned`) | tile padding of `W_local` must not enter `ss` | refinement masks the last chunk's last tile column range (reader zero-fill of padded columns or a mask tile); Phase 0 rejects non-aligned via SUPPORTED |
| Trace / host sync | the op runs inside traces | the only host sync is the one-time creation of global semaphores + board + Ethernet probe (cached on first call, before capture); per call only RT args change |
| `ttnn.Topology` at import | circular import | `from ttnn.operations.ccl import Topology` |
| SUPPORTED gamma "none" | canonical no-gamma cell | SUPPORTED lists `"none"` for `gamma_dtype` and `gamma_layout` |

### Structural impossibilities

None beyond `feature_spec.py`'s INVALID.

### Catalog entries that informed the knobs

| Entry | Knob it informed |
|-------|------------------|
| `width_split` | split `w_local` across cores when rows cannot fill the grid (decode) |
| `tensix_all_reduce` | root (leader) reduce + fan-out back for a tiny payload on a 1-D line; every core works including the root |
| `row_reduce_accumulate`, `eltwise_l1_vs_dest_accumulate` | DEST-accumulate the squares first, reduce once (`sumsq_block`) |
| `compute_fusion` | multiplies stay on the FPU (no SFPU dest-reuse multiply); SFPU only for rsqrt in the reduce post-op |
| `compute_block_size` | one helper call per block, not per tile row |
| `double_buffer`, `noc_placement` | batched reads, reads NoC0 / writes NoC1, row-major spread of workers |
| `fabric_all_gather` / `fabric_reduce_scatter` | port placement, ring/line schedule, inc-every-8, ready fence, arrival re-arm |
