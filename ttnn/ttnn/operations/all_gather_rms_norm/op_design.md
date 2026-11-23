# Operation Design: all_gather_rms_norm

## Overview

| Field | Value |
|-------|-------|
| Classification | fused CCL: a local row reduction on each chip, a cross-chip exchange of per-row statistics, a local normalize, then a cross-chip all-gather of the normalized slices over Fabric 2D. One `ttnn.generic_op` per call (one `MeshProgramDescriptor`). |
| Goal | Every device of a `cluster_axis` group gets the full normalized `(..., M, W)` row (`W = G·W_local`) from its own `(..., M, W_local)` slice: RMSNorm, or LayerNorm with `norm_type="layer_norm"`. Phase 0 is multi-core on every chip and multi-link on the fabric from day 1. |
| Math (rms_norm) | `full = concat_p x[p]`, `out[q][..., h, w] = full[..., h, w] · rsqrt((1/W)·Σ_{w'<W} full[..., h, w']² + eps) · gamma[w]` for every group member q |
| Math (layer_norm) | `mean = (1/W)·Σ full`, `var = (1/W)·Σ (full − mean)²`, `out = (full − mean) · rsqrt(var + eps) · gamma + beta` (gamma / beta terms omitted when absent) |
| Scheme in one line | **Stats-first, gather-normalized (SF-GN):** each chip reduces its own slice to per-row partials while the per-call fabric fence runs. The partials (Float32) go around the ring on the link-0 port cores. Each chip then normalizes **only its own slice** and writes it to its own output. Its port cores send the normalized tiles **straight into the peers' final output pages**: the routers write them, and no core on the receiving chip runs any compute or data-movement step for them. |
| References | base op `ttnn/ttnn/operations/base_ops/rms_norm/` (`op_design.md`, `changelog.md`, kernels); `ttnn/ttnn/operations/examples/{fabric_all_gather, fabric_ring_exchange, fabric_reduce_scatter}/`; `.claude/references/{blocking-model.md, l1-footprint-discipline.md, blackhole-fabric.md, blackhole-ccl-kernels.md, precision_convention.md}`; `ttnn/cpp/ttnn/kernel_lib/` helpers; golden suite `eval/golden_tests/all_gather_rms_norm/` |

### Catalog entries → knobs they informed

| Entry (measured) | Fact used | Knob / decision it set |
|---|---|---|
| `fabric_all_gather` (TRANSPORT) | t ≈ 8.4 µs + bytes / 77.2 GB/s per chip (1 link). 2 links: 0.53–0.63× from 256 tiles. A ring is 1.25–1.8× faster than a line. Balancing pays from ~64 tiles per chip. `inc_every` 8 and a batch of 4 chunks are best or within 5–16% everywhere. Port cores go below their Ethernet core, sends on NoC1. One port per (link, direction). | Bulk transport = the example's port reader / sender and relay with arrival counters: `inc_every = 8`, `batch_chunks = 4`, `port_cb_bytes = 2·batch_chunks·payload`, `balance = slot_tiles ≥ 64`, `num_links` = all usable, probe-based port placement. |
| `fabric_ring_exchange` (TRANSPORT) | Minimal correct exchange (ready fence + header-only items): ~5.0 µs traced. A 128 B relay is +0.34 µs, 8 KiB +1 µs. Direct multi-hop sends are 1.1–1.6 µs slower than relay (the 2-hop fence). A 2nd link does not help payloads ≤ 32 KiB. | Stats exchange = one-hop relay on **link 0 only**. One fence per port per call, shared by the stats and the bulk. Direct multi-hop sends are `rejected`. |
| `blackhole-fabric.md` rules 1, 4, 9 | Packet-rate bound below ~5 KiB (1 tile/packet caps a link at 12.8 GB/s). A fused inc on every packet halves the rate. Read runs of pages contiguous in one bank. | Packets carry **runs contiguous at the destination**, up to `scatter_chunks` runs per packet (see Mechanism caps). Increments on every 8th packet and on each entry's last. |
| base op `width_split` / `_choose_k` (local) | Splitting W across k members fills the grid when tile-rows are few. | The `k` (W members) cost model, re-derived for the fused grid (Work Distribution). |
| base op `eltwise_l1_vs_dest_accumulate`, `compute_fusion`, `sfpu_tile_scope` | DEST-accumulated x² is 2.01× faster. Dest-reuse gamma. Column-0 finalize. | RMS local compute = the base op's pass 1 / normalize, ported unchanged. |
| base op Perf 1/2 | Compact gamma page (row 0 of faces 0/1) + raw row-broadcast dest-reuse multiply. Row partials shipped as 128 B column-0 records into ⌈k/32⌉ gather tiles. | `cb_gamma` / `cb_beta` page format. The 128 B record + gather-tile layout of the stats exchange. |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| input_tensor | ttnn.Tensor (mesh) | yes | TILE, rank ≥ 2, identical per-device shape / dtype / placement | — | addresses: common RT |
| cluster_axis | int (kw) | yes | 0 or 1; mesh size along it ≥ 2 | — | host (plan) |
| gamma | Optional[Tensor] (kw) | no | full-W, replicated. TILE `(1,1,1,W)` or ROW_MAJOR sticks `(1,1,ceil(W/32),32)`. dtype float32 / bfloat16 | None | `has_gamma`, `gamma_is_sticks` CT; address common RT |
| beta | Optional[Tensor] (kw) | no | layer_norm only, only with gamma, same dtype + layout + shape rule as gamma | None | `has_beta` CT; address common RT |
| norm_type | str (kw) | no | `"rms_norm"` / `"layer_norm"` | `"rms_norm"` | `is_layer_norm` CT |
| epsilon | float (kw) | no | ≥ 0, finite | 1e-6 | fp32 bits, common RT |
| topology | Topology (kw) | no | Linear, Ring (Ring needs the wrap link; G = 2 Ring runs as Linear) | Linear | host (schedule) |
| num_links | Optional[int] (kw) | no | 1 ≤ n ≤ min over the group's hops of `len(get_forwarding_link_indices)` | None = that min | host (port count) |
| compute_kernel_config | ComputeConfigDescriptor (kw) | no | `fp32_dest_acc_en` gated, fidelity / approx any value | `default_compute_kernel_config()` | compute KernelDescriptor `config` |
| memory_config | Optional[MemoryConfig] (kw) | no | output placement: DRAM/L1 INTERLEAVED, or a HEIGHT/WIDTH shard spec built for the gathered shape | DRAM interleaved | host |
| output_tensor | Optional[Tensor] (kw) | no | per-device shape `(..., M, W)`, input dtype, TILE | None | address common RT |

Entry-point contract:

| Item | Decision |
|------|----------|
| Module | `ttnn/ttnn/operations/all_gather_rms_norm/__init__.py` exports `all_gather_rms_norm`, `default_compute_kernel_config`, `INPUT_TAGGERS`, `SUPPORTED`, `EXCLUSIONS`. Topology enum at import time: `from ttnn.operations.ccl import Topology`. |
| `default_compute_kernel_config()` | Factory, fresh per call: HiFi4, `fp32_dest_acc_en=True`, `math_approx_mode=False`. `None` resolves only through it (`axes.py` imports it). |
| First line | `validate(...)` (registry gate), then caller-error checks, then plan / program build, then one `ttnn.generic_op(tensors, mesh_program_descriptor)`. |
| Caller errors (ValueError) | Input rank < 2. Gamma shape ≠ its layout's format for the full W. Beta without gamma, beta with rms_norm, or an unknown norm_type. Beta shape / dtype / layout ≠ gamma's. `output_tensor` per-device shape / dtype / layout mismatch. num_links > discovered. Ring on an axis with no wrap link, or on a fabric config that does not wrap that axis (TORUS_X wraps axis 1, TORUS_Y axis 0, XY both). Cluster axis of size 1. |
| Capacity errors (RuntimeError) | The worker footprint does not fit L1 at the chosen split. The stats table exceeds `STATS_TABLE_CAP`. Fewer than 2L free cores for ports (sharded input covering the grid). No golden INPUT reaches any of them (Mechanism caps). |
| Support refusals | `UnsupportedAxisValue` / `ExcludedCell` from `ttnn.operations._op_contract`, raised by `validate()` (SUPPORTED per axis, then EXCLUSIONS). |
| INPUT_TAGGERS | exactly two: `alignment`, on the per-device input `inputs[0]` (`w_non_aligned` if `W_local % 32`, else `h_non_aligned` if `H % 32`, else `tile_aligned`), and `rank` (= `len(inputs[0])`). gamma_dtype / gamma_layout are projected inside `validate()` (real value, or `"none"`), not registered as taggers (the base op's reason: a tagger key stops TARGET iteration). |
| Recommended SUPPORTED (implementer owns it) | the from-base Phase 0 list: dtype {float32, bfloat16, bfloat8_b}; fp32_dest_acc_en {True, False}; layout {TILE}; alignment {all three}; rank {2,3,4}; gamma_mode {gamma, no_gamma}; gamma_dtype {float32, bfloat16, "none"}; gamma_layout {TILE, ROW_MAJOR, "none"}; norm_type {rms_norm, layer_norm}; beta_mode {beta, no_beta}; memory_layout {INTERLEAVED, HEIGHT_SHARDED, WIDTH_SHARDED}; cluster_axis {0, 1}; topology {Linear, Ring}; num_links {1, 2}. |
| Recommended EXCLUSIONS | `{dtype: float32, fp32_dest_acc_en: False}` (precision convention). `{dtype: bfloat8_b, alignment: w_non_aligned}` and `{dtype: bfloat8_b, alignment: h_non_aligned}`: carried over from the base op, because block-float tails cannot be element-masked or element-shifted. Carry the last two only while the base op's reason holds; R_U's element-shift assembly cannot address 16-datum shared-exponent blocks. |
| Index axes | none (the gathered and normalized dim is fixed at −1) |

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | per device `(..., M, W_local)`, rank ≥ 2; leading dims fold into rows |
| Dtype | float32, bfloat16, bfloat8_b |
| Layout | TILE |
| Memory | DRAM/L1 INTERLEAVED, or L1 HEIGHT_SHARDED / WIDTH_SHARDED (per device, over that device's cores) |
| gamma / beta | full-W, replicated: TILE `(1,1,1,W)` (row 0 of each tile valid) or ROW_MAJOR sticks `(1,1,ceil(W/32),32)` (stick j = `gamma[32j .. 32j+31]`, tail zero-padded) |

### Output

| Property | Value |
|----------|-------|
| Shape | per device `(..., M, G·W_local)`; slice p at columns `[p·W_local, (p+1)·W_local)` (dense, no padding between slices) |
| Dtype | input dtype |
| Layout | TILE |
| Memory | `output_tensor`'s if given, else `memory_config`, else DRAM interleaved. Allocated as a mesh tensor, so it has the **same address on every chip**. |

Geometry (single source: host `derive_geometry()`, alignment-aware, per image):

| Symbol | Formula |
|---|---|
| `W_local`, `W` | `shape[-1]`, `G·W_local` |
| `num_images` | `prod(shape[:-2])` (1 for rank 2) |
| `tensor_row_tiles` (R) | `num_images · ceil(M / 32)` |
| `slice_w_tiles` (Wlt) | `ceil(W_local / 32)`: the local slice's tile-columns, last one partial if `W_local % 32` |
| `out_w_tiles` (Wt) | `ceil(W / 32)` |
| `w_valid_in_last_tile` | `W_local − 32·(Wlt − 1)` |

## Blocking Model

The op has **two kinds of work**, and both are blocked. The *local* work (reduce, normalize) is the base op's blocking re-run on the grid left after the ports. The *cross-device* work is the exchange: a stats round, then a bulk gather. Its blocks are packets and slots, labelled **TRANSPORT** (how bytes cross the fabric) or **OVERLAP** (what is hidden behind the exchange).

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| row (tile-rows of the local slice, leading dims folded) | **independent**: each output row depends only on its own row | `block_row_tiles` | the whole per-group assignment `core_row_tiles` (one block per worker). The stats round is one exchange per call for all rows, so x must stay resident from pass 1 to normalize. | host `select_split()` → RT `core_row_tiles`, `row_start` | row-groups over the **worker grid** (grid − 2L port cores); sharded: from the shard spec | knob-turn (grid); a row-windowed stats round is a scheme-change (`deferred` R_WIN) |
| W_local (tile-columns of the slice) | **dependent**: Σ spans the full gathered row | `core_w_tiles` = `ceil(Wlt / k)` (last member: remainder) | `k` from the fused cost model (Work Distribution); focus case k = 4 → 3 tiles | host `select_split()` → RT `w_tile_start`, `core_w_tiles`; CT `w_split_members` (k) | k workers per row-group. They never talk to each other directly: their partials meet in the stats table (no intra-chip combine round). | knob-turn (k ≤ `P_CAP/G`) |
| group position p (device axis, the gathered W) | **dependent** for the stats (Σ spans all G slices). Each slice's **output** is independent once the stats are known. | one slot = one chip's slice | whole slice per chip (not cut) | mesh plan | slice p is computed only on chip p; it reaches the others over the fabric | — (fixed by the problem) |
| stats partition j (k members × G chips) | **dependent** (summed) | `P = G·k` columns in `gather_tiles = ceil(P/32)` gather tiles | P = 16 (focus) | derived from k, G | every member of a row-group combines all P partials of its rows itself (redundant, no second round) | — |
| bulk packet (TRANSPORT) | independent (pages land at fixed addresses) | `scatter_chunks` runs per packet, each ≤ `payload_pages` | 4 runs, ≤ `payload_pages` pages (interleaved output); sharded output runs are already long, so 1–2 | host CT | each (link, direction) port core owns a lane set (DRAM banks, or tile-rows for sharded) | knob-turn |
| port batch (TRANSPORT) | — (pipeline quantum) | `batch_chunks` | 4 packets (port CB = 2 batches) | host CT | per port | knob-turn |

Gamma and beta are **reuse-shared along the row axis** (they do not vary by row). They vary along W, and chip p needs only its own columns `[p·W_local, (p+1)·W_local)`, so each member reads only its slice of them.

Operand-reuse check per (operand, chosen split):

| Operand | Split | Varies along it? | Consequence |
|---|---|---|---|
| x[p] | row-groups / W members | yes / yes | read once from DRAM (interleaved) or consumed in place (sharded) |
| gamma, beta | row-groups | **no** | each row-group re-reads its member slice (≤ 1/32 of x bytes per worker). Read-once + `ttnn.Mcast` is a `deferred` row, as in the base op. |
| gamma, beta | W members, chips | yes | member m on chip p reads columns `p·W_local + w_tile_start·32 …` only |
| per-row stats | W members, chips | **no**: every member of every chip needs the same per-row result | the stats exchange: the k·G partials of a row meet in every member |
| normalized slice p | chips | **no**: every chip needs it | the bulk all-gather (fabric, relay, balanced ring) |

Step-1 table, one row per axis of every data object:

| Data | Axis | Character | Block factor (Phase 0) | Core assignment | Buffer depth | Why |
|---|---|---|---|---|---|---|
| x | row | independent | `core_row_tiles` (whole) | row-groups over the worker grid | 1 (resident) | normalize needs x after the stats round |
| x | W_local | dependent | `core_w_tiles` | k members | same CB | residency avoids a second read of x |
| gamma / beta | W | reuse-shared over row | member slice, program-resident | each member its own | 1 | ≤ 1/32 of x bytes |
| gamma / beta | row | absent | n/a | n/a | n/a | one row |
| partial records | row | independent | `core_row_tiles` | the producing member | 1 block | one 128 B record per (tile-row, stat) |
| stats table | chip × row × member | chips/members dependent, row independent | whole table (all rows of one call) | **communication role only**: the table lives on the 2 link-0 port cores, which do no arithmetic. Every consuming worker combines its own rows (no idle-group stage). | 1 call | one fabric round per call |
| gathered partials | row × partition | partition summed | `core_row_tiles × gather_tiles` | every member of a row-group combines redundantly | 1 block | a 2nd round to scatter the combine costs a fabric/NoC round for 1–7 tile-ops |
| inv / mean / rstd | row | independent | `core_row_tiles` | every member of the group | 1 block | consumed by normalize |
| out (own slice) | row, W_local | as x | `core_row_tiles × core_w_tiles` | same worker as x | **whole block resident until program end** | port cores pull it from worker L1 (no DRAM re-read) |
| out (remote slices) | row, W | independent | packet = ≤ `scatter_chunks` runs | port (link, direction) lanes | port CB 2 × `batch_chunks` packets | lands at final addresses: no receive-side core touches it |

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_x` | `x_block_depth` | 1 (one resident block) | — (the block must live across the exchange) |
| `cb_out_block` | — | 1 whole block | must stay intact for the ports' pull (correctness, not overlap) |
| `cb_port_chunks` | `batch_chunks` × 2 | 4 × 2 packets (112 KiB at 14 KiB payload) | reader fills batch b+1 while the sender sends batch b (catalog: 4/batch best or within 5–16%) |
| `cb_ln_sq` | `ln_sq_depth` | 2 tiles | chain packs the next centered square while the reduce consumes |
| all other worker CBs | — | 1 block | producer/consumer hand-off within one block |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| One fabric stats round per call (x resident from pass 1 to normalize) | per-worker block = whole assignment | footprint(worker) ≤ L1 budget, else RuntimeError (R_WIN deferred) | silent overflow of L1 / CB overlap |
| Stats table on the 2 hub cores | `G · R · k · nstats · 128 B` | ≤ `STATS_TABLE_CAP` = 256 KiB (largest golden: G = 8, R = 64, k = 1, nstats = 3 → 196 KiB). `select_split` clamps k to `stats_k_cap(R, G, nstats)` (verifier fix: the cost model picked k = 3 for LN at R = 64, a 288 KiB table); RuntimeError only if k = 1 does not fit | hub L1 overflow |
| Gather tile columns | P = G·k | `k ≤ P_CAP / G`, `P_CAP = 128` (≤ 4 gather tiles per row-tile) | combine cost grows with P on every member |
| Packet payload | pages per packet | `payload_pages = floor(get_tt_fabric_max_payload_size_bytes() / page_bytes)`; a run is cut at `payload_pages` | router rejects or corrupts an oversized packet |
| Scatter write | runs per packet | `scatter_chunks ≤ NOC_SCATTER_WRITE_MAX_CHUNKS = 4` (`fabric_edm_packet_header.hpp:194`); an **increment-carrying** packet ≤ 2 runs (`NocUnicastScatterAtomicIncFusedCommandHeader` is exactly 2 chunks, `:254`), 1 run → fused unicast + inc | ASSERT in the header code / wrong chunk sizes |
| Destination contiguity | a run | consecutive pages whose destination NoC addresses are consecutive in one bank (interleaved) or one shard core (sharded), detected with the output `TensorAccessor` | a multi-page write across a bank boundary lands in the wrong bank |
| Header pool | headers per RISC | ≤ 12 (`blackhole-ccl-kernels.md` §3.3): `batch_chunks` + 1 ready + stats ≤ 12 | `PACKET HEADER POOL EXHAUSTION` hang |
| One producer per router channel | port cores | exactly one sender core per (link, direction); stats ride the link-0 ports | undefined router behaviour |
| Arrival counters: one ordered sender per counter | bulk / stats counters | one global semaphore per (port, kind); the threshold rule of `blackhole-ccl-kernels.md` §6.1 | a relay reads unlanded pages |
| Counters return to 0 every call (trace) | all global semaphores | consumers decrement exactly what they consumed (`0u − n`) + `noc_async_atomic_barrier()` | the second traced call hangs or reads early |
| Identical CB addresses on every core of a chip | every CB | one CB descriptor list on the **full program core range** (workers ∪ ports) | the port computes a worker's `cb_out_block` address from its own instance: a mismatch pulls garbage |
| Same address on every chip | output, stats table, staging scratch, global semaphores | output / staging / global semaphores mesh-allocated (MeshBuffer / `create_global_semaphore`); the stats table is a program-scoped CB on the full core range (verifier fix: identical CB lists give the same address on every core of every chip, and peers write into it only after the ready fence). Worker / DRAM NoC coordinates are translated (identical across chips). | a peer write lands at a wrong address |
| fp32 DEST capacity (4 tiles half-sync) | none on the block | helpers clamp via `DEST_AUTO_LIMIT` (`dest_helpers.hpp:103`). The LN combine holds ≤ 3 live DEST slots. | n/a |
| W padding of the slice | last tile-column of the last member | RMS: reader zero-fills columns ≥ `w_valid_in_last_tile` (base op, `fill_l1_range`, `l1_helpers.hpp:90`). LN: the centered square is masked by the reduce's `only_partial()` mask; zero-fill is not enough because (0 − K)² ≠ 0. | padding leaks into Σ (golden poisons padding with 1000) |
| UnpackToDestFp32 / Accurate reduce need fp32 DEST | LN exact-centering path | only when `fp32_dest_acc_en=True`; False → FPU centering (16-bit DEST is the caller's choice) | compile / wrong-precision path |

### Regimes

Named memory boundary: the **per-chip DRAM** plus the **fabric links**. Minimum per chip, from the roofline: x read once (S), the full output written once (G·S), and each remote slice received once (G−1 slices into each chip, ring busiest link direction (G−1)/2 slices). gamma/beta: their W_local columns once per row-group (reuse excess ≤ 1/32 of x).

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **R_A interleaved: stats-first, gather-normalized, final-address landing** | **built** (Phase 0) | `W_local % 32 == 0` and input INTERLEAVED | `core_row_tiles × core_w_tiles` per worker; packet = ≤ 4 destination-contiguous runs | DRAM per chip: x 1·S, own output write 1·S, landed (G−1)·S (router writes), **relay reads**: ring G = 4 balanced 1·S (every slice a chip forwards is read back from its DRAM once) → 6S vs 5S. The relay read is the excess, removed by `fabric_mcast` (deferred). Own slice is pulled from worker L1, not DRAM (saves 2S). Links at the minimum (balanced). | Per block: one x read, one pass-1 chain + one collapse reduce, one records write, one combine, one normalize chain (one init each). Per packet: one router header + slot wait; more runs per packet means fewer router packets per byte (rule 1). |
| **R_A HEIGHT_SHARDED** | **built** (Phase 0, from-base) | `W_local % 32 == 0` and input HEIGHT_SHARDED | the resident shard (k = 1) | x never crosses DRAM (zero-copy `cb_x` on the shard). Output per `memory_config`: a HEIGHT shard spec for the gathered shape lands in L1 at final addresses (runs = `Wlt` tiles, contiguous in one shard core). | as R_A |
| **R_A WIDTH_SHARDED** | **built** (Phase 0, from-base) | `W_local % 32 == 0` and input WIDTH_SHARDED | one group, k = number of input shards, block = the shard | as above. Stats P = G·k, k from the shard grid (clamped by `P_CAP/G`, else RuntimeError). | as R_A |
| **R_U unaligned staged** | **built** (Phase 0, from-base) | `W_local % 32 != 0` (any placement) | as R_A, plus `assemble_block` = output tile-rows of a worker | Slices land at non-tile-aligned offsets, so an output tile mixes chips and no chip owns whole output pages. Each chip's normalized padded slice (`R × Wlt` tiles) is gathered into a **slot-contiguous DRAM staging scratch** (the example's dim −2 layout: full-length bank runs, full packets). Each chip then assembles its own output from the G slots by element-shifted copies. DRAM above minimum: staging landed (G−1)·S' + read back G·S' (S' = padded slice). The assembly is serialized after the last landing (exposed). | one staging read per output tile-row instead of per tile |
| R_WIN: row-windowed stats rounds (x streamed in row windows, one stats round per window) | **deferred**: no golden INPUT exceeds the residency cap (largest per-worker block: 2048×1280 bf16, k = 1 → 80 KiB x + 80 KiB out), and the capacity error names it. Reachable: `block_row_tiles` already exists as a knob, and the stats table already indexes by tile-row. A window is a subset of rows with its own slot and counter thresholds. | footprint > budget | `block_row_tiles < core_row_tiles` | one fabric stats round per window instead of per call | — |
| R_RAW: raw-slice gather into receiver L1, every chip normalizes full rows | **deferred**: the bulk could start right after the fence (no stats round in front of it). But each chip computes 4× the tiles, and remote raw data must land in L1 of the normalizing cores. That needs either a fused increment on every packet (one per destination worker, measured 2× rate loss, rule 4) or a landing hub whose NoC fan-out to ~100 workers is a hotspot. Positive reason: SF-GN's only extra serial term is the stats round (+0.3–1 µs, `fabric_ring_exchange`), which is smaller than either cost. Reachable: the ports, fence, schedule and arrival counters are shared. Only the landing address (worker L1 instead of output) and the worker's input source change. | — | — | same DRAM minimum (no relay re-read if forwarded from L1); +3× local compute | — |
| `fabric_mcast`: chip-level multicast of a slice to the ring's next G/2 chips (router-forwarded) | **deferred**: removes the relay DRAM re-read (6S → 5S) and the per-hop relay turnaround. It is `[in-tree]`/unmeasured (`blackhole-ccl-kernels.md` §4: "no wrap handling in the route setter"), and a ring's wrap is exactly where a 4-ring needs it. Reachable: an entry's source and pages are unchanged, only its route header differs. | — | — | DRAM 5S | — |
| direct multi-hop unicast (no relay) | **rejected**: superseded by relay. Measured 1.1–1.6 µs slower (its fence waits for the 2-hop destination), and it sends a far slice twice over the first hop (2 vs 1.5 slices on the busiest link of a 4-ring). | — | — | +33% link load | — |
| DRAM-landing, then normalize in place (gather raw slices into the output, re-read, normalize, re-write) | **rejected**, dead end: about 12S DRAM per chip (above the link bound). Nothing built on it survives into SF-GN. | — | — | 12S | — |
| port re-reads own normalized slice from its own output DRAM (the example's exact source) | **rejected**: superseded by the pull from worker L1, which needs the same "own slice ready" signal and saves 2S DRAM reads per chip | — | — | +2S | — |
| two-round LayerNorm (mean round, then centered-variance round) | **rejected**: superseded by the single-round shifted-sum combine (Chan / shifted sums, below), which is exact to fp32 with one round | — | — | +1 fabric round (~0.3–1 µs) | — |
| single core / single port / single link | **rejected**: Phase 0 is multi-core and uses all usable links (catalog: 2 links 0.53–0.63×) | — | — | — | — |
| hub-side fp32 stats combine (port-core TRISCs combine with fp32 DEST and return final per-row stats) | **deferred**: only needed if `fp32_dest_acc_en=False` must still sum in fp32 (see Key Risks). Reachable: the hub already holds the whole table. | `fp32_dest_acc_en == False` | — | +1 local NoC round | — |

### Regime selection (exact, host-side)

```
aligned  = (W_local % 32 == 0)
regime   = R_A if aligned else R_U          # placement only changes how cb_x / the output are backed
k, groups = select_split(...)                # Work Distribution; WIDTH_SHARDED: k = #input shards
if footprint(worker) > budget or stats_table_bytes > STATS_TABLE_CAP: raise RuntimeError(capacity)
```

Regime-pinned tests (acceptance test): R_A interleaved (focus 640×384 rms, 640×32 LN), R_A with k > 1 (640×384 → k = 4), R_U (17×50, 32×50), R_A multi-batch rank-4 (2×1×128×256), rank-3 and rank-2 shapes. The predicates use only `W_local % 32` and the input placement, so they select the same regime on every grid. k depends on the worker count, so the k > 1 pin uses a shape that gives k > 1 for every worker count ≥ 40 (640×384: k ≥ 2).

### Traffic ranking

Per chip, G = 4 balanced ring, interleaved, S = one slice. Crossings counted per tier, not timed.

| Candidate scheme / split | DRAM | Fabric (busiest link direction) | Cross-core NoC | Local compute | Verdict |
|---|---|---|---|---|---|
| **SF-GN, final-address landing (R_A)** | x 1, out-own 1, landed 3, relay re-read 1 = **6S** | 1.5S (minimum) + stats `R·k·nstats·128 B` per slot | ports pull own 2S from worker L1; records G·k·128 B per tile-row | own slice only (1×) | **chosen** for aligned W |
| SF-GN + fabric multicast | 5S | 1.5S | same | 1× | cheapest; `deferred` (unmeasured on the wrap) |
| R_RAW (raw gather into receiver L1) | x 1 (+0 if pulled), out 4 = 5S | 1.5S | 3S landed to L1 + fan-out | 4× | `deferred` (notification / fan-out costs) |
| raw gather into DRAM + normalize in place | ~12S | 1.5S | — | 4× | `rejected` (DRAM-bound) |
| SF-GN via slot-contiguous staging (R_U path used for aligned W) | ~12S (staging write + read back) | 1.5S | — | 1× | used only where final-address landing is impossible (W unaligned) |
| direct multi-hop unicast instead of relay | 5S | 2S | — | 1× | `rejected` |

Local split ranking (within a chip): row-groups only (k = 1) vs row-groups × W members (k > 1) are DRAM-identical. k > 1 adds only P = G·k records per row-tile on the NoC (G·k·128 B). That buys occupancy, and it shortens the **critical-path normalize** (the only local work after the stats round). Occupancy is not used as a traffic argument here; it enters through the cost model in Work Distribution.

### Block schedule

Logical schedule per chip (W = worker, H = hub = link-0 port core of each direction, P = every port core). Cross-device items carry their kind.

```cpp
// ---- t = 0: everything below this line that is local runs concurrently with the fence ----
P: fabric_fence();                               // TRANSPORT  ready inc to the peer port that writes into me; wait mine
W: load_gamma_beta_slice();                      // reader, program-resident
W: load_x_block();                               // reader (interleaved) / shard in place (sharded)
W: local_stats_block();                          // compute: RMS Σx² | LN (m_i, T_i, K_i)  → cb_partial
W: publish_records_block();                      // writer: col-0 → 128 B records → both hubs' tables (+ local inc)
H: exchange_stats();                             // TRANSPORT  own slot after fence + records; relay upstream slots
H: announce_stats_ready();                       // OVERLAP    multicast inc to every worker once own + expected slots landed
W: gather_partials_block();                      // reader: P records of my rows → cb_gathered (col j = partition j)
W: combine_stats_block();                        // compute: Σ / Chan → inv (RMS) | mean, rstd (LN)
W: normalize_block();                            // compute: → cb_out_block (whole block, resident)
W: store_own_block();  W: announce_own_ready();  // writer: own pages → own output; inc every port's own-ready
P: gather_normalized_slices();                   // TRANSPORT  own slot (pulled from worker L1) + relays, balanced ring
P: await_landings();                             // TRANSPORT  wait every expected arrival; re-arm counters
// R_U only:
P: announce_landed();                            // OVERLAP    multicast inc to workers
W: assemble_block();                             // reader/writer: staging slots → shifted output tiles → output
```

| Block operation | Block shape | Resident across it | Intended frequency of fixed costs |
|---|---|---|---|
| `fabric_fence` (P) | 1 header-only packet each way per port | — | once per call per port. Connections opened once per call. |
| `load_gamma_beta_slice` (W) | `core_w_tiles` compact pages (each of gamma, beta) | — | once per program. Sticks: one 64 B (bf16) / 128 B (fp32) page read per tile-column at stick index `p·Wlt + w_tile_start + j`. TILE: the base op's 2 × 64 B row-0 reads. R_U: element-shifted gather (below). |
| `load_x_block` (W) | `core_row_tiles × core_w_tiles` | gamma, beta | one reserve, one barrier per tile-row, one push per tile-row. RMS W-tail zero-fill on the last member. |
| `local_stats_block` (W) | RMS: chain `grid(core_row_tiles, core_w_tiles)` + `reduce of(core_row_tiles, 1)`. LN: `reduce of(core_row_tiles, core_w_tiles)` (S1→m), `unary_bcast` (K), centered-square chain + masked `reduce` (T). | cb_x (not popped) | one chain init + one reduce call per block per stat |
| `publish_records_block` (W) | `core_row_tiles × nstats` records of 128 B | cb_x | one NoC write per record per hub (2 hubs), one write barrier, one local inc per hub per worker |
| `exchange_stats` (H) | slots: `R·k·nstats·128 B` each, ⌈slot/payload⌉ packets, inc on the slot's last packet | — | ring: fwd sends G/2 slots (own + relays), bwd G/2 − 1. Line: fwd sends p+1 slots, bwd G−p. One arrival wait per relayed slot. |
| `announce_stats_ready` (H, reader) | 1 multicast semaphore inc over the program grid rectangle | — | once per call per hub (2 per chip) |
| `gather_partials_block` (W) | `core_row_tiles × nstats × gather_tiles` gather tiles from G slot reads per tile-row | cb_x | G NoC reads of `k·nstats·128 B` per tile-row, one barrier per block; zero-init gather tiles once per block |
| `combine_stats_block` (W) | `reduce of(core_row_tiles, gather_tiles)` (+ LN chain) | cb_x | one reduce per block per stat; finalize post-op per tile-row |
| `normalize_block` (W) | `grid(core_row_tiles, core_w_tiles)` | cb_x (popped at end), cb_gamma, cb_beta | one chain init per block |
| `store_own_block` / `announce_own_ready` (W) | `core_row_tiles × core_w_tiles` own pages | cb_out_block (kept, never overwritten) | per-tile write, one barrier per tile-row; one inc per port core (2L) after the final barrier |
| `gather_normalized_slices` (P) | packets of ≤ `scatter_chunks` runs, ≤ `payload_pages` pages | — | per batch of 4 packets: one reader barrier, one sender flush. Inc every 8th packet + each entry's last. Own entry waits once on own-ready = number of active workers. |
| `await_landings` (P) | — | — | one wait on the total expected incs, one decrement |
| `assemble_block` (R_U, W) | output tile-rows `[row_start, row_start + core_row_tiles)` × the worker's output tile-column range | — | one staging read run per (tile-row, slot), one shifted copy per (row, source segment), one write per output tile |

**Stall-shadow check.** Three stages wait on something they do not produce:

1. **The stats exchange waits for the fence (~5 µs, `fabric_ring_exchange`) and for the peers' records.** Independent of it are the x read, pass 1, record publishing and the gamma/beta loads, which are all scheduled into it. That is the point of starting the local work at t = 0, on cores other than the ports.
   - RMS (lamp): `x·gamma` could also be lifted into this window (commutativity `x·r·γ = (x·γ)·r`), leaving one multiply after the stats arrive. It is not built in Phase 0: it needs a block-sized fp32 buffer, and the base op measured the gamma dest-reuse at ~40 ns/tile.
   - LN: nothing else is independent. Centering needs the global mean.
2. **The bulk send waits for the normalized own slice.** Independent of it: the **relay entries**, which wait on upstream arrivals instead. They cannot run earlier because every upstream also waits for its stats. The bulk is a single dependency chain: stats → normalize → own entry → peers' relays.
3. **R_U assembly waits for all landings.** Nothing is independent of it. Accepted: R_U serves decode-size shapes, and finer per-slot announcements are a lamp.

**Launch skew:** a late chip delays its stats slot, and every chip waits for it. The fence already absorbs it. The bulk never waits on more than the stats round has already waited for.

### Perf lamps

| Lamp | Kind | Why the default may be wrong here | Nearby alternative to measure |
|------|------|-----------------------------------|-------------------------------|
| Bulk start granularity | OVERLAP | The own entry waits until **all** workers announce own-ready, which serializes the slowest worker's normalize in front of the first packet | per-row-group ready counters (the port walks its own entry in row-group order and waits only for the groups a batch touches) |
| Scatter runs per packet | TRANSPORT | `scatter_chunks = 4` is `[unmeasured]`. At W_local = 384 (Wlt = 12) the interleaved runs are 1–2 pages: 1 run/packet ≈ 3 KiB packets (packet-rate bound). | `scatter_chunks` ∈ {1, 2, 4}. Sharded output (runs of Wlt). |
| Stats ride on link 0 only | TRANSPORT | The second link's ports idle during the stats round | split the slot between the two links' ports (catalog: a 2nd link does not help ≤ 32 KiB, so expect null) |
| Grid synchronization (k) | OVERLAP | k = 4 on the focus case puts 80 workers' records on 2 hubs (80 + 80 local incs, 80·4 reads of the table) | k = 2 / k = 1 (fewer records, longer normalize) |
| x·gamma in the stall shadow (RMS) | OVERLAP | the gamma multiply sits on the critical path after the stats arrive | precompute `x·γ` into an fp32 block during the fence |
| Balance threshold | TRANSPORT | `balance` from the example's measured threshold (64 tiles per chip); the focus case is 240 | balance off at 240 tiles |
| Relay re-read from DRAM | TRANSPORT | every relayed slot is read back from DRAM (1S per chip) | `fabric_mcast` regime (deferred row) |
| Overlap (one block per worker) | OVERLAP | x read and pass 1 do not overlap within a worker | `block_row_tiles = ceil(core_row_tiles/2)`, depth 2 (the read is already hidden behind the fence: expect null) |
| R_U assembly tail | OVERLAP | assembly starts only after every landing | per-slot `announce_landed` (assemble the tiles whose sources have landed) |
| Writer per-tile write | local | own-slice writes are per tile | batch per tile-row (base op measured flat) |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| gamma/beta DRAM → `cb_gamma` / `cb_beta` (W reader) | gamma dtype, compact page = row 0 of faces 0/1 (the base op's Perf 2 page) | Sticks: one page read per tile-column (stick `p·Wlt + w_tile_start + j`), split into its two 16-element halves. TILE: the base op's two 64 B row-0 reads. R_U: element-shifted. Global column `c = p·W_local + local col`. Read the covering stick(s) / tile row-0 segment(s) and copy element-wise into the compact page; zero columns ≥ W_local of the last tile. | `beta` added at its own global column (prompt rule): same mapping as gamma |
| x → `cb_x` | input dtype tiles | interleaved: TensorAccessor reads, tile id `t·Wlt + w_tile_start + j`. HEIGHT/WIDTH sharded: `cb_x` backed on the shard (`ttnn.cb_descriptor_from_sharded_tensor`, zero-copy). | W-tail zero-fill for both norms (RMS Σx², LN S1). It must not mutate a caller's shard: copy the last tile into a 1-tile scratch (base op). LN's T additionally masks the tail with the reduce's `only_partial` mask, because (0 − K)² ≠ 0. |
| compute → `cb_partial` | Float32 (wire contract) | RMS: chain → `cb_sumsq_tiles` → collapse reduce. LN: three stats (below). | column 0 holds the per-row value |
| `cb_partial` → hub tables | 128 B fp32 records | W writer: copy column 0 (face 0 words `r·16`, face 2 words `256 + (r−16)·16`) into `cb_record_staging`, then `noc_async_write` to both hubs at `table + slot(p) + ((tr·k + m)·nstats + s)·128`; barrier; `noc_semaphore_inc(hub, sem_records_local, 1)` | slot(p) = p · R · k · nstats · 128 |
| hub table → peer hub table | 128 B records, slot-contiguous | H sender: `fabric_unicast_noc_unicast_write` of the slot (⌈slot/payload⌉ packets). Last packet fused with an inc of the peer hub's `stats_arrived` counter. Relay = send slot q from the local table after `stats_arrived ≥ index(q)+1`. | the table is a mesh L1 sharded scratch over the union of all chips' hub cores: same address everywhere |
| hub → workers | semaphore | H reader: wait `sem_records_local == active_workers` and `stats_arrived == expected_slots`; then `noc_semaphore_inc_multicast` of `sem_stats_ready` over the program rectangle; then decrement `stats_arrived` by `expected_slots` | every core of the rectangle is in the program and declares the semaphore |
| hub tables → `cb_gathered` | Float32 gather tiles: column j = partition j = `q·k + m` | W reader: wait `sem_stats_ready == 2`. Per tile-row, G reads of `k·nstats·128 B` (slot q from the fwd hub if q is behind me or me, else the bwd hub) into `cb_record_landing`. Zero the gather tiles, then scatter each record's 32 words into column j (face 0/2 rows, face 1/3 for j ≥ 16). | tiles with j ≥ P stay 0 |
| compute → `cb_out_block` | output dtype | normalize chain | whole block, never overwritten during the call |
| `cb_out_block` → own output | output dtype pages | W writer: TensorAccessor writes, page `(row_start + r)·Wt + p·Wlt + w_tile_start + j`; final barrier; `noc_semaphore_inc` on every port core's `sem_own_ready` | own slot done locally: no port touches the own output pages |
| worker L1 → port CB → peer output (own entry) | output pages | P reader: wait `sem_own_ready == active_workers`. For each packet of its lane set, NoC-read each page from its owning worker's `cb_out_block` (address = the port's own `cb_out_block` base, identical on every core, + `((r − row_start)·core_w_tiles_max + (j − w_tile_start))·page_bytes`), packed in run order. P sender: scatter-write the runs to the peer's output pages (`dst = out_accessor.get_noc_addr(page)`). | the routers write the final pages: no receive-side kernel |
| own output → port CB → next peer (relay entries) | output pages | P reader: wait the arrival threshold (`blackhole-ccl-kernels.md` §6.1), read the runs from the local output (DRAM or L1 shard), sender forwards | the example's relay, unchanged |
| R_U: normalized padded slice → staging slot | output dtype tiles | staging = mesh DRAM interleaved scratch `[G · R · Wlt]` pages, slot q contiguous. Own slot is written by the workers, remote slots land via the ports (example's `for_each_chunk` bank-run walk, full packets). | then `announce_landed` |
| R_U: staging → output | output pages | W reader reads the source tiles for an output tile-row. For output tile (tr, oc), the global columns `32·oc + c'` come from slot `q = col // W_local`, local tile `(col − q·W_local) // 32`. Element segments are copied per row, split at face boundaries (16 columns) on both sides; columns ≥ W of the last output tile are zeroed. W writer writes the page. | dataflow-RISC copies, dtype-size elements (bf16 / fp32; bf8b excluded) |

Placement axis (`memory_layout`) as the physical realization of the logical shards:

| Value | Axis it cuts | Character | Unlock class | Realization |
|---|---|---|---|---|
| INTERLEAVED | — | — | built (Phase 0) | R_A / R_U as above |
| HEIGHT_SHARDED | row | independent | knob-turn (built Phase 0) | workers = input shard cores, k = 1, block = the shard, `cb_x` zero-copy. Port cores are chosen from cores **not** in the input shard grid. The output (gathered HEIGHT spec) lands in L1 at final pages. |
| WIDTH_SHARDED | W_local | dependent | scheme-change already designed (the stats table absorbs the k members, built Phase 0) | one group, k = shard count, block = shard. The W tail is masked in compute, never zeroed in the caller's shard. The output is per its WIDTH spec. |
| BLOCK_SHARDED | — | — | not in TARGET | — |

Structural impossibilities: none beyond the INVALID entries already in `feature_spec.py`.

### Deviations from the base op (why)

| Base op mechanism | Here | Why |
|---|---|---|
| intra-chip combine of the k members (symmetric unicast all-gather / owner combine, R2/R3) | removed: the members' partials go into the stats table, combined with the G chips' in one gather tile (`P = G·k` columns) | the cross-chip round exists anyway, and a second (intra-chip) round would add a handshake on the critical path |
| `_choose_k`: `COMBINE_TILE_COST = 3`, slice ≥ 4 tiles | cost `rows·(slice + COMB_TILE_COST_FUSED·ceil(G·k/32))`, `COMB_TILE_COST_FUSED = 1`, slice ≥ 1 | the combine is one gather-tile reduce per row-tile regardless of k ≤ 32/G, and the normalize after the stats round is the critical path, so smaller slices pay |
| streamed output (`cb_out` 2 × unit) | `cb_out_block` = whole block | the ports pull the normalized tiles from worker L1 |
| gamma `(1,1,1,W)` single-row RM | sticks `(1,1,ceil(W/32),32)`: one page per tile-column | the model's format (prompt) |
| RM input paths, BLOCK_SHARDED, RM flat / narrow-flat regimes | not ported | layout TILE only; BLOCK_SHARDED is not in TARGET |
| RMS only | + LayerNorm: three-stat local pass, fp32-exact combine and centering | prompt |

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | a block: `core_row_tiles` tile-rows × the member's `core_w_tiles` |
| Grid | `device.compute_with_storage_grid_size()` (runtime value, never a literal). **Worker grid** = grid − port cores (2L per chip; HEIGHT/WIDTH sharded: workers = input shard cores). |
| Port cores | 2L per chip, one per (direction ∈ {fwd, bwd}, link l), placed as in `fabric_all_gather.plan()`: probe the Ethernet core, take the free worker with the fewest NoC1 hops (`_worker_nearest_noc1`). Receive-only ends of a line keep their port. Hubs = (fwd, 0) and (bwd, 0). |
| `select_split(R, Wlt, Nw, G)` (interleaved) | for k in `1 .. min(Wlt, P_CAP // G)`: `groups = min(Nw // k, R)` (stop if 0), `rows = ceil(R / groups)`, `slice = ceil(Wlt / k)`, `cost = rows · (slice + COMB_TILE_COST_FUSED · ceil(G·k / 32))`. Pick the min cost, ties → smaller k. Normalize `k = ceil(Wlt / ceil(Wlt / k))` (no empty member). |
| Rows over groups | `ttnn.split_work_to_cores`-style two classes over `groups`: `core_row_tiles` = ceil / floor. Group g member m = worker index `g·k + m` in the ordered worker list (row-major logical order, ports removed). Members of a group need no adjacency (they never exchange directly). |
| Focus check on the fused grid | QB2 11×10 = 110 cores, L = 2 → 106 workers. **640×384** (R = 20, Wlt = 12, G = 4): k = 4 → 20 groups × 4 = 80 workers × 3 tiles (P = 16). The base op's rule on the full 110-core grid gives k = 3 × 4 tiles (60 cores): no cliff, the fused cost model only prefers the next k. **640×32 (LN)**: k = 1 → 20 workers × 1 tile. **640×1792**: k = 5 → 20 × 5 = 100 workers × 12 tiles. **32×1792** (decode): k = 16 → 16 workers × 4 tiles (P = 64, 2 gather tiles). |
| Remainder | rows: two classes; W: the last member gets `Wlt − (k−1)·ceil(Wlt/k)`; geometry by `ceil` per image. CB capacities use `core_row_tiles_max`, `core_w_tiles_max`. |
| Inactive workers | workers beyond `groups·k` run the kernels with 0 blocks: they still declare every CB and semaphore (multicast rectangle), and do not publish records or own-ready |
| Port lane split (TRANSPORT) | interleaved output: link l owns DRAM banks `l, l+L, …` of the destination (the example). Balanced ring: the far slot's lanes are split between fwd (first half) and bwd (second half). Sharded output: link l owns tile-rows `tr ≡ l (mod L)`. R_U staging: bank lanes. |
| Runtime args | buffer addresses (input, output, gamma, beta, scratches) as **common** RT args, the only per-call change. Worker identity from `get_absolute_logical_x/y()` plus a CT list of port positions, so worker kernels need no per-core RT args (dispatch: "per-core RT args cost a flat ~3 µs"). Port cores: per-core RT args (schedule, fabric connection args appended LAST). Core sets merged into rectangles (`CoreRangeSet.merge`). |
| Program cache | `custom_program_hash` over the plan (port cores, schedule, k, regime) and the CT flags. Per-chip programs differ only in RT args and port positions. |

## Circular Buffers

T_in / T_out = page bytes of the input / output dtype. T_g = compact gamma page. T32 = 4096 B (Float32 tile). T_acc = Float32 if `fp32_dest_acc_en` else Float16_b. **Every CB is declared on the full program core range** (Mechanism caps). Worker-only CBs are unused on ports and vice versa.

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_x` | 0 | T_in | `core_row_tiles_max · core_w_tiles_max` (sharded: the shard, zero-copy) | spans row × W of the block; resident across the exchange | input dtype | reader | compute | program (one block) |
| `cb_gamma` | 1 | T_g | `core_w_tiles_max` | spans W; row absent | gamma dtype | reader | compute | program |
| `cb_beta` (LN + beta) | 2 | T_g | `core_w_tiles_max` | as gamma | gamma dtype | reader | compute | program |
| `cb_reduce_scaler` | 3 | bf16 tile | 2 (scaler 1.0, W-tail 0/1 mask) | constants | Float16_b | reader | compute | program |
| `cb_sumsq_tiles` (RMS) | 4 | T_acc | `core_row_tiles_max` | spans row; W folded in DEST | T_acc | compute | compute | pass 1 → collapse |
| `cb_ln_mean_local` (LN) | 5 | T32 | `core_row_tiles_max` | spans row (m_i in col 0) | Float32 | compute | compute | stats pass |
| `cb_ln_shift` (LN) | 6 | T32 | `core_row_tiles_max` | spans row (K_i broadcast to all columns) | Float32 | compute | compute | stats pass |
| `cb_ln_sq` (LN) | 7 | T_acc | `ln_sq_depth` = 2 | streams over W | T_acc | compute | compute | per tile |
| `cb_partial` | 8 | T32 | `core_row_tiles_max · nstats` | spans row × stat | **Float32** (wire contract) | compute | writer | stats pass → publish |
| `cb_record_staging` | 9 | 128 B | `nstats · core_row_tiles_max` | dataflow scratch for 128 B records | raw fp32 | writer | writer (self scratch) | publish |
| `cb_record_landing` | 10 | 128 B | `G · k · nstats` | one tile-row's records | raw fp32 | reader | reader (self scratch) | gather, per tile-row |
| `cb_gathered` | 11 | T32 | `core_row_tiles_max · nstats · gather_tiles` | spans row × stat × partition tile | Float32 | reader | compute | gather → combine |
| `cb_stat_consts` (LN) | 12 | T32 | 1 (col j = n_j) | constant | Float32 | reader | compute | program |
| `cb_inv` | 13 | T32 | `core_row_tiles_max` | spans row (RMS inv / LN rstd, col 0) | T_acc | compute | compute | combine → normalize end |
| `cb_mean_bcast` (LN) | 14 | T32 | `core_row_tiles_max` | spans row (exact mean in all columns) | Float32 | compute | compute | combine → normalize end |
| `cb_rstd_bcast` (LN) | 15 | T32 | `core_row_tiles_max` | spans row | T_acc | compute | compute | combine → normalize end |
| `cb_ln_scratch` (LN) | 16 | T32 | 2 | split-broadcast / combine intermediates (m_hi, m_lo) | Float32 | compute | compute | per tile-row |
| `cb_out_block` | 17 | T_out | `core_row_tiles_max · core_w_tiles_max` | spans the block; pulled by ports | output dtype | compute | writer | program (never overwritten) |
| `cb_assemble_scratch` (R_U) | 18 | T_out | `2 + 1` (two source tiles, one output tile) | dataflow scratch | output dtype | reader | writer | per output tile |
| `cb_port_chunks` (ports) | 19 | `payload_pages · T_out` | `2 · batch_chunks` | 2 batches of packets | output dtype pages | port reader | port sender | per batch |

Semaphores. Program-local, on the full core range: `sem_records_local` (hubs), `sem_stats_ready` (workers), `sem_own_ready` (ports), `sem_landed` (workers, R_U). Global, mesh-allocated over the union of all chips' port cores: `ready`, `bulk_arrived`, `stats_arrived` (hubs). Declare the program-local ones **before** `setup_fabric_connection` (it takes the lowest free ids).

Push/wait parity:
- `cb_x`: pushes per tile-row; compute waits the whole block.
- `cb_partial`: compute pushes `core_row_tiles·nstats`; the writer waits / pops the same.
- `cb_gathered`: the reader pushes `core_row_tiles·nstats·gather_tiles`; compute's reduce waits / pops per stat block (`BulkWaitBulkPop`).
- `cb_out_block`: compute pushes per tile; the writer waits / pops per tile. Popped pages stay in place, because the capacity equals the whole block and nothing pushes again within the call.
- `cb_port_chunks`: example batch rule (a batch ends at a multiple of `batch_chunks` or at an entry end).

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 0 | startup | — | API | cb_x, cb_x, cb_out_block | — | first statement of compute MAIN |
| 1 | `local_stats_block` RMS (a) Σx² | `grid(core_row_tiles, core_w_tiles)` | yes (base op chain) | cb_x, block, None/None | cb_sumsq_tiles, 1 per tile-row | cb_x kept |
| 2 | RMS (b) collapse | `ReduceInputBlockShape::of(core_row_tiles, 1)` | yes | cb_sumsq_tiles, BulkWaitBulkPop | cb_partial, `core_row_tiles` | raw Σx² in col 0 |
| 3 | LN (a) S1 → m_i | `of(core_row_tiles, core_w_tiles)`, `WaitUpfrontNoPop`, post-op ×(1/n_i) | yes | cb_x (kept) | cb_ln_mean_local | — |
| 4 | LN (b) shift K_i | `unary_bcast<Col>` over `core_row_tiles` tiles | yes | cb_ln_mean_local (kept until the records) | cb_ln_shift | K_i = tf32(m_i), exact in fp32 |
| 5 | LN (c) T_i = Σ(x − K_i)² | per tile-row: chain `Sub(x, cb_ln_shift[row])` → `Square` → pack cb_ln_sq; `reduce` AccumulateViaAdd + `only_partial` mask on the last tile | yes | cb_x (kept), cb_ln_shift | cb_partial (stat T) | padding masked |
| 6 | LN (d) records m_i, K_i | copy col 0 of cb_ln_mean_local / cb_ln_shift into cb_partial (stats m, K) | yes (`copy`) | cb_ln_mean_local, cb_ln_shift (popped) | cb_partial | 3 stats per tile-row |
| 7 | `publish_records_block` | `core_row_tiles · nstats` records | raw dataflow | cb_partial (popped) | hub tables | — |
| 8 | `gather_partials_block` | `core_row_tiles · nstats · gather_tiles` | raw dataflow | hub tables | cb_gathered | — |
| 9 | `combine_stats_block` RMS | `of(core_row_tiles, gather_tiles)`, `ReduceWithinTile::Collapse` over the partition columns, finalize `·(1/W) + eps → rsqrt` | yes | cb_gathered | cb_inv | — |
| 10 | `combine_stats_block` LN | (i) mean = Σ_j n_j·m_j / W: chain `Mul(G_m, N)` → reduce (Accurate). (ii) `split_broadcast(mean)` → cb_mean_bcast: m_hi = `unary_bcast<Col>(mean)`, m_lo = mean − m_hi (SFPU), `+ unary_bcast<Col>(m_lo)` (SFPU add). (iii) M2 = Σ_j [T_j − n_j(m_j − K_j)² + n_j(m_j − mean)²]: SFPU chain on G_m, G_K, G_T, N, cb_mean_bcast → reduce (Accurate), finalize `·(1/W) + eps → rsqrt` → cb_inv. (iv) `unary_bcast<Col>(rstd)` → cb_rstd_bcast. | yes (chain + reduce + bcast) | cb_gathered, cb_stat_consts | cb_mean_bcast, cb_inv, cb_rstd_bcast | — |
| 11 | `normalize_block` RMS | `grid(core_row_tiles, core_w_tiles)`: `Mul(x, cb_inv Col)` → dest-reuse ×gamma (row bcast) | yes + raw (base op `MulGammaRowBcast`) | cb_x, cb_inv, cb_gamma | cb_out_block | caller pops cb_x, cb_inv |
| 12 | `normalize_block` LN (fp32 DEST) | per tile: CopyTile x (UnpackToDestFp32) → D0; CopyTile cb_mean_bcast[row] → D1; `SubBinary` (SFPU); CopyTile cb_rstd_bcast[row] → D1; `MulBinary`; dest-reuse ×gamma, +beta (row bcast, raw) | yes + raw | cb_x, cb_mean_bcast, cb_rstd_bcast, cb_gamma, cb_beta | cb_out_block | pops at block end |
| 12' | `normalize_block` LN (16-bit DEST) | per tile-row: `BinaryFpu<Sub, cb_x, cb_mean_bcast[row]>` → `DestReuseBinary<Mul, cb_rstd_bcast[row], DEST_TO_SRCA>` (rstd is already broadcast, so no `BroadcastDim` is needed) → ×gamma, +beta (raw row-bcast dest-reuse) | yes + raw | cb_x, cb_mean_bcast, cb_rstd_bcast, cb_gamma, cb_beta | cb_out_block | pops at block end |
| 13 | `store_own_block` | `core_row_tiles · core_w_tiles` pages | raw dataflow (TensorAccessor) | cb_out_block | output | data stays in place |
| 14 | `gather_normalized_slices` | packets | raw dataflow + fabric | worker cb_out_block / own output | peer outputs | — |
| 15 | `assemble_block` (R_U) | worker's output tiles | raw dataflow | staging (DRAM) | output | — |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| startup | raw_api (required boot) | `compute_kernel_hw_startup` | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:60` | (cb_x, cb_x, cb_out_block) | — | — | — |
| RMS Σx² | helper | `eltwise_chain` + `BinaryFpu<Mul, A=B=cb_x, Dst::D0, DestAccumulation::PerRow>` + `PackTile<output(cb_sumsq_tiles, PerOuter, PerOuter, …, PerRow)>` | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp:533`, `BinaryFpu :491`, `PackTile :504`, `DestAccumulation :283` | the base op's call, unchanged | cb_x | cb_sumsq_tiles | `grid(core_row_tiles, core_w_tiles)` |
| RMS collapse / LN S1 / combines | helper | `compute_kernel_lib::reduce` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp:632` | `<SUM, REDUCE_ROW, in, cb_reduce_scaler, out, policy, INPUT_AND_OUTPUT, fp32_mode, algorithm, within_tile>`. Collapse: AccumulateViaAdd. LN S1: ReduceTile, `fp32_mode = Accurate` for float32 x (`reduce_helpers_common.hpp:18`). Combines: AccumulateViaAdd + Collapse, Accurate when `fp32_dest_acc_en`. | per row | per row | `core_row_tiles`, `core_w_tiles`, `gather_tiles` |
| finalize post-op | raw_api inside the helper's hook | `binop_with_scalar_tile_init`, `mul_unary_tile`, `add_unary_tile`, `rsqrt_tile_init`, `rsqrt_tile` | `tt_metal/hw/inc/api/compute/eltwise_unary/binop_with_scalar.h`, `eltwise_unary/rsqrt.h` (base op cites `:60, :35, :156`, `:44, :21`) | 1/W, eps as fp32 bits (common RT) | DST | DST | — |
| LN shift / broadcasts | helper | `UnaryBcast<BroadcastDim::Col, …>` / `unary_bcast` | `ttnn/cpp/ttnn/kernel_lib/eltwise/broadcast/bcast.hpp:25`, `:28` | Col | col-0 tile | bcast tile | `core_row_tiles` |
| LN centered square | helper | `eltwise_chain(BinaryFpu<Sub, cb_x, cb_ln_shift>, Square<>, PackTile<cb_ln_sq>)` | `chain.hpp:533`; `Square` `eltwise/unary/misc.hpp:35` | per tile-row `grid(1, core_w_tiles)` (Offset addressing into cb_x) | cb_x, cb_ln_shift | cb_ln_sq | `core_w_tiles` |
| LN T reduce with mask | helper | `reduce<…, AccumulateViaAdd>` + `ReducePartialScaler::only_partial()` | `reduce_helpers_compute.hpp:632`; mask contract `:306-331`, `ReducePartialScaler :340` | mask tile at cb_reduce_scaler index 1 | cb_ln_sq | cb_partial | — |
| LN combine / centering SFPU | helper | `CopyTile` (with `UnpackToDestFp32` on cb_x / cb_mean_bcast / cb_gathered via the kernel's `unpack_to_dest_mode`), `SubBinary`, `MulBinary`, `AddBinary`, `Square` | `chain.hpp:483` (CopyTile); `eltwise/binary/sfpu/basic.hpp:20, :26, :28` | fp32 | — | — | — |
| RMS normalize | helper + raw | `BinaryFpu<Mul, cb_x, input(cb_inv, Col)>` + raw row-broadcast dest-reuse gamma multiply (port `MulGammaRowBcast`, base op `kernels/rms_norm_compute.cpp:314`) | `chain.hpp:491` | — | cb_x, cb_inv, cb_gamma | cb_out_block | `grid(core_row_tiles, core_w_tiles)` |
| scaler / mask CB | helper | `prepare_reduce_scaler<PoolType::SUM, ReduceDim::REDUCE_ROW>` (pool-type-aware, bf16 packed) + mask fill | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp:59` | — | — | cb_reduce_scaler | — |
| W-tail zero fill | helper | `fill_l1_range` | `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:90` | — | cb_x | cb_x | — |
| records / gather / own store / pull / assembly | raw_api (dataflow) | `noc_async_read/write`, barriers, `TensorAccessor`, `noc_semaphore_inc`, `noc_semaphore_wait_min` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:1980` (wait_min); `tech_reports/tensor_accessor/tensor_accessor.md` | — | — | — | — |
| stats-ready / landed signal | helper (preferred) / raw | `McastArgs<>::sender(noc).send_signal()` + host `ttnn.Mcast`. Raw fallback: `noc_semaphore_inc_multicast` | `ttnn/cpp/ttnn/kernel_lib/mcast/kernel/mcast_args.hpp:65`; `dataflow_api.h:2324` | one signal per hub per call | — | — | — |
| fabric fence / stats / bulk | raw_api (fabric) | `WorkerToFabricEdmSender::build_from_args`, `fabric_unicast_noc_unicast_write`, `fabric_unicast_noc_scatter_write`, fused inc variants (`to_noc_fused_unicast_write_atomic_inc`, `to_noc_fused_unicast_scatter_write_atomic_inc`), `fabric_unicast_noc_unicast_atomic_inc` | `tt_metal/fabric/hw/inc/mesh/api.h:57, :237, :400`; `tt_metal/fabric/fabric_edm_packet_header.hpp:196, :254, :635, :716` | route by destination node (2D) | port CB | peer memory | `scatter_chunks`, `batch_chunks`, `inc_every` |
| host fabric | host API | `ttnn.get_forwarding_link_indices`, `ttnn.setup_fabric_connection` (append LAST), `ttnn.get_tt_fabric_max_payload_size_bytes`, `ttnn.create_global_semaphore` | `fabric_all_gather/program_descriptor_with_inline_kernels.py:784`, `plan()` `:717` | — | — | — | — |

Helpers considered and rejected (raw_api entries):

| raw_api entry | Candidate helper | Mismatch (file:line) | Reason |
|---|---|---|---|
| fabric sends (all) | `ccl_helpers_dataflow.hpp` `FabricStreamSender` | `blackhole-ccl-kernels.md` §12: `[unmeasured]`, routes are 1D hop counts, "on this branch, not on main" | the measured transport is the example's raw sender. Porting it keeps the cited measurements valid. |
| records publish / gather | `ttnn.Mcast` data channel (`mcast/host/mcast.hpp`) | one sender per round; here ~80 workers each write 2 × `core_row_tiles·nstats` 128 B records to 2 hubs (fan-in, not fan-out) | a multicast does not express many-to-one. The gather is G reads per tile-row (fan-in pull). |
| own-slice pull | `local_copy_helpers_dataflow.hpp` | it copies within one core; the pull is a cross-core NoC read into a packet buffer whose order is set by the destination runs | wrong shape |
| R_U assembly | `tilize` / `untilize` helpers | the shift is by `W_local mod 32` elements at a non-tile offset: untilize → shifted stick copy → tilize is 3 passes, and still needs the same element copy | the dataflow copy is one pass; a compute-based assembly is a lamp only if R_U becomes hot |
| finalize post-op | `eltwise_chain` `Rsqrt` (`eltwise/unary/math.hpp:38`) | `reduce()` exposes the post-reduce hook as a `dst_idx` callable (`reduce_helpers_compute.hpp:632`) | the base op's reason, unchanged |
| RMS gamma dest-reuse with row broadcast | `DestReuseBinary` (`chain.hpp:500`) | `DestReuseBinary` takes no `BroadcastDim` (`chain.hpp:499-501`) | the base op built the raw `MulGammaRowBcast`; port it (and its Add twin for beta) |

## Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast Dim |
|-------|-----|-------------------|-------------------|---------------|
| RMS Σx² | Mul (acc_to_dest) | cb_x: All | cb_x: All | None |
| RMS normalize 1 | Mul | cb_x: All | cb_inv: Col0 | Col |
| RMS / LN gamma | DestReuse Mul | DEST: All | cb_gamma: Row0 (faces 0/1) | Row (raw) |
| LN beta | DestReuse Add | DEST: All | cb_beta: Row0 | Row (raw) |
| LN shift | UnaryBcast | cb_ln_mean_local: Col0 | — | Col |
| LN centered square | Sub | cb_x: All | cb_ln_shift: All (already broadcast) | None |
| LN combine (i) | Mul | cb_gathered (m): All (cols ≥ P are 0) | cb_stat_consts: All (cols ≥ P are 0) | None |
| LN combine (iii) | SFPU binaries | gathered tiles: All | cb_mean_bcast: All | None (pre-broadcast) |
| LN normalize | SFPU Sub / Mul | x: All | cb_mean_bcast / cb_rstd_bcast: All | None (pre-broadcast) |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| LN precision at large mean (suite `big_offset`: fp32, mean 50, std 1, tolerance 0.005) | The FPU truncates srcA/srcB to tf32 (~2⁻¹¹): an fp32 x ≈ 50 loses ~0.009 rms absolute, and a truncated mean shifts each row by up to 0.031, so the result fails at 0.02. E[x²] − mean² cancels. | (1) The variance uses shifted sums with an exactly representable shift K_i = tf32(m_i) (`unary_bcast` output): T_i = Σ(x − K_i)² is exact up to x's own truncation. The combine uses M2 = Σ[T − n(m − K)² + n(m − mean)²] in fp32 SFPU, with no cancellation. (2) Σx on fp32 x uses the `Accurate` (SFPU) reduce. (3) Output centering is in SFPU fp32: x via UnpackToDestFp32, mean via the hi/lo split broadcast. Test `test_layer_norm_distribution[big_offset]` early. |
| `fp32_dest_acc_en=False` vs "partials MUST stay fp32 on the wire and when summed" | 16-bit DEST rounds every DEST-resident sum | Records are always Float32 pages (the wire contract), and the combine runs `Fast` in the caller's 16-bit DEST: the caller chose 16-bit accumulation, and the base op's Refinement 1 tolerances apply. If the verifier reads the rule as binding under False, the remedy is the `deferred` hub-side fp32 combine row. |
| Scatter write is `[unmeasured]` | Phase 0 uses `scatter_chunks = 4` on interleaved outputs | Verify on the 1-tile and focus shapes first. `scatter_chunks = 1` is legal at every shape (knob): it is the example's plain write path. |
| Ports pull from a worker whose kernels already exited | the pull happens after the worker's writer finished | Safe while the program runs: dispatch launches the next program only after every core of this one completes. `cb_out_block` capacity = whole block, so nothing overwrites it. |
| Port and worker CB addresses differ | the port derives the worker's `cb_out_block` address from its own instance | one CB descriptor list on the full core range. Read the base address at kernel start (before any push/pop). |
| Stats readiness without polling remote memory | workers cannot `noc_semaphore_wait` on a hub's L1 | the hub reader multicasts `sem_stats_ready` over the program rectangle. Every core in it is in the program and declares the semaphore (inactive workers included). |
| Counters not back at 0 → second traced call hangs | global semaphores persist across calls | each consumer decrements what it consumed (`0u − n`) + `noc_async_atomic_barrier()`. Ready: one sent / one consumed per port per call. Test: determinism (4 calls) and the output_tensor reuse test. |
| Program-cache aliasing between chips / variants | the generic_op cache does not key per-core RT-arg layouts or semaphore ids (`fabric_ring_exchange` gotcha) | `custom_program_hash` over the plan + CT flags |
| Semaphore ids | `setup_fabric_connection` takes the lowest free ids | declare the op's semaphores first |
| Sharded output L1 addresses on peers | remote pages land in the peer's shard-core L1 | output is mesh-allocated (same address). NoC coordinates are translated (identical logical → NoC on every chip). Verify on HEIGHT_SHARDED early. |
| HEIGHT_SHARDED input on a would-be port core | ports need cores next to Ethernet cores | port candidates exclude input shard cores. Fewer than 2L free → RuntimeError (capacity). |
| W-tail padding (golden poisons padding with 1000) | RMS Σx², LN S1, T | RMS: reader zero-fill (copy-to-scratch for sharded input). LN: zero-fill for S1, plus the reduce `only_partial` mask for T. 1/W and 1/n_i use logical widths. |
| `beta` at its own global column | prompt rule: "never reused per slice" | gamma / beta columns are `p·W_local + local`, the same mapping (sticks: page `p·Wlt + j` when aligned; element-shifted in R_U). `test_layer_norm_gamma_beta_ramp`. |
| R_U element-shifted copies across face boundaries | a 16-column face boundary splits segments on both sides | copy per (row, segment) with both source and destination split at 16. Zero output columns ≥ W in the last output tile. |
| Persistent resources (global semaphores, staging) under trace | creation needs `synchronize_device` once | create on the first call (outside capture), cached per (mesh, port-core union, sizes). The handles are kept alive in a module cache (generic_op keeps only addresses); a cached staging buffer that is no longer allocated is re-created. **Verifier fix:** the stats table is no longer persistent — a per-size persistent L1 table leaked one buffer per distinct table size (top-down L1, colliding with the CB region after a few shapes: 404 golden failures), and the golden conftest's post-failure deallocation of frame-held tensors freed it under the cache. It is now `cb_stats_table` (CB 25). |
| Line ends / G = 2 | receive-only ports; a 2-chip ring is a line | every chip has 2L ports. Ring with G = 2 runs the line schedule. |
| DISPATCH-BOUND | per-core RT args on ~100 workers | worker kernels take common RT args only; core ranges merged |


## Implementation notes (ttnn-implementer, Phase 0)

Built as designed (SF-GN, R_A final-address landing + R_U staging, hubs on the link-0 ports, workers on grid − ports
with common runtime args, relay ring with arrival counters, balanced far slot, scatter packets). Departures, all in
the advisory layer (CB set, helper choices, mechanism details); see `l1_ledger.md` "As built" and `changelog.md`:

| Design item | As built | Why |
|---|---|---|
| Gather: G reads per tile-row, RISC-V scatter of each record into column j | Writers store each (slot, row, stat) block half-split; one contiguous hub read per slot per row; local NoC copies land the records as ROWS of `cb_gather_rows`; compute transposes exactly in DEST (`copy_tile` from an UnpackToDestFp32 CB + `transpose_dest`, raw API — no helper transposes in DEST) | the RISC-V column scatter cost 3.6 µs (focus); reading each record separately from the hubs put ~1300 requests on each hub (48.8 µs) |
| `announce_stats_ready`: `noc_semaphore_inc_multicast` | the hub's sender flags its own reader (local semaphore), the reader (NoC0) multicasts a semaphore *set* over the program grid; one semaphore id per hub | the inc multicast forbids a sender inside the rectangle; the hubs are inside the grid |
| LN split_broadcast (m_hi / m_lo) of the exact mean | centering as `(x − K) − d`, d = mean − K (exact SFPU), K the worker's own exact tf32 shift; the combine's `(m_j − mean)` uses the same identity | no exact fp32 mean broadcast is needed at all (a dataflow broadcast cost ~4 µs) |
| LN `fp32_mode = Accurate` S1 on x | for non-Float32 x an exact Float32 copy `cb_x32` (UnpackToDestFp32) feeds the Accurate reduce and the SFPU centering | the Accurate (SFPU) reduce path is selected only for a Float32 input CB, and a tagged CB cannot be read through srcA/srcB |
| LN T reduce mask in the bf16 scaler CB | `cb_mask` is Float32 | the helper's masked fold reads the mask as srcB in the input CB's format (no reconfig) |
| Port placement by probe program | the Ethernet channel comes from the connection's own runtime args (`setup_fabric_connection(...)[0]`), mapped to the physical Ethernet core; then `fabric_all_gather._worker_nearest_noc1` | no extra dispatch on the first call |
| LN k | capped at 32 / G (one gather tile per stat) | the exact LN combine copies gathered tiles to DEST (UnpackToDestFp32); add_tiles over several gather tiles would read them through srcA/srcB |
| num_links None | min(usable links per hop, 2) | SUPPORTED num_links is {1, 2} |

Not landed in Phase 0: HEIGHT / WIDTH sharded input, `fp32_dest_acc_en=False` (see `changelog.md`).

## Implementation notes (Refinement 1: sharded input, 16-bit DEST)

| Design item | As built | Why |
|---|---|---|
| Worker identity (common RT args only) | sharded input: a packed byte map `lin -> worker index` (shard order) in the common args (`CRT_WORKER_MAP`); interleaved keeps "grid − ports, row-major" | shard grids may be multi-range / column-major; still no per-core RT args (dispatch) |
| Split from the shard spec | HEIGHT: k = 1, one group per shard, ragged last shard → `last_crt` CT arg; WIDTH: one group, k = #shards, only the stats-table cap binds (P_CAP / the LN one-gather-tile rule stay the interleaved split's cost choices). Ragged WIDTH shards only at R = 1 | the shard is the block (zero-copy `cb_x`) |
| Placement | ports chosen from the grid minus the input shard cores (`build_plan(..., excluded)`, cached per exclusion set) | port cores run no worker kernels |
| W tail on a sharded input | the caller's shard is never zeroed. On the tail member every sum over x (RMS x², LN S1 and T) multiplies its block by a **mask block** (`cb_mask`: cw row-0 masks, ones but the tail tile's valid columns, built with `prepare_reduce_mask`) via the row-broadcast DestReuse multiply already used for gamma, then a plain ReduceTile | the helper's AccumulateViaAdd `only_partial` fold of a lone tail tile (cw = 1) accumulates onto an unseeded DEST when crt > 1 (measured: rel RMS 0.14 → 0.016 → pass); and it cannot read an UnpackToDestFp32-tagged x block |
| LN x block on a sharded input | `cb_xs = cb_x` (no `cb_x32` copy, untagged); `cb_ln_sq` bf16 for a 16-bit-mantissa x | bf16 / bf8b x lands in fp32 DEST exactly through srcA; frees `cwt` Float32 tiles next to the caller's resident in/out shards (HEIGHT LN at W_local 1792–2048 did not fit) |
| LN combine with several gather tiles (WIDTH k > 32 / G) | chains over `grid(crt, gather_tiles)` with Strided Block addressing, `n_j` in `gather_tiles` tiles, and the exact ReduceTile (Accurate) collapse the RMS combine uses; one gather tile keeps the Phase 0 AccumulateViaAdd collapse | AccumulateViaAdd's cross-tile `add_tiles` reads the UnpackToDestFp32-tagged `cb_ln_tmp` through srcA (garbage) |
| Sharded output landing | port walk mode `shard_w > 0`: lanes are tile-rows (link l owns rows l, l + L, …), a run is a slot row's columns inside one shard core (split at `shard_w` columns), relay reads the run from the local shard. HEIGHT in + HEIGHT out with one tile-row per shard on the same cores: `cb_out_block` IS the own slice of the output shard (`cb_descriptor_from_sharded_tensor(..., address_offset = p·Wlt·page)`), no own write | runs are long (Wlt pages) so 1–2 runs per packet; in-place output frees `cwt` pages (fp32 HEIGHT 640×1536 needed it) |
| 16-bit DEST | `T_acc` for `cb_sumsq`, `cb_inv`, `cb_rstd_b`, `cb_ln_sq`, `cb_mask`; no UnpackToDestFp32 tags; LN reads `cb_x` directly and every reduce runs Fast; records stay Float32 pages (wire contract); the hub-side fp32 combine stays `deferred` | `helpers.TOLERANCES[(bf16, False)]` holds on the whole gate tier (188/188 interleaved cells) |

Not landed (Refinement 1b): WIDTH_SHARDED at large R × k — the stats table (G·R·k·nstats records) exceeds
`STATS_TABLE_CAP`, or the per-member gather / LN per-row CBs (`crt = R` rows × P partitions) exceed L1.
Measured: forcing an aligned slice through the slot-contiguous staging regime (long fabric runs + tile assembly)
gave 40.3 µs vs 35.5 µs for final-address landing on the focus case: the R_A choice holds on this box.
