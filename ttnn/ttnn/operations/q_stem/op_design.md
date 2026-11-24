# Operation Design: q_stem

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (two matmuls around a cross-device all-to-all over the fabric, plus the rope part's pass-through / rotation) |
| Goal | Turn the TP-gathered q_a latent of a prefill chunk into the attention query in **one** `ttnn.generic_op` dispatch per call: project with this device's q_b_proj column slice, move the TP split from heads to rows (all-to-all over the group along `cluster_axis`, Fabric 2D), absorb wkv_b1 into the nope part, pass through (NoPE) or rotate (RoPE) the rope part, write `(B, H, S/G, N + Dr)`. |
| Math | per group of G devices, position p holds `q_b[p]`: `q[p] = x @ q_b[p]`; `heads[p] = q[p].view(B,S,Hl,Dn+Dr).permute(0,2,1,3)`; device q' receives row quarter q' of every p (output head `h = p*Hl + j`); `out[q'][b,h,s,:N] = nope[q'][b,h,s,:] @ wkv_b1[0,h]`, `out[q'][b,h,s,N:] = rope[q'][b,h,s,:]` (NoPE) or `rope*cos + rotate_half_meta(rope)*sin` (RoPE). |
| References | `eval/golden_tests/q_stem/{feature_spec.py,helpers.py,roofline.py,topology.py}`; transport: `ttnn/ttnn/operations/examples/fabric_all_gather/` (plan, probe placement, fence, arrival counters, chunk walk), `fabric_ring_exchange/` (direct multi-hop sends, `_first_hop_dir`, fence cost), `fabric_reduce_scatter/` (different slice per destination, mux); `tt_metal/third_party/tt_ops_code_gen/references/blackhole-fabric.md`; `.../precision_convention.md`; helpers `ttnn/cpp/ttnn/kernel_lib/{matmul_block_helpers.hpp,untilize_helpers.hpp,eltwise/api/convenience.hpp,mcast/...}`; RoPE math: `ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/kernels/compute/rotary_embedding_llama.cpp` (used by `deepseek_prefill/rotary_embedding_indexed`); catalog `ttnn/ttnn/operations/examples/master.md` entries `shared_input_reuse`, `mcast_topology`, `noc_placement`, `double_buffer`, `fabric_*`. |

Symbols used throughout (all derived on host from the call; **one source each**, see Blocking Model):

| Symbol | Definition |
|--------|------------|
| `G` | mesh extent along `cluster_axis` (=4 on every target ring) |
| `B, S, Q` | latent per-device shape `(B, 1, S, Q)` |
| `H, Dn, N` | `wkv_b1_weight.shape[1:]`; `Hl = H / G`; `Dr = q_b_proj_weight.shape[3] / Hl − Dn` |
| `Q_t, Dn_t, Dr_t, N_t` | `Q/32, Dn/32, Dr/32, N/32` (model dims, whole tiles by contract) |
| `head_w_tiles` | `Dn_t + Dr_t` — one head's column width in q |
| `tensor_q_col_tiles` (`Cq_t`) | `Hl * head_w_tiles` — q's width in tiles on one device |
| `tensor_row_tiles` | `B * ceil(S/32)` — q's height in tiles (per image, ceil) |
| `quarter_row_tiles` | `ceil((S/G)/32)` per batch image; Phase 0 requires `(S/G) % 32 == 0` (alignment = tile_aligned) |
| `tile_bytes` | 2048 for bf16 pages (1088 for bfloat8_b weights) |
| `pkt_tiles` | `floor(ttnn.get_tt_fabric_max_payload_size_bytes() / tile_bytes)` — 3 at 7232 B, 7 at 14400 B |
| `gx, gy` | `device.compute_with_storage_grid_size()` (11 × 10 on the QuietBox 2 / Galaxy Blackhole) |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | ttnn.Tensor (mesh) | yes | per device `(B,1,S,Q)`, TILE, interleaved DRAM/L1, bf16 | — | RT (addresses) |
| `q_b_proj_weight` | ttnn.Tensor (mesh) | yes | per device `(1,1,Q,Hl*(Dn+Dr))`, TILE, DRAM interleaved | — | RT |
| `wkv_b1_weight` | ttnn.Tensor (mesh) | yes | `(1,H,Dn,N)`, same on every device, TILE, DRAM interleaved | — | RT |
| `cluster_axis` | int | yes (kw) | 1 | — | host (plan) |
| `cos`, `sin` | ttnn.Tensor \| None | no | both or neither; per device `(1,1,S/G,Dr)` bf16 TILE DRAM | None (NoPE) | RT; `rope_mode` CT |
| `topology` | `ttnn.Topology` | no | Ring | `Topology.Ring` (import `from ttnn.operations.ccl import Topology`) | host |
| `num_links` | int | no | `0 < L ≤ min over hops of len(get_forwarding_link_indices)`; SUPPORTED {2} | 2 | host (ports) |
| `output_layout` | ttnn.Layout | no | TILE / ROW_MAJOR (Phase 0: TILE) | TILE | CT |
| `compute_kernel_config` | `ttnn.ComputeConfigDescriptor` | no | any fidelity / approx; `fp32_dest_acc_en` gated | `default_compute_kernel_config()` (HiFi4, fp32 DEST, approx off) | CT |

Index axes: none (`cluster_axis` is fixed at 1 in TARGET; no canonicalization needed).

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | latent `(B,1,S,Q)`; q_b_proj `(1,1,Q,Hl*(Dn+Dr))`; wkv_b1 `(1,H,Dn,N)`; cos/sin `(1,1,S/G,Dr)` |
| Dtype | latent bf16; q_b_proj, wkv_b1 bf16 (Phase 0) / bfloat8_b (refinement) |
| Layout | TILE (all) |
| Memory | latent interleaved DRAM (Phase 0) or L1 (refinement); weights + cos/sin DRAM interleaved |

### Output

| Property | Value |
|----------|-------|
| Shape | per device `(B, H, S/G, N + Dr)` |
| Dtype | bf16 |
| Layout | `output_layout` (TILE in Phase 0; ROW_MAJOR refinement written by the op's own writer after an in-kernel untilize) |
| Memory | DRAM interleaved, every device |

## Blocking Model

The first design section. Everything below is downstream of these decisions. Semantics:
`.claude/references/blocking-model.md`.

### Per-chip core roles (fixed by the transport, decided before the split)

| Role | Cores (logical) | Count | Why there |
|------|-----------------|-------|-----------|
| Port cores | in **port row = logical row 0**, each the free row-0 core with the fewest NoC1 hops to its link's Ethernet core (probe as `fabric_all_gather` `probe_ethernet_cores` AG:436 + `_worker_nearest_noc1` AG:581, `allowed` = row 0) | `2 * num_links` (fwd/bwd × link) | blackhole-fabric.md §6 rules 6–8: one producer per (link, direction), directly below its ERISC, sends on NoC1, no other work on the core |
| x injector | one free row-0 core (the middle-most non-port core of row 0) | 1 | multicast sender **outside** the receiver rectangle (`mcast_host.hpp:389` "fixed sender may be inside or outside the receiver grid") |
| Compute rectangle | rows `1 .. gy−1`, all `gx` columns | `C = gx*(gy−1)` (99) | **identical on every chip** (port positions differ per chip, the compute rect does not) — required because the landing scratch is a mesh-uniform L1 sharded tensor over it |

The work split below runs on the compute rectangle — the grid that is left after the ports and the
injector are placed (6 row-0 cores stay idle in Phase 0; see lamp L9).

### Axes

Stage 1 (`q = x @ q_b`), the exchange, and stage 3 (`prefix = nope @ wkv_b1`, rope) each have axes; the
intermediate q and the landed rows are re-blocked (Step-4 rule: every stage's data gets its own rows).

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| S1·rows `m` (q rows = `tensor_row_tiles`, structured as batch × quarter `d` × row-in-quarter) | independent (each q row is its own dot products) — but it is the **send axis**: quarter `d` is the destination chip | `s1_block_m_tiles` | **1** — the block is the exchange's send unit; every remote direction must have data in flight from the first block (overlap lamp L1 records the alternative `quarter_row_tiles`) | host const `S1_BLOCK_M_TILES` → CT arg of injector/compute/writer | **not split across cores**: every stage-1 core walks all row blocks in the **send order** (below) in lockstep with the x multicast | knob-turn |
| S1·cols `n` (`Cq_t`) | independent (each column is its own dot products) | `core_cols` (per core, ragged) | `split_work_to_cores(C1, Cq_t)`: `ceil` or `floor` of `Cq_t/C1` (Kimi 2 / 1) — the whole per-core assignment is ONE block along n | `split_work_to_cores` on host → RT arg per core | **primary split**: columns over the stage-1 rectangle `W1 × R1`, `W1 = min(gx, Cq_t)`, `R1 = min(gy−1, max(1, floor(Cq_t/W1)))`, `C1 = W1*R1` (every stage-1 core owns ≥ 1 column) | knob-turn (core count) |
| S1·K (`Q_t`) | dependent (results span K: accumulation) | `s1_kb` | largest divisor of `Q_t` ≤ `X_ROUND_TILES_CAP` (=64): whole K for every INPUTS shape except Q=4096 (2 K-blocks) | host const `X_ROUND_TILES_CAP` → derived CT arg | not split (whole K per core; accumulated in DEST, spill to `cb_q_partial` only when `Q_t/s1_kb > 1`) | scheme-change (K-split, deferred regime R6) |
| S1·x operand | **reuse-shared** by construction of the n split (x does not vary along n — operand-reuse check) | x round = `s1_block_m_tiles × s1_kb` tiles | 48 tiles (Kimi) | derived | read once from DRAM/L1 by the injector, **multicast** to the stage-1 rectangle | knob-turn (injector count, lamp L3) |
| S1·q_b operand | independent along n (each column read by exactly one core); reused across all row blocks of that core | resident slice `Q_t × core_cols` | whole slice resident (read once) | derived from `core_cols` | follows the column split | — |
| X·quarter `d` (exchange) | independent — each destination's rows are disjoint; this is the all-to-all's destination axis | segment = one head-row (`head_w_tiles` tiles at the destination) | 1 head-row per segment, `ceil(head_w_tiles/pkt_tiles)` packets | derived from `pkt_tiles` | per remote destination chip: the port(s) in the first-hop direction (`_first_hop_dir`, RX:57); link chosen by the **destination core** (`link = k mod L`) so each arrival counter has exactly one sending stream | TRANSPORT knob-turns (lamps L4–L6) / relay = deferred R4 |
| S3·heads `h` (H) | independent (per-head matmul with its own wkv_b1[h]) | `heads_per_core` (`hpc`) | `ceil(H/C)` (Kimi 1, DSV3.2 2) | `split_work_to_cores(C, H)` → RT | **primary split** of stage 3: heads over **all** compute cores, filled from the cores with the least stage-1 work first | knob-turn |
| S3·rows (`B*quarter_row_tiles` per head) | independent (rows of the landed quarter are separate output rows) | `s3_block_m_tiles` | **1** — rows land progressively; a row is processed as soon as its arrival count covers it (overlap; lamp L2) | host const → CT | not split (the head owner processes all rows of its heads) | knob-turn |
| S3·N (`N_t`) | independent (prefix columns) | `s3_block_n_tiles` | `N_t` when `wkv_resident` (R1), else largest divisor of `N_t` ≤ `S3_BLOCK_N_CAP` (=16) (R2) | host predicate → CT | not split across cores in Phase 0 | scheme addition (N split across cores, deferred R8) |
| S3·Dn (K of stage 3) | dependent (accumulation) | — | whole `Dn_t` (one K-block, ≤ 6 tiles) | — | not split — Dn ≤ 192 makes splitting pointless | — |
| S3·wkv_b1 operand | independent along heads; reused across all rows of a head | resident per head | `min(2,hpc)` heads resident (R1) / 2 N-blocks streamed (R2) | derived | follows the head split; prefetched during stage 1 (stall shadow) | — |
| S3·rope columns (`Dr_t`) | independent (elementwise) | `Dr_t` per row block | whole `Dr_t` | — | follows the head split | — |

**Send order** (one source: host function `send_order(p)`, shared by injector, stage-1 writers, ports and
the stage-3 arrival thresholds):
`[(b, d, r) for b in range(B) for r in range(quarter_row_tiles) for d in remote_order(p)] + [(b, p, r) for b for r]`,
`remote_order(p) = [p+1, p−1, p+2, p−2, …] mod G` (ring distance 1 both ways, then 2, …; for G=4: p+1, p−1, p+2).
x row tile of block `(b, d, r)` = `b*ceil(S/32) + d*quarter_row_tiles + r`. Remote rows first, interleaved across
destinations (every direction busy from the first block); the own quarter last (it needs no fabric and feeds the
local heads that fill the stage-3 wait — stall-shadow check below).

**Stall-shadow check.** Stages that wait on something they do not produce: (a) a port's first send waits on the
ready fence → stage 1 (x multicast, q_b reads, compute) and the wkv_b1 prefetch need no remote data and run
under it; (b) a stage-3 row waits on its remote arrival → filled by the own-quarter stage-1 blocks (scheduled
last for that reason), by the wkv_b1 prefetch (issued between x rounds), and by the **local heads**, which are
consumed first on any core that owns both (they are complete when stage 1 ends). The permission to process a head
row-by-row is algebraic: `prefix` rows are independent dot products (`(P·nope) @ W = P·(nope @ W)` for any row
selection P), and the rope part is elementwise — no floating-point order changes (each tile's K accumulation is
unchanged), so no precision gate applies.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_x_round` | `x_round_depth` | 2 rounds | injector reads round r+1 from DRAM while round r multicasts; receivers compute round r while r+1 lands |
| `cb_qb_slice` | — (resident, 1) | 1 | read once, reused by every row block |
| `cb_q_slice` | — (resident: all row blocks of the core) | `tensor_row_tiles` blocks | no ack path from ports back to stage-1 cores; q stays on chip (roofline assumption) |
| `cb_port_stage` | `port_batch_segments` | 2 × 4 segments | port reader pulls the next 4 head-row segments while the sender sends 4 (fabric_all_gather: 4 chunks per batch, 2 batches) |
| `cb_landing` | — (resident: every row this core receives) | all | remote writes never wait on the consumer (no credit return over the fabric) |
| `cb_wkv` | `wkv_depth` | `min(2,hpc)` heads (R1) / 2 N-blocks (R2) | next head's (next N-block's) weights stream in while the current one multiplies |
| `cb_prefix_out` | `out_depth` | 2 × `s3_block_n_tiles` | writer drains row block i while compute produces i+1 |
| `cb_rope_out` | `out_depth` | 2 × `Dr_t` | same |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Fabric max payload (router, set at mesh open) | packet ≤ `pkt_tiles` tiles | `pkt_tiles = floor(get_tt_fabric_max_payload_size_bytes()/tile_bytes)`, never hard-coded | router rejects / truncates the packet → corrupt landing or hang |
| Packet contiguity at the destination | a packet covers tiles contiguous in **one** destination head-row segment | packets per segment = `ceil(head_w_tiles / pkt_tiles)`; never span two segments (different dest cores/offsets) | tiles land in the wrong head / row |
| Arrival-counter stream ordering | each `(dest core k, source p)` counter must be incremented by exactly **one** in-order stream | remote: link = `k mod L`, direction = first hop of the route p→q'; local (p = q'): the consumer waits for the **total** local count of the head (all rows) before consuming any row | a per-row threshold reached by another stream's increments → stage 3 reads unlanded rows (silent wrong results) |
| `PacketHeaderPool` (≤ 12 headers per RISC, RX port.cpp:68) | pre-routed headers per port sender | one header per (destination chip) + `port_batch_segments` data headers ≤ 12 | header pool overflow |
| Multicast transaction size (≤ 16 KiB per multicast, estimate.py refusal) | x round transfer | a round is sent as ≤ 16 KiB linked transactions by the Mcast helper | refused / split by the NoC |
| `matmul_block` DEST capacity (`out_subblock_h*out_subblock_w ≤ DEST_AUTO_LIMIT`, `matmul_block_helpers.inl:158`; 4 tiles fp32 half-sync) | stage-1 subblock within `1 × core_cols`; stage-3 subblock within `1 × s3_block_n_tiles` | subblock factors chosen by the implementer/auto-tuner from the block size + `fp32_dest_acc_en` | helper assert |
| x round L1 | `s1_block_m_tiles*s1_kb ≤ X_ROUND_TILES_CAP` (64) | `s1_kb` = largest divisor of `Q_t` ≤ cap / `s1_block_m_tiles` | L1 OOM at deep Q |
| wkv_b1 residency | `min(2,hpc)*Dn_t*N_t*tile_bytes(wkv) ≤ WKV_RESIDENT_BYTES` (256 KiB) | regime R2: `s3_block_n_tiles` ≤ 16, streamed | L1 OOM (wide N: 1 MiB per head) |
| Per-core L1 total (ledger total) | all resident terms | host predicate `fits_l1`; RuntimeError naming the term if it fails (no INPUTS shape fails: max 1032 KiB, long chunk) | program build OOM |
| Tile alignment of the quarter | `(S/G) % 32 == 0` for whole-tile segments | `alignment = tile_aligned` (SUPPORTED); m_non_aligned = deferred R9 | segments would straddle tiles → wrong rows |
| Runtime-arg count per core | port send lists, threshold tables | per-port descriptors are **regular** (per destination: a head list + row stride); the device derives per-row offsets; a core-coordinate table (≤ C words) is passed once | args overflow the per-core limit → truncated plan |

### Regimes

Named boundary for "minimum": **each input crosses DRAM once, the output crosses DRAM once, q never touches
DRAM** (the roofline's model), and each remote q tile crosses the fabric along its shortest ring path once.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **R1 fused · column-split stage 1 · direct exchange into L1 landing · wkv-resident stage 3** | **built** | `alignment == tile_aligned ∧ fits_l1 ∧ wkv_resident` where `wkv_resident := min(2,hpc)*Dn_t*N_t*tile_bytes(wkv) ≤ 256 KiB` | S1: `s1_block_m_tiles × core_cols × (Q_t in s1_kb K-blocks)`; X: one head-row segment; S3: `(head, s3_block_m_tiles rows, N_t cols)` + `Dr_t` rope | **minimum** on DRAM (x 1×, q_b 1×, wkv_b1 1×, out 1×; q 0×). NoC above minimum: x multicast fan-out `C1`, own-quarter local delivery (q/G), port pulls of remote q (3q/4). Fabric: the antipodal quarter takes the routing table's direction (not balanced; see R4). | S1: one x round (one handshake, one multicast reservation), one matmul re-init and one port block-ready increment per block — a larger `s1_block_m_tiles` amortizes these over more rows but delays the first send. S3: one matmul init + one rope copy init + one writer barrier per row block. Intended frequency: matmul init once per stage per core (helper `InitMode::Short` per call is accepted), one x handshake per round, one fused arrival increment per `inc_every_packets` packets per destination core and on each segment-final packet that completes a threshold row, writer barrier per output row block. |
| **R2 fused · as R1 · wkv N-streamed stage 3** | **built** (wide-N INPUTS shape `N=4096` needs it) | `alignment == tile_aligned ∧ fits_l1 ∧ ¬wkv_resident` | S3: `(head, s3_block_m_tiles rows, s3_block_n_tiles cols)`; nope row retained across the N-blocks | wkv_b1 crosses DRAM `B*quarter_row_tiles/s3_block_m_tiles` times (=1 for the only INPUTS shape here: S/G = 32 rows); everything else as R1 | as R1, plus one wkv_b1 N-block read per (row block, N-block) — a larger `s3_block_m_tiles` divides the wkv re-read factor |
| R3 stage-1 streaming (q_b not resident, q ring with port acks) | deferred — every INPUTS shape satisfies `fits_l1` (max 1032 KiB per core); a shape beyond it raises RuntimeError naming the term. Reachable: the per-block port hand-off already exists; a credit semaphore (port → stage-1 writer) turns `cb_q_slice` into a bounded ring without touching the loop nest | `¬fits_l1` | S1 as R1 with `cb_q_slice` = `q_depth` blocks | q_b re-read per quarter (G×) if the slice cannot be resident | — |
| R4 balanced relay of the antipodal quarter (half each way, store-and-forward through the neighbour's Tensix) | deferred (TRANSPORT) — measured on this ring: the 4-ring all-to-all of q_nope took 53–56 µs at 7232 B "whether the far quarter went direct or every packet was a 1-hop send" (prompt), so the built direct path covers it at equal measured cost with no relay storage. Reachable: the port send list is host-built; relay = extra entries + an arrival-gated relay reader on the middle chip's port (fabric_all_gather relay, AG:149) | Galaxy torus where the route table sends every antipodal quarter the same way | relay segment | busiest directed link: direct worst case 3 quarter-crossings vs balanced 2 (roofline `q/2`) ; relay adds a store-and-forward hop | — |
| R5 2-band stage 1 (rows split in 2 bands, x multicast per band, q_b read twice) | deferred — op model (non-designs mode) ranks it 62.2 µs vs 68.9 µs one band, but the `--designs` search (linked resident multicast, the scheme R1 builds) puts one band at 57.1 µs (73 cores) and it moves +7.5 MB (bfp8) / +14.2 MB (bf16) of q_b DRAM traffic on a DRAM-bound op. Reachable: band = a second x-multicast family + band offset in the send order | — | S1 per band | q_b 2× DRAM | — |
| R6 K-split stage 1 (Q split across 2 core groups, partial sums combined over the NoC) | deferred — op model `--designs` best: 50.3 µs on 110 cores vs 57.1 µs for R1's split (−12%); stage 1 is hidden behind the DRAM/transport bound in the perf focus (DRAM roofline 73.8 µs, exchange ~55–75 µs), and the combine adds `q` fp32 partials (11.8 MB) of NoC traffic. Reachable: a combine block op between `s1_matmul_block` and `s1_handoff_block` | — | S1 `(block_m, core_cols, Q_t/2)` | + partial-sum NoC traffic | — |
| R7 q through DRAM (stage 1 writes q, ports read it like fabric_all_gather) | **rejected** — dead end: +4.4 MB write + 4.4 MB read per chip (~+17 µs of DRAM time on a DRAM-bound op); superseded everywhere by R1's L1-resident q pulled by the ports | — | — | q 2× DRAM | — |
| R8 stage-3 N split across cores (head × N-group units) | deferred — only the wide-N / small-H shapes under-fill (H=16 heads on 99 cores); stage-3 FLOPs there are 0.5 GFLOP. Reachable: an N-group sibling pulls the owner's landed nope rows over the NoC after the owner's arrival count covers them | `H < C/2` | S3 `(head, rows, N-group)` | nope rows re-read over NoC per sibling | — |
| R9 m_non_aligned rows (`(S/G) % 32 ≠ 0`) | deferred (alignment refinement) — not in Phase 0 SUPPORTED. Mechanism: the source sends the (up to 2) tile rows that cover the quarter; the destination realigns rows with a 0/1 row-shift matrix applied by matmul (`out_tile = U_δ·tile_a + D_δ·tile_b`, exact), applied to nope and rope **before** stage 3 — legal because a left row permutation commutes with `@ wkv_b1` and with the elementwise rope | `alignment == m_non_aligned` | as R1 + shift block op | +≤ 2 boundary tile rows per quarter on the fabric | — |
| R10 every stage-1 core reads x itself | **rejected** — `C1×` x DRAM traffic (99 × 1.97 MB); superseded by the x multicast (stepping stone not needed: the multicast is in Phase 0) | — | — | x 99× DRAM | — |
| R11 (head × quarter) 2-D stage-1 split, q_b read per quarter band | **rejected** — G× q_b DRAM (57 MB bf16), and all quarters finish together (no early send); superseded by R1 | — | — | q_b 4× | — |

Regime selection function (host, exact): `regime = R1 if wkv_resident else R2`, after asserting
`alignment == tile_aligned` (validate) and `fits_l1` (RuntimeError otherwise). R1 and R2 are one code path with
`s3_block_n_tiles` either `N_t` (weights loaded once per head, retained across rows) or `< N_t` (weights streamed
per row block); **regime-pinned tests are required**: `kimi_k31_640` (R1) and `wide_n_streamed_wkv` (R2) in the
acceptance test, plus `deep_q_k_blocked` pinning the `Q_t/s1_kb = 2` K-blocked stage 1.

### Traffic ranking

Per chip, Kimi K3.1 S=640, bf16 Phase 0 (model config bytes in parentheses). DRAM crossings first, then cross-core
NoC, then fabric.

| Candidate split | DRAM (per chip) | Cross-core NoC | Fabric | Verdict |
|-----------------|-----------------|----------------|--------|---------|
| **S1 columns over cores, x multicast, q on chip (R1)** | x 1.97 + q_b 14.16 (7.52) + wkv 12.58 + out 17.69 = **46.4 MB (37.8)** — the minimum | x mcast 1.97 MB × 99 receivers; port pulls 4.42 MB; local quarter 1.47 MB | 4.42 MB sent | **chosen** — minimum DRAM, compute stays on 72-effective cores (op-model curve flat from 73 cores) |
| S1 rows in 2 bands (R5) | +q_b 1× (+14.16 / +7.52 MB) | x mcast per band (half fan-out each) | same | deferred: more DRAM bytes on a DRAM-bound op |
| S1 K split + combine (R6) | minimum | + fp32 partials 11.8 MB | same | deferred: more NoC bytes; op model −12% on stage 1 alone |
| S1 (head × quarter) 2-D (R11) | +q_b 3× (+42.5 MB) | x per band | same | rejected |
| q through DRAM (R7) | +8.8 MB | — | same | rejected |
| Exchange direct (R1) vs relay (R4) | relay via DRAM: +2.95 MB; via L1: 0 | relay: +1 quarter pulled on the middle chip | same bytes; per-direction balance differs | direct built (measured parity on this ring) |

Op model, stage 1 alone (the matmul it covers; stage 3 is a batched per-head matmul the op model does not
express — 2.01 GFLOP, 13.2 µs at HiFi4 on 110 cores, counted in the roofline):

```
python3 .claude/tools/perf_calculator/op_model.py matmul 20 48 144 --fidelity HiFi2 --b-dtype bfp8_b --bands 1,2,4 --kb 4,8,16 --in0 dram,mcast --top 8
python3 .claude/tools/perf_calculator/op_model.py matmul 20 48 144 --fidelity HiFi2 --b-dtype bfp8_b --designs --c-mem l1 --kb 4,8,16 --top 8 --workers 16
```

| Rank (`--designs`, c in L1) | µs | cores | bands | ksplit | kb | in0 | lower bounds (DRAM / compute) |
|------|----|-------|-------|--------|----|-----|------|
| 1 | 50.3 | 110 | 1 | 2 | 4 | mcast (resident, linked) | 18.5 / 36.4 |
| 2–8 | 50.3–51.7 | 110 | 1 | 2 | 4 | mcast | same |

Curve (best µs per core count, `--designs`): 49 → 82.2, 64 → 82.0, **73 → 57.1**, 77 → 57.9, 81 → 57.2, 91 → 57.2,
99 → 57.7, 110 → 50.3 (K-split). Without K-split the curve is **flat from 73 cores** — exactly where every core
owns ≤ 2 of the 144 columns. Phase 0 takes R1's column split over the 99-core compute rectangle (max 2 columns
per core, same time as 73 cores; the extra cores own 1 column and start stage 3 earlier) and records the K-split
as R6. Non-designs mode shortlist: 2 bands × 51 cores 62.2 µs (q_b read twice) vs one band 68.9 µs — R5. Per-core
launch cost (~0.65 µs/core one-call-at-a-time) is hidden in a traced model and does not change the choice at
this op size (~100 µs).

### Data-movement search

`estimate.py sweep … --record ttnn/ttnn/operations/q_stem/dm_estimates.json --phase Planner` (sweep files and
full shortlists in `l1_ledger.md` → "Data-movement search"). Phase 0 takes rank 1 or its smallest-footprint tie:

| Group | Phase 0 knobs | `pass_ns` (rank 1) | Note |
|-------|---------------|--------------------|------|
| q_b slice read | NoC0, batch 8, 99 stage-1 cores | 32 795 (96 cores, batch 4) | all four shortlisted tie |
| wkv_b1 per-head read | NoC0, batch 16 | 29 582 | all tie |
| output write | NoC1, batch 9 | 41 668 | all tie |
| x multicast | 16 KiB linked transactions, 2-round depth, NoC0 | 89 171 (unlinked) | disagrees with the op model's linked design (57 µs stage 1) → lamp L3 |

### Block schedule

Logical schedule per chip (reader / compute / writer / port / injector kernels run concurrently; adjacent blocks
pipeline):

```cpp
// injector (row 0)
for (uint32_t block_idx = 0; block_idx < num_blocks_s1; ++block_idx)   // send order
    x_round_multicast_block(block_idx);          // s1_block_m_tiles x s1_kb tiles per round, Q_t/s1_kb rounds

// every stage-1 core (compute rectangle W1 x R1)
qb_slice_load_block();                            // Q_t x core_cols, once
for (uint32_t block_idx = 0; block_idx < num_blocks_s1; ++block_idx) {
    s1_matmul_block(block_idx);                   // (s1_block_m_tiles x Q_t) @ (Q_t x core_cols) -> cb_q_slice
    s1_handoff_block(block_idx);                  // remote: bump port block-ready; own quarter: deliver to local landing
}

// every port core (row 0): realizes the exchange for its (direction, link)
ready_fence_block();                              // readies out to every chip that writes into me, wait for mine
for (uint32_t seg_idx = 0; seg_idx < num_segments_this_port; ++seg_idx)
    send_segment_block(seg_idx);                  // pull head-row from source cores, packetize, fused arrival inc

// every compute core with heads (stage 3), heads ordered: local first, then remote by expected completion
for (uint32_t head_idx = 0; head_idx < heads_this_core; ++head_idx) {
    wkv_load_block(head_idx);                     // R1: Dn_t x N_t once per head (prefetched during stage 1)
    for (uint32_t row_block = 0; row_block < num_row_blocks; ++row_block) {
        land_row_block(head_idx, row_block);      // wait arrival threshold, hand the landed tiles to compute
        prefix_matmul_block(head_idx, row_block); // (s3_block_m x Dn_t) @ (Dn_t x s3_block_n) [x N_t/s3_block_n]
        rope_block(head_idx, row_block);          // NoPE: pass-through copy of Dr_t tiles; RoPE: rotate
        store_row_block(head_idx, row_block);     // TILE: tiles to output; ROW_MAJOR: untilized sticks
    }
}
rearm_counters();                                 // every consumer returns its counters to 0
```

| Block operation | Block shape | Resident across it | Intended fixed-cost frequency |
|-----------------|-------------|--------------------|-------------------------------|
| `x_round_multicast_block` | `s1_block_m_tiles × s1_kb` tiles | injector's 2-round buffer | one DRAM/L1 read batch + one Mcast `send` (handshake + linked ≤16 KiB transactions) per round; `Q_t/s1_kb` rounds per block |
| `qb_slice_load_block` | `Q_t × core_cols` | — (becomes resident) | once per call per core |
| `s1_matmul_block` | M=`s1_block_m_tiles`, K=`Q_t` (in `s1_kb` K-blocks), N=`core_cols` | q_b slice, `cb_q_slice` (all blocks) | matmul (re)init once per block (helper `Short`); x CB wait/pop per K-block; spill only if `Q_t/s1_kb > 1` |
| `s1_handoff_block` | the block's `s1_block_m_tiles × core_cols` tiles | `cb_q_slice` (never popped before the end) | remote: one NoC atomic per interested port per block; own quarter: one NoC write per (head-row piece) + one atomic per piece's dest core |
| `ready_fence_block` | — | — | once per call per port: one header-only ready per chip that writes into this chip through this port's link/direction; wait for this port's own readies |
| `send_segment_block` | one head-row segment (`head_w_tiles`) for one dest core | port staging batch | NoC0 pulls (one per source run), `ceil(head_w_tiles/pkt_tiles)` packets, fused increment on every `inc_every_packets`-th packet to that dest core and on threshold rows; `noc_async_writes_flushed` once per batch |
| `wkv_load_block` | R1: `Dn_t × N_t`; R2: `Dn_t × s3_block_n_tiles` | R1: head's weights for all its rows | R1 once per head (issued between x rounds); R2 once per (row block, N-block) |
| `land_row_block` | `s3_block_m_tiles × head_w_tiles` | landing scratch | one semaphore wait per row block (remote) / one wait per head (local); credit push, no data move |
| `prefix_matmul_block` | M=`s3_block_m_tiles`, K=`Dn_t`, N=`s3_block_n_tiles` | wkv (R1) | matmul init per row block |
| `rope_block` | `s3_block_m_tiles × Dr_t` | (RoPE: cos/sin rows of the quarter + transformation tile, once per core) | copy init per row block |
| `store_row_block` | `s3_block_m_tiles × (N_t + Dr_t)` tiles (TILE) / `32*s3_block_m_tiles` sticks (RM) | — | one write barrier per `out_batch` (9) tiles |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| L1 overlap: `s1_block_m_tiles = 1` | M=1 blocks halve in0 reuse in the matmul and pay one x round per row; the exchange may not need sends this early at HiFi4 | `s1_block_m_tiles = quarter_row_tiles` (5) with the same interleaved destination order (OVERLAP) |
| L2 overlap: `s3_block_m_tiles = 1` | per-row matmul init + writer barrier ×5 per head | `s3_block_m_tiles = quarter_row_tiles` once arrival is per quarter (OVERLAP) |
| L3 x multicast injector count | `estimate.py` puts an unlinked 16 KiB-transaction multicast of x to 77 cores at 89 µs (path reservation bound), above stage-1 compute; the op model's linked resident multicast at 57 µs | 2 rotating injectors (`Mcast2DRotatingSenderConfig`, `mcast_host.hpp:380`) on different NoCs |
| L4 arrival increment interval `inc_every_packets = 8` | Kimi head-rows are 2 packets: one increment per row = every 2nd packet halves the receiving router's rate (blackhole-fabric §6 rule 4); every 8th delays stage-3 start by ~4 rows | increment per row (`1`) vs per quarter (TRANSPORT/OVERLAP) |
| L5 port batch `port_batch_segments = 4` | fabric_all_gather's 4-chunk batch was measured at 14 KiB packets; at 7232 B a segment is 2 packets | 2 / 8 segments (TRANSPORT) |
| L6 direct antipodal route | balance set by the routing table; on a Galaxy torus row the busiest direction may carry 1.5× | R4 relay (TRANSPORT, measured in the op) |
| L7 stage-1 core count | full-rect ragged split (99 cores, max 2 cols) vs `C1 = ceil(Cq_t/2) = 72` cores freeing 27 cores to start stage 3 (output writes) at t≈0 | 72-core stage-1 rectangle (grid-synchronization lamp: fewer participants per x round) |
| L8 head placement weighting | heads land last on cores whose stage 1 ends last | weight `hpc` toward stage-1-free cores |
| L9 idle row-0 cores | 6 row-0 cores do nothing | use them as extra x injectors (L3) or stage-3 owners (they are outside the landing scratch's rect — needs a second shard grid) |
| L10 dm shortlist: q_b read `batch=8`, wkv `batch=16`, out `batch=9` (ties) | ranks are ties within 10% | rank-1 alternatives in `dm_estimates.json` (q_b batch 4 @96 cores, wkv batch 64) |

Catalog entries that informed the knobs: `shared_input_reuse` / `mcast_topology` (x multicast instead of per-core
re-reads; 1-D column split needs a 2-D rectangle multicast), `noc_placement` (reads on NoC0, writes on NoC1; port
pulls on NoC0, fabric sends on NoC1), `double_buffer` (depth 2 on streaming CBs), `fabric_all_gather` (probe
placement, increment every 8 chunks, 4-chunk batches, ring over line, 2 links), `fabric_ring_exchange` (direct
multi-hop sends cost +1.3 µs of fence only; ~5 µs fence floor), `fabric_reduce_scatter` (a different slice per
destination; mux only for >1 worker per link — not needed: one port per link/direction pulls from all sources).

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| x → injector | bf16 tiles (DRAM or L1 interleaved) | `TensorAccessor` reads (NoC0) in send-order row tiles, one round = `s1_block_m_tiles × s1_kb` tiles | latent in L1 (model config) uses the same accessor over L1 banks |
| injector → stage-1 cores | bf16 | `ttnn.Mcast` 2-D, fixed sender outside rect `W1×R1`; `McastArgs::sender(noc).send(src, dst, size)` (`mcast_pipe.hpp:74`), receivers `receive(round)` | `cb_x_round` is declared on rect ∪ injector so `dst = get_write_ptr` is the same address everywhere |
| q_b → stage-1 cores | bf16 / bfp8 tiles | each core reads its `Q_t × core_cols` slice (page `k*Cq_t + c`), NoC0, `batch` 8 | read once, resident |
| stage-1 compute | DEST fp32 (Phase 0) → q packed **bf16** | `matmul_block` | q stored as the exchanged format; bit-exact from here on (never re-rounded) |
| q → port (remote quarters) | bf16 | port reader NoC0 `noc_async_read` from the source cores' `cb_q_slice` (same CB address: CB declared on rect ∪ ports) into `cb_port_stage`, gated by the port's block-ready semaphore | segment = one head-row, assembled contiguous in staging |
| port → remote chip | bf16, `pkt_tiles` per packet | `WorkerToFabricEdmSender` (NoC1), `fabric_set_unicast_route(hdr, chip, mesh)` (2D, first hop = port direction, also for 2-hop destinations), `to_noc_unicast_write` / `to_noc_fused_unicast_write_atomic_inc` (AG:277-284) into the destination core's landing address | ready fence first (AG:242-255, RX port.cpp:156-169 for multi-chip readies) |
| q → local landing (own quarter) | bf16 | stage-1 writer `noc_async_write` (NoC1) to the dest core's landing + `noc_semaphore_inc(arrive[k][p])` after `noc_async_writes_flushed` | no fabric, no fence |
| landing → stage-3 compute | bf16 | `cb_landing` **backed by the landing scratch shard** (`ttnn.cb_descriptor_from_sharded_tensor`); reader waits `arrive[k][p] ≥ threshold`, then `cb_push_back(head_w_tiles × s3_block_m)` (credit only, zero-copy) | consumption order = landing layout order |
| wkv_b1 → stage-3 | bf16 / bfp8 | reader `TensorAccessor` reads of head h's contiguous `Dn_t × N_t` pages (R1) or N-block columns (R2) | issued between x rounds (prefetch) |
| stage-3 compute | DEST → bf16 | `matmul_block` (prefix) + `copy` (NoPE rope) → `cb_prefix_out`, `cb_rope_out` | RoPE: transformation-tile matmul + mul/add (refinement) |
| → output | bf16 TILE (Phase 0) / RM sticks | writer NoC1 tile writes at `((b*H + h)*quarter_row_tiles + r)*(N_t+Dr_t) + c` | RM: untilize in compute, sticks of `(N+Dr)*2` B, prefix and rope as two segments of each page |

**Fabric protocol (TRANSPORT), per call, trace-safe:**

| Item | Where | Protocol |
|------|-------|----------|
| Global semaphores (cached per mesh device, created once with `ttnn.create_global_semaphore(mesh, crs, 0)` + one `synchronize_device` at creation, as AG:1099) | `ready` on port row; `arrive[0..G−1]` on the compute rect | address passed as RT args; identical on every chip/core |
| Program semaphores | `block_ready` on port cores (intra-chip) | `SemaphoreDescriptor`, `get_semaphore(id)` |
| Ready fence | each port that writes into chip X waits for one ready from X per call; chip X's port in direction `first_hop(X→writer)` with the writer's link sends a header-only `to_noc_unicast_atomic_inc` | host computes per port: readies to send (targets) and readies to expect; the port consumes with `noc_semaphore_inc(ready, 0u − n_expected)` (RX port.cpp:166) |
| Arrival | `arrive[k][p]` on dest core k: remote increments fused on packets; local increments by stage-1 writers | consumer (stage-3 reader of k) re-arms `arrive[k][p] −= total_expected` after its last wait (AG:297-302) |
| Block-ready | stage-1 writers `noc_semaphore_inc(port, 1)` per block for each port that carries a column of that block's destination | port re-arms at end |
| Mcast handshake | `ttnn.Mcast` helper semaphores | helper resets per round |
| Scratch reuse | landing scratch written by remote chips only after the fence; no call reads a previous call's rows (thresholds are re-armed totals) | — |
| `custom_program_hash` | over shapes, dtypes, layouts, fidelity/fp32, rope_mode, payload size, num_links, plan (port coords, routes), all knob values | generic_op caches on per-core arg-grid size only (prompt note; RX:224-229) |

**TARGET placement axis.** `memory_layout` ∈ {INTERLEAVED} only, so no physical-shard scheme is in TARGET. The
latent's `input_buffer = L1` is a source change for the injector's accessor (knob-turn, no new algorithm). The
landing scratch is an internal L1 height-sharded scratch tensor (consumed in place through a CB backed on it —
never re-read over the NoC).

**Refinement mechanisms that this structure keeps reachable** (each is a value of a TARGET axis):

| TARGET value | Change | Class |
|--------------|--------|-------|
| `q_b_proj_dtype = bfloat8_b`, `wkv_b1_dtype = bfloat8_b` | in1 page format Bfp8_b in `cb_qb_slice` / `cb_wkv`, accessor page size 1088 B; L1 terms shrink | knob-turn |
| `fp32_dest_acc_en = False` | `cb_q_partial` page Float16_b; subblock capacity 8 | knob-turn |
| `output_layout = ROW_MAJOR` | stage-3 adds `untilize_row_block` (`untilize<s3_block_n_tiles,…>` on `cb_prefix_out`, `untilize<Dr_t,…>` on `cb_rope_out`) into RM CBs; writer writes stick segments | scheme addition (new block op, same nest) |
| `rope_mode = rope` | `rope_block` = `rope @ trans_tile` (matmul, `SrcOrder::Reverse`, as rotary_embedding_llama.cpp:120-131) then `*sin`, `+ rope*cos` (eltwise `mul`/`add`, `convenience.hpp:45,51`); cos/sin rows of the quarter (`quarter_row_tiles × Dr_t` each) read once per core; the transformation tile generated on device by the reader (no host tensor) | scheme addition |
| `alignment = m_non_aligned` | R9 | scheme addition |
| `input_buffer = L1` | injector accessor over L1 | knob-turn |

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | S1: one row block × the core's columns; X: one head-row segment; S3: one (head, row block) |
| Grid | per chip: port row 0 (`2L` ports + 1 injector), compute rect rows `1..gy−1` (C = 99); stage-1 rect `W1 × R1` ⊆ compute rect anchored at logical (0,1) |
| Per-core work | S1 columns: `split_work_to_cores(C1, Cq_t)` (Kimi: 45 cores × 2, 54 × 1); S3 heads: `split_work_to_cores(C, H)` assigned starting from the cores with the fewest stage-1 columns (Kimi: 96 cores × 1 head) |
| Remainder | ragged columns/heads use the same block ops with a smaller runtime extent; CB push/wait counts stay nominal per core; tile counts use `ceil` per image (`tensor_row_tiles = B*ceil(S/32)`); quarter rows require `(S/G)%32 == 0` in Phase 0 |
| Head → dest core map | `k3(h)` identical on every chip (host function); landing offset of `(u, b, r, c)` on core k = `((u*B + b)*quarter_row_tiles + r)*head_w_tiles + c`, heads `u` in the core's consumption order (local first) |
| Link / direction of a remote segment | direction = first hop of the route `p → q'` (`ttnn.get_eth_forwarding_direction` compared with the ring neighbours, RX `_first_hop_dir` :57); link index = `ttnn.get_forwarding_link_indices(src, first_hop_neighbour)[k3 mod L]` |
| Regime selection | `R1 if wkv_resident else R2` (exact predicate in Regimes); regime-pinned tests in the acceptance test |

## Circular Buffers

Indices are any distinct slots; listed in **allocation order** (address identity matters for the two CBs other
cores address).

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|----------|
| `cb_q_slice` | 0 | tile (bf16) | `tensor_row_tiles × core_cols_max` | spans S1·rows (all blocks, resident) × S1·cols; declared on compute rect ∪ ports so the ports' `get_write_ptr` = sources' address (ports never push/pop it) | Float16_b | compute | writer (stage-1 core) | whole call |
| `cb_x_round` | 1 | tile (bf16) | `x_round_depth × s1_block_m_tiles × s1_kb` | streams S1·rows and S1·K, spans one round; declared on stage-1 rect ∪ injector (multicast destination address) | Float16_b | reader (receivers: mcast receive + push) / reader (injector: DRAM read) | compute (receivers) / writer (injector: multicast) | stage 1 |
| `cb_qb_slice` | 2 | tile (q_b dtype) | `Q_t × core_cols_max` | spans S1·K × S1·cols (resident operand) | Float16_b (Bfp8_b refinement) | reader | compute | stage 1 |
| `cb_q_partial` | 3 | tile (fp32) | `s1_block_m_tiles × core_cols_max` (allocated only when `Q_t/s1_kb > 1`) | one block's partial sums across K-blocks | Float32 (fp32 DEST) | compute | compute | stage 1 |
| `cb_landing` | 4 | tile (bf16) | `hpc × B × quarter_row_tiles × head_w_tiles` | spans S3·heads × S3·rows × head columns: every row this core receives (remote writes have no credit path) — **backed by the landing scratch shard** | Float16_b | reader (credits after arrival) | compute | whole call |
| `cb_wkv` | 5 | tile (wkv dtype) | R1: `min(2,hpc) × Dn_t × N_t`; R2: `2 × Dn_t × s3_block_n_tiles` | R1 spans S3·N, streams S3·heads (depth 2 heads); R2 streams N-blocks | Float16_b (Bfp8_b refinement) | reader | compute | stage 3 (filled from stage 1 on) |
| `cb_prefix_out` | 6 | tile (bf16) | `out_depth × s3_block_m_tiles × s3_block_n_tiles` | streams S3·rows, spans one N-block | Float16_b (output ABI) | compute | writer | stage 3 |
| `cb_rope_out` | 7 | tile (bf16) | `out_depth × s3_block_m_tiles × Dr_t` | streams S3·rows | Float16_b (output ABI) | compute | writer | stage 3 |
| `cb_port_stage` | 8 | tile (bf16) | `2 × port_batch_segments × head_w_tiles` | port only: two batches of segments | Float16_b (raw bytes) | reader (port) | writer (port sender) | port lifetime |

RoPE / ROW_MAJOR refinements add `cb_cos`, `cb_sin`, `cb_trans`, `cb_rope_tmp`, `cb_prefix_rm`, `cb_rope_rm`
(recorded there; not Phase 0).

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `x_round_multicast_block` | `s1_block_m_tiles × s1_kb` | Mcast helper (dataflow) | injector `cb_x_round` (round pages, filled by injector reader) | receivers' `cb_x_round` (same pages) | injector pops after `send` returns (SourceL1Guard) |
| 2 | `qb_slice_load_block` | `Q_t × core_cols` | no (TensorAccessor reads) | DRAM | `cb_qb_slice` (all pages, pushed once) | resident |
| 3 | `s1_matmul_block` | `s1_block_m_tiles × core_cols`, K=`Q_t` | `matmul_block` | `cb_x_round` (`s1_block_m_tiles*s1_kb` per K-block, wait/pop), `cb_qb_slice` (NoWaitNoPop + `In1BaseOffsetFn` = `kblock*s1_kb*core_cols`) | `cb_q_slice` (`s1_block_m_tiles*core_cols` per block) | `cb_q_slice` grows; never popped by compute |
| 4 | `s1_handoff_block` | one block | no (dataflow) | `cb_q_slice` (writer waits the block) | remote landing via ports / local landing via NoC | writer pops all at end of call |
| 5 | `ready_fence_block` | — | no (fabric API) | — | — | — |
| 6 | `send_segment_block` | `head_w_tiles` | no (fabric API, copied from AG sender) | source cores' `cb_q_slice` (remote reads) | `cb_port_stage` → fabric | staging popped per batch after `noc_async_writes_flushed` |
| 7 | `wkv_load_block` | R1 `Dn_t × N_t` / R2 `Dn_t × s3_block_n_tiles` | no (TensorAccessor) | DRAM | `cb_wkv` | R1 held across the head's rows (popped after its last row) |
| 8 | `land_row_block` | `s3_block_m_tiles × head_w_tiles` | no (semaphore wait + `cb_push_back`) | landing scratch | `cb_landing` (credits) | — |
| 9 | `prefix_matmul_block` | M=`s3_block_m_tiles`, K=`Dn_t`, N=`s3_block_n_tiles` | `matmul_block` (`num_k_blocks=1`, interm = out) | `cb_landing` (`s3_block_m*Dn_t`, retained across N-blocks in R2, popped after the last), `cb_wkv` (R1: NoWaitNoPop across rows) | `cb_prefix_out` | — |
| 10 | `rope_block` | `s3_block_m_tiles × Dr_t` | `copy` (eltwise convenience) | `cb_landing` (`s3_block_m*Dr_t`, wait/pop) | `cb_rope_out` | `cb_landing` front advances to the next row's nope |
| 11 | `store_row_block` | `s3_block_m_tiles × (N_t + Dr_t)` | no (TensorAccessor writes) | `cb_prefix_out`, `cb_rope_out` | DRAM output | popped |

Landing order per row = `[nope Dn_t][rope Dr_t]` — exactly q's column order of one head, so `matmul_block` pops
the nope tiles and the rope copy pops the rope tiles of the same row from the front of `cb_landing`.

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| (boot) | helper prerequisite | `compute_kernel_hw_startup<SrcOrder::Reverse>(in0, in1, out)` | `matmul_block_helpers.hpp:100-106,235-239` (matmul puts in0 on SrcB) | once at the top of the compute kernel, before any helper | — | — | — |
| `s1_matmul_block` | helper | `matmul_block<false, false, LastBlockTarget::Out, OutputCBLayout::SubblockMajor, InitMode::Short, WaitAndPopPerKBlock, NoWaitNoPop, …, In1BaseOffsetFn>` | `matmul_block_helpers.hpp:339-371`, `MatmulBlockShape::of` `:160-169`, `In1BaseOffsetFn` `:226-232`, `InputPolicy` `:77` | `MatmulBlockShape::of(in0_nsb, in1_nsb, sb_h, sb_w, s1_kb, Q_t/s1_kb, batch)` with `in0_nsb*sb_h = s1_block_m_tiles`, `in1_nsb*sb_w = core_cols`; interm = `cb_q_partial` when `Q_t/s1_kb>1` else `cb_q_slice` | `cb_x_round`, `cb_qb_slice` | `cb_q_slice` | block = `s1_block_m_tiles × core_cols` (M×N), `s1_kb`; subblocks derived from the block + `fp32_dest_acc_en` (≤ `DEST_AUTO_LIMIT`, `dest_helpers.hpp:103`) |
| `prefix_matmul_block` | helper | `matmul_block` (same header), in1 policy `NoWaitNoPop` in R1 (reader-owned head lifetime), `WaitAndPopPerKBlock` in R2; in0 `WaitAndRetainOnLastBlock` in R2 | as above | `MatmulBlockShape::of(…, Dn_t, 1)`; M = `s3_block_m_tiles`, N = `s3_block_n_tiles` | `cb_landing`, `cb_wkv` | `cb_prefix_out` | `s3_block_m_tiles`, `s3_block_n_tiles` |
| `rope_block` (NoPE) | helper | `copy<input(cb_landing), output(cb_rope_out)>(IterationShape::tiles(s3_block_m_tiles*Dr_t))` | `eltwise/api/convenience.hpp:87` | caller owns hw startup (done) | `cb_landing` | `cb_rope_out` | tile count |
| `rope_block` (RoPE, refinement) | helper + raw | `matmul_block` with the trans tile (1-tile in1) then `mul`/`add` | `convenience.hpp:45,51`; reference math `rotary_embedding_llama.cpp:120-164` | — | `cb_landing`, `cb_trans`, `cb_cos`, `cb_sin` | `cb_rope_out` | — |
| `untilize_row_block` (ROW_MAJOR refinement) | helper | `untilize<block_width_tiles, in, out>(num_blocks)` | `untilize_helpers.hpp:145-154` | width = `s3_block_n_tiles` / `Dr_t` | `cb_prefix_out` / `cb_rope_out` | `cb_prefix_rm` / `cb_rope_rm` | width |
| `x_round_multicast_block` | helper (dataflow) | `McastArgs<…>::sender(noc).send(src_l1, dst_l1, size)` / `receiver(noc).receive(round)` | `mcast/kernel/mcast_pipe.hpp:70-74`, `mcast/kernel/mcast_args.hpp:43-52`; host `ttnn.Mcast` 2-D fixed sender outside the grid `mcast/host/mcast_host.hpp:389` | `base_sem_id` 0 | injector `cb_x_round` | receivers' `cb_x_round` | round size |
| `qb_slice_load_block`, `wkv_load_block`, `store_row_block`, x injector reads | raw_api | `TensorAccessor` + `noc_async_read_tile`/`noc_async_write_tile` + barriers | `tech_reports/tensor_accessor/tensor_accessor.md` | batch per dm shortlist (8 / 16 / 9) | — | — | batch |
| `send_segment_block` (1-hop destinations), `ready_fence_block` (1-hop readies) | helper (dataflow) | `FabricStreamSender<DirectConn>` → `open(unicast_route(1))` → `arm_fused_write_inc(page_size, 1)` / `arm_unicast_write(page_size)` / `arm_inc(1)`; issue `write_fused` / `write` / `inc`; `close()` drains | `kernel_lib/ccl/ccl_helpers_dataflow.hpp:27-46` (stage types), `:190-204` (`DirectConn`), `:299-307` (`unicast_route`), `:388-397` (`FusedWriteIncChannel`), `:435` (`arm_inc`) | payload = segment piece ≤ `pkt_tiles*tile_bytes`; increment value 1 | port `cb_port_stage` | fabric | `pkt_tiles` (cap), `inc_every_packets`, `port_batch_segments` |
| `send_segment_block` (2-hop destinations: the antipodal quarter, sent direct) | raw_api | `PacketHeaderPool::allocate_header`, `fabric_set_unicast_route(hdr, dst_chip, dst_mesh)`, `to_noc_unicast_write` / `to_noc_fused_unicast_write_atomic_inc`, `send_current_slot_non_blocking` on the **same** connection (`DirectConn::sender()`, `ccl_helpers_dataflow.hpp:204`); readies to 2-hop chips: `to_noc_unicast_atomic_inc` + `send_payload_flush_blocking_from_address` | fabric_all_gather sender `program_descriptor_with_inline_kernels.py:172-305` (open 235-241, fence 242-255, send 277-284, re-arm 297-302); RX `kernels/port.cpp:41-50,156-169`; blackhole-fabric.md §5 | as above | port `cb_port_stage` | fabric | as above |
| `land_row_block`, `s1_handoff_block` | raw_api | `noc_semaphore_wait_min`, `noc_semaphore_inc`, `cb_reserve_back`/`cb_push_back`, `noc_async_write`, `noc_async_writes_flushed` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h` | thresholds from host | — | `cb_landing` | — |

**Helpers considered and rejected (raw_api rows):**

| Raw block op | Helper candidate | Mismatch (file:line) | Why it cannot express the shape |
|--------------|------------------|----------------------|---------------------------------|
| `qb_slice_load_block`, `wkv_load_block`, `store_row_block` | none in `kernel_lib` for interleaved DRAM tile reads/writes (`l1_helpers.hpp`, `local_copy_helpers_dataflow.hpp` are L1→L1) | `local_copy_helpers_dataflow.hpp` (local L1 copies only) | DRAM page reads through `TensorAccessor` are the dataflow mechanism; the helper library has no DRAM reader block op |
| `send_segment_block` / readies for **2-hop** destinations | `kernel_lib/ccl/ccl_helpers_dataflow.hpp` `FabricStream` | `ccl_helpers_dataflow.hpp:299-307` (`unicast_route(num_hops)` builds a 1-D hop-count `line_unicast_route_info_t`; `dst_mesh_id` unused) and `:108-110` (built on the LINEAR 1-D fabric API) | a 2-hop packet routed as "2 hops in my first hop's direction" leaves the snake-mock board and never arrives (prompt, Topology semantics); the antipodal chip must be addressed by fabric node through the 2D routing table (`fabric_set_unicast_route(hdr, chip, mesh)`), which the helper's route type cannot express. Gap to close: a node-addressed route overload of `FabricStream::open`. The 1-hop traffic on the same port uses the helper. |
| `land_row_block` | `dfb_helpers_dataflow.hpp` | — | a credit push after a semaphore threshold on an externally written buffer; no helper owns cross-chip arrival counters |

## Broadcast Verification

Phase 0 has no binary op. The RoPE refinement's binary ops:

| Phase | Op | CB_A (semantic name) Valid Region | CB_B (semantic name) Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| rope (refinement) | mul | `cb_rope_tmp` (rotated) All | `cb_sin` (rows of the quarter, `Dr_t` cols) All | None |
| rope (refinement) | mul | `cb_landing` rope tiles All | `cb_cos` All | None |
| rope (refinement) | add | `cb_rope_tmp` All | product All | None |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Arrival thresholds reached by the wrong stream | two ports (two links) or two source chips incrementing one counter makes per-row thresholds meaningless | one counter per (dest core, source position); link chosen by dest core; local heads wait for their total |
| Remote write before the destination started this call | landing scratch persists across calls | per-port ready fence before the first send; header-only readies; re-armed to 0 |
| Same-address assumptions | ports read `cb_q_slice` of other cores; the injector multicasts into `cb_x_round` | both CBs are declared on a core-range set including the remote reader / sender (one address per CB); the cross-chip landing uses an explicit scratch-tensor address, not a CB address |
| 2-hop route on the snake mock | "2 hops in my first hop's direction" leaves the board and hangs | `fabric_set_unicast_route(hdr, dst_chip, dst_mesh)` (2D routing table) on the port whose direction is the route's first hop (`_first_hop_dir`) |
| Payload size hard-coded | packet size differs per model (7232 / 12352 / 14400 B) | `pkt_tiles` from `get_tt_fabric_max_payload_size_bytes()` only |
| x multicast lockstep | one slow receiver (a reader busy with q_b / wkv reads) stalls every core | reader order: q_b slice first, then wkv reads issued in chunks *between* x rounds (async, barrier before push) |
| `cb_landing` consumed by two block ops | nope (matmul) and rope (copy) both pop from it | both are in the compute kernel (one consumer thread); pop order fixed: Dn_t then Dr_t per row |
| Runtime-arg overflow on ports | per-segment lists for ~120 segments × source runs | regular descriptors (per dest chip: head list, row stride) + one core-coordinate table; device derives offsets |
| generic_op program-cache aliasing | per-core arg layouts differ with the plan | `custom_program_hash` over the full plan and knob set |
| q stored bf16 under fp32 DEST | the ledger's audit 2 flags 16-bit pages under fp32 DEST | deliberate: the exchanged q is bf16 (link bytes, prompt allows bf16 q within default tolerances); bit-exact after packing |
| Stage-1 cores with 0 columns | a rect larger than `Cq_t` | `W1 = min(gx, Cq_t)`, `R1 = floor(Cq_t/W1)` ⇒ every stage-1 core owns ≥ 1 column |
| Dispatch-bound | ~100+ cores with per-core args | merge identical-kernel core sets into rectangles (`CoreRangeSet.merge`); common args where possible |

### Structural impossibilities

None beyond `feature_spec.py`'s empty `INVALID`.
