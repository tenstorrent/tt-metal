# Operation Design: a2a_wkv_b2

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (CCL all-to-all over a 4-chip ring + per-head batched matmul), one `ttnn.generic_op` over a `ttnn.MeshProgramDescriptor` |
| Goal | On every chip of every ring along `cluster_axis`, apply `W` to this chip's 160 tokens of all 96 heads **first** (matmul-first), then send each destination chip its 24 heads' 128-wide bfloat8_b result over the fabric, writing it straight into the destination's L1-interleaved output tiles. |
| Math | `out_r[0, h, 160*j + t, v] = sum_l attn_j[0, 24*r + h, t, l] * W[0, 24*r + h, l, v]`, h∈[0,24), j∈[0,n), t∈[0,160), v∈[0,128); n = ring size (4) |
| Unfused equivalent | `all_to_all_async_generic(attn, in_dim=2, out_dim=1, cluster_axis=1, num_links=2)` then `ttnn.linear(y, W[:, 24r:24r+24])`. Neither is called. |
| References | `eval/golden_tests/a2a_wkv_b2/{feature_spec.py,helpers.py,test_regression.py}`; `ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_async_generic/device/` (read only: ring offset folding and antipode split, `all_to_all_async_generic_program_factory.cpp:150-205`; 2D route `kernels/all_to_all_sender_writer.cpp:600-640`); `tests/ttnn/unit_tests/operations/debug/test_generic_op.py:138-439` (fabric generic_op example); `ttnn/cpp/ttnn/kernel_lib/ccl/ccl_helpers_dataflow.hpp` (fabric egress helpers); `ttnn/cpp/ttnn/kernel_lib/{tilize_helpers.hpp,tilize_helpers_dataflow.hpp,matmul_block_helpers.hpp}`; `ttnn/ttnn/operations/examples/compute_block_size/` (tilize feeding matmul_block, `program_descriptor_with_inline_kernels.py:125-180`) |

## Structure: `pipeline_spec.json`

The perf case's black box (`spec`), pipeline (`pipeline`) and open choices (`knobs`) are in
`pipeline_spec.json` beside this file. `pipeline_spec.py` **refuses** it (the send axis is a ring:
"only a line (no wraparound) on the send axis is supported yet", all 24 combinations). The design is
therefore estimated with an `opgraph.py` builder for the ring, `perf/ring_builder.py` +
`perf/ring_problem.json` (the tool the skill names for a multi-chip op), and the reader schedules with
`estimate.py sweep` (`perf/sweep_attn.json`, `perf/sweep_w.json`, recorded in `dm_estimates.json`).

| Perf case / regime | File | Sweep command |
|--------------------|------|---------------|
| LOOSE_CASES[0] (the only case), regime R1 | `pipeline_spec.json` (spec of record; refused by `pipeline_spec.py`) | `python3 .claude/tools/perf_calculator/pipeline_spec.py ttnn/ttnn/operations/a2a_wkv_b2/pipeline_spec.json --sweep` → 0 estimated, 24 refused (ring) |
| same, ring estimate | `perf/ring_problem.json` + `perf/ring_builder.py` | `python3 .claude/tools/perf_calculator/opgraph.py ttnn/ttnn/operations/a2a_wkv_b2/perf/ring_problem.json ttnn/ttnn/operations/a2a_wkv_b2/perf/ring_builder.py --opt mt_block=1 --opt cores=96 --opt antipode_split=True` (line rate edited to 25 in the problem for the 25 GB/s rows) |
| worker reader schedules | `perf/sweep_attn.json`, `perf/sweep_w.json` | `python3 .claude/tools/perf_calculator/estimate.py sweep <file> --record ttnn/ttnn/operations/a2a_wkv_b2/dm_estimates.json --phase Planner` |

## Bounds

| Bound | Value ns | Binding resource | Built by |
|-------|----------|------------------|----------|
| Speed of light | 43,776 | DRAM: 15,728,640 B attn + 6,684,672 B W at 512 GB/s | `feature_spec.py` LOOSE_CASES extras, `.claude/references/speed-of-light.md` |
| Roofline | ≈ 49,500 (DRAM-bound); ≈ 52,000–60,000 if the NoC delivery estimate of the reads holds | DRAM, then the NoC links that carry the reads into 96 worker cores | `.claude/references/roofline-analysis.md`, table below |
| Calculator estimate of the chosen design | 51,710 (50 GB/s links), 53,880 (25 GB/s); reads priced at 512 GB/s | DRAM (85% busy), tail of the last round | `opgraph.py … ring_builder.py --opt mt_block=1 --opt cores=96` |

Roofline table (loads from `pipeline_spec.json` `spec`):

| Resource | Load (best algorithm) | Rate | Rate source | Time ns |
|---|---|---|---|---|
| DRAM | 22,413,312 B (attn once + all 96 heads of W once) | 512 GB/s | 8 GDDR6 ch × 64 GB/s, `tech_reports/CCLs/CCL_bandwidth.md` via feature_spec | 43,776 |
| NoC delivery of the reads (96 cores) | attn 15.7 MB in 1024 B sticks + W 6.7 MB in 1088 B tiles | estimate.py (silicon-checked read model) | `dm_estimates.json` sweeps eb378294 / d1adf26d | NoC 0: 38,012 + 16,236 = 54,248; split NoC: 32,287 + 13,870 = 46,157 (carries the tool's split-NoC caveat) |
| Fabric, worst link direction | 522,240 B per channel (one 24-head bf8 chunk on each of the 2 channels of a direction = feature_spec's "2 chunks over 2 channels"; see Traffic ranking) | 50 GB/s per channel | feature_spec (BH rating; NOT measured on this box) | 10,445 (25 GB/s: 20,890) |
| Matmul | 30,720 tile-mults / 96 cores × 32 cycles (HiFi2) | 1.35 GHz | feature_spec (AICLK assumed) | 7,585 |
| Fixed | launch 1,400 + open + entry handshake (2 hops) ≈ 3,140 | — | opgraph `fabric()` steps | 3,140 |
| Drain | last mt round's tilize + matmul + send + hop + land ≈ 2,600 (mt_block = 1) | — | opgraph critical path | 2,600 |

Roofline = DRAM 43,776 + fixed ≈ 3,100 + drain ≈ 2,600 ≈ 49,500 ns. The fabric (10.4–20.9 us) and
the matmul (7.6 us) hide under the DRAM stream because results are sent one mt round at a time.

Ranking of the opgraph estimate (`perf/ring_builder.py`, 50 GB/s unless stated):

| Rank | Knobs | us | vs best |
|---|---|---|---|
| 1 | mt_block=1, cores=96 or 110, antipode split | 51.44–51.71 | 1.00 (tie) |
| — | same at 25 GB/s | 53.62–53.88 | 1.04 (tie) |
| 3 | mt_block=5 (whole head per block), cores=96 | 70.23 | 1.36 |
| 4 | mt_block=5 at 25 GB/s | 81.00 | 1.57 |

Chosen: rank 1 with **96** cores (the 96-vs-110 difference is 0.5%, and 110 cores would need some head
split across cores, which re-reads W — see Traffic ranking). The antipode split shows no difference in
the graph because the links are ≤21% busy; it is kept because it is what keeps every link direction at
one chunk (the gap opens at a lower link rate or a larger V), and it costs nothing structurally.

Split search: `op_model.py` is not used — the op is a fused multi-chip op (the skill's tool for it is
`opgraph.py`, above). The time-vs-cores question it would answer is the 96 vs 110 cores rows above
(flat: 0.5%); with ~0.65 us per extra core of launch cost not modelled, the smaller count is taken.

Data-movement search (`estimate.py sweep`, `--phase Planner`, recorded in `dm_estimates.json`; the
fabric sends and the worker → fabric_tx pulls are L1-to-L1 / fabric schemes the tool refuses, ranked on
bytes above):

| Group | Rank | Knobs (cores, bytes, batch, noc) | pass_ns | tie_with_best | L1 bytes |
|---|---|---|---|---|---|
| attn sticks (id eb378294) | 1 | 96, 1024, 16, split | 32,287 | yes | 16,384 |
| | 2 | 96, 1024, 32, split | 32,390 | yes | 32,768 |
| | 3 | 96, 1024, 32, 0 | 38,012 | no | 32,768 |
| | 4 | 96, 1024, 8, 0 | 38,013 | no | 8,192 |
| W tiles (id d1adf26d) | 1 | 96, 1088, 64, split | 13,870 | yes | 69,632 |
| | 4 | 96, 1088, 32, 0 | 16,236 | no | 34,816 |
| | 5 | 96, 1088, 64, 0 | 16,237 | no | 69,632 |

Phase 0 builds the NoC 0 entries (attn batch 32 = one barrier per tile-row block, W batch 64 = one per
head): rank 1 is a split-NoC read on 96 of 120 cores, and the tool's own caveat says that pattern
measured 17–26% slower than estimated and slower than NoC 0 alone, and not to float it on the estimate
— it is the first perf lamp. Batch is flat on NoC 0 (38.01 vs 38.02 us), so the batch follows the block.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| input_tensor | ttnn.Tensor (mesh) | yes | per chip [1, 96, 160, 512] bf16 ROW_MAJOR DRAM interleaved | — | buffer address RT |
| weight | ttnn.Tensor (mesh) | yes | per chip [1, 96, 512, 128] bf8_b TILE DRAM interleaved, all 96 heads | — | buffer address RT |
| cluster_axis | int (kw-only) | no | mesh axis of length 4 (`mesh_device.shape[cluster_axis] == 4`) | 1 | host |
| num_links | int (kw-only) | no | Phase 0: == streams per direction (2 on a 4-ring); else NotImplementedError | 2 | host (selects link per stream) |
| compute_kernel_config | ttnn.ComputeConfigDescriptor (kw-only) | no | fp32_dest_acc_en=False (SUPPORTED); any fidelity | `default_compute_kernel_config()` = HiFi2, fp32_dest_acc_en=False, math_approx_mode=False | compute kernel config |

`default_compute_kernel_config()` is exported from the op module; `None` resolves through it. The
golden axis tagger reads the same factory.

## Tensors

### Input

| Property | attn (`input_tensor`) | W (`weight`) |
|----------|----------------------|--------------|
| Shape (per chip) | [1, 96, 160, 512] | [1, 96, 512, 128] (replicated) |
| Dtype | bfloat16 | bfloat8_b |
| Layout | ROW_MAJOR (page = one token of one head = 1024 B) | TILE (page = 1088 B) |
| Memory | DRAM interleaved | DRAM interleaved |

### Output

| Property | Value |
|----------|-------|
| Shape | per chip [1, H/n, T·n, V] = [1, 24, 640, 128] |
| Dtype | bfloat8_b |
| Layout | TILE |
| Memory | L1 interleaved (allocated by the entry point with `ttnn.allocate_tensor_on_device`, same address on every chip) |

Tile id of output element block `(h_local, source j, mt, nt)` on the destination:
`h_local·(n·Mt·Nt) + (j·Mt + mt)·Nt + nt` with Mt = ceil(T/32) = 5, Nt = ceil(V/32) = 4. T = 160 is a
multiple of 32, so every source's rows start on a tile row; the formula is still written with `ceil`
(`Mt = ceil(T/32)`) and the op's `validate()` refuses `T % 32 != 0` (another source's rows would share a
tile).

## Blocking Model

Semantics: `.claude/references/blocking-model.md`. Per chip the op is a batched matmul over heads,
followed by a routing of each head to its owning chip.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `h` (heads, 96 on the sender) | independent — each head has its own W slice and output rows; also the **destination axis** (head h goes to chip h // 24) | `heads_per_core` | 1 (= ceil(96 / num_workers), num_workers = min(96, 110, grid − tx cores)) | host: `ttnn.split_work_to_cores(worker_grid, H)` | spread over the 96 worker cores, one head per core | knob-turn |
| `m` (token tile rows, Mt = 5 per chip) | independent per output tile, but **reuse-shared for W**: all 5 rows of a head use the same W[h] | `mt_block` | 1 tile row (32 tokens) | host constant → compute/reader CT arg | not split across cores (whole Mt of a head on its core) — splitting it re-reads W[h] per part | knob-turn (raise `mt_block`); scheme-change to split it across cores (needs W multicast) |
| `k` (latent, Kt = 16) | dependent — reduced by the matmul | `kt_block` | 16 = whole K (one K block, accumulated in DEST) | derived: `Kt = ceil(L/32)` | not split (whole K on one core) | scheme-change (K split + combine), rejected below |
| `n` (v_head_dim, Nt = 4) | independent | `nt_block` | 4 = whole N | derived: `Nt = ceil(V/32)` | not split | knob-turn |
| `j` (ring position / destination chip, n = 4) | independent — each destination's output is disjoint | `streams` (one per (direction, hop count) pair) | n − 1 = 3 destinations → 4 streams (antipode split in two) | host route plan from `mesh_device.shape[cluster_axis]` | one fabric sender core (`fabric_tx`) per stream; a stream = one (direction, link) | knob-turn within the ring |
| `fabric packet` (tiles per packet, derived from `n`×`m`) | — (transport unit) | `tiles_per_packet` | 4 = Nt (one mt block of one head) | host: `min(Nt, 4 scatter chunks, floor(max_payload / 1088))` | — | knob-turn (bank-contiguous packing, lamp) |

Intermediate data re-run through the table: the output block of one (head, mt) is 1 × Nt tiles; its
axes `h` and `m` stay independent, so every stage that handles it (send, land) is spread over the 4
fabric_tx cores by destination — no stage is left on a single coordinating core. The fabric_tx cores
are a **communication role**; they do no compute.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_attn_rm` | `attn_rm_depth` | 2 blocks (2 × Kt pages) | the reader fetches mt+1 while compute tilizes mt |
| `cb_attn_tiled` | — | 1 block | tilize and matmul run on the same TRISCs one after another; a second block buys nothing |
| `cb_weight` | — | 1 head (resident across `m`) | residency, not overlap (heads_per_core = 1) |
| `cb_out` | — | all of the core's blocks (heads_per_core × Mt × Nt) | blocks stay until fabric_tx pulls them; no back-pressure handshake |
| `cb_tx_stage` (fabric_tx) | `tx_buffers · tx_batch` | 2 batches of 8 packets | pull of batch b+1 (one barrier) overlaps the fabric send of batch b (one flush); polls cached per head (Refinement 1) |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Fabric scatter write (`NocUnicastScatterCommandHeader`, `NOC_SCATTER_WRITE_MAX_CHUNKS` = 4, `tt_metal/fabric/hw/inc/api_common.h:281`; `ScatterWriteChannel` 2..4 chunks, `ccl_helpers_dataflow.hpp:136`) | tiles per packet | `tiles_per_packet = min(Nt, 4)`; Nt = 4 here | chunks past 4 are dropped: tiles silently missing |
| Fabric max payload (`ttnn.get_tt_fabric_max_payload_size_bytes()`) | bytes per packet | `tiles_per_packet = min(.., floor(max_payload / 1088))`; host asserts ≥ 1 | packet larger than the channel buffer: corrupt/hang |
| DEST capacity, half-sync bf16 = 8 tiles (`dest_helpers.hpp:89-103`) | `mt_block × nt_block` accumulators live at once | out subblock 1 × 4 (helper legality, `matmul_block_helpers.inl:158`) — the matmul helper's own subblocking; this design only requires mt_block·Nt ≤ 8 or the helper subblocks it | assert / wrong tiles |
| `tilize` fast path (`tilize_helpers.inl:65-69`) | block width | 16 tiles < 256 ✓ | slow path, still correct |
| Fabric routers: one worker sender channel per (direction, link) router | streams per (direction, link) | Phase 0 requires `num_links == streams per direction` (2); otherwise NotImplementedError | two workers on one router channel: connection refused / hang |
| Source tile-row alignment | T | `validate()` requires `T % 32 == 0` | two sources write the same output tile: data race |
| Uniform CB addresses | every CB's core range | every CB descriptor spans **all program cores** (workers ∪ fabric_tx) | fabric_tx pulls from the wrong L1 address on a worker |
| Matmul di/dt | matmul cores | ≤ 110 | power limit |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **R1 matmul-first, head split, streamed rounds** | built | the only cell: per chip attn [1,96,160,512] bf16 RM, W [1,96,512,128] bf8_b, ring of n = 4 along `cluster_axis`, `num_links == 2` | per worker: W[h] resident (Kt×Nt = 64 tiles); attn one mt row (32 sticks = 16 tiles) per block; out 1×Nt tiles per block = one packet | DRAM: attn crosses once, W crosses once per chip (all 96 heads: this order needs them); minimum for this order. Fabric: each 24-head bf8 chunk crosses its route once (neighbour 1 hop, antipode 2 hops) = the minimal byte-hops, 1 chunk per link direction. Output: written once into L1 (local by NoC, remote by fabric) | `mt_block` up to 5 removes 4 tilize↔matmul re-inits per head but makes the first packet wait for the whole head: opgraph 70.2 vs 51.7 us. Larger packets (bank-contiguous) amortize the per-packet header (64 B / 4352 B = 1.5%) and EDM slot handling — lamp |
| R2 transfer-first (send 512-wide attn, multiply on arrival) | rejected — superseded by R1 | — | — | DRAM 15.7 MB + 1.67 MB W (24 heads) = 34 us, but fabric 2 chunks of 3,932,160 B per link direction = 78.6 us at 50 GB/s (157 us at 25): 7.5× the link bytes of R1. A dead end, not a stepping stone: nothing of R1 is built on top of it | — |
| R3 mt split across cores (use all 110 matmul cores) | rejected — superseded by R1 | — | — | each extra core on a head re-reads W[h] (69,632 B) from DRAM; DRAM-bound op, so +W bytes = +time; 96 heads already fill 96 cores | — |
| R4 K split across cores + combine | rejected — superseded by R1 | — | — | adds partial-sum NoC traffic and a combine round, saves nothing (attn is read once already, K = 512 fits one core's DEST accumulation) | — |
| R5 W multicast across chips/cores | rejected | — | — | W is replicated in every chip's DRAM and each head's W is used by exactly one core per chip: no operand is re-read, so there is nothing for a multicast to save (operand-reuse check below) | — |
| R6 more than one stream per router (num_links < streams per direction, or a larger ring) | built in Refinement 3, parked: `MUX_OVERFLOW_HEADS` (default 0) moves the last x heads of each neighbour stream onto link 1 as a second stream, link 1 then shared with the antipode stream through a fabric mux core (`ttnn.experimental.fabric_mux`, `MuxConn<N>` in the tx writer behind `A2A_TX_MUX`). Measured x = 6: device span 71 → 63.5 µs, but the launch gap between back-to-back programs 11.5 → 17.5 µs, so per call is flat (82.0 vs 82.7 µs) | `num_links < ceil((n−1)/2)+1 per direction` | same block | same bytes; several fabric_tx cores share one link through `MuxConn<N>` (`ccl_helpers_dataflow.hpp:231`) | — ; reachable: the route plan already produces a list of streams, R6 only maps several onto one link through the mux policy, the fabric_tx kernel is unchanged (the ConnT template parameter) |

Single-cell domain: TARGET has one value per axis and INPUTS one shape, so R1 covers the whole space; R2
loses by link bytes (78.6 vs 10.4 us at 50 GB/s against a 43.8 us DRAM floor), R3/R4 by DRAM bytes.

### Traffic ranking

Per chip, bytes per tier (structural, not ns):

| Candidate split | DRAM | Fabric (worst link direction) | Cross-core NoC | Verdict |
|---|---|---|---|---|
| **Heads over cores, matmul first (R1)** | 22.4 MB (attn 15.7 + W 6.7, each once) | 522,240 B | out blocks pulled worker → fabric_tx: 72 heads × 21,760 B = 1.57 MB; local 24 heads written to local L1 0.52 MB | **chosen** — fewest link bytes, DRAM at the minimum for this order |
| Heads over cores, transfer first (R2) | 17.4 MB | 3,932,160 B × 2 = 7.86 MB | arriving attn to compute cores | 7.5× link bytes; fabric-bound at 78.6 us |
| (head, mt) over 110 cores (R3) | 22.4 MB + 69.6 KB per extra core on a head | same as R1 | same | strictly more DRAM bytes |
| K split + combine (R4) | 22.4 MB | same | + partials 2 B × 160 × 128 per head per part | strictly more NoC bytes |

Operand-reuse check (chosen split = heads over cores): attn varies along h ✓; W varies along h ✓ → no
operand is reuse-shared by the split. W is reuse-shared along `m` **within** a core and is kept resident
across it (read once per head). No broadcast row is needed.

Stall-shadow check: (1) each fabric_tx core waits on workers' blocks; what runs in that window is the
fabric send of the previous blocks — the rounds are mt-major so the first packets leave after
W + one mt row is read (~44% of the DRAM stream) instead of after the whole head. (2) The entry
handshake (ready from the destination) is waited on by fabric_tx only, while the workers already read
and compute: data-independent of the wait. (3) The exit wait (done from the peer) overlaps the last
local writes. No reorder changes floating-point order (each output tile is one core's K=16
accumulation in fixed order).

### Block schedule

Per worker core (reader / compute / writer kernels, pipelined):

```cpp
for (uint32_t head_idx = 0; head_idx < heads_this_core; ++head_idx) {           // heads_per_core
    load_weight_block(head_idx);                                                 // W[h]: Kt x Nt tiles, resident across mt
    for (uint32_t mt_idx = 0; mt_idx < num_mt_blocks; ++mt_idx) {               // num_mt_blocks = ceil(Mt / mt_block)
        load_attn_block(head_idx, mt_idx);                                       // mt_block*32 sticks x 1024 B
        tilize_attn_block(mt_idx);                                               // mt_block x Kt tiles (bf16)
        project_block(mt_idx);                                                   // [mt_block x Kt] @ [Kt x Nt] -> mt_block x Nt bf8 tiles
        publish_block(head_idx, mt_idx);                                         // local head: write to local output L1; remote: bump sem_blocks_done
    }
    release_weight_block(head_idx);
}
```

Per fabric_tx core (one per stream = (direction, link); reader / writer kernels):

```cpp
announce_ready();                       // fabric atomic inc -> peer fabric_tx sem_ready on the destination
await_ready();                          // local wait sem_ready >= 1, reset 0
for (uint32_t mt_idx = 0; mt_idx < num_mt_blocks; ++mt_idx)
    for (uint32_t s = 0; s < heads_in_stream; ++s) {
        pull_block(s, mt_idx);          // poll worker's sem_blocks_done >= mt_idx+1, NoC-read its 4 tiles into cb_tx_stage
        send_block(s, mt_idx);          // one scatter packet: 4 tiles to their 4 output-tile addresses on the destination
    }
announce_done();                        // fabric atomic inc (flush) -> peer fabric_tx sem_done on the destination
await_done();                           // local wait sem_done >= 1 (peer's data has landed here), reset 0
```

| Operation | Block shape | Resident across it | Intended frequency of inits / sync |
|---|---|---|---|
| `load_weight_block` | Kt × Nt = 16 × 4 bf8 tiles (69,632 B) | — | once per head; one barrier per head (64 reads in flight) |
| `load_attn_block` | mt_block·32 sticks × 1024 B | W[h] | one barrier per block (32 reads in flight) |
| `tilize_attn_block` | mt_block × Kt tiles | W[h] | tilize init/uninit once per block (alternates with matmul) |
| `project_block` | out mt_block × Nt, K = Kt in one block | W[h] (retained, never popped by the helper) | matmul short-init once per block; DEST accumulates all of K |
| `publish_block` | mt_block × Nt tiles | the block itself stays in `cb_out` until the program ends | local: one write barrier per block; remote: one local semaphore bump per block |
| `pull_block` | Nt tiles (4,352 B) from one worker | — | one NoC read + barrier per packet; one remote semaphore poll per packet |
| `send_block` | one packet = tiles_per_packet tiles | — | one fabric packet; route and chunk sizes armed once per stream; destination NoC addresses from the stream's address table (`cb_tx_addrs`, filled once before the first send, Refinement 3). Packets come in the reader's batch order (`TX_ORDER`), named by a `cb_tx_meta` page per batch |
| handshakes | 1 inc + 1 wait each | — | once per call per stream |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Reader NoC | estimate.py ranks the split-NoC read first (attn 32.3 vs 38.0 us, W 13.9 vs 16.2 us) but carries a caveat: on 96 cores a split-NoC read measured 17–26% slower than estimated. Phase 0 uses NoC 0 | issue alternate requests of `load_attn_block` / `load_weight_block` on NoC 1 (the worker writer's NoC is idle for 72 of 96 heads). **Refinement 2, measured:** per-request alternation on one RISC is unsafe in dedicated-NoC mode, so it is split by core instead (`READ_NOC_SPLIT`; reader on NoC 1 + writer on NoC 0 for one core set). `halves` (right half on NoC 1): reader end 60.5 → 55.8 µs, 103.3 → 93.9 µs per call; `halves_left`: reader 53.8 µs, 95.9 µs per call; `rows`: flat; `cols`: 107 µs per call; `checker`: reader 54 µs but 183 µs per call (48 single-core ranges defeat dispatch multicast); `all` (NoC 1 only): reader 105 µs |
| Overlap: `mt_block = 1` | 5 tilize↔matmul re-inits per head; the compute is 7.6 us of a 44 us op so this is unlikely to bind | `mt_block = 5` re-measured only if compute shows on the critical path (opgraph: 70 us, i.e. overlap matters more) |
| Last-round tail (~2.6 us) | every worker finishes its last mt at the end of the DRAM stream, then the last round is sent | read order that ends on a smaller block (e.g. last W-free round split), or local heads' last round scheduled last. **Refinement 3, measured:** worker cores finish at very different times (last block ~24 to ~60 µs by core position), so the tail is the fabric send of late remote heads, not a per-round drain. Built: `TX_ORDER = "ready"` (send whatever is published; 89.3 → 84.2 µs per call) and `HEAD_PLACEMENT = "by_stream"` (local heads on the slowest core rows; 84.2 → 82.4 µs) |
| Packet size | 4 tiles (4,352 B) per packet via scatter; per-packet EDM overhead may cap a link below 50 GB/s | bank-contiguous packets: output tiles i and i + num_L1_banks are adjacent in one bank, so a packet of k same-bank tiles is one contiguous write of k·1088 B |
| Worker grid placement | worker cores taken row-major; DRAM-read NoC links are the binding resource in estimate.py | place workers with `assign = bank_spread` style order, or near DRAM columns. Refinement 3: head → core placement by stream (`HEAD_PLACEMENT`), see the tail row |
| Polling | fabric_tx polls each worker's semaphore over the NoC | worker pushes a "block ready" atomic to fabric_tx with a per-round counter if polling shows up |

## Dataflow Strategy

`pipeline_spec.json` stage names: `read_w`, `read_attn`, `tilize`, `mm`, `send`. The `send` stage is
realized by two kernels on each fabric_tx core (pull + send), and `publish` is the worker writer.

| Stage (`pipeline_spec.json` name) | Format | Mechanism (helper / NoC API) | Notes |
|-----------------------------------|--------|------------------------------|-------|
| `read_w` (worker reader, NCRISC, NoC 0) | bf8_b tiles 1088 B, DRAM → `cb_weight` | `TensorAccessor` + `noc_async_read_page`, tile ids `h·Kt·Nt .. +Kt·Nt−1` | once per head, before the first attn block |
| `read_attn` (worker reader) | bf16 sticks 1024 B, DRAM → `cb_attn_rm` (tile-sized pages, 16 per 32-stick block) | `dataflow_kernel_lib::read_sticks_for_tilize<cb_attn_rm, TilizeGranularity::TILE>(accessor, rows, 1024, start_page)` (`tilize_helpers_dataflow.hpp:88`) | page id of (h, t) = h·T + t; one block = mt_block·32 sticks |
| `tilize` (worker compute) | RM → bf16 tiles in `cb_attn_tiled` | `compute_kernel_lib::tilize<Kt, cb_attn_rm, cb_attn_tiled>(in, out, mt_block)` (`tilize_helpers.hpp:206`) | in-kernel tilize (no Python layout conversion) |
| `mm` (worker compute) | bf16 × bf8_b → bf8_b tiles in `cb_out` | `compute_kernel_lib::matmul_block<…, in1_policy = WaitAndRetainOnLastBlock>` (`matmul_block_helpers.hpp:358`), `MatmulBlockShape::of(1, 1, mt_block, Nt, Kt, 1)` (`:160`) | W never popped by the helper; compute pops `cb_weight` (Kt·Nt) after the head's last block |
| publish (worker writer, BRISC, NoC 1) | bf8_b tiles | local head: `noc_async_write_page` via `TensorAccessor` on the output; remote head: `noc_semaphore_inc` on its own `sem_blocks_done` (local) after `cb_wait_front` | never pops a remote block before the end of the kernel |
| `send` part 1: pull (fabric_tx reader, NCRISC, NoC 0) | bf8_b tiles → `cb_tx_stage` | poll worker `sem_blocks_done` with `noc_async_read` of 4 B; `noc_async_read` 4,352 B from worker's `cb_out` at `get_read_ptr(cb_out) + block_offset` (uniform CB address) | order: mt-major, then the stream's heads in head order |
| `send` part 2: fabric egress (fabric_tx writer, BRISC) | bf8_b tiles over Ethernet | `dataflow_kernel_lib::ccl::FabricStreamSender<DirectConn>` → `open(route{dst_mesh, dst_chip})` → `arm_scatter_write(1088, tiles_per_packet)` → `write_scatter(addrs, n, l1)`; handshakes `arm_inc()` → `inc(peer_sem_noc_addr)` (`ccl_helpers_dataflow.hpp:486-540`, `:136`, `:141`) | destination NoC addresses: `tt::tt_fabric::linear::addrgen_detail::get_noc_address(output_accessor, tile_id, 0)` (same use as `all_to_all_sender_writer.cpp:639`) |
| land | bf8_b tiles in the destination's L1 output | the fabric router performs the NoC writes on the destination | no receiver kernel |

### Ring route plan (host, derived — never hard-coded)

| Item | Derivation |
|---|---|
| n, r | `n = mesh_device.shape[cluster_axis]`; `r = coord[cluster_axis]` for each `MeshCoordinate` of the mesh |
| ring members | same coordinate with `coord[cluster_axis] = (r + k) mod n`; fabric node ids via `mesh_device.get_fabric_node_id(coord)` |
| destination offsets | k = 1..n−1: `2k < n` → forward (+1 side), hops k, all 24 heads; `2k > n` → backward, hops n−k, all heads; `2k == n` (antipode) → **two streams**, forward with heads [0, hpc/2), backward with heads [hpc/2, hpc) |
| streams → links | within a direction, streams sorted by hop count get link 0, 1, … (on n = 4: link 0 = neighbour, link 1 = antipode half). Each link direction then carries exactly one 24-head chunk: link 0 the neighbour chunk, link 1 this chip's antipode half (hop 1) + the previous chip's antipode half relayed by the router (hop 2). |
| stream's fabric connection | `ttnn._ttnn.ccl_host.build_ccl_fabric_rt_args(mesh_device, my_node, neighbour_node_in_direction, link, pd, tx_core, is_forward)` placed FIRST in the fabric_tx writer's RT args |
| route | `line_unicast_route_info_t{dst_mesh_id, dst_chip_id}` of the destination (2D fabric: `fabric_set_unicast_route(hdr, chip, mesh)`, `worker_routing_utils.hpp:60-70`); the packet is injected into the chosen direction's router, so the antipode half takes the chosen arc (precedent: `all_to_all_async_generic_program_factory.cpp:188-205`) |
| peer | the fabric_tx core of the destination whose stream targets this chip: opposite direction, same hop count, same link. Its virtual NoC coords + the global semaphore address give the handshake NoC address. Symmetric on every chip, so each stream has exactly one peer and every count is 1 |

The two rings (rows) are independent: each chip's plan only names members of its own ring.

### Synchronization contract (persistent state)

| Semaphore | Kind | Lives on | Incremented by | Waited by | Reset |
|---|---|---|---|---|---|
| block counter (one per worker head slot; was `sem_blocks_done`) | `GlobalSemaphore` per head slot (`heads_per_core` of them), created once per mesh device and cached (first, eager call), zero at creation | every worker (same address per slot) | the worker writer, +1 per published block of a **remote** head (local heads have no consumer and never bump it) | that head's fabric_tx reader, remote poll `>= mt_idx + 1` | the consuming fabric_tx reader, `noc_inline_dw_write(…, 0)` + write barrier after the head's last block |

**Refinement 2 change (race fix).** `sem_blocks_done` was a program semaphore, which dispatch re-initialises when *the worker core* launches. Launch is skewed across cores (fabric_tx cores start 0.25–0.6 µs in, workers up to 0.5 µs), so a fabric_tx reader could read the previous call's final count (5) before the worker's launch reset it, and pull blocks not yet produced. The per-head counters are persistent, and their only consumer returns them to 0, so every call starts from 0 whatever the launch order. The same refinement fixed a second fabric_tx race: the scatter channel's single packet header was re-programmed for packet i+1 before the non-blocking header write of packet i had left L1. The writer now rotates a ring of armed headers, reused only after a flush.
| `sem_ready` | `GlobalSemaphore` on the fabric_tx cores, created once per mesh device and cached by the module (first, eager call), same address on every chip | fabric_tx | the peer fabric_tx (fabric atomic inc) at its start | own fabric_tx writer before the first data packet: `noc_semaphore_wait_min(sem_ready, 1)` | `noc_semaphore_set(sem_ready, 0)` right after the wait, before any data is sent |
| `sem_done` | `GlobalSemaphore`, same | fabric_tx | the peer fabric_tx, one inc with flush after its last data packet to this chip | own fabric_tx writer before it exits | set 0 right after the wait |

Ordering argument (why the next call sees 0): the peer's ready for call N+1 is sent only after the
peer's chip starts call N+1, which needs that chip's call N to finish, which needs its `sem_done` wait,
which needs this core's done for call N — sent after this core reset `sem_ready`. Symmetrically for
`sem_done`. No host reset, no host read; trace replay reuses the cached semaphores.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one head (all Mt token rows, all of K and N) per worker; one stream per fabric_tx core |
| Grid | `grid = device.compute_with_storage_grid_size()`; cores enumerated row-major as logical CoreCoords; fabric_tx = the last `num_streams` (4) cores; workers = the first `num_workers = min(H, 110, grid_cores − num_streams)` cores |
| Per-core work | `ttnn.split_work_to_cores(worker_core_range, H)` → heads_per_core (1 on 120 cores) and the remainder group; a core's heads are contiguous |
| Remainder | handled by split_work_to_cores' second group; tile geometry `Mt = ceil(T/32)`, `Kt = ceil(L/32)`, `Nt = ceil(V/32)` per head |
| Head → stream | head h's destination `d = h // hpc`; `d == r` → local (worker writes); else the stream whose (direction, hop, head range) covers `(d, h mod hpc)`; the stream's head list in head order is a RT arg of its fabric_tx reader as (worker core NoC x, y, block offset) triples |

One regime: no selection function; regime-pinned tests are the golden cell itself.

## Circular Buffers

All CB descriptors span **all program cores** (workers ∪ fabric_tx) so every CB has the same address
on every core (`pull_block` relies on it). Indices are any distinct values.

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_weight` | 0 | 1088 | `Kt · Nt` (64) | spans k, n of one head; streams h | Bfp8_b | reader (worker) | compute | one head (resident across m) |
| `cb_attn_rm` | 1 | 2048 (tile-sized page holding 64 B × 32 rows of RM data; 16 pages = 32 sticks × 1024 B) | `attn_rm_depth · mt_block · Kt` (32) | spans k, one mt block; streams m | Float16_b | reader (worker) | compute | per mt block |
| `cb_attn_tiled` | 2 | 2048 | `mt_block · Kt` (16) | spans k of one block; streams m | Float16_b | compute (tilize) | compute (matmul) | per mt block |
| `cb_out` | 3 | 1088 | `heads_per_core · Mt · Nt` (20) | spans m, n of the core's heads (bounded: heads_per_core = ceil(H/num_workers) ≤ 1 at ≥ 96 workers) | Bfp8_b | compute | writer (worker) | whole program (fabric_tx reads it in place) |
| `cb_tx_stage` | 4 | 1088 | `tx_buffers · tx_batch · tiles_per_packet` (64) | one packet per slot, batches of `tx_batch`; streams the stream's blocks | Bfp8_b | reader (fabric_tx) | writer (fabric_tx) | whole program |

`cb_out` has one CB producer (compute) and one CB consumer (the worker writer). fabric_tx reads its
pages over the NoC, not through CB credit; ordering comes from `sem_blocks_done` (bumped by the writer
after `cb_wait_front`, i.e. after the pack completed). The writer pops nothing until its last block.

Packet headers come from the helper's `PacketHeaderPool` (no CB).

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_weight_block` | Kt × Nt | partly (TensorAccessor) | DRAM W | `cb_weight`, Kt·Nt pushed once | resident until #6 |
| 2 | `load_attn_block` | mt_block·32 sticks | yes, `read_sticks_for_tilize` | DRAM attn | `cb_attn_rm`, mt_block·Kt pages pushed | — |
| 3 | `tilize_attn_block` | mt_block × Kt | yes, `tilize` | `cb_attn_rm`, mt_block·Kt pages, popped | `cb_attn_tiled`, mt_block·Kt | — |
| 4 | `project_block` | [mt_block × Kt] @ [Kt × Nt] | yes, `matmul_block` | `cb_attn_tiled` (popped), `cb_weight` (retained) | `cb_out`, mt_block·Nt | `cb_weight` unchanged |
| 5 | `publish_block` | mt_block × Nt | no helper (one NoC write per tile or one semaphore inc) | `cb_out` wait_front of the cumulative count, not popped | output L1 (local) / `sem_blocks_done` | `cb_out` holds every block till kernel end |
| 6 | `release_weight_block` | Kt × Nt | raw `cb_pop_front` | `cb_weight` | — | empty for the next head |
| 7 | `pull_block` | Nt tiles | no (remote NoC read) | worker `cb_out` (remote, by address) | `cb_tx_stage`, tiles_per_packet | — |
| 8 | `send_block` | tiles_per_packet | yes, `ScatterWriteChannel::write_scatter` | `cb_tx_stage`, popped after the write is issued (helper's flow control) | destination output L1 | — |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| boot (compute) | raw_api | `compute_kernel_hw_startup<SrcOrder::Reverse>(cb_attn_tiled, cb_weight, cb_out)` | `matmul_block_helpers.hpp:100` (requirement) | once at top | — | — | — |
| `load_attn_block` | helper | `dataflow_kernel_lib::read_sticks_for_tilize<cb_attn_rm, TilizeGranularity::TILE>` | `tilize_helpers_dataflow.hpp:88` | `(attn_accessor, mt_block*32, 1024, h*T + mt*32)` | DRAM | `cb_attn_rm` | `mt_block` (rows) |
| `tilize_attn_block` | helper | `compute_kernel_lib::tilize<Kt, cb_attn_rm, cb_attn_tiled>` | `tilize_helpers.hpp:206` | `(cb_attn_rm, cb_attn_tiled, mt_block)`, symmetric mode, default `InitAndUninit` | `cb_attn_rm` | `cb_attn_tiled` | `num_blocks = mt_block` |
| `project_block` | helper | `compute_kernel_lib::matmul_block<false, false, LastBlockTarget::Out, OutputCBLayout::SubblockMajor, InitMode::Short, InputPolicy::WaitAndPopPerKBlock, InputPolicy::WaitAndRetainOnLastBlock>` | `matmul_block_helpers.hpp:358`; shape `:160`; policy `:77` | `(cb_attn_tiled, cb_weight, cb_out, cb_out /*interm, K in one block*/, MatmulBlockShape::of(1, 1, mt_block, Nt, Kt, 1), {}, {}, Nt)` | `cb_attn_tiled`, `cb_weight` | `cb_out` | block size `mt_block × Nt`; `in0_block_k = Kt`; subblocking left to the helper's legality (1 × 4 ≤ DEST 8) |
| `load_weight_block` | raw_api | `TensorAccessor` + `noc_async_read_page`, `noc_async_read_barrier` | `tech_reports/tensor_accessor/tensor_accessor.md` | tile ids `h·Kt·Nt + i` | DRAM | `cb_weight` | — |
| `release_weight_block` | raw_api | `cb_pop_front(cb_weight, Kt*Nt)` | — | — | `cb_weight` | — | — |
| `publish_block` (local) | raw_api | `noc_async_write_page` with an output `TensorAccessor`, `noc_async_write_barrier` | `tensor_accessor.md` | tile id `h_local·n·Mt·Nt + (r·Mt + mt)·Nt + nt` | `cb_out` | output L1 | — |
| `publish_block` (remote) | raw_api | `noc_semaphore_inc(get_noc_addr(get_semaphore(sem_blocks_done)), 1)` | `api/dataflow/dataflow_api.h` | — | `cb_out` | — | — |
| `pull_block` | raw_api | `noc_async_read(worker_noc_addr(sem), …)` poll + `noc_async_read(worker cb_out addr, cb_tx_stage write ptr, 4352)` | dataflow_api | — | — | `cb_tx_stage` | `tiles_per_packet` |
| `send_block` | helper | `ccl::FabricStreamSender<DirectConn>(arg_idx, is_forward, l1_alignment).open(route)` → `arm_scatter_write(1088, tiles_per_packet)` → `write_scatter(addrs, n, l1)` | `ccl_helpers_dataflow.hpp:486-507`, `:136`, scatter channel `:338` | route `{dst_mesh_id, dst_chip_id}` | `cb_tx_stage` | remote output | `tiles_per_packet` |
| handshakes | helper | `stream.arm_inc(1)` → `inc(peer_sem_noc_addr)` for ready and for done; local `noc_semaphore_wait_min` + `noc_semaphore_set(…, 0)` | `ccl_helpers_dataflow.hpp:141`, contract `:120-131` of the file header | — | — | — | — |
| host: fabric connection | helper (host) | `ttnn._ttnn.ccl_host.build_ccl_fabric_rt_args` | `ttnn/cpp/ttnn-nanobind/ccl_host.cpp:64-90` | first in fabric_tx writer RT args | — | — | — |
| host: cross-chip semaphores | helper (host) | `ttnn._ttnn.ccl_host.make_ccl_semaphore(mesh_device, initial_value=0)` (or `ttnn.create_global_semaphore(mesh_device, tx_cores, 0)`), cached per mesh device | `ccl_host.cpp:54-63` | created on the first (eager) call only — it runs a host Synchronize, which a trace capture must not contain | — | — | — |

> **Perf 1 (as built now):** `load_attn_block` / `load_weight_block` are raw `noc_async_read` on BOTH NoCs from NCRISC (per-bank NoC mask in the core's common-arg word, `NOC1_BANK_MASKS`; transaction-id barriers; worker reader + writer in `DM_DYNAMIC_NOC` mode), bypassing `read_sticks_for_tilize` (single NoC, plain barrier). `send_block` keeps the helper's `DirectConn` but issues packets with the tt_fabric linear `set_state` / adapter calls and destination addresses pre-encoded in `cb_tx_addrs`, bypassing `FabricStream::arm_*` / `write_scatter`. Worker kernels take only common runtime args (a per-core table indexed by logical core). Measurements and the helper-bypass table: `changelog.md` Perf 1.

Helpers considered and rejected:

| Raw entry | Helper considered | Mismatch (file:line) |
|---|---|---|
| `load_weight_block` | `read_sticks_for_tilize` | reads row-major sticks into a tilize CB (`tilize_helpers_dataflow.hpp:88`); W is already tiled bf8_b. No tile-reader helper in `kernel_lib/` (`ls kernel_lib`: only the tilize stick reader, local copy and mcast dataflow helpers) |
| `publish_block` (local) | `local_copy_helpers_dataflow.hpp` | copies within one core's L1 / CBs; the target is an interleaved L1 tensor on other cores addressed per tile |
| `pull_block` | `mcast/kernel/mcast_args.hpp` (`McastArgs` receiver) | the mcast protocol is sender-driven multicast with a per-round receiver handshake; this is a one-to-one pull of one worker's block by address, no multicast |
| `pull_block` | `ccl::FabricStream` | fabric egress only; the file header states the receive/ingress and local reads stay op-owned (`ccl_helpers_dataflow.hpp:127-131`) |
| `release_weight_block` | `matmul_block` in1 policy `WaitAndPopPerKBlock` | that pops W after every call (`matmul_block_helpers.hpp:64-77`), i.e. after every mt block; W must survive the head's Mt blocks |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Antipode half on the "wrong" arc | Under the 2D fabric the route is the destination's action map; the antipode packet is injected into the chosen direction's router and relies on the intermediate chip forwarding it on (its own 1-hop route to the destination) | First implementation stage: a probe with only the two antipode streams, checking both halves land; precedent `all_to_all_async_generic_program_factory.cpp:188-205`. Fallback if it does not hold: the intermediate chip's fabric_tx on the same link relays the half (store-and-forward), same link load |
| `done` arriving before the data it follows | fabric data writes and the atomic inc land through the destination router's NoC | arm the inc with flush (`arm_inc` programs Val|Flush, `ccl_helpers_dataflow.hpp:139-141`); the inc goes on the same stream after the last packet |
| Writing into a chip still running its previous program | the output L1 may overlap the previous program's CBs on a slower chip | no data packet before `sem_ready` from the destination's peer (sent at program start) |
| Global semaphore creation inside a trace | `make_ccl_semaphore` runs a host Synchronize | create on the first call (eager; the regression test always calls eagerly first) and cache per mesh device; never create during capture |
| CB address mismatch between worker and fabric_tx | `pull_block` reads `cb_out` by address | every CB descriptor uses the same CoreRangeSet (all program cores) |
| `cb_out` pages overwritten | no back-pressure from fabric_tx to the worker | `cb_out` holds all of the core's blocks (heads_per_core·Mt·Nt); bounded because heads_per_core ≤ ceil(96/num_workers) |
| bf8_b output packed from a bf16 interm | `matmul_block_helpers.inl:407-409, 461-465`: with K>1 blocks and no fp32 dest the pack format is not switched | `num_k_blocks = 1` (`kt_block = Kt`), interm = `cb_out`; never raise the K block count without packer_l1_acc |
| Precision at fp32_dest_acc_en = False | K = 512 accumulated in 16-bit DEST, output rounded to bf8_b | gate 0.9995 (feature_spec estimate 0.9999 for this accumulation); first stage measures PCC on one chip before the fabric is wired |
| Router worker channels | one worker connection per (direction, link) router | exactly one fabric_tx per (direction, link) by default; with `MUX_OVERFLOW_HEADS` > 0 the streams sharing a link go through one fabric mux core (R6, Refinement 3) |
| Hard-coded topology | rule: derive ring, neighbours, grid | every quantity from `mesh_device.shape`, coords, `get_fabric_node_id`, `compute_with_storage_grid_size()`, `worker_core_from_logical_core` |
| Head mapping across the two rings | each row is its own ring | the plan only iterates coordinates sharing every index except `cluster_axis` |

## Structural impossibilities

None beyond `INVALID = []`.

## Catalog entries used

| Entry (`ttnn/ttnn/operations/examples/master.md`) | Knob it informed |
|---|---|
| `compute_block_size` (:220-243) | tilize → matmul_block in one kernel with W retained (`WaitAndRetainOnLastBlock`, `num_k_blocks=1`) |
| `noc_placement` (:20) | reads on NoC 0 (worker reader), local writes on NoC 1 |
| `trid_double_issue` (:67) / `double_buffer` (:51) | `attn_rm_depth = 2`, one barrier per 32-stick block |
| `matmul_output_subblock` (:159-163) | block 1 × 4 within DEST (subblocking left to the helper) |
| No fabric entry exists | packet size and stream layout from first principles + `all_to_all_async_generic` |
