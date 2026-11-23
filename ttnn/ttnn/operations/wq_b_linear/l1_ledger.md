# L1 Ledger: wq_b_linear

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes: `m` (`block_m_tiles` = 5), `k` (`block_k_tiles` = 8, `num_k_blocks` = 6 — implementer turned the knob from the design's 4/12, see Implementation notes), `n` (`block_n_tiles` = 2). `fp32_dest_acc_en = False` (16-bit DEST).

## Worker cores (64)

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_x_resident` | `block_m_tiles · block_k_tiles · x_depth_rounds`, `x_depth_rounds = num_k_blocks` → 240 × 2048 B = 491,520 B | 240 worst case: the sender may land every round before compute pops any (no back-pressure in the handshake-free scheme) | `{m: spans → block_m_tiles (= tensor_m_tiles), k: spans → block_k_tiles·num_k_blocks (= tensor_k_tiles, resident), n: — (x does not vary with n)}` | Float16_b (bf16 activation as stored; never passes through DEST) | writer (push per received round; bytes written by the sender's multicast) | compute | whole program | Not shared: live for the whole program, concurrent with every other CB. Capacity = live set by construction (rounds are never reused — that is what removes the ack round trip). |
| `cb_w_block` | `block_k_tiles · block_n_tiles · w_depth_blocks`, `w_depth_blocks = num_k_blocks` → 96 × 1088 B = 104,448 B | 96 worst case (reader runs ahead of compute while compute waits for x round 0) | `{m: — (W does not vary with m), k: spans → block_k_tiles·w_depth_blocks, n: spans → block_n_tiles}` | Bfp8_b (weight as stored; DRAM page format) | reader | compute | whole program | Not shared: concurrent with `cb_x_resident` and the accumulator. Capacity above the 8-page minimum is the stall-shadow decision (Blocking Model, buffer depths); lamp measures depth 3–4. |
| `cb_interm_acc` | `block_m_tiles · block_n_tiles` → 10 × 2048 B = 20,480 B | 10 (the whole output block's partial sum, accumulated in place) | `{m: spans → block_m_tiles, k: streams (accumulated), n: spans → block_n_tiles}` | Float16_b — equals 16-bit DEST width; packing DEST to it is lossless | compute | compute | K-steps 0..num_k_blocks−1 | Could alias `cb_out_block` (helper permits a shared region for SubblockMajor, `matmul_block_helpers.hpp:250-251`; TileRowMajor requires its OWN region, `:252`). This design uses TileRowMajor, so it cannot share. |
| `cb_out_block` | `block_m_tiles · block_n_tiles · out_depth_blocks`, `out_depth_blocks = 1` → 10 × 2048 B = 20,480 B | 10 | `{m: spans → block_m_tiles, k: streams (written once after the last K-step), n: spans → block_n_tiles}` | Float16_b (output dtype bf16 = DEST width) | compute | writer | last K-step → store | Not shared with `cb_interm_acc` (TileRowMajor, above). Not packed straight into the output tensor: y is L1-interleaved across all cores' banks, not this core's L1. |

## Sender core (1)

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_x_resident` (same descriptor as on workers, so the address matches) | 240 × 2048 B = 491,520 B | 240: each round's slot is the multicast source and is never rewritten | `{m: spans → block_m_tiles, k: spans → tensor_k_tiles, n: —}` | Float16_b | x-reader kernel (NCRISC / NoC0, DRAM reads, push per round) | multicast kernel (BRISC / NoC1, wait/pop per round; the slot is the multicast source) | whole program | Not shared: it is the only buffer on the sender core. A 2-slot ring would need the consumer handshake (rejected regime). |

## Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `tensor_m_tiles` | = 5 | `validate()`: `x.shape == [1,1,160,1536]` |
| `tensor_k_tiles` | = 48 | same + `W.shape == [1536,4096]` |
| `tensor_n_tiles` | = 128 | `W.shape == [1536,4096]` |
| `num_k_blocks` | = `tensor_k_tiles / block_k_tiles` = 12 | host assert `tensor_k_tiles % block_k_tiles == 0` |
| `x_depth_rounds`, `w_depth_blocks` | = `num_k_blocks` | scheme (resident x, stall-shadow W) |
| `out_depth_blocks` | = 1 | one output block per core |

## Footprint

Worker: `2048·block_m_tiles·block_k_tiles·num_k_blocks` (x, = 2048·tensor_m_tiles·tensor_k_tiles) `+ 1088·block_k_tiles·block_n_tiles·num_k_blocks` (W slice, scales with `block_n_tiles`) `+ 2·2048·block_m_tiles·block_n_tiles` (acc + out, scale with `block_n_tiles`) = 491,520 + 104,448 + 40,960 = **636,928 B** (< ~1.4 MB usable L1). `block_k_tiles` does not change the total (both resident CBs scale with `block_k_tiles · num_k_blocks = tensor_k_tiles`). Sender: **491,520 B**.

## Data-movement budget

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| x (491,520 B) | 1 | read once by the single sender core; workers receive it by multicast and keep it resident | one multicast tree, 12 rounds × 40,960 B to 64 receivers (31.5 MB delivered, 491,520 B injected) + 12 Counter increments |
| W (6,684,672 B) | 1 | each worker reads only its `block_n_tiles` columns; no M bands, so no re-read | none |
| y (1,310,720 B) | 0 | output is L1 interleaved | 640 tile writes core → L1 bank over the NoC |

Totals: DRAM 7,176,192 B (= SOL minimum; 16,018 ns at 448 GB/s on p100a). NoC: 491,520 B multicast injection, 6,684,672 B DRAM→core, 1,310,720 B core→L1. Per worker NoC in: 104,448 B W + 491,520 B x; out: 20,480 B.

> Cheapest-traffic split considered: N-split with x multicast — DRAM 7.18 MB (minimum), no extra NoC beyond the x tree. Implemented: N-split with x multicast. It is the cheapest; the K-split (+2.62 MB NoC partials) and 2-D split (second tree) are `deferred` rows in the regime table.

### Split search (op model)

`op_model.py matmul 5 48 128 --designs --c-mem l1 --fidelity HiFi2 --a-dtype bf16 --b-dtype bfp8_b --c-dtype bf16` — shortlist and curve in `op_design.md` (Blocking Model). Phase 0: 64 workers + 1 sender, kb 4, rank-4 config (17.39 us, tie with rank 1 at 16.53 us).

### Data-movement search (`estimate.py sweep --phase Planner`, recorded in `dm_estimates.json`)

| Sweep (record id) | Rank | Knobs | pass_ns | tie_with_best | L1 bytes/core |
|-------------------|------|-------|---------|---------------|---------------|
| W read, 64 cores, 1088 B pages, column page map (`db941791`) | 1 | batch 8, NoC split | 15,594 | yes | 8,704 |
| | 2 | batch 4, NoC split | 15,600 | yes | 4,352 |
| | 5 | **batch 8, NoC 0** ← Phase 0 | 15,673 | yes | 8,704 |
| x read by the sender, 2048 B pages (`baa09d76`) | 1 | **batch 80, NoC 0/1** | 7,329 | yes | 163,840 |
| | 3 | batch 40 | 8,373 | no | 81,920 |
| x multicast to the 8×8 rect, sender (10,9) (`6f9e7cf5`) | 1 | **16 KB packets, 10 between barriers, NoC 1** | 14,439 | yes | 163,840 |
| | 2 | 16 KB, 4 between barriers, NoC 1 | 16,851 | no | 65,536 |

Phase 0 choices from the sweeps: W reader batch 8 (one K-step per barrier) on NoC0 — tie with rank 1; NoC split for W is excluded because the op model shows W on NoC1 colliding with the x multicast tree (22.6 vs 16.6 us). x: the sender issues one round (20 pages) per barrier and overlaps the next round's reads with the current multicast; the sweep's batch-80 figure (7.3 us for all of x) is a lower bound on the sender's read, which is off the critical path per the op model (ablating x: 15.0 vs 17.4 us). The multicast estimate (14.4 us standalone, path-reserving, non-posted) runs concurrently with the 15.6 us W stream on the other NoC. An earlier exploratory W sweep (`0d5aab10`) varied the core count; it is superseded because the core count is fixed by the split. No refusals.

## Implementation notes (ttnn-implementer)

- `block_k_tiles` 4 → 8 (`num_k_blocks` 12 → 6). Totals unchanged: both resident CBs scale with
  `block_k_tiles · num_k_blocks = tensor_k_tiles`. Measured: each Counter-mode multicast round has
  ~1 µs fixed cost (write barrier + counter increment over 64 receivers), so fewer, larger rounds win.
  Device kernel time (p150, 11×10, 8 banks): kb 4 → 31.8 µs, 6 → 27.4, 8 → 26.4, 12 → 26.5 (split sender).
- Sender core is split over its two RISCs (design perf option "x sender read NoC"): NCRISC reads x on NoC0
  and pushes each round to `cb_x_resident`; BRISC waits for each round and multicasts it on NoC1. One producer
  and one consumer for the CB on that core. Single-RISC sender measured 30.5 µs at kb 12, split 26.5 µs.
- Data-movement budget unchanged (x 1×, W 1× from DRAM; y to L1).
