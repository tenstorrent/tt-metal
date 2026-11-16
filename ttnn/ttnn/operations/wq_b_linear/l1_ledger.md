# L1 Ledger: wq_b_linear

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes from `op_design.md`
Blocking Model: `m` (`block_m_tiles`), `n` (`core_n_tiles`), `k` (block `tensor_k_tiles`, streamed in
`k_block_tiles`). `fp32_dest_acc_en = False` (16-bit DEST).

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_x_resident` | `block_m_tiles · k_block_tiles · num_k_blocks` (= 5·3·16 = 240; whole-K resident, writer reserves all rounds up front) | 240: one write-once landing slot per multicast round (`num_x_rounds` = 4 slots of `x_round_k_blocks` = 4 K-blocks) (x is resident; rounds land ahead of compute) | `{m: spans → block_m_tiles, n: streams → replica, independent of core_n_tiles, k: spans → k_block_tiles · num_k_blocks}` | Float16_b (bf16 input; dataflow landing, unpacked to SrcB) | writer | compute | whole kernel (first round lands → last K-block popped) | none: concurrent with `cb_w_stream` and `cb_y_tiles` for the whole kernel. Write-once slots are what let Refinement 1 drop the ready handshake: each round has its own fixed-sender Flag channel (`X_MCAST_CHANNELS="per_round"`, one semaphore per round), fired once per launch. A 2-slot ring (R3) would need the handshake back; not taken because x is bounded at 491,520 B and residency keeps rounds ahead of compute |
| `cb_w_stream` | `k_block_tiles · num_k_blocks · core_n_tiles_max` (= 3·16·2 = 96; whole-K resident, reader reserves the whole slice up front) | `48 · core_n_tiles` (96 on the 63 two-column cores, 48 on the 2 single-column cores at the Refinement 2 defaults): whole slice resident; `w_reads_in_flight` (shipped 1) K-blocks outstanding. With `W_NOC1_K_BLOCKS > 0` (parked at 0) the writer fills the last K-block slots over NoC1 and the reader pushes them after a local handoff semaphore; the reader stays the only producer | `{m: streams → absent from W, n: spans → core_n_tiles_max, k: spans → k_block_tiles · num_k_blocks}` | Bfp8_b (bf8b weight, unpacked to SrcA) | reader | compute | whole kernel | none: concurrent with the other two. Capacity > live set on single-column cores (52,224 B) because one descriptor serves both groups; per-group descriptors are the fix if L1 ever binds |
| `cb_y_tiles` | `block_m_tiles · core_n_tiles_max` (= 10) | `5 · core_n_tiles`: the accumulator over all K-blocks | `{m: spans → block_m_tiles, n: spans → core_n_tiles_max, k: streams → accumulated in place over num_k_blocks}` | Float16_b (matches 16-bit DEST and the bf16 output) | compute | writer | first K-block pack → store | none: it is the destination and the accumulator at once (pack-into-destination + packer L1 accumulation replace the `interm` spill buffer); concurrent with both inputs |
| `cb_x_slice` (R4, `K_REMAINDER_SPLIT`, parked off) | alias of `cb_x_resident` (second format descriptor on the same CB allocation, 0 extra bytes) | the K-blocks of this core's K-slice | `{m: spans, n: replica, k: spans}` (same memory as `cb_x_resident`) | Float16_b | writer (pushes every K-block on both indices) | compute (skips `kb_start` K-blocks, consumes the slice) | whole kernel | shares `cb_x_resident`'s memory by construction: the slice matmul needs x K-blocks the main matmul has already popped or not yet reached, so it reads them through its own FIFO pointers instead of a copy |
| `cb_w_slice` (R4) | `k_block_tiles · ceil(num_k_blocks / num_slices)` (= 3·3 = 9) | this core's W K-slice of one leftover column, read first | `{m: absent, n: 1 leftover column, k: spans → slice K-blocks}` | Bfp8_b | reader | compute | kernel start → slice matmul | none: separate from `cb_w_stream` so the slice can be read first and consumed mid-stream without reordering the main FIFO |
| `cb_partial` (R4) | `block_m_tiles` (= 5) | the `(5, 1)` K-slice partial | `{m: spans, n: 1, k: accumulated in place over the slice}` | Float16_b (same format and rounding as the `cb_y_tiles` L1 accumulation, no extra rounding pass) | compute | writer (slice > 0: unicast to owner) / compute (owner: combine) | slice matmul → send or combine | none: concurrent with the main accumulator in `cb_y_tiles`; 10 KB |
| `cb_landing` (R4) | `block_m_tiles · (num_slices − 1)` (= 25), on the full grid so its address is identical on every core | landed partials, tile `t` of slice `s` at slot `t · (num_slices − 1) + s − 1` | `{m: spans, n: 1, k: one partial per remote slice}` | Float16_b | owner writer (waits `LANDING_SEMAPHORE_ID` = num_slices − 1 remote writes, then pushes) | owner compute (combine) | remote unicasts → combine | none: remote writers target it before the owner reaches the combine; only owners use it, but it is allocated grid-wide for the shared address (51 KB) |

R4 changes `cb_y_tiles` to `block_m_tiles · (core_n_tiles_max + 1)` (= 10): the owner holds its own
column plus the combined leftover column (own partial copied, landed partials L1-accumulated in slice order).
One more semaphore (`LANDING_SEMAPHORE_ID`, id 1) is declared in both regimes; the x channels start at id 2.
R4 per-core footprint at p100a: 491,520 + 52,224 (W main) + 9,792 + 20,480 + 10,240 + 51,200 = 635,456 B.

Reuse patterns applied: pack-into-destination and fold-into-accumulator (packer L1 acc) eliminate the
matmul `interm` CB; no phase boundary in this op is storage. In-place transform and aliasing: no
disjoint lifetimes exist (all three CBs are live for the whole kernel).

## Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `block_m_tiles` = `tensor_m_tiles` | = 5 | `validate()`: `x.shape == [1,1,160,1536]` |
| `tensor_k_tiles` | = 48 | same |
| `tensor_n_tiles` | = 128 | `validate()`: `W.shape == [1536, 4096]` |
| `k_block_tiles` | divisor of 48, ≤ `K_BLOCK_TILES` = 3 | host clamp (mechanism cap) |
| `num_k_blocks` | = 48 / `k_block_tiles` = 16; ≥ 1 | derived |
| `num_x_rounds` | = `num_k_blocks / x_round_k_blocks` = 4; ≤ `MAX_X_MCAST_CHANNELS` = 8 | host: `_x_round_k_blocks` raises the round size until it fits |
| `num_cores` | 16 ≤ grid; `num_cores` ≤ min(grid, `tensor_n_tiles`, ceil(`tensor_n_tiles` / `CORE_N_TILES`)) (65 on p100a: 63 bank-coalesced two-column cores + 2 single-column) | `validate()` raises `RuntimeError` below 16 grid cores; host `_core_n_counts` |
| `core_n_tiles_max` | = `ceil(128 / num_cores)` ≤ 8 (2 on p100a, 3 on 8×7 WH) | follows from `num_cores ≥ 16` and `CORE_N_TILES = 2` |

## Total per-core footprint

```
L1_cb = 2048 · block_m_tiles · tensor_k_tiles                  (cb_x_resident, fixed by contract)
      + 1088 · tensor_k_tiles · core_n_tiles_max                (cb_w_stream, scales with n share)
      + 2048 · block_m_tiles · core_n_tiles_max                 (cb_y_tiles, scales with n share)
      = 491,520 + 62,464 · core_n_tiles_max  bytes
```

p100a (`core_n_tiles_max = 2`): 616,448 B. Worst case under the bound (`core_n_tiles_max = 8`):
991,232 B. Plus the core's share of the L1-interleaved output y (1,310,720 B over the device's L1
banks, ≈ 12 KB per bank on p100a). `k_block_tiles` does not change the footprint (x and W are
whole-K resident); it changes only push quanta.

## Data-movement budget (R1, p100a, 110 cores)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| W (6,684,672 B) | 1 | each core reads only its own n columns, all k, once, into a whole-K resident CB; m is not split, so no core re-reads W | 0 (W lands directly in its consumer) |
| x (491,520 B) | 1 | each K-block round is read once from DRAM by its injector | multicast to every other core: 491,520 B inbound per core, ≈ 53.6 MB delivered over 109 receivers (NoC1) |
| y (1,310,720 B) | 0 | output is allocated in L1 | unicast L1 writes, ≤ 20,480 B per core (NoC1) |

Totals: DRAM 7,176,192 B (the SOL byte count; 16,018 ns at 448 GB/s). Cross-core ≈ 54.9 MB
delivered; busiest-core NoC inbound ≤ 104,448 (W, NoC0) + 491,520 (x, NoC1) B.

> Cheapest-traffic split considered: R5 `mn_2d_w_bcast_along_m` — DRAM: 0 B delta; cross-core:
> ≈ −18 MB delivered (≈ 37 MB vs ≈ 55 MB), busiest-core inbound ≈ 411 KB vs ≤ 595 KB.
> Implemented: R1 `n_split_x_bcast_resident`. Deferred because R5's saving is in the cross-core
> tier, which is below the binding DRAM term in both splits, while R5 makes the binding W stream
> worse: W DRAM reads concentrate on 22 injectors (~20 GB/s each instead of ~4–6.5 GB/s on 110
> consumers) and every W byte takes a multicast hop before compute. R1's x-broadcast channel and
> per-core W reader are the two pieces R5 reuses (as the per-m-row x channel and the
> column-segment W injector), so R1 does not foreclose it.
>
> Verifier measurement (Phase 0) contradicts the deferral premise: the cross-core tier binds.
> x broadcast + y store alone (no W reads, no compute) is 25.4 µs of the 26.6 µs op; W-only is
> 20.8 µs. Refinement 1 in `op_requirements.md` attacks the x broadcast with knob-turns first
> (round granularity, protocol); an x-topology scheme-change (R5 or concurrent banded injectors)
> is its named escalation if the knob-turns leave x above the W stream.
>
> After Refinement 1 (per-round fixed-sender channels, no handshake, diagonal injectors) the
> x path no longer binds: x broadcast + y store alone is 16.0 µs vs 21.0 µs for the W stream alone,
> and the full op (23.8 µs) sits 0.9 µs above the W + compute ablation with a trivial x (22.95 µs).
> The R5 escalation is not needed for x; the binding term is the W stream plus the compute tail on
> the 2-column cores (Refinement 2).
>
> Refinement 2 (`CORE_N_TILES = 2`, `W_BANK_COALESCE`): the split is now 65 active cores (63 cores
> own a bank-strided column pair `(n, n + 7)`, 2 cores own one column) instead of 18 × 2 + 92 × 1.
> DRAM crossings are unchanged (W ×1, x ×1). W arrives in 2176 B single-bank transfers on the
> paired cores. x still multicasts to the full grid (45 idle receivers; the rectangle cost is per
> link, not per receiver). One extra semaphore (`W_HANDOFF_SEMAPHORE_ID`, id 0) is declared for
> the parked NoC1 W share; the x channels start at id 1. The W stream alone is 19.8 µs (≈ 343 GB/s
> from reader start, 77 % of the 448 GB/s peak) and binds via NoC0 row-link load next to the DRAM
> endpoints, not DRAM banks: the same stream on NoC1 takes 36 µs, and adding a NoC1 share drops
> W-only to 17.9 µs.

>
> Refinement 3 (R4 `n_split_k_remainder_combine`, built, parked behind `K_REMAINDER_SPLIT = False`):
> 110 cores each own one full column; the 18 leftover columns are cut into 6 K-slices (2–3 K-blocks)
> over 108 cores. DRAM crossings unchanged (W ×1, x ×1). Cross-core adds 18 × 5 × 5 tiles × 2048 B
> = 921,600 B of partials (unicast, NoC1). Busiest core W 104,448 → 62,016 B, compute 480 → 285
> tile-matmuls. Measured 22.7 µs vs 22.4 µs for the Refinement 2 split: the W stream on 110 cores
> of single-tile (1088 B) reads ends ≈ 0.8 µs later than on 65 cores of 2176 B coalesced reads,
> which outweighs the ≈ 0.7 µs shorter compute tail.
