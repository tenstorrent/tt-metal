# L1 Ledger: wq_b_linear

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes from `op_design.md`
Blocking Model: `m` (`block_m_tiles`), `n` (`core_n_tiles`), `k` (block `tensor_k_tiles`, streamed in
`k_block_tiles`). `fp32_dest_acc_en = False` (16-bit DEST).

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_x_resident` | `block_m_tiles · k_block_tiles · x_cb_k_blocks` (= 5·6·8 = 240; `x_cb_k_blocks = num_k_blocks`) | 240: one write-once landing slot per multicast round (x is resident; rounds land ahead of compute) | `{m: spans → block_m_tiles, n: streams → replica, independent of core_n_tiles, k: spans → k_block_tiles · num_k_blocks}` | Float16_b (bf16 input; dataflow landing, unpacked to SrcB) | writer | compute | whole kernel (first round lands → last K-block popped) | none: concurrent with `cb_w_stream` and `cb_y_tiles` for the whole kernel. The shipped multicast (`X_MCAST_LINKED=True`: Flag + ready handshake, chosen because it issues linked multicasts, measured 19.1 µs vs 26.3 µs for the bare broadcast) would also permit a 2-slot ring (R3); not taken because x is bounded at 491,520 B and residency keeps rounds ahead of compute |
| `cb_w_stream` | `k_block_tiles · w_cb_k_blocks · core_n_tiles_max` (= 6·8·2 = 96; `w_cb_k_blocks = num_k_blocks`) | `48 · core_n_tiles` (96 on core_group_1, 48 on core_group_2): whole slice resident; `w_reads_in_flight` (shipped 1) K-blocks outstanding | `{m: streams → absent from W, n: spans → core_n_tiles_max, k: spans → k_block_tiles · num_k_blocks}` | Bfp8_b (bf8b weight, unpacked to SrcA) | reader | compute | whole kernel | none: concurrent with the other two. Capacity > live set on core_group_2 (52,224 B) because one descriptor serves both groups; per-group descriptors are the fix if L1 ever binds |
| `cb_y_tiles` | `block_m_tiles · core_n_tiles_max` (= 10) | `5 · core_n_tiles`: the accumulator over all K-blocks | `{m: spans → block_m_tiles, n: spans → core_n_tiles_max, k: streams → accumulated in place over num_k_blocks}` | Float16_b (matches 16-bit DEST and the bf16 output) | compute | writer | first K-block pack → store | none: it is the destination and the accumulator at once (pack-into-destination + packer L1 accumulation replace the `interm` spill buffer); concurrent with both inputs |

Reuse patterns applied: pack-into-destination and fold-into-accumulator (packer L1 acc) eliminate the
matmul `interm` CB; no phase boundary in this op is storage. In-place transform and aliasing: no
disjoint lifetimes exist (all three CBs are live for the whole kernel).

## Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `block_m_tiles` = `tensor_m_tiles` | = 5 | `validate()`: `x.shape == [1,1,160,1536]` |
| `tensor_k_tiles` | = 48 | same |
| `tensor_n_tiles` | = 128 | `validate()`: `W.shape == [1536, 4096]` |
| `k_block_tiles` | divisor of 48, ≤ `K_BLOCK_TILES` = 6 | host clamp (mechanism cap) |
| `num_k_blocks` | = 48 / `k_block_tiles` = 8; ≥ 1 | derived |
| `num_cores` | 16 ≤ `num_cores` ≤ 128 | `validate()` raises `RuntimeError` below 16; `split_work_to_cores` uses ≤ `tensor_n_tiles` cores |
| `core_n_tiles_max` | = `ceil(128 / num_cores)` ≤ 8 (2 on p100a, 3 on 8×7 WH) | follows from `num_cores ≥ 16` |

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
