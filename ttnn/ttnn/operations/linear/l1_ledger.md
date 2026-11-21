# L1 Ledger: linear

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking: `op_design.md` § Blocking Model.
Target: `quasar` (4 MiB L1 per cluster, DFB ring ≤ 1,048,560 B), bfloat16 tiles (`tile_bytes` = 2048 B).
Block axes: **rows** (`block_h`), **columns** (`block_w`), **K** (`block_k`), and the thread plan's **C**
(compute threads; capacities of rings with a BLOCKED or STRIDED compute side are counted per NEO × C).

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_act_tiles` | `ACT_DEPTH * block_h * block_k` | `block_h * block_k` (the chunk all C NEOs are multiplying; ALL: one copy serves every NEO) | `{rows: spans → block_h, K: spans → block_k (streams over num_k_chunks), columns: streams (shared by the round's C·block_w columns, not × C)}` | bf16, the dtype of `a`. A dataflow payload unpacked to SrcB, never a DEST product, so audit 2 does not apply | `reader_act` (1 hart) | `compute` (ALL, C NEOs) | whole kernel; one chunk per K step | none. Live with `cb_weight_tiles` during every chunk (both feed the same `matmul_block`) and pipelined against the next chunk; capacity = 2 × live set for that double buffering (`ACT_DEPTH`) |
| `cb_weight_tiles` | `WEIGHT_DEPTH * C * block_k * block_w` | `C * block_k * block_w` (one block per NEO) | `{rows: streams (re-read per row block), columns: spans → block_w per NEO × C, K: spans → block_k (streams over num_k_chunks)}` | bf16 (dtype of `a`; dataflow payload to SrcA) | `reader_weight` (1 hart) | `compute` (BLOCKED, `block_size = block_k*block_w`) | whole kernel | none. Concurrent with `cb_act_tiles` (same matmul) and with the next chunk; capacity = 2 × live set (`WEIGHT_DEPTH`). Different producer kernel from `cb_act_tiles`, so the two cannot be one ring (one producer per ring) |
| `cb_bias_tiles` (only when `has_bias`) | `BIAS_DEPTH * C * block_w` | `C * block_w` (one bias block per NEO, read once at round start) | `{rows: streams (broadcast: one bias row serves all block_h rows), columns: spans → block_w per NEO × C, K: streams (independent of K)}` | bf16 (dtype of `a`; B2D datacopy into DEST) | `reader_weight` | `compute` (BLOCKED, `block_size = block_w`) | first op of every round | none. Its live range (round start) overlaps the previous round's last weight chunk in flight; it has a different page role from `cb_weight_tiles`, and sharing a ring would interleave two block sizes on one BLOCKED binding (one `block_size` per binding). Capacity = 2 × live set so round r+1's bias lands during round r |
| `cb_output_tiles` | `OUT_DEPTH * C * block_h * block_w` | `C * block_h * block_w` (one packed block per NEO) | `{rows: spans → block_h, columns: spans → block_w per NEO × C, K: streams (folded into DEST; never stored)}` | bf16, the output dtype (R9: the result has `a`'s dtype): an output ABI requirement. Under fp32 DEST the pack rounds once, at the final store, which is the contract | `compute` (STRIDED, C NEOs) | `writer` (STRIDED, 2 harts) | per round | none. It is the destination itself (Rule 3.1: the product packs straight here, no scratch). Concurrent with every input ring; capacity = 2 × live set (`OUT_DEPTH`) so packing overlaps writing |

**Inventory record (Rules 1 and 3, applied before any solve).** There is no product ring: the matmul accumulates the whole K in DEST and packs into `cb_output_tiles` (pattern 1). There is no bias-add intermediate: the bias is copied into DEST first and the matmul accumulates onto it (pattern 4). There is no partial-sum ring: no spill, since a block equals the DEST capacity. Pattern 2 (in place) and pattern 3 (aliasing) found nothing: all four rings are live at once in the round pipeline. The only budget predicate is the `block_k` choice (`select_blocking` step 5); it acts on this minimal inventory and picks the coarsest K chunk.

## Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `block_h`, `block_w` | `block_h * block_w <= dest_tiles` (4 with `fp32_dest_acc_en`, 8 without); `block_h \| tensor_m_tiles`, `block_w \| tensor_n_tiles` | `select_blocking` steps 1-2 (DEST cap, `llk_unpack_matmul.h:238`) |
| `block_k` | `block_k \| tensor_k_tiles`; `ACT_DEPTH*block_h*block_k*tile_bytes <= DFB_RING_CAP_BYTES`; `WEIGHT_DEPTH*C*block_k*block_w*tile_bytes <= DFB_RING_CAP_BYTES`; `block_k*block_w <= 1023`; total ≤ `budget` | `select_blocking` step 5 |
| `C` | ∈ {1, 2, 4}; Phase 0 = 4 | `THREAD_PLAN`, checked on the host (C ≤ 4 NEOs, R + W ≤ 6, W \| C) |
| `ACT_DEPTH`, `WEIGHT_DEPTH`, `BIAS_DEPTH`, `OUT_DEPTH` | 2 each (Phase 0) | host constants |
| `has_bias` | 0 or 1 | the `bias` axis ≠ None |
| `tile_bytes` | 2048 (bf16) | `a.buffer_page_size()` |
| `DFB_RING_CAP_BYTES` | 1,048,560 | target page, *Cluster model* (no run-time query exists) |
| `budget` | `ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES` (`L1_RESERVE_BYTES = 65536`); below 4,194,304 B on this target, the exact value is the query's | device query |
| `tensor_k_tiles` | appears only through `block_k \| tensor_k_tiles`; no capacity scales with it beyond `block_k` | — |

## Total per-core footprint

```
footprint = tile_bytes * ( ACT_DEPTH    * block_h * block_k                 // scales with block_h, block_k
                         + WEIGHT_DEPTH * C * block_k * block_w             // scales with C, block_k, block_w
                         + has_bias * BIAS_DEPTH * C * block_w              // scales with C, block_w
                         + OUT_DEPTH    * C * block_h * block_w )           // scales with C, block_h*block_w (≤ C*dest_tiles)
          <= budget
```

Worked on this target: ResNet (`block 1×1, block_k 32, C 4, bias`) = 2048 × (64 + 256 + 8 + 8) = 688,128 B.
Qwen (`1×1, block_k 8`) = 2048 × (16 + 64 + 8 + 8) = 196,608 B. The largest single ring is ResNet's
`cb_weight_tiles`, 524,288 B, under the 1,048,560 B ring cap. `block_k = 64` would need 1,048,576 B, which is why
`select_blocking` settles on 32.

## Data-movement budget

Crossings of the tensor's home tier: DRAM for the weight and bias; DRAM or the L1 shards of another cluster for
the activation (`memory_config` axis); DRAM or L1 shards for the result (`output_memory` axis).

**ResNet-50 fc** (`1x1x32x2048 @ 1x1x2048x1024 + [1,1,1,1024]`, 8 clusters, `num_row_blocks = 1`, `rounds_per_row_block = 8`):

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| weight (2048 tiles, 4 MiB) | 1 | one row block: every column set is read by exactly one cluster | none |
| activation (64 tiles, 128 KiB) | 8 (1 MiB total) | read once per round (each cluster's round delivers it to its 4 NEOs through the ALL ring). In ResNet's call it lives in L1 shards, so these are L1→L1 NoC reads, not DRAM | none beyond those reads |
| bias (32 tiles) | 1 | one row block | none |
| output (32 tiles) | 1 | each output tile is packed once (no padded slots: 32 / 4 exact) | NoC writes to the result's shard cluster when width-sharded |

Totals: 4 MiB weight + 64 KiB bias from DRAM; 1 MiB activation from its home; 64 KiB written.

**Qwen3-VL** (`1x1x128x256 @ 256x512 (+ [512])`, 16 clusters, `num_row_blocks = 4`, `rounds_per_row_block = 4`):

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| weight (128 tiles, 256 KiB) | 4 (1 MiB) | re-read once per row block (`block_h = 1` chosen for occupancy) | none |
| activation (32 tiles, 64 KiB) | 4 (256 KiB) | read once per round; 4 rounds per row block | none |
| bias (16 tiles) | 4 | once per row block | none |
| output (64 tiles) | 1 | — | none (DRAM result) |

Totals: 1 MiB + 256 KiB + 128 KiB read, 128 KiB written.

> Cheapest-traffic split considered: 2-D output blocks with two 1-D multicasts (weight along the row-block axis,
> activation along the column-round axis; regimes R-mcast-weight + R-mcast-act). Every input crosses its home once:
> ResNet −896 KiB of activation reads (1 MiB → 128 KiB), Qwen −768 KiB weight and −192 KiB activation, with the same
> bytes added as NoC multicast payloads. Implemented: R-2D-stream (the same 2-D split, every cluster reading its own
> operands). Deferred because the multicast layers onto the built structure unchanged (only the two readers of
> non-injector clusters become receivers; compute, rings and the split stay), and multicast between clusters is
> unexercised on this part at this base (`.claude/references/quasar_op_examples/README.md`, *Not covered*). Within
> the built split, the traffic-minimal block (`block_h = M_t`: Qwen weight 1×, 256 KiB) loses to the occupancy-first
> choice (16 vs 4 clusters). That trade is the "occupancy-first blocks" perf lamp in `op_design.md`.

## Implementation record (Phase 0, implementer)

Built as planned: the four rings above with the capacities above, each computed in
`linear_program_artifacts.py` (`BlockPlan.*_entries`) from `block_h`, `block_w`, `block_k`, `THREAD_PLAN.compute`
and the `*_DEPTH` constants; `cb_bias_tiles` is declared only when a bias is passed. No ring was added, merged or
removed. Bindings: `cb_act_tiles` STRIDED producer (1 hart) → **ALL** compute consumer; `cb_weight_tiles` STRIDED
producer (1 hart) → **BLOCKED** compute consumer (`block_size = block_k * block_w`); `cb_bias_tiles` the same with
`block_size = block_w`; `cb_output_tiles` STRIDED compute producer (C NEOs) → STRIDED writer consumer (W harts).
The host accepted every ring with implicit sync on the data-movement side; no depth had to be raised. The
data-movement budget above holds unchanged (block extents and the split are the design's: the 32-cluster
`select_blocking` outputs match the design's worked table, pinned in `test_linear_plan.py`).
