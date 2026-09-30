# L1 Ledger: mhc_pre

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes (from `op_design.md` Blocking Model):
**token** (extent `block_token_tiles`), **K** = stream-column × stream (extent `core_k_tiles`, the whole rank slice; its y-side projection is **C** with extent `core_c_tiles`), **rank** (group ranks, `group_cores`), **slot** (coefficient slots, whole tile), **stream** (n, whole).

Page sizes: `xT` = X tile bytes (fp32 4096, bf16 2048); `wT` = W tile bytes; `yT = xT`; `fT = 4096` (fp32); `hT = 2048` (bf16).

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_x_resident` | `x_block_depth · block_token_tiles · core_k_tiles_max` | same. Block b is held from projection through y-mix while block b+1 prefetches. | {token: spans → block_token_tiles·x_block_depth, K: spans → core_k_tiles_max, rank: streams (one rank per core), slot: n/a (0), stream: spans (inside K)} | X dtype | reader | compute | per block | none. It is the residency that makes X read once. Capacity above one block is the prefetch depth (stall shadow of the combine round trip). |
| `cb_weight` | `core_k_tiles_max` | same (resident all kernel). Pushed by the reader in `W_CHUNK_TILES` chunks. | {token: streams (reused by every block), K: spans → core_k_tiles_max, rank: streams, slot: 0, stream: spans (inside K)} | W dtype (fp32 W: UnpackToDestFp32, read only by the split's `copy_tile`) | reader | compute | whole kernel | none. Concurrent with every other CB. Re-reading per block would double DRAM traffic. |
| `cb_weight_split` (fp32 W only) | `2 · core_k_tiles_max` bf16 pages | same (resident all kernel) | {token: streams, K: spans → core_k_tiles_max (× 2 pieces), rank: streams, slot: 0, stream: spans} | Float16_b, pages [W_hi(k), W_lo(k)] | compute (`w_split_block`) | compute (`project_block_pieces`) | whole kernel, from before block 0 | **Aliases `cb_weight`** (a second buffer index in the same CBDescriptor, so it costs 0 extra bytes). Tile k of fp32 W is rewritten in place into its bf16 pair, which occupies the same 4096 B. That is safe because tile k is fully unpacked into DEST before its pair is packed. |
| `cb_bias_coef` | 1 | 1 (constant: 24 slots × 32 lanes of b_k) | {token: streams (constant), K: 0, rank: 0, slot: spans → 1 tile, stream: 0} | Float32 (UnpackToDestFp32) | reader | compute | whole kernel | none: it is a persistent operand of every `coefficients_block`. Different lifetime from all per-block CBs. |
| `cb_reduce_scaler` | 1 | 1 (constant) | {token: 0, K: 0, rank: 0, slot: 0, stream: 0} | Float16_b | reader | compute | whole kernel | none: its page format differs from every fp32 CB, and it is read by every reduce. |
| `cb_sq_acc` | 1 | 1 (Q of the current token row) | {token: streams (one row at a time), K: streams (collapsed into DEST), rank: 0, slot: 0, stream: streams} | Float32 (UnpackToDestFp32: the Accurate SFPU row collapse copies Q to DEST; Default would truncate it to tf32) | compute | compute | within `sumsq_block`, per row | Not in place: Q differs in shape from `cb_partial`'s pages (32×32 elementwise sum vs col-0 result). It cannot pack into `cb_partial` because the reduce reads Q while writing `cb_partial`. Folding into DEST was considered: the FPU accumulate and the Accurate (SFPU) row collapse are separate helper calls, and DEST does not persist across them. 1 tile. |
| `cb_partial` | `2 · block_token_tiles` | same (the block's mix + Σx² partials, sent in one transfer) | {token: spans → block_token_tiles, K: streams (reduced), rank: 0, slot: 0, stream: 0} | Float32 | compute | writer | per block | Could alias `cb_combined` (same size): on the root both are live at once (partial in flight while combine waits), so no. Non-roots could, but the address must stay uniform. |
| `cb_gathered` | `group_cores · 2 · block_token_tiles` | same on the root (all ranks' partials); **0 on non-roots** | {token: spans → block_token_tiles, K: 0, rank: spans → group_cores, slot: 0, stream: 0} | Float32 (UnpackToDestFp32) | writer (root; remote fill) | compute (root) | per block | Allocated grid-wide so its address is uniform: remote senders address it with their own base. Non-root capacity is unused, which is the stated cost of the uniform address. It cannot alias `cb_x_resident` / `cb_weight` (concurrent). Capped by `group_cores ≤ 32`. |
| `cb_combined` | `2 · block_token_tiles` | same | {token: spans → block_token_tiles, K: 0, rank: 0 (already summed), slot: 0, stream: 0} | Float32 | root: compute / non-root: remote mcast | writer | per block | **Shared**: the root's compute output and the non-roots' mcast landing are one allocation. The roles are disjoint per core, and the mcast destination address equals the source address. |
| `cb_coef_in` | `block_token_tiles` | same | {token: spans → block_token_tiles, K: 0, rank: 0, slot: spans → 1 tile, stream: 0} | Float32 (UnpackToDestFp32) | writer | compute | per block | Not `cb_combined` in place: the layout differs (coefficient-major scatter). A non-root's landing is also overwritten by the next mcast only after the receive handshake, so the scatter must copy out. |
| `cb_coef_out` | `block_token_tiles` | same | {token: spans, K: 0, rank: 0, slot: spans → 1 tile, stream: 0} | Float32 | compute | writer | per block | Not in place into `cb_coef_in`: the writer is its consumer, while compute is `cb_coef_in`'s consumer (the ownership rule needs a separate CB). |
| `cb_logits_coef` | `block_token_tiles` | owned rows of the block (≤ block_token_tiles) | {token: spans → block_token_tiles, K: 0, rank: 0, slot: spans, stream: 0} | Float32 (UnpackToDestFp32) | compute | compute | `coefficients_block` → `sinkhorn_block` (across `ymix_block`) | Not `cb_coef_out`: that CB's consumer is the writer, which pops it before the Sinkhorn runs (the stall-shadow reorder). The CB ownership rule forbids a second consumer. |
| `cb_comb_coef` | `block_token_tiles` | owned rows | {token: spans, K: 0, rank: 0, slot: spans, stream: 0} | Float32 | compute | writer | per block | Could transform `cb_logits_coef` in place, but that CB is compute-consumed and this one writer-consumed, so there would be two consumers. |
| `cb_pre_cols` | `n · block_token_tiles` | same | {token: spans → block_token_tiles, K: 0, rank: 0, slot: 0, stream: spans → n} | Float32 | writer | compute | `expand_pre_block` → `ymix_block` | none: concurrent with `cb_x_resident` (both are y-mix operands), and the layout differs from the coefficient tile (FPU col-bcast needs col 0). |
| `cb_y_out` | `y_depth · y_chunk_tiles` | `y_chunk_tiles` (one window being packed while one drains) | {token: streams (block rows one after another), K→C: streams → y_chunk_tiles window, rank: 0, slot: 0, stream: streams (accumulated in DEST)} | y dtype | compute | writer | per chunk | none. Streaming window: sizing it to the block (`block_token_tiles·core_c_tiles`) would add up to 84 KB with no reuse. Double buffering = overlap of pack and DRAM write. |
| `cb_out_stage` | 2 | 2 (one post tile, one comb tile being assembled / in flight) | {token: streams (one owned row at a time), K: 0, rank: 0, slot: 0, stream: 0} | Float32 | writer | writer | per owned row | No sharing. The candidates are all live when a post/comb row is staged: `cb_coef_out` is the source being read, `cb_comb_coef` is the source of the comb tile, and `cb_coef_in` may already hold the next scatter. Aliasing any of them would need a cross-CB lifetime proof for 8 KB. The padding columns are zeroed once, and they are why the stage is persistent rather than per-row scratch. |

## Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `n` | = 4 (W: n(n+2)=24); host asserts n(n+2)+1 ≤ 32 | W shape; Mechanism caps (coefficient slots) |
| `group_cores` | 1 ≤ G_k ≤ min(32, Ct) | group geometry derivation, `group_cores_cap = 32` |
| `core_k_tiles_max` | `n · ceil(Ct / group_cores)`; ≤ 84 for every TARGET/INPUTS shape on an 11-wide grid (C=7168); ≤ 112 on an 8-wide grid | group geometry |
| `core_c_tiles_max` | `ceil(Ct / group_cores)` | same |
| `block_token_tiles` | 1 ≤ bt ≤ min(`BLOCK_TOKEN_TILES_CAP`, `core_token_tiles_max`), and the total below ≤ `l1_budget` | selection function (op_design Mechanism caps). Implementation: `BLOCK_TOKEN_TILES_CAP = 1` (measured, perf lamp L1: finer token blocks overlap the X read of block b+1 with block b's combine / y-mix / y stores; the K slice stays whole) |
| `x_block_depth` | ∈ {2, 1}; 1 only when the total with 2 and bt=1 exceeds `l1_budget` | selection function |
| `y_chunk_tiles` | `min(core_c_tiles_max, 8)` | host constant (catalog `double_buffer`: 4–8 in flight) |
| `y_depth` | 2 | host constant |
| `l1_budget` | device L1 per core minus allocator-reserved base (queried from device/allocator at call time; BH ≈ 1.4 MB usable) | host query |

## Total per-core footprint (identical on every core; the uniform-address CBs are counted everywhere)

```
L1 = x_block_depth · block_token_tiles · core_k_tiles_max · xT         # scales with token and K knobs, depth
   + core_k_tiles_max · wT                                              # scales with K
   + block_token_tiles · fT · (2 + 2·group_cores + 2 + 1 + 1 + 1 + 1 + n)   # partial, gathered, combined, coef_in/out, logits, comb, pre_cols
   + y_depth · y_chunk_tiles · yT
   + fT·(1 + 1 + 2) + hT                                                # bias, sq_acc, out_stage, scaler
```

Worst TARGET case (T=640/4096, C=7168, fp32, 11-wide group, bt=1, depth 2): 688,128 + 344,064 + 4096·(2+22+2+4+4) = 139,264 + 65,536 + 16,384 + 2,048 ≈ **1.26 MB** (fits BH). The same case on an 8-wide grid (WH): `core_k_tiles_max` = 112 → 1.60 MB with depth 2 → the selection falls back to depth 1 → 1.14 MB.

## Data-movement budget

Reference shape T=640, C=7168, n=4, fp32 X and W, 11×10 grid (G_k = 11, G_t = 10).

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| X (73.4 MB) | 1 | Each rank's block stays resident in `cb_x_resident` from projection to y-mix, and blocks are pipelined per group. The resident unit is `block_token_tiles × core_k_tiles`, so read-once holds for every T. | none |
| W (3.67 MB) | 10 (= G_t) | Each group's rank r reads its own W slice once and keeps it resident all kernel (`cb_weight`). It is not broadcast across groups in Phase 0 (R2 deferred). | none (R2 would add a 3.67 MB column mcast) |
| b (4 KB) | 110 (once per core) | read into `cb_bias_coef` once | none |
| y (18.4 MB) | 1 | written once from `cb_y_out` | none |
| post (80 KB tiles) | 1 | written once by the owning rank | none |
| comb (80 KB tiles) | 1 | written once by the owning rank | none |
| partials | 0 | on-chip only | gather: `Mt·G_k·2` = 440 tiles = 1.8 MB (≈ 0.18 MB into each of 10 roots) |
| combined S | 0 | on-chip only | mcast: `Mt·2` tiles × (G_k−1) receivers = 400 tile deliveries = 1.6 MB |

Totals: DRAM ≈ 73.4 + 36.7 + 0.4 + 18.4 + 0.16 ≈ **129 MB** (fp32). The DRAM minimum is 95.9 MB. The bf16-X perf-focus case is 36.7 + 36.7 + 9.2 + … ≈ 83 MB vs a 50 MB minimum. NoC: ≈ 3.4 MB combine traffic (≈ 5 % of X).

> Cheapest-traffic split considered: tokens × stream-column **with W broadcast down the group-rank columns** (R2) — −33 MB DRAM (W 36.7 MB → 3.67 MB), +3.67 MB × (G_t−1) NoC column mcast, one-shot at kernel start. Implemented: tokens × stream-column (R1) with direct per-group W reads. Deferred because R2 is a stepping-stone successor: it changes only the fill path of `cb_weight` (reader DRAM read → `Mcast1D(PerColumn)` receive). Compute, every other CB and the group combine are unchanged. Phase 0 validates the single cross-core mechanism (the group combine) in isolation first. The structure keeps R2 reachable: `cb_weight` is filled once, before block 0, by the reader alone, and the same-rank cores of all groups already share an identical W slice (identical `c_start`/`core_c_tiles` per rank).

## Implementation notes (ttnn-implementer)

- `l1_budget` = `ttnn.get_max_worker_l1_unreserved_size()` − 64 KB margin (1,531,904 B unreserved on BH p150).
- Every CB push is ring-aligned: `cb_x_resident` is pushed/popped by the NOMINAL `block_token_tiles · core_k_tiles_max`
  pages (uneven ranks and the ragged last block write fewer tiles into the same slot); `cb_combined` / `cb_gathered` are
  pushed at the nominal `2 · block_token_tiles` per rank, so the root's read pointer and every non-root's landing are the
  CB base on every block. `cb_y_out` is drained in `y_chunk_tiles` windows clipped at the ring end (wrap-aware writer).
- `cb_out_stage` has no push/pop: it is writer-private scratch at the CB base (zeroed once over the NoC).
- The bias tile is staged in the first, not-yet-pushed `cb_x_resident` slot (disjoint lifetime: before block 0's X read).
- Data-movement budget unchanged: X once, W once per group (G_t×, R2 still deferred), outputs once. Measured: the reader
  is DRAM-bound (~110 MB in ~300 µs at T=640, C=7168 fp32), so the W re-read (≈ 1/3 of the bytes) is the largest
  remaining lever.

## Verifier notes (Phase 0 verification)

- **Single source of CB sizes.** Every row above is now produced by one host function,
  `mhc_pre_program_descriptor._cb_table(...)`. Both the L1 selection function (`_l1_bytes`, which is affine
  in `block_token_tiles`) and the ProgramDescriptor's CB list read it. Before this, `_l1_bytes` restated the
  page counts by hand, so a knob turn could change the CBs without changing the fit test.
- **Currency check.** The table matches the code row by row. The measured `device_l1_peak_bytes` on the golden
  run is 1,255,424 B at C=7168 fp32 (kmax=84, G=11, bt=1, depth 2), which equals the closed form above.
- **bf16 W** (now in SUPPORTED): `cb_weight` pages are 2048 B. At C=7168 that is 172 KB instead of 344 KB.

## Refinement 1 (bf16 streams, W hi/lo split)

- `cb_weight_split` aliases `cb_weight`, so the footprint formula is unchanged. For bf16 X, `xT` = 2048, which
  halves the `cb_x_resident` term. The data-movement budget is also unchanged: the split is on-chip only.
